"""
Stable resumable Stage-1 comparison v3: train every model, calculate filename-group bootstrap
confidence intervals for all main metrics, select the operational Stage-1
model using VALIDATION results only, and export predictions for the next stage.

Recommended location:
    <project-root>\\src\\stage1_compare_select_stable_v3.py

Run from the project root:
    python -m src.stage1_compare_select_stable_v3 ^
        --project-root "<project-root>" ^
        --device cuda ^
        --reps 2000 ^
        --resume

Important design:
1. The fixed exploratory test set is never used to select the model or threshold.
2. Every model receives group-bootstrap 95% CIs for:
   ROC-AUC, PR-AUC, accuracy, balanced accuracy, sensitivity,
   specificity, F1 and Brier score.
3. The threshold is selected on the validation set.
4. The operational model is selected on validation performance only.
5. After selection is frozen, the test set is evaluated once.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.pipeline import (
    frames_for_task,
    handcrafted,
    loader,
    make_model,
    predict_torch,
    seed_all,
    train_baseline,
    train_deep,
)


SEED = 20260719

BASELINE_MODELS = (
    "intensity_logistic",
    "hog_linear_svm",
)

DEEP_MODELS = (
    "small_cnn",
    "resnet18",
    "efficientnet_b0",
    "pneunet",
    "pneunet_no_attention",
    "pneunet_avg_pool",
)

ENSEMBLE_NAME = "deep_ensemble"
ENSEMBLE_COMPONENTS = (
    "resnet18",
    "efficientnet_b0",
    "pneunet",
)

METRIC_NAMES = (
    "roc_auc",
    "pr_auc",
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "f1",
    "brier",
)

HIGHER_IS_BETTER = {
    "roc_auc": True,
    "pr_auc": True,
    "accuracy": True,
    "balanced_accuracy": True,
    "sensitivity": True,
    "specificity": True,
    "f1": True,
    "brier": False,
}


@dataclass
class PredictionSet:
    model: str
    split: str
    frame: pd.DataFrame
    inference_seconds: float
    model_kind: str
    model_file: Optional[str]


def ensure_dirs(root: Path) -> None:
    for relative in (
        "artifacts",
        "models/stage1",
        "reports",
        "reports/figures",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)


def validate_manifest(manifest: pd.DataFrame) -> None:
    required = {"path", "patient", "split", "stage1", "subtype"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"manifest.csv缺少字段: {missing}")

    if manifest["path"].duplicated().any():
        examples = manifest.loc[manifest["path"].duplicated(), "path"].head().tolist()
        raise ValueError(f"manifest.csv存在重复path，例如: {examples}")

    if manifest["patient"].isna().any():
        examples = manifest.loc[manifest["patient"].isna(), "path"].head().tolist()
        raise ValueError(f"manifest.csv存在缺失filename group，例如: {examples}")

    split_values = set(manifest["split"].astype(str))
    required_splits = {"train", "val", "test"}
    if not required_splits.issubset(split_values):
        raise ValueError(
            f"manifest.csv必须包含train/val/test，实际为: {sorted(split_values)}"
        )

    # This checks the namespace-specific grouping used by the current project.
    group_splits = manifest.groupby("patient")["split"].nunique()
    if (group_splits > 1).any():
        bad = group_splits[group_splits > 1].head().index.tolist()
        raise ValueError(f"同一filename group跨越数据子集，例如: {bad}")


def score_to_probability(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    if np.all((score >= 0.0) & (score <= 1.0)):
        return score
    return 1.0 / (1.0 + np.exp(-np.clip(score, -30.0, 30.0)))


def cached_handcrafted(
    frame: pd.DataFrame,
    task: str,
    kind: str,
    split: str,
    root: Path,
) -> np.ndarray:
    path = root / "artifacts" / f"features_{task}_{kind}_{split}.npy"
    if path.exists():
        values = np.load(path)
        if len(values) != len(frame):
            raise ValueError(
                f"缓存特征数量与{split}集不一致: {path}, "
                f"features={len(values)}, rows={len(frame)}"
            )
        return values

    values = handcrafted(frame, kind)
    np.save(path, values)
    return values


def predict_baseline_split(
    model_name: str,
    frame: pd.DataFrame,
    split: str,
    root: Path,
) -> PredictionSet:
    model_path = root / "models" / "stage1" / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"未找到基线模型: {model_path}")

    kind = "intensity" if model_name == "intensity_logistic" else "hog"
    features = cached_handcrafted(frame, "stage1", kind, split, root)
    classifier = joblib.load(model_path)

    start = time.perf_counter()
    if hasattr(classifier[-1], "predict_proba"):
        score = classifier.predict_proba(features)[:, 1]
    else:
        score = classifier.decision_function(features)
    elapsed = time.perf_counter() - start
    probability = score_to_probability(score)

    pred = pd.DataFrame(
        {
            "path": frame["path"].astype(str).tolist(),
            "y_true": frame["stage1"].astype(int).tolist(),
            "probability": probability,
        }
    )
    return PredictionSet(
        model=model_name,
        split=split,
        frame=pred,
        inference_seconds=elapsed,
        model_kind="baseline",
        model_file=str(model_path),
    )


def load_deep_checkpoint(
    model_name: str,
    root: Path,
    device: torch.device,
) -> torch.nn.Module:
    checkpoint_path = root / "models" / "stage1" / f"{model_name}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"未找到深度模型checkpoint: {checkpoint_path}")

    model = make_model(model_name).to(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    state = checkpoint["state"] if isinstance(checkpoint, dict) and "state" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()
    return model


def predict_deep_split(
    model_name: str,
    frame: pd.DataFrame,
    split: str,
    root: Path,
    device: torch.device,
    batch_size: int,
) -> PredictionSet:
    workers = 0 if os.name == "nt" else 4
    model = load_deep_checkpoint(model_name, root, device)
    data_loader = loader(
        frame=frame,
        task="stage1",
        train=False,
        batch=batch_size,
        workers=workers,
    )
    y_true, probability, paths, elapsed = predict_torch(model, data_loader, device)

    del model, data_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()

    pred = pd.DataFrame(
        {
            "path": paths,
            "y_true": y_true.astype(int),
            "probability": probability.astype(float),
        }
    )
    return PredictionSet(
        model=model_name,
        split=split,
        frame=pred,
        inference_seconds=elapsed,
        model_kind="deep",
        model_file=str(root / "models" / "stage1" / f"{model_name}.pt"),
    )


def validate_prediction_frame(
    pred: pd.DataFrame,
    expected: pd.DataFrame,
    model_name: str,
    split: str,
) -> pd.DataFrame:
    required = {"path", "y_true", "probability"}
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(f"{model_name}/{split}预测缺少字段: {sorted(missing)}")

    if pred["path"].duplicated().any():
        raise ValueError(f"{model_name}/{split}存在重复预测path")

    if pred["probability"].isna().any():
        raise ValueError(f"{model_name}/{split}存在空概率")

    if not pred["probability"].between(0.0, 1.0).all():
        bad = pred.loc[
            ~pred["probability"].between(0.0, 1.0),
            ["path", "probability"],
        ].head()
        raise ValueError(f"{model_name}/{split}概率不在[0,1]:\n{bad}")

    expected_check = expected[["path", "stage1"]].rename(
        columns={"stage1": "expected_y"}
    )
    merged = expected_check.merge(
        pred[["path", "y_true", "probability"]],
        on="path",
        how="left",
        validate="one_to_one",
    )
    if merged["probability"].isna().any():
        missing_paths = merged.loc[merged["probability"].isna(), "path"].head().tolist()
        raise ValueError(
            f"{model_name}/{split}缺少{merged['probability'].isna().sum()}张预测，例如: "
            f"{missing_paths}"
        )

    if not np.array_equal(
        merged["expected_y"].astype(int).to_numpy(),
        merged["y_true"].astype(int).to_numpy(),
    ):
        raise ValueError(f"{model_name}/{split}的y_true与manifest不一致")

    return merged[["path", "y_true", "probability"]].copy()


def combine_ensemble(
    predictions: Mapping[str, PredictionSet],
    split: str,
    components: Sequence[str],
) -> PredictionSet:
    missing = [name for name in components if name not in predictions]
    if missing:
        raise ValueError(f"无法构建ensemble，缺少模型: {missing}")

    merged: Optional[pd.DataFrame] = None
    total_seconds = 0.0
    for name in components:
        frame = predictions[name].frame[
            ["path", "y_true", "probability"]
        ].rename(columns={"probability": name})
        merged = (
            frame
            if merged is None
            else merged.merge(
                frame[["path", name]],
                on="path",
                how="inner",
                validate="one_to_one",
            )
        )
        total_seconds += predictions[name].inference_seconds

    assert merged is not None
    probability = merged[list(components)].mean(axis=1)
    frame = pd.DataFrame(
        {
            "path": merged["path"],
            "y_true": merged["y_true"].astype(int),
            "probability": probability.astype(float),
        }
    )
    return PredictionSet(
        model=ENSEMBLE_NAME,
        split=split,
        frame=frame,
        inference_seconds=total_seconds,
        model_kind="ensemble",
        model_file=None,
    )


def binary_roc_auc_numpy(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> float:
    """Tie-aware binary ROC-AUC without repeatedly calling sklearn."""
    y_true = np.asarray(y_true, dtype=np.int8)
    probability = np.asarray(probability, dtype=np.float64)

    positive = y_true == 1
    negative = y_true == 0
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    if n_positive == 0 or n_negative == 0:
        return float("nan")

    order = np.argsort(probability, kind="mergesort")
    sorted_probability = probability[order]
    sorted_y = y_true[order]

    concordant = 0.0
    prior_negative = 0.0
    start = 0
    while start < len(sorted_y):
        end = start + 1
        while (
            end < len(sorted_y)
            and sorted_probability[end] == sorted_probability[start]
        ):
            end += 1

        block = sorted_y[start:end]
        block_positive = float((block == 1).sum())
        block_negative = float((block == 0).sum())
        concordant += block_positive * (
            prior_negative + 0.5 * block_negative
        )
        prior_negative += block_negative
        start = end

    return float(concordant / (n_positive * n_negative))


def average_precision_numpy(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> float:
    """Binary average precision using descending stable score order."""
    y_true = np.asarray(y_true, dtype=np.int8)
    probability = np.asarray(probability, dtype=np.float64)
    n_positive = int((y_true == 1).sum())
    if n_positive == 0:
        return float("nan")

    order = np.argsort(-probability, kind="mergesort")
    ranked_y = y_true[order]
    cumulative_positive = np.cumsum(ranked_y == 1)
    ranks = np.arange(1, len(ranked_y) + 1, dtype=np.float64)
    precision_at_rank = cumulative_positive / ranks
    return float(
        precision_at_rank[ranked_y == 1].sum() / n_positive
    )


def threshold_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int8)
    probability = np.asarray(probability, dtype=np.float64)
    y_pred = (probability >= threshold).astype(np.int8)

    n_positive = int((y_true == 1).sum())
    n_negative = int((y_true == 0).sum())
    if n_positive == 0 or n_negative == 0:
        raise ValueError("指标计算需要同时包含正常和肺炎类别")

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / len(y_true)
    balanced_accuracy = 0.5 * (sensitivity + specificity)
    f1_denominator = 2 * tp + fp + fn
    f1 = (
        2 * tp / f1_denominator
        if f1_denominator > 0
        else 0.0
    )

    return {
        "roc_auc": binary_roc_auc_numpy(y_true, probability),
        "pr_auc": average_precision_numpy(y_true, probability),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
        "brier": float(np.mean((probability - y_true) ** 2)),
    }


def confusion_rates_numpy(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> Tuple[float, float, float, float]:
    """Return sensitivity, specificity, balanced accuracy and F1."""
    y_true = np.asarray(y_true, dtype=np.int8)
    probability = np.asarray(probability, dtype=np.float64)
    y_pred = probability >= float(threshold)

    tp = int(((y_true == 1) & y_pred).sum())
    fn = int(((y_true == 1) & (~y_pred)).sum())
    tn = int(((y_true == 0) & (~y_pred)).sum())
    fp = int(((y_true == 0) & y_pred).sum())

    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    balanced = 0.5 * (sensitivity + specificity)
    denominator = 2 * tp + fp + fn
    f1 = 2 * tp / denominator if denominator else 0.0
    return (
        float(sensitivity),
        float(specificity),
        float(balanced),
        float(f1),
    )


def choose_threshold(
    y_true: np.ndarray,
    probability: np.ndarray,
    rule: str,
    target_sensitivity: float,
) -> float:
    """
    Select a threshold using validation data only.

    This implementation deliberately uses NumPy only. It avoids repeated
    sklearn classification-metric calls, which can trigger a native Windows
    crash in some sklearn/NumPy builds.
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    probability = np.asarray(probability, dtype=np.float64)

    if len(y_true) != len(probability):
        raise ValueError("y_true与probability长度不一致")
    if len(np.unique(y_true)) < 2:
        raise ValueError("阈值选择需要同时包含正常和肺炎类别")
    if not np.isfinite(probability).all():
        raise ValueError("probability包含NaN或无穷值")
    if not np.all((probability >= 0.0) & (probability <= 1.0)):
        raise ValueError("probability必须位于[0,1]")

    if rule == "fixed_0.5":
        return 0.5

    # With prediction defined as probability >= threshold, every distinct
    # classification state is represented by one observed probability.
    thresholds = np.unique(np.r_[0.0, probability, 1.0]).astype(np.float64)

    rows: List[Tuple[float, float, float, float, float]] = []
    # row = sensitivity, specificity, balanced accuracy, F1, threshold
    for threshold in thresholds:
        sensitivity, specificity, balanced, f1 = confusion_rates_numpy(
            y_true,
            probability,
            float(threshold),
        )
        rows.append(
            (
                sensitivity,
                specificity,
                balanced,
                f1,
                float(threshold),
            )
        )

    if rule == "youden":
        # Youden J = sensitivity + specificity - 1.
        # Tie-break: higher balanced accuracy, then higher sensitivity,
        # then higher threshold.
        best = max(
            rows,
            key=lambda row: (
                row[0] + row[1] - 1.0,
                row[2],
                row[0],
                row[4],
            ),
        )
        return float(best[4])

    if rule == "f1":
        # Tie-break: balanced accuracy, sensitivity, specificity, threshold.
        best = max(
            rows,
            key=lambda row: (
                row[3],
                row[2],
                row[0],
                row[1],
                row[4],
            ),
        )
        return float(best[4])

    if rule == "balanced_accuracy":
        # Tie-break: sensitivity, specificity, threshold.
        best = max(
            rows,
            key=lambda row: (
                row[2],
                row[0],
                row[1],
                row[4],
            ),
        )
        return float(best[4])

    if rule == "sensitivity":
        # Screening-first threshold:
        # among thresholds meeting the requested sensitivity, maximize
        # specificity, then balanced accuracy. If none meet the target,
        # maximize sensitivity first.
        candidates = [
            row for row in rows if row[0] >= target_sensitivity
        ]

        if candidates:
            best = max(
                candidates,
                key=lambda row: (
                    row[1],  # specificity
                    row[2],  # balanced accuracy
                    row[3],  # F1
                    row[4],  # threshold
                ),
            )
        else:
            best = max(
                rows,
                key=lambda row: (
                    row[0],  # sensitivity
                    row[1],  # specificity
                    row[2],  # balanced accuracy
                    row[3],  # F1
                    row[4],  # threshold
                ),
            )
        return float(best[4])

    raise ValueError(f"未知threshold rule: {rule}")



def generate_group_bootstrap_indices(
    groups: Sequence[str],
    reps: int,
    seed: int,
) -> List[np.ndarray]:
    group_array = pd.Series(groups, dtype="string")
    if group_array.isna().any():
        raise ValueError("bootstrap group中存在缺失值")

    unique_groups = sorted(group_array.unique().tolist())
    group_to_code = {group: index for index, group in enumerate(unique_groups)}
    codes = np.asarray(
        [group_to_code[str(group)] for group in group_array],
        dtype=np.int64,
    )

    rng = np.random.default_rng(seed)
    indices: List[np.ndarray] = []
    row_ids = np.arange(len(codes), dtype=np.int64)

    while len(indices) < reps:
        sampled = rng.integers(
            0,
            len(unique_groups),
            size=len(unique_groups),
        )
        counts = np.bincount(
            sampled,
            minlength=len(unique_groups),
        )
        index = np.repeat(row_ids, counts[codes])
        indices.append(index)

    return indices


def bootstrap_metric_intervals(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    bootstrap_indices: Sequence[np.ndarray],
) -> Dict[str, Tuple[float, float]]:
    samples: Dict[str, List[float]] = {
        name: [] for name in METRIC_NAMES
    }

    for index in bootstrap_indices:
        y_boot = y_true[index]
        p_boot = probability[index]
        if len(np.unique(y_boot)) < 2:
            continue
        values = threshold_metrics(y_boot, p_boot, threshold)
        for name, value in values.items():
            if np.isfinite(value):
                samples[name].append(float(value))

    intervals: Dict[str, Tuple[float, float]] = {}
    for name in METRIC_NAMES:
        values = np.asarray(samples[name], dtype=float)
        if len(values) == 0:
            intervals[name] = (float("nan"), float("nan"))
        else:
            low, high = np.quantile(values, [0.025, 0.975])
            intervals[name] = (float(low), float(high))
    return intervals


def evaluate_prediction_set(
    prediction: PredictionSet,
    threshold: float,
    manifest: pd.DataFrame,
    bootstrap_indices: Sequence[np.ndarray],
) -> Dict[str, object]:
    group_map = manifest.set_index("path")["patient"]
    frame = prediction.frame.copy()
    frame["group"] = frame["path"].map(group_map)

    if frame["group"].isna().any():
        examples = frame.loc[frame["group"].isna(), "path"].head().tolist()
        raise ValueError(
            f"{prediction.model}/{prediction.split}无法匹配group，例如: {examples}"
        )

    y_true = frame["y_true"].astype(int).to_numpy()
    probability = frame["probability"].astype(float).to_numpy()

    point = threshold_metrics(y_true, probability, threshold)
    intervals = bootstrap_metric_intervals(
        y_true,
        probability,
        threshold,
        bootstrap_indices,
    )

    row: Dict[str, object] = {
        "split": prediction.split,
        "model": prediction.model,
        "model_kind": prediction.model_kind,
        "n_images": int(len(frame)),
        "n_filename_groups": int(frame["group"].nunique()),
        "threshold": float(threshold),
        "inference_seconds": float(prediction.inference_seconds),
        "inference_ms_per_image": (
            float(prediction.inference_seconds / len(frame) * 1000.0)
            if len(frame)
            else float("nan")
        ),
        "model_file": prediction.model_file or "",
    }

    for name in METRIC_NAMES:
        row[name] = point[name]
        row[f"{name}_ci_low"] = intervals[name][0]
        row[f"{name}_ci_high"] = intervals[name][1]

    y_pred = (probability >= threshold).astype(int)
    row.update(
        {
            "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
            "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
            "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
            "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        }
    )
    return row


def select_operational_model(
    validation_results: pd.DataFrame,
    minimum_sensitivity: float,
    minimum_specificity: float,
    allow_ensemble: bool,
) -> Tuple[pd.Series, pd.DataFrame, str]:
    candidates = validation_results.copy()

    if not allow_ensemble:
        candidates = candidates[candidates["model_kind"] != "ensemble"]

    eligible = candidates[
        (candidates["sensitivity"] >= minimum_sensitivity)
        & (candidates["specificity"] >= minimum_specificity)
    ].copy()

    if len(eligible) == 0:
        selection_pool = candidates.copy()
        reason = (
            "No model met both validation operating constraints; "
            "all allowed models were ranked."
        )
    else:
        selection_pool = eligible
        reason = (
            f"Eligible models met validation sensitivity >= "
            f"{minimum_sensitivity:.3f} and specificity >= "
            f"{minimum_specificity:.3f}."
        )

    # Selection uses VALIDATION results only.
    # Main criterion: robust discrimination (lower 95% AUC bound).
    # Tie-breakers: lower balanced-accuracy bound, lower PR-AUC bound,
    # point sensitivity, point specificity, then inference speed.
    selection_pool = selection_pool.sort_values(
        by=[
            "roc_auc_ci_low",
            "balanced_accuracy_ci_low",
            "pr_auc_ci_low",
            "sensitivity",
            "specificity",
            "inference_ms_per_image",
            "model",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
            True,
            True,
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    selection_pool["selection_rank"] = np.arange(
        1, len(selection_pool) + 1
    )
    winner = selection_pool.iloc[0]
    return winner, selection_pool, reason


def paired_auc_selected_vs_others(
    selected_model: str,
    test_predictions: Mapping[str, PredictionSet],
    bootstrap_indices: Sequence[np.ndarray],
) -> pd.DataFrame:
    selected = test_predictions[selected_model].frame[
        ["path", "y_true", "probability"]
    ].rename(columns={"probability": "selected_probability"})

    rows: List[Dict[str, object]] = []
    for other_name, prediction in test_predictions.items():
        if other_name == selected_model:
            continue

        other = prediction.frame[
            ["path", "y_true", "probability"]
        ].rename(columns={"probability": "other_probability"})

        merged = selected.merge(
            other[["path", "other_probability"]],
            on="path",
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != len(selected):
            raise ValueError(
                f"{selected_model}与{other_name}的测试样本不完全一致"
            )

        y_true = merged["y_true"].astype(int).to_numpy()
        selected_p = merged["selected_probability"].astype(float).to_numpy()
        other_p = merged["other_probability"].astype(float).to_numpy()

        point = float(
            binary_roc_auc_numpy(y_true, selected_p)
            - binary_roc_auc_numpy(y_true, other_p)
        )

        differences: List[float] = []
        for index in bootstrap_indices:
            y_boot = y_true[index]
            if len(np.unique(y_boot)) < 2:
                continue
            difference = (
                binary_roc_auc_numpy(y_boot, selected_p[index])
                - binary_roc_auc_numpy(y_boot, other_p[index])
            )
            differences.append(float(difference))

        low, high = np.quantile(
            np.asarray(differences),
            [0.025, 0.975],
        )

        rows.append(
            {
                "selected_model": selected_model,
                "other_model": other_name,
                "auc_difference_selected_minus_other": point,
                "ci95_low": float(low),
                "ci95_high": float(high),
                "conclusive": bool(low > 0.0 or high < 0.0),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "auc_difference_selected_minus_other",
        ascending=False,
    )


def save_prediction(
    prediction: PredictionSet,
    threshold: float,
    path: Path,
) -> None:
    frame = prediction.frame.copy()
    frame["threshold"] = float(threshold)
    frame["y_pred"] = (
        frame["probability"].astype(float) >= threshold
    ).astype(int)
    frame.to_csv(path, index=False)


def make_comparison_plot(
    test_results: pd.DataFrame,
    selected_model: str,
    output_base: Path,
) -> None:
    plot = test_results.sort_values("roc_auc", ascending=True).copy()
    y_position = np.arange(len(plot))

    lower_error = plot["roc_auc"] - plot["roc_auc_ci_low"]
    upper_error = plot["roc_auc_ci_high"] - plot["roc_auc"]

    fig, ax = plt.subplots(figsize=(8.0, max(4.8, 0.48 * len(plot) + 1.5)))
    ax.errorbar(
        plot["roc_auc"],
        y_position,
        xerr=np.vstack([lower_error, upper_error]),
        fmt="o",
        capsize=3,
        linewidth=1.2,
    )
    ax.set_yticks(y_position)
    ax.set_yticklabels(plot["model"])
    ax.set_xlabel("Locked-test ROC-AUC (95% filename-group bootstrap CI)")
    ax.set_title(
        f"Stage-1 model comparison; validation-selected model: {selected_model}"
    )
    ax.grid(axis="x", alpha=0.25)

    for y, (_, row) in zip(y_position, plot.iterrows()):
        if row["model"] == selected_model:
            ax.text(
                row["roc_auc_ci_high"] + 0.001,
                y,
                "selected",
                va="center",
                fontweight="bold",
            )

    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=400)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)


def write_selected_manifest(
    selected_test: PredictionSet,
    threshold: float,
    manifest: pd.DataFrame,
    output: Path,
) -> None:
    details = manifest[
        ["path", "patient", "original_split", "subtype", "stage1"]
    ].copy()
    selected = selected_test.frame.merge(
        details,
        on="path",
        how="left",
        validate="one_to_one",
    )
    selected["stage1_threshold"] = float(threshold)
    selected["stage1_prediction"] = (
        selected["probability"] >= threshold
    ).astype(int)
    selected["route_to_stage2"] = selected["stage1_prediction"] == 1
    selected.to_csv(output, index=False)


def load_existing_prediction(
    path: Path,
    model_name: str,
    split: str,
    model_kind: str,
    model_file: Optional[str],
) -> Optional[PredictionSet]:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    required = {"path", "y_true", "probability"}
    if not required.issubset(frame.columns):
        return None
    return PredictionSet(
        model=model_name,
        split=split,
        frame=frame[["path", "y_true", "probability"]].copy(),
        inference_seconds=0.0,
        model_kind=model_kind,
        model_file=model_file,
    )


def save_raw_prediction_immediately(
    prediction: PredictionSet,
    path: Path,
) -> None:
    """Persist inference before bootstrap so a later failure loses no work."""
    prediction.frame[["path", "y_true", "probability"]].to_csv(
        path,
        index=False,
    )
    print(
        f"PREDICTION_SAVED {prediction.split} {prediction.model} "
        f"n={len(prediction.frame)} -> {path}",
        flush=True,
    )


def train_or_reuse_all_models(
    root: Path,
    manifest: pd.DataFrame,
    device: torch.device,
    resume: bool,
    batch_size: int,
) -> Tuple[
    Dict[str, PredictionSet],
    Dict[str, PredictionSet],
]:
    val_df = frames_for_task(manifest, "stage1", "val")
    test_df = frames_for_task(manifest, "stage1", "test")

    validation_predictions: Dict[str, PredictionSet] = {}
    test_predictions: Dict[str, PredictionSet] = {}

    for model_name in BASELINE_MODELS:
        print(f"\n[Stage1 baseline] {model_name}", flush=True)
        model_path = root / "models" / "stage1" / f"{model_name}.joblib"
        val_path = (
            root / "artifacts" / f"val_predictions_stage1_{model_name}.csv"
        )
        test_path = (
            root / "artifacts" / f"predictions_stage1_{model_name}.csv"
        )

        val_existing = (
            load_existing_prediction(
                val_path,
                model_name,
                "val",
                "baseline",
                str(model_path),
            )
            if resume
            else None
        )
        test_existing = (
            load_existing_prediction(
                test_path,
                model_name,
                "test",
                "baseline",
                str(model_path),
            )
            if resume
            else None
        )

        if val_existing is not None and test_existing is not None:
            validation_predictions[model_name] = val_existing
            test_predictions[model_name] = test_existing
            print(
                f"PREDICTIONS_REUSED stage1 {model_name}",
                flush=True,
            )
            continue

        stale_prediction = test_path
        if not model_path.exists() and stale_prediction.exists():
            stale_prediction.unlink()

        train_baseline(model_name, "stage1", manifest, root)

        validation_predictions[model_name] = (
            val_existing
            if val_existing is not None
            else predict_baseline_split(
                model_name,
                val_df,
                "val",
                root,
            )
        )
        test_predictions[model_name] = (
            test_existing
            if test_existing is not None
            else predict_baseline_split(
                model_name,
                test_df,
                "test",
                root,
            )
        )

        validation_predictions[model_name].frame = validate_prediction_frame(
            validation_predictions[model_name].frame,
            val_df,
            model_name,
            "val",
        )
        test_predictions[model_name].frame = validate_prediction_frame(
            test_predictions[model_name].frame,
            test_df,
            model_name,
            "test",
        )
        save_raw_prediction_immediately(
            validation_predictions[model_name],
            val_path,
        )
        save_raw_prediction_immediately(
            test_predictions[model_name],
            test_path,
        )

    for model_name in DEEP_MODELS:
        print(f"\n[Stage1 deep] {model_name}", flush=True)
        checkpoint_path = (
            root / "models" / "stage1" / f"{model_name}.pt"
        )
        val_path = (
            root / "artifacts" / f"val_predictions_stage1_{model_name}.csv"
        )
        test_path = (
            root / "artifacts" / f"predictions_stage1_{model_name}.csv"
        )

        val_existing = (
            load_existing_prediction(
                val_path,
                model_name,
                "val",
                "deep",
                str(checkpoint_path),
            )
            if resume
            else None
        )
        test_existing = (
            load_existing_prediction(
                test_path,
                model_name,
                "test",
                "deep",
                str(checkpoint_path),
            )
            if resume
            else None
        )

        if val_existing is not None and test_existing is not None:
            validation_predictions[model_name] = val_existing
            test_predictions[model_name] = test_existing
            print(
                f"PREDICTIONS_REUSED stage1 {model_name}",
                flush=True,
            )
            continue

        effective_resume = bool(resume and checkpoint_path.exists())
        train_deep(
            model_name,
            "stage1",
            manifest,
            root,
            quick=False,
            resume=effective_resume,
        )

        validation_predictions[model_name] = (
            val_existing
            if val_existing is not None
            else predict_deep_split(
                model_name,
                val_df,
                "val",
                root,
                device,
                batch_size,
            )
        )

        # Save validation inference immediately. This is the expensive result
        # needed for threshold/model selection.
        validation_predictions[model_name].frame = validate_prediction_frame(
            validation_predictions[model_name].frame,
            val_df,
            model_name,
            "val",
        )
        save_raw_prediction_immediately(
            validation_predictions[model_name],
            val_path,
        )

        test_predictions[model_name] = (
            test_existing
            if test_existing is not None
            else predict_deep_split(
                model_name,
                test_df,
                "test",
                root,
                device,
                batch_size,
            )
        )
        test_predictions[model_name].frame = validate_prediction_frame(
            test_predictions[model_name].frame,
            test_df,
            model_name,
            "test",
        )
        save_raw_prediction_immediately(
            test_predictions[model_name],
            test_path,
        )

    # Validate reused files as well.
    for model_name in list(validation_predictions):
        validation_predictions[model_name].frame = validate_prediction_frame(
            validation_predictions[model_name].frame,
            val_df,
            model_name,
            "val",
        )
        test_predictions[model_name].frame = validate_prediction_frame(
            test_predictions[model_name].frame,
            test_df,
            model_name,
            "test",
        )

    validation_predictions[ENSEMBLE_NAME] = combine_ensemble(
        validation_predictions,
        "val",
        ENSEMBLE_COMPONENTS,
    )
    test_predictions[ENSEMBLE_NAME] = combine_ensemble(
        test_predictions,
        "test",
        ENSEMBLE_COMPONENTS,
    )
    save_raw_prediction_immediately(
        validation_predictions[ENSEMBLE_NAME],
        root / "artifacts" / f"val_predictions_stage1_{ENSEMBLE_NAME}.csv",
    )
    save_raw_prediction_immediately(
        test_predictions[ENSEMBLE_NAME],
        root / "artifacts" / f"predictions_stage1_{ENSEMBLE_NAME}.csv",
    )

    return validation_predictions, test_predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train/compare all Stage-1 models, calculate group-bootstrap "
            "confidence intervals and select the operational model using "
            "validation results only."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=2000,
        help="Filename-group bootstrap repetitions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed checkpoints and predictions when available.",
    )
    parser.add_argument(
        "--threshold-rule",
        choices=[
            "sensitivity",
            "youden",
            "f1",
            "balanced_accuracy",
            "fixed_0.5",
        ],
        default="sensitivity",
    )
    parser.add_argument(
        "--threshold-target-sensitivity",
        type=float,
        default=0.95,
        help=(
            "Used only when --threshold-rule sensitivity. "
            "Threshold selection is performed on validation data."
        ),
    )
    parser.add_argument(
        "--minimum-val-sensitivity",
        type=float,
        default=0.95,
        help="Eligibility requirement for validation-based model selection.",
    )
    parser.add_argument(
        "--minimum-val-specificity",
        type=float,
        default=0.90,
        help="Eligibility requirement for validation-based model selection.",
    )
    parser.add_argument(
        "--allow-ensemble-selection",
        action="store_true",
        help=(
            "Allow the equal-weight deep ensemble to become the operational "
            "Stage-1 model. By default it is compared but not selected."
        ),
    )
    parser.add_argument(
        "--run-paired-comparison",
        action="store_true",
        help=(
            "Optionally run selected-vs-all paired AUC bootstrap after all "
            "primary outputs have already been saved. Disabled by default "
            "because it is not required for model selection."
        ),
    )
    parser.add_argument(
        "--paired-reps",
        type=int,
        default=500,
        help=(
            "Bootstrap repetitions for the optional selected-vs-all paired "
            "AUC analysis. Used only with --run-paired-comparison."
        ),
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    ensure_dirs(root)
    seed_all(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了cuda，但当前PyTorch检测不到CUDA")

    manifest_path = root / "data" / "processed" / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"未找到manifest: {manifest_path}\n"
            "请先运行数据清单构建流程。"
        )

    manifest = pd.read_csv(manifest_path)
    validate_manifest(manifest)

    val_frame = frames_for_task(manifest, "stage1", "val").reset_index(drop=True)
    test_frame = frames_for_task(manifest, "stage1", "test").reset_index(drop=True)

    print(
        f"Stage-1 validation: {len(val_frame)} images, "
        f"{val_frame['patient'].nunique()} filename groups",
        flush=True,
    )
    print(
        f"Stage-1 locked test: {len(test_frame)} images, "
        f"{test_frame['patient'].nunique()} filename groups",
        flush=True,
    )

    validation_predictions, test_predictions = train_or_reuse_all_models(
        root=root,
        manifest=manifest,
        device=device,
        resume=args.resume,
        batch_size=args.batch_size,
    )

    # Use identical bootstrap group draws for every model within a split.
    val_bootstrap = generate_group_bootstrap_indices(
        val_frame["patient"].astype(str).tolist(),
        reps=args.reps,
        seed=args.seed,
    )
    test_bootstrap = generate_group_bootstrap_indices(
        test_frame["patient"].astype(str).tolist(),
        reps=args.reps,
        seed=args.seed + 1,
    )

    thresholds: Dict[str, float] = {}
    validation_rows: List[Dict[str, object]] = []
    test_rows: List[Dict[str, object]] = []

    for model_name in validation_predictions:
        print(
            f"\n[EVAL_START] {model_name}: "
            f"{args.reps} group-bootstrap draws",
            flush=True,
        )
        try:
            val_pred = validation_predictions[model_name]
            test_pred = test_predictions[model_name]

            threshold = choose_threshold(
                val_pred.frame["y_true"].astype(int).to_numpy(),
                val_pred.frame["probability"].astype(float).to_numpy(),
                rule=args.threshold_rule,
                target_sensitivity=args.threshold_target_sensitivity,
            )
            thresholds[model_name] = threshold

            validation_row = evaluate_prediction_set(
                val_pred,
                threshold,
                manifest,
                val_bootstrap,
            )
            test_row = evaluate_prediction_set(
                test_pred,
                threshold,
                manifest,
                test_bootstrap,
            )
            validation_rows.append(validation_row)
            test_rows.append(test_row)

            save_prediction(
                val_pred,
                threshold,
                root / "artifacts" / f"val_predictions_stage1_{model_name}.csv",
            )
            save_prediction(
                test_pred,
                threshold,
                root / "artifacts" / f"predictions_stage1_{model_name}.csv",
            )

            # Partial reports survive any later interruption.
            pd.DataFrame(validation_rows).to_csv(
                root / "reports" / "stage1_validation_ci_partial.csv",
                index=False,
            )
            pd.DataFrame(test_rows).to_csv(
                root / "reports" / "stage1_locked_test_ci_partial.csv",
                index=False,
            )
            print(
                f"[EVAL_DONE] {model_name}: "
                f"val_auc={validation_row['roc_auc']:.6f}, "
                f"test_auc={test_row['roc_auc']:.6f}",
                flush=True,
            )
        except Exception as error:
            error_path = (
                root / "reports" / "stage1_compare_select_error.txt"
            )
            import traceback
            error_path.write_text(
                traceback.format_exc(),
                encoding="utf-8",
            )
            print(
                f"[EVAL_FAILED] {model_name}: {error}\n"
                f"完整错误已保存到: {error_path}",
                flush=True,
            )
            raise

    validation_results = pd.DataFrame(validation_rows)
    test_results = pd.DataFrame(test_rows)

    winner, ranking, selection_reason = select_operational_model(
        validation_results=validation_results,
        minimum_sensitivity=args.minimum_val_sensitivity,
        minimum_specificity=args.minimum_val_specificity,
        allow_ensemble=args.allow_ensemble_selection,
    )

    selected_model = str(winner["model"])
    selected_threshold = float(thresholds[selected_model])

    validation_results["selected_operational_model"] = (
        validation_results["model"] == selected_model
    )
    test_results["selected_operational_model"] = (
        test_results["model"] == selected_model
    )

    validation_results = validation_results.sort_values(
        "roc_auc", ascending=False
    )
    test_results = test_results.sort_values(
        "roc_auc", ascending=False
    )

    validation_results.to_csv(
        root / "reports" / "stage1_all_models_validation_ci.csv",
        index=False,
    )
    test_results.to_csv(
        root / "reports" / "stage1_all_models_locked_test_ci.csv",
        index=False,
    )
    ranking.to_csv(
        root / "reports" / "stage1_validation_selection_ranking.csv",
        index=False,
    )

    selected_val = validation_predictions[selected_model]
    selected_test = test_predictions[selected_model]

    save_prediction(
        selected_val,
        selected_threshold,
        root / "artifacts" / "val_predictions_stage1_selected.csv",
    )
    save_prediction(
        selected_test,
        selected_threshold,
        root / "artifacts" / "predictions_stage1_selected.csv",
    )

    write_selected_manifest(
        selected_test=selected_test,
        threshold=selected_threshold,
        manifest=manifest,
        output=root
        / "artifacts"
        / "stage1_selected_test_routing_to_stage2.csv",
    )

    selected_test_row = test_results[
        test_results["model"] == selected_model
    ].iloc[0].to_dict()

    selected_info = {
        "selected_model": selected_model,
        "selected_threshold": selected_threshold,
        "selection_split": "validation",
        "selection_used_locked_test": False,
        "selection_rule": {
            "threshold_rule": args.threshold_rule,
            "threshold_target_sensitivity": args.threshold_target_sensitivity,
            "minimum_validation_sensitivity": args.minimum_val_sensitivity,
            "minimum_validation_specificity": args.minimum_val_specificity,
            "allow_ensemble_selection": args.allow_ensemble_selection,
            "ranking": [
                "validation roc_auc lower 95% CI",
                "validation balanced_accuracy lower 95% CI",
                "validation pr_auc lower 95% CI",
                "validation sensitivity",
                "validation specificity",
                "inference speed",
            ],
            "reason": selection_reason,
        },
        "selected_validation_metrics": {
            key: (
                value.item()
                if isinstance(value, np.generic)
                else value
            )
            for key, value in winner.to_dict().items()
        },
        "selected_locked_test_metrics": {
            key: (
                value.item()
                if isinstance(value, np.generic)
                else value
            )
            for key, value in selected_test_row.items()
        },
        "bootstrap": {
            "unit": "namespace-specific filename group",
            "repetitions": args.reps,
            "seed_validation": args.seed,
            "seed_test": args.seed + 1,
            "confidence_interval": "percentile 95%",
        },
        "operational_artifacts": {
            "validation_predictions": (
                "artifacts/val_predictions_stage1_selected.csv"
            ),
            "test_predictions": (
                "artifacts/predictions_stage1_selected.csv"
            ),
            "test_routing_manifest": (
                "artifacts/stage1_selected_test_routing_to_stage2.csv"
            ),
            "model_file": selected_test.model_file,
            "ensemble_components": (
                list(ENSEMBLE_COMPONENTS)
                if selected_model == ENSEMBLE_NAME
                else []
            ),
        },
    }

    (root / "reports" / "stage1_selected_model.json").write_text(
        json.dumps(selected_info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Optional secondary analysis. All primary tables, selected predictions,
    # routing output and selected-model metadata have already been saved above.
    paired_output = (
        root / "reports" / "stage1_selected_vs_all_paired_auc.csv"
    )
    if args.run_paired_comparison:
        print(
            f"\n[PAIRED_AUC_START] selected={selected_model}, "
            f"reps={args.paired_reps}",
            flush=True,
        )
        try:
            paired_bootstrap = generate_group_bootstrap_indices(
                test_frame["patient"].astype(str).tolist(),
                reps=args.paired_reps,
                seed=args.seed + 2,
            )
            paired = paired_auc_selected_vs_others(
                selected_model=selected_model,
                test_predictions=test_predictions,
                bootstrap_indices=paired_bootstrap,
            )
            paired.to_csv(paired_output, index=False)
            print(
                f"[PAIRED_AUC_DONE] -> {paired_output}",
                flush=True,
            )
        except Exception:
            import traceback
            paired_error = (
                root
                / "reports"
                / "stage1_paired_auc_error.txt"
            )
            paired_error.write_text(
                traceback.format_exc(),
                encoding="utf-8",
            )
            print(
                "[PAIRED_AUC_FAILED] Primary results remain complete. "
                f"See: {paired_error}",
                flush=True,
            )
    else:
        pd.DataFrame(
            columns=[
                "selected_model",
                "other_model",
                "auc_difference_selected_minus_other",
                "ci95_low",
                "ci95_high",
                "conclusive",
            ]
        ).to_csv(paired_output, index=False)
        print(
            "\n[PAIRED_AUC_SKIPPED] Optional paired comparison was not "
            "requested. Primary model comparison and selection are complete.",
            flush=True,
        )

    make_comparison_plot(
        test_results=test_results,
        selected_model=selected_model,
        output_base=root
        / "reports"
        / "figures"
        / "stage1_all_models_roc_auc_ci",
    )

    print("\n" + "=" * 80)
    print("Stage-1全部模型验证集比较（选择只基于验证集）")
    print(
        validation_results[
            [
                "model",
                "threshold",
                "roc_auc",
                "roc_auc_ci_low",
                "roc_auc_ci_high",
                "sensitivity",
                "specificity",
                "balanced_accuracy",
                "f1",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 80)
    print(f"最终选择模型: {selected_model}")
    print(f"验证集确定阈值: {selected_threshold:.8f}")
    print(selection_reason)
    print("锁定测试结果:")
    print(
        test_results.loc[
            test_results["model"] == selected_model,
            [
                "roc_auc",
                "roc_auc_ci_low",
                "roc_auc_ci_high",
                "accuracy",
                "accuracy_ci_low",
                "accuracy_ci_high",
                "sensitivity",
                "sensitivity_ci_low",
                "sensitivity_ci_high",
                "specificity",
                "specificity_ci_low",
                "specificity_ci_high",
                "f1",
                "f1_ci_low",
                "f1_ci_high",
            ],
        ].to_string(index=False)
    )

    print("\n输出文件:")
    for relative in (
        "reports/stage1_all_models_validation_ci.csv",
        "reports/stage1_all_models_locked_test_ci.csv",
        "reports/stage1_validation_selection_ranking.csv",
        "reports/stage1_selected_vs_all_paired_auc.csv",
        "reports/stage1_selected_model.json",
        "artifacts/val_predictions_stage1_selected.csv",
        "artifacts/predictions_stage1_selected.csv",
        "artifacts/stage1_selected_test_routing_to_stage2.csv",
        "reports/figures/stage1_all_models_roc_auc_ci.png",
        "reports/figures/stage1_all_models_roc_auc_ci.pdf",
    ):
        print(f"  {root / relative}")


if __name__ == "__main__":
    main()
