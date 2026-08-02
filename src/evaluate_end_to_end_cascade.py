r"""
Evaluate the selected two-stage cascade on all validation/test images.

Stage 1:
    Normal (0) vs pneumonia (1), using the already selected Stage-1
    prediction files and the validation-selected Stage-1 threshold.

Stage 2:
    Bacterial (0) vs viral (1), using the validation-selected equal-weight
    ensemble. Stage-2 inference is run on EVERY image routed by Stage 1,
    including normal false positives. This is essential for a valid end-to-end
    three-class evaluation.

Final classes:
    0 = normal
    1 = bacterial pneumonia
    2 = viral pneumonia

Recommended path:
    <repository>\src\evaluate_end_to_end_cascade.py

Run:
    python -m src.evaluate_end_to_end_cascade ^
        --project-root "<repository>" ^
        --device cuda ^
        --batch-size 8 ^
        --reps 2000
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)

from src.pipeline import loader, make_model, predict_torch


CLASS_NAMES = ("normal", "bacterial", "viral")
CLASS_TO_ID = {
    "normal": 0,
    "bacterial": 1,
    "viral": 2,
}
SEED = 20260722


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"缺少JSON文件: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_manifest(manifest: pd.DataFrame) -> None:
    required = {
        "path",
        "patient",
        "split",
        "stage1",
        "subtype",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"manifest.csv缺少字段: {missing}")

    if manifest["path"].duplicated().any():
        examples = manifest.loc[
            manifest["path"].duplicated(), "path"
        ].head().tolist()
        raise ValueError(
            f"manifest.csv存在重复path，例如: {examples}"
        )

    if manifest["patient"].isna().any():
        raise ValueError("manifest.csv存在缺失filename group")

    unexpected = sorted(
        set(manifest["subtype"].astype(str))
        - set(CLASS_NAMES)
    )
    if unexpected:
        raise ValueError(
            f"manifest.csv包含未知subtype: {unexpected}"
        )

    group_split_count = manifest.groupby(
        "patient"
    )["split"].nunique()
    if (group_split_count > 1).any():
        examples = group_split_count[
            group_split_count > 1
        ].head().index.tolist()
        raise ValueError(
            "同一filename group跨越train/val/test，例如: "
            f"{examples}"
        )


def read_stage1_selected_predictions(
    root: Path,
    split: str,
    expected: pd.DataFrame,
) -> pd.DataFrame:
    filename = (
        "val_predictions_stage1_selected.csv"
        if split == "val"
        else "predictions_stage1_selected.csv"
    )
    path = root / "artifacts" / filename
    if not path.exists():
        raise FileNotFoundError(
            f"缺少Stage-1 selected预测: {path}"
        )

    frame = pd.read_csv(path)
    required = {"path", "probability"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"{path.name}缺少字段: {missing}"
        )

    frame = frame[["path", "probability"]].copy()
    frame["path"] = frame["path"].astype(str)
    frame["probability"] = pd.to_numeric(
        frame["probability"],
        errors="coerce",
    )

    if frame["path"].duplicated().any():
        raise ValueError(
            f"{path.name}存在重复path"
        )
    if frame["probability"].isna().any():
        raise ValueError(
            f"{path.name}存在非法概率"
        )
    if not frame["probability"].between(0.0, 1.0).all():
        raise ValueError(
            f"{path.name}概率超出[0,1]"
        )

    aligned = expected[
        ["path"]
    ].merge(
        frame,
        on="path",
        how="left",
        validate="one_to_one",
    )

    if aligned["probability"].isna().any():
        missing_count = int(
            aligned["probability"].isna().sum()
        )
        raise ValueError(
            f"{path.name}缺少{missing_count}张图像的预测"
        )

    extra = set(frame["path"]) - set(expected["path"])
    if extra:
        raise ValueError(
            f"{path.name}包含不属于{split}集的预测，例如: "
            f"{sorted(extra)[:5]}"
        )

    return aligned


def load_checkpoint_state(path: Path) -> Mapping[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"缺少checkpoint: {path}")

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )
    if isinstance(checkpoint, dict) and "state" in checkpoint:
        return checkpoint["state"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError(
        f"无法识别checkpoint格式: {path}"
    )


@torch.inference_mode()
def predict_stage2_member(
    root: Path,
    member: str,
    routed_frame: pd.DataFrame,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    checkpoint_path = (
        root
        / "models"
        / "stage2"
        / f"{member}.pt"
    )
    state = load_checkpoint_state(checkpoint_path)

    model = make_model(member).to(device)
    model.load_state_dict(state)
    model.eval()

    data_loader = loader(
        frame=routed_frame,
        task="stage2",
        train=False,
        batch=batch_size,
        workers=0 if os.name == "nt" else 4,
    )

    _, probability, paths, elapsed = predict_torch(
        model,
        data_loader,
        device,
    )

    output = pd.DataFrame(
        {
            "path": [str(path) for path in paths],
            f"probability_{member}": probability.astype(
                np.float64
            ),
        }
    )

    if output["path"].duplicated().any():
        raise ValueError(
            f"{member}路由预测存在重复path"
        )

    print(
        f"[STAGE2_MEMBER_DONE] {member}: "
        f"n={len(output)}, seconds={elapsed:.3f}",
        flush=True,
    )

    del model, data_loader, state
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return output


def predict_stage2_ensemble_for_routed(
    root: Path,
    routed_frame: pd.DataFrame,
    members: Sequence[str],
    weights: Mapping[str, float],
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    if len(routed_frame) == 0:
        return pd.DataFrame(
            {
                "path": pd.Series(dtype="string"),
                "stage2_probability_viral": pd.Series(
                    dtype=float
                ),
            }
        )

    merged = routed_frame[["path"]].copy()
    merged["path"] = merged["path"].astype(str)

    for member in members:
        prediction = predict_stage2_member(
            root=root,
            member=member,
            routed_frame=routed_frame,
            device=device,
            batch_size=batch_size,
        )
        merged = merged.merge(
            prediction,
            on="path",
            how="left",
            validate="one_to_one",
        )

    probability = np.zeros(
        len(merged),
        dtype=np.float64,
    )
    weight_sum = 0.0

    for member in members:
        weight = float(weights.get(member, 0.0))
        if weight <= 0:
            raise ValueError(
                f"Stage-2成员{member}的权重无效: {weight}"
            )
        column = f"probability_{member}"
        if merged[column].isna().any():
            raise ValueError(
                f"Stage-2成员{member}存在缺失路由预测"
            )
        probability += (
            weight
            * merged[column].to_numpy(
                dtype=np.float64
            )
        )
        weight_sum += weight

    if not np.isclose(weight_sum, 1.0, atol=1e-6):
        probability /= weight_sum

    merged["stage2_probability_viral"] = probability
    return merged


def build_cascade_predictions(
    root: Path,
    split: str,
    manifest: pd.DataFrame,
    stage1_threshold: float,
    stage2_threshold: float,
    members: Sequence[str],
    weights: Mapping[str, float],
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    frame = manifest[
        manifest["split"].astype(str) == split
    ].copy().reset_index(drop=True)

    stage1 = read_stage1_selected_predictions(
        root,
        split,
        frame,
    ).rename(
        columns={
            "probability": "stage1_probability_pneumonia"
        }
    )

    frame = frame.merge(
        stage1,
        on="path",
        how="left",
        validate="one_to_one",
    )
    frame["stage1_route_to_stage2"] = (
        frame["stage1_probability_pneumonia"]
        >= stage1_threshold
    )

    routed = frame[
        frame["stage1_route_to_stage2"]
    ].copy().reset_index(drop=True)

    print(
        f"\n[{split.upper()}] total={len(frame)}, "
        f"routed_to_stage2={len(routed)}, "
        f"not_routed={len(frame)-len(routed)}",
        flush=True,
    )

    routed_prediction = (
        predict_stage2_ensemble_for_routed(
            root=root,
            routed_frame=routed,
            members=members,
            weights=weights,
            device=device,
            batch_size=batch_size,
        )
    )

    frame = frame.merge(
        routed_prediction[
            ["path", "stage2_probability_viral"]
        ],
        on="path",
        how="left",
        validate="one_to_one",
    )

    # Non-routed images have no Stage-2 result by design.
    predicted_class = np.zeros(
        len(frame),
        dtype=np.int8,
    )
    routed_mask = frame[
        "stage1_route_to_stage2"
    ].to_numpy(dtype=bool)

    routed_viral = (
        frame.loc[
            routed_mask,
            "stage2_probability_viral",
        ].to_numpy(dtype=np.float64)
        >= stage2_threshold
    )
    predicted_class[routed_mask] = np.where(
        routed_viral,
        CLASS_TO_ID["viral"],
        CLASS_TO_ID["bacterial"],
    )

    frame["y_true_3class"] = (
        frame["subtype"]
        .astype(str)
        .map(CLASS_TO_ID)
        .astype(np.int8)
    )
    frame["y_pred_3class"] = predicted_class
    frame["predicted_subtype"] = [
        CLASS_NAMES[int(value)]
        for value in predicted_class
    ]

    true = frame["y_true_3class"].to_numpy()
    pred = frame["y_pred_3class"].to_numpy()

    error_type = np.full(
        len(frame),
        "correct",
        dtype=object,
    )

    stage1_false_positive = (
        (true == CLASS_TO_ID["normal"])
        & routed_mask
    )
    stage1_false_negative = (
        (true != CLASS_TO_ID["normal"])
        & (~routed_mask)
    )
    stage2_subtype_error = (
        (true != CLASS_TO_ID["normal"])
        & routed_mask
        & (pred != true)
    )

    error_type[stage1_false_positive] = (
        "stage1_false_positive_normal_routed"
    )
    error_type[stage1_false_negative] = (
        "stage1_false_negative_pneumonia_stopped"
    )
    error_type[stage2_subtype_error] = (
        "stage2_subtype_error_after_correct_route"
    )
    frame["cascade_error_type"] = error_type

    frame["stage1_threshold"] = float(
        stage1_threshold
    )
    frame["stage2_threshold"] = float(
        stage2_threshold
    )

    return frame


def metric_point(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.asarray(y_pred, dtype=np.int8)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2],
    )

    precision, recall, class_f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            zero_division=0,
        )
    )

    total = float(matrix.sum())
    rows: Dict[str, float] = {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "macro_recall_balanced_accuracy": float(
            recall_score(
                y_true,
                y_pred,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=[0, 1, 2],
                average="weighted",
                zero_division=0,
            )
        ),
    }

    for class_id, name in enumerate(CLASS_NAMES):
        tp = float(matrix[class_id, class_id])
        fn = float(matrix[class_id, :].sum() - tp)
        fp = float(matrix[:, class_id].sum() - tp)
        tn = total - tp - fn - fp

        rows[f"{name}_precision"] = float(
            precision[class_id]
        )
        rows[f"{name}_recall_sensitivity"] = float(
            recall[class_id]
        )
        rows[f"{name}_specificity"] = float(
            tn / (tn + fp)
            if (tn + fp) > 0
            else np.nan
        )
        rows[f"{name}_f1"] = float(
            class_f1[class_id]
        )
        rows[f"{name}_support"] = float(
            support[class_id]
        )

    return rows


def group_bootstrap_indices(
    groups: Sequence[str],
    reps: int,
    seed: int,
) -> List[np.ndarray]:
    group_series = pd.Series(
        groups,
        dtype="string",
    )
    if group_series.isna().any():
        raise ValueError(
            "bootstrap filename group存在缺失值"
        )

    unique_groups = sorted(
        group_series.unique().tolist()
    )
    group_to_rows = {
        group: np.flatnonzero(
            group_series.to_numpy() == group
        )
        for group in unique_groups
    }

    rng = np.random.default_rng(seed)
    bootstrap: List[np.ndarray] = []

    for _ in range(reps):
        sampled = rng.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True,
        )
        index = np.concatenate(
            [group_to_rows[group] for group in sampled]
        )
        bootstrap.append(index.astype(np.int64))

    return bootstrap


def metric_intervals(
    frame: pd.DataFrame,
    reps: int,
    seed: int,
) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
    y_true = frame[
        "y_true_3class"
    ].to_numpy(dtype=np.int8)
    y_pred = frame[
        "y_pred_3class"
    ].to_numpy(dtype=np.int8)

    point = metric_point(y_true, y_pred)
    bootstrap = group_bootstrap_indices(
        frame["patient"].astype(str).tolist(),
        reps=reps,
        seed=seed,
    )

    values: Dict[str, List[float]] = {
        key: []
        for key in point
        if not key.endswith("_support")
    }

    for index in bootstrap:
        y_boot = y_true[index]
        # Skip draws lacking any final class; this stabilizes macro metrics.
        if len(np.unique(y_boot)) < 3:
            continue

        boot_point = metric_point(
            y_boot,
            y_pred[index],
        )
        for key in values:
            value = boot_point[key]
            if np.isfinite(value):
                values[key].append(float(value))

    intervals: Dict[str, Tuple[float, float]] = {}
    for key, collected in values.items():
        if len(collected) == 0:
            intervals[key] = (
                float("nan"),
                float("nan"),
            )
        else:
            low, high = np.quantile(
                np.asarray(collected),
                [0.025, 0.975],
            )
            intervals[key] = (
                float(low),
                float(high),
            )

    return point, intervals


def make_metric_table(
    split: str,
    frame: pd.DataFrame,
    point: Mapping[str, float],
    intervals: Mapping[
        str,
        Tuple[float, float],
    ],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for metric, value in point.items():
        row: Dict[str, object] = {
            "split": split,
            "metric": metric,
            "value": float(value),
        }
        if metric in intervals:
            row["ci95_low"] = intervals[metric][0]
            row["ci95_high"] = intervals[metric][1]
        else:
            row["ci95_low"] = np.nan
            row["ci95_high"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def confusion_table(
    split: str,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    matrix = confusion_matrix(
        frame["y_true_3class"],
        frame["y_pred_3class"],
        labels=[0, 1, 2],
    )

    rows: List[Dict[str, object]] = []
    for true_id, true_name in enumerate(CLASS_NAMES):
        for pred_id, pred_name in enumerate(CLASS_NAMES):
            rows.append(
                {
                    "split": split,
                    "true_class": true_name,
                    "predicted_class": pred_name,
                    "count": int(
                        matrix[true_id, pred_id]
                    ),
                }
            )
    return pd.DataFrame(rows)


def error_summary(
    split: str,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    counts = (
        frame["cascade_error_type"]
        .value_counts(dropna=False)
        .rename_axis("error_type")
        .reset_index(name="count")
    )
    counts.insert(0, "split", split)
    counts["fraction"] = (
        counts["count"] / len(frame)
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    reports = root / "reports"
    artifacts = root / "artifacts"
    reports.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    manifest_path = (
        root / "data" / "processed" / "manifest.csv"
    )
    manifest = pd.read_csv(manifest_path)
    validate_manifest(manifest)

    stage1_info = load_json(
        reports / "stage1_selected_model.json"
    )
    stage2_info = load_json(
        reports
        / "stage2_equal_weight_ensemble_selected.json"
    )

    stage1_model = str(
        stage1_info["selected_model"]
    )
    stage1_threshold = float(
        stage1_info["selected_threshold"]
    )

    members = [
        str(member)
        for member in stage2_info["members"]
    ]
    weights = {
        str(key): float(value)
        for key, value in stage2_info["weights"].items()
    }
    stage2_threshold = float(
        stage2_info["selected_threshold"]
    )

    print("=" * 96)
    print("END-TO-END CASCADE CONFIGURATION")
    print(
        f"Stage 1 selected model: {stage1_model}"
    )
    print(
        f"Stage 1 threshold: {stage1_threshold:.8f}"
    )
    print(
        "Stage 2 equal-weight members: "
        + ", ".join(members)
    )
    print(
        f"Stage 2 threshold: {stage2_threshold:.8f}"
    )
    print("=" * 96)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "指定了cuda，但PyTorch检测不到CUDA"
        )

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = False

    all_metrics: List[pd.DataFrame] = []
    all_confusions: List[pd.DataFrame] = []
    all_errors: List[pd.DataFrame] = []
    split_summaries: Dict[str, object] = {}

    for split, seed_offset in (
        ("val", 0),
        ("test", 1),
    ):
        start = time.perf_counter()
        output = build_cascade_predictions(
            root=root,
            split=split,
            manifest=manifest,
            stage1_threshold=stage1_threshold,
            stage2_threshold=stage2_threshold,
            members=members,
            weights=weights,
            device=device,
            batch_size=args.batch_size,
        )
        elapsed = time.perf_counter() - start

        point, intervals = metric_intervals(
            output,
            reps=args.reps,
            seed=args.seed + seed_offset,
        )

        metric_table = make_metric_table(
            split,
            output,
            point,
            intervals,
        )
        confusion = confusion_table(
            split,
            output,
        )
        errors = error_summary(
            split,
            output,
        )

        prediction_name = (
            "val_predictions_cascade_selected.csv"
            if split == "val"
            else "predictions_cascade_selected.csv"
        )
        output.to_csv(
            artifacts / prediction_name,
            index=False,
        )

        all_metrics.append(metric_table)
        all_confusions.append(confusion)
        all_errors.append(errors)

        split_summaries[split] = {
            "n_images": int(len(output)),
            "n_filename_groups": int(
                output["patient"].nunique()
            ),
            "n_routed_to_stage2": int(
                output[
                    "stage1_route_to_stage2"
                ].sum()
            ),
            "inference_seconds": float(elapsed),
            "point_metrics": {
                key: float(value)
                for key, value in point.items()
            },
            "ci95": {
                key: {
                    "low": float(value[0]),
                    "high": float(value[1]),
                }
                for key, value in intervals.items()
            },
            "confusion_matrix": (
                confusion.pivot(
                    index="true_class",
                    columns="predicted_class",
                    values="count",
                )
                .reindex(
                    index=CLASS_NAMES,
                    columns=CLASS_NAMES,
                )
                .fillna(0)
                .astype(int)
                .to_dict()
            ),
            "error_counts": {
                str(row.error_type): int(row.count)
                for row in errors.itertuples()
            },
        }

        print("\n" + "=" * 96)
        print(
            f"{split.upper()} CASCADE RESULT"
        )
        print(
            f"Accuracy={point['accuracy']:.6f}"
        )
        print(
            "Macro balanced accuracy="
            f"{point['macro_recall_balanced_accuracy']:.6f}"
        )
        print(
            f"Macro F1={point['macro_f1']:.6f}"
        )
        print(
            "Normal recall="
            f"{point['normal_recall_sensitivity']:.6f}"
        )
        print(
            "Bacterial recall="
            f"{point['bacterial_recall_sensitivity']:.6f}"
        )
        print(
            "Viral recall="
            f"{point['viral_recall_sensitivity']:.6f}"
        )
        print("\nConfusion matrix:")
        print(
            confusion.pivot(
                index="true_class",
                columns="predicted_class",
                values="count",
            )
            .reindex(
                index=CLASS_NAMES,
                columns=CLASS_NAMES,
            )
            .fillna(0)
            .astype(int)
            .to_string()
        )
        print("\nError decomposition:")
        print(errors.to_string(index=False))

    metrics_output = pd.concat(
        all_metrics,
        ignore_index=True,
    )
    confusions_output = pd.concat(
        all_confusions,
        ignore_index=True,
    )
    errors_output = pd.concat(
        all_errors,
        ignore_index=True,
    )

    metrics_output.to_csv(
        reports / "cascade_selected_metrics_ci.csv",
        index=False,
    )
    confusions_output.to_csv(
        reports
        / "cascade_selected_confusion_matrices.csv",
        index=False,
    )
    errors_output.to_csv(
        reports
        / "cascade_selected_error_decomposition.csv",
        index=False,
    )

    summary = {
        "status": "complete",
        "final_classes": list(CLASS_NAMES),
        "stage1": {
            "selected_model": stage1_model,
            "selected_threshold": stage1_threshold,
            "selection_source": (
                "reports/stage1_selected_model.json"
            ),
        },
        "stage2": {
            "selected_ensemble": stage2_info.get(
                "selected_ensemble"
            ),
            "members": members,
            "weights": weights,
            "selected_threshold": stage2_threshold,
            "selection_source": (
                "reports/"
                "stage2_equal_weight_ensemble_selected.json"
            ),
        },
        "bootstrap": {
            "unit": (
                "namespace-specific filename group"
            ),
            "repetitions": args.reps,
            "confidence_interval": "percentile 95%",
            "seed_validation": args.seed,
            "seed_test": args.seed + 1,
        },
        "splits": split_summaries,
        "interpretation_warning": (
            "The current test split has already been inspected during "
            "earlier model development. Treat the end-to-end test estimate "
            "as exploratory until confirmed on an external test set or "
            "nested group cross-validation."
        ),
        "files": {
            "validation_predictions": (
                "artifacts/"
                "val_predictions_cascade_selected.csv"
            ),
            "test_predictions": (
                "artifacts/"
                "predictions_cascade_selected.csv"
            ),
            "metrics": (
                "reports/"
                "cascade_selected_metrics_ci.csv"
            ),
            "confusion_matrices": (
                "reports/"
                "cascade_selected_confusion_matrices.csv"
            ),
            "error_decomposition": (
                "reports/"
                "cascade_selected_error_decomposition.csv"
            ),
        },
    }

    (
        reports / "cascade_selected_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 96)
    print("END-TO-END CASCADE EVALUATION COMPLETE")
    print(
        reports / "cascade_selected_summary.json"
    )
    print(
        reports / "cascade_selected_metrics_ci.csv"
    )
    print(
        reports
        / "cascade_selected_confusion_matrices.csv"
    )
    print(
        reports
        / "cascade_selected_error_decomposition.csv"
    )


if __name__ == "__main__":
    main()
