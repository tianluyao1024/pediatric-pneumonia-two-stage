"""Nested group-disjoint whole-image diagnostic for Stage 2.

This script fixes two issues in the first whole-image OOF experiment:

1. It uses a dedicated whole-image classifier instead of the generic
   multi-region fusion head.
2. The outer held-out fold is never used for checkpoint/epoch selection.
   Epoch selection occurs on an inner StratifiedGroupKFold split. The model is
   then reinitialized and retrained on the full outer-training set for the
   selected number of epochs before one-time outer-fold evaluation.

Supported initialization:
    ssl       Fold-specific pediatric SimSiam encoder.
    imagenet  TorchVision ImageNet EfficientNet-B0.
    random    Random EfficientNet-B0.

Outputs:
    artifacts/oof_predictions_stage2_<run>.csv
    artifacts/predictions_stage2_<run>.csv
    reports/stage2_<run>_fold_metrics.csv
    reports/stage2_<run>_summary.json
    models/stage2_nested/<run>_fold*.pt

The current test set has already been inspected in earlier development, so its
metrics remain exploratory.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile, ImageOps
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GroupKFold,
    StratifiedGroupKFold,
)
from torch.utils.data import (
    DataLoader,
    Dataset,
    WeightedRandomSampler,
)
from torchvision import models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True
SEED = 20260729


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


class WholeTransform:
    def __init__(
        self,
        image_size: int,
        train: bool,
    ) -> None:
        operations: List[object] = []

        if train:
            operations.extend(
                [
                    transforms.RandomResizedCrop(
                        image_size,
                        scale=(0.78, 1.00),
                        ratio=(0.88, 1.12),
                        antialias=True,
                    ),
                    transforms.RandomRotation(
                        5,
                        fill=0,
                    ),
                    transforms.RandomAffine(
                        degrees=0,
                        translate=(0.025, 0.025),
                        scale=(0.97, 1.03),
                        fill=0,
                    ),
                    transforms.ColorJitter(
                        brightness=0.10,
                        contrast=0.15,
                    ),
                ]
            )
        else:
            operations.append(
                transforms.Resize(
                    (image_size, image_size),
                    antialias=True,
                )
            )

        operations.extend(
            [
                transforms.Grayscale(
                    num_output_channels=3
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )
        self.transform = transforms.Compose(
            operations
        )

    def __call__(
        self,
        image: Image.Image,
    ) -> torch.Tensor:
        return self.transform(image)


class WholeDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_size: int,
        train: bool,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = WholeTransform(
            image_size,
            train,
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, float, str]:
        row = self.frame.iloc[index]
        path = str(row["path"])

        with Image.open(path) as image:
            gray = image.convert("L")
            background = int(
                np.median(
                    np.asarray(
                        gray.resize(
                            (64, 64),
                            Image.Resampling.BILINEAR,
                        ),
                        dtype=np.uint8,
                    )
                )
            )
            square = ImageOps.pad(
                gray,
                (512, 512),
                method=Image.Resampling.BILINEAR,
                color=background,
                centering=(0.5, 0.5),
            )

        return (
            self.transform(square),
            float(row["y"]),
            path,
        )


class EfficientNetEncoder(nn.Module):
    def __init__(
        self,
        imagenet: bool,
    ) -> None:
        super().__init__()
        weights = (
            models.EfficientNet_B0_Weights.DEFAULT
            if imagenet
            else None
        )
        backbone = models.efficientnet_b0(
            weights=weights
        )
        self.features = backbone.features
        self.pool = backbone.avgpool
        self.dim = 1280

    def forward(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        return torch.flatten(
            self.pool(self.features(image)),
            1,
        )


class WholeClassifier(nn.Module):
    def __init__(
        self,
        initialization: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = EfficientNetEncoder(
            imagenet=(
                initialization == "imagenet"
            )
        )
        self.head = nn.Sequential(
            nn.LayerNorm(
                self.encoder.dim
            ),
            nn.Dropout(dropout),
            nn.Linear(
                self.encoder.dim,
                1,
            ),
        )

    def forward(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(
            self.encoder(image)
        ).reshape(-1)


def load_fold_ssl(
    model: WholeClassifier,
    checkpoint: Path,
) -> None:
    saved = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    state = saved.get(
        "encoder",
        saved,
    )
    missing, unexpected = (
        model.encoder.load_state_dict(
            state,
            strict=False,
        )
    )
    if missing or unexpected:
        raise RuntimeError(
            "SSL encoder mismatch: "
            f"missing={missing}, "
            f"unexpected={unexpected}"
        )


def build_model(
    initialization: str,
    dropout: float,
    ssl_checkpoint: Path | None,
    seed: int,
) -> WholeClassifier:
    seed_all(seed)
    model = WholeClassifier(
        initialization=initialization,
        dropout=dropout,
    )
    if initialization == "ssl":
        if (
            ssl_checkpoint is None
            or not ssl_checkpoint.exists()
        ):
            raise FileNotFoundError(
                f"Missing SSL checkpoint: "
                f"{ssl_checkpoint}"
            )
        load_fold_ssl(
            model,
            ssl_checkpoint,
        )
    return model


def make_loader(
    frame: pd.DataFrame,
    image_size: int,
    batch_size: int,
    workers: int,
    train: bool,
) -> DataLoader:
    dataset = WholeDataset(
        frame,
        image_size=image_size,
        train=train,
    )

    if train:
        labels = frame[
            "y"
        ].to_numpy(dtype=np.int64)
        counts = np.bincount(
            labels,
            minlength=2,
        )
        weights = (
            1.0 / np.maximum(counts, 1)
        )[labels]
        sampler = WeightedRandomSampler(
            torch.as_tensor(
                weights,
                dtype=torch.double,
            ),
            num_samples=len(frame),
            replacement=True,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=workers > 0,
            drop_last=True,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=False,
    )


def make_optimizer(
    model: WholeClassifier,
    encoder_lr: float,
    head_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [
            {
                "params": (
                    model.encoder.parameters()
                ),
                "lr": encoder_lr,
            },
            {
                "params": (
                    model.head.parameters()
                ),
                "lr": head_lr,
            },
        ],
        weight_decay=weight_decay,
        foreach=False,
        fused=False,
    )


def train_epoch(
    model: WholeClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    label_smoothing: float,
) -> float:
    model.train()
    losses: List[float] = []

    for images, labels, _ in loader:
        images = images.to(
            device,
            non_blocking=True,
        )
        labels = labels.float().to(
            device,
            non_blocking=True,
        )
        targets = (
            labels * (1.0 - label_smoothing)
            + 0.5 * label_smoothing
        )

        optimizer.zero_grad(
            set_to_none=True
        )
        logits = model(images)
        loss = (
            nn.functional
            .binary_cross_entropy_with_logits(
                logits,
                targets,
            )
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite training loss"
            )

        loss.backward()
        nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0,
        )
        optimizer.step()

        losses.append(
            float(
                loss.detach().cpu()
            )
        )

    return float(
        np.mean(losses)
    )


@torch.inference_mode()
def predict(
    model: WholeClassifier,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    model.eval()
    labels: List[int] = []
    probabilities: List[float] = []
    paths: List[str] = []

    for images, batch_labels, batch_paths in loader:
        images = images.to(
            device,
            non_blocking=True,
        )
        probability = torch.sigmoid(
            model(images)
        ).cpu().numpy()

        labels.extend(
            batch_labels.numpy().astype(int)
        )
        probabilities.extend(
            probability.tolist()
        )
        paths.extend(batch_paths)

    return (
        np.asarray(
            labels,
            dtype=np.int8,
        ),
        np.asarray(
            probabilities,
            dtype=np.float64,
        ),
        paths,
    )


def optimal_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    false_positive, true_positive, thresholds = (
        roc_curve(
            labels,
            probabilities,
        )
    )
    index = int(
        np.argmax(
            true_positive - false_positive
        )
    )
    return float(
        thresholds[index]
    )


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    predicted = (
        probabilities >= threshold
    ).astype(np.int8)
    return {
        "roc_auc": float(
            roc_auc_score(
                labels,
                probabilities,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                labels,
                probabilities,
            )
        ),
        "accuracy": float(
            accuracy_score(
                labels,
                predicted,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                labels,
                predicted,
            )
        ),
        "f1": float(
            f1_score(
                labels,
                predicted,
            )
        ),
        "threshold": float(threshold),
    }


def inner_split(
    frame: pd.DataFrame,
    seed: int,
    inner_folds: int,
) -> Tuple[np.ndarray, np.ndarray]:
    labels = frame[
        "y"
    ].to_numpy(dtype=np.int8)
    groups = frame[
        "patient"
    ].astype(str).to_numpy()

    try:
        splitter = StratifiedGroupKFold(
            n_splits=inner_folds,
            shuffle=True,
            random_state=seed,
        )
        train_index, validation_index = next(
            splitter.split(
                frame,
                labels,
                groups,
            )
        )
    except Exception:
        splitter = GroupKFold(
            n_splits=inner_folds
        )
        train_index, validation_index = next(
            splitter.split(
                frame,
                labels,
                groups,
            )
        )

    overlap = (
        set(
            frame.iloc[
                train_index
            ]["patient"].astype(str)
        )
        & set(
            frame.iloc[
                validation_index
            ]["patient"].astype(str)
        )
    )
    if overlap:
        raise RuntimeError(
            "Inner group leakage"
        )

    return (
        train_index,
        validation_index,
    )


def select_epoch(
    outer_train: pd.DataFrame,
    initialization: str,
    ssl_checkpoint: Path | None,
    device: torch.device,
    args,
    fold: int,
) -> Tuple[int, List[Dict[str, float]]]:
    train_index, validation_index = inner_split(
        outer_train,
        seed=args.seed + fold * 101,
        inner_folds=args.inner_folds,
    )
    inner_train = outer_train.iloc[
        train_index
    ].reset_index(drop=True)
    inner_validation = outer_train.iloc[
        validation_index
    ].reset_index(drop=True)

    model_seed = (
        args.seed + fold * 1000 + 11
    )
    model = build_model(
        initialization,
        args.dropout,
        ssl_checkpoint,
        model_seed,
    ).to(device)

    train_loader = make_loader(
        inner_train,
        args.image_size,
        args.batch_size,
        args.workers,
        train=True,
    )
    validation_loader = make_loader(
        inner_validation,
        args.image_size,
        args.batch_size,
        args.workers,
        train=False,
    )

    optimizer = make_optimizer(
        model,
        args.encoder_lr,
        args.head_lr,
        args.weight_decay,
    )
    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=args.max_epochs,
            eta_min=(
                args.encoder_lr * 0.05
            ),
        )
    )

    best_auc = -math.inf
    best_epoch = 1
    bad_epochs = 0
    history: List[
        Dict[str, float]
    ] = []

    for epoch in range(
        1,
        args.max_epochs + 1,
    ):
        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.label_smoothing,
        )
        labels, probabilities, _ = predict(
            model,
            validation_loader,
            device,
        )
        auc = float(
            roc_auc_score(
                labels,
                probabilities,
            )
        )
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "inner_loss": loss,
                "inner_auc": auc,
            }
        )
        print(
            f"[INNER] fold={fold} "
            f"epoch={epoch:02d} "
            f"loss={loss:.6f} "
            f"auc={auc:.6f}",
            flush=True,
        )

        if auc > best_auc + args.min_delta:
            best_auc = auc
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            break

    del (
        model,
        optimizer,
        scheduler,
        train_loader,
        validation_loader,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return (
        int(best_epoch),
        history,
    )


def fit_outer_model(
    outer_train: pd.DataFrame,
    initialization: str,
    ssl_checkpoint: Path | None,
    selected_epochs: int,
    device: torch.device,
    args,
    fold: int,
) -> WholeClassifier:
    model_seed = (
        args.seed + fold * 1000 + 11
    )
    model = build_model(
        initialization,
        args.dropout,
        ssl_checkpoint,
        model_seed,
    ).to(device)

    train_loader = make_loader(
        outer_train,
        args.image_size,
        args.batch_size,
        args.workers,
        train=True,
    )
    optimizer = make_optimizer(
        model,
        args.encoder_lr,
        args.head_lr,
        args.weight_decay,
    )
    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=args.max_epochs,
            eta_min=(
                args.encoder_lr * 0.05
            ),
        )
    )

    for epoch in range(
        1,
        selected_epochs + 1,
    ):
        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.label_smoothing,
        )
        scheduler.step()
        print(
            f"[OUTER_TRAIN] fold={fold} "
            f"epoch={epoch:02d}/"
            f"{selected_epochs:02d} "
            f"loss={loss:.6f}",
            flush=True,
        )

    del train_loader, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--initialization",
        choices=(
            "ssl",
            "imagenet",
            "random",
        ),
        required=True,
    )
    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--inner-folds",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
    )
    parser.add_argument(
        "--encoder-lr",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--head-lr",
        type=float,
        default=5e-4,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )
    parser.add_argument(
        "--device",
        choices=(
            "cuda",
            "cpu",
        ),
        default="cuda",
    )
    parser.add_argument(
        "--fold-assignment-csv",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--ssl-template",
        type=str,
        default="",
    )
    args = parser.parse_args()

    seed_all(args.seed)

    root = args.project_root.resolve()
    reports = root / "reports"
    artifacts = root / "artifacts"
    model_dir = (
        root / "models" / "stage2_nested"
    )

    for directory in (
        reports,
        artifacts,
        model_dir,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    manifest = pd.read_csv(
        root
        / "data"
        / "processed"
        / "manifest.csv"
    )
    development = manifest[
        (manifest["stage1"] == 1)
        & manifest["split"].isin(
            ["train", "val"]
        )
    ].copy().reset_index(drop=True)
    test = manifest[
        (manifest["stage1"] == 1)
        & (manifest["split"] == "test")
    ].copy().reset_index(drop=True)

    development["y"] = (
        development["subtype"].astype(str)
        == "viral"
    ).astype(np.int8)
    test["y"] = (
        test["subtype"].astype(str)
        == "viral"
    ).astype(np.int8)

    assignment_path = (
        args.fold_assignment_csv
        or reports
        / "stage2_groupfold_assignments.csv"
    )
    assignment = pd.read_csv(
        assignment_path
    )
    aligned = development[
        ["path"]
    ].merge(
        assignment[
            ["path", "fold"]
        ],
        on="path",
        how="left",
        validate="one_to_one",
    )
    if aligned["fold"].isna().any():
        raise ValueError(
            "Fold assignment incomplete"
        )
    fold_values = aligned[
        "fold"
    ].astype(int).to_numpy()

    ssl_template = (
        args.ssl_template
        or str(
            root
            / "models"
            / "ssl_groupfold"
            / (
                "pediatric_simsiam_"
                "efficientnet_b0_"
                "fold{fold}_best.pt"
            )
        )
    )

    device = torch.device(
        args.device
        if args.device == "cpu"
        or torch.cuda.is_available()
        else "cpu"
    )

    oof = np.full(
        len(development),
        np.nan,
        dtype=np.float64,
    )
    test_probabilities = []
    fold_rows = []
    history_rows = []

    for fold in range(
        1,
        args.folds + 1,
    ):
        outer_validation_index = np.flatnonzero(
            fold_values == fold
        )
        outer_train_index = np.flatnonzero(
            fold_values != fold
        )

        outer_train = development.iloc[
            outer_train_index
        ].reset_index(drop=True)
        outer_validation = development.iloc[
            outer_validation_index
        ].reset_index(drop=True)

        overlap = (
            set(
                outer_train["patient"].astype(str)
            )
            & set(
                outer_validation[
                    "patient"
                ].astype(str)
            )
        )
        if overlap:
            raise RuntimeError(
                f"Outer fold {fold} group leakage"
            )

        ssl_checkpoint = (
            Path(
                ssl_template.format(
                    fold=fold
                )
            )
            if args.initialization == "ssl"
            else None
        )

        selected_epochs, history = (
            select_epoch(
                outer_train,
                args.initialization,
                ssl_checkpoint,
                device,
                args,
                fold,
            )
        )

        for item in history:
            history_rows.append(
                {
                    "fold": fold,
                    **item,
                }
            )

        model = fit_outer_model(
            outer_train,
            args.initialization,
            ssl_checkpoint,
            selected_epochs,
            device,
            args,
            fold,
        )

        outer_validation_loader = make_loader(
            outer_validation,
            args.image_size,
            args.batch_size,
            args.workers,
            train=False,
        )
        test_loader = make_loader(
            test,
            args.image_size,
            args.batch_size,
            args.workers,
            train=False,
        )

        labels, probabilities, paths = predict(
            model,
            outer_validation_loader,
            device,
        )
        test_labels, test_probability, _ = predict(
            model,
            test_loader,
            device,
        )

        fold_auc = float(
            roc_auc_score(
                labels,
                probabilities,
            )
        )
        oof[
            outer_validation_index
        ] = probabilities
        test_probabilities.append(
            test_probability
        )

        checkpoint_path = (
            model_dir
            / (
                f"{args.run_name}_"
                f"fold{fold}.pt"
            )
        )
        torch.save(
            {
                "state": model.state_dict(),
                "fold": fold,
                "selected_epochs": (
                    selected_epochs
                ),
                "initialization": (
                    args.initialization
                ),
                "ssl_checkpoint": str(
                    ssl_checkpoint or ""
                ),
                "outer_fold_auc": fold_auc,
                "config": vars(args)
                | {
                    "project_root": str(root),
                    "assignment_path": str(
                        assignment_path
                    ),
                },
            },
            checkpoint_path,
        )

        fold_rows.append(
            {
                "fold": fold,
                "n_outer_train": len(
                    outer_train
                ),
                "n_outer_validation": len(
                    outer_validation
                ),
                "n_outer_train_groups": int(
                    outer_train[
                        "patient"
                    ].nunique()
                ),
                "n_outer_validation_groups": int(
                    outer_validation[
                        "patient"
                    ].nunique()
                ),
                "selected_epochs": int(
                    selected_epochs
                ),
                "outer_auc": fold_auc,
                "ssl_checkpoint": str(
                    ssl_checkpoint or ""
                ),
            }
        )

        print(
            f"[OUTER_RESULT] fold={fold} "
            f"selected_epochs={selected_epochs} "
            f"auc={fold_auc:.6f}",
            flush=True,
        )

        del (
            model,
            outer_validation_loader,
            test_loader,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if np.isnan(oof).any():
        raise RuntimeError(
            "OOF predictions incomplete"
        )

    test_probability_mean = np.mean(
        np.stack(
            test_probabilities,
            axis=0,
        ),
        axis=0,
    )
    development_labels = development[
        "y"
    ].to_numpy(dtype=np.int8)
    test_labels = test[
        "y"
    ].to_numpy(dtype=np.int8)

    threshold = optimal_threshold(
        development_labels,
        oof,
    )
    oof_metrics = classification_metrics(
        development_labels,
        oof,
        threshold,
    )
    test_metrics = classification_metrics(
        test_labels,
        test_probability_mean,
        threshold,
    )

    fold_frame = pd.DataFrame(
        fold_rows
    )
    mean_fold_auc = float(
        fold_frame[
            "outer_auc"
        ].mean()
    )
    std_fold_auc = float(
        fold_frame[
            "outer_auc"
        ].std(ddof=1)
    )

    pd.DataFrame(
        {
            "path": development["path"],
            "patient": development["patient"],
            "y_true": development_labels,
            "fold": fold_values,
            "probability": oof,
        }
    ).to_csv(
        artifacts
        / (
            "oof_predictions_stage2_"
            f"{args.run_name}.csv"
        ),
        index=False,
    )
    pd.DataFrame(
        {
            "path": test["path"],
            "patient": test["patient"],
            "y_true": test_labels,
            "probability": (
                test_probability_mean
            ),
        }
    ).to_csv(
        artifacts
        / (
            "predictions_stage2_"
            f"{args.run_name}.csv"
        ),
        index=False,
    )
    fold_frame.to_csv(
        reports
        / (
            f"stage2_{args.run_name}_"
            "fold_metrics.csv"
        ),
        index=False,
    )
    pd.DataFrame(
        history_rows
    ).to_csv(
        artifacts
        / (
            f"history_stage2_"
            f"{args.run_name}.csv"
        ),
        index=False,
    )

    result = {
        "model": args.run_name,
        "architecture": (
            "EfficientNet-B0 dedicated "
            "whole-image head"
        ),
        "initialization": (
            args.initialization
        ),
        "evaluation": (
            "nested group-disjoint "
            "epoch selection"
        ),
        "folds": args.folds,
        "inner_folds": args.inner_folds,
        "mean_outer_fold_auc": (
            mean_fold_auc
        ),
        "std_outer_fold_auc": (
            std_fold_auc
        ),
        "development_oof": oof_metrics,
        "exploratory_test": test_metrics,
        "fold_assignment_csv": str(
            assignment_path
        ),
        "selection_used_outer_fold": False,
        "selection_used_test": False,
        "warning": (
            "The current test set was inspected "
            "during earlier development and remains "
            "exploratory."
        ),
    }
    (
        reports
        / (
            f"stage2_{args.run_name}_"
            "summary.json"
        )
    ).write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
