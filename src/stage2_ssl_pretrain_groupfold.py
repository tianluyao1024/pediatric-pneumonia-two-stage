"""Strict fold-specific pediatric chest-X-ray SimSiam pretraining.

Purpose
-------
Create one self-supervised EfficientNet-B0 encoder for each outer Stage-2
GroupKFold split. The held-out outer-fold images/groups are excluded from that
fold's SSL pool, so downstream OOF predictions remain inductive.

SSL pool for fold k
-------------------
All manifest rows with split == "train", except rows whose filename-group
(`patient` column) belongs to the held-out Stage-2 fold.

No bacterial/viral labels are used by the SSL objective.

Outputs
-------
reports/stage2_groupfold_assignments.csv
models/ssl_groupfold/pediatric_simsiam_efficientnet_b0_fold{1..K}.pt
models/ssl_groupfold/pediatric_simsiam_efficientnet_b0_fold{1..K}_best.pt
artifacts/history_ssl_groupfold_fold{1..K}.csv
reports/ssl_groupfold_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True
SEED = 20260728


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def stable_hash(values: List[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class TwoViewTransform:
    """Anatomy-preserving pediatric CXR augmentations.

    Horizontal flipping is intentionally omitted because downstream models use
    explicit left/right lung views.
    """

    def __init__(self, size: int) -> None:
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size,
                    scale=(0.70, 1.00),
                    ratio=(0.86, 1.14),
                    antialias=True,
                ),
                transforms.RandomRotation(
                    6,
                    fill=0,
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.035, 0.035),
                    scale=(0.96, 1.04),
                    fill=0,
                ),
                transforms.ColorJitter(
                    brightness=0.18,
                    contrast=0.22,
                ),
                transforms.RandomApply(
                    [
                        transforms.GaussianBlur(
                            kernel_size=7,
                            sigma=(0.1, 1.2),
                        )
                    ],
                    p=0.30,
                ),
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

    def __call__(
        self,
        image: Image.Image,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.transform(image),
            self.transform(image),
        )


class SSLDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_size: int,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = TwoViewTransform(image_size)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        path = str(self.frame.iloc[index]["path"])
        with Image.open(path) as image:
            gray = image.convert("L")
        first, second = self.transform(gray)
        return first, second, path


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = models.efficientnet_b0(weights=None)
        self.features = backbone.features
        self.pool = backbone.avgpool
        self.dim = 1280

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.flatten(
            self.pool(self.features(image)),
            1,
        )


def projection_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(
            input_dim,
            hidden_dim,
            bias=False,
        ),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
        ),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(
            hidden_dim,
            output_dim,
            bias=False,
        ),
        nn.BatchNorm1d(
            output_dim,
            affine=False,
        ),
    )


class SimSiam(nn.Module):
    def __init__(
        self,
        projection_dim: int = 2048,
        prediction_dim: int = 512,
    ) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.projector = projection_mlp(
            self.encoder.dim,
            2048,
            projection_dim,
        )
        self.predictor = nn.Sequential(
            nn.Linear(
                projection_dim,
                prediction_dim,
                bias=False,
            ),
            nn.BatchNorm1d(prediction_dim),
            nn.ReLU(inplace=True),
            nn.Linear(
                prediction_dim,
                projection_dim,
            ),
        )

    def forward(
        self,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        representation = self.projector(
            self.encoder(image)
        )
        prediction = self.predictor(
            representation
        )
        return prediction, representation.detach()


def negative_cosine(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    prediction = nn.functional.normalize(
        prediction,
        dim=1,
    )
    target = nn.functional.normalize(
        target,
        dim=1,
    )
    return -(
        prediction * target
    ).sum(dim=1).mean()


def create_fold_assignments(
    manifest: pd.DataFrame,
    folds: int,
) -> pd.DataFrame:
    development = manifest[
        (manifest["stage1"] == 1)
        & manifest["split"].isin(
            ["train", "val"]
        )
    ].copy().reset_index(drop=True)

    groups = development[
        "patient"
    ].astype(str).to_numpy()
    labels = (
        development["subtype"].astype(str)
        == "viral"
    ).astype(int).to_numpy()

    splitter = GroupKFold(
        n_splits=folds
    )
    assigned = np.full(
        len(development),
        -1,
        dtype=np.int16,
    )

    for fold, (_, validation_index) in enumerate(
        splitter.split(
            development,
            labels,
            groups,
        ),
        start=1,
    ):
        assigned[validation_index] = fold

    if np.any(assigned < 1):
        raise RuntimeError(
            "GroupKFold assignment incomplete"
        )

    output = development[
        [
            "path",
            "patient",
            "split",
            "subtype",
        ]
    ].copy()
    output["fold"] = assigned
    return output


def save_checkpoint(
    path: Path,
    model: SimSiam,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_loss: float,
    fold: int,
    pool_hash: str,
    config: Dict[str, object],
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "encoder": model.encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": int(epoch),
            "best_loss": float(best_loss),
            "fold": int(fold),
            "ssl_pool_hash": pool_hash,
            "config": config,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
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
        "--lr",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )
    parser.add_argument(
        "--only-fold",
        type=int,
        default=0,
        help=(
            "0 trains all folds; 1..K trains only "
            "the selected fold."
        ),
    )
    args = parser.parse_args()

    seed_all(args.seed)

    root = args.project_root.resolve()
    reports = root / "reports"
    artifacts = root / "artifacts"
    model_dir = root / "models" / "ssl_groupfold"

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

    required = {
        "path",
        "patient",
        "split",
        "stage1",
        "subtype",
    }
    missing = sorted(
        required - set(manifest.columns)
    )
    if missing:
        raise ValueError(
            f"manifest missing columns: {missing}"
        )

    assignment = create_fold_assignments(
        manifest,
        args.folds,
    )
    assignment_path = (
        reports
        / "stage2_groupfold_assignments.csv"
    )
    assignment.to_csv(
        assignment_path,
        index=False,
    )

    device = torch.device(
        args.device
        if args.device == "cpu"
        or torch.cuda.is_available()
        else "cpu"
    )

    requested_folds = (
        [args.only_fold]
        if args.only_fold
        else list(range(1, args.folds + 1))
    )

    if any(
        fold < 1 or fold > args.folds
        for fold in requested_folds
    ):
        raise ValueError(
            "--only-fold must be 0 or 1..folds"
        )

    summaries = []

    for fold in requested_folds:
        fold_seed = args.seed + fold * 1000
        seed_all(fold_seed)

        heldout = assignment[
            assignment["fold"] == fold
        ].copy()
        heldout_groups = set(
            heldout["patient"].astype(str)
        )

        ssl_pool = manifest[
            manifest["split"].astype(str)
            == "train"
        ].copy()
        ssl_pool = ssl_pool[
            ~ssl_pool["patient"]
            .astype(str)
            .isin(heldout_groups)
        ].reset_index(drop=True)

        if len(ssl_pool) < 2:
            raise RuntimeError(
                f"Fold {fold} SSL pool too small"
            )

        pool_hash = stable_hash(
            ssl_pool["path"]
            .astype(str)
            .tolist()
        )
        heldout_hash = stable_hash(
            heldout["path"]
            .astype(str)
            .tolist()
        )

        dataset = SSLDataset(
            ssl_pool,
            args.image_size,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.workers > 0,
            drop_last=True,
        )

        model = SimSiam().to(device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
            nesterov=True,
            foreach=False,
        )
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs,
                eta_min=args.lr * 0.001,
            )
        )

        checkpoint = (
            model_dir
            / (
                "pediatric_simsiam_"
                f"efficientnet_b0_fold{fold}.pt"
            )
        )
        best_checkpoint = (
            model_dir
            / (
                "pediatric_simsiam_"
                f"efficientnet_b0_fold{fold}_best.pt"
            )
        )
        history_path = (
            artifacts
            / f"history_ssl_groupfold_fold{fold}.csv"
        )

        start_epoch = 0
        best_loss = float("inf")
        history: List[Dict[str, object]] = []

        if args.resume and checkpoint.exists():
            saved = torch.load(
                checkpoint,
                map_location=device,
                weights_only=True,
            )
            if (
                str(saved.get("ssl_pool_hash", ""))
                != pool_hash
            ):
                raise RuntimeError(
                    f"Fold {fold} resume pool hash mismatch"
                )

            model.load_state_dict(
                saved["model"]
            )
            optimizer.load_state_dict(
                saved["optimizer"]
            )
            scheduler.load_state_dict(
                saved["scheduler"]
            )
            start_epoch = (
                int(saved["epoch"]) + 1
            )
            best_loss = float(
                saved.get(
                    "best_loss",
                    best_loss,
                )
            )

            if history_path.exists():
                history = pd.read_csv(
                    history_path
                ).to_dict("records")

            print(
                f"[SSL_RESUME] fold={fold} "
                f"next_epoch={start_epoch + 1} "
                f"best_loss={best_loss:.6f}",
                flush=True,
            )

        config = {
            "project_root": str(root),
            "folds": args.folds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "image_size": args.image_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "fold_seed": fold_seed,
            "device": str(device),
            "horizontal_flip": False,
            "ssl_pool_rule": (
                "manifest split=train excluding "
                "held-out Stage-2 filename groups"
            ),
        }

        for epoch in range(
            start_epoch,
            args.epochs,
        ):
            model.train()
            losses = []
            start_time = time.perf_counter()

            for first, second, _ in loader:
                first = first.to(
                    device,
                    non_blocking=True,
                )
                second = second.to(
                    device,
                    non_blocking=True,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                prediction_first, target_first = model(
                    first
                )
                prediction_second, target_second = model(
                    second
                )

                loss = 0.5 * (
                    negative_cosine(
                        prediction_first,
                        target_second,
                    )
                    + negative_cosine(
                        prediction_second,
                        target_first,
                    )
                )

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "Non-finite SSL loss"
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

            scheduler.step()

            mean_loss = float(
                np.mean(losses)
            )
            elapsed = (
                time.perf_counter()
                - start_time
            )

            history.append(
                {
                    "fold": fold,
                    "epoch": epoch + 1,
                    "loss": mean_loss,
                    "lr": optimizer.param_groups[
                        0
                    ]["lr"],
                    "seconds": elapsed,
                    "n_ssl_images": len(
                        ssl_pool
                    ),
                }
            )
            pd.DataFrame(history).to_csv(
                history_path,
                index=False,
            )

            improved = (
                mean_loss < best_loss
            )
            if improved:
                best_loss = mean_loss

            save_checkpoint(
                checkpoint,
                model,
                optimizer,
                scheduler,
                epoch,
                best_loss,
                fold,
                pool_hash,
                config,
            )

            if improved:
                save_checkpoint(
                    best_checkpoint,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_loss,
                    fold,
                    pool_hash,
                    config,
                )

            print(
                f"[SSL] fold={fold} "
                f"epoch={epoch + 1:03d} "
                f"loss={mean_loss:.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.7f} "
                f"seconds={elapsed:.1f}",
                flush=True,
            )

        summaries.append(
            {
                "fold": fold,
                "n_ssl_images": int(
                    len(ssl_pool)
                ),
                "n_heldout_stage2_images": int(
                    len(heldout)
                ),
                "n_heldout_groups": int(
                    heldout["patient"].nunique()
                ),
                "ssl_pool_hash": pool_hash,
                "heldout_hash": heldout_hash,
                "best_ssl_loss": float(
                    best_loss
                ),
                "checkpoint": str(
                    checkpoint
                ),
                "best_checkpoint": str(
                    best_checkpoint
                ),
                "history": str(
                    history_path
                ),
            }
        )

        del (
            model,
            optimizer,
            scheduler,
            loader,
            dataset,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = (
        reports
        / "ssl_groupfold_summary.json"
    )

    existing = {}
    if summary_path.exists():
        try:
            existing = json.loads(
                summary_path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            existing = {}

    old_folds = {
        int(item["fold"]): item
        for item in existing.get(
            "fold_summaries",
            [],
        )
        if "fold" in item
    }
    for item in summaries:
        old_folds[int(item["fold"])] = item

    summary = {
        "status": (
            "complete"
            if len(old_folds) == args.folds
            else "partial"
        ),
        "method": (
            "fold-specific SimSiam "
            "for strict Stage-2 OOF"
        ),
        "folds": args.folds,
        "assignment_file": str(
            assignment_path
        ),
        "assignment_hash": stable_hash(
            (
                assignment["path"].astype(str)
                + "|"
                + assignment["fold"].astype(str)
            ).tolist()
        ),
        "fold_summaries": [
            old_folds[key]
            for key in sorted(old_folds)
        ],
        "leakage_control": (
            "Each held-out Stage-2 filename group "
            "is excluded from that fold's SSL pool."
        ),
        "label_use": (
            "No bacterial/viral labels are used "
            "during SSL optimization."
        ),
    }

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
