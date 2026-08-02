r"""
Train next-generation Stage-2 models for pediatric bacterial-vs-viral pneumonia.

Models:
    xrv_densenet121       Chest-X-ray-pretrained DenseNet121 from TorchXRayVision
    convnext_tiny_384     ImageNet-pretrained ConvNeXt-Tiny at 384x384
    efficientnet_b3_384   ImageNet-pretrained EfficientNet-B3 at 384x384

Positive class:
    viral pneumonia = 1

Negative class:
    bacterial pneumonia = 0

The script:
1. Reads the existing fixed manifest.
2. Uses only reference-positive pneumonia images for Stage 2.
3. Preserves the existing train/validation/test split.
4. Keeps aspect ratio by median-padded square resizing for ImageNet models.
5. Uses the official TorchXRayVision normalization/crop/resize chain for XRV.
6. Trains the classification head first, then fine-tunes the final backbone block.
7. Selects the checkpoint only by validation ROC-AUC.
8. Saves both validation and test probabilities for the existing comparison tools.
9. Supports completed-run reuse with --resume.

Recommended location:
    <repository>\src\stage2_train_nextgen.py

Example:
    python -m src.stage2_train_nextgen ^
        --project-root "<repository>" ^
        --model xrv_densenet121 ^
        --device cuda ^
        --resume
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


DEFAULT_SEED = 20260722
MODEL_NAMES = (
    "xrv_densenet121",
    "convnext_tiny_384",
    "efficientnet_b3_384",
)


@dataclass
class TrainConfig:
    model: str
    image_size: int
    seed: int
    batch_size: int
    workers: int
    warmup_epochs: int
    finetune_epochs: int
    patience: int
    head_lr: float
    backbone_lr: float
    weight_decay: float
    mixup_alpha: float
    mixup_probability: float
    random_erasing_probability: float
    use_pos_weight: bool
    xrv_weights: str
    device: str


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Match the stable setting already used by the project.
    torch.backends.cudnn.benchmark = False


def binary_roc_auc(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> float:
    """Pure NumPy AUC with correct handling of tied scores."""
    y_true = np.asarray(y_true, dtype=np.int8)
    probability = np.asarray(probability, dtype=np.float64)

    n_positive = int((y_true == 1).sum())
    n_negative = int((y_true == 0).sum())
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

    pneumonia = manifest[manifest["stage1"] == 1].copy()
    allowed = {"bacterial", "viral"}
    unexpected = sorted(
        set(pneumonia["subtype"].astype(str)) - allowed
    )
    if unexpected:
        raise ValueError(
            f"Stage-2肺炎数据包含异常subtype: {unexpected}"
        )

    group_split_count = pneumonia.groupby(
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


def stage2_split(
    manifest: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    frame = manifest[
        (manifest["stage1"] == 1)
        & (manifest["split"].astype(str) == split)
    ].copy()
    frame["y_true"] = (
        frame["subtype"].astype(str) == "viral"
    ).astype(np.int8)
    return frame.reset_index(drop=True)


class MedianLetterbox:
    """Resize while preserving aspect ratio, padding with image median."""

    def __init__(
        self,
        size: int,
        interpolation: InterpolationMode,
    ) -> None:
        self.size = int(size)
        self.interpolation = interpolation

    def __call__(self, image: Image.Image) -> Image.Image:
        gray = image.convert("L")
        median = int(np.median(np.asarray(gray)))
        return ImageOps.pad(
            gray,
            (self.size, self.size),
            method=(
                Image.Resampling.BICUBIC
                if self.interpolation
                == InterpolationMode.BICUBIC
                else Image.Resampling.BILINEAR
            ),
            color=median,
            centering=(0.5, 0.5),
        )


class Stage2ImageNetDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        train: bool,
        image_size: int,
        interpolation: InterpolationMode,
        random_erasing_probability: float,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.train = bool(train)
        self.image_size = int(image_size)

        geometric: List[object] = []
        if train:
            geometric.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(
                        degrees=7,
                        interpolation=interpolation,
                        fill=0,
                    ),
                    transforms.RandomAffine(
                        degrees=0,
                        translate=(0.035, 0.035),
                        scale=(0.96, 1.04),
                        interpolation=interpolation,
                        fill=0,
                    ),
                ]
            )

        tensor_ops: List[object] = [
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]

        if train and random_erasing_probability > 0:
            tensor_ops.append(
                transforms.RandomErasing(
                    p=random_erasing_probability,
                    scale=(0.02, 0.07),
                    ratio=(0.5, 2.0),
                    value=0.0,
                )
            )

        self.transform = transforms.Compose(
            [
                *geometric,
                MedianLetterbox(
                    image_size,
                    interpolation=interpolation,
                ),
                *tensor_ops,
            ]
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, int, str]:
        row = self.frame.iloc[index]
        with Image.open(row.path) as image:
            x = self.transform(image.convert("L"))
        return x, int(row.y_true), str(row.path)


class Stage2XRVDataset(Dataset):
    """Official XRV intensity normalization and 224x224 crop/resize."""

    def __init__(
        self,
        frame: pd.DataFrame,
        train: bool,
        random_erasing_probability: float,
    ) -> None:
        try:
            import torchxrayvision as xrv
        except ImportError as error:
            raise ImportError(
                "未安装torchxrayvision。请执行: "
                "python -m pip install torchxrayvision==1.5.2"
            ) from error

        self.xrv = xrv
        self.frame = frame.reset_index(drop=True)
        self.train = bool(train)
        self.random_erasing_probability = float(
            random_erasing_probability
        )
        self.center_crop = xrv.datasets.XRayCenterCrop()
        self.resize = xrv.datasets.XRayResizer(224)

    def __len__(self) -> int:
        return len(self.frame)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        if not self.train:
            return x

        if torch.rand(()) < 0.5:
            x = TF.hflip(x)

        angle = float(
            torch.empty(1).uniform_(-7.0, 7.0).item()
        )
        x = TF.rotate(
            x,
            angle=angle,
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )

        max_dx = int(round(0.035 * x.shape[-1]))
        max_dy = int(round(0.035 * x.shape[-2]))
        translate = [
            int(
                torch.randint(
                    -max_dx,
                    max_dx + 1,
                    (1,),
                ).item()
            ),
            int(
                torch.randint(
                    -max_dy,
                    max_dy + 1,
                    (1,),
                ).item()
            ),
        ]
        scale = float(
            torch.empty(1).uniform_(0.96, 1.04).item()
        )
        x = TF.affine(
            x,
            angle=0.0,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )

        if (
            self.random_erasing_probability > 0
            and torch.rand(())
            < self.random_erasing_probability
        ):
            eraser = transforms.RandomErasing(
                p=1.0,
                scale=(0.02, 0.07),
                ratio=(0.5, 2.0),
                value=0.0,
            )
            x = eraser(x)

        return x

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, int, str]:
        row = self.frame.iloc[index]

        with Image.open(row.path) as image:
            array = np.asarray(
                image.convert("L"),
                dtype=np.float32,
            )

        # Official XRV preprocessing:
        # 8-bit [0,255] -> [-1024,1024], one channel,
        # center crop, and resize to 224.
        array = self.xrv.datasets.normalize(array, 255)
        array = array[None, ...]
        array = self.center_crop(array)
        array = self.resize(array)

        x = torch.from_numpy(
            np.ascontiguousarray(array)
        ).float()
        x = self._augment(x)

        return x, int(row.y_true), str(row.path)


class XRVBinaryClassifier(nn.Module):
    def __init__(self, weights_name: str) -> None:
        super().__init__()
        try:
            import torchxrayvision as xrv
        except ImportError as error:
            raise ImportError(
                "未安装torchxrayvision。"
            ) from error

        self.backbone = xrv.models.DenseNet(
            weights=weights_name,
            apply_sigmoid=False,
        )
        # Disable pathology-specific operating-point normalization.
        self.backbone.op_threshs = None
        self.backbone.apply_sigmoid = False

        in_features = int(
            self.backbone.classifier.in_features
        )
        self.head = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(in_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.features2(x)
        return self.head(features).reshape(-1)

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.head.parameters():
            parameter.requires_grad = True

    def unfreeze_last_block(self) -> None:
        self.freeze_backbone()
        for module_name in (
            "denseblock4",
            "norm5",
        ):
            module = getattr(
                self.backbone.features,
                module_name,
            )
            for parameter in module.parameters():
                parameter.requires_grad = True

    def head_parameters(self) -> Iterable[nn.Parameter]:
        return self.head.parameters()

    def backbone_trainable_parameters(
        self,
    ) -> Iterable[nn.Parameter]:
        return (
            parameter
            for parameter in self.backbone.parameters()
            if parameter.requires_grad
        )


class ConvNeXtTinyBinary(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.DEFAULT
        )
        in_features = int(
            self.backbone.classifier[2].in_features
        )
        # We explicitly use the feature map and build our own head.
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(in_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        # ConvNeXt classifier begins with channel-first LayerNorm.
        x = self.backbone.classifier[0](x)
        return self.head(x).reshape(-1)

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.head.parameters():
            parameter.requires_grad = True

    def unfreeze_last_block(self) -> None:
        self.freeze_backbone()
        for module in self.backbone.features[-2:]:
            for parameter in module.parameters():
                parameter.requires_grad = True
        for parameter in self.backbone.classifier[
            0
        ].parameters():
            parameter.requires_grad = True

    def head_parameters(self) -> Iterable[nn.Parameter]:
        return self.head.parameters()

    def backbone_trainable_parameters(
        self,
    ) -> Iterable[nn.Parameter]:
        return (
            parameter
            for parameter in self.backbone.parameters()
            if parameter.requires_grad
        )


class EfficientNetB3Binary(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.DEFAULT
        )
        in_features = int(
            self.backbone.classifier[1].in_features
        )
        self.head = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(in_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return self.head(x).reshape(-1)

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.head.parameters():
            parameter.requires_grad = True

    def unfreeze_last_block(self) -> None:
        self.freeze_backbone()
        for module in self.backbone.features[-3:]:
            for parameter in module.parameters():
                parameter.requires_grad = True

    def head_parameters(self) -> Iterable[nn.Parameter]:
        return self.head.parameters()

    def backbone_trainable_parameters(
        self,
    ) -> Iterable[nn.Parameter]:
        return (
            parameter
            for parameter in self.backbone.parameters()
            if parameter.requires_grad
        )


def build_model(
    model_name: str,
    xrv_weights: str,
) -> nn.Module:
    if model_name == "xrv_densenet121":
        return XRVBinaryClassifier(xrv_weights)
    if model_name == "convnext_tiny_384":
        return ConvNeXtTinyBinary()
    if model_name == "efficientnet_b3_384":
        return EfficientNetB3Binary()
    raise ValueError(f"未知模型: {model_name}")


def image_size_for_model(model_name: str) -> int:
    if model_name == "xrv_densenet121":
        return 224
    return 384


def make_dataset(
    frame: pd.DataFrame,
    model_name: str,
    train: bool,
    image_size: int,
    random_erasing_probability: float,
) -> Dataset:
    if model_name == "xrv_densenet121":
        return Stage2XRVDataset(
            frame,
            train=train,
            random_erasing_probability=(
                random_erasing_probability
            ),
        )

    interpolation = (
        InterpolationMode.BICUBIC
        if model_name == "efficientnet_b3_384"
        else InterpolationMode.BILINEAR
    )
    return Stage2ImageNetDataset(
        frame,
        train=train,
        image_size=image_size,
        interpolation=interpolation,
        random_erasing_probability=(
            random_erasing_probability
        ),
    )


def make_loader(
    dataset: Dataset,
    train: bool,
    batch_size: int,
    workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=False,
    )


def maybe_mixup(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    probability: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if (
        alpha <= 0
        or probability <= 0
        or images.shape[0] < 2
        or torch.rand((), device=images.device)
        >= probability
    ):
        return images, targets

    beta = torch.distributions.Beta(alpha, alpha)
    lam = beta.sample().to(images.device)
    permutation = torch.randperm(
        images.shape[0],
        device=images.device,
    )
    mixed_images = (
        lam * images
        + (1.0 - lam) * images[permutation]
    )
    mixed_targets = (
        lam * targets
        + (1.0 - lam) * targets[permutation]
    )
    return mixed_images, mixed_targets


@torch.inference_mode()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[str], float]:
    model.eval()
    labels: List[int] = []
    probabilities: List[float] = []
    paths: List[str] = []

    start = time.perf_counter()
    for images, target, batch_paths in loader:
        images = images.to(
            device,
            non_blocking=True,
        )
        logits = model(images).reshape(-1)
        probability = torch.sigmoid(logits)

        labels.extend(target.numpy().astype(int).tolist())
        probabilities.extend(
            probability.cpu().numpy().astype(float).tolist()
        )
        paths.extend([str(path) for path in batch_paths])

    elapsed = time.perf_counter() - start
    return (
        np.asarray(labels, dtype=np.int8),
        np.asarray(probabilities, dtype=np.float64),
        paths,
        elapsed,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
    mixup_alpha: float,
    mixup_probability: float,
) -> float:
    model.train()
    losses: List[float] = []

    for images, target, _ in loader:
        images = images.to(
            device,
            non_blocking=True,
        )
        target = target.float().to(
            device,
            non_blocking=True,
        )

        images, target = maybe_mixup(
            images,
            target,
            alpha=mixup_alpha,
            probability=mixup_probability,
        )

        optimizer.zero_grad(set_to_none=True)

        # FP32 is the project default because the local Windows/CUDA
        # configuration previously showed unstable depthwise-convolution
        # behavior under FP16.
        logits = model(images).reshape(-1)
        loss = loss_function(logits, target)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"训练损失出现非有限值: {loss.item()}"
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            max_norm=5.0,
        )
        optimizer.step()

        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses))


def build_loss(
    train_frame: pd.DataFrame,
    device: torch.device,
    use_pos_weight: bool,
) -> Tuple[nn.Module, float]:
    n_positive = int((train_frame.y_true == 1).sum())
    n_negative = int((train_frame.y_true == 0).sum())

    if n_positive == 0 or n_negative == 0:
        raise ValueError(
            "Stage-2训练集必须同时包含细菌和病毒病例"
        )

    pos_weight_value = (
        n_negative / n_positive
        if use_pos_weight
        else 1.0
    )
    pos_weight = torch.tensor(
        [pos_weight_value],
        device=device,
        dtype=torch.float32,
    )

    return (
        nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        ),
        float(pos_weight_value),
    )


def count_parameters(
    parameters: Iterable[nn.Parameter],
) -> int:
    return sum(
        parameter.numel()
        for parameter in parameters
        if parameter.requires_grad
    )


def save_predictions(
    path: Path,
    labels: np.ndarray,
    probabilities: np.ndarray,
    paths: Sequence[str],
) -> None:
    pd.DataFrame(
        {
            "path": list(paths),
            "y_true": labels.astype(int),
            "probability": probabilities.astype(float),
        }
    ).to_csv(path, index=False)


def train_model(
    root: Path,
    config: TrainConfig,
    resume: bool,
) -> Dict[str, object]:
    model_name = config.model
    output_model_name = model_name

    artifact_dir = root / "artifacts"
    report_dir = root / "reports"
    checkpoint_dir = root / "models" / "stage2"

    for directory in (
        artifact_dir,
        report_dir,
        checkpoint_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    best_checkpoint = (
        checkpoint_dir / f"{output_model_name}.pt"
    )
    complete_marker = (
        artifact_dir
        / f"train_complete_stage2_{output_model_name}.txt"
    )
    validation_prediction_path = (
        artifact_dir
        / f"val_predictions_stage2_{output_model_name}.csv"
    )
    test_prediction_path = (
        artifact_dir
        / f"predictions_stage2_{output_model_name}.csv"
    )
    history_path = (
        artifact_dir
        / f"history_stage2_{output_model_name}.csv"
    )
    summary_path = (
        report_dir
        / f"stage2_training_summary_{output_model_name}.json"
    )

    if (
        resume
        and complete_marker.exists()
        and best_checkpoint.exists()
        and validation_prediction_path.exists()
        and test_prediction_path.exists()
    ):
        print(
            f"COMPLETED_RUN_REUSED {output_model_name}",
            flush=True,
        )
        return json.loads(
            summary_path.read_text(encoding="utf-8")
        )

    manifest_path = (
        root / "data" / "processed" / "manifest.csv"
    )
    manifest = pd.read_csv(manifest_path)
    validate_manifest(manifest)

    train_frame = stage2_split(manifest, "train")
    validation_frame = stage2_split(manifest, "val")
    test_frame = stage2_split(manifest, "test")

    print(
        f"\n{output_model_name}: "
        f"train={len(train_frame)}, "
        f"val={len(validation_frame)}, "
        f"test={len(test_frame)}",
        flush=True,
    )
    print(
        "Train class counts: "
        f"bacterial={(train_frame.y_true == 0).sum()}, "
        f"viral={(train_frame.y_true == 1).sum()}",
        flush=True,
    )

    train_dataset = make_dataset(
        train_frame,
        model_name,
        train=True,
        image_size=config.image_size,
        random_erasing_probability=(
            config.random_erasing_probability
        ),
    )
    validation_dataset = make_dataset(
        validation_frame,
        model_name,
        train=False,
        image_size=config.image_size,
        random_erasing_probability=0.0,
    )
    test_dataset = make_dataset(
        test_frame,
        model_name,
        train=False,
        image_size=config.image_size,
        random_erasing_probability=0.0,
    )

    train_loader = make_loader(
        train_dataset,
        train=True,
        batch_size=config.batch_size,
        workers=config.workers,
    )
    validation_loader = make_loader(
        validation_dataset,
        train=False,
        batch_size=config.batch_size,
        workers=config.workers,
    )
    test_loader = make_loader(
        test_dataset,
        train=False,
        batch_size=config.batch_size,
        workers=config.workers,
    )

    device = torch.device(config.device)
    model = build_model(
        model_name,
        xrv_weights=config.xrv_weights,
    ).to(device)

    loss_function, pos_weight_value = build_loss(
        train_frame,
        device,
        use_pos_weight=config.use_pos_weight,
    )

    history: List[Dict[str, object]] = []
    best_validation_auc = -math.inf
    best_epoch = 0
    global_epoch = 0

    phases = (
        (
            "head_warmup",
            config.warmup_epochs,
        ),
        (
            "last_block_finetune",
            config.finetune_epochs,
        ),
    )

    for phase_name, epochs in phases:
        if epochs <= 0:
            continue

        if phase_name == "head_warmup":
            model.freeze_backbone()
            optimizer = torch.optim.AdamW(
                model.head_parameters(),
                lr=config.head_lr,
                weight_decay=config.weight_decay,
            )
        else:
            model.unfreeze_last_block()
            backbone_parameters = list(
                model.backbone_trainable_parameters()
            )
            head_parameters = list(
                model.head_parameters()
            )
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": backbone_parameters,
                        "lr": config.backbone_lr,
                    },
                    {
                        "params": head_parameters,
                        "lr": config.head_lr * 0.20,
                    },
                ],
                weight_decay=config.weight_decay,
            )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs),
            eta_min=min(
                config.backbone_lr,
                config.head_lr,
            )
            * 0.05,
        )

        trainable_count = count_parameters(
            model.parameters()
        )
        print(
            f"\nPHASE_START {phase_name}: "
            f"epochs={epochs}, "
            f"trainable_parameters={trainable_count:,}",
            flush=True,
        )

        bad_epochs = 0

        for phase_epoch in range(1, epochs + 1):
            global_epoch += 1

            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                loss_function,
                device,
                mixup_alpha=config.mixup_alpha,
                mixup_probability=(
                    config.mixup_probability
                ),
            )

            validation_y, validation_p, _, _ = predict(
                model,
                validation_loader,
                device,
            )
            validation_auc = binary_roc_auc(
                validation_y,
                validation_p,
            )

            current_lrs = [
                float(group["lr"])
                for group in optimizer.param_groups
            ]

            row = {
                "global_epoch": global_epoch,
                "phase": phase_name,
                "phase_epoch": phase_epoch,
                "train_loss": train_loss,
                "validation_auc": validation_auc,
                "learning_rates": json.dumps(
                    current_lrs
                ),
                "trainable_parameters": trainable_count,
            }
            history.append(row)
            pd.DataFrame(history).to_csv(
                history_path,
                index=False,
            )

            print(
                f"EPOCH {global_epoch:02d} "
                f"phase={phase_name} "
                f"loss={train_loss:.6f} "
                f"val_auc={validation_auc:.6f} "
                f"lr={current_lrs}",
                flush=True,
            )

            if validation_auc > best_validation_auc + 1e-4:
                best_validation_auc = validation_auc
                best_epoch = global_epoch
                bad_epochs = 0

                torch.save(
                    {
                        "state": model.state_dict(),
                        "model_name": output_model_name,
                        "task": "stage2",
                        "positive_class": "viral",
                        "negative_class": "bacterial",
                        "best_validation_auc": (
                            best_validation_auc
                        ),
                        "best_epoch": best_epoch,
                        "config": asdict(config),
                        "pos_weight": pos_weight_value,
                    },
                    best_checkpoint,
                )
                print(
                    "BEST_CHECKPOINT_SAVED "
                    f"epoch={best_epoch} "
                    f"val_auc={best_validation_auc:.6f} "
                    f"-> {best_checkpoint}",
                    flush=True,
                )
            else:
                bad_epochs += 1

            scheduler.step()

            if (
                phase_name == "last_block_finetune"
                and bad_epochs >= config.patience
            ):
                print(
                    "EARLY_STOP "
                    f"phase={phase_name} "
                    f"bad_epochs={bad_epochs}",
                    flush=True,
                )
                break

    if not best_checkpoint.exists():
        raise RuntimeError(
            "训练结束但没有生成最佳checkpoint"
        )

    checkpoint = torch.load(
        best_checkpoint,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["state"])

    validation_y, validation_p, validation_paths, val_seconds = (
        predict(
            model,
            validation_loader,
            device,
        )
    )
    test_y, test_p, test_paths, test_seconds = predict(
        model,
        test_loader,
        device,
    )

    save_predictions(
        validation_prediction_path,
        validation_y,
        validation_p,
        validation_paths,
    )
    save_predictions(
        test_prediction_path,
        test_y,
        test_p,
        test_paths,
    )

    final_validation_auc = binary_roc_auc(
        validation_y,
        validation_p,
    )
    final_test_auc = binary_roc_auc(
        test_y,
        test_p,
    )

    summary: Dict[str, object] = {
        "status": "complete",
        "model": output_model_name,
        "positive_class": "viral",
        "negative_class": "bacterial",
        "best_epoch": best_epoch,
        "best_validation_auc_during_training": (
            best_validation_auc
        ),
        "saved_validation_auc": final_validation_auc,
        "saved_test_auc": final_test_auc,
        "n_train": len(train_frame),
        "n_validation": len(validation_frame),
        "n_test": len(test_frame),
        "train_bacterial": int(
            (train_frame.y_true == 0).sum()
        ),
        "train_viral": int(
            (train_frame.y_true == 1).sum()
        ),
        "pos_weight": pos_weight_value,
        "validation_inference_seconds": val_seconds,
        "test_inference_seconds": test_seconds,
        "validation_ms_per_image": (
            val_seconds / len(validation_frame) * 1000
        ),
        "test_ms_per_image": (
            test_seconds / len(test_frame) * 1000
        ),
        "config": asdict(config),
        "files": {
            "checkpoint": str(best_checkpoint),
            "history": str(history_path),
            "validation_predictions": str(
                validation_prediction_path
            ),
            "test_predictions": str(
                test_prediction_path
            ),
        },
        "interpretation_note": (
            "Model/checkpoint selection used validation ROC-AUC only. "
            "The existing test set has already been inspected in earlier "
            "experiments, so new-model test results should be treated as "
            "exploratory until externally confirmed."
        ),
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    complete_marker.write_text(
        "training completed; best checkpoint selected by "
        "validation ROC-AUC only\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 88)
    print(f"TRAINING_COMPLETE {output_model_name}")
    print(
        f"Best validation AUC: "
        f"{final_validation_auc:.6f}"
    )
    print(
        f"Exploratory test AUC: "
        f"{final_test_auc:.6f}"
    )
    print(f"Checkpoint: {best_checkpoint}")
    print(
        "Validation predictions: "
        f"{validation_prediction_path}"
    )
    print(f"Test predictions: {test_prediction_path}")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--model",
        choices=(*MODEL_NAMES, "all"),
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
        "--workers",
        type=int,
        default=0,
        help=(
            "Windows建议使用0；Linux可尝试2-4。"
        ),
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--head-lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=2e-5,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--mixup-probability",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--random-erasing-probability",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--no-pos-weight",
        action="store_true",
    )
    parser.add_argument(
        "--xrv-weights",
        default="densenet121-res224-all",
        help=(
            "TorchXRayVision DenseNet weights. "
            "Recommended initial experiment: "
            "densenet121-res224-all"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse a fully completed model run. "
            "Interrupted mid-training runs restart safely."
        ),
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    if not (
        root / "data" / "processed" / "manifest.csv"
    ).exists():
        raise FileNotFoundError(
            "未找到 "
            f"{root / 'data' / 'processed' / 'manifest.csv'}"
        )

    if (
        args.device == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "指定了cuda，但PyTorch检测不到CUDA"
        )

    seed_all(args.seed)

    selected_models = (
        MODEL_NAMES
        if args.model == "all"
        else (args.model,)
    )

    summaries: List[Dict[str, object]] = []

    for model_name in selected_models:
        image_size = image_size_for_model(model_name)
        config = TrainConfig(
            model=model_name,
            image_size=image_size,
            seed=args.seed,
            batch_size=args.batch_size,
            workers=args.workers,
            warmup_epochs=args.warmup_epochs,
            finetune_epochs=args.finetune_epochs,
            patience=args.patience,
            head_lr=args.head_lr,
            backbone_lr=args.backbone_lr,
            weight_decay=args.weight_decay,
            mixup_alpha=args.mixup_alpha,
            mixup_probability=args.mixup_probability,
            random_erasing_probability=(
                args.random_erasing_probability
            ),
            use_pos_weight=not args.no_pos_weight,
            xrv_weights=args.xrv_weights,
            device=args.device,
        )

        summaries.append(
            train_model(
                root,
                config,
                resume=args.resume,
            )
        )

        if args.device == "cuda":
            torch.cuda.empty_cache()

    print(
        json.dumps(
            {
                "status": "all_requested_models_complete",
                "models": [
                    summary["model"]
                    for summary in summaries
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
