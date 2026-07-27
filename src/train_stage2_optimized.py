"""High-resolution, staged fine-tuning experiments for pneumonia etiology."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.io import read_image
from torchvision.transforms import functional as TF

from src.pipeline import GeM, SEED, bootstrap_ci, metrics, optimal_threshold, seed_all


def letterbox(image, size):
    _, height, width = image.shape
    scale = size / max(height, width)
    new_height, new_width = max(1, round(height * scale)), max(1, round(width * scale))
    image = TF.resize(image, [new_height, new_width], antialias=True)
    left=(size-new_width)//2; right=size-new_width-left
    top=(size-new_height)//2; bottom=size-new_height-top
    return TF.pad(image,[left,top,right,bottom],fill=0)


def lung_crop_equalize(image):
    _, height, width=image.shape
    image=image[:,round(.04*height):round(.96*height),round(.06*width):round(.94*width)]
    return TF.equalize(image)


class Stage2XRays(Dataset):
    def __init__(self, frame: pd.DataFrame, train: bool, size: int, preserve_aspect: bool = False,
                 contrast_normalize: bool = False):
        self.frame = frame.reset_index(drop=True)
        ops = [
            transforms.Lambda(lambda z: z.repeat(3, 1, 1) if z.shape[0] == 1 else transforms.functional.rgb_to_grayscale(z, 3)),
            transforms.Lambda(lung_crop_equalize) if contrast_normalize else transforms.Lambda(lambda z:z),
            transforms.Lambda(lambda z: letterbox(z, size)) if preserve_aspect else transforms.Resize((size, size), antialias=True),
        ]
        if train:
            ops += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(5),
                    transforms.RandomAffine(0, translate=(0.025, 0.025), scale=(0.97, 1.03))]
        ops += [transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize([.485, .456, .406], [.229, .224, .225])]
        self.tf = transforms.Compose(ops)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        return self.tf(read_image(row.path)), int(row.subtype == "viral"), row.path


class DenseEtiologyNet(nn.Module):
    def __init__(self, gem: bool):
        super().__init__()
        base = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.features = base.features
        self.pool = GeM() if gem else nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(.35), nn.Linear(1024, 1))

    def forward(self, x):
        x = nn.functional.relu(self.features(x), inplace=True)
        return self.head(self.pool(x)).reshape(-1)


class ConvNextEtiologyNet(nn.Module):
    def __init__(self):
        super().__init__()
        base=models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        self.features=base.features; self.pool=GeM()
        self.head=nn.Sequential(nn.Flatten(),nn.LayerNorm(768),nn.Dropout(.35),nn.Linear(768,1))
    def forward(self,x): return self.head(self.pool(self.features(x))).reshape(-1)


class EfficientEtiologyNet(nn.Module):
    def __init__(self):
        super().__init__(); base=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.features=base.features; self.pool=GeM()
        self.head=nn.Sequential(nn.Flatten(),nn.Dropout(.35),nn.Linear(1280,1))
    def forward(self,x): return self.head(self.pool(self.features(x))).reshape(-1)


@torch.inference_mode()
def predict(model, loader, device, tta: bool):
    model.eval(); ys=[]; probs=[]; paths=[]
    for x, y, p in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probability = torch.sigmoid(logits)
        if tta:
            probability = (probability + torch.sigmoid(model(torch.flip(x, dims=[3])))) / 2
        ys.extend(y.numpy()); probs.extend(probability.cpu().numpy()); paths.extend(p)
    return np.asarray(ys), np.asarray(probs), paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--name", required=True, choices=["pneunet_hr", "densenet121_avg", "pneunet_v2", "pneunet_eq"])
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=14)
    args = parser.parse_args()
    root = Path(args.project_root); seed_all(); device = torch.device("cuda")
    df = pd.read_csv(root / "data" / "processed" / "manifest.csv")
    df = df[df.stage1 == 1]
    train = df[df.split == "train"]; val = df[df.split == "val"]; test = df[df.split == "test"]
    preserve_aspect=args.name in ("pneunet_v2","pneunet_eq"); contrast_normalize=args.name=="pneunet_eq"
    batch_size=4 if args.name=="pneunet_v2" else 6
    train_loader = DataLoader(Stage2XRays(train, True, args.size, preserve_aspect, contrast_normalize), batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(Stage2XRays(val, False, args.size, preserve_aspect, contrast_normalize), batch_size=8, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(Stage2XRays(test, False, args.size, preserve_aspect, contrast_normalize), batch_size=8, shuffle=False,
                             num_workers=0, pin_memory=True)
    model = (ConvNextEtiologyNet() if args.name=="pneunet_v2" else EfficientEtiologyNet() if args.name=="pneunet_eq"
             else DenseEtiologyNet(gem=args.name == "pneunet_hr")).to(device)
    for parameter in model.features.parameters(): parameter.requires_grad = False
    counts = train.subtype.value_counts(); pos_weight = torch.tensor([counts.bacterial / counts.viral], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    checkpoint = root / "models" / "stage2" / f"{args.name}.pt"
    history_file = root / "artifacts" / f"history_stage2_{args.name}.csv"
    history = pd.read_csv(history_file).to_dict("records") if history_file.exists() else []
    best=-1.; bad=0; start_epoch=0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(saved["state"])
        start_epoch = int(saved.get("epoch", 2)) + 1
        best = float(saved.get("best_val_auc", -1.0))
        print(f"{args.name} resumed_at_epoch={start_epoch + 1} best_val_auc={best:.6f}", flush=True)
    for epoch in range(start_epoch, args.epochs):
        if epoch == 2:
            for parameter in model.features.parameters(): parameter.requires_grad = True
        optimizer = torch.optim.AdamW([
            {"params": model.features.parameters(), "lr": (1e-5 if args.name in ("pneunet_v2","pneunet_eq") else 2e-5) if epoch >= 2 else 0.0},
            {"params": list(model.pool.parameters()) + list(model.head.parameters()), "lr": 1e-4 if args.name in ("pneunet_v2","pneunet_eq") else 2e-4},
        ], weight_decay=1e-4)
        model.train(); losses=[]
        for x, y, _ in train_loader:
            x=x.to(device, non_blocking=True); y=y.float().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True); loss=loss_fn(model(x), y); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step(); losses.append(loss.item())
        vy, vp, _ = predict(model, val_loader, device, True); auc=roc_auc_score(vy, vp)
        history.append({"epoch":epoch+1,"loss":float(np.mean(losses)),"val_auc":auc})
        print(f"{args.name} epoch={epoch+1} loss={np.mean(losses):.5f} val_auc={auc:.6f}", flush=True)
        if auc > best + 1e-4:
            best=auc; bad=0; checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state":model.state_dict(),"name":args.name,"size":args.size,
                        "epoch":epoch,"best_val_auc":best}, checkpoint)
        else:
            bad += 1
        pd.DataFrame(history).to_csv(history_file, index=False)
        if epoch >= 4 and bad >= 4: break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["state"])
    vy, vp, _ = predict(model, val_loader, device, True); threshold=optimal_threshold(vy, vp)
    y, p, paths = predict(model, test_loader, device, True)
    pd.DataFrame({"path":paths,"y_true":y,"probability":p}).to_csv(
        root / "artifacts" / f"predictions_stage2_{args.name}.csv", index=False)
    result=metrics("stage2",args.name,y,p,0.0,threshold=threshold,reps=2000)
    pd.DataFrame([result]).to_csv(root / "artifacts" / f"metrics_stage2_{args.name}.csv", index=False)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
