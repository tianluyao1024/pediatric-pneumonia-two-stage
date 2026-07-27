"""Single-stage three-class PneuNet baseline on the fixed filename-group split."""
from pathlib import Path
import json, os, sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.io import read_image

ROOT = Path(os.environ.get("PNEUMONIA_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
from src.pipeline import CBAM, GeM, seed_all


class ThreeClassXRays(Dataset):
    label = {"normal": 0, "bacterial": 1, "viral": 2}
    def __init__(self, frame, train, size=224):
        self.frame = frame.reset_index(drop=True)
        ops = [transforms.Lambda(lambda z: z.repeat(3, 1, 1) if z.shape[0] == 1 else transforms.functional.rgb_to_grayscale(z, 3)),
               transforms.Resize((size, size), antialias=True)]
        if train:
            ops += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(5),
                    transforms.RandomAffine(0, translate=(0.025, 0.025), scale=(0.97, 1.03))]
        ops += [transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize([.485, .456, .406], [.229, .224, .225])]
        self.tf = transforms.Compose(ops)
    def __len__(self): return len(self.frame)
    def __getitem__(self, i):
        row = self.frame.iloc[i]
        return self.tf(read_image(row.path)), self.label[row.subtype], row.path


class PneuNet3(nn.Module):
    def __init__(self):
        super().__init__(); base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.features = base.features; self.attention = CBAM(1280); self.pool = GeM()
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(.35), nn.Linear(1280, 3))
    def forward(self, x): return self.head(self.pool(self.attention(self.features(x))))


@torch.inference_mode()
def predict(model, loader, device):
    model.eval(); ys=[]; probs=[]; paths=[]
    for x, y, p in loader:
        logits = model(x.to(device, non_blocking=True)); probability = torch.softmax(logits, 1)
        ys.extend(y.numpy()); probs.extend(probability.cpu().numpy()); paths.extend(p)
    return np.asarray(ys), np.asarray(probs), paths


def main():
    seed_all(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(ROOT / "data" / "processed" / "manifest.csv")
    train, val, test = [df[df.split == split] for split in ("train", "val", "test")]
    train_loader = DataLoader(ThreeClassXRays(train, True), batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(ThreeClassXRays(val, False), batch_size=24, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(ThreeClassXRays(test, False), batch_size=24, shuffle=False, num_workers=0, pin_memory=True)
    model = PneuNet3().to(device)
    counts = train.subtype.map({"normal":0,"bacterial":1,"viral":2}).value_counts().sort_index().to_numpy()
    weights = torch.tensor(len(train) / (3 * counts), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    checkpoint = ROOT / "models" / "single_stage" / "pneunet_3class.pt"; checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best = -1.; bad = 0; history = []
    for epoch in range(12):
        model.train(); losses=[]
        for x, y, _ in train_loader:
            x=x.to(device, non_blocking=True); y=y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True); loss=loss_fn(model(x), y); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step(); losses.append(loss.item())
        vy, vp, _ = predict(model, val_loader, device); vf1 = f1_score(vy, vp.argmax(1), average="macro")
        history.append({"epoch":epoch+1, "loss":float(np.mean(losses)), "val_macro_f1":vf1})
        print(f"epoch={epoch+1} loss={np.mean(losses):.5f} val_macro_f1={vf1:.5f}", flush=True)
        if vf1 > best + 1e-4:
            best=vf1; bad=0; torch.save({"state":model.state_dict(),"epoch":epoch,"best_val_macro_f1":best},checkpoint)
        else: bad += 1
        if epoch >= 5 and bad >= 3: break
    pd.DataFrame(history).to_csv(ROOT / "artifacts" / "history_single_stage_pneunet_3class.csv", index=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["state"])
    y, p, paths = predict(model, test_loader, device); pred = p.argmax(1); rec = recall_score(y,pred,labels=[0,1,2],average=None)
    result = {"model":"single_stage_pneunet_3class","n":len(y),"accuracy":accuracy_score(y,pred),
              "macro_f1":f1_score(y,pred,average="macro"),"normal_recall":rec[0],
              "bacterial_recall":rec[1],"viral_recall":rec[2],"best_val_macro_f1":best}
    pd.DataFrame({"path":paths,"y_true":y,"p_normal":p[:,0],"p_bacterial":p[:,1],"p_viral":p[:,2],"y_pred":pred}).to_csv(
        ROOT / "artifacts" / "predictions_single_stage_pneunet_3class.csv", index=False)
    (ROOT / "reports" / "single_stage_pneunet_3class_metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__": main()
