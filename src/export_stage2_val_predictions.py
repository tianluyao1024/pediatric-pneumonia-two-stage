"""Export validation probabilities for leakage-free stacking."""

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.pipeline import frames_for_task, loader, make_model, predict_torch, seed_all
from src.train_stage2_optimized import ConvNextEtiologyNet, DenseEtiologyNet, EfficientEtiologyNet, Stage2XRays, predict
from torch.utils.data import DataLoader


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--project-root",required=True); args=parser.parse_args()
    root=Path(args.project_root); seed_all(); device=torch.device("cuda")
    manifest=pd.read_csv(root/"data"/"processed"/"manifest.csv"); val=frames_for_task(manifest,"stage2","val")
    for name in ("resnet18","efficientnet_b0","pneunet","pneunet_no_attention","pneunet_avg_pool"):
        out=root/"artifacts"/f"val_predictions_stage2_{name}.csv"
        if out.exists(): continue
        model=make_model(name).to(device)
        model.load_state_dict(torch.load(root/"models"/"stage2"/f"{name}.pt",map_location=device,weights_only=True)["state"])
        y,p,paths,_=predict_torch(model,loader(val,"stage2",False,8,0),device)
        pd.DataFrame({"path":paths,"y_true":y,"probability":p}).to_csv(out,index=False)
        print(f"exported {name} n={len(y)}",flush=True)
        del model; torch.cuda.empty_cache()
    for name in ("pneunet_hr","pneunet_v2","pneunet_eq"):
        out=root/"artifacts"/f"val_predictions_stage2_{name}.csv"
        if out.exists(): continue
        preserve=name in ("pneunet_v2","pneunet_eq"); contrast=name=="pneunet_eq"
        model=(ConvNextEtiologyNet() if name=="pneunet_v2" else EfficientEtiologyNet() if contrast else DenseEtiologyNet(True)).to(device)
        model.load_state_dict(torch.load(root/"models"/"stage2"/f"{name}.pt",map_location=device,weights_only=True)["state"])
        dl=DataLoader(Stage2XRays(val,False,320,preserve,contrast),batch_size=8,shuffle=False,num_workers=0,pin_memory=True)
        y,p,paths=predict(model,dl,device,True)
        pd.DataFrame({"path":paths,"y_true":y,"probability":p}).to_csv(out,index=False)
        print(f"exported {name} n={len(y)}",flush=True)
        del model; torch.cuda.empty_cache()


if __name__=="__main__": main()
