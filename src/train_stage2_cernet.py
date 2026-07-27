"""Patient-disjoint CER-Net stage-2 adaptation based on the supplied thesis."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.io import read_image

from src.pipeline import metrics, seed_all


class CERDataset(Dataset):
    def __init__(self, frame, train=False):
        self.frame=frame.reset_index(drop=True)
        ops=[transforms.Lambda(lambda z:z.repeat(3,1,1) if z.shape[0]==1 else transforms.functional.rgb_to_grayscale(z,3)),
             transforms.Resize((224,224),antialias=True)]
        if train:
            ops += [transforms.RandomHorizontalFlip(.5),transforms.RandomRotation(8),
                    transforms.RandomAffine(0,translate=(.03,.03),scale=(.97,1.03)),
                    transforms.ColorJitter(brightness=.15,contrast=.15),
                    transforms.RandomAdjustSharpness(1.5,p=.3)]
        ops += [transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize([.485,.456,.406],[.229,.224,.225])]
        self.tf=transforms.Compose(ops)
    def __len__(self): return len(self.frame)
    def __getitem__(self,index):
        row=self.frame.iloc[index]
        return self.tf(read_image(row.path)),int(row.subtype=="viral"),row.path


class CERStage2(nn.Module):
    def __init__(self):
        super().__init__(); base=models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        base.fc=nn.Identity(); self.backbone=base
        self.head=nn.Sequential(nn.Linear(2048,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(.50),
                                nn.Linear(512,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(.30),
                                nn.Linear(128,2))
    def forward(self,x): return self.head(self.backbone(x))


def mixup(x,y,alpha=.1):
    lam=np.random.beta(alpha,alpha); order=torch.randperm(x.size(0),device=x.device)
    return lam*x+(1-lam)*x[order],y,y[order],lam


@torch.inference_mode()
def predict(model,dl,device,temperature=1.0):
    model.eval(); ys=[]; probs=[]; confidences=[]; paths=[]
    for x,y,p in dl:
        logits=model(x.to(device))/temperature; soft=torch.softmax(logits,1)
        ys.extend(y.numpy()); probs.extend(soft[:,1].cpu().numpy()); confidences.extend(soft.max(1).values.cpu().numpy()); paths.extend(p)
    return np.asarray(ys),np.asarray(probs),np.asarray(confidences),paths


def pr_threshold(y,p):
    precision,recall,thresholds=precision_recall_curve(y,p)
    f1=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-9)
    return float(thresholds[np.nanargmax(f1)])


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--project-root",required=True); parser.add_argument("--epochs",type=int,default=18)
    parser.add_argument("--variant",choices=["paper","tuned"],default="paper"); args=parser.parse_args()
    root=Path(args.project_root); seed_all(); device=torch.device("cuda")
    df=pd.read_csv(root/"data"/"processed"/"manifest.csv"); df=df[df.stage1==1]
    splits={s:df[df.split==s] for s in ("train","val","test")}
    ytrain=(splits["train"].subtype=="viral").astype(int).to_numpy(); counts=np.bincount(ytrain,minlength=2)
    sampler=WeightedRandomSampler(torch.as_tensor((1/np.maximum(counts,1))[ytrain],dtype=torch.double),len(ytrain),replacement=True)
    train_dl=DataLoader(CERDataset(splits["train"],True),batch_size=32,sampler=sampler,num_workers=0,pin_memory=True,drop_last=True)
    val_dl=DataLoader(CERDataset(splits["val"]),batch_size=32,shuffle=False,num_workers=0,pin_memory=True)
    test_dl=DataLoader(CERDataset(splits["test"]),batch_size=32,shuffle=False,num_workers=0,pin_memory=True)
    model_name="cernet_resnet50" if args.variant=="paper" else "cernet_resnet50_tuned"
    model=CERStage2().to(device); checkpoint=root/"models"/"stage2"/f"{model_name}.pt"; history_file=root/"artifacts"/f"history_stage2_{model_name}.csv"
    start=0; best=-1.; bad=0; history=[]
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location=device,weights_only=True); model.load_state_dict(saved["state"])
        start=int(saved.get("epoch",-1))+1; best=float(saved.get("best_val_auc",-1.)); history=pd.read_csv(history_file).to_dict("records") if history_file.exists() else []
        print(f"CER-Net resume_epoch={start+1} best_val_auc={best:.6f}",flush=True)
    loss_fn=nn.CrossEntropyLoss(weight=torch.tensor([1.3,.6] if args.variant=="paper" else [.6,1.3],device=device))
    for epoch in range(start,args.epochs):
        optimizer=torch.optim.AdamW([{"params":model.backbone.parameters(),"lr":1e-6 if args.variant=="paper" else 1e-5},
                                     {"params":model.head.parameters(),"lr":5e-5}],weight_decay=2e-4)
        model.train(); losses=[]
        for x,y,_ in train_dl:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); optimizer.zero_grad(set_to_none=True)
            if epoch<10:
                xm,ya,yb,lam=mixup(x,y,.1); logits=model(xm); loss=lam*loss_fn(logits,ya)+(1-lam)*loss_fn(logits,yb)
            else: loss=loss_fn(model(x),y)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step(); losses.append(loss.item())
        vy,vp,_,_=predict(model,val_dl,device); auc=roc_auc_score(vy,vp)
        history.append({"epoch":epoch+1,"loss":float(np.mean(losses)),"val_auc":auc}); pd.DataFrame(history).to_csv(history_file,index=False)
        print(f"CER-Net epoch={epoch+1} loss={np.mean(losses):.5f} val_auc={auc:.6f}",flush=True)
        if auc>best+1e-4:
            best=auc; bad=0; torch.save({"state":model.state_dict(),"epoch":epoch,"best_val_auc":best},checkpoint)
        else: bad+=1
        if bad>=5: break
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True)["state"])
    # Thesis settings: fixed temperature T=1.3, validation PR threshold and 0.57 confidence rejection.
    vy,vp,vc,_=predict(model,val_dl,device,temperature=1.3); threshold=pr_threshold(vy,vp)
    y,p,confidence,paths=predict(model,test_dl,device,temperature=1.3)
    result=metrics("stage2",model_name,y,p,0.,threshold=threshold,reps=2000)
    accepted=confidence>=.57; pred=(p>=threshold).astype(int)
    result.update({"validation_auc":roc_auc_score(vy,vp),"temperature":1.3,"pr_threshold":threshold,
                   "uncertainty_threshold":.57,"coverage":float(accepted.mean()),
                   "selective_accuracy":accuracy_score(y[accepted],pred[accepted]) if accepted.any() else np.nan,
                   "selective_f1":f1_score(y[accepted],pred[accepted]) if accepted.any() else np.nan})
    pd.DataFrame({"path":paths,"y_true":y,"probability":p,"confidence":confidence,"accepted":accepted}).to_csv(
        root/"artifacts"/f"predictions_stage2_{model_name}.csv",index=False)
    pd.DataFrame([result]).to_csv(root/"artifacts"/f"metrics_stage2_{model_name}.csv",index=False)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__": main()
