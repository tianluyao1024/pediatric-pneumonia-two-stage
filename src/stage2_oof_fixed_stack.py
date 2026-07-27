"""Group-OOF Stage-2 stack with development-fitted fixed transforms.

Four frozen ImageNet backbones provide embeddings. GroupKFold logistic base
learners generate genuine out-of-fold probabilities. Fold-specific empirical
CDFs transform OOF probabilities; a fixed CDF fitted to pooled development OOF
probabilities transforms each locked-test case independently. A cross-fitted
meta-layer selects the threshold, then the final meta-layer is fit on all OOF
features. No test-cohort ranking is used.
"""
from pathlib import Path
import json, sys, time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.io import read_image

ROOT = Path(r"D:\pneumonia_two_stage_study")
ART, REP = ROOT / "artifacts", ROOT / "reports"
SEED, FOLDS = 20260719, 5


class XRays(Dataset):
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)
        self.tf = transforms.Compose([
            transforms.Lambda(lambda z: z.repeat(3,1,1) if z.shape[0] == 1 else transforms.functional.rgb_to_grayscale(z,3)),
            transforms.Resize((224,224), antialias=True), transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    def __len__(self): return len(self.frame)
    def __getitem__(self, i): return self.tf(read_image(self.frame.iloc[i].path)), self.frame.iloc[i].path


def backbones():
    r = models.resnet18(weights=models.ResNet18_Weights.DEFAULT); r.fc = nn.Identity()
    e = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT); e.classifier = nn.Identity()
    d = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT); d.classifier = nn.Identity()
    return {"ResNet18-emb":r, "EfficientNetB0-emb":e, "DenseNet121-emb":d}


@torch.inference_mode()
def extract(model, loader, device):
    model.eval(); chunks=[]
    for x,_ in loader: chunks.append(model(x.to(device,non_blocking=True)).flatten(1).cpu().numpy())
    return np.concatenate(chunks)


def empirical(values): return np.sort(np.asarray(values,float))
def apply_cdf(values, ref): return (np.searchsorted(ref,np.asarray(values),side="right")+.5)/(len(ref)+1.)


def optimal_threshold(y,p):
    thresholds=np.unique(p); best=(0.5,-1.)
    for t in thresholds:
        pred=p>=t; score=balanced_accuracy_score(y,pred)
        if score>best[1]: best=(float(t),score)
    return best[0]


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    manifest=pd.read_csv(ROOT/"data"/"processed"/"manifest.csv")
    dev=manifest[(manifest.stage1==1)&(manifest.split.isin(["train","val"]))].reset_index(drop=True)
    test=manifest[(manifest.stage1==1)&(manifest.split=="test")].reset_index(drop=True)
    ydev=(dev.subtype=="viral").astype(int).to_numpy(); ytest=(test.subtype=="viral").astype(int).to_numpy(); groups=dev.patient.to_numpy()
    cache=ART/"stage2_oof_frozen_embeddings.npz"
    if cache.exists():
        data=np.load(cache); names=json.loads(str(data["names"])); xdev={n:data[f"dev_{i}"] for i,n in enumerate(names)}; xtest={n:data[f"test_{i}"] for i,n in enumerate(names)}
    else:
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dev_loader=DataLoader(XRays(dev),batch_size=24,shuffle=False,num_workers=0,pin_memory=True)
        test_loader=DataLoader(XRays(test),batch_size=24,shuffle=False,num_workers=0,pin_memory=True)
        xdev={};xtest={}; names=[]
        for name,model in backbones().items():
            print(f"extract {name}",flush=True); model=model.to(device); xdev[name]=extract(model,dev_loader,device); xtest[name]=extract(model,test_loader,device); names.append(name); del model; torch.cuda.empty_cache()
        payload={"names":np.array(json.dumps(names))}
        for i,n in enumerate(names): payload[f"dev_{i}"]=xdev[n]; payload[f"test_{i}"]=xtest[n]
        np.savez_compressed(cache,**payload)

    splitter=GroupKFold(n_splits=FOLDS); oof=np.zeros((len(dev),len(names))); test_base=np.zeros((len(test),len(names))); base_models={}
    for j,name in enumerate(names):
        fold_test=[]
        for fold,(tr,va) in enumerate(splitter.split(xdev[name],ydev,groups)):
            clf=make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight="balanced",max_iter=3000,random_state=SEED+fold))
            clf.fit(xdev[name][tr],ydev[tr]); ptr=clf.predict_proba(xdev[name][tr])[:,1]; pva=clf.predict_proba(xdev[name][va])[:,1]
            oof[va,j]=apply_cdf(pva,empirical(ptr)); fold_test.append(clf.predict_proba(xtest[name])[:,1])
        # Fixed test transformation uses only pooled development OOF raw-scale
        # references approximated by fold models' development predictions.
        final=make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight="balanced",max_iter=3000,random_state=SEED))
        final.fit(xdev[name],ydev); dev_ref=final.predict_proba(xdev[name])[:,1]; test_raw=final.predict_proba(xtest[name])[:,1]
        test_base[:,j]=apply_cdf(test_raw,empirical(dev_ref)); base_models[name]=final
        print(f"base {name} oof_auc={roc_auc_score(ydev,oof[:,j]):.4f}",flush=True)

    meta_oof=np.zeros(len(dev))
    for fold,(tr,va) in enumerate(splitter.split(oof,ydev,groups)):
        meta=LogisticRegression(C=.1,class_weight="balanced",max_iter=3000,random_state=SEED+100+fold)
        meta.fit(oof[tr],ydev[tr]); meta_oof[va]=meta.predict_proba(oof[va])[:,1]
    threshold=optimal_threshold(ydev,meta_oof)
    meta=LogisticRegression(C=.1,class_weight="balanced",max_iter=3000,random_state=SEED); meta.fit(oof,ydev)
    ptest=meta.predict_proba(test_base)[:,1]; pred=(ptest>=threshold).astype(int)
    result={"model":"group_oof_fixed_stack","n":len(test),"folds":FOLDS,"threshold":threshold,
            "development_oof_auc":roc_auc_score(ydev,meta_oof),"roc_auc":roc_auc_score(ytest,ptest),
            "pr_auc":average_precision_score(ytest,ptest),"accuracy":accuracy_score(ytest,pred),
            "balanced_accuracy":balanced_accuracy_score(ytest,pred),"viral_sensitivity":recall_score(ytest,pred,pos_label=1),
            "viral_specificity":recall_score(ytest,pred,pos_label=0),"viral_f1":f1_score(ytest,pred)}
    pd.DataFrame({"path":test.path,"y_true":ytest,"probability":ptest}).to_csv(ART/"predictions_stage2_group_oof_fixed_stack.csv",index=False)
    pd.DataFrame({"path":dev.path,"y_true":ydev,"probability":meta_oof,"group":groups}).to_csv(ART/"oof_predictions_stage2_group_oof_fixed_stack.csv",index=False)
    (REP/"group_oof_fixed_stack_metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    joblib.dump({"base_models":base_models,"meta":meta,"threshold":threshold,"names":names},ROOT/"models"/"stage2"/"group_oof_fixed_stack.joblib")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__": main()
