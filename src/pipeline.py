from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import chi2
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from skimage.feature import hog
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.io import read_image
from tqdm import tqdm


SEED = 20260718


def seed_all(seed: int = SEED) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    # The local RTX 5080/driver combination is stable with fixed cuDNN
    # algorithms; benchmark mode intermittently selects a crashing kernel.
    torch.backends.cudnn.benchmark = False


def patient_id(name: str) -> str:
    n=name.lower(); m = re.match(r"(person\d+)", n)
    # Numeric person identifiers are reused across bacterial and viral filename
    # namespaces in this release, so the etiology prefix is required to avoid
    # incorrectly merging unrelated cases. Repeated images within one namespace
    # remain grouped.
    if m:
        namespace="bacterial" if "bacteria" in n else "viral" if "virus" in n else "pneumonia"
        return f"{namespace}_{m.group(1)}"
    return Path(name).stem.lower()


def subtype(name: str) -> str:
    n = name.lower()
    if "bacteria" in n: return "bacterial"
    if "virus" in n: return "viral"
    return "normal"


def locate_dataset(root: Path) -> Path:
    hits = [p.parent for p in root.rglob("train") if p.is_dir() and (p / "NORMAL").exists() and "__MACOSX" not in p.parts]
    if not hits: raise FileNotFoundError("Could not find extracted chest_xray/train/NORMAL")
    return min(hits,key=lambda p:len(p.parts))


def build_manifest(data_root: Path, out: Path) -> pd.DataFrame:
    rows = []
    for original_split in ("train", "val", "test"):
        for cls in ("NORMAL", "PNEUMONIA"):
            folder = data_root / original_split / cls
            for p in sorted(folder.glob("*.jpeg")):
                rows.append({"path": str(p), "file": p.name, "patient": patient_id(p.name),
                             "original_split": original_split, "stage1": int(cls == "PNEUMONIA"),
                             "subtype": subtype(p.name)})
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(SEED)
    # Rebuild a reproducible 80:10:10 split across the complete dataset.  Grouping
    # by patient identifier prevents multiple radiographs from the same patient
    # from crossing split boundaries.
    df["split"] = ""
    for label in ("normal", "bacterial", "viral"):
        mask = df.subtype == label
        patients = df.loc[mask, "patient"].drop_duplicates().to_numpy()
        rng.shuffle(patients)
        n_test = max(1, round(len(patients) * .10)); n_val = max(1, round(len(patients) * .10))
        test_ids=set(patients[:n_test]); val_ids=set(patients[n_test:n_test+n_val])
        ids=df.loc[mask,"patient"]
        df.loc[mask,"split"]=np.where(ids.isin(test_ids),"test",np.where(ids.isin(val_ids),"val","train"))
    groups={s:set(df[df.split==s].patient) for s in ("train","val","test")}
    assert not (groups["train"]&groups["val"] or groups["train"]&groups["test"] or groups["val"]&groups["test"])
    out.parent.mkdir(parents=True, exist_ok=True); df.to_csv(out, index=False)
    return df


class XRays(Dataset):
    def __init__(self, frame: pd.DataFrame, task: str, train: bool, size: int = 224):
        self.df = frame.reset_index(drop=True); self.task = task
        norm = transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        aug = [transforms.Lambda(lambda z: z.repeat(3,1,1) if z.shape[0]==1 else transforms.functional.rgb_to_grayscale(z,3)),
               transforms.Resize((size, size),antialias=True)]
        if train: aug += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(7),
                          transforms.RandomAffine(0, translate=(.04, .04), scale=(.95, 1.05))]
        aug += [transforms.ConvertImageDtype(torch.float32), norm]
        self.tf = transforms.Compose(aug)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = self.tf(read_image(r.path))
        y = int(r.stage1) if self.task == "stage1" else int(r.subtype == "viral")
        return x, y, r.path


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(*sum(([nn.Conv2d(a,b,3,padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2)]
                                            for a,b in [(3,32),(32,64),(64,128),(128,256)]), []))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(.3), nn.Linear(256, 1))
    def forward(self, x): return self.head(self.features(x)).squeeze(1)


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__(); hidden=max(channels//reduction,16)
        self.mlp=nn.Sequential(nn.Conv2d(channels,hidden,1),nn.ReLU(),nn.Conv2d(hidden,channels,1))
        self.spatial=nn.Conv2d(2,1,7,padding=3)
    def forward(self,x):
        ca=torch.sigmoid(self.mlp(nn.functional.adaptive_avg_pool2d(x,1))+self.mlp(nn.functional.adaptive_max_pool2d(x,1)))
        x=x*ca; sa=torch.sigmoid(self.spatial(torch.cat([x.mean(1,keepdim=True),x.amax(1,keepdim=True)],1)))
        return x*sa


class GeM(nn.Module):
    def __init__(self,p=3.): super().__init__(); self.p=nn.Parameter(torch.tensor(p))
    def forward(self,x): return nn.functional.adaptive_avg_pool2d(x.clamp_min(1e-6).pow(self.p),1).pow(1/self.p)


class PneuNet(nn.Module):
    """EfficientNet-B0 backbone with CBAM and generalized-mean pooling."""
    def __init__(self, attention=True, gem=True):
        super().__init__(); base=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.features=base.features; self.attention=CBAM(1280) if attention else nn.Identity()
        self.pool=GeM() if gem else nn.AdaptiveAvgPool2d(1)
        self.head=nn.Sequential(nn.Flatten(),nn.Dropout(.35),nn.Linear(1280,1))
    def forward(self,x): return self.head(self.pool(self.attention(self.features(x)))).squeeze(1)


def make_model(name: str) -> nn.Module:
    if name == "small_cnn": return SmallCNN()
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT); m.fc = nn.Linear(m.fc.in_features, 1); return m
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1); return m
    if name == "pneunet": return PneuNet(attention=True,gem=True)
    if name == "pneunet_no_attention": return PneuNet(attention=False,gem=True)
    if name == "pneunet_avg_pool": return PneuNet(attention=True,gem=False)
    raise ValueError(name)


def frames_for_task(df: pd.DataFrame, task: str, split: str) -> pd.DataFrame:
    q = df[df.split == split]
    return q if task == "stage1" else q[q.stage1 == 1]


def loader(frame, task, train, batch=64, workers=4):
    ds = XRays(frame, task, train)
    sampler = None
    if train:
        y = frame.stage1.to_numpy() if task == "stage1" else (frame.subtype == "viral").astype(int).to_numpy()
        counts = np.bincount(y, minlength=2); weights = 1 / np.maximum(counts, 1)
        sampler = WeightedRandomSampler(torch.as_tensor(weights[y], dtype=torch.double), len(y), replacement=True)
    return DataLoader(ds, batch_size=batch, sampler=sampler, shuffle=train and sampler is None,
                      num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)


@torch.inference_mode()
def predict_torch(model, dl, device):
    model.eval(); ys=[]; ps=[]; paths=[]
    start=time.perf_counter()
    for x,y,p in dl:
        # reshape(-1) preserves a one-element batch; squeeze(-1) would turn the
        # final batch into a scalar whenever the split size is not divisible by
        # the evaluation batch size.
        prob=torch.sigmoid(model(x.to(device, non_blocking=True)).reshape(-1)).cpu().numpy()
        ys.extend(y.numpy()); ps.extend(prob); paths.extend(p)
    elapsed=time.perf_counter()-start
    return np.asarray(ys), np.asarray(ps), paths, elapsed


def train_deep(name, task, df, root, quick=False, resume=False):
    print(f"DEEP_START {task} {name} resume={resume}",flush=True)
    train_marker=root/"artifacts"/f"train_complete_{task}_{name}.txt"
    prediction_file=root/"artifacts"/f"predictions_{task}_{name}.csv"
    # A completed checkpoint may already have a persisted independent-test
    # inference.  Reuse it on resume so an unrelated later crash never forces
    # repeated model construction/inference for finished experiments.
    if resume and train_marker.exists() and prediction_file.exists():
        saved=pd.read_csv(prediction_file)
        print(f"DEEP_TEST_REUSED {task} {name} n={len(saved)}",flush=True)
        return (saved.y_true.to_numpy(dtype=int), saved.probability.to_numpy(dtype=float),
                saved.path.tolist(), 0.0)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df=frames_for_task(df,task,"train"); val_df=frames_for_task(df,task,"val")
    test_df=frames_for_task(df,task,"test")
    workers=0 if os.name=="nt" else 4
    dl_train=loader(train_df,task,True,8,workers)
    dl_val=loader(val_df,task,False,8,workers); dl_test=loader(test_df,task,False,8,workers)
    model=make_model(name).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    # FP16 backward is unstable for depthwise convolutions on the local
    # Torch 2.8/CUDA 12.8/RTX 5080 stack, so experiments use reproducible FP32.
    use_amp=False
    scaler=torch.amp.GradScaler("cuda",enabled=use_amp)
    loss_fn=nn.BCEWithLogitsLoss(); best=-1.; bad=0; epochs=2 if quick else 12
    ckpt=root/"models"/task/f"{name}.pt"; ckpt.parent.mkdir(parents=True,exist_ok=True)
    hist=[]
    start_epoch=epochs if (resume and ckpt.exists() and train_marker.exists()) else 0
    for epoch in range(start_epoch,epochs):
        model.train(); losses=[]
        # Disable per-batch terminal rendering: on Windows a long PTY progress
        # stream can fill the output buffer and stall otherwise healthy CUDA work.
        for x,y,_ in tqdm(dl_train,desc=f"{task}/{name} e{epoch+1}",leave=False,disable=True):
            x=x.to(device,non_blocking=True); y=y.float().to(device,non_blocking=True); opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda",enabled=use_amp):
                loss=loss_fn(model(x).squeeze(-1),y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); losses.append(loss.item())
        vy,vp,_,_=predict_torch(model,dl_val,device); auc=roc_auc_score(vy,vp)
        hist.append({"epoch":epoch+1,"loss":float(np.mean(losses)),"val_auc":auc})
        if auc>best+1e-4: best=auc; bad=0; torch.save({"state":model.state_dict(),"name":name,"task":task},ckpt)
        else: bad+=1
        if bad>=3: break
    train_marker.write_text("training completed; best checkpoint selected by validation ROC-AUC\n",encoding="utf-8")
    model.load_state_dict(torch.load(ckpt,map_location=device,weights_only=True)["state"])
    y,p,paths,elapsed=predict_torch(model,dl_test,device)
    pd.DataFrame({"path":paths,"y_true":y,"probability":p}).to_csv(prediction_file,index=False)
    print(f"DEEP_TEST_DONE {task} {name} n={len(y)}",flush=True)
    if hist: pd.DataFrame(hist).to_csv(root/"artifacts"/f"history_{task}_{name}.csv",index=False)
    del model, opt, dl_train, dl_val, dl_test
    if device.type=="cuda": torch.cuda.empty_cache()
    return y,p,paths,elapsed


def handcrafted(frame, kind):
    feats=[]
    for path in tqdm(frame.path,desc=kind,leave=False,disable=True):
        x=read_image(path).float().mean(0,keepdim=True).unsqueeze(0)/255.
        a=nn.functional.interpolate(x,size=(128,128),mode="bilinear",align_corners=False)[0,0].numpy()
        if kind=="intensity":
            hist,_=np.histogram(a,bins=32,range=(0,1),density=True)
            feats.append(np.r_[hist,a.mean(),a.std(),np.percentile(a,[5,25,50,75,95])])
        else: feats.append(hog(a,orientations=9,pixels_per_cell=(16,16),cells_per_block=(2,2),feature_vector=True))
    return np.asarray(feats)


def train_baseline(name,task,df,root):
    prediction_file=root/"artifacts"/f"predictions_{task}_{name}.csv"
    if prediction_file.exists():
        saved=pd.read_csv(prediction_file)
        return (saved.y_true.to_numpy(dtype=int), saved.probability.to_numpy(dtype=float),
                saved.path.tolist(), 0.0, {"fit_seconds":0.0})
    tr=frames_for_task(df,task,"train"); te=frames_for_task(df,task,"test")
    ytr=tr.stage1.to_numpy() if task=="stage1" else (tr.subtype=="viral").astype(int).to_numpy()
    yte=te.stage1.to_numpy() if task=="stage1" else (te.subtype=="viral").astype(int).to_numpy()
    kind="intensity" if name=="intensity_logistic" else "hog"
    cache_train=root/"artifacts"/f"features_{task}_{kind}_train.npy"; cache_test=root/"artifacts"/f"features_{task}_{kind}_test.npy"
    if cache_train.exists() and cache_test.exists(): xtr=np.load(cache_train); xte=np.load(cache_test)
    else:
        xtr=handcrafted(tr,kind); xte=handcrafted(te,kind); np.save(cache_train,xtr); np.save(cache_test,xte)
    clf=(make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,class_weight="balanced",C=1.0)) if kind=="intensity"
         else make_pipeline(StandardScaler(),LinearSVC(class_weight="balanced",C=.1)))
    start=time.perf_counter(); clf.fit(xtr,ytr); fit=time.perf_counter()-start
    start=time.perf_counter(); score=clf.predict_proba(xte)[:,1] if hasattr(clf[-1],"predict_proba") else clf.decision_function(xte)
    infer=time.perf_counter()-start
    if score.min()<0 or score.max()>1: score=1/(1+np.exp(-np.clip(score,-30,30)))
    out=root/"models"/task/f"{name}.joblib"; out.parent.mkdir(parents=True,exist_ok=True); joblib.dump(clf,out)
    return yte,score,te.path.tolist(),infer,{"fit_seconds":fit}


def optimal_threshold(y,p):
    fpr,tpr,t=roc_curve(y,p); return float(t[np.argmax(tpr-fpr)])


def bootstrap_ci(y,p,fn,reps=2000):
    rng=np.random.default_rng(SEED); vals=[]
    for _ in range(reps):
        idx=rng.integers(0,len(y),len(y))
        if len(np.unique(y[idx]))<2: continue
        vals.append(fn(y[idx],p[idx]))
    return np.percentile(vals,[2.5,97.5]).tolist()


def metrics(task,model,y,p,seconds,threshold=.5,reps=2000):
    pred=(p>=threshold).astype(int)
    row={"task":task,"model":model,"n":len(y),"threshold":threshold,"accuracy":accuracy_score(y,pred),
         "balanced_accuracy":balanced_accuracy_score(y,pred),"f1":f1_score(y,pred),"roc_auc":roc_auc_score(y,p),
         "pr_auc":average_precision_score(y,p),"sensitivity":((pred[y==1]==1).mean()),
         "specificity":((pred[y==0]==0).mean()),"brier":brier_score_loss(y,p),
         "inference_ms_per_image":seconds/len(y)*1000}
    row["roc_auc_ci95"]=json.dumps(bootstrap_ci(y,p,lambda a,b:roc_auc_score(a,b),reps))
    return row


def plot_results(preds, root):
    import matplotlib.pyplot as plt
    out=root/"reports"/"figures"; out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"Arial","font.size":9,"axes.linewidth":0.8,"figure.facecolor":"white",
                         "axes.facecolor":"white","savefig.bbox":"tight","savefig.facecolor":"white"})
    palette=["#3B4CC0","#688AE8","#9EBEFF","#F7A889","#D64F43","#7A1E2C","#222222"]
    for task in ("stage1","stage2"):
        fig,axs=plt.subplots(1,2,figsize=(11,4.5))
        for i,(name,(y,p,_)) in enumerate(preds[task].items()):
            color=palette[i%len(palette)]; lw=2.2 if name in ("pneunet","deep_ensemble") else 1.2
            fpr,tpr,_=roc_curve(y,p); axs[0].plot(fpr,tpr,label=f"{name} ({roc_auc_score(y,p):.3f})",color=color,lw=lw)
            pr,re,_=precision_recall_curve(y,p); axs[1].plot(re,pr,label=f"{name} ({average_precision_score(y,p):.3f})",color=color,lw=lw)
        axs[0].plot([0,1],[0,1],'k--',lw=.8); axs[0].set(xlabel="False-positive rate",ylabel="True-positive rate",title=f"{task}: ROC")
        axs[1].set(xlabel="Recall",ylabel="Precision",title=f"{task}: precision-recall")
        for ax in axs:
            ax.legend(fontsize=7,frameon=False); ax.grid(False); ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(out/f"{task}_roc_pr.png",dpi=180); plt.close(fig)


def architecture_figure(root):
    import matplotlib.pyplot as plt
    out=root/"reports"/"figures"; out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"Arial","font.size":9,"figure.facecolor":"white"})
    fig,ax=plt.subplots(figsize=(12,4.8)); ax.set_xlim(0,12); ax.set_ylim(0,5); ax.axis("off")
    def box(x,y,w,h,text,color,ec="#333333"):
        from matplotlib.patches import FancyBboxPatch
        p=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.04,rounding_size=0.08",fc=color,ec=ec,lw=.9)
        ax.add_patch(p); ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=8.5)
    def arrow(x1,y1,x2,y2): ax.annotate("",(x2,y2),(x1,y1),arrowprops=dict(arrowstyle="-|>",lw=1,color="#444444"))
    box(.2,2.0,1.25,.8,"Chest X-ray\n224 × 224","#F2F2F2")
    box(1.8,2.0,1.45,.8,"EfficientNet-B0\nbackbone","#DCE8F2")
    box(3.6,2.0,1.35,.8,"Channel\nattention","#D8EBCF")
    box(5.3,2.0,1.35,.8,"Spatial\nattention","#D8EBCF")
    box(7.0,2.0,1.15,.8,"GeM\npooling","#F7E3BD")
    box(8.5,2.0,1.2,.8,"Dropout +\nlinear","#F4CDD0")
    box(10.1,3.2,1.5,.8,"Stage 1\nNormal / pneumonia","#C9D7F0")
    box(10.1,.8,1.5,.8,"Stage 2\nBacterial / viral","#F2C7C1")
    for a,b in [((1.45,2.4),(1.8,2.4)),((3.25,2.4),(3.6,2.4)),((4.95,2.4),(5.3,2.4)),((6.65,2.4),(7,2.4)),((8.15,2.4),(8.5,2.4))]: arrow(*a,*b)
    arrow(9.7,2.55,10.1,3.45); arrow(9.7,2.25,10.1,1.35)
    ax.text(.2,4.45,"PneuNet: two-stage attention-guided classification",fontsize=13,weight="bold",color="#1F374C")
    ax.text(.2,.2,"CBAM = channel and spatial attention; GeM = generalized-mean pooling. Each stage is trained independently.",fontsize=8,color="#555555")
    fig.savefig(out/"pneunet_architecture.png",dpi=300); plt.close(fig)


def evaluate_cascades(preds, df, root):
    """Evaluate the clinically interpretable three-class cascade on the untouched test set."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    label_map={"normal":0,"bacterial":1,"viral":2}; rows=[]
    common=sorted(set(preds["stage1"]) & set(preds["stage2"]))
    test=df[df.split=="test"].copy(); truth={r.path:label_map[r.subtype] for r in test.itertuples()}
    for name in common:
        y1,p1,paths1=preds["stage1"][name]; _,p2,paths2=preds["stage2"][name]
        subprob=dict(zip(paths2,p2)); yt=[]; yp=[]
        for path,prob1 in zip(paths1,p1):
            yt.append(truth[path])
            yp.append(0 if prob1<.5 else (2 if subprob.get(path,.5)>=.5 else 1))
        cm=confusion_matrix(yt,yp,labels=[0,1,2])
        rows.append({"model":name,"n":len(yt),"accuracy":accuracy_score(yt,yp),
                     "macro_f1":f1_score(yt,yp,average="macro"),"weighted_f1":f1_score(yt,yp,average="weighted")})
        fig,ax=plt.subplots(figsize=(5.2,4.4)); sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",cbar=False,ax=ax,
            xticklabels=["Normal","Bacterial","Viral"],yticklabels=["Normal","Bacterial","Viral"])
        ax.set(xlabel="Predicted",ylabel="Reference",title=f"Cascade confusion matrix: {name}")
        fig.tight_layout(); fig.savefig(root/"reports"/"figures"/f"cascade_confusion_{name}.png",dpi=180); plt.close(fig)
    pd.DataFrame(rows).to_csv(root/"reports"/"cascade_metrics.csv",index=False)


def ancillary_analyses(preds, df, root, reps=2000):
    """Calibration, source-folder stress tests, and paired bootstrap comparisons."""
    import matplotlib.pyplot as plt
    out=root/"reports"/"figures"; subgroup=[]; comparisons=[]; rng=np.random.default_rng(SEED)
    for task in ("stage1","stage2"):
        fig,ax=plt.subplots(figsize=(5,4.3)); ax.plot([0,1],[0,1],"--",color="#777777",lw=1)
        for name in [n for n in ("intensity_logistic","hog_linear_svm","resnet18","efficientnet_b0","pneunet","deep_ensemble") if n in preds[task]]:
            y,p,paths=preds[task][name]; frac,mean=calibration_curve(y,p,n_bins=10,strategy="quantile")
            ax.plot(mean,frac,marker="o",ms=3,lw=1.2,label=name)
            meta=df.set_index("path").loc[paths]
            for source in ("train","val","test"):
                idx=np.where(meta.original_split.to_numpy()==source)[0]
                if len(idx)>=20 and len(np.unique(y[idx]))==2:
                    subgroup.append({"task":task,"model":name,"original_source_folder":source,"n":len(idx),"roc_auc":roc_auc_score(y[idx],p[idx])})
        ax.set(xlabel="Mean predicted probability",ylabel="Observed fraction",title=f"{task}: calibration")
        ax.spines[["top","right"]].set_visible(False); ax.legend(frameon=False,fontsize=7); fig.tight_layout()
        fig.savefig(out/f"{task}_calibration.png",dpi=220); plt.close(fig)
        if "pneunet" in preds[task]:
            base="hog_linear_svm" if "hog_linear_svm" in preds[task] else "intensity_logistic"
            y,p_new,_=preds[task]["pneunet"]; p_base=preds[task][base][1]; diffs=[]
            for _ in range(reps):
                idx=rng.integers(0,len(y),len(y))
                if len(np.unique(y[idx]))==2: diffs.append(roc_auc_score(y[idx],p_new[idx])-roc_auc_score(y[idx],p_base[idx]))
            comparisons.append({"task":task,"comparison":f"pneunet - {base}","auc_difference":roc_auc_score(y,p_new)-roc_auc_score(y,p_base),
                                "ci95_low":np.percentile(diffs,2.5),"ci95_high":np.percentile(diffs,97.5)})
    pd.DataFrame(subgroup).to_csv(root/"reports"/"subgroup_metrics.csv",index=False)
    pd.DataFrame(comparisons).to_csv(root/"reports"/"paired_bootstrap_comparisons.csv",index=False)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--project-root",type=Path,required=True); ap.add_argument("--quick",action="store_true"); ap.add_argument("--resume",action="store_true")
    args=ap.parse_args(); root=args.project_root.resolve(); os.chdir(root); seed_all()
    for d in ("artifacts","models","reports/figures","paper"): (root/d).mkdir(parents=True,exist_ok=True)
    data_root=locate_dataset(root/"data"/"raw"/"extracted")
    manifest=root/"data"/"processed"/"manifest.csv"; df=build_manifest(data_root,manifest)
    (root/"artifacts"/"dataset_summary.json").write_text(json.dumps({"total":len(df),"counts":df.groupby(["split","subtype"]).size().unstack(fill_value=0).to_dict()},indent=2),encoding="utf-8")
    names=["small_cnn"] if args.quick else ["small_cnn","resnet18","efficientnet_b0","pneunet","pneunet_no_attention","pneunet_avg_pool"]
    baselines=["intensity_logistic"] if args.quick else ["intensity_logistic","hog_linear_svm"]
    rows=[]; preds={"stage1":{},"stage2":{}}
    reps=100 if args.quick else 2000
    for task in ("stage1","stage2"):
        for name in baselines:
            print(f"BASELINE_START {task} {name}",flush=True)
            y,p,paths,secs,extra=train_baseline(name,task,df,root); preds[task][name]=(y,p,paths)
            rows.append(metrics(task,name,y,p,secs,reps=reps)|extra)
        for name in names:
            y,p,paths,secs=train_deep(name,task,df,root,args.quick,args.resume); preds[task][name]=(y,p,paths)
            rows.append(metrics(task,name,y,p,secs,reps=reps))
        ensemble_names=[n for n in ("resnet18","efficientnet_b0","pneunet") if n in preds[task]]
        if len(ensemble_names)>1:
            y=preds[task][ensemble_names[0]][0]; paths=preds[task][ensemble_names[0]][2]
            p=np.mean([preds[task][n][1] for n in ensemble_names],axis=0); preds[task]["deep_ensemble"]=(y,p,paths)
            rows.append(metrics(task,"deep_ensemble",y,p,0.,reps=reps))
        for name,(y,p,paths) in preds[task].items():
            pd.DataFrame({"path":paths,"y_true":y,"probability":p}).to_csv(root/"artifacts"/f"predictions_{task}_{name}.csv",index=False)
    pd.DataFrame(rows).to_csv(root/"reports"/"metrics.csv",index=False)
    plot_results(preds,root)
    architecture_figure(root)
    evaluate_cascades(preds,df,root)
    ancillary_analyses(preds,df,root,reps)
    print(json.dumps({"status":"complete","gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU","metrics":str(root/'reports'/'metrics.csv')},ensure_ascii=False))


if __name__ == "__main__": main()
