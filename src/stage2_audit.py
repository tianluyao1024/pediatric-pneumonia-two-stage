"""Stage-2 label/source/image audit. This script never changes labels or files."""
from __future__ import annotations
import argparse, hashlib, json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def avg_hash(gray: np.ndarray, size: int = 16) -> str:
    small = np.asarray(Image.fromarray(gray).resize((size,size), Image.Resampling.BILINEAR), dtype=np.float32)
    return np.packbits((small >= small.mean()).reshape(-1)).tobytes().hex()

def hamming_hex(a: str, b: str) -> int:
    return sum(int(x ^ y).bit_count() for x, y in zip(bytes.fromhex(a), bytes.fromhex(b)))

def image_stats(path: Path) -> Dict[str, object]:
    try:
        with Image.open(path) as im:
            fmt, mode, (w,h) = im.format or "", im.mode, im.size
            gray = np.asarray(im.convert("L").resize((256,256), Image.Resampling.BILINEAR), dtype=np.uint8)
        x = gray.astype(np.float32) / 255.0
        by, bx = max(1, int(.05*x.shape[0])), max(1, int(.05*x.shape[1]))
        border = np.r_[x[:by,:].ravel(), x[-by:,:].ravel(), x[:,:bx].ravel(), x[:,-bx:].ravel()]
        q = np.percentile(x, [1,5,25,50,75,95,99])
        return {
            "read_ok": True, "read_error": "", "format": fmt, "mode": mode,
            "width": w, "height": h, "aspect_ratio": w/max(h,1),
            "mean_intensity": x.mean(), "std_intensity": x.std(),
            "p01": q[0], "p05": q[1], "p25": q[2], "median": q[3],
            "p75": q[4], "p95": q[5], "p99": q[6],
            "dark_fraction": np.mean(x <= .03), "bright_fraction": np.mean(x >= .97),
            "border_dark_fraction": np.mean(border <= .03),
            "border_bright_fraction": np.mean(border >= .97),
            "average_hash_16": avg_hash(gray),
        }
    except Exception as e:
        return {"read_ok": False, "read_error": repr(e)}

def source_of(row: pd.Series, path: Path) -> str:
    if "original_split" in row.index and pd.notna(row["original_split"]):
        return str(row["original_split"])
    parts = [p.lower() for p in path.parts]
    for value in ("train","val","test"):
        if value in parts: return value
    return "unknown"

def load_available_predictions(root: Path, split: str) -> pd.DataFrame | None:
    a = root/"artifacts"
    names = (
        ["val_predictions_stage2_equal_weight_ensemble_selected.csv",
         "val_predictions_stage2_selected.csv",
         "oof_predictions_stage2_group_oof_fixed_stack.csv"]
        if split == "val" else
        ["predictions_stage2_equal_weight_ensemble_selected.csv",
         "predictions_stage2_selected.csv",
         "predictions_stage2_group_oof_fixed_stack.csv"]
    )
    for name in names:
        p = a/name
        if p.exists():
            z = pd.read_csv(p)
            if not {"path","probability"} <= set(z.columns):
                raise ValueError(f"{name}缺少path/probability")
            return z[["path","probability"]].rename(columns={"probability":f"probability_viral_{split}"})
    return None

def group_summary(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    numeric = ["width","height","aspect_ratio","file_size_kb","mean_intensity",
               "std_intensity","border_dark_fraction","border_bright_fraction"]
    g = df.groupby(by, dropna=False)
    out = g.size().rename("n").to_frame()
    for c in numeric:
        if c in df:
            out = out.join(g[c].agg(["mean","std","median","min","max"]).rename(columns=lambda s:f"{c}_{s}"))
    return out.reset_index()

def near_duplicates(df: pd.DataFrame, max_distance: int) -> pd.DataFrame:
    valid = df[df.average_hash_16.notna()]
    buckets = defaultdict(list)
    for i, h in zip(valid.index, valid.average_hash_16.astype(str)):
        buckets[h[:8]].append(i)
    rows = []
    for ids in buckets.values():
        for a_pos, a in enumerate(ids):
            for b in ids[a_pos+1:]:
                d = hamming_hex(str(df.at[a,"average_hash_16"]), str(df.at[b,"average_hash_16"]))
                if d <= max_distance:
                    rows.append({
                        "path_a":df.at[a,"path"],"path_b":df.at[b,"path"],
                        "subtype_a":df.at[a,"subtype"],"subtype_b":df.at[b,"subtype"],
                        "split_a":df.at[a,"split"],"split_b":df.at[b,"split"],
                        "patient_a":df.at[a,"patient"],"patient_b":df.at[b,"patient"],
                        "hash_hamming_distance":d,
                        "cross_label":df.at[a,"subtype"] != df.at[b,"subtype"],
                        "cross_split":df.at[a,"split"] != df.at[b,"split"],
                    })
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",type=Path,required=True)
    ap.add_argument("--high-confidence",type=float,default=.90)
    ap.add_argument("--max-near-duplicate-hamming",type=int,default=8)
    ap.add_argument("--skip-near-duplicate-search",action="store_true")
    args=ap.parse_args()
    root=args.project_root.resolve(); rep=root/"reports"; rep.mkdir(parents=True,exist_ok=True)
    m=pd.read_csv(root/"data"/"processed"/"manifest.csv")
    required={"path","patient","split","stage1","subtype"}
    if not required <= set(m.columns): raise ValueError(f"manifest缺少{sorted(required-set(m.columns))}")
    df=m[m.stage1==1].copy().reset_index(drop=True)
    if set(df.subtype.astype(str))-{"bacterial","viral"}: raise ValueError("Stage-2 subtype异常")
    rows=[]
    for i,r in df.iterrows():
        p=Path(str(r.path)); x=r.to_dict(); x["path"]=str(p); x["source"]=source_of(r,p)
        x["extension"]=p.suffix.lower(); x["file_exists"]=p.exists()
        x["file_size_kb"]=p.stat().st_size/1024 if p.exists() else np.nan
        x["sha256"]=sha256_file(p) if p.exists() else ""
        x.update(image_stats(p) if p.exists() else {"read_ok":False,"read_error":"missing"})
        x["y_true_stage2"]=int(str(r.subtype)=="viral"); rows.append(x)
        if (i+1)%250==0 or i+1==len(df): print(f"[AUDIT] {i+1}/{len(df)}",flush=True)
    audit=pd.DataFrame(rows)
    for split in ("val","test"):
        pred=load_available_predictions(root,split)
        if pred is not None: audit=audit.merge(pred,on="path",how="left",validate="one_to_one")
    audit["available_probability_viral"]=np.nan; audit["prediction_split"]=""
    for split in ("val","test"):
        c=f"probability_viral_{split}"
        if c in audit:
            mask=audit[c].notna(); audit.loc[mask,"available_probability_viral"]=audit.loc[mask,c]
            audit.loc[mask,"prediction_split"]=split
    p=pd.to_numeric(audit.available_probability_viral,errors="coerce")
    y=audit.y_true_stage2.to_numpy(int)
    audit["model_label_discordance"]=np.where(p.notna(),np.where(y==1,1-p,p),np.nan)
    audit["high_confidence_opposite_prediction"]=audit.model_label_discordance>=args.high_confidence

    exact=(audit[audit.sha256!=""].groupby("sha256").agg(
        n=("path","size"),paths=("path",lambda v:"|".join(v)),
        subtypes=("subtype",lambda v:"|".join(sorted(set(map(str,v))))),
        splits=("split",lambda v:"|".join(sorted(set(map(str,v))))),
        patients=("patient",lambda v:"|".join(sorted(set(map(str,v)))))
    ).reset_index())
    exact=exact[exact.n>1].copy()
    exact["cross_label"]=exact.subtypes.str.contains(r"\|",regex=True)
    exact["cross_split"]=exact.splits.str.contains(r"\|",regex=True)

    groups=audit.groupby("patient").agg(
        n=("path","size"),subtypes=("subtype",lambda v:"|".join(sorted(set(map(str,v))))),
        splits=("split",lambda v:"|".join(sorted(set(map(str,v))))),
        sources=("source",lambda v:"|".join(sorted(set(map(str,v))))),
        paths=("path",lambda v:"|".join(v))
    ).reset_index()
    conflicts=groups[groups.subtypes.str.contains(r"\|",regex=True)]
    leakage=groups[groups.splits.str.contains(r"\|",regex=True)]
    near=pd.DataFrame() if args.skip_near_duplicate_search else near_duplicates(audit,args.max_near_duplicate_hamming)
    discord=audit[audit.high_confidence_opposite_prediction].sort_values("model_label_discordance",ascending=False)

    audit.to_csv(rep/"stage2_audit_table.csv",index=False)
    group_summary(audit,["split","subtype"]).to_csv(rep/"stage2_audit_class_summary.csv",index=False)
    group_summary(audit,["source","split","subtype"]).to_csv(rep/"stage2_audit_source_summary.csv",index=False)
    exact.to_csv(rep/"stage2_audit_duplicate_groups.csv",index=False)
    near.to_csv(rep/"stage2_audit_near_duplicate_candidates.csv",index=False)
    conflicts.to_csv(rep/"stage2_audit_group_conflicts.csv",index=False)
    leakage.to_csv(rep/"stage2_audit_split_leakage.csv",index=False)
    discord.to_csv(rep/"stage2_audit_prediction_discordance.csv",index=False)

    summary={
        "n_images":len(audit),"n_groups":audit.patient.nunique(),
        "class_counts":audit.subtype.value_counts().to_dict(),
        "split_counts":audit.split.value_counts().to_dict(),
        "source_counts":audit.source.value_counts().to_dict(),
        "missing_files":int((~audit.file_exists).sum()),
        "unreadable_images":int((~audit.read_ok.fillna(False)).sum()),
        "exact_duplicate_groups":len(exact),
        "exact_duplicate_cross_label_groups":int(exact.cross_label.sum()) if len(exact) else 0,
        "exact_duplicate_cross_split_groups":int(exact.cross_split.sum()) if len(exact) else 0,
        "near_duplicate_candidates":len(near),
        "near_duplicate_cross_label_candidates":int(near.cross_label.sum()) if len(near) else 0,
        "near_duplicate_cross_split_candidates":int(near.cross_split.sum()) if len(near) else 0,
        "filename_group_label_conflicts":len(conflicts),
        "filename_group_split_leakage":len(leakage),
        "high_confidence_discordance":len(discord),
        "warning":"模型不一致仅用于人工复核，禁止自动改标签或删除。",
    }
    (rep/"stage2_audit_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__": main()
