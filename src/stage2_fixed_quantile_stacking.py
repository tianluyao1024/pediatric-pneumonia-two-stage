"""Validation-fitted stack with a fixed development-set empirical CDF.

Unlike the historical within-test-cohort rank transform, every locked-test
probability is transformed independently against a CDF fitted on validation
predictions only.
"""
from pathlib import Path
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(os.environ.get("PNEUMONIA_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
from src.pipeline import metrics, optimal_threshold
NAMES = ["resnet18", "efficientnet_b0", "pneunet", "pneunet_no_attention",
         "pneunet_avg_pool", "pneunet_hr", "pneunet_v2", "pneunet_eq",
         "pneunet_embedding_svm"]


def merge_predictions(prefix):
    frames = []
    for name in NAMES:
        frame = pd.read_csv(ROOT / "artifacts" / f"{prefix}predictions_stage2_{name}.csv")
        frames.append(frame[["path", "y_true", "probability"]].rename(columns={"probability": name}))
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["path", "y_true"], how="inner", validate="one_to_one")
    return merged


def fit_cdf(values):
    return np.sort(np.asarray(values, dtype=float))


def apply_cdf(values, reference):
    # Fixed midpoint empirical quantile in (0, 1); each case is transformed
    # independently, without access to other test cases or test labels.
    return (np.searchsorted(reference, np.asarray(values), side="right") + 0.5) / (len(reference) + 1.0)


def main():
    val = merge_predictions("val_"); test = merge_predictions("")
    references = {name: fit_cdf(val[name]) for name in NAMES}
    xv = np.column_stack([apply_cdf(val[name], references[name]) for name in NAMES])
    xt = np.column_stack([apply_cdf(test[name], references[name]) for name in NAMES])
    yv = val.y_true.to_numpy(int); yt = test.y_true.to_numpy(int)

    trials = []; fitted = {}
    for c in (.001, .003, .01, .03, .1, 1.0):
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=5000, random_state=20260719)
        model.fit(xv, yv); pv = model.predict_proba(xv)[:, 1]
        trials.append({"C": c, "validation_auc": roc_auc_score(yv, pv)})
        fitted[c] = (model, pv)
    best = max(trials, key=lambda row: (row["validation_auc"], -row["C"]))
    model, pv = fitted[best["C"]]
    pt = model.predict_proba(xt)[:, 1]
    threshold = optimal_threshold(yv, pv)
    result = metrics("stage2", "fixed_quantile_stacked_ensemble", yt, pt, 0.0,
                     threshold=threshold, reps=2000)
    result.update({"validation_auc": best["validation_auc"], "meta_C": best["C"],
                   "transform": "fixed validation empirical CDF",
                   "coefficients": json.dumps(dict(zip(NAMES, model.coef_[0].tolist())))})
    pd.DataFrame(trials).to_csv(ROOT / "reports" / "stage2_fixed_quantile_stack_validation.csv", index=False)
    pd.DataFrame({"path": test.path, "y_true": yt, "probability": pt}).to_csv(
        ROOT / "artifacts" / "predictions_stage2_fixed_quantile_stacked_ensemble.csv", index=False)
    pd.DataFrame([result]).to_csv(
        ROOT / "artifacts" / "metrics_stage2_fixed_quantile_stacked_ensemble.csv", index=False)
    joblib.dump({"model": model, "cdf_references": references, "names": NAMES},
                ROOT / "models" / "stage2" / "fixed_quantile_stacked_ensemble.joblib")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
