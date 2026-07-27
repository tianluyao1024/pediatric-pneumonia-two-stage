"""Assemble the prespecified 50:50 Stage-2 development OOF probability fusion.

The two component files must be genuine paired outer-fold predictions produced
by ``stage2_whole_nested_diagnostic``.  This utility never searches weights,
thresholds, or test-set information; it only averages the matched OOF scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def load_component(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"path", "y_true", "probability"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    out = frame[["path", "y_true", "probability"]].copy()
    out = out.rename(columns={"probability": label})
    if out.duplicated(["path", "y_true"]).any():
        raise ValueError(f"{path} contains duplicate OOF rows")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fixed 50:50 paired OOF fusion results.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--imagenet-oof", type=Path, default=None)
    parser.add_argument("--ssl-oof", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    artifacts, reports = root / "artifacts", root / "reports"
    image_path = args.imagenet_oof or artifacts / "oof_predictions_stage2_whole_nested_imagenet_simple.csv"
    ssl_path = args.ssl_oof or artifacts / "oof_predictions_stage2_whole_nested_ssl_simple.csv"
    merged = load_component(image_path, "probability_imagenet").merge(
        load_component(ssl_path, "probability_ssl"), on=["path", "y_true"], how="inner", validate="one_to_one"
    )
    if len(merged) == 0:
        raise ValueError("No paired OOF rows were found")
    merged["probability"] = 0.5 * merged.probability_imagenet + 0.5 * merged.probability_ssl
    y, p = merged.y_true.astype(int), merged.probability
    summary = {
        "model": "fixed_50_50_imagenet_ssl_oof_fusion",
        "evaluation": "paired development OOF point estimates",
        "weight_search": False,
        "exploratory_test_information_used": False,
        "n": int(len(merged)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
    }
    artifacts.mkdir(parents=True, exist_ok=True); reports.mkdir(parents=True, exist_ok=True)
    merged.to_csv(artifacts / "oof_predictions_stage2_fixed_50_50_fusion.csv", index=False)
    (reports / "stage2_fixed_50_50_fusion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
