"""Regenerate the manuscript's model-comparison forest plots.

Reads only sanitised public result tables and expresses all uncertainty as
percentile 95% filename-group bootstrap confidence intervals.
"""
from pathlib import Path
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("PNEUMONIA_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
RESULTS = ROOT / "results"
OUT = ROOT / "figures"
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7, "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.8, "legend.frameon": False})
COLORS = {"stage1": "#477EAF", "stage2": "#A35C56", "oof": "#6E5AA5", "source_train": "#5D9A75", "source_test": "#D58968"}


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight")


def forest(ax, frame, metric, labels, color, title, xlim):
    values, low, high = (frame[key].to_numpy() for key in (metric, f"{metric}_ci_low", f"{metric}_ci_high"))
    y = np.arange(len(frame))
    ax.errorbar(values, y, xerr=np.vstack([values - low, high - values]), fmt="o", color=color, ecolor=color, elinewidth=1.05, capsize=2.2, markersize=4.5)
    ax.set_yticks(y, labels, fontsize=6.2); ax.invert_yaxis(); ax.set_xlim(*xlim); ax.set_title(title, loc="left", fontweight="bold", fontsize=8); ax.set_xlabel("Estimate (95% CI)"); ax.grid(axis="x", color="#D8DEE3", lw=0.6)
    for value, yy in zip(values, y): ax.text(min(value + 0.003, xlim[1] - 0.001), yy, f"{value:.3f}", va="center", fontsize=5.8)


def model_comparisons():
    data = pd.read_csv(RESULTS / "main_tables_complete_group_ci.csv")
    s1, s2 = data[data.table.eq("table2_stage1")], data[data.table.eq("table3_stage2")]
    s1_labels = ["Deep ensemble", "PneuNet", "ResNet-18", "EfficientNet-B0", "HOG linear SVM"]
    s2_labels = ["Group OOF stack", "Embedding SVM", "PneuNet-v2", "EfficientNet-B0", "Tuned CER-Net", "CER-Net baseline", "HOG linear SVM"]
    fig, axes = plt.subplots(2, 2, figsize=(7.20, 5.25), constrained_layout=True)
    forest(axes[0, 0], s1, "roc_auc", s1_labels, COLORS["stage1"], "a  Stage 1 ROC-AUC (n = 575; 424 groups)", (0.95, 1.002))
    forest(axes[0, 1], s1, "pr_auc", s1_labels, COLORS["stage1"], "b  Stage 1 PR-AUC", (0.95, 1.002))
    forest(axes[1, 0], s2, "roc_auc", s2_labels, COLORS["stage2"], "c  Conditional Stage 2 ROC-AUC (n = 417; 266 groups)", (0.65, 0.93))
    forest(axes[1, 1], s2, "pr_auc", s2_labels, COLORS["stage2"], "d  Conditional Stage 2 PR-AUC", (0.50, 0.90))
    save(fig, "figure3_stage1_stage2_model_comparisons")


def oof_and_source_audit():
    oof, src = pd.read_csv(RESULTS / "stage2_whole_nested_oof_group_ci.csv"), pd.read_csv(RESULTS / "source_directory_grouped_ci.csv")
    fig = plt.figure(figsize=(7.20, 3.10), constrained_layout=True); grid = fig.add_gridspec(1, 2, width_ratios=[1.03, 1.25])
    ax = fig.add_subplot(grid[0, 0]); labels, y = ["Pediatric\nSimSiam", "ImageNet", "Fixed 50:50\nfusion"], np.arange(3)
    for offset, metric, color, label in [(-.13, "roc_auc", "#6E5AA5", "ROC-AUC"), (.13, "pr_auc", "#4D8C86", "PR-AUC")]:
        values, low, high = (oof[key].to_numpy() for key in (metric, f"{metric}_ci_low", f"{metric}_ci_high")); ax.errorbar(values, y + offset, xerr=np.vstack([values-low, high-values]), fmt="o", color=color, ecolor=color, capsize=2, markersize=4, label=label)
    ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlim(.62, .90); ax.set_xlabel("Development OOF estimate (95% CI)"); ax.set_title("a  Whole-image development OOF (n = 3,856; 2,387 groups)", loc="left", fontweight="bold", fontsize=8); ax.grid(axis="x", color="#D8DEE3", lw=.6); ax.legend(loc="lower right", fontsize=6)
    ax = fig.add_subplot(grid[0, 1]); models, display, y = ["pneunet_v2", "group_oof_fixed_stack", "cernet_resnet50_tuned"], ["PneuNet-v2", "Group OOF stack", "Tuned CER-Net"], np.arange(3)
    for offset, source, color, label in [(-.13, "train", COLORS["source_train"], "Original train"), (.13, "test", COLORS["source_test"], "Original test")]:
        d = src[(src.source.eq(source)) & (src.model.isin(models))].copy(); d["model"] = pd.Categorical(d["model"], models, ordered=True); d = d.sort_values("model"); values, low, high = d.roc_auc.to_numpy(), d.ci95_low.to_numpy(), d.ci95_high.to_numpy(); ax.errorbar(values, y + offset, xerr=np.vstack([values-low, high-values]), fmt="o", color=color, ecolor=color, capsize=2, markersize=4, label=label)
    ax.set_yticks(y, display); ax.invert_yaxis(); ax.set_xlim(.68, 1.01); ax.set_xlabel("ROC-AUC (95% CI)"); ax.set_title("b  Source-directory stress test", loc="left", fontweight="bold", fontsize=8); ax.grid(axis="x", color="#D8DEE3", lw=.6); ax.legend(loc="lower left", fontsize=6)
    save(fig, "figure4_oof_and_source_audit")


if __name__ == "__main__":
    model_comparisons(); oof_and_source_audit()
