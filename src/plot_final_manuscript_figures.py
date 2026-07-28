"""Final manuscript figures generated from audited, public numerical results.

All uncertainty bars are percentile 95% intervals from 2,000 bootstrap
resamples of subtype-namespaced filename groups. Numerical labels are omitted
from plotted marks to prevent label/marker and label/error-bar collisions; the
complete values are available in the linked CSV tables.
"""
from pathlib import Path
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("PNEUMONIA_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
REPORTS = ROOT / "results"
OUT = ROOT / "figures"

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": .8,
    "legend.frameon": False,
})

BLUE, RED, PURPLE, TEAL, GREEN, ORANGE = "#477EAF", "#A35C56", "#6E5AA5", "#4D8C86", "#5D9A75", "#D58968"


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def forest(ax, data, metric, labels, color, title, xlim):
    values = data[metric].to_numpy()
    low, high = data[f"{metric}_ci_low"].to_numpy(), data[f"{metric}_ci_high"].to_numpy()
    y = np.arange(len(values))
    ax.errorbar(values, y, xerr=np.vstack([values-low, high-values]), fmt="o", color=color,
                ecolor=color, elinewidth=1.0, capsize=2.3, markersize=4.4)
    ax.set_yticks(y, labels, fontsize=6.2); ax.invert_yaxis(); ax.set_xlim(*xlim)
    ax.set_title(title, loc="left", fontsize=8, fontweight="bold")
    ax.set_xlabel("Estimate (95% CI)"); ax.grid(axis="x", color="#D8DEE3", linewidth=.6)


def figure2_model_comparisons():
    data = pd.read_csv(REPORTS / "main_tables_complete_group_ci.csv")
    stage1 = data[data.table.eq("table2_stage1")]
    stage2 = data[data.table.eq("table3_stage2")]
    stage1_labels = ["Deep ensemble", "PneuNet", "ResNet-18", "EfficientNet-B0", "HOG linear SVM"]
    stage2_labels = ["Group OOF stack", "Embedding SVM", "PneuNet-v2", "EfficientNet-B0", "Tuned CER-Net", "CER-Net baseline", "HOG linear SVM"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25), constrained_layout=True)
    forest(axes[0, 0], stage1, "roc_auc", stage1_labels, BLUE, "a  Stage 1 ROC-AUC (n = 575; 424 groups)", (.95, 1.002))
    forest(axes[0, 1], stage1, "pr_auc", stage1_labels, BLUE, "b  Stage 1 PR-AUC", (.95, 1.002))
    forest(axes[1, 0], stage2, "roc_auc", stage2_labels, RED, "c  Conditional Stage 2 ROC-AUC (n = 417; 266 groups)", (.65, .93))
    forest(axes[1, 1], stage2, "pr_auc", stage2_labels, RED, "d  Conditional Stage 2 PR-AUC", (.50, .90))
    save(fig, "figure2_model_comparisons")


def figure3_end_to_end():
    cascade = json.loads((REPORTS / "end_to_end_cascade_metrics.json").read_text(encoding="utf-8"))
    single = json.loads((REPORTS / "single_stage_pneunet_3class_metrics.json").read_text(encoding="utf-8"))
    labels, cm = ["Normal", "Bacterial", "Viral"], np.asarray(cascade["confusion_matrix"])
    fig = plt.figure(figsize=(7.2, 3.02), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.17, 1.05, 1.23])
    ax = fig.add_subplot(grid[0, 0]); ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8, fontweight="bold", color="white" if cm[i, j] > cm.max()*.55 else "#24313D")
    ax.set_xticks(range(3), labels, fontsize=6.3); ax.set_yticks(range(3), labels, fontsize=6.3); ax.tick_params(length=0)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("Reference class"); ax.set_title("a  End-to-end confusion matrix", loc="left", fontsize=8, fontweight="bold")
    for spine in ax.spines.values(): spine.set_visible(False)
    ax = fig.add_subplot(grid[0, 1]); keys = ["normal_recall", "bacterial_recall", "viral_recall"]
    values = np.array([cascade[k] for k in keys]); low = np.array([cascade[f"{k}_ci_low"] for k in keys]); high = np.array([cascade[f"{k}_ci_high"] for k in keys]); y = np.arange(3)
    ax.errorbar(values, y, xerr=np.vstack([values-low, high-values]), fmt="o", color=BLUE, ecolor=BLUE, capsize=2.4, markersize=5)
    ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlim(.42, 1.01); ax.set_xlabel("Recall (95% CI)"); ax.set_title("b  Cascade class recall", loc="left", fontsize=8, fontweight="bold"); ax.grid(axis="x", color="#D8DEE3", linewidth=.6)
    ax = fig.add_subplot(grid[0, 2]); summary = [("Accuracy", "accuracy"), ("Macro-F1", "macro_f1"), ("Normal recall", "normal_recall"), ("Bacterial recall", "bacterial_recall"), ("Viral recall", "viral_recall")]; y = np.arange(len(summary))
    for offset, obj, color, label in [(-.14, cascade, BLUE, "Cascade"), (.14, single, ORANGE, "Single-stage")]:
        vals = np.array([obj[k] for _, k in summary]); lo = np.array([obj[f"{k}_ci_low"] for _, k in summary]); hi = np.array([obj[f"{k}_ci_high"] for _, k in summary])
        ax.errorbar(vals, y+offset, xerr=np.vstack([vals-lo, hi-vals]), fmt="o", color=color, ecolor=color, capsize=2, markersize=4, label=label)
    ax.set_yticks(y, [label.replace(" ", "\n", 1) if "recall" in label else label for label, _ in summary], fontsize=5.7); ax.invert_yaxis(); ax.set_xlim(.42, 1.01); ax.set_xlabel("Estimate (95% CI)"); ax.set_title("c  Cascade versus single-stage", loc="left", fontsize=8, fontweight="bold"); ax.grid(axis="x", color="#D8DEE3", linewidth=.6); ax.legend(loc="upper center", bbox_to_anchor=(.5, -.17), fontsize=5.8, ncol=2, handletextpad=.3, columnspacing=.7)
    save(fig, "figure3_end_to_end_cascade")


def figure4_robustness():
    oof, source = pd.read_csv(REPORTS / "stage2_whole_nested_oof_group_ci.csv"), pd.read_csv(REPORTS / "source_directory_grouped_ci.csv")
    fig = plt.figure(figsize=(7.2, 3.3), constrained_layout=True); grid = fig.add_gridspec(1, 2, width_ratios=[1.02, 1.25])
    ax = fig.add_subplot(grid[0, 0]); labels, y = ["Pediatric\nSimSiam", "ImageNet", "Fixed 50:50\nfusion"], np.arange(3)
    for offset, metric, color, label in [(-.13, "roc_auc", PURPLE, "ROC-AUC"), (.13, "pr_auc", TEAL, "PR-AUC")]:
        v, lo, hi = oof[metric].to_numpy(), oof[f"{metric}_ci_low"].to_numpy(), oof[f"{metric}_ci_high"].to_numpy(); ax.errorbar(v, y+offset, xerr=np.vstack([v-lo, hi-v]), fmt="o", color=color, ecolor=color, capsize=2, markersize=4, label=label)
    ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlim(.62, .90); ax.set_xlabel("Development OOF estimate (95% CI)"); ax.set_title("a  Whole-image development OOF (n = 3,856; 2,387 groups)", loc="left", fontsize=8, fontweight="bold"); ax.grid(axis="x", color="#D8DEE3", linewidth=.6); ax.legend(loc="upper center", bbox_to_anchor=(.5, -.16), ncol=2, fontsize=5.8)
    ax = fig.add_subplot(grid[0, 1]); models, display, y = ["pneunet_v2", "group_oof_fixed_stack", "cernet_resnet50_tuned"], ["PneuNet-v2", "Group OOF stack", "Tuned CER-Net"], np.arange(3)
    for offset, directory, color, label in [(-.13, "train", GREEN, "Original train"), (.13, "test", ORANGE, "Original test")]:
        d = source[(source.source.eq(directory)) & source.model.isin(models)].copy(); d["model"] = pd.Categorical(d["model"], models, ordered=True); d = d.sort_values("model"); v, lo, hi = d.roc_auc.to_numpy(), d.ci95_low.to_numpy(), d.ci95_high.to_numpy(); ax.errorbar(v, y+offset, xerr=np.vstack([v-lo, hi-v]), fmt="o", color=color, ecolor=color, capsize=2, markersize=4, label=label)
    ax.set_yticks(y, display); ax.invert_yaxis(); ax.set_xlim(.68, 1.01); ax.set_xlabel("ROC-AUC (95% CI)"); ax.set_title("b  Source-directory stress test", loc="left", fontsize=8, fontweight="bold"); ax.grid(axis="x", color="#D8DEE3", linewidth=.6); ax.legend(loc="upper center", bbox_to_anchor=(.5, -.16), ncol=2, fontsize=5.8)
    save(fig, "figure4_robustness_audits")


if __name__ == "__main__":
    figure2_model_comparisons(); figure3_end_to_end(); figure4_robustness()
