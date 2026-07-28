"""Publication figure for fixed exploratory-test end-to-end cascade evaluation."""
from pathlib import Path
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r"D:\pneumonia_two_stage_study")
REPORTS = ROOT / "reports"
OUT = REPORTS / "figures" / "nature_redesign"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.8,
})


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=400, bbox_inches="tight")


def main():
    metrics = json.loads((REPORTS / "end_to_end_cascade_metrics.json").read_text(encoding="utf-8"))
    single = json.loads((REPORTS / "single_stage_pneunet_3class_metrics.json").read_text(encoding="utf-8"))
    cm = np.asarray(metrics["confusion_matrix"])
    labels = ["Normal", "Bacterial", "Viral"]
    recalls = np.array([metrics["normal_recall"], metrics["bacterial_recall"], metrics["viral_recall"]])
    recall_low = np.array([metrics["normal_recall_ci_low"], metrics["bacterial_recall_ci_low"], metrics["viral_recall_ci_low"]])
    recall_high = np.array([metrics["normal_recall_ci_high"], metrics["bacterial_recall_ci_high"], metrics["viral_recall_ci_high"]])
    recall_xerr = np.vstack([recalls - recall_low, recall_high - recalls])

    fig = plt.figure(figsize=(7.20, 2.72), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.34, 1.05, 0.90])

    ax = fig.add_subplot(grid[0, 0])
    image = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.55 else "#24313D", fontsize=8, fontweight="bold")
    ax.set_xticks(range(3), labels); ax.set_yticks(range(3), labels)
    ax.set_xlabel("Predicted final class"); ax.set_ylabel("Reference class")
    ax.set_title("a  End-to-end confusion matrix (n = 575)", loc="left", fontweight="bold", fontsize=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values(): spine.set_visible(False)

    ax = fig.add_subplot(grid[0, 1])
    colors = ["#6F9FC8", "#7FAE8A", "#D88770"]
    bars = ax.barh(np.arange(3), recalls, color=colors, height=0.58,
                   xerr=recall_xerr, error_kw={"ecolor": "#24313D", "lw": 0.8, "capsize": 2})
    ax.set_yticks(np.arange(3), labels); ax.invert_yaxis(); ax.set_xlim(0.5, 1.01)
    ax.set_xlabel("Recall (95% group-bootstrap CI)"); ax.set_title("b  Error propagation by class", loc="left", fontweight="bold", fontsize=8)
    ax.axvline(0.8, color="#AAB3BB", lw=0.8, ls="--")
    for bar, value in zip(bars, recalls):
        ax.text(value + 0.012, bar.get_y() + bar.get_height()/2, f"{value:.3f}", va="center", fontsize=7)

    ax = fig.add_subplot(grid[0, 2])
    ax.set_title("c  Cascade versus single-stage", loc="left", fontweight="bold", fontsize=8)
    summary = [("Accuracy","accuracy"),("Macro-F1","macro_f1"),("Normal recall","normal_recall"),("Bacterial recall","bacterial_recall"),("Viral recall","viral_recall")]
    y = np.arange(len(summary))
    cascade_values = np.array([metrics[key] for _, key in summary])
    single_values = np.array([single[key] for _, key in summary])
    cascade_err = np.vstack([
        cascade_values - np.array([metrics[f"{key}_ci_low"] for _, key in summary]),
        np.array([metrics[f"{key}_ci_high"] for _, key in summary]) - cascade_values,
    ])
    single_err = np.vstack([
        single_values - np.array([single[f"{key}_ci_low"] for _, key in summary]),
        np.array([single[f"{key}_ci_high"] for _, key in summary]) - single_values,
    ])
    ax.errorbar(cascade_values, y - 0.13, xerr=cascade_err, fmt="o", ms=3.6, color="#2F78B7",
                ecolor="#2F78B7", elinewidth=0.8, capsize=1.8, label="Cascade")
    ax.errorbar(single_values, y + 0.13, xerr=single_err, fmt="o", ms=3.6, color="#D88770",
                ecolor="#D88770", elinewidth=0.8, capsize=1.8, label="Single-stage")
    ax.set_yticks(y, [label.replace(" recall", "\nrecall") for label, _ in summary], fontsize=5.6)
    ax.invert_yaxis(); ax.set_xlim(0.42, 1.02); ax.set_xticks([0.5, 0.75, 1.0])
    ax.set_xlabel("Estimate (95% CI)", fontsize=6.2)
    ax.grid(axis="x", color="#D8DEE3", lw=0.55)
    ax.legend(loc="lower left", fontsize=5.2, frameon=False, handletextpad=0.3, borderaxespad=0.2)
    ax.tick_params(axis="x", labelsize=5.8, length=2)
    save(fig, "figure4_end_to_end_cascade")


if __name__ == "__main__":
    main()
