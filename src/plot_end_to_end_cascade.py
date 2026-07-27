"""Publication figure for locked-test end-to-end cascade evaluation."""
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
    bars = ax.barh(np.arange(3), recalls, color=colors, height=0.58)
    ax.set_yticks(np.arange(3), labels); ax.invert_yaxis(); ax.set_xlim(0.5, 1.01)
    ax.set_xlabel("Recall"); ax.set_title("b  Error propagation by class", loc="left", fontweight="bold", fontsize=8)
    ax.axvline(0.8, color="#AAB3BB", lw=0.8, ls="--")
    for bar, value in zip(bars, recalls):
        ax.text(value + 0.012, bar.get_y() + bar.get_height()/2, f"{value:.3f}", va="center", fontsize=7)

    ax = fig.add_subplot(grid[0, 2]); ax.axis("off")
    ax.set_title("c  Single-stage comparison", loc="left", fontweight="bold", fontsize=8)
    ax.text(0.74, 0.94, "Cascade", ha="right", transform=ax.transAxes, fontsize=6.0, color="#657482")
    ax.text(0.98, 0.94, "Single", ha="right", transform=ax.transAxes, fontsize=6.2, color="#657482")
    summary = [("Accuracy","accuracy"),("Macro-F1","macro_f1"),("Normal recall","normal_recall"),("Bacterial recall","bacterial_recall"),("Viral recall","viral_recall")]
    y = 0.79
    for label, key in summary:
        ax.text(0.02, y, label, color="#657482", transform=ax.transAxes, fontsize=5.8)
        ax.text(0.74, y, f"{metrics[key]:.3f}", ha="right", color="#24313D", transform=ax.transAxes, fontsize=7.0, fontweight="bold")
        ax.text(0.98, y, f"{single[key]:.3f}", ha="right", color="#24313D", transform=ax.transAxes, fontsize=7.2, fontweight="bold")
        ax.plot([0.02, 0.98], [y-0.08, y-0.08], transform=ax.transAxes, color="#D8DEE3", lw=0.6)
        y -= 0.145
    ax.text(0.02, 0.01, "Both use the same locked split.\nGroup-bootstrap intervals are reported in text.",
            transform=ax.transAxes, fontsize=5.8, color="#657482", va="bottom")
    save(fig, "figure4_end_to_end_cascade")


if __name__ == "__main__":
    main()
