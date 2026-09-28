"""
pipeline_plots.py
===================
Every figure in the paper, built from the results DataFrame. The notebook
only calls these functions, so a figure can always be traced back to
all_results.csv.

Style rules (kept identical in every figure):
- Inside each factor study the baseline C1 is always dark grey with circles,
  the first variant blue with squares, the second orange with triangles.
  Marker shape + line style make the figures readable in greyscale print.
- Sizes fit the IEEE template: one column = 3.5 in, two columns = 7.16 in.
- Each figure is saved as PDF (vector, for LaTeX) and PNG (for quick viewing).
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from pipeline_config import (
    BASELINE_ID, CLASS_NAMES, CONFIGS, FACTOR_STUDIES, FIGURES_DIR, PRETTY_NAMES,
)
from pipeline_metrics import mean_confusion_matrix

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

COL_W, PAGE_W = 3.5, 7.16      # IEEE column / full page width in inches

INK = "#0b0b0b"                 # text and axes
MUTED = "#898781"               # gridlines, reference lines
ROLE_STYLE = {                  # colour follows the role in the study, never the rank
    "baseline": {"color": "#52514e", "marker": "o", "ls": "-"},
    "variant1": {"color": "#2a78d6", "marker": "s", "ls": "--"},
    "variant2": {"color": "#eb6834", "marker": "^", "ls": ":"},
}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
})

STUDY_TITLES = {"architecture": "Architecture", "augmentation": "Augmentation",
                "optimizer": "Optimiser"}


def _roles(factor: str) -> dict:
    """Maps each config in a study to baseline / variant1 / variant2."""
    variants = [c for c in FACTOR_STUDIES[factor] if c != BASELINE_ID]
    return {BASELINE_ID: "baseline", variants[0]: "variant1", variants[1]: "variant2"}


def _label(config_id: str, factor: str) -> str:
    return f"{config_id} {PRETTY_NAMES[CONFIGS[config_id][factor]]}"


def _save(fig, name: str) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, f"{name}.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, f"{name}.png"))


# ---------------------------------------------------------------------------
# Fig. 1 -- effect size vs seed variation
# ---------------------------------------------------------------------------

def plot_effect_sizes(df, summary, name: str = "fig1_effect_sizes"):
    """Mean test accuracy ± std per config (large marker + error bar) with
    the three individual seed runs as small dots, one panel per study. All
    panels share the y-axis, so effect sizes are comparable across factors.
    The dashed line is the baseline C1 mean; the x-axis label of each variant
    gives its difference to C1 in accuracy points."""
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, 2.4), sharey=True)
    s = summary.set_index("config_id")
    base = s.loc[BASELINE_ID, "accuracy_mean"] * 100

    for ax, factor in zip(axes, FACTOR_STUDIES):
        roles = _roles(factor)
        for x, cid in enumerate(FACTOR_STUDIES[factor]):
            st = ROLE_STYLE[roles[cid]]
            runs = df.loc[df["config_id"] == cid, "test_accuracy"].to_numpy() * 100
            jitter = np.linspace(-0.12, 0.12, len(runs))
            ax.scatter(x + 0.22 + jitter, runs, s=10, color=st["color"], alpha=0.55,
                       marker=st["marker"], linewidths=0, zorder=2)
            m, sd = s.loc[cid, "accuracy_mean"] * 100, s.loc[cid, "accuracy_std"] * 100
            ax.errorbar(x, m, yerr=sd, fmt=st["marker"], color=st["color"], ms=5,
                        capsize=3, elinewidth=1.2, zorder=3)
        ax.axhline(base, color=MUTED, ls="--", lw=0.8, zorder=1)
        ax.set_xticks(range(3))
        ticks = []
        for c in FACTOR_STUDIES[factor]:
            name_ = PRETTY_NAMES[CONFIGS[c][factor]].replace(
                "Crop + flip + AutoAugment + erasing", "Strong").replace("SGD (Nesterov)", "SGD")
            delta = "baseline" if c == BASELINE_ID else f"{s.loc[c, 'delta_vs_C1_pp']:+.2f} pp"
            ticks.append(f"{c}\n{name_}\n{delta}")
        ax.set_xticklabels(ticks)
        ax.set_xlim(-0.5, 2.6)
        ax.set_title(STUDY_TITLES[factor])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Test accuracy (%)")
    fig.tight_layout()
    _save(fig, name)
    return fig


# ---------------------------------------------------------------------------
# Fig. 2 -- training curves
# ---------------------------------------------------------------------------

def _curve_stats(df, config_id: str, key: str):
    runs = np.array([h[key] for h in df.loc[df["config_id"] == config_id, "history"]])
    return runs.mean(axis=0), runs.std(axis=0, ddof=1)


def plot_val_accuracy_curves(df, name: str = "fig2a_val_accuracy"):
    """Validation accuracy per epoch, mean over seeds with a ± std band,
    one panel per study (C1 appears in every panel)."""
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, 2.3), sharey=True)
    for ax, factor in zip(axes, FACTOR_STUDIES):
        roles = _roles(factor)
        for cid in FACTOR_STUDIES[factor]:
            st = ROLE_STYLE[roles[cid]]
            mean, sd = _curve_stats(df, cid, "val_acc")
            ep = np.arange(1, len(mean) + 1)
            ax.plot(ep, mean * 100, color=st["color"], ls=st["ls"], lw=1.5,
                    label=_label(cid, factor).replace(
                        "Crop + flip + AutoAugment + erasing", "Strong"))
            ax.fill_between(ep, (mean - sd) * 100, (mean + sd) * 100,
                            color=st["color"], alpha=0.15, linewidth=0)
        ax.set_title(STUDY_TITLES[factor])
        ax.set_xlabel("Epoch")
        ax.legend(loc="lower right", handlelength=2.2)
    axes[0].set_ylabel("Validation accuracy (%)")
    axes[0].set_ylim(30, 100)
    fig.tight_layout()
    _save(fig, name)
    return fig


def plot_augmentation_losses(df, name: str = "fig2b_augmentation_loss"):
    """Train (dashed) vs validation (solid) loss for the augmentation study.
    Shows overfitting without augmentation (train loss -> 0, val loss stays
    high) and train loss above val loss under strong augmentation."""
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, 2.1), sharey=True)
    roles = _roles("augmentation")
    for ax, cid in zip(axes, FACTOR_STUDIES["augmentation"]):
        st = ROLE_STYLE[roles[cid]]
        for key, ls, lab in [("train_loss", "--", "train"), ("val_loss", "-", "validation")]:
            mean, sd = _curve_stats(df, cid, key)
            ep = np.arange(1, len(mean) + 1)
            ax.plot(ep, mean, color=st["color"], ls=ls, lw=1.5, label=lab)
            ax.fill_between(ep, mean - sd, mean + sd, color=st["color"], alpha=0.15, linewidth=0)
        ax.set_title(_label(cid, "augmentation").replace(
            "Crop + flip + AutoAugment + erasing", "Strong aug."))
        ax.set_xlabel("Epoch")
        ax.legend(loc="upper right")
    axes[0].set_ylabel("Cross-entropy loss")
    fig.tight_layout()
    _save(fig, name)
    return fig


# ---------------------------------------------------------------------------
# Fig. 3 -- confusion matrices, best vs worst
# ---------------------------------------------------------------------------

def plot_confusion_pair(df, best_id: str, worst_id: str, name: str = "fig3_confusion"):
    """Row-normalised confusion matrices (each row sums to 100 %, so the
    diagonal is per-class recall), element-wise mean over the 3 seeds.
    One shared colour scale for both panels."""
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 3.6), sharey=True)
    for ax, cid, tag in [(axes[0], best_id, "best"), (axes[1], worst_id, "worst")]:
        cm = mean_confusion_matrix(df.loc[df["config_id"] == cid, "confusion_matrix"].tolist())
        pct = cm / cm.sum(axis=1, keepdims=True) * 100
        im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                v = pct[i, j]
                if i == j or v >= 1.0:          # hide the near-zero clutter
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.5,
                            color="white" if v > 55 else INK)
        ax.set_xticks(range(10)); ax.set_yticks(range(10))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
        ax.set_yticklabels(CLASS_NAMES)
        ax.tick_params(axis="y", labelleft=(ax is axes[0]), left=(ax is axes[0]))
        ax.set_xlabel("Predicted class")
        arch = PRETTY_NAMES[CONFIGS[cid]["architecture"]]
        ax.set_title(f"{cid} {arch} ({tag})")
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_ylabel("True class")
    fig.subplots_adjust(wspace=0.06)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("% of true class")
    cbar.outline.set_visible(False)
    _save(fig, name)
    return fig


# ---------------------------------------------------------------------------
# Fig. 4 (optional) -- accuracy vs training cost
# ---------------------------------------------------------------------------

# C1, C6 and C7 sit almost on top of each other; nudge their labels apart.
LABEL_OFFSETS = {"C7": (-5, 3, "right"), "C6": (5, -3, "left"), "C1": (0, 9, "center")}


def plot_accuracy_vs_cost(summary, name: str = "fig4_accuracy_vs_cost"):
    """Mean test accuracy ± std against mean time per epoch (log scale),
    one point per config, labelled directly."""
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    for _, r in summary.iterrows():
        x, y, sd = r["time_per_epoch_mean"], r["accuracy_mean"] * 100, r["accuracy_std"] * 100
        ax.errorbar(x, y, yerr=sd, fmt="o", color=ROLE_STYLE["variant1"]["color"],
                    ms=4, capsize=2, elinewidth=1)
        dx, dy, ha = LABEL_OFFSETS.get(r["config_id"], (5, 0, "left"))
        ax.annotate(r["config_id"], (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", fontsize=7)
    ax.set_xscale("log")
    ax.set_xticks([4, 6, 10, 20, 30]); ax.set_xticklabels(["4", "6", "10", "20", "30"])
    ax.minorticks_off()
    ax.set_xlabel("Training time per epoch (s, log scale, A100)")
    ax.set_ylabel("Test accuracy (%)")
    fig.tight_layout()
    _save(fig, name)
    return fig
