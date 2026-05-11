"""
ece_plot.py
Week 11 — Riktika (M1)

Reads grid_search_results.json and plots reliability diagrams
for all 5 weight combos side by side.

A reliability diagram shows:
- X axis: confidence score (what the system predicted)
- Y axis: actual accuracy (what really happened)
- Perfect calibration = points sit on the diagonal line
- Gap between bar and diagonal = miscalibration

Usage:
    python src/ece_plot.py
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")   # no display needed — saves to file
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_reliability_diagrams(results_path=None, out_dir=None):
    """
    Load grid search results and plot one reliability diagram per weight combo.
    All 5 diagrams are saved in a single figure side by side.
    """
    # ── Paths ──────────────────────────────────────────────────────────────
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if results_path is None:
        results_path = os.path.join(
            base, "results", "grid_search", "grid_search_results.json"
        )
    if out_dir is None:
        out_dir = os.path.join(base, "results", "grid_search")

    os.makedirs(out_dir, exist_ok=True)

    print(f"[ece_plot] Loading results from: {results_path}")
    with open(results_path) as f:
        results = json.load(f)

    combo_names = list(results.keys())
    n_combos    = len(combo_names)

    # ── Figure setup ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        1, n_combos,
        figsize=(5 * n_combos, 5),
        sharey=True
    )
    if n_combos == 1:
        axes = [axes]

    fig.suptitle(
        "Reliability Diagrams — Week 11 Grid Search\n"
        "Dual-Source Grounded Clinical RAG",
        fontsize=13, fontweight="bold", y=1.02
    )

    colours = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    # ── Plot each combo ────────────────────────────────────────────────────
    summary_rows = []

    for ax, name, colour in zip(axes, combo_names, colours):
        res      = results[name]
        ece      = res["ece"]
        avg_conf = res["avg_conf"]
        weights  = res["weights"]
        bin_data = res["bin_data"]

        # Extract bin info — skip empty bins
        bin_mids   = []
        avg_confs  = []
        avg_correct = []
        counts     = []

        for b in bin_data:
            if b["count"] == 0 or b["avg_conf"] is None:
                continue
            mid = (b["bin_lower"] + b["bin_upper"]) / 2
            bin_mids.append(mid)
            avg_confs.append(b["avg_conf"])
            avg_correct.append(b["avg_correct"])
            counts.append(b["count"])

        bin_width = 0.09   # slightly narrower than 0.1 for visual gap

        # Draw bars = actual accuracy per bin
        bars = ax.bar(
            avg_confs,
            avg_correct,
            width=bin_width,
            color=colour,
            alpha=0.75,
            label="Actual accuracy",
            zorder=2
        )

        # Draw gap between bar top and diagonal (miscalibration area)
        for ac, co in zip(avg_confs, avg_correct):
            lo = min(ac, co)
            hi = max(ac, co)
            ax.bar(
                ac, hi - lo, bottom=lo,
                width=bin_width,
                color="red", alpha=0.25,
                zorder=3
            )

        # Perfect calibration diagonal
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.2,
                label="Perfect calibration", zorder=4)

        # Formatting
        w = weights
        ax.set_title(
            f"{name}\n"
            f"α={w[0]:.2f} β={w[1]:.2f} γ={w[2]:.2f}\n"
            f"ECE = {ece:.4f} | AvgConf = {avg_conf:.4f}",
            fontsize=8.5
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence", fontsize=9)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        if ax == axes[0]:
            ax.set_ylabel("Actual Accuracy", fontsize=9)

        summary_rows.append((name, w, ece, avg_conf, res.get("penalties", "?")))

    # ── Legend ────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(color=colours[0], alpha=0.75, label="Actual accuracy"),
        mpatches.Patch(color="red",      alpha=0.25, label="Miscalibration gap"),
        plt.Line2D([0], [0], color="black", linestyle="--", label="Perfect calibration")
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.08)
    )

    plt.tight_layout()

    # ── Save figure ────────────────────────────────────────────────────────
    out_path = os.path.join(out_dir, "reliability_diagrams.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ece_plot] Reliability diagram saved to: {out_path}")

    # ── Print summary table ────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("ECE SUMMARY TABLE")
    print(f"{'='*75}")
    print(f"{'Combo':<25} {'Weights':<28} {'ECE':>6} {'AvgConf':>9} {'Penalties':>10}")
    print("-"*75)

    best_ece  = float("inf")
    best_name = None

    for name, w, ece, avg_conf, penalties in summary_rows:
        w_str = f"({w[0]:.2f},{w[1]:.2f},{w[2]:.2f})"
        print(f"{name:<25} {w_str:<28} {ece:>6.4f} {avg_conf:>9.4f} {penalties:>10}")
        if ece < best_ece:
            best_ece  = ece
            best_name = name

    print(f"\n🏆 Best combo: {best_name}  ECE = {best_ece}")
    return best_name, best_ece


# ── Also plot a single combo in detail (optional) ─────────────────────────
def plot_single_combo(combo_name, results_path=None):
    """
    Plot a single reliability diagram in larger detail.
    Useful for including in the thesis PDF.
    """
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if results_path is None:
        results_path = os.path.join(
            base, "results", "grid_search", "grid_search_results.json"
        )

    with open(results_path) as f:
        results = json.load(f)

    res      = results[combo_name]
    bin_data = res["bin_data"]
    ece      = res["ece"]
    weights  = res["weights"]

    avg_confs   = []
    avg_correct = []

    for b in bin_data:
        if b["count"] == 0 or b["avg_conf"] is None:
            continue
        avg_confs.append(b["avg_conf"])
        avg_correct.append(b["avg_correct"])

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.bar(avg_confs, avg_correct, width=0.08,
           color="#4C72B0", alpha=0.8, label="Actual accuracy")

    for ac, co in zip(avg_confs, avg_correct):
        lo, hi = min(ac, co), max(ac, co)
        ax.bar(ac, hi - lo, bottom=lo, width=0.08,
               color="red", alpha=0.3, label="_gap")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")
    ax.set_title(
        f"Reliability Diagram — {combo_name}\n"
        f"weights=({weights[0]:.2f},{weights[1]:.2f},{weights[2]:.2f})  ECE={ece:.4f}",
        fontsize=11
    )
    ax.set_xlabel("Confidence", fontsize=11)
    ax.set_ylabel("Actual Accuracy", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()

    out_dir  = os.path.join(base, "results", "grid_search")
    out_path = os.path.join(out_dir, f"reliability_{combo_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ece_plot] Single diagram saved to: {out_path}")


if __name__ == "__main__":
    best_name, best_ece = plot_reliability_diagrams()

    # Also save a large single diagram for the best combo
    print(f"\n[ece_plot] Saving detailed diagram for best combo: {best_name}")
    plot_single_combo(best_name)

    print("\n[ece_plot] Done! Check results/grid_search/ for your PNG files.")
