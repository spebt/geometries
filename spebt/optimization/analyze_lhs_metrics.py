#!/usr/bin/env python3
"""
Analyze LHS sweep results: metric patterns, design-metric relationships, Pareto front.

Generates:
  1. Pairwise metric correlations (4x4 scatter matrix)
  2. Design parameter → metric relationships (4x4 grid)
  3. Parallel coordinates plot (all objectives, colored by dominance)
  4. Baseline comparison table

Usage:
  python analyze_lhs_metrics.py --csv results/results_summary_mobo.csv --out_dir results/analysis_plots
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from itertools import combinations


# Baseline config
BASELINE = {
    "aperture_diam_mm": 0.4,
    "n_apertures": 180,
    "n_det_ring1": 480,
    "n_det_ring2": 720,
}

METRIC_COLS = ["fwhm_mean", "asci_pct", "sensitivity_mean", "mpxi_mean"]
METRIC_LABELS = ["FWHM (mm)", "ASCI (%)", "Sensitivity", "MPXI"]
# Direction: -1 = minimize (lower is better), +1 = maximize (higher is better)
METRIC_DIRS = [-1, 1, 1, -1]

DESIGN_COLS = ["aperture_diam_mm", "n_apertures", "n_det_ring1", "n_det_ring2"]
DESIGN_LABELS = ["Aperture Diam (mm)", "N Apertures", "N Det Ring 1", "N Det Ring 2"]


def is_pareto_dominant(objectives):
    """
    Find Pareto-optimal points.
    objectives: (n, m) array where all objectives are to be MAXIMIZED.
    Returns boolean mask of non-dominated points.
    """
    n = len(objectives)
    is_optimal = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j >= i on all objectives and j > i on at least one
            if np.all(objectives[j] >= objectives[i]) and np.any(objectives[j] > objectives[i]):
                is_optimal[i] = False
                break
    return is_optimal


def plot_metric_correlations(df, out_dir):
    """Plot 1: Pairwise metric scatter matrix with Pareto coloring."""
    n = len(METRIC_COLS)
    fig, axes = plt.subplots(n, n, figsize=(14, 12))
    fig.suptitle("Pairwise Metric Correlations (LHS Sweep)", fontsize=14, y=0.98)

    # Compute Pareto front
    obj_matrix = df[METRIC_COLS].values.copy()
    for i, d in enumerate(METRIC_DIRS):
        obj_matrix[:, i] *= d  # flip so all are "maximize"
    pareto_mask = is_pareto_dominant(obj_matrix)

    colors = np.where(pareto_mask, "#e74c3c", "#3498db")
    sizes = np.where(pareto_mask, 60, 30)

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                # Histogram on diagonal
                ax.hist(df[METRIC_COLS[i]], bins=15, color="#3498db", alpha=0.7, edgecolor="white")
                ax.set_xlabel(METRIC_LABELS[i], fontsize=8)
            else:
                for k in range(len(df)):
                    ax.scatter(df[METRIC_COLS[j]].iloc[k], df[METRIC_COLS[i]].iloc[k],
                               c=colors[k], s=sizes[k], alpha=0.7, edgecolors="white", linewidth=0.5)

                # Correlation coefficient
                corr = df[METRIC_COLS[j]].corr(df[METRIC_COLS[i]])
                ax.annotate(f"r={corr:.2f}", xy=(0.05, 0.92), xycoords="axes fraction",
                            fontsize=8, color="black",
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

            if j == 0:
                ax.set_ylabel(METRIC_LABELS[i], fontsize=8)
            if i == n - 1:
                ax.set_xlabel(METRIC_LABELS[j], fontsize=8)
            ax.tick_params(labelsize=7)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=8, label='Pareto optimal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=8, label='Dominated'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    path = os.path.join(out_dir, "metric_correlations.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def plot_design_vs_metrics(df, out_dir):
    """Plot 2: How each design parameter affects each metric."""
    n_design = len(DESIGN_COLS)
    n_metric = len(METRIC_COLS)

    fig, axes = plt.subplots(n_metric, n_design, figsize=(16, 12))
    fig.suptitle("Design Parameters → Metrics (LHS Sweep)", fontsize=14, y=0.98)

    for i, (mcol, mlabel) in enumerate(zip(METRIC_COLS, METRIC_LABELS)):
        for j, (dcol, dlabel) in enumerate(zip(DESIGN_COLS, DESIGN_LABELS)):
            ax = axes[i, j]
            ax.scatter(df[dcol], df[mcol], c="#2ecc71", s=35, alpha=0.7, edgecolors="white", linewidth=0.5)

            # Add baseline marker
            if dcol in BASELINE:
                baseline_metric = None  # we don't know baseline metric values
                ax.axvline(BASELINE[dcol], color="red", linestyle="--", alpha=0.5, linewidth=1)

            # Trend line
            if len(df) >= 5:
                z = np.polyfit(df[dcol], df[mcol], 1)
                x_line = np.linspace(df[dcol].min(), df[dcol].max(), 50)
                ax.plot(x_line, np.polyval(z, x_line), color="gray", linestyle="--", alpha=0.6, linewidth=1)

                corr = df[dcol].corr(df[mcol])
                ax.annotate(f"r={corr:.2f}", xy=(0.05, 0.92), xycoords="axes fraction",
                            fontsize=8, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

            if j == 0:
                ax.set_ylabel(mlabel, fontsize=9)
            if i == n_metric - 1:
                ax.set_xlabel(dlabel, fontsize=9)
            ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, "design_vs_metrics.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def plot_parallel_coordinates(df, out_dir):
    """Plot 3: Parallel coordinates with Pareto coloring."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Parallel Coordinates — All Objectives (LHS Sweep)", fontsize=13)

    # Normalize each metric to [0, 1] for visualization
    # Flip sign for minimization objectives so "up" is always "better"
    obj_vals = df[METRIC_COLS].values.copy()
    display_vals = obj_vals.copy()
    for i, d in enumerate(METRIC_DIRS):
        if d == -1:
            display_vals[:, i] = -display_vals[:, i]

    # Normalize to [0,1]
    mins = display_vals.min(axis=0)
    maxs = display_vals.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    normed = (display_vals - mins) / ranges

    # Pareto front
    obj_maximized = obj_vals.copy()
    for i, d in enumerate(METRIC_DIRS):
        obj_maximized[:, i] *= d
    pareto_mask = is_pareto_dominant(obj_maximized)

    x_positions = np.arange(len(METRIC_COLS))

    # Draw dominated first (behind)
    for idx in range(len(df)):
        if not pareto_mask[idx]:
            ax.plot(x_positions, normed[idx], color="#3498db", alpha=0.25, linewidth=1)

    # Draw Pareto on top
    for idx in range(len(df)):
        if pareto_mask[idx]:
            ax.plot(x_positions, normed[idx], color="#e74c3c", alpha=0.8, linewidth=2.5)

    # Axis labels
    direction_arrows = ["↓ better" if d == -1 else "↑ better" for d in METRIC_DIRS]
    labels = [f"{ml}\n({da})" for ml, da in zip(METRIC_LABELS, direction_arrows)]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Normalized value (up = better)", fontsize=10)

    # Add raw value annotations on axes
    for i in range(len(METRIC_COLS)):
        col = obj_vals[:, i]
        if METRIC_DIRS[i] == -1:
            ax.annotate(f"{col.min():.3f}", xy=(i, 1.0), xycoords=("data", "axes fraction"),
                        fontsize=7, ha="center", va="bottom", color="green")
            ax.annotate(f"{col.max():.3f}", xy=(i, 0.0), xycoords=("data", "axes fraction"),
                        fontsize=7, ha="center", va="top", color="red")
        else:
            ax.annotate(f"{col.max():.3g}", xy=(i, 1.0), xycoords=("data", "axes fraction"),
                        fontsize=7, ha="center", va="bottom", color="green")
            ax.annotate(f"{col.min():.3g}", xy=(i, 0.0), xycoords=("data", "axes fraction"),
                        fontsize=7, ha="center", va="top", color="red")

    legend_elements = [
        Line2D([0], [0], color='#e74c3c', linewidth=2.5, label=f'Pareto optimal ({pareto_mask.sum()})'),
        Line2D([0], [0], color='#3498db', linewidth=1, alpha=0.5, label=f'Dominated ({(~pareto_mask).sum()})'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "parallel_coordinates.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def print_summary_table(df):
    """Print metric summary with baseline comparison."""
    # Pareto
    obj_vals = df[METRIC_COLS].values.copy()
    for i, d in enumerate(METRIC_DIRS):
        obj_vals[:, i] *= d
    pareto_mask = is_pareto_dominant(obj_vals)

    print("\n" + "=" * 70)
    print("METRIC SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<20} {'Min':>10} {'Max':>10} {'Mean':>10} {'Baseline':>10}")
    print("-" * 70)
    # We don't have baseline metric values, so just show design params
    for col, label in zip(METRIC_COLS, METRIC_LABELS):
        vals = df[col]
        print(f"{label:<20} {vals.min():>10.4f} {vals.max():>10.4f} {vals.mean():>10.4f} {'—':>10}")

    print(f"\nTotal configs: {len(df)}")
    print(f"Pareto optimal: {pareto_mask.sum()}")

    print("\n" + "=" * 70)
    print("PARETO OPTIMAL CONFIGS")
    print("=" * 70)
    pareto_df = df[pareto_mask]
    display_cols = DESIGN_COLS + METRIC_COLS
    available = [c for c in display_cols if c in pareto_df.columns]
    print(pareto_df[available].to_string(index=False))

    # Pairwise correlations
    print("\n" + "=" * 70)
    print("METRIC PAIRWISE CORRELATIONS")
    print("=" * 70)
    for (c1, l1), (c2, l2) in combinations(zip(METRIC_COLS, METRIC_LABELS), 2):
        corr = df[c1].corr(df[c2])
        strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"
        print(f"  {l1:>15} vs {l2:<15}  r={corr:+.3f}  ({strength})")


def plot_key_tradeoffs(df, out_dir):
    """Plot: Two key trade-off scatter plots side by side.
    Left:  FWHM vs Sensitivity (resolution-sensitivity trade-off)
    Right: ASCI vs MPXI (angular completeness-multiplexing trade-off)
    Pareto front highlighted, baseline marked.
    """
    obj_matrix = df[METRIC_COLS].values.copy()
    for i, d in enumerate(METRIC_DIRS):
        obj_matrix[:, i] *= d
    pareto_mask = is_pareto_dominant(obj_matrix)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Left: FWHM vs Sensitivity ---
    fwhm = df["fwhm_mean"].values
    sens = df["sensitivity_mean"].values

    # Dominated points
    ax1.scatter(fwhm[~pareto_mask], sens[~pareto_mask],
                c="#94b8d4", s=50, alpha=0.6, edgecolors="white", linewidth=0.5,
                label="Dominated", zorder=2)
    # Pareto points
    ax1.scatter(fwhm[pareto_mask], sens[pareto_mask],
                c="#e74c3c", s=80, alpha=0.9, edgecolors="white", linewidth=0.8,
                label="Pareto optimal", zorder=3)

    # Connect Pareto front (sorted by FWHM)
    pareto_idx = np.where(pareto_mask)[0]
    pareto_fwhm = fwhm[pareto_idx]
    pareto_sens = sens[pareto_idx]
    sort_order = np.argsort(pareto_fwhm)
    ax1.plot(pareto_fwhm[sort_order], pareto_sens[sort_order],
             "r--", alpha=0.4, linewidth=1.2, zorder=2)

    # Baseline marker
    baseline_row = df[
        (df["aperture_diam_mm"].between(0.39, 0.41)) &
        (df["n_apertures"].between(178, 182))
    ]
    if len(baseline_row) > 0:
        ax1.scatter(baseline_row["fwhm_mean"].values, baseline_row["sensitivity_mean"].values,
                    c="gold", s=200, marker="*", edgecolors="black", linewidth=1,
                    label="Baseline", zorder=5)

    # Correlation annotation
    corr = df["fwhm_mean"].corr(df["sensitivity_mean"])
    ax1.annotate(f"r = {corr:+.2f}", xy=(0.95, 0.95), xycoords="axes fraction",
                 fontsize=11, ha="right", va="top",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

    # Arrow annotations showing trade-off direction
    ax1.annotate("Better resolution\n(narrower beams)", xy=(0.02, 0.02), xycoords="axes fraction",
                 fontsize=8, ha="left", va="bottom", color="gray", style="italic")
    ax1.annotate("Higher sensitivity\n(more photons)", xy=(0.98, 0.02), xycoords="axes fraction",
                 fontsize=8, ha="right", va="bottom", color="gray", style="italic")

    ax1.set_xlabel("FWHM (mm)  ←  better", fontsize=11)
    ax1.set_ylabel("Sensitivity  →  better", fontsize=11)
    ax1.set_title("Resolution–Sensitivity Trade-off", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.2)

    # --- Right: ASCI vs MPXI ---
    asci = df["asci_pct"].values
    mpxi = df["mpxi_mean"].values

    ax2.scatter(asci[~pareto_mask], mpxi[~pareto_mask],
                c="#94b8d4", s=50, alpha=0.6, edgecolors="white", linewidth=0.5,
                label="Dominated", zorder=2)
    ax2.scatter(asci[pareto_mask], mpxi[pareto_mask],
                c="#e74c3c", s=80, alpha=0.9, edgecolors="white", linewidth=0.8,
                label="Pareto optimal", zorder=3)

    # Connect Pareto front
    pareto_asci = asci[pareto_idx]
    pareto_mpxi = mpxi[pareto_idx]
    sort_order2 = np.argsort(pareto_asci)
    ax2.plot(pareto_asci[sort_order2], pareto_mpxi[sort_order2],
             "r--", alpha=0.4, linewidth=1.2, zorder=2)

    if len(baseline_row) > 0:
        ax2.scatter(baseline_row["asci_pct"].values, baseline_row["mpxi_mean"].values,
                    c="gold", s=200, marker="*", edgecolors="black", linewidth=1,
                    label="Baseline", zorder=5)

    corr2 = df["asci_pct"].corr(df["mpxi_mean"])
    ax2.annotate(f"r = {corr2:+.2f}", xy=(0.95, 0.05), xycoords="axes fraction",
                 fontsize=11, ha="right", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

    ax2.annotate("Better angular\ncoverage", xy=(0.98, 0.95), xycoords="axes fraction",
                 fontsize=8, ha="right", va="top", color="gray", style="italic")
    ax2.annotate("Less signal\nambiguity", xy=(0.02, 0.05), xycoords="axes fraction",
                 fontsize=8, ha="left", va="bottom", color="gray", style="italic")

    ax2.set_xlabel("ASCI (%)  →  better", fontsize=11)
    ax2.set_ylabel("MPXI  ←  better", fontsize=11)
    ax2.set_title("Angular Completeness–Multiplexing Trade-off", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(out_dir, "key_tradeoffs.png")
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_radar_configs(df, out_dir):
    """Plot: Radar/spider chart comparing baseline vs top Pareto configs.
    Shows 3-4 configs overlaid: baseline, best-resolution, best-sensitivity, best-balanced.
    """
    from matplotlib.patches import FancyBboxPatch

    obj_matrix = df[METRIC_COLS].values.copy()
    for i, d in enumerate(METRIC_DIRS):
        obj_matrix[:, i] *= d
    pareto_mask = is_pareto_dominant(obj_matrix)

    # Select representative configs
    configs = {}

    # Baseline
    baseline_row = df[
        (df["aperture_diam_mm"].between(0.39, 0.41)) &
        (df["n_apertures"].between(178, 182))
    ]
    if len(baseline_row) > 0:
        configs["Baseline\n(d=0.4, n=180)"] = baseline_row.iloc[0]

    # Best resolution (lowest FWHM among Pareto)
    pareto_df = df[pareto_mask]
    if len(pareto_df) > 0:
        configs["Best Resolution"] = pareto_df.loc[pareto_df["fwhm_mean"].idxmin()]

        # Best sensitivity (highest sensitivity among Pareto)
        configs["Best Sensitivity"] = pareto_df.loc[pareto_df["sensitivity_mean"].idxmax()]

        # Best balanced (highest sum of normalized objectives)
        obj_pareto = obj_matrix[pareto_mask]
        mins = obj_pareto.min(axis=0)
        maxs = obj_pareto.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1
        normed = (obj_pareto - mins) / ranges
        balanced_idx = normed.sum(axis=1).argmax()
        balanced_row = pareto_df.iloc[balanced_idx]
        # Avoid duplicating if balanced == one of the others
        if (balanced_row.name != configs.get("Best Resolution", pd.Series()).name and
                balanced_row.name != configs.get("Best Sensitivity", pd.Series()).name):
            configs["Best Balanced"] = balanced_row

    if len(configs) < 2:
        print("Not enough configs for radar chart, skipping")
        return

    # Normalize all metrics to [0, 1] where 1 = best
    # For minimize objectives, invert so higher = better
    all_vals = df[METRIC_COLS].values
    col_min = all_vals.min(axis=0)
    col_max = all_vals.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1

    def normalize_row(row):
        vals = row[METRIC_COLS].values.astype(float)
        normed = (vals - col_min) / col_range
        # Flip minimize objectives so 1 = best
        for i, d in enumerate(METRIC_DIRS):
            if d == -1:
                normed[i] = 1.0 - normed[i]
        return normed

    # Radar chart
    categories = ["Resolution\n(FWHM)", "Angular\nCompleteness\n(ASCI)",
                   "Sensitivity", "Low\nMultiplexing\n(MPXI)"]
    n_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = ["#2c3e50", "#e74c3c", "#27ae60", "#3498db", "#f39c12"]
    linestyles = ["-", "-", "-", "-", "--"]

    for idx, (name, row) in enumerate(configs.items()):
        values = normalize_row(row).tolist()
        values += values[:1]  # close polygon
        color = colors[idx % len(colors)]
        ls = linestyles[idx % len(linestyles)]

        ax.plot(angles, values, color=color, linewidth=2.5, linestyle=ls, label=name)
        ax.fill(angles, values, color=color, alpha=0.08)
        # Mark vertices
        ax.scatter(angles[:-1], values[:-1], color=color, s=40, zorder=5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "Best"], fontsize=8, color="gray")
    ax.set_title("SC-SPECT Design Performance Comparison", fontsize=13,
                 fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9,
              frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "radar_comparison.png")
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_sensitivity_heatmap(df, out_dir):
    """Plot: Compact heatmap of design parameter → metric correlations.
    One clean 4x4 grid with r-values annotated and color-coded.
    """
    # Compute correlation matrix (design params vs metrics)
    corr_matrix = np.zeros((len(METRIC_COLS), len(DESIGN_COLS)))
    for i, mcol in enumerate(METRIC_COLS):
        for j, dcol in enumerate(DESIGN_COLS):
            corr_matrix[i, j] = df[mcol].corr(df[dcol])

    fig, ax = plt.subplots(figsize=(8, 6))

    # Use diverging colormap centered at 0
    vmax = max(abs(corr_matrix.min()), abs(corr_matrix.max()), 0.8)
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    # Annotate each cell with r-value
    for i in range(len(METRIC_COLS)):
        for j in range(len(DESIGN_COLS)):
            val = corr_matrix[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            weight = "bold" if abs(val) > 0.4 else "normal"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    fontsize=13, color=color, fontweight=weight)

    ax.set_xticks(range(len(DESIGN_COLS)))
    ax.set_xticklabels(["Aperture\nDiameter", "Number of\nApertures",
                         "Ring 1\nCrystals", "Ring 2\nCrystals"],
                        fontsize=10, ha="center")
    ax.set_yticks(range(len(METRIC_COLS)))
    metric_labels_with_dir = [
        "FWHM (↓ better)", "ASCI (↑ better)",
        "Sensitivity (↑ better)", "MPXI (↓ better)"
    ]
    ax.set_yticklabels(metric_labels_with_dir, fontsize=10)

    ax.set_title("Design Parameter Influence on Imaging Metrics",
                 fontsize=13, fontweight="bold", pad=15)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Pearson Correlation (r)", fontsize=10)

    # Add interpretive note
    ax.text(0.5, -0.18,
            "Strong |r| > 0.5 indicates the design parameter significantly affects the metric",
            transform=ax.transAxes, fontsize=9, ha="center", color="gray", style="italic")

    plt.tight_layout()
    path = os.path.join(out_dir, "design_sensitivity_heatmap.png")
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze LHS metrics")
    parser.add_argument("--csv", type=str, required=True, help="Results CSV path")
    parser.add_argument("--out_dir", type=str, default="results/analysis_plots",
                        help="Output directory for plots")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=METRIC_COLS)
    print(f"Loaded {len(df)} feasible configs from {args.csv}")

    if len(df) < 3:
        print("ERROR: Need at least 3 feasible configs for analysis")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    # Original plots (kept)
    plot_metric_correlations(df, args.out_dir)
    plot_design_vs_metrics(df, args.out_dir)
    plot_parallel_coordinates(df, args.out_dir)

    # New publication-quality plots
    plot_key_tradeoffs(df, args.out_dir)
    plot_radar_configs(df, args.out_dir)
    plot_sensitivity_heatmap(df, args.out_dir)

    print_summary_table(df)

    print(f"\nAll plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
