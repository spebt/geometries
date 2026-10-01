#!/usr/bin/env python3
"""
MOBO Convergence Analysis and Plotting.

Reads results_summary_mobo.csv and produces:
  1. Hypervolume vs iteration (convergence curve)
  2. Pareto front expansion (LHS vs LHS+MOBO)
  3. Pairwise objective scatter with Pareto coloring (combined data)
  4. Parallel coordinates (all configs, LHS vs MOBO colored)
  5. Summary table of top Pareto configs

Usage:
  python analyze_mobo_convergence.py --csv results/results_summary_mobo.csv
  python analyze_mobo_convergence.py --csv results/results_summary_mobo.csv --out_dir plots/mobo
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from itertools import combinations


# Imported rather than restated. This copy previously drifted and produced a
# hypervolume curve for metrics the optimizer had already stopped using, which
# is worse than useless for judging convergence.
from mobo_agent import OBJ_COLUMNS as METRIC_COLS
from mobo_agent import OBJ_SHORT as METRIC_LABELS
from mobo_agent import OBJ_DIRECTIONS as METRIC_DIRS

DESIGN_COLS = ["aperture_diam_mm", "n_apertures", "n_det_ring1", "n_det_ring2"]
DESIGN_LABELS = ["Aperture Diam (mm)", "N Apertures", "N Det Ring 1", "N Det Ring 2"]

# Reference point for hypervolume (worst acceptable values in MAXIMIZATION space)
# These should be worse than any observed value after sign-flipping
# Hypervolume reference point, derived from the data rather than hardcoded.
# Hardcoded values silently go stale whenever an objective is redefined, and a
# reference point that no longer bounds the data produces a meaningless
# hypervolume. REF_MARGIN pushes it just past the worst observed value in each
# objective's minimisation-space direction.
REF_MARGIN = 0.05

# Number of points to evaluate along the hypervolume curve. Exact 5D hypervolume
# is expensive enough that computing it at every prefix of a 200+ config archive
# takes hours; ~40 samples gives the same curve in minutes.
HV_CURVE_POINTS = 40


def reference_point(obj_max):
    """Reference point in maximization space, just below the worst observation."""
    worst = obj_max.min(axis=0)
    spread = obj_max.max(axis=0) - worst
    spread[spread == 0] = 1.0
    return worst - REF_MARGIN * spread


def is_pareto_optimal(objectives):
    """Find Pareto-optimal points (all objectives maximized)."""
    n = len(objectives)
    is_optimal = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(objectives[j] >= objectives[i]) and np.any(objectives[j] > objectives[i]):
                is_optimal[i] = False
                break
    return is_optimal


def to_maximization(df):
    """Convert objectives to maximization space."""
    obj = df[METRIC_COLS].values.copy()
    for i, d in enumerate(METRIC_DIRS):
        obj[:, i] *= d
    return obj


def compute_hypervolume_2d(points, ref_point):
    """Compute 2D hypervolume (area dominated by Pareto front above ref_point)."""
    # Filter points that dominate the reference point
    mask = np.all(points > ref_point, axis=1)
    if not mask.any():
        return 0.0
    pts = points[mask]
    # Sort by first objective descending
    pts = pts[pts[:, 0].argsort()[::-1]]
    hv = 0.0
    prev_y = ref_point[1]
    for p in pts:
        hv += (p[0] - ref_point[0]) * (p[1] - prev_y)
        prev_y = max(prev_y, p[1])
    return hv


def compute_hypervolume_nd(points, ref_point):
    """
    Approximate hypervolume via Monte Carlo for 4D.
    For exact HV, would need pygmo or botorch.utils.multi_objective.
    """
    try:
        from botorch.utils.multi_objective.hypervolume import Hypervolume
        import torch
        hv_calculator = Hypervolume(ref_point=torch.tensor(ref_point, dtype=torch.double))
        pareto_mask = is_pareto_optimal(points)
        pareto_pts = points[pareto_mask]
        if len(pareto_pts) == 0:
            return 0.0
        result = hv_calculator.compute(torch.tensor(pareto_pts, dtype=torch.double))
        return float(result)
    except ImportError:
        # Fallback: product of ranges (crude approximation)
        pareto_mask = is_pareto_optimal(points)
        pareto_pts = points[pareto_mask]
        if len(pareto_pts) == 0:
            return 0.0
        ranges = pareto_pts.max(axis=0) - np.array(ref_point)
        return float(np.prod(np.maximum(ranges, 0)))


def plot_hypervolume_convergence(df, n_lhs, out_dir):
    """Plot 1: Hypervolume vs iteration number."""
    obj_max = to_maximization(df)
    # Fixed across the whole curve, so successive hypervolumes are comparable.
    ref = reference_point(obj_max)

    # Exact hypervolume in 5D costs roughly O(N^(M-1)), and this recomputes from
    # scratch at every prefix, so evaluating all 208 points takes hours. The
    # curve is smooth and monotone, so sampling every few points is visually
    # identical for a fraction of the work. Always include the final point so
    # the endpoint is exact.
    n = len(df)
    stride = max(1, n // HV_CURVE_POINTS)
    idx = sorted(set(list(range(1, n + 1, stride)) + [n]))
    hvs = [compute_hypervolume_nd(obj_max[:i], ref) for i in idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(idx, hvs, "b-o", markersize=3, linewidth=1.5)
    ax.axvline(x=n_lhs + 0.5, color="red", linestyle="--", alpha=0.6,
               label=f"LHS ({n_lhs}) → MOBO")
    # NB: this is the running hypervolume over configs that have ALL objectives,
    # in CSV (chronological) order. Configs missing any objective -- e.g. the
    # 4-objective-era runs with no CNR -- are absent, so the x axis is not the
    # MOBO iteration number.
    ax.set_xlabel(f"Configs with all {len(METRIC_COLS)} objectives "
                  f"(chronological order)", fontsize=11)
    ax.set_ylabel("Hypervolume", fontsize=11)
    ax.set_title(f"MOBO Convergence: {len(METRIC_COLS)}-Objective Hypervolume",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "hypervolume_convergence.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_pareto_expansion(df, n_lhs, out_dir):
    """Plot 2: Pairwise Pareto front comparison — LHS only vs LHS+MOBO."""
    obj_max = to_maximization(df)

    # LHS-only Pareto
    lhs_obj = obj_max[:n_lhs]
    lhs_pareto = is_pareto_optimal(lhs_obj)

    # Combined Pareto
    all_pareto = is_pareto_optimal(obj_max)

    # Axis labels are DERIVED, never written out. The hardcoded versions went
    # stale twice: they still said "Sensitivity" for what is now ppds_ring1, and
    # "MPXI (neg.)" after MPXI was changed to a maximized objective. A mislabelled
    # axis on a figure is worse than a missing one, because it gets believed.
    def axis_label(i):
        base = METRIC_LABELS[i]
        return base if METRIC_DIRS[i] > 0 else f"{base} (neg.)"

    # Index pairs only. The last two panels carry CNR, the outcome objective.
    pairs = [(0, 2), (1, 3), (3, 4), (2, 4)]
    pairs = [(i, j) for i, j in pairs
             if i < len(METRIC_LABELS) and j < len(METRIC_LABELS)]

    fig, axes = plt.subplots(1, len(pairs), figsize=(7 * len(pairs), 6))
    if len(pairs) == 1:
        axes = [axes]

    for ax, (i, j) in zip(axes, pairs):
        xlabel, ylabel = axis_label(i), axis_label(j)
        # LHS points
        ax.scatter(lhs_obj[:, i], lhs_obj[:, j], c="#3498db", s=40, alpha=0.6,
                   edgecolors="white", linewidth=0.5, label=f"LHS ({n_lhs})", zorder=2)
        # MOBO points
        mobo_obj = obj_max[n_lhs:]
        if len(mobo_obj) > 0:
            ax.scatter(mobo_obj[:, i], mobo_obj[:, j], c="#e74c3c", s=50, alpha=0.8,
                       edgecolors="white", linewidth=0.5, marker="D",
                       label=f"MOBO ({len(mobo_obj)})", zorder=3)

        # Mark Pareto-optimal points (combined). With 5 objectives a point is
        # dominated only if another beats it on all five, which is rare -- the
        # front typically covers most of the data, so the count is in the label
        # to keep that visible rather than implied.
        pareto_pts = obj_max[all_pareto]
        pct = 100.0 * all_pareto.sum() / len(all_pareto)
        ax.scatter(pareto_pts[:, i], pareto_pts[:, j], facecolors="none",
                   edgecolors="gold", s=100, linewidth=2, zorder=4,
                   label=f"Pareto front ({all_pareto.sum()}/{len(all_pareto)}, {pct:.0f}%)")

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Pareto Front Expansion: LHS → MOBO", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, "pareto_expansion.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_metric_scatter_combined(df, n_lhs, out_dir):
    """Plot 3: 4x4 pairwise metric scatter, LHS vs MOBO colored."""
    n = len(METRIC_COLS)
    obj_max = to_maximization(df)
    all_pareto = is_pareto_optimal(obj_max)

    fig, axes = plt.subplots(n, n, figsize=(14, 12))
    fig.suptitle("Pairwise Objectives (LHS + MOBO)", fontsize=14, y=0.98)

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                ax.hist(df[METRIC_COLS[i]].iloc[:n_lhs], bins=12, alpha=0.5,
                        color="#3498db", label="LHS", edgecolor="white")
                ax.hist(df[METRIC_COLS[i]].iloc[n_lhs:], bins=12, alpha=0.5,
                        color="#e74c3c", label="MOBO", edgecolor="white")
                ax.set_xlabel(METRIC_LABELS[i], fontsize=8)
            else:
                # LHS
                ax.scatter(df[METRIC_COLS[j]].iloc[:n_lhs],
                           df[METRIC_COLS[i]].iloc[:n_lhs],
                           c="#3498db", s=25, alpha=0.6, edgecolors="white", linewidth=0.3)
                # MOBO
                ax.scatter(df[METRIC_COLS[j]].iloc[n_lhs:],
                           df[METRIC_COLS[i]].iloc[n_lhs:],
                           c="#e74c3c", s=35, alpha=0.8, edgecolors="white", linewidth=0.3,
                           marker="D")
                # Pareto ring
                pareto_idx = np.where(all_pareto)[0]
                ax.scatter(df[METRIC_COLS[j]].iloc[pareto_idx],
                           df[METRIC_COLS[i]].iloc[pareto_idx],
                           facecolors="none", edgecolors="gold", s=70, linewidth=1.5)

                corr = df[METRIC_COLS[j]].corr(df[METRIC_COLS[i]])
                ax.annotate(f"r={corr:.2f}", xy=(0.05, 0.92), xycoords="axes fraction",
                            fontsize=7, bbox=dict(boxstyle="round,pad=0.2",
                                                  facecolor="white", alpha=0.8))

            if j == 0:
                ax.set_ylabel(METRIC_LABELS[i], fontsize=8)
            if i == n - 1:
                ax.set_xlabel(METRIC_LABELS[j], fontsize=8)
            ax.tick_params(labelsize=6)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=8, label='LHS'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#e74c3c', markersize=8, label='MOBO'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='gold',
               markersize=10, markeredgewidth=2, label='Pareto optimal'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    path = os.path.join(out_dir, "metric_scatter_combined.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_parallel_coordinates(df, n_lhs, out_dir):
    """Plot 4: Parallel coordinates, LHS vs MOBO, Pareto highlighted."""
    obj_max = to_maximization(df)
    all_pareto = is_pareto_optimal(obj_max)

    # Normalize for display (flip minimization so up=better)
    display_vals = df[METRIC_COLS].values.copy()
    for i, d in enumerate(METRIC_DIRS):
        if d == -1:
            display_vals[:, i] = -display_vals[:, i]
    mins = display_vals.min(axis=0)
    maxs = display_vals.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    normed = (display_vals - mins) / ranges

    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(METRIC_COLS))

    # Draw dominated LHS (behind)
    for idx in range(n_lhs):
        if not all_pareto[idx]:
            ax.plot(x_pos, normed[idx], color="#3498db", alpha=0.15, linewidth=0.8)

    # Draw dominated MOBO
    for idx in range(n_lhs, len(df)):
        if not all_pareto[idx]:
            ax.plot(x_pos, normed[idx], color="#e74c3c", alpha=0.2, linewidth=1)

    # Draw Pareto (LHS)
    for idx in range(n_lhs):
        if all_pareto[idx]:
            ax.plot(x_pos, normed[idx], color="#3498db", alpha=0.7, linewidth=2)

    # Draw Pareto (MOBO) on top
    for idx in range(n_lhs, len(df)):
        if all_pareto[idx]:
            ax.plot(x_pos, normed[idx], color="#e74c3c", alpha=0.9, linewidth=2.5)

    direction_arrows = ["\u2193 better" if d == -1 else "\u2191 better" for d in METRIC_DIRS]
    labels = [f"{ml}\n({da})" for ml, da in zip(METRIC_LABELS, direction_arrows)]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Normalized value (up = better)", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="x", alpha=0.3)

    # Raw value annotations
    raw_vals = df[METRIC_COLS].values
    for i in range(len(METRIC_COLS)):
        col = raw_vals[:, i]
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

    n_pareto_lhs = all_pareto[:n_lhs].sum()
    n_pareto_mobo = all_pareto[n_lhs:].sum()
    legend_elements = [
        Line2D([0], [0], color='#3498db', linewidth=2, label=f'LHS Pareto ({n_pareto_lhs})'),
        Line2D([0], [0], color='#3498db', linewidth=0.8, alpha=0.3, label=f'LHS dominated'),
        Line2D([0], [0], color='#e74c3c', linewidth=2.5, label=f'MOBO Pareto ({n_pareto_mobo})'),
        Line2D([0], [0], color='#e74c3c', linewidth=1, alpha=0.3, label=f'MOBO dominated'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    ax.set_title("Parallel Coordinates: LHS + MOBO Objectives", fontsize=13)

    plt.tight_layout()
    path = os.path.join(out_dir, "parallel_coordinates_combined.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def print_summary(df, n_lhs):
    """Print summary with top Pareto configs."""
    obj_max = to_maximization(df)
    all_pareto = is_pareto_optimal(obj_max)

    print("\n" + "=" * 80)
    print("MOBO CONVERGENCE SUMMARY")
    print("=" * 80)
    print(f"LHS configs: {n_lhs}")
    print(f"MOBO configs: {len(df) - n_lhs}")
    print(f"Total feasible: {len(df)}")
    print(f"Pareto-optimal: {all_pareto.sum()} "
          f"(LHS: {all_pareto[:n_lhs].sum()}, MOBO: {all_pareto[n_lhs:].sum()})")

    print(f"\n{'Metric':<20} {'LHS Best':>12} {'Combined Best':>15} {'Direction':>10}")
    print("-" * 60)
    for col, label, d in zip(METRIC_COLS, METRIC_LABELS, METRIC_DIRS):
        lhs_vals = df[col].iloc[:n_lhs]
        all_vals = df[col]
        if d == -1:
            lhs_best = lhs_vals.min()
            all_best = all_vals.min()
        else:
            lhs_best = lhs_vals.max()
            all_best = all_vals.max()
        improved = "+" if (all_best - lhs_best) * d > 0 else "="
        print(f"{label:<20} {lhs_best:>12.4f} {all_best:>15.4f} {'min' if d == -1 else 'max':>10} {improved}")

    print(f"\nTOP PARETO CONFIGS (combined):")
    print("-" * 80)
    pareto_df = df[all_pareto].copy()
    display_cols = DESIGN_COLS + METRIC_COLS
    available = [c for c in display_cols if c in pareto_df.columns]
    if "config_name" in pareto_df.columns:
        available = ["config_name"] + available
    elif "config" in pareto_df.columns:
        available = ["config"] + available
    print(pareto_df[available].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="MOBO Convergence Analysis")
    parser.add_argument("--csv", type=str, required=True, help="Results CSV")
    parser.add_argument("--out_dir", type=str, default="results/mobo_plots",
                        help="Output directory for plots")
    parser.add_argument("--n_lhs", type=int, default=None,
                        help="Number of LHS seed points (auto-detected if not specified)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=METRIC_COLS)
    print(f"Loaded {len(df)} feasible configs from {args.csv}")

    # Auto-detect LHS vs MOBO split
    n_lhs = args.n_lhs
    if n_lhs is None:
        if "config_name" in df.columns:
            n_lhs = (~df["config_name"].str.startswith("mobo_")).sum()
        elif "config" in df.columns:
            n_lhs = (~df["config"].str.startswith("mobo_")).sum()
        else:
            n_lhs = 21  # fallback
    print(f"LHS: {n_lhs}, MOBO: {len(df) - n_lhs}")

    os.makedirs(args.out_dir, exist_ok=True)

    plot_hypervolume_convergence(df, n_lhs, args.out_dir)
    plot_pareto_expansion(df, n_lhs, args.out_dir)
    plot_metric_scatter_combined(df, n_lhs, args.out_dir)
    plot_parallel_coordinates(df, n_lhs, args.out_dir)
    print_summary(df, n_lhs)

    print(f"\nAll plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
