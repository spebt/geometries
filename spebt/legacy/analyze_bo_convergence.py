#!/usr/bin/env python3
"""
BO Convergence Analysis and Plotting.

Reads results_summary.csv and produces:
  1. Best JI vs iteration (convergence curve)
  2. GP surrogate surface (2D heatmap at each slice)
  3. ARD lengthscale evolution
  4. Design space coverage (scatter of evaluated configs)

Usage:
  python analyze_bo_convergence.py --results_csv results/results_summary.csv
  python analyze_bo_convergence.py --results_csv results/results_summary.csv --output_dir plots/
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood

PARAM_NAMES = ["aperture_diam_mm", "n_apertures", "scint_radial_thickness_mm", "ring_thickness_mm"]
BOUNDS_MIN = [0.2, 60.0, 3.0, 1.0]
BOUNDS_MAX = [1.0, 360.0, 12.0, 5.0]


def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["JI"])
    return df


def fit_gp(df: pd.DataFrame):
    """Fit GP on results data, return model and training data."""
    bounds = torch.tensor([BOUNDS_MIN, BOUNDS_MAX], dtype=torch.double)
    train_x = torch.tensor(df[PARAM_NAMES].values, dtype=torch.double)
    train_y = torch.tensor(df[["JI"]].values, dtype=torch.double)

    # Handle infeasible points
    feasible_mask = train_y.squeeze() > 1e-9
    if not feasible_mask.all() and feasible_mask.any():
        feas_min = train_y[feasible_mask].min().item()
        train_y[~feasible_mask] = feas_min * 0.1

    train_x_norm = normalize(train_x, bounds)
    y_mean, y_std = train_y.mean(), train_y.std() + 1e-8
    train_y_std = (train_y - y_mean) / y_std

    gp = SingleTaskGP(train_x_norm, train_y_std)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    return gp, bounds, y_mean, y_std


def plot_convergence(df: pd.DataFrame, output_dir: str):
    """Plot best JI vs evaluation number."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ji_vals = df["JI"].values
    best_so_far = np.maximum.accumulate(np.where(ji_vals > 0, ji_vals, 0))

    # Left: best JI vs iteration
    ax1.plot(range(len(best_so_far)), best_so_far, "b-o", markersize=3, linewidth=1.5)
    ax1.set_xlabel("Evaluation Number")
    ax1.set_ylabel("Best JI")
    ax1.set_title("BO Convergence: Best JI vs Iteration")
    ax1.grid(True, alpha=0.3)

    # Highlight LHS phase vs BO phase if applicable
    # Assume first N points without "bo_" prefix are LHS
    lhs_mask = ~df["config"].str.startswith("bo_")
    n_lhs = lhs_mask.sum()
    if n_lhs > 0 and n_lhs < len(df):
        ax1.axvline(x=n_lhs - 0.5, color="red", linestyle="--", alpha=0.5, label=f"LHS→BO ({n_lhs} pts)")
        ax1.legend()

    # Right: JI per evaluation (raw)
    colors = ["green" if ji > 0 else "red" for ji in ji_vals]
    ax2.scatter(range(len(ji_vals)), ji_vals, c=colors, s=20, alpha=0.7)
    ax2.set_xlabel("Evaluation Number")
    ax2.set_ylabel("JI")
    ax2.set_title("JI per Evaluation (red = infeasible)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "convergence.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_design_space(df: pd.DataFrame, output_dir: str):
    """Pairwise scatter plots of evaluated configs colored by JI (4D)."""
    import itertools

    feasible = df[df["JI"] > 0]
    infeasible = df[df["JI"] <= 0]

    # All available parameter pairs
    available_params = [p for p in PARAM_NAMES if p in df.columns]
    pairs = list(itertools.combinations(available_params, 2))
    n_pairs = len(pairs)

    if n_pairs == 0:
        print("  Skipping design space plot: no parameter columns found")
        return

    ncols = min(3, n_pairs)
    nrows = (n_pairs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if n_pairs == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, (px, py) in enumerate(pairs):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        if len(feasible) > 0 and px in feasible.columns and py in feasible.columns:
            sc = ax.scatter(
                feasible[px], feasible[py],
                c=feasible["JI"], cmap="viridis", s=40, edgecolors="black", linewidth=0.3,
                zorder=2,
            )
            fig.colorbar(sc, ax=ax, label="JI", shrink=0.8)

        if len(infeasible) > 0 and px in infeasible.columns and py in infeasible.columns:
            ax.scatter(
                infeasible[px], infeasible[py],
                c="red", marker="x", s=40, label=f"Infeasible ({len(infeasible)})",
                zorder=3,
            )

        # Feasibility boundary for aperture_diam vs n_apertures
        if px == "aperture_diam_mm" and py == "n_apertures":
            n_range = np.linspace(60, 360, 300)
            max_diam = 2 * 35.0 * np.sin(np.pi / n_range)
            ax.plot(max_diam, n_range, "r--", linewidth=1.5, alpha=0.7)

        # Mark best config
        if len(feasible) > 0 and px in feasible.columns and py in feasible.columns:
            best = feasible.loc[feasible["JI"].idxmax()]
            ax.scatter(
                best[px], best[py],
                s=200, facecolors="none", edgecolors="gold", linewidth=3, zorder=4,
            )

        ax.set_xlabel(px)
        ax.set_ylabel(py)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(n_pairs, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle("Design Space Exploration (4D)", fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "design_space.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_gp_surface(df: pd.DataFrame, output_dir: str):
    """GP posterior mean surface — 2D slices through best config for each param pair."""
    feasible = df[df["JI"] > 0]
    available_params = [p for p in PARAM_NAMES if p in df.columns]

    if len(feasible) < 5 or len(available_params) < 2:
        print("  Skipping GP surface: too few feasible points or parameters")
        return

    gp, bounds, y_mean, y_std = fit_gp(df)

    # Best config values (used to fix non-plotted dimensions)
    best_row = feasible.loc[feasible["JI"].idxmax()]
    best_vals = [float(best_row.get(p, (BOUNDS_MIN[i] + BOUNDS_MAX[i]) / 2))
                 for i, p in enumerate(PARAM_NAMES) if p in df.columns]

    # Plot primary slice: aperture_diam vs n_apertures (fixing other 2 at best)
    n_grid = 80
    dim = len(available_params)

    # Determine which pair to plot as primary (first two params)
    p0, p1 = 0, 1
    d_range = np.linspace(BOUNDS_MIN[p0], BOUNDS_MAX[p0], n_grid)
    n_range = np.linspace(BOUNDS_MIN[p1], BOUNDS_MAX[p1], n_grid)
    D, N = np.meshgrid(d_range, n_range)

    # Build test_x: fill all dims with best values, vary p0 and p1
    test_x = torch.full((n_grid * n_grid, dim), 0.0, dtype=torch.double)
    for i in range(dim):
        test_x[:, i] = best_vals[i]
    test_x[:, p0] = torch.tensor(D.ravel(), dtype=torch.double)
    test_x[:, p1] = torch.tensor(N.ravel(), dtype=torch.double)
    test_x_norm = normalize(test_x, bounds)

    gp.eval()
    with torch.no_grad():
        posterior = gp.posterior(test_x_norm)
        pred_mean = posterior.mean.squeeze().numpy()
        pred_std = posterior.variance.sqrt().squeeze().numpy()

    pred_mean_phys = pred_mean * y_std.item() + y_mean.item()
    pred_std_phys = pred_std * y_std.item()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    Z_mean = pred_mean_phys.reshape(n_grid, n_grid)
    c1 = ax1.contourf(D, N, Z_mean, levels=30, cmap="viridis")
    plt.colorbar(c1, ax=ax1, label="GP Mean JI")

    if "aperture_diam_mm" in feasible.columns and "n_apertures" in feasible.columns:
        ax1.scatter(feasible["aperture_diam_mm"], feasible["n_apertures"],
                    c="white", edgecolors="black", s=20, zorder=3)

    # Feasibility boundary
    n_bnd = np.linspace(60, 360, 300)
    max_d = 2 * 35.0 * np.sin(np.pi / n_bnd)
    ax1.plot(max_d, n_bnd, "r--", linewidth=1.5)

    ax1.set_xlabel(PARAM_NAMES[p0])
    ax1.set_ylabel(PARAM_NAMES[p1])
    slice_info = ", ".join(f"{PARAM_NAMES[i]}={best_vals[i]:.2f}" for i in range(dim) if i not in (p0, p1))
    ax1.set_title(f"GP Posterior Mean (slice: {slice_info})")
    ax1.set_xlim(BOUNDS_MIN[p0] - 0.05, BOUNDS_MAX[p0] + 0.05)
    ax1.set_ylim(BOUNDS_MIN[p1] - 10, BOUNDS_MAX[p1] + 10)

    Z_std = pred_std_phys.reshape(n_grid, n_grid)
    c2 = ax2.contourf(D, N, Z_std, levels=30, cmap="Oranges")
    plt.colorbar(c2, ax=ax2, label="GP Std JI")
    if "aperture_diam_mm" in feasible.columns and "n_apertures" in feasible.columns:
        ax2.scatter(feasible["aperture_diam_mm"], feasible["n_apertures"],
                    c="white", edgecolors="black", s=20, zorder=3)
    ax2.plot(max_d, n_bnd, "r--", linewidth=1.5)
    ax2.set_xlabel(PARAM_NAMES[p0])
    ax2.set_ylabel(PARAM_NAMES[p1])
    ax2.set_title(f"GP Posterior Uncertainty (slice: {slice_info})")
    ax2.set_xlim(BOUNDS_MIN[p0] - 0.05, BOUNDS_MAX[p0] + 0.05)
    ax2.set_ylim(BOUNDS_MIN[p1] - 10, BOUNDS_MAX[p1] + 10)

    plt.tight_layout()
    path = os.path.join(output_dir, "gp_surface.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # Log ARD lengthscales
    try:
        ls = gp.covar_module.base_kernel.lengthscale.detach().cpu().numpy().flatten()
        print(f"  ARD lengthscales: {dict(zip(available_params, ls))}")
    except Exception:
        pass


def print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n--- Results Summary ---")
    print(f"Total evaluations: {len(df)}")
    feasible = df[df["JI"] > 0]
    infeasible = df[df["JI"] <= 0]
    print(f"Feasible: {len(feasible)}, Infeasible: {len(infeasible)}")

    if len(feasible) > 0:
        best = feasible.loc[feasible["JI"].idxmax()]
        print(f"\nBest config: {best.get('config', 'N/A')}")
        print(f"  aperture_diam = {best['aperture_diam_mm']:.4f} mm")
        print(f"  n_apertures   = {int(best['n_apertures'])}")
        if "scint_radial_thickness_mm" in best.index:
            print(f"  scint_radial  = {best.get('scint_radial_thickness_mm', 'N/A')} mm")
        if "ring_thickness_mm" in best.index:
            print(f"  ring_thickness= {best.get('ring_thickness_mm', 'N/A')} mm")
        print(f"  JI            = {best['JI']:.6e}")
        if "fwhm_mean" in best.index:
            print(f"  FWHM          = {best.get('fwhm_mean', 'N/A')}")
            print(f"  Sensitivity   = {best.get('sensitivity_mean', 'N/A')}")
            print(f"  ASCI          = {best.get('asci_pct', 'N/A')}%")

        print(f"\nTop 5 configs:")
        top5 = feasible.nlargest(5, "JI")
        for _, row in top5.iterrows():
            sr = f"  sr={row.get('scint_radial_thickness_mm', 6.0):.2f}" if "scint_radial_thickness_mm" in row.index else ""
            rt = f"  rt={row.get('ring_thickness_mm', 2.5):.2f}" if "ring_thickness_mm" in row.index else ""
            print(f"  {row.get('config', '?'):40s}  d={row['aperture_diam_mm']:.4f}  "
                  f"n={int(row['n_apertures']):3d}{sr}{rt}  JI={row['JI']:.4e}")


def main():
    parser = argparse.ArgumentParser(description="BO Convergence Analysis")
    parser.add_argument("--results_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="plots/convergence")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = load_results(args.results_csv)

    print(f"Loaded {len(df)} results from {args.results_csv}")

    print_summary(df)
    print("\nGenerating plots...")
    plot_convergence(df, args.output_dir)
    plot_design_space(df, args.output_dir)
    plot_gp_surface(df, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
