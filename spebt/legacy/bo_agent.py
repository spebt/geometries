#!/usr/bin/env python3
"""
Bayesian Optimization Agent for SAI SC-SPECT.

Adapted from Kirtiraj's bo_agent.py for 2D hardware design space:
  - Design vector: (aperture_diam_mm, n_apertures)
  - SingleTaskGP with Matérn 5/2 + ARD
  - Sequential q=1 Expected Improvement (matching Kirtiraj's proven approach)

Usage:
  from bo_agent import get_next_candidate
  diam, n_ap = get_next_candidate(results_csv)
"""
import os
import logging
import warnings
import torch
import pandas as pd
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import normalize, unnormalize
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("BO_Agent_SAI")
warnings.filterwarnings("ignore", category=UserWarning)

# --- Design space bounds ---
# [aperture_diam_mm, n_apertures, scint_radial_thickness_mm, ring_thickness_mm]
# n_apertures treated as continuous, rounded to int for geometry generation
PARAM_NAMES = ["aperture_diam_mm", "n_apertures", "scint_radial_thickness_mm", "ring_thickness_mm"]
BOUNDS_MIN = [0.2, 60.0, 3.0, 1.0]
BOUNDS_MAX = [1.0, 360.0, 12.0, 5.0]
DIM = len(PARAM_NAMES)

RESULTS_FILE = None  # set by caller or CLI


def get_next_candidate(results_csv: str = None):
    """
    1. Reads existing results CSV.
    2. Fits a Gaussian Process (GP).
    3. Optimizes qEI (Monte Carlo Expected Improvement), q=1.
    4. Returns the next (aperture_diam_mm, n_apertures).

    Matches Kirtiraj's bo_agent.get_next_candidate() structure exactly.
    """
    csv_path = results_csv or RESULTS_FILE
    logger.info("--- Starting Bayesian Optimization Step ---")

    # 1. Load Data
    if not os.path.exists(csv_path):
        logger.error(f"File not found: {csv_path}")
        raise FileNotFoundError(f"Could not find {csv_path}.")

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["JI"])

    # Keep infeasible (JI=0) points but assign a small penalty value
    # so the GP learns to avoid those regions of the design space
    n_infeasible = (df["JI"] <= 1e-9).sum()
    if n_infeasible > 0:
        feasible_min = df.loc[df["JI"] > 1e-9, "JI"].min() if (df["JI"] > 1e-9).any() else 1e-10
        penalty_val = feasible_min * 0.1  # 10% of worst feasible JI
        df.loc[df["JI"] <= 1e-9, "JI"] = penalty_val
        logger.info(f"Replaced {n_infeasible} infeasible (JI=0) points with penalty={penalty_val:.4e}")

    n_points = len(df)
    logger.info(f"Loaded {n_points} data points from history.")

    if n_points < 5:
        logger.warning("Very few data points found (<5). GP model may be unstable.")

    # Prepare Tensors
    train_x = torch.tensor(df[PARAM_NAMES].values, dtype=torch.double)
    train_y = torch.tensor(df[["JI"]].values, dtype=torch.double)

    current_best_ji = train_y.max().item()
    logger.info(f"Current Best JI: {current_best_ji:.6e}")

    # 2. Normalize Inputs & Standardize Outputs
    bounds = torch.tensor([BOUNDS_MIN, BOUNDS_MAX], dtype=torch.double)
    train_x_norm = normalize(train_x, bounds)
    train_y_std = (train_y - train_y.mean()) / (train_y.std() + 1e-6)

    # 3. Fit Gaussian Process
    logger.info("Fitting Gaussian Process model...")
    gp = SingleTaskGP(train_x_norm, train_y_std)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    logger.info("GP Model trained successfully.")

    # Log ARD lengthscales
    try:
        ls = gp.covar_module.base_kernel.lengthscale.detach().cpu().numpy().flatten()
        for name, l in zip(PARAM_NAMES, ls):
            logger.info(f"  ARD lengthscale {name}: {l:.4f}")
    except Exception:
        pass

    # 4. Optimize Acquisition Function (qLogEI, q=1)
    logger.info("Optimizing Acquisition Function (Log Expected Improvement)...")
    MC_EI = qLogExpectedImprovement(
        model=gp,
        best_f=train_y_std.max(),
    )

    candidate_norm, _ = optimize_acqf(
        acq_function=MC_EI,
        bounds=torch.stack([torch.zeros(DIM), torch.ones(DIM)]).double(),
        q=1,
        num_restarts=10,
        raw_samples=512,
    )

    # 5. Un-normalize
    candidate_physical = unnormalize(candidate_norm, bounds)
    next_diam = candidate_physical[0, 0].item()
    next_n_ap = candidate_physical[0, 1].item()
    next_scint_radial = candidate_physical[0, 2].item()
    next_ring_thick = candidate_physical[0, 3].item()

    # Round n_apertures to nearest integer
    next_n_ap = int(round(next_n_ap))

    logger.info(f"Optimization Complete.")
    logger.info(f"SUGGESTION -> Aperture Diam: {next_diam:.4f} mm | N Apertures: {next_n_ap} | "
                f"Scint Radial: {next_scint_radial:.4f} mm | Ring Thickness: {next_ring_thick:.4f} mm")

    return next_diam, next_n_ap, next_scint_radial, next_ring_thick


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BO Agent for SAI SC-SPECT")
    parser.add_argument("--results_csv", type=str, required=True,
                        help="Path to results_summary.csv")
    args = parser.parse_args()

    diam, n_ap, scint_rad, ring_thick = get_next_candidate(args.results_csv)
    print(f"\nSuggested next config:")
    print(f"  aperture_diam        = {diam:.4f} mm")
    print(f"  n_apertures          = {n_ap}")
    print(f"  scint_radial_thick   = {scint_rad:.4f} mm")
    print(f"  ring_thickness       = {ring_thick:.4f} mm")
