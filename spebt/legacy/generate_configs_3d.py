#!/usr/bin/env python3
"""
Generate Latin Hypercube Sampling (LHS) configurations for 4D MOBO.

Design vector: (aperture_diam_mm, n_apertures, n_det_ring1, n_det_ring2)
  - aperture_diam_mm  ∈ [0.2, 1.0]   (collimator aperture diameter)
  - n_apertures        ∈ [60, 360]    (number of apertures on HR ring)
  - n_det_ring1        ∈ [120, 480]   (crystals on detector ring 1, must be even)
  - n_det_ring2        ∈ [180, 720]   (crystals on detector ring 2, must be even)

Feasibility constraint:
  aperture_diam < circumference / n_apertures
  circumference = 2π × 35.0 mm ≈ 219.9 mm  (at HR ring mid-radius)

Infeasible samples are resampled until all configs are valid.

Output: configs_manifest_3d.csv
"""
import argparse
import math
import os
import numpy as np
from scipy.stats.qmc import LatinHypercube


BOUNDS_MIN = np.array([0.2, 60.0, 120.0, 180.0])
BOUNDS_MAX = np.array([1.0, 270.0, 660.0, 960.0])  # n_ap max: MIN_SPACING=0.8mm → n≤274
PARAM_NAMES = ["aperture_diam_mm", "n_apertures", "n_det_ring1", "n_det_ring2"]
DIM = len(PARAM_NAMES)

# HR collimator ring geometry (fixed)
HR_RING_INNER_DIAM_MM = 67.5
HR_RING_THICKNESS_MM = 2.5
HR_R_CENTER = HR_RING_INNER_DIAM_MM / 2.0 + HR_RING_THICKNESS_MM / 2.0  # 35.0 mm
HR_CIRCUMFERENCE = 2.0 * math.pi * HR_R_CENTER  # ~219.9 mm


def is_feasible(aperture_diam, n_apertures):
    """Check if aperture config fits on the HR ring without overlap."""
    max_diam = HR_CIRCUMFERENCE / n_apertures
    return aperture_diam < max_diam * 0.95  # 5% safety margin


def generate_lhs_configs(n_samples: int, seed: int = 2026) -> np.ndarray:
    """Generate n_samples feasible configs via LHS in 4D."""
    # Oversample to account for infeasible rejection
    oversample_factor = 3
    rng = np.random.default_rng(seed)

    all_feasible = []
    attempt = 0

    while len(all_feasible) < n_samples:
        attempt += 1
        n_draw = (n_samples - len(all_feasible)) * oversample_factor
        sampler = LatinHypercube(d=DIM, seed=rng.integers(0, 2**31))
        unit_samples = sampler.random(n=n_draw)
        configs = BOUNDS_MIN + unit_samples * (BOUNDS_MAX - BOUNDS_MIN)

        # Round integer params
        configs[:, 1] = np.round(configs[:, 1]).astype(int)          # n_apertures
        configs[:, 2] = (np.round(configs[:, 2] / 2) * 2).astype(int)  # n_det_ring1 (even)
        configs[:, 3] = (np.round(configs[:, 3] / 2) * 2).astype(int)  # n_det_ring2 (even)

        # Filter feasible
        for row in configs:
            if is_feasible(row[0], row[1]):
                all_feasible.append(row)
                if len(all_feasible) >= n_samples:
                    break

        if attempt > 20:
            print(f"[warn] Only found {len(all_feasible)} feasible after {attempt} attempts")
            break

    result = np.array(all_feasible[:n_samples])
    n_total_checked = attempt * n_draw // oversample_factor
    print(f"Generated {len(result)} feasible configs (checked ~{n_total_checked})")
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate LHS configs for 4D MOBO")
    parser.add_argument("--n_samples", type=int, default=25,
                        help="Number of LHS samples (default: 25)")
    parser.add_argument("--seed", type=int, default=2026,
                        help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory for manifest CSV")
    parser.add_argument("--base_work_dir", type=str,
                        default="/vscratch/grp-rutaoyao/Omer/spebt/spebt/optimization/results",
                        help="Base work directory on HPC")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    configs = generate_lhs_configs(args.n_samples, args.seed)

    out_path = os.path.join(args.output_dir, "configs_manifest_3d.csv")
    with open(out_path, "w") as f:
        f.write("idx,aperture_diam_mm,n_apertures,n_det_ring1,n_det_ring2,work_dir\n")
        for i, row in enumerate(configs):
            ap_d, n_ap, n_det1, n_det2 = row
            n_ap_int = int(n_ap)
            n_det1_int = int(n_det1)
            n_det2_int = int(n_det2)
            work_dir = os.path.join(
                args.base_work_dir,
                f"lhs4d_{i:04d}_ap{ap_d:.4f}_nap{n_ap_int}_nd1_{n_det1_int}_nd2_{n_det2_int}"
            )
            f.write(f"{i},{ap_d:.6f},{n_ap_int},{n_det1_int},{n_det2_int},{work_dir}\n")

    print(f"Saved {len(configs)} configs -> {out_path}")
    print(f"Bounds: {dict(zip(PARAM_NAMES, zip(BOUNDS_MIN, BOUNDS_MAX)))}")
    print(f"Feasibility: aperture_diam < {HR_CIRCUMFERENCE:.1f} / n_apertures (with 5% margin)")
    print(f"\nSample statistics:")
    for j, name in enumerate(PARAM_NAMES):
        col = configs[:, j]
        print(f"  {name}: min={col.min():.4f}  max={col.max():.4f}  mean={col.mean():.4f}")


if __name__ == "__main__":
    main()
