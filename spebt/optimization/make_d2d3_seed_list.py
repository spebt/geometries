#!/usr/bin/env python3
"""
Generate designs that actually vary D2/D3, to unstick the 6D search.

The campaign is nominally 6D but its training data is constant along the two new
axes: across 183 rows, d3_inner_mm has a standard deviation of 0.015 mm over a
255 mm range, and d2_inner_mm has 8 values inside a 106 mm window. Every point
sits at the legacy 390/520 layout because that is what the pre-expansion configs
were stamped with.

A GP cannot learn from a constant. With no variance along a dimension it fits a
very long lengthscale, concludes the objective does not depend on it, and the
acquisition sees no expected improvement from moving -- so it proposes the
default again, and the data stays constant. The loop cannot bootstrap out of
this on its own.

This breaks the cycle by evaluating a handful of designs chosen to span D2/D3
while holding the other four parameters at values already known to perform well.
Holding them fixed matters: it means any difference in the objectives is
attributable to the ring geometry rather than confounded with aperture size.

Every generated design is checked against the same feasibility rules the
optimizer uses, so none can fail geometry generation.

Usage:
  python make_d2d3_seed_list.py --results_csv results/results_summary_mobo.csv
  python make_d2d3_seed_list.py --results_csv ... --n_seeds 8
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

import mobo_agent as ma


def main():
    ap = argparse.ArgumentParser(description="Generate D2/D3-spanning seed designs")
    ap.add_argument("--results_csv", required=True)
    ap.add_argument("--out", default="d2d3_seeds.csv")
    ap.add_argument("--n_seeds", type=int, default=6)
    ap.add_argument("--base_config", default=None,
                    help="Config whose 4D parameters to hold fixed "
                         "(default: the best by cnr_sector_mean)")
    args = ap.parse_args()

    if not os.path.exists(args.results_csv):
        print(f"ERROR: results CSV not found: {args.results_csv}")
        sys.exit(1)
    df = pd.read_csv(args.results_csv)

    # Anchor on a design already known to be good, so the seeds explore ring
    # geometry from a sensible starting point rather than a random one.
    have = df.dropna(subset=["cnr_sector_mean"] + ma.PARAM_NAMES)
    if args.base_config:
        row = have[have["config"] == args.base_config]
        if row.empty:
            print(f"ERROR: {args.base_config} not found with full data")
            sys.exit(1)
        base = row.iloc[0]
    else:
        base = have.loc[have["cnr_sector_mean"].idxmax()]

    diam = float(base["aperture_diam_mm"])
    n_ap = int(base["n_apertures"])
    nd1 = int(base["n_det_ring1"])
    nd2 = int(base["n_det_ring2"])
    print(f"Anchoring on {base['config']}  (CNR {base['cnr_sector_mean']:.3f})")
    print(f"  holding aperture={diam:.4f} n_ap={n_ap} nd1={nd1} nd2={nd2}\n")

    # Grid across the feasible D2/D3 region, then keep a spread-out subset.
    d2_lo, d2_hi = ma.BOUNDS_MIN[4], ma.BOUNDS_MAX[4]
    d3_lo, d3_hi = ma.BOUNDS_MIN[5], ma.BOUNDS_MAX[5]
    cand = []
    for d2 in np.linspace(d2_lo, d2_hi, 7):
        for d3 in np.linspace(d3_lo, d3_hi, 7):
            if not ma.is_feasible_full(diam, n_ap, nd1, nd2, d2, d3):
                continue
            # Skip anything effectively at the legacy layout -- that region is
            # already saturated in the training data.
            if abs(d2 - 390.0) < 15.0 and abs(d3 - 520.0) < 15.0:
                continue
            cand.append((float(d2), float(d3)))

    if not cand:
        print("ERROR: no feasible D2/D3 combinations found for this anchor design.")
        sys.exit(1)

    # Greedy max-min selection so the seeds spread out rather than cluster.
    pts = np.array(cand)
    span = np.array([d2_hi - d2_lo, d3_hi - d3_lo])
    norm = pts / span
    chosen = [int(np.argmax(norm.sum(axis=1)))]
    while len(chosen) < min(args.n_seeds, len(pts)):
        d = np.min(np.linalg.norm(norm[:, None, :] - norm[chosen][None, :, :], axis=2), axis=1)
        chosen.append(int(np.argmax(d)))

    rows = []
    for i, idx in enumerate(chosen):
        d2, d3 = pts[idx]
        rows.append({
            "config_name": f"seed6d_{i:02d}_ap{diam:.4f}_nap{n_ap}"
                           f"_nd1_{nd1}_nd2_{nd2}_d2_{d2:.0f}_d3_{d3:.0f}",
            "aperture_diam_mm": diam, "n_apertures": n_ap,
            "n_det_ring1": nd1, "n_det_ring2": nd2,
            "d2_inner_mm": round(d2, 3), "d3_inner_mm": round(d3, 3),
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)

    print(f"{len(out)} seed designs spanning D2 {out.d2_inner_mm.min():.0f}-"
          f"{out.d2_inner_mm.max():.0f} mm, D3 {out.d3_inner_mm.min():.0f}-"
          f"{out.d3_inner_mm.max():.0f} mm\n")
    print(out[["d2_inner_mm", "d3_inner_mm"]].to_string(index=False))
    print(f"\nWrote {args.out}")
    print(f"\nSubmit with:\n  sbatch --array=0-{len(out) - 1}%4 submit_d2d3_seeds.sh")
    print("\nAll designs pass is_feasible_full, so none can fail geometry generation.")


if __name__ == "__main__":
    main()
