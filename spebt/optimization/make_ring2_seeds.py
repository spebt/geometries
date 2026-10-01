#!/usr/bin/env python3
"""
Seed designs in the region opened by raising the n_det_ring2 bound.

The bound was 960, matching ring 3's fixed count, and every top design in the
271-config archive sat exactly on it. That is the signature of a search stopped
by its box rather than by physics: the packing constraint allows 1012 crystals
at d2 = 400 mm and 1361 at d2 = 540 mm.

Raising the bound alone is not enough. The archive contains NO design above 960,
so the surrogate has no observations there, fits a lengthscale from data that
stops at the old wall, and the acquisition sees no reason to cross it. This is
the same failure that froze the 6D expansion at the legacy 390/520 ring layout
until seeds were placed across the range. A GP cannot learn from a region it has
never seen.

There is a real tension worth understanding before reading the results. More
ring-2 crystals REQUIRE a larger d2, because the packing limit scales with
radius. But the search's best designs sit at d2 ~ 393-414, where the ceiling is
only ~995-1020. So the newly opened space is mostly at LARGER d2, which the
5-objective campaign found less attractive. The seeds therefore walk the packing
frontier rather than sitting at one d2: if extra crystals help enough to pay for
moving d2 outward, that shows up as a trend across them.

Other parameters are held at the best known design's values so any difference is
attributable to the ring-2 geometry rather than confounded with aperture size.

Usage:
  python make_ring2_seeds.py --results_csv results/results_summary_mobo.csv
  python make_ring2_seeds.py --results_csv ... --n_seeds 8 --out ring2_seeds.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

import mobo_agent as ma

# Fraction of the packing limit to sit at. 1.0 would be exactly tangential
# crystals with zero clearance, which is_ring_packing_ok rejects (strict <).
PACKING_FRACTION = 0.97


def main():
    ap = argparse.ArgumentParser(description="Seed the region above the old ring-2 bound")
    ap.add_argument("--results_csv", required=True)
    ap.add_argument("--out", default="ring2_seeds.csv")
    ap.add_argument("--n_seeds", type=int, default=8)
    ap.add_argument("--prefix", default="ring2_")
    ap.add_argument("--results_dir", default="results")
    args = ap.parse_args()

    if not os.path.exists(args.results_csv):
        print(f"ERROR: results CSV not found: {args.results_csv}")
        sys.exit(1)

    df = pd.read_csv(args.results_csv).dropna(subset=["cnr_sector_mean"] + ma.PARAM_NAMES)
    if df.empty:
        print("ERROR: no configs with CNR and a full design vector")
        sys.exit(1)

    best = df.loc[df["cnr_sector_mean"].idxmax()]
    print(f"Holding non-ring-2 parameters at the best known design:")
    print(f"  {best['config']}  CNR {best['cnr_sector_mean']:.4f}")
    print(f"  aperture {best['aperture_diam_mm']:.4f} mm, {int(best['n_apertures'])} apertures, "
          f"ring-1 {int(best['n_det_ring1'])}")
    print(f"  its ring 2: {int(best['n_det_ring2'])} crystals at d2 = {best['d2_inner_mm']:.0f} mm\n")

    diam = float(best["aperture_diam_mm"])
    n_ap = int(best["n_apertures"])
    nd1 = int(best["n_det_ring1"])

    # Walk d2 outward, taking as many crystals as packing allows at each step.
    d2_lo = max(float(best["d2_inner_mm"]), ma.BOUNDS_MIN[4])
    d2_grid = np.linspace(d2_lo, ma.BOUNDS_MAX[4], args.n_seeds)

    rows, skipped = [], []
    for d2 in d2_grid:
        nd2 = int(ma.max_crystals_on_ring(d2) * PACKING_FRACTION)
        nd2 -= nd2 % 2                      # two crystals per cell
        nd2 = min(nd2, int(ma.BOUNDS_MAX[3]))
        if nd2 <= 960:
            skipped.append((d2, nd2, "packing ceiling still at or below the old bound"))
            continue

        # Hold d3 at the best design's value wherever the clearance rule allows,
        # and only push it outward when d2 forces it. Setting d3 = d2 + minimum
        # separation for every seed would change d3 as well as ring 2, so the
        # seeds nearest the good pocket -- the ones that matter most -- would
        # differ from the reference design on two axes at once.
        d3_ref = float(best["d3_inner_mm"])
        d3 = max(d3_ref, d2 + ma.MIN_DIAM_SEPARATION_MM)
        d3 = min(max(d3, ma.BOUNDS_MIN[5]), ma.BOUNDS_MAX[5])
        if not ma.is_feasible_full(diam, n_ap, nd1, nd2, d2, d3):
            skipped.append((d2, nd2, "failed a feasibility rule"))
            continue
        rows.append((float(diam), n_ap, nd1, nd2, float(d2), float(d3)))

    if not rows:
        print("ERROR: no feasible seeds above the old 960 bound.")
        print("At every d2 tried, the packing ceiling was still <= 960, which would")
        print("mean the bound was never the binding constraint after all.")
        for d2, nd2, why in skipped:
            print(f"  d2={d2:.0f}: nd2={nd2} -- {why}")
        sys.exit(1)

    out = pd.DataFrame(rows, columns=ma.PARAM_NAMES)
    out.insert(0, "config", [f"{args.prefix}{i:03d}" for i in range(len(out))])

    clashes = [c for c in out["config"]
               if os.path.isdir(os.path.join(args.results_dir, str(c)))]
    if clashes:
        print(f"ERROR: work directories already exist for {clashes[:3]}. "
              f"Pass a different --prefix.")
        sys.exit(1)

    print(f"{'config':<12} {'n_det_ring2':>12} {'d2 (mm)':>9} {'d3 (mm)':>9} {'packing max':>12}")
    print("-" * 60)
    for _, r in out.iterrows():
        print(f"{r['config']:<12} {int(r['n_det_ring2']):>12} {r['d2_inner_mm']:>9.0f} "
              f"{r['d3_inner_mm']:>9.0f} {ma.max_crystals_on_ring(r['d2_inner_mm']):>12.0f}")
    if skipped:
        print(f"\n{len(skipped)} candidate(s) skipped:")
        for d2, nd2, why in skipped:
            print(f"  d2={d2:.0f}, nd2={nd2}: {why}")

    out.to_csv(args.out, index=False)
    print(f"\nwrote {len(out)} seeds to {args.out}")
    print(f"ring-2 counts span {int(out['n_det_ring2'].min())} to "
          f"{int(out['n_det_ring2'].max())}, all above the old 960 bound.")


if __name__ == "__main__":
    main()
