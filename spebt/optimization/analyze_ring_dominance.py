#!/usr/bin/env python3
"""
Test RY's explanation for why the inner detector ring dominates PPDS.

His point (Jul 2026): with a fixed collimator ring configuration, the impact of
each detector ring declines outward, so ring 1's dominance is expected. Outer
rings should only matter when the inner rings have a large opening.

That is a testable claim. If outer-ring contribution only becomes relevant for
large apertures, then:
  1. ring 4's share of total PPDS should rise with aperture diameter
  2. ring 4's negative correlation with CNR should be driven by the large-aperture
     configs -- i.e. ring-4 PPDS is largely a proxy for aperture size, which
     itself costs resolution
  3. within a narrow aperture band, the ring correlations should weaken

Reads only columns already in the results CSV; nothing is recomputed.

Usage:
  python analyze_ring_dominance.py --results_csv results/results_summary_mobo.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

RING_COLS = [f"ppds_ring{i}" for i in range(1, 5)]


def spearman(a, b):
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def main():
    ap = argparse.ArgumentParser(description="Test the ring-dominance explanation")
    ap.add_argument("--results_csv", required=True)
    ap.add_argument("--cnr_col", default="cnr_mean",
                    help="CNR column to correlate against "
                         "(use cnr_sector_mean for the equally-weighted definition)")
    ap.add_argument("--n_bands", type=int, default=3,
                    help="Number of aperture-size bands to split into (default 3)")
    args = ap.parse_args()

    if not os.path.exists(args.results_csv):
        print(f"ERROR: results CSV not found: {args.results_csv}")
        sys.exit(1)

    df = pd.read_csv(args.results_csv)
    cnr_col = args.cnr_col
    need = RING_COLS + [cnr_col, "aperture_diam_mm", "n_apertures", "fwhm_mean"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        print(f"ERROR: missing columns: {missing}")
        sys.exit(1)

    d = df[need].dropna().copy()
    n = len(d)
    if n < 10:
        print(f"Only {n} complete rows; not enough to say anything.")
        sys.exit(1)

    d["ppds_total"] = d[RING_COLS].sum(axis=1)
    for i, c in enumerate(RING_COLS, start=1):
        d[f"share{i}"] = d[c] / d["ppds_total"]
    # Total open aperture area is what actually lets flux through to the far rings
    d["open_area"] = d["n_apertures"] * (d["aperture_diam_mm"] ** 2)

    crit = 1.96 / np.sqrt(n - 1)
    print("=" * 74)
    print("IS OUTER-RING PPDS JUST A PROXY FOR APERTURE OPENING?")
    print("=" * 74)
    print(f"\nn = {n};  approximate 5% critical |rho| = {crit:.3f}")

    print("\n1. Does each ring's SHARE of total PPDS track the aperture opening?")
    print(f"\n{'':<10} {'vs aperture_diam':>18} {'vs n_apertures':>16} {'vs open area':>14}")
    print("-" * 62)
    for i in range(1, 5):
        s = d[f"share{i}"]
        print(f"  ring {i}   {spearman(s, d['aperture_diam_mm']):>18.3f} "
              f"{spearman(s, d['n_apertures']):>16.3f} "
              f"{spearman(s, d['open_area']):>14.3f}")

    print(f"\n2. How does each ring correlate with {cnr_col}, and with FWHM?")
    print(f"\n{'':<10} {'rho vs CNR':>12} {'rho vs FWHM':>13}")
    print("-" * 38)
    for i, c in enumerate(RING_COLS, start=1):
        print(f"  ring {i}   {spearman(d[c], d[cnr_col]):>12.3f} "
              f"{spearman(d[c], d['fwhm_mean']):>13.3f}")
    print(f"\n  aperture_diam vs CNR : {spearman(d['aperture_diam_mm'], d[cnr_col]):.3f}")
    print(f"  aperture_diam vs FWHM: {spearman(d['aperture_diam_mm'], d['fwhm_mean']):.3f}")

    print("\n3. Within narrow aperture bands, does the ring signal survive?")
    print("   (if the ring effect is really an aperture effect, it should weaken)")
    d["band"] = pd.qcut(d["aperture_diam_mm"], args.n_bands,
                        labels=[f"band{i+1}" for i in range(args.n_bands)])
    print(f"\n{'band':<8} {'n':>4} {'aperture range':>18} "
          + " ".join(f"{'r'+str(i):>7}" for i in range(1, 5)))
    print("-" * 62)
    for band, g in d.groupby("band", observed=True):
        if len(g) < 8:
            print(f"{str(band):<8} {len(g):>4}   (too few rows)")
            continue
        rng = f"{g['aperture_diam_mm'].min():.2f}-{g['aperture_diam_mm'].max():.2f}"
        rhos = " ".join(f"{spearman(g[c], g[cnr_col]):>7.3f}" for c in RING_COLS)
        print(f"{str(band):<8} {len(g):>4} {rng:>18} {rhos}")

    print(f"""
How to read this:
  - If ring 4's share climbs with aperture opening, that supports the picture of
    flux only reaching the outer rings once the inner openings are large.
  - If ring 4 correlates with CNR about as strongly as aperture diameter does,
    ring-4 PPDS is mostly re-measuring aperture size rather than adding
    information.
  - If the per-band correlations collapse toward zero, the ring effect was an
    aperture effect all along. If ring 1 stays positive within every band, its
    signal is real and independent of aperture size.
""")


if __name__ == "__main__":
    main()
