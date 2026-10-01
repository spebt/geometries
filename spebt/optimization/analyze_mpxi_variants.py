#!/usr/bin/env python3
"""
Does MPXI's relationship with ASCI survive a matched FWHM window?

RY, Aug 2026, on the MPXI vs windowed-ASCI correlation: "This is puzzling and
reveals a mismatch. It would be relevant to assess the correlation between
windowed MPXI against windowed ASCI."

He is right that the comparison was mismatched. mpxi_mean counts every beam a
detector sees, including wide poorly-collimated ones. asci_pct_fwhm0p45 counts
only beams narrower than 0.45 mm. Correlating them compares two different beam
populations, so part of the +0.71 could be an artifact of the window rather
than a property of multiplexing.

This recomputes MPXI over the SAME windowed beam set and re-measures. Two
outcomes, both informative:

  - the correlation largely survives  -> it is a physical relationship between
    multiplexing and angular sampling, and the window was not driving it
  - it collapses                      -> the original number was the mismatch,
    and windowed MPXI is the honest quantity to use from here

It also reports the active-detector variant RY approved, which stops blind
detectors (k=0) being averaged into the score.

Correlations are printed in PHYSICAL units, not maximization space. The
max-space convention negates every minimized metric, which made a +0.33 between
measured MPXI and CNR print as -0.33 and read as though more multiplexing hurt
image quality -- the opposite of what the data says.

Usage:
  python analyze_mpxi_variants.py --results_csv results/results_summary_mobo.csv
  python analyze_mpxi_variants.py --results_csv ... --out results/mpxi_variants.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

import compute_metrics as cm

# What the variants are measured against. All in physical units.
OUTCOME_CNR = "cnr_sector_mean"
REFERENCE_COLS = ["asci_pct_fwhm0p45", "cnr_sector_mean", "fwhm_weighted_mean",
                  "ppds_ring1"]
VARIANT_COLS = ["mpxi_mean", "mpxi_active_mean", "mpxi_windowed_mean",
                "mpxi_windowed_active_mean"]

LABELS = {
    "mpxi_mean": "MPXI (original)",
    "mpxi_active_mean": "MPXI active-only",
    "mpxi_windowed_mean": "MPXI windowed",
    "mpxi_windowed_active_mean": "MPXI windowed+active",
}


def main():
    ap = argparse.ArgumentParser(description="Compare MPXI definitions")
    ap.add_argument("--results_csv", required=True)
    ap.add_argument("--out", default=None,
                    help="Write the per-config variants here so this expensive "
                         "pass is done once.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N configs (for a smoke test)")
    args = ap.parse_args()

    if not os.path.exists(args.results_csv):
        print(f"ERROR: results CSV not found: {args.results_csv}")
        sys.exit(1)

    df = pd.read_csv(args.results_csv)
    if "work_dir" not in df.columns:
        print("ERROR: results CSV has no work_dir column; cannot locate the "
              "beam files the variants are computed from.")
        sys.exit(1)

    df = df.dropna(subset=["work_dir"])
    if args.limit:
        df = df.head(args.limit)
    print(f"computing MPXI variants for {len(df)} configs\n", flush=True)

    rows, missing = [], 0
    for i, (_, r) in enumerate(df.iterrows(), 1):
        wd = str(r["work_dir"])
        if not os.path.isdir(wd):
            missing += 1
            continue
        v = cm.compute_mpxi_variants(wd)
        v["config"] = r.get("config")
        rows.append(v)
        if i % 25 == 0:
            print(f"  {i}/{len(df)} configs", flush=True)

    if not rows:
        print("ERROR: no config work_dirs were readable; nothing to compare.")
        sys.exit(1)
    if missing:
        print(f"\n[warn] {missing} configs had no readable work_dir and were skipped")

    var = pd.DataFrame(rows)
    merged = df.merge(var, on="config", how="inner", suffixes=("", "_new"))
    if args.out:
        var.to_csv(args.out, index=False)
        print(f"\nwrote per-config variants to {args.out}")

    have = [c for c in VARIANT_COLS if c in merged.columns]
    refs = [c for c in REFERENCE_COLS if c in merged.columns]

    print("\n" + "=" * 78)
    print("MPXI DEFINITIONS vs THE OTHER METRICS (Spearman, PHYSICAL units)")
    print("=" * 78)
    print("\n+ means the two measured quantities rise together. FWHM is a width,")
    print("so a NEGATIVE entry against FWHM means more multiplexing goes with")
    print("NARROWER beams -- which is the 'splitting' effect RY described.\n")

    hdr = f"{'definition':<24}" + "".join(f"{c.replace('_', ' ')[:16]:>18}" for c in refs)
    print(hdr)
    print("-" * len(hdr))
    for v in have:
        sub = merged[[v] + refs].dropna()
        if len(sub) < 10:
            print(f"{LABELS.get(v, v):<24}  too few configs ({len(sub)})")
            continue
        cells = "".join(f"{sub[v].corr(sub[c], method='spearman'):>18.2f}" for c in refs)
        print(f"{LABELS.get(v, v):<24}{cells}")

    print(f"\n(n = {len(merged[have + refs].dropna())} configs with every column)")

    # Is "maximize MPXI" actually right, or does it have an optimum?
    #
    # Spearman measures MONOTONE association. A metric that helps up to a point
    # and hurts past it can still show a healthy positive rho while "maximize"
    # is the wrong instruction. The scatter of CNR against MPXI looks like an
    # inverted U, so this bins it rather than trusting the eye or the
    # coefficient.
    key = "mpxi_windowed_active_mean"
    if key in merged.columns and OUTCOME_CNR in merged.columns:
        sub = merged[[key, OUTCOME_CNR]].dropna()
        if len(sub) >= 40:
            print("\n" + "=" * 78)
            print("IS MORE MPXI ALWAYS BETTER?")
            print("=" * 78)
            print(f"\nMean CNR within bins of {key}, {len(sub)} designs:\n")
            # Quantile bins keep the counts comparable; equal-width bins would
            # put almost everything in the first bin, since MPXI is right-skewed.
            try:
                bins = pd.qcut(sub[key], q=8, duplicates="drop")
            except ValueError:
                bins = pd.cut(sub[key], bins=8)
            g = sub.groupby(bins, observed=True)[OUTCOME_CNR].agg(["count", "mean", "max"])
            print(f"{'MPXI range':<24} {'n':>5} {'mean CNR':>10} {'best CNR':>10}")
            print("-" * 52)
            for interval, row in g.iterrows():
                lo, hi = interval.left, interval.right
                print(f"{f'{lo:.2f} to {hi:.2f}':<24} {int(row['count']):>5} "
                      f"{row['mean']:>10.3f} {row['max']:>10.3f}")

            peak = g["mean"].idxmax()
            top = g.index[-1]
            print(f"\nHighest mean CNR is in the {peak.left:.2f} to {peak.right:.2f} bin.")
            if peak != top:
                drop = g["mean"].iloc[-1] - g["mean"].max()
                print(f"The highest-MPXI bin is {abs(drop):.3f} BELOW that peak.")
                print("\n-> MPXI has an optimum, it is not monotone in CNR. Maximizing it")
                print("   without bound walks into designs that score well on MPXI and")
                print("   badly as scanners. Treating it as a target near the peak, or")
                print("   pairing it with CNR so CNR pushes back, is the safer use.")
            else:
                print("\n-> CNR is still rising in the top MPXI bin, so maximizing is")
                print("   defensible over the range explored so far.")

    # The direct question: how much of the original ASCI relationship was the
    # window mismatch rather than physics?
    if {"mpxi_mean", "mpxi_windowed_mean"}.issubset(merged.columns) and \
            "asci_pct_fwhm0p45" in merged.columns:
        sub = merged[["mpxi_mean", "mpxi_windowed_mean", "asci_pct_fwhm0p45"]].dropna()
        if len(sub) >= 10:
            a = sub["mpxi_mean"].corr(sub["asci_pct_fwhm0p45"], method="spearman")
            b = sub["mpxi_windowed_mean"].corr(sub["asci_pct_fwhm0p45"], method="spearman")
            print("\n" + "=" * 78)
            print("THE MISMATCH TEST")
            print("=" * 78)
            print(f"\n  unwindowed MPXI vs windowed ASCI   {a:+.2f}")
            print(f"  windowed   MPXI vs windowed ASCI   {b:+.2f}")
            print(f"  change                             {b - a:+.2f}")
            if abs(b - a) < 0.15:
                print("\n-> The relationship survives the matched window, so it is a")
                print("   property of multiplexing rather than an artifact of")
                print("   comparing different beam populations.")
            else:
                print("\n-> The window mismatch was carrying a substantial part of the")
                print("   original number. Windowed MPXI is the quantity to use.")
            agree = sub["mpxi_mean"].corr(sub["mpxi_windowed_mean"], method="spearman")
            print(f"\n  How much the two MPXI definitions agree with each other: {agree:+.2f}")
            print("  Note this does NOT mean the choice of definition is unimportant.")
            print("  Two metrics can rank designs almost identically and still")
            print("  differ sharply in how well they track the outcome, if the")
            print("  disagreement is concentrated among the best designs -- which")
            print("  are the only ones selection actually turns on. Compare the CNR")
            print("  column above across definitions before concluding it does not")
            print("  matter.")


if __name__ == "__main__":
    main()
