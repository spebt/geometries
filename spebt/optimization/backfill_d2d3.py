#!/usr/bin/env python3
"""
Stamp the fixed D2/D3 ring diameters onto configs evaluated before the D2/D3
expansion, so they can warm-start the 6D MOBO loop.

Every config evaluated up to the 180-iteration 5-objective campaign was built
with the hard-coded ring layout [260, 390, 520, 650], i.e. d2_inner_mm = 390 and
d3_inner_mm = 520. They are therefore valid 6D samples that happen to lie on one
slice of the new design space.

This matters more than it looks: mobo_agent does df[PARAM_NAMES].dropna(), so if
the two new columns are absent or NaN, EVERY historical row is dropped and the
6D loop starts with no training data at all. Run this before the first 6D
iteration.

Only rows that already have PPDF-derived metrics are stamped -- a row with no
metrics is not a real evaluation and should not become a training point.

Usage:
  python backfill_d2d3.py --results_csv results/results_summary_mobo.csv
  python backfill_d2d3.py --results_csv ... --dry_run
"""
import argparse
import os
import shutil
import sys
import time

import pandas as pd

# The ring layout every pre-expansion config was generated with.
LEGACY_D2_INNER_MM = 390.0
LEGACY_D3_INNER_MM = 520.0


def main():
    parser = argparse.ArgumentParser(
        description="Stamp legacy D2/D3 diameters onto pre-expansion configs")
    parser.add_argument("--results_csv", required=True,
                        help="Path to results CSV to update in place")
    parser.add_argument("--d2_inner_mm", type=float, default=LEGACY_D2_INNER_MM)
    parser.add_argument("--d3_inner_mm", type=float, default=LEGACY_D3_INNER_MM)
    parser.add_argument("--dry_run", action="store_true",
                        help="Report what would change without writing")
    args = parser.parse_args()

    if not os.path.exists(args.results_csv):
        print(f"ERROR: results CSV not found: {args.results_csv}")
        sys.exit(1)

    df = pd.read_csv(args.results_csv)
    print(f"Loaded {len(df)} rows from {args.results_csv}")

    for col, val in (("d2_inner_mm", args.d2_inner_mm),
                     ("d3_inner_mm", args.d3_inner_mm)):
        if col not in df.columns:
            df[col] = float("nan")
            print(f"Added empty {col} column.")

    # A row is a real evaluation if it produced any PPDF-derived metric. Rows
    # that are entirely empty (or were force-zeroed before metrics ran) should
    # not be stamped, since they are not usable training points anyway.
    evaluated = df["fwhm_mean"].notna() if "fwhm_mean" in df.columns else df.index.to_series().notna()
    needs_d2 = df["d2_inner_mm"].isna() & evaluated
    needs_d3 = df["d3_inner_mm"].isna() & evaluated
    n_d2, n_d3 = int(needs_d2.sum()), int(needs_d3.sum())

    print(f"\nRows with metrics:            {int(evaluated.sum())}")
    print(f"Rows needing d2_inner_mm:     {n_d2}  -> {args.d2_inner_mm}")
    print(f"Rows needing d3_inner_mm:     {n_d3}  -> {args.d3_inner_mm}")

    already = int((df["d2_inner_mm"].notna() & evaluated).sum())
    if already:
        print(f"Rows that already have D2/D3: {already} (left untouched)")

    if args.dry_run:
        print("\n--dry_run set; nothing written.")
        return

    if n_d2 == 0 and n_d3 == 0:
        print("\nNothing to do.")
        return

    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = args.results_csv.replace(".csv", f".bak.{stamp}.csv")
    shutil.copy(args.results_csv, backup)
    print(f"\nBackup written: {backup}")

    df.loc[needs_d2, "d2_inner_mm"] = args.d2_inner_mm
    df.loc[needs_d3, "d3_inner_mm"] = args.d3_inner_mm
    df.to_csv(args.results_csv, index=False)

    print(f"Updated CSV:    {args.results_csv}")

    # The 6D agent needs every design column present to use a row for training.
    design_cols = ["aperture_diam_mm", "n_apertures", "n_det_ring1",
                   "n_det_ring2", "d2_inner_mm", "d3_inner_mm"]
    have_all = [c for c in design_cols if c in df.columns]
    n_usable = len(df[have_all].dropna()) if len(have_all) == len(design_cols) else 0
    print(f"\nRows with a complete 6D design vector: {n_usable}")
    if n_usable == 0:
        print("WARNING: no usable training rows -- the 6D loop would start cold.")


if __name__ == "__main__":
    main()
