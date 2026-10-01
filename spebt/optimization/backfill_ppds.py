#!/usr/bin/env python3
"""
Backfill PPDS (Projection Probability Density Sensitivity) for previously
evaluated SAI SC-SPECT configurations.

Reads the existing results CSV, walks each row's work_dir, and computes
PPDS using compute_metrics.compute_ppds (no re-simulation needed — the
PPDF and beam-mask HDF5 files are already stored).

The original CSV is renamed to *.bak.<timestamp>.csv, and the new CSV with
an added `ppds_mean` column is written in place. Idempotent: rows that
already have a numeric ppds_mean are kept untouched unless --force is set.

Usage:
  python backfill_ppds.py \
      --results_csv results/results_summary_mobo.csv

  # Force recompute even for rows that already have a value:
  python backfill_ppds.py \
      --results_csv results/results_summary_mobo.csv --force
"""
import argparse
import os
import shutil
import sys
import time
import pandas as pd

from compute_metrics import compute_ppds


def main():
    parser = argparse.ArgumentParser(description="Backfill PPDS for existing MOBO results")
    parser.add_argument("--results_csv", required=True,
                        help="Path to results CSV to backfill in place")
    parser.add_argument("--force", action="store_true",
                        help="Recompute PPDS even if the row already has a value")
    parser.add_argument("--skip_missing_dir", action="store_true", default=True,
                        help="Skip rows whose work_dir does not exist")
    args = parser.parse_args()

    if not os.path.exists(args.results_csv):
        print(f"ERROR: results CSV not found: {args.results_csv}")
        sys.exit(1)

    df = pd.read_csv(args.results_csv)
    n = len(df)
    print(f"Loaded {n} rows from {args.results_csv}")

    if "ppds_mean" not in df.columns:
        df["ppds_mean"] = float("nan")
        print("Added empty ppds_mean column.")

    if "work_dir" not in df.columns:
        print("ERROR: results CSV has no 'work_dir' column; cannot locate PPDF files.")
        sys.exit(1)

    # Snapshot original
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = args.results_csv.replace(".csv", f".bak.{stamp}.csv")
    shutil.copy(args.results_csv, backup)
    print(f"Backup written: {backup}")

    n_done = n_skip_have = n_skip_missing = n_failed = 0
    for i, row in df.iterrows():
        config = row.get("config", f"row_{i}")
        work_dir = row.get("work_dir")
        existing = row.get("ppds_mean")

        if not args.force and pd.notna(existing):
            n_skip_have += 1
            continue
        if not isinstance(work_dir, str) or not work_dir:
            n_skip_missing += 1
            continue
        if not os.path.isdir(work_dir):
            print(f"  [skip] {config}: work_dir does not exist ({work_dir})")
            n_skip_missing += 1
            continue

        try:
            ppds = compute_ppds(work_dir)
        except Exception as e:
            print(f"  [fail] {config}: compute_ppds raised: {e}")
            n_failed += 1
            continue

        df.at[i, "ppds_mean"] = ppds
        if pd.notna(ppds):
            print(f"  [ok]   {config}: PPDS = {ppds:.4e}")
            n_done += 1
        else:
            print(f"  [nan]  {config}: PPDS = NaN (missing PPDF or mask files)")
            n_failed += 1

        # Save progress every 10 rows so a crash doesn't lose work
        if (n_done + n_failed) % 10 == 0 and (n_done + n_failed) > 0:
            df.to_csv(args.results_csv, index=False)

    df.to_csv(args.results_csv, index=False)

    print()
    print(f"Done. Computed: {n_done}, already had PPDS: {n_skip_have}, "
          f"skipped missing dir: {n_skip_missing}, failed: {n_failed}")
    print(f"Updated CSV: {args.results_csv}")
    print(f"Backup:      {backup}")


if __name__ == "__main__":
    main()
