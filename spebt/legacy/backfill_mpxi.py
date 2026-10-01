#!/usr/bin/env python3
"""
Backfill MPXI values for existing results in the CSV.

Reads each row's work_dir, computes MPXI from beam mask files,
and writes an updated CSV with the mpxi_mean column.

Usage:
  python backfill_mpxi.py --input results/results_summary.csv --output results/results_summary_mobo.csv
  python backfill_mpxi.py --input results/results_summary_4d.csv --output results/results_summary_mobo.csv
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_metrics import compute_mpxi


def main():
    parser = argparse.ArgumentParser(description="Backfill MPXI into results CSV")
    parser.add_argument("--input", type=str, required=True, help="Input CSV path")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    mpxi_values = []
    for i, row in df.iterrows():
        work_dir = row.get("work_dir", "")
        if pd.isna(work_dir) or not work_dir or not os.path.isdir(str(work_dir)):
            print(f"  [{i}] {row.get('config', '?')}: work_dir missing or inaccessible -> NaN")
            mpxi_values.append(float("nan"))
            continue

        mpxi = compute_mpxi(str(work_dir))
        status = f"{mpxi:.4f}" if not pd.isna(mpxi) else "NaN (no mask files)"
        print(f"  [{i}] {row.get('config', '?')}: MPXI = {status}")
        mpxi_values.append(mpxi)

    df["mpxi_mean"] = mpxi_values

    n_valid = sum(1 for v in mpxi_values if not pd.isna(v))
    print(f"\nComputed MPXI for {n_valid}/{len(df)} configs")

    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
