#!/usr/bin/env python3
"""
One-time cleanup of LHS results CSV:
  1. Remove duplicate rows (keep first occurrence per config)
  2. Add JI=0 rows for infeasible configs (10, 11, 20)
  3. Backup original file

Usage:
  python cleanup_lhs_results.py --csv results/results_summary.csv
"""
import argparse
import os
import shutil
import pandas as pd
import numpy as np


INFEASIBLE_CONFIGS = [
    {"config": "lhs_10_ap0.879741_nap308", "aperture_diam_mm": 0.879741, "n_apertures": 308,
     "work_dir": "/vscratch/grp-rutaoyao/Omer/spebt/optimization/config_0010_ap0.8797_nap308"},
    {"config": "lhs_11_ap0.904938_nap313", "aperture_diam_mm": 0.904938, "n_apertures": 313,
     "work_dir": "/vscratch/grp-rutaoyao/Omer/spebt/optimization/config_0011_ap0.9049_nap313"},
    {"config": "lhs_20_ap0.762011_nap350", "aperture_diam_mm": 0.762011, "n_apertures": 350,
     "work_dir": "/vscratch/grp-rutaoyao/Omer/spebt/optimization/config_0020_ap0.7620_nap350"},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    # Backup
    backup = args.csv + ".bak"
    shutil.copy2(args.csv, backup)
    print(f"Backup saved to {backup}")

    df = pd.read_csv(args.csv)
    print(f"Original: {len(df)} rows")

    # Deduplicate
    df = df.drop_duplicates(subset=["config"], keep="first")
    print(f"After dedup: {len(df)} rows")

    # Add infeasible configs
    for cfg in INFEASIBLE_CONFIGS:
        if cfg["config"] not in df["config"].values:
            row = {
                "fwhm_mean": np.nan,
                "sensitivity_total": np.nan,
                "sensitivity_mean": np.nan,
                "asci_pct": np.nan,
                "n_ppdf_files": 0,
                "JI": 0.0,
                **cfg,
            }
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            print(f"  Added infeasible: {cfg['config']}")

    # Sort by config name for readability
    df = df.sort_values("config").reset_index(drop=True)

    df.to_csv(args.csv, index=False)
    print(f"Final: {len(df)} rows written to {args.csv}")

    # Summary
    feasible = df[df["JI"] > 0]
    print(f"\nFeasible: {len(feasible)}, Infeasible: {len(df) - len(feasible)}")
    if len(feasible) > 0:
        best = feasible.loc[feasible["JI"].idxmax()]
        print(f"Best: {best['config']} | JI={best['JI']:.6e} | d={best['aperture_diam_mm']:.4f} n={int(best['n_apertures'])}")


if __name__ == "__main__":
    main()
