#!/usr/bin/env python3
"""
Fold the D2/D3 seed evaluations into the main results CSV.

The seeds write to per-task files rather than the main CSV, because the
controller is normally running at the same time and compute_cnr.py does a full
read-modify-write -- concurrent writers would silently drop rows. This merges
them once the seed array has finished.

Run this while the controller is NOT mid-write if possible; it takes a backup
either way. The merge is idempotent: a config already present in the main CSV is
skipped rather than duplicated.

Usage:
  python merge_d2d3_seeds.py --results_csv results/results_summary_mobo.csv
  python merge_d2d3_seeds.py --results_csv ... --dry_run
"""
import argparse
import glob
import os
import shutil
import sys
import time

import pandas as pd

import mobo_agent as ma


def main():
    ap = argparse.ArgumentParser(description="Merge D2/D3 seed results")
    ap.add_argument("--results_csv", required=True)
    ap.add_argument("--seed_dir", default="results/d2d3_seed_out")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.results_csv):
        print(f"ERROR: results CSV not found: {args.results_csv}")
        sys.exit(1)

    files = sorted(glob.glob(os.path.join(args.seed_dir, "task_*.csv")))
    if not files:
        print(f"No seed results in {args.seed_dir}. Has the array finished?")
        sys.exit(1)

    main_df = pd.read_csv(args.results_csv)
    existing = set(main_df["config"].dropna()) if "config" in main_df.columns else set()

    parts, skipped = [], 0
    for f in files:
        try:
            d = pd.read_csv(f)
        except Exception as e:
            print(f"  [warn] could not read {f}: {e}")
            continue
        for _, r in d.iterrows():
            if r.get("config") in existing:
                skipped += 1
                continue
            parts.append(r)

    if not parts:
        print(f"Nothing new to merge ({skipped} already present).")
        return

    new = pd.DataFrame(parts)
    print(f"Merging {len(new)} seed rows ({skipped} already present)\n")

    obj = [c for c in ma.OBJ_COLUMNS if c in new.columns]
    show = ["config", "d2_inner_mm", "d3_inner_mm"] + obj
    print(new[[c for c in show if c in new.columns]].to_string(index=False))

    complete = len(new.dropna(subset=[c for c in ma.OBJ_COLUMNS if c in new.columns]))
    print(f"\n{complete} of {len(new)} seeds have all five objectives")
    if complete < len(new):
        print("  (incomplete rows still merge; they just will not train the GP)")

    if args.dry_run:
        print("\n--dry_run set; nothing written.")
        return

    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = args.results_csv.replace(".csv", f".bak.{stamp}.csv")
    shutil.copy(args.results_csv, backup)

    # Align to the main CSV's columns so the merge cannot shift fields
    for c in main_df.columns:
        if c not in new.columns:
            new[c] = float("nan")
    new = new.reindex(columns=main_df.columns)
    pd.concat([main_df, new], ignore_index=True).to_csv(args.results_csv, index=False)

    print(f"\nBackup: {backup}")
    print(f"Updated: {args.results_csv}")

    merged = pd.read_csv(args.results_csv).dropna(subset=ma.OBJ_COLUMNS + ma.PARAM_NAMES)
    print(f"\nTraining rows now: {len(merged)}")
    for p in ("d2_inner_mm", "d3_inner_mm"):
        v = merged[p]
        print(f"  {p:<16} unique={v.nunique():>4}  std={v.std():>8.3f}  "
              f"range={v.min():.1f}-{v.max():.1f}")
    print("\nRestart the controller so it retrains on the new spread:")
    print("  scancel <controller job>; sbatch submit_mobo.sh")


if __name__ == "__main__":
    main()
