#!/usr/bin/env python3
"""
List the configs worth reconstructing so they can carry the section-mean CNR.

Switching the objective to the equally-weighted section mean costs us any config
without a saved reconstruction, since the sections are computed from the recon
image. Configs from the pre-in-loop era have cnr_mean from the old offline
pipeline but no recon file, so they would silently drop out of the training set.

This finds the ones worth recovering: they have PPDF files (so a reconstruction
is possible), have the other objectives already, and lack both a saved recon and
a section-mean value. Writes one work_dir per line for the SLURM array.

Usage:
  python make_recon_backfill_list.py --results_csv results/results_summary_mobo.csv
"""
import argparse
import glob
import os
import sys

import pandas as pd

OBJ_COLS = ["fwhm_mean", "asci_pct", "mpxi_mean"]


def has_recon(work_dir: str) -> bool:
    if os.path.exists(os.path.join(work_dir, "cnr_inloop", "recon_mlem_T8.npz")):
        return True
    return bool(glob.glob(os.path.join(work_dir, "cnr_repeat_seed*", "recon_mlem_T8.npz")))


def main():
    ap = argparse.ArgumentParser(description="List configs needing reconstruction")
    ap.add_argument("--results_csv", required=True)
    ap.add_argument("--out", default="recon_backfill_configs.txt")
    # store_true with default=True could never be switched off, so the wider
    # set of configs -- those with no CNR at all -- was unreachable.
    ap.add_argument("--require_cnr", dest="require_cnr", action="store_true",
                    default=True,
                    help="Only configs that already have a CNR value (default)")
    ap.add_argument("--all_missing", dest="require_cnr", action="store_false",
                    help="Also include configs with no CNR at all. These have "
                         "real FWHM/ASCI/MPXI/PPDS but no reconstruction, so "
                         "reconstructing them makes their rows fully usable "
                         "instead of merely excluded from training.")
    args = ap.parse_args()

    if not os.path.exists(args.results_csv):
        print(f"ERROR: results CSV not found: {args.results_csv}")
        sys.exit(1)

    df = pd.read_csv(args.results_csv)
    print(f"Loaded {len(df)} rows")

    have_sector = df["cnr_sector_mean"].notna() if "cnr_sector_mean" in df.columns else pd.Series(False, index=df.index)
    have_cnr = df["cnr_mean"].notna() if "cnr_mean" in df.columns else pd.Series(False, index=df.index)
    have_objs = df[[c for c in OBJ_COLS if c in df.columns]].notna().all(axis=1)

    picks, no_ppdf, no_dir = [], 0, 0
    for i, row in df.iterrows():
        if have_sector.iloc[i]:
            continue
        if args.require_cnr and not have_cnr.iloc[i]:
            continue
        if not have_objs.iloc[i]:
            continue
        wd = row.get("work_dir")
        if not isinstance(wd, str) or not os.path.isdir(wd):
            no_dir += 1
            continue
        if has_recon(wd):
            continue   # recompute_cnr_sectors can already use it
        if not glob.glob(os.path.join(wd, "position_*_ppdfs_t8_*.hdf5")):
            no_ppdf += 1
            continue
        picks.append(wd)

    with open(args.out, "w") as f:
        f.write("\n".join(picks) + ("\n" if picks else ""))

    print(f"\nAlready have section CNR : {int(have_sector.sum())}")
    print(f"Need reconstruction      : {len(picks)}")
    print(f"Skipped, no PPDF files   : {no_ppdf}")
    print(f"Skipped, work_dir gone   : {no_dir}")
    print(f"\nWrote {args.out}")
    if picks:
        print(f"\nSubmit with:\n  sbatch --array=0-{len(picks) - 1} submit_recon_backfill.sh")
        print("\nThen fold the new reconstructions in:")
        print("  python3 recompute_cnr_sectors.py --results_csv results/results_summary_mobo.csv")


if __name__ == "__main__":
    main()
