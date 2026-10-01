#!/usr/bin/env python3
"""
Recompute CNR as the equally-weighted mean over rod-size sections.

RY (Jul 2026): each rod size should carry the same weight, because the CNR of
the rods at a given size represents imaging performance at that object size
(tied to the PPDF FWHM). So the objective should be

    CNR = mean_i CNR_i,   i = 1..6 rod-size sections

Our existing `overall_cnr` instead pools every hot pixel into one number, which
weights each size by its pixel area — the 0.225 mm rods therefore dominate and
small-rod performance, where resolution actually bites, is underweighted.

This does NOT re-run ML-EM. Every config evaluated through the in-loop CNR step
saved its reconstruction (cnr_inloop/recon_mlem_T8.npz), and the section CNRs
are a cheap masking operation on top of that — seconds per config rather than
the ~22 minutes a reconstruction costs.

Requires the sector-mask wrap fix in run_recon_comparison.compute_cnr; before
that fix the three sections centred past 180 degrees over-collected and their
values were meaningless.

Writes cnr_sector_mean plus the six per-section values, then reports whether the
change in definition reorders the designs — which is what actually matters, since
a monotone rescaling would leave every conclusion from the campaign intact.

Usage:
  python recompute_cnr_sectors.py --results_csv results/results_summary_mobo.csv
  python recompute_cnr_sectors.py --results_csv ... --analyze_only
"""
import argparse
import glob
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..", "recon")))
import run_recon_comparison as rrc  # noqa: E402

DEFAULT_PHANTOM = ("/vscratch/grp-rutaoyao/Omer/spebt/spebt/spebt/data/"
                   "sai_10mm/hot_rods_phantom_10.0_mm_x_10.0_mm.pt")
N_SECTORS = 6
MEAN_COL = "cnr_sector_mean"
SECTOR_COLS = [f"cnr_sector{i}" for i in range(N_SECTORS)]


def find_recon(work_dir: str):
    """Newest saved reconstruction for a config, or None.

    Prefers the in-loop reconstruction; falls back to any repeat-run seed so
    configs that were only ever reconstructed by the repeat study still count.
    """
    candidates = [os.path.join(work_dir, "cnr_inloop", "recon_mlem_T8.npz")]
    candidates += sorted(glob.glob(os.path.join(work_dir, "cnr_repeat_seed*",
                                                "recon_mlem_T8.npz")))
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def report(df: pd.DataFrame) -> None:
    if MEAN_COL not in df.columns or "cnr_mean" not in df.columns:
        print("\n(need both cnr_mean and cnr_sector_mean to compare)")
        return
    d = df[["config", "cnr_mean", MEAN_COL]].dropna()
    if len(d) < 5:
        print(f"\n(only {len(d)} rows have both definitions)")
        return

    print()
    print("=" * 74)
    print("DOES THE NEW CNR DEFINITION REORDER THE DESIGNS?")
    print("=" * 74)
    rho = d["cnr_mean"].corr(d[MEAN_COL], method="spearman")
    pear = d["cnr_mean"].corr(d[MEAN_COL], method="pearson")
    print(f"\nn = {len(d)}")
    print(f"Spearman(old, new) = {rho:.4f}   Pearson = {pear:.4f}")
    print(f"old  mean {d['cnr_mean'].mean():.3f}  range {d['cnr_mean'].min():.2f}-{d['cnr_mean'].max():.2f}")
    print(f"new  mean {d[MEAN_COL].mean():.3f}  range {d[MEAN_COL].min():.2f}-{d[MEAN_COL].max():.2f}")

    old_rank = d["cnr_mean"].rank(ascending=False)
    new_rank = d[MEAN_COL].rank(ascending=False)
    moved = (old_rank - new_rank).abs()
    print(f"\nrank movement: mean {moved.mean():.1f} places, max {moved.max():.0f}")

    print("\nTop 8 by the NEW definition (old rank in brackets):")
    top_new = d.assign(old_rank=old_rank, new_rank=new_rank).sort_values(MEAN_COL, ascending=False).head(8)
    for _, r in top_new.iterrows():
        print(f"  {r['config'][:46]:<46} new {r[MEAN_COL]:.3f}  "
              f"old {r['cnr_mean']:.3f}  [rank {int(r['old_rank'])} -> {int(r['new_rank'])}]")

    print("\nTop 8 by the OLD definition (for comparison):")
    top_old = d.assign(old_rank=old_rank, new_rank=new_rank).sort_values("cnr_mean", ascending=False).head(8)
    for _, r in top_old.iterrows():
        print(f"  {r['config'][:46]:<46} old {r['cnr_mean']:.3f}  "
              f"new {r[MEAN_COL]:.3f}  [rank {int(r['old_rank'])} -> {int(r['new_rank'])}]")

    sec = [c for c in SECTOR_COLS if c in df.columns]
    if len(sec) == N_SECTORS:
        s = df[sec].dropna()
        if len(s):
            print(f"\nPer-section CNR across {len(s)} configs (section 0 = smallest rods):")
            for i, c in enumerate(sec):
                print(f"  section {i}:  mean {s[c].mean():6.3f}   "
                      f"range {s[c].min():5.2f} - {s[c].max():5.2f}")

    print(f"""
How to read this:
  - If Spearman(old, new) is near 1.0, the definitions rank designs the same way
    and every conclusion from the campaign carries over unchanged.
  - If it is meaningfully below 1.0, the equal-weighting changes which designs
    look best, and the campaign's top designs need re-examining under the new
    objective before the 6D run.
  - Section 0 holds the smallest rods. It will have the lowest CNR; the point of
    the new definition is that it now counts as much as the largest.
""")


def main():
    ap = argparse.ArgumentParser(description="Recompute CNR as an equally-weighted section mean")
    ap.add_argument("--results_csv", required=True)
    ap.add_argument("--phantom_path", default=DEFAULT_PHANTOM)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--analyze_only", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.results_csv):
        print(f"ERROR: results CSV not found: {args.results_csv}")
        sys.exit(1)

    df = pd.read_csv(args.results_csv)
    print(f"Loaded {len(df)} rows from {args.results_csv}")

    if args.analyze_only:
        report(df)
        return

    if not os.path.exists(args.phantom_path):
        print(f"ERROR: phantom not found: {args.phantom_path}")
        sys.exit(1)
    if "work_dir" not in df.columns:
        print("ERROR: results CSV has no 'work_dir' column.")
        sys.exit(1)

    for c in [MEAN_COL] + SECTOR_COLS:
        if c not in df.columns:
            df[c] = float("nan")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = args.results_csv.replace(".csv", f".bak.{stamp}.csv")
    shutil.copy(args.results_csv, backup)
    print(f"Backup written: {backup}\n")

    n_done = n_norecon = n_fail = n_skip = 0
    t0 = time.time()
    for i, row in df.iterrows():
        if args.limit is not None and n_done >= args.limit:
            print(f"\nReached --limit {args.limit}; stopping early.")
            break

        config = row.get("config", f"row_{i}")
        work_dir = row.get("work_dir")

        if not args.force and pd.notna(row.get(MEAN_COL)):
            n_skip += 1
            continue
        if not isinstance(work_dir, str) or not os.path.isdir(work_dir):
            n_skip += 1
            continue

        recon_path = find_recon(work_dir)
        if recon_path is None:
            n_norecon += 1
            continue

        try:
            res = rrc.compute_cnr(recon_path, args.phantom_path,
                                  os.path.dirname(recon_path))
        except Exception as e:
            print(f"  [fail] {config}: {e}")
            n_fail += 1
            continue

        sectors = np.asarray(res["sector_cnrs"], dtype=float)
        if sectors.size != N_SECTORS or not np.isfinite(sectors).all():
            print(f"  [warn] {config}: {np.isnan(sectors).sum()} non-finite sections")
        for j in range(min(N_SECTORS, sectors.size)):
            df.at[i, SECTOR_COLS[j]] = sectors[j]
        df.at[i, MEAN_COL] = float(np.nanmean(sectors))

        print(f"  [ok]   {config[:44]:<44} sector-mean={np.nanmean(sectors):.3f}  "
              f"(pooled={res['overall_cnr']:.3f})")
        n_done += 1

        if (n_done + n_fail) % 20 == 0 and (n_done + n_fail) > 0:
            df.to_csv(args.results_csv, index=False)

    df.to_csv(args.results_csv, index=False)
    elapsed = time.time() - t0
    print(f"\nDone. Computed: {n_done}, no saved reconstruction: {n_norecon}, "
          f"skipped: {n_skip}, failed: {n_fail}")
    if n_done:
        print(f"Elapsed: {elapsed/60:.1f} min ({elapsed/n_done:.1f} s per config)")
    print(f"Updated CSV: {args.results_csv}")

    report(df)


if __name__ == "__main__":
    main()
