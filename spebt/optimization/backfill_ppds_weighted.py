#!/usr/bin/env python3
"""
Store per-ring PPDS contributions, then search ring weightings against CNR.

Background: overall sensitivity is a poor objective because wide, poorly
collimated PPDFs raise it while blurring the image and degrading CNR (Spearman
-0.919 across 109 configs). RY's Jul 2026 proposal was to weight each detector's
PPDF contribution by its ring, 1/2/3/4 from inner to outer, so counts on the
better-collimated outer rings count for more.

Measured, that weighting made things worse rather than better:

    PPDS unweighted     +0.140   (below the 0.189 significance threshold)
    PPDS 1,2,3,4        -0.319   (significantly NEGATIVE)
    sensitivity         -0.919

So rather than guess further weightings at ~70 min per attempt, this stores the
four per-ring partial sums. PPDS is a plain sum over detectors and ring weights
are constant within a ring, so

    PPDS_weighted = sum_r  w_r * ppds_ring_r

and any weighting becomes a dot product over stored columns -- evaluated in
milliseconds instead of re-reading gigabytes of HDF5. The per-ring correlations
also show directly which rings help and which hurt, which is what the single
aggregate number could not.

Usage:
  python backfill_ppds_weighted.py --results_csv results/results_summary_mobo.csv
  python backfill_ppds_weighted.py --results_csv ... --analyze_only
  python backfill_ppds_weighted.py --results_csv ... --analyze_only --search
"""
import argparse
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

from compute_metrics import compute_ppds_per_ring

RING_COLS = [f"ppds_ring{i}" for i in range(1, 5)]
WEIGHTED_COL = "ppds_weighted_mean"

# Weightings worth reporting. RY's proposal plus its reverse and a few shapes
# that test whether the inner or outer rings carry the CNR-relevant signal.
CANDIDATE_WEIGHTS = {
    "unweighted    (1,1,1,1)": (1.0, 1.0, 1.0, 1.0),
    "RY proposal   (1,2,3,4)": (1.0, 2.0, 3.0, 4.0),
    "reversed      (4,3,2,1)": (4.0, 3.0, 2.0, 1.0),
    "steep outer   (1,2,4,8)": (1.0, 2.0, 4.0, 8.0),
    "steep inner   (8,4,2,1)": (8.0, 4.0, 2.0, 1.0),
    "inner only    (1,0,0,0)": (1.0, 0.0, 0.0, 0.0),
    "ring2 only    (0,1,0,0)": (0.0, 1.0, 0.0, 0.0),
    "ring3 only    (0,0,1,0)": (0.0, 0.0, 1.0, 0.0),
    "outer only    (0,0,0,1)": (0.0, 0.0, 0.0, 1.0),
}


def _spearman(a: pd.Series, b: pd.Series) -> float:
    return float(a.corr(b, method="spearman"))


def report(df: pd.DataFrame, do_search: bool = False,
           cnr_col: str = "cnr_mean") -> None:
    if cnr_col not in df.columns:
        print(f"\n(no {cnr_col} column; skipping correlation report)")
        return
    have = [c for c in RING_COLS if c in df.columns]
    if len(have) != 4:
        print(f"\n(per-ring columns missing: {set(RING_COLS) - set(have)})")
        return

    sub = df[RING_COLS + [cnr_col]].dropna()
    n = len(sub)
    if n < 3:
        print(f"\n(only {n} rows with per-ring PPDS and CNR; nothing to correlate)")
        return

    crit = 1.96 / np.sqrt(n - 1)
    print()
    print("=" * 72)
    print("WHICH RINGS CARRY CNR-RELEVANT SIGNAL?")
    print("=" * 72)
    print(f"\nn = {n};  approximate 5% critical |rho| = {crit:.3f}\n")

    print(f"{'per-ring contribution':<28} {'rho vs CNR':>11}")
    print("-" * 42)
    for i, col in enumerate(RING_COLS, start=1):
        rho = _spearman(sub[col], sub[cnr_col])
        flag = "" if abs(rho) >= crit else "   (n.s.)"
        print(f"  ring {i} (inner->outer){'':<7} {rho:>+11.3f}{flag}")

    print(f"\n{'weighting':<28} {'rho vs CNR':>11}")
    print("-" * 42)
    rows = []
    for label, w in CANDIDATE_WEIGHTS.items():
        combined = sub[RING_COLS].values @ np.asarray(w, dtype=np.float64)
        rho = _spearman(pd.Series(combined, index=sub.index), sub[cnr_col])
        rows.append((rho, label))
        flag = "" if abs(rho) >= crit else "   (n.s.)"
        print(f"  {label:<26} {rho:>+11.3f}{flag}")

    if "sensitivity_mean" in df.columns:
        s = df[["sensitivity_mean", cnr_col]].dropna()
        if len(s) >= 3:
            print(f"\n  {'sensitivity (incumbent)':<26} "
                  f"{_spearman(s['sensitivity_mean'], s[cnr_col]):>+11.3f}")

    if do_search:
        rng = np.random.default_rng(0)
        # Random search over the simplex. Spearman of a weighted sum is not
        # linear in the weights, so there is no closed form; 20k dot products
        # over a few hundred rows is instant anyway.
        best_rho, best_w = -2.0, None
        X = sub[RING_COLS].values
        y = sub[cnr_col]
        for _ in range(20000):
            w = rng.dirichlet(np.ones(4))
            rho = _spearman(pd.Series(X @ w, index=sub.index), y)
            if rho > best_rho:
                best_rho, best_w = rho, w
        print(f"\nBest weighting found by random search over the simplex:")
        print(f"  weights = ({', '.join(f'{v:.3f}' for v in best_w)})")
        print(f"  rho vs CNR = {best_rho:+.3f}")
        if best_rho < crit:
            print("  -> even the best weighting is not significant; no linear")
            print("     combination of ring contributions tracks CNR.")

    best = max(rows)
    print(f"""
How to read this:
  - The per-ring rows are the diagnostic. If the inner and outer rings carry
    opposite signs, that explains why a monotonic weighting like 1,2,3,4 can be
    worse than no weighting at all.
  - We want a weighting strongly POSITIVE against CNR, so it can replace
    sensitivity without rewarding the wide, blurring PPDFs sensitivity rewards.
  - Best candidate here: {best[1].strip()} at rho = {best[0]:+.3f}
  - If nothing clears {crit:.3f}, PPDS in any ring weighting is not a usable
    stand-in for sensitivity, and that is the result to take back to RY.
""")


def main():
    parser = argparse.ArgumentParser(
        description="Store per-ring PPDS and search ring weightings against CNR")
    parser.add_argument("--results_csv", required=True)
    parser.add_argument("--cnr_col", default="cnr_mean",
                        help="CNR column to correlate against")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if the row already has values")
    parser.add_argument("--analyze_only", action="store_true",
                        help="Only print the report; compute nothing")
    parser.add_argument("--search", action="store_true",
                        help="Also random-search the weight simplex for the best rho")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after this many computed rows (for timing)")
    args = parser.parse_args()

    if not os.path.exists(args.results_csv):
        print(f"ERROR: results CSV not found: {args.results_csv}")
        sys.exit(1)

    df = pd.read_csv(args.results_csv)
    print(f"Loaded {len(df)} rows from {args.results_csv}")

    if args.analyze_only:
        report(df, args.search, args.cnr_col)
        return

    for required in ("work_dir", "n_det_ring1"):
        if required not in df.columns:
            print(f"ERROR: results CSV has no '{required}' column.")
            sys.exit(1)

    for c in RING_COLS + [WEIGHTED_COL]:
        if c not in df.columns:
            df[c] = float("nan")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = args.results_csv.replace(".csv", f".bak.{stamp}.csv")
    shutil.copy(args.results_csv, backup)
    print(f"Backup written: {backup}\n")

    n_done = n_skip = n_fail = 0
    t_start = time.time()
    for i, row in df.iterrows():
        if args.limit is not None and n_done >= args.limit:
            print(f"\nReached --limit {args.limit}; stopping early.")
            break

        config = row.get("config", f"row_{i}")
        work_dir = row.get("work_dir")
        n_det_ring1 = row.get("n_det_ring1")

        if not args.force and all(pd.notna(row.get(c)) for c in RING_COLS):
            n_skip += 1
            continue
        if not isinstance(work_dir, str) or not os.path.isdir(work_dir):
            n_skip += 1
            continue
        if pd.isna(n_det_ring1):
            n_skip += 1
            continue

        try:
            comps = compute_ppds_per_ring(work_dir, int(n_det_ring1))
        except Exception as e:
            print(f"  [fail] {config}: {e}")
            n_fail += 1
            continue

        if comps is None:
            print(f"  [nan]  {config}: missing PPDF or mask files")
            n_fail += 1
            continue

        for c, v in zip(RING_COLS, comps):
            df.at[i, c] = float(v)
        # Keep the aggregate column in sync with RY's original proposal so the
        # earlier numbers remain reproducible from the CSV alone.
        df.at[i, WEIGHTED_COL] = float(np.dot((1.0, 2.0, 3.0, 4.0), comps))
        print(f"  [ok]   {config[:46]:<46} "
              + "  ".join(f"r{j+1}={v:.3e}" for j, v in enumerate(comps)))
        n_done += 1

        if (n_done + n_fail) % 10 == 0 and (n_done + n_fail) > 0:
            df.to_csv(args.results_csv, index=False)

    df.to_csv(args.results_csv, index=False)

    elapsed = time.time() - t_start
    print(f"\nDone. Computed: {n_done}, skipped: {n_skip}, failed: {n_fail}")
    if n_done:
        print(f"Elapsed: {elapsed / 60:.1f} min ({elapsed / n_done:.1f} s per config)")
    print(f"Updated CSV: {args.results_csv}")
    print(f"Backup:      {backup}")

    report(df, args.search, args.cnr_col)


if __name__ == "__main__":
    main()
