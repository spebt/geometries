#!/usr/bin/env python3
"""
Does an FWHM-windowed PPDS carry signal where the unwindowed one goes flat?

RY proposed this in Jul 2026. Ring-1 PPDS was adopted as the sensitivity
replacement at rho +0.60 against CNR, but split by aperture band it correlates
0.73 and 0.80 in the two larger bands and -0.03 in the smallest, 0.20 to
0.45 mm, which is exactly where every one of our best designs sits. So it
separates good regions from bad and says nothing inside the good one. By Aug it
had decayed to +0.21 over the full archive.

This is the last objective we have not re-examined, and every other one needed
it: sensitivity was removed, ASCI was windowed, FWHM was reweighted, MPXI was
redefined and its sign reversed.

Method, deliberately the same as the ASCI window sweep that produced the
0.45 mm threshold: recompute PPDS counting only beams narrower than each
candidate threshold, and correlate against reconstructed CNR. A threshold is
worth adopting only if it beats the unwindowed metric BOTH overall and inside
the small-aperture band, since that band is the one that matters.

Usage:
  python analyze_ppds_window.py --results_csv results/results_summary_mobo.csv
  python analyze_ppds_window.py --results_csv ... --limit 60
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

import compute_metrics as cm

THRESHOLDS = (0.40, 0.45, 0.50, 0.60, 0.80, 1.00)
OUTCOME = "cnr_sector_mean"
# The band containing every top design. Ring-1 PPDS is uninformative here, which
# is the whole reason for this test.
SMALL_APERTURE = (0.20, 0.45)


def configs_with_raw_files(df, work_dir_col="work_dir"):
    """Rows whose PPDF files still exist on disk.

    The vscratch purge strips raw simulation output from older designs, leaving
    the row in the CSV and a cnr/ subdirectory behind. Those rows look complete
    and recompute to NaN. Filtering them out here is what keeps a sweep from
    reporting a full table of NaN as though it were a measurement.
    """
    import glob as _glob
    keep = df[work_dir_col].astype(str).map(
        lambda w: bool(_glob.glob(os.path.join(w, "position_*_ppdfs_t8_*.hdf5"))))
    return df[keep]


def stratified_sample(df, limit, band=SMALL_APERTURE, col="aperture_diam_mm",
                      random_state=0):
    """Sample `limit` rows, keeping half inside the aperture band that matters.

    A plain head() picks the oldest rows, which are the ones the purge stripped.
    A plain random sample can leave too few designs inside the 0.20 to 0.45 mm
    band, and that band is the whole point: unwindowed ring-1 PPDS is
    uninformative there, so a sweep that under-samples it cannot answer the
    question it was run to answer.
    """
    if limit is None or len(df) <= limit:
        return df
    small = df[(df[col] >= band[0]) & (df[col] <= band[1])]
    rest = df[~df.index.isin(small.index)]
    n_small = min(len(small), max(limit // 2, 1))
    n_rest = min(len(rest), limit - n_small)
    return pd.concat([small.sample(n_small, random_state=random_state),
                      rest.sample(n_rest, random_state=random_state)])


def enough_values(series, minimum=10):
    """True when a column has enough non-NaN entries to correlate.

    Correlating a mostly-empty column returns NaN, and a table of NaN reads
    like a result. Callers should stop rather than print one.
    """
    return int(pd.Series(series).notna().sum()) >= minimum


def main():
    ap = argparse.ArgumentParser(description="Sweep an FWHM window for PPDS")
    ap.add_argument("--results_csv", required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N configs (each takes ~20 s)")
    ap.add_argument("--out", default=None, help="Write per-config values here")
    args = ap.parse_args()

    df = pd.read_csv(args.results_csv)
    need = ["work_dir", "n_det_ring1", OUTCOME, "aperture_diam_mm"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        print(f"ERROR: results CSV lacks {missing}")
        sys.exit(1)
    df = df.dropna(subset=need)

    # Only configs whose PPDF files still exist can be recomputed. The vscratch
    # purge stripped the raw files from the older lhs4d_* designs, leaving just
    # their cnr/ subdirectory, and head(limit) picks exactly those because they
    # are first in the CSV. Sampling the wrong subset produced a full table of
    # NaN that looked like a result.
    n_before = len(df)
    df = configs_with_raw_files(df)
    print(f"{len(df)} of {n_before} configs still have their PPDF files "
          f"({n_before - len(df)} stripped by the scratch purge)")
    if df.empty:
        print("\nERROR: no config has the raw files this test needs. Nothing to do.")
        sys.exit(1)

    if args.limit and len(df) > args.limit:
        df = stratified_sample(df, args.limit)
        in_band = int(((df.aperture_diam_mm >= SMALL_APERTURE[0])
                       & (df.aperture_diam_mm <= SMALL_APERTURE[1])).sum())
        print(f"sampled {len(df)}: {in_band} in the 0.20-0.45 mm aperture band, "
              f"{len(df) - in_band} outside it")

    print(f"recomputing PPDS at {len(THRESHOLDS)} thresholds for {len(df)} configs")
    print("(about 20 s per config per threshold, so this is not quick)\n", flush=True)

    rows = []
    for i, (_, r) in enumerate(df.iterrows(), 1):
        wd = str(r["work_dir"])
        if not os.path.isdir(wd):
            continue
        rec = {"config": r.get("config"), OUTCOME: r[OUTCOME],
               "aperture_diam_mm": r["aperture_diam_mm"]}
        base = cm.compute_ppds_per_ring(wd, int(r["n_det_ring1"]))
        rec["ppds_ring1"] = float(base[0]) if base is not None and len(base) else np.nan
        for t in THRESHOLDS:
            comps = cm.compute_ppds_per_ring(wd, int(r["n_det_ring1"]), fwhm_max=t)
            rec[f"ppds_ring1_w{t}"] = (float(comps[0]) if comps is not None
                                       and len(comps) else np.nan)
        rows.append(rec)
        if i % 10 == 0:
            print(f"  {i}/{len(df)}", flush=True)

    if not rows:
        print("ERROR: no readable work_dirs")
        sys.exit(1)
    out = pd.DataFrame(rows)
    if args.out:
        out.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")

    if not enough_values(out["ppds_ring1"]):
        print(f"\nERROR: only {int(out['ppds_ring1'].notna().sum())} configs produced "
              f"a PPDS value.\nWithout usable inputs the correlations below would all "
              f"be NaN, which reads\nlike a result and is not one. Stopping.")
        sys.exit(1)

    small = out[(out.aperture_diam_mm >= SMALL_APERTURE[0])
                & (out.aperture_diam_mm <= SMALL_APERTURE[1])]
    print("\n" + "=" * 68)
    print("PPDS RING 1 vs CNR, BY FWHM WINDOW")
    print("=" * 68)
    print(f"\n{'window (mm)':<14}{'rho, all':>12}{'rho, small ap':>16}{'n small':>10}")
    print("-" * 52)

    def rho(frame, col):
        sub = frame[[col, OUTCOME]].dropna()
        return sub[col].corr(sub[OUTCOME], method="spearman") if len(sub) >= 10 else np.nan

    base_all, base_small = rho(out, "ppds_ring1"), rho(small, "ppds_ring1")
    print(f"{'unwindowed':<14}{base_all:>12.2f}{base_small:>16.2f}{len(small):>10}")
    best = None
    for t in THRESHOLDS:
        c = f"ppds_ring1_w{t}"
        a, s = rho(out, c), rho(small, c)
        print(f"{t:<14.2f}{a:>12.2f}{s:>16.2f}")
        if np.isfinite(s) and (best is None or s > best[1]):
            best = (t, s, a)

    print("""
How to read this. The 'small ap' column is the one that decides it: every top
design sits in the 0.20 to 0.45 mm aperture band, and unwindowed ring-1 PPDS is
uninformative there. A window is worth adopting only if it beats the unwindowed
metric in that column, not merely overall.""")
    if best and np.isfinite(base_small):
        t, s, a = best
        if s > base_small + 0.15:
            print(f"\n-> The {t:.2f} mm window lifts the small-aperture correlation "
                  f"from {base_small:+.2f} to {s:+.2f}. Worth adopting, and worth "
                  f"testing\n   the same way the ASCI window was: confirm it peaks "
                  f"rather than\n   rising monotonically, which would just mean the "
                  f"threshold is arbitrary.")
        else:
            cv = 100 * small["ppds_ring1"].std() / small["ppds_ring1"].mean()
            print(f"\n-> No window helps materially. Best is {t:.2f} mm at "
                  f"{s:+.2f} against {base_small:+.2f} unwindowed.\n   Windowing "
                  f"does not change ring-1 PPDS's relationship to CNR, so there is "
                  f"no\n   reason to adopt the windowed form.")
            print(f"\n   Note what this does NOT say. Ring-1 PPDS still varies "
                  f"widely in this band\n   ({cv:.0f}% of its mean), so it is not a "
                  f"dead metric -- its variation is simply\n   unrelated to CNR. "
                  f"That makes it an INDEPENDENT objective, which in a\n   "
                  f"multi-objective search is a reason to keep it if we want it for "
                  f"its own\n   sake. Whether detector utilisation is a design goal "
                  f"in itself is a\n   judgement call, not something this "
                  f"correlation can settle.")


if __name__ == "__main__":
    main()
