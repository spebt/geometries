#!/usr/bin/env python3
"""
Compare three definitions of the FWHM objective.

fwhm_mean currently averages every beam with a non-NaN width, while ASCI counts
only beams above 1% of the layout's peak sensitivity. On mobo_0069 that gap is
large and systematic -- 0.586 unfiltered against 0.459 filtered, with a quarter
of beams below the threshold -- because weak beams are also wide. So the two
objectives are describing different populations of beams.

Three candidates:

  unfiltered   every beam equally, as now. Treats a beam carrying 0.1% of the
               signal as mattering as much as one carrying all of it.
  filtered     beams above 1% of peak sensitivity, matching ASCI's selection.
  weighted     every beam, each width weighted by that beam's sensitivity. The
               most physically direct: a beam's width affects the image in
               proportion to the signal it contributes.

What decides it is not which looks nicer but (a) whether the choice reorders
designs at all, and (b) which best predicts CNR, since FWHM is a proxy for
resolution and CNR is the outcome we actually care about.

Reads stored beam-property files only; nothing is re-simulated.

Usage:
  python analyze_fwhm_definition.py --results_csv results/results_summary_mobo.csv
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np
import pandas as pd

SENS_FLOOR_FRAC = 0.01
FWHM_COL, SENS_COL = 4, 7


def fwhm_variants(work_dir: str):
    """(unfiltered, filtered, sensitivity-weighted) mean FWHM, pooled over layouts."""
    files = sorted(glob.glob(os.path.join(work_dir, "beams_properties_configuration_*.hdf5")))
    if not files:
        return None

    all_fw, all_sens = [], []
    for f in files:
        try:
            with h5py.File(f, "r") as h:
                bp = h["beam_properties"][:]
        except Exception:
            return None
        if bp.shape[0] == 0:
            continue
        fw = bp[:, FWHM_COL].astype(np.float64)
        sens = bp[:, SENS_COL].astype(np.float64)
        ok = np.isfinite(fw) & (fw > 0) & np.isfinite(sens)
        all_fw.append(fw[ok])
        all_sens.append(sens[ok])

    if not all_fw:
        return None
    fw = np.concatenate(all_fw)
    sens = np.concatenate(all_sens)
    if fw.size == 0:
        return None

    unfiltered = float(fw.mean())
    smax = sens.max()
    strong = sens > smax * SENS_FLOOR_FRAC if smax > 0 else np.ones_like(sens, bool)
    filtered = float(fw[strong].mean()) if strong.any() else float("nan")
    wsum = sens.sum()
    weighted = float(np.dot(fw, sens) / wsum) if wsum > 0 else float("nan")
    return unfiltered, filtered, weighted, int(fw.size), int(strong.sum())


def main():
    ap = argparse.ArgumentParser(description="Compare FWHM objective definitions")
    ap.add_argument("--results_csv", required=True)
    ap.add_argument("--cnr_col", default="cnr_sector_mean")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--write", action="store_true",
                    help="Write fwhm_weighted_mean into the CSV. Needed once, so "
                         "existing rows carry the column the optimizer now reads; "
                         "new rows get it from compute_metrics directly.")
    args = ap.parse_args()

    if not os.path.exists(args.results_csv):
        print(f"ERROR: results CSV not found: {args.results_csv}")
        sys.exit(1)

    df = pd.read_csv(args.results_csv)
    print(f"Loaded {len(df)} rows")

    rows = []
    for _, r in df.iterrows():
        if args.limit and len(rows) >= args.limit:
            break
        wd = r.get("work_dir")
        if not isinstance(wd, str) or not os.path.isdir(wd):
            continue
        v = fwhm_variants(wd)
        if v is None:
            continue
        rows.append({
            "config": r.get("config"),
            "unfiltered": v[0], "filtered": v[1], "weighted": v[2],
            "n_beams": v[3], "n_strong": v[4],
            "stored": r.get("fwhm_mean"),
            "cnr": r.get(args.cnr_col),
            "aperture_diam_mm": r.get("aperture_diam_mm"),
            "n_apertures": r.get("n_apertures"),
        })

    if len(rows) < 5:
        print(f"Only {len(rows)} configs readable; not enough to compare.")
        sys.exit(1)
    d = pd.DataFrame(rows)
    print(f"Computed all three definitions for {len(d)} configs\n")

    if args.write:
        import shutil, time as _time
        stamp = _time.strftime("%Y%m%d_%H%M%S")
        backup = args.results_csv.replace(".csv", f".bak.{stamp}.csv")
        shutil.copy(args.results_csv, backup)
        if "fwhm_weighted_mean" not in df.columns:
            df["fwhm_weighted_mean"] = float("nan")
        by_config = dict(zip(d["config"], d["weighted"]))
        n_set = 0
        for i, r in df.iterrows():
            v = by_config.get(r.get("config"))
            if v is not None and np.isfinite(v):
                df.at[i, "fwhm_weighted_mean"] = float(v)
                n_set += 1
        df.to_csv(args.results_csv, index=False)
        print(f"Wrote fwhm_weighted_mean for {n_set} rows")
        print(f"Backup: {backup}\n")

    print("=" * 74)
    print("DOES THE FWHM DEFINITION REORDER THE DESIGNS?")
    print("=" * 74)
    print(f"\n{'definition':<14} {'mean':>8} {'std':>8} {'rho vs stored':>14} {'rho vs CNR':>12}")
    print("-" * 62)
    for col in ("unfiltered", "filtered", "weighted"):
        sub = d[[col, "cnr"]].dropna()
        rho_cnr = sub[col].corr(sub["cnr"], method="spearman") if len(sub) >= 3 else float("nan")
        s2 = d[[col, "stored"]].dropna()
        rho_stored = s2[col].corr(s2["stored"], method="spearman") if len(s2) >= 3 else float("nan")
        print(f"{col:<14} {d[col].mean():>8.4f} {d[col].std():>8.4f} "
              f"{rho_stored:>+14.4f} {rho_cnr:>+12.4f}")

    print("\nAgreement between definitions (Spearman):")
    for a, b in (("unfiltered", "filtered"), ("unfiltered", "weighted"), ("filtered", "weighted")):
        sub = d[[a, b]].dropna()
        print(f"  {a:<11} vs {b:<11} {sub[a].corr(sub[b], method='spearman'):+.4f}")

    print(f"\nWeak-beam fraction: mean {(1 - d.n_strong / d.n_beams).mean():.1%}, "
          f"range {(1 - d.n_strong / d.n_beams).min():.1%} - "
          f"{(1 - d.n_strong / d.n_beams).max():.1%}")

    # A uniform offset is harmless; one that varies with the design is a bias.
    d["gap"] = d["unfiltered"] - d["filtered"]
    print(f"unfiltered - filtered: mean {d.gap.mean():.4f}, std {d.gap.std():.4f}")
    for col in ("aperture_diam_mm", "n_apertures"):
        sub = d[["gap", col]].dropna()
        if len(sub) >= 3:
            print(f"  gap vs {col:<18} rho = {sub['gap'].corr(sub[col], method='spearman'):+.3f}")

    print("""
How to read this:
  - If all three agree at rho ~1.0, the choice does not reorder designs and the
    current definition is fine to keep; only its absolute scale is arbitrary.
  - "rho vs CNR" is the tiebreaker. FWHM is a proxy for resolution and CNR is
    the outcome, so the definition tracking CNR best is the more useful
    objective. Note FWHM is MINIMISED, so a more negative rho is better.
  - The gap correlations matter most. If the unfiltered-minus-filtered gap tracks
    aperture size or aperture count, then the weak beams are biasing the
    objective in a design-dependent way, which is worse than a constant offset
    and argues for changing the definition.
""")


if __name__ == "__main__":
    main()
