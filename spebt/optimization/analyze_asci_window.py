#!/usr/bin/env python3
"""
Sweep an FWHM window over the ASCI metric and pick the threshold from data.

ASCI is currently saturated: ~64% of configurations sit at exactly 100%, and the
best value was already reached during the initial random sampling, so MOBO has
never been able to improve it. RY's Jul 2026 note is that with ring geometry the
angular sampling is abundant enough that saturation is expected without an FWHM
window, and that adding one is the direction to go.

Rather than guess a threshold, this computes windowed ASCI at several candidate
thresholds for every evaluated config, then reports for each threshold:
  - how saturated the metric still is (fraction of configs pinned at 100%)
  - its spread across configs (a flat metric cannot steer an optimizer)
  - its Spearman correlation with CNR
so the threshold can be chosen on evidence.

Windowed ASCI keeps the existing definition -- fraction of (FOV pixel, angular
bin) cells covered by at least one beam, combined across layouts -- and adds one
filter: only beams with FWHM <= threshold count.

Everything is recomputed from stored beams_properties_*.hdf5 and
beams_masks_*.hdf5, so no re-simulation is needed.

Usage:
  python analyze_asci_window.py --results_csv results/results_summary_mobo.csv
  python analyze_asci_window.py --results_csv ... --thresholds 0.6,0.8,1.0,1.5
  python analyze_asci_window.py --results_csv ... --analyze_only
"""
import argparse
import glob
import os
import shutil
import sys
import time

import h5py
import numpy as np
import pandas as pd

# Must match sai_analyze_asci.py
SAI_N_PIXELS = (200, 200)
N_BINS = 360
SENSITIVITY_FLOOR_FRAC = 0.01   # beams below 1% of the layout's peak are ignored

DEFAULT_THRESHOLDS = (0.6, 0.8, 1.0, 1.5, 2.0)


def _layout_id(path):
    for part in os.path.basename(path).replace(".hdf5", "").split("_"):
        if part.isdigit() and len(part) == 3:
            return part
    return None


def windowed_asci(work_dir: str, thresholds) -> dict:
    """ASCI (%) at each FWHM threshold for one config.

    Returns {threshold: pct}. Values are NaN if the config's files are missing.
    Combining across layouts before taking the percentage matches
    compute_metrics.compute_fwhm_and_asci, which sums the per-layout histograms.
    """
    prop_files = sorted(glob.glob(os.path.join(work_dir, "beams_properties_configuration_*.hdf5")))
    mask_files = sorted(glob.glob(os.path.join(work_dir, "beams_masks_configuration_*.hdf5")))
    if not prop_files or not mask_files:
        return {t: float("nan") for t in thresholds}

    mask_by_lid = {}
    for f in mask_files:
        lid = _layout_id(f)
        if lid is not None:
            mask_by_lid[lid] = f

    n_fov = SAI_N_PIXELS[0] * SAI_N_PIXELS[1]
    # One coverage map per threshold, accumulated across every layout.
    filled = {t: np.zeros((n_fov, N_BINS), dtype=bool) for t in thresholds}

    # Same binning as sai_analyze_asci: torch.bucketize(..., right=False) - 1 is
    # equivalent to np.searchsorted(..., side="left") - 1.
    boundaries = np.arange(N_BINS + 1) / 180.0 * np.pi

    any_layout = False
    for prop_file in prop_files:
        lid = _layout_id(prop_file)
        mask_file = mask_by_lid.get(lid)
        if mask_file is None:
            continue
        try:
            with h5py.File(prop_file, "r") as f:
                bp = f["beam_properties"][:]
            with h5py.File(mask_file, "r") as f:
                masks = f["beam_mask"][:]
        except Exception as e:
            print(f"  [warn] ASCI: failed reading layout {lid}: {e}")
            continue
        if bp.shape[0] == 0:
            continue

        angles = bp[:, 3].astype(np.float64)
        fwhms = bp[:, 4].astype(np.float64)
        sens = bp[:, 7].astype(np.float64)
        det_ids = bp[:, 1].astype(np.int64)
        beam_ids = bp[:, 2].astype(np.int64)

        n_det = masks.shape[0]
        if det_ids.size and det_ids.min() >= 1 and det_ids.max() <= n_det:
            det_ids = det_ids - 1

        keep = ~np.isnan(angles)
        if keep.sum() == 0:
            continue
        # Sensitivity floor, computed on the angle-valid subset exactly as the
        # original script does.
        smax = sens[keep].max() if np.isfinite(sens[keep]).any() else 0.0
        if smax > 0:
            keep &= sens > smax * SENSITIVITY_FLOOR_FRAC
        keep &= np.isfinite(fwhms) & (fwhms > 0)
        if keep.sum() == 0:
            continue

        bins = np.searchsorted(boundaries, angles, side="left") - 1
        keep &= (bins >= 0) & (bins < N_BINS)
        if keep.sum() == 0:
            continue

        any_layout = True

        # Per detector, map beam_id -> (angular bin, FWHM), then scatter all of
        # that detector's lit pixels at once. Doing this per beam instead would
        # mean a full 40k-element comparison per beam.
        order = np.argsort(det_ids[keep], kind="stable")
        kd = det_ids[keep][order]
        kb = beam_ids[keep][order]
        kbin = bins[keep][order]
        kf = fwhms[keep][order]

        starts = np.searchsorted(kd, np.arange(n_det), side="left")
        ends = np.searchsorted(kd, np.arange(n_det), side="right")

        for det_i in range(n_det):
            lo, hi = starts[det_i], ends[det_i]
            if lo == hi:
                continue
            row = masks[det_i]
            lit = np.nonzero(row)[0]
            if lit.size == 0:
                continue
            # beam_mask is stored as float in the real HDF5 files, so cast before
            # using the ids as lookup indices.
            row_beams = row[lit].astype(np.int64)

            # Lookup arrays indexed by beam id for this detector
            max_bid = int(max(kb[lo:hi].max(), row_beams.max()))
            bin_lut = np.full(max_bid + 1, -1, dtype=np.int64)
            fwhm_lut = np.full(max_bid + 1, np.inf, dtype=np.float64)
            bids = kb[lo:hi]
            in_range = bids <= max_bid
            bin_lut[bids[in_range]] = kbin[lo:hi][in_range]
            fwhm_lut[bids[in_range]] = kf[lo:hi][in_range]

            valid_beam = (row_beams >= 0) & (row_beams <= max_bid)
            if not valid_beam.any():
                continue
            pix = lit[valid_beam]
            pb = row_beams[valid_beam]
            pbin = bin_lut[pb]
            pfwhm = fwhm_lut[pb]
            ok = pbin >= 0
            if not ok.any():
                continue
            pix, pbin, pfwhm = pix[ok], pbin[ok], pfwhm[ok]

            for t in thresholds:
                sel = pfwhm <= t
                if sel.any():
                    filled[t][pix[sel], pbin[sel]] = True

    if not any_layout:
        return {t: float("nan") for t in thresholds}

    total = n_fov * N_BINS
    return {t: float(filled[t].sum()) / total * 100.0 for t in thresholds}


def col_for(t) -> str:
    return f"asci_pct_fwhm{t:g}".replace(".", "p")


def report(df: pd.DataFrame, thresholds, cnr_col: str = "cnr_mean") -> None:
    print()
    print("=" * 78)
    print("WHICH FWHM WINDOW MAKES ASCI USEFUL AGAIN?")
    print("=" * 78)
    print(f"\n{'metric':<26} {'n':>4} {'mean':>8} {'std':>8} {'%at100':>8} {'rho vs ' + cnr_col:>16}")
    print("-" * 70)

    rows = [("asci_pct", "ASCI (current, no window)")]
    rows += [(col_for(t), f"ASCI (FWHM <= {t:g} mm)") for t in thresholds]

    for col, label in rows:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        at100 = 100.0 * (vals >= 99.999).sum() / len(vals)
        sub = df[[col, cnr_col]].dropna() if cnr_col in df.columns else pd.DataFrame()
        rho = (sub[col].corr(sub[cnr_col], method="spearman")
               if len(sub) >= 3 else float("nan"))
        print(f"{label:<26} {len(vals):>4} {vals.mean():>8.2f} {vals.std():>8.2f} "
              f"{at100:>7.1f}% {rho:>+11.3f}")

    print("""
How to read this:
  - %at100 is the saturation problem. The current metric pins ~64% of configs at
    100%, so it cannot discriminate between them. A useful window brings this
    near zero.
  - std is whether the metric varies enough to steer an optimizer at all.
  - rho vs CNR should ideally be positive: ASCI is maximised, so a window that
    anti-correlates with CNR would repeat the mistake sensitivity is making.
  - Pick the largest threshold that de-saturates, so we keep as much angular
    coverage information as possible.
""")


def main():
    ap = argparse.ArgumentParser(description="Sweep an FWHM window over ASCI")
    ap.add_argument("--results_csv", required=True)
    ap.add_argument("--thresholds", type=str, default=None,
                    help="Comma-separated FWHM thresholds in mm "
                         f"(default: {','.join(str(t) for t in DEFAULT_THRESHOLDS)})")
    ap.add_argument("--cnr_col", default="cnr_mean",
                    help="CNR column to correlate against")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after this many computed rows (for timing)")
    ap.add_argument("--analyze_only", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.results_csv):
        print(f"ERROR: results CSV not found: {args.results_csv}")
        sys.exit(1)

    thresholds = (tuple(float(t) for t in args.thresholds.split(","))
                  if args.thresholds else DEFAULT_THRESHOLDS)

    df = pd.read_csv(args.results_csv)
    print(f"Loaded {len(df)} rows from {args.results_csv}")

    if args.analyze_only:
        report(df, thresholds, args.cnr_col)
        return

    print(f"FWHM thresholds (mm): {thresholds}")
    if "work_dir" not in df.columns:
        print("ERROR: results CSV has no 'work_dir' column.")
        sys.exit(1)

    cols = [col_for(t) for t in thresholds]
    for c in cols:
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

        if not args.force and all(pd.notna(row.get(c)) for c in cols):
            n_skip += 1
            continue
        if not isinstance(work_dir, str) or not os.path.isdir(work_dir):
            n_skip += 1
            continue

        try:
            res = windowed_asci(work_dir, thresholds)
        except Exception as e:
            print(f"  [fail] {config}: {e}")
            n_fail += 1
            continue

        for t in thresholds:
            df.at[i, col_for(t)] = res[t]
        if np.isnan(res[thresholds[0]]):
            print(f"  [nan]  {config}: missing beam files")
            n_fail += 1
        else:
            summary = "  ".join(f"{t:g}mm={res[t]:.1f}%" for t in thresholds)
            print(f"  [ok]   {config[:48]:<48} {summary}")
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

    report(df, thresholds, args.cnr_col)


if __name__ == "__main__":
    main()
