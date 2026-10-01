#!/usr/bin/env python3
"""
Find configs whose metrics were computed from an incomplete file set.

Step 2 of run_sai_pipeline.sh ran the two layouts' beam analysis in background
subshells without checking exit codes, and the metric aggregation globs whatever
files happen to exist. A failure in one layout therefore produced FWHM, ASCI,
MPXI and PPDS from one layout instead of two, written to the CSV looking valid.
ASCI is the most exposed: combining one histogram instead of two directly lowers
the filled-bin count.

Both the shell and compute_metrics now guard against this, but rows evaluated
before those guards need checking. This audits every work_dir against the
expected file counts and reports which rows cannot be trusted.

Usage:
  python audit_input_completeness.py --results_csv results/results_summary_mobo.csv
  python audit_input_completeness.py --results_csv ... --write_list bad_rows.txt
"""
import argparse
import glob
import os
import sys

import pandas as pd

N_LAYOUTS = 2
N_PPDF = 16
PER_LAYOUT = {
    "masks": "beams_masks_configuration_*.hdf5",
    "props": "beams_properties_configuration_*.hdf5",
    "asci": "asci_histogram_*.hdf5",
}
METRIC_COLS = ["fwhm_mean", "asci_pct", "asci_pct_fwhm0p45", "mpxi_mean", "ppds_ring1"]


def main():
    ap = argparse.ArgumentParser(description="Audit per-config input completeness")
    ap.add_argument("--results_csv", required=True)
    ap.add_argument("--write_list", default=None,
                    help="Write the affected work_dirs, one per line, for re-running")
    args = ap.parse_args()

    if not os.path.exists(args.results_csv):
        print(f"ERROR: results CSV not found: {args.results_csv}")
        sys.exit(1)

    df = pd.read_csv(args.results_csv)
    print(f"Auditing {len(df)} rows\n")

    bad, missing_dir, clean, no_metrics = [], 0, 0, 0
    for _, row in df.iterrows():
        wd = row.get("work_dir")
        config = row.get("config", "?")
        if not isinstance(wd, str) or not os.path.isdir(wd):
            missing_dir += 1
            continue

        # Only rows that actually produced metrics can be corrupted by this
        have_metrics = any(pd.notna(row.get(c)) for c in METRIC_COLS if c in df.columns)
        if not have_metrics:
            no_metrics += 1
            continue

        counts = {k: len(glob.glob(os.path.join(wd, p))) for k, p in PER_LAYOUT.items()}
        counts["ppdf"] = len(glob.glob(os.path.join(wd, "position_*_ppdfs_t8_*.hdf5")))
        expected = {"masks": N_LAYOUTS, "props": N_LAYOUTS, "asci": N_LAYOUTS, "ppdf": N_PPDF}

        problems = [f"{k}={counts[k]}/{expected[k]}" for k in expected if counts[k] != expected[k]]
        if problems:
            bad.append((config, wd, problems))
        else:
            clean += 1

    print(f"Clean (full file set)          : {clean}")
    print(f"INCOMPLETE -- metrics suspect  : {len(bad)}")
    print(f"No metrics (nothing to corrupt): {no_metrics}")
    print(f"work_dir gone                  : {missing_dir}")

    if bad:
        print("\nAffected configs:")
        for config, _, problems in bad:
            print(f"  {config[:52]:<52} {', '.join(problems)}")
        print("\nThese rows' FWHM / ASCI / MPXI / PPDS were aggregated from a")
        print("partial file set. Re-run the pipeline for them, or drop the rows.")
        if args.write_list:
            with open(args.write_list, "w") as f:
                f.write("\n".join(wd for _, wd, _ in bad) + "\n")
            print(f"\nWrote {len(bad)} work_dirs to {args.write_list}")
    else:
        print("\nNo affected rows: every config with metrics has its full file set,")
        print("so the missing exit-code check never actually corrupted a result.")


if __name__ == "__main__":
    main()
