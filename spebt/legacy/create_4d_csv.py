#!/usr/bin/env python3
"""
One-time script to extract 4D LHS results from the mixed results_summary.csv
into a clean results_summary_4d.csv with proper 12-column header.

The issue: results_summary.csv has a 10-column header (from 2D BO) but
4D rows appended 12 values, causing column misalignment.

Usage:
  python create_4d_csv.py
  python create_4d_csv.py --input results/results_summary.csv --output results/results_summary_4d.csv
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Extract 4D results into clean CSV")
    parser.add_argument("--input", type=str, default="results/results_summary.csv")
    parser.add_argument("--output", type=str, default="results/results_summary_4d.csv")
    args = parser.parse_args()

    header = ("fwhm_mean,sensitivity_total,sensitivity_mean,asci_pct,"
              "n_ppdf_files,JI,config,work_dir,aperture_diam_mm,n_apertures,"
              "scint_radial_thickness_mm,ring_thickness_mm")

    lines = open(args.input).readlines()
    out = [header + "\n"]
    for line in lines[1:]:  # skip old header
        # 4D rows have _sr and _rt in config name
        if "_sr" in line and "_rt" in line:
            out.append(line)

    with open(args.output, "w") as f:
        f.writelines(out)

    n_rows = len(out) - 1
    print(f"Extracted {n_rows} 4D rows from {args.input}")
    print(f"Saved to {args.output}")

    # Quick summary
    feasible = sum(1 for line in out[1:] if not line.startswith(","))
    infeasible = n_rows - feasible
    print(f"Feasible: {feasible}, Infeasible: {infeasible}")


if __name__ == "__main__":
    main()
