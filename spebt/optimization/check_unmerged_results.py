#!/usr/bin/env python3
"""Find finished batch evaluations that were never merged into the archive.

Seed sweeps (`make_ring2_seeds.py`, `make_d2d3_seed_list.py`,
`make_lhs6d_seeds.py`) write one CSV per array task into a
`results/<name>_seed_out/` directory. Merging those rows into
`results_summary_mobo.csv` is a separate manual step, and nothing ever checked
that it happened.

It did not happen once. The `ring2_000..007` sweep -- the whole point of which
was to test whether n_det_ring2 had headroom above 960 -- finished on 12 Aug
2026 with all 8 configs complete, and sat unread in `results/ring2_seed_out/`
for three weeks. The optimizer never trained on it and nobody saw the answer.

A config counts as merged if it appears in ANY known archive, not just the main
one: the head-to-head arms keep their own `results_h2h_*/results_summary_mobo.csv`,
and the 21 shared LHS seeds legitimately live there. A checker that reports 21
false positives is a checker nobody reads.

Deliberate exclusions go in `unmerged_ignore.txt`, one config name per line with
a `#` comment saying why, so "we meant to leave that out" is recorded rather
than remembered.

Run this after any seed sweep, and before quoting the archive as complete:

    python check_unmerged_results.py
    python check_unmerged_results.py --results_dir results --out_csv results/results_summary_mobo.csv

Exit status is 1 when unmerged rows exist, so it can gate a submit script.
"""
import argparse
import glob
import os
import sys

import pandas as pd


def batch_output_dirs(results_dir: str) -> list:
    """Directories holding per-task batch CSVs, oldest name order."""
    return sorted(
        d for d in glob.glob(os.path.join(results_dir, "*_out"))
        if os.path.isdir(d) and glob.glob(os.path.join(d, "task_*.csv"))
    )


def configs_in_batch_dir(batch_dir: str) -> set:
    """Config names recorded in one batch output directory.

    A task CSV that is empty or truncated (the array job died mid-write) is
    skipped with a warning rather than crashing the check: a partial batch is
    exactly when you most want the rest of the report.
    """
    names = set()
    for f in sorted(glob.glob(os.path.join(batch_dir, "task_*.csv"))):
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"  [warn] unreadable, skipping: {f} ({e})")
            continue
        if "config" not in df.columns:
            print(f"  [warn] no 'config' column, skipping: {f}")
            continue
        names.update(df["config"].dropna().astype(str))
    return names


def known_archives(out_csv: str, extra=()) -> list:
    """Every results CSV a config could legitimately have landed in.

    The head-to-head campaigns write their own archive per arm and replicate,
    beside the main one, and the shared LHS seeds are merged into those.
    """
    paths = [out_csv]
    parent = os.path.dirname(os.path.dirname(os.path.abspath(out_csv)))
    paths += sorted(glob.glob(os.path.join(parent, "results_h2h_*",
                                           os.path.basename(out_csv))))
    paths += list(extra)
    seen, out = set(), []
    for p in paths:
        rp = os.path.abspath(p)
        if rp not in seen and os.path.exists(rp):
            seen.add(rp)
            out.append(p)
    return out


def configs_in_archives(paths) -> set:
    """Union of config names across every archive."""
    names = set()
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"  [warn] unreadable archive, skipping: {p} ({e})")
            continue
        if "config" in df.columns:
            names.update(df["config"].dropna().astype(str))
    return names


def load_ignore(path: str) -> set:
    """Config names deliberately left out, from a `#`-commented list."""
    if not path or not os.path.exists(path):
        return set()
    out = set()
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                out.add(line)
    return out


def find_unmerged(results_dir: str, out_csv: str, extra_archives=(),
                  ignore_path: str = None) -> dict:
    """Map batch dir -> sorted config names in no archive and not ignored."""
    merged = configs_in_archives(known_archives(out_csv, extra_archives))
    ignored = load_ignore(ignore_path)

    unmerged = {}
    for d in batch_output_dirs(results_dir):
        missing = configs_in_batch_dir(d) - merged - ignored
        if missing:
            unmerged[d] = sorted(missing)
    return unmerged


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_csv", default="results/results_summary_mobo.csv")
    ap.add_argument("--archive", action="append", default=[],
                    help="Additional results CSV to count as merged. Repeatable.")
    ap.add_argument("--ignore_file", default="unmerged_ignore.txt",
                    help="Config names deliberately left unmerged, with reasons.")
    args = ap.parse_args()

    archives = known_archives(args.out_csv, args.archive)
    print(f"Checking against {len(archives)} archive(s).")
    unmerged = find_unmerged(args.results_dir, args.out_csv, args.archive,
                             args.ignore_file)
    if not unmerged:
        print(f"All batch outputs under {args.results_dir}/ are merged into "
              f"{args.out_csv}.")
        return 0

    total = sum(len(v) for v in unmerged.values())
    print(f"{total} evaluated config(s) are NOT in {args.out_csv}:\n")
    for d, names in unmerged.items():
        print(f"  {d}  ({len(names)})")
        for n in names:
            print(f"      {n}")
    print("\nThese cost compute and produced results nobody has read. Merge them "
          "or record why they are being left out.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
