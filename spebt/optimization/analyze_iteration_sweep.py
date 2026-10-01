#!/usr/bin/env python3
"""
Does the ML-EM iteration count change which design wins?

RY (Jul 2026): contrast and noise both depend on the iteration count, and a
given count favours a particular rod size. Every design in the campaign was
reconstructed at a fixed 150 iterations.

Absolute CNR will certainly move with the iteration count. The question that
matters for optimization is whether the RANKING of designs is stable. If it is,
the fixed choice is harmless for design selection even though the absolute
values are arbitrary. If designs swap places, then 150 iterations is an
unexamined assumption sitting under every conclusion we have drawn.

Reads the per-section CNRs written by submit_iteration_sweep.sh into
<work_dir>/iter_<N>/cnr_results.npz. Reports, per iteration count, each design's
section-mean CNR and its rank, plus which rod size peaks -- the size-dependence
RY specifically flagged.

Usage:
  python analyze_iteration_sweep.py
  python analyze_iteration_sweep.py --results_dir results
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

CONFIGS = [
    "mobo_0069_ap0.3138_nap124_nd1_612_nd2_230",
    "mobo_0177_ap0.3512_nap97_nd1_604_nd2_584",
    "mobo_0133_ap0.4008_nap70_nd1_660_nd2_562",
]
ROD_RADII_MM = [0.100, 0.125, 0.150, 0.175, 0.200, 0.225]  # ascending, matches sector order


def short(config):
    m = re.match(r"(mobo_\d+)", config)
    return m.group(1) if m else config[:12]


def collect(results_dir, config):
    """{iteration_count: (sector_mean, sectors)} for one design."""
    out = {}
    for path in glob.glob(os.path.join(results_dir, config, "iter_*", "cnr_results.npz")):
        m = re.search(r"iter_(\d+)", path)
        if not m:
            continue
        d = np.load(path, allow_pickle=True)
        sectors = np.asarray(d["sector_cnrs"], dtype=float)
        out[int(m.group(1))] = (float(np.nanmean(sectors)), sectors)
    return out


def main():
    ap = argparse.ArgumentParser(description="Iteration-count stability of the design ranking")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--configs", default=None,
                    help="Comma-separated config names (default: the built-in three)")
    args = ap.parse_args()

    configs = args.configs.split(",") if args.configs else CONFIGS
    data = {c: collect(args.results_dir, c) for c in configs}
    data = {c: v for c, v in data.items() if v}
    if not data:
        print("No iteration-sweep results found. Has submit_iteration_sweep.sh run?")
        sys.exit(1)

    iters = sorted({n for v in data.values() for n in v})

    print("=" * 74)
    print("DOES THE ITERATION COUNT CHANGE WHICH DESIGN WINS?")
    print("=" * 74)
    print(f"\n{'iters':>6}  " + "  ".join(f"{short(c):>12}" for c in data))
    print("-" * (8 + 14 * len(data)))
    rank_rows = []
    for n in iters:
        vals, line = [], f"{n:>6}  "
        for c in data:
            v = data[c].get(n)
            vals.append(v[0] if v else np.nan)
            line += f"{v[0]:>12.3f}  " if v else f"{'--':>12}  "
        print(line)
        order = np.argsort(-np.asarray(vals))
        ranks = np.empty(len(vals), dtype=int)
        ranks[order] = np.arange(1, len(vals) + 1)
        rank_rows.append((n, ranks))

    print(f"\n{'iters':>6}  " + "  ".join(f"{short(c):>12}" for c in data) + "   (rank)")
    print("-" * (8 + 14 * len(data)))
    for n, ranks in rank_rows:
        print(f"{n:>6}  " + "  ".join(f"{r:>12d}" for r in ranks))

    all_same = all(np.array_equal(rank_rows[0][1], r) for _, r in rank_rows)
    print()
    if all_same:
        print("RANKING IS STABLE across every iteration count tested.")
        print("The fixed 150-iteration choice does not affect which design wins,")
        print("even though the absolute CNR values move.")
    else:
        print("RANKING CHANGES with the iteration count.")
        print("The fixed 150-iteration choice is then an assumption underneath")
        print("every design comparison we have made, and needs justifying.")

    print("\nWhich rod size peaks, per design and iteration count:")
    print("(RY's point: a given iteration count favours a particular size)")
    print(f"\n{'iters':>6}  " + "  ".join(f"{short(c):>14}" for c in data))
    print("-" * (8 + 16 * len(data)))
    for n in iters:
        line = f"{n:>6}  "
        for c in data:
            v = data[c].get(n)
            if v is None:
                line += f"{'--':>14}  "
            else:
                line += f"{ROD_RADII_MM[int(np.nanargmax(v[1]))]:>13.3f}mm  "
        print(line)

    print("""
How to read this:
  - The rank table is the answer. Identical rows mean the iteration count is
    irrelevant to design selection.
  - The peak-size table shows the effect RY described: as iterations increase,
    contrast recovery reaches progressively smaller features, so the peak should
    drift toward the smaller rods. That is expected and not itself a problem --
    it only matters if it reorders designs.
""")


if __name__ == "__main__":
    main()
