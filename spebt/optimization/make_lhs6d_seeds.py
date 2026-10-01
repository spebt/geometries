#!/usr/bin/env python3
"""
Fresh 6D Latin hypercube seed designs for a head-to-head campaign comparison.

Why fresh seeds rather than reusing the archive: the replay
(replay_objective_subsets.py) can only ever be suggestive, because its candidate
pool was chosen by the five-objective campaign. Seeding a new campaign from that
same archive carries the same contamination. A live comparison needs a start
that neither objective set had a hand in choosing, so both arms begin from
identical, neutral ground and the only difference is what they optimize.

Two properties matter and are checked, not assumed:

  SPAN. The training data must vary along every axis. The 6D expansion stalled
  once already because every row sat at the legacy 390/520 ring layout: d3 had a
  standard deviation of 0.015 mm across a 255 mm range. A GP fits a long
  lengthscale to a constant dimension, concludes nothing depends on it, and
  never proposes a move. The seeds are rejected and resampled if any dimension
  fails a minimum spread.

  FEASIBILITY. Every design is checked against the same rules the optimizer
  uses, so none can fail geometry generation and waste a pipeline slot.

Usage:
  python make_lhs6d_seeds.py --n_seeds 21 --out lhs6d_seeds.csv
  python make_lhs6d_seeds.py --n_seeds 21 --seed 7 --out lhs6d_seeds.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

import mobo_agent as ma

# A dimension whose sampled spread falls below this fraction of its allowed
# range is treated as degenerate. 0.5 is loose enough that a valid LHS passes
# easily and tight enough to catch a feasibility filter collapsing an axis --
# which is the real risk, since the aperture and packing constraints correlate
# several dimensions.
MIN_SPREAD_FRAC = 0.5


def sample_lhs(n, rng, dim):
    """Latin hypercube on the unit cube: one sample per stratum per dimension."""
    cuts = np.linspace(0, 1, n + 1)
    out = np.empty((n, dim))
    for d in range(dim):
        pts = rng.uniform(cuts[:-1], cuts[1:])
        out[:, d] = rng.permutation(pts)
    return out


def to_physical(u):
    """Unit cube to design space, with the integer parameters rounded properly."""
    lo = np.asarray(ma.BOUNDS_MIN, dtype=float)
    hi = np.asarray(ma.BOUNDS_MAX, dtype=float)
    x = lo + u * (hi - lo)
    rows = []
    for r in x:
        diam, n_ap, nd1, nd2, d2, d3 = r
        nd1, nd2 = int(round(nd1)), int(round(nd2))
        # Two crystals per cell, so detector counts must be even.
        nd1 += nd1 % 2
        nd2 += nd2 % 2
        rows.append((float(diam), int(round(n_ap)), nd1, nd2, float(d2), float(d3)))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Generate fresh 6D LHS seed designs")
    ap.add_argument("--n_seeds", type=int, default=21,
                    help="Matches the 21-design start the existing campaign used")
    ap.add_argument("--out", default="lhs6d_seeds.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefix", default="lhs6d_",
                    help="Config-name prefix. MUST differ between replicates: "
                         "the pipeline derives its work directory from the "
                         "config name, so a shared prefix makes two replicates "
                         "evaluate different designs into the same directories.")
    ap.add_argument("--results_dir", default="results",
                    help="Where pipeline work directories live; checked "
                         "for config-name collisions.")
    ap.add_argument("--max_tries", type=int, default=20000)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    dim = len(ma.PARAM_NAMES)

    # Oversample and filter: the aperture, ordering and packing constraints
    # reject a large fraction of the unit cube, so an exact-size LHS would come
    # back short. Sampling in blocks until enough survive keeps the stratification
    # meaningful while still landing on n_seeds feasible designs.
    kept, tries = [], 0
    while len(kept) < args.n_seeds and tries < args.max_tries:
        block = to_physical(sample_lhs(max(args.n_seeds * 4, 64), rng, dim))
        tries += len(block)
        for d in block:
            if len(kept) >= args.n_seeds:
                break
            if ma.is_feasible_full(*d):
                kept.append(d)

    if len(kept) < args.n_seeds:
        print(f"ERROR: only {len(kept)} feasible designs after {tries} samples.\n"
              f"The feasible region may be smaller than the bounds suggest; "
              f"check BOUNDS_MIN/MAX against the constraint rules.")
        sys.exit(1)

    df = pd.DataFrame(kept, columns=ma.PARAM_NAMES)

    # Reject a seed set that is degenerate along any axis. This is the failure
    # that stalled the 6D search before, and it is silent: the campaign runs
    # normally and simply never explores.
    lo = np.asarray(ma.BOUNDS_MIN, dtype=float)
    hi = np.asarray(ma.BOUNDS_MAX, dtype=float)
    spread = (df.max().values - df.min().values) / (hi - lo)
    print(f"{len(df)} feasible seed designs from {tries} samples\n")
    print(f"{'parameter':<20} {'min':>10} {'max':>10} {'span/range':>12}")
    print("-" * 56)
    bad = []
    for name, s, mn, mx in zip(ma.PARAM_NAMES, spread, df.min().values, df.max().values):
        flag = "" if s >= MIN_SPREAD_FRAC else "  <-- DEGENERATE"
        if s < MIN_SPREAD_FRAC:
            bad.append(name)
        print(f"{name:<20} {mn:>10.2f} {mx:>10.2f} {s:>11.1%}{flag}")

    if bad:
        print(f"\nERROR: {bad} span less than {MIN_SPREAD_FRAC:.0%} of their range.")
        print("A GP cannot learn from a near-constant dimension: it fits a long")
        print("lengthscale, concludes the objectives do not depend on that axis,")
        print("and the acquisition never proposes a move along it. Rerun with a")
        print("different --seed, or widen the bounds if the constraints are")
        print("genuinely collapsing this axis.")
        sys.exit(1)

    # Config names must be unique ACROSS replicates. run_sai_pipeline.sh derives
    # its work directory from the config name, so two replicates using
    # lhs6d_000..020 would evaluate different designs into the same 21
    # directories -- concurrently, and without erroring. The numbers that came
    # out would look entirely plausible.
    df.insert(0, "config", [f"{args.prefix}{i:03d}" for i in range(len(df))])

    # Refuse to emit names whose work directories already exist. This is the
    # guard that would have caught replicates 1 and 2 being generated with the
    # default prefix: they would have evaluated into replicate 0's directories,
    # concurrently, silently.
    clashes = [c for c in df["config"]
               if os.path.isdir(os.path.join(args.results_dir, str(c)))]
    if clashes:
        print(f"\nERROR: {len(clashes)} config names already have work directories "
              f"under {args.results_dir}, e.g. {clashes[:3]}")
        print("The pipeline derives its work directory from the config name, so "
              "these\nwould overwrite existing evaluations. Pass a distinct "
              "--prefix, e.g.\n  --prefix lhs6d_r3_")
        sys.exit(1)

    df.to_csv(args.out, index=False)
    print(f"\nwrote {len(df)} seeds to {args.out}")
    print("\nEvery design satisfies the same feasibility rules the optimizer uses,")
    print("so none should fail geometry generation.")


if __name__ == "__main__":
    main()
