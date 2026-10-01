#!/usr/bin/env python3
"""
Is this really a five-objective problem?

After the Jul 2026 objective revision, four of the five correlate strongly with
reconstructed CNR: FWHM -0.93, ASCI@0.45mm +0.80, PPDS ring 1 +0.60. MPXI was
assumed independent on the strength of a ~0.00 correlation against CNR, but that
was measured against pooled cnr_mean on ~100 configs and says nothing about how
MPXI relates to the OTHER four. That raises a question worth answering before
the next campaign: if four objectives largely restate the same thing, the
multi-objective formulation may be doing less work than it appears, and a
smaller set would search the same space with a sharper acquisition signal.

Symptoms already visible: 94 of 208 designs are Pareto-optimal. With five
objectives a point is dominated only if another beats it on all five, which
almost never happens, so the front stops discriminating and design selection
falls back to CNR with the rest as constraints.

This tests the question against the existing archive, with no new simulation:

  1. Correlation structure. Pairwise Spearman between every objective, and the
     effective dimensionality from a PCA of the standardised objective matrix.
  2. Pareto overlap. Which designs are non-dominated under the full set versus
     under smaller subsets. If the five-objective front is mostly the union of
     the two-objective front plus points nothing dominates by accident, the
     extra objectives are not defining trade-offs.
  3. Ranking agreement. Whether a subset would have selected the same top
     designs, which is the question that actually matters for the campaign.

Usage:
  python analyze_objective_redundancy.py --results_csv results/results_summary_mobo.csv
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

import mobo_agent as ma

# Subsets worth testing. CNR is the outcome we care about; MPXI is the objective
# least aligned with it, so it is the natural second axis to test.
# Short tags keep the section-3 table readable -- the full labels all start with
# "CNR" and truncate to nothing useful.
# The outcome every objective is ultimately judged against.
OUTCOME = "cnr_sector_mean"

SUBSETS = {
    "all five": ma.OBJ_COLUMNS,
    "CNR + MPXI": ["cnr_sector_mean", "mpxi_windowed_active_mean"],
    "CNR + MPXI + FWHM": ["cnr_sector_mean", "mpxi_windowed_active_mean",
                          "fwhm_weighted_mean"],
    "CNR only": ["cnr_sector_mean"],
    "no CNR (the 4 proxies)": [c for c in ma.OBJ_COLUMNS if c != "cnr_sector_mean"],
}
SUBSET_TAGS = {
    "all five": "all5",
    "CNR + MPXI": "C+M",
    "CNR + MPXI + FWHM": "C+M+F",
    "CNR only": "CNRonly",
    "no CNR (the 4 proxies)": "noCNR",
}


def pareto_mask(obj_max: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows, all objectives maximized."""
    n = len(obj_max)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated = np.all(obj_max >= obj_max[i], axis=1) & np.any(obj_max > obj_max[i], axis=1)
        if dominated.any():
            keep[i] = False
    return keep


def to_max(df: pd.DataFrame, cols) -> np.ndarray:
    """Objective matrix in maximization space."""
    dirs = [ma.OBJ_DIRECTIONS[ma.OBJ_COLUMNS.index(c)] for c in cols]
    return df[cols].values * np.asarray(dirs, dtype=float)


def main():
    ap = argparse.ArgumentParser(description="Test whether the objective set is redundant")
    ap.add_argument("--results_csv", required=True)
    args = ap.parse_args()

    if not os.path.exists(args.results_csv):
        print(f"ERROR: results CSV not found: {args.results_csv}")
        sys.exit(1)

    df = pd.read_csv(args.results_csv)
    ma.require_objective_columns(df, args.results_csv)
    df = df.dropna(subset=ma.OBJ_COLUMNS + ma.PARAM_NAMES)
    if {"d2_inner_mm", "d3_inner_mm"}.issubset(df.columns):
        df = df[df.apply(lambda r: ma.is_ring_ordering_ok(r.d2_inner_mm, r.d3_inner_mm), axis=1)]
    df = df.reset_index(drop=True)
    n = len(df)
    print(f"{n} designs with all five objectives and valid geometry\n")

    short = {c: l.split(" (")[0] for c, l in zip(ma.OBJ_COLUMNS, ma.OBJ_NAMES)}

    print("=" * 74)
    print("1. HOW MUCH DO THE OBJECTIVES OVERLAP?")
    print("=" * 74)
    print("\nPairwise Spearman, in maximization space (so + means they agree):\n")
    mx = pd.DataFrame(to_max(df, ma.OBJ_COLUMNS), columns=[short[c] for c in ma.OBJ_COLUMNS])
    corr = mx.corr(method="spearman")
    print(corr.round(2).to_string())

    off = corr.values[np.triu_indices(len(corr), k=1)]
    print(f"\nmean |rho| between objectives: {np.abs(off).mean():.2f}")
    print("(near 0 would mean genuinely independent objectives)")

    # The same numbers in physical units. The maximization-space table above is
    # what the acquisition function sees, but it silently negates every
    # minimized metric (FWHM, MPXI), so a reader reasoning about the PHYSICS
    # reads the sign backwards. That happened: a -0.33 for MPXI vs CNR in max
    # space is +0.33 between the measured quantities, i.e. more multiplexing
    # goes WITH better CNR -- the opposite of what the max-space sign suggests.
    print("\nThe same correlations in physical units (raw measured values,")
    print("no direction flips) -- read this one for the physics:\n")
    raw = df[ma.OBJ_COLUMNS].copy()
    raw.columns = [short[c] for c in ma.OBJ_COLUMNS]
    print(raw.corr(method="spearman").round(2).to_string())
    print("\n  Minimized metrics (FWHM, MPXI) are NOT negated here, so a + means")
    print("  the two measured quantities rise together, whatever we do with them.")
    mins = [n for n, d in zip(ma.OBJ_NAMES, ma.OBJ_DIRECTIONS) if d < 0]
    print(f"  Currently minimized: {', '.join(m.split(' (')[0] for m in mins)}")

    # PCA on the standardised objectives: how many directions carry the variance
    z = (mx - mx.mean()) / mx.std().replace(0, 1)
    ev = np.linalg.eigvalsh(np.cov(z.values.T))[::-1]
    ev = ev / ev.sum()
    print("\nVariance explained by each principal direction:")
    cum = 0.0
    for i, e in enumerate(ev, 1):
        cum += e
        print(f"  PC{i}: {e:6.1%}   cumulative {cum:6.1%}")
    n_eff = int(np.searchsorted(np.cumsum(ev), 0.90) + 1)
    print(f"\n-> {n_eff} directions capture 90% of the variance across 5 objectives")

    # Is each objective's DIRECTION right, not just its correlation?
    #
    # Spearman measures monotone association. An objective that helps up to a
    # point and hurts past it can still post a healthy rho while "maximize" is
    # the wrong instruction. That is exactly what happened with MPXI: +0.60
    # against CNR, but the best designs cluster near 2.0 and the top of the
    # range collapses to CNR 1.78. Binning catches the shape that a coefficient
    # averages away. Run for every objective, since being wrong about one was
    # enough to justify checking all of them.
    print()
    print("=" * 74)
    print("1b. IS EACH OBJECTIVE'S DIRECTION ACTUALLY RIGHT?")
    print("=" * 74)
    print("\nMean CNR in bins of each objective. An objective whose best CNR")
    print("sits in a MIDDLE bin has an optimum, not a direction.\n")
    print(f"{'objective':<24} {'best bin':>22} {'top bin below peak by':>23}")
    print("-" * 72)
    for col, name, d in zip(ma.OBJ_COLUMNS, ma.OBJ_NAMES, ma.OBJ_DIRECTIONS):
        if col == OUTCOME:
            continue
        sub = df[[col, OUTCOME]].dropna()
        if len(sub) < 40:
            continue
        try:
            bins = pd.qcut(sub[col], q=8, duplicates="drop")
        except ValueError:
            continue
        g = sub.groupby(bins, observed=True)[OUTCOME].mean()
        if len(g) < 3:
            continue
        peak = g.idxmax()
        # "Preferred end" is the top bin for a maximized objective, the bottom
        # bin for a minimized one.
        end = g.index[-1] if d > 0 else g.index[0]
        shortfall = g.max() - g.loc[end]
        flag = "" if peak == end else "   <-- OPTIMUM, not a direction"
        print(f"{name.split(' (')[0]:<24} "
              f"{f'{peak.left:.3g} to {peak.right:.3g}':>22} "
              f"{shortfall:>23.3f}{flag}")
    print("\nA large shortfall means the objective is pushing the search past")
    print("where CNR actually peaks, toward designs that score well on the")
    print("metric and badly as scanners.")

    print()
    print("=" * 74)
    print("2. DOES A SMALLER SET PICK THE SAME DESIGNS?")
    print("=" * 74)
    full_mask = pareto_mask(to_max(df, ma.OBJ_COLUMNS))
    full_set = set(df.index[full_mask])
    # A smaller front is always a subset of a larger one, so "how much of the
    # subset front is in the full front" is trivially 100% and says nothing.
    # The informative direction is the reverse: how much of the five-objective
    # front exists ONLY because of the extra objectives.
    print(f"\n{'objective set':<26} {'Pareto':>7} {'% of all':>9} {'adds to 5-obj front':>20}")
    print("-" * 66)
    for label, cols in SUBSETS.items():
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue
        m = pareto_mask(to_max(df, cols))
        s = set(df.index[m])
        extra = len(full_set - s)
        print(f"{label:<26} {len(s):>7} {len(s)/n:>8.1%} "
              f"{extra:>13} designs")

    print("""
A five-objective front covering most of the archive is the tell: with five
objectives a design is dominated only if another beats it everywhere, so almost
nothing is dominated and the front stops being a filter.""")

    print()
    print("=" * 74)
    print("3. WOULD A SMALLER SET HAVE CHOSEN THE SAME TOP DESIGNS?")
    print("=" * 74)
    print("\nTop 5 by CNR, and their rank under each subset's hypervolume")
    print("contribution (higher contribution = more valuable to that front):\n")

    top = df.nlargest(5, "cnr_sector_mean")
    masks = {lab: pareto_mask(to_max(df, [c for c in cols if c in df.columns]))
             for lab, cols in SUBSETS.items()}
    print(f"{'design':<30} {'CNR':>7} " +
          " ".join(f"{SUBSET_TAGS[lab]:>10}" for lab in SUBSETS))
    print("-" * (40 + 11 * len(SUBSETS)))
    for idx, row in top.iterrows():
        pos = df.index.get_loc(idx)
        cells = ["front" if masks[lab][pos] else "  --" for lab in SUBSETS]
        cfg = str(row["config"])[:28]
        print(f"{cfg:<30} {row['cnr_sector_mean']:>7.3f} " +
              " ".join(f"{c:>10}" for c in cells))

    print("""
How to read this:
  - If the top CNR designs sit on the front under every subset, the extra
    objectives are not what put them there, and a smaller set would have found
    them too.
  - Compare the number of principal directions carrying 90% of the variance
    against the 5 objectives declared. A gap means the formulation is asking for
    more trade-offs than the physics actually offers. That is not a failure --
    it is worth knowing, because a smaller set gives the acquisition function a
    sharper signal and a front that actually discriminates.
  - Sign matters in the correlation table. It is in maximization space, so a
    NEGATIVE entry is a genuine trade-off between two objectives, and a strongly
    positive one means the pair is largely restating a single quantity.
  - The "no CNR" row is the useful control: if the four proxies alone pick the
    same designs, they are doing their job as cheap stand-ins for an expensive
    reconstruction.
""")


if __name__ == "__main__":
    main()
