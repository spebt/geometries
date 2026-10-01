#!/usr/bin/env python3
"""
Two objectives against five, run live from identical seeds.

The replay (replay_objective_subsets.py) could only ever be suggestive: the
designs it searched were chosen by the five-objective campaign, so a smaller set
winning there was winning on someone else's map. This compares two campaigns
that started from the same 21 fresh 6D designs and differ only in what they
optimise.

Both arms are scored on CNR regardless of what they optimise. That is the point:
CNR is the outcome we care about, and an arm that does not include it as an
objective is still judged by it.

Comparability notes that matter for reading the output:

  - Hypervolume is NOT compared across arms. The two arms have different
    objective spaces, so their hypervolumes are different quantities with
    different units and comparing them would be meaningless.
  - The x axis is evaluations, not wall time. The arms cost very different
    amounts per iteration, which is reported separately and is part of the
    result rather than a confound to remove.
  - Both arms share the same 21 seed designs, so every trajectory starts from
    the same best-so-far value. A gap at iteration 0 would mean the arms were
    not seeded identically and the comparison is invalid.

Usage:
  python compare_headtohead.py
  python compare_headtohead.py --arms results_h2h_2obj results_h2h_5obj --plot out.png
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

OUTCOME = "cnr_sector_mean"
DEFAULT_ARMS = ["results_h2h_2obj", "results_h2h_5obj"]
RESULTS_NAME = "results_summary_mobo.csv"
MANIFEST_NAME = "mobo_manifest.csv"

# Measured over 5-seed repeats of 8 designs (Aug 2026): per-run std is
# 0.05-0.12, pooled ~0.08. The earlier 0.15 came from a single two-run
# spread and was about twice too conservative, which made real design
# differences read as "inside noise".
CNR_NOISE_SD = 0.08

# Fewest evaluations a run needs before a cross-arm comparison means
# anything. Below this the truncation just compares two campaigns that
# have barely started.
MIN_BUDGET = 10


def load_arm(arm_dir):
    """Best-CNR-so-far against evaluation number for one arm."""
    res_path = os.path.join(arm_dir, RESULTS_NAME)
    man_path = os.path.join(arm_dir, MANIFEST_NAME)
    if not os.path.exists(res_path):
        return None, f"no results CSV at {res_path}"

    res = pd.read_csv(res_path)
    if OUTCOME not in res.columns:
        return None, f"{res_path} has no {OUTCOME} column"

    seeds = res[res["config"].astype(str).str.startswith("lhs6d_")]
    seed_best = seeds[OUTCOME].max() if len(seeds) else np.nan

    # Evaluation order comes from the manifest, not from row order in the
    # results CSV: compute_cnr.py rewrites that file in place and rows can move.
    if not os.path.exists(man_path):
        return None, f"no manifest at {man_path} (arm may not have started)"
    man = pd.read_csv(man_path)
    if "idx" not in man.columns or "config_name" not in man.columns:
        return None, f"{man_path} has an unexpected schema"

    order = man.sort_values("idx")[["idx", "config_name"]]
    merged = order.merge(res[["config", OUTCOME]],
                         left_on="config_name", right_on="config", how="left")

    cnr = merged[OUTCOME].values.astype(float)
    # A failed or still-running iteration has no CNR. Treat it as "no
    # improvement" rather than dropping it, so the x axis stays evaluation
    # count. Dropping them would credit an arm for iterations it wasted.
    running = np.maximum.accumulate(np.where(np.isnan(cnr), -np.inf, cnr))
    running = np.where(np.isinf(running), np.nan, running)
    if np.isfinite(seed_best):
        running = np.maximum(running, seed_best)

    # Separate "finished but produced nothing" from "still running". The
    # controller records no_row / all_nan for the former since Aug 2026; older
    # manifests only have "completed", so fall back to treating the final row as
    # in-flight when the status column cannot distinguish them.
    status = man.sort_values("idx").get("status")
    if status is not None:
        st = status.astype(str).values
        n_lost = int(((st == "no_row") | (st == "all_nan")).sum())
        n_pending = int(((st == "submitted") | (st == "running")).sum())
    else:
        n_lost, n_pending = 0, 0
    missing = len(merged) - int(np.isfinite(cnr).sum())
    if n_lost == 0 and n_pending == 0 and missing > 0:
        # Legacy manifest: assume at most one in-flight row, rest are losses.
        n_pending = 1 if missing >= 1 else 0
        n_lost = max(0, missing - n_pending)

    return {
        "n_iters": len(merged),
        "n_evaluated": int(np.isfinite(cnr).sum()),
        "n_lost": n_lost,
        "n_pending": n_pending,
        "seed_best": seed_best,
        "curve": running,
        "final": float(np.nanmax(running)) if np.isfinite(running).any() else np.nan,
    }, None


def short(arm_dir):
    """Directory basename, so the tables stay readable with absolute paths."""
    return os.path.basename(os.path.normpath(arm_dir)).replace("results_h2h_", "")


def evals_to_reach(curve, target):
    hit = np.where(np.asarray(curve) >= target)[0]
    return int(hit[0] + 1) if len(hit) else None


def arm_label(arm_dir):
    """'2obj' from results_h2h_2obj_r1, so replicates group together."""
    s = short(arm_dir)
    return re.sub(r"_r\d+$", "", s)


def report_replicates(args):
    """Aggregate paired replicates: per-arm mean and spread, not single runs.

    One trajectory per arm cannot separate these formulations. The replay's own
    spread was 12 +/- 9 against 26 +/- 18, so a single run lands almost anywhere
    in that range. This groups results_h2h_<arm>_r<N> directories by arm and
    reports the spread across replicates.
    """
    groups = {}
    for a in args.arms:
        data, err = load_arm(a)
        if err:
            print(f"  {short(a)}: {err}")
            continue
        groups.setdefault(arm_label(a), []).append((a, data))

    if not groups:
        print("\nNo arm has usable data yet.")
        sys.exit(1)

    print("=" * 74)
    print("HEAD-TO-HEAD ACROSS REPLICATES")
    print("=" * 74)

    # Paired design: within a replicate both arms share a seed set, so a seed
    # best that differs between arms of the SAME replicate invalidates it.
    by_rep = {}
    for label, runs in groups.items():
        for a, d in runs:
            m = re.search(r"_r(\d+)$", short(a))
            by_rep.setdefault(m.group(1) if m else "0", {})[label] = d["seed_best"]
    for rep, seeds in sorted(by_rep.items()):
        vals = {round(v, 6) for v in seeds.values() if np.isfinite(v)}
        if len(vals) > 1:
            print(f"\n*** WARNING: replicate {rep} arms do not share a seed best "
                  f"({seeds}). That replicate is not paired and should be excluded.")

    # Compare at a budget every run reached, so a short run cannot flatter an arm.
    budget = min(d["n_iters"] for runs in groups.values() for _, d in runs)

    print("\nProgress per run (manifest rows / with a CNR value):")
    for label, runs in sorted(groups.items()):
        counts = ", ".join(f"{short(a)}={d['n_iters']}/{d['n_evaluated']}"
                           for a, d in runs)
        print(f"  {label:<8} {counts}")

    # An iteration with no CNR is treated as "no improvement" and still counted,
    # which is right for a FAILED evaluation and wrong for one that never ran.
    # Flag it rather than silently crediting an arm for iterations it did not do.
    #
    # Read the manifest STATUS rather than inferring from a missing CNR. A
    # running controller always has one iteration in flight whose pipeline has
    # not written its metrics yet, and counting that as a loss made every
    # healthy arm emit a warning -- which is how a real signal gets ignored.
    for label, runs in sorted(groups.items()):
        for a, d in runs:
            lost, pending = d["n_lost"], d["n_pending"]
            if lost > 0:
                print(f"  [warn] {short(a)}: {lost} of {d['n_iters']} iterations "
                      f"finished WITHOUT metrics.\n         This arm is credited "
                      f"with evaluations it never completed. Compare hit RATES, "
                      f"not totals.")
            elif pending:
                print(f"  {short(a)}: {pending} iteration(s) still in flight "
                      f"(counted as incomplete, not lost)")

    # A common budget set by a just-started replicate makes every column read
    # "not reached" and every mean NaN, which looks like a result and is not.
    # Report progress and stop instead.
    if budget < MIN_BUDGET:
        print(f"\nCommon budget is only {budget} evaluation(s), set by the "
              f"shortest run above.")
        print("Truncating to that would compare campaigns before either has")
        print("done anything. Nothing to conclude yet -- rerun once every run")
        print(f"has at least {MIN_BUDGET} evaluations.")
        return

    print(f"\nAll runs truncated to their common budget of {budget} evaluations,")
    print("so an arm that ran longer gets no advantage.\n")

    targets = [4.2, 4.4, 4.6]
    hdr = f"{'arm':<10} {'runs':>5} " + "".join(f"{f'-> {t}':>16}" for t in targets) + f"{'best CNR':>16}"
    print(hdr)
    print("-" * len(hdr))
    for label, runs in sorted(groups.items()):
        cells = ""
        for t in targets:
            got = [evals_to_reach(d["curve"][:budget], t) for _, d in runs]
            hit = [g for g in got if g]
            cells += (f"{f'{np.mean(hit):.0f}+/-{np.std(hit):.0f} ({len(hit)}/{len(got)})':>16}"
                      if hit else f"{'not reached':>16}")
        finals = [float(np.nanmax(d["curve"][:budget])) for _, d in runs]
        print(f"{label:<10} {len(runs):>5} {cells}"
              f"{f'{np.mean(finals):.3f}+/-{np.std(finals):.3f}':>16}")

    n_runs = min(len(r) for r in groups.values())
    if n_runs < 3:
        print(f"\nOnly {n_runs} run(s) per arm. That is not enough to separate these")
        print("formulations: single trajectories vary by more than the difference")
        print("being measured. Treat any ordering here as provisional until there")
        print("are at least 3 paired replicates per arm.")


def main():
    ap = argparse.ArgumentParser(description="Compare head-to-head campaign arms")
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    ap.add_argument("--plot", default=None, help="Write a trajectory plot here")
    ap.add_argument("--replicates", action="store_true",
                    help="Group results_h2h_<arm>_r<N> dirs by arm and report "
                         "mean and spread across replicates")
    args = ap.parse_args()

    if args.replicates:
        report_replicates(args)
        return

    arms, problems = {}, []
    for a in args.arms:
        data, err = load_arm(a)
        if err:
            problems.append(f"  {a}: {err}")
        else:
            arms[a] = data

    for p in problems:
        print(p)
    if not arms:
        print("\nNo arm has usable data yet. Both arms need at least one "
              "completed iteration.")
        sys.exit(1)

    print("=" * 74)
    print("HEAD-TO-HEAD: TWO OBJECTIVES vs FIVE, FROM IDENTICAL SEEDS")
    print("=" * 74)

    # If the arms were not seeded identically the comparison is invalid, so
    # check it before reporting anything that depends on it.
    seed_vals = {a: d["seed_best"] for a, d in arms.items()}
    uniq = {round(v, 6) for v in seed_vals.values() if np.isfinite(v)}
    print(f"\nBest CNR among the shared seeds: "
          f"{', '.join(f'{short(a)}={v:.4f}' for a, v in seed_vals.items())}")
    if len(uniq) > 1:
        print("\n*** WARNING: the arms do not share a seed best. They were not")
        print("*** seeded identically, so any difference below may be the")
        print("*** starting data rather than the objective set. Re-seed with")
        print("*** merge_lhs6d_seeds.py before drawing conclusions.")

    print(f"\n{'arm':<22} {'iters':>7} {'evaluated':>10} {'best CNR':>10}")
    print("-" * 52)
    for a, d in arms.items():
        print(f"{short(a):<22} {d['n_iters']:>7} {d['n_evaluated']:>10} {d['final']:>10.4f}")

    # Evaluations to reach shared thresholds. Using thresholds both arms can
    # reach avoids the trap of scoring each arm against its own best, which
    # would flatter whichever arm got lucky.
    finals = [d["final"] for d in arms.values() if np.isfinite(d["final"])]
    if finals:
        lo = min(finals)
        targets = [t for t in (4.2, 4.4, 4.6, round(lo, 3)) if t <= lo + 1e-9]
        targets = sorted(set(targets))
        if targets:
            print(f"\nEvaluations to reach each CNR level "
                  f"(only levels both arms reached):\n")
            hdr = f"{'arm':<22}" + "".join(f"{t:>10.3f}" for t in targets)
            print(hdr)
            print("-" * len(hdr))
            for a, d in arms.items():
                cells = ""
                for t in targets:
                    n = evals_to_reach(d["curve"], t)
                    cells += f"{n:>10}" if n else f"{'--':>10}"
                print(f"{short(a):<22}{cells}")

    if len(finals) == 2:
        gap = abs(finals[0] - finals[1])
        print(f"\nDifference in best CNR between arms: {gap:.4f}")
        if gap < CNR_NOISE_SD:
            print(f"That is below the {CNR_NOISE_SD} single-reconstruction noise")
            print("level, so the arms have found designs of indistinguishable")
            print("quality. Any claim has to rest on HOW FAST each got there,")
            print("not on which final number is larger.")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for a, d in arms.items():
                ax.plot(np.arange(1, len(d["curve"]) + 1), d["curve"],
                        label=a.replace("results_h2h_", ""), lw=2)
            ax.set_xlabel("evaluations after the shared seeds")
            ax.set_ylabel("best CNR sector-mean so far")
            ax.set_title("Two objectives vs five, identical 21-design start")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(args.plot, dpi=150)
            print(f"\nwrote {args.plot}")
        except Exception as e:
            print(f"\n[warn] plot failed: {e}")


if __name__ == "__main__":
    main()
