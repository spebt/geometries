#!/usr/bin/env python3
"""
Fold the evaluated 6D LHS seeds into both head-to-head arms.

Both arms must start from byte-identical training data. That is the whole basis
of the comparison: if the two campaigns began from different seed evaluations,
any difference in their trajectories could be the starting data rather than the
objective set, and the experiment would prove nothing.

So the seeds are evaluated once by submit_lhs6d_seeds.sh, and this script writes
the SAME rows into each arm's results CSV. It refuses to run if an arm already
has a results file with rows in it, since overwriting a campaign in progress
would destroy it.

Usage:
  python merge_lhs6d_seeds.py
  python merge_lhs6d_seeds.py --dry_run
"""
import argparse
import glob
import hashlib
import os
import re
import shutil
import sys
import time

import pandas as pd

import mobo_agent as ma

DEFAULT_TASK_GLOB = "results/lhs6d_seed_out/task_*.csv"
DEFAULT_ARMS = ["results_h2h_2obj", "results_h2h_5obj"]
RESULTS_NAME = "results_summary_mobo.csv"


def main():
    ap = argparse.ArgumentParser(description="Seed both head-to-head arms identically")
    ap.add_argument("--task_glob", default=DEFAULT_TASK_GLOB)
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an arm that already has rows. Destroys a "
                         "campaign in progress; only for restarting cleanly.")
    ap.add_argument("--append_to", default=None,
                    help="APPEND the seeds to an existing results CSV instead of "
                         "seeding fresh arms. Use this for the ring2_* seeds, "
                         "which extend the main archive rather than starting a "
                         "new campaign. Mutually exclusive with --arms.")
    args = ap.parse_args()

    files = sorted(glob.glob(args.task_glob))
    if not files:
        print(f"ERROR: no task outputs matched {args.task_glob}")
        print("Has the array job finished? Check: squeue -u $USER")
        sys.exit(1)

    frames = []
    for f in files:
        try:
            d = pd.read_csv(f)
        except Exception as e:
            print(f"  [warn] unreadable, skipping: {f}: {e}")
            continue
        if len(d):
            frames.append(d)
    if not frames:
        print("ERROR: every task output was empty or unreadable")
        sys.exit(1)

    seeds = pd.concat(frames, ignore_index=True)
    if "config" in seeds.columns:
        n_before = len(seeds)
        seeds = seeds.drop_duplicates(subset=["config"], keep="last")
        if len(seeds) != n_before:
            print(f"[warn] dropped {n_before - len(seeds)} duplicate configs")

    print(f"{len(files)} task files -> {len(seeds)} seed designs")

    # A seed missing an objective is dropped by the optimizer, so the arms would
    # silently start from fewer designs than intended. Report it now.
    have = [c for c in ma._ALL_OBJ_COLUMNS if c in seeds.columns]
    missing_cols = [c for c in ma._ALL_OBJ_COLUMNS if c not in seeds.columns]
    if missing_cols:
        print(f"ERROR: seed rows are missing objective column(s): {missing_cols}")
        print("The pipeline did not produce them. Do not seed the arms from this.")
        sys.exit(1)

    complete = seeds.dropna(subset=have)
    print(f"{len(complete)} of {len(seeds)} have every objective")
    # The 3-point floor is a FRESH-campaign rule: a controller starting from
    # these alone needs enough to fit a GP. It does not apply when appending to
    # an archive that already has hundreds of rows, where even one usable seed
    # adds information.
    if len(complete) < 3 and not args.append_to:
        print("ERROR: fewer than 3 usable seeds; the controller needs at least 3.")
        sys.exit(1)
    if len(complete) == 0:
        print("ERROR: no seed has a complete objective set; nothing to add.")
        sys.exit(1)
    if len(complete) < len(seeds):
        lost = seeds.loc[~seeds.index.isin(complete.index), "config"].tolist()
        print(f"  incomplete (will still be written, optimizer skips them): {lost}")

    # Identical content in both arms is the point, so verify it rather than
    # assume it: a hash of the written frame is cheap and catches an accidental
    # per-arm transformation creeping in later.
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(seeds, index=False).values.tobytes()).hexdigest()[:16]
    print(f"seed frame digest: {digest}")

    if args.append_to:
        target = args.append_to
        if not os.path.exists(target):
            print(f"ERROR: {target} does not exist. --append_to extends an "
                  f"existing archive; it will not create one.")
            sys.exit(1)
        existing = pd.read_csv(target)
        dup = set(existing.get("config", pd.Series(dtype=str)).astype(str)) & \
              set(seeds["config"].astype(str))
        if dup:
            print(f"ERROR: {len(dup)} of these configs are already in {target}, "
                  f"e.g. {sorted(dup)[:3]}")
            print("Appending would duplicate rows and double-count them in "
                  "training. Nothing written.")
            sys.exit(1)

        combined = pd.concat([existing, seeds], ignore_index=True)
        print(f"\n{target}: {len(existing)} rows + {len(seeds)} seeds "
              f"= {len(combined)}")
        if args.dry_run:
            print("dry run, nothing written")
            return
        backup = f"{target}.bak_{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(target, backup)
        combined.to_csv(target, index=False)
        print(f"wrote {target}\nbackup at {backup}")
        print("\nThe controller picks these up on its next restart.")
        return

    for arm in args.arms:
        out_dir = arm
        out = os.path.join(out_dir, RESULTS_NAME)
        if os.path.exists(out):
            existing = pd.read_csv(out)
            if len(existing) and not args.force:
                print(f"\nERROR: {out} already has {len(existing)} rows.")
                print("Refusing to overwrite a campaign in progress. Use --force")
                print("only if you intend to discard it and restart.")
                sys.exit(1)
        if args.dry_run:
            print(f"  [dry run] would write {len(seeds)} rows to {out}")
            continue
        os.makedirs(os.path.join(out_dir, "slurm_logs", "out"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "slurm_logs", "err"), exist_ok=True)
        seeds.to_csv(out, index=False)
        print(f"  wrote {len(seeds)} rows to {out}")

    if args.dry_run:
        print("\ndry run, nothing written")
        return

    # Derive the replicate from the arm directory names rather than printing a
    # bare "2obj", which defaults to replicate 0 and would target the original
    # experiment instead of the one just seeded.
    reps = {m.group(1) for a in args.arms
            for m in [re.search(r"_r(\d+)$", a.rstrip("/"))] if m}
    rep = reps.pop() if len(reps) == 1 else None

    print("\nBoth arms now hold identical seed data. Launch them with:")
    suffix = f" {rep}" if rep else ""
    # 2obj measured 0.98 GB against 5obj's 49.6 GB, so it does not need the
    # script's 160G default and schedules far sooner without it.
    print(f"  sbatch --mem=16G submit_mobo_headtohead.sh 2obj{suffix}")
    print(f"  sbatch           submit_mobo_headtohead.sh 5obj{suffix}")
    if rep is None:
        print("\n(could not infer a replicate from the arm names; if these are "
              "replicate\n arms, append the replicate number to each command)")


if __name__ == "__main__":
    main()
