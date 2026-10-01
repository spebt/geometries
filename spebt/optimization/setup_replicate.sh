#!/bin/bash
# Prepare one PAIRED replicate of the head-to-head experiment.
#
# Why replicates: the first live run gave 2obj 22 evaluations to reach CNR 4.6
# and 5obj 14, which reverses what the replay predicted. Both are single
# trajectories. The replay's own spread was 12 +/- 9 against 26 +/- 18, wide
# enough that one run per arm is close to uninformative. Three paired replicates
# per arm turn a point into a spread.
#
# Why PAIRED: replicate N generates ONE LHS seed set and gives it to BOTH arms,
# so the starting data is blocked out rather than confounded with the objective
# set. Different replicates use different seed sets, so the conclusion does not
# rest on one lucky or unlucky start.
#
# This script only prepares the seeds. It prints the two sbatch lines to run
# once the seed array has finished, because the campaigns cannot start until
# their training data exists.
#
# Usage:
#   ./setup_replicate.sh 1        # prepare replicate 1
#   ./setup_replicate.sh 2

set -euo pipefail

REP="${1:-}"
if ! [[ "$REP" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: usage: ./setup_replicate.sh <replicate>   (1 or higher)" >&2
    echo "Replicate 0 is the original experiment and is already set up." >&2
    exit 2
fi

BASE=/vscratch/grp-rutaoyao/Omer/spebt/spebt/spebt
cd "$BASE/optimization"

SEED_CSV="lhs6d_seeds_r${REP}.csv"
TASK_DIR="results/lhs6d_seed_out_r${REP}"
ARM_2="results_h2h_2obj_r${REP}"
ARM_5="results_h2h_5obj_r${REP}"

for d in "$ARM_2" "$ARM_5"; do
    if [[ -s "$d/results_summary_mobo.csv" ]]; then
        echo "ERROR: $d already has results. Refusing to overwrite a replicate" >&2
        echo "that has already run. Pick a different replicate number." >&2
        exit 1
    fi
done

# A different --seed per replicate is the whole point: replicate 0 used seed 0,
# so reusing it would produce an identical start and measure nothing new.
#
# --prefix is equally load-bearing and less obvious. run_sai_pipeline.sh derives
# its work directory from the config name, so without a per-replicate prefix
# every replicate evaluates into results/lhs6d_000..020 -- the same 21
# directories, concurrently, with different designs and no error.
echo "=== generating seed designs for replicate ${REP} ==="
python make_lhs6d_seeds.py --n_seeds 21 --seed "$REP" \
    --prefix "lhs6d_r${REP}_" --out "$SEED_CSV"

mkdir -p "$TASK_DIR"

echo
echo "=== next steps ==="
echo
echo "1. Evaluate the seeds (about 4 hours):"
echo
echo "   SEED_CSV=${SEED_CSV} TASK_DIR=${TASK_DIR} sbatch --array=0-20%8 submit_lhs6d_seeds.sh"
echo
echo "2. When that array finishes, seed both arms identically:"
echo
echo "   python merge_lhs6d_seeds.py --task_glob '${TASK_DIR}/task_*.csv' \\"
echo "     --arms ${ARM_2} ${ARM_5} --dry_run"
echo
echo "   (drop --dry_run once the counts look right)"
echo
echo "3. Launch both arms:"
echo
echo "   sbatch submit_mobo_headtohead.sh 2obj ${REP}"
echo "   sbatch submit_mobo_headtohead.sh 5obj ${REP}"
echo
echo "Both arms must start from the SAME seed evaluations. merge_lhs6d_seeds.py"
echo "prints a digest of the seed frame; it should match between the arms."
