#!/bin/bash
# One arm of the two-objective vs five-objective head-to-head.
#
# Both arms start from the SAME fresh 6D LHS seeds and differ only in what they
# optimize, which is what the replay could not establish: the replay's candidate
# pool was chosen by the five-objective campaign, so a subset winning there was
# always searching a space something else mapped.
#
# Usage:
#   sbatch submit_mobo_headtohead.sh 2obj
#   sbatch submit_mobo_headtohead.sh 5obj
#
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
# MEASURED, 4 cores: the 2obj arm used 44 min of CPU across 71 h, the 5obj arm
# 14 h across 72 h. Both spend most of their life asleep waiting on pipeline
# jobs, so 4 cores is generous already.
#SBATCH --cpus-per-task=4
#
# MEASURED peak RSS, same design space and ~100 training points:
#     2obj arm (25671477)   0.98 GB
#     5obj arm (25671478)  49.60 GB      <- 50x, purely from objective count
#
# qLogNEHVI's hypervolume partition is what costs this, and it grows with both
# the objective count and the training-set size. 160G is ~3x the 5obj peak at
# 100 points and leaves room as the archive grows; the main campaign OOM-killed
# at 96G with 189 points and five objectives.
#
# SBATCH directives cannot branch on the arm, so this default is sized for the
# expensive case. When submitting a 2obj arm, override it -- 160G for a job that
# uses 1 GB just queues behind nodes it does not need:
#
#     sbatch --mem=16G submit_mobo_headtohead.sh 2obj 1
#     sbatch           submit_mobo_headtohead.sh 5obj 1
#
#SBATCH --mem=160G
#SBATCH --time=72:00:00
# Controllers send no mail on success (there are only a handful, and they run
# for days), but a controller dying is the expensive failure: in Aug 2026 four
# of them died silently when /vscratch filled and it went unnoticed for two
# days. FAIL and TIMEOUT are worth an email.
#SBATCH --mail-user=syedomer@buffalo.edu
#SBATCH --mail-type=FAIL,TIMEOUT
#SBATCH --job-name=mobo_h2h
#SBATCH --output=results/slurm_logs/out/h2h_%j.out
#SBATCH --error=results/slurm_logs/err/h2h_%j.err

set -euo pipefail

ARM="${1:-}"
if [[ "$ARM" != "2obj" && "$ARM" != "5obj" ]]; then
    echo "ERROR: pass an arm: sbatch submit_mobo_headtohead.sh {2obj|5obj} [replicate]" >&2
    exit 2
fi

# Replicate index. One run per arm cannot separate these formulations: the
# replay's own spread was 12 +/- 9 against 26 +/- 18, so single trajectories are
# mostly noise. Replicates are PAIRED -- replicate N gives both arms the same
# LHS seed set, and different replicates use different seed sets, so the start
# is blocked out rather than confounded with the objective set.
REP="${2:-0}"
if ! [[ "$REP" =~ ^[0-9]+$ ]]; then
    echo "ERROR: replicate must be a non-negative integer, got '$REP'" >&2
    exit 2
fi

BASE=/vscratch/grp-rutaoyao/Omer/spebt/spebt/spebt
# The venv lives in HOME, not /vscratch. Scratch is auto-purged and in
# Aug 2026 it removed .venv/bin/python mid-campaign. Override with
# SPEBT_VENV if the environment moves again.
source "${SPEBT_VENV:-$HOME/spebt-venv}/bin/activate"
cd "$BASE/optimization"

# Each arm and replicate gets its own results directory: manifest, results CSV,
# singleton lock and logs. Sharing one would make two campaigns claim the same
# manifest indices and interleave rows into a single CSV, unrecoverable after
# the fact. Replicate 0 keeps the original unsuffixed names so the first
# experiment's directories stay valid.
if [[ "$REP" == "0" ]]; then
    export MOBO_RESULTS_DIR="$BASE/optimization/results_h2h_${ARM}"
else
    export MOBO_RESULTS_DIR="$BASE/optimization/results_h2h_${ARM}_r${REP}"
fi

if [[ "$ARM" == "2obj" ]]; then
    export MOBO_OBJECTIVES="cnr_sector_mean,mpxi_windowed_active_mean"
else
    # Unset means all five. Set explicitly so the log records what ran rather
    # than leaving it to whatever the default was on the day.
    export MOBO_OBJECTIVES="fwhm_weighted_mean,asci_pct_fwhm0p45,ppds_ring1,mpxi_windowed_active_mean,cnr_sector_mean"
fi

mkdir -p "$MOBO_RESULTS_DIR/slurm_logs/out" "$MOBO_RESULTS_DIR/slurm_logs/err"

echo "arm         : $ARM"
echo "replicate   : $REP"
echo "results dir : $MOBO_RESULTS_DIR"
echo "objectives  : $MOBO_OBJECTIVES"

# 80 optimized iterations on top of the 21 shared LHS seeds. The seeds live in
# the results CSV as training data but are NOT manifest rows, so --max_iters
# counts optimized iterations only.
#
# Both arms get the same budget. The comparison is best CNR against iteration
# number, so an unequal budget would make the curves incomparable past the
# shorter one.
python run_mobo_loop.py --max_iters 80
