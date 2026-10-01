#!/bin/bash
# Submit the sequential MOBO controller for the SAI SC-SPECT design search.
#
# The controller (run_mobo_loop.py) reads the manifest to figure out how many
# iterations have already been completed and picks up from there — this script
# is safe to re-submit any time; it will only propose the delta up to
# --max_iters.
#
# Committed to git so it survives vscratch auto-cleanups.
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --cpus-per-task=4
# Sized against qLogNEHVI's hypervolume computation, which scales badly in the
# number of X_baseline points. 96 GB sufficed at 120 training points (58.7 GB
# peak) but OOM-killed at 189. If the training set grows much past ~250 this
# will recur, and the real fix is capping X_baseline rather than the node.
#SBATCH --mem=160G
#SBATCH --time=72:00:00
# Controllers send no mail on success (there are only a handful, and they run
# for days), but a controller dying is the expensive failure: in Aug 2026 four
# of them died silently when /vscratch filled and it went unnoticed for two
# days. FAIL and TIMEOUT are worth an email.
#SBATCH --mail-user=syedomer@buffalo.edu
#SBATCH --mail-type=FAIL,TIMEOUT
#SBATCH --job-name=mobo_loop
#SBATCH --output=results/slurm_logs/out/mobo_%j.out
#SBATCH --error=results/slurm_logs/err/mobo_%j.err

set -euo pipefail

# The venv lives in HOME, not /vscratch. Scratch is auto-purged and in
# Aug 2026 it removed .venv/bin/python mid-campaign. Override with
# SPEBT_VENV if the environment moves again.
source "${SPEBT_VENV:-$HOME/spebt-venv}/bin/activate"
cd /vscratch/grp-rutaoyao/Omer/spebt/spebt/spebt/optimization

mkdir -p results/slurm_logs/out results/slurm_logs/err

# Bump this ceiling as needed. Controller no-ops once the manifest already has
# --max_iters rows, so it's safe to set higher than we plan to run in one go.
# 180 closed out the 4D campaign; 260 gave 80 iterations of 6D search; 300 was
# reached at the point MPXI was corrected (Aug 4). 380 gives ~100 iterations on
# the corrected objective set, where MPXI is windowed+active and maximized.
# The existing 224 designs stay valid as training data: the design-to-metric
# mapping did not change, only which column is read and its sign.
# 420 gives ~100 iterations past the 322 the campaign reached, now that
# n_det_ring2 can exceed 960 and the ring2_* seeds have put training data in
# the newly opened region.
python run_mobo_loop.py --max_iters 420
