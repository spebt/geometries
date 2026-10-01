#!/bin/bash
# Reconstruct a few designs at several ML-EM iteration counts.
#
# RY (Jul 2026): contrast and noise in an ML-EM reconstruction both depend on the
# iteration count, and a given count favours a particular rod size. Every design
# in the campaign was reconstructed at a fixed 150 iterations, so that choice is
# baked into every comparison we have made.
#
# What matters for optimization is not whether absolute CNR shifts with the
# iteration count -- it certainly will -- but whether the RANKING of designs
# holds. If the ranking is stable, the fixed choice is harmless for design
# selection even though the absolute numbers are arbitrary.
#
# Each task is one (design, iteration count) pair, writing into
# <work_dir>/iter_<N>/ so the runs do not overwrite each other or the in-loop
# reconstruction.
#
# Usage:
#   sbatch --array=0-14 submit_iteration_sweep.sh
#   python3 analyze_iteration_sweep.py --results_csv results/results_summary_mobo.csv
#
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=iter_sweep
#SBATCH --output=results/slurm_logs/out/iter_sweep_%A_%a.out
#SBATCH --error=results/slurm_logs/err/iter_sweep_%A_%a.err

set -euo pipefail

# The venv lives in HOME, not /vscratch. Scratch is auto-purged and in
# Aug 2026 it removed .venv/bin/python mid-campaign. Override with
# SPEBT_VENV if the environment moves again.
VENV_PY="${SPEBT_VENV:-$HOME/spebt-venv}/bin/python3"
CODE_DIR=/vscratch/grp-rutaoyao/Omer/spebt/spebt/spebt
cd "${CODE_DIR}/optimization"

PHANTOM_PATH="${PHANTOM_PATH:-${CODE_DIR}/data/sai_10mm/hot_rods_phantom_10.0_mm_x_10.0_mm.pt}"
RESULTS_DIR="${CODE_DIR}/optimization/results"

# Designs spanning the CNR range: the best, a mid-tier one, and a poor one.
# If the ranking is going to break anywhere it will be between designs that are
# close, so the two leaders are both included.
CONFIGS=(
  "mobo_0069_ap0.3138_nap124_nd1_612_nd2_230"
  "mobo_0177_ap0.3512_nap97_nd1_604_nd2_584"
  "mobo_0133_ap0.4008_nap70_nd1_660_nd2_562"
)
ITERATIONS=(25 50 100 150 300)

N_ITERS=${#ITERATIONS[@]}
cfg_idx=$(( SLURM_ARRAY_TASK_ID / N_ITERS ))
itr_idx=$(( SLURM_ARRAY_TASK_ID % N_ITERS ))
CONFIG="${CONFIGS[$cfg_idx]}"
NITER="${ITERATIONS[$itr_idx]}"

echo "[array ${SLURM_ARRAY_TASK_ID}] ${CONFIG} @ ${NITER} iterations"

# Fixed seed so that differences across iteration counts are the iteration
# count, not a fresh Poisson draw. The same seed is used for every design so the
# comparison between designs is paired.
mkdir -p results/iter_sweep
"${VENV_PY}" compute_cnr.py \
    --work_dir "${RESULTS_DIR}/${CONFIG}" \
    --phantom_path "${PHANTOM_PATH}" \
    --out_csv "results/iter_sweep/task_${SLURM_ARRAY_TASK_ID}.csv" \
    --config_name "${CONFIG}__iter${NITER}" \
    --out_subdir "iter_${NITER}" \
    --seed 0 \
    --iterations "${NITER}"
