#!/bin/bash
# Evaluate the fresh 6D LHS seed designs shared by both head-to-head arms.
#
# These are evaluated ONCE and then merged into both arms' results files, so the
# two campaigns start from byte-identical training data and the only difference
# between them is what they optimize. Evaluating separately per arm would let
# Poisson noise in the reconstructions differ between the arms and contaminate
# the comparison at its starting point.
#
# Usage:
#   python make_lhs6d_seeds.py --n_seeds 21 --out lhs6d_seeds.csv
#   sbatch --array=0-20%8 submit_lhs6d_seeds.sh
#   python merge_lhs6d_seeds.py           # after all tasks finish
#
# The %8 concurrency cap is deliberate. run_recon reads 16 HDF5 files inside its
# 150-iteration loop, which is 2,400 reads per reconstruction; at higher
# concurrency the I/O contention timed out 15 of 67 tasks in a previous array.
#
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
# MEASURED (job 25704120_0): 22.2 cores of 36 used, 17.0 GB of 32.
# The PPDF ray tracing does parallelise, so this is the one job where a
# large core count earns its keep. 24 matches observed use and schedules
# sooner than 36 on a busy partition.
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=lhs6d_seed
#SBATCH --output=results/slurm_logs/out/lhs6d_%A_%a.out
#SBATCH --error=results/slurm_logs/err/lhs6d_%A_%a.err

set -euo pipefail

CODE_DIR=/vscratch/grp-rutaoyao/Omer/spebt/spebt/spebt
cd "${CODE_DIR}/optimization"

SEED_CSV="${SEED_CSV:-lhs6d_seeds.csv}"

# One output file per task. compute_cnr.py does a full read-modify-write of the
# results CSV, so concurrent array tasks writing one file silently drop rows.
#
# Override the DIRECTORY, not the file: RESULTS_CSV has to embed
# SLURM_ARRAY_TASK_ID, which does not exist when sbatch captures the
# environment, so exporting RESULTS_CSV from outside leaves a literal
# "${SLURM_ARRAY_TASK_ID}" in the path and every task writes the same file.
TASK_DIR="${TASK_DIR:-results/lhs6d_seed_out}"
mkdir -p "${TASK_DIR}"
RESULTS_CSV="${CODE_DIR}/optimization/${TASK_DIR}/task_${SLURM_ARRAY_TASK_ID}.csv"

# Row 0 of the CSV is the header, so task N maps to line N+2
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${SEED_CSV}")
if [[ -z "${LINE}" ]]; then
  echo "No entry at task ${SLURM_ARRAY_TASK_ID} of ${SEED_CSV}"
  exit 0
fi

IFS=, read -r CONFIG_NAME APERTURE_DIAM N_APERTURES N_DET_RING1 N_DET_RING2 D2_INNER D3_INNER <<< "${LINE}"

WORK_DIR="${CODE_DIR}/optimization/results/${CONFIG_NAME}"
mkdir -p "${WORK_DIR}"

echo "[lhs6d ${SLURM_ARRAY_TASK_ID}] ${CONFIG_NAME}"
echo "  aperture=${APERTURE_DIAM} n_ap=${N_APERTURES}"
echo "  nd1=${N_DET_RING1} nd2=${N_DET_RING2} d2=${D2_INNER} d3=${D3_INNER}"

# The same pipeline the MOBO loop uses, so these rows are directly comparable to
# every other config in the archive.
export WORK_DIR APERTURE_DIAM N_APERTURES N_DET_RING1 N_DET_RING2 \
       D2_INNER D3_INNER CODE_DIR RESULTS_CSV CONFIG_NAME
export A_MM=0.2 B_MM=0.2

bash "${CODE_DIR}/optimization/run_sai_pipeline.sh"
