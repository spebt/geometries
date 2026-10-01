#!/bin/bash
# Evaluate the D2/D3-spanning seed designs from make_d2d3_seed_list.py.
#
# These exist to give the surrogate variance along the two new design axes. The
# campaign's training data is constant there (d3 std = 0.015 mm over a 255 mm
# range), so the GP cannot learn any dependence and the acquisition never
# proposes anything off the legacy layout.
#
# Each task runs the full pipeline for one seed: geometry, PPDF, beam analysis,
# metrics, CNR. Results go straight into the main results CSV so the next
# controller restart picks them up as training data.
#
# Usage:
#   python make_d2d3_seed_list.py --results_csv results/results_summary_mobo.csv
#   sbatch --array=0-5%4 submit_d2d3_seeds.sh
#
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --cpus-per-task=36
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=d2d3_seed
#SBATCH --output=results/slurm_logs/out/d2d3_seed_%A_%a.out
#SBATCH --error=results/slurm_logs/err/d2d3_seed_%A_%a.err

set -euo pipefail

CODE_DIR=/vscratch/grp-rutaoyao/Omer/spebt/spebt/spebt
cd "${CODE_DIR}/optimization"

SEED_CSV="${SEED_CSV:-d2d3_seeds.csv}"

# Deliberately NOT the main results CSV. The controller is normally running
# while these evaluate, and compute_cnr.py does a full read-modify-write, so
# concurrent writers would silently drop rows. Each task writes its own file;
# merge_d2d3_seeds.py folds them in once everything has finished.
mkdir -p results/d2d3_seed_out
RESULTS_CSV="${RESULTS_CSV:-${CODE_DIR}/optimization/results/d2d3_seed_out/task_${SLURM_ARRAY_TASK_ID}.csv}"

# Row 0 of the CSV is the header, so task N maps to line N+2
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${SEED_CSV}")
if [[ -z "${LINE}" ]]; then
  echo "No entry at task ${SLURM_ARRAY_TASK_ID} of ${SEED_CSV}"
  exit 0
fi

IFS=, read -r CONFIG_NAME APERTURE_DIAM N_APERTURES N_DET_RING1 N_DET_RING2 D2_INNER D3_INNER <<< "${LINE}"

WORK_DIR="${CODE_DIR}/optimization/results/${CONFIG_NAME}"
mkdir -p "${WORK_DIR}"

echo "[seed ${SLURM_ARRAY_TASK_ID}] ${CONFIG_NAME}"
echo "  d2=${D2_INNER} d3=${D3_INNER}"

# Run the same pipeline the MOBO loop uses, so these rows are directly
# comparable to every other config in the CSV.
export WORK_DIR APERTURE_DIAM N_APERTURES N_DET_RING1 N_DET_RING2 \
       D2_INNER D3_INNER CODE_DIR RESULTS_CSV CONFIG_NAME
export A_MM=0.2 B_MM=0.2

bash "${CODE_DIR}/optimization/run_sai_pipeline.sh"
