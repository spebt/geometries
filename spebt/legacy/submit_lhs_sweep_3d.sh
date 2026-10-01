#!/bin/bash
# Submit all 4D LHS configs from configs_manifest_3d.csv as SLURM jobs.
#
# Usage:
#   cd /vscratch/grp-rutaoyao/Omer/spebt/spebt/optimization
#   python generate_configs_3d.py --n_samples 25
#   bash submit_lhs_sweep_3d.sh
#
# Reads: results/configs_manifest_3d.csv
# Submits: one sbatch job per row using run_sai_pipeline.sh

set -euo pipefail

CODE_DIR="/vscratch/grp-rutaoyao/Omer/spebt/spebt"
RESULTS_DIR="${CODE_DIR}/optimization/results"
RESULTS_CSV="${RESULTS_DIR}/results_summary_mobo.csv"
SLURM_SCRIPT="${CODE_DIR}/optimization/run_sai_pipeline.sh"
MANIFEST="${RESULTS_DIR}/configs_manifest_3d.csv"
LOG_DIR="${RESULTS_DIR}/slurm_logs"

# Fixed parameters
A_MM=0.2
B_MM=0.2

# Create log directories
mkdir -p "${LOG_DIR}/out" "${LOG_DIR}/err"

if [ ! -f "${MANIFEST}" ]; then
  echo "ERROR: ${MANIFEST} not found. Run generate_configs_3d.py first."
  exit 1
fi

echo "=========================================="
echo "4D LHS Sweep Submission (MOBO)"
echo "  Design: aperture_diam, n_apertures, n_det_ring1, n_det_ring2"
echo "  Manifest: ${MANIFEST}"
echo "  Pipeline: ${SLURM_SCRIPT}"
echo "  Results:  ${RESULTS_CSV}"
echo "=========================================="

# Read CSV, skip header
n_submitted=0
tail -n +2 "${MANIFEST}" | while IFS=',' read -r idx aperture_diam n_apertures n_det_ring1 n_det_ring2 work_dir; do
  config_name="lhs4d_${idx}_ap${aperture_diam}_nap${n_apertures}_nd1_${n_det_ring1}_nd2_${n_det_ring2}"

  echo "Submitting config ${idx}: d=${aperture_diam} n=${n_apertures} nd1=${n_det_ring1} nd2=${n_det_ring2}"

  job_id=$(sbatch --parsable \
    --output="${LOG_DIR}/out/${config_name}_%j.out" \
    --error="${LOG_DIR}/err/${config_name}_%j.err" \
    --export="ALL,WORK_DIR=${work_dir},APERTURE_DIAM=${aperture_diam},N_APERTURES=${n_apertures},N_DET_RING1=${n_det_ring1},N_DET_RING2=${n_det_ring2},A_MM=${A_MM},B_MM=${B_MM},CODE_DIR=${CODE_DIR},RESULTS_CSV=${RESULTS_CSV},CONFIG_NAME=${config_name}" \
    "${SLURM_SCRIPT}")

  echo "  -> Job ${job_id}"
  n_submitted=$((n_submitted + 1))
done

echo "=========================================="
echo "Submitted jobs. Monitor with: squeue -u \$USER"
echo "Results will be appended to: ${RESULTS_CSV}"
echo "=========================================="
