#!/bin/bash
# Submit all LHS configs from configs_manifest.csv as SLURM jobs.
#
# Usage:
#   cd /vscratch/grp-rutaoyao/Omer/spebt/spebt/optimization
#   bash submit_lhs_sweep.sh
#
# Reads: results/configs_manifest.csv
# Submits: one sbatch job per row using run_sai_pipeline.sh

set -euo pipefail

CODE_DIR="/vscratch/grp-rutaoyao/Omer/spebt/spebt"
RESULTS_DIR="${CODE_DIR}/optimization/results"
RESULTS_CSV="${RESULTS_DIR}/results_summary.csv"
SLURM_SCRIPT="${CODE_DIR}/optimization/run_sai_pipeline.sh"
MANIFEST="${RESULTS_DIR}/configs_manifest.csv"
LOG_DIR="${RESULTS_DIR}/slurm_logs"

# Fixed T8 parameters
A_MM=0.2
B_MM=0.2

# Create log directories
mkdir -p "${LOG_DIR}/out" "${LOG_DIR}/err"

if [ ! -f "${MANIFEST}" ]; then
  echo "ERROR: ${MANIFEST} not found. Run generate_configs.py first."
  exit 1
fi

echo "=========================================="
echo "LHS Sweep Submission"
echo "  Manifest: ${MANIFEST}"
echo "  Pipeline: ${SLURM_SCRIPT}"
echo "  Results:  ${RESULTS_CSV}"
echo "=========================================="

# Read CSV, skip header
n_submitted=0
tail -n +2 "${MANIFEST}" | while IFS=',' read -r idx aperture_diam n_apertures scint_radial ring_thickness work_dir; do
  config_name="lhs_${idx}_ap${aperture_diam}_nap${n_apertures}_sr${scint_radial}_rt${ring_thickness}"

  echo "Submitting config ${idx}: aperture_diam=${aperture_diam} n_apertures=${n_apertures} scint_radial=${scint_radial} ring_thickness=${ring_thickness}"

  job_id=$(sbatch --parsable \
    --output="${LOG_DIR}/out/${config_name}_%j.out" \
    --error="${LOG_DIR}/err/${config_name}_%j.err" \
    --export="ALL,WORK_DIR=${work_dir},APERTURE_DIAM=${aperture_diam},N_APERTURES=${n_apertures},SCINT_RADIAL_MM=${scint_radial},RING_THICKNESS_MM=${ring_thickness},A_MM=${A_MM},B_MM=${B_MM},CODE_DIR=${CODE_DIR},RESULTS_CSV=${RESULTS_CSV},CONFIG_NAME=${config_name}" \
    "${SLURM_SCRIPT}")

  echo "  -> Job ${job_id}"
  n_submitted=$((n_submitted + 1))
done

echo "=========================================="
echo "Submitted ${n_submitted} jobs. Monitor with: squeue -u \$USER"
echo "Results will be appended to: ${RESULTS_CSV}"
echo "=========================================="
