#!/bin/bash
#SBATCH --job-name=brain_ppdf
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --mem=32G
#SBATCH --array=0-11
#SBATCH --output=logs/brain_ppdf_%A_%a.out
#SBATCH --error=logs/brain_ppdf_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tridevme@buffalo.edu

set -euo pipefail
mkdir -p logs

SCRIPT_DIR="/user/tridevme/spebt_brain/spebt/pymatcal"
LAYOUT_FILE="/vscratch/grp-rutaoyao/Tridev/brain_spect/geometry/brain_spect_layouts_e90446e80e2b0ae1da75fd55d98c8d37_rot2_ang1p0deg_trans1x1_step0p0x0p0.tensor"
OUTPUT_DIR="/vscratch/grp-rutaoyao/Tridev/brain_spect/data"
PYTHON="/cvmfs/soft.ccr.buffalo.edu/versions/2023.01/easybuild/software/avx512/Compiler/gcccore/11.2.0/python/3.9.6/bin/python"

# Array index 0-11 maps to layout 0-1 x panel 0-5
LAYOUT_IDX=$((SLURM_ARRAY_TASK_ID / 6))
PANEL_IDX=$((SLURM_ARRAY_TASK_ID % 6))

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

echo "Array=${SLURM_ARRAY_TASK_ID} Layout=${LAYOUT_IDX} Panel=${PANEL_IDX}"
$PYTHON "${SCRIPT_DIR}/arg_ppdf_brain.py" \
    ${LAYOUT_IDX} \
    --layout_file "${LAYOUT_FILE}" \
    --output_dir  "${OUTPUT_DIR}" \
    --fov_npix 200 200 \
    --fov_size_mm 200.0 200.0 \
    --sfov_subdiv 2 2 \
    --crystal_subs 1 1 \
    --single_panel ${PANEL_IDX}

echo "Done layout=${LAYOUT_IDX} panel=${PANEL_IDX}"
