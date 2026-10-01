#!/bin/bash
# Reconstruct the configs that have CNR but no saved reconstruction, so they can
# carry the section-mean CNR objective.
#
# RY redefined CNR as the equally-weighted mean over rod-size sections. That is
# computed from a saved reconstruction, and 86 configs from the pre-in-loop era
# never saved one -- they have cnr_mean from the old offline pipeline but no
# recon file. Of those, the ones that also have CNR are usable training points
# we would otherwise drop when switching the objective.
#
# Each task runs one config: forward projection + 150-iteration ML-EM + CNR,
# writing into <work_dir>/cnr_inloop/. recompute_cnr_sectors.py then picks the
# reconstructions up without re-running ML-EM.
#
# Usage:
#   python make_recon_backfill_list.py --results_csv results/results_summary_mobo.csv
#   sbatch --array=0-$(($(wc -l < recon_backfill_configs.txt) - 1)) submit_recon_backfill.sh
#
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --job-name=recon_bf
#SBATCH --output=results/slurm_logs/out/recon_bf_%A_%a.out
#SBATCH --error=results/slurm_logs/err/recon_bf_%A_%a.err

set -euo pipefail

# The venv lives in HOME, not /vscratch. Scratch is auto-purged and in
# Aug 2026 it removed .venv/bin/python mid-campaign. Override with
# SPEBT_VENV if the environment moves again.
VENV_PY="${SPEBT_VENV:-$HOME/spebt-venv}/bin/python3"
CODE_DIR=/vscratch/grp-rutaoyao/Omer/spebt/spebt/spebt
cd "${CODE_DIR}/optimization"

PHANTOM_PATH="${PHANTOM_PATH:-${CODE_DIR}/data/sai_10mm/hot_rods_phantom_10.0_mm_x_10.0_mm.pt}"
CONFIG_LIST="${CONFIG_LIST:-recon_backfill_configs.txt}"

# One work_dir per line
WORK_DIR=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${CONFIG_LIST}")
if [[ -z "${WORK_DIR}" ]]; then
  echo "No entry at line $((SLURM_ARRAY_TASK_ID + 1)) of ${CONFIG_LIST}"
  exit 0
fi
CONFIG_NAME=$(basename "${WORK_DIR}")

echo "[array ${SLURM_ARRAY_TASK_ID}] ${CONFIG_NAME}"

# Write to a scratch CSV: the reconstruction is what we want, and the main CSV
# must not be touched by 15 concurrent tasks doing read-modify-write.
mkdir -p results/recon_backfill
"${VENV_PY}" compute_cnr.py \
    --work_dir "${WORK_DIR}" \
    --phantom_path "${PHANTOM_PATH}" \
    --out_csv "results/recon_backfill/task_${SLURM_ARRAY_TASK_ID}.csv" \
    --config_name "${CONFIG_NAME}" \
    --iterations 150
