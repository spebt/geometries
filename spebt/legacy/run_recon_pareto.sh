#!/bin/bash
#SBATCH --job-name=recon_pareto
#SBATCH --cluster=ub-hpc
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=results/slurm_logs/out/recon_%j.out
#SBATCH --error=results/slurm_logs/err/recon_%j.err

set -euo pipefail

# ============================================================
# Run ML-EM reconstruction on top Pareto configs + baseline.
#
# Usage:
#   # Submit with the results CSV:
#   sbatch run_recon_pareto.sh
#
#   # Or run directly:
#   bash run_recon_pareto.sh
# ============================================================

CODE_DIR="/vscratch/grp-rutaoyao/Omer/spebt/spebt"
RESULTS_CSV="${CODE_DIR}/optimization/results/results_summary_mobo.csv"
RECON_SCRIPT="${CODE_DIR}/recon/run_recon_comparison.py"
PHANTOM_PATH="${CODE_DIR}/data/sai_10mm/hot_rods_phantom_10.0_mm_x_10.0_mm.pt"
RECON_OUT_DIR="${CODE_DIR}/optimization/results/recon"

# The venv lives in HOME, not /vscratch. Scratch is auto-purged and in
# Aug 2026 it removed .venv/bin/python mid-campaign. Override with
# SPEBT_VENV if the environment moves again.
source "${SPEBT_VENV:-$HOME/spebt-venv}/bin/activate"
export PYTHONPATH="${CODE_DIR}/pymatcal:${PYTHONPATH:-}"

mkdir -p "${RECON_OUT_DIR}"

echo "=================================================="
echo "Pareto Recon Pipeline | $(date)"
echo "=================================================="

# Step 1: Identify top Pareto configs
echo "[1/3] Identifying top Pareto configs..."
TOP_CONFIGS=$(python3 -c "
import pandas as pd
import numpy as np

df = pd.read_csv('${RESULTS_CSV}')
cols = ['fwhm_mean', 'asci_pct', 'sensitivity_mean', 'mpxi_mean']
df = df.dropna(subset=cols)

# Pareto optimality
obj = df[cols].values.copy()
obj[:, 0] *= -1  # negate FWHM (minimize)
obj[:, 3] *= -1  # negate MPXI (minimize)

n = len(obj)
is_optimal = np.ones(n, dtype=bool)
for i in range(n):
    for j in range(n):
        if i == j: continue
        if np.all(obj[j] >= obj[i]) and np.any(obj[j] > obj[i]):
            is_optimal[i] = False
            break

pareto_df = df[is_optimal]

# Pick top 5 by hypervolume contribution (approx: sort by sum of normalized objectives)
obj_pareto = obj[is_optimal]
mins = obj_pareto.min(axis=0)
maxs = obj_pareto.max(axis=0)
ranges = maxs - mins
ranges[ranges == 0] = 1
normed = (obj_pareto - mins) / ranges
scores = normed.sum(axis=1)
top_idx = np.argsort(scores)[-5:][::-1]

for idx in top_idx:
    row = pareto_df.iloc[idx]
    name_col = 'config_name' if 'config_name' in row.index else 'config'
    print(row[name_col])
")

echo "Top Pareto configs:"
echo "${TOP_CONFIGS}"

# Step 2: Run baseline reconstruction
BASELINE_PPDF_DIR="${CODE_DIR}/data/sai_10mm"
BASELINE_OUT="${RECON_OUT_DIR}/baseline"
if [ ! -f "${BASELINE_OUT}/recon_mlem_T8.npz" ]; then
    echo ""
    echo "[2/3] Reconstructing baseline..."
    python "${RECON_SCRIPT}" \
        --config baseline \
        --ppdf_dir "${BASELINE_PPDF_DIR}" \
        --phantom_path "${PHANTOM_PATH}" \
        --output_dir "${BASELINE_OUT}"
else
    echo "[2/3] Baseline recon already exists, skipping."
fi

# Step 3: Reconstruct each top Pareto config
echo ""
echo "[3/3] Reconstructing top Pareto configs..."
RESULTS_DIR="${CODE_DIR}/optimization/results"

for CONFIG_NAME in ${TOP_CONFIGS}; do
    CONFIG_DIR="${RESULTS_DIR}/${CONFIG_NAME}"
    CONFIG_OUT="${RECON_OUT_DIR}/${CONFIG_NAME}"

    if [ ! -d "${CONFIG_DIR}" ]; then
        echo "  SKIP ${CONFIG_NAME}: work dir not found at ${CONFIG_DIR}"
        continue
    fi

    if [ -f "${CONFIG_OUT}/recon_mlem_T8.npz" ]; then
        echo "  SKIP ${CONFIG_NAME}: recon already exists"
        continue
    fi

    echo ""
    echo "  Reconstructing ${CONFIG_NAME}..."
    python "${RECON_SCRIPT}" \
        --config bo_optimized \
        --ppdf_dir "${CONFIG_DIR}" \
        --phantom_path "${PHANTOM_PATH}" \
        --output_dir "${CONFIG_OUT}" \
        --bo_label "${CONFIG_NAME}"
done

# Step 4: Generate comparison plots (each Pareto vs baseline)
echo ""
echo "[4/4] Generating comparison plots..."
for CONFIG_NAME in ${TOP_CONFIGS}; do
    CONFIG_OUT="${RECON_OUT_DIR}/${CONFIG_NAME}"
    COMPARE_OUT="${RECON_OUT_DIR}/compare_${CONFIG_NAME}"

    if [ ! -f "${CONFIG_OUT}/recon_mlem_T8.npz" ]; then
        echo "  SKIP comparison for ${CONFIG_NAME}: recon not available"
        continue
    fi

    python "${RECON_SCRIPT}" \
        --compare \
        --baseline_dir "${BASELINE_OUT}" \
        --bo_dir "${CONFIG_OUT}" \
        --output_dir "${COMPARE_OUT}" \
        --baseline_label "Baseline (d=0.4, n=180)" \
        --bo_label "${CONFIG_NAME}"
done

echo ""
echo "=================================================="
echo "RECON PIPELINE COMPLETE | $(date)"
echo "All results in: ${RECON_OUT_DIR}"
echo "=================================================="
