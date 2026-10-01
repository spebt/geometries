#!/bin/bash
#SBATCH --job-name=recon_compare
#SBATCH --cluster=ub-hpc
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/vscratch/grp-rutaoyao/Omer/spebt/spebt/recon/slurm_logs/recon_compare_%j.out
#SBATCH --error=/vscratch/grp-rutaoyao/Omer/spebt/spebt/recon/slurm_logs/recon_compare_%j.err
#SBATCH --mail-user=syedomer@buffalo.edu
# Mail on FAILURE only, never on END. A campaign submits one of these per
# iteration, so ~480 across six arms, and an inbox full of successes is an inbox
# nobody reads. Failures are rare and worth seeing: since Aug 2026 this script
# exits 1 when the geometry generator CRASHES (as opposed to legitimately
# rejecting an infeasible design, which exits 0 and is data), so a FAIL mail
# means the toolchain is broken.
#SBATCH --mail-type=FAIL,TIMEOUT

# ============================================================
# Reconstruction comparison: Baseline vs BO-Optimized vs LHS_16
#
# Noise: Harsh's approach (physical emission rates + Poisson)
# ============================================================

set -euo pipefail

# The venv lives in HOME, not /vscratch. Scratch is auto-purged and in
# Aug 2026 it removed .venv/bin/python mid-campaign. Override with
# SPEBT_VENV if the environment moves again.
source "${SPEBT_VENV:-$HOME/spebt-venv}/bin/activate"

CODE_DIR="/vscratch/grp-rutaoyao/Omer/spebt/spebt"
RECON_SCRIPT="${CODE_DIR}/recon/run_recon_comparison.py"
RESULTS_BASE="${CODE_DIR}/recon/recon_results"
PHANTOM="/vscratch/grp-rutaoyao/Omer/spebt/spebt/data/sai_10mm/hot_rods_phantom_10.0_mm_x_10.0_mm.pt"

# PPDF directories
BASELINE_PPDF="/vscratch/grp-rutaoyao/Omer/spebt/data/sai_10mm"
BO_PPDF="/vscratch/grp-rutaoyao/Omer/spebt/spebt/optimization/results/bo_0013_ap0.5300_nap232"
LHS16_PPDF="/vscratch/grp-rutaoyao/Omer/spebt/optimization/config_0016_ap0.3496_nap260"

# Physical noise parameters (same as Harsh's fake_projection_v3.py)
T_SEC=10
E_HOT=10
E_BG=2

mkdir -p "${RESULTS_BASE}"

echo "=================================================="
echo "Reconstruction Comparison (Harsh noise model) | $(date)"
echo "  T=${T_SEC}s  e_hot=${E_HOT}  e_bg=${E_BG}"
echo "=================================================="

# Step 1: Baseline
echo ""
echo "=== BASELINE (d=0.4mm, n=180) ==="
python "${RECON_SCRIPT}" \
  --config baseline \
  --ppdf_dir "${BASELINE_PPDF}" \
  --phantom_path "${PHANTOM}" \
  --T_sec "${T_SEC}" --e_hot "${E_HOT}" --e_bg "${E_BG}" \
  --output_dir "${RESULTS_BASE}/baseline"

# Step 2: BO-optimized
echo ""
echo "=== BO-OPTIMIZED (d=0.53mm, n=232) ==="
python "${RECON_SCRIPT}" \
  --config bo_optimized \
  --ppdf_dir "${BO_PPDF}" \
  --phantom_path "${PHANTOM}" \
  --T_sec "${T_SEC}" --e_hot "${E_HOT}" --e_bg "${E_BG}" \
  --output_dir "${RESULTS_BASE}/bo_optimized"

# Step 3: LHS_16
echo ""
echo "=== LHS_16 (d=0.35mm, n=260) ==="
python "${RECON_SCRIPT}" \
  --config bo_optimized \
  --ppdf_dir "${LHS16_PPDF}" \
  --phantom_path "${PHANTOM}" \
  --T_sec "${T_SEC}" --e_hot "${E_HOT}" --e_bg "${E_BG}" \
  --output_dir "${RESULTS_BASE}/lhs_16"

# Step 4: Comparison — Baseline vs BO
echo ""
echo "=== COMPARISON: Baseline vs BO ==="
python "${RECON_SCRIPT}" \
  --compare \
  --baseline_dir "${RESULTS_BASE}/baseline" \
  --bo_dir "${RESULTS_BASE}/bo_optimized" \
  --baseline_label "Baseline (d=0.4, n=180)" \
  --bo_label "BO-Optimized (d=0.53, n=232)" \
  --output_dir "${RESULTS_BASE}/comparison_bo"

# Step 5: Comparison — Baseline vs LHS_16
echo ""
echo "=== COMPARISON: Baseline vs LHS_16 ==="
python "${RECON_SCRIPT}" \
  --compare \
  --baseline_dir "${RESULTS_BASE}/baseline" \
  --bo_dir "${RESULTS_BASE}/lhs_16" \
  --baseline_label "Baseline (d=0.4, n=180)" \
  --bo_label "LHS_16 (d=0.35, n=260)" \
  --output_dir "${RESULTS_BASE}/comparison_lhs16"

echo ""
echo "=================================================="
echo "ALL DONE | $(date)"
echo "=================================================="
