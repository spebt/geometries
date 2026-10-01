#!/bin/bash
# Repeat-run CNR measurement for the top designs, to establish the noise floor.
#
# The forward projection applies unseeded Poisson noise, so a single CNR value
# is one draw from a distribution. Two runs of mobo_0069 gave 4.4405 and 4.5688
# (spread 0.128) while the gap between mobo_0069 and mobo_0177 is 0.017 --
# i.e. the ranking among the top designs is not resolvable at n=1.
#
# This array runs N_SEEDS seeded repeats of each top config so we can report
# CNR as mean +/- std instead of a bare point estimate.
#
# Usage:  sbatch submit_cnr_repeats.sh
# Then:   python3 analyze_cnr_repeats.py
#
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
# MEASURED (job 25706256_0): 1.06 cores of 8 used, 10.7 GB of 32.
# ML-EM reconstruction is single-threaded and I/O-bound -- run_mlem re-reads
# all 16 PPDF files inside its 150-iteration loop, so extra cores idle.
# 2 cores leaves headroom for the torch/BLAS threads that do fire.
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=cnr_repeat
#SBATCH --array=0-14
#SBATCH --output=results/slurm_logs/out/cnr_repeat_%A_%a.out
#SBATCH --error=results/slurm_logs/err/cnr_repeat_%A_%a.err

set -euo pipefail

# The venv lives in HOME, not /vscratch. Scratch is auto-purged and in
# Aug 2026 it removed .venv/bin/python mid-campaign. Override with
# SPEBT_VENV if the environment moves again.
source "${SPEBT_VENV:-$HOME/spebt-venv}/bin/activate"
cd /vscratch/grp-rutaoyao/Omer/spebt/spebt/spebt/optimization

PHANTOM_PATH="/vscratch/grp-rutaoyao/Omer/spebt/spebt/spebt/data/sai_10mm/hot_rods_phantom_10.0_mm_x_10.0_mm.pt"

# One CSV per array task: _write_cnr_to_csv does a read-modify-write, so 15
# concurrent tasks sharing one file would silently drop rows. The analysis
# script globs these back together (and can also read the per-seed npz files).
#
# REPEATS_DIR keeps a new measurement out of an old one's files. Task ids
# restart at 0 every submission, so a second array writes task_0.csv on top of
# the first array's task_0.csv. _write_cnr_to_csv adds rows rather than
# truncating, so nothing is lost, but the two runs end up interleaved in one
# directory and analyze_cnr_repeats.py globs them together as if they belonged
# to the same experiment. Give an unrelated measurement its own directory:
#
#     REPEATS_DIR=results/cnr_ring2_recheck CONFIG_LIST=... sbatch ...
REPEATS_DIR="${REPEATS_DIR:-results/cnr_repeats}"
mkdir -p "${REPEATS_DIR}"
OUT_CSV="${REPEATS_DIR}/task_${SLURM_ARRAY_TASK_ID}.csv"

# Configs to repeat. Override with CONFIG_LIST=<file> (one config name per line)
# and size the array to match: --array=0-$((n_configs*N_SEEDS - 1)).
# Default: the top designs from the 180-iteration campaign.
# CONFIG_LIST must live on shared storage, not /tmp: compute nodes each have
# their own /tmp, so a list written on the login node is simply absent when the
# task runs. That failed as "CONFIGS[$cfg_idx]: unbound variable" 60 lines
# later rather than saying the file was missing.
if [[ -n "${CONFIG_LIST:-}" ]]; then
  if [[ ! -r "${CONFIG_LIST}" ]]; then
    echo "ERROR: CONFIG_LIST '${CONFIG_LIST}' is not readable from this node." >&2
    echo "       Put it on shared storage (e.g. alongside this script), not /tmp." >&2
    exit 2
  fi
  mapfile -t CONFIGS < <(grep -v '^\s*$' "${CONFIG_LIST}")
  if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    echo "ERROR: CONFIG_LIST '${CONFIG_LIST}' has no config names in it." >&2
    exit 2
  fi
else
  CONFIGS=(
    "mobo_0069_ap0.3138_nap124_nd1_612_nd2_230"
    "mobo_0177_ap0.3512_nap97_nd1_604_nd2_584"
    "mobo_0173_ap0.3500_nap117_nd1_446_nd2_236"
  )
fi
N_SEEDS="${N_SEEDS:-5}"

cfg_idx=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
seed=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))
if (( cfg_idx >= ${#CONFIGS[@]} )); then
  echo "ERROR: array task ${SLURM_ARRAY_TASK_ID} maps to config index ${cfg_idx}," >&2
  echo "       but only ${#CONFIGS[@]} config(s) were given with N_SEEDS=${N_SEEDS}." >&2
  echo "       Size the array as --array=0-\$(( ${#CONFIGS[@]} * N_SEEDS - 1 ))." >&2
  exit 2
fi
config="${CONFIGS[$cfg_idx]}"

echo "[array ${SLURM_ARRAY_TASK_ID}] config=${config} seed=${seed}"

python3 compute_cnr.py \
    --work_dir "results/${config}" \
    --phantom_path "${PHANTOM_PATH}" \
    --out_csv "${OUT_CSV}" \
    --config_name "${config}__seed${seed}" \
    --out_subdir "cnr_repeat_seed${seed}" \
    --seed "${seed}" \
    --iterations 150
