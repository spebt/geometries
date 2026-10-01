#!/bin/bash
# Replay the design archive under smaller objective subsets.
#
# Read-only against the archive — safe to run while the MOBO controller is live.
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
# MEASURED (job 25615829): 2.8 cores of 4 used. BoTorch GP fitting is mostly
# single-threaded with some BLAS parallelism. 4 is right.
#SBATCH --cpus-per-task=4
# 32G was OOM-killed after 4.5 h. The cost is not the point count, as that first
# estimate assumed — it is qLogNEHVI's hypervolume partition, which grows with
# the size of the Pareto front and the objective count. The script now chunks
# candidate scoring and uses approximate partitioning at m=5, which measured
# 0.83 GB peak on a synthetic archive of the same shape.
#
# MEASURED on the real archive (job 25582343): 585 MB RSS mid-run, against the
# 0.83 GB the synthetic probe predicted. The worry that a 46% Pareto front would
# blow past that estimate was wrong — chunking plus approximate partitioning
# bounds the peak regardless of front size. 128G was requested for that run and
# was a ~200x over-correction after the 32G OOM.
# 16G is ~15x the measured peak: enough margin for a larger archive without
# queueing behind a request nothing needs.
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --job-name=replay_subsets
#SBATCH --output=results/slurm_logs/out/replay_%j.out
#SBATCH --error=results/slurm_logs/err/replay_%j.err

set -euo pipefail

# The venv lives in HOME, not /vscratch. Scratch is auto-purged and in
# Aug 2026 it removed .venv/bin/python mid-campaign. Override with
# SPEBT_VENV if the environment moves again.
source "${SPEBT_VENV:-$HOME/spebt-venv}/bin/activate"
cd /vscratch/grp-rutaoyao/Omer/spebt/spebt/spebt/optimization

mkdir -p results/slurm_logs/out results/slurm_logs/err

python replay_objective_subsets.py \
    --results_csv results/results_summary_mobo.csv \
    --n_repeats 10 \
    --n_steps 80
