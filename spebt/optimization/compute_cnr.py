#!/usr/bin/env python3
"""
Compute reconstruction CNR for a single SAI SC-SPECT configuration.

Wraps the ML-EM reconstruction pipeline (forward projection -> ML-EM ->
CNR) into a single CLI entry point that the SLURM job (run_sai_pipeline.sh)
calls AFTER compute_metrics.py, so each MOBO evaluation contributes a CNR
value the optimizer can target.

Adds the cnr_mean column to the results CSV in place. If the column does
not exist yet, it is created. If the row for --config_name does not exist
yet (e.g. compute_metrics.py was not run first), a minimal row is appended.

Usage:
  python compute_cnr.py \
      --work_dir <path-to-config-work-dir> \
      --phantom_path <path-to-hot_rods_phantom.pt> \
      --out_csv results/results_summary_mobo.csv \
      --config_name mobo_0099_ap... \
      [--iterations 150]

Design notes:
  - 150 ML-EM iterations is the default. CNR ranking is stable well before
    convergence; 500 iter is only needed for clean final plots.
  - Recon artefacts (projections, recon, CNR npz) are written to
    <work_dir>/cnr_inloop/ so the per-config folder remains self-contained.
  - The function returns NaN if PPDF files are missing rather than raising,
    so an infeasible config does not crash the SLURM job.
"""
import argparse
import hashlib
import os
import sys
import time
import numpy as np
import pandas as pd
import torch

# Reuse the recon implementation that already exists for offline comparison
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..", "recon")))
import run_recon_comparison as rrc  # noqa: E402


def seed_from_config_name(config_name: str) -> int:
    """Stable per-config noise seed derived from the config name.

    Python's built-in hash() is salted per process (PYTHONHASHSEED), so it
    would give a different seed on every SLURM task and defeat the point.
    sha256 is stable across processes, machines and Python versions.

    Distinct configs get effectively independent draws, so the GP still sees
    honest across-design observation noise; re-running one config reproduces
    its value exactly. Repeat measurements of a single design should still pass
    explicit --seed 0..4 rather than relying on this.
    """
    digest = hashlib.sha256(config_name.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def compute_cnr_for_work_dir(
    work_dir: str,
    phantom_path: str,
    iterations: int = 150,
    T_sec: float = 10.0,
    e_hot: float = 10.0,
    e_bg: float = 2.0,
    out_subdir: str = "cnr_inloop",
    seed: int = None,
) -> float:
    """Run forward project -> ML-EM -> CNR for one work_dir, return overall CNR.

    Returns NaN if PPDF files are missing (e.g. infeasible config).

    The forward projection applies Poisson noise (rrc.forward_project ->
    torch.poisson), so CNR is a random variable: repeat runs of the same
    design differ by ~0.1 CNR. Pass `seed` to fix the noise realisation and
    make a run reproducible. The pipeline passes --seed_from_config, which
    derives a stable seed from the config name: different designs still get
    independent draws, so the GP sees honest across-design observation noise,
    but a config re-run reproduces its own value. Leaving `seed` None gives a
    fresh irreproducible draw, which is what the archive rows written before
    Sep 2026 contain.
    """
    out_dir = os.path.join(work_dir, out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    if seed is not None:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed) % (2 ** 32))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        flist_path, flist = rrc.generate_flist(work_dir, out_dir)
    except FileNotFoundError as e:
        print(f"  [cnr] no PPDF files in {work_dir}: {e}")
        return float("nan"), float("nan"), np.array([])
    if not flist:
        # A NaN with no explanation is what made the Aug 2026 failures take days
        # to find: the archive gained an empty row and nothing said why.
        print(f"  [cnr] file list is empty for {work_dir}; no PPDFs to project. "
              f"Returning NaN.")
        return float("nan"), float("nan"), np.array([])

    projs_path = rrc.forward_project(
        flist, phantom_path, out_dir, device,
        T_sec=T_sec, e_hot=e_hot, e_bg=e_bg,
    )

    # Temporarily override ML-EM iteration count for fast in-loop runs
    saved_iter = rrc.N_ITERATIONS
    rrc.N_ITERATIONS = int(iterations)
    try:
        recon_path = rrc.run_mlem(flist, projs_path, out_dir, device)
    finally:
        rrc.N_ITERATIONS = saved_iter

    results = rrc.compute_cnr(recon_path, phantom_path, out_dir)
    sectors = np.asarray(results["sector_cnrs"], dtype=float)
    # RY (Jul 2026): weight every rod size equally, since the CNR at a size
    # represents imaging performance for that object size. overall_cnr pools all
    # hot pixels and is therefore area-weighted, letting the largest rods
    # dominate. Both are returned so the older column stays reproducible.
    sector_mean = float(np.nanmean(sectors)) if sectors.size else float("nan")
    return float(results["overall_cnr"]), sector_mean, sectors


def _write_cnr_to_csv(out_csv: str, config_name: str, cnr: float,
                      sector_mean: float = float("nan"),
                      sectors=None) -> None:
    """Set the CNR columns for the row matching config_name.

    cnr_sector_mean is the objective the optimizer reads; cnr_mean is kept so
    the pre-Jul-2026 numbers stay comparable. Creates rows or columns as needed.
    """
    vals = {
        "cnr_mean": cnr,
        "cnr_sector_mean": sector_mean,
    }
    if sectors is not None:
        for i, v in enumerate(np.asarray(sectors, dtype=float)):
            vals[f"cnr_sector{i}"] = float(v)

    if not os.path.exists(out_csv):
        pd.DataFrame([{"config": config_name, **vals}]).to_csv(out_csv, index=False)
        return

    df = pd.read_csv(out_csv)
    for col in vals:
        if col not in df.columns:
            df[col] = float("nan")

    if "config" in df.columns and (df["config"] == config_name).any():
        for col, v in vals.items():
            df.loc[df["config"] == config_name, col] = v
    else:
        new_row = {col: np.nan for col in df.columns}
        new_row["config"] = config_name
        new_row.update(vals)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(out_csv, index=False)


def main():
    parser = argparse.ArgumentParser(description="In-loop CNR computation for SAI SC-SPECT")
    parser.add_argument("--work_dir", required=True,
                        help="Per-config directory containing position_*_ppdfs_t8_*.hdf5 files")
    parser.add_argument("--phantom_path", required=True,
                        help="Hot-rod phantom .pt file used to simulate projections")
    parser.add_argument("--out_csv", required=True,
                        help="Metrics CSV to update with cnr_mean (created if missing)")
    parser.add_argument("--config_name", required=True,
                        help="Config identifier (must match the row written by compute_metrics.py)")
    parser.add_argument("--iterations", type=int, default=150,
                        help="ML-EM iteration count for in-loop recon (default: 150)")
    parser.add_argument("--T_sec", type=float, default=10.0,
                        help="Scan time in seconds for forward projection")
    parser.add_argument("--e_hot", type=float, default=10.0,
                        help="Emission rate per hot voxel (counts/sec)")
    parser.add_argument("--e_bg", type=float, default=2.0,
                        help="Emission rate per background voxel (counts/sec)")
    parser.add_argument("--force_nan", action="store_true",
                        help="Write NaN row (for infeasible configs flagged upstream)")
    parser.add_argument("--reason", type=str, default="")
    parser.add_argument("--out_subdir", type=str, default="cnr_inloop",
                        help="Subdirectory of --work_dir for recon artefacts "
                             "(default: cnr_inloop). Use a distinct value for "
                             "repeat runs so they do not overwrite each other.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed the Poisson noise in forward projection. "
                             "Takes precedence over --seed_from_config.")
    parser.add_argument("--seed_from_config", action="store_true",
                        help="Derive the noise seed from --config_name so the run "
                             "is reproducible while different designs still get "
                             "independent draws. This is what the pipeline uses.")
    args = parser.parse_args()

    if args.force_nan:
        print(f"[{args.config_name}] FORCE_NAN: {args.reason}")
        _write_cnr_to_csv(args.out_csv, args.config_name, float("nan"))
        return

    seed = args.seed
    if seed is None and args.seed_from_config:
        seed = seed_from_config_name(args.config_name)
        print(f"[{args.config_name}] noise seed {seed} (derived from config name)")

    t0 = time.time()
    cnr, sector_mean, sectors = compute_cnr_for_work_dir(
        args.work_dir, args.phantom_path,
        iterations=args.iterations,
        T_sec=args.T_sec, e_hot=args.e_hot, e_bg=args.e_bg,
        out_subdir=args.out_subdir, seed=seed,
    )
    elapsed = time.time() - t0

    if np.isnan(cnr):
        print(f"[{args.config_name}] CNR = NaN ({elapsed:.1f} s)")
    else:
        print(f"[{args.config_name}] CNR sector-mean = {sector_mean:.4f}  "
              f"(pooled {cnr:.4f})  ({elapsed:.1f} s, {args.iterations} ML-EM iter)")

    _write_cnr_to_csv(args.out_csv, args.config_name, cnr, sector_mean, sectors)


if __name__ == "__main__":
    main()
