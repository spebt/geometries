#!/usr/bin/env python3
"""
Sequential BO Controller for SAI SC-SPECT.

Adapted from Kirtiraj's 12_run_bo_loop_checkpointed.py.
  - Sequential q=1: propose one config, evaluate on HPC, collect JI, repeat.
  - Resumes from existing manifest (checkpointed).
  - Design vector: (aperture_diam_mm, n_apertures).

Usage:
  python run_bo_loop.py                   # run with defaults
  python run_bo_loop.py --max_iters 50    # custom iteration count
"""
import os
import sys
import time
import subprocess
import argparse
import pandas as pd

from bo_agent import get_next_candidate

from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.table import Table
from rich.panel import Panel

# =========================
# PATHS (edit for your HPC setup)
# =========================
CODE_DIR = "/vscratch/grp-rutaoyao/Omer/spebt/spebt"
RESULTS_DIR = os.path.join(CODE_DIR, "optimization", "results")
MANIFEST_FILE = os.path.join(RESULTS_DIR, "experiment_manifest.csv")
RESULTS_CSV = os.path.join(RESULTS_DIR, "results_summary_4d.csv")
SLURM_SCRIPT = os.path.join(CODE_DIR, "optimization", "run_sai_pipeline.sh")
LOG_DIR = os.path.join(RESULTS_DIR, "slurm_logs")

console = Console()


def ensure_dirs():
    """Create output directories."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(LOG_DIR, "out"), exist_ok=True)
    os.makedirs(os.path.join(LOG_DIR, "err"), exist_ok=True)


def ensure_manifest_header():
    """Create manifest CSV if it doesn't exist."""
    if not os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "w") as f:
            f.write("idx,config_name,aperture_diam_mm,n_apertures,scint_radial_thickness_mm,ring_thickness_mm,work_dir,job_id,status\n")


def get_next_manifest_index():
    """Get next available index from manifest."""
    if not os.path.exists(MANIFEST_FILE):
        return 0
    with open(MANIFEST_FILE, "r") as f:
        lines = [ln for ln in f.readlines() if ln.strip()]
    return max(0, len(lines) - 1)


def append_manifest_row(idx, config_name, diam, n_ap, scint_rad, ring_thick, work_dir, job_id, status="submitted"):
    """Append a row to the experiment manifest."""
    with open(MANIFEST_FILE, "a") as f:
        f.write(f"{idx},{config_name},{diam:.6f},{n_ap},{scint_rad:.4f},{ring_thick:.4f},{work_dir},{job_id},{status}\n")


def patch_manifest_status(idx, job_id, status):
    """Update the status and job_id of the last manifest row."""
    with open(MANIFEST_FILE, "r") as f:
        lines = f.readlines()
    last_i = max(i for i, ln in enumerate(lines) if ln.strip())
    parts = lines[last_i].rstrip("\n").split(",")
    if len(parts) >= 9:
        parts[7] = str(job_id)
        parts[8] = status
        lines[last_i] = ",".join(parts) + "\n"
        with open(MANIFEST_FILE, "w") as f:
            f.writelines(lines)


def is_job_running(job_id: str) -> bool:
    """Check if a SLURM job is still running."""
    try:
        r = subprocess.run(["squeue", "--job", str(job_id)], capture_output=True, text=True)
        return str(job_id) in r.stdout
    except Exception:
        return True  # assume running if squeue fails


def assert_initial_data():
    """Check that initial LHS data exists in results CSV."""
    if not os.path.exists(RESULTS_CSV):
        console.print("[bold red]ERROR:[/bold red] results_summary.csv not found.")
        console.print(f"  Expected: {RESULTS_CSV}")
        console.print("  Run the initial LHS sweep first, then re-run this script.")
        sys.exit(1)
    df = pd.read_csv(RESULTS_CSV)
    df = df.dropna(subset=["JI"])
    df = df[df["JI"] > 1e-9]
    n = len(df)
    if n < 5:
        console.print(f"[bold yellow]WARNING:[/bold yellow] Only {n} valid data points. GP may be unstable.")
    else:
        console.print(f"[green]Loaded {n} existing data points.[/green]")
    return n


def print_status(idx, config_name):
    """Print current optimization status."""
    try:
        df = pd.read_csv(RESULTS_CSV)
        df = df.dropna(subset=["JI"])
        df = df[df["JI"] > 1e-9]
        best_row = df.loc[df["JI"].idxmax()]
        best_cfg = best_row.get("config", "N/A")
        best_ji = float(best_row["JI"])
        best_diam = float(best_row.get("aperture_diam_mm", 0))
        best_nap = int(best_row.get("n_apertures", 0))
        best_sr = float(best_row.get("scint_radial_thickness_mm", 6.0))
        best_rt = float(best_row.get("ring_thickness_mm", 2.5))
    except Exception:
        best_cfg, best_ji, best_diam, best_nap, best_sr, best_rt = "N/A", 0.0, 0, 0, 6.0, 2.5

    t = Table(title="Optimization Status")
    t.add_column("Iteration", style="dim")
    t.add_column("Latest Config", style="cyan")
    t.add_column("Best Config", style="green")
    t.add_column("Best JI", style="magenta")
    t.add_column("Best Params", style="yellow")
    t.add_row(
        str(idx), config_name, str(best_cfg),
        f"{best_ji:.6e}", f"d={best_diam:.3f} n={best_nap} sr={best_sr:.2f} rt={best_rt:.2f}"
    )
    console.print(t)


def main():
    parser = argparse.ArgumentParser(description="Sequential BO loop for SAI SC-SPECT")
    parser.add_argument("--max_iters", type=int, default=100,
                        help="Total BO iterations (default: 100)")
    args = parser.parse_args()

    TOTAL_ITERATIONS = args.max_iters

    console.print(Panel.fit(
        "[bold green]SAI SC-SPECT BO Controller (4D)[/bold green]\n"
        "Sequential q=1 | Design: (aperture_diam, n_apertures, scint_radial, ring_thickness)",
        subtitle=f"Max iterations: {TOTAL_ITERATIONS}"
    ))

    ensure_dirs()
    ensure_manifest_header()
    n_initial = assert_initial_data()

    start_idx = get_next_manifest_index()
    if start_idx >= TOTAL_ITERATIONS:
        console.print(f"[green]Done.[/green] Already have {start_idx} entries (>= {TOTAL_ITERATIONS}).")
        return

    n_to_run = TOTAL_ITERATIONS - start_idx

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:

        task = progress.add_task("[green]BO Loop", total=n_to_run)

        for _ in range(n_to_run):
            idx = get_next_manifest_index()

            console.print(f"\n[bold yellow]=== BO Iteration {idx} ===[/bold yellow]")

            # 1. Ask BO agent for next candidate
            console.log("Calling bo_agent.get_next_candidate()...")
            try:
                diam, n_ap, scint_rad, ring_thick = get_next_candidate(RESULTS_CSV)
            except Exception as e:
                console.print(f"[bold red]BO Agent Failed:[/bold red] {e}")
                break

            console.log(f"[cyan]Candidate:[/cyan] aperture_diam={diam:.4f}mm  n_apertures={n_ap}  "
                         f"scint_radial={scint_rad:.4f}mm  ring_thickness={ring_thick:.4f}mm")

            # 2. Create work directory
            config_name = f"bo_{idx:04d}_ap{diam:.4f}_nap{n_ap}_sr{scint_rad:.2f}_rt{ring_thick:.2f}"
            work_dir = os.path.join(RESULTS_DIR, config_name)
            os.makedirs(work_dir, exist_ok=True)

            # 3. Append to manifest (before submission)
            append_manifest_row(idx, config_name, diam, n_ap, scint_rad, ring_thick, work_dir, "", "pending")

            # 4. Submit SLURM job
            if not os.path.exists(SLURM_SCRIPT):
                console.print(f"[bold red]Missing SLURM script:[/bold red] {SLURM_SCRIPT}")
                break

            env_vars = (
                f"ALL,"
                f"WORK_DIR={work_dir},"
                f"APERTURE_DIAM={diam},"
                f"N_APERTURES={n_ap},"
                f"SCINT_RADIAL_MM={scint_rad},"
                f"RING_THICKNESS_MM={ring_thick},"
                f"A_MM=0.2,B_MM=0.2,"
                f"CODE_DIR={CODE_DIR},"
                f"RESULTS_CSV={RESULTS_CSV},"
                f"CONFIG_NAME={config_name}"
            )

            sbatch_cmd = [
                "sbatch",
                "--parsable",
                f"--output={LOG_DIR}/out/{config_name}_%j.out",
                f"--error={LOG_DIR}/err/{config_name}_%j.err",
                "--export", env_vars,
                SLURM_SCRIPT,
            ]

            try:
                job_id = subprocess.check_output(sbatch_cmd, text=True).strip().split(";")[0]
                console.log(f"[bold green]Submitted job:[/bold green] {job_id}")
            except Exception as e:
                console.print(f"[bold red]SLURM Submission Failed:[/bold red] {e}")
                patch_manifest_status(idx, "", "failed")
                break

            patch_manifest_status(idx, job_id, "running")

            # 5. Wait for job to finish
            console.log(f"Waiting for job {job_id}...")
            wait_start = time.time()
            last_print = 0

            while is_job_running(job_id):
                time.sleep(60)
                elapsed_min = int((time.time() - wait_start) / 60)
                if elapsed_min >= last_print + 10:
                    console.log(f"[yellow]Job {job_id} still running... ({elapsed_min} min)[/yellow]")
                    last_print = elapsed_min

            elapsed_total = int((time.time() - wait_start) / 60)
            console.log(f"Job {job_id} finished in {elapsed_total} min.")
            patch_manifest_status(idx, job_id, "completed")

            # 6. Print status
            print_status(idx, config_name)
            progress.advance(task)

    console.print("\n[bold green]BO Loop finished.[/bold green]")
    print_status(get_next_manifest_index() - 1, "final")


if __name__ == "__main__":
    main()
