#!/usr/bin/env python3
"""
Sequential MOBO Controller for SAI SC-SPECT.

Multi-objective BO loop: propose one config, evaluate on HPC, collect metrics, repeat.
CNR is produced in-loop by compute_cnr.py step [4/4] of run_sai_pipeline.sh.
Uses mobo_agent (ModelListGP + qLogNEHVI).

The objective set lives in mobo_agent.OBJ_COLUMNS / OBJ_DIRECTIONS and is
deliberately NOT restated here. This docstring used to enumerate it and drifted
badly: it still named sensitivity, retired in Jul 2026, and MPXI as minimized,
reversed in Aug 2026. A stale list in a docstring is read as documentation and
believed.

Usage:
  python run_mobo_loop.py                   # run with defaults
  python run_mobo_loop.py --max_iters 50    # custom iteration count
"""
import os
import sys
import time
import errno
import fcntl
import socket
import subprocess
import argparse
import pandas as pd

from mobo_agent import get_next_candidate

from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.table import Table
from rich.panel import Panel

# =========================
# PATHS (overridable via env; defaults to the current 3-spebt repo layout
# on CCR: /vscratch/grp-rutaoyao/Omer/spebt/spebt/spebt/)
# =========================
CODE_DIR = os.environ.get(
    "MOBO_CODE_DIR",
    "/vscratch/grp-rutaoyao/Omer/spebt/spebt/spebt",
)
RESULTS_DIR = os.environ.get(
    "MOBO_RESULTS_DIR",
    os.path.join(CODE_DIR, "optimization", "results"),
)
MANIFEST_FILE = os.path.join(RESULTS_DIR, "mobo_manifest.csv")
LOCK_FILE = os.path.join(RESULTS_DIR, ".mobo_loop.lock")
# Holds the flock fd for the process lifetime. See acquire_singleton_lock().
_LOCK_FD = None
RESULTS_CSV = os.path.join(RESULTS_DIR, "results_summary_mobo.csv")
SLURM_SCRIPT = os.path.join(CODE_DIR, "optimization", "run_sai_pipeline.sh")
LOG_DIR = os.path.join(RESULTS_DIR, "slurm_logs")

# Imported, not restated. This was a hand-synced copy; a duplicated definition
# that drifted from mobo_agent's is what produced 46 wasted iterations in Jul
# 2026, so the objective set now has exactly one home.
import mobo_agent
from mobo_agent import OBJ_COLUMNS, OBJ_DIRECTIONS, OBJ_NAMES, OBJ_SHORT

# Rich defaults to 80 columns when stdout is not a terminal, which silently
# truncated the candidate line and hid the d2/d3 values in the SLURM logs.
console = Console(width=int(os.environ.get("MOBO_LOG_WIDTH", "160")))


def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(LOG_DIR, "out"), exist_ok=True)
    os.makedirs(os.path.join(LOG_DIR, "err"), exist_ok=True)


def acquire_singleton_lock():
    """Refuse to start if another controller already owns the manifest.

    Nothing in the loop is safe to run twice concurrently: get_next_manifest_index()
    reads the file with no reservation, so two controllers claim the SAME index,
    append duplicate rows, and submit two pipelines writing to one config
    directory. Re-submitting submit_mobo.sh while a controller is alive is easy
    to do by accident -- the script is documented as safe to re-submit, and it is,
    but only once the previous one has exited.

    flock is released automatically when the process dies, so a killed or
    OOM-ed controller leaves no stale lock to clear by hand.

    Two details that look like style but are not:

    - The fd is parked in a module global, not returned for the caller to hold.
      A returned file object whose value is discarded is garbage-collected
      immediately, which closes the fd and drops the lock -- the guard would
      then pass its own tests while protecting nothing.
    - The file is opened without O_TRUNC and truncated only AFTER the lock is
      won. Opening "w" truncates before flock is attempted, so a refused
      process would erase the running controller's identity on its way to
      reporting it.
    """
    global _LOCK_FD
    if _LOCK_FD is not None:
        return
    fd = os.open(LOCK_FILE, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        os.close(fd)
        if e.errno not in (errno.EACCES, errno.EAGAIN):
            raise
        try:
            with open(LOCK_FILE) as fh:
                holder = fh.read().strip() or "unknown"
        except OSError:
            holder = "unknown"
        console.print(
            f"[bold red]Another MOBO controller is already running[/bold red] "
            f"({holder}).\nRefusing to start: two controllers would claim the same "
            f"manifest index and collide on config directories.\n"
            f"Cancel this job, or wait for the running one to finish."
        )
        sys.exit(1)
    os.ftruncate(fd, 0)
    os.write(fd, f"job={os.environ.get('SLURM_JOB_ID', 'local')} "
                 f"host={socket.gethostname()} pid={os.getpid()}\n".encode())
    _LOCK_FD = fd


# Each evaluated design leaves ~8.9 GB of simulation output behind, against
# ~77 KB of metrics that actually get used. On Aug 26 2026 /vscratch reached
# 100% and every running campaign died in a way that looked like a code bug:
# pipeline jobs failed with no logs because SLURM could not create their output
# files, and controllers died with no traceback because stderr could not be
# written either. Checking before each iteration turns that into a clear message.
MIN_FREE_GB = 60.0


def free_gb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def check_disk_space(where="before iteration"):
    """Warn, and refuse to start a new iteration, when the disk is nearly full."""
    try:
        avail = free_gb(RESULTS_DIR)
    except OSError as e:
        console.print(f"[yellow]Could not check free space: {e}[/yellow]")
        return True
    if avail < MIN_FREE_GB:
        console.print(
            f"[bold red]Only {avail:.0f} GB free on the results filesystem "
            f"({where}).[/bold red]\n"
            f"Each design needs roughly 9 GB, so this iteration would likely "
            f"fail\nhalfway and take the controller with it. Stopping cleanly "
            f"instead.\n"
            f"Free space with prune_ppdfs.py, then resubmit; the manifest makes "
            f"this\nsafe to resume."
        )
        return False
    if avail < MIN_FREE_GB * 4:
        console.print(f"[yellow]{avail:.0f} GB free; roughly "
                      f"{int(avail // 9)} more designs will fit.[/yellow]")
    return True


def ensure_manifest_header():
    if not os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "w") as f:
            f.write("idx,config_name,aperture_diam_mm,n_apertures,"
                    "n_det_ring1,n_det_ring2,d2_inner_mm,d3_inner_mm,"
                    "work_dir,job_id,status\n")


def get_next_manifest_index():
    if not os.path.exists(MANIFEST_FILE):
        return 0
    with open(MANIFEST_FILE, "r") as f:
        lines = [ln for ln in f.readlines() if ln.strip()]
    return max(0, len(lines) - 1)


def append_manifest_row(idx, config_name, diam, n_ap, n_det1, n_det2,
                        d2_inner, d3_inner, work_dir, job_id, status="submitted"):
    with open(MANIFEST_FILE, "a") as f:
        f.write(f"{idx},{config_name},{diam:.6f},{n_ap},"
                f"{n_det1},{n_det2},{d2_inner:.3f},{d3_inner:.3f},"
                f"{work_dir},{job_id},{status}\n")


def patch_manifest_status(idx, job_id, status):
    """Update the manifest row whose first column equals `idx`.

    Previously this patched whichever row was last, which happened to be correct
    only because the caller always appends immediately before patching. Anchoring
    on `idx` makes the function robust to controller restarts, orphan rows, or
    any future concurrent update.
    """
    with open(MANIFEST_FILE, "r") as f:
        lines = f.readlines()
    target = str(idx)
    target_i = None
    # Iterate in reverse: if multiple rows share the same idx (should not happen,
    # but see previous stuck-row bugs), the most recent one wins.
    for i in range(len(lines) - 1, -1, -1):
        ln = lines[i].strip()
        if not ln or ln.startswith("idx,"):
            continue
        if ln.split(",", 1)[0] == target:
            target_i = i
            break
    if target_i is None:
        # Fallback: patch the last non-empty row (legacy behaviour) and warn
        console.print(f"[yellow]patch_manifest_status: no row with idx={idx}, "
                      f"falling back to last row[/yellow]")
        target_i = max(i for i, ln in enumerate(lines) if ln.strip())

    parts = lines[target_i].rstrip("\n").split(",")
    # job_id and status are the final two columns in every manifest schema. Use
    # negative indices rather than fixed positions so that rows written before
    # d2_inner_mm/d3_inner_mm were added (9 columns) are still patched correctly
    # alongside new 11-column rows.
    if len(parts) >= 9:
        parts[-2] = str(job_id)
        parts[-1] = status
        lines[target_i] = ",".join(parts) + "\n"
        with open(MANIFEST_FILE, "w") as f:
            f.writelines(lines)


def is_job_running(job_id: str) -> bool:
    """Return True if the given SLURM job id is still in the queue.

    Uses a word-boundary check so that job 12345 does not match against 123456
    when substring-scanning squeue output. squeue itself already filters by
    `--job`, so under normal operation r.stdout contains only that job's line;
    the word-boundary check just makes the failure mode explicit.
    """
    import re
    try:
        r = subprocess.run(
            ["squeue", "--job", str(job_id), "--noheader", "-o", "%i"],
            capture_output=True, text=True,
        )
        # Each line of stdout is a plain job id; match on word boundaries
        pattern = rf"(?<!\d){re.escape(str(job_id))}(?!\d)"
        return re.search(pattern, r.stdout) is not None
    except Exception:
        return True


def assert_initial_data():
    if not os.path.exists(RESULTS_CSV):
        console.print("[bold red]ERROR:[/bold red] Results CSV not found.")
        console.print(f"  Expected: {RESULTS_CSV}")
        console.print("  Run the initial LHS sweep first (with MPXI), then re-run.")
        sys.exit(1)
    df = pd.read_csv(RESULTS_CSV)
    mobo_agent.require_objective_columns(df, RESULTS_CSV)
    df = df.dropna(subset=OBJ_COLUMNS)
    n = len(df)
    if n < 3:
        console.print(f"[bold red]ERROR:[/bold red] Need >= 3 feasible points for MOBO, got {n}.")
        sys.exit(1)

    # Report the number that will actually TRAIN, not just the number with
    # objectives. get_next_candidate additionally drops designs whose rings
    # violate the radial clearance rule, so this pre-flight line previously
    # printed a count 4 higher than the training set and read as though those
    # designs were being used.
    n_train = n
    if {"d2_inner_mm", "d3_inner_mm"}.issubset(df.columns):
        ok = df.apply(
            lambda r: (pd.isna(r["d2_inner_mm"]) or pd.isna(r["d3_inner_mm"])
                       or mobo_agent.is_ring_ordering_ok(r["d2_inner_mm"],
                                                         r["d3_inner_mm"])),
            axis=1)
        n_train = int(ok.sum())

    console.print(f"[green]Loaded {n} designs with all {len(OBJ_COLUMNS)} objectives; "
                  f"{n_train} will train "
                  f"({n - n_train} excluded for ring clearance).[/green]")
    return n_train


def print_status(idx, config_name):
    """Print Pareto front summary."""
    try:
        df = pd.read_csv(RESULTS_CSV).dropna(subset=OBJ_COLUMNS)
        t = Table(title=f"MOBO Status (iter {idx})")
        t.add_column("Metric", style="cyan")
        t.add_column("Best", style="green")
        t.add_column("Worst", style="red")
        t.add_column("Mean", style="yellow")
        for col, label, sign in zip(OBJ_COLUMNS, OBJ_SHORT, OBJ_DIRECTIONS):
            vals = df[col]
            best = vals.min() if sign < 0 else vals.max()
            worst = vals.max() if sign < 0 else vals.min()
            fmt = ".4e" if col == "ppds_ring1" else ".4f"
            t.add_row(label, f"{best:{fmt}}", f"{worst:{fmt}}", f"{vals.mean():{fmt}}")
        t.add_row("Total configs", str(len(df)), "", "")
        console.print(t)
    except Exception as e:
        console.print(f"[yellow]Could not print status: {e}[/yellow]")


def main():
    parser = argparse.ArgumentParser(description="Sequential MOBO loop for SAI SC-SPECT")
    parser.add_argument("--max_iters", type=int, default=50,
                        help="Total MOBO iterations (default: 50)")
    args = parser.parse_args()

    TOTAL_ITERATIONS = args.max_iters

    # Built from OBJ_NAMES, never hand-written. The hardcoded version of this
    # banner still read "MPXI (min)" for a full run after MPXI was changed to
    # windowed+active and MAXIMIZED, so the log contradicted the optimizer for
    # anyone reading it later.
    console.print(Panel.fit(
        "[bold green]SAI SC-SPECT MOBO Controller[/bold green]\n"
        f"Design: ({', '.join(mobo_agent.PARAM_NAMES)})\n"
        f"{len(OBJ_NAMES)} objectives: {', '.join(OBJ_NAMES)}\n"
        "ModelListGP + qLogNEHVI | Sequential q=1",
        subtitle=f"Max iterations: {TOTAL_ITERATIONS}"
    ))

    ensure_dirs()
    # Must be taken before anything reads the manifest, since the index race
    # starts at the first read. The fd is held in a module global, not here.
    acquire_singleton_lock()
    if not check_disk_space("at startup"):
        sys.exit(1)
    ensure_manifest_header()
    assert_initial_data()

    start_idx = get_next_manifest_index()
    if start_idx >= TOTAL_ITERATIONS:
        console.print(f"[green]Done.[/green] Already have {start_idx} entries.")
        return

    n_to_run = TOTAL_ITERATIONS - start_idx

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:

        task = progress.add_task("[green]MOBO Loop", total=n_to_run)

        for _ in range(n_to_run):
            if not check_disk_space():
                break
            idx = get_next_manifest_index()
            console.print(f"\n[bold yellow]=== MOBO Iteration {idx} ===[/bold yellow]")

            # 1. Ask MOBO agent for next candidate
            console.log("Calling mobo_agent.get_next_candidate()...")
            try:
                diam, n_ap, n_det1, n_det2, d2_inner, d3_inner = get_next_candidate(RESULTS_CSV)
            except Exception as e:
                console.print(f"[bold red]MOBO Agent Failed:[/bold red] {e}")
                break

            console.log(f"[cyan]Candidate:[/cyan] d={diam:.4f}mm  n={n_ap}  "
                        f"nd1={n_det1}  nd2={n_det2}  "
                        f"d2={d2_inner:.1f}mm  d3={d3_inner:.1f}mm")

            # 2. Create work directory
            config_name = (f"mobo_{idx:04d}_ap{diam:.4f}_nap{n_ap}"
                           f"_nd1_{n_det1}_nd2_{n_det2}"
                           f"_d2_{d2_inner:.0f}_d3_{d3_inner:.0f}")
            work_dir = os.path.join(RESULTS_DIR, config_name)
            os.makedirs(work_dir, exist_ok=True)

            # 3. Append to manifest
            append_manifest_row(idx, config_name, diam, n_ap, n_det1, n_det2,
                                d2_inner, d3_inner, work_dir, "", "pending")

            # 4. Submit SLURM job
            env_vars = (
                f"ALL,"
                f"WORK_DIR={work_dir},"
                f"APERTURE_DIAM={diam},"
                f"N_APERTURES={n_ap},"
                f"N_DET_RING1={n_det1},"
                f"N_DET_RING2={n_det2},"
                f"D2_INNER={d2_inner},"
                f"D3_INNER={d3_inner},"
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

            # "The SLURM job exited" is not "the evaluation succeeded". Marking
            # it completed on exit status alone is what hid a broken geometry
            # generator for five days: four campaigns ran at 51-69% of their
            # nominal budget with every manifest row reading "completed".
            #
            # Check the archive actually gained an objective value for this
            # config before claiming the iteration happened.
            outcome = "completed"
            try:
                df_chk = pd.read_csv(RESULTS_CSV)
                row = df_chk[df_chk["config"].astype(str) == config_name]
                if row.empty:
                    outcome = "no_row"
                elif row[OBJ_COLUMNS].isna().all(axis=1).iloc[0]:
                    outcome = "all_nan"
            except Exception as e:
                console.print(f"[yellow]Could not verify {config_name}: {e}[/yellow]")
                outcome = "unverified"

            if outcome != "completed":
                console.print(
                    f"[bold red]Iteration {idx} produced no usable metrics "
                    f"({outcome}).[/bold red] The pipeline job exited cleanly but "
                    f"the archive gained nothing. This is a LOST iteration, not a "
                    f"measurement. Check {LOG_DIR}/out for this config before "
                    f"trusting the campaign's budget."
                )
            patch_manifest_status(idx, job_id, outcome)

            # 6. Print status
            print_status(idx, config_name)
            progress.advance(task)

    console.print("\n[bold green]MOBO Loop finished.[/bold green]")
    print_status(get_next_manifest_index() - 1, "final")


if __name__ == "__main__":
    main()
