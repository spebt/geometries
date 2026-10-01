# CCR Cleanup & Symlink Recipe

Run this on CCR **after** the current MOBO loop finishes (or when you are
ready to briefly stop it). None of these steps modify any data — they only
reorganise files and add a convenience symlink.

## What this changes

- Removes the 11 archived legacy files (`bo_agent.py`, `run_bo_loop.py`,
  `backfill_mpxi.py`, LHS sweeps, etc.) from `optimization/` and puts them
  under a new `spebt/legacy/` folder.
- Removes the untracked scratch files (`direction1_pipeline_figures.py`,
  `pymatcal/debug_*`, `pymatcal/test_subdivision_fix.py`) from the working
  tree — they still exist under `spebt/legacy/` if you need them.
- Adds a convenience symlink so you can `cd $HOME/sai/opt` instead of the
  triple-`spebt` path. Actual paths in code and CSVs are unchanged, so
  nothing else needs updating.

## What this does NOT change

- The MOBO pipeline continues to run against the same actual paths
  (`/vscratch/grp-rutaoyao/Omer/spebt/spebt/spebt/optimization/…`).
- `results_summary_mobo.csv`, the manifest, the per-config `work_dir`
  values — all untouched.
- No git repo restructure; the inner `spebt/` folder inside the repo stays
  where it is because moving it would require rewriting every hardcoded
  path in the codebase and re-migrating the CSV, which is more risk than
  the redundant naming is worth.

## Steps

### 1. Verify the MOBO loop is stopped

```bash
squeue -u $USER | grep mobo_loo
# If a mobo_loo job is running, wait for it or scancel it — pulling mid-loop
# could confuse the per-config sbatch machinery.
```

### 2. Pull the audit + cleanup commit

```bash
cd /vscratch/grp-rutaoyao/Omer/spebt/spebt/spebt
git pull
```

The pull will:
- Move 11 legacy files from `optimization/` to `legacy/` (as git renames)
- Update 4 source files with the flagged audit fixes
- Add `AUDIT_REPORT.md`, `tests/test_pipeline.py`, `CCR_CLEANUP_RECIPE.md`

### 3. Run the test suite once to confirm the pull works on CCR

```bash
. /vscratch/grp-rutaoyao/Omer/.venv/bin/activate
python3 -m pytest tests/test_pipeline.py -v
```

All 25 tests should pass. If any fail, do NOT restart the MOBO loop —
tell me the failure and we debug.

### 4. Add the convenience symlink for shorter navigation

```bash
# One-time setup — create a short alias in your home directory
ln -sfn /vscratch/grp-rutaoyao/Omer/spebt/spebt/spebt $HOME/sai
```

Now you can `cd ~/sai/optimization` instead of typing the triple-spebt
path. This is purely for interactive convenience — actual pipeline paths
in code, CSVs, and SLURM scripts are unchanged.

If you also want a shorter results path:

```bash
ln -sfn /vscratch/grp-rutaoyao/Omer/spebt/spebt/spebt/optimization/results $HOME/sai_results
```

### 5. Restart the MOBO loop (still 4-obj until you flip to 5-obj)

```bash
cd ~/sai/optimization      # or the long path if you skipped step 4
sbatch submit_mobo.sh
squeue -u $USER
```

The controller picks up where it left off from the manifest.

## After all this

Local Mac tree becomes:

```
Users/Omer/Desktop/RA/omer/
├── mph/                       # unrelated MPH project (untouched)
└── spebt/
    ├── AUDIT_REPORT.md
    ├── CCR_CLEANUP_RECIPE.md   ← this file
    ├── context.md
    ├── data/
    ├── geometry/
    ├── legacy/                 ← NEW: archived one-off + old-BO code
    │   ├── analyze_bo_convergence.py
    │   ├── backfill_mpxi.py
    │   ├── bo_agent.py
    │   ├── cleanup_lhs_results.py
    │   ├── create_4d_csv.py
    │   ├── direction1_pipeline_figures.py
    │   ├── generate_configs.py
    │   ├── generate_configs_3d.py
    │   ├── pymatcal/           ← subdivision debug scripts
    │   ├── run_bo_loop.py
    │   ├── run_recon_pareto.sh
    │   ├── submit_lhs_sweep.sh
    │   └── submit_lhs_sweep_3d.sh
    ├── optimization/           ← MIC-critical MOBO scripts only
    │   ├── backfill_cnr.py
    │   ├── backfill_ppds.py
    │   ├── CNR_PIPELINE_INTEGRATION.md
    │   ├── compute_cnr.py
    │   ├── compute_metrics.py
    │   ├── D2_D3_EXPANSION_PLAN.md
    │   ├── mobo_agent.py
    │   ├── run_mobo_loop.py
    │   ├── run_sai_pipeline.sh
    │   ├── sai_analyze_asci.py
    │   ├── sai_extract_masks.py
    │   ├── sai_extract_props.py
    │   ├── submit_mobo.sh
    │   ├── analyze_lhs_metrics.py         # analysis, kept
    │   ├── analyze_mobo_convergence.py    # analysis, kept
    │   └── configs/
    ├── plots/
    ├── pymatana/
    ├── pymatcal/
    ├── recon/
    │   └── run_recon_comparison.py         # (email drafts moved to ~/MIC/)
    └── tests/
        └── test_pipeline.py
```

CCR tree becomes (paths as they exist on disk):

```
/vscratch/grp-rutaoyao/Omer/spebt/spebt/spebt/    ← repo root, same as before
├── … (as above)
└── (all data / results / plots stay here, under .gitignore now)

$HOME/sai -> /vscratch/…/spebt/spebt/spebt/       ← NEW: convenience symlink
$HOME/sai_results -> …/optimization/results/       ← optional
```

## Why not do a full flatten now?

Flattening the repo (moving `spebt/*` up to the git root so a clone gives
you `/…/spebt-clone/optimization/` directly, only one `spebt` on CCR)
would require:

- Rewriting every hardcoded path in `run_mobo_loop.py`, `run_sai_pipeline.sh`,
  `submit_mobo.sh`, `run_recon_comparison.py`, `configs/bo_config.yml`, and
  the CCR migration doc.
- A `sed` migration on the CSV's `work_dir` column across ~100 rows.
- Cancelling and restarting the MOBO loop.
- Coordinating both the git-repo restructure AND the on-disk restructure
  in a single window without leaving zombie sbatches behind.

That's a real project — worth doing once the MIC results are locked and
we're not iterating actively on the pipeline. Not worth doing this week
while the CNR-in-loop runs are producing headline data.
