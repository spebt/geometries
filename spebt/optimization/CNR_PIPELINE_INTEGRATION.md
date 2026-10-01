# CNR-in-the-loop integration — what to change on CCR next week

This note captures the small set of CCR-side changes needed to wire the
new `compute_cnr.py` into the existing per-evaluation SLURM pipeline.
The Python code changes have already been pushed in this round (revert
to 4-objective MOBO + new `compute_cnr.py` + `backfill_cnr.py`).

## 1. Patch `run_sai_pipeline.sh`

Add a CNR step **after** the existing `compute_metrics.py` invocation,
inside the same SLURM job, so each evaluated configuration writes a
`cnr_mean` value into the same CSV row as its other metrics:

```bash
# --- existing block (geometry -> ppdf -> beam analysis -> metrics) ---
python compute_metrics.py \
    --work_dir "$WORK_DIR" \
    --out_csv  "$RESULTS_CSV" \
    --config_name "$CONFIG_NAME" \
    --aperture_diam_mm "$APERTURE_DIAM" \
    --n_apertures "$N_APERTURES" \
    --n_det_ring1 "$N_DET_RING1" \
    --n_det_ring2 "$N_DET_RING2"

# --- NEW: append CNR to the same row ---
python compute_cnr.py \
    --work_dir "$WORK_DIR" \
    --phantom_path "/vscratch/grp-rutaoyao/Omer/spebt/spebt/data/sai_10mm/hot_rods_phantom_10.0_mm_x_10.0_mm.pt" \
    --out_csv  "$RESULTS_CSV" \
    --config_name "$CONFIG_NAME" \
    --iterations 150
```

If `$CONFIG_NAME` was force-zeroed upstream (infeasible config),
echo a `--force_nan` call instead so the CSV still gets a row:

```bash
if [ "$FORCE_ZERO" = "1" ]; then
  python compute_cnr.py --force_nan --reason "$REASON" \
      --work_dir "$WORK_DIR" --phantom_path "$PHANTOM" \
      --out_csv "$RESULTS_CSV" --config_name "$CONFIG_NAME"
fi
```

## 2. Bump SLURM `--time` and `--mem`

`compute_cnr.py` adds roughly 5–10 minutes per config (150 ML-EM
iterations on CPU; faster if CUDA is allocated). Update the per-config
SLURM script:

```
#SBATCH --time=01:00:00       # was probably 00:30:00
#SBATCH --mem=24G             # was 16G; recon needs a bit more headroom
```

If a GPU is available on the partition, request one — ML-EM scales
linearly with iter count and is the new dominant cost. Otherwise CPU is
fine for ranking purposes; 150 iterations on CPU finishes in ~7 min.

## 3. Update `mobo_agent.py` to add `cnr_mean` as a 5th objective

Once `cnr_mean` is reliably present in the CSV (verified via
`backfill_cnr.py` for the existing 17 reconstructed configs), flip the
objective set:

```python
OBJ_COLUMNS    = ["fwhm_mean", "asci_pct", "sensitivity_mean", "mpxi_mean", "cnr_mean"]
OBJ_DIRECTIONS = [-1.0,        1.0,        1.0,                -1.0,        1.0       ]
OBJ_NAMES      = ["FWHM (min)","ASCI (max)","Sensitivity (max)","MPXI (min)","CNR (max)"]
```

Also propagate the same change to `run_mobo_loop.py` status table and
the controller banner. Hold off on this flip until backfill confirms
the column is populated; otherwise the GP fit will fail with mostly-NaN
training data.

## 4. Recommended bring-up order on CCR

1. `git pull` to pick up the reverted 4-objective agent + new CNR
   scripts.
2. `python backfill_cnr.py --results_csv results/results_summary_mobo.csv
   --recon_root results/recon_results` — populates `cnr_mean` for the
   ~17 already-reconstructed configs. **No MOBO changes yet.**
3. Restart the 4-objective MOBO loop (no CNR objective yet) so we
   continue making progress on the established setup while step 4 is
   prepared:
   ```bash
   sbatch optimization/submit_mobo.sh
   ```
4. In parallel, patch `run_sai_pipeline.sh` (Section 1) and the SLURM
   resource line (Section 2). Test the patched pipeline on a single
   known configuration:
   ```bash
   bash run_sai_pipeline.sh   # with WORK_DIR pointing to e.g. mobo_0069's dir
   ```
5. Confirm the CSV row for that test config gains a `cnr_mean` value
   matching (within Poisson noise) what `backfill_cnr.py` wrote.
6. Once the in-loop CNR is verified, flip the MOBO objective set
   (Section 3) and restart the loop.

## 5. After CNR-in-the-loop is producing values

Move on to the D2/D3 design-space expansion described in
`D2_D3_EXPANSION_PLAN.md` (separate file in this folder).
