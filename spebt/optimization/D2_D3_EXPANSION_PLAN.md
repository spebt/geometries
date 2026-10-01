# D2 / D3 expansion plan — 4D -> 6D MOBO design space

Triggered by Dr. Yao's comment on the PPDS-validation reply, referencing
the TMI brain-SPECT protocol (steps 5–6 of the paper screenshot): after
CNR-in-the-loop is producing values, the next move is to expand the
MOBO design space to include the inner diameters of the middle detector
rings as design parameters, with the geometric ordering enforced as a
constraint.

## Current state

4D design space (`mobo_agent.py`, `BOUNDS_MIN`/`BOUNDS_MAX`):

| parameter | min | max | unit |
|---|---|---|---|
| aperture_diam_mm | 0.2 | 1.0 | mm |
| n_apertures | 60 | 270 | count |
| n_det_ring1 | 120 | 660 | count |
| n_det_ring2 | 180 | 960 | count |

Detector ring **inner diameters are fixed** in
`generate_mph_scanner_circularfov.py`:

```python
RING_INNER_DIAMS_MM = [260.0, 390.0, 520.0, 650.0]   # rings 1..4
```

so the optimizer currently cannot reshape the radial arrangement of the
detector stack.

## Proposed change — sweep D2 and D3

Following the TMI protocol shown in the screenshot, hold the innermost
(D1 = 260 mm) and outermost (D4 = 650 mm) rings fixed and let MOBO move
the two middle rings:

| parameter | min | max | unit | notes |
|---|---|---|---|---|
| d2_inner_mm | 280 | 540 | mm | new — Ring 2 inner Ø |
| d3_inner_mm | 320 | 640 | mm | new — Ring 3 inner Ø |

Plus the ordering constraint **D1 < D2 < D3 < D4**, i.e.
260 mm < d2_inner_mm < d3_inner_mm < 650 mm. Also enforce a minimum
radial gap between rings (say 20 mm) so the rings don't visually
collapse onto each other:

```
d2_inner_mm >= D1 + 20     ⇒  d2_inner_mm >= 280
d3_inner_mm >= d2_inner_mm + 20
d4_inner_mm >= d3_inner_mm + 20
```

Bounds above already respect the first and third inequalities; the
middle one (d3 - d2 >= 20) is handled as an MOBO feasibility
constraint inside `is_feasible_norm()`, exactly like the aperture
overlap constraint already in `mobo_agent.py`.

## Code changes required

### 1. `generate_mph_scanner_circularfov.py`

- Add `--d2_inner` and `--d3_inner` CLI arguments (default 390, 520 so
  legacy invocations stay identical).
- Replace the hard-coded `RING_INNER_DIAMS_MM = [260, 390, 520, 650]`
  with `[260, cli_args.d2_inner, cli_args.d3_inner, 650]`.
- The geometry validation block at the bottom needs no change; the
  existing per-ring overlap check already runs at the supplied
  diameters.

### 2. `mobo_agent.py`

- Extend the design-space constants:
  ```python
  PARAM_NAMES = [
      "aperture_diam_mm", "n_apertures",
      "n_det_ring1", "n_det_ring2",
      "d2_inner_mm", "d3_inner_mm",
  ]
  BOUNDS_MIN = [0.2,  60.0, 120.0, 180.0, 280.0, 320.0]
  BOUNDS_MAX = [1.0, 270.0, 660.0, 960.0, 540.0, 640.0]
  DIM = len(PARAM_NAMES)
  ```
- Add to `is_feasible_norm(x_norm)` an ordering check:
  ```python
  d2 = 280.0 + x_norm[..., 4] * (540.0 - 280.0)
  d3 = 320.0 + x_norm[..., 5] * (640.0 - 320.0)
  ordering_ok = (d3 - d2) >= 20.0
  return (aperture_overlap_ok) & ordering_ok
  ```
- The dedup block already normalises by parameter ranges so it handles
  6D without code change.

### 3. `run_mobo_loop.py`

- Update `append_manifest_row()` and the manifest CSV header to include
  `d2_inner_mm,d3_inner_mm`.
- Pass `--d2_inner $D2_INNER` and `--d3_inner $D3_INNER` to the
  per-config SLURM job via the `--export` environment block.

### 4. `run_sai_pipeline.sh`

- Forward `$D2_INNER` and `$D3_INNER` to the geometry generation step:
  ```bash
  python generate_mph_scanner_circularfov.py \
      --aperture_diam "$APERTURE_DIAM" \
      --n_apertures "$N_APERTURES" \
      --n_det_ring1 "$N_DET_RING1" \
      --n_det_ring2 "$N_DET_RING2" \
      --d2_inner "$D2_INNER" \
      --d3_inner "$D3_INNER" \
      --output_dir "$WORK_DIR"
  ```
- Plumb the two new env vars all the way through compute_metrics.py and
  compute_cnr.py invocations so they end up in the CSV alongside the
  other design parameters.

### 5. `compute_metrics.py`

- Add `--d2_inner_mm` and `--d3_inner_mm` CLI args (mirroring the
  existing `--n_det_ring1/2` pattern); write them into the CSV row so
  MOBO can re-read them.

## Backfill considerations

The 86 existing configurations all use D2=390 and D3=520. The backfill
script for the new columns is trivial:

```bash
python -c "
import pandas as pd
df = pd.read_csv('results/results_summary_mobo.csv')
if 'd2_inner_mm' not in df: df['d2_inner_mm'] = 390.0
if 'd3_inner_mm' not in df: df['d3_inner_mm'] = 520.0
df.to_csv('results/results_summary_mobo.csv', index=False)
"
```

The 4D points act as 6D training samples that happen to lie on the
fixed-D2=390, fixed-D3=520 slice. The GP will not extrapolate confidently
into the rest of the 6D space until the new loop visits it, but the
warm-start helps anchor the surrogate.

## Test plan before launching 6D loop

1. Sanity test: invoke `generate_mph_scanner_circularfov.py` with one
   D2/D3 combination different from the defaults and visually confirm
   via `draw_scanner_with_insets.py` that the rings shift correctly.
2. Run one full pipeline pass with a non-default D2/D3 to confirm
   PPDF, beam-mask, and CNR computations all succeed at the new
   diameters.
3. Run the MOBO loop for ~5 iterations with the 6D agent to confirm
   feasibility and dedup logic work; check that no proposed point
   violates `d3 - d2 >= 20`.
4. Once confirmed, queue ~50 iterations for the 6D sweep.

## Open question for Dr. Yao

The TMI screenshot suggests sweeping in the range `D1 < D2 < D3 < D4`
without further constraint than ordering. Do we also want to constrain
the *axial* arrangement (the detector rings have to physically share
the same axial extent) so that adjacent rings can't be closer than the
crystal radial thickness (6 mm) plus some mounting clearance? The 20 mm
minimum gap I've proposed is a conservative default — worth confirming
with him before committing the bound to code.
