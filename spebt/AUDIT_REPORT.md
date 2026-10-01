# SAI SC-SPECT MOBO Pipeline — Audit Report

Focused rigor pass on the MIC-critical pipeline before scaling the CNR-in-loop
runs. Covers: correctness bugs, dead/bloated code, missing error handling,
consistency risks. All findings below are either fixed in this pass or flagged
for a decision.

## Test suite added

`tests/test_pipeline.py` — 25 tests, all passing (~5 s runtime).

| Coverage area | Tests |
|---|---|
| `backfill_cnr._find_cnr_npz` matcher (3 fallback layers) | 5 |
| `mobo_agent.is_feasible` aperture-overlap math | 4 |
| `compute_metrics._per_beam_radial_fwhm` projection math | 5 |
| `compute_metrics.compute_ppds` end-to-end on synthetic HDF5 | 4 |
| `compute_cnr._write_cnr_to_csv` row/column update semantics | 5 |
| `compute_metrics.main()` CSV column alignment on append | 2 |

Run with `python3 -m pytest tests/test_pipeline.py -v` from the repo root.

---

## Fixes applied (ready to push)

### HIGH — correctness

- **`run_mobo_loop.py::patch_manifest_status()` now anchors on `idx`.**
  Previous behaviour patched the last non-empty row regardless of the `idx`
  argument. Worked only because the caller always appended immediately before
  patching, so any interruption between append and patch (e.g. controller
  wall-time kill) could leave the wrong row updated later. Fixed by scanning
  the manifest in reverse for the row whose first column equals `idx`; falls
  back to previous behaviour with a warning if no match.

- **`mobo_agent.py` docstring corrected.** The design vector was still listed
  as `(aperture_diam_mm, n_apertures, scint_radial_thickness_mm, ring_thickness_mm)`
  from an older 3D formulation; actual is `(aperture_diam_mm, n_apertures,
  n_det_ring1, n_det_ring2)`. Also fixed the "all 4 objectives" line to be
  generic (now references `OBJ_COLUMNS`) so future flips don't require another
  docstring edit.

- **`mobo_agent.py` unused import removed.**
  `from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood`
  had no callers.

- **`mobo_agent.py` CLI help updated** to list all five objective columns
  including `cnr_mean`.

### MED — clarity / correctness of comments

- **`compute_metrics.py` docstring updated** to list `PPDS` alongside the other
  metrics, with a note that CNR is added by `compute_cnr.py` in a separate
  step (so future readers don't wonder where CNR comes from).

- **`compute_metrics.compute_fwhm_and_asci()` inline comment fixed.** The old
  comment described a wrong 5-column schema for `beam_properties`; the real
  schema (defined in `pymatana/scanner_modeling/beam_property_io.py`) is 11
  columns and column 4 is `FWHM (mm)`. New comment lists all 11 columns.

- **`run_recon_comparison.py` — dead `SPROJ = None` module variable** replaced
  with a comment explaining that `sproj` is auto-detected per PPDF file inside
  `forward_project` / `run_mlem` and cannot be a constant (varies across
  configs when `n_det_ring1/2` differ).

- **`run_recon_comparison.py` — stale phantom fallback path.** The
  auto-detection list only had the old 2-spebt phantom location, which is
  gone after the vscratch recovery. Added the 3-spebt path as the primary
  fallback (kept the 2-spebt entry as a low-priority backup for legacy
  ad-hoc invocations).

- **`run_recon_comparison.py` — "reward hacking" debug print removed.**
  Legacy tag from the JI-vs-CNR sanity-check era; the current message wasn't
  useful in the compare-mode output. Replaced with a cleaner summary using
  the user-supplied labels.

---

## Flagged for decision (NOT fixed in this pass)

These are not correctness bugs and don't affect the running MOBO loop. Bring
them up if / when time permits.

### `run_mobo_loop.py`

- **`is_job_running()` uses substring match on job IDs.** With 8-digit CCR
  IDs the odds of a false positive are effectively zero, but a strict
  word-boundary check would be more defensible.
- **Sequential q=1 loop wastes wall-time waiting.** By design (each iteration
  is a separate SLURM job). Only worth changing if we want batch acquisition,
  which is a much bigger change.
- **No timeout on `is_job_running` polling loop.** In practice the SLURM
  wall-time kills the child job, which clears squeue. But a hung squeue
  command would freeze the controller silently.

### `mobo_agent.py`

- **Penalty calculation** (`col.min() * 0.5` / `col.max() * 2.0`) is
  sign-fragile. Works because all metrics are strictly positive today.
  Documenting this constraint would be safer than changing the code.
- **`warnings.filterwarnings("ignore", category=UserWarning)`** globally
  silences ALL UserWarnings from botorch/gpytorch. Some of those flag
  numerical issues in GP fitting. Currently intentional (log noise reduction)
  but worth revisiting if a GP fit ever gives obviously-wrong lengthscales.

### `run_sai_pipeline.sh`

- **`set -uo pipefail` without `-e`** is intentional (the script wants to
  handle failures per-step), but silent-failure risk is real: any Python
  crash in step 3 or step 4 is invisible unless the user tails the `.err`
  file. `write_zero_ji` only covers KNOWN infeasibility, not arbitrary crashes.
- The legacy `write_zero_ji` function name (from the JI compound-metric era)
  is still accurate about what it does but confusing to a new reader.

### `run_recon_comparison.py`

- The `--config {baseline,bo_optimized}` CLI argument is accepted but never
  branches on. Dead arg, safe to remove — but external ad-hoc callers might
  still pass it.

---

## Legacy files (candidates for archive)

These are on the disk but not on the MIC-critical path. Consider moving to a
`legacy/` subfolder to reduce cognitive load in the `optimization/` directory:

- `bo_agent.py` — old 2D BO agent
- `run_bo_loop.py` — old 2D BO controller
- `analyze_bo_convergence.py` — 2D BO analysis
- `backfill_mpxi.py` — one-off MPXI backfill (all rows now have MPXI)
- `create_4d_csv.py` — one-off migration from 2D to 4D CSV schema
- `cleanup_lhs_results.py` — one-off cleanup
- `generate_configs.py`, `generate_configs_3d.py` — LHS generators, ran once
  at project start

None of these are imported by the live pipeline; removing or archiving them
is purely cosmetic.

---

## What did NOT need changes

- **`compute_cnr.py`** — cleanly written, correctly reuses `run_recon_comparison`
  functions, monkey-patches `N_ITERATIONS` in a try/finally block, handles
  missing PPDFs with an explicit `FileNotFoundError` catch.
- **`backfill_cnr.py`** — the three-layer matcher (exact → prefix-glob →
  design-signature) is exactly right for our naming variance across LHS vs
  MOBO configs; tests confirm all three layers work as advertised.
- **`backfill_ppds.py`** — simple, correct, idempotent. Checkpoints every 10
  rows so a crash doesn't lose work.
- **`submit_mobo.sh`** — small, correct, committed to git so it survives
  vscratch cleanup.
- **`compute_metrics.compute_ppds()`** — verified end-to-end with synthetic
  HDF5 tests. Note the PPDS metric is not currently used as a MOBO objective
  (per Dr. Yao's earlier decision); the computation is kept in place for
  potential future reformulation.
- **`sai_extract_masks.py`, `sai_extract_props.py`, `sai_analyze_asci.py`** —
  thin adapters over the existing `pymatana/ppdf-analysis/beam-analysis`
  library. Small, single-purpose, no issues found.
- **`generate_mph_scanner_circularfov.py`** — geometry generator was already
  reviewed during the SNMMI slide-15 visualization work; nothing critical
  has changed since.

---

## Verification

All fixes verified against the test suite:

```
$ python3 -m pytest tests/test_pipeline.py -v
...
============================== 25 passed in 4.64s ==============================
```

Same 25 tests pass both before AND after the fixes; no behavioral regressions.

---

## Recommendation for the CNR runs

The pipeline is in a good state to keep the current 4-objective loop running
overnight. When the CNR row count reaches ~35-45 (currently 37, growing at
~1/hour), we can push the 5-objective flip and the reverted `mobo_agent.py` /
`run_mobo_loop.py` edits together with all the audit fixes as one clean commit.
The test suite catches the most likely regression paths (matcher, column
alignment, PPDS math), so future edits have a safety net.

---

# Fresh-eyes audit — 2 Sep 2026

Re-checked the code and the results that the MIC claims rest on, with no
assumption that earlier passes got it right. Four things changed as a result;
two of them change what we should say to Dr. Yao.

## 1. The headline reproduces exactly

Recomputed from the raw per-seed CNR outputs rather than from any summary:

| design | n seeds | CNR sector-mean | sd |
|---|---|---|---|
| `mobo_0296` (ours) | 5 | 4.7224 | 0.0694 |
| `tmi_reference_000` | 5 | 3.5887 | 0.0553 |

Gap 1.1336, SE 0.0397, t = 28.6, **+31.6%**. Same seeds 0-4 on both sides,
150 ML-EM iterations on both sides, identical `rod_radii_mm`
(0.100-0.225 mm). No numerical code changed between the two evaluations: the
only commits since 1 Aug touching geometry/PPDF/recon/CNR were `9e37419`
(mail settings), `bc68303` (venv path) and `545aa09` (error handling on the
failure branch only). The comparison is like-for-like.

## 2. The head-to-head conclusion was overstated — corrected

Five paired replicates completed, not three. Best CNR per arm:

| rep | 2obj | 5obj | 2obj - 5obj | feasible 2obj / 5obj |
|---|---|---|---|---|
| r0 | 4.7811 | 4.7365 | +0.0446 | 101 / 101 |
| r1 | 4.8599 | 4.7192 | +0.1408 | 75 / 62 |
| r2 | 4.7940 | 4.6887 | +0.1053 | 75 / 70 |
| r4 | 4.7591 | 4.7640 | -0.0049 | 99 / 100 |
| r5 | 4.8233 | 4.8300 | -0.0067 | 99 / 100 |

All five: mean difference +0.056, t = 1.89, 2obj ahead in 3 of 5.
**Restricted to the three replicates with complete data (r0, r4, r5): mean
difference +0.011, t = 0.65, 2obj ahead in 1 of 3.**

The entire apparent two-objective advantage sits in r1 and r2 — the two
replicates that overlapped the crash window, where 25-38% of evaluations were
lost and the five-objective arm lost more of them (62 and 70 feasible against
75 and 75). Once the crashes are excluded the formulations are
indistinguishable. Keep the five-objective set: it costs nothing in CNR and it
is the one that reports the physics trade-offs.

## 3. Cross-run convergence — a stronger result than the headline

Eleven independent searches (10 head-to-head arms + the main campaign), from
different LHS seed sets and under two different objective sets, recover the
same aperture:

| parameter | mean +/- sd | sd as % of searched range |
|---|---|---|
| aperture_diam_mm | 0.287 +/- 0.014 | **1.8%** |
| n_apertures | 186 +/- 25 | 11.8% |
| n_det_ring1 | 499 +/- 62 | 11.5% |
| n_det_ring2 | 821 +/- 145 | 12.2% |
| d2_inner_mm | 382 +/- 34 | 13.6% |
| d3_inner_mm | 489 +/- 65 | 27.9% |

`fwhm_weighted_mean` at those eleven optima spans 0.482-0.487 — under 1%.

Aperture diameter is pinned to ~1.8% of its range; the ring geometry is barely
constrained at all. This answers the obvious reviewer question ("is this a
lucky local optimum?") with reproducibility rather than assertion, and it says
the design lever is the aperture, not the rings.

## 4. The n_det_ring2 headroom test ran and was never read

`ring2_000..007` completed on 12 Aug (8 configs, 16 PPDFs each, full beam
outputs) and the results sat in `results/ring2_seed_out/` **without ever being
merged into `results_summary_mobo.csv`**. Consequences: the optimizer never
trained on them, and nobody read the answer. Holding the `mobo_0296` aperture
fixed and sweeping ring 2 outward:

| n_det_ring2 | d2_inner (mm) | CNR sector-mean |
|---|---|---|
| 964 | 393.2 | 4.476 |
| 1016 | 414.1 | 4.483 |
| 1066 | 435.1 | 4.544 |
| 1116 | 456.1 | 4.485 |
| 1168 | 477.1 | 4.507 |
| 1218 | 498.0 | 4.455 |
| 1268 | 519.0 | 4.277 |
| 1320 | 540.0 | 4.266 |

Flat from 964 to 1218, then falling. **No headroom above the 960 ceiling** —
consistent with finding 3, where the ring parameters are the unconstrained
ones. This also explains why the archive has 104 rows pinned at exactly 960
and none above it: raising the bound opened a region that has nothing in it.

Caveat, unresolved: `ring2_000` is `mobo_0296` with 4 extra ring-2 crystals,
yet it reads 4.476 against `mobo_0296`'s 4.722 +/- 0.069 — about 3 sigma low,
and the whole batch sits ~0.25 below the archive scale. The deterministic
metrics reproduce (`ppds_ring1` identical to 10 significant figures,
`fwhm_weighted_mean` within 0.3%, ASCI within 1.7%), so the geometry and PPDF
path are fine and it is isolated to the CNR step. **Do not merge these 8 rows
into the archive** until this is settled; a 5-seed re-measurement of
`ring2_000` decides it. The within-batch conclusion (no headroom) does not
depend on it, since all 8 ran identically on the same day.

## 5. In-loop CNR is an unseeded single draw

`run_sai_pipeline.sh` calls `compute_cnr.py` without `--seed`, so every
`cnr_sector_mean` in the archive is one irreproducible Poisson realisation
(sigma ~= 0.08). This is deliberate and documented in the docstring, and it is
fine for ranking, but two consequences are worth stating plainly:

- The archive CNR column cannot be reproduced exactly.
- Selecting the maximum over ~100 draws inflates it. Measured directly: for
  all five designs that were re-measured over seeds, the archive value exceeds
  the 5-seed mean (+0.116, +0.038, +0.211, +0.046, +0.107; mean +0.104). Five
  of five positive is the selection effect, not a bug.

The headline is unaffected because both sides use re-measured 5-seed means,
and the reference was never selected on.

## 6. Silent NaN paths closed

Four returns produced NaN with no message — `compute_cnr.py:78`,
`compute_metrics.py:70`, `:214`, `:612`. Both of August's expensive incidents
were NaN appearing without explanation. Each now names its reason. No bare
`except:` and no `fillna` remain anywhere in the optimization package.
Committed as `35b5a2f`; 112 tests pass.

## Bounds check

323 archive rows, no duplicate config names, no partially-NaN rows (52 are
all-NaN, all of them crash-window or legacy 4D seeds). Six rows lie outside
the declared bounds; four are all-NaN legacy `lhs4d_*` seeds, and two
(`mobo_0190`, `mobo_0193`) carry real data at `d3_inner_mm` = 640 against a
bound of 618 — proposed under the earlier wider bound and never re-flagged.
They are GP training points outside the current search box, which is harmless
extrapolation, and neither is near the optimum.

## Open, not resolvable from the files

Whether the SAI baseline (0.4 mm, 180 apertures) was itself the product of the
systematic resolution-variance screening Dr. Yao described. The +31.6% claim
rests on his characterisation of it, and it is the one thing a reviewer will
push on.

## Fixes implemented (2 Sep 2026)

| # | Fix | Commit |
|---|---|---|
| 1 | Every silent NaN return now names its reason | `35b5a2f` |
| 2 | In-loop CNR seed derived from the config name | `7981723` |
| 3 | Training points outside the current bounds are reported | `7981723` |
| 4 | `check_unmerged_results.py` finds finished-but-unmerged batches | `7981723` |
| 5 | Checker counts sibling h2h archives, reads `unmerged_ignore.txt` | `0927144` |
| 6 | `REPEATS_DIR` keeps a repeat measurement out of an older one's files | `2a9e818` |

Test suite 112 -> 133.

**On fix 2.** `run_sai_pipeline.sh` now passes `--seed_from_config`, and the
seed comes from sha256 of the config name rather than Python's `hash()`, which
is salted per process and would have produced a different seed on every SLURM
task. Different designs still draw independent noise, so the GP sees the same
honest across-design variance it always did; only re-running one config is now
deterministic. Explicit `--seed` still wins, so the 5-seed repeat harness is
unaffected. Rows written before today keep their unseeded values.

**What the checker found.** 36 evaluated configs were absent from
`results_summary_mobo.csv`. 21 of those are the shared LHS seeds, which
correctly live in the head-to-head arms' own archives, and one is the TMI
reference, deliberately excluded so a fixed benchmark never becomes GP training
data. After teaching the checker about both, 6 remain:

`seed6d_00..05` (the July D2/D3 expansion seeds) were evaluated, and their raw
files all survive — 16 PPDFs and 2 mask files each — but they predate the
August MPXI change and so have no `mpxi_windowed_active_mean` column. Merging
them as-is would add six partially-measured rows, which `get_next_candidate`
excludes from training anyway. To actually use them they need the documented
backfill (`analyze_mpxi_variants.py` then `backfill_mpxi_variants.py`) first.
That is worth doing: they are six deliberately-placed points at the d2/d3
corners, which is exactly where the archive is thin.

**Gating both merges.** Job `25952484` re-measures `ring2_000` over 5 seeds
into `results/cnr_ring2_recheck/`. If it returns ~4.72 the Aug 12 batch value
was wrong and the sweep should be re-read; if it returns ~4.48 the difference
is real and belongs to the geometry. Either answer settles whether the ring2
and seed6d batches sit on the archive's CNR scale, which is what decides if
they can be merged. `results/cnr_repeats/` was backed up first and verified
byte-identical.
