# Bayesian Optimization for SC-SPECT SAI Hardware Design

**Target:** IEEE MIC 2026 abstract (deadline May 12, 2026)

---

## 1. Problem

Evaluating one SC-SPECT configuration requires ~2 hours of PPDF ray-tracing on HPC (25 CPUs). Grid search over even a modest 20×20 parameter grid = 400 configs × 2h = 800 hours. Bayesian Optimization finds near-optimal solutions in ~25–50 evaluations by building a surrogate model of the objective.

## 2. Design Vector — 2D Hardware-Only

| Parameter | Range | Baseline | Physical Effect | Literature |
|-----------|-------|----------|-----------------|-----------|
| `aperture_diam_mm` | [0.2, 1.0] | 0.4 | Resolution↔sensitivity tradeoff. Smaller = sharper beams, fewer photons. | Van Audenhaege 2015: "dominant factor" |
| `n_apertures` | [60, 360] | 180 | Angular sampling density. More = higher ASCI, but more multiplexing. | Han/Yao JNM 2026: ASCI governed by aperture count |

**Why these two:** They are the most impactful and most tightly coupled hardware parameters. Together they determine the open-fraction of the collimator ring: `180 × 0.4 / (2π × 35) ≈ 33%`. This drives the resolution-sensitivity-multiplexing tradeoff.

**Why not T8 translations:** (a, b) are acquisition parameters that can be configured manually in minutes. Hardware parameters require full PPDF recomputation to evaluate — this is where BO provides real value.

**Physical constraint:** `aperture_diam < chord_spacing = 2 × r_center × sin(π / n_apertures)`. At n=360, chord≈0.61mm so diam must be <0.61. The GP learns this from low-JI configs at infeasible combinations.

**Analogy to Kirtiraj:** Direct match — his (Diameter_mm, Displacement_mm) → our (aperture_diam_mm, n_apertures). Same dimensionality, same SingleTaskGP, same sequential approach.

## 3. Objective — JI (Joint Index)

```
JI = (sensitivity_mean / FWHM²) × ASCI_pct / 100
```

- Validated by Harsh (SNMMI 2026): JI-optimal configs produce best reconstruction quality
- Same formula as Kirtiraj → direct comparison possible
- Computed from beam analysis only (no reconstruction) → fast

## 4. Method — Sequential BO (Matching Kirtiraj)

```
1. Evaluate f at N₀ initial points (LHS, N₀ = 25)
2. Fit GP surrogate: SingleTaskGP(Matérn 5/2 + ARD)
3. For t = 1 to 100:
   a. x_new = argmax EI(x)     (Expected Improvement, q=1)
   b. Generate geometry with (aperture_diam, n_apertures)
   c. Run PPDF + beam analysis on HPC → compute JI
   d. Append (x_new, JI) to dataset
   e. Refit GP
4. Return best config found
```

**Key choices:**
- **Sequential q=1** (not batch): each observation improves the next proposal. Simpler, proven by Kirtiraj.
- **SingleTaskGP**: standard exact GP. Works well for 2D with 25–125 points.
- **Matérn 5/2 + ARD**: C² smooth kernel. ARD learns which parameter matters more (expect aperture_diam has shorter lengthscale).
- **Expected Improvement**: balances exploration (high uncertainty) and exploitation (high predicted JI).

## 5. Pipeline Per Configuration

```
(aperture_diam, n_apertures)
  → Step 0: Geometry generation (~1 min)
      generate_mph_scanner_circularfov.py --aperture_diam <d> --n_apertures <n>
      → scanner_layouts_<md5>.tensor
  → Step 1: PPDF ray-tracing (~2 hours, 2 SLURM array jobs)
      2 layouts × 8 T8 poses = 16 HDF5 files (a=0.2, b=0.2 fixed)
  → Steps 2-4: Beam analysis (~5 min)
      Extract masks → Extract properties → ASCI histograms
  → Step 5: Compute JI
      → Append to results_summary.csv
```

## 6. Implementation Status

### Completed
| File | What |
|------|------|
| `geometry/generate_mph_scanner_circularfov.py` | Added `--aperture_diam` and `--n_apertures` CLI args |
| `optimization/bo_agent.py` | SingleTaskGP + EI, q=1, 2D bounds. Matches Kirtiraj's structure. |
| `optimization/generate_configs.py` | LHS in 2D → configs_manifest.csv |
| `optimization/compute_ji.py` | JI for SAI (200×200, 16 files, aperture_diam + n_apertures tracking) |
| `optimization/configs/bo_config.yml` | Bounds, paths, SLURM settings |

### Pending
| File | What |
|------|------|
| `optimization/run_bo_loop.py` | Sequential BO orchestrator (adapt Kirtiraj's 12_run_bo_loop_checkpointed.py) |
| `optimization/run_sai_pipeline.sh` | Per-config SLURM script (geom → PPDF → beam analysis → JI) |
| Beam analysis scripts | Uncomment + adapt for T8 16-file aggregation |

## 7. Benchmark (from CCR execution 2026-04-05)

| Metric | Value |
|--------|-------|
| Wall time per config (2 layouts × 8 T8 poses) | ~2 hours |
| CPU utilization (25 cores) | 99.74% |
| Peak memory | 4.6 GB |
| Output per config | 16 files × 538 MB = ~8.6 GB |
| Output PPDF shape | (3360 crystals, 40000 pixels) per file |

**BO budget estimate:** 25 initial (LHS) + 50 BO iterations = 75 configs × 2h = ~150 hours compute = ~6 days sequential. Can parallelize LHS sweep via SLURM array.

## 8. Timeline

| Week | Dates | Milestone |
|------|-------|-----------|
| 1 | Apr 6–12 | LHS 25 configs, submit sweep, collect JI. Build run_bo_loop.py. |
| 2 | Apr 13–19 | Run sequential BO (~50 iterations), convergence analysis |
| 3 | Apr 20–26 | Validate top-5 with MLEM reconstruction |
| 4 | Apr 27–May 3 | Generate figures, sensitivity analysis |
| 5–6 | May 4–12 | Write + submit IEEE MIC abstract |

## 9. Expected Results

- GP R² > 0.85 on held-out data (2D is ideal GP regime)
- JI improvement of 20–50% over baseline (aperture_diam=0.4, n_apertures=180)
- ARD lengthscales reveal which parameter dominates
- Convergence within 30–50 BO iterations
- Reconstruction validation: visibly sharper hot rod phantoms for BO-optimal config

## 10. Key References

- Van Audenhaege 2015 (PMC5148182): aperture diameter = most impactful SPECT parameter
- Han, Tripathi, Yao (JNM 2026): ASCI as SC-SPECT optimization metric
- Furenlid/Barrett (PMC3703762): aperture position = dominant parameter
- PMC4337004: pinhole geometry = critical factor in image quality
- Harsh SNMMI 2026: JI-optimal → near-peak CNR
