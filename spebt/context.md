# SC-SPECT Research Context

**Last updated:** 2026-04-22
**Maintainer:** Omer Shah (syedomer@buffalo.edu)
**Rule:** Update this file whenever new progress, results, decisions, or context emerge.

---

## 1. People & Roles

| Person | Role | Contribution |
|--------|------|-------------|
| **Dr. Rutao Yao** | PI, Senior Member IEEE | Directs research, defined SC-SPECT concept, senior author on TMI 2021 paper |
| **Omer Shah** | Graduate RA (this work) | BO implementation, pipeline engineering, GPU acceleration |
| **Kirtiraj** | Labmate | Built original BO pipeline for MPH SPECT (pinhole diam + displacement). Template for Omer's work. |
| **Harsh Tripathi** | Labmate | Defined JI metric (SNMMI 2026 abstract). Validated JI-optimal = best reconstruction quality. |
| **Soham** | Labmate | Also working on BO optimization |
| **Tianyu Ma** | TMI 2021 first author | Designed MATRICES architecture, ran original parameter sweeps |

## 2. Research Goal

Optimize the hardware configuration of a Self-Collimating SPECT (SC-SPECT) system using Bayesian Optimization, replacing the exhaustive grid search used in the 2021 TMI paper. Target: **IEEE MIC 2026 abstract (deadline May 12, 2026)**.

**Key insight from Dr. Yao (2026-04-22):** Reproduce the optimization from the 2021 TMI paper (Figure 3 — sweeping 4 hardware parameters) using our BO pipeline, then identify and apply a different/better optimization algorithm.

## 3. The SC-SPECT System

### 3.1 Self-Collimation Concept (from 2021 TMI Paper)
Unlike conventional SPECT which uses a metal collimator to define photon pathways, SC-SPECT uses **multi-layer interspaced mosaic detectors (MATRICES)** where the detectors themselves act as both collimators and sensors. Photon absorption on one detector forms the collimation for other detectors. This achieves high-resolution collimation without sacrificing sensitivity.

### 3.2 SAI (Small Animal Imaging) System — Our System
| Component | Specification |
|-----------|--------------|
| **Detector rings** | 4 concentric, fixed. Ring 1: 480 crystals (r=263mm), Ring 2: 720 (r=393mm), Ring 3: 960 (r=523mm), Ring 4: 1200 (r=653mm). **Total: 3360 crystals.** |
| **Scintillator crystal** | 0.84mm tangential width, 6.0mm radial thickness, 20.0mm axial length, 0.84mm intra-cell gap |
| **Collimator (HR ring)** | Tungsten, inner r=33.75mm, thickness=2.5mm, outer r=36.25mm |
| **Apertures** | Baseline: 180 apertures, 0.4mm diameter, evenly distributed at r_center ~35mm |
| **FOV** | Circular, 10mm diameter, 200x200 pixels at 0.05mm/pixel |
| **Motion** | Collimator rotates (2 angular positions: 0 deg and 1 deg). T8 protocol: 8 elliptical bed positions (a=0.2mm, b=0.2mm, phase=0 deg) |
| **Files per config** | 2 layouts x 8 T8 poses = 16 HDF5 files |
| **PPDF shape** | (3360 crystals, 40000 pixels) per file |

### 3.3 TMI 2021 Paper — Key Optimization Details

**Paper:** "Self-Collimating SPECT With Multi-Layer Interspaced Mosaic Detectors" (IEEE TMI, Vol. 40, No. 8, August 2021)

**Figure 3 (page 2155)** shows the optimization strategy for the **mouse SPECT** system:
- **Parameters swept (Section II.D, pp. 2154-2155):**
  1. `r` — aperture opening ratio (width of tungsten section between neighboring apertures)
  2. `D_i`, `D_m` — inner diameters of inner and middle detector layers
  3. `r_i`, `r_m` — radial thickness of inner and middle detector layers
  4. `v` — number of crystals in inner and middle layers

- **Optimization approach:** One-at-a-time parameter sweeps. They optimize one parameter at a time, holding others fixed, and iterate:
  - Step 1: Set D_i = 50.4mm, D_m = 80.6mm (inner and middle layers evenly spaced). Sweep r (aperture ratio) and v (number of crystals) in 2D. Select combination with best sensitivity-resolution tradeoff.
  - Step 2: Fix r_i = 1.5mm. Sweep v. Select best.
  - Step 3: Fix chosen v. Sweep r_i and r_m in 2D. Select best.

- **Figure 4 (mouse SPECT):** Shows k vs v curves for different opening ratios. Metric `k = avg deviation bound` which captures resolution-sensitivity tradeoff.

- **Figure 5 (mouse SPECT):** Shows optimization across multiple parameters simultaneously with 2D heatmaps.

- **Key quote (Section IV.C, p. 2165):** "the approach in Section II.D does not guarantee a global optimization of the parameter set {r, D2, D3, t2, t3}. Considering the complicated geometry of a MATRICES SPECT system and the long computational time for matrix inverse, it is extremely hard to find the global optimal solution within the realistic time."

**This is exactly the gap our BO work fills** — the paper acknowledges their grid search doesn't guarantee global optimality, and BO is designed to find near-optimal solutions efficiently.

### 3.4 Physical Constraints
- `aperture_diam < chord_spacing = 2 * r_center * sin(pi / n_apertures)` — apertures can't overlap
- At n=360: max diam ~ 0.61mm
- At n=180: max diam ~ 1.22mm

## 4. Metrics

### 4.1 JI (Joint Index) — Primary BO Objective
```
JI = (sensitivity_mean / FWHM^2) * ASCI_pct / 100
```
- Defined by Harsh Tripathi (SNMMI 2026 abstract)
- Validated: JI-optimal configs produce near-peak CNR in reconstruction
- Same formula used by Kirtiraj for MPH BO

### 4.2 Component Metrics
| Metric | Definition | Better |
|--------|-----------|--------|
| **FWHM** | Beam width from PPDF convex hull arc profiles (mm) | Lower |
| **Sensitivity** | Sum of PPDF values across all crystals per pixel, averaged over FOV | Higher |
| **ASCI** | N_filled_bins / (200*200*360) as percentage | Higher (100% = full angular coverage) |
| **CNR** | Contrast-to-Noise Ratio from ML-EM reconstruction | Higher (>5 = visible) |

### 4.3 TMI Paper Metric
The 2021 TMI paper uses a different metric: **k (average deviation bound)** — captures resolution-sensitivity tradeoff via bias-gradient analysis. They sweep `k` vs `v` curves and look for the parameter value that yields the lowest k at a desired resolution-sensitivity operating point.

## 5. Bayesian Optimization Implementation

### 5.1 Design Vector

#### 5.1a Phase 1: 2D (completed)
| Parameter | Range | Baseline | Physical Effect |
|-----------|-------|----------|----------------|
| `aperture_diam_mm` | [0.2, 1.0] | 0.4 | Resolution-sensitivity tradeoff |
| `n_apertures` | [60, 360] | 180 | Angular sampling density, ASCI |

#### 5.1b Phase 2: 4D (in progress — per Dr. Yao directive, TMI paper Figure 3)
| Parameter | Range | Baseline | TMI Paper Equivalent | Physical Effect |
|-----------|-------|----------|---------------------|----------------|
| `aperture_diam_mm` | [0.2, 1.0] | 0.4 | `v_s` (aperture opening ratio) | Resolution-sensitivity tradeoff |
| `n_apertures` | [60, 360] | 180 | `v` (crystal/aperture count) | Angular sampling density, ASCI |
| `scint_radial_thickness_mm` | [3.0, 12.0] | 6.0 | `r_i`/`r_m` (crystal thickness) | Self-collimation depth, sensitivity |
| `ring_thickness_mm` | [1.0, 5.0] | 2.5 | Collimator geometry | Attenuation, aperture angular width |

### 5.2 BO Configuration
- **GP:** SingleTaskGP + Matern 5/2 kernel + ARD (Automatic Relevance Determination)
- **Acquisition:** qLogExpectedImprovement (q=1, sequential)
- **Initial data:** 25-point LHS (2D) / 50-point LHS (4D)
- **Infeasible handling:** JI=0 replaced with penalty (10% of min feasible JI) so GP learns constraint boundary
- **Normalization:** Input normalized to [0,1], output standardized (mean=0, std=1)

### 5.3 Pipeline Per Configuration
```
(aperture_diam, n_apertures, scint_radial_thickness, ring_thickness)
  -> Step 0: Geometry generation (~1 min) — all 4 params passed as CLI args
  -> Step 1: PPDF ray-tracing (16 poses, ~2h with parallel execution)
  -> Step 2: Beam analysis (~5 min) - masks, properties, ASCI
  -> Step 3: Compute JI -> append to results_summary.csv
```

## 6. Results (as of 2026-04-19)

### 6.1 LHS Phase (25 configs)
- 22 feasible, 3 infeasible (configs 10, 11, 20 — aperture too wide for chord spacing)
- Best LHS: lhs_8 (diam=0.694, n=191), JI = 1.290

### 6.2 BO Phase (50 iterations)
- Total evaluations: 76 (26 LHS rows including 4 duplicates + 50 BO)
- 5 additional infeasible configs discovered by BO at boundaries
- **Best config found at iteration 13:**

| Metric | Value |
|--------|-------|
| Config | bo_0013_ap0.5300_nap232 |
| aperture_diam | 0.530 mm |
| n_apertures | 232 |
| FWHM | 0.502 mm |
| Sensitivity | 0.361 |
| ASCI | 100% |
| **JI** | **1.432** |

- GP converged by iteration ~20 (proposals varied by <0.001mm after that)
- ARD lengthscales: n_apertures (0.147) more sensitive than aperture_diam (0.246)

### 6.3 Baseline Comparison
| Metric | Baseline (d=0.4, n=180) | BO-Optimized (d=0.53, n=232) | Change |
|--------|------------------------|------------------------------|--------|
| FWHM | 0.489 mm | 0.502 mm | +2.7% |
| Sensitivity | 0.187 | 0.361 | **+93%** |
| ASCI | 100% | 100% | same |
| **JI** | **0.783** | **1.432** | **+83%** |

**Key finding:** BO sacrificed 2.7% resolution for 93% sensitivity gain, yielding 83% higher JI.

### 6.4 Convergence Plots
Generated and saved at `optimization/plots/convergence/`:
- `convergence.png` — Best JI vs iteration + per-evaluation scatter
- `design_space.png` — Config scatter with feasibility boundary
- `gp_surface.png` — GP posterior mean + uncertainty heatmaps

## 7. GPU Acceleration

### 7.1 CUDA Kernel
- Repo: https://github.com/Syedomershah99/gpu-ppdf-sai
- One thread per (crystal, pixel) pair — 3360 * 40000 = 134.4M threads
- Bounding-box edge culling (GPU-friendly alternative to CPU's convex hull)
- Z dimension set to 1 for 2D-to-3D conversion

### 7.2 Benchmarks
| Platform | Time/pose | Time/config (16 poses) | Speedup |
|----------|-----------|----------------------|---------|
| CPU (25 cores) | ~15 min | ~2 hours | 1x |
| Colab T4 GPU | 51.2s | ~13.7 min | **9x** |
| CCR H100/H200 | TBD | TBD (expected 5-10x more) | ~50-90x est. |

## 8. Codebase

### 8.1 Repository Structure
```
/Users/Omer/Desktop/RA/           (workspace root, NOT a git repo)
  omer/                           -> github.com/Syedomershah99/spebt.git
    spebt/
      geometry/                   Geometry generation
      pymatcal/                   PPDF ray-tracing (local copy)
      pymatana/                   Beam analysis (local copy)
      optimization/               BO pipeline (main work)
      recon/                      ML-EM reconstruction
      data/                       Output data
  pymatcal/                       -> github.com/Syedomershah99/pymatcal.git
  pymatana/                       -> github.com/spebt/pymatana.git
  pyrecon/                        -> github.com/spebt/pyrecon.git
  pydetgen/                       -> github.com/spebt/pydetgen.git
  gpu-ppdf-sai/                   -> github.com/Syedomershah99/gpu-ppdf-sai
  Bayesian-Optimization.../       Kirtiraj's reference BO code
  papers/                         Reference papers
```

### 8.2 Key Files in optimization/
| File | Purpose |
|------|---------|
| `bo_agent.py` | GP + LogEI acquisition, infeasible penalty handling |
| `run_bo_loop.py` | Sequential BO orchestrator with SLURM submission |
| `run_sai_pipeline.sh` | Per-config SLURM pipeline (geom -> PPDF -> beam -> JI) |
| `generate_configs.py` | LHS sampling |
| `compute_ji.py` | JI computation with --force_zero for infeasible configs |
| `submit_lhs_sweep.sh` | Batch LHS job submission |
| `sai_extract_masks.py` | Beam mask extraction (T8 aggregation) |
| `sai_extract_props.py` | Beam property extraction (FWHM, sensitivity) |
| `sai_analyze_asci.py` | ASCI histogram computation |
| `analyze_bo_convergence.py` | Convergence plots, GP surface visualization |
| `cleanup_lhs_results.py` | One-time CSV dedup + infeasible row addition |
| `PLAN_direction1_bo.md` | Full BO plan document |

### 8.3 Data Pipeline
```
pydetgen (layout generation)
  -> pymatcal (PPDF ray-tracing, outputs .hdf5)
    -> pymatana (beam analysis: masks, properties, ASCI)
      -> compute_ji.py (JI metric)
        -> bo_agent.py (GP fit + next candidate)
```

## 9. HPC / CCR Setup

| Item | Value |
|------|-------|
| Cluster | UB-HPC |
| User | syedomer |
| Account | rutaoyao |
| Partition | general-compute |
| QOS | general-compute |
| Code path | /vscratch/grp-rutaoyao/Omer/spebt/spebt/ |
| venv | /vscratch/grp-rutaoyao/Omer/.venv/ |
| PPDF output | /vscratch/grp-rutaoyao/Omer/spebt/optimization/ |
| Max walltime | 72 hours |
| Typical job | 25 CPUs, 40G mem, 6h walltime |

**Gotchas:** vscratch auto-deletes untouched files every 2 months. Always use `${PYTHONPATH:-}` with `set -u`. Create log dirs before sbatch.

## 10. Key Literature

| Paper | Key Finding |
|-------|------------|
| **Ma et al. TMI 2021** | SC-SPECT with MATRICES. Grid search optimization of 4 params. Acknowledges doesn't guarantee global optimum. |
| Van Audenhaege 2015 (PMC5148182) | Aperture diameter = most impactful SPECT parameter |
| Han, Tripathi, Yao JNM 2026 | ASCI as SC-SPECT optimization metric |
| Harsh SNMMI 2026 | JI-optimal = near-peak CNR in reconstruction |
| Furenlid/Barrett (PMC3703762) | Resolution-variance tradeoff framework |
| PMC4337004 | Pinhole geometry = critical factor in image quality |

## 11. Current Task (2026-04-22)

**From Dr. Yao:** Select 4 hardware parameters from Figure 3 of the 2021 TMI paper and reproduce all steps of the current BO process. Then identify and work on a different optimization algorithm.

### What this means:
1. Read the TMI paper's optimization strategy (Section II.D) — they sweep parameters one-at-a-time
2. Identify 4 hardware parameters from Figure 3/4/5 that we can optimize
3. Extend our BO pipeline from 2D to 4D (or build a new one)
4. Run BO on the 4-parameter space
5. Compare BO results against the paper's grid search results
6. Then explore alternative optimization algorithms (e.g., multi-objective, evolutionary, etc.)

### TMI Paper's Parameters (from Section II.D, Figures 3-5):
The paper optimizes for the **mouse SPECT** system:
- `r` (or equivalently `v_s`) — aperture opening ratio / width of tungsten between apertures
- `D_2`, `D_3` — inner diameters of inner and middle detector layers  
- `r_2`, `r_3` (or `t_2`, `t_3`) — radial thickness of inner and middle layers
- `v` — number of crystals in inner and middle layers

**We need to map these to our SAI system parameters** and select 4 that are compatible with our geometry generator and ray-tracing pipeline.

## 12. Decision Log

| Date | Decision | Reason |
|------|----------|--------|
| 2026-03-12 | Shift from PINN to BO optimization | Dr. Yao: too many uncertainties in PINN approach |
| 2026-04-06 | 2D hardware-only design vector | T8 translations not worth BO, batch BO overengineered |
| 2026-04-06 | Sequential q=1, not batch q=8 | Matches Kirtiraj, better sample efficiency |
| 2026-04-13 | Switch qEI to qLogEI | Numerical stability warning from BoTorch |
| 2026-04-13 | Add infeasible penalty to GP | GP was ignoring JI=0 points, kept proposing infeasible configs |
| 2026-04-22 | Extend to 4D per TMI paper Figure 3 | Dr. Yao directive: reproduce paper's optimization with BO |
| 2026-04-22 | 4D params: aperture_diam, n_apertures, scint_radial_thickness, ring_thickness | Option A chosen — maps to TMI paper's v_s, v, r_i/r_m, collimator geometry |
| 2026-04-22 | 50 LHS samples for 4D (up from 25 for 2D) | Higher dimensionality needs more initial data for GP fit |

## 13. Pending / Next Steps

- [x] Identify which 4 parameters from TMI paper map to our SAI system (Option A chosen)
- [x] Extend BO pipeline to 4D (all files updated: geometry generator, bo_agent, generate_configs, run_sai_pipeline, compute_ji, run_bo_loop, submit_lhs_sweep, analyze_bo_convergence)
- [ ] Generate 4D LHS configs (50 samples) and submit sweep on CCR
- [ ] Run 4D BO loop (~100 iterations) and analyze convergence
- [ ] Compare 4D BO results with TMI paper's grid search results
- [ ] Explore alternative optimization algorithms
- [ ] ML-EM reconstruction comparison (baseline vs BO-optimized) using mlem_torch_gpf_nonmpi.py
- [ ] CNR computation (borrow from Kirtiraj's pipeline)
- [ ] Generate CNR comparison figures
- [ ] Draft IEEE MIC abstract (deadline May 12)
