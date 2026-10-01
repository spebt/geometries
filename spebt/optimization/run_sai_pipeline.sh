#!/bin/bash
#SBATCH --job-name=sai_pipeline
#SBATCH --cluster=ub-hpc
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/out/pipeline_%A_%a.out
#SBATCH --error=slurm_logs/err/pipeline_%A_%a.err
#SBATCH --mail-user=syedomer@buffalo.edu
# Mail on FAILURE only, never on END. A campaign submits one of these per
# iteration, so ~480 across six arms, and an inbox full of successes is an inbox
# nobody reads. Failures are rare and worth seeing: since Aug 2026 this script
# exits 1 when the geometry generator CRASHES (as opposed to legitimately
# rejecting an infeasible design, which exits 0 and is data), so a FAIL mail
# means the toolchain is broken.
#SBATCH --mail-type=FAIL,TIMEOUT

set -uo pipefail
# Note: -e intentionally omitted so we can handle errors per-step

# ============================================================
# SAI SC-SPECT per-config pipeline
# Called by run_bo_loop.py or submit_lhs_sweep.sh.
#
# Robustness features:
#   - Infeasible geometry → writes JI=0 to CSV and exits cleanly
#   - Corrupt HDF5 files → detected and deleted before resume
#   - Parallel PPDF poses (12 concurrent) with per-process throttling
#   - Stale beam analysis outputs cleaned before re-run
# ============================================================

# The venv lives in HOME, not /vscratch. Scratch is auto-purged and in
# Aug 2026 it removed .venv/bin/python mid-campaign. Override with
# SPEBT_VENV if the environment moves again.
source "${SPEBT_VENV:-$HOME/spebt-venv}/bin/activate"
export PYTHONPATH="${CODE_DIR}/pymatcal:${PYTHONPATH:-}"
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

A_MM="${A_MM:-0.2}"
B_MM="${B_MM:-0.2}"
SCINT_RADIAL_MM="${SCINT_RADIAL_MM:-6.0}"
RING_THICKNESS_MM="${RING_THICKNESS_MM:-2.5}"
N_DET_RING1="${N_DET_RING1:-480}"
N_DET_RING2="${N_DET_RING2:-720}"
# Inner diameters of detector rings 2 and 3 (MOBO design variables from the D2/D3
# expansion). Defaults reproduce the original fixed [260, 390, 520, 650] layout,
# so a run without these set behaves exactly as before.
D2_INNER="${D2_INNER:-390.0}"
D3_INNER="${D3_INNER:-520.0}"
MAX_PARALLEL=16
# Phantom + ML-EM settings for the in-loop CNR step (Section 4 below).
# Default matches the current 3-spebt repo layout on CCR after the recovery.
PHANTOM_PATH="${PHANTOM_PATH:-/vscratch/grp-rutaoyao/Omer/spebt/spebt/spebt/data/sai_10mm/hot_rods_phantom_10.0_mm_x_10.0_mm.pt}"
CNR_ITERATIONS="${CNR_ITERATIONS:-150}"

mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

echo "=================================================="
echo "SAI Pipeline | $(date)"
echo "Config: ${CONFIG_NAME}"
echo "  aperture_diam    = ${APERTURE_DIAM} mm"
echo "  n_apertures      = ${N_APERTURES}"
echo "  n_det_ring1      = ${N_DET_RING1}"
echo "  n_det_ring2      = ${N_DET_RING2}"
echo "  d2_inner         = ${D2_INNER} mm"
echo "  d3_inner         = ${D3_INNER} mm"
echo "  scint_radial_mm  = ${SCINT_RADIAL_MM} mm (fixed)"
echo "  ring_thickness   = ${RING_THICKNESS_MM} mm (fixed)"
echo "  a_mm=${A_MM}  b_mm=${B_MM}"
echo "  work_dir = ${WORK_DIR}"
echo "  CPUs = ${SLURM_CPUS_PER_TASK}"
echo "=================================================="

# -------------------------------------------------------
# Helper: write JI=0 for infeasible/failed configs
# -------------------------------------------------------
write_zero_ji() {
  local reason="$1"
  echo "[INFEASIBLE] ${reason}"
  echo "  Writing JI=0 to results CSV..."
  python "${CODE_DIR}/optimization/compute_metrics.py" \
    --work_dir "${WORK_DIR}" \
    --out_csv "${RESULTS_CSV}" \
    --config_name "${CONFIG_NAME}" \
    --aperture_diam_mm "${APERTURE_DIAM}" \
    --n_apertures "${N_APERTURES}" \
    --n_det_ring1 "${N_DET_RING1}" \
    --n_det_ring2 "${N_DET_RING2}" \
    --d2_inner_mm "${D2_INNER}" \
    --d3_inner_mm "${D3_INNER}" \
    --force_zero --reason "${reason}"
  # Also write NaN for cnr_mean so the CSV row is complete for MOBO
  python "${CODE_DIR}/optimization/compute_cnr.py" \
    --work_dir "${WORK_DIR}" \
    --phantom_path "${PHANTOM_PATH}" \
    --out_csv "${RESULTS_CSV}" \
    --config_name "${CONFIG_NAME}" \
    --force_nan --reason "${reason}"
  echo "=================================================="
  echo "PIPELINE COMPLETE (infeasible) | $(date)"
  echo "=================================================="
  exit 0
}

# -------------------------------------------------------
# Step 0: Generate geometry (skip if .tensor already exists)
# -------------------------------------------------------
shopt -s nullglob
TENSORS=("${WORK_DIR}"/*.tensor)
if [ ${#TENSORS[@]} -gt 0 ]; then
  TENSOR_FILE="${TENSORS[0]}"
  echo "[0/4] Geometry already exists: ${TENSOR_FILE}"
else
  echo "[0/4] Generating scanner geometry..."
  if ! python "${CODE_DIR}/geometry/generate_mph_scanner_circularfov.py" \
    --aperture_diam "${APERTURE_DIAM}" \
    --n_apertures "${N_APERTURES}" \
    --scint_radial_mm "${SCINT_RADIAL_MM}" \
    --ring_thickness "${RING_THICKNESS_MM}" \
    --n_det_ring1 "${N_DET_RING1}" \
    --n_det_ring2 "${N_DET_RING2}" \
    --d2_inner "${D2_INNER}" \
    --d3_inner "${D3_INNER}" \
    --output_dir "${WORK_DIR}" 2>&1 | tee "${WORK_DIR}/geometry_gen.log"; then
    # Distinguish a genuinely infeasible DESIGN from a broken TOOLCHAIN. Both
    # exit non-zero, and treating them alike is how a purged helper.py cost four
    # campaigns 30-50% of their iterations for five days: every crash was
    # recorded as "aperture too wide or ring ordering violated" and written to
    # the archive as an infeasible design.
    #
    # An infeasible design is data. A crash is a lost iteration, and the run
    # must stop rather than poison the archive with a NaN row that the optimizer
    # will read as "this region is bad".
    if grep -qE "ModuleNotFoundError|ImportError|No such file or directory|SyntaxError|Traceback" \
         "${WORK_DIR}/geometry_gen.log"; then
      echo "[FATAL] The geometry generator CRASHED; it did not reject the design." >&2
      echo "        This is a broken environment, not an infeasible geometry." >&2
      echo "        Not writing a NaN row -- that would teach the optimizer this" >&2
      echo "        design region is bad when it was never evaluated." >&2
      sed -n "1,40p" "${WORK_DIR}/geometry_gen.log" >&2
      exit 1
    fi
    write_zero_ji "Geometry generation failed (aperture too wide for n_apertures, or ring ordering violated)"
  fi

  TENSORS=("${WORK_DIR}"/*.tensor)
  if [ ${#TENSORS[@]} -eq 0 ]; then
    write_zero_ji "No .tensor file produced"
  fi
  TENSOR_FILE="${TENSORS[0]}"
fi
echo "  Tensor file: ${TENSOR_FILE}"

# -------------------------------------------------------
# Step 1: PPDF computation (2 layouts × 8 T8 poses = 16 files)
#   - Validates existing HDF5 files (deletes corrupt ones)
#   - Parallelizes up to MAX_PARALLEL poses
# -------------------------------------------------------
echo "[1/4] Computing PPDFs (2 layouts × 8 T8 poses, parallel)..."

# Validate existing HDF5 files — delete corrupt ones
echo "  Checking existing HDF5 integrity..."
python3 -c "
import h5py, glob, os
for f in glob.glob('${WORK_DIR}/position_*_ppdfs_t8_*.hdf5'):
    try:
        with h5py.File(f, 'r') as h:
            _ = h['ppdfs'].shape
    except:
        print(f'  Deleting corrupt: {os.path.basename(f)}')
        os.remove(f)
"

n_running=0
n_skipped=0
n_launched=0
PIDS=()

for layout_idx in 0 1; do
  for pose_idx in $(seq 0 7); do
    OUT_FILE="${WORK_DIR}/position_$(printf '%03d' ${layout_idx})_ppdfs_t8_$(printf '%02d' ${pose_idx}).hdf5"

    if [ -f "${OUT_FILE}" ]; then
      n_skipped=$((n_skipped + 1))
      continue
    fi

    # Throttle
    while [ ${n_running} -ge ${MAX_PARALLEL} ]; do
      wait -n 2>/dev/null || true
      n_running=$((n_running - 1))
    done

    echo "  Launching layout=${layout_idx} pose=${pose_idx}..."
    python "${CODE_DIR}/pymatcal/arg_ppdf_t8.py" \
      "${layout_idx}" \
      --layout_file "${TENSOR_FILE}" \
      --output_dir "${WORK_DIR}" \
      --a_mm "${A_MM}" \
      --b_mm "${B_MM}" \
      --pose_idx "${pose_idx}" &

    PIDS+=($!)
    n_running=$((n_running + 1))
    n_launched=$((n_launched + 1))
  done
done

echo "  Launched ${n_launched} poses, skipped ${n_skipped} (already exist)"
echo "  Waiting for all PPDF processes to finish..."

FAIL=0
for pid in "${PIDS[@]}"; do
  wait "${pid}" || FAIL=$((FAIL + 1))
done

if [ ${FAIL} -gt 0 ]; then
  echo "[WARNING] ${FAIL} PPDF process(es) failed"
fi

# Verify 16 HDF5 files
N_HDF5=$(ls "${WORK_DIR}"/position_*_ppdfs_t8_*.hdf5 2>/dev/null | wc -l)
echo "  Total PPDF files: ${N_HDF5} (expected 16)"
if [ "${N_HDF5}" -lt 16 ]; then
  write_zero_ji "Only ${N_HDF5}/16 PPDF files produced"
fi

echo "  Step 1 complete at $(date)"

# -------------------------------------------------------
# Step 2: Beam analysis (masks, properties, ASCI)
# Clean stale outputs first, then run per-layout
# -------------------------------------------------------
echo "[2/4] Beam analysis (masks -> properties -> ASCI)..."
export PYTHONPATH="${CODE_DIR}/pymatana/ppdf-analysis/beam-analysis:${PYTHONPATH:-}"

# Remove stale beam analysis files (force fresh computation)
rm -f "${WORK_DIR}"/beams_masks_configuration_*.hdf5
rm -f "${WORK_DIR}"/beams_properties_configuration_*.hdf5
rm -f "${WORK_DIR}"/asci_histogram_*.hdf5

# The three steps are strictly sequential per layout -- properties need masks,
# ASCI needs both -- so `set -e` inside the subshell stops a failure from
# cascading into garbage downstream. Exit codes are collected per layout,
# because a bare `wait` discards them and the metric aggregation downstream
# globs "whatever files exist", which would silently compute FWHM/ASCI/MPXI
# from one layout instead of two.
BEAM_PIDS=()
for layout_idx in 0 1; do
  (
    set -e
    echo "  Layout ${layout_idx}: extracting masks..."
    python "${CODE_DIR}/optimization/sai_extract_masks.py" \
      --layout_idx "${layout_idx}" \
      --work_dir "${WORK_DIR}" \
      --tensor_file "${TENSOR_FILE}"

    echo "  Layout ${layout_idx}: extracting properties..."
    python "${CODE_DIR}/optimization/sai_extract_props.py" \
      --layout_idx "${layout_idx}" \
      --work_dir "${WORK_DIR}" \
      --tensor_file "${TENSOR_FILE}"

    echo "  Layout ${layout_idx}: computing ASCI histogram..."
    python "${CODE_DIR}/optimization/sai_analyze_asci.py" \
      --layout_idx "${layout_idx}" \
      --work_dir "${WORK_DIR}"
  ) &
  BEAM_PIDS+=("$!")
done

BEAM_FAIL=0
for pid in "${BEAM_PIDS[@]}"; do
  wait "${pid}" || BEAM_FAIL=$((BEAM_FAIL + 1))
done
if [ "${BEAM_FAIL}" -gt 0 ]; then
  write_zero_ji "${BEAM_FAIL} of 2 beam-analysis layouts failed"
fi

for layout_idx in 0 1; do
  for f in "beams_masks_configuration" "beams_properties_configuration" "asci_histogram"; do
    if [ ! -f "${WORK_DIR}/${f}_$(printf '%03d' "${layout_idx}").hdf5" ]; then
      write_zero_ji "Missing ${f}_$(printf '%03d' "${layout_idx}").hdf5 after beam analysis"
    fi
  done
done

# -------------------------------------------------------
# Step 3: Compute metrics (FWHM, ASCI, sensitivity, MPXI, PPDS) and append to CSV
# -------------------------------------------------------
echo "[3/4] Computing metrics..."
python "${CODE_DIR}/optimization/compute_metrics.py" \
  --work_dir "${WORK_DIR}" \
  --out_csv "${RESULTS_CSV}" \
  --config_name "${CONFIG_NAME}" \
  --aperture_diam_mm "${APERTURE_DIAM}" \
  --n_apertures "${N_APERTURES}" \
  --n_det_ring1 "${N_DET_RING1}" \
  --n_det_ring2 "${N_DET_RING2}" \
  --d2_inner_mm "${D2_INNER}" \
  --d3_inner_mm "${D3_INNER}"

# -------------------------------------------------------
# Step 4: In-loop CNR — forward-project + ML-EM + CNR, append to CSV row
# Adds ~5-10 min per config on CPU (much less on GPU). Uses 150 ML-EM
# iterations by default (see CNR_ITERATIONS env var), which is enough for
# ranking; 500 is only needed for clean publication images.
#
# --seed_from_config derives the Poisson noise seed from the config name.
# Before Sep 2026 this step ran unseeded, so every archive CNR was a single
# irreproducible draw (sigma ~= 0.08) and no row could be re-derived from its
# work_dir. Different configs still get independent draws, so the GP sees the
# same honest across-design noise it always did; only re-running one config
# is now deterministic. Rows written earlier keep their unseeded values.
# -------------------------------------------------------
echo "[4/4] Computing in-loop CNR (${CNR_ITERATIONS} ML-EM iterations)..."
python "${CODE_DIR}/optimization/compute_cnr.py" \
  --work_dir "${WORK_DIR}" \
  --phantom_path "${PHANTOM_PATH}" \
  --out_csv "${RESULTS_CSV}" \
  --config_name "${CONFIG_NAME}" \
  --iterations "${CNR_ITERATIONS}" \
  --seed_from_config

echo "=================================================="
echo "PIPELINE COMPLETE | $(date)"
echo "=================================================="
