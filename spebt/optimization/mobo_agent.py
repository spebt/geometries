#!/usr/bin/env python3
"""
Multi-Objective Bayesian Optimization Agent for SAI SC-SPECT.

The objective set is defined ONCE below, in OBJ_COLUMNS / OBJ_DIRECTIONS /
OBJ_NAMES / OBJ_SHORT, and is deliberately not restated here. This docstring
used to enumerate it and went stale twice: it still named sensitivity months
after that objective was retired for correlating -0.92 with reconstructed CNR,
and still described MPXI as minimized after the sign was reversed. A stale list
in a docstring is read as documentation and believed.

Read the code, and see the comment block above OBJ_COLUMNS for why each
objective is what it is. MOBO_OBJECTIVES can restrict the set at runtime for
head-to-head comparisons.

CNR is the outcome objective: reconstructed contrast-to-noise from the in-loop
150-iteration ML-EM run on the hot-rod phantom, computed by compute_cnr.py and
appended by run_sai_pipeline.sh step [4/4]. It exists because the cheap proxy
metrics do not always align with reconstructed image quality, which is the
lesson this file's history mostly records.

Uses ModelListGP (one SingleTaskGP per objective) + qLogNEHVI.

Design vector: (aperture_diam_mm, n_apertures, n_det_ring1, n_det_ring2,
                d2_inner_mm, d3_inner_mm)

Usage:
  python mobo_agent.py --results_csv results/results_summary.csv
"""
import math
import os
import logging
import warnings
import torch
import pandas as pd
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import normalize, unnormalize
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MOBO_Agent_SAI")
# Selective warning suppression: BoTorch emits an InputDataWarning every fit
# because our train_x is normalised to [0,1] (which is what it wants but the
# check misfires). Everything else — numerical warnings from GPyTorch about
# jitter or ill-conditioning — we want to see.
warnings.filterwarnings(
    "ignore",
    message="Data (input features) is not contained to the unit cube.*",
    category=UserWarning,
)

# --- Design space bounds ---
# 6D: aperture_diam, n_apertures, n_det_ring1, n_det_ring2, d2_inner, d3_inner
# n_det values must be even (2 crystals per cell). Rounded after acquisition.
# n_apertures max: geometry generator enforces MIN_SPACING=0.8mm between
# aperture centers → chord = 2*R*sin(π/n) >= 0.8 → n_max ≈ 274 at R=35mm.
# Use 270 with safety margin.
# d2_inner / d3_inner are the inner diameters of detector rings 2 and 3; rings 1
# and 4 stay fixed at 260 / 650 mm.
# d2 lower / d3 upper bounds come from RADIAL clearance against the fixed rings:
# a ring occupies r_in .. r_in + 6 mm, so 10 mm of clearance past ring 1 needs
# d2 >= 260 + 2*(6+10) = 292, and clearance before ring 4 needs d3 <= 618.
# The earlier values (270, 640) were derived from a diameter difference rather
# than a radial gap and admitted designs whose rings interpenetrated by 1 mm.
#
# d3's lower bound is set by something else entirely: ring 3 carries a fixed 960
# crystals, which do not fit below ~379 mm (see max_crystals_on_ring). 385 keeps
# a small margin over that hard floor.
#
# n_det_ring2's upper bound was 960, matching ring 3's fixed count. That number
# was never physical, and it was binding: EVERY top design in the 271-config
# archive sat exactly on it, which is the signature of a search constrained by
# the box rather than converged. The real limit is the packing constraint, and
# it is well above 960 across the whole d2 range:
#
#     d2 = 292 mm  ->  743 crystals        d2 = 450 mm  ->  1137
#     d2 = 400 mm  ->  1012                d2 = 540 mm  ->  1361
#
# Raised to 1362 so is_ring_packing_ok binds instead of the box. That is the
# packing limit at the largest allowed d2 (1361.4) rounded up to the next even
# count, since cells hold two crystals. 1360 would still have clipped the very
# top of the range by 1.4 crystals, which is exactly the kind of not-quite-right
# bound this change exists to remove. At the d2 ~ 400 pocket where the best
# designs live the effective ceiling becomes ~1010, so this opens roughly 5%
# more crystals there and much more at larger d2.
#
# NOTE FOR RY: the real hardware is 480/720/960/1200, so ring 2 carrying more
# than 960 is a design-study question rather than a build-ready one. Worth
# confirming there is no manufacturing reason for the old cap.
PARAM_NAMES = ["aperture_diam_mm", "n_apertures", "n_det_ring1", "n_det_ring2",
               "d2_inner_mm", "d3_inner_mm"]
BOUNDS_MIN = [0.2, 60.0, 120.0, 180.0, 292.0, 385.0]
BOUNDS_MAX = [1.0, 270.0, 660.0, 1362.0, 540.0, 618.0]
DIM = len(PARAM_NAMES)

# --- Feasibility constraints ---
# 1) Aperture overlap: aperture_diam < circumference / n_apertures
HR_R_CENTER = 67.5 / 2.0 + 2.5 / 2.0  # 35.0 mm
HR_CIRCUMFERENCE = 2.0 * math.pi * HR_R_CENTER  # ~219.9 mm
SAFETY_MARGIN = 0.95

# 2) Detector ring ordering: D1 < D2 < D3 < D4 with a minimum radial gap.
# RY (Jul 2026): 10 mm is reasonable given mechanical mounting and cooling.
# Rings stay concentric and share an axial extent — he wants the method
# established before more variables are added, so there is no axial term here.
D1_INNER_MM = 260.0
D4_INNER_MM = 650.0
MIN_RING_GAP_MM = 10.0

# Minimum separation between two rings' INNER DIAMETERS that yields
# MIN_RING_GAP_MM of radial clearance. A ring occupies r_in .. r_in + 6 mm, so
# clearing the previous ring costs 6 mm of radius plus the gap, and diameters
# are twice radii. Derived rather than written as a literal, because having the
# radial rule in one place and a hand-computed diameter threshold in another is
# exactly how the two drifted apart before.
MIN_DIAM_SEPARATION_MM = 2.0 * (6.0 + MIN_RING_GAP_MM)   # 32 mm

# 3) Detector cell packing. Each ring holds n_scint/2 cells, and every cell needs
# 2*W + gap of arc length. Shrinking a ring while keeping its crystal count
# raises the packing density, so diameter and detector count interact: the
# geometry generator raises ValueError once clearance reaches zero, which would
# burn a whole MOBO iteration on a config that could never be built. These
# constants mirror build_sc_spect_detector_rings in the geometry generator.
SCINT_TANGENTIAL_MM = 0.84
INTRA_CELL_GAP_MM = 0.84
SCINT_RADIAL_MM = 6.0
CELL_SPAN_MM = 2.0 * SCINT_TANGENTIAL_MM + INTRA_CELL_GAP_MM  # 2.52 mm
# Rings 3 and 4 carry fixed crystal counts (geometry generator: 40*12*2, 40*15*2)
N_DET_RING3 = 960
N_DET_RING4 = 1200


def is_feasible(diam, n_ap):
    """Check if aperture config fits on the HR ring without overlap."""
    return diam < SAFETY_MARGIN * HR_CIRCUMFERENCE / n_ap


def _ring_radial_span(inner_diam_mm):
    """(inner, outer) radius of a ring's scintillator layer."""
    r_in = inner_diam_mm / 2.0
    return r_in, r_in + SCINT_RADIAL_MM


def is_ring_ordering_ok(d2_inner, d3_inner):
    """Check the rings are ordered with MIN_RING_GAP_MM of RADIAL clearance.

    This compares radial spans, not inner diameters. The earlier version tested
    `d2_inner - D1_INNER_MM >= 10`, which is a difference of DIAMETERS: 10 mm of
    diameter is only 5 mm of radius, and each crystal is 6 mm deep, so rings
    satisfying it interpenetrated by 1 mm. The geometry generator did not catch
    this either -- its overlap check is tangential (arc length per cell within a
    ring) and nothing tested radial separation between rings.

    Clearance is measured between the outer surface of the inner ring and the
    inner surface of the next one out, which is the gap an engineer would need
    for mounting and cooling.
    """
    r1_out = _ring_radial_span(D1_INNER_MM)[1]
    r2_in, r2_out = _ring_radial_span(d2_inner)
    r3_in, r3_out = _ring_radial_span(d3_inner)
    r4_in = _ring_radial_span(D4_INNER_MM)[0]
    return (
        r2_in - r1_out >= MIN_RING_GAP_MM
        and r3_in - r2_out >= MIN_RING_GAP_MM
        and r4_in - r3_out >= MIN_RING_GAP_MM
    )


def max_crystals_on_ring(inner_diam_mm):
    """Largest crystal count a ring of this diameter can hold without overlap."""
    r_center = inner_diam_mm / 2.0 + SCINT_RADIAL_MM / 2.0
    return 4.0 * math.pi * r_center / CELL_SPAN_MM


def is_ring_packing_ok(n_det1, n_det2, d2_inner, d3_inner):
    """Check every ring's crystals fit at its diameter.

    Ring 1 is fixed at 260 mm, which caps n_det_ring1 at ~663 -- the reason the
    existing bound is 660. Ring 4 is fixed in both diameter and count. Only
    ring 2 (both variable) and ring 3 (fixed count, variable diameter) can be
    driven infeasible by the D2/D3 expansion.
    """
    return (
        n_det1 < max_crystals_on_ring(D1_INNER_MM)
        and n_det2 < max_crystals_on_ring(d2_inner)
        and N_DET_RING3 < max_crystals_on_ring(d3_inner)
        and N_DET_RING4 < max_crystals_on_ring(D4_INNER_MM)
    )


def report_out_of_bounds(train_x) -> int:
    """Log every training point outside the current bounds; return how many.

    Returns the count of ROWS with at least one out-of-range parameter, not the
    count of violations, so it can be compared against len(train_x).
    """
    lo = torch.tensor(BOUNDS_MIN, dtype=torch.double)
    hi = torch.tensor(BOUNDS_MAX, dtype=torch.double)
    below, above = train_x < lo, train_x > hi
    bad_rows = (below | above).any(dim=1)
    n_bad = int(bad_rows.sum())
    if n_bad:
        for j, name in enumerate(PARAM_NAMES):
            col = train_x[:, j]
            n_lo, n_hi = int((col < lo[j]).sum()), int((col > hi[j]).sum())
            if n_lo or n_hi:
                logger.warning(
                    f"  {name}: {n_lo} below {BOUNDS_MIN[j]}, {n_hi} above "
                    f"{BOUNDS_MAX[j]} (observed {col.min():.4g} to {col.max():.4g})")
    return n_bad


def is_feasible_full(diam, n_ap, n_det1, n_det2, d2_inner, d3_inner):
    """All design constraints together."""
    return (
        is_feasible(diam, n_ap)
        and is_ring_ordering_ok(d2_inner, d3_inner)
        and is_ring_packing_ok(n_det1, n_det2, d2_inner, d3_inner)
    )


# --- Objective columns (as they appear in the CSV) ---
# Revised Jul 2026 after measuring every metric against CNR across 109 designs.
# Two of the previous five were being MAXIMISED while pulling against image
# quality; both were replaced (Spearman against section-mean CNR in brackets):
#
#   sensitivity_mean (-0.92)  ->  ppds_ring1 (+0.60)
#       Overall sensitivity rewards wide, poorly collimated PPDFs. Per-ring PPDS
#       showed the inner ring is the only one that helps; ring 4 is at -0.82 and
#       is largely re-measuring aperture size.
#   asci_pct (-0.75)          ->  asci_pct_fwhm0p45 (+0.80)
#       Unwindowed ASCI saturates at 100% for 64% of designs and was already
#       maxed before optimization began. Restricting to beams under 0.45 mm
#       de-saturates it completely.
#   cnr_mean                  ->  cnr_sector_mean
#       Equal weight per rod size rather than pooling all hot pixels, which was
#       area-weighted and let the largest rods dominate. Ranks designs almost
#       identically (Spearman 0.999) but is the defensible definition.
#
#   mpxi_mean (min)  ->  mpxi_windowed_active_mean (MAX)   [RY approved Aug 2026]
#       Two separate errors, found together.
#
#       DIRECTION. MPXI was minimized on the assumption that multiplexing costs
#       image quality. Measured on 228 designs, in physical units, the opposite
#       holds: multiplexing correlates +0.29 with CNR, +0.71 with windowed ASCI,
#       and -0.43 with FWHM. More splitting means more angular sampling and
#       narrower stripes. We were minimizing a quantity every measurement says
#       to increase. RY predicted this from the physics before it was measured.
#
#       DEFINITION. The old count included every beam, while windowed ASCI counts
#       only beams under 0.45 mm, so the two were measured over different beam
#       populations. Matching the window and the sensitivity floor strengthens
#       every relationship: +0.84 vs ASCI, +0.47 vs CNR, -0.62 vs FWHM. Also
#       restricting to detectors that see something (blind detectors contributed
#       k=0 to a then-minimized mean) gives +0.91, +0.60, -0.73.
#
#       Note the active-detector fix ALONE makes things worse (+0.71 -> +0.60 vs
#       ASCI): blind detectors carry real information in an unwindowed count,
#       since a design with many of them is a poor design. The two changes are
#       only correct together.
#
#       Consequence: corrected MPXI is no longer independent of the other
#       objectives -- it is now among the most redundant. Better metric, worse
#       fifth objective. See analyze_objective_redundancy.py.
#   fwhm_mean (-0.83)         ->  fwhm_weighted_mean (-0.93)
#       The unweighted mean counted a beam carrying 0.1% of the signal as much
#       as one carrying all of it. Worse, its gap from the sensitivity-filtered
#       mean tracks aperture diameter at -0.69, so it carried an
#       aperture-dependent artifact -- and aperture diameter is itself the
#       strongest CNR predictor. Weighting each width by that beam's sensitivity
#       removes the artifact and predicts CNR better.
# THE definition of the objective set. Other modules import these rather than
# restating them: run_mobo_loop.py and analyze_mobo_convergence.py both used to
# carry their own copy under a "keep in sync" comment, and a duplicated rule
# that drifted is what made 46 consecutive iterations propose an unbuildable
# geometry in Jul 2026. One copy, imported.
_ALL_OBJ_COLUMNS = ["fwhm_weighted_mean", "asci_pct_fwhm0p45", "ppds_ring1",
                    "mpxi_windowed_active_mean", "cnr_sector_mean"]
# Directions: +1 = maximize, -1 = minimize (we negate minimization objectives)
_ALL_OBJ_DIRECTIONS = [-1.0, 1.0, 1.0, 1.0, 1.0]
_ALL_OBJ_NAMES = ["FWHM weighted (min)", "ASCI@0.45mm (max)", "PPDS ring1 (max)",
                  "MPXI windowed+active (max)", "CNR sector-mean (max)"]
# Short labels for status tables and plots, in OBJ_COLUMNS order.
_ALL_OBJ_SHORT = ["FWHM wtd (mm)", "ASCI@0.45mm (%)", "PPDS ring1",
                  "MPXI win+act", "CNR sector-mean"]

assert (len(_ALL_OBJ_COLUMNS) == len(_ALL_OBJ_DIRECTIONS)
        == len(_ALL_OBJ_NAMES) == len(_ALL_OBJ_SHORT))


def _select_objectives():
    """Restrict the objective set via MOBO_OBJECTIVES, for head-to-head campaigns.

    Set it to a comma-separated list of column names to run a campaign on a
    subset, e.g.

        MOBO_OBJECTIVES=cnr_sector_mean,mpxi_windowed_active_mean

    Unset means all five. Direction, display name and short label are carried
    across from the master lists, so a subset campaign cannot end up with a
    label that disagrees with its own sign.

    Pair it with MOBO_RESULTS_DIR so the subset campaign writes its own
    manifest, results CSV, lock file and logs. Two campaigns sharing a results
    directory would interleave rows and claim the same manifest indices.
    """
    sel = os.environ.get("MOBO_OBJECTIVES", "").strip()
    if not sel:
        return list(range(len(_ALL_OBJ_COLUMNS)))

    wanted = [s.strip() for s in sel.split(",") if s.strip()]
    unknown = [w for w in wanted if w not in _ALL_OBJ_COLUMNS]
    if unknown:
        raise SystemExit(
            f"\nMOBO_OBJECTIVES names unknown objective(s): {unknown}\n"
            f"Valid names: {_ALL_OBJ_COLUMNS}\n")
    if len(set(wanted)) != len(wanted):
        raise SystemExit(f"\nMOBO_OBJECTIVES repeats an objective: {wanted}\n")
    if len(wanted) < 2:
        # The controller optimizes qLogNEHVI, whose hypervolume is undefined for
        # a single objective. It raises "ref_point has length 1" deep inside
        # BoTorch, which killed a 5-hour job before this check existed.
        raise SystemExit(
            f"\nMOBO_OBJECTIVES needs at least 2 objectives, got {len(wanted)}.\n"
            f"The controller uses qLogNEHVI and hypervolume is undefined for one\n"
            f"objective. For a single-objective run the acquisition would have to\n"
            f"change to qLogNEI.\n")
    return [_ALL_OBJ_COLUMNS.index(w) for w in wanted]


_OBJ_IDX = _select_objectives()
OBJ_COLUMNS = [_ALL_OBJ_COLUMNS[i] for i in _OBJ_IDX]
OBJ_DIRECTIONS = [_ALL_OBJ_DIRECTIONS[i] for i in _OBJ_IDX]
OBJ_NAMES = [_ALL_OBJ_NAMES[i] for i in _OBJ_IDX]
OBJ_SHORT = [_ALL_OBJ_SHORT[i] for i in _OBJ_IDX]


def require_objective_columns(df, results_csv: str):
    """Fail with an actionable message when the archive predates an objective change.

    Changing OBJ_COLUMNS makes every previously-written row missing the new
    column, and pandas' dropna(subset=...) then raises a bare KeyError naming
    the column but not what to do about it. Anyone hitting that mid-campaign
    needs to know it is a backfill step, not a corrupted archive.
    """
    missing = [c for c in OBJ_COLUMNS if c not in df.columns]
    if not missing:
        return
    raise SystemExit(
        f"\n{results_csv} has no column(s): {missing}\n"
        f"\nThe objective set references a metric this archive predates. The"
        f"\narchive is fine; it just needs the column filled in. For the Aug 2026"
        f"\nMPXI change that is:\n"
        f"\n  python analyze_mpxi_variants.py --results_csv {results_csv} \\"
        f"\n      --out results/mpxi_variants.csv"
        f"\n  python backfill_mpxi_variants.py --results_csv {results_csv} \\"
        f"\n      --variants_csv results/mpxi_variants.csv\n"
        f"\nRun the backfill with --dry_run first to see the coverage.\n")


def get_next_candidate(results_csv: str):
    """
    1. Load results CSV with all OBJ_COLUMNS objectives.
    2. Fit ModelListGP (one GP per objective).
    3. Optimize qLogNEHVI, q=1.
    4. Return next design point.
    """
    logger.info("--- Starting Multi-Objective BO Step ---")

    if not os.path.exists(results_csv):
        raise FileNotFoundError(f"Could not find {results_csv}")

    df = pd.read_csv(results_csv)
    require_objective_columns(df, results_csv)
    # Unfiltered copy: every design ever attempted, valid or not. Used only for
    # deduplication, so a design that failed still counts as "already tried".
    df_all = df.copy()

    # Separate feasible, genuinely-failed, and partially-measured rows.
    #
    # A row missing SOME objectives is not the same as a failed design. Configs
    # evaluated before in-loop CNR existed have real FWHM, ASCI, MPXI and PPDS
    # but no reconstruction, so they carry four good measurements and one gap.
    # Treating them as failures fabricated penalty values for the four we
    # actually measured -- at one point 67 of 176 rows had a real FWHM near
    # 0.5 mm replaced by the penalty value of 9.84, i.e. over a third of the
    # FWHM surrogate's training data was invented.
    #
    # Only rows with EVERY objective missing are genuine failures: the
    # force_zero path in run_sai_pipeline.sh writes NaN across the board when
    # geometry cannot be built, and that absence really is information.
    # Drop rows whose geometry cannot physically be built. Configs proposed
    # before the ring-clearance fix used a diameter comparison rather than a
    # radial one, so some have rings interpenetrating by ~1 mm. The generator
    # built them anyway and the ray tracer modelled absorbers occupying the same
    # space, which makes their PPDFs wrong rather than merely unattractive.
    # Filtering here rather than editing the CSV keeps the record intact and
    # covers any future bounds change automatically.
    if {"d2_inner_mm", "d3_inner_mm"}.issubset(df.columns):
        geom_ok = df.apply(
            lambda r: (pd.isna(r["d2_inner_mm"]) or pd.isna(r["d3_inner_mm"])
                       or is_ring_ordering_ok(r["d2_inner_mm"], r["d3_inner_mm"])),
            axis=1,
        )
        n_bad_geom = int((~geom_ok).sum())
        if n_bad_geom:
            logger.warning(f"Excluding {n_bad_geom} rows whose rings violate the "
                           f"{MIN_RING_GAP_MM:.0f} mm radial clearance rule")
            df = df[geom_ok].reset_index(drop=True)

    # Deduplication must see EVERY design already attempted, including the ones
    # just excluded from training. Filtering them out of the dedup reference set
    # makes the optimizer forget it has tried a point: after the clearance fix it
    # re-proposed the same unbuildable d2=540/d3=550 for 46 consecutive
    # iterations because each failed attempt was hidden from the next.
    df_attempted = df_all[PARAM_NAMES].dropna() if set(PARAM_NAMES).issubset(df_all.columns) else df[PARAM_NAMES].dropna()

    df_valid = df.dropna(subset=OBJ_COLUMNS)
    all_missing = df[OBJ_COLUMNS].isna().all(axis=1)
    some_missing = df[OBJ_COLUMNS].isna().any(axis=1)
    df_failed = df[all_missing]
    n_total = len(df)
    n_valid = len(df_valid)
    n_failed = len(df_failed)
    n_partial = int((some_missing & ~all_missing).sum())
    logger.info(f"Loaded {n_total} rows, {n_valid} complete, {n_failed} failed "
                f"(all objectives missing), {n_partial} partially measured")
    if n_partial:
        logger.info(f"  {n_partial} partially-measured rows are EXCLUDED from training "
                    f"rather than penalised; reconstruct them to recover their data")

    if n_valid < 3:
        raise ValueError(f"Need at least 3 feasible points for MOBO, got {n_valid}")

    # --- Assign penalty values to failed configs so the GP learns to avoid them ---
    # Use worst observed value (in the "maximize" direction) with a margin.
    #
    # IMPORTANT: this multiplicative scheme assumes every metric in OBJ_COLUMNS
    # is strictly positive across the observed data. All current metrics
    # (every column in OBJ_COLUMNS; MPXI is a beam count, so non-negative)
    # satisfy that. If a future metric can go negative, "col.min() * 0.5" would
    # make the penalty MORE favourable than the worst observation —
    # subtract-a-margin would be needed instead. The assertion below guards
    # against a silent regression.
    if n_failed > 0 and len(df_failed[PARAM_NAMES].dropna()) > 0:
        valid_vals = df_valid[OBJ_COLUMNS].values
        assert (valid_vals >= 0).all(), (
            "penalty calculation assumes strictly non-negative objective values; "
            "negative value detected — rework the penalty scheme before proceeding"
        )
        penalty_row = []
        for i, d in enumerate(OBJ_DIRECTIONS):
            col = valid_vals[:, i]
            if d > 0:  # maximize → penalty is well below minimum
                penalty_row.append(float(col.min()) * 0.5)
            else:  # minimize → penalty is well above maximum
                penalty_row.append(float(col.max()) * 2.0)
        logger.info(f"Penalty values for {n_failed} failed configs: {penalty_row}")

        # Build combined training data: valid + failed-with-penalty
        df_failed_with_params = df_failed.dropna(subset=PARAM_NAMES)
        failed_x = df_failed_with_params[PARAM_NAMES].values
        failed_y = [penalty_row] * len(df_failed_with_params)

        train_x = torch.tensor(
            pd.concat([df_valid[PARAM_NAMES], df_failed_with_params[PARAM_NAMES]]).values,
            dtype=torch.double)
        train_y_raw = torch.tensor(
            list(df_valid[OBJ_COLUMNS].values) + failed_y,
            dtype=torch.double)
        logger.info(f"Training with {n_valid} feasible + {len(df_failed_with_params)} penalized = "
                     f"{len(train_x)} total points")
    else:
        train_x = torch.tensor(df_valid[PARAM_NAMES].values, dtype=torch.double)
        train_y_raw = torch.tensor(df_valid[OBJ_COLUMNS].values, dtype=torch.double)

    # Apply direction signs (negate FWHM and MPXI so all objectives are maximized)
    directions = torch.tensor(OBJ_DIRECTIONS, dtype=torch.double)
    train_y = train_y_raw * directions

    bounds = torch.tensor([BOUNDS_MIN, BOUNDS_MAX], dtype=torch.double)

    # Designs proposed under an earlier, wider box stay in the archive and keep
    # their measurements, so training points can sit outside the CURRENT bounds
    # -- mobo_0190 and mobo_0193 carry real data at d3_inner_mm = 640 against a
    # ceiling of 618. normalize() does not clamp, so those land outside the unit
    # cube the acquisition optimises over. The GP extrapolates them harmlessly
    # and dropping them would throw away real measurements, but it should never
    # be invisible: a bounds change that silently strands half the training set
    # outside the search box is exactly the kind of thing that goes unnoticed
    # for weeks.
    n_oob = report_out_of_bounds(train_x)
    if n_oob:
        logger.warning(f"{n_oob} of {len(train_x)} training points lie outside the "
                       f"current search box; kept as training data (see above)")

    train_x_norm = normalize(train_x, bounds)

    # Log current objective ranges
    for i, name in enumerate(OBJ_NAMES):
        col_vals = train_y_raw[:, i]
        logger.info(f"  {name}: min={col_vals.min():.4f}, max={col_vals.max():.4f}, "
                     f"mean={col_vals.mean():.4f}")

    # --- Standardize each objective independently ---
    y_mean = train_y.mean(dim=0, keepdim=True)
    y_std = train_y.std(dim=0, keepdim=True).clamp(min=1e-6)
    train_y_std = (train_y - y_mean) / y_std

    # --- Fit ModelListGP (one SingleTaskGP per objective) ---
    logger.info("Fitting ModelListGP (one GP per objective)...")
    models = []
    for i, name in enumerate(OBJ_NAMES):
        gp = SingleTaskGP(train_x_norm, train_y_std[:, i:i+1])
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        models.append(gp)

        # Log ARD lengthscales. These are the main diagnostic for whether a
        # design dimension is being used at all: a lengthscale orders of
        # magnitude above the others means the GP has concluded the objective
        # does not vary along it, so the acquisition sees no reason to explore
        # there. That is exactly what happens when the training data is constant
        # in a dimension, as ours is in d2_inner_mm / d3_inner_mm.
        #
        # The kernel layout differs across BoTorch versions (newer SingleTaskGP
        # defaults changed the wrapping), so try the known shapes rather than
        # swallowing the failure -- this was previously `except: pass`, which
        # silently discarded the diagnostic and made the log look as if the
        # lengthscales had simply never been computed.
        ls = None
        for accessor in (
            lambda g: g.covar_module.base_kernel.lengthscale,
            lambda g: g.covar_module.lengthscale,
        ):
            try:
                ls = accessor(gp).detach().cpu().numpy().flatten()
                break
            except (AttributeError, RuntimeError):
                continue
        if ls is not None and len(ls) == len(PARAM_NAMES):
            ls_str = ", ".join(f"{p}={l:.3f}" for p, l in zip(PARAM_NAMES, ls))
            logger.info(f"  GP[{name}] lengthscales: {ls_str}")
        else:
            logger.warning(f"  GP[{name}] lengthscales unavailable "
                           f"(kernel layout not recognised); dimension-usage "
                           f"diagnostic is unavailable for this objective")

    model = ModelListGP(*models)
    logger.info("ModelListGP trained successfully.")

    # --- Reference point (in standardized space) ---
    # Use worst observed value minus a small margin for each objective
    ref_point_std = train_y_std.min(dim=0).values - 0.1
    logger.info(f"Reference point (standardized): {ref_point_std.tolist()}")

    # --- Optimize qLogNEHVI ---
    logger.info("Optimizing qLogNEHVI acquisition function...")
    acqf = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point_std.tolist(),
        X_baseline=train_x_norm,
        prune_baseline=True,
        cache_root=False,
    )

    # Feasibility constraint in normalized space:
    # aperture_diam < 0.95 * circumference / n_apertures
    # Physical: diam * n_ap < 0.95 * circumference = threshold
    diam_min, diam_range = BOUNDS_MIN[0], BOUNDS_MAX[0] - BOUNDS_MIN[0]
    nap_min, nap_range = BOUNDS_MIN[1], BOUNDS_MAX[1] - BOUNDS_MIN[1]
    nd2_min, nd2_range = BOUNDS_MIN[3], BOUNDS_MAX[3] - BOUNDS_MIN[3]
    d2_min, d2_range = BOUNDS_MIN[4], BOUNDS_MAX[4] - BOUNDS_MIN[4]
    d3_min, d3_range = BOUNDS_MIN[5], BOUNDS_MAX[5] - BOUNDS_MIN[5]
    threshold = SAFETY_MARGIN * HR_CIRCUMFERENCE

    def is_feasible_norm(x_norm):
        """Check feasibility in normalized [0,1] space. x_norm shape: (..., DIM)

        Three constraints:
          1. apertures must not overlap on the HR ring
          2. the detector rings must stay ordered with the minimum radial gap
          3. ring 2's crystals must fit at its proposed diameter

        The bounds already guarantee d2 >= D1 + gap, d3 <= D4 - gap, and that
        ring 3's fixed 960 crystals fit (d3 >= 385), so only the middle ordering
        inequality and ring 2's packing need checking here.
        """
        diam = diam_min + x_norm[..., 0] * diam_range
        n_ap = nap_min + x_norm[..., 1] * nap_range
        aperture_ok = diam * n_ap < threshold

        d2 = d2_min + x_norm[..., 4] * d2_range
        d3 = d3_min + x_norm[..., 5] * d3_range
        # Separation of inner diameters, which must cover ring 2's 6 mm depth
        # plus the clearance. Comparing d3 - d2 against MIN_RING_GAP_MM directly
        # accepts overlapping rings -- that mistake let the acquisition propose
        # d2=540/d3=550 for 46 consecutive iterations, none of which could be
        # built.
        ordering_ok = (d3 - d2) >= MIN_DIAM_SEPARATION_MM

        n_det2 = nd2_min + x_norm[..., 3] * nd2_range
        max_nd2 = 4.0 * math.pi * (d2 / 2.0 + SCINT_RADIAL_MM / 2.0) / CELL_SPAN_MM
        packing_ok = n_det2 < max_nd2

        return aperture_ok & ordering_ok & packing_ok

    def feasible_ic_generator(acq_function, bounds, num_restarts, raw_samples, options=None, **kwargs):
        """Generate feasible initial conditions for constrained acquisition optimization."""
        # Use Sobol sequence for better coverage of the 4D space
        n_candidates = 2048
        sobol = torch.quasirandom.SobolEngine(dimension=DIM, scramble=True)
        X_rnd = sobol.draw(n_candidates, dtype=bounds.dtype).unsqueeze(1)
        feasible_mask = is_feasible_norm(X_rnd.squeeze(1))
        X_feasible = X_rnd[feasible_mask]
        if len(X_feasible) < num_restarts:
            raise RuntimeError(f"Only {len(X_feasible)} feasible ICs found, need {num_restarts}")
        # Evaluate acquisition in batches to limit memory
        n_eval = min(len(X_feasible), 256)
        with torch.no_grad():
            acq_values = acq_function(X_feasible[:n_eval])
        top_indices = torch.argsort(acq_values, descending=True)[:num_restarts]
        return X_feasible[top_indices]

    candidate_norm, acq_value = optimize_acqf(
        acq_function=acqf,
        bounds=torch.stack([torch.zeros(DIM), torch.ones(DIM)]).double(),
        q=1,
        num_restarts=10,
        raw_samples=256,
        ic_generator=feasible_ic_generator,
    )

    # --- Un-normalize ---
    candidate_physical = unnormalize(candidate_norm, bounds)
    next_diam = candidate_physical[0, 0].item()
    next_n_ap = int(round(candidate_physical[0, 1].item()))
    next_n_det1 = int(round(candidate_physical[0, 2].item()))
    next_n_det2 = int(round(candidate_physical[0, 3].item()))
    next_d2 = candidate_physical[0, 4].item()
    next_d3 = candidate_physical[0, 5].item()
    # n_det values must be even (2 crystals per cell)
    if next_n_det1 % 2 != 0:
        next_n_det1 += 1
    if next_n_det2 % 2 != 0:
        next_n_det2 += 1

    # Final feasibility check (belt and suspenders)
    if not is_feasible(next_diam, next_n_ap):
        logger.warning(f"Candidate infeasible (d={next_diam:.4f}, n={next_n_ap}), clamping aperture_diam")
        next_diam = min(next_diam, SAFETY_MARGIN * HR_CIRCUMFERENCE / next_n_ap - 0.01)
    if not is_ring_ordering_ok(next_d2, next_d3):
        # Repair by separating the diameters enough to clear ring 2's depth, then
        # pulling d2 in if d3 has hit its bound. The previous version used
        # MIN_RING_GAP_MM as a diameter offset, so for d2=540 it produced
        # d3=550 -- still overlapping, and deterministic, which is why the same
        # unbuildable design was proposed 46 iterations running.
        before = (next_d2, next_d3)
        next_d3 = min(next_d2 + MIN_DIAM_SEPARATION_MM, BOUNDS_MAX[5])
        next_d2 = min(next_d2, next_d3 - MIN_DIAM_SEPARATION_MM)
        next_d2 = max(next_d2, BOUNDS_MIN[4])
        logger.warning(f"Candidate violated ring clearance "
                       f"(d2={before[0]:.1f}, d3={before[1]:.1f}); "
                       f"repaired to d2={next_d2:.1f}, d3={next_d3:.1f}")
        if not is_ring_ordering_ok(next_d2, next_d3):
            # Repair cannot succeed from every starting point. Falling back to
            # the legacy layout is better than emitting a design the pipeline
            # will refuse to build and burning the iteration on a NaN row.
            logger.error(f"Repair failed; falling back to the legacy 390/520 layout")
            next_d2, next_d3 = 390.0, 520.0
    if not is_ring_packing_ok(next_n_det1, next_n_det2, next_d2, next_d3):
        # Trim ring 2's crystal count to what actually fits, rounding down to an
        # even number. Better a slightly smaller ring than a geometry crash that
        # costs the whole iteration.
        max_nd2 = int(max_crystals_on_ring(next_d2))
        trimmed = min(next_n_det2, max_nd2 - 2)
        trimmed -= trimmed % 2
        logger.warning(f"Ring 2 packing infeasible (n_det2={next_n_det2} at d2={next_d2:.1f} mm, "
                       f"max {max_nd2}); trimming n_det_ring2 to {trimmed}")
        next_n_det2 = max(int(BOUNDS_MIN[3]), trimmed)

    # --- Deduplication: reject if too close to a previously tried config ---
    all_x = torch.tensor(df_attempted.values, dtype=torch.double)
    candidate_vec = torch.tensor([next_diam, float(next_n_ap), float(next_n_det1),
                                  float(next_n_det2), next_d2, next_d3],
                                 dtype=torch.double)
    # Normalize distances by parameter ranges to compare fairly
    ranges = bounds[1] - bounds[0]
    norm_dists = ((all_x - candidate_vec) / ranges).norm(dim=1)
    min_dist = norm_dists.min().item() if len(norm_dists) > 0 else 1.0

    if min_dist < 0.02:  # less than 2% of design space away = duplicate
        logger.warning(f"Candidate too close to existing config (dist={min_dist:.4f}), searching for diverse alternative")

        # Generate Sobol set, filter for feasibility + distance from all existing
        sobol_dedup = torch.quasirandom.SobolEngine(dimension=DIM, scramble=True)
        n_dedup = 2048
        X_sobol = sobol_dedup.draw(n_dedup, dtype=torch.double)
        # Unnormalize
        X_phys = X_sobol * ranges + bounds[0]

        # Filter feasible (aperture overlap, ring ordering, ring packing)
        feasible_mask = torch.tensor(
            [is_feasible_full(X_phys[i, 0].item(), X_phys[i, 1].item(),
                              X_phys[i, 2].item(), X_phys[i, 3].item(),
                              X_phys[i, 4].item(), X_phys[i, 5].item())
             for i in range(n_dedup)]
        )
        X_feasible = X_phys[feasible_mask]
        if len(X_feasible) == 0:
            logger.warning("No feasible candidates for dedup — keeping original")
        else:
            # Compute min distance from each candidate to all existing configs
            all_x_norm = (all_x - bounds[0]) / ranges
            X_feas_norm = (X_feasible - bounds[0]) / ranges
            # Pairwise distances
            dists_to_existing = torch.cdist(X_feas_norm, all_x_norm)  # (n_feasible, n_existing)
            min_dists = dists_to_existing.min(dim=1).values  # (n_feasible,)

            # Evaluate acquisition on candidates with min_dist >= 0.02
            diverse_mask = min_dists >= 0.02
            if diverse_mask.sum() == 0:
                # Relax threshold
                diverse_mask = min_dists >= 0.01
                logger.warning("Relaxed dedup threshold to 0.01")

            if diverse_mask.sum() > 0:
                X_diverse = X_feasible[diverse_mask]
                diverse_dists = min_dists[diverse_mask]

                # Normalize for acquisition evaluation
                X_diverse_norm = ((X_diverse - bounds[0]) / ranges).unsqueeze(1)
                n_eval = min(len(X_diverse_norm), 256)
                with torch.no_grad():
                    acq_vals = acqf(X_diverse_norm[:n_eval])
                # Pick candidate with best acquisition value
                best_idx = acq_vals.argmax()
                best_cand = X_diverse[best_idx]

                next_diam = best_cand[0].item()
                next_n_ap = int(round(best_cand[1].item()))
                next_n_det1 = int(round(best_cand[2].item()))
                next_n_det2 = int(round(best_cand[3].item()))
                next_d2 = best_cand[4].item()
                next_d3 = best_cand[5].item()
                if next_n_det1 % 2 != 0:
                    next_n_det1 += 1
                if next_n_det2 % 2 != 0:
                    next_n_det2 += 1
                logger.info(f"Dedup: selected diverse candidate (acq={acq_vals[best_idx]:.4f}, "
                            f"dist={diverse_dists[best_idx]:.4f}): "
                            f"d={next_diam:.4f} n={next_n_ap} nd1={next_n_det1} "
                            f"nd2={next_n_det2} d2={next_d2:.1f} d3={next_d3:.1f}")
            else:
                logger.warning("No diverse candidates found — keeping original")

    logger.info(f"Acquisition value: {acq_value.item():.6f}")
    logger.info(f"SUGGESTION -> aperture_diam={next_diam:.4f} mm | "
                f"n_apertures={next_n_ap} | "
                f"n_det_ring1={next_n_det1} | n_det_ring2={next_n_det2} | "
                f"d2_inner={next_d2:.1f} mm | d3_inner={next_d3:.1f} mm")

    return next_diam, next_n_ap, next_n_det1, next_n_det2, next_d2, next_d3


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MOBO Agent for SAI SC-SPECT")
    parser.add_argument("--results_csv", type=str, required=True,
                        help="Path to results CSV with all OBJ_COLUMNS "
                             "(fwhm_mean, asci_pct, sensitivity_mean, mpxi_mean, cnr_mean)")
    args = parser.parse_args()

    diam, n_ap, n_det1, n_det2, d2_inner, d3_inner = get_next_candidate(args.results_csv)
    print(f"\nSuggested next config:")
    print(f"  aperture_diam   = {diam:.4f} mm")
    print(f"  n_apertures     = {n_ap}")
    print(f"  n_det_ring1     = {n_det1}")
    print(f"  n_det_ring2     = {n_det2}")
    print(f"  d2_inner        = {d2_inner:.1f} mm")
    print(f"  d3_inner        = {d3_inner:.1f} mm")
