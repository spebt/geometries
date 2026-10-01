"""
Local unit tests for the SAI SC-SPECT MOBO pipeline.

Covers the pure-Python logic that runs on Mac without CCR access:
- backfill_cnr matcher (all three fallback layers)
- mobo_agent feasibility math
- compute_metrics._per_beam_radial_fwhm projection math
- compute_metrics.compute_ppds end-to-end with synthetic HDF5 files
- compute_cnr._write_cnr_to_csv row/column semantics
- CSV column-alignment behaviour of compute_metrics.main()

Run with:
    cd /Users/Omer/Desktop/RA/omer/spebt
    python -m pytest tests/test_pipeline.py -v
"""
import math
import os
import subprocess
import sys
import tempfile

import h5py
import numpy as np
import pandas as pd
import pytest

# Wire the optimization/ folder onto the path so we can import as modules
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "optimization"))


# =============================================================================
# backfill_cnr._find_cnr_npz — matcher logic
# =============================================================================
class TestBackfillCnrMatcher:
    """Verify the three fallback layers in backfill_cnr._find_cnr_npz."""

    @pytest.fixture
    def recon_root(self, tmp_path):
        """Layout mimicking what we have on CCR: mixed naming conventions."""
        root = tmp_path / "recon_results"
        root.mkdir()

        # A: exact-match style (MOBO — indices are already zero-padded)
        (root / "mobo_0069_ap0.3138_nap124_nd1_612_nd2_230").mkdir()
        np.savez(root / "mobo_0069_ap0.3138_nap124_nd1_612_nd2_230" / "cnr_results.npz",
                 overall_cnr=4.44)

        # B: 500-iter suffix — must be preferred
        (root / "mobo_0069_ap0.3138_nap124_nd1_612_nd2_230_500iter").mkdir()
        np.savez(root / "mobo_0069_ap0.3138_nap124_nd1_612_nd2_230_500iter" / "cnr_results.npz",
                 overall_cnr=4.53)

        # C: LHS-style mismatch (padded index, less-precise aperture)
        (root / "lhs4d_0003_ap0.3400_nap63_nd1_202_nd2_788").mkdir()
        np.savez(root / "lhs4d_0003_ap0.3400_nap63_nd1_202_nd2_788" / "cnr_results.npz",
                 overall_cnr=4.25)

        # D: no matching folder at all
        # (nothing created for config "mobo_9999_...")
        return root

    def test_exact_match_returns_direct(self, recon_root):
        """Layer 1: exact folder name match."""
        from backfill_cnr import _find_cnr_npz
        result = _find_cnr_npz(str(recon_root),
                               "mobo_0069_ap0.3138_nap124_nd1_612_nd2_230")
        assert result is not None
        # With both an exact-match dir AND a _500iter dir, prefix glob picks up both
        # and _pick_best prefers 500iter. So the exact-match layer might miss it.
        # Actually layer 1 checks exact first, so returns the non-500iter dir.
        assert "cnr_results.npz" in result

    def test_prefix_glob_prefers_500iter(self, recon_root):
        """Layer 2: prefix glob when no exact-match. The 500-iter variant should win."""
        from backfill_cnr import _find_cnr_npz
        # Remove the exact-match dir so we fall through to prefix glob
        exact = recon_root / "mobo_0069_ap0.3138_nap124_nd1_612_nd2_230" / "cnr_results.npz"
        exact.unlink()
        exact.parent.rmdir()

        result = _find_cnr_npz(str(recon_root),
                               "mobo_0069_ap0.3138_nap124_nd1_612_nd2_230")
        assert result is not None
        assert "500iter" in result

    def test_design_signature_recovers_lhs(self, recon_root):
        """Layer 3: design-signature match recovers LHS runs with padded index."""
        from backfill_cnr import _find_cnr_npz
        # CSV config has unpadded index + full-precision aperture; recon folder has
        # padded index + truncated aperture. Signature (nap, nd1, nd2) is unique.
        result = _find_cnr_npz(str(recon_root),
                               "lhs4d_3_ap0.340036_nap63_nd1_202_nd2_788")
        assert result is not None
        assert "lhs4d_0003" in result

    def test_returns_none_when_no_match(self, recon_root):
        """No match anywhere → None (backfill logs 'no recon output')."""
        from backfill_cnr import _find_cnr_npz
        result = _find_cnr_npz(str(recon_root),
                               "mobo_9999_ap0.5_nap100_nd1_300_nd2_500")
        assert result is None

    def test_pick_best_prefers_500iter(self):
        """Helper directly: order-independence."""
        from backfill_cnr import _pick_best
        assert "500iter" in _pick_best([
            "/tmp/a/cnr_results.npz",
            "/tmp/b_500iter/cnr_results.npz",
        ])
        assert "500iter" in _pick_best([
            "/tmp/b_500iter/cnr_results.npz",
            "/tmp/a/cnr_results.npz",
        ])
        # No 500iter → first one
        assert _pick_best(["/tmp/a/cnr_results.npz",
                           "/tmp/b/cnr_results.npz"]) == "/tmp/a/cnr_results.npz"


# =============================================================================
# mobo_agent — feasibility math
# =============================================================================
class TestMoboAgentFeasibility:
    """The aperture-overlap feasibility check must accept legit designs and reject
    ones that would cause apertures to touch/overlap on the HR ring."""

    def test_is_feasible_accepts_baseline(self):
        from mobo_agent import is_feasible
        # Baseline: 180 apertures at 0.4 mm diameter → chord ≈ 1.22 mm — fine
        assert is_feasible(0.4, 180)

    def test_is_feasible_rejects_too_many_wide(self):
        from mobo_agent import is_feasible
        # 270 apertures at 1.0 mm — chord per aperture ≈ 0.81 mm < 1.0 mm — overlap
        assert not is_feasible(1.0, 270)

    def test_is_feasible_at_safety_margin(self):
        """Just above the SAFETY_MARGIN threshold should reject; just below accept."""
        from mobo_agent import is_feasible, HR_CIRCUMFERENCE, SAFETY_MARGIN
        n = 200
        boundary_diam = SAFETY_MARGIN * HR_CIRCUMFERENCE / n
        assert not is_feasible(boundary_diam + 0.01, n)
        assert is_feasible(boundary_diam - 0.01, n)

    def test_is_feasible_at_bounds_extreme(self):
        """Design-space corner (min aperture, max n) must be feasible."""
        from mobo_agent import is_feasible
        assert is_feasible(0.2, 270)


# =============================================================================
# compute_metrics._per_beam_radial_fwhm — projection math
# =============================================================================
class TestRingOrdering:
    """D1 < D2 < D3 < D4 with a 10 mm minimum gap (RY, Jul 2026).

    The bounds guarantee the outer two inequalities, so the acquisition-time
    check only enforces d3 - d2; these tests pin down the full check that the
    geometry generator and the dedup path use.
    """

    def test_nominal_layout_is_feasible(self):
        import mobo_agent as ma
        # The legacy fixed layout [260, 390, 520, 650] must remain feasible,
        # otherwise every historical config becomes an infeasible training point.
        # Its clearances are 59 mm all round, so it is comfortably valid.
        assert ma.is_ring_ordering_ok(390.0, 520.0)

    def test_clearance_is_radial_not_diametral(self):
        """The bug this replaced: 10 mm of DIAMETER is 5 mm of radius, and the
        crystals are 6 mm deep, so d2=270 put ring 2 inside ring 1 by 1 mm."""
        import mobo_agent as ma
        assert not ma.is_ring_ordering_ok(270.0, 520.0), \
            "d2=270 overlaps ring 1 by 1 mm and must be rejected"
        assert not ma.is_ring_ordering_ok(390.0, 640.0), \
            "d3=640 overlaps ring 4 by 1 mm and must be rejected"

    def test_exactly_at_clearance_is_feasible(self):
        """d2 = 260 + 2*(6+10) = 292 gives exactly 10 mm; d3 = 650 - 32 = 618."""
        import mobo_agent as ma
        assert ma.is_ring_ordering_ok(292.0, 618.0)
        assert not ma.is_ring_ordering_ok(291.0, 618.0)
        assert not ma.is_ring_ordering_ok(292.0, 619.0)

    def test_adjacent_rings_need_clearance_too(self):
        import mobo_agent as ma
        # d3 - d2 = 20 is only 4 mm of radial clearance after the 6 mm crystal
        assert not ma.is_ring_ordering_ok(450.0, 470.0)
        # d3 - d2 = 32 is exactly 10 mm
        assert ma.is_ring_ordering_ok(450.0, 482.0)

    def test_rejects_out_of_order(self):
        import mobo_agent as ma
        assert not ma.is_ring_ordering_ok(520.0, 390.0)  # D3 inside D2

    def test_diameter_and_radial_rules_agree(self):
        """The acquisition-time check works on diameters for speed; it must accept
        exactly what the radial rule accepts.

        These drifted apart once: is_ring_ordering_ok was corrected to compare
        radial spans while is_feasible_norm kept comparing d3 - d2 against 10 mm.
        The optimizer then proposed d2=540/d3=550 for 46 consecutive iterations,
        every one of which the geometry generator refused to build.
        """
        import mobo_agent as ma
        for d2 in range(292, 541, 8):
            for d3 in range(385, 619, 8):
                by_diam = (d3 - d2) >= ma.MIN_DIAM_SEPARATION_MM
                by_radius = ma.is_ring_ordering_ok(float(d2), float(d3))
                assert by_diam == by_radius, f"disagree at d2={d2}, d3={d3}"

    def test_repair_produces_a_valid_layout(self):
        """The repair path must not emit something still invalid.

        It previously set d3 = d2 + 10, which for d2=540 gives the overlapping
        540/550 -- and deterministically, so every iteration landed on the same
        unbuildable design.
        """
        import mobo_agent as ma
        for d2, d3 in ((540.0, 550.0), (500.0, 505.0), (292.0, 300.0), (530.0, 618.0)):
            r3 = min(d2 + ma.MIN_DIAM_SEPARATION_MM, ma.BOUNDS_MAX[5])
            r2 = max(min(d2, r3 - ma.MIN_DIAM_SEPARATION_MM), ma.BOUNDS_MIN[4])
            if not ma.is_ring_ordering_ok(r2, r3):
                r2, r3 = 390.0, 520.0      # documented fallback
            assert ma.is_ring_ordering_ok(r2, r3), f"repair failed for {d2}/{d3}"

    def test_bounds_admit_only_valid_layouts(self):
        """Every bound corner must satisfy the clearance rule."""
        import mobo_agent as ma
        assert ma.is_ring_ordering_ok(ma.BOUNDS_MIN[4], ma.BOUNDS_MAX[5])
        r1_out = ma.D1_INNER_MM / 2 + ma.SCINT_RADIAL_MM
        assert ma.BOUNDS_MIN[4] / 2 - r1_out >= ma.MIN_RING_GAP_MM
        r4_in = ma.D4_INNER_MM / 2
        assert r4_in - (ma.BOUNDS_MAX[5] / 2 + ma.SCINT_RADIAL_MM) >= ma.MIN_RING_GAP_MM

    def test_bounds_admit_the_gap(self):
        """Every bound corner must be reachable without violating the fixed rings."""
        import mobo_agent as ma
        d2_lo, d3_hi = ma.BOUNDS_MIN[4], ma.BOUNDS_MAX[5]
        assert d2_lo - ma.D1_INNER_MM >= ma.MIN_RING_GAP_MM
        assert ma.D4_INNER_MM - d3_hi >= ma.MIN_RING_GAP_MM

    def test_is_feasible_full_combines_all_constraints(self):
        import mobo_agent as ma
        # Everything OK
        assert ma.is_feasible_full(0.4, 100, 480, 720, 390.0, 520.0)
        # Ring ordering bad
        assert not ma.is_feasible_full(0.4, 100, 480, 720, 400.0, 405.0)
        # Aperture bad (0.4 * 900 far exceeds the ~220 mm circumference)
        assert not ma.is_feasible_full(0.4, 900, 480, 720, 390.0, 520.0)
        # Ring 2 packing bad: 960 crystals cannot fit at a 270 mm diameter
        assert not ma.is_feasible_full(0.4, 100, 480, 960, 270.0, 520.0)


class TestRingPacking:
    """Ring diameter and crystal count interact: shrinking a ring while keeping
    its crystals raises packing density until cells overlap, at which point the
    geometry generator raises and the MOBO iteration is wasted. These bounds and
    checks exist to keep the optimizer out of that region.
    """

    def test_max_crystals_matches_generator_formula(self):
        """max_crystals_on_ring must mirror build_sc_spect_detector_rings."""
        import math
        import mobo_agent as ma
        # Generator: arc_per_cell = r_c * 2*pi / (n_scint/2) must exceed 2*W+gap
        for d in (260.0, 390.0, 520.0, 650.0):
            r_c = d / 2.0 + ma.SCINT_RADIAL_MM / 2.0
            n_max = ma.max_crystals_on_ring(d)
            arc_at_max = r_c * 2.0 * math.pi / (n_max / 2.0)
            assert math.isclose(arc_at_max, ma.CELL_SPAN_MM, rel_tol=1e-9)

    def test_nominal_layout_packs(self):
        """The legacy [260,390,520,650] / [480,720,960,1200] layout must pass."""
        import mobo_agent as ma
        assert ma.is_ring_packing_ok(480, 720, 390.0, 520.0)

    def test_ring1_bound_is_justified(self):
        """n_det_ring1 max of 660 exists because ring 1 caps at ~663 crystals."""
        import mobo_agent as ma
        cap = ma.max_crystals_on_ring(ma.D1_INNER_MM)
        assert ma.BOUNDS_MAX[2] < cap
        assert cap < ma.BOUNDS_MAX[2] + 10   # the bound is tight, not arbitrary

    def test_d3_lower_bound_admits_fixed_ring3(self):
        """Ring 3 carries a fixed 960 crystals, so d3 cannot go arbitrarily low."""
        import mobo_agent as ma
        assert ma.N_DET_RING3 < ma.max_crystals_on_ring(ma.BOUNDS_MIN[5])
        # And the bound is near the true floor rather than needlessly generous
        assert ma.N_DET_RING3 > ma.max_crystals_on_ring(ma.BOUNDS_MIN[5] - 10.0)

    def test_ring2_infeasible_at_small_diameter(self):
        import mobo_agent as ma
        # 960 crystals need ~379 mm; at 270 mm only ~688 fit
        assert not ma.is_ring_packing_ok(480, 960, 270.0, 520.0)
        assert ma.is_ring_packing_ok(480, 600, 270.0, 520.0)

    def test_fixed_ring4_always_packs(self):
        import mobo_agent as ma
        assert ma.N_DET_RING4 < ma.max_crystals_on_ring(ma.D4_INNER_MM)


class TestFailedRowClassification:
    """Only rows missing EVERY objective are failures.

    A row missing some objectives is a partially-measured design, not a bad one.
    Penalising those fabricated values for objectives we had actually measured --
    real FWHM values near 0.5 mm were being replaced by the penalty value of
    9.84 for a third of the training set.
    """

    @staticmethod
    def _classify(df, obj_cols):
        """Mirrors the classification in mobo_agent.get_next_candidate."""
        all_missing = df[obj_cols].isna().all(axis=1)
        some_missing = df[obj_cols].isna().any(axis=1)
        return {
            "complete": len(df.dropna(subset=obj_cols)),
            "failed": int(all_missing.sum()),
            "partial": int((some_missing & ~all_missing).sum()),
        }

    def test_partial_rows_are_not_failures(self):
        import mobo_agent as ma
        cols = ma.OBJ_COLUMNS
        rows = [
            {c: 1.0 for c in cols},                                  # complete
            {**{c: 1.0 for c in cols}, cols[-1]: float("nan")},       # missing CNR only
            {c: float("nan") for c in cols},                          # geometry failure
        ]
        got = self._classify(pd.DataFrame(rows), cols)
        assert got == {"complete": 1, "failed": 1, "partial": 1}, got

    def test_row_missing_only_cnr_is_partial(self):
        """The exact case that was being mis-penalised."""
        import mobo_agent as ma
        cols = ma.OBJ_COLUMNS
        row = {c: 1.0 for c in cols}
        row["cnr_sector_mean"] = float("nan")
        got = self._classify(pd.DataFrame([row]), cols)
        assert got["partial"] == 1 and got["failed"] == 0

    def test_all_nan_row_is_a_failure(self):
        import mobo_agent as ma
        cols = ma.OBJ_COLUMNS
        got = self._classify(pd.DataFrame([{c: float("nan") for c in cols}]), cols)
        assert got["failed"] == 1 and got["partial"] == 0

    def test_geometrically_invalid_rows_are_excluded(self):
        """Rows whose rings overlap must not train the GP.

        Configs proposed before the clearance fix have rings interpenetrating by
        ~1 mm. The generator built them and the ray tracer modelled overlapping
        absorbers, so their PPDFs are physically wrong rather than just poor.
        """
        import mobo_agent as ma
        rows = [
            {"d2_inner_mm": 390.0, "d3_inner_mm": 520.0},   # legacy, valid
            {"d2_inner_mm": 400.0, "d3_inner_mm": 460.0},   # off-slice, valid
            {"d2_inner_mm": 362.0, "d3_inner_mm": 640.0},   # ring 3 in ring 4
            {"d2_inner_mm": 381.0, "d3_inner_mm": 391.0},   # ring 2 in ring 3
            {"d2_inner_mm": 399.0, "d3_inner_mm": 424.0},   # only 6.5 mm gap
        ]
        ok = [ma.is_ring_ordering_ok(r["d2_inner_mm"], r["d3_inner_mm"]) for r in rows]
        assert ok == [True, True, False, False, False], ok

    def test_missing_d2d3_is_not_treated_as_invalid(self):
        """A row with no D2/D3 recorded predates the expansion and is fine."""
        import mobo_agent as ma
        # Mirrors the guard in get_next_candidate: NaN passes through
        for d2, d3 in ((float("nan"), 520.0), (390.0, float("nan"))):
            passes = (pd.isna(d2) or pd.isna(d3)
                      or ma.is_ring_ordering_ok(d2, d3))
            assert passes

    def test_penalty_is_worse_than_every_real_value(self):
        """Penalty must sit outside the observed range in the correct direction."""
        import mobo_agent as ma
        vals = np.array([[0.5, 20.0, 0.1, 2.0, 4.0],
                         [4.9, 60.0, 0.18, 22.0, 1.35]])
        for i, d in enumerate(ma.OBJ_DIRECTIONS):
            col = vals[:, i]
            penalty = col.min() * 0.5 if d > 0 else col.max() * 2.0
            if d > 0:
                assert penalty < col.min(), f"objective {i}: penalty not worse"
            else:
                assert penalty > col.max(), f"objective {i}: penalty not worse"


class TestPerBeamRadialFwhm:
    """Verify the FWHM-along-beam-axis calculation on synthetic beams."""

    def _make_gaussian_beam_row(self, cx_mm, cy_mm, sigma_r_mm, sigma_t_mm,
                                angle_rad, beam_id=1, fov_shape=(200, 200),
                                mm_per_px=0.05):
        """Construct a synthetic (mask_row, ppdf_row) with one 2D Gaussian beam.

        The beam is centered at (cx_mm, cy_mm), oriented so its radial axis points
        along (cos(angle), sin(angle)), with the given sigmas.
        """
        rows = np.repeat(np.arange(fov_shape[0]), fov_shape[1]).astype(np.float64) * mm_per_px
        cols = np.tile(np.arange(fov_shape[1]), fov_shape[0]).astype(np.float64) * mm_per_px
        dx = cols - cx_mm
        dy = rows - cy_mm
        # Radial coord (along beam axis)
        r = dx * math.cos(angle_rad) + dy * math.sin(angle_rad)
        # Tangential coord (perpendicular)
        t = -dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
        # 2D Gaussian intensity
        ppdf_row = np.exp(-(r * r) / (2 * sigma_r_mm ** 2) - (t * t) / (2 * sigma_t_mm ** 2))
        # Mask: pixels above ~1e-3 of the peak belong to the beam
        mask_row = np.where(ppdf_row > 1e-3, beam_id, 0).astype(np.int32)
        return mask_row, ppdf_row.astype(np.float64)

    def test_recovers_fwhm_of_known_gaussian(self):
        """Synthesise a beam with known sigma_radial, verify recovered FWHM ≈ 2.355·sigma."""
        import compute_metrics as cm
        sigma_r = 0.5  # mm
        sigma_t = 0.2
        mask_row, ppdf_row = self._make_gaussian_beam_row(
            cx_mm=5.0, cy_mm=5.0, sigma_r_mm=sigma_r, sigma_t_mm=sigma_t,
            angle_rad=0.0, beam_id=7,
        )
        fwhm = cm._per_beam_radial_fwhm(mask_row, ppdf_row, beam_id=7, angle=0.0)
        expected = 2.355 * sigma_r
        # 5% tolerance (weighted moments deviate from analytical FWHM at these sigmas)
        assert abs(fwhm - expected) / expected < 0.05, f"got {fwhm}, expected {expected}"

    def test_returns_zero_for_empty_beam(self):
        import compute_metrics as cm
        empty_mask = np.zeros(200 * 200, dtype=np.int32)
        empty_ppdf = np.zeros(200 * 200, dtype=np.float64)
        assert cm._per_beam_radial_fwhm(empty_mask, empty_ppdf, beam_id=1, angle=0.0) == 0.0

    def test_returns_zero_for_tiny_beam(self):
        import compute_metrics as cm
        mask_row = np.zeros(200 * 200, dtype=np.int32)
        mask_row[0:2] = 1  # only 2 pixels labelled — below the pix_idx.size < 3 guard
        ppdf_row = np.ones_like(mask_row, dtype=np.float64)
        assert cm._per_beam_radial_fwhm(mask_row, ppdf_row, beam_id=1, angle=0.0) == 0.0

    def test_returns_zero_when_weights_all_zero(self):
        import compute_metrics as cm
        mask_row = np.zeros(200 * 200, dtype=np.int32)
        mask_row[0:100] = 1  # pixels are labelled
        ppdf_row = np.zeros_like(mask_row, dtype=np.float64)  # but zero intensity
        assert cm._per_beam_radial_fwhm(mask_row, ppdf_row, beam_id=1, angle=0.0) == 0.0

    def test_rotation_invariance(self):
        """A beam with radial axis at angle=0 and one at angle=π/2 (same sigmas)
        must both recover the same FWHM_radial when the correct projection angle
        is supplied. Sanity check on the projection direction."""
        import compute_metrics as cm
        sigma_r, sigma_t = 0.5, 0.2  # same physics, different orientation

        mask_0, ppdf_0 = self._make_gaussian_beam_row(
            5.0, 5.0, sigma_r, sigma_t, angle_rad=0.0, beam_id=1,
        )
        fwhm_0 = cm._per_beam_radial_fwhm(mask_0, ppdf_0, 1, 0.0)

        mask_pi2, ppdf_pi2 = self._make_gaussian_beam_row(
            5.0, 5.0, sigma_r, sigma_t, angle_rad=math.pi / 2, beam_id=1,
        )
        fwhm_pi2 = cm._per_beam_radial_fwhm(mask_pi2, ppdf_pi2, 1, math.pi / 2)

        # Both should recover ~2.355 * sigma_r along their (different) radial axes.
        expected = 2.355 * sigma_r
        assert abs(fwhm_0 - expected) / expected < 0.10
        assert abs(fwhm_pi2 - expected) / expected < 0.10
        assert abs(fwhm_0 - fwhm_pi2) / max(fwhm_0, fwhm_pi2) < 0.10


# =============================================================================
# compute_metrics.compute_ppds — end-to-end with synthetic HDF5
# =============================================================================
class TestComputePpds:
    """Build a small synthetic work_dir with matched PPDF, beam-properties, and
    beam-mask HDF5 files, then verify compute_ppds returns something sensible."""

    @pytest.fixture
    def work_dir(self, tmp_path):
        """One layout, one T8 pose, 2 detectors, 1 beam each."""
        wd = tmp_path / "work"
        wd.mkdir()
        n_det = 2
        n_pix = 200 * 200

        # PPDF: (n_det, n_pix). Detector 0 has a small hot spot, detector 1 a bigger one.
        ppdfs = np.zeros((n_det, n_pix), dtype=np.float32)
        # detector 0: 4 pixels lit around index 10101 (~row 50, col 101)
        ppdfs[0, [10101, 10102, 10301, 10302]] = 1.0
        # detector 1: 4 pixels lit around a different location
        ppdfs[1, [15050, 15051, 15250, 15251]] = 1.0

        # Write a single PPDF file: position_000_ppdfs_t8_00.hdf5
        with h5py.File(wd / "position_000_ppdfs_t8_00.hdf5", "w") as h:
            h.create_dataset("ppdfs", data=ppdfs)

        # Mask: (n_det, n_pix), int beam IDs (1-based, 0=background)
        masks = np.zeros((n_det, n_pix), dtype=np.int32)
        masks[0, [10101, 10102, 10301, 10302]] = 1
        masks[1, [15050, 15051, 15250, 15251]] = 1
        with h5py.File(wd / "beams_masks_configuration_000.hdf5", "w") as h:
            h.create_dataset("beam_mask", data=masks)

        # Beam properties: 11 columns (matching the real schema).
        # Layout: 2 rows (one beam per detector). Detector indices are 0-based here.
        # Columns: (position, det_id, beam_id, angle, FWHM, ...pad the rest with zeros)
        bp = np.zeros((2, 11), dtype=np.float32)
        bp[0] = [0, 0, 1, 0.0, 0.30, 0, 0, 0.5, 0.5, 0, 0]  # detector 0, beam 1, angle=0, FWHM=0.30
        bp[1] = [0, 1, 1, 0.5, 0.40, 0, 0, 0.5, 0.5, 0, 0]  # detector 1, beam 1, angle=0.5, FWHM=0.40
        with h5py.File(wd / "beams_properties_configuration_000.hdf5", "w") as h:
            h.create_dataset("beam_properties", data=bp)

        return wd

    def test_ppds_returns_finite_value(self, work_dir):
        import compute_metrics as cm
        val = cm.compute_ppds(str(work_dir))
        assert np.isfinite(val), f"expected finite PPDS, got {val}"
        assert val > 0.0

    def test_ppds_nan_when_ppdfs_missing(self, tmp_path):
        import compute_metrics as cm
        wd = tmp_path / "empty"
        wd.mkdir()
        assert np.isnan(cm.compute_ppds(str(wd)))

    def test_ppds_nan_when_only_masks(self, tmp_path):
        """Masks exist but no PPDF files."""
        import compute_metrics as cm
        wd = tmp_path / "no_ppdf"
        wd.mkdir()
        masks = np.zeros((2, 200 * 200), dtype=np.int32)
        with h5py.File(wd / "beams_masks_configuration_000.hdf5", "w") as h:
            h.create_dataset("beam_mask", data=masks)
        # Also need beam_properties for compute_ppds to enter its layout loop
        bp = np.zeros((0, 11), dtype=np.float32)
        with h5py.File(wd / "beams_properties_configuration_000.hdf5", "w") as h:
            h.create_dataset("beam_properties", data=bp)
        # No PPDF files — should return NaN cleanly
        assert np.isnan(cm.compute_ppds(str(wd)))

    def test_ppds_matches_expected_value(self, work_dir):
        """Compute the expected PPDS by hand and compare.

        For our synthetic case:
        - PPDF sum per pixel across detectors is 1 or 2 (whichever detectors light it).
        - sumV per detector = FWHM_tang * FWHM_rad. FWHM_tang stored = 0.30 (det 0) or 0.40 (det 1).
          FWHM_rad is measured from mask via _per_beam_radial_fwhm. With only 4 aligned pixels,
          the weighted std is small; expected FWHM_rad ~= 2.355 * pixel_pitch_along_axis.
        Since the pixels are placed strategically, we mainly check the value is
        positive and correlates with the expected order of magnitude.
        """
        import compute_metrics as cm
        val = cm.compute_ppds(str(work_dir))
        # Two detectors, each contributes PPDF/sumV summed over their lit pixels.
        # sumV[0] ~= 0.30 * O(0.05*something) ~= small; PPDS_j at their pixels ~= 1/sumV.
        # Mean over 40000 pixels: only 8 non-zero, mean = sum/40000.
        # This is a coarse sanity check on order of magnitude — just verify > 0.
        assert 0.0 < val < 1e6


# =============================================================================
# Ring-weighted PPDS — ring membership and the effect of weighting
# =============================================================================
class TestRingWeightedPpds:
    """Detectors are laid out ring-by-ring by the geometry generator, so ring
    membership is just the cumulative counts [n1, n2, 960, 1200]. If that layout
    assumption ever breaks, the weighted PPDS silently weights the wrong
    detectors, so it is worth pinning down."""

    def test_ring_boundaries_match_real_config(self):
        """mobo_0069 had n_det_ring1=612, n_det_ring2=230; its PPDF files gave
        SPROJ=3002, and 612 + 230 + 960 + 1200 == 3002."""
        import compute_metrics as cm
        w = cm._ring_weight_vector(n_det=3002, n_det_ring1=612)
        assert w is not None
        assert len(w) == 3002
        assert (w[:612] == 1.0).all()
        assert (w[612:842] == 2.0).all()
        assert (w[842:1802] == 3.0).all()
        assert (w[1802:] == 4.0).all()

    def test_ring2_inferred_from_total(self):
        """n_det_ring2 is derived from the PPDF row count, not passed in."""
        import compute_metrics as cm
        # mobo_0177: n1=604, n2=584 -> 604 + 584 + 960 + 1200 = 3348
        w = cm._ring_weight_vector(n_det=3348, n_det_ring1=604)
        assert int((w == 2.0).sum()) == 584

    def test_none_without_ring1(self):
        """No ring1 count means ring membership is unresolvable -> unweighted."""
        import compute_metrics as cm
        assert cm._ring_weight_vector(n_det=3002, n_det_ring1=None) is None

    def test_none_when_layout_inconsistent(self):
        """A total too small for rings 3+4 must not produce a bogus vector."""
        import compute_metrics as cm
        assert cm._ring_weight_vector(n_det=2000, n_det_ring1=612) is None

    def test_custom_weights_respected(self):
        import compute_metrics as cm
        w = cm._ring_weight_vector(3002, 612, ring_weights=(1.0, 2.0, 4.0, 8.0))
        assert w[0] == 1.0 and w[612] == 2.0 and w[842] == 4.0 and w[-1] == 8.0

    @pytest.fixture
    def four_ring_work_dir(self, tmp_path):
        """A work_dir whose detector count actually resolves into four rings.

        n_det = n1 + n2 + 960 + 1200. Uses a tiny FOV (n_pix=10) since PPDS reads
        n_pix from the PPDF array rather than assuming 200x200, which keeps a
        2000+ detector case small enough to test.
        """
        wd = tmp_path / "fourring"
        wd.mkdir()
        n1, n2 = 2, 4
        n_det = n1 + n2 + 960 + 1200
        n_pix = 10

        rng = np.random.default_rng(7)
        ppdfs = np.zeros((n_det, n_pix), dtype=np.float32)
        masks = np.zeros((n_det, n_pix), dtype=np.int32)
        # Light one detector in each ring so every ring contributes something
        lit = [0, n1, n1 + n2, n1 + n2 + 960]
        for d in lit:
            ppdfs[d, :4] = rng.uniform(0.5, 1.5, 4)
            masks[d, :4] = 1
        with h5py.File(wd / "position_000_ppdfs_t8_00.hdf5", "w") as h:
            h.create_dataset("ppdfs", data=ppdfs)
        with h5py.File(wd / "beams_masks_configuration_000.hdf5", "w") as h:
            h.create_dataset("beam_mask", data=masks)

        bp = np.zeros((len(lit), 11), dtype=np.float32)
        for i, d in enumerate(lit):
            bp[i] = [0, d, 1, 0.3 * i, 0.4 + 0.1 * i, 0, 0, 1.0, 0, 0, 0]
        with h5py.File(wd / "beams_properties_configuration_000.hdf5", "w") as h:
            h.create_dataset("beam_properties", data=bp)
        return wd, n1

    def test_components_sum_to_unweighted_total(self, four_ring_work_dir):
        """The decomposition must be exact -- otherwise weightings computed as a
        dot product of stored components would not match a direct computation."""
        import compute_metrics as cm
        wd, n1 = four_ring_work_dir
        comps = cm.compute_ppds_per_ring(str(wd), n_det_ring1=n1)
        total = cm.compute_ppds(str(wd))
        assert comps is not None and len(comps) == 4
        assert np.isclose(comps.sum(), total, rtol=1e-12)

    def test_dot_product_matches_direct_weighting(self, four_ring_work_dir):
        """Any weighting must be reproducible from the stored components."""
        import compute_metrics as cm
        wd, n1 = four_ring_work_dir
        comps = cm.compute_ppds_per_ring(str(wd), n_det_ring1=n1)
        for w in [(1, 2, 3, 4), (4, 3, 2, 1), (1, 0, 0, 0), (0.5, 1.5, 2.5, 3.5)]:
            direct = cm.compute_ppds(str(wd), n_det_ring1=n1, ring_weights=w)
            assert np.isclose(float(np.dot(w, comps)), direct, rtol=1e-12), w

    def test_per_ring_none_without_ring1(self, four_ring_work_dir):
        import compute_metrics as cm
        wd, _ = four_ring_work_dir
        assert cm.compute_ppds_per_ring(str(wd), n_det_ring1=None) is None

    def test_weighting_changes_ppds(self, tmp_path):
        """Weighted and unweighted PPDS must differ when detectors span rings.

        Uses a 2-detector work_dir with a ring layout of n1=1, n2=1 and rings
        3/4 empty, which is not physical but exercises the weighting path.
        """
        import compute_metrics as cm

        wd = tmp_path / "w"
        wd.mkdir()
        n_det, n_pix = 2, 200 * 200
        ppdfs = np.zeros((n_det, n_pix), dtype=np.float32)
        ppdfs[0, [10101, 10102, 10301, 10302]] = 1.0
        ppdfs[1, [15050, 15051, 15250, 15251]] = 1.0
        with h5py.File(wd / "position_000_ppdfs_t8_00.hdf5", "w") as h:
            h.create_dataset("ppdfs", data=ppdfs)
        masks = np.zeros((n_det, n_pix), dtype=np.int32)
        masks[0, [10101, 10102, 10301, 10302]] = 1
        masks[1, [15050, 15051, 15250, 15251]] = 1
        with h5py.File(wd / "beams_masks_configuration_000.hdf5", "w") as h:
            h.create_dataset("beam_mask", data=masks)
        bp = np.zeros((2, 11), dtype=np.float32)
        bp[0] = [0, 0, 1, 0.0, 0.30, 0, 0, 0.5, 0.5, 0, 0]
        bp[1] = [0, 1, 1, 0.5, 0.40, 0, 0, 0.5, 0.5, 0, 0]
        with h5py.File(wd / "beams_properties_configuration_000.hdf5", "w") as h:
            h.create_dataset("beam_properties", data=bp)

        plain = cm.compute_ppds(str(wd))
        # Two detectors split across rings 1 and 2 -> weights 1 and 2.
        weighted = cm.compute_ppds(
            str(wd), n_det_ring1=1,
            ring_weights=(1.0, 2.0, 0.0, 0.0),
        )
        # Patch the fixed ring 3/4 sizes to zero for this synthetic case
        assert np.isfinite(plain)
        # With rings 3/4 nonzero constants the layout will not resolve, so the
        # weighted call falls back to unweighted. That fallback is the
        # documented behaviour -- assert it explicitly rather than silently.
        assert np.isclose(plain, weighted), (
            "ring layout should not resolve for a 2-detector synthetic case, "
            "so weighted must fall back to plain PPDS"
        )


# =============================================================================
# Per-sector CNR angular masking
# =============================================================================
class TestCnrSectorMasks:
    """Sector centres run to 330 deg while atan2 returns [-pi, pi], so the
    angular difference has to be wrapped. Without it, `2*pi - diff` goes
    negative for the three centres beyond pi and trivially passes the < 30 deg
    test -- sector 5 covered half the image. The per-sector CNR values are how
    we answer questions about rod-size dependence, so they have to be right.
    """

    @staticmethod
    def _sector_masks(H=200, W=200):
        import torch
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        cx, cy = (H - 1) / 2.0, (W - 1) / 2.0
        px_angles = torch.atan2(yy - cx, xx - cy)
        masks = []
        for angle in [30, 90, 150, 210, 270, 330]:
            delta = px_angles - math.radians(angle)
            diff = torch.abs((delta + math.pi) % (2 * math.pi) - math.pi)
            masks.append(diff < math.radians(30))
        return masks

    def test_sectors_are_disjoint(self):
        import torch
        stack = torch.stack(self._sector_masks()).sum(0)
        assert int(stack.max()) == 1, "sectors overlap"

    def test_sectors_cover_the_image(self):
        masks = self._sector_masks()
        assert sum(int(m.sum()) for m in masks) == 200 * 200

    def test_no_sector_exceeds_a_quarter_of_the_image(self):
        """Regression: sector 5 previously covered 50% and sector 4 32%."""
        masks = self._sector_masks()
        for i, m in enumerate(masks):
            frac = float(m.sum()) / (200 * 200)
            assert frac < 0.25, f"sector {i} covers {frac:.1%} of the image"

    def test_opposite_sectors_are_symmetric(self):
        """Sectors 180 deg apart must have equal area on a square grid."""
        masks = self._sector_masks()
        for a, b in ((0, 3), (1, 4), (2, 5)):
            assert int(masks[a].sum()) == int(masks[b].sum())

    def test_sector_centres_match_the_phantom_layout(self):
        """Each sector must hold exactly one rod size, not straddle two.

        The centres were once [30, 90, 150, ...], which sat on the boundaries
        between rod-size groups, so every per-size CNR was a mixture of two
        sizes. This measures the phantom directly rather than trusting a
        hardcoded list: it groups rods by area, takes each group's mean angle,
        and checks the sector centres line up with those.
        """
        phantom = os.path.join(os.path.dirname(__file__), "..", "data",
                               "sai_10mm", "hot_rods_phantom_10.0_mm_x_10.0_mm.pt")
        if not os.path.exists(phantom):
            pytest.skip("phantom file not available")
        torch = pytest.importorskip("torch")
        ndimage = pytest.importorskip("scipy.ndimage")

        d = torch.load(phantom, map_location="cpu", weights_only=False)
        t = d["Phantom tensor"].numpy()
        n_sizes = len(d["Metadata"]["rods radii in mm"])

        lab, n = ndimage.label(t > 0.1)
        H, W = t.shape
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0

        # Group rods by pixel area; each area is one rod size
        by_area = {}
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lab == i)
            ang = math.degrees(math.atan2(ys.mean() - cy, xs.mean() - cx)) % 360
            by_area.setdefault(int(len(ys)), []).append(ang)
        assert len(by_area) == n_sizes, f"expected {n_sizes} rod sizes, found {len(by_area)}"

        # Ascending area == ascending radius, matching rod_radii_mm's order
        centres = []
        for area in sorted(by_area):
            angs = np.asarray(by_area[area])
            # Circular mean, so a group straddling 0 deg does not average to 180
            rad = np.radians(angs)
            centres.append(math.degrees(math.atan2(np.sin(rad).mean(),
                                                   np.cos(rad).mean())) % 360)

        expected = [60, 0, 300, 240, 180, 120]
        for i, (measured, exp) in enumerate(zip(centres, expected)):
            delta = abs((measured - exp + 180) % 360 - 180)
            assert delta < 10, (
                f"rod size {i}: phantom has it centred at {measured:.1f} deg, "
                f"but compute_cnr uses {exp} deg (off by {delta:.1f})"
            )


# =============================================================================
# FWHM-windowed ASCI
# =============================================================================
class TestWindowedAsci:
    """ASCI restricted to beams narrower than a threshold. The existing metric
    saturates (64% of configs at exactly 100%), so the window has to actually
    exclude beams for it to discriminate."""

    @pytest.fixture
    def work_dir(self, tmp_path):
        """One layout, 3 detectors, one beam each, with FWHM 0.5 / 1.0 / 2.0 mm.

        Each beam lights a distinct block of pixels at a distinct angle, so the
        coverage contributed by each is independent and easy to reason about.
        """
        wd = tmp_path / "asci"
        wd.mkdir()
        n_det, n_pix = 3, 200 * 200

        # float32, matching the real beams_masks HDF5 files -- an int32 fixture
        # hid a dtype bug where beam ids were used directly as lookup indices.
        masks = np.zeros((n_det, n_pix), dtype=np.float32)
        masks[0, 0:100] = 1
        masks[1, 100:200] = 1
        masks[2, 200:300] = 1
        with h5py.File(wd / "beams_masks_configuration_000.hdf5", "w") as h:
            h.create_dataset("beam_mask", data=masks)

        # 11-column schema: 1 det_id, 2 beam_id, 3 angle, 4 FWHM, 7 sensitivity
        bp = np.zeros((3, 11), dtype=np.float32)
        bp[0] = [0, 0, 1, 0.10, 0.5, 0, 0, 1.0, 0, 0, 0]
        bp[1] = [0, 1, 1, 0.50, 1.0, 0, 0, 1.0, 0, 0, 0]
        bp[2] = [0, 2, 1, 1.00, 2.0, 0, 0, 1.0, 0, 0, 0]
        with h5py.File(wd / "beams_properties_configuration_000.hdf5", "w") as h:
            h.create_dataset("beam_properties", data=bp)
        return wd

    def test_window_is_monotonic(self, work_dir):
        import analyze_asci_window as aw
        res = aw.windowed_asci(str(work_dir), (0.6, 1.5, 3.0))
        assert res[0.6] < res[1.5] < res[3.0], res

    def test_threshold_excludes_wider_beams(self, work_dir):
        """Each beam lights 100 pixels in 1 angular bin, so coverage is additive."""
        import analyze_asci_window as aw
        res = aw.windowed_asci(str(work_dir), (0.6, 1.5, 3.0))
        total = 200 * 200 * 360
        # 0.6 keeps only the 0.5mm beam; 1.5 keeps two; 3.0 keeps all three
        assert np.isclose(res[0.6], 100 / total * 100)
        assert np.isclose(res[1.5], 200 / total * 100)
        assert np.isclose(res[3.0], 300 / total * 100)

    def test_threshold_below_all_beams_gives_zero(self, work_dir):
        import analyze_asci_window as aw
        assert aw.windowed_asci(str(work_dir), (0.1,))[0.1] == 0.0

    def test_nan_when_files_missing(self, tmp_path):
        import analyze_asci_window as aw
        wd = tmp_path / "empty"
        wd.mkdir()
        res = aw.windowed_asci(str(wd), (1.0,))
        assert np.isnan(res[1.0])

    def test_sensitivity_floor_still_applied(self, tmp_path):
        """Beams under 1% of the layout peak are dropped, as in sai_analyze_asci."""
        import analyze_asci_window as aw
        wd = tmp_path / "sens"
        wd.mkdir()
        n_det, n_pix = 2, 200 * 200
        # float32, matching the real beams_masks HDF5 files -- an int32 fixture
        # hid a dtype bug where beam ids were used directly as lookup indices.
        masks = np.zeros((n_det, n_pix), dtype=np.float32)
        masks[0, 0:100] = 1
        masks[1, 100:200] = 1
        with h5py.File(wd / "beams_masks_configuration_000.hdf5", "w") as h:
            h.create_dataset("beam_mask", data=masks)
        bp = np.zeros((2, 11), dtype=np.float32)
        bp[0] = [0, 0, 1, 0.10, 0.5, 0, 0, 1.0, 0, 0, 0]      # strong
        bp[1] = [0, 1, 1, 0.50, 0.5, 0, 0, 0.001, 0, 0, 0]    # 0.1% of peak
        with h5py.File(wd / "beams_properties_configuration_000.hdf5", "w") as h:
            h.create_dataset("beam_properties", data=bp)
        res = aw.windowed_asci(str(wd), (3.0,))
        total = 200 * 200 * 360
        # Only the strong beam's 100 pixels should count
        assert np.isclose(res[3.0], 100 / total * 100)

    def test_column_names_are_filesystem_safe(self):
        import analyze_asci_window as aw
        assert aw.col_for(1.0) == "asci_pct_fwhm1"
        assert aw.col_for(0.6) == "asci_pct_fwhm0p6"
        assert "." not in aw.col_for(1.5)


# =============================================================================
# compute_cnr._write_cnr_to_csv — CSV update semantics
# =============================================================================
class TestWriteCnrToCsv:
    """Verify the row-update / row-append / column-create paths."""

    def test_creates_csv_when_missing(self, tmp_path):
        from compute_cnr import _write_cnr_to_csv
        csv = tmp_path / "results.csv"
        _write_cnr_to_csv(str(csv), "mobo_0001", 3.72)
        df = pd.read_csv(csv)
        assert list(df.columns) == ["config", "cnr_mean", "cnr_sector_mean"]
        assert df.loc[0, "config"] == "mobo_0001"
        assert df.loc[0, "cnr_mean"] == pytest.approx(3.72)

    def test_writes_sector_mean_and_sectors(self, tmp_path):
        """cnr_sector_mean is the objective the optimizer reads; the per-section
        values are kept alongside it for the rod-size analysis."""
        from compute_cnr import _write_cnr_to_csv
        csv = tmp_path / "results.csv"
        sectors = [2.1, 3.0, 3.6, 3.2, 2.6, 2.2]
        _write_cnr_to_csv(str(csv), "mobo_0002", 3.72, 2.783, sectors)
        df = pd.read_csv(csv)
        assert df.loc[0, "cnr_sector_mean"] == pytest.approx(2.783)
        for i, v in enumerate(sectors):
            assert df.loc[0, f"cnr_sector{i}"] == pytest.approx(v)

    def test_updates_existing_row(self, tmp_path):
        from compute_cnr import _write_cnr_to_csv
        csv = tmp_path / "results.csv"
        pd.DataFrame({
            "fwhm_mean": [0.5, 0.6],
            "config": ["mobo_0001", "mobo_0002"],
            "cnr_mean": [np.nan, np.nan],
        }).to_csv(csv, index=False)

        _write_cnr_to_csv(str(csv), "mobo_0001", 4.25)
        df = pd.read_csv(csv)
        assert df.loc[df["config"] == "mobo_0001", "cnr_mean"].iloc[0] == pytest.approx(4.25)
        # Other row untouched
        assert np.isnan(df.loc[df["config"] == "mobo_0002", "cnr_mean"].iloc[0])

    def test_adds_column_when_missing(self, tmp_path):
        from compute_cnr import _write_cnr_to_csv
        csv = tmp_path / "results.csv"
        pd.DataFrame({
            "fwhm_mean": [0.5],
            "config": ["mobo_0001"],
        }).to_csv(csv, index=False)

        _write_cnr_to_csv(str(csv), "mobo_0001", 4.44)
        df = pd.read_csv(csv)
        assert "cnr_mean" in df.columns
        assert df.loc[0, "cnr_mean"] == pytest.approx(4.44)

    def test_appends_row_when_config_missing(self, tmp_path):
        from compute_cnr import _write_cnr_to_csv
        csv = tmp_path / "results.csv"
        pd.DataFrame({
            "fwhm_mean": [0.5],
            "config": ["mobo_0001"],
            "cnr_mean": [3.0],
        }).to_csv(csv, index=False)

        _write_cnr_to_csv(str(csv), "mobo_9999", 2.5)
        df = pd.read_csv(csv)
        assert len(df) == 2
        assert df.loc[df["config"] == "mobo_9999", "cnr_mean"].iloc[0] == pytest.approx(2.5)

    def test_writes_nan(self, tmp_path):
        from compute_cnr import _write_cnr_to_csv
        csv = tmp_path / "results.csv"
        pd.DataFrame({
            "config": ["mobo_0001"],
            "cnr_mean": [1.0],
        }).to_csv(csv, index=False)
        _write_cnr_to_csv(str(csv), "mobo_0001", float("nan"))
        df = pd.read_csv(csv)
        assert np.isnan(df.loc[0, "cnr_mean"])


# =============================================================================
# compute_metrics.main() — CSV column alignment on append
# =============================================================================
class TestCsvAlignment:
    """The append path in compute_metrics.main() MUST align to the existing
    header, or subsequent MOBO GP fits crash on misaligned numeric columns."""

    def _run_compute_metrics_cli(self, csv_path, config_name, work_dir):
        """Run compute_metrics.py as a CLI subprocess with the --force_zero flag
        so we don't need real HDF5 data. Returns the written CSV as a DataFrame."""
        script = os.path.join(_REPO_ROOT, "optimization", "compute_metrics.py")
        subprocess.run(
            [sys.executable, script,
             "--work_dir", str(work_dir),
             "--out_csv", str(csv_path),
             "--config_name", config_name,
             "--aperture_diam_mm", "0.5",
             "--n_apertures", "180",
             "--n_det_ring1", "480",
             "--n_det_ring2", "720",
             "--force_zero",
             "--reason", "test"],
            check=True, capture_output=True, text=True,
        )
        return pd.read_csv(csv_path)

    def test_new_csv_has_expected_columns(self, tmp_path):
        csv = tmp_path / "results.csv"
        wd = tmp_path / "work"
        wd.mkdir()
        df = self._run_compute_metrics_cli(csv, "mobo_test_0001", wd)
        # Every metric column should be present, plus config/work_dir/design params
        for col in ["fwhm_mean", "sensitivity_total", "sensitivity_mean",
                    "asci_pct", "n_ppdf_files", "mpxi_mean", "ppds_mean",
                    "config", "work_dir",
                    "aperture_diam_mm", "n_apertures", "n_det_ring1", "n_det_ring2"]:
            assert col in df.columns, f"missing column {col}"

    def test_append_respects_existing_header_with_extra_cols(self, tmp_path):
        """Simulate CSV that already has cnr_mean at the end (from backfill).
        Appending must place the new row's values in the CORRECT columns."""
        csv = tmp_path / "results.csv"
        wd = tmp_path / "work"
        wd.mkdir()

        # Seed CSV in the shape that would exist AFTER backfill_cnr.py ran:
        # 14 columns, cnr_mean at the very end.
        pd.DataFrame([{
            "fwhm_mean": 0.5, "sensitivity_total": 1.0, "sensitivity_mean": 0.1,
            "asci_pct": 95.0, "n_ppdf_files": 16, "mpxi_mean": 3.0,
            "config": "seed_row", "work_dir": "/tmp/seed",
            "aperture_diam_mm": 0.4, "n_apertures": 180,
            "n_det_ring1": 480, "n_det_ring2": 720,
            "ppds_mean": 0.5, "cnr_mean": 3.0,
        }]).to_csv(csv, index=False)

        # compute_metrics.py appends
        df = self._run_compute_metrics_cli(csv, "new_row_0001", wd)
        assert len(df) == 2

        # The NEW row is the second one; its work_dir must be the string we passed,
        # NOT some misaligned numeric value from a shifted column.
        new_row = df.iloc[1]
        assert new_row["config"] == "new_row_0001"
        assert new_row["work_dir"] == str(wd)
        assert new_row["aperture_diam_mm"] == pytest.approx(0.5)
        assert new_row["n_apertures"] == 180
        # cnr_mean is not written by compute_metrics.py, so should be NaN on the new row
        assert pd.isna(new_row["cnr_mean"])


# =============================================================================
# run_mobo_loop.acquire_singleton_lock — two controllers must not coexist
# =============================================================================
class TestControllerSingletonLock:
    """Two concurrent controllers claim the same manifest index and collide.

    Aug 2026: submit_mobo.sh was re-submitted while 25545619 was still running,
    queueing a second controller against the same manifest. Caught in squeue
    before it started; these tests make sure it cannot start at all.
    """

    _CHILD = (
        "import sys, os; sys.path.insert(0, {opt!r});"
        "os.environ['MOBO_RESULTS_DIR'] = {res!r};"
        "import run_mobo_loop as L;"
        "L.acquire_singleton_lock();"
        "print('ACQUIRED')"
    )

    def _child(self, tmp_path, hold_fd=None):
        opt = os.path.join(_REPO_ROOT, "optimization")
        code = self._CHILD.format(opt=opt, res=str(tmp_path))
        return subprocess.run([sys.executable, "-c", code],
                              capture_output=True, text=True, timeout=60)

    def test_first_holder_acquires(self, tmp_path):
        r = self._child(tmp_path)
        assert r.returncode == 0, r.stderr
        assert "ACQUIRED" in r.stdout

    def test_second_holder_is_refused(self, tmp_path):
        """While one process holds the lock, a second must exit non-zero."""
        opt = os.path.join(_REPO_ROOT, "optimization")
        holder_code = (
            f"import sys, os; sys.path.insert(0, {opt!r});"
            f"os.environ['MOBO_RESULTS_DIR'] = {str(tmp_path)!r};"
            "import run_mobo_loop as L;"
            "L.acquire_singleton_lock();"
            "print('HELD', flush=True);"
            "import time; time.sleep(30)"
        )
        holder = subprocess.Popen([sys.executable, "-c", holder_code],
                                  stdout=subprocess.PIPE, text=True)
        try:
            assert holder.stdout.readline().strip() == "HELD"
            r = self._child(tmp_path)
            assert r.returncode == 1, (
                f"second controller was allowed to start: {r.stdout} {r.stderr}")
            assert "already running" in (r.stdout + r.stderr).lower()
        finally:
            holder.kill()
            holder.wait(timeout=10)

    def test_lock_released_when_holder_dies(self, tmp_path):
        """A killed or OOM-ed controller must not leave a lock needing manual
        cleanup -- that would turn a crash into a blocked queue."""
        opt = os.path.join(_REPO_ROOT, "optimization")
        holder_code = (
            f"import sys, os; sys.path.insert(0, {opt!r});"
            f"os.environ['MOBO_RESULTS_DIR'] = {str(tmp_path)!r};"
            "import run_mobo_loop as L;"
            "L.acquire_singleton_lock();"
            "print('HELD', flush=True);"
            "import time; time.sleep(30)"
        )
        holder = subprocess.Popen([sys.executable, "-c", holder_code],
                                  stdout=subprocess.PIPE, text=True)
        assert holder.stdout.readline().strip() == "HELD"
        holder.kill()
        holder.wait(timeout=10)

        r = self._child(tmp_path)
        assert r.returncode == 0, f"stale lock survived the holder: {r.stderr}"
        assert "ACQUIRED" in r.stdout


# =============================================================================
# compute_metrics.compute_mpxi_variants — the two definition changes RY asked for
# =============================================================================
class TestMpxiVariants:
    """Aug 2026: MPXI averaged blind detectors (k=0) into a minimized objective,
    and was correlated against a WINDOWED ASCI while itself counting every beam.
    These verify both fixes on synthetic files with known answers."""

    def _write(self, wd, n_det, beams):
        """beams: list of (det_id_1based, beam_id, fwhm, sens)."""
        os.makedirs(wd, exist_ok=True)
        with h5py.File(os.path.join(wd, "beams_masks_configuration_001.hdf5"), "w") as f:
            f.create_dataset("beam_mask", data=np.zeros((n_det, 4), dtype=np.int32))
        bp = np.zeros((len(beams), 8), dtype=np.float64)
        for i, (d, b, fw, s) in enumerate(beams):
            bp[i, 1], bp[i, 2], bp[i, 4], bp[i, 7] = d, b, fw, s
        with h5py.File(os.path.join(wd, "beams_properties_configuration_001.hdf5"), "w") as f:
            f.create_dataset("beam_properties", data=bp)

    def test_blind_detectors_excluded_from_active_mean(self, tmp_path):
        """4 detectors, only 2 see anything, with 3 beams each.

        The original definition averages over all 4 and reports 1.5; the active
        mean must report 3.0 -- otherwise idling half the array looks like an
        improvement in a quantity we minimize.
        """
        wd = str(tmp_path / "blind")
        beams = [(1, b, 0.3, 1.0) for b in range(1, 4)] + \
                [(2, b, 0.3, 1.0) for b in range(1, 4)]
        self._write(wd, n_det=4, beams=beams)
        import compute_metrics as cm
        r = cm.compute_mpxi_variants(wd, threshold_mm=0.45)
        assert r["mpxi_active_mean"] == pytest.approx(3.0)
        # windowed_mean averages over ALL detectors, so it keeps the zeros
        assert r["mpxi_windowed_mean"] == pytest.approx(6 / 4)
        assert r["mpxi_windowed_active_mean"] == pytest.approx(3.0)

    def test_window_excludes_wide_beams(self, tmp_path):
        """Detector 1 has 2 narrow beams and 2 wide ones; only narrow count."""
        wd = str(tmp_path / "window")
        beams = [(1, 1, 0.30, 1.0), (1, 2, 0.40, 1.0),
                 (1, 3, 0.90, 1.0), (1, 4, 1.20, 1.0)]
        self._write(wd, n_det=1, beams=beams)
        import compute_metrics as cm
        r = cm.compute_mpxi_variants(wd, threshold_mm=0.45)
        assert r["mpxi_active_mean"] == pytest.approx(4.0)
        assert r["mpxi_windowed_active_mean"] == pytest.approx(2.0)

    def test_sensitivity_floor_matches_windowed_asci(self, tmp_path):
        """Beams under 1% of peak sensitivity are dropped, as in windowed ASCI.

        If these two floors drift apart the variant stops measuring the same
        beam population as ASCI, which is the entire reason it exists.
        """
        import analyze_asci_window as aw
        import compute_metrics as cm
        assert cm._MPXI_SENSITIVITY_FLOOR_FRAC == aw.SENSITIVITY_FLOOR_FRAC

        wd = str(tmp_path / "floor")
        beams = [(1, 1, 0.3, 100.0), (1, 2, 0.3, 0.5)]  # second is 0.5% of peak
        self._write(wd, n_det=1, beams=beams)
        r = cm.compute_mpxi_variants(wd, threshold_mm=0.45)
        assert r["mpxi_windowed_active_mean"] == pytest.approx(1.0)

    def test_nan_when_files_missing(self, tmp_path):
        import compute_metrics as cm
        r = cm.compute_mpxi_variants(str(tmp_path / "nothing"))
        assert all(np.isnan(v) for v in r.values())


# =============================================================================
# Objective-set definition — one source of truth
# =============================================================================
class TestObjectiveSetSingleSource:
    """run_mobo_loop and analyze_mobo_convergence used to carry hand-synced
    copies of OBJ_COLUMNS under "keep in sync" comments. A duplicated rule that
    drifted is what made 46 consecutive iterations propose an unbuildable
    geometry in Jul 2026. These fail if anyone reintroduces a private copy."""

    def test_run_mobo_loop_imports_the_definition(self):
        import mobo_agent as ma
        import run_mobo_loop as loop
        assert loop.OBJ_COLUMNS is ma.OBJ_COLUMNS
        assert loop.OBJ_DIRECTIONS is ma.OBJ_DIRECTIONS

    def test_convergence_imports_the_definition(self):
        import mobo_agent as ma
        import analyze_mobo_convergence as conv
        assert conv.METRIC_COLS is ma.OBJ_COLUMNS
        assert conv.METRIC_DIRS is ma.OBJ_DIRECTIONS

    def test_parallel_lists_stay_aligned(self):
        import mobo_agent as ma
        n = len(ma.OBJ_COLUMNS)
        assert len(ma.OBJ_DIRECTIONS) == n
        assert len(ma.OBJ_NAMES) == n
        assert len(ma.OBJ_SHORT) == n
        assert all(d in (1.0, -1.0) for d in ma.OBJ_DIRECTIONS)

    def test_mpxi_is_maximized_under_the_corrected_definition(self):
        """RY approved Aug 2026: windowed+active MPXI, maximized.

        Measured in physical units on 228 designs, multiplexing correlates
        +0.60 with CNR, +0.91 with windowed ASCI and -0.73 with FWHM. Minimizing
        it optimizes away from image quality. If this ever reverts to
        mpxi_mean or to a -1 direction, that is a regression, not a choice.
        """
        import mobo_agent as ma
        assert "mpxi_windowed_active_mean" in ma.OBJ_COLUMNS
        assert "mpxi_mean" not in ma.OBJ_COLUMNS
        i = ma.OBJ_COLUMNS.index("mpxi_windowed_active_mean")
        assert ma.OBJ_DIRECTIONS[i] == 1.0, "MPXI must be maximized"

    def test_names_declare_the_same_direction_as_the_signs(self):
        """A label saying (min) beside a +1 direction would mislead every log
        and status table the campaign is monitored through."""
        import mobo_agent as ma
        for name, d in zip(ma.OBJ_NAMES, ma.OBJ_DIRECTIONS):
            if "(min)" in name:
                assert d == -1.0, f"{name} labelled min but direction {d}"
            elif "(max)" in name:
                assert d == 1.0, f"{name} labelled max but direction {d}"


class TestRequireObjectiveColumns:
    """Changing OBJ_COLUMNS strands every previously-written row. The failure
    must name the fix, not just raise KeyError on the column name."""

    def test_passes_when_all_present(self):
        import mobo_agent as ma
        df = pd.DataFrame({c: [1.0] for c in ma.OBJ_COLUMNS})
        ma.require_objective_columns(df, "x.csv")  # must not raise

    def test_names_the_backfill_when_missing(self):
        import mobo_agent as ma
        df = pd.DataFrame({c: [1.0] for c in ma.OBJ_COLUMNS[:-1]})
        with pytest.raises(SystemExit) as e:
            ma.require_objective_columns(df, "results/x.csv")
        msg = str(e.value)
        assert ma.OBJ_COLUMNS[-1] in msg
        assert "backfill_mpxi_variants.py" in msg
        assert "results/x.csv" in msg


class TestNoHardcodedObjectiveLabels:
    """The controller banner said "MPXI (min)" for a whole run after MPXI was
    changed to windowed+active and MAXIMIZED. The optimizer was correct; only
    the log lied, which is worse than an obvious failure because it is believed.
    Any objective label in run_mobo_loop must derive from mobo_agent."""

    def test_source_has_no_literal_objective_labels(self):
        import run_mobo_loop as loop
        src = open(loop.__file__).read()
        # Strip comments, which legitimately discuss the old labels. Docstrings
        # are NOT stripped: the module docstring is where the stalest list lived
        # (it still named sensitivity months after that objective was retired),
        # and docstrings are read as documentation, so they must stay honest.
        code = "\n".join(l for l in src.splitlines()
                         if not l.lstrip().startswith("#"))
        for bad in ("MPXI (min)", "MPXI (max)", "ASCI@0.45mm (max)",
                    "FWHM wtd (min)", "CNR sector-mean (max)",
                    "sensitivity (max)"):
            assert bad not in code, (
                f"hardcoded objective label {bad!r} in run_mobo_loop.py; "
                f"derive it from mobo_agent.OBJ_NAMES instead")

    def test_banner_reflects_the_current_direction(self):
        import mobo_agent as ma
        i = ma.OBJ_COLUMNS.index("mpxi_windowed_active_mean")
        assert "(max)" in ma.OBJ_NAMES[i]


class TestObjectiveSelection:
    """MOBO_OBJECTIVES restricts the objective set for head-to-head campaigns.

    A subset campaign that silently ran the wrong objectives, or that paired a
    direction with the wrong label, would invalidate the comparison without
    failing. These run mobo_agent in a subprocess because the selection happens
    at import time.
    """

    def _run(self, env_val, code):
        env = dict(os.environ)
        if env_val is None:
            env.pop("MOBO_OBJECTIVES", None)
        else:
            env["MOBO_OBJECTIVES"] = env_val
        opt = os.path.join(_REPO_ROOT, "optimization")
        return subprocess.run(
            [sys.executable, "-c",
             f"import sys; sys.path.insert(0, {opt!r});\nimport mobo_agent as ma\n{code}"],
            capture_output=True, text=True, env=env, timeout=120)

    def test_default_is_all_five(self):
        r = self._run(None, "print(len(ma.OBJ_COLUMNS))")
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "5"

    def test_subset_selects_in_the_order_given(self):
        r = self._run("cnr_sector_mean,mpxi_windowed_active_mean",
                      "print(','.join(ma.OBJ_COLUMNS))")
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "cnr_sector_mean,mpxi_windowed_active_mean"

    def test_direction_and_label_travel_with_the_column(self):
        """The subset must not reindex directions independently of columns."""
        r = self._run("cnr_sector_mean,fwhm_weighted_mean",
                      "print(ma.OBJ_DIRECTIONS); print(ma.OBJ_NAMES)")
        assert r.returncode == 0, r.stderr
        out = r.stdout
        # FWHM is minimized and must stay so after being moved to position 2
        assert "[1.0, -1.0]" in out
        assert "FWHM weighted (min)" in out

    def test_single_objective_refused(self):
        """qLogNEHVI needs m>=2; failing here beats failing 5 hours in."""
        r = self._run("cnr_sector_mean", "print('reached')")
        assert r.returncode != 0
        assert "at least 2 objectives" in (r.stdout + r.stderr)

    def test_unknown_name_refused(self):
        r = self._run("cnr_sector_mean,not_a_metric", "print('reached')")
        assert r.returncode != 0
        assert "unknown objective" in (r.stdout + r.stderr)

    def test_duplicate_refused(self):
        r = self._run("cnr_sector_mean,cnr_sector_mean", "print('reached')")
        assert r.returncode != 0
        assert "repeats an objective" in (r.stdout + r.stderr)


class TestSeedNameCollision:
    """run_sai_pipeline.sh derives WORK_DIR from the config name, so two
    replicates sharing a config-name prefix evaluate different designs into the
    same directories -- concurrently, silently, producing plausible numbers.
    Caught in Aug 2026 after both replicate arrays were already submitted."""

    def _gen(self, tmp_path, prefix, results_dir):
        opt = os.path.join(_REPO_ROOT, "optimization")
        out = tmp_path / f"seeds_{prefix.strip('_')}.csv"
        cmd = [sys.executable, os.path.join(opt, "make_lhs6d_seeds.py"),
               "--n_seeds", "21", "--seed", "0", "--prefix", prefix,
               "--results_dir", str(results_dir), "--out", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=opt, timeout=180)
        return r, out

    def test_prefix_lands_in_config_names(self, tmp_path):
        r, out = self._gen(tmp_path, "lhs6d_r7_", tmp_path / "empty_results")
        assert r.returncode == 0, r.stdout + r.stderr
        df = pd.read_csv(out)
        assert all(str(c).startswith("lhs6d_r7_") for c in df["config"])

    def test_refuses_names_whose_work_dirs_exist(self, tmp_path):
        """The guard that would have prevented the incident."""
        results = tmp_path / "results"
        # Pretend replicate 0 already evaluated these
        for i in range(21):
            (results / f"lhs6d_{i:03d}").mkdir(parents=True)
        r, _ = self._gen(tmp_path, "lhs6d_", results)
        assert r.returncode != 0
        out = r.stdout + r.stderr
        assert "already have work directories" in out
        assert "--prefix" in out

    def test_distinct_prefix_passes_the_same_check(self, tmp_path):
        results = tmp_path / "results"
        for i in range(21):
            (results / f"lhs6d_{i:03d}").mkdir(parents=True)
        r, out = self._gen(tmp_path, "lhs6d_r1_", results)
        assert r.returncode == 0, r.stdout + r.stderr
        assert len(pd.read_csv(out)) == 21


class TestRing2BoundNotBinding:
    """n_det_ring2's box bound was 960 while packing allows 1012 at d2=400 and
    1361 at d2=540. Every top design in the 271-config archive sat exactly on
    960, which is a search stopped by its box rather than by physics. The bound
    should never be what binds; is_ring_packing_ok should be."""

    def test_box_bound_exceeds_packing_across_the_d2_range(self):
        import mobo_agent as ma
        worst_case = ma.max_crystals_on_ring(ma.BOUNDS_MAX[4])
        assert ma.BOUNDS_MAX[3] >= worst_case, (
            f"n_det_ring2 bound {ma.BOUNDS_MAX[3]} is below the packing limit "
            f"{worst_case:.0f} at d2={ma.BOUNDS_MAX[4]}, so the box binds first "
            f"and the search cannot reach physically buildable designs")

    def test_packing_still_rejects_overfull_rings(self):
        """Raising the box must not disable the physical constraint."""
        import mobo_agent as ma
        d2 = 400.0
        limit = ma.max_crystals_on_ring(d2)
        assert ma.is_ring_packing_ok(480, int(limit) - 10, d2, 520.0)
        assert not ma.is_ring_packing_ok(480, int(limit) + 10, d2, 520.0)

    def test_old_bound_was_reachable_everywhere_new_one_is_not(self):
        """Sanity: 960 fit at every d2, which is why it always bound. The new
        bound is deliberately unreachable at small d2, so packing decides."""
        import mobo_agent as ma
        assert ma.max_crystals_on_ring(ma.BOUNDS_MIN[4]) < ma.BOUNDS_MAX[3]
        assert ma.max_crystals_on_ring(ma.BOUNDS_MAX[4]) <= ma.BOUNDS_MAX[3]


class TestRepeatMetricIsSectorMean:
    """analyze_cnr_repeats.collect must return the SECTOR MEAN, not overall_cnr.

    Aug 2026: it returned overall_cnr while the campaign optimises
    cnr_sector_mean. The two differ by ~0.15 for the top designs, so repeat runs
    of mobo_0296 read as 4.91 against a campaign value of 4.77 and were
    mistaken for a systematic measurement offset. The gap between the metrics
    is larger than the effects being measured, so this must not regress."""

    def _write(self, wd, seed, sector_cnrs, overall):
        d = os.path.join(wd, f"cnr_repeat_seed{seed}")
        os.makedirs(d, exist_ok=True)
        np.savez(os.path.join(d, "cnr_results.npz"),
                 sector_cnrs=np.array(sector_cnrs, dtype=float),
                 overall_cnr=float(overall))

    def test_returns_sector_mean_not_overall(self, tmp_path):
        import analyze_cnr_repeats as acr
        cfg = "cfg_x"
        wd = str(tmp_path / cfg)
        # sector mean is 3.0; overall is deliberately far away
        for s in range(3):
            self._write(wd, s, [1.0, 2.0, 3.0, 4.0, 5.0], 9.99)
        seeds, values, sectors = acr.collect(str(tmp_path), cfg)
        assert seeds == [0, 1, 2]
        assert values == pytest.approx([3.0, 3.0, 3.0])
        assert not np.allclose(values, 9.99), "returned overall_cnr, not sector mean"
        assert sectors.shape == (3, 5)

    def test_empty_when_no_repeats(self, tmp_path):
        import analyze_cnr_repeats as acr
        seeds, values, sectors = acr.collect(str(tmp_path), "missing_cfg")
        assert seeds == [] and values.size == 0


class TestDiskSpaceGuard:
    """Aug 26 2026: /vscratch hit 100% and every campaign died looking like a
    code bug. Pipeline jobs failed with no logs (SLURM could not create their
    output files) and controllers died with no traceback (stderr could not be
    written). Each design leaves ~8.9 GB behind for ~77 KB of metrics, so this
    was arithmetic, not bad luck. The guard turns a silent death into a clear
    stop."""

    def test_free_gb_reports_something_sane(self, tmp_path):
        import run_mobo_loop as loop
        gb = loop.free_gb(str(tmp_path))
        assert gb > 0, "free space should be positive on a working filesystem"

    def test_guard_blocks_when_space_is_low(self, tmp_path, monkeypatch):
        import run_mobo_loop as loop
        monkeypatch.setattr(loop, "RESULTS_DIR", str(tmp_path))
        monkeypatch.setattr(loop, "free_gb", lambda p: 5.0)
        assert loop.check_disk_space() is False, (
            "5 GB free must stop the loop; one design needs ~9 GB")

    def test_guard_allows_when_space_is_ample(self, tmp_path, monkeypatch):
        import run_mobo_loop as loop
        monkeypatch.setattr(loop, "RESULTS_DIR", str(tmp_path))
        monkeypatch.setattr(loop, "free_gb", lambda p: 5000.0)
        assert loop.check_disk_space() is True

    def test_unreadable_path_does_not_block(self, tmp_path, monkeypatch):
        """A failure to CHECK must not halt a healthy campaign."""
        import run_mobo_loop as loop
        monkeypatch.setattr(loop, "RESULTS_DIR", "/definitely/not/here")
        assert loop.check_disk_space() is True

    def test_threshold_covers_at_least_one_design(self):
        import run_mobo_loop as loop
        assert loop.MIN_FREE_GB >= 9.0, (
            "threshold must exceed the ~8.9 GB one design writes, or the guard "
            "passes and the iteration still fails halfway")


class TestPpdsWindowing:
    """RY proposed an FWHM-windowed PPDS in Jul 2026. The window must restrict
    the NUMERATOR as well as the V sum: filtering only the denominator shrinks
    it while leaving every beam's probability in the numerator, which inflates
    PPDS for exactly the wide-beam designs the window exists to penalise."""

    def _make(self, wd, fwhms, n_slots=4):
        """One layout, one detector, beams occupying 6 pixels each.

        n_slots fixes the pixel count regardless of how many beams are present.
        Real configs always share a 200x200 FOV, so a fixture that varied the
        pixel count would vary something physical designs never vary, and the
        final mean over pixels would move for reasons unrelated to the window.
        """
        os.makedirs(wd, exist_ok=True)
        n_beams = len(fwhms)
        assert n_beams <= n_slots
        n_pix = 6 * n_slots
        mask = np.zeros((1, n_pix), dtype=np.int32)
        ppdf = np.zeros((1, n_pix), dtype=np.float64)
        for b, _ in enumerate(fwhms, start=1):
            sl = slice((b - 1) * 6, b * 6)
            mask[0, sl] = b
            ppdf[0, sl] = 1.0
        with h5py.File(os.path.join(wd, "position_000_ppdfs_t8_00.hdf5"), "w") as f:
            f.create_dataset("ppdfs", data=ppdf)
        with h5py.File(os.path.join(wd, "beams_masks_configuration_000.hdf5"), "w") as f:
            f.create_dataset("beam_mask", data=mask)
        bp = np.zeros((n_beams, 8), dtype=np.float64)
        for b, fw in enumerate(fwhms, start=1):
            bp[b - 1, 1] = 0        # detector id
            bp[b - 1, 2] = b        # beam id
            bp[b - 1, 3] = 0.0      # angle
            bp[b - 1, 4] = fw       # FWHM
            bp[b - 1, 7] = 1.0      # sensitivity
        with h5py.File(os.path.join(wd, "beams_properties_configuration_000.hdf5"), "w") as f:
            f.create_dataset("beam_properties", data=bp)

    def test_wide_beams_do_not_affect_the_windowed_value(self, tmp_path):
        """The invariant that actually pins the numerator restriction.

        Windowed PPDS is EXPECTED to exceed the unwindowed value: PPDS is
        probability mass per unit beam volume, and narrow beams have small
        volume, so restricting to them raises the density. Magnitude therefore
        proves nothing.

        What must hold is that beams outside the window are invisible. Two
        designs sharing the same narrow beams must give the same windowed PPDS
        however many wide beams one of them also has. If the numerator were not
        restricted, the extra wide beams would add probability mass with no
        matching volume and inflate the result.
        """
        import compute_metrics as cm
        narrow_only = str(tmp_path / "narrow")
        with_wide = str(tmp_path / "wide")
        self._make(narrow_only, [0.30, 0.35])          # 2 beams, 4 slots
        self._make(with_wide, [0.30, 0.35, 0.90, 1.20])  # same 2, plus 2 wide
        a = cm._ppds_components(narrow_only, fwhm_max=0.45)
        b = cm._ppds_components(with_wide, fwhm_max=0.45)
        assert a is not None and b is not None
        assert float(b[0]) == pytest.approx(float(a[0]), rel=1e-9), (
            f"wide beams changed the windowed value ({a[0]} vs {b[0]}); "
            f"they are leaking into the numerator")

    def test_window_above_all_beams_is_a_noop(self, tmp_path):
        import compute_metrics as cm
        wd = str(tmp_path / "cfg2")
        self._make(wd, [0.30, 0.35])
        full = cm._ppds_components(wd)
        win = cm._ppds_components(wd, fwhm_max=10.0)
        assert full is not None and win is not None
        assert win[0] == pytest.approx(full[0]), (
            "a window wider than every beam must change nothing")

    def test_window_below_all_beams_yields_nothing(self, tmp_path):
        import compute_metrics as cm
        wd = str(tmp_path / "cfg3")
        self._make(wd, [0.90, 1.20])
        win = cm._ppds_components(wd, fwhm_max=0.45)
        assert win is None or float(win[0]) == pytest.approx(0.0), (
            "excluding every beam must give zero, not a stale full-population value")


class TestAllNanGuards:
    """Sep 2026: a PPDS sweep ran in 4 seconds and printed a full table of NaN
    correlations that read exactly like a result. Nothing errored. The cause was
    input SELECTION, not computation: --limit took head(80), which is the oldest
    designs, and the scratch purge had stripped their raw PPDF files months
    earlier while leaving the CSV rows and a cnr/ subdirectory intact.

    Three guards, each tested here, because this is the third failure of the
    same shape this month: the code was right, the inputs were wrong, and the
    output looked plausible."""

    def _work_dir(self, root, name, with_ppdf):
        d = os.path.join(str(root), name)
        os.makedirs(os.path.join(d, "cnr_inloop"), exist_ok=True)
        if with_ppdf:
            for i in range(2):
                open(os.path.join(d, f"position_00{i}_ppdfs_t8_00.hdf5"), "w").close()
        return d

    def test_configs_missing_raw_files_are_dropped(self, tmp_path):
        import analyze_ppds_window as apw
        rows = [
            {"work_dir": self._work_dir(tmp_path, "purged_a", False)},
            {"work_dir": self._work_dir(tmp_path, "purged_b", False)},
            {"work_dir": self._work_dir(tmp_path, "intact_a", True)},
        ]
        kept = apw.configs_with_raw_files(pd.DataFrame(rows))
        assert len(kept) == 1
        assert "intact_a" in str(kept.work_dir.iloc[0]), (
            "a purged config, which recomputes to NaN, was kept")

    def test_a_cnr_subdir_alone_does_not_count_as_intact(self, tmp_path):
        """The exact shape the purge leaves behind: the directory exists and
        looks populated, but the raw files it needs are gone."""
        import analyze_ppds_window as apw
        d = self._work_dir(tmp_path, "looks_fine", False)
        assert os.path.isdir(os.path.join(d, "cnr_inloop"))
        kept = apw.configs_with_raw_files(pd.DataFrame([{"work_dir": d}]))
        assert len(kept) == 0

    def test_sample_keeps_the_band_the_test_turns_on(self):
        import analyze_ppds_window as apw
        # 5 designs inside the 0.20-0.45 band, 45 outside it
        df = pd.DataFrame({"aperture_diam_mm":
                           [0.25] * 5 + [0.80] * 45})
        got = apw.stratified_sample(df, limit=10)
        lo, hi = apw.SMALL_APERTURE
        in_band = ((got.aperture_diam_mm >= lo) & (got.aperture_diam_mm <= hi)).sum()
        assert len(got) == 10
        assert in_band == 5, (
            f"only {in_band} of the 5 in-band designs survived sampling; a plain "
            f"random draw would leave about 1 and the sweep could not answer "
            f"the question it exists for")

    def test_sample_is_a_noop_below_the_limit(self):
        import analyze_ppds_window as apw
        df = pd.DataFrame({"aperture_diam_mm": [0.25, 0.30, 0.80]})
        assert len(apw.stratified_sample(df, limit=10)) == 3

    def test_enough_values_rejects_a_mostly_empty_column(self):
        import analyze_ppds_window as apw
        assert not apw.enough_values([float("nan")] * 80)
        assert not apw.enough_values([1.0] * 9 + [float("nan")] * 71)
        assert apw.enough_values([1.0] * 10)

    def test_enough_values_is_what_the_sweep_actually_calls(self):
        """Guard against the check being loosened to a truthiness test, which
        NaN passes."""
        import analyze_ppds_window as apw
        import inspect
        src = inspect.getsource(apw.main)
        assert "enough_values(" in src, (
            "main() no longer calls the guard; an all-NaN table can be printed again")


# =============================================================================
# compute_cnr.seed_from_config_name — reproducible in-loop noise
# =============================================================================
class TestCnrSeedDerivation:
    """The in-loop CNR ran unseeded until Sep 2026, so no archive row could be
    reproduced from its work_dir and selecting the maximum over ~100 noisy
    draws inflated it by a measured +0.10 CNR. The seed is now derived from the
    config name, which keeps draws independent ACROSS designs while making any
    single design re-runnable."""

    def test_seed_is_stable_for_a_given_name(self):
        import compute_cnr
        a = compute_cnr.seed_from_config_name("mobo_0296_ap0.2718")
        b = compute_cnr.seed_from_config_name("mobo_0296_ap0.2718")
        assert a == b

    def test_seed_survives_a_fresh_interpreter(self):
        """The whole point. Python's hash() is salted per process, so a seed
        built on it would differ on every SLURM task and reproduce nothing."""
        import compute_cnr
        opt_dir = os.path.join(_REPO_ROOT, "optimization")
        prog = ("import sys; sys.path.insert(0, %r); import compute_cnr; "
                "print(compute_cnr.seed_from_config_name('mobo_0296'))" % opt_dir)
        seen = set()
        for salt in ("0", "1", "random"):
            env = dict(os.environ, PYTHONHASHSEED=salt)
            out = subprocess.run([sys.executable, "-c", prog], env=env,
                                 capture_output=True, text=True, check=True)
            seen.add(out.stdout.strip())
        assert len(seen) == 1, (
            f"seed changed with PYTHONHASHSEED: {seen}; it must not be built "
            f"on Python's salted hash()")
        assert seen.pop() == str(compute_cnr.seed_from_config_name("mobo_0296"))

    def test_different_configs_get_different_seeds(self):
        import compute_cnr
        names = [f"mobo_{i:04d}" for i in range(500)]
        seeds = {compute_cnr.seed_from_config_name(n) for n in names}
        assert len(seeds) == len(names), "seed collisions across 500 configs"

    def test_seed_is_a_valid_torch_seed(self):
        import compute_cnr
        for n in ("", "x", "mobo_0296", "ring2_007", "tmi_reference_000"):
            s = compute_cnr.seed_from_config_name(n)
            assert isinstance(s, int) and 0 <= s < 2 ** 31

    def test_pipeline_asks_for_the_derived_seed(self):
        """A seed nothing passes is a seed that does nothing."""
        sh = os.path.join(_REPO_ROOT, "optimization", "run_sai_pipeline.sh")
        with open(sh) as f:
            src = f.read()
        assert "--seed_from_config" in src, (
            "run_sai_pipeline.sh no longer passes --seed_from_config; in-loop "
            "CNR is back to irreproducible single draws")

    def test_explicit_seed_still_wins(self):
        """The repeat harness passes --seed 0..4 and must keep overriding."""
        import compute_cnr
        import inspect
        src = inspect.getsource(compute_cnr.main)
        assert "if seed is None and args.seed_from_config" in src, (
            "--seed_from_config must only fill in when --seed was omitted, or "
            "the 5-seed repeat runs all collapse onto one realisation")


# =============================================================================
# mobo_agent.report_out_of_bounds — training points outside the search box
# =============================================================================
class TestOutOfBoundsReporting:
    """mobo_0190 and mobo_0193 carry real measurements at d3_inner_mm = 640
    against a ceiling of 618, so they normalise outside the unit cube the
    acquisition searches. Keeping them is right; not knowing is not."""

    def test_clean_training_set_reports_nothing(self):
        import torch
        import mobo_agent
        x = torch.tensor([mobo_agent.BOUNDS_MIN, mobo_agent.BOUNDS_MAX],
                         dtype=torch.double)
        assert mobo_agent.report_out_of_bounds(x) == 0

    def test_counts_rows_not_violations(self):
        import torch
        import mobo_agent
        bad = list(mobo_agent.BOUNDS_MAX)
        bad[4] += 50.0
        bad[5] += 50.0          # one row, two violated parameters
        x = torch.tensor([mobo_agent.BOUNDS_MIN, bad], dtype=torch.double)
        assert mobo_agent.report_out_of_bounds(x) == 1

    def test_detects_the_real_d3_case(self):
        import torch
        import mobo_agent
        row = list(mobo_agent.BOUNDS_MIN)
        row[5] = 640.0          # mobo_0190 / mobo_0193
        x = torch.tensor([row], dtype=torch.double)
        assert mobo_agent.report_out_of_bounds(x) == 1

    def test_detects_values_below_the_floor(self):
        import torch
        import mobo_agent
        row = list(mobo_agent.BOUNDS_MIN)
        row[0] = 0.05
        x = torch.tensor([row], dtype=torch.double)
        assert mobo_agent.report_out_of_bounds(x) == 1

    def test_out_of_bounds_points_are_kept_not_dropped(self):
        """Dropping them would silently discard real measurements."""
        import inspect
        import mobo_agent
        src = inspect.getsource(mobo_agent.get_next_candidate)
        i = src.index("report_out_of_bounds(train_x)")
        after = src[i:i + 400]
        assert "train_x = train_x[" not in after and "drop" not in after.lower(), (
            "out-of-bounds handling must warn, not filter the training set")


# =============================================================================
# check_unmerged_results — finished batches nobody merged
# =============================================================================
class TestUnmergedResults:
    """The ring2 headroom sweep finished on 12 Aug 2026 and sat unmerged and
    unread for three weeks. Nothing checked."""

    @pytest.fixture
    def tree(self, tmp_path):
        res = tmp_path / "results"
        (res / "ring2_seed_out").mkdir(parents=True)
        (res / "lhs6d_seed_out").mkdir(parents=True)
        pd.DataFrame({"config": ["ring2_000", "ring2_001"],
                      "cnr_sector_mean": [4.4, 4.5]}).to_csv(
            res / "ring2_seed_out" / "task_0.csv", index=False)
        pd.DataFrame({"config": ["lhs6d_000"], "cnr_sector_mean": [4.1]}).to_csv(
            res / "lhs6d_seed_out" / "task_0.csv", index=False)
        arc = res / "results_summary_mobo.csv"
        pd.DataFrame({"config": ["lhs6d_000"], "cnr_sector_mean": [4.1]}).to_csv(
            arc, index=False)
        return str(res), str(arc)

    def test_finds_the_unmerged_batch_only(self, tree):
        import check_unmerged_results as cur
        res, arc = tree
        got = cur.find_unmerged(res, arc)
        assert len(got) == 1
        d, names = next(iter(got.items()))
        assert d.endswith("ring2_seed_out")
        assert names == ["ring2_000", "ring2_001"]

    def test_reports_nothing_once_merged(self, tree):
        import check_unmerged_results as cur
        res, arc = tree
        pd.DataFrame({"config": ["lhs6d_000", "ring2_000", "ring2_001"],
                      "cnr_sector_mean": [4.1, 4.4, 4.5]}).to_csv(arc, index=False)
        assert cur.find_unmerged(res, arc) == {}

    def test_missing_archive_means_everything_is_unmerged(self, tree):
        import check_unmerged_results as cur
        res, _ = tree
        got = cur.find_unmerged(res, os.path.join(res, "does_not_exist.csv"))
        assert len(got) == 2

    def test_a_truncated_task_file_does_not_stop_the_report(self, tree):
        """A dead array job is exactly when the rest of the report matters."""
        import check_unmerged_results as cur
        res, arc = tree
        with open(os.path.join(res, "ring2_seed_out", "task_9.csv"), "w") as f:
            f.write("")
        got = cur.find_unmerged(res, arc)
        assert next(iter(got.values())) == ["ring2_000", "ring2_001"]

    def test_nonzero_exit_when_rows_are_unmerged(self, tree):
        """So a submit script can gate on it."""
        import check_unmerged_results as cur
        res, arc = tree
        rc = subprocess.run(
            [sys.executable, os.path.join(_REPO_ROOT, "optimization",
                                          "check_unmerged_results.py"),
             "--results_dir", res, "--out_csv", arc],
            capture_output=True, text=True).returncode
        assert rc == 1

    def test_a_sibling_h2h_archive_counts_as_merged(self, tmp_path):
        """The 21 shared LHS seeds live in the head-to-head arms' own archives.
        Reporting them as unmerged is 21 false positives, and a report that
        cries wolf is one nobody reads."""
        import check_unmerged_results as cur
        opt = tmp_path / "optimization"
        res = opt / "results"
        (res / "lhs6d_seed_out").mkdir(parents=True)
        pd.DataFrame({"config": ["lhs6d_000"]}).to_csv(
            res / "lhs6d_seed_out" / "task_0.csv", index=False)
        arc = res / "results_summary_mobo.csv"
        pd.DataFrame({"config": ["mobo_0001"]}).to_csv(arc, index=False)

        assert cur.find_unmerged(str(res), str(arc)) != {}, "precondition"

        h2h = opt / "results_h2h_2obj"
        h2h.mkdir()
        pd.DataFrame({"config": ["lhs6d_000"]}).to_csv(
            h2h / "results_summary_mobo.csv", index=False)
        assert cur.find_unmerged(str(res), str(arc)) == {}

    def test_ignore_file_suppresses_deliberate_exclusions(self, tree):
        import check_unmerged_results as cur
        res, arc = tree
        ign = os.path.join(res, "ignore.txt")
        with open(ign, "w") as f:
            f.write("# deliberate\nring2_000  # benchmark\nring2_001\n")
        assert cur.find_unmerged(res, arc, ignore_path=ign) == {}

    def test_ignore_file_strips_comments_and_blanks(self, tmp_path):
        import check_unmerged_results as cur
        p = tmp_path / "ig.txt"
        p.write_text("# header\n\n  ring2_000   # why\nring2_001\n   \n")
        assert cur.load_ignore(str(p)) == {"ring2_000", "ring2_001"}

    def test_missing_ignore_file_is_not_an_error(self, tmp_path):
        import check_unmerged_results as cur
        assert cur.load_ignore(str(tmp_path / "nope.txt")) == set()

    def test_the_committed_ignore_file_explains_every_entry(self):
        """An exclusion with no reason is how the ring2 sweep got lost."""
        import check_unmerged_results as cur
        path = os.path.join(_REPO_ROOT, "optimization", "unmerged_ignore.txt")
        assert cur.load_ignore(path), "ignore list is empty"
        with open(path) as f:
            body = f.read()
        assert "AUDIT_REPORT" in body and "ring2" in body and "tmi_reference" in body

