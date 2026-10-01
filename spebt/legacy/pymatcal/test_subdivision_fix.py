#!/usr/bin/env python3
"""
Test script for the self-attenuation fix in ppdf_2d_local.

Tests across edge cases: (1,1), (5,1), (1,5), (3,3), (1,3), (3,1)
Simplified geometry: one crystal, one pixel, no plates/other crystals.

Key invariant: total PPDF from all subdivisions should approximately equal
the (1,1) case (since the crystal physics doesn't change).

Two bugs fixed:
  1. Missing self-attenuation through sibling subdivisions
  2. Target sub path length wrong when ray endpoint is inside the polygon
"""
from __future__ import annotations
import sys
import os
import torch
import numpy as np
from torch import (
    Tensor, arange, bmm, cat, stack, tensor, where, pi,
    linspace, meshgrid, zeros,
)
from torch import abs as torch_abs


# ---- Minimal self-contained helper functions ----

def polygon_edges_from_vertices_2d_batch(polygon_batch: Tensor) -> Tensor:
    return stack((polygon_batch, polygon_batch.roll(-1, dims=1)), dim=2)


def points_to_refs_angle_2d_batch(points: Tensor, refs: Tensor) -> Tensor:
    n_refs = refs.shape[0]
    n_points = points.shape[0]
    diff = points.unsqueeze(0).expand(n_refs, -1, -1) - refs.unsqueeze(1).expand(-1, n_points, -1)
    return torch.atan2(diff[:, :, 1], diff[:, :, 0])


def polygon_to_points_angular_span_2d_batch(
    polygon_vertices_batch: Tensor, ref_points_batch: Tensor
) -> Tensor:
    polygon_vertices_rads = points_to_refs_angle_2d_batch(
        polygon_vertices_batch.view(-1, 2), ref_points_batch
    ).view(
        ref_points_batch.shape[0],
        polygon_vertices_batch.shape[0],
        polygon_vertices_batch.shape[1],
    )
    span = (
        polygon_vertices_rads.max(dim=2).values
        - polygon_vertices_rads.min(dim=2).values
    )
    return where(span > pi, 2 * pi - span, span)


def subdivision_grid_rectangle(n_sub) -> Tensor:
    grid = stack(
        meshgrid(
            linspace(0, 1, int(n_sub[0]) + 1),
            linspace(0, 1, int(n_sub[1]) + 1),
            indexing="ij",
        ),
        dim=-1,
    )
    return stack(
        (grid[:-1, :-1], grid[1:, :-1], grid[1:, 1:], grid[:-1, 1:]), dim=-2
    ).view(-1, 4, 2)


def subdivision_vertices_rectangle(vertices: Tensor, grid: Tensor) -> Tensor:
    origin = vertices[0]
    v_matrix = stack((vertices[1] - origin, vertices[3] - origin))
    return bmm(grid, v_matrix.unsqueeze(0).expand(grid.shape[0], -1, -1)) + origin


def rays_2d_batch(pa_batch: Tensor, pb_batch: Tensor) -> Tensor:
    npa = pa_batch.shape[0]
    npb = pb_batch.shape[0]
    return stack(
        (
            pa_batch.unsqueeze(1).expand(-1, npb, -1),
            pb_batch.unsqueeze(0).expand(npa, -1, -1),
        ),
        dim=2,
    )


def line_segments_t(ls_a: Tensor, ls_b: Tensor, eps=1e-9) -> Tensor:
    n_ls_a = ls_a.shape[0]
    n_ls_b = ls_b.shape[0]
    va = (ls_a[:, 1] - ls_a[:, 0]).unsqueeze(1).expand(-1, n_ls_b, -1)
    vb = (ls_b[:, 0] - ls_b[:, 1]).unsqueeze(0).expand(n_ls_a, -1, -1)
    v3 = ls_b[:, 0].unsqueeze(0).expand(n_ls_a, -1, -1) - ls_a[:, 0].view(
        n_ls_a, 1, 2
    ).expand(-1, n_ls_b, -1)
    det = va[:, :, 0] * vb[:, :, 1] - va[:, :, 1] * vb[:, :, 0]
    t = where(torch_abs(det) > eps,
        (v3[:, :, 0] * vb[:, :, 1] - v3[:, :, 1] * vb[:, :, 0]) / det, -1.0)
    s = where(torch_abs(det) > eps,
        (va[:, :, 0] * v3[:, :, 1] - va[:, :, 1] * v3[:, :, 0]) / det, -1.0)
    valid = (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
    return where(valid, t, -1.0)


def rays_edges_t_subdivisions(ls_a: Tensor, ls_b: Tensor, eps=1e-9) -> Tensor:
    """Each ray[i,j] tested against its matching subdivision j's edges only."""
    n_edges = ls_b.shape[1]
    n_pa = ls_a.shape[0]
    n_pb = ls_b.shape[0]
    va = (ls_a[:, :, 1] - ls_a[:, :, 0]).unsqueeze(2).expand(-1, -1, n_edges, -1)
    vb = (ls_b[:, :, 0] - ls_b[:, :, 1]).unsqueeze(0).expand(n_pa, -1, -1, -1)
    v3 = ls_b[:, :, 0].unsqueeze(0).expand(n_pa, -1, -1, -1) - ls_a[:, :, 0].view(
        n_pa, n_pb, 1, 2).expand(-1, -1, n_edges, -1)
    det = va[:, :, :, 0] * vb[:, :, :, 1] - va[:, :, :, 1] * vb[:, :, :, 0]
    t = where(torch_abs(det) > eps,
        (v3[:, :, :, 0] * vb[:, :, :, 1] - v3[:, :, :, 1] * vb[:, :, :, 0]) / det, -1.0)
    s = where(torch_abs(det) > eps,
        (va[:, :, :, 0] * v3[:, :, :, 1] - va[:, :, :, 1] * v3[:, :, :, 0]) / det, -1.0)
    valid = (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
    return where(valid, t, -1.0)


def rays_intersection_lengths(rays: Tensor, rays_t: Tensor) -> Tensor:
    rays = rays.view(-1, 2, 2)
    rays_t_reshaped = rays_t.reshape(rays.shape[0], -1, 4)
    rays_t_sorted = rays_t_reshaped.sort(dim=2).values
    rays_t_diff = rays_t_sorted[:, :, -1] - rays_t_sorted[:, :, -2]
    length = rays_t_diff * (rays[:, 1] - rays[:, 0]).norm(dim=1).view(-1, 1)
    return length


# ---- FIXED ppdf function (both bugs fixed) ----

def ppdf_2d_local_fixed(
    sfov_pixels_batch, crystal_objects_vertices, reduced_plate_edges,
    reduced_crystal_edges, subdivision_grid, mu_dict,
):
    sub_crystals_vertices = subdivision_vertices_rectangle(
        crystal_objects_vertices[0], subdivision_grid
    )
    sub_crystals_edges = polygon_edges_from_vertices_2d_batch(sub_crystals_vertices)

    pa_batch = sfov_pixels_batch[0]
    pb_batch = sub_crystals_vertices.mean(dim=1)
    n_pix = pa_batch.shape[0]
    n_subs = pb_batch.shape[0]

    # Original rays (pixel → centroid) for angular span and external attenuation
    rays = rays_2d_batch(pa_batch, pb_batch)

    # --- FIX 1: Extended rays for correct target sub path length ---
    # Extend each ray beyond the centroid so it fully traverses the target sub
    # endpoint = 2 * centroid - pixel (same direction, double distance)
    pb_ext = 2 * pb_batch.unsqueeze(0).expand(n_pix, -1, -1) \
           - pa_batch.unsqueeze(1).expand(-1, n_subs, -1)
    rays_ext = stack((
        pa_batch.unsqueeze(1).expand(-1, n_subs, -1),
        pb_ext
    ), dim=2)  # (n_pix, n_subs, 2, 2)

    # Target sub path: extended ray vs its own sub edges (diagonal only)
    rays_ext_sub_t = rays_edges_t_subdivisions(rays_ext, sub_crystals_edges)
    intersection_length_target = rays_intersection_lengths(
        rays_ext, rays_ext_sub_t
    ).view(n_pix, n_subs)

    # --- FIX 2: Self-attenuation through sibling subs ---
    # Original rays vs ALL sub edges (off-diagonal = correct sibling paths)
    all_sub_edges_flat = sub_crystals_edges.view(-1, 2, 2)
    rays_all_subs_t = line_segments_t(rays.view(-1, 2, 2), all_sub_edges_flat)
    intersection_length_all_subs = rays_intersection_lengths(
        rays, rays_all_subs_t
    ).view(n_pix, n_subs, n_subs)

    # Off-diagonal sum = correct sibling paths (subs after target contribute 0)
    diag_idx = arange(n_subs)
    wrong_diagonal = intersection_length_all_subs[:, diag_idx, diag_idx]
    self_atten_lengths = intersection_length_all_subs.sum(dim=2) - wrong_diagonal

    # External attenuation (plates + other crystals)
    if reduced_plate_edges.numel() > 0:
        rpt = line_segments_t(rays.view(-1, 2, 2), reduced_plate_edges.view(-1, 2, 2))
        il_plates = rays_intersection_lengths(rays, rpt).view(n_pix, n_subs, -1)
        sum_plate_exp = (il_plates * mu_dict[0]).sum(dim=2)
    else:
        sum_plate_exp = zeros(n_pix, n_subs)

    if reduced_crystal_edges.numel() > 0:
        rct = line_segments_t(rays.view(-1, 2, 2), reduced_crystal_edges.view(-1, 2, 2))
        il_crystals = rays_intersection_lengths(rays, rct).view(n_pix, n_subs, -1)
        sum_crystal_exp = (il_crystals * mu_dict[1]).sum(dim=2)
    else:
        sum_crystal_exp = zeros(n_pix, n_subs)

    subdivision_rads_span = polygon_to_points_angular_span_2d_batch(
        sub_crystals_vertices, pa_batch
    )

    subdivision_exponent = intersection_length_target * mu_dict[1]
    self_atten_exponent = self_atten_lengths * mu_dict[1]
    angular_term = subdivision_rads_span / (2 * pi)

    return (
        (-sum_plate_exp - sum_crystal_exp - self_atten_exponent).exp()
        * (1 - (-subdivision_exponent).exp())
        * angular_term
    ).sum(dim=1)


# ---- ORIGINAL buggy ppdf function (for comparison) ----

def ppdf_2d_local_original(
    sfov_pixels_batch, crystal_objects_vertices, reduced_plate_edges,
    reduced_crystal_edges, subdivision_grid, mu_dict,
):
    sub_crystals_vertices = subdivision_vertices_rectangle(
        crystal_objects_vertices[0], subdivision_grid
    )
    sub_crystals_edges = polygon_edges_from_vertices_2d_batch(sub_crystals_vertices)

    pa_batch = sfov_pixels_batch[0]
    pb_batch = sub_crystals_vertices.mean(dim=1)
    n_pix = pa_batch.shape[0]
    n_subs = pb_batch.shape[0]

    rays = rays_2d_batch(pa_batch, pb_batch)

    # Original: each ray vs its own sub only, no self-attenuation
    rays_sub_t = rays_edges_t_subdivisions(rays, sub_crystals_edges)
    il_subs = rays_intersection_lengths(rays, rays_sub_t).view(n_pix, n_subs)

    if reduced_plate_edges.numel() > 0:
        rpt = line_segments_t(rays.view(-1, 2, 2), reduced_plate_edges.view(-1, 2, 2))
        il_plates = rays_intersection_lengths(rays, rpt).view(n_pix, n_subs, -1)
        sum_plate_exp = (il_plates * mu_dict[0]).sum(dim=2)
    else:
        sum_plate_exp = zeros(n_pix, n_subs)

    if reduced_crystal_edges.numel() > 0:
        rct = line_segments_t(rays.view(-1, 2, 2), reduced_crystal_edges.view(-1, 2, 2))
        il_crystals = rays_intersection_lengths(rays, rct).view(n_pix, n_subs, -1)
        sum_crystal_exp = (il_crystals * mu_dict[1]).sum(dim=2)
    else:
        sum_crystal_exp = zeros(n_pix, n_subs)

    subdivision_rads_span = polygon_to_points_angular_span_2d_batch(
        sub_crystals_vertices, pa_batch
    )

    subdivision_exponent = il_subs * mu_dict[1]
    angular_term = subdivision_rads_span / (2 * pi)

    return (
        (-sum_plate_exp - sum_crystal_exp).exp()
        * (1 - (-subdivision_exponent).exp())
        * angular_term
    ).sum(dim=1)


# ---- Test Geometry ----

def make_test_geometry(pixel_pos, crystal_center, crystal_w, crystal_h, crystal_angle_deg=0.0):
    angle = np.radians(crystal_angle_deg)
    t_hat = tensor([np.sin(angle), np.cos(angle)], dtype=torch.float32)
    r_hat = tensor([np.cos(angle), -np.sin(angle)], dtype=torch.float32)

    cc = tensor(crystal_center, dtype=torch.float32)
    W, H = crystal_w, crystal_h

    v0 = cc + (W / 2) * t_hat + (H / 2) * r_hat
    v1 = cc - (W / 2) * t_hat + (H / 2) * r_hat
    v2 = cc - (W / 2) * t_hat - (H / 2) * r_hat
    v3 = cc + (W / 2) * t_hat - (H / 2) * r_hat

    crystal_vertices = stack([v0, v1, v2, v3]).unsqueeze(0)
    pixel = tensor(pixel_pos, dtype=torch.float32)
    sfov_pixels = pixel.view(1, 1, 2)
    empty_edges = zeros((0, 2, 2), dtype=torch.float32)

    return sfov_pixels, crystal_vertices, empty_edges, empty_edges


def compute_ppdf(fn, sub_config, pixel_pos, crystal_center, crystal_w, crystal_h,
                 mu_dict, crystal_angle_deg=0.0):
    sfov_pixels, crystal_verts, plate_edges, other_crystal_edges = \
        make_test_geometry(pixel_pos, crystal_center, crystal_w, crystal_h, crystal_angle_deg)
    grid = subdivision_grid_rectangle(sub_config)
    ppdf = fn(sfov_pixels, crystal_verts, plate_edges, other_crystal_edges, grid, mu_dict)
    return ppdf.item()


# ---- Main test runner ----

def run_tests():
    crystal_w = 0.84
    crystal_h = 6.0
    crystal_center = (35.0, 0.0)
    crystal_angle = 0.0
    mu_dict = tensor([3.5, 0.5], dtype=torch.float32)

    pixel_positions = [
        (0.0, 0.0),
        (0.0, 0.2),
        (1.0, 0.0),
        (0.5, 0.5),
    ]

    subdivision_configs = [
        (1, 1),
        (5, 1),
        (1, 5),
        (3, 3),
        (1, 3),
        (3, 1),
        (4, 4),
        (5, 5),
    ]

    print("=" * 90)
    print("SELF-ATTENUATION + TARGET PATH FIX — EDGE CASE TESTS")
    print("=" * 90)
    print(f"Crystal: {crystal_w}mm (tang) x {crystal_h}mm (radial) at {crystal_center}")
    print(f"mu_dict: plates={mu_dict[0].item()}, crystals={mu_dict[1].item()}")
    print()

    all_passed = True
    radial_tolerance = 0.10   # 10% for configs with radial subs (the fix target)
    tangential_tolerance = 0.40  # 40% for tangential-only (known angular approx. limitation)

    # Configs with radial subdivisions > 1 (what the fix addresses)
    radial_configs = {(1, 5), (3, 3), (1, 3)}
    # Tangential-only configs (pre-existing angular span approximation)
    tangential_only_configs = {(5, 1), (3, 1)}

    results_fixed = {}
    results_original = {}

    for px_pos in pixel_positions:
        results_fixed[px_pos] = {}
        results_original[px_pos] = {}
        print(f"--- Pixel at {px_pos} ---")
        print(f"  {'Config':>7s}  {'Original':>12s}  {'Fixed':>12s}  {'GT (1,1)':>12s}  {'Orig Err%':>10s}  {'Fix Err%':>10s}  Status")

        for config in subdivision_configs:
            val_fixed = compute_ppdf(
                ppdf_2d_local_fixed, config, px_pos, crystal_center,
                crystal_w, crystal_h, mu_dict, crystal_angle
            )
            val_original = compute_ppdf(
                ppdf_2d_local_original, config, px_pos, crystal_center,
                crystal_w, crystal_h, mu_dict, crystal_angle
            )
            results_fixed[px_pos][config] = val_fixed
            results_original[px_pos][config] = val_original

        gt = results_fixed[px_pos][(1, 1)]

        for config in subdivision_configs:
            vf = results_fixed[px_pos][config]
            vo = results_original[px_pos][config]

            rel_orig = (vo - gt) / gt * 100 if gt > 0 else 0
            rel_fix = (vf - gt) / gt * 100 if gt > 0 else 0

            if config in tangential_only_configs:
                tol = tangential_tolerance
                note = " (angular approx.)" if abs(rel_fix) > radial_tolerance * 100 else ""
            else:
                tol = radial_tolerance
                note = ""

            passed = abs(rel_fix) < tol * 100
            if not passed:
                all_passed = False
            status = "PASS" if passed else "FAIL"

            print(f"  {str(config):>7s}  {vo:12.8f}  {vf:12.8f}  {gt:12.8f}  {rel_orig:+9.2f}%  {rel_fix:+9.2f}%  [{status}]{note}")

        print()

    print("=" * 90)
    print("NOTE: Tangential-only configs (5,1), (3,1) have a pre-existing angular span")
    print("      approximation error for off-center pixels. This is unrelated to the fix.")
    print("      Radial configs are the target of the fix and must be within 10%.")
    print()
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 90)

    return all_passed, results_fixed, results_original


def create_visualization(results_fixed, results_original):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # --- Panel 1: Before fix diagram ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Before: Two Bugs", fontsize=11, fontweight='bold')
    for y_start, label, color in [
        (0.0, "R0 (front)", "#ff6b6b"), (2.0, "R1 (middle)", "#ff6b6b"),
        (4.0, "R2 (target)", "#ff6b6b"),
    ]:
        rect = patches.FancyBboxPatch((0.2, y_start + 0.05), 1.6, 1.9,
            boxstyle="round,pad=0.02", facecolor=color, edgecolor='black', alpha=0.6)
        ax1.add_patch(rect)
        ax1.text(1.0, y_start + 1.0, label, ha='center', va='center', fontsize=8, fontweight='bold')
    ax1.annotate('', xy=(1.0, 5.0), xytext=(-1.0, 3.0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax1.text(-1.1, 3.0, 'Ray', color='blue', fontsize=9, fontweight='bold')
    ax1.text(-1.5, -0.3, 'Bug 1: R0,R1 not attenuated', color='red', fontsize=7.5)
    ax1.text(-1.5, -0.8, 'Bug 2: R2 path length wrong\n'
             '(ray ends inside → bad absorption)', color='red', fontsize=7.5)
    ax1.set_xlim(-1.8, 2.5); ax1.set_ylim(-1.5, 6.5); ax1.set_aspect('equal'); ax1.axis('off')

    # --- Panel 2: After fix diagram ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("After: Both Fixed", fontsize=11, fontweight='bold')
    for y_start, label, color in [
        (0.0, "R0 (atten)", "#ffa94d"), (2.0, "R1 (atten)", "#ffa94d"),
        (4.0, "R2 (target)", "#51cf66"),
    ]:
        rect = patches.FancyBboxPatch((0.2, y_start + 0.05), 1.6, 1.9,
            boxstyle="round,pad=0.02", facecolor=color, edgecolor='black', alpha=0.6)
        ax1.add_patch(rect) if False else ax2.add_patch(rect)
        ax2.text(1.0, y_start + 1.0, label, ha='center', va='center', fontsize=8, fontweight='bold')
    # Original ray
    ax2.annotate('', xy=(1.0, 5.0), xytext=(-1.0, 3.0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    # Extended ray (dashed)
    ax2.annotate('', xy=(1.3, 6.3), xytext=(1.0, 5.0),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5, linestyle='dashed'))
    ax2.text(-1.1, 3.0, 'Ray', color='blue', fontsize=9, fontweight='bold')
    ax2.text(1.4, 6.0, 'Extended\n(for target\npath)', color='green', fontsize=7)
    ax2.text(-1.5, -0.3, 'Fix 1: Sibling attenuation added', color='#e67700', fontsize=7.5)
    ax2.text(-1.5, -0.8, 'Fix 2: Extended ray gives correct\n'
             'entry+exit for target sub', color='green', fontsize=7.5)
    ax2.set_xlim(-1.8, 2.5); ax2.set_ylim(-1.5, 6.5); ax2.set_aspect('equal'); ax2.axis('off')

    # --- Panel 3: Formula ---
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title("Formula Change", fontsize=11, fontweight='bold')
    ax3.axis('off')
    ax3.text(0.05, 0.88, "BEFORE (two bugs):", fontsize=10, fontweight='bold', color='red',
             transform=ax3.transAxes)
    ax3.text(0.05, 0.73,
        r"$e^{-\mu_p L_p - \mu_c L_{other}}$"
        r"$\times (1 - e^{-\mu_c \tilde{L}_{target}})$"  "\n"
        r"$\times\; \Delta\theta / 2\pi$" "\n\n"
        r"$\tilde{L}_{target}$" " = wrong (ray ends inside sub)",
        fontsize=9, transform=ax3.transAxes, verticalalignment='top')

    ax3.text(0.05, 0.42, "AFTER (both fixed):", fontsize=10, fontweight='bold', color='green',
             transform=ax3.transAxes)
    ax3.text(0.05, 0.27,
        r"$e^{-\mu_p L_p - \mu_c L_{other} \mathbf{- \mu_c L_{sib}}}$"
        r"$\times (1 - e^{-\mu_c L_{target}})$" "\n"
        r"$\times\; \Delta\theta / 2\pi$" "\n\n"
        r"$L_{sib}$" " = sibling sub paths (original rays)\n"
        r"$L_{target}$" " = correct path (extended rays)",
        fontsize=9, transform=ax3.transAxes, verticalalignment='top')

    # --- Panel 4: Original vs Fixed bar chart (center pixel) ---
    ax4 = fig.add_subplot(gs[1, 0:2])
    ax4.set_title("Original (Buggy) vs Fixed PPDF — Pixel at (0,0)", fontsize=11, fontweight='bold')

    center_fixed = results_fixed.get((0.0, 0.0), {})
    center_orig = results_original.get((0.0, 0.0), {})
    if center_fixed:
        configs = list(center_fixed.keys())
        gt = center_fixed[(1, 1)]
        labels = [f"({c[0]},{c[1]})" for c in configs]
        x = np.arange(len(configs))
        w = 0.35

        vals_orig = [center_orig[c] for c in configs]
        vals_fixed = [center_fixed[c] for c in configs]

        bars1 = ax4.bar(x - w/2, vals_orig, w, label='Original (buggy)', color='#ff6b6b',
                        edgecolor='black', alpha=0.8)
        bars2 = ax4.bar(x + w/2, vals_fixed, w, label='Fixed', color='#51cf66',
                        edgecolor='black', alpha=0.8)
        ax4.axhline(gt, color='#339af0', linestyle='--', lw=1.5,
                    label=f'(1,1) ground truth = {gt:.8f}')

        for bar, val in zip(bars1, vals_orig):
            rel = (val - gt) / gt * 100 if gt > 0 else 0
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f'{rel:+.1f}%', ha='center', va='bottom', fontsize=7, color='red')
        for bar, val in zip(bars2, vals_fixed):
            rel = (val - gt) / gt * 100 if gt > 0 else 0
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f'{rel:+.1f}%', ha='center', va='bottom', fontsize=7, color='green')

        ax4.set_xticks(x)
        ax4.set_xticklabels(labels)
        ax4.set_ylabel("PPDF Value")
        ax4.set_xlabel("Subdivision Config (tangential, radial)")
        ax4.legend(fontsize=9)
        ax4.grid(axis='y', alpha=0.3)

    # --- Panel 5: Relative error across pixels ---
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_title("Fixed: Relative Error vs (1,1)", fontsize=11, fontweight='bold')

    test_configs = [(5, 1), (1, 5), (3, 3), (1, 3), (3, 1)]
    config_labels = [f"({c[0]},{c[1]})" for c in test_configs]
    x_pos = np.arange(len(test_configs))
    width = 0.18

    for i, (px_pos, px_results) in enumerate(results_fixed.items()):
        gt = px_results.get((1, 1), 1.0)
        rel_diffs = [(px_results.get(c, gt) - gt) / gt * 100 if gt > 0 else 0 for c in test_configs]
        ax5.bar(x_pos + i * width, rel_diffs, width, label=f"px {px_pos}", alpha=0.8)

    ax5.axhline(0, color='black', lw=0.8)
    ax5.axhline(10, color='red', linestyle=':', lw=1, alpha=0.5, label='10% tolerance')
    ax5.axhline(-10, color='red', linestyle=':', lw=1, alpha=0.5)
    ax5.set_xticks(x_pos + width * 1.5)
    ax5.set_xticklabels(config_labels, fontsize=9)
    ax5.set_ylabel("Relative Diff (%)")
    ax5.set_xlabel("Subdivision Config")
    ax5.legend(fontsize=7, loc='best')
    ax5.grid(axis='y', alpha=0.3)

    fig.suptitle("ppdf_2d_local: Self-Attenuation + Target Path Fix Validation",
                 fontsize=14, fontweight='bold', y=0.98)

    out_path = os.path.join(os.path.dirname(__file__), "subdivision_fix_validation.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"\nVisualization saved to: {out_path}")
    plt.close()


if __name__ == "__main__":
    all_passed, results_fixed, results_original = run_tests()
    create_visualization(results_fixed, results_original)
    sys.exit(0 if all_passed else 1)