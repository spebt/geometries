"""
generate_brain_spect_polygon.py

Generates the .tensor geometry file for the Brain SPECT polygon-panel system.

System description:
  - Collimator: cylindrical ring at inner radius 120 mm, 2 mm thick (tungsten).
    Sampling is achieved by rotating the collimator ring while detectors and
    object remain stationary.
  - Detector panels: N_PANELS flat panels arranged as a regular polygon.
    Each panel has 3 crystal layers (inner, middle, outer) following the
    MATRICES self-collimation architecture.
    Crystal occupancy on inner/middle layers is sparse (opening ratio r=1/3):
    1 mm crystal, 2 mm gap, repeating. Outer layer is dense.
  - FOV: 200 mm diameter circle, 200x200 voxels at 1 mm/voxel.

Usage:
  python generate_brain_spect_polygon.py [options]

  python generate_brain_spect_polygon.py \
      --n_panels 6 \
      --collimator_inner_r 120.0 \
      --collimator_thickness 2.0 \
      --n_apertures 200 \
      --aperture_size_mm 2.0 \
      --panel_length_mm 120.0 \
      --crystal_size_mm 3.0 \
      --output_dir .
"""

import os
import math
import argparse
import torch
from typing import List, Tuple
from torch import Tensor, save as torch_save

try:
    from helper import (
        generate_md5_from_tensors,
        plot_polygons_from_vertices_2d_mpl,
        positions_parameters,
        transform_to_positions_2d_batch,
        OutDataDict,
    )
except ImportError as e:
    print(f"Error importing from helper.py: {e}")
    raise


# ─────────────────────────────────────────────────────────────────────────────
# Geometry builders
# ─────────────────────────────────────────────────────────────────────────────

def build_polygon_detector_panels(
    n_panels: int,
    panel_distance_mm: float,        # radial distance from origin to panel face (inner surface)
    panel_length_mm: float,          # tangential extent of each panel
    crystal_size_mm: float,          # crystal pitch (tangential and radial)
    n_layers: int = 3,               # number of crystal layers per panel
    layer_gap_mm: float = 0.0,       # radial gap between layers (air gap)
    opening_ratio: float = 1.0/3.0,  # fraction of tangential space occupied by crystal (inner/mid layers)
    dtype=torch.float32,
    device: str = "cpu",
) -> Tuple[Tensor, dict]:
    """
    Build (N,4,2) crystal rectangles for polygon-panel detector.

    Crystal layout:
      - Outer layer (layer index = n_layers-1): dense, one crystal every crystal_size_mm
      - Inner/middle layers: sparse mosaic, opening_ratio crystal then gap
        so for opening_ratio=1/3 and crystal_size_mm=3mm:
          crystal occupies 1mm, gap = 2mm, pitch = 3mm (same as outer)
          → crystals at positions 0, 3, 6, 9 ... mm (centre positions)
          → each crystal is 1mm wide, so inner layers have 1/3 the tangential count

    Each panel is placed at angle theta = k * (2*pi/n_panels), k=0..n_panels-1,
    oriented so its face is perpendicular to the radial direction.
    """
    crystal_w = crystal_size_mm      # tangential size of crystal cell pitch
    crystal_h = crystal_size_mm      # radial thickness of each layer

    # For inner/middle layers: crystal is opening_ratio fraction of the cell
    inner_crystal_tang = crystal_w * opening_ratio   # actual crystal width (tangential)
    inner_cell_pitch = crystal_w                     # pitch stays same as outer layer

    polys: List[Tensor] = []
    crystal_meta: List[dict] = []

    n_outer_per_panel = int(panel_length_mm / crystal_w)

    for panel_idx in range(n_panels):
        theta = panel_idx * (2.0 * math.pi / n_panels)

        # Unit vectors for this panel
        # r_hat: outward radial (normal to panel face, points away from origin)
        # t_hat: tangential (along panel length)
        r_hat = torch.tensor([math.cos(theta), math.sin(theta)], dtype=dtype)
        t_hat = torch.tensor([-math.sin(theta), math.cos(theta)], dtype=dtype)

        # Panel inner face centre (at panel_distance_mm from origin)
        panel_face_centre = r_hat * panel_distance_mm

        for layer_idx in range(n_layers):
            # Radial offset of this layer's centre from the panel face
            r_offset = (layer_idx + 0.5) * crystal_h + layer_idx * layer_gap_mm
            layer_centre_radial = panel_face_centre + r_hat * r_offset

            is_outer = (layer_idx == n_layers - 1)
            if is_outer:
                # Dense outer layer
                n_crystals = n_outer_per_panel
                tang_size = crystal_w
                for ci in range(n_crystals):
                    # crystal tangential centre relative to panel left edge
                    tang_offset = (ci + 0.5) * crystal_w - panel_length_mm / 2.0
                    c = layer_centre_radial + t_hat * tang_offset
                    v1 = c + (tang_size/2.0)*t_hat + (crystal_h/2.0)*r_hat
                    v2 = c - (tang_size/2.0)*t_hat + (crystal_h/2.0)*r_hat
                    v3 = c - (tang_size/2.0)*t_hat - (crystal_h/2.0)*r_hat
                    v4 = c + (tang_size/2.0)*t_hat - (crystal_h/2.0)*r_hat
                    polys.append(torch.stack([v1, v2, v3, v4]))
                    crystal_meta.append({
                        "panel": panel_idx, "layer": layer_idx,
                        "crystal_in_layer": ci, "type": "outer_dense"
                    })
            else:
                # Sparse inner/middle layer: crystal at every pitch, but crystal
                # width = inner_crystal_tang, leaving gap = pitch - inner_crystal_tang
                n_crystals = n_outer_per_panel  # same number of positions
                tang_size = inner_crystal_tang
                for ci in range(n_crystals):
                    tang_offset = (ci + 0.5) * inner_cell_pitch - panel_length_mm / 2.0
                    c = layer_centre_radial + t_hat * tang_offset
                    v1 = c + (tang_size/2.0)*t_hat + (crystal_h/2.0)*r_hat
                    v2 = c - (tang_size/2.0)*t_hat + (crystal_h/2.0)*r_hat
                    v3 = c - (tang_size/2.0)*t_hat - (crystal_h/2.0)*r_hat
                    v4 = c + (tang_size/2.0)*t_hat - (crystal_h/2.0)*r_hat
                    polys.append(torch.stack([v1, v2, v3, v4]))
                    crystal_meta.append({
                        "panel": panel_idx, "layer": layer_idx,
                        "crystal_in_layer": ci, "type": "inner_sparse"
                    })

    detector_units = torch.stack(polys, dim=0).to(dtype=dtype)
    stats = {
        "n_panels": n_panels,
        "n_layers": n_layers,
        "n_outer_crystals_per_layer_per_panel": n_outer_per_panel,
        "n_inner_crystals_per_layer_per_panel": n_outer_per_panel,
        "total_crystals": detector_units.shape[0],
        "opening_ratio": opening_ratio,
    }
    return detector_units, stats


def build_cylindrical_collimator_ring(
    r_in: float,             # inner radius of collimator ring (mm)
    thickness: float,        # radial thickness (mm)
    n_apertures: int,        # number of apertures distributed around ring
    aperture_size_mm: float, # angular extent of each aperture (arc length at ring centre)
    dtype=torch.float32,
    device: str = "cpu",
) -> Tuple[Tensor, Tensor]:
    """
    Build the cylindrical collimator ring as wedge-shaped quad segments.

    The ring is modeled as alternating:
      - Tungsten wall segments (solid, high attenuation)
      - Open aperture gaps (no material)

    Returns:
      plate_segments: (M, 4, 2) quad vertices for tungsten walls
      aperture_centers: (n_apertures, 2) centre positions of apertures
    """
    r_out = r_in + thickness
    r_center = (r_in + r_out) / 2.0

    # Angular spacing between aperture centres
    dtheta_cell = 2.0 * math.pi / n_apertures
    # Angular half-width of each aperture opening
    dtheta_open = aperture_size_mm / r_center  # arc length / radius = angle

    def make_wedge_quad(r1: float, r2: float, th1: float, th2: float) -> Tensor:
        v1 = torch.tensor([r1 * math.cos(th1), r1 * math.sin(th1)], dtype=dtype)
        v2 = torch.tensor([r2 * math.cos(th1), r2 * math.sin(th1)], dtype=dtype)
        v3 = torch.tensor([r2 * math.cos(th2), r2 * math.sin(th2)], dtype=dtype)
        v4 = torch.tensor([r1 * math.cos(th2), r1 * math.sin(th2)], dtype=dtype)
        return torch.stack([v1, v2, v3, v4])

    segments: List[Tensor] = []
    aperture_centres: List[Tensor] = []

    for i in range(n_apertures):
        th_center = i * dtheta_cell
        th_open_left  = th_center - 0.5 * dtheta_open
        th_open_right = th_center + 0.5 * dtheta_open
        th_left_mid   = th_center - 0.5 * dtheta_cell
        th_right_mid  = th_center + 0.5 * dtheta_cell

        # Left wall: from cell left edge to aperture left edge
        segments.append(make_wedge_quad(r_in, r_out, th_left_mid, th_open_left))
        # Right wall: from aperture right edge to cell right edge
        segments.append(make_wedge_quad(r_in, r_out, th_open_right, th_right_mid))

        # Aperture centre position
        cx = r_center * math.cos(th_center)
        cy = r_center * math.sin(th_center)
        aperture_centres.append(torch.tensor([cx, cy], dtype=dtype))

    plate_segments = torch.stack(segments, dim=0)
    aperture_centers = torch.stack(aperture_centres, dim=0)
    return plate_segments, aperture_centers


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Brain SPECT polygon-panel geometry")
    parser.add_argument("--n_panels", type=int, default=6,
                        help="Number of detector panels (default: 6 = hexagon)")
    parser.add_argument("--collimator_inner_r", type=float, default=120.0,
                        help="Collimator ring inner radius in mm (default: 120.0)")
    parser.add_argument("--collimator_thickness", type=float, default=2.0,
                        help="Collimator ring radial thickness in mm (default: 2.0)")
    parser.add_argument("--n_apertures", type=int, default=200,
                        help="Number of apertures on collimator ring (default: 200)")
    parser.add_argument("--aperture_size_mm", type=float, default=2.0,
                        help="Aperture arc length at ring centre in mm (default: 2.0)")
    parser.add_argument("--panel_length_mm", type=float, default=120.0,
                        help="Tangential length of each detector panel in mm (default: 120 = 5 modules x 8 x 3mm)")
    parser.add_argument("--crystal_size_mm", type=float, default=3.0,
                        help="Crystal size (tangential pitch = radial thickness) in mm (default: 3.0)")
    parser.add_argument("--n_layers", type=int, default=3,
                        help="Number of crystal layers per panel (default: 3)")
    parser.add_argument("--opening_ratio", type=float, default=1.0/3.0,
                        help="Crystal occupancy fraction for inner/middle layers (default: 1/3)")
    parser.add_argument("--layer_gap_mm", type=float, default=0.0,
                        help="Air gap between crystal layers in mm (default: 0.0)")
    parser.add_argument("--n_rotations", type=int, default=2,
                        help="Number of collimator rotation steps (default: 2)")
    parser.add_argument("--rotation_step_deg", type=float, default=1.0,
                        help="Angular step between collimator rotations in deg (default: 1.0)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for .tensor file (default: current dir)")
    args = parser.parse_args()

    # ── Derived parameters ────────────────────────────────────────────────────
    COLLIMATOR_INNER_R   = args.collimator_inner_r
    COLLIMATOR_THICKNESS = args.collimator_thickness
    COLLIMATOR_OUTER_R   = COLLIMATOR_INNER_R + COLLIMATOR_THICKNESS

    # Detector panel inner face sits just outside the collimator outer radius
    # with a small gap (1 mm) for clearance
    PANEL_CLEARANCE_MM = 1.0
    PANEL_INNER_R = COLLIMATOR_OUTER_R + PANEL_CLEARANCE_MM

    N_PANELS       = args.n_panels
    PANEL_LENGTH   = args.panel_length_mm
    CRYSTAL_SIZE   = args.crystal_size_mm
    N_LAYERS       = args.n_layers
    OPENING_RATIO  = args.opening_ratio
    LAYER_GAP      = args.layer_gap_mm

    N_APERTURES    = args.n_apertures
    APERTURE_SIZE  = args.aperture_size_mm
    FOV_DIAMETER   = 200.0   # mm — fixed for brain SPECT

    print("=" * 60)
    print("Brain SPECT Polygon-Panel Geometry")
    print("=" * 60)
    print(f"  N panels:              {N_PANELS}")
    print(f"  Collimator inner r:    {COLLIMATOR_INNER_R} mm")
    print(f"  Collimator thickness:  {COLLIMATOR_THICKNESS} mm")
    print(f"  Collimator outer r:    {COLLIMATOR_OUTER_R} mm")
    print(f"  Panel inner r:         {PANEL_INNER_R} mm")
    print(f"  Panel length:          {PANEL_LENGTH} mm")
    print(f"  Crystal size:          {CRYSTAL_SIZE} mm")
    print(f"  N crystal layers:      {N_LAYERS}")
    print(f"  Opening ratio (inner): {OPENING_RATIO:.3f}")
    print(f"  N apertures:           {N_APERTURES}")
    print(f"  Aperture arc size:     {APERTURE_SIZE} mm")
    print(f"  FOV diameter:          {FOV_DIAMETER} mm")
    print(f"  Collimator rotations:  {args.n_rotations} x {args.rotation_step_deg}°")

    # Validate panel fits within polygon inscribed circle
    panel_inscribed_r = PANEL_INNER_R  # inner face
    polygon_inradius = PANEL_INNER_R   # we place panel faces at exactly this radius
    print(f"\n  Panel face radius:     {polygon_inradius} mm")
    print(f"  FOV radius:            {FOV_DIAMETER/2} mm")
    assert polygon_inradius > FOV_DIAMETER / 2.0, \
        f"Panel face ({polygon_inradius} mm) must be outside FOV radius ({FOV_DIAMETER/2} mm)"

    # ── Build detector panels ─────────────────────────────────────────────────
    print("\n[1/4] Building detector panels...")
    detector_units, det_stats = build_polygon_detector_panels(
        n_panels=N_PANELS,
        panel_distance_mm=PANEL_INNER_R,
        panel_length_mm=PANEL_LENGTH,
        crystal_size_mm=CRYSTAL_SIZE,
        n_layers=N_LAYERS,
        layer_gap_mm=LAYER_GAP,
        opening_ratio=OPENING_RATIO,
    )
    print(f"  Detector tensor: {tuple(detector_units.shape)}")
    print(f"  Total crystals:  {det_stats['total_crystals']}")
    n_crystals_per_panel = det_stats['total_crystals'] // N_PANELS
    print(f"  Crystals/panel:  {n_crystals_per_panel}")
    print(f"  Layout: {N_PANELS} panels x {N_LAYERS} layers x {det_stats['n_outer_crystals_per_layer_per_panel']} positions/layer")

    # ── Build collimator ring ─────────────────────────────────────────────────
    print("\n[2/4] Building cylindrical collimator ring...")
    base_plate_segments, base_aperture_centers = build_cylindrical_collimator_ring(
        r_in=COLLIMATOR_INNER_R,
        thickness=COLLIMATOR_THICKNESS,
        n_apertures=N_APERTURES,
        aperture_size_mm=APERTURE_SIZE,
    )
    print(f"  Plate segments:    {tuple(base_plate_segments.shape)}")
    print(f"  Aperture centers:  {tuple(base_aperture_centers.shape)}")

    # Validate aperture spacing vs size
    r_center_ap = COLLIMATOR_INNER_R + COLLIMATOR_THICKNESS / 2.0
    pitch_arc = 2.0 * math.pi * r_center_ap / N_APERTURES
    wall_arc = pitch_arc - APERTURE_SIZE
    opening_ratio_actual = APERTURE_SIZE / pitch_arc
    print(f"  Arc pitch:         {pitch_arc:.3f} mm")
    print(f"  Wall arc:          {wall_arc:.3f} mm")
    print(f"  Actual open ratio: {opening_ratio_actual:.3f}")
    assert wall_arc > 0, "Apertures overlap — reduce n_apertures or aperture_size_mm"

    # ── MD5 for unique geometry ID ────────────────────────────────────────────
    print("\n[3/4] Computing geometry MD5...")
    centers_with_r = torch.cat(
        [base_aperture_centers,
         torch.full((N_APERTURES, 1), float(APERTURE_SIZE / 2.0))],
        dim=1
    )
    md5 = generate_md5_from_tensors(detector_units, centers_with_r)
    print(f"  MD5: {md5}")

    # ── Collimator rotation motion ────────────────────────────────────────────
    print("\n[4/4] Generating collimator rotation positions...")
    motion_positions = positions_parameters(
        n_rotations=args.n_rotations,
        angle_step_deg=args.rotation_step_deg,
        n_shifts=[1, 1],
        shift_step=[0.0, 0.0],
    )
    n_positions = int(motion_positions.shape[0])
    print(f"  Total positions: {n_positions}")

    # Rotate aperture centers
    transformed_centers_flat = transform_to_positions_2d_batch(
        motion_positions, base_aperture_centers.reshape(-1, 2)
    )
    transformed_centers = transformed_centers_flat.reshape(
        n_positions, N_APERTURES, 2
    )

    # Rotate plate segments
    transformed_segments_flat = transform_to_positions_2d_batch(
        motion_positions, base_plate_segments.reshape(-1, 2)
    )
    transformed_plate_segments = transformed_segments_flat.reshape(
        n_positions,
        base_plate_segments.shape[0],
        base_plate_segments.shape[1],
        2,
    )

    # ── Package and save ──────────────────────────────────────────────────────
    out_data: OutDataDict = {  # type: ignore
        "scanner MD5": md5,
        "brain_spect_config": {
            "n_panels": N_PANELS,
            "collimator_inner_r_mm": COLLIMATOR_INNER_R,
            "collimator_thickness_mm": COLLIMATOR_THICKNESS,
            "panel_inner_r_mm": PANEL_INNER_R,
            "panel_length_mm": PANEL_LENGTH,
            "crystal_size_mm": CRYSTAL_SIZE,
            "n_layers": N_LAYERS,
            "opening_ratio": OPENING_RATIO,
            "n_apertures": N_APERTURES,
            "aperture_size_mm": APERTURE_SIZE,
            "fov_diameter_mm": FOV_DIAMETER,
            "total_crystals": int(det_stats["total_crystals"]),
        },
        "motion_parameters": {
            "description": "Collimator ring rotates; detectors and object are stationary",
            "n_rotational_steps": args.n_rotations,
            "rotation_step_deg": args.rotation_step_deg,
            "generated_n_positions": n_positions,
        },
        "layouts": {},
    }

    for p in range(n_positions):
        layout_key = f"position {p:03d}"
        out_data["layouts"][layout_key] = {
            "position": motion_positions[p],
            "detector units": detector_units,          # FIXED (stationary)
            "plate circles": {
                "centers": transformed_centers[p],     # ROTATED collimator
                "radius_mm": float(APERTURE_SIZE / 2.0),
            },
            "plate segments": transformed_plate_segments[p],  # ROTATED
        }

    def _fmt(x):
        return f"{x}".replace("-", "m").replace(".", "p")

    motion_id = (
        f"rot{args.n_rotations}"
        f"_ang{_fmt(args.rotation_step_deg)}deg"
        f"_trans1x1_step0p0x0p0"
    )
    out_filename = f"brain_spect_layouts_{md5}_{motion_id}.tensor"
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_filename = os.path.join(args.output_dir, out_filename)

    print(f"\nSaving to: {out_filename}")
    torch_save(out_data, out_filename)
    print("Saved ✅")

    # ── Visualization ─────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle as MplCircle, Wedge

    fig, ax = plt.subplots(figsize=(12, 12))

    # Detector crystals
    plot_polygons_from_vertices_2d_mpl(
        detector_units, ax,
        facecolor="lightblue", edgecolor="blue", alpha=0.5,
        label=f"Detector crystals ({det_stats['total_crystals']} total)"
    )

    # Collimator ring annulus
    ann = Wedge(
        (0, 0), COLLIMATOR_OUTER_R, 0, 360,
        width=COLLIMATOR_THICKNESS,
        facecolor="0.75", edgecolor="0.3", linewidth=1.0, zorder=2,
        label=f"Collimator ring (r={COLLIMATOR_INNER_R}–{COLLIMATOR_OUTER_R} mm)"
    )
    ax.add_patch(ann)

    # Apertures as white circles
    for cx, cy in base_aperture_centers.tolist():
        ax.add_patch(MplCircle(
            (cx, cy), APERTURE_SIZE / 2.0,
            facecolor="white", edgecolor="black", linewidth=0.8, zorder=3
        ))

    # FOV circle
    ax.add_patch(MplCircle(
        (0, 0), FOV_DIAMETER / 2.0,
        edgecolor="red", facecolor="none", linestyle="--", linewidth=2,
        label=f"FOV (D={FOV_DIAMETER} mm)", zorder=4
    ))

    ax.set_aspect("equal")
    lim = PANEL_INNER_R + N_LAYERS * CRYSTAL_SIZE + 20
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(
        f"Brain SPECT — {N_PANELS}-panel polygon + cylindrical collimator\n"
        f"(base pose, {det_stats['total_crystals']} crystals, {N_APERTURES} apertures)"
    )
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)

    fig_path = "brain_spect_geometry.png"
    if args.output_dir:
        fig_path = os.path.join(args.output_dir, fig_path)
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved geometry figure: {fig_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== GEOMETRY SUMMARY ===")
    print(f"  Panels:          {N_PANELS} (hexagon)")
    print(f"  Panel inner r:   {PANEL_INNER_R} mm")
    print(f"  Total crystals:  {det_stats['total_crystals']}")
    print(f"    per panel:     {n_crystals_per_panel}")
    print(f"    per layer:     {det_stats['n_outer_crystals_per_layer_per_panel']} (outer dense)")
    print(f"    inner/mid:     {det_stats['n_inner_crystals_per_layer_per_panel']} positions, {OPENING_RATIO:.0%} occupied")
    print(f"  Collimator:      {N_APERTURES} apertures, {APERTURE_SIZE} mm each")
    print(f"  Open ratio:      {opening_ratio_actual:.3f}")
    print(f"  Positions:       {n_positions} (collimator rotations)")
    print(f"  FOV:             {FOV_DIAMETER} mm diameter")
    print("GEOMETRY OK ✅")
