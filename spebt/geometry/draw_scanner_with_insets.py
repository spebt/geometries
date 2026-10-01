#!/usr/bin/env python3
"""SAI SC-SPECT top-view drawing with insets that reveal the fine features.

Adds zoomed-in panels for:
  (a) HR collimator ring  — shows individual apertures + tungsten walls
  (b) Detector ring 1     — shows cell pairs and the cell-to-cell gaps
  (c) Detector ring 2     — same, smaller cell spacing
  (d) Detector ring 3     — same

Geometry parameters mirror generate_mph_scanner_circularfov.py exactly.
The original file is NOT modified. Run this script from the geometry/ folder.
"""
import math
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, PatchCollection
from matplotlib.patches import Wedge, Circle, Rectangle, ConnectionPatch
from matplotlib.path import Path


# ---------- parameters (same as the production geometry script) ----------
HR_INNER_DIAM_MM = 67.5
HR_THICK_MM      = 2.5
APERTURE_DIAM_MM = 0.4
APERTURE_COUNT   = 180

RING_INNER_DIAMS_MM = [260.0, 390.0, 520.0, 650.0]
DETS_PER_RING       = [480, 720, 960, 1200]
SCINT_TANGENT_MM    = 0.84       # W
SCINT_RADIAL_MM     = 6.0        # H
INTRA_CELL_GAP_MM   = 0.84       # gap between the 2 crystals inside one cell

FOV_DIAM_MM = 10.0


# ---------- geometry builders (mirror the originals, no torch) ----------
def build_ring_polygons(inner_d, n_scint, W=SCINT_TANGENT_MM, H=SCINT_RADIAL_MM,
                        gap=INTRA_CELL_GAP_MM):
    """Return (N, 4, 2) array of crystal rectangle vertices (x, y)."""
    n_cells = int(n_scint) // 2
    r_in = inner_d / 2.0
    r_c = r_in + H / 2.0
    dtheta = 2.0 * math.pi / n_cells
    half_pair = (W + gap) / 2.0

    verts = np.empty((n_cells * 2, 4, 2), dtype=np.float64)
    for i in range(n_cells):
        th = i * dtheta
        t = np.array([-math.sin(th), math.cos(th)])
        r = np.array([ math.cos(th), math.sin(th)])
        slot_center = r * r_c
        for k, sgn in enumerate((-1.0, 1.0)):
            c = slot_center + sgn * half_pair * t
            v1 = c + (W / 2) * t + (H / 2) * r
            v2 = c - (W / 2) * t + (H / 2) * r
            v3 = c - (W / 2) * t - (H / 2) * r
            v4 = c + (W / 2) * t - (H / 2) * r
            verts[2 * i + k] = np.stack([v1, v2, v3, v4])
    return verts, r_c, dtheta


def build_tungsten_segments(n, r_in, r_out, ap_diam):
    """Return (2*n, 4, 2) wedge polygons forming the walls between apertures."""
    r_c = (r_in + r_out) / 2.0
    dth_cell = 2.0 * math.pi / n
    dth_open = ap_diam / r_c

    def wedge(th1, th2):
        return np.array([
            [r_in  * math.cos(th1), r_in  * math.sin(th1)],
            [r_out * math.cos(th1), r_out * math.sin(th1)],
            [r_out * math.cos(th2), r_out * math.sin(th2)],
            [r_in  * math.cos(th2), r_in  * math.sin(th2)],
        ])

    segs = []
    for i in range(n):
        th_c = i * dth_cell
        segs.append(wedge(th_c - 0.5 * dth_cell, th_c - 0.5 * dth_open))
        segs.append(wedge(th_c + 0.5 * dth_open, th_c + 0.5 * dth_cell))
    return np.stack(segs, axis=0)


def aperture_centers(n, r_c):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.column_stack([r_c * np.cos(a), r_c * np.sin(a)])


def aperture_centers_random(n, r_c, min_chord_mm, seed=2025, max_attempts=200_000):
    """Random aperture angles on the ring center circle, with a minimum chord
    spacing constraint. Matches the paper's 'randomized aperture arrangement'
    on the HR plate. Rejection sampling is used; raises if it cannot place n
    apertures within max_attempts (means the constraint is infeasible)."""
    rng = random.Random(seed)
    min_angle = 2.0 * math.asin(min_chord_mm / (2.0 * r_c))
    angles = []
    attempts = 0
    while len(angles) < n and attempts < max_attempts:
        cand = rng.uniform(0.0, 2.0 * math.pi)
        ok = True
        for a in angles:
            d = abs(cand - a)
            if d > math.pi:
                d = 2.0 * math.pi - d
            if d < min_angle:
                ok = False
                break
        if ok:
            angles.append(cand)
        attempts += 1
    if len(angles) < n:
        raise RuntimeError(f"only placed {len(angles)}/{n} apertures (chord {min_chord_mm} mm)")
    angles = np.array(sorted(angles))
    return np.column_stack([r_c * np.cos(angles), r_c * np.sin(angles)])


# ---------- compute geometry ----------
HR_R_IN  = HR_INNER_DIAM_MM / 2.0
HR_R_OUT = HR_R_IN + HR_THICK_MM
HR_R_C   = HR_R_IN + HR_THICK_MM / 2.0
HR_R_AP  = APERTURE_DIAM_MM / 2.0

# Paper specifies a *randomized* aperture arrangement on the HR plate, with a
# minimum aperture-to-aperture chord spacing of 0.8 mm (MIN_SPACING_MM).
APERTURE_MIN_CHORD_MM = 0.8
APERTURE_SEED = int(os.getenv("HR_APERTURE_SEED", "2025"))
ap_centers = aperture_centers_random(APERTURE_COUNT, HR_R_C,
                                     min_chord_mm=APERTURE_MIN_CHORD_MM,
                                     seed=APERTURE_SEED)

ring_data = []  # list of (verts, r_c, dtheta, label, color)
RING_COLORS = ["#1F77B4", "#2CA02C", "#FF7F0E", "#9467BD"]
for i, (d, n) in enumerate(zip(RING_INNER_DIAMS_MM, DETS_PER_RING)):
    verts, r_c, dtheta = build_ring_polygons(d, n)
    ring_data.append((verts, r_c, dtheta, f"Ring {i+1}", RING_COLORS[i]))

print(f"Apertures: {APERTURE_COUNT} (randomized, min chord {APERTURE_MIN_CHORD_MM} mm, seed {APERTURE_SEED})")
_d_ap = np.linalg.norm(ap_centers[:, None, :] - ap_centers[None, :, :], axis=-1)
_d_ap[np.arange(APERTURE_COUNT), np.arange(APERTURE_COUNT)] = np.inf
print(f"  realized min chord: {_d_ap.min():.3f} mm")
for verts, r_c, dtheta, label, _ in ring_data:
    print(f"  {label}: {len(verts)} crystals, r_c={r_c:.2f} mm, "
          f"arc/cell={r_c*dtheta:.3f} mm, "
          f"cell-to-cell gap={r_c*dtheta - (2*SCINT_TANGENT_MM + INTRA_CELL_GAP_MM):.3f} mm")


# ---------- figure layout: single square main plot with two in-graph insets ----------
fig, ax_main = plt.subplots(figsize=(11, 11))

# ---------- draw main overview ----------
legend_handles = []
for verts, _, _, label, color in ring_data:
    pc = PolyCollection(verts, facecolor=color, edgecolor=color, linewidths=0.1, alpha=0.85)
    ax_main.add_collection(pc)
    legend_handles.append(plt.Line2D([0], [0], marker='s', color='w',
                                     markerfacecolor=color, markersize=10, label=label))

# HR ring as annulus (apertures invisible at this scale)
ax_main.add_patch(Wedge((0, 0), HR_R_OUT, 0, 360, width=HR_THICK_MM,
                        facecolor='0.78', edgecolor='0.40', lw=1.2, zorder=3))

# FOV
fov_circle = Circle((0, 0), FOV_DIAM_MM / 2.0, edgecolor='red', facecolor='none',
                    linestyle='--', linewidth=1.6, zorder=4)
ax_main.add_patch(fov_circle)
legend_handles.append(plt.Line2D([0], [0], color='red', linestyle='--',
                                  label=f'Effective FOV (D={FOV_DIAM_MM:.0f} mm)'))
legend_handles.append(plt.Line2D([0], [0], marker='s', color='w',
                                  markerfacecolor='0.78', markeredgecolor='0.40',
                                  markersize=10, label='HR collimator ring'))

ax_main.set_aspect('equal', adjustable='box')
lim = max(d / 2.0 for d in RING_INNER_DIAMS_MM) + SCINT_RADIAL_MM + 15
ax_main.set_xlim(-lim, lim)
ax_main.set_ylim(-lim, lim)
ax_main.grid(True, alpha=0.3)
ax_main.set_xlabel('X (mm)')
ax_main.set_ylabel('Y (mm)')
ax_main.legend(handles=legend_handles, loc='lower right', fontsize=9, framealpha=0.92)


CONN_COLOR = "#1F9E45"  # green like the mph_geometry reference


def add_circular_inset(ax_main, bounds, zoom_xy_data, title, draw_content):
    """Place a circular inset inside ax_main and draw two tangent dashed
    connector lines from zoom_xy_data (data coords) to the inset's circle.
    `bounds` is [x0, y0, w, h] in ax_main axes-fraction coords.
    `draw_content(ax_in)` is called to draw the zoomed geometry."""
    ax_in = ax_main.inset_axes(bounds)
    bx, by, bw, bh = bounds
    icx, icy = bx + bw / 2.0, by + bh / 2.0
    ir = min(bw, bh) / 2.0

    # rectangular axes machinery off
    for spine in ax_in.spines.values():
        spine.set_visible(False)
    ax_in.set_xticks([]); ax_in.set_yticks([])
    ax_in.patch.set_visible(False)

    # subtle circular background so the inset reads as a distinct viewing area
    bg_circle = Circle((0.5, 0.5), 0.5, transform=ax_in.transAxes,
                       facecolor='#F4F4F4', edgecolor='none', zorder=-5)
    ax_in.add_patch(bg_circle)

    # snapshot existing children so we know what's added by draw_content
    pre_ids = {id(a) for a in ax_in.get_children()}
    draw_content(ax_in)
    new_artists = [a for a in ax_in.get_children() if id(a) not in pre_ids]

    # build a circular Path (axes-fraction coords) and apply as clip
    n = 96
    th = np.linspace(0.0, 2.0 * math.pi, n)
    verts = np.column_stack([0.5 + 0.5 * np.cos(th), 0.5 + 0.5 * np.sin(th)])
    clip_path_obj = Path(verts)
    for artist in new_artists:
        try:
            artist.set_clip_path(clip_path_obj, ax_in.transAxes)
        except Exception:
            pass

    # circular border on top
    border = Circle((0.5, 0.5), 0.5, transform=ax_in.transAxes,
                    facecolor='none', edgecolor='black', lw=1.2, zorder=200)
    ax_in.add_patch(border)

    # title above the circle
    if title:
        ax_in.set_title(title, fontsize=10, fontweight='bold', pad=3)

    # zoom dot location in ax_main axes fraction
    xlim = ax_main.get_xlim(); ylim = ax_main.get_ylim()
    fx = (zoom_xy_data[0] - xlim[0]) / (xlim[1] - xlim[0])
    fy = (zoom_xy_data[1] - ylim[0]) / (ylim[1] - ylim[0])

    # tangent lines from external point (fx, fy) to circle at (icx, icy), r=ir
    dx, dy = fx - icx, fy - icy
    dist = math.hypot(dx, dy)
    if dist > ir:
        phi = math.atan2(dy, dx)
        alpha = math.acos(ir / dist)
        for sign in (+1.0, -1.0):
            ang = phi + sign * alpha
            tp = (icx + ir * math.cos(ang), icy + ir * math.sin(ang))
            con = ConnectionPatch(
                xyA=zoom_xy_data, coordsA=ax_main.transData,
                xyB=tp,           coordsB=ax_main.transAxes,
                arrowstyle='-', linestyle='--', linewidth=1.4,
                color=CONN_COLOR, alpha=0.92, zorder=50,
            )
            ax_main.add_artist(con)

    # small marker at the zoom origin
    ax_main.plot([zoom_xy_data[0]], [zoom_xy_data[1]],
                 marker='o', color=CONN_COLOR, markersize=4.5, zorder=51)

    return ax_in


# ---------- inset (a): HR collimator zoom, upper-right of main plot ----------
def _draw_hr_inset(ax):
    # tight window so the tungsten annulus fills the circular inset
    cx, cy, half = HR_R_C, 0.0, 1.7
    ax.add_patch(Wedge((0, 0), HR_R_OUT, 0, 360, width=HR_THICK_MM,
                       facecolor='#6e6e6e', edgecolor='#2a2a2a',
                       linewidth=0.9, zorder=3))
    for px, py in ap_centers:
        if abs(px - cx) < half + 1.0 and abs(py - cy) < half + 1.0:
            ax.add_patch(Circle((px, py), HR_R_AP, facecolor='white',
                                edgecolor='black', linewidth=0.9, zorder=5))
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect('equal')


def _draw_ring1_inset(ax):
    verts, r_c, dtheta, label, color = ring_data[0]
    cx, cy, half = r_c, 0.0, 4.5
    in_win = np.any((verts[:, :, 0] >= cx - half - 1) &
                    (verts[:, :, 0] <= cx + half + 1) &
                    (verts[:, :, 1] >= cy - half - 1) &
                    (verts[:, :, 1] <= cy + half + 1), axis=1)
    ax.add_collection(PolyCollection(verts[in_win], facecolor=color,
                                     edgecolor=color, linewidths=0.6, alpha=0.90))
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect('equal')


INSET_SIZE = 0.22  # axes fraction
# Insets are stacked vertically on the LEFT side so the two cones fan to the
# left and don't cross. (a) HR sits lower (closer to the data origin), (b)
# Ring 1 sits higher (farther from origin).
ax_a = add_circular_inset(
    ax_main, [0.05, 0.40, INSET_SIZE, INSET_SIZE],
    zoom_xy_data=(HR_R_C, 0.0),
    title='(a) HR collimator',
    draw_content=_draw_hr_inset,
)
ax_b = add_circular_inset(
    ax_main, [0.05, 0.72, INSET_SIZE, INSET_SIZE],
    zoom_xy_data=(ring_data[0][1], 0.0),
    title='(b) Detector ring 1',
    draw_content=_draw_ring1_inset,
)


# ---------- save ----------
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "scspect_with_insets.png")
plt.savefig(out_path, dpi=200, facecolor='white')
print(f"Saved: {out_path}")
