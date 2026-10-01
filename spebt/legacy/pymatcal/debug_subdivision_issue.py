#!/usr/bin/env python3
"""
Diagnostic plots explaining the crystal subdivision sensitivity bug.

ROOT CAUSE:  CRYSTAL_SUBS = (1, 5) → 5 radial slices
  - Each slice subtends ~same angular span as full crystal → 5x angular overcounting
  - No self-attenuation between slices → each independently absorbs photons
  - Combined overestimate: ~2.3x → pushes sensitivity above 1.0

FIX:         CRYSTAL_SUBS = (5, 1) → 5 tangential strips
  - Angular spans partition the crystal → no overcounting
  - No self-shielding issue → strips are side-by-side
  - Sum recovers exact correct PPDF
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Arc, Wedge
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

# ── System parameters ──────────────────────────────────────────────
W = 0.84        # tangential width (mm)
H = 6.0         # radial thickness (mm)
MU = 0.5        # mu_crystal (mm⁻¹)
R_RING1 = 263.0 # center radius of ring 1 (mm)
R_COLL  = 35.0  # collimator center radius (mm)
N_SUBS  = 5     # number of subdivisions

# ── Color palette ──────────────────────────────────────────────────
COLORS_RADIAL = plt.cm.Blues(np.linspace(0.3, 0.85, N_SUBS))
COLORS_TANGENTIAL = plt.cm.Oranges(np.linspace(0.3, 0.85, N_SUBS))
COLOR_CORRECT = '#2ca02c'
COLOR_BUG     = '#d62728'
COLOR_PIXEL   = '#ff7f0e'


# ====================================================================
# FIGURE 1: Crystal subdivision geometry comparison
# ====================================================================
def fig1_subdivision_geometry():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))

    def draw_crystal(ax, title, subdivisions, colors, labels=True):
        """Draw a crystal with its subdivisions."""
        ax.set_xlim(-1.0, 1.5)
        ax.set_ylim(-0.8, 7.0)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=12)

        n_tang, n_rad = subdivisions
        dw = W / n_tang
        dh = H / n_rad

        for it in range(n_tang):
            for ir in range(n_rad):
                x0 = -W/2 + it * dw
                y0 = ir * dh
                idx = ir if n_rad > 1 else it
                rect = plt.Rectangle((x0, y0), dw, dh,
                                     facecolor=colors[idx], edgecolor='black',
                                     linewidth=1.5, alpha=0.85)
                ax.add_patch(rect)
                cx, cy = x0 + dw/2, y0 + dh/2
                label = f"Sub {idx+1}" if (n_tang == 1 or ir == 0) and (n_rad == 1 or it == 0) else ""
                if label:
                    ax.text(cx, cy, label, ha='center', va='center',
                            fontsize=9, fontweight='bold', color='white')

        # Dimension annotations
        # Width arrow
        y_arrow = -0.4
        ax.annotate('', xy=(W/2, y_arrow), xytext=(-W/2, y_arrow),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
        ax.text(0, y_arrow - 0.25, f'W = {W} mm\n(tangential)',
                ha='center', va='top', fontsize=9)

        # Height arrow
        x_arrow = W/2 + 0.25
        ax.annotate('', xy=(x_arrow, H), xytext=(x_arrow, 0),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
        ax.text(x_arrow + 0.1, H/2, f'H = {H} mm\n(radial / depth)',
                ha='left', va='center', fontsize=9, rotation=90)

        # Sub-element dimensions
        if n_rad > 1:
            x_sub = -W/2 - 0.15
            ax.annotate('', xy=(x_sub, dh), xytext=(x_sub, 0),
                        arrowprops=dict(arrowstyle='<->', color=COLOR_BUG, lw=1.2))
            ax.text(x_sub - 0.1, dh/2, f'{dh:.1f}\nmm',
                    ha='right', va='center', fontsize=8, color=COLOR_BUG)
        if n_tang > 1:
            y_sub = H + 0.15
            ax.annotate('', xy=(-W/2 + dw, y_sub), xytext=(-W/2, y_sub),
                        arrowprops=dict(arrowstyle='<->', color=COLOR_CORRECT, lw=1.2))
            ax.text(-W/2 + dw/2, y_sub + 0.2, f'{dw:.3f} mm',
                    ha='center', va='bottom', fontsize=8, color=COLOR_CORRECT)

        # Radial direction label
        ax.annotate('', xy=(0, H + 0.8), xytext=(0, H + 0.2),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
        ax.text(0, H + 1.0, '← toward detector outer face\n(away from FOV)',
                ha='center', va='bottom', fontsize=8, color='gray')
        ax.annotate('', xy=(0, -0.2), xytext=(0, -0.8),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
        ax.text(0, -0.85, 'toward FOV (photon source) →',
                ha='center', va='top', fontsize=8, color='gray')

        ax.set_xlabel('Tangential (mm)', fontsize=10)
        ax.set_ylabel('Radial depth (mm)', fontsize=10)
        ax.grid(True, alpha=0.2)

    # Panel A: No subdivision
    draw_crystal(axes[0], '(A) No Subdivision (1,1)', (1, 1),
                 [plt.cm.Greens(0.5)])

    # Panel B: Radial (1,5) — THE BUG
    draw_crystal(axes[1], '(B) Radial (1,5) — CURRENT (BUG)',
                 (1, N_SUBS), COLORS_RADIAL)

    # Panel C: Tangential (5,1) — THE FIX
    draw_crystal(axes[2], '(C) Tangential (5,1) — FIX',
                 (N_SUBS, 1), COLORS_TANGENTIAL)

    fig.suptitle('Crystal Subdivision Schemes\n'
                 f'Crystal: {W} mm (tangential) × {H} mm (radial depth)',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig('debug_fig1_subdivision_geometry.png', dpi=200, bbox_inches='tight')
    print("Saved: debug_fig1_subdivision_geometry.png")
    plt.close(fig)


# ====================================================================
# FIGURE 2: Angular span overlap problem (zoomed schematic)
# ====================================================================
def fig2_angular_span_overlap():
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # --- Schematic parameters (scaled for visualization) ---
    pixel_pos = np.array([0.0, 0.0])
    crystal_dist = 8.0       # scaled distance for visualization
    crystal_w = 1.2           # scaled width
    crystal_h = 4.0           # scaled depth
    n_subs = 5

    for panel_idx, (ax, scheme, title) in enumerate(zip(
        axes,
        ['radial', 'tangential'],
        ['(A) Radial Subs (1,5) — OVERLAPPING SPANS',
         '(B) Tangential Subs (5,1) — PARTITIONED SPANS']
    )):
        ax.set_xlim(-3.5, 14)
        ax.set_ylim(-5.5, 5.5)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=13, fontweight='bold',
                     color=COLOR_BUG if panel_idx == 0 else COLOR_CORRECT, pad=12)

        # Draw pixel
        ax.plot(*pixel_pos, 'o', color=COLOR_PIXEL, markersize=12, zorder=10)
        ax.text(pixel_pos[0], pixel_pos[1] - 0.7, 'FOV pixel',
                ha='center', fontsize=10, color=COLOR_PIXEL, fontweight='bold')

        if scheme == 'radial':
            colors = COLORS_RADIAL
            dh = crystal_h / n_subs

            for i in range(n_subs):
                x0 = crystal_dist + i * dh
                y0 = -crystal_w / 2
                rect = plt.Rectangle((x0, y0), dh, crystal_w,
                                     facecolor=colors[i], edgecolor='black',
                                     linewidth=1.2, alpha=0.85, zorder=5)
                ax.add_patch(rect)
                ax.text(x0 + dh/2, 0, f'S{i+1}', ha='center', va='center',
                        fontsize=8, fontweight='bold', color='white', zorder=6)

                # Angular span arc for each subdivision (nearly identical!)
                # All subs at roughly same angle → overlapping arcs
                sub_center_x = x0 + dh / 2
                half_angle = np.degrees(np.arctan2(crystal_w / 2, sub_center_x))
                arc_radius = 2.0 + i * 0.4
                arc = Arc(pixel_pos, 2*arc_radius, 2*arc_radius,
                         angle=0, theta1=-half_angle, theta2=half_angle,
                         color=colors[i], linewidth=2.5, zorder=4)
                ax.add_patch(arc)

                # Dashed ray to center of each subdivision
                ax.plot([0, sub_center_x], [0, 0], '--', color=colors[i],
                        alpha=0.4, linewidth=0.8)

            # Label overlapping arcs
            ax.annotate('All 5 arcs ≈ same\nangle → 5× overcounting!',
                        xy=(3.5, 1.5), fontsize=10, color=COLOR_BUG,
                        fontweight='bold', ha='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffcccc',
                                  edgecolor=COLOR_BUG, alpha=0.9))

            # Self-attenuation arrows
            for i in range(1, n_subs):
                x_arr = crystal_dist + i * dh
                ax.annotate('', xy=(x_arr + 0.05, -crystal_w/2 - 0.4),
                           xytext=(crystal_dist + 0.05, -crystal_w/2 - 0.4),
                           arrowprops=dict(arrowstyle='->', color=COLOR_BUG,
                                          lw=1.5, linestyle='--'))
            ax.text(crystal_dist + crystal_h/2, -crystal_w/2 - 0.8,
                    'Photon must traverse S1→S2→...→S5\nbut code ignores this attenuation!',
                    ha='center', fontsize=9, color=COLOR_BUG, style='italic')

        else:  # tangential
            colors = COLORS_TANGENTIAL
            dw = crystal_w / n_subs

            for i in range(n_subs):
                x0 = crystal_dist
                y0 = -crystal_w/2 + i * dw
                rect = plt.Rectangle((x0, y0), crystal_h, dw,
                                     facecolor=colors[i], edgecolor='black',
                                     linewidth=1.2, alpha=0.85, zorder=5)
                ax.add_patch(rect)
                ax.text(x0 + crystal_h/2, y0 + dw/2, f'S{i+1}',
                        ha='center', va='center', fontsize=8,
                        fontweight='bold', color='white', zorder=6)

                # Angular span arc for each subdivision (each is 1/5 of total!)
                sub_center_y = y0 + dw / 2
                angle_to_sub = np.degrees(np.arctan2(sub_center_y, crystal_dist + crystal_h/2))
                half_sub_angle = np.degrees(np.arctan2(dw / 2, crystal_dist + crystal_h/2))
                arc_radius = 3.5
                arc = Arc(pixel_pos, 2*arc_radius, 2*arc_radius,
                         angle=0,
                         theta1=angle_to_sub - half_sub_angle,
                         theta2=angle_to_sub + half_sub_angle,
                         color=colors[i], linewidth=3.0, zorder=4)
                ax.add_patch(arc)

            # Label partitioned arcs
            ax.annotate('5 arcs partition the\ntotal angle → correct sum!',
                        xy=(3.5, 2.5), fontsize=10, color=COLOR_CORRECT,
                        fontweight='bold', ha='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#ccffcc',
                                  edgecolor=COLOR_CORRECT, alpha=0.9))

            ax.text(crystal_dist + crystal_h/2, -crystal_w/2 - 0.8,
                    'No self-shielding: strips are side-by-side,\nphotons reach each strip independently ✓',
                    ha='center', fontsize=9, color=COLOR_CORRECT, style='italic')

        # Common labels
        ax.annotate('', xy=(crystal_dist + crystal_h + 0.5, 0),
                    xytext=(crystal_dist - 0.5, 0),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1))
        ax.text(crystal_dist + crystal_h/2, crystal_w/2 + 1.5,
                f'Radial depth ({crystal_h}→{H} mm)', ha='center',
                fontsize=9, color='gray')

        ax.set_xlabel('Radial direction (toward detector)', fontsize=10)
        ax.set_ylabel('Tangential direction', fontsize=10)
        ax.grid(True, alpha=0.15)

    fig.suptitle('Angular Span: Radial vs Tangential Subdivisions\n'
                 '(view from above, photon traveling from FOV pixel → crystal)',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig('debug_fig2_angular_span_overlap.png', dpi=200, bbox_inches='tight')
    print("Saved: debug_fig2_angular_span_overlap.png")
    plt.close(fig)


# ====================================================================
# FIGURE 3: Self-attenuation and PPDF overestimate (quantitative)
# ====================================================================
def fig3_self_attenuation():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ── Panel A: Photon survival through radial slices ──
    ax = axes[0]
    depths = np.linspace(0, H, 200)
    survival = np.exp(-MU * depths)
    ax.fill_between(depths, survival, alpha=0.15, color='steelblue')
    ax.plot(depths, survival, 'steelblue', linewidth=2.5,
            label=f'Survival: exp(-{MU}×d)')

    # Mark subdivision boundaries
    for i in range(N_SUBS + 1):
        d = i * H / N_SUBS
        ax.axvline(d, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)

    # Show what fraction reaches each subdivision
    for i in range(N_SUBS):
        d_start = i * H / N_SUBS
        d_end = (i + 1) * H / N_SUBS
        d_mid = (d_start + d_end) / 2
        frac_arriving = np.exp(-MU * d_start)
        frac_absorbed = frac_arriving * (1 - np.exp(-MU * H / N_SUBS))
        ax.annotate(f'S{i+1}: {frac_absorbed:.1%}\nabsorbed',
                    xy=(d_mid, frac_arriving),
                    xytext=(d_mid, frac_arriving + 0.08),
                    fontsize=7.5, ha='center', color=COLORS_RADIAL[i],
                    fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=COLORS_RADIAL[i], lw=1))
        # Shade this subdivision
        d_range = np.linspace(d_start, d_end, 50)
        ax.fill_between(d_range, np.exp(-MU * d_start), np.exp(-MU * d_range),
                        alpha=0.3, color=COLORS_RADIAL[i])

    total_correct = 1 - np.exp(-MU * H)
    ax.axhline(1 - total_correct, color=COLOR_CORRECT, linestyle=':', linewidth=1.5)
    ax.text(H * 0.95, 1 - total_correct + 0.03,
            f'Total absorbed (correct): {total_correct:.1%}',
            ha='right', fontsize=9, color=COLOR_CORRECT)

    ax.set_xlabel('Depth into crystal (mm)', fontsize=11)
    ax.set_ylabel('Fraction of photons surviving', fontsize=11)
    ax.set_title('(A) Photon Attenuation Through Crystal Depth\n'
                 f'μ = {MU} mm⁻¹, H = {H} mm',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(0, H)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.2)

    # ── Panel B: Bar chart — per-subdivision PPDF contribution ──
    ax = axes[1]
    x_pos = np.arange(N_SUBS)
    bar_width = 0.35

    # Bug values (no self-attenuation)
    bug_per_sub = (1 - np.exp(-MU * H / N_SUBS))
    bug_values = np.full(N_SUBS, bug_per_sub)

    # Correct values (with self-attenuation)
    correct_values = []
    for i in range(N_SUBS):
        d_start = i * H / N_SUBS
        arriving = np.exp(-MU * d_start)
        absorbed = arriving * (1 - np.exp(-MU * H / N_SUBS))
        correct_values.append(absorbed)
    correct_values = np.array(correct_values)

    bars1 = ax.bar(x_pos - bar_width/2, bug_values, bar_width,
                   color=COLOR_BUG, alpha=0.8, label='Code (no self-atten.)', edgecolor='black')
    bars2 = ax.bar(x_pos + bar_width/2, correct_values, bar_width,
                   color=COLOR_CORRECT, alpha=0.8, label='Correct (with self-atten.)', edgecolor='black')

    # Annotate each bar
    for i, (bv, cv) in enumerate(zip(bug_values, correct_values)):
        ax.text(i - bar_width/2, bv + 0.01, f'{bv:.3f}', ha='center', fontsize=8, color=COLOR_BUG)
        ax.text(i + bar_width/2, cv + 0.01, f'{cv:.3f}', ha='center', fontsize=8, color=COLOR_CORRECT)

    # Sum lines
    bug_total = bug_values.sum()
    correct_total = correct_values.sum()
    ax.axhline(bug_total / N_SUBS, color=COLOR_BUG, linestyle='--', alpha=0.0)  # hidden, just for scale

    ax.set_xlabel('Subdivision index', fontsize=11)
    ax.set_ylabel('Interaction probability per sub', fontsize=11)
    ax.set_title('(B) Per-Subdivision Interaction Probability\n'
                 f'(1 − exp(−μ × L_sub)), L_sub = {H/N_SUBS:.1f} mm',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'S{i+1}' for i in range(N_SUBS)])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')

    # ── Panel C: Total PPDF comparison ──
    ax = axes[2]

    # Compare different subdivision schemes
    schemes = ['(1,1)\nNo sub', '(1,3)\nRef code', '(1,5)\nCurrent', '(5,1)\nFix']
    n_rads = [1, 3, 5, 1]

    bug_totals = []
    correct_val = 1 - np.exp(-MU * H)  # always the same correct answer

    for n_r in n_rads:
        L_sub = H / n_r
        per_sub = 1 - np.exp(-MU * L_sub)
        # Bug: n_r subs × per_sub (includes angular overcounting for radial)
        if n_r > 1:
            # For radial subs: angular span overcounted by n_r
            # Total = n_r × per_sub × (θ/2π) but should be 1 × correct_val × (θ/2π)
            bug_totals.append(n_r * per_sub)
        else:
            bug_totals.append(per_sub)

    bug_totals = np.array(bug_totals)
    correct_totals = np.full(len(schemes), correct_val)

    x_pos = np.arange(len(schemes))
    bars1 = ax.bar(x_pos - bar_width/2, bug_totals, bar_width,
                   color=[COLOR_CORRECT, COLOR_BUG, COLOR_BUG, COLOR_CORRECT],
                   alpha=0.8, label='What code computes', edgecolor='black')
    bars2 = ax.bar(x_pos + bar_width/2, correct_totals, bar_width,
                   color=COLOR_CORRECT, alpha=0.3, label='Correct value',
                   edgecolor=COLOR_CORRECT, linewidth=2, hatch='//')

    # Add overestimate ratios
    for i, (bv, cv) in enumerate(zip(bug_totals, correct_totals)):
        ratio = bv / cv
        color = COLOR_BUG if ratio > 1.05 else COLOR_CORRECT
        ax.text(i, max(bv, cv) + 0.08, f'{ratio:.2f}×',
                ha='center', fontsize=12, fontweight='bold', color=color)

    # Horizontal line at 1.0
    ax.axhline(1.0, color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax.text(len(schemes) - 0.5, 1.02, 'physical limit', fontsize=8, color='black', alpha=0.5)

    ax.set_xlabel('Subdivision scheme (tangential, radial)', fontsize=11)
    ax.set_ylabel('Total interaction term\n(× θ/2π gives PPDF per crystal)', fontsize=10)
    ax.set_title('(C) Total PPDF Factor: Code vs Correct\n'
                 f'μ = {MU} mm⁻¹, H = {H} mm',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(schemes, fontsize=10)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_ylim(0, max(bug_totals) * 1.25)

    fig.suptitle('Crystal Subdivision Self-Attenuation Bug — Quantitative Analysis',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig('debug_fig3_self_attenuation.png', dpi=200, bbox_inches='tight')
    print("Saved: debug_fig3_self_attenuation.png")
    plt.close(fig)


# ====================================================================
# FIGURE 4: Full system context — ring cross-section with rays
# ====================================================================
def fig4_system_cross_section():
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for panel_idx, (ax, scheme, title) in enumerate(zip(
        axes,
        [(1, 5), (5, 1)],
        ['(A) Radial (1,5) — PPDF overcounted per crystal',
         '(B) Tangential (5,1) — correct PPDF per crystal']
    )):
        is_bug = panel_idx == 0

        # Draw a zoomed view: pixel at center, collimator ring, one crystal
        # Use much smaller scale for visibility
        scale_R = 10.0   # scaled crystal distance
        scale_coll = 3.0  # scaled collimator distance
        scale_W = 1.5     # scaled crystal width
        scale_H = 3.0     # scaled crystal depth
        ap_width = 0.6    # scaled aperture width

        ax.set_xlim(-2, 15)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=12, fontweight='bold',
                     color=COLOR_BUG if is_bug else COLOR_CORRECT, pad=10)

        # Pixel
        ax.plot(0, 0, 'o', color=COLOR_PIXEL, markersize=10, zorder=10)
        ax.text(0, -0.6, 'pixel j', ha='center', fontsize=10,
                color=COLOR_PIXEL, fontweight='bold')

        # Collimator plate (with aperture gap)
        plate_y_half = 2.5
        # Left wall
        ax.fill_between([scale_coll - 0.3, scale_coll + 0.3],
                        [-plate_y_half, -plate_y_half],
                        [-ap_width/2, -ap_width/2],
                        color='gray', alpha=0.6, zorder=3)
        ax.fill_between([scale_coll - 0.3, scale_coll + 0.3],
                        [ap_width/2, ap_width/2],
                        [plate_y_half, plate_y_half],
                        color='gray', alpha=0.6, zorder=3)
        ax.text(scale_coll, plate_y_half + 0.3, 'W plate\n(μ=3.5)',
                ha='center', fontsize=8, color='gray')

        # Aperture label
        ax.annotate('', xy=(scale_coll + 0.5, ap_width/2),
                    xytext=(scale_coll + 0.5, -ap_width/2),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=1))
        ax.text(scale_coll + 0.8, 0, 'aperture\n0.4 mm',
                ha='left', va='center', fontsize=7)

        # Draw subdivisions
        n_tang, n_rad = scheme
        dw = scale_W / n_tang
        dh = scale_H / n_rad
        colors = COLORS_RADIAL if is_bug else COLORS_TANGENTIAL

        sub_centers = []
        for it in range(n_tang):
            for ir in range(n_rad):
                x0 = scale_R + ir * dh
                y0 = -scale_W/2 + it * dw
                idx = ir if n_rad > 1 else it
                rect = plt.Rectangle((x0, y0), dh, dw,
                                     facecolor=colors[idx], edgecolor='black',
                                     linewidth=1.2, alpha=0.85, zorder=5)
                ax.add_patch(rect)
                cx, cy = x0 + dh/2, y0 + dw/2
                sub_centers.append((cx, cy, idx))
                ax.text(cx, cy, f'S{idx+1}', ha='center', va='center',
                        fontsize=7, fontweight='bold', color='white', zorder=6)

        # Draw rays from pixel through aperture to each subdivision center
        for cx, cy, idx in sub_centers:
            ax.plot([0, cx], [0, cy], '-', color=colors[idx],
                    alpha=0.4, linewidth=1.0, zorder=2)

        # Angular span visualization (arcs from pixel)
        for cx, cy, idx in sub_centers:
            # Skip duplicates for tangential case (same arc per radial position)
            if not is_bug and idx > 0:
                # For tangential subs, show each strip's arc
                pass
            half_angle = np.degrees(np.arctan2(dw/2, cx))
            center_angle = np.degrees(np.arctan2(cy, cx))
            arc_r = 1.8 + idx * 0.3 if is_bug else 2.0
            arc = Arc((0, 0), 2*arc_r, 2*arc_r, angle=0,
                     theta1=center_angle - half_angle,
                     theta2=center_angle + half_angle,
                     color=colors[idx], linewidth=2.5, zorder=4)
            ax.add_patch(arc)

        # Info box
        if is_bug:
            info = (f'Each sub: θ ≈ full crystal angle\n'
                    f'Sum of 5 θ/2π ≈ 5× actual\n'
                    f'Each sub: P_interact = {1-np.exp(-MU*H/N_SUBS):.1%}\n'
                    f'Code total = {N_SUBS*(1-np.exp(-MU*H/N_SUBS)):.3f}\n'
                    f'Correct total = {1-np.exp(-MU*H):.3f}')
            box_color = '#ffcccc'
            edge_color = COLOR_BUG
        else:
            info = (f'Each sub: θ = crystal angle / 5\n'
                    f'Sum of 5 θ/2π = total angle / 2π ✓\n'
                    f'Each sub: P_interact = {1-np.exp(-MU*H):.1%}\n'
                    f'Code total = {1-np.exp(-MU*H):.3f} ✓\n'
                    f'Correct total = {1-np.exp(-MU*H):.3f} ✓')
            box_color = '#ccffcc'
            edge_color = COLOR_CORRECT

        ax.text(7.5, -2.8, info, fontsize=9, fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=box_color,
                          edgecolor=edge_color, alpha=0.9),
                ha='center', va='center')

        ax.set_xlabel('Radial direction (mm, scaled)', fontsize=10)
        ax.set_ylabel('Tangential direction (mm, scaled)', fontsize=10)
        ax.grid(True, alpha=0.15)

    fig.suptitle('System Cross-Section: Ray Tracing Through Aperture to Crystal Subdivisions',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig('debug_fig4_system_cross_section.png', dpi=200, bbox_inches='tight')
    print("Saved: debug_fig4_system_cross_section.png")
    plt.close(fig)


# ====================================================================
# FIGURE 5: Sensitivity impact summary
# ====================================================================
def fig5_sensitivity_impact():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Panel A: Overestimate factor vs number of radial subdivisions ──
    ax = axes[0]
    n_range = np.arange(1, 11)
    overestimate = np.array([n * (1 - np.exp(-MU * H / n)) / (1 - np.exp(-MU * H)) for n in n_range])

    ax.bar(n_range, overestimate, color='steelblue', alpha=0.8, edgecolor='black')
    ax.axhline(1.0, color=COLOR_CORRECT, linewidth=2, linestyle='--', label='Correct (1.0×)')

    for i, (n, ov) in enumerate(zip(n_range, overestimate)):
        color = COLOR_BUG if ov > 1.05 else COLOR_CORRECT
        ax.text(n, ov + 0.05, f'{ov:.2f}×', ha='center', fontsize=9,
                fontweight='bold', color=color)

    # Highlight current and reference
    ax.get_children()[4].set_color(COLOR_BUG)    # n=5 (current)
    ax.get_children()[4].set_alpha(1.0)
    ax.get_children()[2].set_color('#e377c2')     # n=3 (reference)
    ax.get_children()[2].set_alpha(1.0)

    ax.annotate('Current\n(1,5)', xy=(5, overestimate[4]),
                xytext=(7, overestimate[4] + 0.4),
                fontsize=10, fontweight='bold', color=COLOR_BUG,
                arrowprops=dict(arrowstyle='->', color=COLOR_BUG, lw=1.5))
    ax.annotate('Reference\n(1,3)', xy=(3, overestimate[2]),
                xytext=(1, overestimate[2] + 0.5),
                fontsize=10, fontweight='bold', color='#e377c2',
                arrowprops=dict(arrowstyle='->', color='#e377c2', lw=1.5))

    ax.set_xlabel('Number of radial subdivisions (n_sub[1])', fontsize=11)
    ax.set_ylabel('PPDF overestimate factor', fontsize=11)
    ax.set_title(f'(A) Overestimate Factor vs Radial Subdivisions\n'
                 f'μ_crystal = {MU} mm⁻¹, H = {H} mm',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(n_range)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_ylim(0, max(overestimate) * 1.25)

    # ── Panel B: Estimated sensitivity before/after fix ──
    ax = axes[1]

    # From the user's sensitivity map: mean ~0.97, max ~1.35
    observed_mean = 0.97
    observed_max = 1.35
    overestimate_15 = 5 * (1 - np.exp(-MU * H / 5)) / (1 - np.exp(-MU * H))

    corrected_mean = observed_mean / overestimate_15
    corrected_max = observed_max / overestimate_15

    categories = ['Observed\n(1,5) BUG', 'Corrected\n(5,1) FIX']
    means = [observed_mean, corrected_mean]
    maxs = [observed_max, corrected_max]

    x_pos = np.arange(len(categories))
    bars_mean = ax.bar(x_pos - 0.18, means, 0.35,
                       color=[COLOR_BUG, COLOR_CORRECT], alpha=0.8,
                       edgecolor='black', label='Mean sensitivity')
    bars_max = ax.bar(x_pos + 0.18, maxs, 0.35,
                      color=[COLOR_BUG, COLOR_CORRECT], alpha=0.4,
                      edgecolor='black', hatch='//', label='Max sensitivity')

    for i, (m, mx) in enumerate(zip(means, maxs)):
        ax.text(i - 0.18, m + 0.02, f'{m:.3f}', ha='center', fontsize=10, fontweight='bold')
        ax.text(i + 0.18, mx + 0.02, f'{mx:.3f}', ha='center', fontsize=10)

    ax.axhline(1.0, color='black', linewidth=2, linestyle='--', alpha=0.7)
    ax.text(1.5, 1.03, 'Physical limit (100%)', fontsize=9, ha='right', alpha=0.7)

    ax.set_ylabel('Sensitivity (detection probability)', fontsize=11)
    ax.set_title('(B) Estimated Sensitivity Before & After Fix\n'
                 '(approximate, based on observed map values)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_ylim(0, 1.7)

    fig.suptitle('Sensitivity Impact of Crystal Subdivision Bug',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig('debug_fig5_sensitivity_impact.png', dpi=200, bbox_inches='tight')
    print("Saved: debug_fig5_sensitivity_impact.png")
    plt.close(fig)


# ====================================================================
# Run all figures
# ====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Generating diagnostic plots for subdivision sensitivity bug")
    print("=" * 60)

    fig1_subdivision_geometry()
    fig2_angular_span_overlap()
    fig3_self_attenuation()
    fig4_system_cross_section()
    fig5_sensitivity_impact()

    print("\n" + "=" * 60)
    print("All diagnostic plots saved.")
    print("=" * 60)
    print("\nSummary:")
    print(f"  CRYSTAL_SUBS = (1, 5) → 5 radial slices → {5*(1-np.exp(-MU*H/5))/(1-np.exp(-MU*H)):.2f}× overestimate")
    print(f"  CRYSTAL_SUBS = (5, 1) → 5 tangential strips → 1.00× (correct)")
    print(f"  CRYSTAL_SUBS = (1, 1) → no subdivision → 1.00× (correct)")