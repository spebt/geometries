#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def t8_offsets(a_mm=0.2, b_mm=0.2, phase_deg=0.0):
    """8 offsets on an ellipse: (a cos θ, b sin θ)."""
    phase = np.deg2rad(phase_deg)
    thetas = np.linspace(0, 2*np.pi, 8, endpoint=False) + phase
    dx = a_mm * np.cos(thetas)
    dy = b_mm * np.sin(thetas)
    return np.stack([dx, dy], axis=1)  # (8,2)

def t4_offsets(shift_mm=0.4):
    """4 offsets like your T4 phantom shifts (corner-style)."""
    s = shift_mm
    return np.array([
        (-s, -s),
        (+s, +s),
        (-s, +s),
        (+s, -s),
    ], dtype=float)

def plot_offsets(ax, offsets, title, circle_r=None, ellipse_ab=None):
    x, y = offsets[:,0], offsets[:,1]

    # Draw reference circle/ellipse if requested
    if circle_r is not None:
        t = np.linspace(0, 2*np.pi, 400)
        ax.plot(circle_r*np.cos(t), circle_r*np.sin(t), linestyle="--", linewidth=1)
    if ellipse_ab is not None:
        a, b = ellipse_ab
        t = np.linspace(0, 2*np.pi, 400)
        ax.plot(a*np.cos(t), b*np.sin(t), linestyle="--", linewidth=1)

    # Origin crosshair
    ax.axhline(0, linewidth=0.8, linestyle=":")
    ax.axvline(0, linewidth=0.8, linestyle=":")

    # Points + labels
    ax.scatter(x, y, s=60)
    for i, (xi, yi) in enumerate(offsets):
        ax.text(xi, yi, f"{i:02d}", fontsize=10, ha="left", va="bottom")

    # Connect in visiting order (nice for “trajectory” look)
    ax.plot(np.r_[x, x[0]], np.r_[y, y[0]], linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("dx (mm)")
    ax.set_ylabel("dy (mm)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.35)

def main():
    # ======= tweak these for your slide =======
    a_mm = 0.2
    b_mm = 0.2
    phase_deg = 0.0

    t4_shift_mm = 0.4
    # =========================================

    offs_t8 = t8_offsets(a_mm=a_mm, b_mm=b_mm, phase_deg=phase_deg)
    offs_t4 = t4_offsets(shift_mm=t4_shift_mm)

    # Determine axis limits that fit both nicely
    max_abs = max(np.abs(offs_t8).max(), np.abs(offs_t4).max()) * 1.25

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")

    plot_offsets(
        axes[0],
        offs_t4,
        title=f"T4 translations (±{t4_shift_mm:.1f} mm)",
        circle_r=None,
        ellipse_ab=None,
    )
    axes[0].set_xlim(-max_abs, max_abs)
    axes[0].set_ylim(-max_abs, max_abs)

    plot_offsets(
        axes[1],
        offs_t8,
        title=f"T8 translations (ellipse a={a_mm:.1f} mm, b={b_mm:.1f} mm)",
        circle_r=(a_mm if abs(a_mm-b_mm) < 1e-9 else None),
        ellipse_ab=(a_mm, b_mm),
    )
    axes[1].set_xlim(-max_abs, max_abs)
    axes[1].set_ylim(-max_abs, max_abs)

    out_png = f"t4_vs_t8_translations_a{a_mm}_b{b_mm}_t4shift{t4_shift_mm}.png"
    fig.suptitle("Sub-pixel translation patterns used for super-resolution sampling", fontsize=14)
    fig.savefig(out_png, dpi=300)
    print("Saved:", out_png)

if __name__ == "__main__":
    main()