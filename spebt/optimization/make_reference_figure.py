#!/usr/bin/env python3
"""
The MIC headline figure: our optimized design against the published TMI reference.

Both sides were measured identically: 5 independent Poisson realisations of the
same 150-iteration ML-EM reconstruction, sector-mean CNR on the hot-rod phantom.
The reference configuration (0.4 mm apertures, 180 apertures, 480/720 detectors,
d2 390 mm, d3 520 mm) had never been run through our pipeline before Aug 2026,
which is why this comparison did not exist until now.

Values are recorded here rather than recomputed because they come from a
5-seed repeat study whose raw outputs live on the cluster. Provenance for each
is in the comment beside it; regenerate with analyze_cnr_repeats.py.

Usage:
  python make_reference_figure.py --out reference_comparison.png
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Rod radii of the six phantom sections, smallest to largest (mm).
ROD_RADII = [0.100, 0.125, 0.150, 0.175, 0.200, 0.225]

# analyze_cnr_repeats.py, tmi_reference_000, n=5 seeds
REF_MEAN = [2.057, 2.247, 2.592, 4.309, 4.769, 5.559]
REF_STD  = [0.114, 0.079, 0.108, 0.181, 0.130, 0.161]
REF_OVERALL = (3.589, 0.055)

# analyze_cnr_repeats.py, mobo_0296, n=5 seeds
OURS_MEAN = [2.498, 3.615, 4.028, 5.717, 6.362, 6.113]
OURS_STD  = [0.248, 0.267, 0.239, 0.134, 0.104, 0.097]
OURS_OVERALL = (4.722, 0.069)

# Validated with the dataviz palette validator (light surface, categorical):
# CVD separation dE 24.7 protan / 32.7 tritan, normal-vision 33.6, all checks pass.
C_REF, C_OURS = "#eb6834", "#2a78d6"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"


def main():
    ap = argparse.ArgumentParser(description="Build the reference-comparison figure")
    ap.add_argument("--out", default="reference_comparison.png")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    gain = [100 * (o / r - 1) for o, r in zip(OURS_MEAN, REF_MEAN)]
    overall_gain = 100 * (OURS_OVERALL[0] / REF_OVERALL[0] - 1)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(len(ROD_RADII))
    # A 2px-equivalent gap between adjacent bars: width 0.38 inside a 0.5 slot.
    w = 0.38
    ax.bar(x - w / 2, REF_MEAN, w, yerr=REF_STD, capsize=3,
           color=C_REF, label="TMI reference", zorder=3,
           error_kw=dict(ecolor=INK_MUTED, lw=1.2, capthick=1.2))
    ax.bar(x + w / 2, OURS_MEAN, w, yerr=OURS_STD, capsize=3,
           color=C_OURS, label="MOBO optimized", zorder=3,
           error_kw=dict(ecolor=INK_MUTED, lw=1.2, capthick=1.2))

    # Direct labels carry the story; a number on every bar would not.
    for xi, (o, s, g) in enumerate(zip(OURS_MEAN, OURS_STD, gain)):
        ax.text(xi, o + s + 0.22, f"+{g:.0f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=INK, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:.3f}" for r in ROD_RADII])
    ax.set_xlabel("Hot-rod radius (mm)", fontsize=11, color=INK)
    # Each bar is ONE rod section, not the sector mean. The sector mean appears
    # only in the title, where it summarises all six.
    ax.set_ylabel("CNR (mean of 5 runs)", fontsize=11, color=INK)
    ax.set_title(
        f"Optimized design beats the published reference by "
        f"{overall_gain:.0f}% overall\n"
        f"sector mean {OURS_OVERALL[0]:.2f} $\\pm$ {OURS_OVERALL[1]:.2f} vs "
        f"{REF_OVERALL[0]:.2f} $\\pm$ {REF_OVERALL[1]:.2f}, largest gains at "
        f"0.125 to 0.150 mm",
        fontsize=12.5, color=INK, pad=14)

    # Recessive grid and axes; no top/right spines.
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=10)
    ax.set_ylim(0, max(OURS_MEAN) + 1.15)

    leg = ax.legend(frameon=False, fontsize=11, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK)

    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, facecolor="white")
    print(f"wrote {args.out}")
    print(f"  overall gain {overall_gain:.1f}%")
    print("  per-rod gain " + ", ".join(f"{g:.0f}%" for g in gain))


if __name__ == "__main__":
    main()
