#!/usr/bin/env python3
"""
MIC figure: MPXI has an optimum, not a direction.

The history this figure exists to settle. MPXI was originally MINIMIZED, on the
assumption that multiplexing costs image quality. Measured in physical units it
correlates +0.55 with CNR, so we were optimizing away from the thing we wanted.
RY predicted that from the splitting physics before it was measured, and the
sign was reversed in Aug 2026.

But "maximize" then overshot. A Spearman coefficient measures MONOTONE
association, and this relationship turns over: CNR climbs with MPXI to about
2.0 and falls away above it. A healthy positive rho was fully consistent with
the wrong instruction, and binning is what exposed the shape.

The figure shows both: every design as a point, and the binned mean as the
trend, so the turnover is visible rather than asserted.

Usage:
  python make_mpxi_optimum_figure.py --data deck_figures/mpxi_cnr_data.csv \
      --out deck_figures/mpxi_optimum.png
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated with the dataviz palette validator (light surface, categorical):
# all six checks pass, CVD dE 24.7 protan.
C_POINTS, C_TREND = "#2a78d6", "#eb6834"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

MPXI = "mpxi_windowed_active_mean"
CNR = "cnr_sector_mean"


def main():
    ap = argparse.ArgumentParser(description="Build the MPXI-optimum figure")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="mpxi_optimum.png")
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    df = pd.read_csv(args.data).dropna(subset=[MPXI, CNR])
    # Quantile bins keep the counts comparable. Equal-width bins would put most
    # designs in the first one, since MPXI is right-skewed.
    df["bin"] = pd.qcut(df[MPXI], q=args.bins, duplicates="drop")
    g = df.groupby("bin", observed=True).agg(
        x=(MPXI, "mean"), y=(CNR, "mean"), sd=(CNR, "std"), n=(CNR, "size"))
    peak = g["y"].idxmax()
    peak_x, peak_y = g.loc[peak, "x"], g.loc[peak, "y"]
    # The top two bins are close, so the honest claim is a REGION rather than a
    # point. Quoting a single peak the trend does not clearly resolve would
    # overstate what 8 bins can say.
    top2 = g.nlargest(2, "y")["x"].sort_values()
    lo_x, hi_x = float(top2.iloc[0]), float(top2.iloc[-1])

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.scatter(df[MPXI], df[CNR], s=26, color=C_POINTS, alpha=0.35,
               edgecolors="none", zorder=2, label=f"designs (n={len(df)})")
    ax.errorbar(g["x"], g["y"], yerr=g["sd"], color=C_TREND, lw=2.4,
                marker="o", ms=8, capsize=4, zorder=4,
                markeredgecolor="white", markeredgewidth=1.4,
                label="binned mean CNR")

    # One annotation carrying the claim, rather than a label on every bin.
    ax.annotate(f"peak {lo_x:.1f} to {hi_x:.1f}",
                xy=(peak_x, peak_y), xytext=(peak_x - 0.05, peak_y + 0.55),
                fontsize=11, fontweight="bold", color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=1.4,
                                shrinkA=2, shrinkB=4))

    ax.set_xlabel("MPXI (windowed, active detectors)", fontsize=11, color=INK)
    ax.set_ylabel("CNR sector mean", fontsize=11, color=INK)
    ax.set_title("MPXI has an optimum, not a direction\n"
                 f"CNR rises with multiplexing to roughly {lo_x:.1f} to {hi_x:.1f}, "
                 f"then falls away",
                 fontsize=12.5, color=INK, pad=14)

    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=10)

    leg = ax.legend(frameon=False, fontsize=10.5, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK)

    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, facecolor="white")
    print(f"wrote {args.out}")
    print(f"  {len(df)} designs, {len(g)} bins, peak bin mean CNR "
          f"{peak_y:.2f} at MPXI {peak_x:.2f}")
    print(f"  highest-MPXI bin is {peak_y - g['y'].iloc[-1]:.2f} below the peak")


if __name__ == "__main__":
    main()
