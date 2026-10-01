#!/usr/bin/env python3
"""
Visualize the combined system matrix (PPDFs) as a heatmap and a 2D sensitivity map.

Usage:
  python plot_system_matrix.py \
      /vscratch/grp-rutaoyao/Omer/spebt/data/sai_10mm/position_000_ppdfs.hdf5 \
      --nx 200 --ny 200 \
      --outdir /vscratch/grp-rutaoyao/Omer/spebt/plots/sai_10mm

If --nx/--ny are omitted, only the matrix heatmap is saved.
"""

import os
import sys
import argparse

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot combined system matrix (PPDFs).")
    parser.add_argument("h5_file", type=str,
                        help="Path to position_XXX_ppdfs.hdf5 (with dataset 'ppdfs').")
    parser.add_argument("--nx", type=int, default=None,
                        help="Number of pixels in x (for reshaping sensitivity map).")
    parser.add_argument("--ny", type=int, default=None,
                        help="Number of pixels in y (for reshaping sensitivity map).")
    parser.add_argument("--outdir", type=str, default=".",
                        help="Directory to save output plots.")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --- 1. Load PPDF matrix ---
    if not os.path.exists(args.h5_file):
        print(f"ERROR: File not found: {args.h5_file}")
        sys.exit(1)

    print(f"[INFO] Loading system matrix from: {args.h5_file}")
    with h5py.File(args.h5_file, "r") as f:
        if "ppdfs" not in f:
            print("ERROR: Dataset 'ppdfs' not found in HDF5 file.")
            print(f"Available datasets: {list(f.keys())}")
            sys.exit(1)
        ppdfs = f["ppdfs"][...]  # shape: (n_det, n_pix)

    A = np.nan_to_num(ppdfs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    n_det, n_pix = A.shape
    print(f"[INFO] Loaded system matrix A with shape: {A.shape} (detectors x pixels)")

    # --- 2. Plot log-scaled heatmap of A ---
    eps = 1e-12
    A_log = np.log10(A + eps)

    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    im = ax.imshow(
        A_log,
        aspect="auto",
        cmap="viridis",
        origin="lower",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10(PPDF value + 1e-12)")

    ax.set_xlabel("Pixel index (0 .. n_pix-1)")
    ax.set_ylabel("Detector index (0 .. n_det-1)")
    ax.set_title("Combined system matrix (all detectors vs all pixels)")

    out_heatmap = os.path.join(args.outdir, "system_matrix_heatmap_log10.png")
    fig.savefig(out_heatmap, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved system matrix heatmap to: {out_heatmap}")

    # --- 3. Optional: 2D sensitivity map = sum over detectors ---
    if args.nx is not None and args.ny is not None:
        nx, ny = args.nx, args.ny
        if nx * ny != n_pix:
            print(
                f"[WARN] nx * ny = {nx * ny} does not match n_pix = {n_pix}. "
                "Skipping 2D sensitivity map."
            )
            return

        # Sum over detectors -> per-pixel sensitivity
        sens_1d = A.sum(axis=0)  # shape: (n_pix,)
        sens_2d = sens_1d.reshape(ny, nx)  # note: (rows, cols) = (y,x)

        fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
        im2 = ax.imshow(
            sens_2d.T,  # transpose to keep x horizontal, y vertical
            origin="lower",
            cmap="magma"
        )
        cbar2 = fig.colorbar(im2, ax=ax)
        cbar2.set_label("Sensitivity (sum over detectors)")

        ax.set_xlabel("Pixel x index")
        ax.set_ylabel("Pixel y index")
        ax.set_title("2D sensitivity map (Σ over detectors)")

        out_sens = os.path.join(args.outdir, "system_matrix_sensitivity_map.png")
        fig.savefig(out_sens, dpi=200)
        plt.close(fig)
        print(f"[INFO] Saved sensitivity map to: {out_sens}")
    else:
        print("[INFO] nx, ny not provided; skipping 2D sensitivity map.")


if __name__ == "__main__":
    main()