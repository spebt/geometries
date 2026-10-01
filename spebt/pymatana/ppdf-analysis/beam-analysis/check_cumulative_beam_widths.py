# #!/usr/bin/env python3
# """
# check_cumulative_beam_widths.py
# --------------------------------
# Reads beam properties from ALL rotation files in a directory and generates a single,
# cumulative histogram of beam FWHM values.

# * **Figure 1:** Cumulative histogram of all valid beam FWHM values (mm) across
#     all configurations, with mean/median markers.
# * Prints the total number of beams found and the count of those within the
#     2–5 mm “good-width” window.

# Run e.g.
# ```bash
# python check_cumulative_beam_widths.py \
#        --props-dir /vscratch/grp-rutaoyao/Omer/spebt/data/sai_10mm \
#        --out       /vscratch/grp-rutaoyao/Omer/spebt/plots/sai_10mm
# ```
# """
# import argparse
# import os
# import glob
# import h5py
# import numpy as np
# import matplotlib.pyplot as plt
# import torch

# # ------------------------------------------------------------------ #

# def read_fwhm_column(h5_path: str) -> np.ndarray:
#     """
#     Return the 'FWHM (mm)' column from a beam_properties HDF5 file.
    
#     Args:
#         h5_path (str): Path to the HDF5 file.

#     Returns:
#         np.ndarray: An array containing the FWHM values.
#     """
#     with h5py.File(h5_path, "r") as f:
#         # Handle case where header might be bytes and needs decoding
#         header_raw = f["beam_properties"].attrs["Header"]
#         header = [h.decode('utf-8') if isinstance(h, bytes) else h for h in header_raw]
        
#         if "FWHM (mm)" not in header:
#             raise RuntimeError(f"Column 'FWHM (mm)' not found in {h5_path}")
            
#         col_index = header.index("FWHM (mm)")
#         # Use numpy directly which is sufficient for this task
#         fwhm_data = f["beam_properties"][:, col_index]
        
#     return fwhm_data

# # ------------------------------------------------------------------ #

# def main(args):
#     """
#     Main function to find files, aggregate data, and generate the plot.
#     """
#     # -- Find all property files in the input directory -------------
#     search_pattern = os.path.join(args.props_dir, "beams_properties_*.hdf5")
#     property_files = sorted(glob.glob(search_pattern))

#     if not property_files:
#         print(f"Error: No 'beams_properties_*.hdf5' files found in '{args.props_dir}'")
#         return

#     print(f"Found {len(property_files)} beam property files to process.")

#     # -- Read FWHM from all files and aggregate --------------------
#     all_fwhm_values = []
#     total_beams_read = 0

#     for file_path in property_files:
#         print(f"  - Reading properties from: {os.path.basename(file_path)}")
#         try:
#             fwhm_raw = read_fwhm_column(file_path)
#             total_beams_read += len(fwhm_raw)
            
#             # IMPORTANT: Filter out NaN values from FWHM for this file.
#             # This handles cases where FWHM calculation may have failed.
#             valid_fwhm_mask = ~np.isnan(fwhm_raw)
#             # valid_fwhm_mask = (~np.isnan(fwhm_raw)) & (fwhm_raw <= 10.0) & (fwhm_raw > 0.0)
#             valid_fwhm = fwhm_raw[valid_fwhm_mask]
            
#             all_fwhm_values.extend(valid_fwhm)
#         except Exception as e:
#             print(f"    Could not process file {os.path.basename(file_path)}: {e}")

#     # Consolidate into a single NumPy array for efficient analysis
#     cumulative_fwhm = np.array(all_fwhm_values)
    
#     print("-" * 50)
#     print(f"Total beams read across all files: {total_beams_read}")
#     print(f"Found {len(cumulative_fwhm)} total beams with a valid FWHM value.")

#     # -- Analyze cumulative data ------------------------------------
#     # if len(cumulative_fwhm) == 0:
#     #     print("No valid FWHM values found across all files. Cannot generate plot.")
#     #     return
        
#     # good_width_mask = (cumulative_fwhm >= 2.0) & (cumulative_fwhm <= 5.0)
#     # n_good_beams = np.sum(good_width_mask)
#     # print(f"Total beams in the 2–5 mm 'good-width' window: {n_good_beams}")

#     # # -------------- Figure 1 – Cumulative FWHM histogram -----------
#     # os.makedirs(args.out, exist_ok=True)
#     # fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    
#     # # Plot the main histogram
#     # ax.hist(cumulative_fwhm, bins=400, color="#4c72b0", alpha=0.9, label=f'{len(cumulative_fwhm)} total beams')

#         # -- Analyze cumulative data ------------------------------------
#     if len(cumulative_fwhm) == 0:
#         print("No valid FWHM values found across all files. Cannot generate plot.")
#         return

#     # 1) Paper's "good-width" window (for comparison with the original work)
#     paper_good_mask = (cumulative_fwhm >= 2.0) & (cumulative_fwhm <= 5.0)
#     n_paper_good = int(np.sum(paper_good_mask))
#     print(f"Total beams in the 2–5 mm 'good-width' window (paper range): {n_paper_good}")

#     # 2) Design-appropriate window for *your* sub-mm beams
#     #    (tweak 0.1 and 0.6 as you like after looking at the histogram)
#     design_plot_mask = (cumulative_fwhm > 0.0) & (cumulative_fwhm <= 2.0)
#     fwhm_for_plot = cumulative_fwhm[design_plot_mask]
#     print(f"Beams used for plotting in 0–2 mm window: {len(fwhm_for_plot)}")

#     # -------------- Figure 1 – Cumulative FWHM histogram -----------
#     os.makedirs(args.out, exist_ok=True)
#     fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")

#     # Plot the main histogram (using design-appropriate range)
#     ax.hist(
#         fwhm_for_plot,
#         bins=200,
#         range=(0.0, 2.0),
#         color="#4c72b0",
#         alpha=0.9,
#         label=f'{len(fwhm_for_plot)} beams (0–2 mm)'
#     )

#     # Stats from the same subset
#     mean_v = fwhm_for_plot.mean()
#     med_v = np.median(fwhm_for_plot)
#     max_v = fwhm_for_plot.max()
#     print(f"Max FWHM in 0–2 mm window: {max_v}")
    
    
#     # Calculate and plot mean/median lines for reference
#     mean_v = cumulative_fwhm.mean()
#     med_v = np.median(cumulative_fwhm)
#     max_v = np.max(cumulative_fwhm)
#     print(f"Max FWHM - {max_v}")
    
#     ax.axvline(mean_v, color="#d62728", ls="--", lw=2, label=f"Mean: {mean_v:.2f} mm")
#     ax.axvline(med_v, color="#2ca02c", ls=":", lw=2, label=f"Median: {med_v:.2f} mm")
    
#     # --- Final plot styling ---
#     ax.legend()
#     ax.set_xlabel("Beam FWHM (mm)")
#     ax.set_ylabel("Number of Beams")
#     ax.set_title(f"Cumulative Beam Width Distribution ({len(property_files)} Rotations)")
#     # ax.set_xlim([0,4.85])
#     ax.grid(axis='y', linestyle='--', alpha=0.7)
    
#     # --- Save the figure ---
#     out_path = os.path.join(args.out, "cumulative_beam_width_histogram.png")
#     fig.savefig(out_path, dpi=300)
#     plt.close(fig)
#     print(f"\nSaved cumulative plot → {out_path}")

# # ------------------------------------------------------------------ #

# if __name__ == "__main__":
#     # --- Argument Parsing ---
#     ap = argparse.ArgumentParser(description="Generate a cumulative beam-width histogram from multiple property files.")
#     ap.add_argument("--props-dir", required=True, help="Directory containing beam_properties_*.hdf5 files.")
#     ap.add_argument("--out", required=True, help="Output directory for the PNG plot.")
#     args = ap.parse_args()
#     main(args)

# beam width t4

#!/usr/bin/env python3
"""
check_cumulative_beam_widths.py
--------------------------------
Reads beam properties from multiple beam_properties HDF5 files and generates
a cumulative histogram of beam FWHM values.

Supports:
  - Non-T4 files: beams_properties_configuration_XX.hdf5
  - T4 files:     beams_properties_configuration_XX_t4_YY.hdf5 

Example:
  # T4, layout 0 only
  python check_cumulative_beam_widths.py --props-dir /path/to/data --out /path/to/plots --t4 --layout-idx 0

  # T4, all layouts found in directory
  python check_cumulative_beam_widths.py --props-dir /vscratch/grp-rutaoyao/Omer/spebt/data/sai_10mm --out /vscratch/grp-rutaoyao/Omer/spebt/plots/sai_10mm --t4

  # Non-T4, all layouts
  python check_cumulative_beam_widths.py --props-dir /path/to/data --out /path/to/plots
"""
import argparse
import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt


def read_fwhm_column(h5_path: str) -> np.ndarray:
    """Return the 'FWHM (mm)' column from a beam_properties HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        header_raw = f["beam_properties"].attrs["Header"]
        header = [h.decode("utf-8") if isinstance(h, bytes) else h for h in header_raw]

        if "FWHM (mm)" not in header:
            raise RuntimeError(f"Column 'FWHM (mm)' not found in {h5_path}")

        col_index = header.index("FWHM (mm)")
        return f["beam_properties"][:, col_index]


def find_property_files(props_dir: str, use_t4: bool, layout_idx: int | None) -> list[str]:
    """
    Find beam property files in props_dir.
    - If use_t4: grabs ..._t4_*.hdf5
    - else: grabs beams_properties_configuration_*.hdf5 but excludes *_t4_*.hdf5
    """
    if use_t4:
        if layout_idx is None:
            pattern = os.path.join(props_dir, "beams_properties_configuration_*_t4_*.hdf5")
        else:
            pattern = os.path.join(props_dir, f"beams_properties_configuration_{layout_idx:02d}_t4_*.hdf5")
        files = sorted(glob.glob(pattern))
        return files

    # non-T4
    if layout_idx is None:
        pattern = os.path.join(props_dir, "beams_properties_configuration_*.hdf5")
        files = sorted(glob.glob(pattern))
        # exclude t4 variants
        files = [p for p in files if "_t4_" not in os.path.basename(p)]
        return files

    # single-layout non-t4 file
    p = os.path.join(props_dir, f"beams_properties_configuration_{layout_idx:02d}.hdf5")
    return [p] if os.path.exists(p) else []


def main(args):
    property_files = find_property_files(args.props_dir, args.t4, args.layout_idx)

    if not property_files:
        mode = "T4" if args.t4 else "non-T4"
        print(f"Error: No {mode} beam property files found in '{args.props_dir}' "
              f"for layout_idx={args.layout_idx}")
        return

    print(f"Found {len(property_files)} beam property files to process:")
    for fp in property_files:
        print("  -", os.path.basename(fp))

    all_fwhm_values = []
    total_beams_read = 0

    for file_path in property_files:
        try:
            fwhm_raw = read_fwhm_column(file_path)
            total_beams_read += len(fwhm_raw)

            valid = fwhm_raw[~np.isnan(fwhm_raw)]
            all_fwhm_values.append(valid.astype(np.float32))
        except Exception as e:
            print(f"[WARN] Could not process {os.path.basename(file_path)}: {e}")

    if not all_fwhm_values:
        print("No valid FWHM arrays loaded. Exiting.")
        return

    cumulative_fwhm = np.concatenate(all_fwhm_values, axis=0)

    print("-" * 60)
    print(f"Total beam rows read across files (including NaNs): {total_beams_read}")
    print(f"Total valid FWHM values (non-NaN): {len(cumulative_fwhm)}")

    # Paper reference window
    paper_good_mask = (cumulative_fwhm >= 2.0) & (cumulative_fwhm <= 5.0)
    print(f"Beams in paper 'good-width' window (2–5 mm): {int(paper_good_mask.sum())}")

    # Plot window (default: 0–2 mm for your sub-mm beams)
    lo, hi = args.plot_min, args.plot_max
    plot_mask = (cumulative_fwhm > lo) & (cumulative_fwhm <= hi)
    fwhm_for_plot = cumulative_fwhm[plot_mask]

    if fwhm_for_plot.size == 0:
        print(f"No beams found in plot window ({lo}–{hi} mm). Try widening --plot-min/--plot-max.")
        return

    print(f"Beams used for plotting in {lo}–{hi} mm window: {len(fwhm_for_plot)}")
    print(f"Plot-window mean:   {fwhm_for_plot.mean():.4f} mm")
    print(f"Plot-window median: {np.median(fwhm_for_plot):.4f} mm")
    print(f"Plot-window max:    {fwhm_for_plot.max():.4f} mm")
    print(f"Global max (all valid): {cumulative_fwhm.max():.4f} mm")

    os.makedirs(args.out, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")

    ax.hist(
        fwhm_for_plot,
        bins=args.bins,
        range=(lo, hi),
        alpha=0.9,
        label=f"{len(fwhm_for_plot)} beams ({lo}–{hi} mm)"
    )

    # mean/median lines for the SAME plotted subset
    ax.axvline(fwhm_for_plot.mean(), ls="--", lw=2, label=f"Mean: {fwhm_for_plot.mean():.3f} mm")
    ax.axvline(np.median(fwhm_for_plot), ls=":", lw=2, label=f"Median: {np.median(fwhm_for_plot):.3f} mm")

    title_mode = "T4" if args.t4 else "non-T4"
    ax.set_title(f"Cumulative Beam Width Distribution ({title_mode}) | files={len(property_files)}")
    ax.set_xlabel("Beam FWHM (mm)")
    ax.set_ylabel("Number of Beams")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend()

    out_path = os.path.join(
        args.out,
        f"cumulative_beam_width_histogram_{title_mode.lower()}.png"
    )
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"\nSaved cumulative plot → {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--props-dir", required=True, help="Directory containing beams_properties_configuration*.hdf5")
    ap.add_argument("--out", required=True, help="Output directory for plots")
    ap.add_argument("--t4", action="store_true", help="Use T4 files beams_properties_configuration_XX_t4_YY.hdf5")
    ap.add_argument("--layout-idx", type=int, default=None, help="Restrict to one layout index (e.g., 0)")
    ap.add_argument("--plot-min", type=float, default=0.0, help="Histogram plot min (mm)")
    ap.add_argument("--plot-max", type=float, default=2.0, help="Histogram plot max (mm)")
    ap.add_argument("--bins", type=int, default=200, help="Number of histogram bins")
    args = ap.parse_args()
    main(args)