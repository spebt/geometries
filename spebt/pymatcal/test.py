# import numpy as np
# import h5py
# import matplotlib.pyplot as plt
# import os

# def load_ppdfs_for_layout(ppdf_dir: str, layout_idx: int, n_xtals_to_load: int = -1):
#     """Loads PPDFs from an HDF5 file for a specific layout."""
#     ppdf_filename = os.path.join(ppdf_dir, f"position_{layout_idx:03d}_ppdfs.hdf5")
#     if not os.path.exists(ppdf_filename):
#         print(f"PPDF file {ppdf_filename} does not exist.")
#         return None
    
#     with h5py.File(ppdf_filename, "r") as f:
#         # Selects all crystals if n_xtals_to_load is -1, otherwise slices the array
#         ppdfs_data = f["ppdfs"][:n_xtals_to_load] if n_xtals_to_load != -1 else f["ppdfs"][:]
#     return ppdfs_data

# def create_and_plot_sensitivity_histogram(sensitivity_map: np.ndarray, output_dir: str, file_suffix: str, title_suffix: str):
#     """Generates and displays a histogram for a given sensitivity map."""
#     # --- 1. Prepare Data: Filter for values within the FOV ---
#     fov_values = sensitivity_map[sensitivity_map > 0].flatten()
#     if fov_values.size == 0:
#         print("Warning: The sensitivity map contains no non-zero values.")
#         return

#     # --- 2. Create and Plot the Histogram ---
#     plt.figure(figsize=(10, 6))
#     plt.hist(fov_values, bins=100, color='royalblue', alpha=0.75)
#     plt.title(f'Histogram of Sensitivity Values - {title_suffix}')
#     plt.xlabel('Sensitivity')
#     plt.ylabel('Number of Pixels (Frequency)')
#     plt.grid(True, linestyle='--', alpha=0.6)
    
#     mean_sensitivity = np.mean(fov_values)
#     plt.axvline(mean_sensitivity, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_sensitivity:.4f}')
#     plt.legend()
#     plt.tight_layout()

#     # --- 3. Save the Histogram Plot ---
#     hist_filename = os.path.join(output_dir, f"sensitivity_histogram_{file_suffix}.png")
#     plt.savefig(hist_filename)
#     print(f"Saved sensitivity histogram to: {hist_filename}")
#     plt.close()


# if __name__ == "__main__":
#     # --- Configuration ---
#     ppdf_files_base_dir = "/vscratch/grp-rutaoyao/Omer/spebt/data/sai_10mm/"
#     num_layouts_to_load = 1
#     layout_indices = list(range(num_layouts_to_load))

#     FOV_PIXELS_X, FOV_PIXELS_Y = 200, 200
#     MM_PER_PIXEL_X, MM_PER_PIXEL_Y = 0.05, 0.05
#     num_crystals_to_sum = -1
#     # --- End Configuration ---

#     if not os.path.isdir(ppdf_files_base_dir):
#         print(f"Error: PPDF directory '{ppdf_files_base_dir}' not found.")
#         exit()

#     # --- Loop through layouts, load, and aggregate ---
#     print(f"Loading and aggregating PPDFs for layouts: {layout_indices}...")
#     aggregated_ppdfs = None
#     successful_loads = 0 # NEW: Counter for successfully loaded layouts

#     for idx in layout_indices:
#         print(f"  - Loading layout {idx:03d}...")
#         ppdfs_for_layout = load_ppdfs_for_layout(ppdf_files_base_dir, idx, num_crystals_to_sum)
        
#         if ppdfs_for_layout is None:
#             print(f"Warning: Could not load data for layout {idx}. Skipping.")
#             continue
            
#         if aggregated_ppdfs is None:
#             aggregated_ppdfs = ppdfs_for_layout.astype(np.float32)
#         else:
#             aggregated_ppdfs += ppdfs_for_layout
        
#         successful_loads += 1 # NEW: Increment the counter on a successful load

#     if aggregated_ppdfs is None:
#         print("Error: No PPDF data was successfully loaded. Exiting.")
#         exit()
    
#     print(f"\nAggregation complete. Successfully loaded {successful_loads} layouts.")

#     # --- Generate and NORMALIZE the sensitivity map ---
#     if successful_loads > 0:
#         # UPDATED: Divide by the number of successful loads to get the average
#         sensitivity_map_1d = np.sum(aggregated_ppdfs, axis=0) / successful_loads
#     else:
#         # This case is already handled by the exit above, but it's good practice
#         sensitivity_map_1d = np.zeros(FOV_PIXELS_X * FOV_PIXELS_Y)

#     sensitivity_map_2d = sensitivity_map_1d.reshape((FOV_PIXELS_Y, FOV_PIXELS_X))
            
#     # Determine plot extent
#     extent = [-(MM_PER_PIXEL_X * FOV_PIXELS_X / 2), (MM_PER_PIXEL_X * FOV_PIXELS_X / 2),
#               -(MM_PER_PIXEL_Y * FOV_PIXELS_Y / 2), (MM_PER_PIXEL_Y * FOV_PIXELS_Y / 2)]

#     # --- Create descriptive names for aggregated output files and titles ---
#     layouts_str_fname = f"aggregated_{successful_loads}layouts_normalized"
#     layouts_str_title = f"Normalized Sensitivity Map ({successful_loads} Layouts)"

#     # Plot and Save the Aggregated and Normalized Sensitivity Map Image
#     plt.figure(figsize=(8, 7))
#     plt.imshow(sensitivity_map_2d, cmap='viridis', origin='lower', extent=extent)
#     # UPDATED: Changed colorbar label to reflect normalization
#     plt.colorbar(label='Average Sensitivity (Normalized)')
#     plt.title(layouts_str_title)
#     plt.xlabel('X (mm)'); plt.ylabel('Y (mm)')
#     plt.axhline(0, color='white', linestyle=':', lw=0.5); plt.axvline(0, color='white', linestyle=':', lw=0.5)
#     plt.tight_layout()
#     map_filename = os.path.join(ppdf_files_base_dir, f"sensitivity_map_{layouts_str_fname}.png")
#     plt.savefig(map_filename)
#     print(f"\nSaved normalized sensitivity map to: {map_filename}")
#     plt.close()
    
#     # NEW: Also create a histogram for the normalized data
#     create_and_plot_sensitivity_histogram(sensitivity_map_2d, ppdf_files_base_dir, layouts_str_fname, layouts_str_title)

# #test_t8

#!/usr/bin/env python3
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os

def load_ppdfs_for_layout_t8(ppdf_dir: str, layout_idx: int, pose_idxs, n_xtals_to_load: int = -1):
    """
    Loads and aggregates T8 PPDFs for a given layout across poses.
    Returns:
      aggregated_ppdfs: (n_crystals, n_pixels) float32  [sum over poses]
      n_loaded_poses: int
    """
    aggregated_ppdfs = None
    n_loaded = 0

    for pose in pose_idxs:
        ppdf_filename = os.path.join(ppdf_dir, f"position_{layout_idx:03d}_ppdfs_t8_{pose:02d}.hdf5")
        if not os.path.exists(ppdf_filename):
            print(f"[MISS] {ppdf_filename}")
            continue

        with h5py.File(ppdf_filename, "r") as f:
            ppdfs_data = f["ppdfs"][:n_xtals_to_load] if n_xtals_to_load != -1 else f["ppdfs"][:]
            ppdfs_data = ppdfs_data.astype(np.float32)

        if aggregated_ppdfs is None:
            aggregated_ppdfs = ppdfs_data
        else:
            aggregated_ppdfs += ppdfs_data

        n_loaded += 1

    return aggregated_ppdfs, n_loaded


def create_and_plot_sensitivity_histogram(sensitivity_map: np.ndarray, output_dir: str, file_suffix: str, title_suffix: str):
    """Generates and saves a histogram for a given sensitivity map."""
    fov_values = sensitivity_map[sensitivity_map > 0].flatten()
    if fov_values.size == 0:
        print("Warning: The sensitivity map contains no non-zero values.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(fov_values, bins=100, color='royalblue', alpha=0.75)
    plt.title(f'Histogram of Sensitivity Values - {title_suffix}')
    plt.xlabel('Sensitivity')
    plt.ylabel('Number of Pixels (Frequency)')
    plt.grid(True, linestyle='--', alpha=0.6)

    mean_sensitivity = np.mean(fov_values)
    plt.axvline(mean_sensitivity, color='red', linestyle='dashed', linewidth=2,
                label=f'Mean: {mean_sensitivity:.6f}')
    plt.legend()
    plt.tight_layout()

    hist_filename = os.path.join(output_dir, f"sensitivity_histogram_{file_suffix}.png")
    plt.savefig(hist_filename, dpi=300)
    print(f"Saved sensitivity histogram to: {hist_filename}")
    plt.close()


if __name__ == "__main__":
    # --- Configuration (match your legacy style) ---
    ppdf_files_base_dir = "/vscratch/grp-rutaoyao/Omer/spebt/data/sai_10mm/"
    layout_indices = [0, 1]               # <-- layouts 000 and 001
    pose_indices = list(range(8))         # <-- T8 poses 00..07

    FOV_PIXELS_X, FOV_PIXELS_Y = 200, 200
    MM_PER_PIXEL_X, MM_PER_PIXEL_Y = 0.05, 0.05
    num_crystals_to_sum = -1              # -1 loads all crystals

    # If True: average over poses inside each layout before averaging over layouts
    # If False: sum over poses (keeps scale larger by factor ~#poses), then average only over layouts
    AVERAGE_OVER_POSES = True
    # --- End Configuration ---

    if not os.path.isdir(ppdf_files_base_dir):
        print(f"Error: PPDF directory '{ppdf_files_base_dir}' not found.")
        exit()

    print(f"Loading and aggregating T8 PPDFs for layouts: {layout_indices} (poses {pose_indices}) ...")

    aggregated_ppdfs = None
    successful_loads = 0

    for layout_idx in layout_indices:
        print(f"\n  - Loading layout {layout_idx:03d} ...")
        ppdfs_layout_sum, n_loaded_poses = load_ppdfs_for_layout_t8(
            ppdf_files_base_dir, layout_idx, pose_indices, num_crystals_to_sum
        )

        if ppdfs_layout_sum is None or n_loaded_poses == 0:
            print(f"Warning: Could not load any poses for layout {layout_idx:03d}. Skipping.")
            continue

        # Optionally average over poses (recommended if you want “per acquisition position” mean)
        if AVERAGE_OVER_POSES:
            ppdfs_layout = ppdfs_layout_sum / float(n_loaded_poses)
        else:
            ppdfs_layout = ppdfs_layout_sum

        # Aggregate across layouts exactly like your legacy script
        if aggregated_ppdfs is None:
            aggregated_ppdfs = ppdfs_layout.astype(np.float32)
        else:
            aggregated_ppdfs += ppdfs_layout.astype(np.float32)

        successful_loads += 1
        print(f"    Loaded poses: {n_loaded_poses}/8")

    if aggregated_ppdfs is None or successful_loads == 0:
        print("Error: No PPDF data was successfully loaded. Exiting.")
        exit()

    print(f"\nAggregation complete. Successfully loaded {successful_loads} layouts.")

    # --- Generate sensitivity map EXACTLY like your legacy code ---
    sensitivity_map_1d = np.sum(aggregated_ppdfs, axis=0) / successful_loads
    sensitivity_map_2d = sensitivity_map_1d.reshape((FOV_PIXELS_Y, FOV_PIXELS_X))

    # Determine plot extent
    extent = [
        -(MM_PER_PIXEL_X * FOV_PIXELS_X / 2), (MM_PER_PIXEL_X * FOV_PIXELS_X / 2),
        -(MM_PER_PIXEL_Y * FOV_PIXELS_Y / 2), (MM_PER_PIXEL_Y * FOV_PIXELS_Y / 2)
    ]

    # Output naming
    poses_tag = "poseMean" if AVERAGE_OVER_POSES else "poseSum"
    layouts_str_fname = f"t8_{poses_tag}_{successful_loads}layouts"
    layouts_str_title = f"T8 Sensitivity Map ({successful_loads} Layouts, {poses_tag})"

    # Plot and save the map
    plt.figure(figsize=(8, 7))
    plt.imshow(sensitivity_map_2d, cmap='viridis', origin='lower', extent=extent)
    plt.colorbar(label='Average Sensitivity (no max-normalization)')
    plt.title(layouts_str_title)
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.axhline(0, color='white', linestyle=':', lw=0.5)
    plt.axvline(0, color='white', linestyle=':', lw=0.5)
    plt.tight_layout()

    map_filename = os.path.join(ppdf_files_base_dir, f"sensitivity_map_{layouts_str_fname}.png")
    plt.savefig(map_filename, dpi=300)
    print(f"\nSaved sensitivity map to: {map_filename}")
    plt.close()

    # Histogram
    create_and_plot_sensitivity_histogram(
        sensitivity_map_2d,
        ppdf_files_base_dir,
        layouts_str_fname,
        layouts_str_title
    )

    # Print quick stats
    nonzero = sensitivity_map_2d[sensitivity_map_2d > 0]
    if nonzero.size:
        print(f"\n[STATS] nonzero mean={nonzero.mean():.6g}, min={nonzero.min():.6g}, max={nonzero.max():.6g}")
    else:
        print("\n[STATS] No nonzero sensitivity pixels found.")