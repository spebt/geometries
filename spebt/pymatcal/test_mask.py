import numpy as np
import h5py
import matplotlib.pyplot as plt
import os


def load_ppdfs_for_layout(ppdf_dir: str, layout_idx: int, n_xtals_to_load: int = -1):
    """Loads PPDFs from an HDF5 file for a specific layout."""
    ppdf_filename = os.path.join(ppdf_dir, f"position_{layout_idx:03d}_ppdfs.hdf5")
    if not os.path.exists(ppdf_filename):
        print(f"PPDF file {ppdf_filename} does not exist.")
        return None

    with h5py.File(ppdf_filename, "r") as f:
        # Selects all crystals if n_xtals_to_load is -1, otherwise slices the array
        ppdfs_data = f["ppdfs"][:n_xtals_to_load] if n_xtals_to_load != -1 else f["ppdfs"][:]
    return ppdfs_data


def create_and_plot_sensitivity_histogram(
    sensitivity_map: np.ndarray,
    fov_mask: np.ndarray,
    output_dir: str,
    file_suffix: str,
    title_suffix: str,
):
    """Generates and saves a histogram for the sensitivity map inside the circular 10-mm FOV."""
    # --- 1. Prepare Data: restrict to circular FOV (r <= 5 mm) ---
    fov_values = sensitivity_map[fov_mask].flatten()
    if fov_values.size == 0:
        print("Warning: The sensitivity map contains no pixels inside the FOV mask.")
        return

    # --- 2. Create and Plot the Histogram ---
    plt.figure(figsize=(10, 6))
    plt.hist(fov_values, bins=100, color="royalblue", alpha=0.75)
    plt.title(f"Histogram of Sensitivity Values - {title_suffix}")
    plt.xlabel("Sensitivity")
    plt.ylabel("Number of Pixels (Frequency)")
    plt.grid(True, linestyle="--", alpha=0.6)

    mean_sensitivity = np.mean(fov_values)
    plt.axvline(
        mean_sensitivity,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Mean: {mean_sensitivity:.4f}",
    )
    plt.legend()
    plt.tight_layout()

    # --- 3. Save the Histogram Plot ---
    hist_filename = os.path.join(output_dir, f"sensitivity_histogram_{file_suffix}.png")
    plt.savefig(hist_filename)
    print(f"Saved sensitivity histogram to: {hist_filename}")
    plt.close()


if __name__ == "__main__":
    # --- Configuration ---
    ppdf_files_base_dir = "/vscratch/grp-rutaoyao/Omer/spebt/data/sai_10mm/"
    num_layouts_to_load = 1
    layout_indices = list(range(num_layouts_to_load))

    FOV_PIXELS_X, FOV_PIXELS_Y = 200, 200
    MM_PER_PIXEL_X, MM_PER_PIXEL_Y = 0.05, 0.05

    # 10-mm diameter FOV -> 5-mm radius (matches paper HR config)
    FOV_DIAMETER_MM = 10.0
    FOV_RADIUS_MM = FOV_DIAMETER_MM / 2.0

    num_crystals_to_sum = -1
    # --- End Configuration ---

    if not os.path.isdir(ppdf_files_base_dir):
        print(f"Error: PPDF directory '{ppdf_files_base_dir}' not found.")
        exit()

    # --- Loop through layouts, load, and aggregate ---
    print(f"Loading and aggregating PPDFs for layouts: {layout_indices}...")
    aggregated_ppdfs = None
    successful_loads = 0  # Counter for successfully loaded layouts

    for idx in layout_indices:
        print(f"  - Loading layout {idx:03d}...")
        ppdfs_for_layout = load_ppdfs_for_layout(ppdf_files_base_dir, idx, num_crystals_to_sum)

        if ppdfs_for_layout is None:
            print(f"Warning: Could not load data for layout {idx}. Skipping.")
            continue

        if aggregated_ppdfs is None:
            aggregated_ppdfs = ppdfs_for_layout.astype(np.float32)
        else:
            aggregated_ppdfs += ppdfs_for_layout

        successful_loads += 1

    if aggregated_ppdfs is None:
        print("Error: No PPDF data was successfully loaded. Exiting.")
        exit()

    print(f"\nAggregation complete. Successfully loaded {successful_loads} layouts.")

    # --- Generate and NORMALIZE the sensitivity map ---
    if successful_loads > 0:
        # Divide by the number of successful loads to get the average over layouts
        sensitivity_map_1d = np.sum(aggregated_ppdfs, axis=0) / successful_loads
    else:
        sensitivity_map_1d = np.zeros(FOV_PIXELS_X * FOV_PIXELS_Y, dtype=np.float32)

    sensitivity_map_2d = sensitivity_map_1d.reshape((FOV_PIXELS_Y, FOV_PIXELS_X))

    # --- Build circular 10-mm FOV mask (r <= 5 mm) ---
    x_coords = np.linspace(-FOV_RADIUS_MM, FOV_RADIUS_MM, FOV_PIXELS_X)
    y_coords = np.linspace(-FOV_RADIUS_MM, FOV_RADIUS_MM, FOV_PIXELS_Y)
    XX, YY = np.meshgrid(x_coords, y_coords)
    fov_mask = (XX**2 + YY**2) <= (FOV_RADIUS_MM**2 + 1e-6)

    # Optionally mask values outside FOV for plotting
    sensitivity_map_2d_masked = np.where(fov_mask, sensitivity_map_2d, 0.0)

    # Determine plot extent
    extent = [
        -FOV_RADIUS_MM,
        FOV_RADIUS_MM,
        -FOV_RADIUS_MM,
        FOV_RADIUS_MM,
    ]

    # --- Create descriptive names for aggregated output files and titles ---
    layouts_str_fname = f"aggregated_{successful_loads}layouts_normalized"
    layouts_str_title = f"Normalized Sensitivity Map ({successful_loads} Layouts)"

    # Plot and Save the Aggregated and Normalized Sensitivity Map Image
    plt.figure(figsize=(8, 7))
    plt.imshow(
        sensitivity_map_2d_masked,
        cmap="viridis",
        origin="lower",
        extent=extent,
    )
    plt.colorbar(label="Average Sensitivity (Normalized)")
    plt.title(layouts_str_title)
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.axhline(0, color="white", linestyle=":", lw=0.5)
    plt.axvline(0, color="white", linestyle=":", lw=0.5)
    plt.tight_layout()
    map_filename = os.path.join(ppdf_files_base_dir, f"sensitivity_map_{layouts_str_fname}.png")
    plt.savefig(map_filename)
    print(f"\nSaved normalized sensitivity map to: {map_filename}")
    plt.close()

    # Create a histogram for the normalized data *inside the circular FOV*
    create_and_plot_sensitivity_histogram(
        sensitivity_map_2d_masked,
        fov_mask,
        ppdf_files_base_dir,
        layouts_str_fname,
        layouts_str_title,
    )
