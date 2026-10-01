import os
import sys
import time
import h5py
from torch import device, arange, tensor, get_num_threads

# Assuming these imports are in your project structure
from scanner_modeling._raytracer_2d._local_functions import (
    ppdf_2d_local,
    reduced_edges_2d_local,
    sfov_properties,
    subdivision_grid_rectangle,
)
from scanner_modeling.geometry_2d import (
    fov_tensor_dict,
    load_scanner_geometry_from_layout,
    load_scanner_layouts,
)

def calculate_ppdf_for_layout(layout_idx: int):
    """
    Calculates and saves the PPDF for all crystals in a single, specified layout.

    This function is designed to be called as a single task in a parallel environment,
    taking a layout index as its primary argument.
    """
    start_time = time.time()
    print(f"--- Starting PPDF calculation for Layout Index: {layout_idx} ---")

    # --- 1. Configuration and Setup ---
    # --- ADD THESE TWO LINES ---
    # Define the path where you want to save the output files.
    output_directory = "/vscratch/grp-rutaoyao/Omer/mph/output" # <-- IMPORTANT: CHANGE THIS TO YOUR VSCRACTH PATH
    os.makedirs(output_directory, exist_ok=True) # Creates the directory if it doesn't exist

    default_device = device("cpu")
    scanner_layout_file_relative_path = (
        "/vscratch/grp-rutaoyao/Omer/mph/geometry/scanner_layouts_636bed318120192d3b39fb33a7655ea8_rot20_trans1x1_step0p0x0p0.tensor"
    )
    scanner_layout_dir = os.path.dirname(scanner_layout_file_relative_path)
    scanner_layout_filename = os.path.basename(scanner_layout_file_relative_path)

    scanner_layouts, layouts_md5 = load_scanner_layouts(
        scanner_layout_dir,
        scanner_layout_filename,
    )
    
    # Check if the requested layout index is valid
    n_layouts_total = len(scanner_layouts)
    if not (0 <= layout_idx < n_layouts_total):
        print(f"Error: Invalid layout_idx {layout_idx}. File contains {n_layouts_total} layouts (indexed 0 to {n_layouts_total-1}).")
        sys.exit(1)

    mu_dict = tensor([3.5, 0.475], device=default_device)
    fov_dict = fov_tensor_dict((128, 128), (70, 70), (0.0, 0.0), (3, 3))
    crystal_n_subs = (3, 3)
    
    sfov_pxs_ids, sfov_pixels_batch, sfov_corners_batch = sfov_properties(fov_dict)
    fov_n_pxs = int(fov_dict["n pixels"].prod())
    n_sfov = int(fov_dict["n subdivisions"].prod())
    sfov_pxs_ids_1d = (
        sfov_pxs_ids[:, :, 0] * fov_dict["n pixels"][0] + sfov_pxs_ids[:, :, 1]
    )
    subdivision_grid = subdivision_grid_rectangle(crystal_n_subs)

    print(f"PyTorch using {get_num_threads()} threads.")

    # --- 2. Load Geometry for the specific layout ---
    (
        plate_objects_vertices,
        crystal_objects_vertices,
        plate_objects_edges,
        crystal_objects_edges,
    ) = load_scanner_geometry_from_layout(layout_idx, scanner_layouts)

    n_crystals_total = crystal_objects_vertices.shape[0]
    crystal_idx_tensor = arange(n_crystals_total)
    n_crystals = int(crystal_idx_tensor.shape[0])
    print(f"Found {n_crystals_total} crystals in layout {layout_idx}.")

    # --- 3. Process all crystals for this layout ---
    filename = f"scanner_layouts_{layouts_md5}_layout_{layout_idx:03d}.hdf5"
    h5_file_path = os.path.join(output_directory, filename)
    
    with h5py.File(h5_file_path, "w") as h5file:
        ppdf_dataset = h5file.create_dataset("ppdfs", (n_crystals, fov_n_pxs), dtype="f")

        for dataset_idx, crystal_idx_tensor_val in enumerate(crystal_idx_tensor):
            crystal_idx = int(crystal_idx_tensor_val.item())
            
            # This part can be further optimized if memory allows, but is correct
            reduced_crystal_edges_sfovs = []
            reduced_plate_edges_sfovs = []
            for sfov_idx in range(n_sfov):
                reduced_plate_edges, reduced_crystal_edges = reduced_edges_2d_local(
                    sfov_idx, crystal_idx, sfov_corners_batch,
                    plate_objects_vertices, plate_objects_edges,
                    crystal_objects_vertices, crystal_objects_edges,
                    default_device,
                )
                reduced_crystal_edges_sfovs.append(reduced_crystal_edges)
                reduced_plate_edges_sfovs.append(reduced_plate_edges)

            for sfov_idx in range(n_sfov):
                ppdf_slice = ppdf_2d_local(
                    sfov_idx, crystal_idx, sfov_pixels_batch,
                    crystal_objects_vertices, reduced_plate_edges_sfovs[sfov_idx],
                    reduced_crystal_edges_sfovs[sfov_idx], subdivision_grid,
                    mu_dict, default_device,
                )
                ppdf_dataset[dataset_idx, sfov_pxs_ids_1d[sfov_idx]] = ppdf_slice.cpu().numpy()

    end_time = time.time()
    duration = end_time - start_time
    print(f"--- Finished Layout {layout_idx} in {duration:.2f} seconds. ---")
    print(f"Results saved to {h5_file_path}")


if __name__ == "__main__":
    # --- Argument Parsing ---
    if len(sys.argv) != 2:
        print("Usage: python arg_ppdf_calculation.py <layout_idx>")
        print("  <layout_idx>: The integer index of the layout to process.")
        sys.exit(1)

    try:
        layout_to_process = int(sys.argv[1])
    except ValueError:
        print(f"Error: Invalid argument. '{sys.argv[1]}' is not an integer.")
        sys.exit(1)

    # --- Run Calculation ---
    calculate_ppdf_for_layout(layout_to_process)