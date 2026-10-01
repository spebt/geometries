#!/usr/bin/env python3
import os
import torch
from torch import save as torch_save, Tensor

# === Import functions from helper.py ===
from helper import (
    generate_md5_from_tensors,
    positions_parameters,
    transform_to_positions_2d_batch,
    plot_polygons_from_vertices_2d_mpl,
    visualize_linear_spect_geometry,
    generate_uniform_linear_geometry,  # ✅ use the new one
    OutDataDict
)

if __name__ == "__main__":
    # ============================================================
    # 1. Generate BASE planar MPH-SPECT geometry
    # ============================================================
    print("\n=== Generating planar (linear) MPH-SPECT geometry ===")

    base_detector_units, base_plate_segments = generate_uniform_linear_geometry(
        aperture_width_x=140.0,
        aperture_height_y=6.0,
        pinhole_diameter_mm=1.6,
        aperture_open_ratio=0.125,
    )

    print(f"Generated {base_detector_units.shape[0]} detector tiles "
          f"and {base_plate_segments.shape[0]} pinhole polygons.")

    base_scanner_md5 = generate_md5_from_tensors(base_detector_units, base_plate_segments)
    print(f"MD5 hash of base scanner geometry: {base_scanner_md5}")

    # ============================================================
    # 2. Visualize the base geometry
    # ============================================================
    print("\nVisualizing planar geometry...")
    visualize_linear_spect_geometry(
        base_detector_units,
        base_plate_segments,
        fov_diameter_mm=16.0,
        save_path="mph_linear_layout_visualized.png"
    )

    # ============================================================
    # 3. Define motion / transformation parameters
    # ============================================================
    n_rotations_for_motion = 20       # Number of angular positions
    angle_step_deg_for_motion = 1.0   # Step size per rotation (deg)
    n_shifts_grid_for_motion = [1, 1] # No translation grid (rotation only)
    shift_step_mm_for_motion = [0.0, 0.0]

    print(f"\nMotion definition: {n_rotations_for_motion} rotations, "
          f"{n_shifts_grid_for_motion} translations (step={shift_step_mm_for_motion} mm)")

    motion_positions = positions_parameters(
        n_rotations_for_motion,
        angle_step_deg_for_motion,
        n_shifts_grid_for_motion,
        shift_step_mm_for_motion
    )
    n_total_positions = motion_positions.shape[0]
    print(f"Total scanner positions to generate: {n_total_positions}")

    # ============================================================
    # 4. Apply transformations (collimator only)
    # ============================================================
    print("\nApplying transformations to aperture plate only...")
    transformed_plate_segments_batch = transform_to_positions_2d_batch(
        motion_positions,
        base_plate_segments.reshape(-1, 2)
    )
    transformed_plate_segments_batch = transformed_plate_segments_batch.reshape(
        n_total_positions, base_plate_segments.shape[0], base_plate_segments.shape[1], 2
    )

    # ============================================================
    # 5. Prepare data for saving
    # ============================================================
    print("\nPreparing data for saving...")
    out_data: OutDataDict = {
        "scanner MD5": base_scanner_md5,
        "motion_parameters": {
            "n_rotational_steps_defined": n_rotations_for_motion,
            "n_translational_shifts_grid": n_shifts_grid_for_motion,
            "translational_step_size_mm": shift_step_mm_for_motion,
            "generated_n_positions": n_total_positions
        },
        "layouts": {}
    }

    layouts_dict = {}
    for i in range(n_total_positions):
        layout_key = f"position {i:03d}"
        layouts_dict[layout_key] = {
            "position": motion_positions[i],
            "detector units": base_detector_units,               # detectors stay fixed
            "plate segments": transformed_plate_segments_batch[i] # collimator moves
        }

    out_data["layouts"] = layouts_dict

    # ============================================================
    # 6. Save transformed layouts
    # ============================================================
    motion_id_str_parts = [
        f"rot{n_rotations_for_motion}",
        f"trans{n_shifts_grid_for_motion[0]}x{n_shifts_grid_for_motion[1]}",
        f"step{shift_step_mm_for_motion[0]}x{shift_step_mm_for_motion[1]}"
    ]
    motion_descriptor = "_".join(str(p).replace('.', 'p') for p in motion_id_str_parts)

    transformed_layouts_unique_id = f"{base_scanner_md5}_{motion_descriptor}"
    out_file_name = f"scanner_layouts_{transformed_layouts_unique_id}.tensor"

    print(f"\nSaving {n_total_positions} transformed layouts to:\n  {out_file_name}")
    torch_save(out_data, out_file_name)
    print("✅ Transformed planar MPH-SPECT layouts saved successfully.\n")

    print("All done — visualization image and tensor file are ready.")
