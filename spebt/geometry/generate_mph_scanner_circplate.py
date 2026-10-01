#!/usr/bin/env python3
import torch
from torch import save as torch_save
from helper import (
    generate_linear_slim_geometry,
    visualize_linear_geometry,
    generate_md5_from_tensors,
    positions_parameters,
    transform_to_positions_2d_batch,
    OutDataDict,
)

if __name__ == "__main__":
    print("\n=== Generating Planar (Linear) MPH-SPECT Tensor ===")

    # ------------------------------------------------------------
    # 1. Generate Base Geometry
    # ------------------------------------------------------------
    base_detectors, base_plate = generate_linear_slim_geometry(
        aperture_width_x=140.0,
        aperture_height_y=6.0,
        n_layers=4,
        layer_spacing_mm=30.0
    )


    md5 = generate_md5_from_tensors(base_detectors, base_plate)
    print(f"Base geometry MD5: {md5}")

    # ------------------------------------------------------------
    # 2. Define Motion / Rotation Parameters
    # ------------------------------------------------------------
    n_rot, angle_step_deg = 20, 1.0
    n_shift_grid, shift_step_mm = [1, 1], [0.0, 0.0]

    motion_positions = positions_parameters(
        n_rot,
        angle_step_deg,
        n_shift_grid,
        shift_step_mm
    )

    n_positions = motion_positions.shape[0]
    print(f"Total positions: {n_positions}")

    # ------------------------------------------------------------
    # 3. Apply Transformations (Aperture only)
    # ------------------------------------------------------------
    transformed_plates = transform_to_positions_2d_batch(
        motion_positions,
        base_plate.reshape(-1, 2)
    )
    transformed_plates = transformed_plates.reshape(
        n_positions, base_plate.shape[0], base_plate.shape[1], 2
    )

    # ------------------------------------------------------------
    # 4. Package Data for Saving
    # ------------------------------------------------------------
    out_data: OutDataDict = {
        "scanner MD5": md5,
        "motion_parameters": {
            "n_rotational_steps_defined": n_rot,
            "n_translational_shifts_grid": n_shift_grid,
            "translational_step_size_mm": shift_step_mm,
            "generated_n_positions": n_positions,
        },
        "layouts": {
            f"position {i:03d}": {
                "position": motion_positions[i],
                "detector units": base_detectors,
                "plate segments": transformed_plates[i],
            } for i in range(n_positions)
        }
    }

    # ------------------------------------------------------------
    # 5. Save Tensor Layouts
    # ------------------------------------------------------------
    motion_id = f"rot{n_rot}_trans{n_shift_grid[0]}x{n_shift_grid[1]}"
    tensor_filename = f"scanner_layouts_{md5}_{motion_id}.tensor"

    torch_save(out_data, tensor_filename)
    print(f"\n✅ Saved {n_positions} planar layouts → {tensor_filename}\n")

    visualize_linear_geometry(base_detectors, base_plate)
    
    from helper import visualize_linear_geometry_with_pinholes

    visualize_linear_geometry_with_pinholes(
        base_detectors,
        base_plate,
        pinhole_diameter_mm=1.6,
        aperture_open_ratio=0.125,
        fov_diameter_mm=16.0,
        save_path="mph_planar_layout_with_pinholes.png",
    )
