import os
import torch
from torch import save as torch_save, cat, Tensor
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

# Assuming helper.py is in the same directory or accessible via PYTHONPATH
try:
    from helper import (
        generate_mph_spect_geometry, 
        generate_md5_from_tensors, 
        plot_polygons_from_vertices_2d_mpl,
        rotate_and_repeat_4gon, 
        positions_parameters,
        transform_to_positions_2d_batch,
        OutDataDict 
    )
except ImportError as e:
    print(f"Error importing from helper.py: {e}")
    print("Ensure helper.py contains all necessary functions, including transformation functions.")
    exit()

if __name__ == "__main__":
    # --- 1. Define Base MPH-SPECT Configuration Parameters ---
    cfg_pinhole_diameter_mm = 3.0
    cfg_n_pinholes = 18
    cfg_collimator_ring_radius_mm = 215.0
    cfg_pinhole_to_detector_distance_mm = 542.0
    cfg_scint_tangential_mm = 3.5                    #arc
    cfg_scint_radial_thickness_mm = 6.0              #radially outward
    cfg_pinhole_channel_length_mm = 20.0             #thickness of cl at hole

    print(f"Generating base MPH-SPECT geometry...")
    
    try:
        base_detector_units, base_plate_segments = generate_mph_spect_geometry(  #(n_scint,4,2) (n_seg, 4, 2)
            pinhole_diameter_mm=cfg_pinhole_diameter_mm,
            n_pinholes=cfg_n_pinholes,
            collimator_ring_radius_mm=cfg_collimator_ring_radius_mm,
            pinhole_to_detector_distance_mm=cfg_pinhole_to_detector_distance_mm,
            scint_tangential_mm=cfg_scint_tangential_mm,
            scint_radial_thickness_mm=cfg_scint_radial_thickness_mm,
            pinhole_channel_length_mm=cfg_pinhole_channel_length_mm
        )
    except ValueError as e:
        print(f"Error during base geometry generation: {e}")
        exit()
        
    base_scanner_md5 = generate_md5_from_tensors(base_detector_units, base_plate_segments)

    # --- 2. Define Motion Parameters ---
    n_rotations_for_motion = 20 
    angle_step_deg_for_motion = 1.0 
    n_shifts_grid_for_motion = [1, 1]  #no translation
    shift_step_mm_for_motion = [0.0, 0.0]  

    # --- 3. Generate Transformation Positions ---
    motion_positions = positions_parameters(
        n_rotations_for_motion,
        angle_step_deg_for_motion, #degree to rad
        n_shifts_grid_for_motion,  #centre grid for sym
        shift_step_mm_for_motion
    )
    n_total_positions = motion_positions.shape[0]

    # --- 4. Transform Components ---
    print("\nApplying transformations to COLLIMATOR ONLY...")
    transformed_plate_segments_batch = transform_to_positions_2d_batch(  #r(thetha)
        motion_positions,
        base_plate_segments.reshape(-1, 2)
    )
    transformed_plate_segments_batch = transformed_plate_segments_batch.reshape(
        n_total_positions, base_plate_segments.shape[0], base_plate_segments.shape[1], 2
    )

    # --- 5. Prepare Data for Saving ---
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
            "detector units": base_detector_units,
            "plate segments": transformed_plate_segments_batch[i],
        }
    out_data["layouts"] = layouts_dict

    # --- 6. Save All Transformed Layouts ---
    motion_id_str_parts = [
        f"rot{n_rotations_for_motion}",
        f"trans{n_shifts_grid_for_motion[0]}x{n_shifts_grid_for_motion[1]}",
        f"step{shift_step_mm_for_motion[0]}x{shift_step_mm_for_motion[1]}"
    ]
    motion_descriptor = "_".join(str(p).replace('.', 'p') for p in motion_id_str_parts)
    transformed_layouts_unique_id = f"{base_scanner_md5}_{motion_descriptor}"
    out_file_name = f"scanner_layouts_{transformed_layouts_unique_id}.tensor"
    
    torch_save(out_data, out_file_name)
    print(f"Saved: {out_file_name}")

    # =====================================================================
    # === VISUALIZATION SECTION (Updated for Labels and Font Size)      ===
    # =====================================================================
    
    # 1. Set Font Sizes globally to 18
    plt.rcParams.update({
        'font.size': 18,            
        'axes.labelsize': 18,       
        'axes.titlesize': 18,       
        'xtick.labelsize': 18,      
        'ytick.labelsize': 18,      
        'legend.fontsize': 18       
    })

    fig, ax = plt.subplots(figsize=(14, 14))

    # 2. Draw Detectors (Label updated to "Detector Ring")
    if base_detector_units.numel() > 0:
        plot_polygons_from_vertices_2d_mpl(base_detector_units, ax, facecolor='lightblue', edgecolor='blue', alpha=0.7, label="Detector Ring")

    # 3. Draw Collimator (Label updated to "Collimator")
    if base_plate_segments.numel() > 0:
        plot_polygons_from_vertices_2d_mpl(base_plate_segments, ax, facecolor='gray', edgecolor='black', label="Collimator")
    
    # 4. Draw FOV (Label updated to "Field of View (FOV)")
    CIRCULAR_FOV_DIAMETER_MM = 70.0 
    circular_fov_radius_mm = CIRCULAR_FOV_DIAMETER_MM / 2.0
    circular_fov_patch = Circle(
        (0, 0), 
        circular_fov_radius_mm,
        edgecolor='red', 
        facecolor='none', 
        linestyle='--', 
        linewidth=2,
        label='Field of View (FOV)'
    )
    ax.add_patch(circular_fov_patch)

    # 5. Reference Circles (Empty labels so they don't show in the legend box)
    coll_ring_circle = plt.Circle((0,0), cfg_collimator_ring_radius_mm, color='darkgrey', fill=False, linestyle=':', label='')
    det_inner_radius = cfg_collimator_ring_radius_mm + cfg_pinhole_to_detector_distance_mm
    det_ring_circle_inner = plt.Circle((0,0), det_inner_radius, color='cyan', fill=False, linestyle=':', label='')
    ax.add_artist(coll_ring_circle)
    ax.add_artist(det_ring_circle_inner)

    # 6. Final Plot Styling
    ax.set_aspect('equal', adjustable='box')
    
    # Calculate Plot Limits
    max_coord_det = torch.abs(base_detector_units).max().item() if base_detector_units.numel() > 0 else 0
    max_coord_coll = torch.abs(base_plate_segments).max().item() if base_plate_segments.numel() > 0 else 0
    plot_limit = max(max_coord_det, max_coord_coll, det_inner_radius + 50) * 1.05    
  
    ax.set_xlim([-plot_limit, plot_limit])
    ax.set_ylim([-plot_limit, plot_limit])
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    
    # Title Update
    ax.set_title("Straight Cylinder Aperture", fontsize=24, fontweight='bold', pad=20)
    
    # Legend Box (Top Right, Font 18)
    ax.legend(loc='upper right', frameon=True)
    
    plt.grid(True)
    plt.savefig("mph_base_layout_with_transforms_generated.png", dpi=300)
    print("\nSaved visualization to mph_base_layout_with_transforms_generated.png")