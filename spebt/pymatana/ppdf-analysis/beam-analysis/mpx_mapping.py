import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def generate_mpxi_voxel_map(layout_idx, input_dir):
    # 1. Define File Paths
    props_fname = os.path.join(input_dir, f"beams_properties_configuration_{layout_idx:02d}.hdf5")
    masks_fname = os.path.join(input_dir, f"beams_masks_configuration_{layout_idx:02d}.hdf5")

    if not os.path.exists(props_fname) or not os.path.exists(masks_fname):
        print(f"Error: Missing files for layout {layout_idx} in {input_dir}")
        return

    # 2. Load Data
    print(f"Loading data for layout {layout_idx}...")
    with h5py.File(props_fname, "r") as f_props:
        properties = f_props["beam_properties"][:]
        header = [h.decode('utf-8') if isinstance(h, bytes) else h for h in f_props["beam_properties"].attrs["Header"]]
    
    with h5py.File(masks_fname, "r") as f_masks:
        masks = f_masks["beam_mask"][:] # Shape: (n_detectors, 512*512)

    # 3. Identify Column Indices
    det_idx_col = header.index('detector unit id')
    beam_idx_col = header.index('beam id')
    mpxi_col = header.index('number of coexisting beams')

    # 4. Create the Map
    # We want a 512x512 map where the value is the max MPXI across all detector views
    n_detectors = masks.shape[0]
    mpxi_map_2d = np.zeros(280 * 280, dtype=np.float32)

    print("Mapping beam properties to voxels...")
    for det_id in range(n_detectors):
        # Filter properties for this specific detector
        det_props = properties[properties[:, det_idx_col] == det_id]
        
        if len(det_props) == 0:
            continue

        # Create a lookup array for this detector's beams
        # beam_id is Column 2, MPXI is Column 10
        # We use a large enough array to index by beam_id
        max_beam_id = int(det_props[:, beam_idx_col].max())
        lookup = np.zeros(max_beam_id + 1)
        for row in det_props:
            lookup[int(row[beam_idx_col])] = row[mpxi_col]

        # Extract the mask for this detector
        detector_mask = masks[det_id] # (512*512,)
        
        # Replace beam IDs in the mask with their actual MPXI value
        # This gives us the MPXI "seen" by this specific detector at every voxel
        current_det_mpxi = lookup[detector_mask.astype(int)]
        
        # Update the global map (Take the MAX multiplexing value seen by any detector)
        mpxi_map_2d = np.maximum(mpxi_map_2d, current_det_mpxi)

    return mpxi_map_2d.reshape(280, 280)

def visualize_mpxi_map(mpxi_map, layout_idx, out_dir):
    plt.figure(figsize=(10, 8))
    
    # Use a discrete color map to highlight the multiplexing levels (1, 2, 3...)
    levels = np.unique(mpxi_map)
    im = plt.imshow(mpxi_map, 
                    extent=[-35, 35, -35, 35], 
                    origin='lower', 
                    cmap='viridis',
                    interpolation='nearest')
    
    plt.colorbar(im, label='Multiplexing Index (Coexisting Beams)')
    plt.title(f"Voxel-wise Multiplexing Index Map - Layout {layout_idx}\n(512x512 @ 0.125mm/px)")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    
    out_path = os.path.join(out_dir, f"mpxi_map_layout_{layout_idx:02d}.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved visualization to: {out_path}")
    plt.show()

if __name__ == "__main__":
    # --- Configuration ---
    L_IDX = 1
    DATA_DIR = "../../../data/system_layout_2mm_36pinholes_rotated/outputs"
    
    mpx_map = generate_mpxi_voxel_map(L_IDX, DATA_DIR)
    
    if mpx_map is not None:
        visualize_mpxi_map(mpx_map, L_IDX, DATA_DIR)