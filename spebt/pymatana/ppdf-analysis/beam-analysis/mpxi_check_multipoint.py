import os
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

# -----------------------------------------------------------------------------
# 1. DATA LOADING & GEOMETRY UTILS
# -----------------------------------------------------------------------------
def load_scanner_resources(base_dir, tensor_path, layout_idx=0):
    """Loads masks, beam properties, and detector polygons."""
    props_path = os.path.join(base_dir, f"beams_properties_configuration_{layout_idx:02d}.hdf5")
    masks_path = os.path.join(base_dir, f"beams_masks_configuration_{layout_idx:02d}.hdf5")

    with h5py.File(props_path, 'r') as f_p, h5py.File(masks_path, 'r') as f_m:
        properties = f_p["beam_properties"][:]
        # header index for 'number of coexisting beams' is 10
        masks = f_m["beam_mask"][:] 

    mpxi_lookup = {(int(r[1]), int(r[2])): int(r[10]) for r in properties}
    
    blob = torch.load(tensor_path, map_location="cpu", weights_only=False)
    det_verts = blob["layouts"][f"position {layout_idx:03d}"]["detector units"]
    
    return masks, mpxi_lookup, det_verts

def mm_to_idx(x_mm, y_mm, nx=280, ny=280, fov_size=70):
    """Converts world mm to 1D voxel index for the 280x280 grid."""
    offset = fov_size / 2
    px_size = fov_size / nx
    ix = int((x_mm + offset) / px_size)
    iy = int((y_mm + offset) / px_size)
    return max(0, min(nx-1, ix)) * ny + max(0, min(ny-1, iy))

def idx_to_mm(v_idx, nx=280, ny=280, fov_size=70):
    """Converts 1D voxel index back to world mm."""
    ix, iy = divmod(v_idx, ny)
    offset = fov_size / 2
    px_size = fov_size / nx
    return (ix - nx/2) * px_size, (iy - ny/2) * px_size

# -----------------------------------------------------------------------------
# 2. CORE GHOST ANALYSIS (Hypothesis Testing)
# -----------------------------------------------------------------------------
def analyze_ghost_hypothesis(source_points_mm, masks, mpxi_lookup, 
                             purity_threshold=0.9, sim_threshold=0.9, 
                             nx=280, ny=280, AND_COND=False):
    """
    Finds ghost regions for a list of source points.
    AND_COND = True: Returns voxels that are ghosts for ALL source points.
    """
    n_dets, n_voxels = masks.shape
    results = {} 

    print(f"Pre-calculating footprints with Purity >= {purity_threshold}...")
    fov_footprints = []
    for v_idx in range(n_voxels):
        beam_ids = masks[:, v_idx]
        active = np.where(beam_ids > 0)[0]
        
        current_fp = []
        mux_count = 0
        for d in active:
            bid = int(beam_ids[d])
            mpxi = mpxi_lookup.get((int(d), bid), 0)
            if mpxi >= 2: mux_count += 1
            current_fp.append((int(d), bid))
        
        purity = mux_count / len(active) if active.size > 0 else 0
        if purity >= purity_threshold:
            fov_footprints.append(set(current_fp))
        else:
            fov_footprints.append(None)

    target_fps = []
    valid_source_points = []
    for ax, ay in source_points_mm:
        a_idx = mm_to_idx(ax, ay, nx, ny)
        target_fp = fov_footprints[a_idx]
        if target_fp is not None:
            target_fps.append(target_fp)
            valid_source_points.append((ax, ay))
        else:
            print(f"Skipping Point A ({ax}, {ay}): Voxel not multiplexed enough.")

    if AND_COND:
        common_ghosts = []
        print(f"Applying AND condition across {len(valid_source_points)} phantom points...")
        for v_idx, fp in enumerate(fov_footprints):
            if fp is None: continue
            
            gx_mm, gy_mm = idx_to_mm(v_idx, nx, ny)
            # Ensure we are looking for a spatial ghost, not just the source vicinity
            too_close = any(np.sqrt((gx_mm - sx)**2 + (gy_mm - sy)**2) < 5.0 for sx, sy in valid_source_points)
            if too_close: continue

            match_all = True
            for t_fp in target_fps:
                sim = len(t_fp.intersection(fp)) / len(t_fp)
                if sim < sim_threshold:
                    match_all = False
                    break
            
            if match_all:
                common_ghosts.append((gx_mm, gy_mm))
        
        results["COMMON_GHOST"] = common_ghosts
        for pt in valid_source_points: results[pt] = [] 
    else:
        for i, (ax, ay) in enumerate(valid_source_points):
            t_fp = target_fps[i]
            ghosts = []
            for v_idx, fp in enumerate(fov_footprints):
                if fp is None or v_idx == mm_to_idx(ax, ay, nx, ny): continue
                sim = len(t_fp.intersection(fp)) / len(t_fp)
                if sim >= sim_threshold:
                    gx_mm, gy_mm = idx_to_mm(v_idx, nx, ny)
                    if np.sqrt((gx_mm - ax)**2 + (gy_mm - ay)**2) > 5.0:
                        ghosts.append((gx_mm, gy_mm))
            results[(ax, ay)] = ghosts

    return results

# -----------------------------------------------------------------------------
# 3. VISUALIZATION
# -----------------------------------------------------------------------------
def visualize_results(results, det_verts, out_dir, layout_idx, AND_COND=False, ZOOM_IN=False):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    if not ZOOM_IN:
        det_coll = PolyCollection(det_verts.tolist(), facecolor='royalblue', 
                                  edgecolor='white', alpha=0.2, label="Detectors")
        ax.add_collection(det_coll)
        ax.set_xlim([-800, 800]); ax.set_ylim([-800, 800])
    else:
        # Zoom strictly to the FOV region
        ax.set_xlim([-35, 35]); ax.set_ylim([-35, 35])

    if AND_COND:
        # Plot source phantom points
        sources = [k for k in results.keys() if isinstance(k, tuple)]
        if sources:
            sx, sy = zip(*sources)
            ax.scatter(sx, sy, color='blue', marker='*', s=150, label="Phantom Points")
        
        # Highlight Common Ghost Region
        common = results.get("COMMON_GHOST", [])
        if common:
            gx, gy = zip(*common)
            # Use specific red color for the AND+ZOOM highlight
            ghost_color = 'red' if ZOOM_IN else 'magenta'
            ax.scatter(gx, gy, color=ghost_color, marker='o', s=40, alpha=0.6, 
                       label=f"Common Ghost (AND Logic)")
    else:
        # Standard logic
        colors = plt.cm.get_cmap('tab10', len(results))
        for i, (point, ghosts) in enumerate(results.items()):
            if point == "COMMON_GHOST": continue
            c = colors(i)
            ax.scatter(point[0], point[1], color=c, marker='*', s=150)
            if ghosts:
                gx, gy = zip(*ghosts)
                ax.scatter(gx, gy, color=c, marker='o', s=30, alpha=0.4)

    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    title_tag = "[ZOOMED]" if ZOOM_IN else "[SYSTEM]"
    ax.set_title(f"Ghost Region Analysis {title_tag} - Layout {layout_idx}")
    
    tag = "and" if AND_COND else "or"
    save_path = os.path.join(out_dir, f"ghost_plot_{tag}_z{int(ZOOM_IN)}_{layout_idx:02d}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    plt.close()

# -----------------------------------------------------------------------------
# 4. MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    DATA_DIR = "../../../data/system_layout_2mm_36pinholes_rotated/filtered_outputs/mpxi_2"
    TENSOR_FILE = "../../../data/scanner_layouts/system_layout_2mm_36pinholes_rotated.tensor"
    
    AND_COND = True 
    ZOOM_IN = True  
    
    PHANTOM_POINTS = [(5.0, 5.0), (-5.0, -5.0)]

    masks, lookup, geometry = load_scanner_resources(DATA_DIR, TENSOR_FILE)
    
    ghost_results = analyze_ghost_hypothesis(
        PHANTOM_POINTS, masks, lookup, 
        purity_threshold=0.7, sim_threshold=0.2, AND_COND=AND_COND
    )
    
    visualize_results(ghost_results, geometry, DATA_DIR, 0, AND_COND, ZOOM_IN)