import torch
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from torch import cat
import numpy as np

# --- Import necessary functions from your local modules ---
from beam_property_extract import (
    sample_ppdf_on_arc_2d_local,
    beams_boundaries_radians,
    get_beams_masks,
    get_beams_weighted_center,
    get_beam_width,
)
from convex_hull_helper import convex_hull_2d, sort_points_for_hull_batch_2d
from geometry_2d_io import load_scanner_layout_geometries, load_scanner_layouts
from geometry_2d_utils import (
    fov_tensor_dict,
    pixels_coordinates,
    pixels_to_detector_unit_rads,
)
from ppdf_io import load_ppdfs_data_from_hdf5


# --- Helper for plotting ---
def plot_polygons_from_vertices_mpl(vertices: torch.Tensor, ax: plt.Axes, **kwargs):
    """Draws a collection of polygons on the given axes."""
    p = PolyCollection(vertices.tolist(), **kwargs)
    ax.add_collection(p)
    return p


# %% [markdown]
# ## 1. Configuration

# %%
# --- User Configuration ---
LAYOUT_INDEX = 0         # Index of the scanner layout (e.g., 0-23)
DETECTOR_UNIT_INDEX = 1100   # Index of the detector unit to analyze
ZOOM_IN_COLLIMATOR = False   # True: Zoom to collimator plates | False: View full detector ring

# --- File Paths ---
SCANNER_LAYOUTS_DIR = "../../../data/scanner_layouts"
SCANNER_LAYOUTS_FILENAME = "mph_hourglass_single_position_base_2mm_20pinholes.tensor"
PPDFS_DATASET_DIR = "../../../data/mph_hourglass_single_position_base_2mm_20pinholes_rotated_18custom_rot/outputs"

# --- FOV Definition ---
FOV_DICT = fov_tensor_dict(
    n_pixels=(512, 512),
    mm_per_pixel=(0.25, 0.25),
    center_coordinates=(0.0, 0.0),
)

# %% [markdown]
# ## 2. Load Geometry and PPDF Data

# %%
# --- Load Scanner Layout ---
scanner_layouts_data, _ = load_scanner_layouts(SCANNER_LAYOUTS_DIR, SCANNER_LAYOUTS_FILENAME)
plates_vertices, detector_units_vertices = load_scanner_layout_geometries(
    LAYOUT_INDEX, scanner_layouts_data
)
detector_unit_verts = detector_units_vertices[DETECTOR_UNIT_INDEX]
detector_unit_center = detector_unit_verts.mean(dim=0)

# --- Load PPDF Data ---
ppdfs_hdf5_filename = f"position_{LAYOUT_INDEX:03d}_ppdfs.hdf5"
all_ppdfs = load_ppdfs_data_from_hdf5(
    PPDFS_DATASET_DIR, ppdfs_hdf5_filename, FOV_DICT
)
ppdf_data_1d = all_ppdfs[DETECTOR_UNIT_INDEX]
ppdf_data_2d = ppdf_data_1d.view(int(FOV_DICT["n pixels"][0]), int(FOV_DICT["n pixels"][1]))

print(f"Data loaded for Layout {LAYOUT_INDEX}, Detector Unit {DETECTOR_UNIT_INDEX}")
print(f"Detector Unit Center: {detector_unit_center.tolist()}")

# %% [markdown]
# ## 3. Run Beam Segmentation and Property Extraction

# %%
# --- 1. Define Sampling Arc ---
# This fov_corners variable will also be used for plotting later
fov_corners = torch.tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * FOV_DICT["size in mm"] * 0.5

hull_points_for_sampling = cat((fov_corners, detector_unit_center.unsqueeze(0)))
sorted_hull_points = sort_points_for_hull_batch_2d(hull_points_for_sampling.unsqueeze(0)).squeeze(0)
hull_2d = convex_hull_2d(sorted_hull_points)

# --- 2. Sample PPDF on Arc ---
sampled_ppdf, sampling_rads, sampling_points = sample_ppdf_on_arc_2d_local(
    ppdf_data_2d, detector_unit_center, hull_2d, FOV_DICT
)

# --- 3. Find Beam Boundaries ---
beam_boundaries_rads = beams_boundaries_radians(sampled_ppdf, sampling_rads, threshold=0.01)

if beam_boundaries_rads.numel() == 0:
    raise ValueError("No beams were found for the selected detector unit with the current thresholds.")

# --- 4. Get Masks and Properties ---
fov_points_xy = pixels_coordinates(FOV_DICT)
fov_points_rads = pixels_to_detector_unit_rads(fov_points_xy, detector_unit_center)

beams_masks = get_beams_masks(fov_points_rads, beam_boundaries_rads)
beams_weighted_centers = get_beams_weighted_center(beams_masks, fov_points_xy, ppdf_data_2d)

# Get FWHM and other line-based properties
beams_fwhm, x_bounds_batch, sampled_beams_data, beam_sp_distance = get_beam_width(
    beams_weighted_centers, detector_unit_center, beams_masks, ppdf_data_2d, FOV_DICT
)

print(f"Found {beams_masks.shape[0]} beams.")
for i in range(beams_fwhm.shape[0]):
    print(f"  - Beam {i+1}: FWHM = {beams_fwhm[i]:.4f} mm, Center = {beams_weighted_centers[i].tolist()}")

# %% [markdown]
# ## 4. Visualization

# %%
# --- Create the Combined Plot ---
fig, axs = plt.subplots(1, 3, figsize=(28, 8), constrained_layout=True)

# -------------------------------------------------------------------
# --- Left Plot: Scanner Geometry with Overlaid Intensity Arc ---
# -------------------------------------------------------------------
ax = axs[0]

# 1. Plot Geometry
plot_polygons_from_vertices_mpl(
    plates_vertices, ax=ax, fc='indigo', ec='black',
    alpha=0.4, label='Collimator Plates'
)
plot_polygons_from_vertices_mpl(
    detector_units_vertices, ax=ax, fc='#FFDAB9', ec='black',
    alpha=0.6, label='All Detector Units'
)

# 2. Plot FOV Boundary (Green Dashed Box)
plot_polygons_from_vertices_mpl(
    fov_corners.unsqueeze(0), ax=ax, fc='none', 
    ec='green', lw=2, label='FOV Boundary'
)

# 3. Plot Sampling Arc Boundaries (NEW: Pink Dashed Lines to Arc Ends)
arc_start = sampling_points[0]
arc_end = sampling_points[-1]
# Line to start of arc
ax.plot(
    [detector_unit_center[0].item(), arc_start[0].item()],
    [detector_unit_center[1].item(), arc_start[1].item()],
    linestyle='--', color='pink', alpha=0.8, lw=1.5, label='Sampling Cone'
)
# Line to end of arc
ax.plot(
    [detector_unit_center[0].item(), arc_end[0].item()],
    [detector_unit_center[1].item(), arc_end[1].item()],
    linestyle='--', color='pink', alpha=0.8, lw=1.5
)

# 4. PPDF Background
fov_size = FOV_DICT["size in mm"]
center_coords = FOV_DICT["center coordinates in mm"]
ppdf_extent = [
    (center_coords[0] - fov_size[0] / 2).item(), (center_coords[0] + fov_size[0] / 2).item(),
    (center_coords[1] - fov_size[1] / 2).item(), (center_coords[1] + fov_size[1] / 2).item(),
]
ax.imshow(
    ppdf_data_2d.T, origin="lower", extent=ppdf_extent,
    cmap="hot_r", aspect='equal'
)

# 5. Selected Unit Highlight
plot_polygons_from_vertices_mpl(
    detector_unit_verts.unsqueeze(0), ax=ax, fc='red',
    ec='black', lw=1.5, label=f'Selected Unit ({DETECTOR_UNIT_INDEX})'
)

# 6. Draw lines from detector to beam centers and mark centers
for i, beam_center in enumerate(beams_weighted_centers):
    line_x = [detector_unit_center[0].item(), beam_center[0].item()]
    line_y = [detector_unit_center[1].item(), beam_center[1].item()]
    ax.plot(line_x, line_y, '--', color=f'C{i}', lw=1.5)

ax.scatter(
    beams_weighted_centers[:, 0], beams_weighted_centers[:, 1],
    c='cyan', marker='x', s=100, zorder=5, label='Beam Centers'
)

# 7. "Polar" intensity profile curve
intensity_scaling_factor = 40.0
direction_vectors = sampling_points - detector_unit_center
norm_vectors = direction_vectors / torch.norm(direction_vectors, dim=1, keepdim=True)
displacements = norm_vectors * sampled_ppdf.unsqueeze(1) * intensity_scaling_factor
visualized_curve_points = sampling_points + displacements

ax.plot(visualized_curve_points[:, 0], visualized_curve_points[:, 1], '-', color='cyan', lw=2, label='PPDF Intensity Profile')

# 8. Zoom / Limits Logic
if ZOOM_IN_COLLIMATOR:
    # Zoom to collimator plates
    all_plate_points = plates_vertices.reshape(-1, 2)
    x_min, y_min = all_plate_points.min(dim=0).values
    x_max, y_max = all_plate_points.max(dim=0).values
    
    # Add a 10% margin relative to the object size
    margin_x = (x_max - x_min) * 0.1
    margin_y = (y_max - y_min) * 0.1
    
    ax.set_xlim(x_min - margin_x, x_max + margin_x)
    ax.set_ylim(y_min - margin_y, y_max + margin_y)
    ax.set_title(f"Scanner Geometry (Zoomed: Collimator) - Unit {DETECTOR_UNIT_INDEX}")

else:
    # Zoom to full Detector Ring + 10 unit margin
    all_det_points = detector_units_vertices.reshape(-1, 2)
    x_min, y_min = all_det_points.min(dim=0).values
    x_max, y_max = all_det_points.max(dim=0).values
    
    margin = 10.0
    
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_title(f"Scanner Geometry (Full System) - Unit {DETECTOR_UNIT_INDEX}")

ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.legend(loc='upper right')
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.5)

# -------------------------------------------------------------------
# --- Middle Plot: Line Profiles and FWHM ---
# -------------------------------------------------------------------
ax = axs[1]
for i in range(sampled_beams_data.shape[0]):
    line_profile = sampled_beams_data[i]
    color = f'C{i}'
    ax.plot(beam_sp_distance, line_profile, label=f'Beam {i+1}', color=color)
    half_max = line_profile.max() / 2
    ax.plot(x_bounds_batch[i], [half_max, half_max], '--', color=color, lw=2)
    ax.text(0, half_max, f'FWHM: {beams_fwhm[i]:.2f} mm', color=color, ha='left', va='bottom', fontsize=12, fontweight='bold')
ax.set_title('Beam Spatial Profiles (FWHM)')
ax.set_xlabel('Distance along sampling line (mm)')
ax.set_ylabel('PPDF Intensity')
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()


# -------------------------------------------------------------------
# --- Right Plot: Angular Profile Visualization ---
# -------------------------------------------------------------------
ax = axs[2]
sampling_degrees = np.rad2deg(sampling_rads.numpy())
ax.plot(sampling_degrees, sampled_ppdf, color='black', lw=1.5, label='Arc PPDF Profile')
for i in range(beam_boundaries_rads.shape[0]):
    start_rad, end_rad = beam_boundaries_rads[i]
    color = f'C{i}'
    beam_mask = (sampling_rads >= start_rad) & (sampling_rads <= end_rad)
    ax.fill_between(
        sampling_degrees, sampled_ppdf, where=beam_mask.numpy(),
        color=color, alpha=0.4, label=f'Beam {i+1}'
    )
    beam_ppdf_slice = sampled_ppdf[beam_mask]
    if beam_ppdf_slice.numel() > 0:
        peak_intensity = beam_ppdf_slice.max()
        peak_angle_deg = sampling_degrees[beam_mask][torch.argmax(beam_ppdf_slice)]
        ax.text(
            peak_angle_deg, peak_intensity, f'{peak_intensity:.2e}',
            ha='center', va='bottom', color=color, fontsize=12, fontweight='bold'
        )
ax.set_title('Beam Angular Profiles')
ax.set_xlabel('Angle (degrees)')
ax.set_ylabel('PPDF Intensity')
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()

# --- Save with a descriptive name ---
filename = f"beam_analysis_layout_{LAYOUT_INDEX}_unit_{DETECTOR_UNIT_INDEX}.png"
plt.savefig(filename)
print(f"Plot saved to {filename}")
# plt.show()