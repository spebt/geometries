import os
import sys
from typing import Dict, List
from _disk_shape import fov_tensor_dict, hot_rods_add_sector
import torch
from matplotlib import pyplot as plt
from torch import arange, cat, cos, ones, pi, sin, tensor, zeros
from torch import float32 as torch_float32
from torch import load as torch_load
from torch import save as torch_save

# --- Configuration ---
fov_size_in_mm = (10.0, 10.0)  # mm
# Using your example pixel size
fov_px_size_in_mm = (0.05, 0.05)  # mm
fov_n_pxs = (
    (tensor(fov_size_in_mm) / tensor(fov_px_size_in_mm)).int().tolist()
)

fov_dict = fov_tensor_dict(fov_n_pxs, fov_size_in_mm)

phantom = zeros(fov_n_pxs, dtype=torch_float32)


# Sector definitions
# radii = tensor([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
radii = tensor([0.10, 0.125, 0.15, 0.175, 0.20, 0.225])   # mm 
shifts = tensor(
    [[1.5, 0.0], [1.5, 0.0], [1.5, 0.0], [1.5, 0.0], [1.5, 0.0], [1.5, 0.0]]
)  # x, y shift in mm
n_x_layers = tensor(
    [5, 5, 4, 3, 3, 3]
)  # Number of layers in the x-direction

# --- NEW: Variable for rod spacing ---
# 1.0 = rods are touching (center-dist = 2 * radius)
# 2.0 = 1-diameter gap (center-dist = 4 * radius), like original script
# We can set different values for each sector
rod_spacing_factors = tensor([2, 2, 2, 2, 2, 2])

# Sector rotation angles
angles = arange(0, 2 * pi, 2 * pi / 6).unsqueeze(-1) + pi / 6

# --- Generation Loop ---
print("Generating phantom...")
sectors_centers_mm = []
for i in range(shifts.shape[0]):
    print(
        f"  Adding sector {i+1}: "
        f"radius={radii[i].item():.1f}mm, "
        f"spacing_factor={rod_spacing_factors[i].item()}"
    )
    transform = cat((angles[i : i + 1], shifts[i : i + 1]), dim=1)
    sector_centers_mm, sector_centers_px = hot_rods_add_sector(
        phantom,
        int(n_x_layers[i].item()),
        radii[i].item(),
        transform,
        fov_dict,
        rod_spacing_factors[i].item(),  # Pass the new spacing factor
    )
    sectors_centers_mm.append(sector_centers_mm)
print("Phantom generation complete.")

# --- Save Output ---
out_dict = {
    "Description": "Hot Rods Phantom",
    "Metadata": {
        "size in mm": fov_dict["size in mm"].tolist(),
        "mm per pixel": fov_dict["mm per pixel"].tolist(),
        "n pixels": fov_dict["n pixels"].tolist(),
        "center coordinates in mm": fov_dict[
            "center coordinates in mm"
        ].tolist(),
        "rods radii in mm": radii.tolist(),
        "rod spacing factors": rod_spacing_factors.tolist(), # Save new metadata
    },
    "Phantom tensor": phantom,
    "Phantom shape": phantom.shape,
    "Phantom dtype": phantom.dtype,
}

out_filename = f'hot_rods_phantom_{fov_dict["size in mm"][0].item()}_mm_x_{fov_dict["size in mm"][1].item()}_mm.pt'
torch_save(
    out_dict,
    out_filename,
)
print(f"Phantom saved to {out_filename}")

# --- Plotting Code ---
print("Generating plot...")
in_filename = out_filename

# Check if the file exists
if not os.path.isfile(in_filename):
    print(f"File {in_filename} does not exist.")
    sys.exit(1)

# Read the file
try:
    data = torch_load(in_filename, map_location="cpu")
except Exception as e:
    print(f"Error reading file {in_filename}: {e}")

radii = tensor(data["Metadata"]["rods radii in mm"])  # mm

angles = arange(0, 2 * 3.14159, 2 * 3.14159 / 6).unsqueeze(-1) + 3.14159 / 6
# anno_radii = ones(6) * 30.25  # mm (Annotation text position)
anno_radii = ones(6) * 3.0 

anno_xy = cat(
    (
        cos(angles) * anno_radii.view(-1, 1),
        sin(angles) * anno_radii.view(-1, 1),
    ),
    -1,
)
# Use 2*radius for diameter
anno_text = [f"{r*2:.2f} mm" for r in radii]

phantom = data["Phantom tensor"]
fov_size_in_mm: List[float] = data["Metadata"]["size in mm"]  # mm
fov_n_pixels = data["Metadata"]["n pixels"]  # px
fov_mm_per_pixel: List[float] = data["Metadata"]["mm per pixel"]  # mm/px

plt.close("all")
fig, ax = plt.subplots(dpi=150, figsize=(12, 12), layout="constrained")
ax.imshow(
    phantom.T,
    cmap="gray_r",
    extent=(
        -fov_size_in_mm[0] / 2,
        fov_size_in_mm[0] / 2,
        -fov_size_in_mm[1] / 2,
        fov_size_in_mm[1] / 2,
    ),
    origin="lower",
    aspect="equal",
    interpolation="none",
    vmin=0,
    vmax=phantom.max()
)

# Ticks setup
# tick_spacing = 20  # mm
tick_spacing = 2.0                                        # sensible tick spacing for 10 mm
x_ticks = arange(
    -int(fov_size_in_mm[0] / 2), int(fov_size_in_mm[0] / 2) + 1, tick_spacing
)
y_ticks = arange(
    -int(fov_size_in_mm[1] / 2), int(fov_size_in_mm[1] / 2) + 1, tick_spacing
)
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
ax.set_xlabel("x (mm)", fontsize=12)
ax.set_ylabel("y (mm)", fontsize=12)
ax.grid(True, linestyle=":", alpha=0.5)

# Annotations
for i, rad in enumerate(angles):
    ax.annotate(
        anno_text[i],
        xy=(0, 0),
        xytext=tuple(anno_xy[i].tolist()),
        textcoords="data",
        fontsize=20,
        ha="center",
        va="center",
        color="w",
        bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="none", alpha=0.5),
    )
    
fig_title = f"Hot Rods Phantom, {fov_size_in_mm[0]} mm X {fov_size_in_mm[1]} mm"
fig_title += (
    f", {fov_n_pixels[0]} px X {fov_n_pixels[1]} px"
    f' ({fov_mm_per_pixel[0]:.4f} mm/px)'
)
fig.suptitle(fig_title, fontsize=20)

out_plot_filename = f"hot_rods_phantom_{fov_size_in_mm[0]}_mm_x_{fov_size_in_mm[1]}_mm.png"
fig.savefig(
    out_plot_filename,
    dpi=150,
)
print(f"Plot saved to {out_plot_filename}")