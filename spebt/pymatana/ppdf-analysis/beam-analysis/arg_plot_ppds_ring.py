#!/usr/bin/env python3
import os
import sys
import h5py
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from geometry_2d_io import load_scanner_layout_geometries, load_scanner_layouts
from geometry_2d_utils import fov_tensor_dict

def _polycollection_from_vertices(vertices: torch.Tensor, **kwargs):
    if vertices is None or vertices.numel() == 0:
        return None
    return PolyCollection(vertices.cpu().tolist(), **kwargs)

def plot_ppds_maps_per_ring(
    layout_selector,                  # int layout index OR string 'AVG'
    scanner_layouts_dir: str,
    scanner_layouts_filename: str,
    ppds_dir: str,
    plot_dir: str,
    cmap: str = "viridis",
    overlay_geometry: bool = True,
    overlay_ring_only: bool = True,
):
    os.makedirs(plot_dir, exist_ok=True)

    # FOV info
    fov = fov_tensor_dict(n_pixels=(200, 200), mm_per_pixel=(0.05, 0.05), center_coordinates=(0.0, 0.0))
    width_mm  = float(fov["size in mm"][0].item())
    height_mm = float(fov["size in mm"][1].item())
    extent = (-width_mm/2, width_mm/2, -height_mm/2, height_mm/2)

    # ring slices (SC-SPECT ordering)
    RING_STARTS = [0, 480, 1200, 2160]
    RING_ENDS   = [480, 1200, 2160, 3360]

    # Geometry overlay
    layouts_data, _ = load_scanner_layouts(scanner_layouts_dir, scanner_layouts_filename)

    if isinstance(layout_selector, str) and layout_selector.upper() == "AVG":
        overlay_idx = 0
        plate_vertices, det_vertices = load_scanner_layout_geometries(overlay_idx, layouts_data)
        ppds_h5 = os.path.join(ppds_dir, "ppds_layout_AVG_perring.hdf5")
        nice_tag = "AVG"
    else:
        layout_idx = int(layout_selector)
        plate_vertices, det_vertices = load_scanner_layout_geometries(layout_idx, layouts_data)
        ppds_h5 = os.path.join(ppds_dir, f"ppds_layout_{layout_idx:03d}_perring.hdf5")
        nice_tag = f"{layout_idx:03d}"

    if not os.path.exists(ppds_h5):
        raise FileNotFoundError(f"Missing {ppds_h5}. Generate it with arg_compute_ppds_per_ring.py first.")

    with h5py.File(ppds_h5, "r") as f:
        PPDS_list = [torch.from_numpy(f[f"PPDS_ring{r}"][...]).to(torch.float64) for r in range(1, 5)]
        SENS_list = [torch.from_numpy(f[f"SENS_ring{r}"][...]).to(torch.float64) for r in range(1, 5)]
        V_i_list = []
        for r in range(1, 5):
            key = f"V_i_ring{r}"
            V_i_list.append(torch.from_numpy(f[key][...]) if key in f else None)

    def _plot_single(img: torch.Tensor, title: str, fname: str, ring_idx: int):
        fig, ax = plt.subplots(figsize=(7.2, 6.6), layout="constrained")
        im = ax.imshow(img.T.cpu().numpy(), extent=extent, origin="lower", cmap=cmap)

        if overlay_geometry:
            pc_plate = _polycollection_from_vertices(
                plate_vertices, facecolor="none", edgecolor="tab:red", linewidth=0.6, alpha=0.8
            )
            if pc_plate is not None:
                ax.add_collection(pc_plate)

            det_plot = det_vertices
            if overlay_ring_only and det_vertices is not None and det_vertices.numel() > 0:
                s = RING_STARTS[ring_idx]
                e = RING_ENDS[ring_idx]
                det_plot = det_vertices[s:e]

            pc_det = _polycollection_from_vertices(
                det_plot, facecolor="none", edgecolor="white", linewidth=0.4, alpha=0.7
            )
            if pc_det is not None:
                ax.add_collection(pc_det)

        fig.colorbar(im, ax=ax)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title(title)

        out_path = os.path.join(plot_dir, fname)
        fig.savefig(out_path, dpi=250)
        plt.close(fig)
        return out_path

    eps = 1e-12
    saved = []

    for r in range(4):
        PPDS = PPDS_list[r]
        SENS = SENS_list[r]

        PPDS_nz = torch.nan_to_num(PPDS, nan=0.0, posinf=0.0, neginf=0.0)
        ppds_max = float(PPDS_nz.max().item())
        PPDS_maxnorm = PPDS / max(ppds_max, eps)
        PPDS_relative = PPDS / (SENS + eps)

        ring_tag = f"{nice_tag}_ring{r+1}"

        saved.append(_plot_single(PPDS,
                                  f"PPDS map (layout {nice_tag}, ring {r+1})",
                                  f"ppds_map_layout_{ring_tag}.png",
                                  r))
        saved.append(_plot_single(PPDS_maxnorm,
                                  f"PPDS (max-normalized) (layout {nice_tag}, ring {r+1})",
                                  f"ppds_maxnorm_layout_{ring_tag}.png",
                                  r))
        saved.append(_plot_single(PPDS_relative,
                                  f"PPDS / SENS (relative) (layout {nice_tag}, ring {r+1})",
                                  f"ppds_relative_layout_{ring_tag}.png",
                                  r))

        summary = os.path.join(plot_dir, f"ppds_summary_layout_{ring_tag}.txt")
        with open(summary, "w") as f:
            f.write(f"Layout {nice_tag} Ring {r+1}\n")
            f.write(f"PPDS max      : {ppds_max:.6g}\n")
            if V_i_list[r] is not None:
                V_i = V_i_list[r]
                nz = V_i[V_i > 0]
                f.write(f"Nonzero V_i   : {int(nz.numel())} / {int(V_i.numel())}\n")
                if nz.numel():
                    f.write(f"V_i mean/median(min,max): "
                            f"{float(nz.mean()):.6g} / {float(nz.median()):.6g} "
                            f"({float(nz.min()):.6g}, {float(nz.max()):.6g})\n")
            else:
                f.write("V_i           : (not present)\n")
        saved.append(summary)

    print("Saved:")
    for p in saved:
        print("  ", p)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python arg_plot_ppds_per_ring.py <layout_idx|avg>")
        sys.exit(1)

    selector = sys.argv[1]
    try:
        layout_selector = int(selector)
    except ValueError:
        if selector.lower() in ("avg", "average", "mean"):
            layout_selector = "AVG"
        else:
            print("Error: layout_idx must be an integer or 'avg'.")
            sys.exit(1)

    # UPDATED: your rot2 tensor lives in /geometry
    SCANNER_LAYOUTS_DIR  = "/vscratch/grp-rutaoyao/Omer/spebt/geometry"
    SCANNER_LAYOUTS_FILE = "scanner_layouts_24a4365260a3f68491dfa8ca55e0ecc2_rot2_ang1p0deg_trans1x1_step0p0x0p0.tensor"
    PPDS_DIR             = "/vscratch/grp-rutaoyao/Omer/spebt/data/sai_10mm"
    PLOT_DIR             = "/vscratch/grp-rutaoyao/Omer/spebt/data/sai_10mm"

    print(f"--- Plotting per-ring PPDS for selector={selector} ---")
    plot_ppds_maps_per_ring(
        layout_selector,
        SCANNER_LAYOUTS_DIR,
        SCANNER_LAYOUTS_FILE,
        PPDS_DIR,
        PLOT_DIR,
        cmap="viridis",
        overlay_geometry=True,
        overlay_ring_only=True,
    )
    print("--- Done ---")