#!/usr/bin/env python3
import os, sys, h5py, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from geometry_2d_io import load_scanner_layout_geometries, load_scanner_layouts
from geometry_2d_utils import fov_tensor_dict

def _polycollection_from_vertices(vertices: torch.Tensor, **kwargs):
    if vertices is None or vertices.numel() == 0:
        return None
    return PolyCollection(vertices.cpu().tolist(), **kwargs)

def plot_ppds_maps_t8(
    layout_selector,                  # int layout index OR string 'AVG'
    scanner_layouts_dir: str,
    scanner_layouts_filename: str,
    ppds_dir: str,
    plot_dir: str,
    cmap: str = "viridis",
    overlay_geometry: bool = True,
):
    os.makedirs(plot_dir, exist_ok=True)

    # FOV info (canonical)
    fov = fov_tensor_dict(n_pixels=(200, 200), mm_per_pixel=(0.05, 0.05), center_coordinates=(0.0, 0.0))
    width_mm  = float(fov["size in mm"][0].item())
    height_mm = float(fov["size in mm"][1].item())
    extent = (-width_mm/2, width_mm/2, -height_mm/2, height_mm/2)

    # Geometry overlay: for AVG we just show layout 0 as a reference
    layouts_data, _ = load_scanner_layouts(scanner_layouts_dir, scanner_layouts_filename)
    if isinstance(layout_selector, str) and layout_selector.upper() == "AVG":
        overlay_idx = 0
        plate_vertices, det_vertices = load_scanner_layout_geometries(overlay_idx, layouts_data)
        ppds_h5 = os.path.join(ppds_dir, "ppds_layout_AVG_t8.hdf5")
        nice_tag = "AVG_T8"
    else:
        layout_idx = int(layout_selector)
        plate_vertices, det_vertices = load_scanner_layout_geometries(layout_idx, layouts_data)
        ppds_h5 = os.path.join(ppds_dir, f"ppds_layout_{layout_idx:03d}_t8.hdf5")
        nice_tag = f"{layout_idx:03d}_T8"

    if not os.path.exists(ppds_h5):
        raise FileNotFoundError(f"Missing {ppds_h5}. Generate it with arg_compute_ppds_t8.py first.")

    with h5py.File(ppds_h5, "r") as f:
        PPDS = torch.from_numpy(f["PPDS"][...]).to(torch.float64)   # (H,W)
        SENS = torch.from_numpy(f["SENS"][...]).to(torch.float64)
        # V_i likely absent for AVG; handle gracefully
        V_i  = torch.from_numpy(f["V_i"][...]) if "V_i" in f else None

    eps = 1e-12
    PPDS_nz  = torch.nan_to_num(PPDS, nan=0.0, posinf=0.0, neginf=0.0)
    ppds_max = float(PPDS_nz.max().item())

    PPDS_maxnorm  = PPDS / max(ppds_max, eps)
    PPDS_relative = PPDS / (SENS + eps)

    def _plot_single(img: torch.Tensor, title: str, fname: str):
        fig, ax = plt.subplots(figsize=(7.2, 6.6), layout="constrained")
        im = ax.imshow(img.T.cpu().numpy(), extent=extent, origin="lower", cmap=cmap)

        if overlay_geometry:
            pc_plate = _polycollection_from_vertices(
                plate_vertices, facecolor="none", edgecolor="tab:red", linewidth=0.6, alpha=0.8
            )
            pc_det = _polycollection_from_vertices(
                det_vertices, facecolor="none", edgecolor="white", linewidth=0.4, alpha=0.7
            )
            if pc_plate: ax.add_collection(pc_plate)
            if pc_det:   ax.add_collection(pc_det)

        fig.colorbar(im, ax=ax)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title(title)

        out_path = os.path.join(plot_dir, fname)
        fig.savefig(out_path, dpi=250)
        plt.close(fig)
        return out_path

    paths = []
    paths.append(_plot_single(
        PPDS,
        f"PPDS map (layout {nice_tag})",
        f"ppds_map_layout_{nice_tag}.png"
    ))
    paths.append(_plot_single(
        PPDS_maxnorm,
        f"PPDS (max-normalized) (layout {nice_tag})",
        f"ppds_maxnorm_layout_{nice_tag}.png"
    ))
    paths.append(_plot_single(
        PPDS_relative,
        f"PPDS / SENS (relative) (layout {nice_tag})",
        f"ppds_relative_layout_{nice_tag}.png"
    ))

    summary = os.path.join(plot_dir, f"ppds_summary_layout_{nice_tag}.txt")
    with open(summary, "w") as f:
        f.write(f"Layout {nice_tag}\n")
        f.write(f"PPDS max      : {ppds_max:.6g}\n")
        if V_i is not None:
            nz = V_i[V_i > 0]
            f.write(f"Nonzero V_i   : {int(nz.numel())} / {V_i.numel()}\n")
            if nz.numel():
                f.write(
                    f"V_i mean/median(min,max): "
                    f"{float(nz.mean()):.6g} / {float(nz.median()):.6g} "
                    f"({float(nz.min()):.6g}, {float(nz.max()):.6g})\n"
                )
        else:
            f.write("V_i           : (not present for AVG map)\n")

    print("Saved:")
    for p in paths:
        print("  ", p)
    print("  ", summary)

if __name__ == "__main__":
    # Usage:
    #   python arg_plot_ppds_maps_t8.py <layout_idx>
    #   python arg_plot_ppds_maps_t8.py avg
    if len(sys.argv) != 2:
        print("Usage: python arg_plot_ppds_maps_t8.py <layout_idx|avg>")
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

    SCANNER_LAYOUTS_DIR  = "/vscratch/grp-rutaoyao/Omer/spebt/data/sai_10mm"
    SCANNER_LAYOUTS_FILE = "scanner_layouts_24a4365260a3f68491dfa8ca55e0ecc2_rot2_ang1p0deg_trans1x1_step0p0x0p0.tensor"
    PPDS_DIR             = "/vscratch/grp-rutaoyao/Omer/spebt/data/sai_10mm"
    PLOT_DIR             = "/vscratch/grp-rutaoyao/Omer/spebt/data/sai_10mm"

    print(f"--- [START] Plotting PPDS T8 for {selector} ---")
    plot_ppds_maps_t8(
        layout_selector,
        SCANNER_LAYOUTS_DIR,
        SCANNER_LAYOUTS_FILE,
        PPDS_DIR,
        PLOT_DIR,
        cmap="viridis",
        overlay_geometry=True,
    )
    print(f"--- [ END ] Done plotting PPDS T8 for {selector} ---")