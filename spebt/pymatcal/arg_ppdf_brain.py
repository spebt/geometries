"""
arg_ppdf_brain.py

Computes the system matrix (ppdfs) for the Brain SPECT polygon-panel system.

Key differences from arg_ppdf_t8.py:
  - FOV: 200x200 mm at 1mm/voxel (brain scale, vs 10x10mm for preclinical)
  - No T8 elliptical bed motion — the collimator ring rotates instead
    (motion is encoded in the layout file, not in source displacement)
  - Source stays at FOV centre; each layout in the .tensor file has a
    different collimator rotation baked in
  - Output filename: brain_spect_ppdfs_layout_{layout_idx:03d}.hdf5

Usage:
  python arg_ppdf_brain.py 0 \
      --layout_file /path/to/brain_spect_layouts_<md5>_rot2_ang1p0deg_trans1x1_step0p0x0p0.tensor \
      --output_dir  /vscratch/grp-rutaoyao/Tridev/brain_spect/data

  # For single-panel mode (compute one panel, rotate for rest):
  python arg_ppdf_brain.py 0 \
      --layout_file /path/to/brain_spect_layouts_*.tensor \
      --output_dir  /vscratch/grp-rutaoyao/Tridev/brain_spect/data \
      --single_panel 0
"""

#!/usr/bin/env python3
import os
import time
import argparse
import h5py
import numpy as np
from torch import device, arange, tensor, get_num_threads

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


def compute_brain_layout(
    *,
    layout_idx: int,
    scanner_layouts,
    layouts_md5: str,
    output_dir: str,
    fov_npix: tuple = (200, 200),
    fov_size_mm: tuple = (200.0, 200.0),
    sfov_subdiv: tuple = (2, 2),
    crystal_subs: tuple = (1, 1),
    single_panel: int = None,
):
    """
    Compute system matrix for one layout (one collimator rotation position).

    Parameters
    ----------
    layout_idx      : which position in the .tensor file to compute
    scanner_layouts : loaded scanner layout dict (from load_scanner_layouts)
    layouts_md5     : geometry hash for traceability
    output_dir      : where to write the HDF5 file
    fov_npix        : (nx, ny) number of FOV voxels
    fov_size_mm     : (Lx, Ly) physical FOV size in mm
    sfov_subdiv     : (sx, sy) sub-FOV subdivision grid for accuracy
    crystal_subs    : (cx, cy) crystal subdivision for accuracy
    single_panel    : if set, compute only crystals belonging to this panel
                      index and write a partial HDF5 (for distributed runs)
    """
    default_device = device("cpu")
    print(f"[Brain SPECT] layout={layout_idx} | FOV={fov_npix} @ {fov_size_mm} mm")

    # Attenuation coefficients: [tungsten, scintillator]
    # Same convention as arg_ppdf_t8.py
    mu_dict = tensor([3.5, 0.5], device=default_device)

    subdivision_grid = subdivision_grid_rectangle(crystal_subs)

    (
        plate_objects_vertices,
        crystal_objects_vertices,
        plate_objects_edges,
        crystal_objects_edges,
    ) = load_scanner_geometry_from_layout(layout_idx, scanner_layouts)

    n_crystals_total = int(crystal_objects_vertices.shape[0])
    print(f"  Total crystals in layout: {n_crystals_total}")

    # Determine which crystals to compute
    if single_panel is not None:
        # Single-panel mode: only compute crystals for the requested panel.
        # Crystals are ordered panel-major in generate_brain_spect_polygon.py,
        # so we can slice them out by panel index.
        # Assumes equal crystals per panel.
        n_per_panel = n_crystals_total  # will be divided below if panels > 1
        # We need to know n_panels — read from scanner_layouts metadata if present
        n_panels = 6  # default; update if geometry changes
        n_per_panel = n_crystals_total // n_panels
        crystal_start = single_panel * n_per_panel
        crystal_end   = crystal_start + n_per_panel
        crystal_indices = arange(crystal_start, crystal_end)
        print(f"  Single-panel mode: panel={single_panel} "
              f"crystals {crystal_start}..{crystal_end-1} ({n_per_panel} crystals)")
        out_suffix = f"panel{single_panel:02d}"
    else:
        crystal_indices = arange(n_crystals_total)
        out_suffix = "all"
        print(f"  Full mode: computing all {n_crystals_total} crystals")

    # FOV — no source displacement for brain SPECT (collimator rotates instead)
    fov_dict = fov_tensor_dict(
        fov_npix,
        fov_size_mm,
        (0.0, 0.0),    # source at FOV centre — no T8 bed motion
        sfov_subdiv,
    )

    sfov_pxs_ids, sfov_pixels_batch, sfov_corners_batch = sfov_properties(fov_dict)
    fov_n_pxs = int(fov_dict["n pixels"].prod())
    n_sfov = int(fov_dict["n subdivisions"].prod())

    sfov_pxs_ids_1d = (
        sfov_pxs_ids[:, :, 0] * fov_dict["n pixels"][0] + sfov_pxs_ids[:, :, 1]
    )

    os.makedirs(output_dir, exist_ok=True)
    out_name = f"brain_spect_ppdfs_layout{layout_idx:03d}_{out_suffix}.hdf5"
    out_path = os.path.join(output_dir, out_name)

    n_crystals_this = int(crystal_indices.shape[0])
    print(f"  → writing {out_path}")
    print(f"  crystals={n_crystals_this} | fov_pxs={fov_n_pxs} | sfov={n_sfov} | threads={get_num_threads()}")

    t0 = time.time()
    with h5py.File(out_path, "w") as h5file:
        h5file.attrs["layout_idx"]     = int(layout_idx)
        h5file.attrs["layouts_md5"]    = str(layouts_md5)
        h5file.attrs["fov_npix_x"]     = int(fov_npix[0])
        h5file.attrs["fov_npix_y"]     = int(fov_npix[1])
        h5file.attrs["fov_size_mm_x"]  = float(fov_size_mm[0])
        h5file.attrs["fov_size_mm_y"]  = float(fov_size_mm[1])
        h5file.attrs["single_panel"]   = int(single_panel) if single_panel is not None else -1
        h5file.attrs["pose_tag"]       = "collimator_rotation"

        # Dataset shape: (n_crystals_this, fov_n_pxs)
        # For full mode: first dim = n_crystals_total
        # For single-panel: first dim = n_per_panel
        dset = h5file.create_dataset("ppdfs", (n_crystals_this, fov_n_pxs), dtype="f")

        for local_idx, crystal_idx_val in enumerate(crystal_indices):
            crystal_idx = int(crystal_idx_val.item())

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
                    crystal_objects_vertices,
                    reduced_plate_edges_sfovs[sfov_idx],
                    reduced_crystal_edges_sfovs[sfov_idx],
                    subdivision_grid,
                    mu_dict,
                    default_device,
                )
                dset[local_idx, sfov_pxs_ids_1d[sfov_idx]] = ppdf_slice.cpu().numpy()

            if (local_idx + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (local_idx + 1) / elapsed
                eta = (n_crystals_this - local_idx - 1) / rate if rate > 0 else 0
                print(f"    {local_idx+1}/{n_crystals_this} crystals "
                      f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

    print(f"  done in {time.time()-t0:.2f} s  →  {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Compute Brain SPECT system matrix (polygon-panel, cylindrical collimator)"
    )
    ap.add_argument("layout_idx", type=int,
                    help="Layout index inside the .tensor file")
    ap.add_argument("--layout_file", type=str, required=True,
                    help="Path to brain_spect_layouts_*.tensor")
    ap.add_argument("--output_dir", type=str,
                    default="/vscratch/grp-rutaoyao/Tridev/brain_spect/data",
                    help="Output directory for HDF5 files")

    # FOV parameters
    ap.add_argument("--fov_npix", type=int, nargs=2, default=[200, 200],
                    metavar=("NX", "NY"),
                    help="FOV voxel count (default: 200 200)")
    ap.add_argument("--fov_size_mm", type=float, nargs=2, default=[200.0, 200.0],
                    metavar=("LX", "LY"),
                    help="FOV physical size in mm (default: 200.0 200.0)")

    # Accuracy parameters
    ap.add_argument("--sfov_subdiv", type=int, nargs=2, default=[2, 2],
                    metavar=("SX", "SY"),
                    help="Sub-FOV subdivision for accuracy (default: 2 2)")
    ap.add_argument("--crystal_subs", type=int, nargs=2, default=[1, 1],
                    metavar=("CX", "CY"),
                    help="Crystal subdivision (default: 1 1)")

    # Single-panel distributed mode
    ap.add_argument("--single_panel", type=int, default=None,
                    help="Compute only this panel index (0-based). "
                         "If omitted, compute all panels.")

    args = ap.parse_args()

    layout_dir   = os.path.dirname(args.layout_file)
    layout_fname = os.path.basename(args.layout_file)
    scanner_layouts, layouts_md5 = load_scanner_layouts(layout_dir, layout_fname)

    n_layouts = len([k for k in scanner_layouts.keys() if k.startswith("position")])
    if not (0 <= args.layout_idx < n_layouts):
        raise ValueError(
            f"layout_idx={args.layout_idx} out of range 0..{n_layouts-1}"
        )

    print(f"--- Brain SPECT PPDF | layout={args.layout_idx} "
          f"| single_panel={args.single_panel} ---")

    compute_brain_layout(
        layout_idx=args.layout_idx,
        scanner_layouts=scanner_layouts,
        layouts_md5=layouts_md5,
        output_dir=args.output_dir,
        fov_npix=tuple(args.fov_npix),
        fov_size_mm=tuple(args.fov_size_mm),
        sfov_subdiv=tuple(args.sfov_subdiv),
        crystal_subs=tuple(args.crystal_subs),
        single_panel=args.single_panel,
    )


if __name__ == "__main__":
    main()
