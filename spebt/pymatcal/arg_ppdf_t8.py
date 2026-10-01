'''
python arg_ppdf_t8.py 0 --layout_file /vscratch/grp-rutaoyao/Omer/spebt/geometry/scanner_layouts_24a4365260a3f68491dfa8ca55e0ecc2_rot2_ang1p0deg_trans1x1_step0p0x0p0.tensor --a_mm 0.8 --b_mm 0.8
python arg_ppdf_t8.py 1 --layout_file /vscratch/grp-rutaoyao/Omer/spebt/geometry/scanner_layouts_24a4365260a3f68491dfa8ca55e0ecc2_rot2_ang1p0deg_trans1x1_step0p0x0p0.tensor --a_mm 0.8 --b_mm 0.8
'''
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

def ellipse_offsets_t8(a_mm: float = 0.2, b_mm: float = 0.2, phase_deg: float = 0.0):
    """8 bed positions on an ellipse (a,b) in mm."""
    phase = np.deg2rad(phase_deg)
    thetas = np.linspace(0, 2*np.pi, 8, endpoint=False) + phase
    return [(float(a_mm*np.cos(t)), float(b_mm*np.sin(t))) for t in thetas]

def compute_pose(
    *,
    layout_idx: int,
    pose_idx: int,
    dx: float,
    dy: float,
    scanner_layouts,
    layouts_md5: str,
    output_dir: str,
):
    default_device = device("cpu")
    print(f"[T8 {pose_idx:02d}] dx={dx:.3f} mm dy={dy:.3f} mm")

    # materials for raytracing (keep consistent with your pipeline)
    mu_dict = tensor([3.5, 0.5], device=default_device)

    # keep these consistent with recon grid
    FOV_NPIX = (200, 200)
    FOV_SIZE_MM = (10, 10)
    SFOV_SUBDIV = (5, 5)
    CRYSTAL_SUBS = (1, 5)
    subdivision_grid = subdivision_grid_rectangle(CRYSTAL_SUBS)

    (
        plate_objects_vertices,
        crystal_objects_vertices,
        plate_objects_edges,
        crystal_objects_edges,
    ) = load_scanner_geometry_from_layout(layout_idx, scanner_layouts)

    n_crystals_total = int(crystal_objects_vertices.shape[0])
    crystal_idx_tensor = arange(n_crystals_total)

    # Translation implemented by shifting FOV center
    fov_dict = fov_tensor_dict(
        FOV_NPIX,          # pixels
        FOV_SIZE_MM,       # mm
        (dx, dy),          # shifted center in mm
        SFOV_SUBDIV,       # sfov grid
    )

    sfov_pxs_ids, sfov_pixels_batch, sfov_corners_batch = sfov_properties(fov_dict)
    fov_n_pxs = int(fov_dict["n pixels"].prod())
    n_sfov = int(fov_dict["n subdivisions"].prod())

    sfov_pxs_ids_1d = (
        sfov_pxs_ids[:, :, 0] * fov_dict["n pixels"][0] + sfov_pxs_ids[:, :, 1]
    )

    os.makedirs(output_dir, exist_ok=True)
    out_name = f"position_{layout_idx:03d}_ppdfs_t8_{pose_idx:02d}.hdf5"
    out_path = os.path.join(output_dir, out_name)

    print(f"  → writing {out_path}")
    print(f"  crystals={n_crystals_total} | sfov={n_sfov} | threads={get_num_threads()}")

    t0 = time.time()
    with h5py.File(out_path, "w") as h5file:
        # attrs for traceability
        h5file.attrs["layout_idx"] = int(layout_idx)
        h5file.attrs["layouts_md5"] = str(layouts_md5)
        h5file.attrs["pose_idx"] = int(pose_idx)
        h5file.attrs["dx_mm"] = float(dx)
        h5file.attrs["dy_mm"] = float(dy)
        h5file.attrs["pose_tag"] = "t8"

        dset = h5file.create_dataset("ppdfs", (n_crystals_total, fov_n_pxs), dtype="f")

        for dataset_idx, crystal_idx_tensor_val in enumerate(crystal_idx_tensor):
            crystal_idx = int(crystal_idx_tensor_val.item())

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
                dset[dataset_idx, sfov_pxs_ids_1d[sfov_idx]] = ppdf_slice.cpu().numpy()

            if (dataset_idx + 1) % 200 == 0:
                print(f"    computed {dataset_idx+1}/{n_crystals_total} crystals...")

    print(f"  done in {time.time()-t0:.2f} s")
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("layout_idx", type=int, help="layout index inside the .tensor file")
    ap.add_argument("--layout_file", type=str, required=True, help="path to scanner_layouts_*.tensor")
    ap.add_argument("--output_dir", type=str, default="/vscratch/grp-rutaoyao/Omer/spebt/data/sai_10mm")
    ap.add_argument("--a_mm", type=float, default=0.2, help="ellipse semi-axis a (X) in mm")
    ap.add_argument("--b_mm", type=float, default=0.2, help="ellipse semi-axis b (Y) in mm")
    ap.add_argument("--phase_deg", type=float, default=0.0, help="phase rotate the 8 positions (deg)")
    ap.add_argument("--pose_idx", type=int, default=None,
                    help="run only this single pose index (0-7). If omitted, run all 8.")
    args = ap.parse_args()

    layout_dir = os.path.dirname(args.layout_file)
    layout_fname = os.path.basename(args.layout_file)
    scanner_layouts, layouts_md5 = load_scanner_layouts(layout_dir, layout_fname)

    if not (0 <= args.layout_idx < len(scanner_layouts)):
        raise ValueError(f"layout_idx={args.layout_idx} out of range 0..{len(scanner_layouts)-1}")

    poses = ellipse_offsets_t8(args.a_mm, args.b_mm, args.phase_deg)

    if args.pose_idx is not None:
        # Single pose mode
        if not (0 <= args.pose_idx < len(poses)):
            raise ValueError(f"pose_idx={args.pose_idx} out of range 0..{len(poses)-1}")
        print(f"--- T8 PPDF | layout={args.layout_idx} | pose={args.pose_idx} | a={args.a_mm} b={args.b_mm} ---")
        dx, dy = poses[args.pose_idx]
        compute_pose(
            layout_idx=args.layout_idx,
            pose_idx=args.pose_idx,
            dx=dx, dy=dy,
            scanner_layouts=scanner_layouts,
            layouts_md5=layouts_md5,
            output_dir=args.output_dir,
        )
    else:
        # All 8 poses (original behavior)
        print(f"--- T8 PPDFs | layout={args.layout_idx} | a={args.a_mm} b={args.b_mm} | poses=8 ---")
        for i, (dx, dy) in enumerate(poses):
            compute_pose(
                layout_idx=args.layout_idx,
                pose_idx=i,
                dx=dx, dy=dy,
                scanner_layouts=scanner_layouts,
                layouts_md5=layouts_md5,
                output_dir=args.output_dir,
            )

if __name__ == "__main__":
    main()