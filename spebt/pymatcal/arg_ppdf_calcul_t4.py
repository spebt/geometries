'''
python arg_ppdf_calcul_t4.py 0 --t4
python arg_ppdf_calcul_t4.py 1 --t4
'''
#!/usr/bin/env python3
import os
import sys
import time
import argparse
import h5py
import torch
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

# Paper T4 positions (x,y) in mm (2D only)
T4_OFFSETS_XY = [
    (-0.4, -0.4),
    ( 0.4,  0.4),
    (-0.4,  0.4),
    ( 0.4, -0.4),
]


def _compute_one_pose_to_file(
    *,
    layout_idx: int,
    pose_idx: int,
    dx: float,
    dy: float,
    scanner_layouts,
    layouts_md5,
    output_dir: str,
):
    """Compute PPDFs for a single (dx,dy) pose and write to one HDF5 file."""
    default_device = device("cpu")

    mu_dict = tensor([3.5, 0.5], device=default_device)
    crystal_n_subs = (5, 5)
    subdivision_grid = subdivision_grid_rectangle(crystal_n_subs)

    # Geometry for this layout (fixed)
    (
        plate_objects_vertices,
        crystal_objects_vertices,
        plate_objects_edges,
        crystal_objects_edges,
    ) = load_scanner_geometry_from_layout(layout_idx, scanner_layouts)

    n_crystals_total = int(crystal_objects_vertices.shape[0])
    crystal_idx_tensor = arange(n_crystals_total)

    # Shift the FOV center (implements the transverse translation)
    fov_dict = fov_tensor_dict(
        (200, 200),       # n_pixels
        (10, 10),         # size in mm
        (dx, dy),         # center shift in mm (T4 translation)
        (5, 5),           # n_subdivisions (sfov)
    )

    sfov_pxs_ids, sfov_pixels_batch, sfov_corners_batch = sfov_properties(fov_dict)
    fov_n_pxs = int(fov_dict["n pixels"].prod())
    n_sfov = int(fov_dict["n subdivisions"].prod())

    sfov_pxs_ids_1d = (
        sfov_pxs_ids[:, :, 0] * fov_dict["n pixels"][0] + sfov_pxs_ids[:, :, 1]
    )

    os.makedirs(output_dir, exist_ok=True)
    out_name = f"position_{layout_idx:03d}_ppdfs_t4_{pose_idx:02d}.hdf5"
    h5_file_path = os.path.join(output_dir, out_name)

    print(f"[POSE {pose_idx:02d}] Writing → {h5_file_path}")
    print(f"[POSE {pose_idx:02d}] dx={dx} mm, dy={dy} mm | crystals={n_crystals_total} | sfov={n_sfov}")

    pose_start = time.time()
    with h5py.File(h5_file_path, "w") as h5file:
        # metadata
        h5file.attrs["layout_idx"] = int(layout_idx)
        h5file.attrs["layouts_md5"] = str(layouts_md5)
        h5file.attrs["pose_idx"] = int(pose_idx)
        h5file.attrs["dx_mm"] = float(dx)
        h5file.attrs["dy_mm"] = float(dy)
        h5file.attrs["use_t4"] = 1

        # dataset
        dset = h5file.create_dataset("ppdfs", (n_crystals_total, fov_n_pxs), dtype="f")

        # Compute PPDFs for each crystal
        for dataset_idx, crystal_idx_tensor_val in enumerate(crystal_idx_tensor):
            crystal_idx = int(crystal_idx_tensor_val.item())

            # Reduce edges per SFOV (depends on sfov_corners -> depends on dx,dy)
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

            # Raytrace each SFOV slice into the correct pixels
            for sfov_idx in range(n_sfov):
                ppdf_slice = ppdf_2d_local(
                    sfov_idx, crystal_idx, sfov_pixels_batch,
                    crystal_objects_vertices, reduced_plate_edges_sfovs[sfov_idx],
                    reduced_crystal_edges_sfovs[sfov_idx], subdivision_grid,
                    mu_dict, default_device,
                )
                dset[dataset_idx, sfov_pxs_ids_1d[sfov_idx]] = ppdf_slice.cpu().numpy()

            if (dataset_idx + 1) % 200 == 0:
                print(f"[POSE {pose_idx:02d}] computed {dataset_idx+1}/{n_crystals_total} crystals...")

    pose_dur = time.time() - pose_start
    print(f"[POSE {pose_idx:02d}] Done in {pose_dur:.2f} s")
    return h5_file_path


def calculate_ppdf_for_layout(layout_idx: int, use_t4: bool):
    start_time = time.time()
    print(f"--- Starting PPDF calculation for Layout Index: {layout_idx} ---")
    print(f"[INFO] use_t4 = {use_t4}")
    print(f"PyTorch using {get_num_threads()} threads.")

    scanner_layout_file = (
        "/vscratch/grp-rutaoyao/Omer/spebt/geometry/"
        "/scanner_layouts_24a4365260a3f68491dfa8ca55e0ecc2_rot2_ang1p0deg_trans1x1_step0p0x0p0.tensor"
    )
    scanner_layout_dir = os.path.dirname(scanner_layout_file)
    scanner_layout_filename = os.path.basename(scanner_layout_file)

    scanner_layouts, layouts_md5 = load_scanner_layouts(scanner_layout_dir, scanner_layout_filename)

    n_layouts_total = len(scanner_layouts)
    if not (0 <= layout_idx < n_layouts_total):
        raise ValueError(
            f"Invalid layout_idx {layout_idx}. File contains {n_layouts_total} layouts (0..{n_layouts_total-1})."
        )

    # Decide which poses to run
    poses_xy = T4_OFFSETS_XY if use_t4 else [(0.0, 0.0)]
    print(f"[INFO] Number of poses to compute: {len(poses_xy)}")
    for pi, (dx, dy) in enumerate(poses_xy):
        print(f"  pose {pi}: dx={dx} mm, dy={dy} mm")

    output_dir = "/vscratch/grp-rutaoyao/Omer/spebt/data/sai_10mm"

    out_files = []
    for pose_idx, (dx, dy) in enumerate(poses_xy):
        # for non-t4, keep your old naming
        if not use_t4:
            # write the single "non-shifted" file as position_XXX_ppdfs.hdf5
            # (pose_idx is always 0 here)
            default_device = device("cpu")
            mu_dict = tensor([3.5, 0.5], device=default_device)
            crystal_n_subs = (5, 5)
            subdivision_grid = subdivision_grid_rectangle(crystal_n_subs)

            (
                plate_objects_vertices,
                crystal_objects_vertices,
                plate_objects_edges,
                crystal_objects_edges,
            ) = load_scanner_geometry_from_layout(layout_idx, scanner_layouts)

            n_crystals_total = int(crystal_objects_vertices.shape[0])
            crystal_idx_tensor = arange(n_crystals_total)

            fov_dict = fov_tensor_dict((200, 200), (10, 10), (0.0, 0.0), (3, 3))
            sfov_pxs_ids, sfov_pixels_batch, sfov_corners_batch = sfov_properties(fov_dict)
            fov_n_pxs = int(fov_dict["n pixels"].prod())
            n_sfov = int(fov_dict["n subdivisions"].prod())
            sfov_pxs_ids_1d = (
                sfov_pxs_ids[:, :, 0] * fov_dict["n pixels"][0] + sfov_pxs_ids[:, :, 1]
            )

            os.makedirs(output_dir, exist_ok=True)
            h5_file_path = os.path.join(output_dir, f"position_{layout_idx:03d}_ppdfs.hdf5")
            print(f"[POSE 00] Writing → {h5_file_path}")

            pose_start = time.time()
            with h5py.File(h5_file_path, "w") as h5file:
                h5file.attrs["layout_idx"] = int(layout_idx)
                h5file.attrs["layouts_md5"] = str(layouts_md5)
                h5file.attrs["pose_idx"] = 0
                h5file.attrs["dx_mm"] = 0.0
                h5file.attrs["dy_mm"] = 0.0
                h5file.attrs["use_t4"] = 0

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
                            crystal_objects_vertices, reduced_plate_edges_sfovs[sfov_idx],
                            reduced_crystal_edges_sfovs[sfov_idx], subdivision_grid,
                            mu_dict, default_device,
                        )
                        dset[dataset_idx, sfov_pxs_ids_1d[sfov_idx]] = ppdf_slice.cpu().numpy()

                    if (dataset_idx + 1) % 200 == 0:
                        print(f"[POSE 00] computed {dataset_idx+1}/{n_crystals_total} crystals...")

            pose_dur = time.time() - pose_start
            print(f"[POSE 00] Done in {pose_dur:.2f} s")
            out_files.append(h5_file_path)
        else:
            out_files.append(
                _compute_one_pose_to_file(
                    layout_idx=layout_idx,
                    pose_idx=pose_idx,
                    dx=dx,
                    dy=dy,
                    scanner_layouts=scanner_layouts,
                    layouts_md5=layouts_md5,
                    output_dir=output_dir,
                )
            )

    duration = time.time() - start_time
    print(f"--- Finished Layout {layout_idx} in {duration:.2f} seconds. ---")
    print("Outputs:")
    for p in out_files:
        print("  ", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("layout_idx", type=int, help="layout index to process")
    ap.add_argument("--t4", action="store_true",
                    help="compute 4 transverse T4 shifts and write 4 separate HDF5 files")
    args = ap.parse_args()

    calculate_ppdf_for_layout(args.layout_idx, use_t4=args.t4)