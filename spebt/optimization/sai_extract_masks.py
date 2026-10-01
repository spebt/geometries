#!/usr/bin/env python3
"""
Extract beam masks for SAI SC-SPECT with T8 aggregation.

Adapted from Kirtiraj's 3_extract_masks.py:
  - FOV: 200×200, 0.05 mm/px (not 280×280, 0.25 mm/px)
  - PPDFs: aggregated across 8 T8 poses per layout
  - PPDF filename: position_NNN_ppdfs_t8_PP.hdf5

Usage:
  python sai_extract_masks.py --layout_idx 0 --work_dir <path> --tensor_file <path>
"""
import argparse
import os
import sys
import h5py
import torch
from torch import cat, tensor

# Add beam-analysis library to path
BEAM_ANALYSIS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "pymatana", "ppdf-analysis", "beam-analysis"
)
sys.path.insert(0, BEAM_ANALYSIS_DIR)

from beam_property_extract import (
    beams_boundaries_radians,
    get_beams_masks,
    get_beams_combined_mask,
    sample_ppdf_on_arc_2d_local,
)
from convex_hull_helper import convex_hull_2d, sort_points_for_hull_batch_2d
from geometry_2d_io import load_scanner_layout_geometries, load_scanner_layouts
from geometry_2d_utils import fov_tensor_dict, pixels_coordinates, pixels_to_detector_unit_rads
from beam_property_io import initialize_beam_masks_hdf5, append_to_hdf5_dataset

# SAI FOV: 200×200 px, 10×10 mm → 0.05 mm/px
SAI_N_PIXELS = (200, 200)
SAI_MM_PER_PIXEL = (0.05, 0.05)
N_T8_POSES = 8


def load_aggregated_ppdfs(work_dir, layout_idx):
    """Load and sum PPDFs across 8 T8 poses for one layout."""
    aggregated = None
    for pose_idx in range(N_T8_POSES):
        fname = f"position_{layout_idx:03d}_ppdfs_t8_{pose_idx:02d}.hdf5"
        fpath = os.path.join(work_dir, fname)
        if not os.path.exists(fpath):
            print(f"[warn] Missing {fname}")
            continue
        with h5py.File(fpath, "r") as f:
            ppdfs = torch.tensor(f["ppdfs"][:], dtype=torch.float32)
        if aggregated is None:
            aggregated = ppdfs
        else:
            aggregated += ppdfs
    if aggregated is None:
        raise FileNotFoundError(f"No T8 PPDF files found for layout {layout_idx} in {work_dir}")
    print(f"  Aggregated {N_T8_POSES} T8 poses for layout {layout_idx}: shape {tuple(aggregated.shape)}")
    return aggregated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout_idx", type=int, required=True)
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--tensor_file", type=str, required=True)
    args = parser.parse_args()

    print(f"[sai_extract_masks] layout={args.layout_idx}")

    # Load geometry
    layouts, _ = load_scanner_layouts(os.path.dirname(args.tensor_file), os.path.basename(args.tensor_file))
    fov_dict = fov_tensor_dict(SAI_N_PIXELS, SAI_MM_PER_PIXEL, (0.0, 0.0))

    # Init output
    out_name = f"beams_masks_configuration_{args.layout_idx:03d}.hdf5"
    out_path = os.path.join(args.work_dir, out_name)
    if os.path.exists(out_path):
        os.remove(out_path)  # overwrite if re-running
    out_file, dset = initialize_beam_masks_hdf5(int(fov_dict["n pixels"].prod()), out_name, args.work_dir)

    # Load aggregated PPDFs (sum over 8 T8 poses)
    ppdfs = load_aggregated_ppdfs(args.work_dir, args.layout_idx)

    # Load geometry for this layout
    plates_verts, det_verts = load_scanner_layout_geometries(args.layout_idx, layouts)
    det_centers = det_verts.mean(dim=1)

    fov_corners = tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * fov_dict["size in mm"] * 0.5
    hull_points = cat((
        fov_corners.unsqueeze(0).expand(det_verts.shape[0], -1, -1),
        det_centers.unsqueeze(1)
    ), dim=1)
    hull_points = sort_points_for_hull_batch_2d(hull_points)

    n_det = det_verts.shape[0]
    for i in range(n_det):
        ppdf_2d = ppdfs[i].view(SAI_N_PIXELS[0], SAI_N_PIXELS[1])
        hull = convex_hull_2d(hull_points[i])
        sampled, rads, _ = sample_ppdf_on_arc_2d_local(ppdf_2d, det_centers[i], hull, fov_dict)
        bounds = beams_boundaries_radians(sampled, rads, threshold=0.01)
        fov_rads = pixels_to_detector_unit_rads(pixels_coordinates(fov_dict), det_centers[i])
        masks = get_beams_masks(fov_rads, bounds)
        append_to_hdf5_dataset(dset, get_beams_combined_mask(masks))

        if (i + 1) % 500 == 0:
            print(f"  processed {i+1}/{n_det} detectors")

    out_file.close()
    print(f"  Saved {out_path}")
