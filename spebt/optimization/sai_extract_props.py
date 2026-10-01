#!/usr/bin/env python3
"""
Extract beam properties (FWHM, angle, sensitivity) for SAI SC-SPECT with T8 aggregation.

Adapted from Kirtiraj's 4_extract_props.py:
  - FOV: 200×200, 0.05 mm/px
  - PPDFs: aggregated across 8 T8 poses per layout

Usage:
  python sai_extract_props.py --layout_idx 0 --work_dir <path> --tensor_file <path>
"""
import argparse
import os
import sys
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
    get_beams_weighted_center,
    get_beam_width,
    get_beams_angle_radian,
    get_beams_basic_properties,
    sample_ppdf_on_arc_2d_local,
)
from convex_hull_helper import convex_hull_2d, sort_points_for_hull_batch_2d
from geometry_2d_io import load_scanner_layout_geometries, load_scanner_layouts
from geometry_2d_utils import fov_tensor_dict, pixels_coordinates, pixels_to_detector_unit_rads
from beam_property_io import initialize_beam_properties_hdf5, append_to_hdf5_dataset, stack_beams_properties

# Reuse the T8 aggregation loader
from sai_extract_masks import load_aggregated_ppdfs, SAI_N_PIXELS, SAI_MM_PER_PIXEL


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout_idx", type=int, required=True)
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--tensor_file", type=str, required=True)
    args = parser.parse_args()

    print(f"[sai_extract_props] layout={args.layout_idx}")

    # Load geometry
    layouts, _ = load_scanner_layouts(os.path.dirname(args.tensor_file), os.path.basename(args.tensor_file))
    fov_dict = fov_tensor_dict(SAI_N_PIXELS, SAI_MM_PER_PIXEL, (0.0, 0.0))

    # Init output
    out_name = f"beams_properties_configuration_{args.layout_idx:03d}.hdf5"
    out_path = os.path.join(args.work_dir, out_name)
    if os.path.exists(out_path):
        os.remove(out_path)
    out_file, dset = initialize_beam_properties_hdf5(out_name, args.work_dir)

    # Load aggregated PPDFs
    ppdfs = load_aggregated_ppdfs(args.work_dir, args.layout_idx)

    # Load geometry
    plates_verts, det_verts = load_scanner_layout_geometries(args.layout_idx, layouts)
    det_centers = det_verts.mean(dim=1)

    fov_corners = tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * fov_dict["size in mm"] * 0.5
    hull_points = sort_points_for_hull_batch_2d(cat((
        fov_corners.unsqueeze(0).expand(det_verts.shape[0], -1, -1),
        det_centers.unsqueeze(1)
    ), dim=1))

    fov_xy = pixels_coordinates(fov_dict)
    n_det = det_verts.shape[0]

    for i in range(n_det):
        ppdf_2d = ppdfs[i].view(SAI_N_PIXELS[0], SAI_N_PIXELS[1])
        hull = convex_hull_2d(hull_points[i])
        sampled, rads, _ = sample_ppdf_on_arc_2d_local(ppdf_2d, det_centers[i], hull, fov_dict)
        bounds = beams_boundaries_radians(sampled, rads, threshold=0.01)
        fov_rads = pixels_to_detector_unit_rads(fov_xy, det_centers[i])
        masks = get_beams_masks(fov_rads, bounds)

        if masks.shape[0] == 0:
            continue

        weighted_centers = get_beams_weighted_center(masks, fov_xy, ppdf_2d)
        fwhm, _, _, _ = get_beam_width(weighted_centers, det_centers[i], masks, ppdf_2d, fov_dict)
        angles = get_beams_angle_radian(weighted_centers, det_centers[i])
        sizes, rel_sens, abs_sens = get_beams_basic_properties(masks, ppdf_2d, fov_xy)

        stacked = stack_beams_properties(args.layout_idx, i, angles, fwhm, sizes, rel_sens, abs_sens, weighted_centers)
        if stacked.numel():
            append_to_hdf5_dataset(dset, stacked)

        if (i + 1) % 500 == 0:
            print(f"  processed {i+1}/{n_det} detectors")

    out_file.close()
    print(f"  Saved {out_path}")
