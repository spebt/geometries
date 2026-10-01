#!/usr/bin/env python3
"""
Compute ASCI histogram for SAI SC-SPECT.

Adapted from Kirtiraj's 5_analyze_asci.py:
  - FOV: 200×200 (not 280×280)
  - Reads beam properties and masks produced by sai_extract_masks.py / sai_extract_props.py

Usage:
  python sai_analyze_asci.py --layout_idx 0 --work_dir <path>
"""
import argparse
import os
import torch
import h5py

# SAI constants
SAI_N_PIXELS = (200, 200)
N_BINS = 360

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout_idx", type=int, required=True)
    parser.add_argument("--work_dir", type=str, required=True)
    args = parser.parse_args()

    print(f"[sai_analyze_asci] layout={args.layout_idx}")

    n_bins = N_BINS
    angular_bin_boundaries = torch.arange(n_bins + 1) / 180 * torch.pi

    # Load beam properties
    props_file = os.path.join(args.work_dir, f"beams_properties_configuration_{args.layout_idx:03d}.hdf5")
    with h5py.File(props_file, "r") as f:
        layout_beams_properties = torch.from_numpy(f["beam_properties"][:])

    # Load beam masks
    masks_file = os.path.join(args.work_dir, f"beams_masks_configuration_{args.layout_idx:03d}.hdf5")
    with h5py.File(masks_file, "r") as f:
        beams_masks = torch.from_numpy(f["beam_mask"][:])

    # Digitize angles into bins
    digitized_angles = torch.bucketize(layout_beams_properties[:, 3], angular_bin_boundaries, right=False)
    layout_beams_properties = torch.cat(
        (layout_beams_properties, (digitized_angles - 1).unsqueeze(1).float()), dim=1
    )

    # Filter out NaN angles
    layout_beams_properties_filtered = layout_beams_properties[
        torch.isnan(layout_beams_properties[:, 3]) == False
    ]

    # Filter by sensitivity threshold (1% of max)
    if layout_beams_properties_filtered[:, 7].numel() > 0:
        beams_sensitivity_max = layout_beams_properties_filtered[:, 7].max()
        layout_beams_properties_filtered = layout_beams_properties_filtered[
            layout_beams_properties_filtered[:, 7] > beams_sensitivity_max * 0.01
        ]

    # Build ASCI histogram: (n_fov_pixels, n_angular_bins)
    n_fov = SAI_N_PIXELS[0] * SAI_N_PIXELS[1]
    asci_histogram = torch.zeros((n_fov, n_bins), dtype=torch.int32)

    for beam_props in layout_beams_properties_filtered:
        detector_idx = int(beam_props[1])
        beam_idx = int(beam_props[2])
        angle_bin_idx = int(beam_props[-1])
        if 0 <= angle_bin_idx < n_bins:
            asci_histogram[beams_masks[detector_idx] == beam_idx, angle_bin_idx] += 1

    # Save
    out_path = os.path.join(args.work_dir, f"asci_histogram_{args.layout_idx:03d}.hdf5")
    with h5py.File(out_path, "w") as f:
        f.create_dataset("asci_histogram", data=asci_histogram.numpy())

    # Summary stats
    filled = (asci_histogram > 0).sum().item()
    total = n_fov * n_bins
    asci_pct = filled / total * 100
    print(f"  ASCI: {filled}/{total} bins filled = {asci_pct:.2f}%")
    print(f"  Saved {out_path}")
