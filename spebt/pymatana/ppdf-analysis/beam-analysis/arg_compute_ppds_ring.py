#!/usr/bin/env python3
import os
import sys
import h5py
import torch
import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt

# --- your modules ----------------------------------------------------
from geometry_2d_io import load_scanner_layout_geometries, load_scanner_layouts
from geometry_2d_utils import fov_tensor_dict, pixels_coordinates, pixels_to_detector_unit_rads
from convex_hull_helper import convex_hull_2d, sort_points_for_hull_batch_2d
from ppdf_io import load_ppdfs_data_from_hdf5
from beam_property_extract import (
    sample_ppdf_on_arc_2d_local,
    beams_boundaries_radians,
    get_beams_masks,
    get_beams_weighted_center,
    get_beam_width,                 # tangential FWHM (perpendicular to axis)
    beam_samples_on_points_batch,   # reuse for radial sampling
    full_width_half_maximum_1d_batch,
)
# ---------------------------------------------------------------------

T4_TAGS = ["t4_00", "t4_01", "t4_02", "t4_03"]

# ------------------------- helpers -----------------------------------
@torch.no_grad()
def _beam_axis_sampling_line_batch(
    detector_center: torch.Tensor,
    beam_centers: torch.Tensor,
    n_samples: int = 4096,
    length_mm: float = 64.0,
):
    """Sampling line *along* the beam axis (radial)."""
    axis = torch.atan2(
        beam_centers[:, 1] - detector_center[1],
        beam_centers[:, 0] - detector_center[0],
    )
    kx, ky = torch.cos(axis), torch.sin(axis)
    dist = torch.linspace(
        -length_mm / 2,
        length_mm / 2,
        n_samples,
        dtype=torch.float32,
        device=beam_centers.device,
    )
    pts = (
        torch.stack((kx, ky), dim=1).unsqueeze(1).expand(-1, n_samples, -1)
        * dist.view(1, n_samples, 1)
        + beam_centers.unsqueeze(1).expand(-1, n_samples, -1)
    )
    return pts, dist


@torch.no_grad()
def _get_beam_width_radial(
    beams_centers: torch.Tensor,
    detector_center: torch.Tensor,
    beams_masks: torch.Tensor,
    ppdf_2d: torch.Tensor,
    fov_dict: dict,
    line_n_samples: int = 4096,
):
    """Radial FWHM by sampling *along* the axis."""
    pts_batch, dist = _beam_axis_sampling_line_batch(
        detector_center, beams_centers, n_samples=line_n_samples, length_mm=64.0
    )
    n_beams = beams_centers.shape[0]
    beams_data_2d_batch = (
        ppdf_2d.view(-1)
        .unsqueeze(0)
        .expand(n_beams, -1)
        .clone()
        .masked_fill_(~beams_masks, 0)
    ).view(n_beams, int(fov_dict["n pixels"][0]), int(fov_dict["n pixels"][1]))

    sampled = beam_samples_on_points_batch(beams_data_2d_batch, pts_batch, fov_dict)
    fwhm, x_bounds = full_width_half_maximum_1d_batch(
        dist.view(1, -1).expand(n_beams, -1), sampled
    )
    return fwhm, x_bounds, sampled, dist
# ---------------------------------------------------------------------


@torch.no_grad()
def compute_ppds_for_layout_per_ring(
    layout_idx: int,
    scanner_layouts_dir: str,
    scanner_layouts_filename: str,
    ppdfs_dir: str,
    out_dir: str,
    threshold_relative: float = 0.01,
    line_n_samples: int = 4096,
):
    os.makedirs(out_dir, exist_ok=True)

    layouts_data, layouts_uid = load_scanner_layouts(scanner_layouts_dir, scanner_layouts_filename)

    # SAI-10mm FOV (matches your 200x200, 0.05 mm/px runs)
    fov = fov_tensor_dict(n_pixels=(200, 200), mm_per_pixel=(0.05, 0.05), center_coordinates=(0.0, 0.0))
    H = int(fov["n pixels"][0])
    W = int(fov["n pixels"][1])

    _, det_units = load_scanner_layout_geometries(layout_idx, layouts_data)
    n_det = int(det_units.shape[0])
    det_centers = det_units.mean(dim=1)

    # ---- SC-SPECT ring mapping (must match build order) --------------
    # Ring-1: 480, Ring-2: 720, Ring-3: 960, Ring-4: 1200 => total 3360
    RING_COUNTS = [480, 720, 960, 1200]
    RING_STARTS = [0, 480, 1200, 2160]
    RING_ENDS   = [480, 1200, 2160, 3360]

    if n_det != sum(RING_COUNTS):
        raise ValueError(f"Unexpected n_det={n_det}. Expected {sum(RING_COUNTS)} for SC-SPECT rings.")

    def ring_id_from_detector_index(i: int) -> int:
        if i < RING_ENDS[0]: return 0
        if i < RING_ENDS[1]: return 1
        if i < RING_ENDS[2]: return 2
        return 3

    print(f"[INFO] layout={layout_idx:03d} | n_det={n_det}")
    print(f"[INFO] ring slices: "
          f"R1[{RING_STARTS[0]}:{RING_ENDS[0]}], "
          f"R2[{RING_STARTS[1]}:{RING_ENDS[1]}], "
          f"R3[{RING_STARTS[2]}:{RING_ENDS[2]}], "
          f"R4[{RING_STARTS[3]}:{RING_ENDS[3]}]")

    fov_corners = (
        torch.tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=torch.float32)
        * fov["size in mm"]
        * 0.5
    )
    hull_points_batch = torch.cat(
        (fov_corners.unsqueeze(0).expand(n_det, -1, -1), det_centers.unsqueeze(1)),
        dim=1
    )
    hull_points_batch = sort_points_for_hull_batch_2d(hull_points_batch)

    # ---- per-ring outputs -------------------------------------------
    PPDS = [torch.zeros(H, W, dtype=torch.float32) for _ in range(4)]
    SENS = [torch.zeros(H, W, dtype=torch.float32) for _ in range(4)]
    V_i  = [torch.zeros(n_det, dtype=torch.float32) for _ in range(4)]  # only ring entries will be nonzero

    for tag in T4_TAGS:
        ppdf_h5 = f"position_{layout_idx:03d}_ppdfs_{tag}.hdf5"
        ppdf_path = os.path.join(ppdfs_dir, ppdf_h5)
        if not os.path.exists(ppdf_path):
            raise FileNotFoundError(f"Missing PPDF file: {ppdf_path}")

        print(f"[LOAD] {ppdf_path}")
        ppdfs = load_ppdfs_data_from_hdf5(ppdfs_dir, ppdf_h5, fov)  # (n_crystals, H, W)

        for i in range(n_det):
            rid = ring_id_from_detector_index(i)

            ppdf2d = ppdfs[i].view(H, W)
            center = det_centers[i]
            hull   = convex_hull_2d(hull_points_batch[i])

            sampled, rads, _ = sample_ppdf_on_arc_2d_local(ppdf2d, center, hull, fov)
            bounds = beams_boundaries_radians(sampled, rads, threshold=threshold_relative)

            SENS[rid] += ppdf2d

            if bounds.numel() == 0:
                continue

            fov_xy   = pixels_coordinates(fov)
            fov_rads = pixels_to_detector_unit_rads(fov_xy, center)
            masks    = get_beams_masks(fov_rads, bounds)
            centers  = get_beams_weighted_center(masks, fov_xy, ppdf2d)

            fwhm_tan, _, _, _ = get_beam_width(
                centers, center, masks, ppdf2d, fov, line_n_samples=line_n_samples
            )
            fwhm_rad, _, _, _ = _get_beam_width_radial(
                centers, center, masks, ppdf2d, fov, line_n_samples=line_n_samples
            )

            eps = 1e-6
            fwhm_tan = torch.nan_to_num(fwhm_tan, nan=0.0).clamp_min(0.0)
            fwhm_rad = torch.nan_to_num(fwhm_rad, nan=0.0).clamp_min(0.0)

            Vi_b = fwhm_tan * fwhm_rad
            Vi   = float(torch.sum(Vi_b).item())
            if Vi <= eps:
                continue

            V_i[rid][i] += Vi
            PPDS[rid] += ppdf2d / Vi

    out_h5 = os.path.join(out_dir, f"ppds_layout_{layout_idx:03d}_perring.hdf5")
    with h5py.File(out_h5, "w") as f:
        for r in range(4):
            f.create_dataset(f"PPDS_ring{r+1}", data=PPDS[r].cpu().numpy())
            f.create_dataset(f"SENS_ring{r+1}", data=SENS[r].cpu().numpy())
            f.create_dataset(f"V_i_ring{r+1}",  data=V_i[r].cpu().numpy())

        f.attrs["layout_idx"] = layout_idx
        f.attrs["layouts_uid"] = layouts_uid
        f.attrs["threshold_relative"] = threshold_relative
        f.attrs["line_n_samples"] = line_n_samples
        f.attrs["t4_tags"] = ",".join(T4_TAGS)

        f.attrs["ring_counts"] = RING_COUNTS
        f.attrs["ring_starts"] = RING_STARTS
        f.attrs["ring_ends"]   = RING_ENDS

    for r in range(4):
        vals = V_i[r][V_i[r] > 0]
        if vals.numel() == 0:
            continue
        fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
        ax.hist(vals.cpu().numpy(), bins=50)
        ax.set_xlabel(r"$V_i$  (mm$^2$)  =  $\sum_b$ FWHM$_{i,b}^{\rm tan}$ × FWHM$_{i,b}^{\rm rad}$")
        ax.set_ylabel("Count")
        ax.set_title(f"Layout {layout_idx:03d} Ring {r+1}: histogram of $V_i$ (n={int(vals.numel())})")
        out_png = os.path.join(out_dir, f"V_i_hist_layout_{layout_idx:03d}_ring{r+1}.png")
        plt.savefig(out_png, dpi=200)
        plt.close(fig)

    print(f"[OK] Wrote: {out_h5}")


@torch.no_grad()
def compute_ppds_average_per_ring(layout_indices, ppds_dir: str, out_dir: str):
    assert len(layout_indices) > 0, "No layout indices provided for averaging."

    sum_PPDS = [None, None, None, None]
    sum_SENS = [None, None, None, None]
    layouts_uid = None
    shape = None
    found = []

    for li in layout_indices:
        h5_path = os.path.join(ppds_dir, f"ppds_layout_{li:03d}_perring.hdf5")
        if not os.path.exists(h5_path):
            print(f"[WARN] Missing {h5_path}; skipping.")
            continue

        with h5py.File(h5_path, "r") as f:
            uid = f.attrs.get("layouts_uid", None)
            PPDS_r = [torch.from_numpy(f[f"PPDS_ring{r}"][...]).to(torch.float32) for r in range(1, 5)]
            SENS_r = [torch.from_numpy(f[f"SENS_ring{r}"][...]).to(torch.float32) for r in range(1, 5)]

            if shape is None:
                shape = PPDS_r[0].shape
                for r in range(4):
                    sum_PPDS[r] = torch.zeros_like(PPDS_r[r], dtype=torch.float32)
                    sum_SENS[r] = torch.zeros_like(SENS_r[r], dtype=torch.float32)
                layouts_uid = uid
            else:
                if PPDS_r[0].shape != shape:
                    raise ValueError(f"Shape mismatch for layout {li}: {PPDS_r[0].shape} vs {shape}")

            for r in range(4):
                sum_PPDS[r] += PPDS_r[r]
                sum_SENS[r] += SENS_r[r]

            found.append(li)

    n = len(found)
    if n == 0:
        raise FileNotFoundError("No per-layout per-ring PPDS files found; nothing to average.")

    mean_PPDS = [sum_PPDS[r] / float(n) for r in range(4)]
    mean_SENS = [sum_SENS[r] / float(n) for r in range(4)]

    os.makedirs(out_dir, exist_ok=True)
    out_h5 = os.path.join(out_dir, "ppds_layout_AVG_perring.hdf5")
    with h5py.File(out_h5, "w") as f:
        for r in range(4):
            f.create_dataset(f"PPDS_ring{r+1}", data=mean_PPDS[r].cpu().numpy())
            f.create_dataset(f"SENS_ring{r+1}", data=mean_SENS[r].cpu().numpy())

        f.attrs["n_layouts"] = n
        f.attrs["layout_indices"] = found
        if layouts_uid is not None:
            f.attrs["layouts_uid"] = layouts_uid

    print(f"[OK] Wrote average per-ring PPDS: {out_h5} (n={n}, layouts={found})")


if __name__ == "__main__":
    # UPDATED: your rot2 tensor lives in /geometry
    SCANNER_LAYOUTS_DIR  = "/vscratch/grp-rutaoyao/Omer/spebt/geometry"
    SCANNER_LAYOUTS_FILE = "scanner_layouts_24a4365260a3f68491dfa8ca55e0ecc2_rot2_ang1p0deg_trans1x1_step0p0x0p0.tensor"
    PPDFS_DIR            = "/vscratch/grp-rutaoyao/Omer/spebt/data/sai_10mm"
    OUT_DIR              = "/vscratch/grp-rutaoyao/Omer/spebt/data/sai_10mm"

    if len(sys.argv) >= 2 and sys.argv[1].lower() in ("avg", "average", "mean"):
        if len(sys.argv) == 3:
            N = int(sys.argv[2])
            indices = list(range(N))
        elif len(sys.argv) == 4:
            start = int(sys.argv[2]); end = int(sys.argv[3])
            indices = list(range(start, end + 1))
        else:
            print("Usage:\n  python arg_compute_ppds_per_ring.py avg <N>\n  python arg_compute_ppds_per_ring.py avg <start> <end>")
            sys.exit(1)

        compute_ppds_average_per_ring(indices, OUT_DIR, OUT_DIR)
        sys.exit(0)

    if len(sys.argv) != 2:
        print("Usage: python arg_compute_ppds_per_ring.py <layout_idx>\n   or : python arg_compute_ppds_per_ring.py avg <N|start end>")
        sys.exit(1)

    layout_idx = int(sys.argv[1])

    compute_ppds_for_layout_per_ring(
        layout_idx,
        SCANNER_LAYOUTS_DIR,
        SCANNER_LAYOUTS_FILE,
        PPDFS_DIR,
        OUT_DIR,
        threshold_relative=0.01,
        line_n_samples=4096,
    )