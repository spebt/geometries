#!/usr/bin/env python3
import os
import argparse
from typing import Optional, Dict, Tuple

import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# -------------------- DEFAULTS --------------------
DEFAULT_DATA_DIR = "/vscratch/grp-rutaoyao/Omer/spebt/data/sai_10mm"
DEFAULT_IMG_NX, DEFAULT_IMG_NY = 200, 200
DEFAULT_MM_PER_PIXEL = 0.05
DEFAULT_N_BINS = 360

# FWHM windows (tune as you like)
FWHM_WINDOWS: Dict[str, Tuple[float, float]] = {
    "w1_0p15_0p25mm": (0.15, 0.25),
    "w2_0p25_0p35mm": (0.25, 0.35),
    "w3_0p35_0p50mm": (0.35, 0.50),
}

# Preferred names for the beam index column
PREFERRED_BEAM_INDEX_NAMES = [
    "beam local id",
    "beam idx",
    "beam index",
    "beam id",
]

# Detector ID ranges for each ring
RING_DET_RANGES = {
    "ring1": (0, 479),
    "ring2": (480, 1199),
    "ring3": (1200, 2159),
    "ring4": (2160, 3359),
}

# T8 suffixes (files you generated)
T8_TAGS = [f"t8_{i:02d}" for i in range(8)]
# --------------------------------------------------


def which_ring(det_id: int) -> Optional[str]:
    for ring_name, (lo, hi) in RING_DET_RANGES.items():
        if lo <= det_id <= hi:
            return ring_name
    return None


def auto_find_beam_index_column(header):
    lower_header = [h.lower() for h in header]

    for cand in PREFERRED_BEAM_INDEX_NAMES:
        if cand in lower_header:
            idx = lower_header.index(cand)
            print(f"[INFO] Using column: '{header[idx]}' (matched preferred '{cand}')")
            return header[idx]

    candidates = []
    for h in header:
        hl = h.lower()
        if "beam" in hl and "id" in hl:
            candidates.append(h)

    if len(candidates) == 1:
        print(f"[INFO] Using column: '{candidates[0]}' (auto-detected)")
        return candidates[0]
    elif len(candidates) > 1:
        print("[WARN] Multiple possible beam index columns found:")
        for c in candidates:
            print(f"       - {c}")
        print(f"[INFO] Defaulting to first candidate: '{candidates[0]}'")
        return candidates[0]

    raise RuntimeError(
        "Could not automatically find a suitable beam index column. "
        "Please inspect header and update logic."
    )


def load_beam_properties(h5_path: str):
    """
    Returns per-beam vectors:
      - fwhm (mm)
      - detector unit id (int)
      - beam index/id (int)
      - angle (rad)
    """
    print(f"\n[LOAD] Reading beam_properties from: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        header_raw = f["beam_properties"].attrs["Header"]
        header = [h.decode("utf-8") if isinstance(h, bytes) else str(h) for h in header_raw]
        data = torch.from_numpy(f["beam_properties"][:])

    def col(name: str) -> torch.Tensor:
        if name not in header:
            raise RuntimeError(f"Column '{name}' not found in {h5_path}. Header={header}")
        return data[:, header.index(name)]

    fwhm = col("FWHM (mm)")
    det_id = col("detector unit id").to(torch.int64)

    beam_index_colname = auto_find_beam_index_column(header)
    beam_idx = col(beam_index_colname).to(torch.int64)

    # NOTE: your header sometimes uses "Angle (rad)" (capital A). Keep as-is.
    angle = col("Angle (rad)")

    valid_ang = ~torch.isnan(angle)
    print(f"[DEBUG] Angle NaNs: {int((~valid_ang).sum())}/{angle.numel()} ({((~valid_ang).float().mean().item()*100):.2f}%)")
    if valid_ang.any():
        a = angle[valid_ang]
        print(f"[DEBUG] Angle range (valid): min={a.min().item():.4f}, max={a.max().item():.4f}")
    else:
        print("[ERROR] All angles are NaN.")

    valid_fwhm = ~torch.isnan(fwhm)
    print(f"[DEBUG] Total beam rows: {data.shape[0]}")
    print(f"[DEBUG] Valid FWHM rows: {valid_fwhm.sum().item()}")

    return fwhm, det_id, beam_idx, angle


def build_lookup(
    fwhm: torch.Tensor,
    det_id: torch.Tensor,
    beam_idx: torch.Tensor,
    angle: torch.Tensor,
    n_bins: int,
):
    # bin edges: 0..2π mapped as degrees 0..360 (same logic as your current)
    angular_bin_boundaries = (
        torch.arange(n_bins + 1, dtype=torch.float32) / 180.0 * torch.pi
    ).contiguous()

    angle = angle.to(torch.float32).contiguous()

    # Keep only rows where BOTH fwhm and angle are valid
    valid = (~torch.isnan(fwhm)) & (~torch.isnan(angle))
    fwhm_v = fwhm[valid]
    det_v  = det_id[valid].to(torch.int64)
    beam_v = beam_idx[valid].to(torch.int64)
    ang_v  = angle[valid]

    # Wrap into [0, 2π)
    ang_v = torch.remainder(ang_v, 2.0 * torch.pi)

    # Bin angles -> [0..n_bins-1]
    angle_bins = torch.bucketize(ang_v, angular_bin_boundaries, right=False) - 1

    lookup = {}
    for w, d, b, abin in zip(fwhm_v.tolist(), det_v.tolist(), beam_v.tolist(), angle_bins.tolist()):
        lookup[(int(d), int(b))] = (float(w), int(abin))

    print(f"[DEBUG] lookup size: {len(lookup)} entries (valid rows only)")
    return lookup


def file_for(layout_idx: int, kind: str, tag: Optional[str]):
    """
    kind: 'props' or 'masks'
    tag: None for base, else 't8_00'..'t8_07'
    """
    if kind == "props":
        base = f"beams_properties_configuration_{layout_idx:02d}"
    elif kind == "masks":
        base = f"beams_masks_configuration_{layout_idx:02d}"
    else:
        raise ValueError("kind must be 'props' or 'masks'")

    if tag is None:
        return base + ".hdf5"
    return base + f"_{tag}.hdf5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--layouts", default="0:1", help="layout range like '0:24' or list like '0,1,2'")
    ap.add_argument("--t8", action="store_true", help="aggregate over t8_00..t8_07 files per layout")
    ap.add_argument("--img-nx", type=int, default=DEFAULT_IMG_NX)
    ap.add_argument("--img-ny", type=int, default=DEFAULT_IMG_NY)
    ap.add_argument("--mm-per-pixel", type=float, default=DEFAULT_MM_PER_PIXEL)
    ap.add_argument("--n-bins", type=int, default=DEFAULT_N_BINS)
    args = ap.parse_args()

    # Parse layouts
    if ":" in args.layouts:
        a, b = args.layouts.split(":")
        layout_seq = range(int(a), int(b))
    else:
        layout_seq = [int(x.strip()) for x in args.layouts.split(",") if x.strip()]

    IMG_NX, IMG_NY = args.img_nx, args.img_ny
    N_PIX = IMG_NX * IMG_NY
    N_BINS = args.n_bins

    FOV_X_MM = IMG_NX * args.mm_per_pixel
    FOV_Y_MM = IMG_NY * args.mm_per_pixel
    EXTENT_MM = (-FOV_X_MM / 2, FOV_X_MM / 2, -FOV_Y_MM / 2, FOV_Y_MM / 2)

    # Allocate ASCI histograms (pixel × angle-bin)
    asci_hist_allrings = {k: torch.zeros((N_PIX, N_BINS), dtype=torch.int32) for k in FWHM_WINDOWS}
    asci_hist_per_ring = {
        ring: {k: torch.zeros((N_PIX, N_BINS), dtype=torch.int32) for k in FWHM_WINDOWS}
        for ring in RING_DET_RANGES
    }

    layouts_used = 0
    poses_used = 0

    pose_tags = T8_TAGS if args.t8 else [None]

    for layout_idx in layout_seq:
        any_pose_loaded_for_layout = False

        for tag in pose_tags:
            props_path = os.path.join(args.data_dir, file_for(layout_idx, "props", tag))
            masks_path = os.path.join(args.data_dir, file_for(layout_idx, "masks", tag))

            if not os.path.exists(props_path):
                print(f"[WARN] Missing props: {props_path}")
                continue
            if not os.path.exists(masks_path):
                print(f"[WARN] Missing masks: {masks_path}")
                continue

            any_pose_loaded_for_layout = True
            poses_used += 1
            print(f"\n[INFO] Processing layout={layout_idx:02d}, tag={(tag or 'base')}")

            fwhm, det_id, beam_idx, angle = load_beam_properties(props_path)
            lookup = build_lookup(fwhm, det_id, beam_idx, angle, N_BINS)

            with h5py.File(masks_path, "r") as f:
                masks = torch.from_numpy(f["beam_mask"][:])  # (n_det, N_pix)

            n_det, n_pix = masks.shape
            if n_pix != N_PIX:
                raise RuntimeError(f"mask pixels {n_pix} != expected {N_PIX}")
            print(f"[DEBUG] beam_mask shape: n_det={n_det}, n_pix={n_pix}")

            # Assume detector index == detector unit id (your pipeline does this)
            for det in range(n_det):
                row = masks[det]
                unique_beams = torch.unique(row)
                unique_beams = unique_beams[unique_beams != 0]
                if unique_beams.numel() == 0:
                    continue

                ring_name = which_ring(det)

                for b in unique_beams.tolist():
                    key = (det, int(b))
                    if key not in lookup:
                        continue

                    fwhm_mm, ang_bin = lookup[key]
                    if not (0 <= ang_bin < N_BINS):
                        continue

                    pix_mask = (row == b)
                    pix_ids = torch.nonzero(pix_mask, as_tuple=False).squeeze(1)

                    for win_name, (w_lo, w_hi) in FWHM_WINDOWS.items():
                        if w_lo <= fwhm_mm < w_hi:
                            asci_hist_allrings[win_name][pix_ids, ang_bin] += 1
                            if ring_name is not None:
                                asci_hist_per_ring[ring_name][win_name][pix_ids, ang_bin] += 1

        if any_pose_loaded_for_layout:
            layouts_used += 1

    print(f"\n[INFO] Finished. layouts_used={layouts_used}, poses_used={poses_used}")

    suffix = "t8agg" if args.t8 else "base"
    out_dir = os.path.join(args.data_dir, f"asci_fwhm_maps_{suffix}")
    os.makedirs(out_dir, exist_ok=True)

    def hist_to_asci_frac(hist_2d: torch.Tensor) -> torch.Tensor:
        return torch.count_nonzero(hist_2d, dim=1).to(torch.float32) / float(N_BINS)

    # 1) ALL-RINGS maps
    for win_name, hist in asci_hist_allrings.items():
        asci_frac = hist_to_asci_frac(hist)
        img2d = asci_frac.view(IMG_NX, IMG_NY).T.cpu().numpy()

        fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
        im = ax.imshow(img2d, origin="lower", cmap="viridis", extent=EXTENT_MM, vmin=0.0, vmax=1.0)
        cbar = fig.colorbar(im, ax=ax, label="ASCI (fraction of angle bins)")
        cbar.formatter = PercentFormatter(xmax=1.0, decimals=1)
        cbar.update_ticks()

        ax.set_title(
            f"ASCI map – ALL RINGS – {win_name}\n"
            f"{suffix} | layouts_used={layouts_used} | poses_used={poses_used} | "
            f"max={asci_frac.max().item():.2%}, min={asci_frac.min().item():.2%}"
        )
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")

        out_path = os.path.join(out_dir, f"asci_allrings_{win_name}_{suffix}.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print("[SAVE]", out_path)

    # 2) Per-ring maps
    for ring_name, ring_hists in asci_hist_per_ring.items():
        for win_name, hist in ring_hists.items():
            asci_frac = hist_to_asci_frac(hist)
            img2d = asci_frac.view(IMG_NX, IMG_NY).T.cpu().numpy()

            fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
            im = ax.imshow(img2d, origin="lower", cmap="viridis", extent=EXTENT_MM, vmin=0.0, vmax=1.0)
            cbar = fig.colorbar(im, ax=ax, label="ASCI (fraction of angle bins)")
            cbar.formatter = PercentFormatter(xmax=1.0, decimals=1)
            cbar.update_ticks()

            pretty_ring = ring_name.replace("ring", "Ring ").upper()
            ax.set_title(
                f"ASCI map – {pretty_ring} – {win_name}\n"
                f"{suffix} | max={asci_frac.max().item():.2%}, min={asci_frac.min().item():.2%}"
            )
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")

            out_path = os.path.join(out_dir, f"asci_{ring_name}_{win_name}_{suffix}.png")
            fig.savefig(out_path, dpi=300)
            plt.close(fig)
            print("[SAVE]", out_path)

    # 3) 2×2 ring panel per window
    ring_order = ["ring1", "ring2", "ring3", "ring4"]
    ring_titles = {
        "ring1": "Ring I (Det 0–479)",
        "ring2": "Ring II (480–1199)",
        "ring3": "Ring III (1200–2159)",
        "ring4": "Ring IV (2160–3359)",
    }

    for win_name in FWHM_WINDOWS.keys():
        imgs = []
        for r in ring_order:
            asci_frac = hist_to_asci_frac(asci_hist_per_ring[r][win_name])
            imgs.append(asci_frac.view(IMG_NX, IMG_NY).T.cpu().numpy())

        fig, axes = plt.subplots(2, 2, figsize=(10, 9), layout="constrained")
        axes = axes.ravel()

        for idx, (r, img) in enumerate(zip(ring_order, imgs)):
            ax = axes[idx]
            im = ax.imshow(img, origin="lower", cmap="viridis", extent=EXTENT_MM, vmin=0.0, vmax=1.0)
            ax.set_title(ring_titles[r])
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")

        cbar = fig.colorbar(im, ax=axes.tolist(), label="ASCI (fraction of angle bins)")
        cbar.formatter = PercentFormatter(xmax=1.0, decimals=1)
        cbar.update_ticks()

        fig.suptitle(f"ASCI map – FWHM window: {win_name} | {suffix}", fontsize=14)

        panel_path = os.path.join(out_dir, f"asci_panel_{win_name}_{suffix}.png")
        fig.savefig(panel_path, dpi=300)
        plt.close(fig)
        print("[SAVE PANEL]", panel_path)


if __name__ == "__main__":
    main()