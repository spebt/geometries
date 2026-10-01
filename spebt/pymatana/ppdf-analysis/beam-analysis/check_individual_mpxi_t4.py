#!/usr/bin/env python3
"""
check_individual_mpxi_t4_like_base.py
------------------------------------
T4 version that follows the SAME logic as the base code:

- For each T4 pose (t4_00..t4_03):
    * load FWHM + detector ids (drop NaN FWHM)
    * load masks -> counts (k per detector)
    * map each beam to k via: beam_k = counts[det_id]   (same as base)

- Then aggregate across poses by concatenating all beams.

Also supports processing multiple layouts via --layouts:
  --layouts 0,1
  --layouts 0:2   (end exclusive)

Outputs per layout (in --out):
  1) beam_width_histogram_t4_layoutXX.png
  2) detector_beam_counts_t4_layoutXX.png
  3) fwhm_by_multiplexing_grid_t4_layoutXX.png
"""
import argparse
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import math

T4_TAGS = ["t4_00", "t4_01", "t4_02", "t4_03"]

# Fixed display range (requested)
FWHM_LO = 0.0
FWHM_HI = 0.6


def read_columns(h5_path: str, desired: list[str]) -> dict[str, torch.Tensor]:
    """Return selected columns from beam_properties HDF5 as tensors."""
    with h5py.File(h5_path, "r") as f:
        header = [
            h.decode("utf-8") if isinstance(h, bytes) else h
            for h in f["beam_properties"].attrs["Header"]
        ]
        data = torch.from_numpy(f["beam_properties"][:])

    cols: dict[str, torch.Tensor] = {}
    for name in desired:
        if name not in header:
            raise RuntimeError(f"Column '{name}' not found in {h5_path}")
        cols[name] = data[:, header.index(name)]
    return cols


def load_props(props_path: str) -> tuple[np.ndarray, torch.Tensor]:
    """Load valid FWHM (numpy) and aligned detector unit IDs (torch int64)."""
    cols = read_columns(props_path, ["FWHM (mm)", "detector unit id"])
    fwhm_raw = cols["FWHM (mm)"].numpy()
    det_id_raw = cols["detector unit id"].to(torch.int64)

    valid = ~np.isnan(fwhm_raw)
    return fwhm_raw[valid], det_id_raw[valid]


def load_masks(masks_path: str) -> torch.Tensor:    #(3360,40000)
    """Load beam_mask (n_det, n_pix)."""   
    with h5py.File(masks_path, "r") as f:
        return torch.from_numpy(f["beam_mask"][:])


def beams_per_detector(masks: torch.Tensor) -> torch.Tensor:
    """counts[i] = number of unique non-zero beam IDs on detector i."""
    return torch.tensor([(row.unique().numel() - 1) for row in masks], dtype=torch.int64)


def nice_grid(n: int) -> tuple[int, int]:
    """Choose a readable grid (rows, cols) for n panels."""
    if n <= 4:
        return 1, n
    cols = min(4, n)
    rows = int(math.ceil(n / cols))
    return rows, cols


def parse_layouts(s: str):
    """Accept '0,1,2' or '0:24' (end exclusive)."""
    s = s.strip()
    if ":" in s:
        a, b = s.split(":")
        return list(range(int(a), int(b)))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def run_one_layout(layout: int, data_dir: str, out_dir: str, bins: int, good_lo: float, good_hi: float):
    os.makedirs(out_dir, exist_ok=True)

    props_paths = [
        os.path.join(data_dir, f"beams_properties_configuration_{layout:02d}_{tag}.hdf5")
        for tag in T4_TAGS
    ]
    masks_paths = [
        os.path.join(data_dir, f"beams_masks_configuration_{layout:02d}_{tag}.hdf5")
        for tag in T4_TAGS
    ]

    for p in props_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing properties file: {p}")
    for m in masks_paths:
        if not os.path.exists(m):
            raise FileNotFoundError(f"Missing masks file: {m}")

    # ================================================================
    # Load + aggregate across T4 poses USING BASE LOGIC PER POSE
    # ================================================================
    all_fwhm = []
    all_detid = []
    all_beam_k = []

    # Also accumulate detector multiplicity distribution across poses
    # (same idea as base Fig2, but aggregated over 4 masks)
    counts_list = []

    for props_path, masks_path in zip(props_paths, masks_paths):
        print(f"\n[POSE] layout={layout:02d} props={os.path.basename(props_path)}")
        fwhm_i, det_i = load_props(props_path)
        print(f"  valid beams: {len(fwhm_i)}")

        print(f"[POSE] layout={layout:02d} masks={os.path.basename(masks_path)}")
        masks = load_masks(masks_path)
        counts = beams_per_detector(masks)   # counts per detector for THIS pose (base logic)
        counts_list.append(counts)

        # Map beam -> k using this pose's counts (exact base logic)
        # (det_i is detector unit id per beam)
        if det_i.numel() > 0 and det_i.max().item() >= counts.numel():
            raise RuntimeError(
                f"Detector IDs in props exceed masks detectors for layout={layout:02d} pose={os.path.basename(masks_path)}. "
                f"max det_id={det_i.max().item()}, n_det={counts.numel()}"
            )

        beam_k_i = counts[det_i].to(torch.int64)

        all_fwhm.append(fwhm_i)
        all_detid.append(det_i)
        all_beam_k.append(beam_k_i)

    fwhm = np.concatenate(all_fwhm, axis=0)
    det_id = torch.cat(all_detid, dim=0)
    beam_k_values = torch.cat(all_beam_k, dim=0)
    beam_k_values_np = beam_k_values.numpy()

    # ---- "good-width" detectors (still printed; range configurable) ----
    good_mask = (fwhm >= good_lo) & (fwhm <= good_hi)
    n_good_det = torch.unique(det_id[good_mask]).numel() if fwhm.size else 0
    print(f"\n[LAYOUT {layout:02d}] Detectors with ≥1 beam in {good_lo:.2f}–{good_hi:.2f} mm window: {n_good_det}")

    # ================================================================
    # Figure 1 – Global FWHM histogram (0–0.6 mm)
    # ================================================================
    fwhm_plot = fwhm[(fwhm >= FWHM_LO) & (fwhm <= FWHM_HI)]

    fig1, ax1 = plt.subplots(figsize=(7, 4), layout="constrained")
    if fwhm_plot.size > 0:
        ax1.hist(
            fwhm_plot,
            bins=bins,
            range=(FWHM_LO, FWHM_HI),
            color="#4c72b0",
            alpha=0.85,
        )
        mean_v, med_v = fwhm_plot.mean(), np.median(fwhm_plot)
        ax1.axvline(mean_v, color="red", ls="--", lw=1.5, label=f"Mean {mean_v:.3f} mm")
        ax1.axvline(med_v, color="green", ls=":", lw=1.5, label=f"Median {med_v:.3f} mm")
        ax1.legend()
    else:
        ax1.text(
            0.5, 0.5, f"No FWHM in {FWHM_LO:.1f}–{FWHM_HI:.1f} mm",
            ha="center", va="center", transform=ax1.transAxes
        )

    ax1.set_xlim(FWHM_LO, FWHM_HI)
    ax1.set_xlabel("Beam FWHM (mm)")
    ax1.set_ylabel("Number of beams")
    ax1.set_title(f"Distribution of Beam Widths (T4 aggregated, layout {layout:02d})")
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    out1 = os.path.join(out_dir, f"beam_width_histogram_t4_layout{layout:02d}.png")
    fig1.savefig(out1, dpi=300)
    plt.close(fig1)
    print(f"[SAVE] {out1}")

    # ================================================================
    # Figure 2 – beams-per-detector distribution (MAX over T4 poses)
    # ================================================================
    counts_stack = torch.stack(counts_list, dim=0)   # (4, n_det=3360)
    counts_final = counts_stack.max(dim=0).values    # (3360,)

    max_k = int(counts_final.max().item()) if counts_final.numel() else 0
    det_per_k = torch.bincount(counts_final, minlength=max_k + 1)  # includes k=0
    ks = np.arange(0, max_k + 1)

    print(f"\n[LAYOUT {layout:02d}] Detectors by beam count (k), T4 aggregation = MAX over poses:")
    for k, c in zip(ks, det_per_k.tolist()):
        print(f"  k={k}: {c} detectors")
    print(f"  Total detectors: {int(det_per_k.sum().item())} (should be 3360)")

    fig2, ax2 = plt.subplots(figsize=(7, 4), layout="constrained")
    bars = ax2.bar(ks, det_per_k.numpy(), color="#55a868", alpha=0.9)
    for bar, val in zip(bars, det_per_k.tolist()):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(val),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax2.set_xlabel("Number of beams per detector (k)")
    ax2.set_ylabel("Number of detectors")
    ax2.set_title(f"Beam Multiplicity Distribution (T4 aggregated: MAX, layout {layout:02d})")
    ax2.set_xticks(ks if max_k <= 15 else np.arange(0, max_k + 1, 2))
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    out2 = os.path.join(out_dir, f"detector_beam_counts_t4_layout{layout:02d}.png")
    fig2.savefig(out2, dpi=300)
    plt.close(fig2)
    print(f"[SAVE] {out2}")

    # ================================================================
    # Figure 3 – FWHM by multiplexing degree k (only non-empty k)
    # ================================================================
    present_ks = sorted(set(int(k) for k in np.unique(beam_k_values_np) if int(k) > 0))
    if len(present_ks) == 0:
        print("\nNo k>0 beams present. Skipping Figure 3.")
        return

    present_ks_nonempty = []
    for k in present_ks:
        fwhm_for_k = fwhm[beam_k_values_np == k]
        fwhm_for_k_plot = fwhm_for_k[(fwhm_for_k >= FWHM_LO) & (fwhm_for_k <= FWHM_HI)]
        if fwhm_for_k_plot.size > 0:
            present_ks_nonempty.append(k)

    if len(present_ks_nonempty) == 0:
        print(f"\nAll k>0 groups are empty in {FWHM_LO:.1f}–{FWHM_HI:.1f} mm window. Skipping Figure 3.")
        return

    rows, cols = nice_grid(len(present_ks_nonempty))
    fig3, axes3 = plt.subplots(
        rows, cols,
        figsize=(cols * 3.8, rows * 3.6),
        layout="constrained",
        sharey=True,
    )
    axes3 = np.array(axes3).reshape(-1)

    for ax, k in zip(axes3, present_ks_nonempty):
        fwhm_for_k = fwhm[beam_k_values_np == k]
        fwhm_for_k_plot = fwhm_for_k[(fwhm_for_k >= FWHM_LO) & (fwhm_for_k <= FWHM_HI)]

        ax.set_title(f"k={k} (N={len(fwhm_for_k_plot)})")
        ax.set_xlabel("FWHM (mm)")
        ax.hist(
            fwhm_for_k_plot,
            bins=min(80, max(10, bins // 4)),
            range=(FWHM_LO, FWHM_HI),
            color="#c44e52",
            alpha=0.85,
        )

        mean_v, med_v = fwhm_for_k_plot.mean(), np.median(fwhm_for_k_plot)
        ax.axvline(mean_v, color="blue", ls="--", lw=1.2, label=f"Mean {mean_v:.3f}")
        ax.axvline(med_v, color="black", ls=":", lw=1.2, label=f"Median {med_v:.3f}")
        ax.legend(fontsize="x-small")
        ax.set_xlim(FWHM_LO, FWHM_HI)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    for ax in axes3[len(present_ks_nonempty):]:
        ax.axis("off")

    axes3[0].set_ylabel("Number of beams")
    fig3.suptitle(
        f"FWHM Distribution by Detector Multiplexing Degree (k) — T4 aggregated (0–0.6 mm) | layout {layout:02d}",
        fontsize=14,
    )

    out3 = os.path.join(out_dir, f"fwhm_by_multiplexing_grid_t4_layout{layout:02d}.png")
    fig3.savefig(out3, dpi=300)
    plt.close(fig3)
    print(f"[SAVE] {out3}")


def main():
    ap = argparse.ArgumentParser(description="FWHM + multiplexing plots aggregated over T4 translations")
    ap.add_argument("--layouts", required=True, help="Layouts list '0,1,2' or range '0:24' (end exclusive)")
    ap.add_argument("--data-dir", required=True, help="Directory containing *_t4_00..03 hdf5 files")
    ap.add_argument("--out", required=True, help="Output directory for PNGs")

    ap.add_argument("--bins", type=int, default=200, help="Histogram bins for global FWHM plot")
    ap.add_argument("--good-lo", type=float, default=2.0, help="Lower bound of good-width window (mm)")
    ap.add_argument("--good-hi", type=float, default=5.0, help="Upper bound of good-width window (mm)")

    args = ap.parse_args()

    layout_list = parse_layouts(args.layouts)

    for layout in layout_list:
        print("\n" + "=" * 72)
        print(f"[RUN] layout {layout:02d}")
        print("=" * 72)
        run_one_layout(
            layout=layout,
            data_dir=args.data_dir,
            out_dir=args.out,
            bins=args.bins,
            good_lo=args.good_lo,
            good_hi=args.good_hi,
        )


if __name__ == "__main__":
    main()