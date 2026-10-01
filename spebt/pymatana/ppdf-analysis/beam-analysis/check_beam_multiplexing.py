#!/usr/bin/env python3
"""
check_beam_multiplexing.py
--------------------------------
* **Figure 1:** Histogram of *all* valid beam FWHM values (mm) with mean/median markers.
* **Figure 2:** Bar-chart showing **how many detector units have *k* beams** (k = 1, 2, 3 …).
* Prints the number of detectors that host at least one beam in the 2–5 mm “good-width” window.

Run e.g.
```bash
python check_beam_multiplexing.py \
       --props  /vscratch/grp-rutaoyao/Omer/spebt/data/sai_10mm/beams_properties_configuration_00.hdf5 \
       --masks  /vscratch/grp-rutaoyao/Omer/spebt/data/sai_10mm/beams_masks_configuration_00.hdf5 \
       --out    /vscratch/grp-rutaoyao/Omer/spebt/plots/sai_10mm
```
"""
import argparse, os, h5py, numpy as np, matplotlib.pyplot as plt
import torch

# ------------------------------------------------------------------ #

def read_columns(h5_path: str, desired: list[str]) -> dict[str, torch.Tensor]:
    """Return selected columns from beam_properties HDF5 as tensors."""
    with h5py.File(h5_path, "r") as f:
        # Handle case where header might be bytes
        header = [h.decode('utf-8') if isinstance(h, bytes) else h for h in f["beam_properties"].attrs["Header"]]
        data   = torch.from_numpy(f["beam_properties"][:])
    cols: dict[str, torch.Tensor] = {}
    for name in desired:
        if name not in header:
            raise RuntimeError(f"Column '{name}' not found in {h5_path}")
        cols[name] = data[:, header.index(name)]
    return cols

# ------------------------------------------------------------------ #

def main(args):
    # -- read FWHM and detector‑ID columns ---------------------------
    print(f"Reading properties from: {args.props}")
    cols   = read_columns(args.props, ["FWHM (mm)", "detector unit id"])
    fwhm_raw = cols["FWHM (mm)"].numpy()
    det_id_raw = cols["detector unit id"].to(torch.int64)

    # --- ADDED: Debugging print to show the raw data ---
    print(f"First 10 raw FWHM values read from file: {fwhm_raw[:10]}")
    # ----------------------------------------------------

    # --- IMPORTANT: Filter out NaN values from FWHM ---
    # This handles cases where FWHM calculation failed for a beam
    valid_fwhm_mask = ~np.isnan(fwhm_raw)
    fwhm = fwhm_raw[valid_fwhm_mask]
    det_id = det_id_raw[valid_fwhm_mask]
    
    print(f"Read {len(fwhm_raw)} total beam entries.")
    print(f"Found {len(fwhm)} beams with a valid FWHM value.")

    # -- detectors that have at least one 2‑5 mm beam ---------------
    if len(fwhm) > 0:
        good_width_mask = (fwhm >= 2.0) & (fwhm <= 5.0)
        # Apply the mask to the valid detector IDs before finding unique ones
        n_good_det = torch.unique(det_id[good_width_mask]).numel()
        print(f"Detectors with ≥1 beam in 2–5 mm window: {n_good_det}")
    else:
        print("No valid FWHM values to analyze.")
        n_good_det = 0

    # ------------------ Figure 1 – FWHM histogram ------------------
    os.makedirs(args.out, exist_ok=True)
    fig1, ax1 = plt.subplots(figsize=(7,4), layout="constrained")
    
    # Plot histogram only with valid FWHM values
    if len(fwhm) > 0:
        ax1.hist(fwhm, bins=30, color="#4c72b0", alpha=0.85)
        mean_v, med_v = fwhm.mean(), np.median(fwhm)
        ax1.axvline(mean_v, color="red",   ls="--", lw=1.5, label=f"Mean {mean_v:.2f} mm")
        ax1.axvline(med_v,  color="green", ls=":",  lw=1.5, label=f"Median {med_v:.2f} mm")
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, "No valid FWHM data to plot", ha='center', va='center')

    ax1.set_xlabel("Beam FWHM (mm)")
    ax1.set_ylabel("Number of beams")
    ax1.set_title("Distribution of Beam Widths (all valid beams)")
    
    out1 = os.path.join(args.out, "beam_width_histogram.png")
    fig1.savefig(out1, dpi=300)
    plt.close(fig1)
    print(f"Saved → {out1}")

    # ------------- Figure 2 – beams‑per‑detector bar chart ----------
    if args.masks:
        print(f"Reading masks from: {args.masks}")
        with h5py.File(args.masks, "r") as f:
            masks = torch.from_numpy(f["beam_mask"][:])
        
        # Count beams per detector by finding unique non-zero IDs in each mask row
        counts = torch.tensor([(row.unique().numel() - 1) for row in masks])

        # How many detectors have k beams? (k = 1…max)
        if counts.numel() > 0 and counts.max() > 0:
            max_k  = int(counts.max().item())
            det_per_k = torch.bincount(counts, minlength=max_k+1)[1:]  # drop k=0 count
            ks = np.arange(1, max_k+1)

            print("\nDetectors by beam count:")
            for k, c in zip(ks, det_per_k.tolist()):
                print(f"  k={k}: {c} detectors")

            fig2, ax2 = plt.subplots(figsize=(6,4), layout="constrained")
            bars = ax2.bar(ks, det_per_k.numpy(), color="#55a868", alpha=0.9)
            for bar, val in zip(bars, det_per_k.tolist()):
                ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                         str(val), ha="center", va="bottom", fontsize=9)
            ax2.set_xlabel("Number of beams per detector (k)")
            ax2.set_ylabel("Number of detectors")
            ax2.set_title("Beam Multiplicity Distribution")
            ax2.set_xticks(ks if max_k < 10 else np.arange(1, max_k + 1, 2)) # Adjust ticks for many bars

            out2 = os.path.join(args.out, "detector_beam_counts.png")
            fig2.savefig(out2, dpi=300)
            plt.close(fig2)
            print(f"Saved → {out2}")
        else:
            print("\nNo beams found in the masks file to generate multiplicity plot.")
    else:
        print("\n--masks file not provided, skipping Figure 2.")

# ------------------------------------------------------------------ #
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Beam-width histogram and detector beam-count plot.")
    ap.add_argument("--props", required=True, help="beam_properties_*.hdf5 file")
    ap.add_argument("--masks", default=None, help="beams_masks_*.hdf5 file (needed for Figure 2)")
    ap.add_argument("--out",   required=True, help="output directory for PNGs")
    args = ap.parse_args()
    main(args)