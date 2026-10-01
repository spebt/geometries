#!/usr/bin/env python3
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

BASE_DIR = "/vscratch/grp-rutaoyao/Omer/spebt/data/sai_10mm/"
LAYOUTS = [0, 1]
POSES = list(range(8))  # T8 -> 0..7

FOV_X, FOV_Y = 200, 200
MM_PER_PX = 0.05
FOV_PIXELS = FOV_X * FOV_Y

def load_ppdfs(path, n_xtals_to_load=-1):
    with h5py.File(path, "r") as f:
        data = f["ppdfs"][:n_xtals_to_load] if n_xtals_to_load != -1 else f["ppdfs"][:]
    return data.astype(np.float32)

def sensitivity_from_ppdfs(ppdfs):
    # sum over crystals -> 1D map (not a probability)
    s1d = np.sum(ppdfs, axis=0)
    assert s1d.size == FOV_PIXELS
    return s1d

def plot_map(s2d, title, out_png, vmin=None, vmax=None):
    extent = [
        -(MM_PER_PX * FOV_X / 2.0), +(MM_PER_PX * FOV_X / 2.0),
        -(MM_PER_PX * FOV_Y / 2.0), +(MM_PER_PX * FOV_Y / 2.0),
    ]
    plt.figure(figsize=(8, 7))
    plt.imshow(s2d, origin="lower", extent=extent, cmap="viridis", vmin=vmin, vmax=vmax)
    plt.colorbar(label="Mean sensitivity (sum over crystals, a.u.)")
    plt.title(title)
    plt.xlabel("X (mm)"); plt.ylabel("Y (mm)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print("Saved:", out_png)

def plot_hist(s2d, title, out_png):
    vals = s2d[s2d > 0].ravel()
    plt.figure(figsize=(10, 6))
    plt.hist(vals, bins=120, alpha=0.8)
    plt.title(title)
    plt.xlabel("Sensitivity (a.u.)")
    plt.ylabel("Pixel count")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print("Saved:", out_png)

def compute_layout_sensitivity(layout_idx, n_xtals_to_load=-1):
    sens_sum = None
    loaded = 0
    for pi in POSES:
        fn = f"position_{layout_idx:03d}_ppdfs_t8_{pi:02d}.hdf5"
        path = os.path.join(BASE_DIR, fn)
        if not os.path.exists(path):
            print("[MISS]", path)
            continue
        ppdfs = load_ppdfs(path, n_xtals_to_load=n_xtals_to_load)
        s1d = sensitivity_from_ppdfs(ppdfs)
        sens_sum = s1d if sens_sum is None else sens_sum + s1d
        loaded += 1

    if loaded == 0:
        raise RuntimeError(f"No T8 files found for layout {layout_idx}")

    sens_mean_1d = sens_sum / float(loaded)   # mean over poses
    return sens_mean_1d.reshape(FOV_Y, FOV_X), loaded

if __name__ == "__main__":
    # optional: set to -1 for all crystals
    n_xtals_to_load = -1

    layout_maps = []
    for li in LAYOUTS:
        s2d, loaded = compute_layout_sensitivity(li, n_xtals_to_load=n_xtals_to_load)
        layout_maps.append(s2d)

        out_map = os.path.join(BASE_DIR, f"sensitivity_t8_layout_{li:03d}_meanOver{loaded}poses.png")
        out_hist = os.path.join(BASE_DIR, f"sensitivity_t8_layout_{li:03d}_hist_meanOver{loaded}poses.png")

        plot_map(s2d, f"T8 Sensitivity (layout {li}) mean over {loaded} poses", out_map)
        plot_hist(s2d, f"T8 Sensitivity Histogram (layout {li})", out_hist)

    # Combined mean over layouts (same “spirit” as your earlier code)
    s2d_all = np.mean(np.stack(layout_maps, axis=0), axis=0)
    out_map_all = os.path.join(BASE_DIR, f"sensitivity_t8_meanOver{len(LAYOUTS)}layouts.png")
    out_hist_all = os.path.join(BASE_DIR, f"sensitivity_t8_hist_meanOver{len(LAYOUTS)}layouts.png")
    plot_map(s2d_all, f"T8 Sensitivity mean over {len(LAYOUTS)} layouts", out_map_all)
    plot_hist(s2d_all, f"T8 Sensitivity Histogram mean over {len(LAYOUTS)} layouts", out_hist_all)