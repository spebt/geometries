#!/usr/bin/env python3
import h5py, glob, numpy as np

BASE = "/vscratch/grp-rutaoyao/Omer/spebt/data/sai_10mm"
files = sorted(glob.glob(f"{BASE}/position_*_ppdfs_t8_*.hdf5"))

if not files:
    raise SystemExit(f"No files found under {BASE}/position_*_ppdfs_t8_*.hdf5")

def pct(x: np.ndarray) -> float:
    return 100.0 * float(x.mean())

print(f"Found {len(files)} T8 files\n")

for path in files:
    with h5py.File(path, "r") as h5:
        P = h5["ppdfs"][:]  # shape (n_crystals, n_pixels)
        dx = float(h5.attrs.get("dx_mm", np.nan))
        dy = float(h5.attrs.get("dy_mm", np.nan))

    pmin = float(np.nanmin(P))
    pmax = float(np.nanmax(P))

    bad_lo = P < 0.0
    bad_hi = P > 1.0
    bad = bad_lo | bad_hi

    # per-pixel sums and max across crystals
    sum_pix = P.sum(axis=0)         # (n_pixels,)
    max_pix = P.max(axis=0)         # (n_pixels,)

    sum_min = float(np.nanmin(sum_pix))
    sum_max = float(np.nanmax(sum_pix))
    max_min = float(np.nanmin(max_pix))
    max_max = float(np.nanmax(max_pix))

    tag = path.split("_")[-1].replace(".hdf5", "")  # 00..07
    base = path.split("/")[-1]

    print(
        f"{base:40s} | pose={tag:>2s} dx={dx:+.3f} dy={dy:+.3f} | "
        f"P[min,max]=[{pmin:.4g},{pmax:.4g}] "
        f"bad={pct(bad):.3f}% (neg={pct(bad_lo):.3f}%, >1={pct(bad_hi):.3f}%) | "
        f"sum_c per-pix [min,max]=[{sum_min:.4g},{sum_max:.4g}] | "
        f"max_c per-pix [min,max]=[{max_min:.4g},{max_max:.4g}]"
    )