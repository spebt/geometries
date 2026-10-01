#!/usr/bin/env python3
"""
psf_mlem_torch_nonmpi.py
------------------------
PSF-modeled MLEM (resolution recovery) for your chunked system matrix setup.

Forward model:    y = H ( B x )
Backprojection:   bp = B^T H^T ( p / y )
Sensitivity:      s  = B^T H^T 1      (precomputed once)

Inputs:
  - flist: text/csv with full paths to matrix files (one per subset)
  - projs: derenzo-projs.npy with shape (len(flist), 3360)
  - each matrix file has dataset "ppdfs" that reshapes to (3360, 200*200)

Outputs:
  - recon_psf_mlem.npz with:
      estimates: [n_saved, 200, 200]
      times:     per-iteration runtime
      meta:      dict with settings + psf params
"""

import os
import time
import argparse
import numpy as np
import h5py
import torch
import torch.nn.functional as F


# ------------------------- IO helpers -------------------------
def read_flist(path: str) -> list[str]:
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f.readlines()]
    lines = [ln for ln in lines if ln]  # drop empty
    return lines


def load_matrix_chunk(path: str, sproj: int, sfov: int, device: torch.device) -> torch.Tensor:
    # returns float32 tensor shape (sproj, sfov)
    with h5py.File(path, "r") as h5f:
        m = torch.from_numpy(h5f["ppdfs"][:])
    m = m.view(sproj, sfov).to(device=device, dtype=torch.float32)
    return m


# ------------------------- PSF (Gaussian) -------------------------
def gaussian_1d_kernel(sigma_px: float, radius: int, device: torch.device) -> torch.Tensor:
    """
    radius: half-width. kernel size = 2*radius + 1.
    """
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-(x * x) / (2.0 * sigma_px * sigma_px))
    k = k / k.sum()
    return k


def make_separable_gaussian_psf(sigma_px: float, device: torch.device, truncate: float = 3.0):
    """
    Build separable conv weights for conv2d:
      - horizontal kernel: (1,1,1,K)
      - vertical kernel:   (1,1,K,1)
    truncate=3.0 => radius ~ 3*sigma
    """
    if sigma_px <= 0:
        raise ValueError("sigma_px must be > 0")

    radius = int(np.ceil(truncate * sigma_px))
    radius = max(radius, 1)
    k1d = gaussian_1d_kernel(sigma_px, radius, device=device)

    kx = k1d.view(1, 1, 1, -1)  # (out=1,in=1,1,K)
    ky = k1d.view(1, 1, -1, 1)  # (out=1,in=1,K,1)
    pad = radius
    return kx, ky, pad


def apply_psf_separable(img_2d: torch.Tensor, kx: torch.Tensor, ky: torch.Tensor, pad: int, pad_mode="reflect"):
    """
    img_2d: (H,W) float32
    returns blurred (H,W)
    """
    x = img_2d.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    # reflect padding to avoid edge darkening artifacts
    x = F.pad(x, (pad, pad, pad, pad), mode=pad_mode)
    x = F.conv2d(x, kx)
    x = F.conv2d(x, ky)
    return x.squeeze(0).squeeze(0)


# ------------------------- MLEM (PSF-modeled) -------------------------
@torch.no_grad()
def precompute_sensitivity(
    flist: list[str],
    sproj: int,
    sfov: int,
    img_dim: int,
    device: torch.device,
    kx: torch.Tensor,
    ky: torch.Tensor,
    pad: int,
    eps: float,
) -> torch.Tensor:
    """
    Compute s_raw = H^T 1, then s = B^T s_raw (≈ B s_raw).
    Returns s (sfov,)
    """
    s_raw = torch.zeros(sfov, device=device, dtype=torch.float32)

    for path in flist:
        m = load_matrix_chunk(path, sproj, sfov, device)
        # H^T 1 == sum over projection bins of each voxel column
        # m: (sproj, sfov) => sum over dim=0 gives (sfov,)
        s_raw += m.sum(dim=0)

    # apply B^T (Gaussian => symmetric)
    s_img = s_raw.view(img_dim, img_dim)
    s_blur = apply_psf_separable(s_img, kx, ky, pad).contiguous().view(-1)
    s_blur = torch.clamp(s_blur, min=eps)
    return s_blur


@torch.no_grad()
def psf_mlem_recon(
    flist: list[str],
    pdata: torch.Tensor,  # (n_files, sproj)
    sproj: int,
    sfov: int,
    img_dim: int,
    device: torch.device,
    iters: int,
    save_every: int,
    kx: torch.Tensor,
    ky: torch.Tensor,
    pad: int,
    eps: float,
    sens: torch.Tensor,  # (sfov,)
    conv_tol: float,
):
    estimate = torch.ones(sfov, device=device, dtype=torch.float32)

    estimates_history = []
    times_history = []

    for it in range(iters):
        t0 = time.time()
        prev = estimate.clone()

        # --- forward uses blurred estimate: x_blur = B x
        x_img = estimate.view(img_dim, img_dim)
        x_blur = apply_psf_separable(x_img, kx, ky, pad).contiguous().view(-1)

        # --- accumulate raw backprojection: bp_raw = H^T (p / (H x_blur))
        bp_raw = torch.zeros(sfov, device=device, dtype=torch.float32)

        for i, path in enumerate(flist):
            m = load_matrix_chunk(path, sproj, sfov, device)  # (sproj, sfov)
            p = pdata[i]  # (sproj,)

            # y = H x_blur
            y = torch.matmul(m, x_blur)  # (sproj,)
            y = torch.clamp(y, min=eps)

            r = p / y  # (sproj,)
            bp_raw += torch.matmul(m.transpose(0, 1), r)  # (sfov,)

        # apply B^T to the backprojection
        bp_img = bp_raw.view(img_dim, img_dim)
        bp = apply_psf_separable(bp_img, kx, ky, pad).contiguous().view(-1)
        bp = torch.clamp(bp, min=eps)

        # MLEM update
        estimate = estimate * (bp / sens)
        estimate = torch.clamp(estimate, min=0.0)

        dt = time.time() - t0
        times_history.append(dt)

        if (it % save_every) == 0:
            estimates_history.append(estimate.view(img_dim, img_dim).detach().cpu().numpy())

        # convergence check
        num = torch.norm(estimate - prev)
        den = torch.clamp(torch.norm(prev), min=eps)
        diff = (num / den).item()

        print(f"[ITER {it+1:04d}/{iters}] time={dt:.2f}s  diff={diff:.3e}")

        if diff < conv_tol:
            print(f"[STOP] Converged at iter {it+1} (diff={diff:.3e} < {conv_tol:.3e})")
            break

    return np.array(estimates_history), np.array(times_history)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Base directory containing dataset_flist.csv and matrix files")
    ap.add_argument("--flist", default="dataset_flist.csv", help="flist filename (relative to data-dir) or full path")
    ap.add_argument("--projs", default="derenzo-projs.npy", help="projections .npy file (relative to data-dir) or full path")
    ap.add_argument("--out", default="recon_psf_mlem.npz", help="output .npz filename (relative to data-dir unless absolute)")

    ap.add_argument("--img-dim", type=int, default=200)
    ap.add_argument("--sproj", type=int, default=3360)

    # PSF specification:
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument("--psf-fwhm-mm", type=float, default=0.25, help="Gaussian PSF FWHM in mm (default 0.25)")
    g.add_argument("--psf-sigma-px", type=float, default=None, help="Gaussian PSF sigma in pixels (overrides fwhm-mm)")

    ap.add_argument("--mm-per-px", type=float, default=0.05, help="mm per pixel (default 0.05)")
    ap.add_argument("--psf-truncate", type=float, default=3.0, help="truncate kernel at truncate*sigma (default 3)")

    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--conv-tol", type=float, default=1e-4)
    ap.add_argument("--eps", type=float, default=1e-8)

    ap.add_argument("--device", default=None, help="cuda or cpu (default auto)")
    args = ap.parse_args()

    # paths
    flist_path = args.flist if os.path.isabs(args.flist) else os.path.join(args.data_dir, args.flist)
    projs_path = args.projs if os.path.isabs(args.projs) else os.path.join(args.data_dir, args.projs)
    out_path   = args.out if os.path.isabs(args.out) else os.path.join(args.data_dir, args.out)

    # device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] device: {device}")

    # sizes
    img_dim = args.img_dim
    sfov = img_dim * img_dim
    sproj = args.sproj

    # load flist
    flist = read_flist(flist_path)
    if len(flist) == 0:
        raise RuntimeError(f"Empty flist: {flist_path}")
    print(f"[INFO] flist entries: {len(flist)}")

    # load projections
    pdata_np = np.load(projs_path)
    if pdata_np.ndim != 2 or pdata_np.shape[1] != sproj:
        raise RuntimeError(f"Expected projs shape (N,{sproj}), got {pdata_np.shape}")
    if pdata_np.shape[0] != len(flist):
        raise RuntimeError(f"Projs rows {pdata_np.shape[0]} != flist entries {len(flist)}")
    pdata = torch.from_numpy(pdata_np).to(device=device, dtype=torch.float32)
    print(f"[INFO] projections shape: {tuple(pdata.shape)}")

    # PSF parameters
    if args.psf_sigma_px is None:
        sigma_mm = args.psf_fwhm_mm / 2.355
        sigma_px = sigma_mm / args.mm_per_px
    else:
        sigma_px = float(args.psf_sigma_px)

    print(f"[INFO] PSF sigma_px={sigma_px:.4f}  (truncate={args.psf_truncate}σ)")

    # build PSF kernels
    kx, ky, pad = make_separable_gaussian_psf(sigma_px, device=device, truncate=args.psf_truncate)

    # precompute sensitivity once
    print("[INFO] Precomputing sensitivity: s = B^T H^T 1 ... (one pass over flist)")
    sens = precompute_sensitivity(
        flist=flist,
        sproj=sproj,
        sfov=sfov,
        img_dim=img_dim,
        device=device,
        kx=kx, ky=ky, pad=pad,
        eps=args.eps,
    )
    print(f"[INFO] sensitivity stats: min={sens.min().item():.3e}, max={sens.max().item():.3e}")

    # run PSF-MLEM
    print("[INFO] Running PSF-modeled MLEM ...")
    estimates, times = psf_mlem_recon(
        flist=flist,
        pdata=pdata,
        sproj=sproj,
        sfov=sfov,
        img_dim=img_dim,
        device=device,
        iters=args.iters,
        save_every=args.save_every,
        kx=kx, ky=ky, pad=pad,
        eps=args.eps,
        sens=sens,
        conv_tol=args.conv_tol,
    )

    meta = {
        "img_dim": img_dim,
        "sfov": sfov,
        "sproj": sproj,
        "n_files": len(flist),
        "iters_requested": args.iters,
        "save_every": args.save_every,
        "conv_tol": args.conv_tol,
        "eps": args.eps,
        "device": str(device),
        "psf_sigma_px": float(sigma_px),
        "psf_fwhm_mm": float(args.psf_fwhm_mm) if args.psf_sigma_px is None else None,
        "mm_per_px": float(args.mm_per_px),
        "psf_truncate": float(args.psf_truncate),
        "flist_path": flist_path,
        "projs_path": projs_path,
    }

    np.savez_compressed(out_path, estimates=estimates, times=times, meta=np.array([meta], dtype=object))
    print(f"[OK] Saved: {out_path}")
    print(f"[OK] estimates saved: {estimates.shape} (frames,H,W)")


if __name__ == "__main__":
    main()
    # --- Optional: also save final PNG (and a couple of snapshots) ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save_png(img2d, title, out_png, mm_per_px=0.05):
        img_dim = img2d.shape[0]
        width_mm = img_dim * mm_per_px
        extent = (-width_mm/2, width_mm/2, -width_mm/2, width_mm/2)

        plt.figure(figsize=(8, 8))
        plt.imshow(img2d.T, cmap="gray_r", origin="lower", extent=extent, interpolation="nearest")
        plt.colorbar(label="Image Intensity")
        plt.title(title)
        plt.xlabel("x (mm)")
        plt.ylabel("y (mm)")
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()

    # last saved frame (not necessarily iter==iters)
    final_img = estimates[-1]
    final_png = out_path.replace(".npz", "_final.png")
    save_png(final_img, "PSF-MLEM Final (last saved frame)", final_png, mm_per_px=args.mm_per_px)

    # also save a mid frame if available
    if estimates.shape[0] > 5:
        mid_img = estimates[len(estimates)//2]
        mid_png = out_path.replace(".npz", "_mid.png")
        save_png(mid_img, "PSF-MLEM Mid (saved frame)", mid_png, mm_per_px=args.mm_per_px)

    print(f"[OK] Saved PNGs: {final_png}")