#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn.functional as F

# ---- inputs ----
NPZ_PATH = "/vscratch/grp-rutaoyao/Omer/spebt/data/sai_10mm/recon_mlem_torch_derenzo_filtered.npz"  # change to your recon npz
OUT_NPY  = "recon_deblur_rl.npy"

MM_PER_PIXEL = 0.05
FWHM_MM = 0.30          # <-- measure this from a point-source recon
N_ITERS = 20            # 10-30 typically

# ---- load image ----
npz = np.load(NPZ_PATH)
img = npz["estimates"][-1].astype(np.float32)   # (H,W)
y = torch.from_numpy(img)[None,None]            # (1,1,H,W)

# ---- PSF kernel (Gaussian) ----
sigma_px = (FWHM_MM / 2.355) / MM_PER_PIXEL
ks = int(max(7, round(sigma_px * 6)))  # ~ +/-3 sigma
if ks % 2 == 0: ks += 1
ax = torch.arange(ks) - ks//2
xx, yy = torch.meshgrid(ax, ax, indexing="ij")
psf = torch.exp(-(xx**2 + yy**2) / (2*sigma_px**2))
psf = (psf / psf.sum()).to(torch.float32)
psf = psf[None,None]  # (1,1,ks,ks)
psf_flip = torch.flip(psf, dims=[-1,-2])

pad = ks//2
eps = 1e-6

# ---- Richardson–Lucy ----
x = torch.clamp(y.clone(), min=0.0) + eps
for _ in range(N_ITERS):
    conv_x = F.conv2d(x, psf, padding=pad)
    ratio  = y / (conv_x + eps)
    corr   = F.conv2d(ratio, psf_flip, padding=pad)
    x = torch.clamp(x * corr, min=0.0)

x_out = x[0,0].cpu().numpy()
np.save(OUT_NPY, x_out)
print(f"[DONE] Saved deblurred recon to {OUT_NPY}  (sigma_px={sigma_px:.3f}, ks={ks}, iters={N_ITERS})")