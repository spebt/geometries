import numpy as np
import matplotlib.pyplot as plt
import os

img_path = "/vscratch/grp-rutaoyao/Omer/spebt/recon/recon_deblur_rl.npy"
output_dir = "/vscratch/grp-rutaoyao/Omer/spebt/plots/sai_10mm"
os.makedirs(output_dir, exist_ok=True)

img = np.load(img_path)  # <-- 2D array (H,W)
print("Loaded:", img.shape, img.dtype, "min/max:", img.min(), img.max())

MM_PER_PX = 0.05
H, W = img.shape
extent = (-W*MM_PER_PX/2, W*MM_PER_PX/2, -H*MM_PER_PX/2, H*MM_PER_PX/2)

def to_phantom_view(img_2d):
    return img_2d.T

plt.figure(figsize=(8,8))
plt.imshow(to_phantom_view(img), cmap="gray_r", origin="lower", extent=extent, interpolation="nearest")
plt.colorbar(label="Image Intensity")
plt.title("Deblurred (Richardson–Lucy)")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
out_path = os.path.join(output_dir, "final_reconstruction_deblur_rl.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print("Saved:", out_path)