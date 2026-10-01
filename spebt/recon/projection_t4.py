import numpy as np
import time
import torch
import os
import h5py
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn, MofNCompleteColumn

def get_flist(input_file: str) -> list:
    with open(input_file, "r") as f:
        flist = [line.strip() for line in f]
    return flist

def shift_image_zeropad(img: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """
    Shift 2D image by (dx, dy) pixels with zero padding (no wrap-around).
    dx > 0: shift right, dx < 0: left
    dy > 0: shift up,    dy < 0: down   (because img[y, x])
    """
    H, W = img.shape
    out = torch.zeros_like(img)

    # source and destination ranges
    x_src_start = max(0, -dx)
    x_src_end   = min(W, W - dx)         # exclusive
    x_dst_start = max(0, dx)
    x_dst_end   = x_dst_start + (x_src_end - x_src_start)

    y_src_start = max(0, -dy)
    y_src_end   = min(H, H - dy)
    y_dst_start = max(0, dy)
    y_dst_end   = y_dst_start + (y_src_end - y_src_start)

    if x_src_end > x_src_start and y_src_end > y_src_start:
        out[y_dst_start:y_dst_end, x_dst_start:x_dst_end] = \
            img[y_src_start:y_src_end, x_src_start:x_src_end]

    return out

if __name__ == "__main__":

    # --- Setup ---
    torch.device("cpu")
    data_dir = "/vscratch/grp-rutaoyao/Omer/spebt/data/sai_10mm"
    outputs_dir = os.path.join(data_dir, "sai_10mm")
    flist = get_flist(os.path.join(data_dir, "dataset_flist.csv"))

    IMG_SIZE = 200
    sfov_expected = IMG_SIZE * IMG_SIZE
    sproj = 3360

    # --- Phantom Loading and Resizing ---
    phantom_filename = os.path.join(
        data_dir, "hot_rods_phantom_10.0_mm_x_10.0_mm.pt"
    )
    phantom_data = torch.load(phantom_filename, map_location="cpu")
    phantom_tensor = phantom_data["Phantom tensor"]

    h, w = phantom_tensor.shape
    pad_h = (IMG_SIZE - h) // 2
    pad_w = (IMG_SIZE - w) // 2

    phantom_padded = torch.nn.functional.pad(
        phantom_tensor, (pad_w, pad_w, pad_h, pad_h), "constant", 0
    )
    print(f"Original phantom shape: {phantom_tensor.shape}")
    print(f"Padded phantom shape:   {phantom_padded.shape}")

    if phantom_padded.numel() != sfov_expected:
        raise ValueError("FATAL: Padded phantom size does not match expected system matrix FOV.")

    # --- Define T4 translations in pixels (±0.4 mm, 0.05 mm/px -> 8 px) ---
    shift_mm = 0.4
    mm_per_px = 0.05
    shift_px = int(round(shift_mm / mm_per_px))  # 8 pixels

    # (dx, dy) in pixels for 4 sub-scans
    t4_shifts_px = [
        (-shift_px, -shift_px),  # (-0.4, -0.4) mm
        ( shift_px,  shift_px),  # (+0.4, +0.4) mm
        (-shift_px,  shift_px),  # (-0.4, +0.4) mm
        ( shift_px, -shift_px),  # (+0.4, -0.4) mm
    ]

    # Precompute flattened phantoms for each T4 shift
    t4_phantoms_flat = []
    for (dx, dy) in t4_shifts_px:
        shifted = shift_image_zeropad(phantom_padded, dx, dy)
        t4_phantoms_flat.append(shifted.view(-1))
    print("Prepared 4 T4-shifted phantoms (±0.4 mm in x,y).")

    # --- Batch Processing with T4 ---
    all_projs = []

    progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )

    with progress:
        task = progress.add_task("[green]Projecting with T4...", total=len(flist))
        for fname in flist:
            with h5py.File(fname, "r") as h5f:
                matrix_chunk = torch.tensor(h5f["ppdfs"][:]).view(1, sproj, sfov_expected)

            # Sum projections over the 4 translated phantoms
            proj_sum = torch.zeros(1, sproj)
            for ph_flat in t4_phantoms_flat:
                proj_sum += torch.matmul(matrix_chunk, ph_flat)

            all_projs.append(proj_sum)
            progress.update(task, advance=1)

    final_projs = torch.cat(all_projs, dim=0)

    # --- Save the final (T4) result ---
    output_path = os.path.join(data_dir, "derenzo-projs_T4.npy")
    np.save(output_path, final_projs.numpy())

    print("\nProjection complete (T4).")
    print(f"Final projection shape: {final_projs.shape}")
    print(f"Saved projections to: {output_path}")