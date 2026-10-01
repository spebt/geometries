import numpy as np
import time
import torch
import os
import h5py
import pandas as pd
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn, MofNCompleteColumn

def get_flist(input_file: str) -> list:
    with open(input_file, "r") as f:
        flist = f.readlines()
        flist = [f.strip() for f in flist]
        return flist

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
    phantom_filename = "/vscratch/grp-rutaoyao/Omer/spebt/data/sai_10mm/hot_rods_phantom_10.0_mm_x_10.0_mm.pt"
    phantom_data = torch.load(phantom_filename)
    phantom_tensor = phantom_data["Phantom tensor"]

    h, w = phantom_tensor.shape
    pad_h = (IMG_SIZE - h) // 2
    pad_w = (IMG_SIZE - w) // 2
    
    phantom_padded = torch.nn.functional.pad(phantom_tensor, (pad_w, pad_w, pad_h, pad_h), "constant", 0)
    
    # Flatten the final, correctly-sized phantom
    phantom_flat = phantom_padded.view(-1)
    
    print(f"Original phantom shape: {phantom_tensor.shape}")
    print(f"Padded phantom shape:   {phantom_padded.shape}")
    if phantom_flat.shape[0] != sfov_expected:
        raise ValueError("FATAL: Padded phantom size does not match expected system matrix FOV.")

    # --- Batch Processing ---
    all_projs = []

    # Setup a nice progress bar
    progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )

    with progress:
        task = progress.add_task("[green]Projecting...", total=len(flist))
        for fname in flist:
            with h5py.File(fname, "r") as h5f:
                # Load one chunk of the matrix
                matrix_chunk = torch.tensor(h5f["ppdfs"][:]).view(1, sproj, sfov_expected)
                # print(f"{fname}  →  shape = {ppdfs.shape}") 
                # Perform matrix multiplication on just this chunk
                proj_chunk = torch.matmul(matrix_chunk, phantom_flat)
                all_projs.append(proj_chunk)
            
            progress.update(task, advance=1)

    # Combine the results from all the chunks
    final_projs = torch.cat(all_projs, dim=0)

    # --- Save the final result ---
    output_path = os.path.join(data_dir, "derenzo-projs.npy")
    np.save(output_path, final_projs.numpy())
    
    print("\nProjection complete!")
    print(f"Final projection shape: {final_projs.shape}")
    print(f"Saved projections to: {output_path}")