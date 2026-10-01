if __name__ == "__main__":
    import os
    import numpy as np

    topdir = "/vscratch/grp-rutaoyao/Omer/spebt/data/sai_10mm"
    
    layout_idxs = [0,1]
    pose_idxs = np.arange(8)
    fnames = [
        f"position_{layout:03d}_ppdfs_t8_{pose:02d}.hdf5" 
        for layout in layout_idxs 
        for pose in pose_idxs
        ]
    
    with open("/vscratch/grp-rutaoyao/Omer/spebt/data/sai_10mm/dataset_flist.csv", "w") as f:
        for fname in fnames:
            f.write(os.path.join(topdir, fname) + "\n")
