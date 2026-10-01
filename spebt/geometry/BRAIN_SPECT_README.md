# Brain SPECT Polygon-Panel Geometry

## System Description

This is the initial implementation of the Brain SPECT polygon-panel
self-collimating system based on the MATRICES architecture (Ma et al., TMI 2021).

### Key differences from the preclinical ring scanner (arg_ppdf_t8.py)

| Parameter | Preclinical Ring | Brain SPECT Polygon |
|-----------|-----------------|-------------------|
| FOV | 10 mm × 10 mm | 200 mm × 200 mm |
| Collimator | Stationary ring, pinholes | **Rotating cylinder**, apertures |
| Source motion | T8 elliptical bed motion | **None** — collimator rotates |
| Detectors | 4 concentric rings | Polygon flat panels |
| Crystal layers | 1 type | 3 layers (inner sparse, mid sparse, outer dense) |

### Collimator sampling strategy

The cylindrical collimator ring rotates while both the object and detector
panels remain stationary. Each rotation step produces one layout in the
`.tensor` file. This avoids moving the subject and simplifies the acquisition
protocol for brain imaging.

---

## Files

| File | Description |
|------|-------------|
| `generate_brain_spect_polygon.py` | Generates the `.tensor` geometry file |
| `../pymatcal/arg_ppdf_brain.py` | Computes the system matrix (ppdfs) |
| `../pymatcal/run_brain_parallel.sh` | SLURM submission script |

---

## Step 1: Generate Geometry

```bash
cd spebt/geometry

python generate_brain_spect_polygon.py \
    --n_panels 6 \
    --collimator_inner_r 120.0 \
    --collimator_thickness 2.0 \
    --n_apertures 200 \
    --aperture_size_mm 2.0 \
    --panel_length_mm 120.0 \
    --crystal_size_mm 3.0 \
    --n_layers 3 \
    --opening_ratio 0.333 \
    --n_rotations 2 \
    --rotation_step_deg 1.0 \
    --output_dir /vscratch/grp-rutaoyao/Tridev/brain_spect/geometry
```

This produces:
- `brain_spect_layouts_<MD5>_rot2_ang1p0deg_trans1x1_step0p0x0p0.tensor`
- `brain_spect_geometry.png` — top-view validation figure

---

## Step 2: Compute System Matrix

### On a single layout (interactive / testing):
```bash
cd spebt/pymatcal

python arg_ppdf_brain.py 0 \
    --layout_file /vscratch/.../brain_spect_layouts_<MD5>_rot2_ang1p0deg_trans1x1_step0p0x0p0.tensor \
    --output_dir /vscratch/grp-rutaoyao/Tridev/brain_spect/data \
    --fov_npix 20 20 \
    --fov_size_mm 200.0 200.0
```
(Use small `fov_npix` for quick testing — same physical size, fewer voxels.)

### On CCR (full 200×200 FOV):
```bash
# Update LAYOUT_FILE path in run_brain_parallel.sh first, then:
sbatch run_brain_parallel.sh
```

---

## Geometry Parameters (Default First Implementation)

### Collimator
- Inner radius: 120 mm
- Thickness: 2 mm (tungsten)
- Apertures: 200, each 2 mm arc length
- Opening ratio: 2 mm / (2π × 121 mm / 200) = ~0.52 → **adjust n_apertures to get ~1/3**
- Rotations: 2 steps at 1° each

### Detector Panels
- N panels: 6 (hexagon)
- Panel inner face radius: 123 mm (120 + 2 + 1 mm clearance)
- Panel length: 120 mm (5 modules × 8 crystals × 3 mm)
- Crystal size: 3 mm × 3 mm
- Layers: 3 (inner sparse, middle sparse, outer dense)
- Opening ratio on inner/middle layers: 1/3 (1 mm crystal, 2 mm gap)
- Total crystals: 6 panels × 3 layers × 40 positions = 720

### FOV
- Diameter: 200 mm
- Voxels: 200 × 200 at 1 mm/voxel
- Axial: 2D single-slice (extend to 3D in future)

---

## Crystal Layer Design (MATRICES Architecture)

```
Radial direction →  [FOV]  [collimator r=120mm]  [inner]  [middle]  [outer]

Inner layer  (sparse): ██░░░██░░░██░░░  (opening_ratio=1/3, 1mm crystal / 2mm gap)
Middle layer (sparse): ░██░░░██░░░██░░  (offset by half pitch from inner)
Outer layer  (dense):  ████████████████  (full crystal coverage)
```

Inner and middle layer crystals act as both detectors AND collimators —
they collimate photons for the outer (highest resolution) layer.
This is the MATRICES self-collimation principle.

---

## What to Change for MOBO Optimization

These parameters in `generate_brain_spect_polygon.py` are the MOBO design variables:

```
--collimator_inner_r    # collimator ring radius
--n_apertures           # number of apertures (controls sensitivity vs resolution)
--aperture_size_mm      # aperture size (opening ratio)
--panel_length_mm       # panel tangential extent (must be multiple of 24 mm)
--opening_ratio         # crystal occupancy on inner/middle layers
```
