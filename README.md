# Main-Project-2D-to-3D-Dental-X-Ray-Reconstruction
# Physics-Based CBCT Reconstruction from Panoramic X-rays

A Python-based experimental project that explores 3D Cone-Beam Computed Tomography (CBCT) volume reconstruction from a 2D panoramic dental X-ray using physics-inspired projection modelling and existing 3D medical imaging templates.

The system uses the Beer–Lambert attenuation model to simulate X-ray projections, selects a suitable CBCT template by comparing simulated projections with the input panoramic X-ray, and refines the selected volume using iterative updates, denoising, and sharpening techniques.

## Features

- **Panoramic X-ray Processing:** Loads, normalizes, and resizes grayscale dental X-ray images.
- **3D Medical Image Processing:** Loads volumetric templates in NIfTI format (`.nii` and `.nii.gz`).
- **Beer–Lambert Projection Model:** Simulates 2D projections from 3D volumes using an attenuation-based model.
- **Template Selection:** Selects the template with the lowest mean squared error (MSE) between its simulated projection and the input image.
- **Iterative Refinement:** Updates the selected volume based on projection error using depth-weighted backprojection.
- **3D Denoising:** Applies total variation denoising to reduce noise while preserving structural details.
- **Volume Sharpening:** Uses a 3D unsharp masking technique to enhance image details.
- **3D Visualization:** Generates axial, sagittal, and coronal views, a maximum intensity projection (MIP), and a simulated panoramic projection.
- **Medical Image Export:** Saves the reconstructed volume in compressed NIfTI format.

## Technologies Used

- Python
- NumPy
- PyTorch
- NiBabel
- Pillow (PIL)
- Matplotlib
- SciPy
- scikit-image

## Project Structure

```text
project/
├── main.py
├── xray.jpeg
├── data/
│   ├── template1.nii.gz
│   └── template2.nii.gz
└── results/
    ├── cbct_physics.nii.gz
    └── physics_reconstruction.png
```

**Note:** `main.py` represents the Python script containing the provided implementation. Replace it with your actual script filename if it differs. The `data/` directory must contain at least one NIfTI volume template.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Replace the URL with your actual GitHub repository link.

### 2. Install the dependencies

```bash
pip install numpy torch nibabel Pillow matplotlib scipy scikit-image
```

A compatible Python environment and PyTorch installation are required.

## Input Requirements

The project expects the following inputs:

- **Panoramic X-ray:** A grayscale-compatible dental X-ray image named `xray.jpeg`.
- **CBCT templates:** One or more 3D medical imaging volumes in NIfTI format (`.nii` or `.nii.gz`) placed inside the `data/` directory.

The input image and template paths can be modified in the `main()` function.

## How to Run

Execute the Python script from the project directory:

```bash
python main.py
```

The application performs the following steps:

1. Loads and normalizes the panoramic X-ray.
2. Loads the available CBCT template volumes and resizes them to a standard resolution.
3. Simulates a 2D projection for each template using the Beer–Lambert model.
4. Selects the template with the lowest projection MSE.
5. Iteratively refines the selected volume using projection error.
6. Applies total variation denoising and 3D sharpening.
7. Saves the reconstructed volume and visualization outputs.

## Output Files

The application generates a `results/` directory containing:

| File | Description |
|---|---|
| `cbct_physics.nii.gz` | Reconstructed 3D volume in compressed NIfTI format |
| `physics_reconstruction.png` | Visualization of the input X-ray, three orthogonal volume views, maximum intensity projection, and simulated projection |

The NIfTI volume can be inspected using compatible medical image visualization software.

## Methodology

### 1. Beer–Lambert Projection

The implementation models attenuation using an exponential relationship:

\[
I = e^{-\sum \mu}
\]

where \(\mu\) represents the scaled voxel values along the selected projection axis.

### 2. Template Matching

Each 3D template is projected into a 2D image and compared with the input panoramic X-ray using mean squared error. The template with the lowest error is selected as the starting volume.

### 3. Iterative Refinement

The selected volume is updated over multiple iterations using the difference between the input X-ray and the simulated projection. A depth-weighting kernel distributes the correction across the volume.

### 4. Denoising and Sharpening

Total variation denoising reduces unwanted variations, while Gaussian-based unsharp masking enhances local details in the resulting volume.

## Limitations

- This is an experimental, template-based reconstruction approach and does not perform conventional CT reconstruction from multiple calibrated projection measurements.
- A single panoramic X-ray does not contain sufficient information to uniquely determine the complete internal 3D anatomy.
- The reconstructed volume is influenced by the anatomy and characteristics of the selected template.
- The Beer–Lambert model and refinement procedure use simplifying assumptions and are not a complete physical model of dental X-ray imaging.
- The output should not be interpreted as a clinically validated CBCT scan or used for diagnosis or treatment planning without appropriate validation.

## Future Improvements

- Incorporate anatomically appropriate template selection.
- Improve projection geometry and attenuation modelling.
- Evaluate reconstruction quality using suitable quantitative metrics.
- Explore validated reconstruction algorithms and additional imaging information.
- Develop a graphical interface for loading images and visualizing reconstructed volumes.

## Disclaimer

This project is intended for educational and research purposes. The generated volume is an experimental reconstruction and is not a substitute for an actual CBCT scan or professional medical evaluation.
