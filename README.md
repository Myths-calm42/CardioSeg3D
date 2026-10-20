# CardioSeg: ACDC Cardiac MRI Segmentation

A research prototype for 3D cardiac MRI segmentation using **MONAI** and a **3D U-Net**. The application accepts a 3D NIfTI volume (`.nii` or `.nii.gz`), applies the project's preprocessing pipeline, predicts cardiac structure labels, and provides visualizations and a downloadable prediction bundle.

> **Research use only. Not for clinical diagnosis or treatment decisions.**

## Features

- Streamlit interface for uploading a 3D cardiac MRI volume.
- Deterministic preprocessing with MONAI.
- 3D U-Net inference using sliding-window inference.
- Slice-wise inspection of MRI intensity, predicted labels, and segmentation overlays.
- Predicted voxel counts/statistics and downloadable `.npz` prediction bundle.
- Training, testing, evaluation, and visualization notebooks/scripts.

## Segmentation labels

| Label ID | Structure |
|---:|---|
| 0 | Background |
| 1 | Right ventricle (RV) |
| 2 | Myocardium (MYO) |
| 3 | Left ventricle (LV) |

## Model architecture

The shared model definition uses a MONAI 3D U-Net with:

- Input channels: 1
- Output classes: 4
- Channels: `(16, 32, 64, 128, 256)`
- Strides: `(2, 2, 2, 2)`
- Residual units per level: 2
- Batch normalization
- Dropout: 0.1

The inference checkpoint must match this architecture and the preprocessing used during training.

## Preprocessing

The shared preprocessing pipeline includes:

1. Load the NIfTI image and ensure channel-first format.
2. Reorient the volume to RAS.
3. Resample to target spacing `(1.5, 1.5, 3.0)` mm.
4. Scale intensities using the 0.5th and 99.5th percentiles to `[0, 1]`.
5. Normalize non-zero intensities.
6. Crop around the foreground with a margin of 15 voxels.
7. Pad to at least `(128, 128, 16)` voxels.

The saved prediction bundle represents the **preprocessed coordinate space**. It is not automatically restored to the original MRI orientation, spacing, or dimensions.

## Repository structure

```text
Monai_ACDC-Project-work/
├── 01_model_training_pipeline.ipynb
├── 02_model_testing.py
├── 03_model_evaluation.ipynb
├── 04_model_segmentation_visualization_and_analysis.ipynb
├── app.py
├── inference.py
├── preprocessing_and_model.py
├── segmentation_visualization.py
├── requirements.txt
├── checkpoints/
│   └── best_model.pt
└── results/
    ├── augmentation_patches.png
    ├── multi_patient_preprocessing.png
    ├── preprocessing_report.csv
    ├── training_curves.png
    └── val_metrics.json
```

Actual files in `results/` may vary. The sample `.npz` prediction file is optional and may be excluded if it is not needed in the repository.

## Requirements

- Python 3.10 or a compatible Python version supported by the installed PyTorch/MONAI releases
- PyTorch
- MONAI
- Streamlit
- nibabel
- NumPy
- SciPy
- Matplotlib

Install the packages listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

### Windows: recommended local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks environment activation, run this for the current PowerShell session and try activation again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

**Version note:** Use compatible PyTorch and MONAI versions. If you use MONAI 1.6.1, it requires PyTorch 2.8.0 or newer. The CPU-only PyTorch build is useful for local smoke testing but may make 3D inference slow.

## Run the Streamlit application

From the project root (the directory containing `app.py`):

```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`.

Upload a 3D NIfTI image (`.nii` or `.nii.gz`) and click **Run 3D segmentation**. The checkpoint must be available at:

```text
checkpoints/best_model.pt
```

If the checkpoint is stored elsewhere, configure the environment variable `ACDC_CHECKPOINT` to the full path to the checkpoint before starting the app. You can also configure `ACDC_PROJECT_DIR` for the project directory, provided the code uses that variable as its default path.

## Run command-line inference

If your `inference.py` imports and checkpoint configuration are up to date, use:

```bash
python inference.py --image path/to/patient_volume.nii.gz
```

Optional arguments include:

```bash
python inference.py --image path/to/patient_volume.nii.gz --checkpoint checkpoints/best_model.pt --output-dir outputs
```

The output is a compressed `.npz` bundle containing the preprocessed image, predicted labels, label names, target spacing, and input filename.

## Results and evaluation

The `results/` directory can contain preprocessing visualizations, augmentation examples, training curves, preprocessing reports, and validation metrics. Review the actual evaluation outputs before reporting performance figures in a paper, portfolio, or resume. Do not treat voxel accuracy alone as a segmentation-quality metric because it includes background voxels; report per-structure Dice and other foreground metrics where available.

## Data and privacy

- Do not commit the complete ACDC dataset or other restricted medical data.
- Do not publish patient-identifiable information.
- Check notebook outputs for personal file paths, credentials, or sensitive information before publishing.
- Only include sample predictions when you have permission to share them.

## Limitations

- This is a research prototype, not a clinically validated medical device.
- Prediction quality depends on the training data, checkpoint, preprocessing, and input volume.
- The downloadable `.npz` output is in preprocessed coordinates and is not automatically converted back to the original NIfTI geometry.
- CPU inference can be slow for 3D volumes.
- A successful app launch does not guarantee that every NIfTI volume will be supported.

## Acknowledgements

Built with [MONAI](https://monai.io/), [PyTorch](https://pytorch.org/), and [Streamlit](https://streamlit.io/). The ACDC dataset is associated with the Automated Cardiac Diagnosis Challenge; follow the dataset's terms and citation requirements when using it.

## License

Add a license file only after choosing a license appropriate for your code and confirming that it is compatible with the licenses and terms of any datasets, weights, and dependencies you redistribute.
