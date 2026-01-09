# DICOM QC Library

Batch quality control review of DICOM data within XNAT or on file system using Jupyter notebooks.

## Features

- Automatic discovery of DICOM files from local directories or XNAT projects
- Geometry and metadata validation checks (slice ordering, gaps, anisotropy, etc.)
- Interactive 3-pane viewer for Jupyter notebooks (axial, coronal, sagittal)
- DICOM header browser with tag filtering
- HTML report generation with thumbnails
- Save/restore progress for large datasets
- Detection of viewer compatibility issues (DTI, perfusion, derived series)

## Installation

### For Local Use

```bash
# Clone the repository
git clone https://github.com/embarklabs-ai/dicom_qc.git
cd dicom_qc

# Install with uv (recommended)
uv pip install .

# Or with pip
pip install .
```

### For XNAT Integration

```bash
# Install with XNAT support
uv pip install ".[xnat]"
```

### For Development

```bash
# Install in editable mode with dev dependencies
uv pip install -e ".[dev]"
```

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/embarklabs-ai/dicom_qc.git
cd dicom_qc
uv pip install .
uv pip install jupyterlab

# 2. Launch JupyterLab
uv run jupyter lab

# 3. Open examples/local.ipynb and follow the instructions
```

See [Running the Notebooks](#running-the-notebooks) for detailed instructions.

## Running the Notebooks

The `examples/` directory contains ready-to-use notebooks:

| Notebook | Description |
|----------|-------------|
| `local.ipynb` | QC review of local DICOM data (recommended starting point) |
| `local_dev.ipynb` | Development notebook (uses source from disk) |
| `xnat.ipynb` | QC review from XNAT project |

### Running `local.ipynb` (Production)

Use this notebook when working with the installed package.

```bash
# 1. Clone and install the package
git clone https://github.com/embarklabs-ai/dicom_qc.git
cd dicom_qc
uv pip install .

# 2. Install JupyterLab
uv pip install jupyterlab

# 3. Launch JupyterLab
uv run jupyter lab

# 4. In JupyterLab, open examples/local.ipynb
# 5. Update DATA_DIR to point to your DICOM data
# 6. Run all cells
```

### Running `local_dev.ipynb` (Development)

Use this notebook when modifying the source code. It loads the package from disk instead of the installed version, so changes take effect after restarting the kernel.

```bash
# 1. Clone the repository
git clone https://github.com/embarklabs-ai/dicom_qc.git
cd dicom_qc

# 2. Install dependencies (editable mode)
uv pip install -e .

# 3. Install JupyterLab
uv pip install jupyterlab

# 4. Launch JupyterLab
uv run jupyter lab

# 5. In JupyterLab, open examples/local_dev.ipynb
# 6. Update DATA_DIR to point to your DICOM data
# 7. Run all cells

# After making code changes:
# - Save your changes
# - Restart the kernel (Kernel > Restart Kernel)
# - Run all cells again
```

### Alternative: Classic Jupyter Notebook

If you prefer the classic notebook interface:

```bash
# Install classic notebook instead of JupyterLab
uv pip install notebook

# Launch
uv run jupyter notebook examples/local.ipynb
```

## Architecture

### Core Components

| Component | File | Description |
|-----------|------|-------------|
| **QuickCheck** | `quickcheck.py` | Main entry point - discovers files, runs QC, generates reports |
| **DicomVolume** | `core/volume.py` | Loaded DICOM volume with metadata and LPS reorientation |
| **GeometryQC** | `core/geometry.py` | Validation checks (slice ordering, gaps, anisotropy) |
| **MultiViewViewer** | `widgets/multiview.py` | Interactive 3-pane Jupyter viewer |
| **DicomHeaderViewer** | `widgets/multiview.py` | DICOM tag browser with filtering |
| **DicomLoader** | `core/dicom_loader.py` | DICOM loading and parsing |

### Dependencies

| Library | Purpose |
|---------|---------|
| **SimpleITK** | Volume loading with correct 3D geometry and orientation |
| **pydicom** | DICOM tag parsing for metadata extraction |
| **numpy** | Array operations |
| **matplotlib** | Visualization rendering |
| **ipympl** | Interactive matplotlib widget backend for Jupyter |
| **ipywidgets** | Jupyter UI controls (sliders, buttons, layouts) |

#### Why both SimpleITK and pydicom?

These libraries serve complementary purposes:

- **SimpleITK** handles the complex task of loading a DICOM series as a properly-oriented 3D volume. It correctly interprets `ImagePositionPatient`, `ImageOrientationPatient`, and slice ordering to produce a spatially-correct numpy array with accurate voxel spacing.

- **pydicom** provides direct access to DICOM tags for metadata extraction (series description, modality, transfer syntax, etc.) and for detecting special cases like JPEG-2000 encoding or missing temporal metadata that SimpleITK abstracts away.

#### How matplotlib and ipympl work together

The interactive viewer uses matplotlib with the ipympl backend:

1. **matplotlib** renders the figure (3 image panels with crosshairs) to an in-memory canvas
2. **ipympl** (`%matplotlib widget`) wraps the canvas as a Jupyter widget, enabling:
   - Live updates without re-rendering the entire cell
   - Mouse event handling (click to move crosshairs, scroll to change slice)
   - Integration with ipywidgets layouts (sliders, buttons)
3. **ipywidgets** provides the UI controls and layout container

The `%matplotlib widget` magic must be called before importing matplotlib to activate the ipympl backend.

## QC Checks

### Geometry Checks (per volume)

| Check | Description |
|-------|-------------|
| Geometry Metadata | Missing DICOM tags (PixelSpacing, ImageOrientation, etc.) |
| 4D Data | Detects fMRI/DTI/DSC datasets (displays first volume only) |
| Reconstructability | Multi-orientation localizers, non-reconstructable data |
| Slice Ordering | Monotonic order, no duplicates |
| Orientation Consistency | Valid direction cosines |
| Frame of Reference | Slices coplanar |
| Gap Detection | Missing slices, irregular spacing |
| Voxel Anisotropy | Flags thick slices (>2x ratio warning, >4x fail) |
| Slice Count | Truncation detection |

### DICOM-Level Checks (per series)

| Check | Description |
|-------|-------------|
| JPEG-2000 Encoding | Multi-layer encoding causes blurry rendering in OHIF |
| Temporal Metadata | Missing timing tags in dynamic/perfusion series |

## HTML Reports

Generate standalone HTML reports for offline review:

```python
qc.generate_html_report('qc_report.html')
```

Reports include:
- Summary statistics by QC status
- Thumbnail grid with 3-pane previews
- Filterable by status (Pass, Warning, Fail, etc.)
- OHIF viewer links (when connected to XNAT)

**Note**: In Jupyter, click "Trust HTML" for filters to work. Open in browser for external links.

## Troubleshooting

### "widget is not a recognized GUI loop"

Restart the kernel after installing ipympl, then run `%matplotlib widget` before any imports.

### Viewer appears blank for 30-60 seconds

This is normal. The ipympl backend transfers pixel data to the browser via WebSocket, which takes time for large volumes. Controls appear immediately; the image follows.

### Import errors

Ensure all dependencies are installed:
```bash
uv pip install ".[xnat]"  # For XNAT support
```

## Series Compatibility Notes

Some series types may have issues in strict DICOM viewers (ITK-SNAP, 3D Slicer):

| Series Type | Issue |
|-------------|-------|
| DTI | Gradient directions in private tags |
| DSC Perfusion | Missing temporal metadata |
| MoCo | Inconsistent geometry from motion correction |
| Derived (ADC, CBV) | May lack spatial metadata |

## License

MIT
