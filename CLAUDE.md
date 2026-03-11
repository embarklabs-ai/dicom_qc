# CLAUDE.md - DICOM QC Library

## Project Overview

`dicom_qc` is a Python library for batch quality control review of de-identified DICOM data. It provides:
- Geometry and metadata validation checks
- Interactive 3-pane viewers for Jupyter notebooks
- HTML report generation for offline review
- Detection of viewer compatibility issues

Primary use case: QC review of de-identified MRI/CT data before ingestion into XNAT or research pipelines.

## Repository Structure

```
dicom_qc/
├── dicom_qc/                     # Main package
│   ├── __init__.py              # Exports QuickCheck, DicomVolume
│   ├── core/
│   │   ├── volume.py            # DicomVolume dataclass, LPS reorientation
│   │   ├── dicom_loader.py      # DICOM loading via SimpleITK and pydicom
│   │   ├── geometry.py          # QC checks (slice ordering, gaps, anisotropy, etc.)
│   │   └── errors.py            # Custom exceptions
│   ├── visualization/
│   │   ├── base.py              # VolumeRenderer base class (LPS orientation, windowing)
│   │   └── snapshots.py         # Thumbnail generation
│   ├── widgets/
│   │   ├── __init__.py          # Exports MultiViewViewer, DicomHeaderViewer
│   │   └── multiview.py         # Interactive viewer and header browser
│   ├── quickcheck.py            # Main QuickCheck class for batch review
│   ├── quickcheck_display.py    # Interactive Jupyter display mixin
│   └── utils/
│       └── errors.py            # Error classes
├── examples/
│   ├── local.ipynb              # Production notebook for local data
│   ├── local_dev.ipynb          # Development notebook (uses local source)
│   ├── xnat.ipynb               # XNAT integration notebook
│   └── xnat_ssl_patch.ipynb     # XNAT with legacy SSL/TLS support
└── pyproject.toml
```

## Key Classes

### QuickCheck (quickcheck.py)
Main entry point for batch review:
```python
qc = QuickCheck(data_dir)
qc.discover()           # Scan for DICOM files, build hierarchy
qc.discover(defaced=True)               # Local: load from DEFACED/ dirs where available, fallback to DICOM/
qc.discover_xnat(project, defaced=True)  # XNAT: load DEFACED resource where available, fallback to DICOM
qc.process_all()        # Run QC checks (parallel, auto-save)
qc.generate_html_report(path)  # Export HTML report
qc.display()            # Interactive Jupyter dashboard (paginated)
qc.save()               # Save state (auto-saves during processing)
qc.reset()              # Clear everything and start fresh
```

### DicomVolume (core/volume.py)
Represents a loaded DICOM volume with:
- `pixel_array`: 3D numpy array [slices, rows, cols]
- `slice_locations`, `image_positions`, `image_orientation`
- `pixel_spacing`, `slice_thickness`
- `get_lps_array()`: Returns volume reoriented to standard LPS

### GeometryQC (core/geometry.py)
Runs validation checks:
- `check_slice_ordering()` - Monotonic order, duplicates
- `check_orientation_consistency()` - Valid direction cosines
- `check_frame_of_reference()` - Slices coplanar
- `check_gap_detection()` - Missing slices, irregular spacing
- `check_voxel_anisotropy()` - Thick slice detection
- `check_slice_count()` - Truncation detection
- `check_series_type()` - DTI/DSC/MoCo detection
- `run_all_checks()` - Returns QCReport

### MultiViewViewer (widgets/multiview.py)
Interactive 3-pane viewer for Jupyter:
- Axial, coronal, sagittal views with crosshairs
- Click to navigate, scroll to change slice
- Window/level presets
- Requires `%matplotlib widget` in notebook

## QC Checks Performed

**Geometry checks (per volume):**
1. Geometry Metadata - missing DICOM tags (PixelSpacing, ImageOrientation, etc.)
2. 4D Data - detects fMRI/DTI/DSC (displays first volume only)
3. Reconstructability - multi-orientation localizers, non-reconstructable data
4. Slice Ordering - monotonic, no duplicates
5. Orientation Consistency - valid unit vectors
6. Frame of Reference - slices coplanar
7. Gap Detection - missing slices, irregular spacing
8. Voxel Anisotropy - ratio > 2x = WARNING
9. Slice Count - truncation detection

**DICOM-level checks (per series):**
10. JPEG-2000 Encoding - flags J2K transfer syntax (some render blurry in OHIF)
11. Temporal Metadata - missing timing for perfusion series

## Viewer Compatibility Notes

Series types that may fail in strict viewers (ITK-SNAP, 3D Slicer):

| Series Type | Issue | Reason |
|-------------|-------|--------|
| DTI | Gradient directions in private tags | Vendors use non-standard encoding |
| DSC Perfusion | Missing temporal metadata | TriggerTime often absent |
| MoCo | Inconsistent geometry | Motion correction warps positions |
| Derived (ADC, CBV) | Secondary capture | May lack spatial metadata |

MicroDICOM is permissive; ITK-SNAP/3D Slicer enforce strict DICOM conformance.

## Defaced Resource Loading

### Feature: `defaced` flag on `discover()` and `discover_xnat()`

```python
qc.discover(defaced=True)               # Local data
qc.discover_xnat(project, defaced=True)  # XNAT data
```

When `defaced=True`:
- Scans **with** DEFACED data: load from DEFACED (purple "DEFACED" badge in `qc.display()` and HTML report)
- Scans **without** DEFACED data: fall back to DICOM (no badge)
- Typically only anatomical scans are defaced; fMRI/other scans load from DICOM as usual
- When `defaced=False` (default): files in DEFACED directories/resources are excluded to avoid double-counting

### Local data layout

Local defaced data uses a `DEFACED/` directory alongside `DICOM/`:
```
scan_folder/DICOM/*.dcm      # Original DICOMs
scan_folder/DEFACED/*.dcm    # Defaced DICOMs (same SeriesInstanceUID)
```
`_is_in_defaced_dir()` checks if any path component is "DEFACED" (case-insensitive).

### Implementation details

- `_fetch_scan_files(scan, defaced)` returns `(files, resource_label)` tuple where `resource_label` is `"DEFACED"` or `"DICOM"`
- `SeriesInfo._xnat_resource` tracks which resource was loaded per series
- `quickcheck_display.py` renders a purple badge when `_xnat_resource == "DEFACED"`

### xnatpy `data_path` caveat

xnatpy only populates `data_path` for "native" archive resources (e.g. DICOM). Pipeline-created resources (e.g. DEFACED) return `None` for `data_path` even when files exist on disk at the expected mount path.

**Fix**: `_find_scan_base_dir()` derives the local mount path from the DICOM resource's `data_path` (which IS populated), then swaps the resource label in the path to locate DEFACED files:
```
/data/projects/{project}/experiments/{session}/SCANS/{scan_id}/DICOM/  -> known via data_path
/data/projects/{project}/experiments/{session}/SCANS/{scan_id}/DEFACED/ -> derived
```

Without this fix, empty `data_path` causes `load_from_xnat()` to fall back from SimpleITK to pydicom manual loading, which produces incorrect orientation in coronal/sagittal views.

## Development Notes

### Dependencies
- `pydicom` - DICOM file parsing
- `SimpleITK` - Volume loading, LPS reorientation
- `numpy` - Array operations
- `matplotlib` + `ipympl` - Visualization
- `ipywidgets` - Jupyter interactivity
- `jinja2` - HTML report templates
- `pillow` - Image processing

### LPS Coordinate System
After `get_lps_array()`:
- Array indexed as `[S, P, L]` (Superior, Posterior, Left)
- Spacing tuple is `(S_spacing, P_spacing, L_spacing)`
- Axial slice: `array[i, :, :]` shows [P, L]
- Coronal slice: `array[:, j, :]` shows [S, L]
- Sagittal slice: `array[:, :, k]` shows [S, P]

### Adding New QC Checks
1. Add method to `GeometryQC` returning `QCResult`
2. Add to `run_all_checks()` results list
3. For DICOM-tag-level checks, add to `QuickCheck._check_*` methods

### Installation

```bash
# Production install
uv pip install .

# Development install (editable)
uv pip install -e .

# With XNAT support
uv pip install ".[xnat]"
```

### Testing Workflow
```bash
uv pip install -e .
jupyter notebook examples/local_dev.ipynb  # Development (uses local source)
# Or for production testing:
jupyter notebook examples/local.ipynb
```

## Common Issues

**"widget is not a recognised GUI loop"**: Restart kernel after `uv pip install ipympl`

**Horizontal scroll in viewer**: Check `overflow='hidden'` on widget layouts

**Blurry images in OHIF**: Some JPEG-2000 encoded data renders blurry; root cause not yet identified

**Slice thickness mismatch**: SliceThickness tag vs calculated spacing from positions - both shown when different

**DEFACED orientation flipped**: If coronal/sagittal views are upside down for DEFACED scans, check that `_find_scan_base_dir()` is deriving local paths correctly. Empty `data_path` from xnatpy causes fallback to pydicom loading which produces wrong orientation.

## File Patterns

- DICOM files: `*.dcm` in data directory
- HTML reports: `*_report.html` in output directory
- Save files: `quickcheck_state.pkl` (pickle format)

## Storage (Scaled Mode)

For large projects, QuickCheck uses disk-based storage in `{data_dir}/_dicom_qc/`:

```
_dicom_qc/
├── qc_database.sqlite3   # SQLite database (series metadata, QC results)
├── thumbnails/           # JPEG thumbnails (~10KB each)
│   ├── ab/
│   │   └── abc123def456.jpg
│   └── cd/
│       └── cde789ghi012.jpg
```

**Benefits at scale:**
- Memory: ~50MB vs 1-5GB for 100K series (thumbnails on disk, not base64 in memory)
- UI: <500ms filter/pagination vs 30-60s full re-render
- Processing: Parallel with auto-save (resume on crash)

**To start fresh:**
```python
qc.reset()  # Deletes _dicom_qc/ directory and clears state
```
