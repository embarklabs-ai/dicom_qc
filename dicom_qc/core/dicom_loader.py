"""DICOM file loading with comprehensive error handling."""

import logging
from typing import List, Tuple, Optional, Any, Union, Dict, Callable, TypeVar
from pathlib import Path

import numpy as np

try:
    import SimpleITK as sitk

    HAS_SIMPLEITK = True
except ImportError:
    HAS_SIMPLEITK = False

from dicom_qc.core.volume import DicomVolume
from dicom_qc.utils.errors import (
    DicomLoadError,
    MissingTagError,
    GeometryError,
)

# Tags to check for missing geometry metadata
GEOMETRY_TAGS = [
    "PixelSpacing",
    "SliceThickness",
    "ImageOrientationPatient",
    "ImagePositionPatient",
]


def _check_missing_geometry_tags(ds) -> List[str]:
    """Check which geometry tags are missing from a DICOM dataset.

    Args:
        ds: pydicom Dataset

    Returns:
        List of missing tag names
    """
    missing = []
    for tag in GEOMETRY_TAGS:
        if not hasattr(ds, tag) or getattr(ds, tag) is None:
            missing.append(tag)
    return missing


logger = logging.getLogger(__name__)

# Type variable for generic file type (str path or XNAT file object)
T = TypeVar("T")


def _filter_4d_generic(
    files: List[T],
    read_metadata: Callable[[T], Any],
) -> Tuple[List[T], Optional[int]]:
    """
    Detect 4D data and filter to first temporal position (shared logic).

    Reads all files for robust detection. This is slower but more reliable
    than sampling, especially for XNAT where file reads may fail intermittently.

    Args:
        files: List of files (paths or XNAT file objects)
        read_metadata: Function that reads pydicom dataset from a file

    Returns:
        Tuple of (filtered_files, num_timepoints)
        If not 4D, returns (files, None)
    """
    if len(files) < 10:
        return files, None

    # Read first file for NumberOfTemporalPositions
    # Try multiple files in case some fail to read
    first_ds = None
    for f in files[:10]:
        first_ds = read_metadata(f)
        if first_ds is not None:
            break

    if first_ds is None:
        # Can't read any files - log warning and return all
        logger.warning(
            "4D detection: Could not read any of first 10 files. "
            "Check XNAT connection or file access."
        )
        return files, None

    # Get slice normal for oblique-safe sorting (Method 2)
    normal = np.array([0, 0, 1])  # Default to Z
    if first_ds is not None and hasattr(first_ds, "ImageOrientationPatient"):
        try:
            ori = [float(x) for x in first_ds.ImageOrientationPatient]
            row_dir = np.array(ori[:3])
            col_dir = np.array(ori[3:])
            normal = np.cross(row_dir, col_dir)
        except (ValueError, IndexError):
            pass

    # Method 1: Check explicit temporal position tags
    num_temporal = getattr(first_ds, "NumberOfTemporalPositions", None)
    if num_temporal is not None and int(num_temporal) > 1:
        num_temporal = int(num_temporal)

        # Group files by TemporalPositionIdentifier (handles any file order)
        temporal_groups: Dict[int, List] = {}
        for f in files:
            ds = read_metadata(f)
            if ds is None:
                continue
            temp_pos = getattr(ds, "TemporalPositionIdentifier", 1)
            temp_pos = int(temp_pos) if temp_pos else 1
            if temp_pos not in temporal_groups:
                temporal_groups[temp_pos] = []
            temporal_groups[temp_pos].append(f)

        if len(temporal_groups) > 1:
            first_timepoint = min(temporal_groups.keys())
            return temporal_groups[first_timepoint], len(temporal_groups)

    # Method 2: Detect by finding duplicate slice locations
    # Read ALL files for robust detection (not sampling)
    slice_positions = []
    file_data = []  # (file, position_tuple, full_position)
    failed_reads = 0

    for f in files:
        ds = read_metadata(f)
        if ds is None:
            failed_reads += 1
            continue
        if hasattr(ds, "ImagePositionPatient"):
            pos = tuple(round(float(x), 2) for x in ds.ImagePositionPatient)
            instance_num = int(getattr(ds, "InstanceNumber", 0) or 0)
            slice_positions.append(pos)
            file_data.append(
                (
                    f,
                    pos,
                    np.array([float(x) for x in ds.ImagePositionPatient]),
                    instance_num,
                )
            )

    # If most files failed to read, log warning
    if failed_reads > len(files) * 0.5:
        logger.warning(
            f"4D detection: {failed_reads}/{len(files)} file reads failed. "
            "Check XNAT connection or local cache."
        )

    if slice_positions:
        unique_positions = set(slice_positions)
        # If many files but few unique positions, likely 4D
        if len(unique_positions) < len(slice_positions) * 0.5:
            from collections import Counter

            pos_counts = Counter(slice_positions)
            most_common_count = pos_counts.most_common(1)[0][1]

            if most_common_count > 1:
                # Sort by InstanceNumber so we consistently pick
                # the first temporal volume for each position
                file_data.sort(key=lambda x: x[3])

                # Take first occurrence of each position
                seen_positions = set()
                file_positions = []

                for f, pos, full_pos, _inst in file_data:
                    if pos not in seen_positions:
                        seen_positions.add(pos)
                        file_positions.append((f, full_pos))

                if file_positions:
                    # Sort by position along slice normal for oblique-safe ordering
                    file_positions.sort(key=lambda x: float(np.dot(x[1], normal)))
                    first_timepoint_files = [f for f, _ in file_positions]

                    estimated_timepoints = len(files) // len(first_timepoint_files)
                    if estimated_timepoints > 1:
                        return first_timepoint_files, estimated_timepoints

    return files, None


class DicomLoader:
    """Load DICOM files with comprehensive error handling."""

    # Required tags for geometry QC
    REQUIRED_TAGS = [
        "PixelData",
        "Rows",
        "Columns",
        "PixelSpacing",
        "ImageOrientationPatient",
        "ImagePositionPatient",
    ]

    # Optional but recommended tags
    OPTIONAL_TAGS = [
        "SliceLocation",
        "SliceThickness",
        "InstanceNumber",
        "PatientPosition",
        "Modality",
        "SeriesDescription",
        "StudyInstanceUID",
        "SeriesInstanceUID",
    ]

    def __init__(self, strict: bool = False):
        """
        Initialize loader.

        Args:
            strict: If True, raise on any error; if False, skip problematic files

        Note: DicomLoader is stateless - errors/warnings are tracked per-load-call
        and returned via DicomVolume, not stored on the loader instance.
        """
        self.strict = strict

    def load_from_xnat(self, files: List[Any]) -> DicomVolume:
        """
        Load DICOM files from XNAT file objects.

        Uses SimpleITK ImageSeriesReader when data_path is available (mounted data).

        Args:
            files: List of xnatpy FileData objects

        Returns:
            DicomVolume with loaded data

        Raises:
            DicomLoadError: If loading fails
        """
        # Use SimpleITK via mounted path if available
        if HAS_SIMPLEITK and files:
            data_path = getattr(files[0], "data_path", None)
            if data_path and Path(data_path).exists():
                return self.load_from_path_simpleitk(Path(data_path).parent)

        # Fall back to manual loading via pydicom
        try:
            import pydicom
        except ImportError:
            raise DicomLoadError(
                "pydicom not installed. Install with: pip install pydicom"
            )

        # Thread-safe: use local variables for errors/warnings
        errors: List[Tuple[str, Exception]] = []
        warnings: List[Tuple[str, str]] = []

        # Detect 4D data and filter to first timepoint
        files, num_timepoints = self._filter_4d_xnat_files(files)

        datasets = []
        orientations = set()
        missing_geometry_tags = []

        for file_obj in files:
            try:
                with file_obj.open() as f:
                    ds = pydicom.dcmread(f)
                    # Track unique orientations for localizer detection
                    if hasattr(ds, "ImageOrientationPatient"):
                        orientation = tuple(
                            round(float(x), 2) for x in ds.ImageOrientationPatient
                        )
                        orientations.add(orientation)
                    # Check missing geometry tags on first file
                    if not datasets:
                        missing_geometry_tags = _check_missing_geometry_tags(ds)
                    ds = self._validate_and_prepare(ds, file_obj.uri, warnings)
                    if ds is not None:
                        datasets.append(ds)
            except Exception as e:
                self._handle_error(file_obj.uri, e, errors)

        volume = self._construct_volume(datasets, len(files), errors, warnings)
        volume.num_timepoints = num_timepoints
        volume.num_orientations = len(orientations) if orientations else 1
        volume.missing_geometry_tags = missing_geometry_tags
        return volume

    def load_from_path_simpleitk(
        self, path: Union[str, Path], series_uid: Optional[str] = None
    ) -> DicomVolume:
        """
        Load DICOM files from a directory using SimpleITK's ImageSeriesReader.

        This is the preferred method as it correctly handles DICOM orientation
        and slice ordering automatically.

        Args:
            path: Directory containing DICOM files
            series_uid: Optional SeriesInstanceUID to filter by

        Returns:
            DicomVolume with loaded data and proper orientation
        """
        if not HAS_SIMPLEITK:
            raise DicomLoadError(
                "SimpleITK not installed. Install with: pip install SimpleITK"
            )

        path = Path(path)
        if not path.is_dir():
            raise DicomLoadError(f"Path is not a directory: {path}")

        # Thread-safe: use local variables for errors/warnings
        errors: List[Tuple[str, Exception]] = []
        warnings: List[Tuple[str, str]] = []

        # Get all series IDs in the directory
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(path))

        if not series_ids:
            raise DicomLoadError(f"No DICOM series found in: {path}")

        # Select series
        if series_uid is not None:
            if series_uid not in series_ids:
                raise DicomLoadError(f"SeriesInstanceUID not found: {series_uid}")
            selected_series = series_uid
        else:
            if len(series_ids) > 1:
                # Get file counts for each series
                series_file_counts = {
                    sid: len(
                        sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(path), sid)
                    )
                    for sid in series_ids
                }
                selected_series = max(series_file_counts, key=series_file_counts.get)
                logger.warning(
                    f"Multiple series found ({len(series_ids)}), using largest "
                    f"({series_file_counts[selected_series]} files). "
                    "Specify series_uid to select a specific series."
                )
            else:
                selected_series = series_ids[0]

        # Get sorted file names for the series
        # CRITICAL: GetGDCMSeriesFileNames sorts by acquisition order, not filename
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
            str(path), selected_series
        )

        # Detect and handle 4D data (fMRI, DTI, DSC perfusion)
        # SimpleITK stacks all files as slices - we need to detect temporal dimension
        num_timepoints = None
        series_file_names, num_timepoints = self._filter_4d_files(series_file_names)

        # Check for multi-orientation data (localizers) BEFORE SimpleITK combines them
        num_orientations = self._count_unique_orientations(series_file_names)

        # Create reader with metadata loading enabled
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(series_file_names)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()

        # Load the volume - this handles orientation automatically
        sitk_image = reader.Execute()

        # Get metadata from first file using pydicom for additional tags
        missing_geometry_tags = []
        try:
            import pydicom

            first_dcm = pydicom.dcmread(series_file_names[0])
            modality = getattr(first_dcm, "Modality", "Unknown")
            series_description = getattr(first_dcm, "SeriesDescription", "")
            patient_position = getattr(first_dcm, "PatientPosition", None)
            study_uid = getattr(first_dcm, "StudyInstanceUID", None)

            # Check for missing geometry tags
            missing_geometry_tags = _check_missing_geometry_tags(first_dcm)

            # Get actual SliceThickness from DICOM tag (not SimpleITK's calculated spacing)
            slice_thickness_tag = getattr(first_dcm, "SliceThickness", None)
            if slice_thickness_tag is not None:
                slice_thickness_tag = float(slice_thickness_tag)
        except Exception:
            modality = "Unknown"
            series_description = ""
            patient_position = None
            study_uid = None
            slice_thickness_tag = None
            missing_geometry_tags = list(GEOMETRY_TAGS)

        return DicomVolume(
            sitk_image=sitk_image,
            modality=modality,
            series_description=series_description,
            patient_position=patient_position,
            study_instance_uid=study_uid,
            series_instance_uid=selected_series,
            errors=errors,
            warnings=warnings,
            num_timepoints=num_timepoints,
            num_orientations=num_orientations,
            missing_geometry_tags=missing_geometry_tags,
        )

    def load_from_path(
        self, path: Union[str, Path], series_uid: Optional[str] = None
    ) -> DicomVolume:
        """
        Load DICOM files from a directory path.

        Args:
            path: Directory containing DICOM files
            series_uid: Optional SeriesInstanceUID to filter by. If None and
                       multiple series are found, loads the series with most files.

        Returns:
            DicomVolume with loaded data
        """
        try:
            import pydicom
        except ImportError:
            raise DicomLoadError(
                "pydicom not installed. Install with: pip install pydicom"
            )

        path = Path(path)
        if not path.is_dir():
            raise DicomLoadError(f"Path is not a directory: {path}")

        # Thread-safe: use local variables for errors/warnings
        errors: List[Tuple[str, Exception]] = []
        warnings: List[Tuple[str, str]] = []

        # Find DICOM files
        dcm_files = list(path.glob("*.dcm")) + list(path.glob("*.DCM"))

        # Also try files without extension (common in DICOM)
        for f in path.iterdir():
            if f.is_file() and f.suffix == "":
                try:
                    # Quick check if it's a DICOM file
                    with open(f, "rb") as fp:
                        fp.seek(128)
                        magic = fp.read(4)
                        if magic == b"DICM":
                            dcm_files.append(f)
                except Exception:
                    pass

        if not dcm_files:
            raise DicomLoadError(f"No DICOM files found in: {path}")

        # Load all datasets and group by SeriesInstanceUID
        all_datasets = []
        series_groups: Dict[str, List[Any]] = {}

        for dcm_file in dcm_files:
            try:
                ds = pydicom.dcmread(str(dcm_file))
                ds = self._validate_and_prepare(ds, str(dcm_file), warnings)
                if ds is not None:
                    all_datasets.append(ds)
                    # Group by SeriesInstanceUID
                    uid = getattr(ds, "SeriesInstanceUID", "unknown")
                    if uid not in series_groups:
                        series_groups[uid] = []
                    series_groups[uid].append(ds)
            except Exception as e:
                self._handle_error(str(dcm_file), e, errors)

        # If multiple series found, filter to requested or largest
        if len(series_groups) > 1:
            if series_uid is not None:
                if series_uid in series_groups:
                    datasets = series_groups[series_uid]
                else:
                    raise DicomLoadError(f"SeriesInstanceUID not found: {series_uid}")
            else:
                # Use the series with most files
                largest_uid = max(
                    series_groups.keys(), key=lambda k: len(series_groups[k])
                )
                datasets = series_groups[largest_uid]
                logger.warning(
                    f"Multiple series found ({len(series_groups)}), using largest "
                    f"({len(datasets)} files). Specify series_uid to select a specific series."
                )
        else:
            datasets = all_datasets

        return self._construct_volume(datasets, len(datasets), errors, warnings)

    def _count_unique_orientations(self, file_names: List[str]) -> int:
        """
        Count unique ImageOrientationPatient values in DICOM files.

        Used to detect multi-orientation localizers before SimpleITK combines them.

        Args:
            file_names: List of DICOM file paths

        Returns:
            Number of unique orientations (1 = normal, >1 = multi-orientation localizer)
        """
        if len(file_names) < 2:
            return 1

        try:
            import pydicom
        except ImportError:
            return 1

        orientations = set()
        for f in file_names:
            try:
                ds = pydicom.dcmread(f, stop_before_pixels=True)
                if hasattr(ds, "ImageOrientationPatient"):
                    # Round to 2 decimal places to handle floating point differences
                    orientation = tuple(
                        round(float(x), 2) for x in ds.ImageOrientationPatient
                    )
                    orientations.add(orientation)
            except Exception:
                continue

        return len(orientations) if orientations else 1

    def _filter_4d_xnat_files(
        self, files: List[Any]
    ) -> Tuple[List[Any], Optional[int]]:
        """
        Detect 4D data and filter XNAT files to first temporal position.

        Args:
            files: List of xnatpy FileData objects

        Returns:
            Tuple of (filtered_files, num_timepoints)
        """
        try:
            import pydicom
        except ImportError:
            return files, None

        def read_xnat_metadata(file_obj):
            """Read pydicom dataset from XNAT file object."""
            try:
                with file_obj.open() as f:
                    return pydicom.dcmread(f, stop_before_pixels=True)
            except Exception:
                return None

        return _filter_4d_generic(files, read_xnat_metadata)

    def _filter_4d_files(
        self, file_names: List[str]
    ) -> Tuple[List[str], Optional[int]]:
        """
        Detect 4D data and filter to first temporal position.

        Args:
            file_names: List of DICOM file paths

        Returns:
            Tuple of (filtered_files, num_timepoints)
        """
        try:
            import pydicom
        except ImportError:
            return file_names, None

        def read_file_metadata(path):
            """Read pydicom dataset from file path."""
            try:
                return pydicom.dcmread(path, stop_before_pixels=True)
            except Exception:
                return None

        return _filter_4d_generic(file_names, read_file_metadata)

    def _validate_and_prepare(
        self, ds: Any, filename: str, warnings: List[Tuple[str, str]]
    ) -> Optional[Any]:
        """Validate required tags and prepare dataset.

        Args:
            ds: pydicom Dataset
            filename: Source filename for error reporting
            warnings: List to append warnings to (thread-safe local variable)
        """
        try:
            import pydicom  # noqa: F401
        except ImportError:
            return None

        # Validate required tags
        missing = []
        for tag in self.REQUIRED_TAGS:
            if not hasattr(ds, tag) or getattr(ds, tag) is None:
                missing.append(tag)

        if missing:
            raise MissingTagError(", ".join(missing))

        # Check optional tags and warn
        for tag in self.OPTIONAL_TAGS:
            if not hasattr(ds, tag) or getattr(ds, tag) is None:
                warnings.append((filename, f"Missing optional tag: {tag}"))

        # Skip derived/secondary captures that might not be image slices
        if hasattr(ds, "ImageType"):
            image_type = (
                list(ds.ImageType)
                if hasattr(ds.ImageType, "__iter__")
                else [ds.ImageType]
            )
            if "DERIVED" in image_type and "SECONDARY" in image_type:
                warnings.append((filename, "Skipping derived/secondary image"))
                return None

        return ds

    def _handle_error(
        self, filename: str, error: Exception, errors: List[Tuple[str, Exception]]
    ) -> None:
        """Handle loading error based on strict mode.

        Args:
            filename: Source filename for error reporting
            error: The exception that occurred
            errors: List to append errors to (thread-safe local variable)
        """
        errors.append((filename, error))
        logger.warning(f"Error loading {filename}: {error}")

        if self.strict:
            raise error

    def _construct_volume(
        self,
        datasets: List[Any],
        total_files: int,
        errors: List[Tuple[str, Exception]],
        warnings: List[Tuple[str, str]],
    ) -> DicomVolume:
        """Construct DicomVolume from sorted datasets.

        Args:
            datasets: List of pydicom Datasets
            total_files: Total number of files attempted
            errors: List of loading errors (thread-safe local variable)
            warnings: List of loading warnings (thread-safe local variable)
        """
        if not datasets:
            raise DicomLoadError("No valid DICOM files could be loaded")

        # Sort slices (skip for single-slice volumes)
        if len(datasets) > 1:
            try:
                datasets = self._sort_slices(datasets)
            except Exception as e:
                raise GeometryError(f"Failed to sort slices: {e}")

        # Extract reference dataset for metadata
        ref_ds = datasets[0]

        # Stack pixel arrays
        try:
            pixel_arrays = []
            for ds in datasets:
                arr = ds.pixel_array
                # Handle 3D+ arrays: multi-frame grayscale or single-frame color
                if arr.ndim > 2:
                    samples = getattr(ds, "SamplesPerPixel", 1)
                    if samples > 1:
                        # Color image (e.g. RGB) — take first channel
                        arr = arr[:, :, 0]
                    else:
                        # Multi-frame grayscale — take first frame
                        arr = arr[0]
                pixel_arrays.append(arr.astype(np.float32))

            volume_data = np.stack(pixel_arrays, axis=0)
        except Exception as e:
            raise DicomLoadError(f"Failed to construct volume: {e}")

        # Extract geometry information
        slice_locations = self._extract_slice_locations(datasets)
        image_positions = self._extract_image_positions(datasets)
        image_orientation = np.array(ref_ds.ImageOrientationPatient, dtype=np.float64)
        pixel_spacing = tuple(float(x) for x in ref_ds.PixelSpacing)

        # Get optional metadata
        slice_thickness = getattr(ref_ds, "SliceThickness", None)
        if slice_thickness is not None:
            slice_thickness = float(slice_thickness)

        modality = getattr(ref_ds, "Modality", "Unknown")
        series_description = getattr(ref_ds, "SeriesDescription", "")
        patient_position = getattr(ref_ds, "PatientPosition", None)
        study_uid = getattr(ref_ds, "StudyInstanceUID", None)
        series_uid = getattr(ref_ds, "SeriesInstanceUID", None)

        # Build SimpleITK image (required for DicomVolume)
        if not HAS_SIMPLEITK:
            raise DicomLoadError(
                "SimpleITK is required. Install with: pip install SimpleITK"
            )

        sitk_image = self._build_sitk_image(
            volume_data,
            image_orientation,
            pixel_spacing,
            slice_thickness,
            slice_locations,
            image_positions,
        )

        return DicomVolume(
            sitk_image=sitk_image,
            modality=modality,
            series_description=series_description,
            patient_position=patient_position,
            study_instance_uid=study_uid,
            series_instance_uid=series_uid,
            errors=errors,
            warnings=warnings,
        )

    def _build_sitk_image(
        self,
        pixel_array: np.ndarray,
        image_orientation: np.ndarray,
        pixel_spacing: tuple,
        slice_thickness: Optional[float],
        slice_locations: np.ndarray,
        image_positions: np.ndarray,
    ) -> Any:
        """
        Build SimpleITK image from volume data with correct direction matrix.

        Args:
            pixel_array: 3D numpy array [slices, rows, cols]
            image_orientation: DICOM ImageOrientationPatient (6 elements)
            pixel_spacing: (row_spacing, col_spacing) in mm
            slice_thickness: Slice thickness in mm (optional)
            slice_locations: Array of slice positions along normal
            image_positions: [N, 3] array of ImagePositionPatient values
        """
        arr = pixel_array.astype(np.float32)

        # Calculate slice spacing from actual positions
        if len(slice_locations) > 1:
            z_spacing = float(np.median(np.abs(np.diff(slice_locations))))
        elif slice_thickness is not None and slice_thickness > 0:
            z_spacing = slice_thickness
        else:
            z_spacing = pixel_spacing[0]  # Fallback to row spacing

        # Build direction vectors from DICOM orientation
        row_dir = image_orientation[0:3]  # Direction of increasing column
        col_dir = image_orientation[3:6]  # Direction of increasing row
        slice_dir = np.cross(row_dir, col_dir)  # Direction of increasing slice

        # Create SimpleITK image
        # GetImageFromArray maps input [z, y, x] to image axes [x, y, z]
        # So: axis 0 (x) = columns, axis 1 (y) = rows, axis 2 (z) = slices
        image = sitk.GetImageFromArray(arr)
        image.SetSpacing(
            [pixel_spacing[1], pixel_spacing[0], z_spacing]
        )  # [col, row, slice]

        # Direction matrix: stored row-major, columns are axis directions
        # Column 0 = x-axis direction (columns) = row_dir
        # Column 1 = y-axis direction (rows) = col_dir
        # Column 2 = z-axis direction (slices) = slice_dir
        direction = [
            row_dir[0],
            col_dir[0],
            slice_dir[0],
            row_dir[1],
            col_dir[1],
            slice_dir[1],
            row_dir[2],
            col_dir[2],
            slice_dir[2],
        ]
        image.SetDirection(direction)
        image.SetOrigin(image_positions[0].tolist())

        return image

    def _sort_slices(self, datasets: List[Any]) -> List[Any]:
        """Sort slices by spatial location."""
        # Try SliceLocation first
        if all(
            hasattr(ds, "SliceLocation") and ds.SliceLocation is not None
            for ds in datasets
        ):
            return sorted(datasets, key=lambda ds: float(ds.SliceLocation))

        # Fall back to ImagePositionPatient
        if all(hasattr(ds, "ImagePositionPatient") for ds in datasets):
            # Use the component along the slice normal
            orientation = datasets[0].ImageOrientationPatient
            row = np.array(orientation[0:3])
            col = np.array(orientation[3:6])
            normal = np.cross(row, col)

            def get_position_along_normal(ds):
                pos = np.array(ds.ImagePositionPatient)
                return np.dot(pos, normal)

            return sorted(datasets, key=get_position_along_normal)

        # Last resort: InstanceNumber
        if all(
            hasattr(ds, "InstanceNumber") and ds.InstanceNumber is not None
            for ds in datasets
        ):
            return sorted(datasets, key=lambda ds: int(ds.InstanceNumber))

        raise GeometryError("Cannot determine slice order: missing location tags")

    def _extract_slice_locations(self, datasets: List[Any]) -> np.ndarray:
        """Extract slice locations from datasets."""
        locations = []

        for ds in datasets:
            if hasattr(ds, "SliceLocation") and ds.SliceLocation is not None:
                locations.append(float(ds.SliceLocation))
            elif hasattr(ds, "ImagePositionPatient"):
                # Calculate position along slice normal
                orientation = ds.ImageOrientationPatient
                row = np.array(orientation[0:3])
                col = np.array(orientation[3:6])
                normal = np.cross(row, col)
                pos = np.array(ds.ImagePositionPatient)
                locations.append(float(np.dot(pos, normal)))
            else:
                # Use index as fallback
                locations.append(float(len(locations)))

        return np.array(locations)

    def _extract_image_positions(self, datasets: List[Any]) -> np.ndarray:
        """Extract image positions from datasets."""
        positions = []

        for ds in datasets:
            if hasattr(ds, "ImagePositionPatient"):
                pos = [float(x) for x in ds.ImagePositionPatient]
                positions.append(pos)
            else:
                # Use zeros as fallback
                positions.append([0.0, 0.0, 0.0])

        return np.array(positions)
