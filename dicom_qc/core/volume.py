"""Data structures for DICOM volumes and scan information."""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

try:
    import SimpleITK as sitk
    HAS_SIMPLEITK = True
except ImportError:
    HAS_SIMPLEITK = False


@dataclass
class ScanInfo:
    """Information about an XNAT scan."""

    id: str
    description: str
    modality: str
    num_files: int
    series_description: Optional[str] = None
    project: Optional[str] = None
    subject: Optional[str] = None
    experiment: Optional[str] = None

    # XNAT scan object reference (for loading)
    _scan_obj: Any = field(default=None, repr=False)


class DicomVolume:
    """
    Represents a loaded DICOM volume with metadata.

    Primary data is stored in sitk_image (SimpleITK Image). The pixel_array
    and geometry properties are derived from it on demand to save memory.

    Attributes:
        sitk_image: SimpleITK image with proper orientation and spacing
        modality: DICOM modality (CT, MR, PT, etc.)
        series_description: Series description from DICOM
        patient_position: Patient position (HFS, FFS, etc.)
        study_instance_uid: DICOM StudyInstanceUID
        series_instance_uid: DICOM SeriesInstanceUID
    """

    def __init__(
        self,
        sitk_image: Any,
        modality: str = 'Unknown',
        series_description: str = '',
        patient_position: Optional[str] = None,
        study_instance_uid: Optional[str] = None,
        series_instance_uid: Optional[str] = None,
        errors: Optional[List[Tuple[str, Exception]]] = None,
        warnings: Optional[List[Tuple[str, str]]] = None,
        num_timepoints: Optional[int] = None,
        num_orientations: int = 1,
        missing_geometry_tags: Optional[List[str]] = None,
    ):
        """
        Initialize DicomVolume from a SimpleITK image.

        Args:
            sitk_image: SimpleITK Image with proper orientation
            modality: DICOM modality
            series_description: Series description
            patient_position: Patient position (HFS, FFS, etc.)
            study_instance_uid: Study UID
            series_instance_uid: Series UID
            errors: List of loading errors
            warnings: List of loading warnings
            num_timepoints: Number of timepoints for 4D data (None for 3D)
            num_orientations: Number of unique orientations (>1 = multi-orientation localizer)
            missing_geometry_tags: List of missing DICOM geometry tags (PixelSpacing, etc.)
        """
        if not HAS_SIMPLEITK:
            raise ImportError("SimpleITK is required. Install with: pip install SimpleITK")

        self.sitk_image = sitk_image
        self.modality = modality
        self.series_description = series_description
        self.patient_position = patient_position
        self.study_instance_uid = study_instance_uid
        self.series_instance_uid = series_instance_uid
        self.errors = errors or []
        self.warnings = warnings or []
        self.num_timepoints = num_timepoints  # For 4D data (fMRI, DTI, etc.)
        self.num_orientations = num_orientations  # For multi-orientation localizers
        self.missing_geometry_tags = missing_geometry_tags or []  # Missing DICOM tags

        # Cache for derived properties
        self._pixel_array: Optional[np.ndarray] = None
        self._lps_cache: Optional[Tuple[np.ndarray, Tuple[float, float, float]]] = None

    @property
    def pixel_array(self) -> np.ndarray:
        """3D numpy array [slices, rows, cols] - derived from sitk_image."""
        if self._pixel_array is None:
            self._pixel_array = sitk.GetArrayFromImage(self.sitk_image).astype(np.float32)
        return self._pixel_array

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return volume shape (slices, rows, cols)."""
        # Get from sitk_image directly to avoid loading full array
        size = self.sitk_image.GetSize()  # (x, y, z) = (cols, rows, slices)
        return (size[2], size[1], size[0])

    @property
    def num_slices(self) -> int:
        """Return number of slices."""
        return self.shape[0]

    @property
    def is_4d(self) -> bool:
        """Return True if this was originally 4D data (fMRI, DTI, etc.)."""
        return self.num_timepoints is not None and self.num_timepoints > 1

    @property
    def pixel_spacing(self) -> Tuple[float, float]:
        """Return (row_spacing, col_spacing) in mm."""
        spacing = self.sitk_image.GetSpacing()  # (x, y, z) = (col, row, slice)
        return (spacing[1], spacing[0])

    @property
    def row_spacing(self) -> float:
        """Return row spacing in mm."""
        return self.pixel_spacing[0]

    @property
    def col_spacing(self) -> float:
        """Return column spacing in mm."""
        return self.pixel_spacing[1]

    @property
    def slice_thickness(self) -> float:
        """Return slice thickness in mm."""
        return self.sitk_image.GetSpacing()[2]

    @property
    def voxel_spacing(self) -> Tuple[float, float, float]:
        """Return voxel spacing (slice, row, col) in mm."""
        spacing = self.sitk_image.GetSpacing()  # (x, y, z) = (col, row, slice)
        return (spacing[2], spacing[1], spacing[0])

    @property
    def image_orientation(self) -> np.ndarray:
        """Return 6-element array of direction cosines [row_dir, col_dir]."""
        direction = self.sitk_image.GetDirection()
        # Direction matrix columns are axis directions
        # Column 0 = row_dir (x-axis), Column 1 = col_dir (y-axis)
        row_dir = [direction[0], direction[3], direction[6]]
        col_dir = [direction[1], direction[4], direction[7]]
        return np.array(row_dir + col_dir, dtype=np.float64)

    @property
    def image_positions(self) -> np.ndarray:
        """Return [N, 3] array of slice positions in patient coordinates."""
        origin = np.array(self.sitk_image.GetOrigin())
        spacing = self.sitk_image.GetSpacing()
        direction = self.sitk_image.GetDirection()
        # Slice direction is column 2 of direction matrix
        slice_dir = np.array([direction[2], direction[5], direction[8]])

        num_slices = self.shape[0]
        positions = np.zeros((num_slices, 3))
        for i in range(num_slices):
            positions[i] = origin + i * spacing[2] * slice_dir
        return positions

    @property
    def slice_locations(self) -> np.ndarray:
        """Return array of slice positions along slice normal."""
        positions = self.image_positions
        slice_normal = self.slice_normal
        return np.array([np.dot(pos, slice_normal) for pos in positions])

    @property
    def row_direction(self) -> np.ndarray:
        """Return row direction cosines (direction of increasing column index)."""
        return self.image_orientation[0:3]

    @property
    def col_direction(self) -> np.ndarray:
        """Return column direction cosines (direction of increasing row index)."""
        return self.image_orientation[3:6]

    @property
    def slice_normal(self) -> np.ndarray:
        """Return slice normal (perpendicular to image plane)."""
        return np.cross(self.row_direction, self.col_direction)

    def get_slice(self, index: int) -> np.ndarray:
        """Get a single axial slice."""
        return self.pixel_array[index]

    def get_coronal_slice(self, index: int) -> np.ndarray:
        """Get a single coronal slice."""
        return self.pixel_array[:, index, :]

    def get_sagittal_slice(self, index: int) -> np.ndarray:
        """Get a single sagittal slice."""
        return self.pixel_array[:, :, index]

    def get_sitk_image(self) -> Any:
        """Get the SimpleITK image for this volume."""
        return self.sitk_image

    def get_reoriented_slices(self) -> Tuple[np.ndarray, Tuple[float, float, float], Any]:
        """
        Get properly oriented slices for axial, coronal, and sagittal views.

        Uses SimpleITK's DICOMOrientImageFilter to reorient to standard LPS.

        Returns:
            Tuple of (lps_array, lps_spacing, lps_image)
        """
        # Reorient to LPS (standard DICOM/radiological orientation)
        orient_filter = sitk.DICOMOrientImageFilter()
        orient_filter.SetDesiredCoordinateOrientation('LPS')
        lps_image = orient_filter.Execute(self.sitk_image)

        # Get array and spacing
        lps_array = sitk.GetArrayFromImage(lps_image)
        lps_spacing = lps_image.GetSpacing()  # (L, P, S)

        return lps_array, lps_spacing, lps_image

    def get_lps_array(self) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        Get the volume reoriented to standard LPS orientation.

        Returns:
            Tuple of (array, spacing) where:
            - array is indexed as [S, P, L] (Superior, Posterior, Left)
            - spacing is (S, P, L) in mm
        """
        if self._lps_cache is None:
            lps_array, lps_spacing, _ = self.get_reoriented_slices()
            # lps_spacing from SimpleITK is (L, P, S), convert to (S, P, L)
            self._lps_cache = (lps_array, (lps_spacing[2], lps_spacing[1], lps_spacing[0]))
        return self._lps_cache

    def get_intensity_range(self) -> Tuple[float, float]:
        """Return (min, max) intensity values."""
        arr = self.pixel_array
        return float(arr.min()), float(arr.max())

    def get_auto_window(self) -> Tuple[float, float]:
        """Calculate automatic window (center, width) based on modality and image statistics."""
        # For CT, use standard soft tissue window as default
        if self.modality == 'CT':
            return (40.0, 400.0)

        # For MR and other modalities, use percentile-based windowing
        data = self.pixel_array
        min_val = float(data.min())
        foreground = data[data > min_val]

        if len(foreground) > 100:
            threshold = np.percentile(foreground, 10)
            tissue = data[data > threshold]

            if len(tissue) > 100:
                p1, p99 = np.percentile(tissue, [1, 99])
                center = float((p1 + p99) / 2)
                width = max(float(p99 - p1), 1.0)
                return (center, width)

        # Fallback: simple percentile on all data
        p2, p98 = np.percentile(data, [2, 98])
        width = max(float(p98 - p2), 1.0)
        center = float((p2 + p98) / 2)
        return (center, width)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'shape': self.shape,
            'num_slices': self.num_slices,
            'pixel_spacing': self.pixel_spacing,
            'slice_thickness': self.slice_thickness,
            'voxel_spacing': self.voxel_spacing,
            'modality': self.modality,
            'series_description': self.series_description,
            'patient_position': self.patient_position,
            'study_instance_uid': self.study_instance_uid,
            'series_instance_uid': self.series_instance_uid,
            'image_orientation': self.image_orientation.tolist(),
            'intensity_range': self.get_intensity_range(),
        }
        if self.num_timepoints is not None:
            result['num_timepoints'] = self.num_timepoints
            result['is_4d'] = True
        return result
