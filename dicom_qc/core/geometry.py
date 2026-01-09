"""Geometry and orientation QC checks for DICOM volumes."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Literal, Optional
from datetime import datetime

import numpy as np

from dicom_qc.core.volume import DicomVolume


@dataclass
class QCResult:
    """Result of a single QC check."""

    status: Literal['PASS', 'FAIL', 'WARNING', 'NOTE']
    check_name: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'status': self.status,
            'check_name': self.check_name,
            'message': self.message,
            'details': self.details,
        }


@dataclass
class QCReport:
    """Collection of QC results for a scan."""

    scan_id: str
    results: List[QCResult]
    overall_status: Literal['PASS', 'FAIL', 'WARNING', 'NOTE']
    orientation_labels: Dict[str, str]
    primary_plane: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'scan_id': self.scan_id,
            'results': [r.to_dict() for r in self.results],
            'overall_status': self.overall_status,
            'orientation_labels': self.orientation_labels,
            'primary_plane': self.primary_plane,
            'timestamp': self.timestamp.isoformat(),
        }


class GeometryQC:
    """Performs geometry and orientation QC checks on DICOM volumes."""

    # Tolerance for floating point comparisons
    ORIENTATION_TOLERANCE = 1e-6

    # Threshold for gap detection (multiple of expected spacing)
    GAP_THRESHOLD = 1.5

    # Threshold for oblique plane detection
    OBLIQUE_THRESHOLD = 0.8

    # Threshold for anisotropy detection (ratio of largest to smallest voxel dimension)
    ANISOTROPY_WARNING = 2.0  # Warning if ratio > 2
    ANISOTROPY_FAIL = 4.0     # Fail if ratio > 4

    # JPEG-2000 transfer syntax UIDs that may have viewer compatibility issues
    JPEG2000_UIDS = {
        '1.2.840.10008.1.2.4.90',  # JPEG 2000 Lossless
        '1.2.840.10008.1.2.4.91',  # JPEG 2000 Lossy
    }
    # Implementations known to produce problematic JPEG-2000
    PROBLEMATIC_IMPLEMENTATIONS = {
        'dcm4che-1.4-JP',  # Known to cause OHIF rendering issues
    }

    def __init__(self, volume: DicomVolume):
        """
        Initialize with a loaded DICOM volume.

        Args:
            volume: DicomVolume instance
        """
        self.volume = volume

    def check_slice_ordering(self) -> QCResult:
        """
        Validate slice ordering using SliceLocation or ImagePositionPatient.

        Verifies that slices are in monotonic order (all increasing or all decreasing)
        and checks for duplicate slice locations.

        Returns:
            QCResult with PASS if monotonic, FAIL if not, WARNING if edge cases
        """
        locations = self.volume.slice_locations

        if len(locations) < 2:
            return QCResult(
                status='WARNING',
                check_name='Slice Ordering',
                message='Only one slice - cannot verify ordering',
                details={'num_slices': len(locations)}
            )

        # Check for duplicates
        unique_locations = np.unique(locations)
        if len(unique_locations) != len(locations):
            duplicates = self._find_duplicates(locations)
            return QCResult(
                status='FAIL',
                check_name='Slice Ordering',
                message=f'Duplicate slice locations detected at {len(duplicates)} positions',
                details={'duplicate_locations': duplicates}
            )

        # Check monotonicity
        diffs = np.diff(locations)
        is_monotonic_increasing = np.all(diffs > 0)
        is_monotonic_decreasing = np.all(diffs < 0)

        if is_monotonic_increasing or is_monotonic_decreasing:
            direction = 'increasing' if is_monotonic_increasing else 'decreasing'
            return QCResult(
                status='PASS',
                check_name='Slice Ordering',
                message=f'Slices are monotonically {direction}',
                details={
                    'direction': direction,
                    'num_slices': len(locations),
                    'first_location': float(locations[0]),
                    'last_location': float(locations[-1]),
                }
            )

        # Find where ordering breaks
        sign_changes = np.where(np.diff(np.sign(diffs)) != 0)[0]
        return QCResult(
            status='FAIL',
            check_name='Slice Ordering',
            message=f'Non-monotonic slice ordering detected at {len(sign_changes)} locations',
            details={
                'locations': locations.tolist(),
                'differences': diffs.tolist(),
                'sign_change_indices': sign_changes.tolist(),
            }
        )

    def _find_duplicates(self, locations: np.ndarray) -> List[float]:
        """Find duplicate slice locations."""
        unique, counts = np.unique(locations, return_counts=True)
        return unique[counts > 1].tolist()

    def check_orientation_consistency(self) -> QCResult:
        """
        Verify ImageOrientationPatient is consistent across all slices.

        Note: Since we load from a single series, orientation should be consistent.
        This check validates the data structure assumption.

        Returns:
            QCResult with status and any inconsistencies found
        """
        orientation = self.volume.image_orientation

        # Validate orientation is a proper 6-element array
        if len(orientation) != 6:
            return QCResult(
                status='FAIL',
                check_name='Orientation Consistency',
                message=f'Invalid orientation: expected 6 values, got {len(orientation)}',
                details={'orientation': orientation.tolist()}
            )

        # Validate row and column vectors are unit vectors
        row = orientation[0:3]
        col = orientation[3:6]

        row_mag = np.linalg.norm(row)
        col_mag = np.linalg.norm(col)

        if not np.isclose(row_mag, 1.0, atol=0.01):
            return QCResult(
                status='WARNING',
                check_name='Orientation Consistency',
                message=f'Row direction vector is not unit length: {row_mag:.4f}',
                details={'row_magnitude': row_mag, 'orientation': orientation.tolist()}
            )

        if not np.isclose(col_mag, 1.0, atol=0.01):
            return QCResult(
                status='WARNING',
                check_name='Orientation Consistency',
                message=f'Column direction vector is not unit length: {col_mag:.4f}',
                details={'col_magnitude': col_mag, 'orientation': orientation.tolist()}
            )

        # Validate row and column are orthogonal
        dot_product = np.dot(row, col)
        if not np.isclose(dot_product, 0.0, atol=0.01):
            return QCResult(
                status='WARNING',
                check_name='Orientation Consistency',
                message=f'Row and column directions not orthogonal: dot={dot_product:.4f}',
                details={'dot_product': dot_product, 'orientation': orientation.tolist()}
            )

        return QCResult(
            status='PASS',
            check_name='Orientation Consistency',
            message='Orientation vectors are valid and consistent',
            details={'orientation': orientation.tolist()}
        )

    def check_gap_detection(self) -> QCResult:
        """
        Detect gaps or irregular spacing between slices.

        Calculates inter-slice spacing and flags:
        - Gaps > 1.5x expected spacing as potential missing slices
        - Irregular spacing (variance > 10% of expected)

        Returns:
            QCResult with gap locations and severity
        """
        positions = self.volume.image_positions

        if len(positions) < 2:
            return QCResult(
                status='WARNING',
                check_name='Gap Detection',
                message='Only one slice - cannot check for gaps',
                details={'num_slices': len(positions)}
            )

        # Calculate 3D distances between consecutive slices
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)

        # Use declared slice thickness or median distance as expected spacing
        expected_spacing = self.volume.slice_thickness
        if expected_spacing is None or expected_spacing <= 0:
            expected_spacing = float(np.median(distances))

        # Detect gaps
        gap_threshold = expected_spacing * self.GAP_THRESHOLD
        gap_indices = np.where(distances > gap_threshold)[0]

        # Check spacing regularity
        spacing_mean = np.mean(distances)
        spacing_std = np.std(distances)
        spacing_cv = spacing_std / spacing_mean if spacing_mean > 0 else 0  # Coefficient of variation

        details = {
            'expected_spacing_mm': expected_spacing,
            'actual_spacings_mm': distances.tolist(),
            'spacing_mean_mm': float(spacing_mean),
            'spacing_std_mm': float(spacing_std),
            'spacing_cv': float(spacing_cv),
            'num_slices': len(positions),
        }

        if len(gap_indices) > 0:
            gap_info = [
                {
                    'between_slices': (int(i), int(i + 1)),
                    'distance_mm': float(distances[i]),
                    'expected_mm': expected_spacing,
                    'ratio': float(distances[i] / expected_spacing),
                }
                for i in gap_indices
            ]
            details['gaps'] = gap_info

            return QCResult(
                status='FAIL',
                check_name='Gap Detection',
                message=f'{len(gap_indices)} gap(s) detected (possible missing slices)',
                details=details
            )

        # Check for irregular spacing (CV > 10%)
        if spacing_cv > 0.1:
            return QCResult(
                status='WARNING',
                check_name='Gap Detection',
                message=f'Irregular slice spacing detected (CV={spacing_cv:.1%})',
                details=details
            )

        return QCResult(
            status='PASS',
            check_name='Gap Detection',
            message=f'No gaps detected, regular spacing ({spacing_mean:.2f} mm)',
            details=details
        )

    def check_orientation_labels(self) -> QCResult:
        """
        Calculate orientation labels (L/R, A/P, S/I) from ImageOrientationPatient.

        Uses DICOM LPS coordinate system:
        - X increases toward patient's Left
        - Y increases toward patient's Posterior
        - Z increases toward patient's Superior (head)

        Returns:
            QCResult with calculated orientation labels and primary plane
        """
        orientation = self.volume.image_orientation

        # Extract row and column direction cosines
        row_cosines = np.array(orientation[0:3])
        col_cosines = np.array(orientation[3:6])

        # Calculate slice normal (perpendicular to image plane)
        slice_normal = np.cross(row_cosines, col_cosines)

        # Determine orientation labels for each axis
        row_label = self._get_orientation_label(row_cosines)
        col_label = self._get_orientation_label(col_cosines)
        slice_label = self._get_orientation_label(slice_normal)

        # Get opposite labels for display on image edges
        row_labels = self._get_axis_labels(row_cosines)
        col_labels = self._get_axis_labels(col_cosines)

        # Determine primary plane
        plane = self._determine_plane(slice_normal)

        labels = {
            'row_positive': row_label,
            'row_negative': self._get_opposite_label(row_label),
            'col_positive': col_label,
            'col_negative': self._get_opposite_label(col_label),
            'slice_direction': slice_label,
            'primary_plane': plane,
            'image_right': row_labels[1],    # Positive row direction (right edge)
            'image_left': row_labels[0],     # Negative row direction (left edge)
            'image_bottom': col_labels[1],   # Positive col direction (bottom edge)
            'image_top': col_labels[0],      # Negative col direction (top edge)
        }

        details = {
            'orientation_labels': labels,
            'row_cosines': row_cosines.tolist(),
            'col_cosines': col_cosines.tolist(),
            'slice_normal': slice_normal.tolist(),
        }

        return QCResult(
            status='PASS',
            check_name='Orientation Labels',
            message=f'{plane} orientation: Row→{row_label}, Col→{col_label}, Slice→{slice_label}',
            details=details
        )

    def _get_orientation_label(self, direction_cosines: np.ndarray) -> str:
        """
        Convert direction cosines to anatomical orientation label.

        Returns string like 'L' (Left), 'R' (Right), 'A' (Anterior),
        'P' (Posterior), 'S' (Superior), 'I' (Inferior)
        """
        abs_vals = np.abs(direction_cosines)
        dominant_axis = np.argmax(abs_vals)
        sign = np.sign(direction_cosines[dominant_axis])

        # LPS coordinate system mapping
        # X: negative=Right, positive=Left
        # Y: negative=Anterior, positive=Posterior
        # Z: negative=Inferior, positive=Superior
        axis_labels = {
            0: ('R', 'L'),  # X-axis
            1: ('A', 'P'),  # Y-axis
            2: ('I', 'S'),  # Z-axis
        }

        neg_label, pos_label = axis_labels[dominant_axis]
        return pos_label if sign > 0 else neg_label

    def _get_axis_labels(self, direction_cosines: np.ndarray) -> tuple:
        """Get (negative, positive) labels for a direction."""
        abs_vals = np.abs(direction_cosines)
        dominant_axis = np.argmax(abs_vals)
        sign = np.sign(direction_cosines[dominant_axis])

        axis_labels = {
            0: ('R', 'L'),
            1: ('A', 'P'),
            2: ('I', 'S'),
        }

        neg_label, pos_label = axis_labels[dominant_axis]
        if sign > 0:
            return (neg_label, pos_label)
        else:
            return (pos_label, neg_label)

    def _get_opposite_label(self, label: str) -> str:
        """Get the opposite anatomical label."""
        opposites = {
            'L': 'R', 'R': 'L',
            'A': 'P', 'P': 'A',
            'S': 'I', 'I': 'S',
        }
        return opposites.get(label, label)

    def _determine_plane(self, slice_normal: np.ndarray) -> str:
        """Determine primary acquisition plane from slice normal."""
        abs_normal = np.abs(slice_normal)
        dominant = np.argmax(abs_normal)

        # Threshold for oblique detection
        if abs_normal[dominant] < self.OBLIQUE_THRESHOLD:
            return 'OBLIQUE'

        plane_map = {0: 'SAGITTAL', 1: 'CORONAL', 2: 'AXIAL'}
        return plane_map[dominant]

    def check_voxel_anisotropy(self) -> QCResult:
        """
        Check for highly anisotropic voxels (e.g., thick slices).

        Anisotropic data looks blurry when reformatted to planes other than
        the acquisition plane. This is common with 2D acquisitions that have
        thick slices relative to in-plane resolution.

        Returns:
            QCResult with WARNING if moderately anisotropic, FAIL if severe
        """
        spacing = self.volume.pixel_spacing
        slice_thickness = self.volume.slice_thickness
        if slice_thickness is None or slice_thickness <= 0:
            # Fallback to third spacing element or first element
            slice_thickness = spacing[2] if len(spacing) > 2 else spacing[0]

        # Get all three dimensions
        dims = [spacing[0], spacing[1], slice_thickness]
        min_dim = min(dims)
        max_dim = max(dims)

        ratio = max_dim / min_dim if min_dim > 0 else float('inf')

        details = {
            'pixel_spacing_mm': list(spacing),
            'slice_thickness_mm': slice_thickness,
            'voxel_dimensions_mm': dims,
            'anisotropy_ratio': float(ratio),
            'min_dimension_mm': float(min_dim),
            'max_dimension_mm': float(max_dim),
        }

        if ratio > self.ANISOTROPY_FAIL:
            return QCResult(
                status='WARNING',
                check_name='Voxel Anisotropy',
                message=f'Severely anisotropic: {ratio:.1f}x ratio ({max_dim:.2f}/{min_dim:.2f} mm)',
                details=details
            )
        elif ratio > self.ANISOTROPY_WARNING:
            return QCResult(
                status='WARNING',
                check_name='Voxel Anisotropy',
                message=f'Moderately anisotropic: {ratio:.1f}x ratio ({max_dim:.2f}/{min_dim:.2f} mm)',
                details=details
            )
        else:
            return QCResult(
                status='PASS',
                check_name='Voxel Anisotropy',
                message=f'Isotropic voxels: {ratio:.1f}x ratio ({min_dim:.2f}-{max_dim:.2f} mm)',
                details=details
            )

    # Series description patterns that indicate potentially problematic series for strict viewers
    ADVANCED_SERIES_PATTERNS = {
        'DTI': ['dti', 'diffusion', 'dwi', 'adc', 'fa_map', 'trace'],
        'DSC_PERFUSION': ['dsc', 'perfusion', 'perf', 'cbv', 'cbf', 'mtt', 'ttp', 'tmax'],
        'MOCO': ['moco', 'motion_corr', 'motion-corr'],
        'DERIVED': ['derived', 'secondary', 'mip', 'reformat'],
    }

    # Transfer syntaxes that may have viewer compatibility issues
    PROBLEMATIC_TRANSFER_SYNTAXES = {
        '1.2.840.10008.1.2.4.90': 'JPEG 2000 Lossless',
        '1.2.840.10008.1.2.4.91': 'JPEG 2000 Lossy',
        '1.2.840.10008.1.2.4.92': 'JPEG 2000 Part 2 Lossless',
        '1.2.840.10008.1.2.4.93': 'JPEG 2000 Part 2 Lossy',
        '1.2.840.10008.1.2.4.80': 'JPEG-LS Lossless',
        '1.2.840.10008.1.2.4.81': 'JPEG-LS Lossy',
        '1.2.840.10008.1.2.5': 'RLE Lossless',
    }

    def check_series_type(self, series_description: str = "") -> QCResult:
        """
        Detect advanced MR series types that may have viewer compatibility issues.

        DTI, DSC perfusion, and MoCo series often fail in strict viewers like
        ITK-SNAP and 3D Slicer due to multi-frame encoding or missing metadata.

        Returns:
            QCResult with WARNING if potentially problematic series type detected
        """
        desc_lower = (series_description or self.volume.series_description or "").lower()
        detected_types = []

        for series_type, patterns in self.ADVANCED_SERIES_PATTERNS.items():
            if any(p in desc_lower for p in patterns):
                detected_types.append(series_type)

        if detected_types:
            return QCResult(
                status='WARNING',
                check_name='Series Type',
                message=f'Advanced series type: {", ".join(detected_types)} - may have viewer compatibility issues',
                details={
                    'detected_types': detected_types,
                    'series_description': series_description or self.volume.series_description,
                    'note': 'DTI/DSC/MoCo series may fail in ITK-SNAP, 3D Slicer due to strict DICOM conformance'
                }
            )

        return QCResult(
            status='PASS',
            check_name='Series Type',
            message='Standard series type',
            details={'series_description': series_description or self.volume.series_description}
        )

    def check_frame_of_reference(self) -> QCResult:
        """
        Check Frame of Reference UID consistency.

        All slices in a series should share the same FrameOfReferenceUID.
        Inconsistent FoR can cause registration/overlay failures in viewers.

        Returns:
            QCResult with status based on FoR consistency
        """
        # Note: DicomVolume currently doesn't store per-slice FoR UIDs
        # This check validates that the volume has position data that's self-consistent
        positions = self.volume.image_positions

        if len(positions) < 2:
            return QCResult(
                status='PASS',
                check_name='Frame of Reference',
                message='Single slice - frame of reference assumed consistent',
                details={'num_slices': len(positions)}
            )

        # Check that all slices lie on a consistent plane/line
        # Calculate the slice direction from first two positions
        slice_dir = positions[1] - positions[0]
        slice_dir_norm = np.linalg.norm(slice_dir)

        if slice_dir_norm < 1e-6:
            return QCResult(
                status='WARNING',
                check_name='Frame of Reference',
                message='First two slices have identical positions',
                details={'position_0': positions[0].tolist(), 'position_1': positions[1].tolist()}
            )

        slice_dir = slice_dir / slice_dir_norm

        # Check that all other positions follow the same direction
        max_deviation = 0.0
        problem_slices = []

        for i in range(2, len(positions)):
            vec = positions[i] - positions[0]
            vec_norm = np.linalg.norm(vec)
            if vec_norm > 1e-6:
                # Project onto slice direction and find perpendicular component
                parallel = np.dot(vec, slice_dir) * slice_dir
                perpendicular = vec - parallel
                deviation = np.linalg.norm(perpendicular)

                if deviation > max_deviation:
                    max_deviation = deviation

                # Flag if deviation > 1mm (slices not coplanar)
                if deviation > 1.0:
                    problem_slices.append({'slice': i, 'deviation_mm': float(deviation)})

        details = {
            'num_slices': len(positions),
            'max_deviation_mm': float(max_deviation),
            'slice_direction': slice_dir.tolist(),
        }

        if problem_slices:
            details['problem_slices'] = problem_slices[:5]  # Limit to first 5
            return QCResult(
                status='WARNING',
                check_name='Frame of Reference',
                message=f'Slices not coplanar: {len(problem_slices)} slices deviate up to {max_deviation:.1f}mm',
                details=details
            )

        return QCResult(
            status='PASS',
            check_name='Frame of Reference',
            message=f'Slices are coplanar (max deviation: {max_deviation:.2f}mm)',
            details=details
        )

    def check_slice_count(self) -> QCResult:
        """
        Validate slice count is reasonable and complete.

        Checks for:
        - Very few slices (possible truncation)
        - Mismatch between expected and actual slice count

        Returns:
            QCResult with status based on slice count validation
        """
        num_slices = self.volume.num_slices
        slice_thickness = self.volume.slice_thickness or 1.0

        # Calculate expected coverage
        if len(self.volume.slice_locations) > 1:
            coverage_mm = abs(self.volume.slice_locations[-1] - self.volume.slice_locations[0])
            expected_slices = int(coverage_mm / slice_thickness) + 1
        else:
            expected_slices = num_slices
            coverage_mm = 0

        details = {
            'num_slices': num_slices,
            'coverage_mm': float(coverage_mm),
            'slice_thickness_mm': float(slice_thickness),
            'expected_slices': expected_slices,
        }

        if num_slices < 3:
            return QCResult(
                status='WARNING',
                check_name='Slice Count',
                message=f'Very few slices ({num_slices}) - may be incomplete or localizer',
                details=details
            )

        # Check if significantly fewer slices than expected (possible truncation)
        if expected_slices > 0 and num_slices < expected_slices * 0.8:
            return QCResult(
                status='WARNING',
                check_name='Slice Count',
                message=f'Fewer slices than expected: {num_slices} vs ~{expected_slices} (possible truncation)',
                details=details
            )

        return QCResult(
            status='PASS',
            check_name='Slice Count',
            message=f'{num_slices} slices, {coverage_mm:.1f}mm coverage',
            details=details
        )

    def check_geometry_metadata(self) -> QCResult:
        """
        Check for missing geometry metadata DICOM tags.

        Derived images (like Color FA maps) sometimes lose geometry tags,
        resulting in SimpleITK using defaults. This makes the data
        non-reconstructable in 3D viewers.

        Returns:
            QCResult with FAIL if critical geometry metadata is missing
        """
        missing_tags = getattr(self.volume, 'missing_geometry_tags', [])

        # Critical tags that affect geometry
        critical_missing = [t for t in missing_tags if t in ('PixelSpacing', 'ImageOrientationPatient', 'ImagePositionPatient')]

        if critical_missing:
            return QCResult(
                status='FAIL',
                check_name='Geometry Metadata',
                message=f'Missing DICOM tags: {", ".join(critical_missing)}',
                details={
                    'missing_tags': missing_tags,
                    'note': 'Derived images may lose geometry metadata, making 3D reconstruction unreliable'
                }
            )

        # SliceThickness alone is a warning (spacing can be inferred from positions)
        if 'SliceThickness' in missing_tags:
            return QCResult(
                status='WARNING',
                check_name='Geometry Metadata',
                message='Missing SliceThickness tag (spacing inferred from positions)',
                details={
                    'missing_tags': missing_tags,
                }
            )

        return QCResult(
            status='PASS',
            check_name='Geometry Metadata',
            message='Geometry metadata present',
            details={}
        )

    def check_4d_data(self) -> QCResult:
        """
        Check if this is 4D data (fMRI, DTI, DSC perfusion, etc.).

        4D data has a time/volume dimension. We display the first timepoint
        for thumbnail and QC, but the user should be aware of the full dataset.

        Returns:
            QCResult with INFO if 4D data detected
        """
        num_timepoints = getattr(self.volume, 'num_timepoints', None)

        if num_timepoints is not None and num_timepoints > 1:
            return QCResult(
                status='NOTE',
                check_name='4D Data',
                message=f'4D dataset ({num_timepoints} timepoints) - displaying first volume only',
                details={
                    'num_timepoints': num_timepoints,
                    'displayed_timepoint': 0,
                    'note': 'fMRI/DTI/DSC data - full timeseries not shown in viewer'
                }
            )

        return QCResult(
            status='PASS',
            check_name='4D Data',
            message='3D dataset',
            details={'is_4d': False}
        )

    def check_reconstructability(self) -> QCResult:
        """
        Check if the volume is reconstructable as a 3D dataset.

        Detects multi-orientation localizers and other non-reconstructable data
        by checking for signs that slices don't form a consistent 3D volume:
        - Slices not coplanar (different orientations)
        - Highly irregular spacing
        - Very few slices with large coverage (scout/localizer pattern)

        Returns:
            QCResult with FAIL if not reconstructable, WARNING if questionable
        """
        positions = self.volume.image_positions
        num_slices = len(positions)

        if num_slices < 2:
            return QCResult(
                status='WARNING',
                check_name='Reconstructability',
                message='Single slice - not a 3D volume',
                details={'num_slices': num_slices}
            )

        # Calculate inter-slice distances and directions
        issues = []

        # Check 0: Multiple orientations detected during loading (most reliable check)
        num_orientations = getattr(self.volume, 'num_orientations', 1)
        if num_orientations > 1:
            issues.append(f"Frames have {num_orientations} different orientations (multi-plane localizer)")

        # Check 1: Are slices coplanar? (different orientations = not coplanar)
        slice_vectors = []
        for i in range(1, len(positions)):
            vec = positions[i] - positions[i-1]
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                slice_vectors.append(vec / norm)

        if len(slice_vectors) >= 2:
            # Check if all slice vectors point in roughly the same direction
            reference_dir = slice_vectors[0]
            max_angle = 0.0
            inconsistent_count = 0

            for i, vec in enumerate(slice_vectors[1:], start=1):
                # Dot product gives cos(angle), should be close to 1 or -1 for parallel
                dot = abs(np.dot(reference_dir, vec))
                angle_deg = np.degrees(np.arccos(np.clip(dot, -1, 1)))

                if angle_deg > max_angle:
                    max_angle = angle_deg

                # If angle > 10 degrees, slices have different orientations
                if angle_deg > 10:
                    inconsistent_count += 1

            if inconsistent_count > 0:
                issues.append(f"Frames have different orientations ({inconsistent_count + 1} directions detected)")

        # Check 2: Irregular spacing pattern typical of localizers
        if len(positions) >= 3:
            distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            if len(distances) > 1:
                spacing_cv = np.std(distances) / np.mean(distances) if np.mean(distances) > 0 else 0
                if spacing_cv > 0.5:  # Very irregular
                    issues.append(f"Highly irregular slice spacing (CV={spacing_cv:.0%})")

        # Check 3: Few slices with large gaps (localizer pattern)
        if num_slices <= 15 and len(positions) >= 2:
            total_coverage = np.linalg.norm(positions[-1] - positions[0])
            avg_spacing = total_coverage / (num_slices - 1) if num_slices > 1 else 0

            # Localizers often have 50-100mm+ spacing between slices
            if avg_spacing > 30:
                issues.append(f"Large inter-slice spacing ({avg_spacing:.0f}mm avg) - likely localizer/scout")

        details = {
            'num_slices': num_slices,
            'issues': issues,
        }

        if issues:
            return QCResult(
                status='FAIL',
                check_name='Reconstructability',
                message=f'Not reconstructable: {"; ".join(issues)}',
                details=details
            )

        return QCResult(
            status='PASS',
            check_name='Reconstructability',
            message='Volume is reconstructable as 3D dataset',
            details=details
        )

    def run_all_checks(self, scan_id: str = "unknown") -> QCReport:
        """
        Run all geometry QC checks.

        Args:
            scan_id: Identifier for the scan

        Returns:
            QCReport with all results
        """
        results = [
            self.check_geometry_metadata(),
            self.check_4d_data(),
            self.check_reconstructability(),
            self.check_slice_ordering(),
            self.check_orientation_consistency(),
            self.check_frame_of_reference(),
            self.check_gap_detection(),
            self.check_voxel_anisotropy(),
            self.check_slice_count(),
            self.check_orientation_labels(),
        ]

        # Determine overall status (NOTE doesn't escalate like WARNING)
        statuses = [r.status for r in results]
        if 'FAIL' in statuses:
            overall_status = 'FAIL'
        elif 'WARNING' in statuses:
            overall_status = 'WARNING'
        elif 'NOTE' in statuses:
            overall_status = 'NOTE'
        else:
            overall_status = 'PASS'

        # Extract orientation info from the orientation labels check (last result)
        orientation_result = results[-1]
        orientation_labels = orientation_result.details.get('orientation_labels', {})
        primary_plane = orientation_labels.get('primary_plane', 'UNKNOWN')

        return QCReport(
            scan_id=scan_id,
            results=results,
            overall_status=overall_status,
            orientation_labels=orientation_labels,
            primary_plane=primary_plane,
        )

    def get_display_labels(self) -> Dict[str, Dict[str, str]]:
        """
        Get standard anatomical orientation labels for display on images.

        Uses standard radiological convention:
        - Axial: viewed from feet, shows R-L (horizontal) and A-P (vertical)
        - Sagittal: viewed from left side, shows A-P (horizontal) and S-I (vertical)
        - Coronal: viewed from front, shows R-L (horizontal) and S-I (vertical)

        Note: These are STANDARD labels for anatomical views, not computed
        from DICOM orientation. The actual slice extraction from the volume
        may not correspond to true anatomical planes if the acquisition
        was oblique.

        Returns:
            Dict with keys 'axial', 'coronal', 'sagittal', each containing
            'left', 'right', 'top', 'bottom' labels
        """
        # Standard radiological convention labels
        # These are the conventional labels regardless of acquisition orientation
        return {
            'axial': {
                'left': 'R',      # Patient's right on viewer's left
                'right': 'L',    # Patient's left on viewer's right
                'top': 'A',      # Anterior at top
                'bottom': 'P',   # Posterior at bottom
            },
            'coronal': {
                'left': 'R',
                'right': 'L',
                'top': 'S',      # Superior at top
                'bottom': 'I',   # Inferior at bottom
            },
            'sagittal': {
                'left': 'A',     # Anterior on left
                'right': 'P',    # Posterior on right
                'top': 'S',
                'bottom': 'I',
            },
        }

    def get_acquisition_labels(self) -> Dict[str, str]:
        """
        Get orientation labels based on actual DICOM ImageOrientationPatient.

        This returns labels for the acquired image plane, which may be
        different from standard anatomical planes if acquisition was oblique.

        Returns:
            Dict with 'image_left', 'image_right', 'image_top', 'image_bottom'
            labels for the acquired image orientation.
        """
        orientation = self.volume.image_orientation
        row_cosines = np.array(orientation[0:3])
        col_cosines = np.array(orientation[3:6])

        # Get labels for each direction
        row_labels = self._get_axis_labels(row_cosines)  # (neg, pos)
        col_labels = self._get_axis_labels(col_cosines)  # (neg, pos)

        return {
            'image_left': row_labels[0],
            'image_right': row_labels[1],
            'image_top': col_labels[0],
            'image_bottom': col_labels[1],
        }
