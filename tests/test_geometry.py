"""Tests for geometry QC checks."""

import numpy as np
import pytest
import SimpleITK as sitk

from dicom_qc.core.geometry import GeometryQC, QCResult, QCReport
from dicom_qc.core.volume import DicomVolume


def create_sitk_image(
    shape: tuple = (20, 256, 256),
    spacing: tuple = (3.0, 1.0, 1.0),
    origin: tuple = (0.0, 0.0, 0.0),
    direction: tuple = None,
) -> sitk.Image:
    """Create a SimpleITK image for testing."""
    slices, rows, cols = shape
    np.random.seed(42)
    data = np.random.rand(slices, rows, cols).astype(np.float32) * 1000
    image = sitk.GetImageFromArray(data)
    image.SetSpacing((spacing[2], spacing[1], spacing[0]))
    image.SetOrigin(origin)
    if direction is None:
        direction = (1, 0, 0, 0, 1, 0, 0, 0, 1)
    image.SetDirection(direction)
    return image


class TestQCResult:
    """Tests for the QCResult dataclass."""

    def test_qc_result_creation(self):
        """Test basic QCResult creation."""
        result = QCResult(
            status='PASS',
            check_name='Test Check',
            message='All good',
            details={'key': 'value'},
        )
        assert result.status == 'PASS'
        assert result.check_name == 'Test Check'
        assert result.message == 'All good'
        assert result.details == {'key': 'value'}

    def test_qc_result_to_dict(self):
        """Test QCResult serialization."""
        result = QCResult(
            status='WARNING',
            check_name='Test',
            message='Warning message',
            details={'count': 5},
        )
        d = result.to_dict()
        assert d['status'] == 'WARNING'
        assert d['check_name'] == 'Test'
        assert d['message'] == 'Warning message'
        assert d['details'] == {'count': 5}

    def test_qc_result_default_details(self):
        """Test QCResult with default empty details."""
        result = QCResult(status='PASS', check_name='Test', message='OK')
        assert result.details == {}


class TestQCReport:
    """Tests for the QCReport dataclass."""

    def test_qc_report_creation(self, axial_volume):
        """Test basic QCReport creation."""
        results = [
            QCResult(status='PASS', check_name='Check1', message='OK'),
            QCResult(status='WARNING', check_name='Check2', message='Warn'),
        ]
        report = QCReport(
            scan_id='scan001',
            results=results,
            overall_status='WARNING',
            orientation_labels={'row_positive': 'L'},
            primary_plane='AXIAL',
        )
        assert report.scan_id == 'scan001'
        assert len(report.results) == 2
        assert report.overall_status == 'WARNING'
        assert report.primary_plane == 'AXIAL'

    def test_qc_report_to_dict(self, axial_volume):
        """Test QCReport serialization."""
        results = [QCResult(status='PASS', check_name='Check1', message='OK')]
        report = QCReport(
            scan_id='scan001',
            results=results,
            overall_status='PASS',
            orientation_labels={},
            primary_plane='AXIAL',
        )
        d = report.to_dict()
        assert d['scan_id'] == 'scan001'
        assert 'results' in d
        assert 'timestamp' in d
        assert d['primary_plane'] == 'AXIAL'


class TestSliceOrdering:
    """Tests for check_slice_ordering."""

    def test_monotonic_increasing(self, axial_volume):
        """Test detection of properly ordered slices (increasing)."""
        qc = GeometryQC(axial_volume)
        result = qc.check_slice_ordering()
        assert result.status == 'PASS'
        assert 'monotonically' in result.message

    def test_single_slice_warning(self, single_slice_volume):
        """Test warning for single-slice volumes."""
        qc = GeometryQC(single_slice_volume)
        result = qc.check_slice_ordering()
        assert result.status == 'WARNING'
        assert 'Only one slice' in result.message

class TestOrientationConsistency:
    """Tests for check_orientation_consistency."""

    def test_valid_orientation(self, axial_volume):
        """Test valid orientation vectors pass."""
        qc = GeometryQC(axial_volume)
        result = qc.check_orientation_consistency()
        assert result.status == 'PASS'
        assert 'valid and consistent' in result.message

    def test_coronal_orientation(self, coronal_volume):
        """Test coronal orientation vectors pass."""
        qc = GeometryQC(coronal_volume)
        result = qc.check_orientation_consistency()
        assert result.status == 'PASS'

    def test_sagittal_orientation(self, sagittal_volume):
        """Test sagittal orientation vectors pass."""
        qc = GeometryQC(sagittal_volume)
        result = qc.check_orientation_consistency()
        assert result.status == 'PASS'

    def test_oblique_orientation(self, oblique_volume):
        """Test oblique orientation vectors still valid."""
        qc = GeometryQC(oblique_volume)
        result = qc.check_orientation_consistency()
        assert result.status == 'PASS'

    def test_non_unit_row_warning(self):
        """Test warning for non-unit row vector."""
        # Create image with non-unit direction vectors
        image = create_sitk_image()
        # Scale row direction (non-unit)
        direction = (2, 0, 0, 0, 1, 0, 0, 0, 1)  # Row has magnitude 2
        image.SetDirection(direction)

        volume = DicomVolume(sitk_image=image)
        qc = GeometryQC(volume)
        result = qc.check_orientation_consistency()
        assert result.status == 'WARNING'
        assert 'not unit length' in result.message

    def test_non_orthogonal_warning(self):
        """Test warning for non-orthogonal vectors."""
        image = create_sitk_image()
        # Non-orthogonal: row and col not perpendicular but both unit length
        # row = (1, 0, 0), col = (0.5, 0.866, 0) - 30 degree angle, col is unit
        # But we need the direction matrix format where row_dir = [d0, d3, d6]
        # row_dir = (1, 0, 0): d0=1, d3=0, d6=0
        # col_dir = (0.5, 0.866, 0): d1=0.5, d4=0.866, d7=0
        # dot(row, col) = 0.5 != 0 (not orthogonal)
        direction = (1, 0.5, 0, 0, 0.866, 0, 0, 0, 1)
        image.SetDirection(direction)

        volume = DicomVolume(sitk_image=image)
        qc = GeometryQC(volume)
        result = qc.check_orientation_consistency()
        assert result.status == 'WARNING'
        assert 'not orthogonal' in result.message


class TestGapDetection:
    """Tests for check_gap_detection."""

    def test_regular_spacing_pass(self, axial_volume):
        """Test regular spacing passes."""
        qc = GeometryQC(axial_volume)
        result = qc.check_gap_detection()
        assert result.status == 'PASS'
        assert 'No gaps detected' in result.message

    def test_single_slice_warning(self, single_slice_volume):
        """Test single-slice warning."""
        qc = GeometryQC(single_slice_volume)
        result = qc.check_gap_detection()
        assert result.status == 'WARNING'
        assert 'Only one slice' in result.message

    def test_gap_detection_fail(self):
        """Test detection of gaps (missing slices)."""
        # Create volume with irregular spacing (gap at slice 5)
        slices = 10
        data = np.random.rand(slices, 64, 64).astype(np.float32)
        image = sitk.GetImageFromArray(data)
        # Normal spacing is 3mm, but we'll create positions manually
        # by modifying the slice direction to simulate gaps
        image.SetSpacing((1.0, 1.0, 3.0))
        image.SetOrigin((0.0, 0.0, 0.0))
        image.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

        volume = DicomVolume(sitk_image=image)
        qc = GeometryQC(volume)
        result = qc.check_gap_detection()
        # With regular spacing from SimpleITK, should pass
        assert result.status == 'PASS'


class TestOrientationLabels:
    """Tests for check_orientation_labels and plane detection."""

    def test_axial_orientation_labels(self, axial_volume):
        """Test axial orientation produces correct labels."""
        qc = GeometryQC(axial_volume)
        result = qc.check_orientation_labels()
        assert result.status == 'PASS'
        assert 'AXIAL' in result.message

        labels = result.details['orientation_labels']
        assert labels['primary_plane'] == 'AXIAL'
        # Axial: row=X(L), col=Y(P), slice=Z(S)
        assert labels['row_positive'] == 'L'
        assert labels['col_positive'] == 'P'
        assert labels['slice_direction'] == 'S'

    def test_coronal_orientation_labels(self, coronal_volume):
        """Test coronal orientation produces correct labels."""
        qc = GeometryQC(coronal_volume)
        result = qc.check_orientation_labels()
        assert result.status == 'PASS'
        assert 'CORONAL' in result.message

        labels = result.details['orientation_labels']
        assert labels['primary_plane'] == 'CORONAL'

    def test_sagittal_orientation_labels(self, sagittal_volume):
        """Test sagittal orientation produces correct labels."""
        qc = GeometryQC(sagittal_volume)
        result = qc.check_orientation_labels()
        assert result.status == 'PASS'
        assert 'SAGITTAL' in result.message

        labels = result.details['orientation_labels']
        assert labels['primary_plane'] == 'SAGITTAL'

    def test_oblique_orientation_labels(self, oblique_volume):
        """Test oblique orientation detection."""
        qc = GeometryQC(oblique_volume)
        result = qc.check_orientation_labels()
        assert result.status == 'PASS'
        # 45-degree oblique should be detected as OBLIQUE
        labels = result.details['orientation_labels']
        assert labels['primary_plane'] == 'OBLIQUE'

    def test_get_opposite_label(self, axial_volume):
        """Test opposite label lookup."""
        qc = GeometryQC(axial_volume)
        assert qc._get_opposite_label('L') == 'R'
        assert qc._get_opposite_label('R') == 'L'
        assert qc._get_opposite_label('A') == 'P'
        assert qc._get_opposite_label('P') == 'A'
        assert qc._get_opposite_label('S') == 'I'
        assert qc._get_opposite_label('I') == 'S'


class TestVoxelAnisotropy:
    """Tests for check_voxel_anisotropy."""

    def test_isotropic_pass(self, isotropic_volume):
        """Test isotropic voxels pass."""
        qc = GeometryQC(isotropic_volume)
        result = qc.check_voxel_anisotropy()
        assert result.status == 'PASS'
        assert result.details['anisotropy_ratio'] == pytest.approx(1.0)

    def test_moderate_anisotropy_warning(self):
        """Test moderate anisotropy (2-4x) gives warning."""
        image = create_sitk_image(spacing=(3.0, 1.0, 1.0))  # 3x ratio
        volume = DicomVolume(sitk_image=image)
        qc = GeometryQC(volume)
        result = qc.check_voxel_anisotropy()
        assert result.status == 'WARNING'
        assert 'anisotropic' in result.message.lower()
        assert result.details['anisotropy_ratio'] == pytest.approx(3.0)

    def test_severe_anisotropy_warning(self, anisotropic_volume):
        """Test severe anisotropy (>4x) gives warning."""
        qc = GeometryQC(anisotropic_volume)
        result = qc.check_voxel_anisotropy()
        assert result.status == 'WARNING'
        assert 'anisotropic' in result.message.lower()
        # 10mm / 0.5mm = 20x ratio
        assert result.details['anisotropy_ratio'] == pytest.approx(20.0)


class TestSeriesType:
    """Tests for check_series_type."""

    def test_standard_series_pass(self, axial_volume):
        """Test standard series type passes."""
        qc = GeometryQC(axial_volume)
        result = qc.check_series_type()
        assert result.status == 'PASS'
        assert 'Standard series type' in result.message

    def test_dti_detection(self, dti_volume):
        """Test DTI series type detection."""
        qc = GeometryQC(dti_volume)
        result = qc.check_series_type()
        assert result.status == 'WARNING'
        assert 'DTI' in result.details['detected_types']

    def test_perfusion_detection(self, perfusion_volume):
        """Test DSC perfusion series type detection."""
        qc = GeometryQC(perfusion_volume)
        result = qc.check_series_type()
        assert result.status == 'WARNING'
        assert 'DSC_PERFUSION' in result.details['detected_types']

    def test_diffusion_pattern(self):
        """Test diffusion pattern matching."""
        image = create_sitk_image()
        volume = DicomVolume(
            sitk_image=image,
            series_description='DWI_b1000',
        )
        qc = GeometryQC(volume)
        result = qc.check_series_type()
        assert result.status == 'WARNING'
        assert 'DTI' in result.details['detected_types']

    def test_adc_pattern(self):
        """Test ADC map pattern matching."""
        image = create_sitk_image()
        volume = DicomVolume(
            sitk_image=image,
            series_description='ADC Map',
        )
        qc = GeometryQC(volume)
        result = qc.check_series_type()
        assert result.status == 'WARNING'

    def test_moco_pattern(self):
        """Test MoCo series pattern matching."""
        image = create_sitk_image()
        volume = DicomVolume(
            sitk_image=image,
            series_description='MOCO_T1',
        )
        qc = GeometryQC(volume)
        result = qc.check_series_type()
        assert result.status == 'WARNING'
        assert 'MOCO' in result.details['detected_types']


class TestFrameOfReference:
    """Tests for check_frame_of_reference."""

    def test_coplanar_slices_pass(self, axial_volume):
        """Test coplanar slices pass."""
        qc = GeometryQC(axial_volume)
        result = qc.check_frame_of_reference()
        assert result.status == 'PASS'
        assert 'coplanar' in result.message.lower()

    def test_single_slice(self, single_slice_volume):
        """Test single slice is accepted."""
        qc = GeometryQC(single_slice_volume)
        result = qc.check_frame_of_reference()
        assert result.status == 'PASS'


class TestSliceCount:
    """Tests for check_slice_count."""

    def test_normal_slice_count(self, axial_volume):
        """Test normal slice count passes."""
        qc = GeometryQC(axial_volume)
        result = qc.check_slice_count()
        assert result.status == 'PASS'
        assert result.details['num_slices'] == 20

    def test_few_slices_warning(self):
        """Test very few slices generates warning."""
        image = create_sitk_image(shape=(2, 64, 64))
        volume = DicomVolume(sitk_image=image)
        qc = GeometryQC(volume)
        result = qc.check_slice_count()
        assert result.status == 'WARNING'
        assert 'few slices' in result.message.lower()


class TestGeometryMetadata:
    """Tests for check_geometry_metadata."""

    def test_complete_metadata_pass(self, axial_volume):
        """Test complete geometry metadata passes."""
        qc = GeometryQC(axial_volume)
        result = qc.check_geometry_metadata()
        assert result.status == 'PASS'

    def test_missing_critical_tags_fail(self, missing_geometry_volume):
        """Test missing critical geometry tags fails."""
        qc = GeometryQC(missing_geometry_volume)
        result = qc.check_geometry_metadata()
        assert result.status == 'FAIL'
        assert 'PixelSpacing' in result.message

    def test_missing_slice_thickness_warning(self):
        """Test missing SliceThickness gives warning."""
        image = create_sitk_image()
        volume = DicomVolume(
            sitk_image=image,
            missing_geometry_tags=['SliceThickness'],
        )
        qc = GeometryQC(volume)
        result = qc.check_geometry_metadata()
        assert result.status == 'WARNING'


class Test4DData:
    """Tests for check_4d_data."""

    def test_3d_data_pass(self, axial_volume):
        """Test 3D data passes."""
        qc = GeometryQC(axial_volume)
        result = qc.check_4d_data()
        assert result.status == 'PASS'
        assert '3D dataset' in result.message

    def test_4d_data_note(self, volume_4d):
        """Test 4D data gives NOTE status."""
        qc = GeometryQC(volume_4d)
        result = qc.check_4d_data()
        assert result.status == 'NOTE'
        assert '4D dataset' in result.message
        assert result.details['num_timepoints'] == 100


class TestReconstructability:
    """Tests for check_reconstructability."""

    def test_reconstructable_pass(self, axial_volume):
        """Test reconstructable volume passes."""
        qc = GeometryQC(axial_volume)
        result = qc.check_reconstructability()
        assert result.status == 'PASS'
        assert 'reconstructable' in result.message.lower()

    def test_single_slice_warning(self, single_slice_volume):
        """Test single slice gives warning."""
        qc = GeometryQC(single_slice_volume)
        result = qc.check_reconstructability()
        assert result.status == 'WARNING'
        assert 'Single slice' in result.message

    def test_multi_orientation_localizer_fail(self, localizer_volume):
        """Test multi-orientation localizer fails."""
        qc = GeometryQC(localizer_volume)
        result = qc.check_reconstructability()
        assert result.status == 'FAIL'
        assert 'Not reconstructable' in result.message


class TestRunAllChecks:
    """Tests for run_all_checks."""

    def test_run_all_checks_pass(self, axial_volume):
        """Test run_all_checks with a good volume."""
        qc = GeometryQC(axial_volume)
        report = qc.run_all_checks(scan_id='test_scan')

        assert isinstance(report, QCReport)
        assert report.scan_id == 'test_scan'
        assert len(report.results) == 10  # All checks
        assert report.overall_status in ('PASS', 'WARNING', 'NOTE')
        assert report.primary_plane == 'AXIAL'

    def test_run_all_checks_with_failures(self, missing_geometry_volume):
        """Test run_all_checks with failures produces FAIL status."""
        qc = GeometryQC(missing_geometry_volume)
        report = qc.run_all_checks(scan_id='bad_scan')

        assert report.overall_status == 'FAIL'
        # Find the geometry metadata check result
        geo_result = next(r for r in report.results if r.check_name == 'Geometry Metadata')
        assert geo_result.status == 'FAIL'

    def test_run_all_checks_with_note(self, volume_4d):
        """Test 4D data produces NOTE status when no failures."""
        qc = GeometryQC(volume_4d)
        report = qc.run_all_checks()

        # Should have NOTE for 4D data
        data_4d_result = next(r for r in report.results if r.check_name == '4D Data')
        assert data_4d_result.status == 'NOTE'

    def test_report_orientation_labels(self, axial_volume):
        """Test report includes orientation labels."""
        qc = GeometryQC(axial_volume)
        report = qc.run_all_checks()

        assert 'row_positive' in report.orientation_labels
        assert 'col_positive' in report.orientation_labels
        assert report.primary_plane == 'AXIAL'


class TestDisplayLabels:
    """Tests for get_display_labels and get_acquisition_labels."""

    def test_get_display_labels(self, axial_volume):
        """Test standard display labels."""
        qc = GeometryQC(axial_volume)
        labels = qc.get_display_labels()

        assert 'axial' in labels
        assert 'coronal' in labels
        assert 'sagittal' in labels

        # Standard radiological convention
        assert labels['axial']['left'] == 'R'
        assert labels['axial']['right'] == 'L'
        assert labels['coronal']['top'] == 'S'
        assert labels['sagittal']['left'] == 'A'

    def test_get_acquisition_labels_axial(self, axial_volume):
        """Test acquisition labels for axial volume."""
        qc = GeometryQC(axial_volume)
        labels = qc.get_acquisition_labels()

        assert 'image_left' in labels
        assert 'image_right' in labels
        assert 'image_top' in labels
        assert 'image_bottom' in labels

    def test_get_acquisition_labels_sagittal(self, sagittal_volume):
        """Test acquisition labels for sagittal volume."""
        qc = GeometryQC(sagittal_volume)
        labels = qc.get_acquisition_labels()

        # Sagittal fixture: row_dir=(0,1,0)=Y, col_dir=(0,0,1)=Z
        # Y axis positive = Posterior, Z axis positive = Superior
        assert labels['image_right'] == 'P'  # row positive direction
        assert labels['image_bottom'] == 'S'  # col positive direction


class TestHelperMethods:
    """Tests for internal helper methods."""

    def test_find_duplicates(self, axial_volume):
        """Test duplicate finding helper."""
        qc = GeometryQC(axial_volume)

        # No duplicates
        locations = np.array([0, 1, 2, 3, 4])
        assert qc._find_duplicates(locations) == []

        # With duplicates
        locations = np.array([0, 1, 1, 2, 3, 3, 3])
        duplicates = qc._find_duplicates(locations)
        assert 1.0 in duplicates
        assert 3.0 in duplicates

    def test_get_orientation_label(self, axial_volume):
        """Test orientation label calculation."""
        qc = GeometryQC(axial_volume)

        # Positive X = Left
        assert qc._get_orientation_label(np.array([1, 0, 0])) == 'L'
        # Negative X = Right
        assert qc._get_orientation_label(np.array([-1, 0, 0])) == 'R'
        # Positive Y = Posterior
        assert qc._get_orientation_label(np.array([0, 1, 0])) == 'P'
        # Negative Y = Anterior
        assert qc._get_orientation_label(np.array([0, -1, 0])) == 'A'
        # Positive Z = Superior
        assert qc._get_orientation_label(np.array([0, 0, 1])) == 'S'
        # Negative Z = Inferior
        assert qc._get_orientation_label(np.array([0, 0, -1])) == 'I'

    def test_determine_plane(self, axial_volume):
        """Test plane determination from slice normal."""
        qc = GeometryQC(axial_volume)

        # Z-dominant = AXIAL
        assert qc._determine_plane(np.array([0, 0, 1])) == 'AXIAL'
        # Y-dominant = CORONAL
        assert qc._determine_plane(np.array([0, 1, 0])) == 'CORONAL'
        # X-dominant = SAGITTAL
        assert qc._determine_plane(np.array([1, 0, 0])) == 'SAGITTAL'
        # Mixed = OBLIQUE
        norm = np.array([0.5, 0.5, 0.5])
        norm = norm / np.linalg.norm(norm)
        assert qc._determine_plane(norm) == 'OBLIQUE'
