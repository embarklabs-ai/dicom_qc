"""Tests for DicomVolume class."""

import numpy as np
import pytest
import SimpleITK as sitk

from dicom_qc.core.volume import DicomVolume, ScanInfo

from tests.helpers import create_sitk_image


class TestScanInfo:
    """Tests for the ScanInfo dataclass."""

    def test_scan_info_creation(self):
        """Test basic ScanInfo creation."""
        info = ScanInfo(
            id='001',
            description='T1 MPRAGE',
            modality='MR',
            num_files=192,
        )
        assert info.id == '001'
        assert info.description == 'T1 MPRAGE'
        assert info.modality == 'MR'
        assert info.num_files == 192

    def test_scan_info_optional_fields(self):
        """Test ScanInfo with optional fields."""
        info = ScanInfo(
            id='001',
            description='CT Head',
            modality='CT',
            num_files=256,
            project='BRAIN01',
            subject='SUBJ001',
            experiment='EXP001',
        )
        assert info.project == 'BRAIN01'
        assert info.subject == 'SUBJ001'
        assert info.experiment == 'EXP001'


class TestDicomVolumeInit:
    """Tests for DicomVolume initialization."""

    def test_basic_init(self, axial_volume):
        """Test basic DicomVolume initialization."""
        assert axial_volume.modality == 'MR'
        assert axial_volume.series_description == 'T1 MPRAGE'
        assert axial_volume.errors == []
        assert axial_volume.warnings == []

    def test_init_with_metadata(self):
        """Test initialization with full metadata."""
        image = create_sitk_image()
        volume = DicomVolume(
            sitk_image=image,
            modality='CT',
            series_description='CT Chest',
            patient_position='HFS',
            study_instance_uid='1.2.3.4',
            series_instance_uid='1.2.3.5',
        )
        assert volume.modality == 'CT'
        assert volume.patient_position == 'HFS'
        assert volume.study_instance_uid == '1.2.3.4'
        assert volume.series_instance_uid == '1.2.3.5'

    def test_init_with_errors_and_warnings(self):
        """Test initialization with errors and warnings."""
        image = create_sitk_image()
        errors = [('file1.dcm', Exception('Corrupt file'))]
        warnings = [('file2.dcm', 'Missing SliceThickness')]
        volume = DicomVolume(
            sitk_image=image,
            errors=errors,
            warnings=warnings,
        )
        assert len(volume.errors) == 1
        assert len(volume.warnings) == 1

    def test_init_4d_data(self, volume_4d):
        """Test initialization with 4D data flags."""
        assert volume_4d.num_timepoints == 100
        assert volume_4d.is_4d is True

    def test_init_missing_geometry_tags(self, missing_geometry_volume):
        """Test initialization with missing geometry tags."""
        assert 'PixelSpacing' in missing_geometry_volume.missing_geometry_tags
        assert 'ImageOrientationPatient' in missing_geometry_volume.missing_geometry_tags


class TestDicomVolumeShape:
    """Tests for shape-related properties."""

    def test_shape(self, axial_volume):
        """Test shape property."""
        shape = axial_volume.shape
        assert len(shape) == 3
        assert shape == (20, 256, 256)  # slices, rows, cols

    def test_num_slices(self, axial_volume):
        """Test num_slices property."""
        assert axial_volume.num_slices == 20

    def test_single_slice_shape(self, single_slice_volume):
        """Test single slice volume shape."""
        assert single_slice_volume.shape == (1, 256, 256)
        assert single_slice_volume.num_slices == 1


class TestDicomVolumeSpacing:
    """Tests for spacing-related properties."""

    def test_pixel_spacing(self, axial_volume):
        """Test pixel_spacing property."""
        spacing = axial_volume.pixel_spacing
        assert len(spacing) == 2
        assert spacing == (1.0, 1.0)  # row, col

    def test_row_spacing(self, axial_volume):
        """Test row_spacing property."""
        assert axial_volume.row_spacing == 1.0

    def test_col_spacing(self, axial_volume):
        """Test col_spacing property."""
        assert axial_volume.col_spacing == 1.0

    def test_slice_thickness(self, axial_volume):
        """Test slice_thickness property."""
        assert axial_volume.slice_thickness == 3.0

    def test_voxel_spacing(self, axial_volume):
        """Test voxel_spacing property."""
        spacing = axial_volume.voxel_spacing
        assert len(spacing) == 3
        assert spacing == (3.0, 1.0, 1.0)  # slice, row, col

    def test_anisotropic_spacing(self, anisotropic_volume):
        """Test anisotropic voxel spacing."""
        spacing = anisotropic_volume.voxel_spacing
        assert spacing == (10.0, 0.5, 0.5)

    def test_isotropic_spacing(self, isotropic_volume):
        """Test isotropic voxel spacing."""
        spacing = isotropic_volume.voxel_spacing
        assert spacing == (1.0, 1.0, 1.0)


class TestDicomVolumeOrientation:
    """Tests for orientation-related properties."""

    def test_image_orientation_axial(self, axial_volume):
        """Test image_orientation for axial volume."""
        orientation = axial_volume.image_orientation
        assert len(orientation) == 6
        # Axial: row=X(1,0,0), col=Y(0,1,0)
        np.testing.assert_array_almost_equal(orientation[0:3], [1, 0, 0])
        np.testing.assert_array_almost_equal(orientation[3:6], [0, 1, 0])

    def test_image_orientation_coronal(self, coronal_volume):
        """Test image_orientation for coronal volume."""
        orientation = coronal_volume.image_orientation
        assert len(orientation) == 6

    def test_image_orientation_sagittal(self, sagittal_volume):
        """Test image_orientation for sagittal volume."""
        orientation = sagittal_volume.image_orientation
        assert len(orientation) == 6

    def test_row_direction(self, axial_volume):
        """Test row_direction property."""
        row_dir = axial_volume.row_direction
        assert len(row_dir) == 3
        np.testing.assert_array_almost_equal(row_dir, [1, 0, 0])

    def test_col_direction(self, axial_volume):
        """Test col_direction property."""
        col_dir = axial_volume.col_direction
        assert len(col_dir) == 3
        np.testing.assert_array_almost_equal(col_dir, [0, 1, 0])

    def test_slice_normal(self, axial_volume):
        """Test slice_normal property."""
        normal = axial_volume.slice_normal
        assert len(normal) == 3
        # For axial: cross(X, Y) = Z
        np.testing.assert_array_almost_equal(normal, [0, 0, 1])

    def test_slice_normal_coronal(self, coronal_volume):
        """Test slice_normal for coronal volume."""
        normal = coronal_volume.slice_normal
        # For coronal: normal should be along Y
        assert np.argmax(np.abs(normal)) == 1

    def test_slice_normal_sagittal(self, sagittal_volume):
        """Test slice_normal for sagittal volume."""
        normal = sagittal_volume.slice_normal
        # For sagittal: normal should be along X
        assert np.argmax(np.abs(normal)) == 0


class TestDicomVolumePositions:
    """Tests for position-related properties."""

    def test_image_positions(self, axial_volume):
        """Test image_positions property."""
        positions = axial_volume.image_positions
        assert positions.shape == (20, 3)  # 20 slices, 3 coordinates

    def test_image_positions_ordering(self, axial_volume):
        """Test that image positions are ordered correctly."""
        positions = axial_volume.image_positions
        # Z should increase (axial volume with identity direction)
        z_values = positions[:, 2]
        assert np.all(np.diff(z_values) > 0)

    def test_slice_locations(self, axial_volume):
        """Test slice_locations property."""
        locations = axial_volume.slice_locations
        assert len(locations) == 20
        # Should be monotonic
        assert np.all(np.diff(locations) > 0) or np.all(np.diff(locations) < 0)

    def test_slice_locations_spacing(self, axial_volume):
        """Test slice location spacing matches slice thickness."""
        locations = axial_volume.slice_locations
        diffs = np.diff(locations)
        # All differences should be approximately equal to slice thickness
        np.testing.assert_array_almost_equal(diffs, 3.0)


class TestDicomVolumePixelArray:
    """Tests for pixel array access."""

    def test_pixel_array_shape(self, axial_volume):
        """Test pixel_array shape."""
        arr = axial_volume.pixel_array
        assert arr.shape == (20, 256, 256)

    def test_pixel_array_dtype(self, axial_volume):
        """Test pixel_array dtype."""
        arr = axial_volume.pixel_array
        assert arr.dtype == np.float32

    def test_pixel_array_cached(self, axial_volume):
        """Test pixel_array caching."""
        arr1 = axial_volume.pixel_array
        arr2 = axial_volume.pixel_array
        assert arr1 is arr2  # Same object (cached)

    def test_get_slice(self, axial_volume):
        """Test get_slice method."""
        slice_data = axial_volume.get_slice(10)
        assert slice_data.shape == (256, 256)

    def test_get_coronal_slice(self, axial_volume):
        """Test get_coronal_slice method."""
        slice_data = axial_volume.get_coronal_slice(128)
        assert slice_data.shape == (20, 256)  # slices x cols

    def test_get_sagittal_slice(self, axial_volume):
        """Test get_sagittal_slice method."""
        slice_data = axial_volume.get_sagittal_slice(128)
        assert slice_data.shape == (20, 256)  # slices x rows

    def test_get_slice_bounds(self, axial_volume):
        """Test slice access at boundaries."""
        # First slice
        first = axial_volume.get_slice(0)
        assert first.shape == (256, 256)
        # Last slice
        last = axial_volume.get_slice(19)
        assert last.shape == (256, 256)

    def test_get_slice_out_of_bounds(self, axial_volume):
        """Test slice access raises on invalid index."""
        with pytest.raises(IndexError):
            axial_volume.get_slice(100)

    def test_get_coronal_slice_out_of_bounds(self, axial_volume):
        """Test coronal slice access raises on invalid index."""
        with pytest.raises(IndexError):
            axial_volume.get_coronal_slice(300)

    def test_get_sagittal_slice_out_of_bounds(self, axial_volume):
        """Test sagittal slice access raises on invalid index."""
        with pytest.raises(IndexError):
            axial_volume.get_sagittal_slice(300)


class TestDicomVolumeLPS:
    """Tests for LPS reorientation."""

    def test_get_lps_array(self, axial_volume):
        """Test get_lps_array method."""
        lps_array, lps_spacing = axial_volume.get_lps_array()
        assert lps_array.ndim == 3
        assert len(lps_spacing) == 3

    def test_get_lps_array_cached(self, axial_volume):
        """Test LPS array caching."""
        result1 = axial_volume.get_lps_array()
        result2 = axial_volume.get_lps_array()
        assert result1[0] is result2[0]  # Same array object

    def test_get_reoriented_slices(self, axial_volume):
        """Test get_reoriented_slices method."""
        lps_array, lps_spacing, lps_image = axial_volume.get_reoriented_slices()
        assert lps_array.ndim == 3
        assert len(lps_spacing) == 3
        assert isinstance(lps_image, sitk.Image)

    def test_lps_spacing_order(self, axial_volume):
        """Test LPS spacing is in (S, P, L) order."""
        _, lps_spacing = axial_volume.get_lps_array()
        # For axial acquisition with 3mm slice, 1mm in-plane:
        # S spacing = 3.0, P spacing = 1.0, L spacing = 1.0
        assert lps_spacing == (3.0, 1.0, 1.0)


class TestDicomVolumeIntensity:
    """Tests for intensity-related methods."""

    def test_get_intensity_range(self, axial_volume):
        """Test get_intensity_range method."""
        min_val, max_val = axial_volume.get_intensity_range()
        assert min_val < max_val
        assert min_val >= 0  # Random data is 0-1000

    def test_get_auto_window_mr(self, axial_volume):
        """Test get_auto_window for MR."""
        center, width = axial_volume.get_auto_window()
        assert width > 0
        # MR uses percentile-based windowing
        assert center > 0

    def test_get_auto_window_ct(self, ct_volume):
        """Test get_auto_window for CT uses standard window."""
        center, width = ct_volume.get_auto_window()
        # CT uses standard soft tissue window
        assert center == 40.0
        assert width == 400.0


class TestDicomVolume4D:
    """Tests for 4D data handling."""

    def test_is_4d_true(self, volume_4d):
        """Test is_4d for 4D volume."""
        assert volume_4d.is_4d is True
        assert volume_4d.num_timepoints == 100

    def test_is_4d_false(self, axial_volume):
        """Test is_4d for 3D volume."""
        assert axial_volume.is_4d is False
        assert axial_volume.num_timepoints is None

    def test_is_4d_single_timepoint(self):
        """Test is_4d with single timepoint (not 4D)."""
        image = create_sitk_image()
        volume = DicomVolume(sitk_image=image, num_timepoints=1)
        assert volume.is_4d is False


class TestDicomVolumeSerialization:
    """Tests for serialization."""

    def test_to_dict(self, axial_volume):
        """Test to_dict method."""
        d = axial_volume.to_dict()
        assert 'shape' in d
        assert 'num_slices' in d
        assert 'pixel_spacing' in d
        assert 'slice_thickness' in d
        assert 'voxel_spacing' in d
        assert 'modality' in d
        assert 'image_orientation' in d
        assert 'intensity_range' in d

    def test_to_dict_values(self, axial_volume):
        """Test to_dict values are correct."""
        d = axial_volume.to_dict()
        assert d['shape'] == (20, 256, 256)
        assert d['num_slices'] == 20
        assert d['modality'] == 'MR'

    def test_to_dict_4d(self, volume_4d):
        """Test to_dict includes 4D info."""
        d = volume_4d.to_dict()
        assert d['is_4d'] is True
        assert d['num_timepoints'] == 100

    def test_to_dict_orientation(self, axial_volume):
        """Test to_dict orientation is list."""
        d = axial_volume.to_dict()
        assert isinstance(d['image_orientation'], list)
        assert len(d['image_orientation']) == 6

    def test_get_sitk_image(self, axial_volume):
        """Test get_sitk_image returns the SimpleITK image."""
        img = axial_volume.get_sitk_image()
        assert isinstance(img, sitk.Image)
        assert img is axial_volume.sitk_image


class TestDicomVolumeEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_series_description(self):
        """Test volume with empty series description."""
        image = create_sitk_image()
        volume = DicomVolume(sitk_image=image, series_description='')
        assert volume.series_description == ''

    def test_num_orientations(self, localizer_volume):
        """Test multi-orientation volume."""
        assert localizer_volume.num_orientations == 3

    @pytest.mark.parametrize("num_slices", [1, 2, 5, 100])
    def test_different_slice_counts(self, num_slices):
        """Test volumes with various slice counts."""
        image = create_sitk_image(shape=(num_slices, 64, 64))
        volume = DicomVolume(sitk_image=image)
        assert volume.num_slices == num_slices

    def test_very_thin_slices(self):
        """Test volume with very thin slices."""
        image = create_sitk_image(
            shape=(100, 256, 256),
            spacing=(0.1, 0.5, 0.5),
        )
        volume = DicomVolume(sitk_image=image)
        assert volume.slice_thickness == pytest.approx(0.1)

    def test_very_thick_slices(self):
        """Test volume with very thick slices."""
        image = create_sitk_image(
            shape=(5, 256, 256),
            spacing=(50.0, 0.5, 0.5),
        )
        volume = DicomVolume(sitk_image=image)
        assert volume.slice_thickness == pytest.approx(50.0)


class TestFilter4D:
    """Tests for 4D data filtering."""

    def test_filter_4d_with_unordered_temporal_positions(self):
        """Test that 4D filtering works with files in arbitrary order.

        This tests the fix for the bug where files were assumed to be
        in sequential or interleaved order, but XNAT files may come
        in any order.
        """
        from dicom_qc.core.dicom_loader import _filter_4d_generic

        # Mock metadata objects with temporal info
        class MockMetadata:
            def __init__(self, temporal_id, slice_pos, num_temporal=4):
                self.NumberOfTemporalPositions = num_temporal
                self.TemporalPositionIdentifier = temporal_id
                self.ImagePositionPatient = [0, 0, slice_pos]

        # Create 12 "files" (3 slices x 4 timepoints) in RANDOM order
        # This simulates how XNAT might return files
        file_data = [
            # (file_id, temporal_position, slice_z)
            ('f1', 2, 10),   # timepoint 2, slice 0
            ('f2', 1, 20),   # timepoint 1, slice 1
            ('f3', 4, 10),   # timepoint 4, slice 0
            ('f4', 1, 10),   # timepoint 1, slice 0  <- should be in result
            ('f5', 3, 30),   # timepoint 3, slice 2
            ('f6', 2, 20),   # timepoint 2, slice 1
            ('f7', 1, 30),   # timepoint 1, slice 2  <- should be in result
            ('f8', 4, 20),   # timepoint 4, slice 1
            ('f9', 3, 10),   # timepoint 3, slice 0
            ('f10', 2, 30),  # timepoint 2, slice 2
            ('f11', 4, 30),  # timepoint 4, slice 2
            ('f12', 3, 20),  # timepoint 3, slice 1
        ]

        files = [f[0] for f in file_data]
        metadata_map = {f[0]: MockMetadata(f[1], f[2]) for f in file_data}

        def read_metadata(file_id):
            return metadata_map.get(file_id)

        filtered, num_timepoints = _filter_4d_generic(files, read_metadata)

        # Should detect 4 timepoints
        assert num_timepoints == 4

        # Should return only files from timepoint 1
        assert len(filtered) == 3

        # All returned files should be from timepoint 1
        expected_files = {'f4', 'f2', 'f7'}  # timepoint 1 files
        assert set(filtered) == expected_files

    def test_filter_4d_with_sequential_files(self):
        """Test 4D filtering with sequentially ordered files."""
        from dicom_qc.core.dicom_loader import _filter_4d_generic

        class MockMetadata:
            def __init__(self, temporal_id, slice_pos, num_temporal=3):
                self.NumberOfTemporalPositions = num_temporal
                self.TemporalPositionIdentifier = temporal_id
                self.ImagePositionPatient = [0, 0, slice_pos]

        # Sequential order: all timepoint 1, then all timepoint 2, etc.
        # Need at least 10 files for 4D detection to run
        file_data = [
            # Timepoint 1 (4 slices)
            ('f1', 1, 0), ('f2', 1, 10), ('f3', 1, 20), ('f4', 1, 30),
            # Timepoint 2
            ('f5', 2, 0), ('f6', 2, 10), ('f7', 2, 20), ('f8', 2, 30),
            # Timepoint 3
            ('f9', 3, 0), ('f10', 3, 10), ('f11', 3, 20), ('f12', 3, 30),
        ]

        files = [f[0] for f in file_data]
        metadata_map = {f[0]: MockMetadata(f[1], f[2]) for f in file_data}

        def read_metadata(file_id):
            return metadata_map.get(file_id)

        filtered, num_timepoints = _filter_4d_generic(files, read_metadata)

        assert num_timepoints == 3
        assert len(filtered) == 4
        assert set(filtered) == {'f1', 'f2', 'f3', 'f4'}

    def test_filter_4d_with_interleaved_files(self):
        """Test 4D filtering with interleaved files."""
        from dicom_qc.core.dicom_loader import _filter_4d_generic

        class MockMetadata:
            def __init__(self, temporal_id, slice_pos, num_temporal=2):
                self.NumberOfTemporalPositions = num_temporal
                self.TemporalPositionIdentifier = temporal_id
                self.ImagePositionPatient = [0, 0, slice_pos]

        # Interleaved: t1s0, t2s0, t1s1, t2s1, ...
        # Need at least 10 files for 4D detection to run
        file_data = [
            ('f1', 1, 0), ('f2', 2, 0),
            ('f3', 1, 10), ('f4', 2, 10),
            ('f5', 1, 20), ('f6', 2, 20),
            ('f7', 1, 30), ('f8', 2, 30),
            ('f9', 1, 40), ('f10', 2, 40),
            ('f11', 1, 50), ('f12', 2, 50),
        ]

        files = [f[0] for f in file_data]
        metadata_map = {f[0]: MockMetadata(f[1], f[2]) for f in file_data}

        def read_metadata(file_id):
            return metadata_map.get(file_id)

        filtered, num_timepoints = _filter_4d_generic(files, read_metadata)

        assert num_timepoints == 2
        assert len(filtered) == 6
        assert set(filtered) == {'f1', 'f3', 'f5', 'f7', 'f9', 'f11'}

    def test_filter_4d_no_temporal_info(self):
        """Test that non-4D data passes through unchanged."""
        from dicom_qc.core.dicom_loader import _filter_4d_generic

        class MockMetadata:
            def __init__(self, slice_pos):
                self.ImagePositionPatient = [0, 0, slice_pos]
                # No NumberOfTemporalPositions or TemporalPositionIdentifier

        # Regular 3D volume - all unique slice positions
        file_data = [('f1', 0), ('f2', 10), ('f3', 20), ('f4', 30)]
        files = [f[0] for f in file_data]
        metadata_map = {f[0]: MockMetadata(f[1]) for f in file_data}

        def read_metadata(file_id):
            return metadata_map.get(file_id)

        filtered, num_timepoints = _filter_4d_generic(files, read_metadata)

        # Should return original files unchanged
        assert num_timepoints is None
        assert filtered == files

    def test_filter_4d_too_few_files(self):
        """Test that small file counts skip 4D detection."""
        from dicom_qc.core.dicom_loader import _filter_4d_generic

        files = ['f1', 'f2', 'f3']  # Only 3 files

        def read_metadata(f):
            return None

        filtered, num_timepoints = _filter_4d_generic(files, read_metadata)

        # Should return original files - too few to be 4D
        assert num_timepoints is None
        assert filtered == files

    def test_filter_4d_by_duplicate_positions(self):
        """Test 4D detection by duplicate slice positions (Method 2).

        When NumberOfTemporalPositions is not available, detect 4D by
        finding duplicate ImagePositionPatient values.
        """
        from dicom_qc.core.dicom_loader import _filter_4d_generic

        class MockMetadata:
            def __init__(self, slice_pos):
                # No temporal tags - rely on position-based detection
                self.ImagePositionPatient = [0, 0, slice_pos]

        # 4 slices repeated 3 times (simulating 3 timepoints)
        # Files in random order
        file_data = [
            ('f1', 20), ('f2', 0), ('f3', 10), ('f4', 30),   # "timepoint 1"
            ('f5', 10), ('f6', 30), ('f7', 0), ('f8', 20),   # "timepoint 2"
            ('f9', 30), ('f10', 20), ('f11', 0), ('f12', 10), # "timepoint 3"
        ]

        files = [f[0] for f in file_data]
        metadata_map = {f[0]: MockMetadata(f[1]) for f in file_data}

        def read_metadata(file_id):
            return metadata_map.get(file_id)

        filtered, num_timepoints = _filter_4d_generic(files, read_metadata)

        # Should detect ~3 timepoints
        assert num_timepoints == 3

        # Should return 4 files (one per unique position)
        assert len(filtered) == 4

        # Files should be sorted by z-position
        positions = [metadata_map[f].ImagePositionPatient[2] for f in filtered]
        assert positions == sorted(positions)
