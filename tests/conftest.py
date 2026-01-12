"""Pytest fixtures for dicom_qc tests."""

import numpy as np
import pytest
import SimpleITK as sitk

from dicom_qc.core.volume import DicomVolume

from tests.helpers import create_sitk_image


@pytest.fixture
def axial_volume() -> DicomVolume:
    """Create a standard axial DicomVolume for testing."""
    image = create_sitk_image(
        shape=(20, 256, 256),
        spacing=(3.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
    )
    return DicomVolume(
        sitk_image=image,
        modality='MR',
        series_description='T1 MPRAGE',
    )


@pytest.fixture
def coronal_volume() -> DicomVolume:
    """Create a coronal DicomVolume for testing."""
    # Coronal: slice normal along Y (A-P axis)
    # Row direction = X (L-R), Col direction = Z (S-I)
    direction = (1, 0, 0, 0, 0, 1, 0, 1, 0)
    image = create_sitk_image(
        shape=(20, 256, 256),
        spacing=(3.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        direction=direction,
    )
    return DicomVolume(
        sitk_image=image,
        modality='MR',
        series_description='Coronal T2',
    )


@pytest.fixture
def sagittal_volume() -> DicomVolume:
    """Create a sagittal DicomVolume for testing."""
    # Sagittal: slice normal along X (L-R axis)
    # row_dir = (0, 1, 0) = Y axis, col_dir = (0, 0, 1) = Z axis
    # cross(Y, Z) = X = SAGITTAL
    direction = (0, 0, 1, 1, 0, 0, 0, 1, 0)
    image = create_sitk_image(
        shape=(20, 256, 256),
        spacing=(3.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        direction=direction,
    )
    return DicomVolume(
        sitk_image=image,
        modality='MR',
        series_description='Sagittal FLAIR',
    )


@pytest.fixture
def oblique_volume() -> DicomVolume:
    """Create an oblique DicomVolume for testing."""
    # 45-degree oblique: neither axial, coronal, nor sagittal
    angle = np.pi / 4  # 45 degrees
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    # Rotate around X axis
    direction = (1, 0, 0, 0, cos_a, sin_a, 0, -sin_a, cos_a)
    image = create_sitk_image(
        shape=(20, 256, 256),
        spacing=(3.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        direction=direction,
    )
    return DicomVolume(
        sitk_image=image,
        modality='MR',
        series_description='Oblique',
    )


@pytest.fixture
def single_slice_volume() -> DicomVolume:
    """Create a single-slice DicomVolume for edge case testing."""
    image = create_sitk_image(
        shape=(1, 256, 256),
        spacing=(5.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
    )
    return DicomVolume(
        sitk_image=image,
        modality='MR',
        series_description='Single slice',
    )


@pytest.fixture
def anisotropic_volume() -> DicomVolume:
    """Create a highly anisotropic DicomVolume (thick slices)."""
    image = create_sitk_image(
        shape=(10, 256, 256),
        spacing=(10.0, 0.5, 0.5),  # 10mm slices, 0.5mm in-plane = 20x ratio
        origin=(0.0, 0.0, 0.0),
    )
    return DicomVolume(
        sitk_image=image,
        modality='MR',
        series_description='Thick slices',
    )


@pytest.fixture
def isotropic_volume() -> DicomVolume:
    """Create an isotropic DicomVolume."""
    image = create_sitk_image(
        shape=(128, 128, 128),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
    )
    return DicomVolume(
        sitk_image=image,
        modality='MR',
        series_description='Isotropic',
    )


@pytest.fixture
def ct_volume() -> DicomVolume:
    """Create a CT DicomVolume for testing CT-specific behavior."""
    slices, rows, cols = 50, 512, 512
    np.random.seed(42)
    # CT-like values: bone ~1000 HU, soft tissue ~40 HU, air ~-1000 HU
    data = np.random.randint(-1000, 1000, size=(slices, rows, cols)).astype(np.float32)

    image = sitk.GetImageFromArray(data)
    image.SetSpacing((0.5, 0.5, 2.0))  # col, row, slice
    image.SetOrigin((0.0, 0.0, 0.0))
    image.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

    return DicomVolume(
        sitk_image=image,
        modality='CT',
        series_description='CT Chest',
    )


@pytest.fixture
def volume_4d() -> DicomVolume:
    """Create a 4D volume (fMRI/DTI) for testing."""
    image = create_sitk_image(
        shape=(30, 64, 64),
        spacing=(4.0, 3.0, 3.0),
        origin=(0.0, 0.0, 0.0),
    )
    return DicomVolume(
        sitk_image=image,
        modality='MR',
        series_description='fMRI BOLD',
        num_timepoints=100,  # 4D data with 100 timepoints
    )


@pytest.fixture
def localizer_volume() -> DicomVolume:
    """Create a localizer/scout volume with multi-orientation slices."""
    # Few slices with large spacing (typical of scout/localizer)
    image = create_sitk_image(
        shape=(3, 256, 256),
        spacing=(100.0, 1.0, 1.0),  # Very large inter-slice spacing
        origin=(0.0, 0.0, 0.0),
    )
    return DicomVolume(
        sitk_image=image,
        modality='MR',
        series_description='Localizer',
        num_orientations=3,  # Multi-orientation
    )


@pytest.fixture
def missing_geometry_volume() -> DicomVolume:
    """Create a volume with missing geometry tags."""
    image = create_sitk_image(
        shape=(20, 256, 256),
        spacing=(3.0, 1.0, 1.0),
    )
    return DicomVolume(
        sitk_image=image,
        modality='MR',
        series_description='Derived ADC',
        missing_geometry_tags=['PixelSpacing', 'ImageOrientationPatient'],
    )


@pytest.fixture
def dti_volume() -> DicomVolume:
    """Create a DTI volume for series type detection."""
    image = create_sitk_image(
        shape=(30, 128, 128),
        spacing=(3.0, 2.0, 2.0),
    )
    return DicomVolume(
        sitk_image=image,
        modality='MR',
        series_description='DTI_64dir_b1000',
        num_timepoints=65,
    )


@pytest.fixture
def perfusion_volume() -> DicomVolume:
    """Create a DSC perfusion volume for series type detection."""
    image = create_sitk_image(
        shape=(20, 128, 128),
        spacing=(5.0, 2.0, 2.0),
    )
    return DicomVolume(
        sitk_image=image,
        modality='MR',
        series_description='DSC Perfusion CBV',
        num_timepoints=60,
    )
