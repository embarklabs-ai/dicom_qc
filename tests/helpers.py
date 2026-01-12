"""Shared test helper functions."""

import numpy as np
import SimpleITK as sitk


def create_sitk_image(
    shape: tuple = (20, 256, 256),
    spacing: tuple = (3.0, 1.0, 1.0),
    origin: tuple = (0.0, 0.0, 0.0),
    direction: tuple = None,
) -> sitk.Image:
    """
    Create a SimpleITK image for testing.

    Args:
        shape: Volume shape (slices, rows, cols)
        spacing: Voxel spacing (slice, row, col) in mm
        origin: Origin position (x, y, z)
        direction: 9-element direction cosine matrix (column-major).
                   If None, uses identity (axial orientation).

    Returns:
        SimpleITK Image
    """
    slices, rows, cols = shape
    # Create random data with some structure
    np.random.seed(42)
    data = np.random.rand(slices, rows, cols).astype(np.float32) * 1000

    # SimpleITK uses (x, y, z) = (cols, rows, slices)
    image = sitk.GetImageFromArray(data)
    image.SetSpacing((spacing[2], spacing[1], spacing[0]))  # (col, row, slice)
    image.SetOrigin(origin)

    if direction is None:
        # Identity matrix = standard axial orientation
        # Row direction = X, Col direction = Y, Slice direction = Z
        direction = (1, 0, 0, 0, 1, 0, 0, 0, 1)
    image.SetDirection(direction)

    return image
