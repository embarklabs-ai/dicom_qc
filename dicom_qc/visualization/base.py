"""Base class for DICOM visualization with proper LPS orientation and windowing."""

from typing import Dict, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

from dicom_qc.core.volume import DicomVolume


class VolumeRenderer:
    """Base class providing LPS-oriented slice extraction and display utilities."""

    # Standard radiological convention orientation labels
    ORIENTATION_LABELS = {
        "axial": {"left": "R", "right": "L", "top": "A", "bottom": "P"},
        "coronal": {"left": "R", "right": "L", "top": "S", "bottom": "I"},
        "sagittal": {"left": "A", "right": "P", "top": "S", "bottom": "I"},
    }

    def __init__(
        self, volume: DicomVolume, window: Optional[Tuple[float, float]] = None
    ):
        """
        Initialize with a DICOM volume.

        Args:
            volume: Loaded DicomVolume instance
            window: (center, width) for windowing, or None for auto
        """
        self.volume = volume
        self.window = window or volume.get_auto_window()

        # Get LPS-reoriented data using SimpleITK for correct anatomical planes
        self.lps_array, self.lps_spacing = volume.get_lps_array()
        # lps_array is indexed [S, P, L] so:
        #   lps_array[i, :, :] = axial slice at superior level i
        #   lps_array[:, j, :] = coronal slice at posterior level j
        #   lps_array[:, :, k] = sagittal slice at left level k
        # lps_spacing is (S, P, L) in mm

    def set_window(self, center: float, width: float) -> None:
        """Set window/level values."""
        self.window = (center, width)

    def apply_window(self, image: np.ndarray) -> np.ndarray:
        """Apply window/level to image data."""
        center, width = self.window
        low = center - width / 2
        high = center + width / 2
        windowed = np.clip(image, low, high)
        return ((windowed - low) / (high - low) * 255).astype(np.uint8)

    def get_num_slices(self, view: str) -> int:
        """Get number of slices for a view.

        After LPS reorientation, array is indexed as [S, P, L].

        Anatomical plane definitions:
        - Axial: perpendicular to Superior-Inferior axis, slice along S (axis 0)
        - Coronal: perpendicular to Anterior-Posterior axis, slice along P (axis 1)
        - Sagittal: perpendicular to Left-Right axis, slice along L (axis 2)
        """
        if view == "axial":
            return self.lps_array.shape[0]  # S axis
        elif view == "coronal":
            return self.lps_array.shape[1]  # P axis
        elif view == "sagittal":
            return self.lps_array.shape[2]  # L axis
        return 0

    def extract_slice(self, view: str, index: int) -> np.ndarray:
        """Extract a slice from the LPS-oriented volume.

        After LPS reorientation, array is [S, P, L]:
        - Axial slice at S=i: array[i, :, :] gives [P, L] view
        - Coronal slice at P=j: array[:, j, :] gives [S, L] view
        - Sagittal slice at L=k: array[:, :, k] gives [S, P] view
        """
        if view == "axial":
            return self.lps_array[index, :, :]  # [P, L]
        elif view == "coronal":
            return self.lps_array[:, index, :]  # [S, L]
        elif view == "sagittal":
            return self.lps_array[:, :, index]  # [S, P]
        raise ValueError(f"Unknown view: {view}")

    def calculate_aspect_ratio(self, view: str) -> float:
        """Calculate display aspect ratio for a view based on LPS spacing.

        lps_spacing is (S, P, L) corresponding to (axis0, axis1, axis2).
        For imshow, aspect = row_spacing / col_spacing.
        """
        s_spacing, p_spacing, l_spacing = self.lps_spacing

        if view == "axial":
            # Axial: array[i,:,:] gives [P, L] -> rows=P, cols=L
            return p_spacing / l_spacing
        elif view == "coronal":
            # Coronal: array[:,j,:] gives [S, L] -> rows=S, cols=L
            return s_spacing / l_spacing
        elif view == "sagittal":
            # Sagittal: array[:,:,k] gives [S, P] -> rows=S, cols=P
            return s_spacing / p_spacing
        return 1.0

    def get_image_origin(self, view: str) -> str:
        """Get matplotlib origin parameter for a view.

        Use origin='lower' for coronal/sagittal so inferior is at bottom.
        Axial uses origin='upper' (anterior at top in radiological convention).
        """
        return "upper" if view == "axial" else "lower"

    def get_orientation_labels(self, view: str) -> Dict[str, str]:
        """Get orientation labels for a view using standard radiological convention."""
        return self.ORIENTATION_LABELS.get(view, {})

    def add_orientation_labels(
        self, ax: plt.Axes, view: str, fontsize: int = 14
    ) -> None:
        """Add L/R, A/P, S/I labels to image edges."""
        labels = self.get_orientation_labels(view)

        props = dict(
            fontsize=fontsize,
            fontweight="bold",
            color="yellow",
            ha="center",
            va="center",
        )

        # Get axis limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Calculate label positions (slightly inside edges)
        margin = 0.05
        x_left = xlim[0] + (xlim[1] - xlim[0]) * margin
        x_right = xlim[1] - (xlim[1] - xlim[0]) * margin
        y_top = ylim[1] - (ylim[0] - ylim[1]) * margin
        y_bottom = ylim[0] + (ylim[0] - ylim[1]) * margin

        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2

        # Add labels
        if "left" in labels:
            ax.text(x_left, y_center, labels["left"], **props)
        if "right" in labels:
            ax.text(x_right, y_center, labels["right"], **props)
        if "top" in labels:
            ax.text(x_center, y_top, labels["top"], **props)
        if "bottom" in labels:
            ax.text(x_center, y_bottom, labels["bottom"], **props)

    def display_slice(
        self,
        ax: plt.Axes,
        view: str,
        slice_index: Optional[int] = None,
        show_labels: bool = True,
        label_fontsize: int = 14,
    ):
        """Display a slice on the given axes with proper orientation and aspect ratio.

        Args:
            ax: Matplotlib axes to draw on
            view: 'axial', 'coronal', or 'sagittal'
            slice_index: Slice index (None for center slice)
            show_labels: Whether to show orientation labels
            label_fontsize: Font size for orientation labels

        Returns:
            The AxesImage object
        """
        num_slices = self.get_num_slices(view)
        idx = slice_index if slice_index is not None else num_slices // 2

        img = self.apply_window(self.extract_slice(view, idx))
        aspect = self.calculate_aspect_ratio(view)
        origin = self.get_image_origin(view)

        im = ax.imshow(
            img, cmap="gray", aspect=aspect, origin=origin, interpolation="bilinear"
        )
        ax.axis("off")

        if show_labels:
            self.add_orientation_labels(ax, view, fontsize=label_fontsize)

        return im

    def generate_mip(self, plane: str = "sagittal") -> np.ndarray:
        """Generate Maximum Intensity Projection for a given anatomical plane.

        Args:
            plane: 'axial', 'coronal', or 'sagittal' - the plane to view (MIP along its normal)

        MIP collapses along the axis perpendicular to the view plane:
        - Axial MIP: collapse along S axis (0)
        - Coronal MIP: collapse along P axis (1)
        - Sagittal MIP: collapse along L axis (2)
        """
        axis = {"axial": 0, "coronal": 1, "sagittal": 2}.get(plane, 2)
        mip = np.max(self.lps_array, axis=axis)
        return self.apply_window(mip)
