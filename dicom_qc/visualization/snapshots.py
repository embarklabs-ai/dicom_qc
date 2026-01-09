"""Generate static PNG snapshots for QC review."""

from io import BytesIO
import base64
from typing import Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from dicom_qc.core.volume import DicomVolume
from dicom_qc.visualization.base import VolumeRenderer


class SnapshotGenerator(VolumeRenderer):
    """Generate static PNG snapshots for QC review."""

    def get_center_slices(self) -> Dict[str, np.ndarray]:
        """
        Get center slices in all three orthogonal planes.

        Returns:
            Dict with 'axial', 'sagittal', 'coronal' keys containing windowed images
        """
        result = {}
        for view in ['axial', 'coronal', 'sagittal']:
            num_slices = self.get_num_slices(view)
            center_idx = num_slices // 2
            result[view] = self.apply_window(self.extract_slice(view, center_idx))
        return result

    def create_single_view(
        self,
        view: str,
        slice_index: Optional[int] = None,
        figsize: Tuple[int, int] = (6, 6),
        show_labels: bool = True,
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Create a single view image.

        Args:
            view: 'axial', 'coronal', 'sagittal', or 'mip'
            slice_index: Slice index (None for center slice)
            figsize: Figure size
            show_labels: Whether to show orientation labels
            save_path: Optional path to save image

        Returns:
            matplotlib Figure
        """
        # Disable interactive mode to prevent auto-display
        was_interactive = plt.isinteractive()
        plt.ioff()

        fig, ax = plt.subplots(figsize=figsize)

        if view == 'mip':
            img = self.generate_mip(plane='sagittal')
            aspect = self.calculate_aspect_ratio('sagittal')
            origin = self.get_image_origin('sagittal')
            title = 'Maximum Intensity Projection'
            label_view = 'sagittal'
        elif view in ('axial', 'coronal', 'sagittal'):
            num_slices = self.get_num_slices(view)
            idx = slice_index if slice_index is not None else num_slices // 2
            img = self.apply_window(self.extract_slice(view, idx))
            aspect = self.calculate_aspect_ratio(view)
            origin = self.get_image_origin(view)
            title = f'{view.capitalize()} (Slice {idx + 1}/{num_slices})'
            label_view = view
        else:
            raise ValueError(f"Unknown view: {view}")

        ax.imshow(img, cmap='gray', aspect=aspect, origin=origin, interpolation='bilinear')
        ax.set_title(title, fontsize=12)
        ax.axis('off')

        if show_labels:
            self.add_orientation_labels(ax, label_view)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='black')

        # Restore interactive mode
        if was_interactive:
            plt.ion()

        return fig

    def create_orientation_figure(
        self,
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Create multi-panel figure with center slices and orientation labels.

        Layout:
        [Axial    ] [Coronal  ]
        [Sagittal ] [MIP      ]

        Each panel includes:
        - Orientation labels (L/R, A/P, S/I)
        - Slice location info
        - Proper aspect ratio

        Args:
            figsize: Figure size
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure
        """
        slices = self.get_center_slices()
        mip = self.generate_mip(plane='sagittal')

        # Disable interactive mode to prevent auto-display
        was_interactive = plt.isinteractive()
        plt.ioff()

        fig, axes = plt.subplots(2, 2, figsize=figsize, facecolor='black')
        fig.suptitle(
            f'{self.volume.modality}: {self.volume.series_description}',
            color='white',
            fontsize=14
        )

        # Standard layout: Axial, Coronal, Sagittal, MIP
        panels = [
            ('Axial', slices['axial'], axes[0, 0], 'axial'),
            ('Coronal', slices['coronal'], axes[0, 1], 'coronal'),
            ('Sagittal', slices['sagittal'], axes[1, 0], 'sagittal'),
            ('MIP', mip, axes[1, 1], 'sagittal'),
        ]

        for title, img, ax, label_view in panels:
            aspect = self.calculate_aspect_ratio(label_view)
            origin = self.get_image_origin(label_view)

            ax.imshow(img, cmap='gray', aspect=aspect, origin=origin, interpolation='bilinear')
            ax.set_facecolor('black')
            ax.axis('off')

            # Add title with slice info
            if title == 'MIP':
                subtitle = 'Through slices'
            else:
                num_slices = self.get_num_slices(title.lower())
                subtitle = f'Slice {num_slices // 2 + 1}/{num_slices}'

            ax.set_title(f'{title}\n{subtitle}', color='white', fontsize=11)

            # Add orientation labels
            self.add_orientation_labels(ax, label_view)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='black')

        # Restore interactive mode
        if was_interactive:
            plt.ion()

        return fig

    def to_base64(self, fig: Figure, format: str = 'png') -> str:
        """Convert matplotlib figure to base64 encoded string."""
        buffer = BytesIO()
        fig.savefig(
            buffer,
            format=format,
            dpi=150,
            bbox_inches='tight',
            facecolor='black'
        )
        buffer.seek(0)
        encoded = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
        return encoded

    def create_tripane_thumbnail(
        self,
        figsize: Tuple[float, float] = (4.5, 1.5),
        dpi: int = 75,
        show_labels: bool = True
    ) -> str:
        """
        Create a compact 3-pane thumbnail (axial, coronal, sagittal) as base64.

        Args:
            figsize: Figure size in inches (width, height)
            dpi: Resolution
            show_labels: Whether to show A/C/S labels

        Returns:
            Base64 encoded PNG string
        """
        was_interactive = plt.isinteractive()
        plt.ioff()

        views = ['axial', 'coronal', 'sagittal']

        fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
        fig.patch.set_facecolor('black')

        for ax, view in zip(axes, views):
            num_slices = self.get_num_slices(view)
            center_idx = num_slices // 2
            img = self.apply_window(self.extract_slice(view, center_idx))
            aspect = self.calculate_aspect_ratio(view)
            origin = self.get_image_origin(view)

            ax.imshow(img, cmap='gray', aspect=aspect, origin=origin, interpolation='bilinear')
            ax.axis('off')
            ax.set_facecolor('black')
            if show_labels:
                ax.text(0.5, 0.02, view[0].upper(), transform=ax.transAxes,
                       fontsize=8, color='white', ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.5))

        plt.subplots_adjust(wspace=0.02, left=0, right=1, top=1, bottom=0)

        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.02, facecolor='black')
        plt.close(fig)
        buf.seek(0)

        if was_interactive:
            plt.ion()

        return base64.b64encode(buf.read()).decode('utf-8')

    def generate_all_snapshots(self) -> Dict[str, str]:
        """
        Generate all standard snapshots as base64 strings.

        Returns:
            Dict with keys 'axial', 'coronal', 'sagittal', 'mip', 'overview'
            Views that can't be generated (e.g., single-slice in that plane) will be None
        """
        snapshots = {}

        # Individual views - skip views that would be degenerate (single slice)
        for view in ['axial', 'coronal', 'sagittal', 'mip']:
            if view == 'mip':
                # MIP needs at least 2 slices in sagittal direction
                if self.get_num_slices('sagittal') < 2:
                    snapshots[view] = None
                    continue
            else:
                # Check if this view has at least 2 slices
                if self.get_num_slices(view) < 2:
                    snapshots[view] = None
                    continue

            try:
                fig = self.create_single_view(view, figsize=(6, 6))
                snapshots[view] = self.to_base64(fig)
            except Exception:
                snapshots[view] = None

        # Overview figure
        try:
            fig = self.create_orientation_figure()
            snapshots['overview'] = self.to_base64(fig)
        except Exception:
            snapshots['overview'] = None

        return snapshots
