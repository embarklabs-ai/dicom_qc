"""Generate animated GIF/MP4 movies for slice scrolling."""

import base64
import os
import tempfile
from typing import Tuple, Optional, Dict

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from dicom_qc.visualization.base import VolumeRenderer


class AnimationGenerator(VolumeRenderer):
    """Generate animated GIF/MP4 movies for slice scrolling."""

    def create_slice_animation(
        self,
        view: str = "axial",
        fps: int = 10,
        output_path: Optional[str] = None,
        figsize: Tuple[float, float] = (6, 6),
        show_labels: bool = True,
    ) -> Optional[bytes]:
        """
        Create animated slice-by-slice scrolling through volume.

        Args:
            view: 'axial', 'coronal', or 'sagittal'
            fps: Frames per second
            output_path: Optional path to save file (must end in .gif)
            figsize: Figure size
            show_labels: Whether to show orientation labels

        Returns:
            Bytes of the GIF animation if output_path is None, else None
        """
        num_slices = self.get_num_slices(view)
        aspect = self.calculate_aspect_ratio(view)
        origin = self.get_image_origin(view)

        # Create figure with black background
        fig, ax = plt.subplots(figsize=figsize, facecolor="black")
        ax.set_facecolor("black")
        ax.axis("off")

        # Get first frame
        first_slice = self.extract_slice(view, 0)
        im = ax.imshow(
            self.apply_window(first_slice),
            cmap="gray",
            aspect=aspect,
            origin=origin,
            interpolation="bilinear",
        )

        # Add slice counter text
        slice_text = ax.text(
            0.02,
            0.98,
            "",
            transform=ax.transAxes,
            fontsize=12,
            color="yellow",
            verticalalignment="top",
            fontweight="bold",
        )

        # Add title
        ax.text(
            0.5,
            0.02,
            f"{view.capitalize()} View",
            transform=ax.transAxes,
            fontsize=12,
            color="white",
            ha="center",
            va="bottom",
        )

        # Add orientation labels (static)
        if show_labels:
            self.add_orientation_labels(ax, view, fontsize=12)

        fig.tight_layout(pad=0.5)

        def animate(frame):
            slice_data = self.extract_slice(view, frame)
            im.set_array(self.apply_window(slice_data))
            slice_text.set_text(f"Slice: {frame + 1}/{num_slices}")
            return [im, slice_text]

        anim = FuncAnimation(
            fig, animate, frames=num_slices, interval=1000 // fps, blit=True
        )

        return self._save_animation(fig, anim, fps, output_path)

    def create_acquisition_plane_animation(
        self,
        fps: int = 10,
        output_path: Optional[str] = None,
        figsize: Tuple[float, float] = (6, 6),
        show_labels: bool = True,
    ) -> Optional[bytes]:
        """
        Create animation in the original acquisition plane (not reoriented).

        This scrolls through slices as they were acquired, which is useful
        for checking the raw data before LPS reorientation.

        Args:
            fps: Frames per second
            output_path: Optional path to save file
            figsize: Figure size
            show_labels: Whether to show orientation labels

        Returns:
            Bytes of the GIF animation if output_path is None
        """
        # Use raw pixel array (acquisition order)
        data = self.volume.pixel_array
        num_slices = data.shape[0]
        row_spacing, col_spacing = self.volume.pixel_spacing
        aspect = row_spacing / col_spacing

        # Get acquisition orientation labels from DICOM
        from dicom_qc.core.geometry import GeometryQC

        geometry_qc = GeometryQC(self.volume)
        acq_labels = geometry_qc.get_acquisition_labels()
        plane = geometry_qc.run_all_checks().primary_plane

        # Create figure with black background
        fig, ax = plt.subplots(figsize=figsize, facecolor="black")
        ax.set_facecolor("black")
        ax.axis("off")

        # Get first frame
        first_slice = data[0, :, :]
        im = ax.imshow(
            self.apply_window(first_slice),
            cmap="gray",
            aspect=aspect,
            origin="upper",
            interpolation="bilinear",
        )

        # Add slice counter text
        slice_text = ax.text(
            0.02,
            0.98,
            "",
            transform=ax.transAxes,
            fontsize=12,
            color="yellow",
            verticalalignment="top",
            fontweight="bold",
        )

        # Add title
        ax.text(
            0.5,
            0.02,
            f"Acquisition Plane ({plane})",
            transform=ax.transAxes,
            fontsize=12,
            color="white",
            ha="center",
            va="bottom",
        )

        # Add orientation labels from DICOM
        if show_labels:
            self._add_acquisition_labels(ax, acq_labels)

        fig.tight_layout(pad=0.5)

        def animate(frame):
            slice_data = data[frame, :, :]
            im.set_array(self.apply_window(slice_data))
            slice_text.set_text(f"Slice: {frame + 1}/{num_slices}")
            return [im, slice_text]

        anim = FuncAnimation(
            fig, animate, frames=num_slices, interval=1000 // fps, blit=True
        )

        return self._save_animation(fig, anim, fps, output_path)

    def _add_acquisition_labels(self, ax: plt.Axes, acq_labels: Dict[str, str]) -> None:
        """Add acquisition plane orientation labels to axes."""
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        margin = 0.08
        x_left = xlim[0] + (xlim[1] - xlim[0]) * margin
        x_right = xlim[1] - (xlim[1] - xlim[0]) * margin
        y_top = ylim[1] - (ylim[0] - ylim[1]) * margin
        y_bottom = ylim[0] + (ylim[0] - ylim[1]) * margin
        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2

        props = dict(
            fontsize=12, fontweight="bold", color="yellow", ha="center", va="center"
        )
        ax.text(x_left, y_center, acq_labels.get("image_left", ""), **props)
        ax.text(x_right, y_center, acq_labels.get("image_right", ""), **props)
        ax.text(x_center, y_top, acq_labels.get("image_top", ""), **props)
        ax.text(x_center, y_bottom, acq_labels.get("image_bottom", ""), **props)

    def create_three_plane_animation(
        self,
        fps: int = 8,
        output_path: Optional[str] = None,
        figsize: Tuple[float, float] = (12, 4),
    ) -> Optional[bytes]:
        """
        Create synchronized animation showing all three anatomical planes.

        Args:
            fps: Frames per second
            output_path: Optional path to save file
            figsize: Figure size

        Returns:
            Bytes of the GIF animation if output_path is None
        """
        views = ["axial", "coronal", "sagittal"]
        num_slices = {view: self.get_num_slices(view) for view in views}

        # Use the minimum dimension for synchronized scrolling
        num_frames = min(num_slices.values())

        fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor="black")

        images = []
        texts = []

        for idx, view in enumerate(views):
            ax = axes[idx]
            ax.set_facecolor("black")
            ax.axis("off")

            aspect = self.calculate_aspect_ratio(view)
            origin = self.get_image_origin(view)

            first_slice = self.extract_slice(view, 0)
            im = ax.imshow(
                self.apply_window(first_slice),
                cmap="gray",
                aspect=aspect,
                origin=origin,
                interpolation="bilinear",
            )
            images.append(im)

            # Slice counter
            text = ax.text(
                0.02,
                0.98,
                "",
                transform=ax.transAxes,
                fontsize=10,
                color="yellow",
                verticalalignment="top",
                fontweight="bold",
            )
            texts.append(text)

            # View title
            ax.set_title(view.capitalize(), color="white", fontsize=11)

            # Orientation labels
            self.add_orientation_labels(ax, view, fontsize=10)

        fig.tight_layout(pad=0.5)

        def animate(frame):
            updated = []
            for idx, view in enumerate(views):
                max_slices = num_slices[view]
                # Scale frame to each axis's range
                slice_idx = int(frame * max_slices / num_frames)
                slice_idx = min(slice_idx, max_slices - 1)

                slice_data = self.extract_slice(view, slice_idx)
                images[idx].set_array(self.apply_window(slice_data))
                texts[idx].set_text(f"{slice_idx + 1}/{max_slices}")
                updated.extend([images[idx], texts[idx]])

            return updated

        anim = FuncAnimation(
            fig, animate, frames=num_frames, interval=1000 // fps, blit=True
        )

        return self._save_animation(fig, anim, fps, output_path)

    def _save_animation(
        self, fig: plt.Figure, anim: FuncAnimation, fps: int, output_path: Optional[str]
    ) -> Optional[bytes]:
        """Save animation to file or return bytes."""
        writer = PillowWriter(fps=fps)

        if output_path:
            anim.save(output_path, writer=writer)
            plt.close(fig)
            return None
        else:
            with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                anim.save(
                    tmp_path, writer=writer, savefig_kwargs={"facecolor": "black"}
                )
                with open(tmp_path, "rb") as f:
                    result = f.read()
            finally:
                os.unlink(tmp_path)
            plt.close(fig)
            return result

    def to_base64(self, animation_bytes: bytes) -> str:
        """Convert animation bytes to base64 for HTML embedding."""
        return base64.b64encode(animation_bytes).decode("utf-8")

    def generate_all_animations(self, fps: int = 10) -> Dict[str, str]:
        """
        Generate all standard animations as base64 strings.

        Args:
            fps: Frames per second

        Returns:
            Dict with keys 'axial', 'coronal', 'sagittal', 'acquisition', 'three_plane'
        """
        animations = {}

        # Anatomical plane animations (LPS-reoriented)
        for view in ["axial", "coronal", "sagittal"]:
            try:
                gif_bytes = self.create_slice_animation(view=view, fps=fps)
                if gif_bytes:
                    animations[view] = self.to_base64(gif_bytes)
            except Exception:
                pass

        # Acquisition plane animation
        try:
            gif_bytes = self.create_acquisition_plane_animation(fps=fps)
            if gif_bytes:
                animations["acquisition"] = self.to_base64(gif_bytes)
        except Exception:
            pass

        # Three-plane animation
        try:
            gif_bytes = self.create_three_plane_animation(fps=fps)
            if gif_bytes:
                animations["three_plane"] = self.to_base64(gif_bytes)
        except Exception:
            pass

        return animations
