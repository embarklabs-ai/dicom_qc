"""Interactive DICOM viewer with windowing and slice navigation."""

from io import BytesIO
from typing import Dict, Tuple
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np

from dicom_qc.core.volume import DicomVolume


class InteractiveViewer:
    """Interactive DICOM viewer with windowing and slice navigation."""

    # Window/level presets (center, width)
    PRESETS = {
        'Auto': None,  # Will be calculated
        'Bone': (300, 1500),
        'Soft Tissue': (40, 400),
        'Lung': (-600, 1500),
        'Brain': (40, 80),
        'Liver': (60, 150),
        'Mediastinum': (50, 350),
    }

    # Standard radiological convention orientation labels
    ORIENTATION_LABELS = {
        'axial': {'left': 'R', 'right': 'L', 'top': 'A', 'bottom': 'P'},
        'coronal': {'left': 'R', 'right': 'L', 'top': 'S', 'bottom': 'I'},
        'sagittal': {'left': 'A', 'right': 'P', 'top': 'S', 'bottom': 'I'},
    }

    def __init__(self, volume: DicomVolume):
        """
        Initialize with a loaded DICOM volume.

        Args:
            volume: DicomVolume instance
        """
        self.volume = volume

        # Get LPS-reoriented data using SimpleITK for correct anatomical planes
        self.lps_array, self.lps_spacing = volume.get_lps_array()
        # lps_array is indexed [S, P, L] so:
        #   lps_array[i, :, :] = axial slice at superior level i
        #   lps_array[:, j, :] = coronal slice at posterior level j
        #   lps_array[:, :, k] = sagittal slice at left level k
        # lps_spacing is (S, P, L) in mm

        # Start with axial view (most common)
        self.current_view = 'axial'

        # Calculate initial window
        data = volume.pixel_array
        self.auto_window = volume.get_auto_window()
        self.window_center = self.auto_window[0]
        self.window_width = self.auto_window[1]

        # Store presets with auto calculated
        self.PRESETS['Auto'] = self.auto_window

        # Determine intensity range for sliders
        self.min_intensity = float(data.min())
        self.max_intensity = float(data.max())

        # Figure and axes for stable display (created on first display)
        self._fig = None
        self._ax = None
        self._im = None
        self._label_texts = {}

        # Create widgets
        self._create_widgets()

        # Wire up observers
        self._setup_observers()

    def _get_num_slices(self, view: str) -> int:
        """Get number of slices for a view.

        After LPS reorientation, SimpleITK GetArrayFromImage returns array indexed as [z,y,x].
        For LPS orientation: x=L (Left), y=P (Posterior), z=S (Superior).
        So the array is indexed as [S, P, L].

        Anatomical plane definitions:
        - Axial: perpendicular to Superior-Inferior axis, so we slice along S (axis 0)
        - Coronal: perpendicular to Anterior-Posterior axis, so we slice along P (axis 1)
        - Sagittal: perpendicular to Left-Right axis, so we slice along L (axis 2)
        """
        if view == 'axial':
            return self.lps_array.shape[0]  # S axis
        elif view == 'coronal':
            return self.lps_array.shape[1]  # P axis
        elif view == 'sagittal':
            return self.lps_array.shape[2]  # L axis
        return 0

    def _extract_slice(self, view: str, index: int) -> np.ndarray:
        """Extract a slice from the LPS-oriented volume.

        After LPS reorientation, array is [S, P, L]:
        - Axial slice at S=i: array[i, :, :] gives [P, L] view
        - Coronal slice at P=j: array[:, j, :] gives [S, L] view
        - Sagittal slice at L=k: array[:, :, k] gives [S, P] view
        """
        if view == 'axial':
            return self.lps_array[index, :, :]  # [P, L]
        elif view == 'coronal':
            return self.lps_array[:, index, :]  # [S, L]
        elif view == 'sagittal':
            return self.lps_array[:, :, index]  # [S, P]
        raise ValueError(f"Unknown view: {view}")

    def _create_widgets(self):
        """Create all widgets."""
        # Get initial slice count for axial view
        num_slices = self._get_num_slices('axial')

        # View selection - standard anatomical views
        view_options = [
            ('Axial', 'axial'),
            ('Coronal', 'coronal'),
            ('Sagittal', 'sagittal'),
        ]

        self.view_dropdown = widgets.Dropdown(
            options=view_options,
            value='axial',
            description='View:',
            style={'description_width': '60px'},
            layout=widgets.Layout(width='180px')
        )

        # Slice slider
        self.slice_slider = widgets.IntSlider(
            value=num_slices // 2,
            min=0,
            max=num_slices - 1,
            step=1,
            description='Slice:',
            continuous_update=True,
            layout=widgets.Layout(width='400px'),
            style={'description_width': '60px'}
        )

        # Slice play button
        self.play_button = widgets.Play(
            value=num_slices // 2,
            min=0,
            max=num_slices - 1,
            step=1,
            interval=100,
            description='Play',
        )
        widgets.jslink((self.play_button, 'value'), (self.slice_slider, 'value'))

        # Window/level sliders
        self.window_center_slider = widgets.FloatSlider(
            value=self.window_center,
            min=self.min_intensity,
            max=self.max_intensity,
            step=1.0,
            description='Center:',
            continuous_update=True,
            layout=widgets.Layout(width='350px'),
            style={'description_width': '60px'}
        )

        self.window_width_slider = widgets.FloatSlider(
            value=self.window_width,
            min=1.0,
            max=self.max_intensity - self.min_intensity,
            step=1.0,
            description='Width:',
            continuous_update=True,
            layout=widgets.Layout(width='350px'),
            style={'description_width': '60px'}
        )

        # Preset buttons
        preset_buttons = []
        for name in ['Auto', 'Bone', 'Soft Tissue', 'Lung', 'Brain']:
            btn = widgets.Button(
                description=name,
                layout=widgets.Layout(width='auto'),
                button_style='' if name != 'Auto' else 'info'
            )
            btn.on_click(self._make_preset_handler(name))
            preset_buttons.append(btn)

        self.preset_buttons = widgets.HBox(preset_buttons)

        # Image widget (updates in-place without flicker)
        self.image_widget = widgets.Image(
            format='png',
            layout=widgets.Layout(width='500px', height='500px')
        )

        # Info display
        self.info_output = widgets.HTML()

    def _setup_observers(self):
        """Set up widget observers."""
        self.view_dropdown.observe(self._on_view_change, names='value')
        self.slice_slider.observe(self._on_slice_change, names='value')
        self.window_center_slider.observe(self._on_window_change, names='value')
        self.window_width_slider.observe(self._on_window_change, names='value')

    def _make_preset_handler(self, preset_name: str):
        """Create a handler for preset button."""
        def handler(button):
            self._apply_preset(preset_name)
        return handler

    def _apply_preset(self, preset_name: str):
        """Apply window/level preset."""
        preset = self.PRESETS.get(preset_name)
        if preset:
            self.window_center_slider.value = preset[0]
            self.window_width_slider.value = preset[1]

    def display(self):
        """Display the viewer widget."""
        header = widgets.HTML(f'''
            <h3 style="margin-bottom: 5px;">Interactive Viewer</h3>
            <p style="color: #666; margin: 0;">
                {self.volume.modality}: {self.volume.series_description}
            </p>
        ''')

        # Slice controls
        slice_controls = widgets.HBox([
            self.view_dropdown,
            self.play_button,
            self.slice_slider,
        ])

        # Window controls
        window_label = widgets.HTML('<b>Window/Level:</b>')
        window_controls = widgets.VBox([
            window_label,
            widgets.HBox([self.window_center_slider, self.window_width_slider]),
            self.preset_buttons,
        ])

        # Layout: controls on left, image on right
        controls = widgets.VBox([
            header,
            slice_controls,
            window_controls,
            self.info_output,
        ], layout=widgets.Layout(width='450px', padding='10px'))

        layout = widgets.HBox([
            controls,
            self.image_widget
        ])

        display(layout)
        self._update_display()

    def _on_view_change(self, change):
        """Handle view direction change."""
        view = change['new']
        self.current_view = view

        # Update slice slider range based on LPS array dimensions
        max_slice = self._get_num_slices(view) - 1
        self.slice_slider.max = max_slice
        self.slice_slider.value = max_slice // 2

        # Update play button
        self.play_button.max = max_slice
        self.play_button.value = max_slice // 2

        self._update_display()

    def _on_slice_change(self, change):
        """Handle slice selection change."""
        self._update_display()

    def _on_window_change(self, change):
        """Handle window/level change."""
        self.window_center = self.window_center_slider.value
        self.window_width = self.window_width_slider.value
        self._update_display()

    def _get_orientation_labels(self) -> Dict[str, str]:
        """Get orientation labels for current view using standard radiological convention."""
        return self.ORIENTATION_LABELS.get(self.current_view, {})

    def _calculate_aspect_ratio(self) -> float:
        """Calculate display aspect ratio for current view based on LPS spacing.

        lps_spacing is (S, P, L) corresponding to (axis0, axis1, axis2).
        For imshow, aspect = row_spacing / col_spacing.
        """
        s_spacing, p_spacing, l_spacing = self.lps_spacing

        if self.current_view == 'axial':
            # Axial: array[i,:,:] gives [P, L] -> rows=P, cols=L
            return p_spacing / l_spacing
        elif self.current_view == 'coronal':
            # Coronal: array[:,j,:] gives [S, L] -> rows=S, cols=L
            return s_spacing / l_spacing
        else:  # sagittal
            # Sagittal: array[:,:,k] gives [S, P] -> rows=S, cols=P
            return s_spacing / p_spacing

    def _update_display(self):
        """Update the displayed image."""
        slice_idx = self.slice_slider.value
        view = self.current_view

        # Extract slice from LPS-reoriented volume
        img = self._extract_slice(view, slice_idx)

        # Apply windowing
        low = self.window_center - self.window_width / 2
        high = self.window_center + self.window_width / 2
        windowed = np.clip(img, low, high)
        windowed = ((windowed - low) / (high - low) * 255).astype(np.uint8)

        # Calculate aspect ratio
        aspect = self._calculate_aspect_ratio()

        # Check if we need a new figure (view changed)
        # Always recreate when view changes to update orientation labels
        last_view = getattr(self, '_last_view', None)
        needs_new_figure = (
            self._fig is None or
            self._im is None or
            self._im.get_array().shape != windowed.shape or
            last_view != view
        )
        self._last_view = view

        if needs_new_figure:
            # Close old figure if exists
            if self._fig is not None:
                plt.close(self._fig)

            # Disable interactive mode to prevent auto-display
            was_interactive = plt.isinteractive()
            plt.ioff()

            # Create figure without displaying it
            self._fig = plt.figure(figsize=(7, 7), facecolor='black')
            self._ax = self._fig.add_subplot(111)
            self._ax.set_facecolor('black')

            # Use origin='lower' for coronal/sagittal so inferior is at bottom
            # Axial uses default origin='upper' (anterior at top in radiological convention)
            img_origin = 'upper' if view == 'axial' else 'lower'
            self._im = self._ax.imshow(
                windowed, cmap='gray', aspect=aspect, origin=img_origin,
                interpolation='bilinear'  # Smooth display like medical viewers
            )

            self._ax.axis('off')

            # Add orientation labels
            labels = self._get_orientation_labels()
            self._setup_orientation_labels(self._ax, labels)

            self._fig.tight_layout(pad=0.5)

            # Restore interactive mode if it was on
            if was_interactive:
                plt.ion()
        else:
            # Just update the image data
            self._im.set_array(windowed)

        # Render figure to PNG bytes and update image widget
        buf = BytesIO()
        self._fig.savefig(buf, format='png', facecolor='black', bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)
        self.image_widget.value = buf.read()

        # Update info
        self._update_info()

    def _setup_orientation_labels(self, ax, labels: Dict[str, str]):
        """Set up orientation labels on axes (called once per view change)."""
        props = dict(fontsize=14, fontweight='bold', color='yellow')

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        margin = 0.05
        x_left = xlim[0] + (xlim[1] - xlim[0]) * margin
        x_right = xlim[1] - (xlim[1] - xlim[0]) * margin
        y_top = ylim[1] - (ylim[0] - ylim[1]) * margin
        y_bottom = ylim[0] + (ylim[0] - ylim[1]) * margin

        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2

        # Store text objects for potential updates
        self._label_texts = {}
        if 'left' in labels:
            self._label_texts['left'] = ax.text(x_left, y_center, labels['left'], ha='left', va='center', **props)
        if 'right' in labels:
            self._label_texts['right'] = ax.text(x_right, y_center, labels['right'], ha='right', va='center', **props)
        if 'top' in labels:
            self._label_texts['top'] = ax.text(x_center, y_top, labels['top'], ha='center', va='top', **props)
        if 'bottom' in labels:
            self._label_texts['bottom'] = ax.text(x_center, y_bottom, labels['bottom'], ha='center', va='bottom', **props)

    def _update_info(self):
        """Update the info display."""
        view = self.current_view
        max_slice = self._get_num_slices(view)

        self.info_output.value = f'''
            <div style="background: #f8f9fa; padding: 10px; border-radius: 4px; margin-top: 10px;">
                <b>{view.capitalize()} View</b><br>
                Slice: {self.slice_slider.value + 1} / {max_slice}<br>
                Window: C={self.window_center:.0f}, W={self.window_width:.0f}<br>
                Voxel: {self.volume.voxel_spacing[0]:.2f} x {self.volume.voxel_spacing[1]:.2f} x {self.volume.voxel_spacing[2]:.2f} mm
            </div>
        '''

    def get_current_window(self) -> Tuple[float, float]:
        """Get current window/level values."""
        return (self.window_center, self.window_width)

    def set_window(self, center: float, width: float):
        """Set window/level values."""
        self.window_center_slider.value = center
        self.window_width_slider.value = width
