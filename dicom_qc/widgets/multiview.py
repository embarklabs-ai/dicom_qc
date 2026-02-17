"""Multi-view DICOM viewer with synchronized crosshair navigation."""

from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import numpy as np

from dicom_qc.core.volume import DicomVolume
from dicom_qc.visualization.base import VolumeRenderer


def _read_dicom_for_header(file_path):
    """Read DICOM file for header viewing (no pixel data).

    Handles both local Path/string paths and XNAT file objects correctly.
    Path.open() defaults to text mode which breaks pydicom, so we pass
    Path/string directly to pydicom which opens in binary mode.

    Args:
        file_path: Path to DICOM file, string path, or XNAT file object

    Returns:
        pydicom Dataset
    """
    import pydicom

    # XNAT file objects have open() but are not Path - use their open() method
    # Path objects also have open() but default to text mode, so pass to pydicom directly
    if hasattr(file_path, 'open') and not isinstance(file_path, Path):
        with file_path.open() as f:
            return pydicom.dcmread(f, stop_before_pixels=True)
    else:
        return pydicom.dcmread(str(file_path), stop_before_pixels=True)


class MultiViewViewer(VolumeRenderer):
    """Interactive DICOM viewer showing all three planes with synchronized navigation."""

    # Window/level presets (center, width)
    PRESETS = {
        'Auto': None,  # Will be calculated
        'Bone': (300, 1500),
        'Soft Tissue': (40, 400),
        'Lung': (-600, 1500),
        'Brain': (40, 80),
        'Liver': (60, 150),
    }

    def __init__(self, volume: DicomVolume, interactive: bool = True,
                 dicom_files: Optional[List[Union[str, Path]]] = None,
                 show_title: bool = True):
        """
        Initialize with a loaded DICOM volume.

        Args:
            volume: DicomVolume instance
            interactive: If True, try to use interactive matplotlib (click to move crosshairs)
            dicom_files: Optional list of DICOM file paths (unused, kept for compatibility)
            show_title: If True, show series title in viewer header
        """
        # Initialize base class (sets up lps_array, lps_spacing, window)
        super().__init__(volume)

        # Store display options
        self._show_title = show_title

        # Current crosshair position in voxel coordinates [S, P, L]
        self.position = [
            self.lps_array.shape[0] // 2,
            self.lps_array.shape[1] // 2,
            self.lps_array.shape[2] // 2,
        ]

        # Store auto window for presets
        self.PRESETS['Auto'] = self.window

        # Determine intensity range for sliders
        data = volume.pixel_array
        self.min_intensity = float(data.min())
        self.max_intensity = float(data.max())

        # Figure and axes references
        self._fig = None
        self._axes = None
        self._click_cid = None
        self._scroll_cid = None

        # References for in-place updates (avoid recreating figure)
        self._images = {}  # view -> imshow object
        self._hlines = {}  # view -> horizontal crosshair line
        self._vlines = {}  # view -> vertical crosshair line
        self._titles = {}  # view -> title object

        # Check if interactive matplotlib is available
        self._interactive = interactive and self._check_interactive_backend()

        # Callback for when rendering is complete
        self._on_ready_callback = None

        # Create widgets
        self._create_widgets()

        # Wire up observers
        self._setup_observers()

    def _check_interactive_backend(self) -> bool:
        """Check if ipympl (interactive matplotlib) is available."""
        try:
            import ipympl  # noqa: F401
            return True
        except ImportError:
            return False

    def _create_widgets(self):
        """Create all widgets."""
        # Window/level sliders
        self.window_center_slider = widgets.FloatSlider(
            value=self.window[0],
            min=self.min_intensity,
            max=self.max_intensity,
            step=1.0,
            description='Center:',
            continuous_update=True,
            layout=widgets.Layout(width='300px'),
            style={'description_width': '60px'}
        )

        self.window_width_slider = widgets.FloatSlider(
            value=self.window[1],
            min=1.0,
            max=self.max_intensity - self.min_intensity,
            step=1.0,
            description='Width:',
            continuous_update=True,
            layout=widgets.Layout(width='300px'),
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

        # Slice position sliders for each view
        self.slice_sliders = {}
        for view, axis_idx in [('axial', 0), ('coronal', 1), ('sagittal', 2)]:
            max_val = self.lps_array.shape[axis_idx] - 1
            slider = widgets.IntSlider(
                value=self.position[axis_idx],
                min=0,
                max=max_val,
                step=1,
                description=f'{view.capitalize()}:',
                continuous_update=True,
                layout=widgets.Layout(width='250px'),
                style={'description_width': '70px'}
            )
            self.slice_sliders[view] = slider

        # Output widget for matplotlib figure
        self.figure_output = widgets.Output(layout=widgets.Layout(
            min_height='250px',
            overflow='hidden',
        ))

        # Info display
        self.info_output = widgets.HTML(layout=widgets.Layout(overflow='hidden'))

    def _setup_observers(self):
        """Set up widget observers."""
        self.window_center_slider.observe(self._on_window_change, names='value')
        self.window_width_slider.observe(self._on_window_change, names='value')

        for view, slider in self.slice_sliders.items():
            slider.observe(self._make_slice_handler(view), names='value')

    def _make_preset_handler(self, preset_name: str):
        """Create a handler for preset button."""
        def handler(button):
            self._apply_preset(preset_name)
        return handler

    def _make_slice_handler(self, view: str):
        """Create a handler for slice slider."""
        def handler(change):
            self._on_slice_change(view, change['new'])
        return handler

    def _apply_preset(self, preset_name: str):
        """Apply window/level preset."""
        preset = self.PRESETS.get(preset_name)
        if preset:
            self.window_center_slider.value = preset[0]
            self.window_width_slider.value = preset[1]

    def _get_slice_index_for_view(self, view: str) -> int:
        """Get the current slice index for a view based on crosshair position."""
        if view == 'axial':
            return self.position[0]  # S
        elif view == 'coronal':
            return self.position[1]  # P
        elif view == 'sagittal':
            return self.position[2]  # L
        return 0

    def _get_crosshair_coords_for_view(self, view: str) -> Tuple[float, float]:
        """Get crosshair (x, y) coordinates for a view in image pixel coords."""
        if view == 'axial':
            # Axial shows [P, L], crosshair at (L, P)
            return (self.position[2], self.position[1])
        elif view == 'coronal':
            # Coronal shows [S, L], crosshair at (L, S)
            return (self.position[2], self.position[0])
        elif view == 'sagittal':
            # Sagittal shows [S, P], crosshair at (P, S)
            return (self.position[1], self.position[0])
        return (0, 0)

    def display(self, on_ready: Optional[callable] = None):
        """Display the viewer widget.

        Args:
            on_ready: Optional callback function called when the canvas is fully rendered.
                      Useful for hiding loading indicators.
        """
        self._on_ready_callback = on_ready

        # Header with title (if enabled) or just hints
        interact_hint = 'Click to move crosshairs, scroll to change slice' if self._interactive else 'Use sliders to navigate'
        if self._show_title:
            header = widgets.HTML(f'''
                <div style="padding:10px 16px;background:#f8fafc;border-bottom:1px solid #e2e8f0;">
                    <div style="font-weight:600;color:#1e293b;font-size:14px;margin-bottom:2px;">{self.volume.modality}: {self.volume.series_description}</div>
                    <div style="font-size:11px;color:#94a3b8;">{interact_hint}</div>
                </div>
            ''')
        else:
            header = None  # No header when embedded - parent provides it

        # Slice controls row
        slice_controls = widgets.HBox([
            self.slice_sliders['axial'],
            self.slice_sliders['coronal'],
            self.slice_sliders['sagittal'],
        ], layout=widgets.Layout(padding='8px 16px', gap='8px', align_items='center', flex_wrap='wrap'))

        # Window controls row
        window_controls = widgets.HBox([
            self.window_center_slider,
            self.window_width_slider,
            self.preset_buttons,
        ], layout=widgets.Layout(align_items='center', padding='0 16px 8px 16px', gap='8px', flex_wrap='wrap'))

        # Build controls list
        controls_children = []
        if header:
            controls_children.append(header)
        controls_children.extend([slice_controls, window_controls])

        controls = widgets.VBox(controls_children, layout=widgets.Layout(background='#ffffff'))

        layout = widgets.VBox([
            controls,
            self.figure_output,
            self.info_output,
        ], layout=widgets.Layout(overflow='hidden', background='#ffffff'))

        # Display layout first (so Output widget is in DOM)
        display(layout)

        if self._interactive:
            # Create figure inside the Output widget
            with self.figure_output:
                self._fig, self._axes = plt.subplots(1, 3, figsize=(8, 3), facecolor='black')

                self._fig.canvas.toolbar_visible = False
                self._fig.canvas.header_visible = False
                self._fig.canvas.footer_visible = False
                self._fig.canvas.capture_scroll = True

                for idx, view in enumerate(['axial', 'coronal', 'sagittal']):
                    self._create_view(self._axes[idx], view)

                self._fig.tight_layout(pad=1.0)

                # Connect event handlers
                self._click_cid = self._fig.canvas.mpl_connect('button_press_event', self._on_click)
                self._scroll_cid = self._fig.canvas.mpl_connect('scroll_event', self._on_scroll)

                plt.show()

            # Fire callback if provided
            if self._on_ready_callback:
                self._on_ready_callback()
                self._on_ready_callback = None

            self._update_info()

        else:
            self._render_figure()

    def _on_slice_change(self, view: str, value: int):
        """Handle slice slider change."""
        if view == 'axial':
            self.position[0] = value
        elif view == 'coronal':
            self.position[1] = value
        elif view == 'sagittal':
            self.position[2] = value
        self._render_figure()

    def _on_window_change(self, change):
        """Handle window/level change."""
        self.window = (self.window_center_slider.value, self.window_width_slider.value)
        self._render_figure()

    def _on_click(self, event):
        """Handle click events on the figure."""
        if event.inaxes is None:
            return

        # Find which view was clicked
        view = None
        for v, ax in zip(['axial', 'coronal', 'sagittal'], self._axes):
            if event.inaxes == ax:
                view = v
                break

        if view is None:
            return

        # Get click coordinates (in data/pixel space)
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        # Convert to voxel coordinates based on view
        # Note: y may need to be inverted depending on origin
        x, y = int(round(x)), int(round(y))

        if view == 'axial':
            # Axial shows [P, L], click gives (L, P)
            new_l = np.clip(x, 0, self.lps_array.shape[2] - 1)
            new_p = np.clip(y, 0, self.lps_array.shape[1] - 1)
            self.position[1] = new_p
            self.position[2] = new_l
        elif view == 'coronal':
            # Coronal shows [S, L], click gives (L, S)
            new_l = np.clip(x, 0, self.lps_array.shape[2] - 1)
            new_s = np.clip(y, 0, self.lps_array.shape[0] - 1)
            self.position[0] = new_s
            self.position[2] = new_l
        elif view == 'sagittal':
            # Sagittal shows [S, P], click gives (P, S)
            new_p = np.clip(x, 0, self.lps_array.shape[1] - 1)
            new_s = np.clip(y, 0, self.lps_array.shape[0] - 1)
            self.position[0] = new_s
            self.position[1] = new_p

        # Update sliders (triggers re-render)
        self.slice_sliders['axial'].value = self.position[0]
        self.slice_sliders['coronal'].value = self.position[1]
        self.slice_sliders['sagittal'].value = self.position[2]

    def _on_scroll(self, event):
        """Handle scroll events to navigate through slices."""
        if event.inaxes is None:
            return

        # Find which view was scrolled over
        view = None
        for v, ax in zip(['axial', 'coronal', 'sagittal'], self._axes):
            if event.inaxes == ax:
                view = v
                break

        if view is None:
            return

        # Scroll direction: up = forward, down = backward
        delta = 1 if event.button == 'up' else -1

        # Update the slice for the scrolled view
        if view == 'axial':
            new_val = np.clip(self.position[0] + delta, 0, self.lps_array.shape[0] - 1)
            self.slice_sliders['axial'].value = new_val
        elif view == 'coronal':
            new_val = np.clip(self.position[1] + delta, 0, self.lps_array.shape[1] - 1)
            self.slice_sliders['coronal'].value = new_val
        elif view == 'sagittal':
            new_val = np.clip(self.position[2] + delta, 0, self.lps_array.shape[2] - 1)
            self.slice_sliders['sagittal'].value = new_val

    def _render_static_preview(self):
        """Render a quick static preview image."""
        views = ['axial', 'coronal', 'sagittal']

        # Create a temporary figure for static rendering
        fig, axes = plt.subplots(1, 3, figsize=(8, 3), facecolor='black')

        for idx, view in enumerate(views):
            ax = axes[idx]
            ax.set_facecolor('black')

            slice_idx = self._get_slice_index_for_view(view)
            img = self.extract_slice(view, slice_idx)
            windowed = self.apply_window(img)
            aspect = self.calculate_aspect_ratio(view)
            origin = self.get_image_origin(view)

            ax.imshow(windowed, cmap='gray', aspect=aspect, origin=origin, interpolation='bilinear')

            # Add crosshairs
            cx, cy = self._get_crosshair_coords_for_view(view)
            ax.axhline(y=cy, color='yellow', linewidth=0.5, alpha=0.7)
            ax.axvline(x=cx, color='yellow', linewidth=0.5, alpha=0.7)

            # Add title
            num_slices = self.get_num_slices(view)
            ax.set_title(f'{view.capitalize()} ({slice_idx + 1}/{num_slices})', color='white', fontsize=10)
            self.add_orientation_labels(ax, view, fontsize=10)
            ax.axis('off')

        fig.tight_layout(pad=1.0)

        # Render to JPEG
        buf = BytesIO()
        fig.savefig(
            buf,
            format='jpeg',
            facecolor='black',
            bbox_inches='tight',
            dpi=100,
            pil_kwargs={'quality': 90}
        )
        buf.seek(0)
        plt.close(fig)

        return buf.read()

    def _render_figure(self):
        """Render/update the figure (called on slider changes)."""
        views = ['axial', 'coronal', 'sagittal']

        if self._interactive and self._fig is not None:
            # Update views in place
            for view in views:
                self._update_view(view)
            self._fig.canvas.draw_idle()
            self._update_info()

        elif not self._interactive:
            # Static mode - render to JPEG in Output widget
            import base64
            from IPython.display import HTML
            static_data = self._render_static_preview()
            b64_data = base64.b64encode(static_data).decode('utf-8')

            with self.figure_output:
                clear_output(wait=True)
                display(HTML(f'<img src="data:image/jpeg;base64,{b64_data}" style="display:block;">'))

            self._update_info()

    def _create_view(self, ax, view: str):
        """Create a single view (first render)."""
        ax.set_facecolor('black')

        # Get slice
        slice_idx = self._get_slice_index_for_view(view)
        img = self.extract_slice(view, slice_idx)
        windowed = self.apply_window(img)

        # Calculate aspect ratio and origin
        aspect = self.calculate_aspect_ratio(view)
        origin = self.get_image_origin(view)

        # Display image and store reference
        im = ax.imshow(
            windowed, cmap='gray', aspect=aspect, origin=origin,
            interpolation='bilinear'
        )
        self._images[view] = im

        # Add crosshairs and store references
        cx, cy = self._get_crosshair_coords_for_view(view)
        hline = ax.axhline(y=cy, color='yellow', linewidth=0.5, alpha=0.7)
        vline = ax.axvline(x=cx, color='yellow', linewidth=0.5, alpha=0.7)
        self._hlines[view] = hline
        self._vlines[view] = vline

        # Add title with slice info and store reference
        num_slices = self.get_num_slices(view)
        title = ax.set_title(
            f'{view.capitalize()} ({slice_idx + 1}/{num_slices})',
            color='white', fontsize=10
        )
        self._titles[view] = title

        # Add orientation labels
        self.add_orientation_labels(ax, view, fontsize=10)

        ax.axis('off')

    def _update_view(self, view: str):
        """Update a view in place (subsequent renders)."""
        # Get new slice data
        slice_idx = self._get_slice_index_for_view(view)
        img = self.extract_slice(view, slice_idx)
        windowed = self.apply_window(img)

        # Update image data
        self._images[view].set_data(windowed)

        # Update crosshairs
        cx, cy = self._get_crosshair_coords_for_view(view)
        self._hlines[view].set_ydata([cy, cy])
        self._vlines[view].set_xdata([cx, cx])

        # Update title
        num_slices = self.get_num_slices(view)
        self._titles[view].set_text(f'{view.capitalize()} ({slice_idx + 1}/{num_slices})')

    def _update_info(self):
        """Update the info display."""
        s, p, lat = self.position

        # Calculate actual slice spacing from slice locations
        slice_locs = self.volume.slice_locations
        if len(slice_locs) > 1:
            calculated_spacing = float(np.median(np.abs(np.diff(slice_locs))))
        else:
            calculated_spacing = None

        # Show slice thickness - both tag and calculated if they differ
        nominal = self.volume.slice_thickness
        if nominal is not None and calculated_spacing is not None:
            if abs(nominal - calculated_spacing) > 0.1:
                thickness_info = f'<span style="color:#64748b;font-weight:500;">Slice:</span> {nominal:.1f}mm / {calculated_spacing:.1f}mm'
            else:
                thickness_info = f'<span style="color:#64748b;font-weight:500;">Slice:</span> {nominal:.1f}mm'
        elif nominal is not None:
            thickness_info = f'<span style="color:#64748b;font-weight:500;">Slice:</span> {nominal:.1f}mm'
        elif calculated_spacing is not None:
            thickness_info = f'<span style="color:#64748b;font-weight:500;">Slice:</span> {calculated_spacing:.1f}mm'
        else:
            thickness_info = ''

        # Use original DICOM pixel spacing (row, col)
        pixel_spacing = self.volume.pixel_spacing
        pixel_info = f'<span style="color:#64748b;font-weight:500;">Pixel:</span> {pixel_spacing[0]:.2f}×{pixel_spacing[1]:.2f}mm'

        self.info_output.value = f'''
            <div style="background:#f1f5f9;padding:8px 16px;font-size:11px;color:#475569;
                        display:flex;flex-wrap:wrap;gap:16px;align-items:center;border-top:1px solid #e2e8f0;">
                <span><span style="color:#64748b;font-weight:500;">Pos:</span> S={s}, P={p}, L={lat}</span>
                <span><span style="color:#64748b;font-weight:500;">Window:</span> C={self.window[0]:.0f}, W={self.window[1]:.0f}</span>
                <span>{thickness_info}</span>
                <span>{pixel_info}</span>
            </div>
        '''

    def set_position(self, s: int, p: int, lat: int):
        """Set crosshair position programmatically."""
        self.position = [
            int(np.clip(s, 0, self.lps_array.shape[0] - 1)),
            int(np.clip(p, 0, self.lps_array.shape[1] - 1)),
            int(np.clip(lat, 0, self.lps_array.shape[2] - 1)),
        ]
        # Update sliders (this will trigger _render_figure via observers)
        self.slice_sliders['axial'].value = self.position[0]
        self.slice_sliders['coronal'].value = self.position[1]
        self.slice_sliders['sagittal'].value = self.position[2]

    def get_current_window(self) -> Tuple[float, float]:
        """Get current window/level values."""
        return self.window

    def set_window(self, center: float, width: float):
        """Set window/level values."""
        self.window_center_slider.value = center
        self.window_width_slider.value = width

    def close(self):
        """Close the viewer and release matplotlib resources."""
        if self._fig is not None:
            # Disconnect event handlers
            if self._click_cid is not None:
                self._fig.canvas.mpl_disconnect(self._click_cid)
                self._click_cid = None
            if self._scroll_cid is not None:
                self._fig.canvas.mpl_disconnect(self._scroll_cid)
                self._scroll_cid = None
            # Close the figure
            plt.close(self._fig)
            self._fig = None
            self._axes = None
        # Clear image references
        self._images.clear()
        self._hlines.clear()
        self._vlines.clear()
        self._titles.clear()


class DicomHeaderViewer:
    """Standalone DICOM header viewer widget - no pixel data loading required."""

    # Common DICOM tag groups for filtering
    TAG_GROUPS = {
        'Patient': ['PatientName', 'PatientID', 'PatientBirthDate', 'PatientSex', 'PatientAge', 'PatientWeight'],
        'Study': ['StudyInstanceUID', 'StudyDate', 'StudyTime', 'StudyDescription', 'AccessionNumber', 'ReferringPhysicianName'],
        'Series': ['SeriesInstanceUID', 'SeriesNumber', 'SeriesDescription', 'Modality', 'BodyPartExamined', 'ProtocolName'],
        'Image': ['SOPInstanceUID', 'InstanceNumber', 'ImageType', 'AcquisitionDate', 'AcquisitionTime', 'ContentDate'],
        'Geometry': ['PixelSpacing', 'SliceThickness', 'SpacingBetweenSlices', 'ImagePositionPatient', 'ImageOrientationPatient', 'SliceLocation'],
        'Pixel Data': ['Rows', 'Columns', 'BitsAllocated', 'BitsStored', 'HighBit', 'PixelRepresentation', 'PhotometricInterpretation', 'SamplesPerPixel'],
        'Acquisition': ['KVP', 'ExposureTime', 'XRayTubeCurrent', 'RepetitionTime', 'EchoTime', 'FlipAngle', 'MagneticFieldStrength'],
    }

    def __init__(self, dicom_files: List[Union[str, Path]], title: str = "DICOM Header",
                 initial_settings: Optional[Dict] = None,
                 show_title: bool = True):
        """
        Initialize header viewer.

        Args:
            dicom_files: List of DICOM file paths
            title: Title to display
            initial_settings: Optional dict with 'quick_filter' and 'selected_tags' to restore
            show_title: If True, show title bar; if False, assume parent provides header
        """
        self._dicom_files = dicom_files or []
        self._title = title
        self._dicom_dataset: Optional[Any] = None
        self._initial_settings = initial_settings
        self._show_title = show_title
        self._create_widgets()

    def _create_widgets(self):
        """Create all widgets."""
        # File selector dropdown
        num_files = len(self._dicom_files)
        file_options = [(f'File {i+1} of {num_files}', i) for i in range(num_files)] if num_files > 0 else [('No files', 0)]
        self.file_selector = widgets.Dropdown(
            options=file_options,
            value=0 if num_files > 0 else 0,
            description='',
            layout=widgets.Layout(width='140px'),
            disabled=num_files <= 1,
        )
        self.file_selector.observe(self._on_file_change, names='value')

        # === Typeahead tag selector ===
        # Search input
        self.tag_search = widgets.Text(
            value='',
            placeholder='Type to search tags...',
            layout=widgets.Layout(width='250px'),
        )
        self.tag_search.observe(self._on_tag_search_change, names='value')

        # Suggestions dropdown (shows matching tags)
        self.tag_suggestions = widgets.SelectMultiple(
            options=[],
            value=[],
            description='',
            layout=widgets.Layout(width='250px', height='120px'),
        )

        # Add button
        self.add_tags_btn = widgets.Button(
            description='Add',
            button_style='primary',
            icon='plus',
            layout=widgets.Layout(width='60px'),
        )
        self.add_tags_btn.on_click(self._on_add_tags)

        # Quick filter buttons
        self.quick_all_btn = widgets.Button(description='All', button_style='primary', layout=widgets.Layout(width='50px'))
        self.quick_all_btn.on_click(lambda _: self._set_quick_filter('all'))
        self.quick_private_btn = widgets.Button(description='Private', layout=widgets.Layout(width='70px'))
        self.quick_private_btn.on_click(lambda _: self._set_quick_filter('private'))
        self.quick_seq_btn = widgets.Button(description='Seq', layout=widgets.Layout(width='50px'))
        self.quick_seq_btn.on_click(lambda _: self._set_quick_filter('sequences'))

        # Preset group buttons
        self.group_buttons = {}
        for group_name in self.TAG_GROUPS.keys():
            btn = widgets.Button(description=group_name, layout=widgets.Layout(width='auto'))
            btn.on_click(lambda _, g=group_name: self._add_group_tags(g))
            self.group_buttons[group_name] = btn

        # Clear button
        self.clear_btn = widgets.Button(description='Clear', button_style='warning', layout=widgets.Layout(width='60px'))
        self.clear_btn.on_click(lambda _: self._set_quick_filter('all'))

        # Selected tags display area (contains chips with remove buttons)
        self.selected_tags_area = widgets.HBox(
            [],
            layout=widgets.Layout(flex_wrap='wrap', gap='4px', min_height='30px')
        )
        self.selected_tags_label = widgets.HTML(
            value='<span style="color:#888;font-size:11px;">Showing all tags</span>',
        )

        # Store selected tags - restore from initial settings if provided
        if self._initial_settings:
            self._selected_tags = list(self._initial_settings.get('selected_tags', []))
            self._quick_filter = self._initial_settings.get('quick_filter', 'all')
        else:
            self._selected_tags = []  # List of (tag_str, keyword) tuples
            self._quick_filter = 'all'  # 'all', 'private', 'sequences', or None

        # All available tags (populated when DICOM loads)
        self._all_tags = []  # List of (tag_str, keyword, display_name) tuples

        # Header content display
        self.header_output = widgets.HTML(
            value='<div style="color:#888;padding:20px;">Loading...</div>',
            layout=widgets.Layout(
                width='100%',
                height='400px',
                overflow='auto',
                border='1px solid #ddd',
                background='white',
            )
        )

    def _on_file_change(self, change):
        """Handle file selection change."""
        self._dicom_dataset = None
        self._load_and_display()

    def _on_tag_search_change(self, change):
        """Update suggestions based on search text."""
        search = change['new'].strip().lower()
        if not search:
            self.tag_suggestions.options = []
            return

        # Filter available tags
        matches = []
        for tag_str, keyword, display_name in self._all_tags:
            if search in tag_str.lower() or search in keyword.lower() or search in display_name.lower():
                # Don't show already selected tags
                if (tag_str, keyword) not in self._selected_tags:
                    matches.append((display_name, (tag_str, keyword)))
                if len(matches) >= 20:  # Limit suggestions
                    break

        self.tag_suggestions.options = matches
        self.tag_suggestions.value = []

    def _on_add_tags(self, btn):
        """Add selected tags from suggestions."""
        for tag_str, keyword in self.tag_suggestions.value:
            if (tag_str, keyword) not in self._selected_tags:
                self._selected_tags.append((tag_str, keyword))

        self._quick_filter = None  # Switch to custom selection
        self._update_selected_tags_display()
        self._update_display()
        self.tag_search.value = ''
        self.tag_suggestions.options = []

    def _set_quick_filter(self, filter_type):
        """Set a quick filter (all, private, sequences)."""
        self._quick_filter = filter_type
        self._selected_tags = []
        self._update_selected_tags_display()
        self._update_display()

    def _add_group_tags(self, group_name):
        """Add all tags from a predefined group."""
        if group_name not in self.TAG_GROUPS:
            return

        # Add all tags from the group (use keyword for matching even if not in current file)
        group_keywords = set(self.TAG_GROUPS[group_name])

        # Build set of keywords already selected (to avoid duplicates)
        existing_keywords = {kw for _, kw in self._selected_tags}

        # First, add tags that exist in current file (with their tag_str)
        found_keywords = set()
        for tag_str, keyword, _display_name in self._all_tags:
            if keyword in group_keywords and keyword not in existing_keywords:
                self._selected_tags.append((tag_str, keyword))
                found_keywords.add(keyword)
                existing_keywords.add(keyword)

        # Also add keywords not found in current file (for persistence across series)
        for keyword in group_keywords - found_keywords:
            if keyword not in existing_keywords:
                self._selected_tags.append(('', keyword))

        self._quick_filter = None
        self._update_selected_tags_display()
        self._update_display()

    def _remove_tag(self, tag_str, keyword):
        """Remove a tag from selection."""
        if (tag_str, keyword) in self._selected_tags:
            self._selected_tags.remove((tag_str, keyword))
            if not self._selected_tags:
                self._quick_filter = 'all'
            self._update_selected_tags_display()
            self._update_display()

    def _update_selected_tags_display(self):
        """Update the selected tags chips display."""
        # Update button styles
        self.quick_all_btn.button_style = 'primary' if self._quick_filter == 'all' else ''
        self.quick_private_btn.button_style = 'primary' if self._quick_filter == 'private' else ''
        self.quick_seq_btn.button_style = 'primary' if self._quick_filter == 'sequences' else ''

        if self._quick_filter == 'all':
            self.selected_tags_label.value = '<span style="color:#888;font-size:11px;">Showing all tags</span>'
            self.selected_tags_area.children = []
        elif self._quick_filter == 'private':
            self.selected_tags_label.value = '<span style="color:#9933cc;font-size:11px;">Showing private tags only</span>'
            self.selected_tags_area.children = []
        elif self._quick_filter == 'sequences':
            self.selected_tags_label.value = '<span style="color:#0066cc;font-size:11px;">Showing sequences only</span>'
            self.selected_tags_area.children = []
        elif self._selected_tags:
            self.selected_tags_label.value = '<span style="font-size:11px;color:#666;">Selected:</span>'
            # Create chip buttons for each selected tag
            chips = []
            for tag_str, keyword in self._selected_tags:
                display = keyword if keyword else tag_str
                tooltip_text = f'Click to remove {keyword or tag_str}'
                chip = widgets.Button(
                    description=f'{display} ×',
                    button_style='',
                    layout=widgets.Layout(width='auto', height='24px', padding='0 8px'),
                    tooltip=tooltip_text
                )
                # Capture tag_str and keyword in closure
                chip.on_click(lambda _, ts=tag_str, kw=keyword: self._remove_tag(ts, kw))
                chips.append(chip)
            self.selected_tags_area.children = chips
        else:
            self.selected_tags_label.value = '<span style="color:#888;font-size:11px;">No tags selected - showing all</span>'
            self.selected_tags_area.children = []

    def _load_and_display(self):
        """Load DICOM file and update display."""
        if not self._dicom_files:
            self.header_output.value = '<div style="color:#888;padding:20px;">No DICOM files available</div>'
            return

        file_idx = self.file_selector.value
        if file_idx >= len(self._dicom_files):
            return

        try:
            file_path = self._dicom_files[file_idx]
            self._dicom_dataset = _read_dicom_for_header(file_path)

            # Build list of all available tags for typeahead
            self._all_tags = []
            for elem in self._dicom_dataset:
                if elem.tag == (0x7FE0, 0x0010):  # Skip pixel data
                    continue
                tag_str = f'({elem.tag.group:04X},{elem.tag.element:04X})'
                keyword = elem.keyword or ''
                if keyword:
                    display = f'{tag_str} {keyword}'
                elif elem.tag.is_private:
                    display = f'{tag_str} [Private]'
                else:
                    display = tag_str
                self._all_tags.append((tag_str, keyword, display))

            self._update_selected_tags_display()  # Update button styles for restored settings
            self._update_display()
        except Exception as e:
            self.header_output.value = f'<div style="color:red;padding:20px;">Error loading DICOM: {e}</div>'

    def _format_element(self, elem, indent: int = 0) -> str:
        """Format a DICOM element as HTML."""
        import html as html_module
        indent_px = indent * 20
        tag_str = f'({elem.tag.group:04X},{elem.tag.element:04X})'

        if elem.keyword:
            name = elem.keyword
        elif elem.tag.is_private:
            name = f'[Private: {elem.tag.element >> 8:02X}xx]'
        else:
            name = '[Unknown]'

        vr = elem.VR if hasattr(elem, 'VR') else '??'

        if elem.VR == 'SQ':
            seq_html = f'<div style="margin-left:{indent_px}px;margin-bottom:4px;">'
            seq_html += f'<span style="color:#666;">{tag_str}</span> '
            seq_html += f'<span style="color:#0066cc;font-weight:bold;">{html_module.escape(name)}</span> '
            seq_html += f'<span style="color:#888;">[SQ] {len(elem.value)} item(s)</span>'
            seq_html += '</div>'

            for i, item in enumerate(elem.value):
                seq_html += f'<div style="margin-left:{indent_px + 20}px;border-left:2px solid #ddd;padding-left:8px;margin-bottom:4px;">'
                seq_html += f'<div style="color:#888;font-size:11px;">Item {i + 1}</div>'
                for sub_elem in item:
                    seq_html += self._format_element(sub_elem, indent + 2)
                seq_html += '</div>'
            return seq_html
        else:
            # Handle encoding errors for binary/non-UTF-8 data
            try:
                value_str = str(elem.value)
            except (UnicodeDecodeError, UnicodeEncodeError):
                if isinstance(elem.value, bytes):
                    value_str = f"[Binary: {elem.value[:20].hex()}{'...' if len(elem.value) > 20 else ''}]"
                else:
                    value_str = "[Unable to decode value]"
            if len(value_str) > 100:
                value_str = value_str[:100] + '...'
            value_str = html_module.escape(value_str)

            name_color = '#9933cc' if elem.tag.is_private else '#0066cc'

            return f'''<div style="margin-left:{indent_px}px;margin-bottom:2px;font-family:monospace;font-size:12px;">
                <span style="color:#666;">{tag_str}</span>
                <span style="color:{name_color};">{html_module.escape(name)}</span>
                <span style="color:#888;">[{vr}]</span>
                <span style="color:#333;">{value_str}</span>
            </div>'''

    def _update_display(self):
        """Update the header display based on filters."""
        if self._dicom_dataset is None:
            return

        ds = self._dicom_dataset

        # Build sets for quick lookup - match by tag_str OR keyword
        selected_tag_strs = {tag_str for tag_str, _ in self._selected_tags if tag_str}
        selected_keywords = {keyword for _, keyword in self._selected_tags if keyword}

        html_parts = []
        for elem in ds:
            if elem.tag == (0x7FE0, 0x0010):
                continue

            tag_str = f'({elem.tag.group:04X},{elem.tag.element:04X})'
            keyword = elem.keyword or ''
            show = False

            if self._quick_filter == 'all':
                show = True
            elif self._quick_filter == 'private':
                show = elem.tag.is_private
            elif self._quick_filter == 'sequences':
                show = elem.VR == 'SQ'
            elif self._selected_tags:
                # Show only specifically selected tags (match by tag_str or keyword)
                show = tag_str in selected_tag_strs or keyword in selected_keywords
            else:
                show = True  # No filter = show all

            if show:
                html_parts.append(self._format_element(elem))

        if html_parts:
            self.header_output.value = f'<div style="padding:10px;">{"".join(html_parts)}</div>'
        else:
            self.header_output.value = '<div style="color:#888;padding:20px;">No matching tags</div>'

    def display(self):
        """Display the header viewer widget."""
        # Build layout children
        layout_children = []

        # Header bar - only if show_title is True
        if self._show_title:
            header_items = [
                widgets.HTML(f'<span style="font-weight:600;color:#1e293b;font-size:14px;">{self._title}</span>'),
                widgets.HTML(f'<span style="color:#94a3b8;font-size:11px;margin-left:10px;">{len(self._dicom_files)} file(s)</span>'),
            ]

            header = widgets.HBox(header_items, layout=widgets.Layout(
                padding='12px 16px',
                background='#f8fafc',
                border_bottom='1px solid #e2e8f0',
                align_items='center',
            ))
            layout_children.append(header)

        # Filter controls - file selector and search
        search_items = [self.file_selector, self.tag_search, self.tag_suggestions, self.add_tags_btn]

        search_row = widgets.HBox(search_items, layout=widgets.Layout(
            margin='12px 16px' if self._show_title else '8px 16px',
            gap='8px',
            align_items='flex-start'
        ))
        layout_children.append(search_row)

        # Quick filter buttons and group presets
        quick_filter_row = widgets.HBox([
            widgets.HTML('<span style="font-size:11px;color:#64748b;margin-right:5px;">Quick:</span>'),
            self.quick_all_btn,
            self.quick_private_btn,
            self.quick_seq_btn,
            widgets.HTML('<span style="color:#e2e8f0;margin:0 8px;">|</span>'),
            widgets.HTML('<span style="font-size:11px;color:#64748b;margin-right:3px;">Groups:</span>'),
        ] + list(self.group_buttons.values()) + [
            widgets.HTML('<span style="color:#e2e8f0;margin:0 8px;">|</span>'),
            self.clear_btn,
        ], layout=widgets.Layout(margin='0 16px 8px 16px', gap='4px', align_items='center', flex_wrap='wrap'))
        layout_children.append(quick_filter_row)

        # Selected tags display
        selected_row = widgets.HBox([
            self.selected_tags_label,
            self.selected_tags_area,
        ], layout=widgets.Layout(margin='0 16px 12px 16px', align_items='center', gap='8px', flex_wrap='wrap'))
        layout_children.append(selected_row)

        # Make header output resizable
        self.header_output.layout = widgets.Layout(
            width='100%',
            height='400px',
            min_height='200px',
            overflow='auto',
            border='1px solid #e2e8f0',
            background='white',
            resize='vertical',
            margin='0 16px 16px 16px',
            border_radius='6px',
        )
        layout_children.append(self.header_output)

        layout = widgets.VBox(layout_children, layout=widgets.Layout(
            border='1px solid #e2e8f0' if self._show_title else 'none',
            border_radius='6px' if self._show_title else '0',
            background='#ffffff',
        ))

        display(layout)
        self._load_and_display()

    def get_settings(self) -> Dict:
        """Get current tag filter settings to restore later.

        Returns:
            Dict with 'quick_filter' and 'selected_tags' keys
        """
        return {
            'quick_filter': self._quick_filter,
            'selected_tags': list(self._selected_tags),
        }
