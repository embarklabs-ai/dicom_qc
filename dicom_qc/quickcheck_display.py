"""Interactive Jupyter display for QuickCheck."""

from typing import Optional


class QuickCheckDisplayMixin:
    """Mixin providing interactive Jupyter display methods for QuickCheck."""

    @staticmethod
    def _series_num_sort_key(n):
        """Sort key that puts numeric values first, then strings."""
        try:
            return (0, int(n))
        except (ValueError, TypeError):
            return (1, str(n or ''))

    def display(self):
        """Display interactive review interface with vertical list and side viewer."""
        import ipywidgets as widgets
        from IPython.display import display, clear_output

        all_series = self.get_all_series()
        counts = self.get_summary()
        c = self.STATUS_COLORS

        # === Build filter options ===
        subject_options = [('All Subjects', None)] + [(p.label, pid) for pid, p in sorted(self.patients.items())]
        session_options = [('All Sessions', None)]

        # Build series number options (sorted numerically)
        series_numbers = sorted(set(s.series_number for s in all_series if s.series_number is not None), key=self._series_num_sort_key)
        series_num_options = [('All', None)] + [(str(n), n) for n in series_numbers]

        # Build status filter options with counts
        status_options = [('All', 'ALL')]
        for status in ['PASS', 'FAIL', 'WARNING', 'ERROR', 'NOTE', 'DERIVED']:
            if counts.get(status, 0) > 0:
                status_options.append((f'{status} ({counts[status]})', status))

        # === Summary - compact badges ===
        summary_parts = [f'<span style="font-weight:600;font-size:14px;">{len(all_series)}</span> series']
        for status in ['PASS', 'FAIL', 'WARNING', 'ERROR', 'NOTE', 'DERIVED']:
            if counts.get(status, 0) > 0:
                text_color = '#333' if status == 'WARNING' else '#fff'
                summary_parts.append(f'<span style="background:{c[status]};color:{text_color};padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;">{counts[status]} {status}</span>')
        summary_html = ' '.join(summary_parts)

        # === Filter widgets (compact) ===
        status_dropdown = widgets.Dropdown(
            options=status_options, value='ALL',
            layout=widgets.Layout(width='120px')
        )
        subject_dropdown = widgets.Dropdown(options=subject_options, value=None, layout=widgets.Layout(width='160px'))
        session_dropdown = widgets.Dropdown(options=session_options, value=None, layout=widgets.Layout(width='180px'))
        series_num_dropdown = widgets.Dropdown(options=series_num_options, value=None, layout=widgets.Layout(width='70px'))
        description_filter = widgets.Text(placeholder='Filter...', layout=widgets.Layout(width='120px'))

        # === Left panel: scrollable list with buttons ===
        list_area = widgets.Output(layout=widgets.Layout(
            height='80vh',
            width='98%',
            overflow_y='auto',
            background='#fafafa'
        ))
        list_area.add_class('qc-list-area')

        # === Right panel: single content area with toggle ===
        content_panel = widgets.Output(layout=widgets.Layout(
            min_height='80vh',
            overflow='auto',
            border='1px solid #e2e8f0',
            border_radius='6px',
            background='#ffffff'
        ))

        # Toggle switch for Image/Header view (displayed inside content panel header)
        view_toggle = widgets.ToggleButtons(
            options=[('Image', 'image'), ('Header', 'header')],
            value='image',
            button_style='',
            layout=widgets.Layout(margin='0 10px')
        )

        # Container for content (toggle is rendered inside by the render functions)
        toggle_container = widgets.VBox(
            [content_panel],
            layout=widgets.Layout(width='100%', min_height='80vh')
        )

        # State tracking
        selected_series = [None]
        # Store card widgets for selection updates without re-rendering
        card_widgets = {}
        # Hidden output widget for running scroll scripts
        script_runner = widgets.Output(layout=widgets.Layout(height='0', overflow='hidden'))

        def scroll_to_card(series):
            """Scroll a card into view using JavaScript."""
            from IPython.display import Javascript
            if series is not None and id(series) in card_widgets:
                card_id = f'qc-card-{id(series)}'
                with script_runner:
                    clear_output()
                    display(Javascript(f'''
                        setTimeout(function() {{
                            var card = document.querySelector('.{card_id}');
                            if (card) card.scrollIntoView({{behavior: 'auto', block: 'nearest'}});
                        }}, 100);
                    '''))

        # Store loaded data so we can switch views without reloading
        loaded_data = {
            'series': None,
            'patient': None,  # Patient/Subject info
            'study': None,    # Study/Session info
            'dicom_files': None,
            'volume': None,
            'image_loaded': False,
            'header_loaded': False,
            'header_viewer': None,  # Reference to current header viewer
            'header_settings': None,  # Persisted header viewer settings
            'image_viewer': None,  # Reference to current image viewer (for cleanup)
        }

        def clear_content():
            """Clear content and reset state."""
            # Save header settings before clearing
            if loaded_data['header_viewer'] is not None:
                loaded_data['header_settings'] = loaded_data['header_viewer'].get_settings()
            # Close image viewer to release matplotlib resources
            if loaded_data['image_viewer'] is not None:
                loaded_data['image_viewer'].close()
            loaded_data['series'] = None
            loaded_data['patient'] = None
            loaded_data['study'] = None
            loaded_data['dicom_files'] = None
            loaded_data['volume'] = None
            loaded_data['image_loaded'] = False
            loaded_data['header_loaded'] = False
            loaded_data['header_viewer'] = None
            loaded_data['image_viewer'] = None
            with content_panel:
                clear_output()

        def update_selection(new_series):
            """Update selection indicator without re-rendering the list."""
            old_series = selected_series[0]

            # Remove highlight from old selection
            if old_series is not None and id(old_series) in card_widgets:
                old_info = card_widgets[id(old_series)]
                old_info['card'].layout.box_shadow = None
                old_info['view_btn'].description = 'View'
                old_info['view_btn'].button_style = 'info'

            # Add highlight to new selection
            selected_series[0] = new_series
            if new_series is not None and id(new_series) in card_widgets:
                new_info = card_widgets[id(new_series)]
                border_color = new_info['border_color']
                new_info['card'].layout.box_shadow = f'0 0 0 3px {border_color}'
                new_info['view_btn'].description = '● View'
                new_info['view_btn'].button_style = 'success'

        def render_header_view():
            """Render the header view in the content panel."""
            from dicom_qc.widgets import DicomHeaderViewer
            series = loaded_data['series']
            patient = loaded_data['patient']
            study = loaded_data['study']
            dicom_files = loaded_data['dicom_files']

            with content_panel:
                clear_output(wait=True)

                # Unified header bar
                display(make_header_bar(series, patient, study))

                if dicom_files:
                    # Create viewer with persisted settings (no title - parent provides header)
                    viewer = DicomHeaderViewer(
                        dicom_files,
                        title=series.label,
                        initial_settings=loaded_data['header_settings'],
                        show_title=False
                    )
                    loaded_data['header_viewer'] = viewer
                    viewer.display()
                else:
                    display(widgets.HTML('<div style="color:#64748b;padding:30px;">No DICOM files available for this series</div>'))

        # Status styles used throughout (bg, border, badge_bg, badge_text, accent, text)
        STATUS_STYLES = {
            'PASS': {'bg': '#f0fdf4', 'border': '#22c55e', 'badge_bg': '#22c55e', 'badge_text': '#fff', 'accent': '#16a34a', 'text': '#166534'},
            'WARNING': {'bg': '#fffbeb', 'border': '#f59e0b', 'badge_bg': '#f59e0b', 'badge_text': '#78350f', 'accent': '#d97706', 'text': '#92400e'},
            'FAIL': {'bg': '#fef2f2', 'border': '#ef4444', 'badge_bg': '#ef4444', 'badge_text': '#fff', 'accent': '#dc2626', 'text': '#991b1b'},
            'ERROR': {'bg': '#f9fafb', 'border': '#6b7280', 'badge_bg': '#6b7280', 'badge_text': '#fff', 'accent': '#4b5563', 'text': '#374151'},
            'DERIVED': {'bg': '#faf5ff', 'border': '#a855f7', 'badge_bg': '#a855f7', 'badge_text': '#fff', 'accent': '#9333ea', 'text': '#6b21a8'},
            'NOTE': {'bg': '#ecfeff', 'border': '#06b6d4', 'badge_bg': '#06b6d4', 'badge_text': '#fff', 'accent': '#0891b2', 'text': '#155e75'},
            'PENDING': {'bg': '#f9fafb', 'border': '#9ca3af', 'badge_bg': '#9ca3af', 'badge_text': '#fff', 'accent': '#6b7280', 'text': '#6b7280'},
        }

        def make_viewer_header(series, patient, study):
            """Create the viewer header HTML."""
            style = STATUS_STYLES.get(series.qc_status, STATUS_STYLES['PENDING'])
            subject_label = patient.label if patient else 'Unknown'
            session_label = study.label if study else 'Unknown'

            return widgets.HTML(f'''
                <div style="flex:1;min-width:0;">
                    <div style="display:flex;align-items:center;gap:8px;margin-bottom:2px;">
                        <span style="color:#1e293b;font-weight:600;font-size:14px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{series.label}</span>
                        <span style="background:{style['badge_bg']};color:{style['badge_text']};padding:3px 10px;border-radius:4px;font-size:10px;font-weight:600;flex-shrink:0;">{series.qc_status}</span>
                    </div>
                    <div style="color:#64748b;font-size:11px;">
                        {subject_label} · {session_label}
                    </div>
                </div>
            ''')

        def get_filtered_series_list():
            """Get list of (series, patient, study) tuples matching current filters."""
            result = []
            for patient_id, patient in sorted(self.patients.items()):
                for study_key, study in sorted(patient.studies.items(), key=lambda x: x[1].date or ''):
                    for series in sorted(study.series.values(), key=lambda s: self._series_num_sort_key(s.series_number)):
                        if matches_filter(series, patient_id, study_key):
                            result.append((series, patient, study))
            return result

        def navigate_series(direction):
            """Navigate to prev (-1) or next (+1) series in filtered list."""
            current = loaded_data['series']
            if current is None:
                return

            filtered = get_filtered_series_list()
            if not filtered:
                return

            # Find current index
            current_idx = None
            for i, (s, p, st) in enumerate(filtered):
                if s is current:
                    current_idx = i
                    break

            if current_idx is None:
                return

            # Calculate new index
            new_idx = current_idx + direction
            if new_idx < 0 or new_idx >= len(filtered):
                return  # At boundary

            # Save header settings before navigating
            if loaded_data['header_viewer'] is not None:
                loaded_data['header_settings'] = loaded_data['header_viewer'].get_settings()

            # Navigate to new series
            new_series, new_patient, new_study = filtered[new_idx]
            update_selection(new_series)
            scroll_to_card(new_series)

            # Update loaded_data and re-render
            loaded_data['series'] = new_series
            loaded_data['patient'] = new_patient
            loaded_data['study'] = new_study
            dicom_files = new_series.files if new_series.files else []
            if not dicom_files and new_series._file_paths:
                dicom_files = [p for p in new_series._file_paths if p]
            loaded_data['dicom_files'] = dicom_files
            loaded_data['volume'] = None
            loaded_data['image_loaded'] = False

            # Re-render current view
            if view_toggle.value == 'header':
                render_header_view()
            else:
                render_image_view()

        def make_header_bar(series, patient, study):
            """Create unified header bar with series info and toggle."""
            style = STATUS_STYLES.get(series.qc_status, STATUS_STYLES['PENDING'])
            header_content = make_viewer_header(series, patient, study)

            # Navigation buttons
            prev_btn = widgets.Button(
                description='',
                icon='chevron-left',
                button_style='',
                tooltip='Previous series',
                layout=widgets.Layout(width='32px', height='32px')
            )
            prev_btn.on_click(lambda _: navigate_series(-1))

            next_btn = widgets.Button(
                description='',
                icon='chevron-right',
                button_style='',
                tooltip='Next series',
                layout=widgets.Layout(width='32px', height='32px')
            )
            next_btn.on_click(lambda _: navigate_series(1))

            nav_buttons = widgets.HBox([prev_btn, next_btn], layout=widgets.Layout(gap='4px'))

            # Close button
            close_btn = widgets.Button(
                description='',
                icon='times',
                button_style='',
                tooltip='Close viewer',
                layout=widgets.Layout(width='32px', height='32px')
            )
            close_btn.on_click(lambda _: close_viewer())

            # Group controls on the right so they don't move when content changes
            controls = widgets.HBox(
                [view_toggle, nav_buttons, close_btn],
                layout=widgets.Layout(gap='8px', flex='0 0 auto', align_items='center')
            )

            header_bar = widgets.HBox(
                [header_content, controls],
                layout=widgets.Layout(
                    padding='12px 16px',
                    background=style['bg'],
                    border_bottom=f'2px solid {style["border"]}',
                    align_items='center',
                    justify_content='space-between',
                )
            )
            return header_bar

        def make_issues_banner(series):
            """Create QC issues banner if there are issues."""
            if not (series.qc_report and series.qc_status in ('FAIL', 'WARNING', 'NOTE')):
                return None
            issues = [r for r in series.qc_report.results if r.status in ('FAIL', 'WARNING', 'NOTE')]
            if not issues:
                return None

            style = STATUS_STYLES.get(series.qc_status, STATUS_STYLES['NOTE'])
            issue_items = ' · '.join(f'<b>{r.check_name}:</b> {r.message}' for r in issues)
            return widgets.HTML(f'''
                <div style="padding:10px 16px;background:{style['bg']};border-bottom:1px solid {style['border']};font-size:11px;color:{style['accent']};">
                    {issue_items}
                </div>
            ''')

        def render_image_view():
            """Render the image view in the content panel."""
            from dicom_qc.widgets import MultiViewViewer
            series = loaded_data['series']
            patient = loaded_data['patient']
            study = loaded_data['study']
            dicom_files = loaded_data['dicom_files']
            volume = loaded_data['volume']

            # Close previous image viewer to release matplotlib resources
            if loaded_data['image_viewer'] is not None:
                loaded_data['image_viewer'].close()
                loaded_data['image_viewer'] = None

            with content_panel:
                clear_output(wait=True)

                # Unified header bar
                display(make_header_bar(series, patient, study))

                # QC issues banner (if any)
                issues_banner = make_issues_banner(series)
                if issues_banner:
                    display(issues_banner)

                # Handle error/derived cases
                if series.error:
                    display(widgets.HTML(f'''
                        <div style="padding:30px;margin:16px;background:#fef2f2;border:1px solid #fecaca;border-radius:6px;">
                            <div style="color:#dc2626;font-weight:600;margin-bottom:8px;">Error Loading Series</div>
                            <div style="color:#991b1b;font-size:13px;">{series.error}</div>
                        </div>
                    '''))
                    return

                if series.is_derived:
                    display(widgets.HTML(f'''
                        <div style="padding:40px;margin:16px;background:#faf5ff;border:1px solid #e9d5ff;border-radius:6px;text-align:center;">
                            <div style="font-size:20px;margin-bottom:10px;color:#7c3aed;font-weight:600;">{series.modality}</div>
                            <div style="color:#6b21a8;">Derived/Non-image data</div>
                            {f'<div style="margin-top:15px;font-size:12px;color:#9333ea;">{series.derived_info}</div>' if series.derived_info else ''}
                        </div>
                    '''))
                    return

                if volume is None:
                    display(widgets.HTML('''
                        <div style="padding:60px;text-align:center;background:#f8fafc;">
                            <div style="color:#64748b;font-size:16px;margin-bottom:16px;">Images not loaded yet</div>
                        </div>
                    '''))
                    load_btn = widgets.Button(description='Load Images', button_style='primary', icon='image')
                    load_btn.on_click(lambda _: load_images())
                    display(load_btn)
                    return

                # Pixel transfer warning
                display(widgets.HTML('''
                    <div style="padding:8px 16px;background:#f8fafc;color:#64748b;font-size:11px;border-bottom:1px solid #e2e8f0;">
                        Note: Image may appear blank for 30-60s while pixel data transfers to browser.
                    </div>
                '''))

                # Show the viewer
                viewer = MultiViewViewer(volume, dicom_files=dicom_files, show_title=False)
                loaded_data['image_viewer'] = viewer
                viewer.display()

        def load_images():
            """Load DICOM images for the current series."""
            import time
            series = loaded_data['series']
            dicom_files = loaded_data['dicom_files']

            # Show loading spinner with light theme
            with content_panel:
                clear_output(wait=True)
                lst = STATUS_STYLES.get(series.qc_status, STATUS_STYLES['PENDING'])
                display(widgets.HTML(f'''
                    <div style="padding:12px 16px;background:{lst['bg']};border-bottom:2px solid {lst['border']};border-radius:6px 6px 0 0;">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <div style="color:#1e293b;font-weight:600;font-size:14px;">{series.label}</div>
                            <span style="background:{lst['badge_bg']};color:{lst['badge_text']};padding:5px 14px;border-radius:4px;font-size:11px;font-weight:600;">{series.qc_status}</span>
                        </div>
                    </div>
                    <div style="padding:60px;text-align:center;background:#f8fafc;">
                        <div style="display:inline-block;width:40px;height:40px;border:3px solid #e2e8f0;border-top-color:#3b82f6;border-radius:50%;animation:spin 1s linear infinite;"></div>
                        <div style="margin-top:20px;font-size:16px;color:#475569;">Loading DICOM data...</div>
                        <div style="margin-top:10px;font-size:12px;color:#94a3b8;">Image may appear blank for 30-60s while pixel data transfers to browser.</div>
                        <style>@keyframes spin {{ to {{ transform: rotate(360deg); }} }}</style>
                    </div>
                '''))

            # Small delay to let browser render the spinner
            time.sleep(0.1)

            # Load DICOM data
            try:
                try:
                    import SimpleITK as sitk
                    sitk.ProcessObject_SetGlobalWarningDisplay(False)
                except Exception:
                    pass

                if series._xnat_files:
                    files = self._get_xnat_files(series)
                    if not files:
                        raise ValueError("No DICOM files found")
                    volume = self.loader.load_from_xnat(files)
                else:
                    # For local files, use the parent directory of the series files
                    # (SimpleITK's GetGDCMSeriesIDs only searches one directory level)
                    from pathlib import Path
                    series_files = series.files if series.files else dicom_files
                    if series_files:
                        first_file = series_files[0]
                        if isinstance(first_file, (str, Path)):
                            series_dir = Path(first_file).parent
                        else:
                            series_dir = self.data_dir
                    else:
                        series_dir = self.data_dir
                    volume = self.loader.load_from_path_simpleitk(series_dir, series_uid=series.uid)

                loaded_data['volume'] = volume
                loaded_data['image_loaded'] = True
                render_image_view()

            except Exception as e:
                with content_panel:
                    clear_output(wait=True)
                    display(widgets.HTML(f'''
                        <div style="padding:30px;margin:16px;background:#fef2f2;border:1px solid #fecaca;border-radius:6px;">
                            <div style="color:#dc2626;font-weight:600;margin-bottom:8px;">Error Loading Images</div>
                            <div style="color:#991b1b;font-size:13px;">{e}</div>
                        </div>
                    '''))

        def on_toggle_change(change):
            """Handle toggle switch change."""
            if loaded_data['series'] is None:
                return
            if change['new'] == 'header':
                render_header_view()
            else:
                render_image_view()

        view_toggle.observe(on_toggle_change, names='value')

        def show_header(series, patient=None, study=None):
            """Show DICOM header viewer for selected series."""
            # If switching series, clear everything
            # Use object identity since uid may be empty in XNAT mode
            if loaded_data['series'] is None or loaded_data['series'] is not series:
                clear_content()
                loaded_data['series'] = series
                loaded_data['patient'] = patient
                loaded_data['study'] = study
                # Get DICOM files
                dicom_files = series.files if series.files else []
                if not dicom_files and series._file_paths:
                    dicom_files = [p for p in series._file_paths if p]
                loaded_data['dicom_files'] = dicom_files

            loaded_data['header_loaded'] = True
            view_toggle.value = 'header'
            render_header_view()

        def show_viewer(series, patient=None, study=None):
            """Show viewer for selected series (View button)."""
            # If switching series, clear everything
            # Use object identity since uid may be empty in XNAT mode
            if loaded_data['series'] is None or loaded_data['series'] is not series:
                clear_content()
                loaded_data['series'] = series
                loaded_data['patient'] = patient
                loaded_data['study'] = study
                # Get DICOM files
                dicom_files = series.files if series.files else []
                if not dicom_files and series._file_paths:
                    dicom_files = [p for p in series._file_paths if p]
                loaded_data['dicom_files'] = dicom_files

            view_toggle.value = 'image'

            # Load images if not already loaded
            if not loaded_data['image_loaded']:
                load_images()
            else:
                render_image_view()

        def update_session_options(_=None):
            """Update session dropdown based on selected subject."""
            subj_id = subject_dropdown.value
            if subj_id is None:
                session_dropdown.options = [('All Sessions', None)]
            else:
                patient = self.patients.get(subj_id)
                if patient:
                    opts = [('All Sessions', None)] + [(s.label, sk) for sk, s in sorted(patient.studies.items())]
                    session_dropdown.options = opts
            session_dropdown.value = None

        def matches_filter(series, patient_id, study_key):
            # Status filter
            if status_dropdown.value != 'ALL' and series.qc_status != status_dropdown.value:
                return False
            # Subject filter
            if subject_dropdown.value is not None and patient_id != subject_dropdown.value:
                return False
            # Session filter
            if session_dropdown.value is not None and study_key != session_dropdown.value:
                return False
            # Series number filter
            if series_num_dropdown.value is not None and series.series_number != series_num_dropdown.value:
                return False
            # Description filter
            desc_query = description_filter.value.strip().lower()
            if desc_query and desc_query not in (series.description or '').lower():
                return False
            return True

        def render_list(_=None):
            """Render the vertical series list with buttons."""
            # Clear card references since we're recreating them
            card_widgets.clear()

            with list_area:
                clear_output(wait=True)

                # Global CSS for list area
                display(widgets.HTML('''<style>
                    .qc-list-area, .qc-list-area * { box-sizing: border-box; }
                    .qc-list-area img { max-width: 100%; height: auto; }
                </style>'''))

                for patient_id, patient in sorted(self.patients.items()):
                    # Check if patient has any matching series
                    patient_has_match = any(
                        matches_filter(s, patient_id, sk)
                        for sk, study in patient.studies.items()
                        for s in study.series.values()
                    )
                    if not patient_has_match:
                        continue

                    for study_key, study in sorted(patient.studies.items(), key=lambda x: x[1].date or ''):
                        study_series = [s for s in study.series.values() if matches_filter(s, patient_id, study_key)]
                        if not study_series:
                            continue

                        # Series cards - modern, clean design (sorted numerically)
                        for series in sorted(study_series, key=lambda s: self._series_num_sort_key(s.series_number)):
                            is_selected = series == selected_series[0]
                            style = STATUS_STYLES.get(series.qc_status, STATUS_STYLES['PENDING'])
                            card_bg = style['bg']
                            border_color = style['border']

                            # Thumbnail - use percentage width so cards resize with panel
                            has_issues = series.qc_status in ('FAIL', 'WARNING', 'NOTE') and series.qc_report
                            if series.thumbnail:
                                img = f'<img src="data:image/png;base64,{series.thumbnail}" style="width:100%;max-width:100%;display:block;border-radius:6px;">'
                            elif series.is_derived:
                                img = f'<div style="height:60px;width:100%;background:linear-gradient(135deg,#4c1d95,#7c3aed);color:white;display:flex;align-items:center;justify-content:center;font-size:13px;border-radius:6px;font-weight:500;">{series.modality}</div>'
                            elif series.error:
                                img = f'<div style="height:60px;width:100%;background:#fee2e2;color:#991b1b;display:flex;align-items:center;justify-content:center;font-size:10px;padding:8px;text-align:center;border-radius:6px;box-sizing:border-box;">{series.error[:35]}...</div>'
                            else:
                                img = f'<div style="height:60px;width:100%;background:#e5e7eb;color:#6b7280;display:flex;align-items:center;justify-content:center;border-radius:6px;font-size:12px;">No preview</div>'

                            # Issue list
                            issue_html = ''
                            if has_issues:
                                issues = [r for r in series.qc_report.results if r.status in ('FAIL', 'WARNING', 'NOTE')]
                                if issues:
                                    issue_list = ', '.join(r.check_name for r in issues)
                                    issue_html = f'<div style="margin-top:8px;padding:6px 10px;font-size:11px;color:{style["text"]};background:rgba(0,0,0,0.04);border-radius:4px;word-wrap:break-word;">{issue_list}</div>'

                            # Badge colors
                            badge_bg = border_color
                            badge_text = '#fff' if series.qc_status != 'WARNING' else '#78350f'

                            # Build complete card HTML - status badge floats top-right
                            card_content = widgets.HTML(
                                f'''
                                <div style="position:relative;width:100%;box-sizing:border-box;">
                                    <span style="position:absolute;top:0;right:0;background:{badge_bg};color:{badge_text};padding:4px 10px;border-radius:4px;font-size:10px;font-weight:600;">{series.qc_status}</span>
                                    <div style="margin-bottom:10px;padding-right:75px;box-sizing:border-box;">
                                        <div style="display:flex;margin-bottom:3px;"><span style="color:#64748b;width:55px;flex-shrink:0;font-size:12px;">Subject</span><span style="font-size:12px;color:#1e293b;font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0;">{patient.label}</span></div>
                                        <div style="display:flex;margin-bottom:3px;"><span style="color:#64748b;width:55px;flex-shrink:0;font-size:12px;">Session</span><span style="font-size:12px;color:#1e293b;font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0;">{study.label}</span></div>
                                        <div style="display:flex;"><span style="color:#64748b;width:55px;flex-shrink:0;font-size:12px;">Series</span><span style="font-size:12px;color:#1e293b;font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0;">{series.label}</span></div>
                                    </div>
                                    {img}
                                    {issue_html}
                                </div>
                                ''',
                                layout=widgets.Layout(width='98%', min_width='0')
                            )

                            # Buttons
                            view_btn = widgets.Button(
                                description='● View' if is_selected else 'View',
                                button_style='success' if is_selected else 'info',
                                layout=widgets.Layout(flex='1', height='32px', min_width='0')
                            )

                            tags_btn = widgets.Button(
                                description='Tags',
                                button_style='',
                                layout=widgets.Layout(flex='1', height='32px', min_width='0')
                            )

                            def on_view_click(btn, s=series, p=patient, st=study):
                                update_selection(s)
                                hide_placeholder()
                                scroll_to_card(s)
                                show_viewer(s, p, st)

                            def on_header_click(btn, s=series, p=patient, st=study):
                                update_selection(s)
                                hide_placeholder()
                                scroll_to_card(s)
                                show_header(s, p, st)

                            view_btn.on_click(on_view_click)
                            tags_btn.on_click(on_header_click)

                            button_row = widgets.HBox(
                                [view_btn, tags_btn],
                                layout=widgets.Layout(margin='10px 0 0 0', gap='8px', width='100%')
                            )

                            # Complete card with background via CSS injection
                            card_id = f'qc-card-{id(series)}'
                            style_tag = widgets.HTML(f'''<style>
                                .{card_id} {{
                                    background: {card_bg} !important;
                                    border-radius: 8px;
                                    box-sizing: border-box !important;
                                    max-width: 100% !important;
                                }}
                                .{card_id} * {{ box-sizing: border-box; }}
                                .{card_id} img {{ max-width: 100%; height: auto; display: block; }}
                                .{card_id} .widget-html-content {{ width: 100%; }}
                            </style>''')

                            card = widgets.VBox(
                                [card_content, button_row],
                                layout=widgets.Layout(
                                    border=f'1px solid {border_color}',
                                    border_left=f'4px solid {border_color}',
                                    border_radius='8px',
                                    padding='12px',
                                    margin='0 0 10px 0',
                                    box_shadow=f'0 0 0 3px {border_color}' if is_selected else None,
                                )
                            )
                            card.add_class(card_id)

                            # Store card reference for selection updates (use id() since SeriesInfo isn't hashable)
                            card_widgets[id(series)] = {
                                'card': card,
                                'view_btn': view_btn,
                                'border_color': border_color
                            }

                            display(style_tag)
                            display(card)

        # Show placeholder message (panels are hidden initially)
        placeholder = widgets.HTML(
            '<div style="color:#64748b;padding:40px;text-align:center;font-size:14px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;min-height:700px;display:flex;align-items:center;justify-content:center;">Select a series to view</div>'
        )
        toggle_container.layout.display = 'none'  # Hide container initially

        # Panel references (will be set after panel creation)
        panel_refs = {'left': None, 'right': None}

        def hide_placeholder():
            """Hide placeholder and show viewer when content is shown."""
            placeholder.layout.display = 'none'
            toggle_container.layout.display = 'flex'
            # Show right panel, constrain left panel
            if panel_refs['right']:
                panel_refs['right'].layout.display = 'flex'
            if panel_refs['left']:
                panel_refs['left'].layout.width = '365px'
                panel_refs['left'].layout.flex = '0 0 auto'

        def close_viewer():
            """Close the viewer and expand thumbnail list to full width."""
            # Save header settings before closing
            if loaded_data['header_viewer'] is not None:
                loaded_data['header_settings'] = loaded_data['header_viewer'].get_settings()
            # Remember which card was selected before clearing
            previously_selected = selected_series[0]
            # Clear selection indicator
            update_selection(None)
            # Hide right panel, expand left panel
            if panel_refs['right']:
                panel_refs['right'].layout.display = 'none'
            if panel_refs['left']:
                panel_refs['left'].layout.width = '100%'
                panel_refs['left'].layout.flex = '1 1 auto'
            # Scroll to keep the previously viewed card visible
            scroll_to_card(previously_selected)

        # Wire up filter changes
        subject_dropdown.observe(update_session_options, names='value')
        subject_dropdown.observe(lambda _: render_list(), names='value')
        session_dropdown.observe(lambda _: render_list(), names='value')
        series_num_dropdown.observe(lambda _: render_list(), names='value')
        status_dropdown.observe(lambda _: render_list(), names='value')
        description_filter.observe(lambda _: render_list(), names='value')

        # Initial render
        render_list()

        # === Header: summary row + filter row ===
        summary_row = widgets.HTML(f'''
            <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:8px 0;">
                {summary_html}
            </div>
        ''')

        # Compact filter labels
        def make_filter(label, widget):
            lbl = widgets.HTML(f'<span style="font-size:11px;color:#64748b;margin-right:4px;">{label}</span>')
            return widgets.HBox([lbl, widget], layout=widgets.Layout(align_items='center'))

        filter_box = widgets.HBox([
            make_filter('Status', status_dropdown),
            make_filter('Subject', subject_dropdown),
            make_filter('Session', session_dropdown),
            make_filter('#', series_num_dropdown),
            description_filter,
        ], layout=widgets.Layout(align_items='center', gap='12px', flex_wrap='wrap'))

        header = widgets.VBox([summary_row, filter_box], layout=widgets.Layout(margin='0 0 12px 0', width='100%'))

        # Start with viewer closed (thumbnails full width)
        left_panel = widgets.VBox([list_area, script_runner], layout=widgets.Layout(width='100%', flex='1 1 auto'))
        right_panel = widgets.VBox([placeholder, toggle_container], layout=widgets.Layout(flex='1 1 auto', min_width='400px', display='none'))

        # Store panel references for close_viewer
        panel_refs['left'] = left_panel
        panel_refs['right'] = right_panel

        main_layout = widgets.HBox([left_panel, right_panel], layout=widgets.Layout(width='100%'))

        display(widgets.VBox([header, main_layout], layout=widgets.Layout(width='100%')))

    def display_grid(self, filter_status: Optional[str] = None):
        """Display thumbnail grid in Jupyter notebook (simple HTML version)."""
        from IPython.display import HTML, display
        display(HTML(self._notebook_grid_html(filter_status)))

    def _notebook_grid_html(self, filter_status: Optional[str] = None) -> str:
        """Generate HTML grid for notebook display."""
        c = self.STATUS_COLORS
        counts = self.get_summary()

        html = f'''
        <style>
            .qc-grid {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0; }}
            .qc-thumb {{ width: 340px; border: 3px solid #ccc; border-radius: 4px;
                         overflow: hidden; cursor: pointer; transition: transform 0.2s; background: #000; }}
            .qc-thumb:hover {{ transform: scale(1.02); box-shadow: 0 4px 12px rgba(0,0,0,0.3); }}
            .qc-thumb img {{ width: 100%; display: block; }}
            .qc-thumb .label {{ font-size: 11px; padding: 6px 8px; background: rgba(0,0,0,0.85);
                                color: white; display: flex; justify-content: space-between; align-items: center; }}
            .qc-thumb .label .desc {{ white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
                                       flex: 1; margin-right: 8px; }}
            .qc-thumb .label .status {{ padding: 2px 6px; border-radius: 8px; font-size: 9px; font-weight: 600; }}
            .qc-thumb.pass {{ border-color: {c['PASS']}; }} .qc-thumb.pass .status {{ background: {c['PASS']}; }}
            .qc-thumb.warning {{ border-color: {c['WARNING']}; }} .qc-thumb.warning .status {{ background: {c['WARNING']}; color: #333; }}
            .qc-thumb.fail {{ border-color: {c['FAIL']}; }} .qc-thumb.fail .status {{ background: {c['FAIL']}; }}
            .qc-thumb.error {{ border-color: {c['ERROR']}; }} .qc-thumb.error .status {{ background: {c['ERROR']}; }}
            .patient-section {{ margin: 20px 0; }}
            .patient-header {{ font-size: 16px; font-weight: bold; margin: 10px 0; padding: 10px 12px;
                               background: #343a40; color: white; border-radius: 4px; }}
            .study-header {{ font-size: 14px; color: #495057; margin: 12px 0 8px 0; padding: 6px 10px;
                             background: #e9ecef; border-radius: 4px; border-left: 4px solid #007bff; }}
            .summary-bar {{ display: flex; gap: 20px; margin: 15px 0; padding: 12px 15px;
                            background: #f8f9fa; border-radius: 4px; border: 1px solid #dee2e6; }}
            .summary-item {{ display: flex; align-items: center; gap: 6px; font-weight: 500; }}
            .summary-dot {{ width: 14px; height: 14px; border-radius: 50%; }}
        </style>
        <div class="summary-bar">'''

        for status, count in counts.items():
            if status != 'PENDING':
                html += f'<div class="summary-item"><div class="summary-dot" style="background:{c[status]}"></div><span>{status}: {count}</span></div>'
        html += '</div>'

        for patient_id, patient in sorted(self.patients.items()):
            html += f'<div class="patient-section"><div class="patient-header">{patient.label}</div>'

            for study_uid, study in sorted(patient.studies.items(), key=lambda x: x[1].date):
                html += f'<div class="study-header">{study.label}</div><div class="qc-grid">'

                for series_uid, series in sorted(study.series.items(), key=lambda item: self._series_num_sort_key(item[1].series_number)):
                    status = series.qc_status.lower()
                    if filter_status and status != filter_status.lower():
                        continue

                    if series.thumbnail:
                        img_src = f'data:image/png;base64,{series.thumbnail}'
                    else:
                        img_src = ('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="340" height="113">'
                                   '<rect fill="%23333" width="340" height="113"/>'
                                   '<text x="170" y="56" fill="%23999" text-anchor="middle" dy=".3em">No image</text></svg>')

                    html += f'''<div class="qc-thumb {status}" title="{series.description}&#10;{len(series.files)} slices">
                        <img src="{img_src}"><div class="label"><span class="desc">{series.label}</span>
                        <span class="status">{series.qc_status}</span></div></div>'''

                html += '</div>'
            html += '</div>'

        return html
