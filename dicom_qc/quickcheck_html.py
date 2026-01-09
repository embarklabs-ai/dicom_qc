"""HTML report generation for QuickCheck."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .quickcheck import SeriesInfo


class QuickCheckHTMLMixin:
    """Mixin providing HTML report generation methods for QuickCheck."""

    def generate_html_report(self, output_path: Path) -> str:
        """Generate a self-contained HTML report with 3-pane thumbnail grid."""
        counts = self.get_summary()
        total_patients = len(self.patients)
        total_studies = sum(len(p.studies) for p in self.patients.values())
        total_series = len(self.get_all_series())

        # Collect unique check names that have issues
        check_names = set()
        for series in self.get_all_series():
            if series.qc_report:
                for r in series.qc_report.results:
                    if r.status in ('FAIL', 'WARNING', 'NOTE'):
                        check_names.add(r.check_name)
        check_names = sorted(check_names)

        html = self._html_header(counts, total_patients, total_studies, total_series, check_names)
        html += self._html_patient_sections()
        html += self._html_footer()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding='utf-8')

        return html

    def _html_header(self, counts: Dict[str, int], n_patients: int, n_studies: int, n_series: int,
                     check_names: List[str] = None) -> str:
        """Generate HTML header with styles and summary."""
        c = self.STATUS_COLORS
        project_prefix = f"{self._xnat_project_id} - " if self._xnat_project_id else ""
        check_names = check_names or []
        return f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{project_prefix}DICOM Quickcheck Report</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 1600px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #343a40, #495057); color: white;
                   padding: 25px; border-radius: 8px; margin-bottom: 20px; }}
        .header h1 {{ margin: 0 0 10px 0; }}
        .header p {{ margin: 5px 0; opacity: 0.9; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                    gap: 15px; margin-bottom: 25px; }}
        .summary-card {{ background: white; padding: 20px; border-radius: 8px; text-align: center;
                         box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .summary-card h2 {{ margin: 0; font-size: 32px; }}
        .summary-card p {{ margin: 5px 0 0 0; color: #666; }}
        .patient-section {{ background: white; border-radius: 8px; margin-bottom: 20px;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1); overflow: hidden; }}
        .patient-header {{ background: #343a40; color: white; padding: 15px 20px;
                           font-size: 18px; font-weight: 600; }}
        .study-section {{ padding: 15px 20px; border-bottom: 1px solid #eee; }}
        .study-section:last-child {{ border-bottom: none; }}
        .study-header {{ font-size: 14px; color: #495057; margin-bottom: 12px; padding: 6px 10px;
                         background: #e9ecef; border-radius: 4px; border-left: 4px solid #007bff; }}
        .qc-grid {{ display: flex; flex-wrap: wrap; gap: 12px; }}
        .qc-thumb {{ width: 340px; border: 3px solid #ccc; border-radius: 6px;
                     overflow: hidden; background: #000; }}
        .qc-thumb img {{ width: 100%; display: block; }}
        .qc-thumb .info {{ padding: 8px 10px; font-size: 11px; background: rgba(0,0,0,0.9);
                           color: white; display: flex; justify-content: space-between; align-items: center; }}
        .qc-thumb .info .series-label {{ flex: 1; white-space: nowrap; overflow: hidden;
                                          text-overflow: ellipsis; margin-right: 10px; }}
        .qc-thumb.pass {{ border-color: {c['PASS']}; }}
        .qc-thumb.warning {{ border-color: {c['WARNING']}; }}
        .qc-thumb.fail {{ border-color: {c['FAIL']}; }}
        .qc-thumb.error {{ border-color: {c['ERROR']}; }}
        .qc-thumb.derived {{ border-color: {c['DERIVED']}; }}
        .qc-thumb.note {{ border-color: {c['NOTE']}; }}
        .status-badge {{ display: inline-block; padding: 3px 8px; border-radius: 10px;
                         font-size: 10px; font-weight: 600; color: white; flex-shrink: 0; }}
        .status-badge.pass {{ background: {c['PASS']}; }}
        .status-badge.warning {{ background: {c['WARNING']}; color: #333; }}
        .status-badge.fail {{ background: {c['FAIL']}; }}
        .status-badge.error {{ background: {c['ERROR']}; }}
        .status-badge.derived {{ background: {c['DERIVED']}; }}
        .status-badge.note {{ background: {c['NOTE']}; }}
        .filter-bar {{ background: white; padding: 15px 20px; border-radius: 8px;
                       margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .filter-bar label {{ margin-right: 15px; cursor: pointer; }}
        .hidden {{ display: none !important; }}
        @media print {{ body {{ background: white; }} .filter-bar {{ display: none; }}
                        .qc-thumb {{ break-inside: avoid; }} }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{project_prefix}DICOM Quickcheck Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>{n_patients} patients | {n_studies} studies | {n_series} series</p>
    </div>
    <div class="summary">
        <div class="summary-card"><h2>{n_series}</h2><p>Total Series</p></div>
        <div class="summary-card" style="border-top: 4px solid {c['PASS']};"><h2>{counts['PASS']}</h2><p>Pass</p></div>
        <div class="summary-card" style="border-top: 4px solid {c['WARNING']};"><h2>{counts['WARNING']}</h2><p>Warning</p></div>
        <div class="summary-card" style="border-top: 4px solid {c['FAIL']};"><h2>{counts['FAIL']}</h2><p>Fail</p></div>
        <div class="summary-card" style="border-top: 4px solid {c['ERROR']};"><h2>{counts['ERROR']}</h2><p>Error</p></div>
        <div class="summary-card" style="border-top: 4px solid {c['NOTE']};"><h2>{counts['NOTE']}</h2><p>Note</p></div>
        <div class="summary-card" style="border-top: 4px solid {c['DERIVED']};"><h2>{counts['DERIVED']}</h2><p>Derived</p></div>
    </div>
    <div class="filter-bar">
        <strong>Filter:</strong>
        <label><input type="checkbox" id="filter-all" checked onchange="toggleAll(this)"> All</label>
        <label><input type="checkbox" checked class="filter-cb" data-status="pass" onchange="updateFilter()"> Pass</label>
        <label><input type="checkbox" checked class="filter-cb" data-status="warning" onchange="updateFilter()"> Warning</label>
        <label><input type="checkbox" checked class="filter-cb" data-status="fail" onchange="updateFilter()"> Fail</label>
        <label><input type="checkbox" checked class="filter-cb" data-status="error" onchange="updateFilter()"> Error</label>
        <label><input type="checkbox" checked class="filter-cb" data-status="note" onchange="updateFilter()"> Note</label>
        <label><input type="checkbox" checked class="filter-cb" data-status="derived" onchange="updateFilter()"> Derived</label>
        <span style="margin-left:20px;">
            <strong>Check:</strong>
            <select id="check-filter" onchange="updateFilter()" style="padding:3px 8px;border-radius:4px;border:1px solid #ccc;">
                <option value="">All Checks</option>
                {''.join(f'<option value="{name}">{name}</option>' for name in check_names)}
            </select>
        </span>
        <span style="color:#888;font-size:12px;margin-left:20px;">(In Jupyter: click "Trust HTML" for filters. Open in browser for links to work.)</span>
    </div>
'''

    def _html_patient_sections(self) -> str:
        """Generate HTML for patient/study/series sections."""
        def series_sort_key(item):
            """Sort by series number numerically (handle str/int/None)."""
            series = item[1]
            try:
                return (0, int(series.series_number))
            except (ValueError, TypeError):
                # Non-numeric or None - sort at the end
                return (1, str(series.series_number or ''))

        html = ''
        for patient_id, patient in sorted(self.patients.items()):
            html += f'    <div class="patient-section">\n'
            html += f'        <div class="patient-header">{patient.label}</div>\n'

            for study_uid, study in sorted(patient.studies.items(), key=lambda x: x[1].date):
                html += f'        <div class="study-section">\n'
                html += f'            <div class="study-header">{study.label}</div>\n'
                html += '            <div class="qc-grid">\n'

                # Build OHIF URL context for this study
                ohif_url = None
                if self._xnat_mode and self._xnat_base_url and self._xnat_project_id:
                    subj_id = patient.xnat_subject_id
                    exp_id = study.xnat_experiment_id
                    exp_label = study.xnat_session_label
                    if subj_id and exp_id:
                        ohif_url = (f"{self._xnat_base_url}/VIEWER/?"
                                    f"subjectId={subj_id}&projectId={self._xnat_project_id}"
                                    f"&experimentId={exp_id}&experimentLabel={exp_label}")

                for series_uid, series in sorted(study.series.items(), key=series_sort_key):
                    html += self._html_series_thumb(series, ohif_url)

                html += '            </div>\n'
                html += '        </div>\n'
            html += '    </div>\n'
        return html

    def _html_series_thumb(self, series: 'SeriesInfo', ohif_url: Optional[str] = None) -> str:
        """Generate HTML for a single series thumbnail.

        Args:
            series: SeriesInfo to render
            ohif_url: Optional OHIF viewer URL - if provided, thumbnail becomes clickable
        """
        status = series.qc_status.lower()
        if series.thumbnail:
            img_src = f'data:image/png;base64,{series.thumbnail}'
        elif series.is_derived:
            # SVG placeholder for derived types
            img_src = (f'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="340" height="113">'
                       f'<rect fill="%232d1f3d" width="340" height="113"/>'
                       f'<text x="170" y="56" fill="%239c27b0" text-anchor="middle" dy=".3em" font-size="16">{series.modality}</text></svg>')
        else:
            img_src = ('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="340" height="113">'
                       '<rect fill="%23333" width="340" height="113"/>'
                       '<text x="170" y="56" fill="%23999" text-anchor="middle" dy=".3em">No image</text></svg>')

        # Build info/reason text
        reason_html = ''
        if series.error:
            reason_html = f'<div class="qc-reason" style="padding:6px 10px;font-size:11px;background:#f8d7da;color:#721c24;">Error: {series.error}</div>'
        elif series.is_derived:
            # Show derived info
            info_parts = []
            if series.derived_info:
                info_parts.append(series.derived_info)
            if series.referenced_series_uid:
                info_parts.append(f'→ {series.referenced_series_uid[:30]}...')
            if info_parts:
                reason_html = f'<div class="qc-reason" style="padding:6px 10px;font-size:11px;background:#2d1f3d;color:#e0c3fc;">{"<br>".join(info_parts)}</div>'
        elif series.qc_report and series.qc_status in ('FAIL', 'WARNING', 'NOTE'):
            issues = [r for r in series.qc_report.results if r.status in ('FAIL', 'WARNING', 'NOTE')]
            if issues:
                issue_lines = [f'<div style="margin:2px 0;"><b>{r.check_name}:</b> {r.message}</div>' for r in issues]
                if series.qc_status == 'FAIL':
                    bg_color, text_color = '#f8d7da', '#721c24'
                elif series.qc_status == 'WARNING':
                    bg_color, text_color = '#fff3cd', '#856404'
                else:  # NOTE
                    bg_color, text_color = '#d1ecf1', '#0c5460'
                reason_html = f'<div class="qc-reason" style="padding:6px 10px;font-size:11px;background:{bg_color};color:{text_color};">{"".join(issue_lines)}</div>'

        # Wrap in link if OHIF URL provided
        link_style = 'cursor:pointer;' if ohif_url else ''
        link_start = f'<a href="{ohif_url}" target="_blank" style="text-decoration:none;color:inherit;">' if ohif_url else ''
        link_end = '</a>' if ohif_url else ''

        # Collect check names for filtering
        series_check_names = []
        if series.qc_report:
            series_check_names = [r.check_name for r in series.qc_report.results if r.status in ('FAIL', 'WARNING', 'NOTE')]
        data_checks = ','.join(series_check_names)

        return f'''                {link_start}<div class="qc-thumb {status}" data-status="{status}" data-checks="{data_checks}" style="{link_style}">
                    <img src="{img_src}" alt="{series.description}">
                    <div class="info">
                        <span class="series-label">{series.label}</span>
                        <span class="status-badge {status}">{series.qc_status}</span>
                    </div>
                    {reason_html}
                </div>{link_end}
'''

    def _html_footer(self) -> str:
        """Generate HTML footer with JavaScript."""
        return '''
    <details style="background:white;border-radius:8px;padding:15px 20px;margin-top:20px;box-shadow:0 2px 4px rgba(0,0,0,0.1);">
        <summary style="cursor:pointer;font-weight:600;font-size:16px;color:#333;">QC Checks Reference</summary>
        <div style="margin-top:15px;font-size:13px;line-height:1.6;">
            <h4 style="margin:15px 0 10px 0;color:#495057;">Geometry Checks</h4>
            <table style="width:100%;border-collapse:collapse;font-size:12px;">
                <tr style="background:#f8f9fa;"><th style="text-align:left;padding:8px;border:1px solid #dee2e6;">Check</th><th style="text-align:left;padding:8px;border:1px solid #dee2e6;">Logic</th><th style="text-align:left;padding:8px;border:1px solid #dee2e6;">Rationale</th></tr>
                <tr><td style="padding:8px;border:1px solid #dee2e6;"><b>Geometry Metadata</b></td><td style="padding:8px;border:1px solid #dee2e6;">FAIL if PixelSpacing, ImageOrientation, or ImagePosition missing</td><td style="padding:8px;border:1px solid #dee2e6;">Derived images may lose geometry tags, making 3D reconstruction unreliable</td></tr>
                <tr><td style="padding:8px;border:1px solid #dee2e6;"><b>4D Data</b></td><td style="padding:8px;border:1px solid #dee2e6;">NOTE if num_timepoints &gt; 1</td><td style="padding:8px;border:1px solid #dee2e6;">fMRI/DTI/DSC has multiple timepoints; viewer shows first volume only</td></tr>
                <tr><td style="padding:8px;border:1px solid #dee2e6;"><b>Reconstructability</b></td><td style="padding:8px;border:1px solid #dee2e6;">FAIL if multiple orientations or slices not parallel</td><td style="padding:8px;border:1px solid #dee2e6;">Multi-plane localizers cannot form coherent 3D volumes</td></tr>
                <tr><td style="padding:8px;border:1px solid #dee2e6;"><b>Slice Ordering</b></td><td style="padding:8px;border:1px solid #dee2e6;">FAIL if non-monotonic or duplicate slice locations</td><td style="padding:8px;border:1px solid #dee2e6;">Non-monotonic order indicates data corruption or mixed series</td></tr>
                <tr><td style="padding:8px;border:1px solid #dee2e6;"><b>Orientation Consistency</b></td><td style="padding:8px;border:1px solid #dee2e6;">WARNING if direction vectors not unit length or orthogonal</td><td style="padding:8px;border:1px solid #dee2e6;">Invalid orientation causes incorrect spatial positioning</td></tr>
                <tr><td style="padding:8px;border:1px solid #dee2e6;"><b>Frame of Reference</b></td><td style="padding:8px;border:1px solid #dee2e6;">WARNING if slices deviate &gt;1mm from expected plane</td><td style="padding:8px;border:1px solid #dee2e6;">Non-coplanar slices cause reconstruction artifacts</td></tr>
                <tr><td style="padding:8px;border:1px solid #dee2e6;"><b>Gap Detection</b></td><td style="padding:8px;border:1px solid #dee2e6;">FAIL if gap &gt;1.5× expected; WARNING if spacing CV &gt;10%</td><td style="padding:8px;border:1px solid #dee2e6;">Missing slices cause interpolation artifacts in reformats</td></tr>
                <tr><td style="padding:8px;border:1px solid #dee2e6;"><b>Voxel Anisotropy</b></td><td style="padding:8px;border:1px solid #dee2e6;">WARNING if max/min voxel ratio &gt;2× (moderate) or &gt;4× (severe)</td><td style="padding:8px;border:1px solid #dee2e6;">Thick slices look blurry in non-acquisition plane reformats</td></tr>
                <tr><td style="padding:8px;border:1px solid #dee2e6;"><b>Slice Count</b></td><td style="padding:8px;border:1px solid #dee2e6;">WARNING if &lt;3 slices or &lt;80% of expected count</td><td style="padding:8px;border:1px solid #dee2e6;">Few slices suggest localizer or truncated transfer</td></tr>
            </table>
            <h4 style="margin:20px 0 10px 0;color:#495057;">DICOM-Level Checks</h4>
            <table style="width:100%;border-collapse:collapse;font-size:12px;">
                <tr style="background:#f8f9fa;"><th style="text-align:left;padding:8px;border:1px solid #dee2e6;">Check</th><th style="text-align:left;padding:8px;border:1px solid #dee2e6;">Logic</th><th style="text-align:left;padding:8px;border:1px solid #dee2e6;">Rationale</th></tr>
                <tr><td style="padding:8px;border:1px solid #dee2e6;"><b>JPEG-2000 Encoding</b></td><td style="padding:8px;border:1px solid #dee2e6;">WARNING if multi-layer (&gt;1) JPEG-2000 encoding detected</td><td style="padding:8px;border:1px solid #dee2e6;">Multi-layer J2K causes blurry rendering in OHIF viewer</td></tr>
                <tr><td style="padding:8px;border:1px solid #dee2e6;"><b>Temporal Metadata</b></td><td style="padding:8px;border:1px solid #dee2e6;">WARNING if dynamic series missing TriggerTime, AcquisitionTime, etc.</td><td style="padding:8px;border:1px solid #dee2e6;">Missing temporal tags cause failures in strict 4D viewers</td></tr>
            </table>
        </div>
    </details>
    <script>
        function updateFilter() {
            var checkboxes = document.querySelectorAll('.filter-cb');
            var activeStatuses = [];
            for (var i = 0; i < checkboxes.length; i++) {
                if (checkboxes[i].checked) {
                    activeStatuses.push(checkboxes[i].getAttribute('data-status'));
                }
            }
            // Get selected check type
            var checkFilter = document.getElementById('check-filter').value;

            var thumbs = document.querySelectorAll('.qc-thumb');
            for (var j = 0; j < thumbs.length; j++) {
                var status = thumbs[j].getAttribute('data-status');
                var checks = thumbs[j].getAttribute('data-checks') || '';

                // Must match status filter
                var statusMatch = activeStatuses.indexOf(status) >= 0;

                // Must match check filter (if set)
                var checkMatch = true;
                if (checkFilter) {
                    checkMatch = checks.split(',').indexOf(checkFilter) >= 0;
                }

                if (statusMatch && checkMatch) {
                    thumbs[j].classList.remove('hidden');
                } else {
                    thumbs[j].classList.add('hidden');
                }
            }
            // Update "All" checkbox state
            var allCheckbox = document.getElementById('filter-all');
            var allChecked = true;
            for (var k = 0; k < checkboxes.length; k++) {
                if (!checkboxes[k].checked) { allChecked = false; break; }
            }
            allCheckbox.checked = allChecked;
        }
        function toggleAll(checkbox) {
            var checkboxes = document.querySelectorAll('.filter-cb');
            for (var i = 0; i < checkboxes.length; i++) {
                checkboxes[i].checked = checkbox.checked;
            }
            updateFilter();
        }
    </script>
</body>
</html>
'''
