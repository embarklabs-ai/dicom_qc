"""HTML report generation for QuickCheck."""

from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import shutil

from .quickcheck import _defaced_badge_html

if TYPE_CHECKING:
    from .quickcheck import SeriesInfo


def _series_sort_key(item: Tuple[str, Any]) -> Tuple[int, Any]:
    """Sort key for series that puts numeric series numbers first, then strings.

    Args:
        item: Tuple of (series_uid, series) from dict.items()

    Returns:
        Sort key tuple (type_priority, value) where numeric values come first
    """
    series = item[1]
    try:
        return (0, int(series.series_number))
    except (ValueError, TypeError):
        return (1, str(series.series_number or ""))


class QuickCheckHTMLMixin:
    """Mixin providing HTML report generation methods for QuickCheck."""

    def _build_ohif_url(self, patient, study) -> Optional[str]:
        """Build OHIF viewer URL for a study if in XNAT mode.

        Args:
            patient: PatientInfo object with xnat_subject_id
            study: StudyInfo object with xnat_experiment_id and xnat_session_label

        Returns:
            OHIF viewer URL string or None if not in XNAT mode
        """
        if not (self._xnat_mode and self._xnat_base_url and self._xnat_project_id):
            return None

        subj_id = patient.xnat_subject_id
        exp_id = study.xnat_experiment_id
        exp_label = study.xnat_session_label

        if not (subj_id and exp_id):
            return None

        return (
            f"{self._xnat_base_url}/VIEWER/?"
            f"subjectId={subj_id}&projectId={self._xnat_project_id}"
            f"&experimentId={exp_id}&experimentLabel={exp_label}"
        )

    def generate_html_report(
        self,
        output_path: Path,
        series_per_page: int = 500,
        embed_thumbnails: bool = None,
    ) -> tuple:
        """Generate HTML report for QC review.

        For small datasets (<=500 series), generates a single self-contained HTML file.
        For large datasets, generates HTML + thumbnails folder, zipped for sharing.

        Args:
            output_path: Output path. For embedded reports, this is the HTML file path.
                        For external thumbnails, a directory is created here.
            series_per_page: Ignored (kept for API compatibility)
            embed_thumbnails: If True, generate single self-contained HTML.
                            If False, HTML + external thumbnails folder + zip.
                            If None, auto-select based on series count (<=500 embeds).

        Returns:
            Tuple of (html_path, zip_path). zip_path is None for embedded single-file reports.
        """
        total_series = len(self.get_all_series())

        # Auto-select embedding for small datasets
        if embed_thumbnails is None:
            embed_thumbnails = total_series <= 500

        if embed_thumbnails:
            # Single self-contained file
            self._generate_single_page_report(output_path)
            return (Path(output_path), None)
        else:
            # Multi-page report with external thumbnails
            output_path = Path(output_path)

            # Use directory name based on output_path
            if output_path.suffix in (".html", ".zip"):
                report_dir = output_path.with_suffix("")
            else:
                report_dir = output_path

            self._generate_multi_page_report(report_dir, series_per_page)

            # Create zip for sharing
            zip_path = report_dir.with_suffix(".zip")
            shutil.make_archive(
                str(report_dir), "zip", report_dir.parent, report_dir.name
            )

            html_path = report_dir / "index.html"
            return (html_path, zip_path)

    def _generate_single_page_report(self, output_path: Path) -> str:
        """Generate single-page self-contained HTML report (legacy)."""
        counts = self.get_summary()
        total_patients = len(self.patients)
        total_studies = sum(len(p.studies) for p in self.patients.values())
        total_series = len(self.get_all_series())

        # Collect unique check names that have issues
        check_names = set()
        for series in self.get_all_series():
            if series.qc_report:
                for r in series.qc_report.results:
                    if r.status in ("FAIL", "WARNING", "NOTE"):
                        check_names.add(r.check_name)
        check_names = sorted(check_names)

        html = self._html_header(
            counts, total_patients, total_studies, total_series, check_names
        )
        html += self._html_patient_sections()
        html += self._html_footer()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")

        return html

    def _generate_multi_page_report(
        self,
        output_dir: Path,
        series_per_page: int = 500,
    ) -> List[Path]:
        """Generate HTML report with external thumbnails.

        Creates:
        - index.html: Full report with all series
        - thumbnails/: Directory with thumbnail files

        Args:
            output_dir: Directory to write report files
            series_per_page: Ignored (kept for API compatibility)

        Returns:
            List of generated file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create thumbnails directory
        thumb_out = output_dir / "thumbnails"
        thumb_out.mkdir(exist_ok=True)

        # Copy thumbnails to output directory
        thumb_paths = {}  # series_uid -> relative path
        thumb_src_dir = (
            self.data_dir / "_dicom_qc" / "thumbnails" if self.data_dir else None
        )
        for series in self.get_all_series():
            thumb_rel_path = None
            if (
                hasattr(series, "_thumbnail_path")
                and series._thumbnail_path
                and thumb_src_dir
            ):
                src = thumb_src_dir / series._thumbnail_path
                if src.exists():
                    thumb_filename = series._thumbnail_path.replace("/", "_")
                    dst = thumb_out / thumb_filename
                    shutil.copy(src, dst)
                    thumb_rel_path = f"thumbnails/{thumb_filename}"
            if not thumb_rel_path and series.thumbnail:
                # Convert base64 to file
                import base64
                import hashlib

                thumb_hash = hashlib.sha256(series.uid.encode()).hexdigest()[:16]
                thumb_filename = f"{thumb_hash}.jpg"
                dst = thumb_out / thumb_filename
                try:
                    data = base64.b64decode(series.thumbnail)
                    dst.write_bytes(data)
                    thumb_rel_path = f"thumbnails/{thumb_filename}"
                except Exception:
                    pass
            thumb_paths[series.uid] = thumb_rel_path

        # Store for use in HTML generation
        self._external_thumb_paths = thumb_paths

        # Generate single HTML file with external thumbnails
        counts = self.get_summary()
        total_patients = len(self.patients)
        total_studies = sum(len(p.studies) for p in self.patients.values())
        total_series = len(self.get_all_series())

        # Collect check names for filter dropdown
        check_names = set()
        for series in self.get_all_series():
            if series.qc_report:
                for r in series.qc_report.results:
                    if r.status in ("FAIL", "WARNING", "NOTE"):
                        check_names.add(r.check_name)
        check_names = sorted(check_names)

        html = self._html_header(
            counts, total_patients, total_studies, total_series, check_names
        )
        html += self._html_patient_sections_external()
        html += self._html_footer()

        # Clean up temp attribute
        del self._external_thumb_paths

        index_path = output_dir / "index.html"
        index_path.write_text(html, encoding="utf-8")

        return [index_path]

    def _html_header(
        self,
        counts: Dict[str, int],
        n_patients: int,
        n_studies: int,
        n_series: int,
        check_names: List[str] = None,
    ) -> str:
        """Generate HTML header with styles and summary."""
        c = self.STATUS_COLORS
        project_prefix = f"{self._xnat_project_id} - " if self._xnat_project_id else ""
        check_names = check_names or []
        return f"""<!DOCTYPE html>
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
        .qc-thumb.pass {{ border-color: {c["PASS"]}; }}
        .qc-thumb.warning {{ border-color: {c["WARNING"]}; }}
        .qc-thumb.fail {{ border-color: {c["FAIL"]}; }}
        .qc-thumb.error {{ border-color: {c["ERROR"]}; }}
        .qc-thumb.derived {{ border-color: {c["DERIVED"]}; }}
        .qc-thumb.note {{ border-color: {c["NOTE"]}; }}
        .status-badge {{ display: inline-block; padding: 3px 8px; border-radius: 10px;
                         font-size: 10px; font-weight: 600; color: white; flex-shrink: 0; }}
        .status-badge.pass {{ background: {c["PASS"]}; }}
        .status-badge.warning {{ background: {c["WARNING"]}; color: #333; }}
        .status-badge.fail {{ background: {c["FAIL"]}; }}
        .status-badge.error {{ background: {c["ERROR"]}; }}
        .status-badge.derived {{ background: {c["DERIVED"]}; }}
        .status-badge.note {{ background: {c["NOTE"]}; }}
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
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p>{n_patients} patients | {n_studies} studies | {n_series} series</p>
    </div>
    <div class="summary">
        <div class="summary-card"><h2>{n_series}</h2><p>Total Series</p></div>
        <div class="summary-card" style="border-top: 4px solid {c["PASS"]};"><h2>{counts["PASS"]}</h2><p>Pass</p></div>
        <div class="summary-card" style="border-top: 4px solid {c["WARNING"]};"><h2>{counts["WARNING"]}</h2><p>Warning</p></div>
        <div class="summary-card" style="border-top: 4px solid {c["FAIL"]};"><h2>{counts["FAIL"]}</h2><p>Fail</p></div>
        <div class="summary-card" style="border-top: 4px solid {c["ERROR"]};"><h2>{counts["ERROR"]}</h2><p>Error</p></div>
        <div class="summary-card" style="border-top: 4px solid {c["NOTE"]};"><h2>{counts["NOTE"]}</h2><p>Note</p></div>
        <div class="summary-card" style="border-top: 4px solid {c["DERIVED"]};"><h2>{counts["DERIVED"]}</h2><p>Derived</p></div>
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
                {"".join(f'<option value="{escape(name, quote=True)}">{escape(name)}</option>' for name in check_names)}
            </select>
        </span>
        <span style="color:#888;font-size:12px;margin-left:20px;">(In Jupyter: click "Trust HTML" for filters. Open in browser for links to work.)</span>
    </div>
"""

    def _html_patient_sections(self) -> str:
        """Generate HTML for patient/study/series sections."""
        html = ""
        for patient_id, patient in sorted(self.patients.items()):
            html += '    <div class="patient-section">\n'
            html += (
                f'        <div class="patient-header">{escape(patient.label)}</div>\n'
            )

            for study_uid, study in sorted(
                patient.studies.items(), key=lambda x: x[1].date or ""
            ):
                html += '        <div class="study-section">\n'
                html += f'            <div class="study-header">{escape(study.label)}</div>\n'
                html += '            <div class="qc-grid">\n'

                ohif_url = self._build_ohif_url(patient, study)

                for series_uid, series in sorted(
                    study.series.items(), key=_series_sort_key
                ):
                    html += self._html_series_thumb(series, ohif_url)

                html += "            </div>\n"
                html += "        </div>\n"
            html += "    </div>\n"
        return html

    def _html_patient_sections_external(self) -> str:
        """Generate HTML for patient/study/series sections with external thumbnails."""
        html = ""
        for patient_id, patient in sorted(self.patients.items()):
            html += '    <div class="patient-section">\n'
            html += (
                f'        <div class="patient-header">{escape(patient.label)}</div>\n'
            )

            for study_uid, study in sorted(
                patient.studies.items(), key=lambda x: x[1].date or ""
            ):
                html += '        <div class="study-section">\n'
                html += f'            <div class="study-header">{escape(study.label)}</div>\n'
                html += '            <div class="qc-grid">\n'

                ohif_url = self._build_ohif_url(patient, study)

                for series_uid, series in sorted(
                    study.series.items(), key=_series_sort_key
                ):
                    thumb_path = self._external_thumb_paths.get(series.uid)
                    html += self._html_series_thumb_external(
                        series, thumb_path, ohif_url
                    )

                html += "            </div>\n"
                html += "        </div>\n"
            html += "    </div>\n"
        return html

    def _html_series_thumb_external(
        self,
        series: "SeriesInfo",
        thumb_path: Optional[str],
        ohif_url: Optional[str] = None,
    ) -> str:
        """Generate HTML for a series thumbnail using external file path."""
        status = series.qc_status.lower()

        if thumb_path:
            img_html = f'<img src="{thumb_path}" alt="{escape(series.description or "", quote=True)}">'
        elif series.is_derived:
            img_html = (
                f'<div style="height:113px;background:#2d1f3d;color:#9c27b0;'
                f'display:flex;align-items:center;justify-content:center;font-size:16px;">'
                f"{escape(series.modality)}</div>"
            )
        elif series.error:
            error_msg = series.error[:50] if len(series.error) > 50 else series.error
            img_html = (
                f'<div style="height:113px;background:#333;color:#999;'
                f"display:flex;align-items:center;justify-content:center;"
                f'font-size:12px;padding:10px;text-align:center;">{escape(error_msg)}</div>'
            )
        else:
            img_html = (
                '<div style="height:113px;background:#333;color:#999;'
                'display:flex;align-items:center;justify-content:center;">No image</div>'
            )

        # Build reason HTML
        reason_html = ""
        if series.error:
            reason_html = (
                f'<div style="padding:6px 10px;font-size:11px;'
                f'background:#f8d7da;color:#721c24;">Error: {escape(series.error[:50])}</div>'
            )
        elif series.is_derived and series.derived_info:
            reason_html = (
                f'<div style="padding:6px 10px;font-size:11px;'
                f'background:#2d1f3d;color:#e0c3fc;">{escape(series.derived_info)}</div>'
            )
        elif series.qc_report and series.qc_status in ("FAIL", "WARNING", "NOTE"):
            issues = [
                r
                for r in series.qc_report.results
                if r.status in ("FAIL", "WARNING", "NOTE")
            ]
            if issues:
                issue_lines = [
                    f"<b>{escape(r.check_name)}:</b> {escape(r.message)}"
                    for r in issues[:3]
                ]
                if series.qc_status == "FAIL":
                    bg_color, text_color = "#f8d7da", "#721c24"
                elif series.qc_status == "WARNING":
                    bg_color, text_color = "#fff3cd", "#856404"
                else:
                    bg_color, text_color = "#d1ecf1", "#0c5460"
                reason_html = (
                    f'<div style="padding:6px 10px;font-size:11px;'
                    f'background:{bg_color};color:{text_color};">'
                    f"{' | '.join(issue_lines)}</div>"
                )

        # Link wrapper
        link_style = "cursor:pointer;" if ohif_url else ""
        link_start = (
            f'<a href="{ohif_url}" target="_blank" style="text-decoration:none;color:inherit;">'
            if ohif_url
            else ""
        )
        link_end = "</a>" if ohif_url else ""

        # Check names for filtering
        series_check_names = []
        if series.qc_report:
            series_check_names = [
                r.check_name
                for r in series.qc_report.results
                if r.status in ("FAIL", "WARNING", "NOTE")
            ]
        data_checks = ",".join(series_check_names)

        defaced_badge = ""
        if getattr(series, "_xnat_resource", None) == "DEFACED":
            defaced_badge = _defaced_badge_html("padding:3px 8px;border-radius:10px;flex-shrink:0;")

        return f'''                {link_start}<div class="qc-thumb {status}" data-status="{status}" data-checks="{data_checks}" style="{link_style}">
                    {img_html}
                    <div class="info">
                        <span class="series-label">{escape(series.label)}</span>
                        {defaced_badge}
                        <span class="status-badge {status}">{series.qc_status}</span>
                    </div>
                    {reason_html}
                </div>{link_end}
'''

    def _html_series_thumb(
        self, series: "SeriesInfo", ohif_url: Optional[str] = None
    ) -> str:
        """Generate HTML for a single series thumbnail.

        Args:
            series: SeriesInfo to render
            ohif_url: Optional OHIF viewer URL - if provided, thumbnail becomes clickable
        """
        status = series.qc_status.lower()

        # Get thumbnail - check disk first, then base64 in memory
        thumb_b64 = None
        mime = "image/jpeg"
        if (
            hasattr(series, "_thumbnail_path")
            and series._thumbnail_path
            and self.data_dir
        ):
            thumb_path = (
                self.data_dir / "_dicom_qc" / "thumbnails" / series._thumbnail_path
            )
            if thumb_path.exists():
                import base64

                thumb_b64 = base64.b64encode(thumb_path.read_bytes()).decode()
                mime = "image/jpeg"
        if not thumb_b64 and series.thumbnail:
            thumb_b64 = series.thumbnail
            mime = "image/jpeg"

        if thumb_b64:
            img_src = f"data:{mime};base64,{thumb_b64}"
        elif series.is_derived:
            # SVG placeholder for derived types
            img_src = (
                f'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="340" height="113">'
                f'<rect fill="%232d1f3d" width="340" height="113"/>'
                f'<text x="170" y="56" fill="%239c27b0" text-anchor="middle" dy=".3em" font-size="16">{escape(series.modality)}</text></svg>'
            )
        else:
            img_src = (
                'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="340" height="113">'
                '<rect fill="%23333" width="340" height="113"/>'
                '<text x="170" y="56" fill="%23999" text-anchor="middle" dy=".3em">No image</text></svg>'
            )

        # Build info/reason text
        reason_html = ""
        if series.error:
            reason_html = f'<div class="qc-reason" style="padding:6px 10px;font-size:11px;background:#f8d7da;color:#721c24;">Error: {escape(series.error)}</div>'
        elif series.is_derived:
            # Show derived info
            info_parts = []
            if series.derived_info:
                info_parts.append(series.derived_info)
            if series.referenced_series_uid:
                info_parts.append(f"→ {series.referenced_series_uid[:30]}...")
            if info_parts:
                reason_html = f'<div class="qc-reason" style="padding:6px 10px;font-size:11px;background:#2d1f3d;color:#e0c3fc;">{"<br>".join(escape(p) for p in info_parts)}</div>'
        elif series.qc_report and series.qc_status in ("FAIL", "WARNING", "NOTE"):
            issues = [
                r
                for r in series.qc_report.results
                if r.status in ("FAIL", "WARNING", "NOTE")
            ]
            if issues:
                issue_lines = [
                    f'<div style="margin:2px 0;"><b>{escape(r.check_name)}:</b> {escape(r.message)}</div>'
                    for r in issues
                ]
                if series.qc_status == "FAIL":
                    bg_color, text_color = "#f8d7da", "#721c24"
                elif series.qc_status == "WARNING":
                    bg_color, text_color = "#fff3cd", "#856404"
                else:  # NOTE
                    bg_color, text_color = "#d1ecf1", "#0c5460"
                reason_html = f'<div class="qc-reason" style="padding:6px 10px;font-size:11px;background:{bg_color};color:{text_color};">{"".join(issue_lines)}</div>'

        # Wrap in link if OHIF URL provided
        link_style = "cursor:pointer;" if ohif_url else ""
        link_start = (
            f'<a href="{ohif_url}" target="_blank" style="text-decoration:none;color:inherit;">'
            if ohif_url
            else ""
        )
        link_end = "</a>" if ohif_url else ""

        # Collect check names for filtering
        series_check_names = []
        if series.qc_report:
            series_check_names = [
                r.check_name
                for r in series.qc_report.results
                if r.status in ("FAIL", "WARNING", "NOTE")
            ]
        data_checks = ",".join(series_check_names)

        defaced_badge = ""
        if getattr(series, "_xnat_resource", None) == "DEFACED":
            defaced_badge = _defaced_badge_html("padding:3px 8px;border-radius:10px;flex-shrink:0;")

        return f'''                {link_start}<div class="qc-thumb {status}" data-status="{status}" data-checks="{data_checks}" style="{link_style}">
                    <img src="{img_src}" alt="{escape(series.description or "", quote=True)}">
                    <div class="info">
                        <span class="series-label">{escape(series.label)}</span>
                        {defaced_badge}
                        <span class="status-badge {status}">{series.qc_status}</span>
                    </div>
                    {reason_html}
                </div>{link_end}
'''

    def _html_footer(self) -> str:
        """Generate HTML footer with JavaScript."""
        return """
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
                <tr><td style="padding:8px;border:1px solid #dee2e6;"><b>JPEG-2000 Encoding</b></td><td style="padding:8px;border:1px solid #dee2e6;">NOTE if JPEG-2000 transfer syntax detected</td><td style="padding:8px;border:1px solid #dee2e6;">Some JPEG-2000 data renders blurry in OHIF viewer</td></tr>
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
"""
