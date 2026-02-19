"""Generate self-contained HTML QC reports."""

from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, BaseLoader

from dicom_qc.core.volume import DicomVolume
from dicom_qc.visualization.snapshots import SnapshotGenerator
from dicom_qc.visualization.animations import AnimationGenerator


# Inline template as fallback if file not found
FALLBACK_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>DICOM QC Report</title>
    <style>
        body { font-family: sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { background: #343a40; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .summary { display: flex; gap: 20px; margin-bottom: 20px; }
        .summary-card { background: white; padding: 20px; border-radius: 8px; text-align: center; flex: 1; border-top: 4px solid #007bff; }
        .scan-section { background: white; border-radius: 8px; margin-bottom: 20px; overflow: hidden; }
        .scan-header { padding: 15px; display: flex; justify-content: space-between; }
        .scan-header.pass { background: #d4edda; }
        .scan-header.fail { background: #f8d7da; }
        .scan-header.flag { background: #fff3cd; }
        .scan-content { padding: 20px; }
        .image-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; }
        .image-container img { max-width: 100%; }
        .status-badge { padding: 5px 15px; border-radius: 20px; color: white; }
        .status-badge.pass { background: #28a745; }
        .status-badge.fail { background: #dc3545; }
        .status-badge.flag { background: #ffc107; }
    </style>
</head>
<body>
    <div class="header">
        <h1>DICOM QC Report</h1>
        <p>Project: {{ session_info.project }} | Subject: {{ session_info.subject }} | Session: {{ session_info.experiment_id }}</p>
        <p>Generated: {{ generated_at }}</p>
    </div>
    <div class="summary">
        <div class="summary-card"><h2>{{ summary.total }}</h2><p>Total</p></div>
        <div class="summary-card" style="border-color:#28a745;"><h2>{{ summary.passed }}</h2><p>Passed</p></div>
        <div class="summary-card" style="border-color:#dc3545;"><h2>{{ summary.failed }}</h2><p>Failed</p></div>
        <div class="summary-card" style="border-color:#ffc107;"><h2>{{ summary.flagged }}</h2><p>Flagged</p></div>
    </div>
    {% for scan in scan_results %}
    <div class="scan-section">
        <div class="scan-header {{ scan.decision|lower }}">
            <div><h2>{{ scan.scan_id }} - {{ scan.description }}</h2></div>
            <span class="status-badge {{ scan.decision|lower }}">{{ scan.decision }}</span>
        </div>
        <div class="scan-content">
            <div class="image-grid">
                {% if scan.axial_image %}<div class="image-container"><img src="data:image/jpeg;base64,{{ scan.axial_image }}"><p>Axial</p></div>{% endif %}
                {% if scan.coronal_image %}<div class="image-container"><img src="data:image/jpeg;base64,{{ scan.coronal_image }}"><p>Coronal</p></div>{% endif %}
                {% if scan.sagittal_image %}<div class="image-container"><img src="data:image/jpeg;base64,{{ scan.sagittal_image }}"><p>Sagittal</p></div>{% endif %}
                {% if scan.mip_image %}<div class="image-container"><img src="data:image/jpeg;base64,{{ scan.mip_image }}"><p>MIP</p></div>{% endif %}
            </div>
            {% if scan.animation_gif %}<div style="text-align:center;margin:20px 0;"><img src="data:image/gif;base64,{{ scan.animation_gif }}" style="max-width:500px;"></div>{% endif %}
            {% if scan.ohif_url %}<a href="{{ scan.ohif_url }}" target="_blank" style="display:inline-block;padding:10px 20px;background:#007bff;color:white;text-decoration:none;border-radius:4px;">View in OHIF</a>{% endif %}
        </div>
    </div>
    {% endfor %}
</body>
</html>
"""


class HTMLReportGenerator:
    """Generate self-contained HTML QC reports."""

    def __init__(self):
        """Initialize the report generator."""
        # Try to load template from file
        template_dir = Path(__file__).parent / "templates"
        template_file = template_dir / "report.html"

        if template_file.exists():
            self.env = Environment(
                loader=FileSystemLoader(str(template_dir)), autoescape=True
            )
            self.template = self.env.get_template("report.html")
        else:
            # Use fallback template
            self.env = Environment(loader=BaseLoader(), autoescape=True)
            self.template = self.env.from_string(FALLBACK_TEMPLATE)

    def generate_report(
        self,
        session_info: Dict[str, str],
        scan_results: List[Dict[str, Any]],
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate complete HTML QC report.

        Args:
            session_info: Session metadata (project, subject, experiment_id)
            scan_results: List of scan QC results with embedded images
            output_path: Optional path to save HTML file

        Returns:
            HTML string
        """
        # Calculate summary statistics
        summary = self._calculate_summary(scan_results)

        # Render template
        html = self.template.render(
            session_info=session_info,
            scan_results=scan_results,
            summary=summary,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            report_version="1.0.0",
        )

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)

        return html

    def _calculate_summary(self, scan_results: List[Dict]) -> Dict:
        """Calculate summary statistics."""
        total = len(scan_results)
        passed = sum(1 for r in scan_results if r.get("decision") == "PASS")
        failed = sum(1 for r in scan_results if r.get("decision") == "FAIL")
        flagged = sum(1 for r in scan_results if r.get("decision") == "FLAG")
        skipped = sum(1 for r in scan_results if r.get("decision") == "SKIP")

        reviewed = passed + failed + flagged + skipped
        pass_rate = (passed / reviewed * 100) if reviewed > 0 else 0

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "flagged": flagged,
            "skipped": skipped,
            "reviewed": reviewed,
            "pass_rate": f"{pass_rate:.1f}",
        }

    def generate_snapshots(
        self, volume: DicomVolume, window: Optional[Tuple[float, float]] = None
    ) -> Dict[str, str]:
        """
        Generate all snapshots for a volume using proper LPS orientation.

        Args:
            volume: DicomVolume object
            window: Optional (center, width) for windowing

        Returns:
            Dict with base64 encoded images for 'axial', 'coronal', 'sagittal', 'mip', 'overview'
        """
        generator = SnapshotGenerator(volume, window=window)
        return generator.generate_all_snapshots()

    def generate_animation(
        self,
        volume: DicomVolume,
        window: Optional[Tuple[float, float]] = None,
        use_acquisition_plane: bool = True,
        fps: int = 10,
    ) -> Optional[str]:
        """
        Generate a slice animation for a volume.

        Args:
            volume: DicomVolume object
            window: Optional (center, width) for windowing
            use_acquisition_plane: If True, animate in acquisition plane; else use axial
            fps: Frames per second

        Returns:
            Base64 encoded GIF string, or None if generation fails
        """
        try:
            generator = AnimationGenerator(volume, window=window)
            if use_acquisition_plane:
                gif_bytes = generator.create_acquisition_plane_animation(fps=fps)
            else:
                gif_bytes = generator.create_slice_animation(view="axial", fps=fps)

            if gif_bytes:
                return generator.to_base64(gif_bytes)
        except Exception:
            pass
        return None

    def prepare_scan_result(
        self,
        scan_info: Any,
        volume: DicomVolume,
        qc_report: Any,
        decision_data: Dict,
        snapshots: Optional[Dict[str, str]] = None,
        animation_gif: Optional[str] = None,
        ohif_url: Optional[str] = None,
        window: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """
        Prepare a single scan result for the report.

        Args:
            scan_info: ScanInfo object
            volume: DicomVolume object
            qc_report: QCReport object
            decision_data: Decision data from QCControls
            snapshots: Dict of base64 encoded images (generated if not provided)
            animation_gif: Optional base64 encoded GIF (generated if not provided)
            ohif_url: Optional OHIF viewer URL
            window: Optional (center, width) for windowing

        Returns:
            Dict ready for template rendering
        """
        # Generate snapshots if not provided
        if snapshots is None:
            snapshots = self.generate_snapshots(volume, window=window)

        # Generate animation if not provided
        if animation_gif is None:
            animation_gif = self.generate_animation(volume, window=window)

        return {
            "scan_id": scan_info.id,
            "description": scan_info.description,
            "modality": volume.modality,
            "num_slices": volume.num_slices,
            "pixel_spacing": volume.pixel_spacing,
            "slice_thickness": volume.slice_thickness,
            "decision": decision_data.get("decision", "Pending"),
            "notes": decision_data.get("notes", ""),
            "issues": decision_data.get("issues", []),
            "qc_checks": [r.to_dict() for r in qc_report.results],
            "orientation_labels": qc_report.orientation_labels,
            "primary_plane": qc_report.primary_plane,
            "axial_image": snapshots.get("axial"),
            "coronal_image": snapshots.get("coronal"),
            "sagittal_image": snapshots.get("sagittal"),
            "mip_image": snapshots.get("mip"),
            "animation_gif": animation_gif,
            "ohif_url": ohif_url,
        }
