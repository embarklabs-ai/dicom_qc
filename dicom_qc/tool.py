"""Main DICOM QC Tool orchestrator."""

from typing import Optional, List, Dict, Any
from pathlib import Path
import logging

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

from dicom_qc.core.connection import XNATSession
from dicom_qc.core.dicom_loader import DicomLoader
from dicom_qc.core.volume import DicomVolume, ScanInfo
from dicom_qc.core.geometry import GeometryQC, QCReport
from dicom_qc.visualization.snapshots import SnapshotGenerator
from dicom_qc.visualization.animations import AnimationGenerator
from dicom_qc.widgets.browser import SessionBrowser
from dicom_qc.widgets.viewer import InteractiveViewer
from dicom_qc.widgets.qc_controls import QCControls
from dicom_qc.widgets.progress import ProgressTracker
from dicom_qc.reports.html_generator import HTMLReportGenerator
from dicom_qc.utils.ohif_links import OHIFLinkGenerator
from dicom_qc.utils.errors import DicomQCError, ErrorDisplay

logger = logging.getLogger(__name__)


class DicomQCTool:
    """
    Main orchestration class for DICOM QC review.

    Usage in Jupyter:
    ```python
    from dicom_qc import DicomQCTool

    # Initialize (auto-connects in XNAT Jupyter environment)
    qc = DicomQCTool()

    # Browse and select session
    qc.browse()

    # Or load specific session directly
    qc.load_session(project='MyProject', subject='Subject01', experiment='Session01')

    # Run QC review
    qc.review()

    # Generate report
    qc.generate_report('qc_report.html')
    ```
    """

    def __init__(
        self, connection: Optional[Any] = None, xnat_url: Optional[str] = None
    ):
        """
        Initialize QC tool.

        Args:
            connection: Existing xnat connection (optional)
            xnat_url: XNAT server URL if not auto-connecting
        """
        self.xnat_session = XNATSession(connection)

        if connection is None:
            try:
                self.xnat_session.connect(xnat_url)
            except Exception as e:
                logger.warning(f"Could not auto-connect to XNAT: {e}")

        # Initialize components
        self.ohif_generator: Optional[OHIFLinkGenerator] = None
        if self.xnat_session._connected:
            self.ohif_generator = OHIFLinkGenerator(self.xnat_session.url)

        # Session state
        self.current_session: Optional[Dict[str, str]] = None
        self.current_scans: List[ScanInfo] = []
        self.scan_results: List[Dict] = []

        # Loaded volumes cache
        self._volumes: Dict[str, DicomVolume] = {}
        self._qc_reports: Dict[str, QCReport] = {}

        # Widget components (created on demand)
        self._browser: Optional[SessionBrowser] = None
        self._viewer: Optional[InteractiveViewer] = None
        self._controls: Optional[QCControls] = None
        self._progress: Optional[ProgressTracker] = None

        # Output areas for review interface
        self._viewer_output: Optional[widgets.Output] = None
        self._controls_output: Optional[widgets.Output] = None
        self._qc_output: Optional[widgets.Output] = None

    def browse(self):
        """Display interactive session browser."""
        self._browser = SessionBrowser(self.xnat_session)
        self._browser.on_scan_selected(self._on_single_scan_selected)
        self._browser.on_all_scans_selected(self._on_all_scans_selected)
        self._browser.display()

    def load_session(
        self,
        project: str,
        subject: str,
        experiment: str,
        scan_filter: Optional[List[str]] = None,
    ):
        """
        Load a specific session for QC review.

        Args:
            project: XNAT project ID
            subject: Subject label
            experiment: Experiment/session ID
            scan_filter: Optional list of scan IDs to include
        """
        self.current_session = {
            "project": project,
            "subject": subject,
            "experiment_id": experiment,
        }

        # Update OHIF generator with correct URL
        if self.xnat_session._connected:
            self.ohif_generator = OHIFLinkGenerator(self.xnat_session.url)

        scans = self.xnat_session.get_scans(project, subject, experiment)

        if scan_filter:
            scans = [s for s in scans if s.id in scan_filter]

        self.current_scans = scans

        # Clear caches
        self._volumes.clear()
        self._qc_reports.clear()
        self.scan_results.clear()

        display(
            HTML(f"""
            <div style="background:#d4edda; padding:15px; border-radius:8px; margin:10px 0;">
                <h3 style="margin-top:0;">Session Loaded</h3>
                <p><strong>Project:</strong> {project}</p>
                <p><strong>Subject:</strong> {subject}</p>
                <p><strong>Session:</strong> {experiment}</p>
                <p><strong>Scans:</strong> {len(scans)}</p>
            </div>
        """)
        )

    def load_from_path(self, path: str):
        """
        Load DICOM data from a local directory path.

        Args:
            path: Path to directory containing DICOM files
        """
        path = Path(path)
        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")

        # Create a fake session info
        self.current_session = {
            "project": "Local",
            "subject": path.parent.name,
            "experiment_id": path.name,
        }

        # Create a single ScanInfo for the directory
        scan_info = ScanInfo(
            id="1",
            description=path.name,
            modality="Unknown",
            num_files=len(list(path.glob("*.dcm"))),
            project="Local",
            subject=path.parent.name,
            experiment=path.name,
        )

        self.current_scans = [scan_info]
        self._volumes.clear()
        self._qc_reports.clear()
        self.scan_results.clear()

        # Load the volume
        loader = DicomLoader()
        volume = loader.load_from_path(path)
        self._volumes[scan_info.id] = volume

        display(
            HTML(f"""
            <div style="background:#d4edda; padding:15px; border-radius:8px; margin:10px 0;">
                <h3 style="margin-top:0;">Local Data Loaded</h3>
                <p><strong>Path:</strong> {path}</p>
                <p><strong>Modality:</strong> {volume.modality}</p>
                <p><strong>Slices:</strong> {volume.num_slices}</p>
            </div>
        """)
        )

    def _on_single_scan_selected(self, scan_info: ScanInfo):
        """Handle single scan selection from browser."""
        self.current_session = {
            "project": scan_info.project,
            "subject": scan_info.subject,
            "experiment_id": scan_info.experiment,
        }
        self.current_scans = [scan_info]
        self._load_and_display_scan(scan_info)

    def _on_all_scans_selected(self, scans: List[ScanInfo]):
        """Handle all scans selection from browser."""
        if not scans:
            return

        first_scan = scans[0]
        self.current_session = {
            "project": first_scan.project,
            "subject": first_scan.subject,
            "experiment_id": first_scan.experiment,
        }
        self.current_scans = scans
        self.review()

    def _load_volume(self, scan_info: ScanInfo) -> DicomVolume:
        """Load a volume for a scan, using cache if available."""
        if scan_info.id in self._volumes:
            return self._volumes[scan_info.id]

        # Load from XNAT
        files = self.xnat_session.get_scan_files(scan_info)
        loader = DicomLoader()
        volume = loader.load_from_xnat(files)
        self._volumes[scan_info.id] = volume

        return volume

    def _get_qc_report(self, scan_info: ScanInfo, volume: DicomVolume) -> QCReport:
        """Get QC report for a scan, using cache if available."""
        if scan_info.id in self._qc_reports:
            return self._qc_reports[scan_info.id]

        geometry_qc = GeometryQC(volume)
        qc_report = geometry_qc.run_all_checks(scan_info.id)
        self._qc_reports[scan_info.id] = qc_report

        return qc_report

    def _load_and_display_scan(self, scan_info: ScanInfo):
        """Load and display a single scan for quick review."""
        try:
            volume = self._load_volume(scan_info)
            qc_report = self._get_qc_report(scan_info, volume)

            # Display QC results
            self._display_qc_results(qc_report)

            # Create viewer
            viewer = InteractiveViewer(volume)
            viewer.display()

            # Generate and display snapshots
            snapshot_gen = SnapshotGenerator(volume)
            fig = snapshot_gen.create_orientation_figure()
            display(fig)

        except DicomQCError as e:
            ErrorDisplay.show_error(e, f"Loading scan {scan_info.id}")

    def _display_qc_results(self, qc_report: QCReport):
        """Display QC check results."""
        results_html = ['<div style="margin:15px 0;">']
        results_html.append(f"<h4>Geometry QC Results: {qc_report.primary_plane}</h4>")

        for result in qc_report.results:
            color = {"PASS": "green", "FAIL": "red", "WARNING": "orange"}[result.status]
            icon = {"PASS": "&#10003;", "FAIL": "&#10007;", "WARNING": "!"}[
                result.status
            ]

            results_html.append(f"""
                <div style="margin:5px 0; padding:8px; background:#f8f9fa; border-radius:4px;">
                    <span style="color:{color}; font-weight:bold;">{icon}</span>
                    <strong>{result.check_name}:</strong> {result.message}
                </div>
            """)

        results_html.append("</div>")
        display(HTML("".join(results_html)))

    def review(self):
        """Start interactive QC review of loaded scans."""
        if not self.current_scans:
            raise ValueError("No scans loaded. Call browse() or load_session() first.")

        # Clear any previous results for new review
        self.scan_results.clear()

        # Initialize progress tracker
        self._progress = ProgressTracker(self.current_scans)
        self._progress.on_scan_change(self._on_progress_scan_change)

        # Create output areas
        self._viewer_output = widgets.Output()
        self._controls_output = widgets.Output()
        self._qc_output = widgets.Output()

        # Create layout
        header = widgets.HTML("<h2>DICOM QC Review</h2>")

        main_content = widgets.HBox(
            [
                widgets.VBox(
                    [
                        self._qc_output,
                        self._viewer_output,
                    ],
                    layout=widgets.Layout(flex="2"),
                ),
                self._controls_output,
            ]
        )

        progress_output = widgets.Output()
        with progress_output:
            self._progress.display()

        layout = widgets.VBox(
            [
                header,
                progress_output,
                main_content,
            ]
        )

        display(layout)

        # Load first scan
        self._load_scan_for_review(0)

    def _on_progress_scan_change(self, index: int, scan_info: ScanInfo):
        """Handle scan change from progress tracker."""
        self._load_scan_for_review(index)

    def _load_scan_for_review(self, index: int):
        """Load a specific scan into the review interface."""
        scan = self.current_scans[index]

        # Clear outputs
        with self._viewer_output:
            clear_output()
        with self._controls_output:
            clear_output()
        with self._qc_output:
            clear_output()

        try:
            # Load volume
            with self._qc_output:
                display(HTML(f"<p>Loading scan {scan.id}...</p>"))

            volume = self._load_volume(scan)
            qc_report = self._get_qc_report(scan, volume)

            # Display QC results
            with self._qc_output:
                clear_output()
                self._display_qc_results(qc_report)

            # Create viewer
            with self._viewer_output:
                self._viewer = InteractiveViewer(volume)
                self._viewer.display()

            # Create controls
            with self._controls_output:
                self._controls = QCControls()
                self._controls.on_decision(
                    lambda d, n, i: self._on_decision_made(
                        scan, volume, qc_report, d, n, i
                    )
                )
                self._controls.on_next(self._on_next_click)
                self._controls.display()

                # Add OHIF link if available
                if self.ohif_generator and self.current_session:
                    ohif_url = self.ohif_generator.generate_scan_link(
                        self.current_session["project"],
                        self.current_session["experiment_id"],
                        scan.id,
                    )
                    display(
                        HTML(f'''
                        <div style="margin-top:15px;">
                            <a href="{ohif_url}" target="_blank"
                               style="display:inline-block; padding:10px 20px;
                                      background:#007bff; color:white;
                                      text-decoration:none; border-radius:4px;">
                                Open in OHIF Viewer
                            </a>
                        </div>
                    ''')
                    )

        except DicomQCError as e:
            with self._qc_output:
                clear_output()
                ErrorDisplay.show_error(e, f"Loading scan {scan.id}")

    def _on_decision_made(
        self,
        scan: ScanInfo,
        volume: DicomVolume,
        qc_report: QCReport,
        decision: str,
        notes: str,
        issues: List[str],
    ):
        """Handle QC decision for a scan."""
        result = {
            "scan_id": scan.id,
            "scan_info": scan,
            "volume": volume,
            "qc_report": qc_report,
            "decision": decision,
            "notes": notes,
            "issues": issues,
        }

        # Update or add result
        existing_idx = next(
            (i for i, r in enumerate(self.scan_results) if r["scan_id"] == scan.id),
            None,
        )
        if existing_idx is not None:
            self.scan_results[existing_idx] = result
        else:
            self.scan_results.append(result)

        # Record in progress tracker
        self._progress.record_result(
            scan.id,
            {
                "decision": decision,
                "notes": notes,
                "issues": issues,
            },
        )

    def _on_next_click(self):
        """Handle next button click."""
        if self._progress:
            current_idx = self._progress.current_index
            if current_idx < len(self.current_scans) - 1:
                self._progress._go_next(None)

    def generate_report(
        self,
        output_path: Optional[str] = None,
        include_animations: bool = True,
        animation_fps: int = 8,
    ) -> str:
        """
        Generate HTML QC report.

        Args:
            output_path: Path to save HTML file (optional)
            include_animations: Include animated GIFs in report
            animation_fps: Frames per second for animations

        Returns:
            HTML string
        """
        if not self.scan_results:
            raise ValueError("No QC results. Complete review first.")

        if not self.current_session:
            raise ValueError("No session info available.")

        generator = HTMLReportGenerator()
        prepared_results = []

        for result in self.scan_results:
            scan_info = result["scan_info"]
            volume = result["volume"]
            qc_report = result["qc_report"]

            # Generate snapshots
            snapshot_gen = SnapshotGenerator(
                volume,
                window=self._viewer.get_current_window() if self._viewer else None,
            )
            snapshots = snapshot_gen.generate_all_snapshots()

            # Generate animation if requested
            animation_gif = None
            if include_animations:
                anim_gen = AnimationGenerator(volume)
                gif_bytes = anim_gen.create_slice_animation(axis=0, fps=animation_fps)
                if gif_bytes:
                    animation_gif = anim_gen.to_base64(gif_bytes)

            # Generate OHIF URL
            ohif_url = None
            if self.ohif_generator:
                ohif_url = self.ohif_generator.generate_scan_link(
                    self.current_session["project"],
                    self.current_session["experiment_id"],
                    scan_info.id,
                )

            prepared_result = generator.prepare_scan_result(
                scan_info=scan_info,
                volume=volume,
                qc_report=qc_report,
                decision_data={
                    "decision": result["decision"],
                    "notes": result["notes"],
                    "issues": result["issues"],
                },
                snapshots=snapshots,
                animation_gif=animation_gif,
                ohif_url=ohif_url,
            )
            prepared_results.append(prepared_result)

        # Generate report
        html = generator.generate_report(
            session_info=self.current_session,
            scan_results=prepared_results,
            output_path=output_path,
        )

        if output_path:
            display(
                HTML(f"""
                <div style="background:#d4edda; padding:15px; border-radius:8px; margin:10px 0;">
                    <p><strong>Report saved to:</strong> <code>{output_path}</code></p>
                </div>
            """)
            )

        return html

    def quick_check(self, scan_info: Optional[ScanInfo] = None):
        """
        Run quick visual check on a single scan without full review.

        Generates and displays snapshots and GIF animation.

        Args:
            scan_info: ScanInfo to check, or uses first loaded scan
        """
        if scan_info is None:
            if not self.current_scans:
                raise ValueError("No scans loaded.")
            scan_info = self.current_scans[0]

        volume = self._load_volume(scan_info)
        qc_report = self._get_qc_report(scan_info, volume)

        # Display info
        display(
            HTML(f"""
            <h3>Quick Check: {scan_info.id} - {scan_info.description}</h3>
            <p>{volume.modality} | {volume.num_slices} slices |
               {volume.pixel_spacing[0]:.2f} x {volume.pixel_spacing[1]:.2f} mm</p>
        """)
        )

        # Display QC results
        self._display_qc_results(qc_report)

        # Generate and display orientation figure
        snapshot_gen = SnapshotGenerator(volume)
        fig = snapshot_gen.create_orientation_figure()
        display(fig)

        # Generate and display animation
        display(HTML("<h4>Slice Animation</h4>"))
        anim_gen = AnimationGenerator(volume)
        gif_bytes = anim_gen.create_slice_animation(axis=0, fps=10)
        if gif_bytes:
            gif_b64 = anim_gen.to_base64(gif_bytes)
            display(
                HTML(
                    f'<img src="data:image/gif;base64,{gif_b64}" style="max-width:500px;">'
                )
            )

    def close(self):
        """Close XNAT connection."""
        if self.xnat_session:
            self.xnat_session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
