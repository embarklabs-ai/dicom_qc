"""Track QC progress across multiple scans."""

from typing import List, Dict, Callable, Optional
import ipywidgets as widgets
from IPython.display import display, HTML as IPHTML

from dicom_qc.core.volume import ScanInfo


class ProgressTracker:
    """Track QC progress across multiple scans."""

    def __init__(self, scans: List[ScanInfo]):
        """
        Initialize with list of scans to review.

        Args:
            scans: List of ScanInfo objects
        """
        self.scans = scans
        self.results: Dict[str, Dict] = {}  # scan_id -> result
        self.current_index = 0
        self._on_scan_change: Optional[Callable[[int, ScanInfo], None]] = None

        self._create_widgets()
        self._setup_handlers()

    def _create_widgets(self):
        """Create all widgets."""
        # Progress bar
        self.progress_bar = widgets.IntProgress(
            value=0,
            min=0,
            max=len(self.scans),
            description="Progress:",
            bar_style="info",
            layout=widgets.Layout(width="400px"),
            style={"description_width": "70px"},
        )

        # Status text
        self.status_text = widgets.HTML(value=self._get_status_html())

        # Navigation buttons
        self.prev_button = widgets.Button(
            description="Previous",
            icon="arrow-left",
            disabled=True,
            layout=widgets.Layout(width="100px"),
        )

        self.next_button = widgets.Button(
            description="Next",
            icon="arrow-right",
            disabled=len(self.scans) <= 1,
            layout=widgets.Layout(width="100px"),
        )

        # Jump dropdown
        scan_options = [
            (f"{i + 1}. {s.id} - {s.description[:30]}", i)
            for i, s in enumerate(self.scans)
        ]
        self.jump_dropdown = widgets.Dropdown(
            options=scan_options,
            value=0,
            description="Jump to:",
            layout=widgets.Layout(width="300px"),
            style={"description_width": "60px"},
        )

        # Summary table
        self.summary_output = widgets.Output()

    def _setup_handlers(self):
        """Set up widget handlers."""
        self.prev_button.on_click(self._go_previous)
        self.next_button.on_click(self._go_next)
        self.jump_dropdown.observe(self._on_jump, names="value")

    def display(self):
        """Display the progress tracker."""
        header = widgets.HTML('<h3 style="margin-bottom: 10px;">Review Progress</h3>')

        nav_buttons = widgets.HBox(
            [self.prev_button, self.next_button, self.jump_dropdown]
        )

        summary_header = widgets.HTML('<h4 style="margin-top: 15px;">Summary</h4>')

        container = widgets.VBox(
            [
                header,
                self.progress_bar,
                self.status_text,
                nav_buttons,
                summary_header,
                self.summary_output,
            ],
            layout=widgets.Layout(padding="10px"),
        )

        display(container)
        self._update_summary()

    def on_scan_change(self, callback: Callable[[int, ScanInfo], None]):
        """Register callback for when scan selection changes."""
        self._on_scan_change = callback

    def record_result(self, scan_id: str, result: Dict):
        """
        Record QC result for a scan.

        Args:
            scan_id: Scan identifier
            result: Result dictionary with decision, notes, issues
        """
        self.results[scan_id] = result
        self._update_progress()
        self._update_summary()

    def _get_status_html(self) -> str:
        """Generate status HTML."""
        reviewed = len(self.results)
        total = len(self.scans)

        # Count by status
        passed = sum(1 for r in self.results.values() if r.get("decision") == "PASS")
        failed = sum(1 for r in self.results.values() if r.get("decision") == "FAIL")
        flagged = sum(1 for r in self.results.values() if r.get("decision") == "FLAG")
        skipped = sum(1 for r in self.results.values() if r.get("decision") == "SKIP")

        return f"""
            <div style="margin: 5px 0;">
                {reviewed} / {total} scans reviewed |
                <span style="color:green;">PASS: {passed}</span> |
                <span style="color:red;">FAIL: {failed}</span> |
                <span style="color:orange;">FLAG: {flagged}</span>
                {f'| <span style="color:gray;">SKIP: {skipped}</span>' if skipped else ""}
            </div>
        """

    def _update_progress(self):
        """Update progress bar and status."""
        reviewed = len(self.results)
        self.progress_bar.value = reviewed
        self.status_text.value = self._get_status_html()

    def _update_summary(self):
        """Update summary table."""
        with self.summary_output:
            from IPython.display import clear_output

            clear_output()

            rows = []
            for i, scan in enumerate(self.scans):
                result = self.results.get(scan.id, {})
                decision = result.get("decision", "Pending")

                color = {
                    "PASS": "green",
                    "FAIL": "red",
                    "FLAG": "orange",
                    "SKIP": "gray",
                    "Pending": "#999",
                }.get(decision, "black")

                icon = {
                    "PASS": "&#10003;",
                    "FAIL": "&#10007;",
                    "FLAG": "&#9873;",
                    "SKIP": "&#8594;",
                    "Pending": "&#8226;",
                }.get(decision, "")

                # Highlight current row
                bg_color = "#e3f2fd" if i == self.current_index else "transparent"

                rows.append(f"""
                    <tr style="background:{bg_color};">
                        <td style="padding:5px; text-align:center;">{i + 1}</td>
                        <td style="padding:5px;">{scan.id}</td>
                        <td style="padding:5px;">{scan.description[:40]}</td>
                        <td style="padding:5px; text-align:center; color:{color}; font-weight:bold;">
                            {icon} {decision}
                        </td>
                    </tr>
                """)

            table_html = f"""
                <div style="max-height:300px; overflow-y:auto;">
                <table style="width:100%; border-collapse:collapse; font-size:0.9em;">
                    <thead>
                        <tr style="background:#f0f0f0; position:sticky; top:0;">
                            <th style="padding:8px; text-align:center; width:40px;">#</th>
                            <th style="padding:8px; text-align:left;">Scan ID</th>
                            <th style="padding:8px; text-align:left;">Description</th>
                            <th style="padding:8px; text-align:center; width:100px;">Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(rows)}
                    </tbody>
                </table>
                </div>
            """

            display(IPHTML(table_html))

    def _go_previous(self, button):
        """Navigate to previous scan."""
        if self.current_index > 0:
            self.current_index -= 1
            self.jump_dropdown.value = self.current_index
            self._navigate()

    def _go_next(self, button):
        """Navigate to next scan."""
        if self.current_index < len(self.scans) - 1:
            self.current_index += 1
            self.jump_dropdown.value = self.current_index
            self._navigate()

    def _on_jump(self, change):
        """Handle jump to specific scan."""
        new_index = change["new"]
        if new_index != self.current_index:
            self.current_index = new_index
            self._navigate()

    def _navigate(self):
        """Handle navigation to current index."""
        self.prev_button.disabled = self.current_index == 0
        self.next_button.disabled = self.current_index >= len(self.scans) - 1
        self._update_summary()

        if self._on_scan_change:
            self._on_scan_change(self.current_index, self.scans[self.current_index])

    def go_to_next_pending(self):
        """Navigate to the next scan without a decision."""
        for i, scan in enumerate(self.scans):
            if scan.id not in self.results:
                self.current_index = i
                self.jump_dropdown.value = i
                self._navigate()
                return True
        return False

    def get_current_scan(self) -> ScanInfo:
        """Get the current scan."""
        return self.scans[self.current_index]

    def get_all_results(self) -> Dict[str, Dict]:
        """Get all recorded results."""
        return self.results.copy()

    def is_complete(self) -> bool:
        """Check if all scans have been reviewed."""
        return len(self.results) == len(self.scans)

    def get_summary_stats(self) -> Dict:
        """Get summary statistics."""
        total = len(self.scans)
        reviewed = len(self.results)

        return {
            "total": total,
            "reviewed": reviewed,
            "pending": total - reviewed,
            "passed": sum(
                1 for r in self.results.values() if r.get("decision") == "PASS"
            ),
            "failed": sum(
                1 for r in self.results.values() if r.get("decision") == "FAIL"
            ),
            "flagged": sum(
                1 for r in self.results.values() if r.get("decision") == "FLAG"
            ),
            "skipped": sum(
                1 for r in self.results.values() if r.get("decision") == "SKIP"
            ),
            "pass_rate": None
            if reviewed == 0
            else sum(1 for r in self.results.values() if r.get("decision") == "PASS")
            / reviewed
            * 100,
        }
