"""Interactive widget for browsing XNAT sessions and scans."""

from typing import Callable, Optional, List
import ipywidgets as widgets
from IPython.display import display, clear_output

from dicom_qc.core.volume import ScanInfo


class SessionBrowser:
    """Interactive widget for browsing XNAT sessions and scans."""

    def __init__(self, xnat_session):
        """
        Initialize with XNAT session wrapper.

        Args:
            xnat_session: XNATSession instance
        """
        self.xnat = xnat_session
        self.selected_scan: Optional[ScanInfo] = None
        self._on_scan_selected: Optional[Callable[[ScanInfo], None]] = None
        self._current_scans: List[ScanInfo] = []

        # Create widgets
        self.project_dropdown = widgets.Dropdown(
            options=[],
            description="Project:",
            layout=widgets.Layout(width="400px"),
            style={"description_width": "80px"},
        )

        self.subject_dropdown = widgets.Dropdown(
            options=[],
            description="Subject:",
            disabled=True,
            layout=widgets.Layout(width="400px"),
            style={"description_width": "80px"},
        )

        self.experiment_dropdown = widgets.Dropdown(
            options=[],
            description="Session:",
            disabled=True,
            layout=widgets.Layout(width="400px"),
            style={"description_width": "80px"},
        )

        self.scan_dropdown = widgets.Dropdown(
            options=[],
            description="Scan:",
            disabled=True,
            layout=widgets.Layout(width="500px"),
            style={"description_width": "80px"},
        )

        self.load_button = widgets.Button(
            description="Load Scan",
            button_style="primary",
            disabled=True,
            icon="download",
        )

        self.load_all_button = widgets.Button(
            description="Load All Scans",
            button_style="info",
            disabled=True,
            icon="folder-open",
        )

        self.refresh_button = widgets.Button(
            description="Refresh", button_style="", icon="refresh"
        )

        self.status_output = widgets.Output()

        # Wire up observers
        self.project_dropdown.observe(self._on_project_change, names="value")
        self.subject_dropdown.observe(self._on_subject_change, names="value")
        self.experiment_dropdown.observe(self._on_experiment_change, names="value")
        self.scan_dropdown.observe(self._on_scan_change, names="value")
        self.load_button.on_click(self._on_load_click)
        self.load_all_button.on_click(self._on_load_all_click)
        self.refresh_button.on_click(self._on_refresh_click)

        # Initialize
        self._load_projects()

    def display(self):
        """Display the browser widget."""
        header = widgets.HTML("""
            <h3 style="margin-bottom: 10px;">XNAT Session Browser</h3>
            <p style="color: #666; margin-bottom: 15px;">
                Select a project, subject, session, and scan to load for QC review.
            </p>
        """)

        buttons = widgets.HBox(
            [self.load_button, self.load_all_button, self.refresh_button]
        )

        container = widgets.VBox(
            [
                header,
                self.project_dropdown,
                self.subject_dropdown,
                self.experiment_dropdown,
                self.scan_dropdown,
                buttons,
                self.status_output,
            ],
            layout=widgets.Layout(padding="10px"),
        )

        display(container)

    def on_scan_selected(self, callback: Callable[[ScanInfo], None]):
        """Register callback for when a scan is loaded."""
        self._on_scan_selected = callback

    def on_all_scans_selected(self, callback: Callable[[List[ScanInfo]], None]):
        """Register callback for when all scans are loaded."""
        self._on_all_scans_selected = callback

    def _load_projects(self):
        """Load available projects."""
        with self.status_output:
            clear_output()
            print("Loading projects...")

        try:
            projects = self.xnat.get_projects()
            self.project_dropdown.options = [("Select project...", None)] + [
                (p, p) for p in projects
            ]
            self.project_dropdown.value = None

            with self.status_output:
                clear_output()
                print(f"Found {len(projects)} projects")
        except Exception as e:
            with self.status_output:
                clear_output()
                print(f"Error loading projects: {e}")

    def _on_project_change(self, change):
        """Handle project selection."""
        project = change["new"]
        if not project:
            return

        # Reset downstream dropdowns
        self.subject_dropdown.options = []
        self.subject_dropdown.disabled = True
        self.experiment_dropdown.options = []
        self.experiment_dropdown.disabled = True
        self.scan_dropdown.options = []
        self.scan_dropdown.disabled = True
        self.load_button.disabled = True
        self.load_all_button.disabled = True

        with self.status_output:
            clear_output()
            print(f"Loading subjects for {project}...")

        try:
            subjects = self.xnat.get_subjects(project)
            self.subject_dropdown.options = [("Select subject...", None)] + [
                (s, s) for s in subjects
            ]
            self.subject_dropdown.value = None
            self.subject_dropdown.disabled = False

            with self.status_output:
                clear_output()
                print(f"Found {len(subjects)} subjects")
        except Exception as e:
            with self.status_output:
                clear_output()
                print(f"Error loading subjects: {e}")

    def _on_subject_change(self, change):
        """Handle subject selection."""
        subject = change["new"]
        if not subject:
            return

        project = self.project_dropdown.value

        # Reset downstream dropdowns
        self.experiment_dropdown.options = []
        self.experiment_dropdown.disabled = True
        self.scan_dropdown.options = []
        self.scan_dropdown.disabled = True
        self.load_button.disabled = True
        self.load_all_button.disabled = True

        with self.status_output:
            clear_output()
            print("Loading sessions...")

        try:
            experiments = self.xnat.get_experiments(project, subject)
            self.experiment_dropdown.options = [("Select session...", None)] + [
                (e, e) for e in experiments
            ]
            self.experiment_dropdown.value = None
            self.experiment_dropdown.disabled = False

            with self.status_output:
                clear_output()
                print(f"Found {len(experiments)} sessions")
        except Exception as e:
            with self.status_output:
                clear_output()
                print(f"Error loading sessions: {e}")

    def _on_experiment_change(self, change):
        """Handle experiment/session selection."""
        experiment = change["new"]
        if not experiment:
            return

        project = self.project_dropdown.value
        subject = self.subject_dropdown.value

        # Reset downstream dropdowns
        self.scan_dropdown.options = []
        self.scan_dropdown.disabled = True
        self.load_button.disabled = True

        with self.status_output:
            clear_output()
            print("Loading scans...")

        try:
            scans = self.xnat.get_scans(project, subject, experiment)
            self._current_scans = scans

            # Format as "ID - Description (N files)"
            scan_options = [("Select scan...", None)]
            for s in scans:
                label = f"{s.id} - {s.description} ({s.num_files} files)"
                scan_options.append((label, s))

            self.scan_dropdown.options = scan_options
            self.scan_dropdown.value = None
            self.scan_dropdown.disabled = False
            self.load_all_button.disabled = len(scans) == 0

            with self.status_output:
                clear_output()
                print(f"Found {len(scans)} scans")
        except Exception as e:
            with self.status_output:
                clear_output()
                print(f"Error loading scans: {e}")

    def _on_scan_change(self, change):
        """Handle scan selection."""
        self.load_button.disabled = change["new"] is None

    def _on_load_click(self, button):
        """Handle load button click."""
        scan_info = self.scan_dropdown.value
        if scan_info is None:
            return

        self.selected_scan = scan_info

        with self.status_output:
            clear_output()
            print(f"Loading scan {scan_info.id}...")

        if self._on_scan_selected:
            self._on_scan_selected(scan_info)

    def _on_load_all_click(self, button):
        """Handle load all scans button click."""
        if not self._current_scans:
            return

        with self.status_output:
            clear_output()
            print(f"Loading {len(self._current_scans)} scans...")

        if hasattr(self, "_on_all_scans_selected") and self._on_all_scans_selected:
            self._on_all_scans_selected(self._current_scans)

    def _on_refresh_click(self, button):
        """Handle refresh button click."""
        self._load_projects()

    def get_selected_scan(self) -> Optional[ScanInfo]:
        """Get the currently selected scan."""
        return self.selected_scan

    def get_current_scans(self) -> List[ScanInfo]:
        """Get all scans in the current session."""
        return self._current_scans
