"""QC decision controls with notes field."""

from typing import Callable, Optional, List, Dict
from datetime import datetime
import ipywidgets as widgets
from IPython.display import display


class QCControls:
    """QC decision controls with notes field and issue checkboxes."""

    # Predefined issue types
    ISSUE_TYPES = [
        "Slice ordering issue",
        "Missing slices / gaps",
        "Orientation mismatch",
        "Image quality issue",
        "Wrong protocol",
        "Incomplete series",
        "Artifacts present",
        "Other (see notes)",
    ]

    def __init__(self):
        """Initialize QC controls."""
        self.current_decision: Optional[str] = None
        self._on_decision: Optional[Callable[[str, str, List[str]], None]] = None
        self._decision_time: Optional[datetime] = None

        self._create_widgets()
        self._setup_handlers()

    def _create_widgets(self):
        """Create all widgets."""
        # Decision buttons
        self.pass_button = widgets.Button(
            description="PASS",
            button_style="success",
            icon="check",
            layout=widgets.Layout(width="100px"),
        )

        self.fail_button = widgets.Button(
            description="FAIL",
            button_style="danger",
            icon="times",
            layout=widgets.Layout(width="100px"),
        )

        self.flag_button = widgets.Button(
            description="FLAG",
            button_style="warning",
            icon="flag",
            layout=widgets.Layout(width="100px"),
        )

        self.skip_button = widgets.Button(
            description="Skip",
            button_style="",
            icon="forward",
            layout=widgets.Layout(width="100px"),
        )

        # Notes field
        self.notes_text = widgets.Textarea(
            placeholder="Enter QC notes here...",
            layout=widgets.Layout(width="100%", height="80px"),
        )

        # Issue checkboxes
        self.issue_checkboxes = []
        for issue in self.ISSUE_TYPES:
            cb = widgets.Checkbox(
                value=False,
                description=issue,
                indent=False,
                layout=widgets.Layout(width="auto"),
            )
            self.issue_checkboxes.append(cb)

        # Status indicator
        self.status_label = widgets.HTML(
            value='<span style="color:gray; font-style:italic;">No decision yet</span>'
        )

        # Next button (after decision)
        self.next_button = widgets.Button(
            description="Next Scan",
            button_style="primary",
            icon="arrow-right",
            disabled=True,
            layout=widgets.Layout(width="120px"),
        )

    def _setup_handlers(self):
        """Set up button handlers."""
        self.pass_button.on_click(lambda b: self._set_decision("PASS"))
        self.fail_button.on_click(lambda b: self._set_decision("FAIL"))
        self.flag_button.on_click(lambda b: self._set_decision("FLAG"))
        self.skip_button.on_click(lambda b: self._set_decision("SKIP"))

    def display(self):
        """Display the QC controls."""
        header = widgets.HTML('<h3 style="margin-bottom: 10px;">QC Decision</h3>')

        buttons = widgets.HBox(
            [self.pass_button, self.fail_button, self.flag_button, self.skip_button],
            layout=widgets.Layout(margin="10px 0"),
        )

        issues_label = widgets.HTML("<b>Common Issues:</b>")
        issues_box = widgets.VBox(
            self.issue_checkboxes,
            layout=widgets.Layout(
                padding="5px 0", max_height="200px", overflow_y="auto"
            ),
        )

        notes_label = widgets.HTML("<b>Notes:</b>")

        container = widgets.VBox(
            [
                header,
                buttons,
                self.status_label,
                issues_label,
                issues_box,
                notes_label,
                self.notes_text,
                widgets.HBox([self.next_button]),
            ],
            layout=widgets.Layout(padding="10px", width="350px"),
        )

        display(container)

    def on_decision(self, callback: Callable[[str, str, List[str]], None]):
        """
        Register callback for when a decision is made.

        Callback receives: (decision, notes, selected_issues)
        """
        self._on_decision = callback

    def on_next(self, callback: Callable[[], None]):
        """Register callback for next button."""
        self.next_button.on_click(lambda b: callback())

    def _set_decision(self, decision: str):
        """Set the QC decision."""
        self.current_decision = decision
        self._decision_time = datetime.now()

        # Update button styles to show selection
        self.pass_button.button_style = "success" if decision == "PASS" else ""
        self.fail_button.button_style = "danger" if decision == "FAIL" else ""
        self.flag_button.button_style = "warning" if decision == "FLAG" else ""
        self.skip_button.button_style = "info" if decision == "SKIP" else ""

        # Update status label
        colors = {"PASS": "green", "FAIL": "red", "FLAG": "orange", "SKIP": "gray"}
        icons = {
            "PASS": "&#10003;",  # checkmark
            "FAIL": "&#10007;",  # x
            "FLAG": "&#9873;",  # flag
            "SKIP": "&#8594;",  # arrow
        }
        self.status_label.value = f"""
            <span style="color:{colors[decision]}; font-weight:bold; font-size:1.2em;">
                {icons[decision]} {decision}
            </span>
        """

        # Enable next button
        self.next_button.disabled = False

        # Collect issues
        selected_issues = self.get_selected_issues()

        # Call callback
        if self._on_decision:
            self._on_decision(decision, self.notes_text.value, selected_issues)

    def get_selected_issues(self) -> List[str]:
        """Get list of selected issue types."""
        return [cb.description for cb in self.issue_checkboxes if cb.value]

    def reset(self):
        """Reset controls for next scan."""
        self.current_decision = None
        self._decision_time = None
        self.notes_text.value = ""

        for cb in self.issue_checkboxes:
            cb.value = False

        # Reset button styles
        self.pass_button.button_style = "success"
        self.fail_button.button_style = "danger"
        self.flag_button.button_style = "warning"
        self.skip_button.button_style = ""

        self.status_label.value = (
            '<span style="color:gray; font-style:italic;">No decision yet</span>'
        )
        self.next_button.disabled = True

    def get_decision(self) -> Dict:
        """Get current decision data."""
        return {
            "decision": self.current_decision,
            "notes": self.notes_text.value,
            "issues": self.get_selected_issues(),
            "timestamp": self._decision_time.isoformat()
            if self._decision_time
            else None,
        }

    def set_enabled(self, enabled: bool):
        """Enable or disable all controls."""
        self.pass_button.disabled = not enabled
        self.fail_button.disabled = not enabled
        self.flag_button.disabled = not enabled
        self.skip_button.disabled = not enabled
        self.notes_text.disabled = not enabled

        for cb in self.issue_checkboxes:
            cb.disabled = not enabled
