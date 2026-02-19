"""Tests for process_all method - covers filtering, parallel processing, and progress tracking."""

import sys
import pytest
from unittest.mock import MagicMock, patch
from typing import List

from dicom_qc.quickcheck import QuickCheck, SeriesInfo, StudyInfo, PatientInfo
from dicom_qc.core.geometry import QCReport, QCResult


# Create mock ipywidgets module for testing
class MockWidgets:
    """Mock ipywidgets module."""

    class IntProgress:
        def __init__(self, **kwargs):
            self.value = kwargs.get("value", 0)
            self.bar_style = kwargs.get("bar_style", "")

    class FloatProgress:
        def __init__(self, **kwargs):
            self.value = kwargs.get("value", 0)
            self.bar_style = kwargs.get("bar_style", "")

    class HTML:
        def __init__(self, value=""):
            self.value = value

    class VBox:
        def __init__(self, children=None, **kwargs):
            self.children = children or []

    class Layout:
        def __init__(self, **kwargs):
            pass


class MockIPython:
    """Mock IPython.display module."""

    @staticmethod
    def display(*args, **kwargs):
        pass

    @staticmethod
    def clear_output(*args, **kwargs):
        pass


@pytest.fixture(autouse=True)
def mock_jupyter_environment():
    """Mock Jupyter environment for all tests."""
    mock_widgets = MockWidgets()
    mock_ipython_display = MagicMock()
    mock_ipython_display.display = MockIPython.display
    mock_ipython_display.clear_output = MockIPython.clear_output

    with patch.dict(
        sys.modules,
        {
            "ipywidgets": mock_widgets,
            "IPython": MagicMock(),
            "IPython.display": mock_ipython_display,
        },
    ):
        yield


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def qc_instance(temp_data_dir):
    """Create a QuickCheck instance with temp directory."""
    return QuickCheck(data_dir=temp_data_dir)


def create_mock_series(
    uid: str,
    series_number: int,
    description: str,
    has_thumbnail: bool = False,
    has_error: bool = False,
    is_derived: bool = False,
) -> SeriesInfo:
    """Create a mock SeriesInfo for testing."""
    series = SeriesInfo(
        uid=uid,
        series_number=series_number,
        description=description,
        modality="MR",
    )
    if has_thumbnail:
        series.thumbnail = "base64_thumbnail_data"
    if has_error:
        series.error = "Previous error"
    if is_derived:
        series.is_derived = True
    return series


def create_qc_with_series(qc_instance, series_list: List[SeriesInfo]) -> QuickCheck:
    """Add series to a QuickCheck instance."""
    patient = PatientInfo("PAT001", "Test Patient")
    study = StudyInfo(uid="study1", date="2024-01-01", description="Test Study")

    for series in series_list:
        study.series[series.uid] = series

    patient.studies["study1"] = study
    qc_instance.patients = {"PAT001": patient}
    return qc_instance


# ============================================================================
# Test Classes
# ============================================================================


class TestProcessAllFiltering:
    """Tests for series filtering in process_all."""

    def test_skips_series_with_thumbnail(self, qc_instance):
        """Test that series with thumbnails are skipped by default."""
        series_list = [
            create_mock_series("1.2.3.1", 1, "Processed", has_thumbnail=True),
            create_mock_series("1.2.3.2", 2, "Pending", has_thumbnail=False),
        ]
        qc = create_qc_with_series(qc_instance, series_list)

        # Mock process_series to track calls
        processed = []

        def mock_process(series, keep_volume=False):
            processed.append(series.uid)
            series.thumbnail = "new_thumb"

        qc.process_series = mock_process

        qc.process_all()

        # Only pending series should be processed
        assert "1.2.3.1" not in processed
        assert "1.2.3.2" in processed

    def test_skips_series_with_thumbnail_path(self, qc_instance):
        """Test that series with _thumbnail_path are skipped."""
        series = create_mock_series("1.2.3.1", 1, "Has disk thumb")
        series._thumbnail_path = "ab/abc123.jpg"

        qc = create_qc_with_series(qc_instance, [series])

        processed = []

        def mock_process(series, keep_volume=False):
            processed.append(series.uid)

        qc.process_series = mock_process

        qc.process_all()

        assert len(processed) == 0

    def test_skips_derived_series(self, qc_instance):
        """Test that derived series are skipped."""
        series_list = [
            create_mock_series("1.2.3.1", 1, "Derived", is_derived=True),
            create_mock_series("1.2.3.2", 2, "Normal"),
        ]
        qc = create_qc_with_series(qc_instance, series_list)

        processed = []

        def mock_process(series, keep_volume=False):
            processed.append(series.uid)
            series.thumbnail = "thumb"

        qc.process_series = mock_process

        qc.process_all()

        assert "1.2.3.1" not in processed
        assert "1.2.3.2" in processed

    def test_skips_series_with_errors(self, qc_instance):
        """Test that series with errors are skipped by default."""
        series_list = [
            create_mock_series("1.2.3.1", 1, "Has error", has_error=True),
            create_mock_series("1.2.3.2", 2, "Normal"),
        ]
        qc = create_qc_with_series(qc_instance, series_list)

        processed = []

        def mock_process(series, keep_volume=False):
            processed.append(series.uid)
            series.thumbnail = "thumb"

        qc.process_series = mock_process

        qc.process_all()

        assert "1.2.3.1" not in processed
        assert "1.2.3.2" in processed

    def test_retry_errors_processes_error_series(self, qc_instance):
        """Test that retry_errors=True processes series with errors."""
        series_list = [
            create_mock_series("1.2.3.1", 1, "Has error", has_error=True),
            create_mock_series("1.2.3.2", 2, "Normal"),
        ]
        qc = create_qc_with_series(qc_instance, series_list)

        processed = []

        def mock_process(series, keep_volume=False):
            processed.append(series.uid)
            series.thumbnail = "thumb"
            series.error = None  # Clear error on success

        qc.process_series = mock_process

        qc.process_all(retry_errors=True)

        # Both should be processed
        assert "1.2.3.1" in processed
        assert "1.2.3.2" in processed

        # Error should be cleared before processing
        error_series = qc.patients["PAT001"].studies["study1"].series["1.2.3.1"]
        assert error_series.error is None

    def test_reprocess_processes_all_series(self, qc_instance):
        """Test that reprocess=True processes all series including those with thumbnails."""
        series_list = [
            create_mock_series("1.2.3.1", 1, "Processed", has_thumbnail=True),
            create_mock_series("1.2.3.2", 2, "Pending"),
        ]
        qc = create_qc_with_series(qc_instance, series_list)

        processed = []

        def mock_process(series, keep_volume=False):
            processed.append(series.uid)
            series.thumbnail = "new_thumb"

        qc.process_series = mock_process

        qc.process_all(reprocess=True)

        # Both should be processed
        assert "1.2.3.1" in processed
        assert "1.2.3.2" in processed

    def test_no_series_to_process(self, qc_instance):
        """Test handling when all series are already processed."""
        series_list = [
            create_mock_series("1.2.3.1", 1, "Processed", has_thumbnail=True),
        ]
        qc = create_qc_with_series(qc_instance, series_list)

        result = qc.process_all()

        # Should return summary without error
        assert isinstance(result, dict)
        assert "PASS" in result


class TestProcessAllParallel:
    """Tests for parallel processing."""

    def test_processes_all_series(self, temp_data_dir):
        """Test that all series are processed."""
        qc = QuickCheck(data_dir=temp_data_dir)
        qc = create_qc_with_series(
            qc, [create_mock_series(f"1.2.3.{i}", i, f"Series {i}") for i in range(5)]
        )

        processed = []

        def mock_process(series, keep_volume=False):
            processed.append(series.uid)
            series.thumbnail = "thumb"

        qc.process_series = mock_process

        qc.process_all(max_workers=2)

        assert len(processed) == 5
        assert set(processed) == {f"1.2.3.{i}" for i in range(5)}


class TestProcessAllDBCheckpointing:
    """Tests for per-series DB checkpointing during process_all."""

    def test_checkpoint_called_for_each_series(self, qc_instance):
        """Test that _checkpoint_series is called for each processed series."""
        series_list = [
            create_mock_series(f"1.2.3.{i}", i, f"Series {i}") for i in range(5)
        ]
        qc = create_qc_with_series(qc_instance, series_list)

        def mock_process(series, keep_volume=False):
            series.thumbnail = "thumb"

        qc.process_series = mock_process

        checkpoint_count = [0]
        original_checkpoint = qc._checkpoint_series

        def mock_checkpoint(series, ctx):
            checkpoint_count[0] += 1
            return original_checkpoint(series, ctx)

        qc._checkpoint_series = mock_checkpoint

        qc.process_all(max_workers=2)

        # Each series should be checkpointed
        assert checkpoint_count[0] == 5


class TestProcessAllErrorHandling:
    """Tests for error handling during processing."""

    def test_error_captured_on_series(self, qc_instance):
        """Test that processing errors are captured on the series."""
        series = create_mock_series("1.2.3.1", 1, "Will fail")
        qc = create_qc_with_series(qc_instance, [series])

        def mock_process(s, keep_volume=False):
            raise ValueError("Processing failed")

        qc.process_series = mock_process

        qc.process_all(max_workers=2)

        # Error should be set on series
        processed_series = qc.patients["PAT001"].studies["study1"].series["1.2.3.1"]
        assert processed_series.error is not None
        assert "Processing failed" in processed_series.error

    def test_continues_after_error(self, qc_instance):
        """Test that processing continues after an error."""
        series_list = [
            create_mock_series("1.2.3.1", 1, "Will fail"),
            create_mock_series("1.2.3.2", 2, "Will succeed"),
        ]
        qc = create_qc_with_series(qc_instance, series_list)

        processed = []

        def mock_process(series, keep_volume=False):
            processed.append(series.uid)
            if series.uid == "1.2.3.1":
                raise ValueError("Processing failed")
            series.thumbnail = "thumb"

        qc.process_series = mock_process

        qc.process_all(max_workers=2)

        # Both should have been attempted
        assert "1.2.3.1" in processed
        assert "1.2.3.2" in processed

    def test_reprocess_clears_stale_error_on_success(self, qc_instance):
        """Test that reprocess=True clears old errors before retrying."""
        series = create_mock_series("1.2.3.1", 1, "Previously failed", has_error=True)
        qc = create_qc_with_series(qc_instance, [series])

        def mock_process(s, keep_volume=False):
            s.thumbnail = "thumb"

        qc.process_series = mock_process

        qc.process_all(reprocess=True, max_workers=1)

        processed_series = qc.patients["PAT001"].studies["study1"].series["1.2.3.1"]
        assert processed_series.error is None
        assert processed_series.qc_status != "ERROR"


class TestProcessAllProgressTracking:
    """Tests for progress tracking functionality."""

    def test_returns_summary_counts(self, qc_instance):
        """Test that process_all returns correct summary counts."""
        series_list = [
            create_mock_series("1.2.3.1", 1, "Series 1"),
            create_mock_series("1.2.3.2", 2, "Series 2"),
        ]
        qc = create_qc_with_series(qc_instance, series_list)

        # Create mock QC report
        qc_report = QCReport(
            scan_id="test",
            results=[QCResult(status="PASS", check_name="Test", message="OK")],
            overall_status="PASS",
            orientation_labels={},
            primary_plane="AXIAL",
        )

        def mock_process(series, keep_volume=False):
            series.thumbnail = "thumb"
            series.qc_report = qc_report

        qc.process_series = mock_process

        result = qc.process_all()

        assert result["PASS"] == 2

    def test_get_summary_reflects_processing(self, qc_instance):
        """Test that get_summary accurately reflects processing state."""
        series_list = [
            create_mock_series("1.2.3.1", 1, "Series 1"),
            create_mock_series("1.2.3.2", 2, "Series 2"),
            create_mock_series("1.2.3.3", 3, "Series 3"),
        ]
        qc = create_qc_with_series(qc_instance, series_list)

        # Initially all pending
        summary = qc.get_summary()
        assert summary["PENDING"] == 3

        # After partial processing
        qc.patients["PAT001"].studies["study1"].series["1.2.3.1"].thumbnail = "thumb"
        qc.patients["PAT001"].studies["study1"].series["1.2.3.1"].qc_report = QCReport(
            scan_id="test",
            results=[QCResult(status="PASS", check_name="Test", message="OK")],
            overall_status="PASS",
            orientation_labels={},
            primary_plane="AXIAL",
        )

        summary = qc.get_summary()
        assert summary["PASS"] == 1
        assert summary["PENDING"] == 2


class TestProcessAllBackwardCompatibility:
    """Tests for backward compatibility."""

    def test_process_all_interactive_alias(self, qc_instance):
        """Test that process_all_interactive is an alias for process_all."""
        assert qc_instance.process_all_interactive == qc_instance.process_all


class TestFormatTime:
    """Tests for time formatting helper."""

    def test_format_time_seconds(self):
        """Test formatting values under 60 seconds."""
        qc = QuickCheck()
        assert qc._format_time(5) == "5s"
        assert qc._format_time(30.4) == "30s"
        assert qc._format_time(59) == "59s"

    def test_format_time_minutes(self):
        """Test formatting values between 1 and 60 minutes."""
        qc = QuickCheck()
        assert qc._format_time(60) == "1m 0s"
        assert qc._format_time(90) == "1m 30s"
        assert qc._format_time(3599) == "59m 59s"

    def test_format_time_hours(self):
        """Test formatting values over 1 hour."""
        qc = QuickCheck()
        assert qc._format_time(3600) == "1h 0m"
        assert qc._format_time(5400) == "1h 30m"
        assert qc._format_time(7261) == "2h 1m"
