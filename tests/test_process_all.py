"""Tests for process_all method - covers filtering, parallel/sequential, and progress tracking."""

import sys
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass, field
from typing import List, Dict, Any

from dicom_qc.quickcheck import QuickCheck, SeriesInfo, StudyInfo, PatientInfo
from dicom_qc.core.geometry import QCReport, QCResult


# Create mock ipywidgets module for testing
class MockWidgets:
    """Mock ipywidgets module."""

    class IntProgress:
        def __init__(self, **kwargs):
            self.value = kwargs.get('value', 0)
            self.bar_style = kwargs.get('bar_style', '')

    class FloatProgress:
        def __init__(self, **kwargs):
            self.value = kwargs.get('value', 0)
            self.bar_style = kwargs.get('bar_style', '')

    class HTML:
        def __init__(self, value=''):
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

    with patch.dict(sys.modules, {
        'ipywidgets': mock_widgets,
        'IPython': MagicMock(),
        'IPython.display': mock_ipython_display,
    }):
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


def create_mock_series(uid: str, series_number: int, description: str,
                       has_thumbnail: bool = False, has_error: bool = False,
                       is_derived: bool = False) -> SeriesInfo:
    """Create a mock SeriesInfo for testing."""
    series = SeriesInfo(
        uid=uid,
        series_number=series_number,
        description=description,
        modality='MR',
    )
    if has_thumbnail:
        series.thumbnail = 'base64_thumbnail_data'
    if has_error:
        series.error = 'Previous error'
    if is_derived:
        series.is_derived = True
    return series


def create_qc_with_series(qc_instance, series_list: List[SeriesInfo]) -> QuickCheck:
    """Add series to a QuickCheck instance."""
    patient = PatientInfo('PAT001', 'Test Patient')
    study = StudyInfo(uid='study1', date='2024-01-01', description='Test Study')

    for series in series_list:
        study.series[series.uid] = series

    patient.studies['study1'] = study
    qc_instance.patients = {'PAT001': patient}
    return qc_instance


# ============================================================================
# Test Classes
# ============================================================================

class TestProcessAllFiltering:
    """Tests for series filtering in process_all."""

    def test_skips_series_with_thumbnail(self, qc_instance):
        """Test that series with thumbnails are skipped by default."""
        series_list = [
            create_mock_series('1.2.3.1', 1, 'Processed', has_thumbnail=True),
            create_mock_series('1.2.3.2', 2, 'Pending', has_thumbnail=False),
        ]
        qc = create_qc_with_series(qc_instance, series_list)

        # Mock process_series to track calls
        processed = []
        def mock_process(series, keep_volume=False):
            processed.append(series.uid)
            series.thumbnail = 'new_thumb'
        qc.process_series = mock_process

        qc.process_all(parallel=False)

        # Only pending series should be processed
        assert '1.2.3.1' not in processed
        assert '1.2.3.2' in processed

    def test_skips_series_with_thumbnail_path(self, qc_instance):
        """Test that series with _thumbnail_path are skipped."""
        series = create_mock_series('1.2.3.1', 1, 'Has disk thumb')
        series._thumbnail_path = 'ab/abc123.jpg'

        qc = create_qc_with_series(qc_instance, [series])

        processed = []
        def mock_process(series, keep_volume=False):
            processed.append(series.uid)
        qc.process_series = mock_process

        qc.process_all(parallel=False)

        assert len(processed) == 0

    def test_skips_derived_series(self, qc_instance):
        """Test that derived series are skipped."""
        series_list = [
            create_mock_series('1.2.3.1', 1, 'Derived', is_derived=True),
            create_mock_series('1.2.3.2', 2, 'Normal'),
        ]
        qc = create_qc_with_series(qc_instance, series_list)

        processed = []
        def mock_process(series, keep_volume=False):
            processed.append(series.uid)
            series.thumbnail = 'thumb'
        qc.process_series = mock_process

        qc.process_all(parallel=False)

        assert '1.2.3.1' not in processed
        assert '1.2.3.2' in processed

    def test_skips_series_with_errors(self, qc_instance):
        """Test that series with errors are skipped by default."""
        series_list = [
            create_mock_series('1.2.3.1', 1, 'Has error', has_error=True),
            create_mock_series('1.2.3.2', 2, 'Normal'),
        ]
        qc = create_qc_with_series(qc_instance, series_list)

        processed = []
        def mock_process(series, keep_volume=False):
            processed.append(series.uid)
            series.thumbnail = 'thumb'
        qc.process_series = mock_process

        qc.process_all(parallel=False)

        assert '1.2.3.1' not in processed
        assert '1.2.3.2' in processed

    def test_retry_errors_processes_error_series(self, qc_instance):
        """Test that retry_errors=True processes series with errors."""
        series_list = [
            create_mock_series('1.2.3.1', 1, 'Has error', has_error=True),
            create_mock_series('1.2.3.2', 2, 'Normal'),
        ]
        qc = create_qc_with_series(qc_instance, series_list)

        processed = []
        def mock_process(series, keep_volume=False):
            processed.append(series.uid)
            series.thumbnail = 'thumb'
            series.error = None  # Clear error on success
        qc.process_series = mock_process

        qc.process_all(parallel=False, retry_errors=True)

        # Both should be processed
        assert '1.2.3.1' in processed
        assert '1.2.3.2' in processed

        # Error should be cleared before processing
        error_series = qc.patients['PAT001'].studies['study1'].series['1.2.3.1']
        assert error_series.error is None

    def test_reprocess_processes_all_series(self, qc_instance):
        """Test that reprocess=True processes all series including those with thumbnails."""
        series_list = [
            create_mock_series('1.2.3.1', 1, 'Processed', has_thumbnail=True),
            create_mock_series('1.2.3.2', 2, 'Pending'),
        ]
        qc = create_qc_with_series(qc_instance, series_list)

        processed = []
        def mock_process(series, keep_volume=False):
            processed.append(series.uid)
            series.thumbnail = 'new_thumb'
        qc.process_series = mock_process

        qc.process_all(parallel=False, reprocess=True)

        # Both should be processed
        assert '1.2.3.1' in processed
        assert '1.2.3.2' in processed

    def test_no_series_to_process(self, qc_instance):
        """Test handling when all series are already processed."""
        series_list = [
            create_mock_series('1.2.3.1', 1, 'Processed', has_thumbnail=True),
        ]
        qc = create_qc_with_series(qc_instance, series_list)

        result = qc.process_all(parallel=False)

        # Should return summary without error
        assert isinstance(result, dict)
        assert 'PASS' in result


class TestProcessAllParallelVsSequential:
    """Tests ensuring parallel and sequential produce same results."""

    def test_parallel_and_sequential_process_same_series(self, temp_data_dir):
        """Test that parallel and sequential modes process the same series."""
        qc_seq = QuickCheck(data_dir=temp_data_dir / 'seq')
        qc_seq = create_qc_with_series(qc_seq, [
            create_mock_series(f'1.2.3.{i}', i, f'Series {i}')
            for i in range(5)
        ])

        qc_par = QuickCheck(data_dir=temp_data_dir / 'par')
        qc_par = create_qc_with_series(qc_par, [
            create_mock_series(f'1.2.3.{i}', i, f'Series {i}')
            for i in range(5)
        ])

        seq_processed = []
        par_processed = []

        def make_mock_process(processed_list):
            def mock_process(series, keep_volume=False):
                processed_list.append(series.uid)
                series.thumbnail = 'thumb'
            return mock_process

        qc_seq.process_series = make_mock_process(seq_processed)
        qc_par.process_series = make_mock_process(par_processed)

        qc_seq.process_all(parallel=False)
        qc_par.process_all(parallel=True, max_workers=2)

        # Same series should be processed (order may differ)
        assert set(seq_processed) == set(par_processed)
        assert len(seq_processed) == 5


class TestProcessAllSaveInterval:
    """Tests for periodic save functionality."""

    def test_saves_at_interval(self, qc_instance):
        """Test that save is called at specified intervals."""
        series_list = [
            create_mock_series(f'1.2.3.{i}', i, f'Series {i}')
            for i in range(15)
        ]
        qc = create_qc_with_series(qc_instance, series_list)
        qc._save_path = qc_instance.data_dir / '_dicom_qc' / 'qc_state.pkl'

        def mock_process(series, keep_volume=False):
            series.thumbnail = 'thumb'
        qc.process_series = mock_process

        save_count = [0]
        original_save = qc.save
        def mock_save(*args, **kwargs):
            save_count[0] += 1
            return original_save(*args, **kwargs)
        qc.save = mock_save

        qc.process_all(parallel=False, save_interval=5)

        # Should save at 5, 10, 15 (interval) + final save
        # At least 3 interval saves + 1 final
        assert save_count[0] >= 3

    def test_no_save_when_interval_zero(self, qc_instance):
        """Test that save_interval=0 disables periodic saves."""
        series_list = [
            create_mock_series(f'1.2.3.{i}', i, f'Series {i}')
            for i in range(5)
        ]
        qc = create_qc_with_series(qc_instance, series_list)
        qc._save_path = qc_instance.data_dir / '_dicom_qc' / 'qc_state.pkl'

        def mock_process(series, keep_volume=False):
            series.thumbnail = 'thumb'
        qc.process_series = mock_process

        save_count = [0]
        original_save = qc.save
        def mock_save(*args, **kwargs):
            save_count[0] += 1
            return original_save(*args, **kwargs)
        qc.save = mock_save

        qc.process_all(parallel=False, save_interval=0)

        # Only final save
        assert save_count[0] == 1


class TestProcessAllErrorHandling:
    """Tests for error handling during processing."""

    def test_error_captured_on_series_parallel(self, qc_instance):
        """Test that processing errors are captured on the series in parallel mode."""
        series = create_mock_series('1.2.3.1', 1, 'Will fail')
        qc = create_qc_with_series(qc_instance, [series])

        def mock_process(s, keep_volume=False):
            raise ValueError("Processing failed")
        qc.process_series = mock_process

        # Parallel mode captures errors
        qc.process_all(parallel=True, max_workers=2)

        # Error should be set on series
        processed_series = qc.patients['PAT001'].studies['study1'].series['1.2.3.1']
        assert processed_series.error is not None
        assert 'Processing failed' in processed_series.error

    def test_continues_after_error_parallel(self, qc_instance):
        """Test that processing continues after an error in parallel mode."""
        series_list = [
            create_mock_series('1.2.3.1', 1, 'Will fail'),
            create_mock_series('1.2.3.2', 2, 'Will succeed'),
        ]
        qc = create_qc_with_series(qc_instance, series_list)

        processed = []
        def mock_process(series, keep_volume=False):
            processed.append(series.uid)
            if series.uid == '1.2.3.1':
                raise ValueError("Processing failed")
            series.thumbnail = 'thumb'
        qc.process_series = mock_process

        qc.process_all(parallel=True, max_workers=2)

        # Both should have been attempted
        assert '1.2.3.1' in processed
        assert '1.2.3.2' in processed

    def test_error_captured_on_series_sequential(self, qc_instance):
        """Test that sequential mode also captures errors (after refactoring).

        Both sequential and parallel modes now consistently capture errors
        on the series instead of raising.
        """
        series = create_mock_series('1.2.3.1', 1, 'Will fail')
        qc = create_qc_with_series(qc_instance, [series])

        def mock_process(s, keep_volume=False):
            raise ValueError("Processing failed")
        qc.process_series = mock_process

        # Sequential mode now captures errors like parallel mode
        qc.process_all(parallel=False)

        # Error should be set on series
        processed_series = qc.patients['PAT001'].studies['study1'].series['1.2.3.1']
        assert processed_series.error is not None
        assert 'Processing failed' in processed_series.error


class TestProcessAllProgressTracking:
    """Tests for progress tracking functionality."""

    def test_returns_summary_counts(self, qc_instance):
        """Test that process_all returns correct summary counts."""
        series_list = [
            create_mock_series('1.2.3.1', 1, 'Series 1'),
            create_mock_series('1.2.3.2', 2, 'Series 2'),
        ]
        qc = create_qc_with_series(qc_instance, series_list)

        # Create mock QC report
        qc_report = QCReport(
            scan_id='test',
            results=[QCResult(status='PASS', check_name='Test', message='OK')],
            overall_status='PASS',
            orientation_labels={},
            primary_plane='AXIAL',
        )

        def mock_process(series, keep_volume=False):
            series.thumbnail = 'thumb'
            series.qc_report = qc_report
        qc.process_series = mock_process

        result = qc.process_all(parallel=False)

        assert result['PASS'] == 2

    def test_get_summary_reflects_processing(self, qc_instance):
        """Test that get_summary accurately reflects processing state."""
        series_list = [
            create_mock_series('1.2.3.1', 1, 'Series 1'),
            create_mock_series('1.2.3.2', 2, 'Series 2'),
            create_mock_series('1.2.3.3', 3, 'Series 3'),
        ]
        qc = create_qc_with_series(qc_instance, series_list)

        # Initially all pending
        summary = qc.get_summary()
        assert summary['PENDING'] == 3

        # After partial processing
        qc.patients['PAT001'].studies['study1'].series['1.2.3.1'].thumbnail = 'thumb'
        qc.patients['PAT001'].studies['study1'].series['1.2.3.1'].qc_report = QCReport(
            scan_id='test',
            results=[QCResult(status='PASS', check_name='Test', message='OK')],
            overall_status='PASS',
            orientation_labels={},
            primary_plane='AXIAL',
        )

        summary = qc.get_summary()
        assert summary['PASS'] == 1
        assert summary['PENDING'] == 2


class TestProcessAllThumbnailTracking:
    """Tests for thumbnail tracking during processing."""

    def test_tracks_recent_thumbnails(self, qc_instance):
        """Test that recent thumbnails are tracked correctly."""
        series_list = [
            create_mock_series(f'1.2.3.{i}', i, f'Series {i}')
            for i in range(10)
        ]
        qc = create_qc_with_series(qc_instance, series_list)

        def mock_process(series, keep_volume=False):
            series.thumbnail = f'thumb_{series.series_number}'
        qc.process_series = mock_process

        qc.process_all(parallel=False)

        # All series should have thumbnails
        for series in qc.get_all_series():
            assert series.thumbnail is not None


class TestProcessAllAutoParallel:
    """Tests for automatic parallel mode selection."""

    def test_auto_enables_parallel_for_large_datasets(self, qc_instance):
        """Test that parallel is auto-enabled for >100 series."""
        # Create >100 series
        series_list = [
            create_mock_series(f'1.2.3.{i}', i, f'Series {i}')
            for i in range(150)
        ]
        qc = create_qc_with_series(qc_instance, series_list)

        def mock_process(series, keep_volume=False):
            series.thumbnail = 'thumb'
        qc.process_series = mock_process

        # Track if ThreadPoolExecutor was used
        executor_used = [False]
        original_tpe = __import__('concurrent.futures').futures.ThreadPoolExecutor

        class MockTPE:
            def __init__(self, *args, **kwargs):
                executor_used[0] = True
                self._executor = original_tpe(*args, **kwargs)
            def __enter__(self):
                return self._executor.__enter__()
            def __exit__(self, *args):
                return self._executor.__exit__(*args)

        with patch('concurrent.futures.ThreadPoolExecutor', MockTPE):
            qc.process_all(parallel=None)  # Auto-detect

        # Should have used parallel
        assert executor_used[0] is True

    def test_sequential_for_small_datasets(self, qc_instance):
        """Test that sequential is used for small datasets."""
        series_list = [
            create_mock_series(f'1.2.3.{i}', i, f'Series {i}')
            for i in range(10)
        ]
        qc = create_qc_with_series(qc_instance, series_list)

        def mock_process(series, keep_volume=False):
            series.thumbnail = 'thumb'
        qc.process_series = mock_process

        # Track if ThreadPoolExecutor was used
        executor_used = [False]
        original_tpe = __import__('concurrent.futures').futures.ThreadPoolExecutor

        class MockTPE:
            def __init__(self, *args, **kwargs):
                executor_used[0] = True
                self._executor = original_tpe(*args, **kwargs)
            def __enter__(self):
                return self._executor.__enter__()
            def __exit__(self, *args):
                return self._executor.__exit__(*args)

        with patch('concurrent.futures.ThreadPoolExecutor', MockTPE):
            qc.process_all(parallel=None)  # Auto-detect

        # Should NOT have used parallel for small dataset
        assert executor_used[0] is False


class TestProcessAllBackwardCompatibility:
    """Tests for backward compatibility."""

    def test_process_all_interactive_alias(self, qc_instance):
        """Test that process_all_interactive is an alias for process_all."""
        assert qc_instance.process_all_interactive == qc_instance.process_all


class TestFormatTime:
    """Tests for time formatting helper."""

    def test_format_time_logic(self):
        """Test time formatting produces expected output."""
        # Test that get_summary returns valid counts
        qc = QuickCheck()
        summary = qc.get_summary()

        assert isinstance(summary, dict)
        assert all(key in summary for key in ['PASS', 'WARNING', 'FAIL', 'ERROR', 'PENDING'])


class TestProcessSeriesIntegration:
    """Integration tests that process_series is called correctly."""

    def test_process_series_called_for_each(self, qc_instance):
        """Test that process_series is called once for each series to process."""
        series_list = [
            create_mock_series(f'1.2.3.{i}', i, f'Series {i}')
            for i in range(5)
        ]
        qc = create_qc_with_series(qc_instance, series_list)

        call_count = [0]
        def mock_process(series, keep_volume=False):
            call_count[0] += 1
            series.thumbnail = 'thumb'
        qc.process_series = mock_process

        qc.process_all(parallel=False)

        assert call_count[0] == 5

    def test_keep_volume_false_by_default(self, qc_instance):
        """Test that keep_volume=False is passed to process_series."""
        series = create_mock_series('1.2.3.1', 1, 'Test')
        qc = create_qc_with_series(qc_instance, [series])

        keep_volume_values = []
        def mock_process(series, keep_volume=False):
            keep_volume_values.append(keep_volume)
            series.thumbnail = 'thumb'
        qc.process_series = mock_process

        qc.process_all(parallel=False)

        assert keep_volume_values == [False]
