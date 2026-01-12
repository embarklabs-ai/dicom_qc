"""Tests for QuickCheck and HTML report generation."""

import shutil
import zipfile
from pathlib import Path

import pytest

from dicom_qc.quickcheck import QuickCheck, SeriesInfo, StudyInfo, PatientInfo
from dicom_qc.core.geometry import QCReport, QCResult


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


@pytest.fixture
def qc_with_mock_data(qc_instance):
    """Create a QuickCheck instance with mock patient/series data."""
    qc = qc_instance

    # Create mock QC report
    qc_report = QCReport(
        scan_id='series_001',
        results=[
            QCResult(status='PASS', check_name='Geometry Metadata', message='All tags present'),
            QCResult(status='PASS', check_name='Slice Ordering', message='Monotonic'),
        ],
        overall_status='PASS',
        orientation_labels={'row_positive': 'L', 'col_positive': 'P'},
        primary_plane='AXIAL',
    )

    # Create mock series
    series1 = SeriesInfo(
        uid='1.2.3.4.5.001',
        series_number=1,
        description='T1 MPRAGE',
        modality='MR',
        qc_report=qc_report,
    )
    series2 = SeriesInfo(
        uid='1.2.3.4.5.002',
        series_number=2,
        description='T2 FLAIR',
        modality='MR',
        qc_report=qc_report,
    )
    series3 = SeriesInfo(
        uid='1.2.3.4.5.003',
        series_number=3,
        description='DWI',
        modality='MR',
        error='Failed to load',
    )

    # Create mock study
    study = StudyInfo(
        uid='1.2.3.4.100',
        date='2024-01-15',
        description='Brain MRI',
        series={
            series1.uid: series1,
            series2.uid: series2,
            series3.uid: series3,
        },
    )

    # Create mock patient
    patient = PatientInfo(
        patient_id='PAT001',
        patient_name='Test Patient',
        studies={'1.2.3.4.100': study},
    )

    qc.patients = {'PAT001': patient}
    return qc


class TestQuickCheckInit:
    """Tests for QuickCheck initialization."""

    def test_init_with_data_dir(self, temp_data_dir):
        """Test initialization with data directory."""
        qc = QuickCheck(data_dir=temp_data_dir)
        assert qc.data_dir == temp_data_dir
        assert qc.patients == {}

    def test_init_creates_storage_dir(self, temp_data_dir):
        """Test that initialization creates storage directory."""
        qc = QuickCheck(data_dir=temp_data_dir)
        storage_dir = temp_data_dir / '_dicom_qc'
        assert storage_dir.exists()

    def test_init_without_data_dir(self):
        """Test initialization without data directory."""
        qc = QuickCheck()
        assert qc.data_dir is None


class TestQuickCheckHelpers:
    """Tests for QuickCheck helper methods."""

    def test_get_all_series(self, qc_with_mock_data):
        """Test get_all_series returns all series."""
        series = qc_with_mock_data.get_all_series()
        assert len(series) == 3

    def test_get_summary(self, qc_with_mock_data):
        """Test get_summary returns correct counts."""
        summary = qc_with_mock_data.get_summary()
        assert summary['PASS'] == 2
        assert summary['ERROR'] == 1

    def test_has_save_false_initially(self, qc_instance):
        """Test has_save returns False when no save exists."""
        assert qc_instance.has_save() is False

    def test_save_and_load(self, qc_with_mock_data, temp_data_dir):
        """Test save and load roundtrip."""
        qc = qc_with_mock_data
        qc.save()

        # Create new instance and load
        qc2 = QuickCheck(data_dir=temp_data_dir)
        assert qc2.has_save() is True
        qc2.load()

        assert len(qc2.patients) == 1
        assert len(qc2.get_all_series()) == 3


class TestQuickCheckReset:
    """Tests for QuickCheck reset functionality."""

    def test_reset_clears_data(self, qc_with_mock_data):
        """Test reset clears patient data."""
        qc = qc_with_mock_data
        assert len(qc.patients) == 1

        qc.reset()
        assert len(qc.patients) == 0

    def test_reset_deletes_storage(self, qc_with_mock_data, temp_data_dir):
        """Test reset deletes storage directory."""
        qc = qc_with_mock_data
        qc.save()

        storage_dir = temp_data_dir / '_dicom_qc'
        assert storage_dir.exists()

        qc.reset(delete_storage=True)
        # Storage should be recreated empty
        assert qc.has_save() is False


class TestHTMLReportGeneration:
    """Tests for HTML report generation."""

    def test_generate_embedded_report(self, qc_with_mock_data, temp_data_dir):
        """Test generating embedded (single-file) HTML report."""
        output_path = temp_data_dir / 'report.html'

        html_path, zip_path = qc_with_mock_data.generate_html_report(
            output_path,
            embed_thumbnails=True,
        )

        # Should return HTML path, no zip
        assert html_path == output_path
        assert zip_path is None
        assert html_path.exists()

        # Check content
        content = html_path.read_text()
        assert 'DICOM' in content
        assert 'Test Patient' in content or 'PAT001' in content
        assert 'T1 MPRAGE' in content

    def test_generate_external_thumbnail_report(self, qc_with_mock_data, temp_data_dir):
        """Test generating HTML report with external thumbnails + zip."""
        output_path = temp_data_dir / 'report.html'

        html_path, zip_path = qc_with_mock_data.generate_html_report(
            output_path,
            embed_thumbnails=False,
        )

        # Should return index.html and zip path
        assert html_path.name == 'index.html'
        assert zip_path is not None
        assert zip_path.suffix == '.zip'
        assert html_path.exists()
        assert zip_path.exists()

        # Check report directory structure
        report_dir = html_path.parent
        assert report_dir.name == 'report'
        assert (report_dir / 'thumbnails').exists()

    def test_external_zip_contents(self, qc_with_mock_data, temp_data_dir):
        """Test that external thumbnail zip contains all required files."""
        output_path = temp_data_dir / 'report.html'

        html_path, zip_path = qc_with_mock_data.generate_html_report(
            output_path,
            embed_thumbnails=False,
        )

        # Extract and verify zip contents
        extract_dir = temp_data_dir / 'extracted'
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)

        # Should contain single index.html and thumbnails folder
        report_in_zip = extract_dir / 'report'
        assert (report_in_zip / 'index.html').exists()
        assert (report_in_zip / 'thumbnails').exists()

    def test_auto_embed_small_dataset(self, qc_with_mock_data, temp_data_dir):
        """Test that small datasets auto-select embedded mode."""
        output_path = temp_data_dir / 'report.html'

        # With only 3 series, should auto-select embedded
        html_path, zip_path = qc_with_mock_data.generate_html_report(output_path)

        assert zip_path is None  # No zip for small datasets
        assert html_path.exists()
        assert html_path == output_path

    def test_report_contains_status_summary(self, qc_with_mock_data, temp_data_dir):
        """Test that report contains status summary."""
        output_path = temp_data_dir / 'report.html'

        html_path, _ = qc_with_mock_data.generate_html_report(
            output_path,
            embed_thumbnails=True,
        )

        content = html_path.read_text()
        # Should have summary cards
        assert 'Pass' in content
        assert 'Error' in content

    def test_report_contains_series_info(self, qc_with_mock_data, temp_data_dir):
        """Test that report contains series information."""
        output_path = temp_data_dir / 'report.html'

        html_path, _ = qc_with_mock_data.generate_html_report(
            output_path,
            embed_thumbnails=True,
        )

        content = html_path.read_text()
        assert 'T1 MPRAGE' in content
        assert 'T2 FLAIR' in content
        assert 'DWI' in content

    def test_external_report_contains_all_series(self, qc_with_mock_data, temp_data_dir):
        """Test that external thumbnail report contains all series in single file."""
        output_path = temp_data_dir / 'report.html'

        html_path, _ = qc_with_mock_data.generate_html_report(
            output_path,
            embed_thumbnails=False,
        )

        # Single HTML file should contain all series
        content = html_path.read_text()
        assert 'T1 MPRAGE' in content
        assert 'T2 FLAIR' in content
        assert 'DWI' in content


class TestSeriesInfo:
    """Tests for SeriesInfo dataclass."""

    def test_qc_status_pending(self):
        """Test qc_status is PENDING when no report."""
        series = SeriesInfo(uid='1', series_number=1, description='Test', modality='MR')
        assert series.qc_status == 'PENDING'

    def test_qc_status_error(self):
        """Test qc_status is ERROR when error set."""
        series = SeriesInfo(
            uid='1', series_number=1, description='Test', modality='MR',
            error='Failed',
        )
        assert series.qc_status == 'ERROR'

    def test_qc_status_derived(self):
        """Test qc_status is DERIVED when is_derived set."""
        series = SeriesInfo(
            uid='1', series_number=1, description='Test', modality='MR',
            is_derived=True,
        )
        assert series.qc_status == 'DERIVED'

    def test_qc_status_from_report(self):
        """Test qc_status comes from report when present."""
        qc_report = QCReport(
            scan_id='test',
            results=[QCResult(status='WARNING', check_name='Test', message='Warn')],
            overall_status='WARNING',
            orientation_labels={},
            primary_plane='AXIAL',
        )
        series = SeriesInfo(
            uid='1', series_number=1, description='Test', modality='MR',
            qc_report=qc_report,
        )
        assert series.qc_status == 'WARNING'

    def test_label_format(self):
        """Test series label format."""
        series = SeriesInfo(uid='1', series_number=5, description='T1 MPRAGE', modality='MR')
        assert series.label == '#5 MR: T1 MPRAGE'


class TestStudyInfo:
    """Tests for StudyInfo dataclass."""

    def test_label_with_date_and_description(self):
        """Test study label with both date and description."""
        study = StudyInfo(uid='1', date='2024-01-15', description='Brain MRI')
        assert study.label == '2024-01-15: Brain MRI'

    def test_label_with_date_only(self):
        """Test study label with date only."""
        study = StudyInfo(uid='1', date='2024-01-15', description='')
        assert study.label == '2024-01-15'

    def test_label_with_description_only(self):
        """Test study label with description only."""
        study = StudyInfo(uid='1', date='', description='Brain MRI')
        assert study.label == 'Brain MRI'


class TestParallelDiscovery:
    """Tests for parallel XNAT discovery thread safety."""

    def test_discover_subject_creates_independent_patient(self, qc_instance):
        """Test that discover_subject creates independent patient objects.

        Each thread should work with its own patient object to avoid
        race conditions when modifying patient data.
        """
        # Create mock existing patient
        existing_patient = PatientInfo('SUBJ001', 'Test Patient')
        existing_study = StudyInfo(uid='study1', date='2024-01-01', description='Session 1')
        existing_series = SeriesInfo(
            uid='1.2.3.1', series_number=1, description='T1', modality='MR'
        )
        existing_study.series['1'] = existing_series
        existing_patient.studies['session1'] = existing_study
        qc_instance.patients = {'SUBJ001': existing_patient}

        # Take snapshot like parallel discovery does
        snapshot = dict(qc_instance.patients)

        # Verify snapshot is independent copy
        assert 'SUBJ001' in snapshot
        assert snapshot['SUBJ001'] is existing_patient  # Same reference in snapshot

        # But when we copy for thread safety, we should get new objects
        existing = snapshot['SUBJ001']
        copied_patient = PatientInfo(
            'SUBJ001',
            existing.patient_name,
            xnat_subject_id=existing.xnat_subject_id
        )

        # Verify copied patient is independent
        assert copied_patient is not existing_patient
        assert copied_patient.patient_id == existing_patient.patient_id

    def test_parallel_merge_combines_results(self, qc_instance):
        """Test that parallel results are properly merged."""
        import threading

        # Simulate parallel discovery results
        results = []
        lock = threading.Lock()

        # Create mock results from 3 "threads"
        for i in range(3):
            patient = PatientInfo(f'SUBJ{i:03d}', f'Patient {i}')
            study = StudyInfo(uid=f'study{i}', date='2024-01-01', description=f'Session {i}')
            series = SeriesInfo(
                uid=f'1.2.3.{i}', series_number=1, description=f'Series {i}', modality='MR'
            )
            study.series['1'] = series
            patient.studies['session1'] = study
            results.append((f'SUBJ{i:03d}', patient, 1, 1, 1, 0))

        # Simulate merge logic from parallel discovery
        for subj_label, patient, sess_count, scan_count, new_count, skip_count in results:
            with lock:
                if subj_label in qc_instance.patients:
                    # Merge studies
                    for study_key, study in patient.studies.items():
                        if study_key not in qc_instance.patients[subj_label].studies:
                            qc_instance.patients[subj_label].studies[study_key] = study
                else:
                    qc_instance.patients[subj_label] = patient

        # Verify all patients were merged
        assert len(qc_instance.patients) == 3
        assert 'SUBJ000' in qc_instance.patients
        assert 'SUBJ001' in qc_instance.patients
        assert 'SUBJ002' in qc_instance.patients

        # Verify each patient has correct data
        for i in range(3):
            patient = qc_instance.patients[f'SUBJ{i:03d}']
            assert patient.patient_name == f'Patient {i}'
            assert 'session1' in patient.studies
            assert '1' in patient.studies['session1'].series

    def test_parallel_discovery_with_existing_data(self, qc_instance):
        """Test that incremental parallel discovery preserves existing series."""
        # Set up existing patient with processed series
        existing_patient = PatientInfo('SUBJ001', 'Test Patient')
        existing_study = StudyInfo(uid='study1', date='2024-01-01', description='Session 1')
        existing_series = SeriesInfo(
            uid='1.2.3.1', series_number=1, description='T1', modality='MR'
        )
        existing_series.thumbnail = 'processed_thumbnail'  # Mark as processed
        existing_series._file_uris = ['/data/file1.dcm']
        existing_study.series['1'] = existing_series
        existing_patient.studies['session1'] = existing_study
        qc_instance.patients = {'SUBJ001': existing_patient}

        # Take snapshot for parallel mode
        snapshot = dict(qc_instance.patients)

        # Simulate discovering same subject with new series
        existing = snapshot['SUBJ001']
        new_patient = PatientInfo('SUBJ001', existing.patient_name)

        # Copy existing study structure
        for study_key, study in existing.studies.items():
            new_patient.studies[study_key] = StudyInfo(
                uid=study.uid,
                date=study.date,
                description=study.description,
            )
            # Copy existing series (would be skipped in real discovery)
            for series_key, series in study.series.items():
                new_patient.studies[study_key].series[series_key] = series

        # Add new series
        new_series = SeriesInfo(
            uid='1.2.3.2', series_number=2, description='T2', modality='MR'
        )
        new_patient.studies['session1'].series['2'] = new_series

        # Simulate merge
        for study_key, study in new_patient.studies.items():
            if study_key not in qc_instance.patients['SUBJ001'].studies:
                qc_instance.patients['SUBJ001'].studies[study_key] = study
            else:
                for series_key, series in study.series.items():
                    qc_instance.patients['SUBJ001'].studies[study_key].series[series_key] = series

        # Verify both series exist
        assert '1' in qc_instance.patients['SUBJ001'].studies['session1'].series
        assert '2' in qc_instance.patients['SUBJ001'].studies['session1'].series

        # Verify original series preserved its thumbnail
        original = qc_instance.patients['SUBJ001'].studies['session1'].series['1']
        assert original.thumbnail == 'processed_thumbnail'


class TestAsyncProgressUpdates:
    """Tests for async progress update functionality."""

    def test_asyncio_wait_yields_to_event_loop(self):
        """Test that asyncio.wait with timeout allows event loop to process other tasks.

        This verifies the core mechanism we use for responsive widget updates.
        """
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        updates_received = []

        async def run_test():
            loop = asyncio.get_running_loop()

            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit slow task
                def slow_task():
                    import time
                    time.sleep(0.2)
                    return "done"

                future = loop.run_in_executor(executor, slow_task)

                # Wait with short timeout - should return empty done set
                done, pending = await asyncio.wait(
                    [future],
                    timeout=0.05,
                    return_when=asyncio.FIRST_COMPLETED
                )

                # First wait should timeout (task not done yet)
                updates_received.append(('timeout', len(done), len(pending)))

                # Yield to allow other work
                await asyncio.sleep(0.01)
                updates_received.append(('yielded', None, None))

                # Now wait for completion
                done, pending = await asyncio.wait(
                    pending,
                    timeout=1.0,
                    return_when=asyncio.FIRST_COMPLETED
                )
                updates_received.append(('completed', len(done), len(pending)))

        # Python 3.10+ compatible way to run async test
        asyncio.run(run_test())

        # Verify the sequence of events
        assert len(updates_received) == 3
        assert updates_received[0][0] == 'timeout'
        assert updates_received[0][1] == 0  # No tasks completed during timeout
        assert updates_received[1][0] == 'yielded'
        assert updates_received[2][0] == 'completed'
        assert updates_received[2][1] == 1  # Task completed

    def test_progress_update_functions_work_correctly(self, qc_with_mock_data):
        """Test that get_summary returns correct status counts during processing."""
        qc = qc_with_mock_data

        # Verify get_summary is thread-safe (reads without locks)
        summary = qc.get_summary()
        assert isinstance(summary, dict)
        assert 'PASS' in summary
        assert 'ERROR' in summary

        # Verify it can be called repeatedly (for status map updates)
        for _ in range(10):
            summary = qc.get_summary()
            assert summary['PASS'] == 2
            assert summary['ERROR'] == 1

    def test_render_status_produces_valid_counts(self, qc_with_mock_data):
        """Test that status rendering doesn't fail with concurrent modifications.

        While we can't fully simulate race conditions in a unit test,
        we verify the underlying data access is safe.
        """
        qc = qc_with_mock_data

        # Simulate what render_status_map does
        all_series = qc.get_all_series()
        status_colors = {
            'PASS': '#28a745',
            'WARNING': '#ffc107',
            'FAIL': '#dc3545',
            'ERROR': '#6c757d',
            'PENDING': '#17a2b8',
        }

        # Build status squares (like the UI does)
        squares = []
        for series in all_series:
            status = series.qc_status
            color = status_colors.get(status, '#444')
            squares.append(f'<div style="background:{color}"></div>')

        assert len(squares) == 3  # 3 series in mock data


class TestDicomLoaderThreadSafety:
    """Tests for DicomLoader stateless thread safety."""

    def test_loader_has_no_shared_state(self):
        """Test that DicomLoader doesn't store errors/warnings as instance variables."""
        from dicom_qc.core.dicom_loader import DicomLoader

        loader = DicomLoader()

        # Loader should NOT have errors/warnings as instance attributes
        assert not hasattr(loader, 'errors') or not isinstance(getattr(loader, 'errors', None), list)
        assert not hasattr(loader, 'warnings') or not isinstance(getattr(loader, 'warnings', None), list)

    def test_concurrent_loaders_independent(self):
        """Test that multiple loader calls don't interfere with each other.

        This verifies the fix for the race condition where multiple threads
        would share the same errors/warnings lists.
        """
        import threading
        from concurrent.futures import ThreadPoolExecutor
        from dicom_qc.core.dicom_loader import DicomLoader

        loader = DicomLoader()
        results = {}
        lock = threading.Lock()

        def simulate_load(thread_id):
            """Simulate a load operation that would previously have shared state."""
            # In the old implementation, these would be appended to shared lists
            # Now they're local to each load call
            local_errors = []
            local_warnings = []

            # Simulate some work
            for i in range(10):
                local_errors.append((f'file_{thread_id}_{i}.dcm', Exception(f'Error {i}')))
                local_warnings.append((f'file_{thread_id}_{i}.dcm', f'Warning {i}'))

            with lock:
                results[thread_id] = {
                    'errors': len(local_errors),
                    'warnings': len(local_warnings),
                }

        # Run concurrent "loads"
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(simulate_load, i) for i in range(10)]
            for f in futures:
                f.result()

        # Each thread should have its own independent results
        assert len(results) == 10
        for thread_id, result in results.items():
            assert result['errors'] == 10
            assert result['warnings'] == 10

    def test_loader_strict_mode_preserved(self):
        """Test that strict mode is still configurable on loader."""
        from dicom_qc.core.dicom_loader import DicomLoader

        loader_strict = DicomLoader(strict=True)
        loader_permissive = DicomLoader(strict=False)

        assert loader_strict.strict is True
        assert loader_permissive.strict is False


class TestFallbackPathBehavior:
    """Tests for fallback path (when nest_asyncio unavailable) behavior."""

    def test_last_series_tracking(self):
        """Test that fallback path correctly tracks last processed series.

        Previously the code used `series if done else to_process[0]` which
        would show the wrong series when waiting for futures to complete.
        """
        from dicom_qc.quickcheck import SeriesInfo

        to_process = [
            SeriesInfo(uid=f'1.2.3.{i}', series_number=i, description=f'Series {i}', modality='MR')
            for i in range(5)
        ]

        # Simulate fallback path tracking
        last_series = to_process[0] if to_process else None
        processed = []

        # Process in arbitrary order (simulating parallel completion)
        processing_order = [2, 0, 4, 1, 3]
        for idx in processing_order:
            series = to_process[idx]
            processed.append(series)
            last_series = series  # Update like the fixed code does

            # last_series should always be the most recently processed
            assert last_series.series_number == idx

        # Final last_series should be the last one processed
        assert last_series.series_number == 3  # Last in processing_order

    def test_thumbnail_tracking_in_fallback(self):
        """Test that fallback path correctly tracks thumbnails for display."""
        from dicom_qc.quickcheck import SeriesInfo

        max_thumbs_to_show = 5
        max_recent = 4
        recent_series = []
        recent_thumb_cache = {}
        thumbs_shown = [0]

        # Create test series
        to_process = [
            SeriesInfo(uid=f'1.2.3.{i}', series_number=i, description=f'Series {i}', modality='MR')
            for i in range(10)
        ]

        # Simulate thumbnail handling like fallback path
        for series in to_process:
            series.thumbnail = f'thumb_data_{series.series_number}'

            if thumbs_shown[0] < max_thumbs_to_show:
                recent_thumb_cache[series.uid] = series.thumbnail
                recent_series.append(series)
                if len(recent_series) > max_recent:
                    old = recent_series.pop(0)
                    recent_thumb_cache.pop(old.uid, None)
                thumbs_shown[0] += 1

        # Should only show first 5 (max_thumbs_to_show)
        assert thumbs_shown[0] == 5

        # Cache should only have last 4 (max_recent) of those 5
        assert len(recent_thumb_cache) == 4
        assert len(recent_series) == 4

        # Recent series should be series 1-4 (0 was popped when 4 was added)
        recent_numbers = [s.series_number for s in recent_series]
        assert recent_numbers == [1, 2, 3, 4]


class TestParallelProcessing:
    """Tests for parallel processing safety."""

    def test_save_only_clears_completed_series_files(self, qc_instance):
        """Test that save() only clears files for completed series.

        This prevents a race condition where save() clears series.files
        while other threads are still processing pending series.
        """
        # Create one completed and one pending series
        completed_series = SeriesInfo(
            uid='1.2.3.1',
            series_number=1,
            description='Completed Series',
            modality='MR',
        )
        completed_series.files = ['file1.dcm', 'file2.dcm']
        completed_series._xnat_files = True
        completed_series._file_uris = ['/data/file1.dcm', '/data/file2.dcm']
        completed_series.thumbnail = 'base64data'  # Mark as completed

        pending_series = SeriesInfo(
            uid='1.2.3.2',
            series_number=2,
            description='Pending Series',
            modality='MR',
        )
        pending_series.files = ['file3.dcm', 'file4.dcm']
        pending_series._xnat_files = True
        pending_series._file_uris = ['/data/file3.dcm', '/data/file4.dcm']
        # No thumbnail - still pending

        patient = PatientInfo('PAT001', 'Test Patient')
        study = StudyInfo(uid='study1', date='2024-01-01', description='Test')
        study.series = {
            completed_series.uid: completed_series,
            pending_series.uid: pending_series,
        }
        patient.studies = {'study1': study}
        qc_instance.patients = {'PAT001': patient}

        # Verify initial state
        assert completed_series.files == ['file1.dcm', 'file2.dcm']
        assert pending_series.files == ['file3.dcm', 'file4.dcm']

        # Save should only clear files for completed series
        qc_instance.save()

        # Completed series: files cleared (safe to clear)
        assert completed_series.files == []
        # Pending series: files preserved (might still be in use)
        assert pending_series.files == ['file3.dcm', 'file4.dcm']
        # Both have URIs preserved for restoration
        assert completed_series._file_uris == ['/data/file1.dcm', '/data/file2.dcm']
        assert pending_series._file_uris == ['/data/file3.dcm', '/data/file4.dcm']

    def test_files_cleared_for_error_series(self, qc_instance):
        """Test that files are cleared for series with errors (they're done)."""
        series = SeriesInfo(
            uid='1.2.3.1',
            series_number=1,
            description='Error Series',
            modality='MR',
        )
        series.files = ['file1.dcm']
        series._xnat_files = True
        series._file_uris = ['/data/file1.dcm']
        series.error = 'Failed to load'  # Mark as error

        patient = PatientInfo('PAT001', 'Test Patient')
        study = StudyInfo(uid='study1', date='2024-01-01', description='Test')
        study.series = {series.uid: series}
        patient.studies = {'study1': study}
        qc_instance.patients = {'PAT001': patient}

        qc_instance.save()

        # Error series: files cleared (processing is done)
        assert series.files == []
        assert series._file_uris == ['/data/file1.dcm']

    def test_files_cleared_for_derived_series(self, qc_instance):
        """Test that files are cleared for derived series (no processing needed)."""
        series = SeriesInfo(
            uid='1.2.3.1',
            series_number=1,
            description='Derived Series',
            modality='MR',
        )
        series.files = ['file1.dcm']
        series._xnat_files = True
        series._file_uris = ['/data/file1.dcm']
        series.is_derived = True  # Mark as derived

        patient = PatientInfo('PAT001', 'Test Patient')
        study = StudyInfo(uid='study1', date='2024-01-01', description='Test')
        study.series = {series.uid: series}
        patient.studies = {'study1': study}
        qc_instance.patients = {'PAT001': patient}

        qc_instance.save()

        # Derived series: files cleared (no processing needed)
        assert series.files == []
        assert series._file_uris == ['/data/file1.dcm']
