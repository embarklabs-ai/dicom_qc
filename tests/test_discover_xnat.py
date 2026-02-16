"""Tests for discover_xnat method - covers both parallel and sequential paths."""

import pytest
from dataclasses import dataclass, field
from typing import Dict, Any

from dicom_qc.quickcheck import QuickCheck


# ============================================================================
# Mock XNAT objects that mimic xnatpy structure
# ============================================================================

@dataclass
class MockXnatFile:
    """Mock XNAT file object."""
    uri: str
    data_path: str = None


@dataclass
class MockXnatResource:
    """Mock XNAT resource object."""
    label: str
    _files: Dict[str, MockXnatFile] = field(default_factory=dict)

    @property
    def files(self):
        return self._files


@dataclass
class MockXnatScan:
    """Mock XNAT scan object."""
    id: str
    series_description: str = ''
    modality: str = 'MR'
    _resources: Dict[str, MockXnatResource] = field(default_factory=dict)

    @property
    def resources(self):
        return self._resources


@dataclass
class MockXnatExperiment:
    """Mock XNAT experiment/session object."""
    label: str
    id: str = None
    _scans: Dict[str, MockXnatScan] = field(default_factory=dict)

    @property
    def scans(self):
        return self._scans


@dataclass
class MockXnatSubject:
    """Mock XNAT subject object."""
    label: str
    id: str = None
    _experiments: Dict[str, MockXnatExperiment] = field(default_factory=dict)

    @property
    def experiments(self):
        return self._experiments


@dataclass
class MockXnatProject:
    """Mock XNAT project object."""
    id: str = 'TEST_PROJECT'
    _subjects: Dict[str, MockXnatSubject] = field(default_factory=dict)
    _session: Any = None

    @property
    def subjects(self):
        return self._subjects


def create_mock_project(
    num_subjects: int = 2,
    sessions_per_subject: int = 1,
    scans_per_session: int = 3,
    files_per_scan: int = 10,
) -> MockXnatProject:
    """Create a mock XNAT project with configurable structure.

    Args:
        num_subjects: Number of subjects to create
        sessions_per_subject: Number of sessions per subject
        scans_per_session: Number of scans per session
        files_per_scan: Number of files per scan

    Returns:
        MockXnatProject with populated hierarchy
    """
    project = MockXnatProject()

    for subj_idx in range(num_subjects):
        subj_label = f'SUBJ{subj_idx:03d}'
        subj_id = f'XNAT_S{subj_idx:05d}'
        subject = MockXnatSubject(label=subj_label, id=subj_id)

        for sess_idx in range(sessions_per_subject):
            sess_label = f'{subj_label}_MR{sess_idx + 1}'
            sess_id = f'XNAT_E{subj_idx:03d}{sess_idx:02d}'
            experiment = MockXnatExperiment(label=sess_label, id=sess_id)

            for scan_idx in range(scans_per_session):
                scan_id = str(scan_idx + 1)
                scan = MockXnatScan(
                    id=scan_id,
                    series_description=f'Series {scan_idx + 1}',
                    modality='MR',
                )

                # Add DICOM resource with files
                dicom_resource = MockXnatResource(label='DICOM')
                for file_idx in range(files_per_scan):
                    file_uri = f'/data/projects/{project.id}/subjects/{subj_label}/experiments/{sess_label}/scans/{scan_id}/resources/DICOM/files/img_{file_idx:04d}.dcm'
                    dicom_resource._files[f'img_{file_idx:04d}.dcm'] = MockXnatFile(
                        uri=file_uri,
                        data_path=f'/mnt/archive{file_uri}',
                    )
                scan._resources['DICOM'] = dicom_resource

                experiment._scans[scan_id] = scan

            subject._experiments[sess_label] = experiment

        project._subjects[subj_label] = subject

    return project


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


@pytest.fixture
def mock_project_small():
    """Create a small mock project (2 subjects, 1 session each, 3 scans)."""
    return create_mock_project(
        num_subjects=2,
        sessions_per_subject=1,
        scans_per_session=3,
        files_per_scan=5,
    )


@pytest.fixture
def mock_project_medium():
    """Create a medium mock project (5 subjects, 2 sessions each, 4 scans)."""
    return create_mock_project(
        num_subjects=5,
        sessions_per_subject=2,
        scans_per_session=4,
        files_per_scan=10,
    )


# ============================================================================
# Test Classes
# ============================================================================

class TestDiscoverXnatBasic:
    """Tests for basic discover_xnat functionality."""

    def test_discover_creates_patient_hierarchy(self, qc_instance, mock_project_small):
        """Test that discover_xnat creates correct patient hierarchy."""
        qc_instance.discover_xnat(mock_project_small, interactive=False, parallel=False)

        # Should have 2 patients
        assert len(qc_instance.patients) == 2
        assert 'SUBJ000' in qc_instance.patients
        assert 'SUBJ001' in qc_instance.patients

    def test_discover_creates_study_hierarchy(self, qc_instance, mock_project_small):
        """Test that discover_xnat creates correct study hierarchy."""
        qc_instance.discover_xnat(mock_project_small, interactive=False, parallel=False)

        # Each patient should have 1 study
        for patient in qc_instance.patients.values():
            assert len(patient.studies) == 1

    def test_discover_creates_series_hierarchy(self, qc_instance, mock_project_small):
        """Test that discover_xnat creates correct series hierarchy."""
        qc_instance.discover_xnat(mock_project_small, interactive=False, parallel=False)

        # Each study should have 3 series
        for patient in qc_instance.patients.values():
            for study in patient.studies.values():
                assert len(study.series) == 3

    def test_discover_stores_xnat_metadata(self, qc_instance, mock_project_small):
        """Test that discover_xnat stores XNAT-specific metadata."""
        qc_instance.discover_xnat(mock_project_small, interactive=False, parallel=False)

        patient = qc_instance.patients['SUBJ000']
        assert patient.xnat_subject_id == 'XNAT_S00000'

        study = list(patient.studies.values())[0]
        assert study.xnat_session_label == 'SUBJ000_MR1'
        assert study.xnat_experiment_id == 'XNAT_E00000'

        series = list(study.series.values())[0]
        assert series.xnat_scan_id is not None
        assert series._xnat_files is True

    def test_discover_stores_file_uris(self, qc_instance, mock_project_small):
        """Test that discover_xnat stores file URIs for later restoration."""
        qc_instance.discover_xnat(mock_project_small, interactive=False, parallel=False)

        # Get first series
        series = qc_instance.get_all_series()[0]

        # Should have file URIs stored
        assert len(series._file_uris) == 5  # 5 files per scan
        assert all('/data/projects/' in uri for uri in series._file_uris)

        # Should have file paths stored
        assert len(series._file_paths) == 5
        assert all('/mnt/archive' in path for path in series._file_paths)

    def test_discover_sets_xnat_mode(self, qc_instance, mock_project_small):
        """Test that discover_xnat sets XNAT mode flag."""
        assert qc_instance._xnat_mode is False

        qc_instance.discover_xnat(mock_project_small, interactive=False, parallel=False)

        assert qc_instance._xnat_mode is True

    def test_discover_stores_project_info(self, qc_instance, mock_project_small):
        """Test that discover_xnat stores project info for OHIF links."""
        qc_instance.discover_xnat(mock_project_small, interactive=False, parallel=False)

        assert qc_instance._xnat_project_id == 'TEST_PROJECT'


class TestDiscoverXnatParallelVsSequential:
    """Tests ensuring parallel and sequential discovery produce identical results."""

    def test_parallel_and_sequential_same_patient_count(self, temp_data_dir, mock_project_medium):
        """Test that parallel and sequential produce same patient count."""
        # Sequential
        qc_seq = QuickCheck(data_dir=temp_data_dir / 'seq')
        qc_seq.discover_xnat(mock_project_medium, interactive=False, parallel=False)

        # Parallel
        qc_par = QuickCheck(data_dir=temp_data_dir / 'par')
        qc_par.discover_xnat(mock_project_medium, interactive=False, parallel=True, max_workers=2)

        assert len(qc_seq.patients) == len(qc_par.patients)
        assert set(qc_seq.patients.keys()) == set(qc_par.patients.keys())

    def test_parallel_and_sequential_same_series_count(self, temp_data_dir, mock_project_medium):
        """Test that parallel and sequential produce same total series count."""
        # Sequential
        qc_seq = QuickCheck(data_dir=temp_data_dir / 'seq')
        qc_seq.discover_xnat(mock_project_medium, interactive=False, parallel=False)

        # Parallel
        qc_par = QuickCheck(data_dir=temp_data_dir / 'par')
        qc_par.discover_xnat(mock_project_medium, interactive=False, parallel=True, max_workers=2)

        seq_series = qc_seq.get_all_series()
        par_series = qc_par.get_all_series()

        assert len(seq_series) == len(par_series)

    def test_parallel_and_sequential_same_file_uris(self, temp_data_dir, mock_project_small):
        """Test that parallel and sequential store same file URIs."""
        # Sequential
        qc_seq = QuickCheck(data_dir=temp_data_dir / 'seq')
        qc_seq.discover_xnat(mock_project_small, interactive=False, parallel=False)

        # Parallel
        qc_par = QuickCheck(data_dir=temp_data_dir / 'par')
        qc_par.discover_xnat(mock_project_small, interactive=False, parallel=True, max_workers=2)

        # Compare file URIs for each series
        seq_series = sorted(qc_seq.get_all_series(), key=lambda s: (s._xnat_subject_label or '', s._xnat_session_label or '', s.xnat_scan_id or ''))
        par_series = sorted(qc_par.get_all_series(), key=lambda s: (s._xnat_subject_label or '', s._xnat_session_label or '', s.xnat_scan_id or ''))

        for seq_s, par_s in zip(seq_series, par_series):
            assert sorted(seq_s._file_uris) == sorted(par_s._file_uris)

    def test_parallel_and_sequential_same_metadata(self, temp_data_dir, mock_project_small):
        """Test that parallel and sequential store same series metadata."""
        # Sequential
        qc_seq = QuickCheck(data_dir=temp_data_dir / 'seq')
        qc_seq.discover_xnat(mock_project_small, interactive=False, parallel=False)

        # Parallel
        qc_par = QuickCheck(data_dir=temp_data_dir / 'par')
        qc_par.discover_xnat(mock_project_small, interactive=False, parallel=True, max_workers=2)

        # Compare metadata
        seq_series = sorted(qc_seq.get_all_series(), key=lambda s: (s._xnat_subject_label or '', s.xnat_scan_id or ''))
        par_series = sorted(qc_par.get_all_series(), key=lambda s: (s._xnat_subject_label or '', s.xnat_scan_id or ''))

        for seq_s, par_s in zip(seq_series, par_series):
            assert seq_s.description == par_s.description
            assert seq_s.modality == par_s.modality
            assert seq_s.xnat_scan_id == par_s.xnat_scan_id


class TestDiscoverXnatParallelErrorHandling:
    """Tests for error handling in parallel discovery."""

    def test_parallel_discovery_keeps_failed_scan_with_error(self, qc_instance, mock_project_small):
        """Test that failed scan tasks are retained as error series (not silently dropped)."""
        original_create = qc_instance._create_series_from_xnat_scan

        def flaky_create(scan, subject_label, session_label):
            if subject_label == 'SUBJ000' and scan.id == '2':
                raise RuntimeError("Injected parallel failure")
            return original_create(scan, subject_label, session_label)

        qc_instance._create_series_from_xnat_scan = flaky_create
        qc_instance.discover_xnat(mock_project_small, interactive=False, parallel=True, max_workers=2)

        # Total count should still match expected scans (2 subjects x 3 scans)
        assert len(qc_instance.get_all_series()) == 6

        failed_series = qc_instance.patients['SUBJ000'].studies['SUBJ000_MR1'].series['2']
        assert failed_series.error is not None
        assert 'Failed to process scan' in failed_series.error
        assert failed_series.xnat_scan_id == '2'
        assert failed_series._xnat_files is True


class TestDiscoverXnatIncremental:
    """Tests for incremental discovery (refresh=False)."""

    def test_incremental_preserves_existing_patients(self, qc_instance, mock_project_small):
        """Test that incremental discovery preserves existing patients."""
        # First discovery
        qc_instance.discover_xnat(mock_project_small, interactive=False, parallel=False, refresh=True)

        # Mark a series as processed
        series = qc_instance.get_all_series()[0]
        series.thumbnail = 'processed'
        # Incremental discovery
        qc_instance.discover_xnat(mock_project_small, interactive=False, parallel=False, refresh=False)

        # Note: In incremental mode, existing series with file_uris are skipped
        assert len(qc_instance.patients) == 2

    def test_incremental_skips_series_with_uris(self, qc_instance, mock_project_small):
        """Test that incremental discovery skips series that already have file URIs."""
        # First discovery
        qc_instance.discover_xnat(mock_project_small, interactive=False, parallel=False, refresh=True)

        # Get initial file URIs
        series = qc_instance.get_all_series()[0]
        original_uris = series._file_uris.copy()
        assert len(original_uris) > 0

        # Incremental discovery
        qc_instance.discover_xnat(mock_project_small, interactive=False, parallel=False, refresh=False)

        # File URIs should be unchanged
        updated_series = qc_instance.get_all_series()[0]
        assert updated_series._file_uris == original_uris

    def test_refresh_true_clears_existing(self, qc_instance, mock_project_small):
        """Test that refresh=True clears existing data."""
        # First discovery
        qc_instance.discover_xnat(mock_project_small, interactive=False, parallel=False, refresh=True)

        # Modify a patient name to detect if it gets cleared
        qc_instance.patients['SUBJ000'].patient_name = 'MODIFIED'

        # Refresh discovery
        qc_instance.discover_xnat(mock_project_small, interactive=False, parallel=False, refresh=True)

        # Patient name should be reset (empty string from XNAT mock)
        assert qc_instance.patients['SUBJ000'].patient_name == ''


class TestDiscoverXnatErrorHandling:
    """Tests for error handling in discover_xnat."""

    def test_handles_missing_series_description(self, qc_instance):
        """Test handling of scans with missing series_description."""
        project = create_mock_project(num_subjects=1, sessions_per_subject=1, scans_per_session=1)
        # Remove series_description
        scan = list(list(project.subjects.values())[0].experiments.values())[0].scans['1']
        scan.series_description = None

        qc_instance.discover_xnat(project, interactive=False, parallel=False)

        series = qc_instance.get_all_series()[0]
        assert series.description == ''  # Should default to empty string

    def test_handles_missing_modality(self, qc_instance):
        """Test handling of scans with missing modality."""
        project = create_mock_project(num_subjects=1, sessions_per_subject=1, scans_per_session=1)
        # Remove modality
        scan = list(list(project.subjects.values())[0].experiments.values())[0].scans['1']
        scan.modality = None

        qc_instance.discover_xnat(project, interactive=False, parallel=False)

        series = qc_instance.get_all_series()[0]
        assert series.modality == '??'  # Should default to ??

class TestDiscoverXnatRetry:
    """Tests for XNAT retry logic."""

    def test_xnat_retry_succeeds_after_transient_error(self):
        """Test that _xnat_retry succeeds after transient errors."""
        from dicom_qc.quickcheck import _xnat_retry

        call_count = [0]

        def flaky_func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("503 Service Temporarily Unavailable")
            return "success"

        result = _xnat_retry(flaky_func, max_retries=3, base_delay=0.01)

        assert result == "success"
        assert call_count[0] == 3

    def test_xnat_retry_raises_on_non_retryable_error(self):
        """Test that _xnat_retry raises immediately for non-retryable errors."""
        from dicom_qc.quickcheck import _xnat_retry

        call_count = [0]

        def always_fail():
            call_count[0] += 1
            raise ValueError("Invalid argument")

        with pytest.raises(ValueError):
            _xnat_retry(always_fail, max_retries=3, base_delay=0.01)

        # Should only be called once (no retry for non-retryable errors)
        assert call_count[0] == 1

    def test_xnat_retry_raises_after_max_retries(self):
        """Test that _xnat_retry raises after exhausting retries."""
        from dicom_qc.quickcheck import _xnat_retry

        call_count = [0]

        def always_503():
            call_count[0] += 1
            raise Exception("503 Service Unavailable")

        with pytest.raises(Exception, match="503"):
            _xnat_retry(always_503, max_retries=2, base_delay=0.01)

        # Should be called max_retries + 1 times
        assert call_count[0] == 3


class TestDiscoverXnatProgressTracking:
    """Tests for progress tracking functionality."""

    def test_get_all_series_count_accurate(self, qc_instance, mock_project_medium):
        """Test that get_all_series returns accurate count."""
        qc_instance.discover_xnat(mock_project_medium, interactive=False, parallel=False)

        # 5 subjects * 2 sessions * 4 scans = 40 series
        assert len(qc_instance.get_all_series()) == 40

    def test_discovery_saves_state(self, qc_instance, mock_project_small, temp_data_dir):
        """Test that discovery saves state to disk."""
        qc_instance.discover_xnat(mock_project_small, interactive=False, parallel=False)

        # Should have saved
        assert qc_instance.has_save()

        # Load into new instance
        qc2 = QuickCheck(data_dir=temp_data_dir)
        qc2.load()

        assert len(qc2.patients) == 2
        assert len(qc2.get_all_series()) == 6


class TestFetchScanFiles:
    """Tests for the _fetch_scan_files helper function."""

    def test_fetch_scan_files_returns_all_dicom_files(self):
        """Test that _fetch_scan_files returns files from all non-SNAPSHOTS resources."""
        from dicom_qc.quickcheck import _fetch_scan_files

        scan = MockXnatScan(id='1', series_description='Test')

        # Add DICOM resource
        dicom = MockXnatResource(label='DICOM')
        dicom._files['file1.dcm'] = MockXnatFile(uri='/file1.dcm')
        dicom._files['file2.dcm'] = MockXnatFile(uri='/file2.dcm')
        scan._resources['DICOM'] = dicom

        # Add secondary resource
        secondary = MockXnatResource(label='secondary')
        secondary._files['file3.dcm'] = MockXnatFile(uri='/file3.dcm')
        scan._resources['secondary'] = secondary

        files = _fetch_scan_files(scan)

        # Should get files from both resources
        assert len(files) == 3

    def test_fetch_scan_files_excludes_snapshots(self):
        """Test that _fetch_scan_files excludes SNAPSHOTS resource."""
        from dicom_qc.quickcheck import _fetch_scan_files

        scan = MockXnatScan(id='1', series_description='Test')

        dicom = MockXnatResource(label='DICOM')
        dicom._files['file1.dcm'] = MockXnatFile(uri='/file1.dcm')
        scan._resources['DICOM'] = dicom

        snapshots = MockXnatResource(label='SNAPSHOTS')
        snapshots._files['snap.jpg'] = MockXnatFile(uri='/snap.jpg')
        scan._resources['SNAPSHOTS'] = snapshots

        files = _fetch_scan_files(scan)

        assert len(files) == 1
        assert files[0].uri == '/file1.dcm'

    def test_fetch_scan_files_case_insensitive_snapshots(self):
        """Test that SNAPSHOTS exclusion is case-insensitive."""
        from dicom_qc.quickcheck import _fetch_scan_files

        scan = MockXnatScan(id='1', series_description='Test')

        dicom = MockXnatResource(label='DICOM')
        dicom._files['file1.dcm'] = MockXnatFile(uri='/file1.dcm')
        scan._resources['DICOM'] = dicom

        # Lowercase snapshots
        snapshots = MockXnatResource(label='snapshots')
        snapshots._files['snap.jpg'] = MockXnatFile(uri='/snap.jpg')
        scan._resources['snapshots'] = snapshots

        files = _fetch_scan_files(scan)

        assert len(files) == 1


class TestDiscoverXnatEdgeCases:
    """Tests for edge cases in discover_xnat."""

    def test_empty_project(self, qc_instance):
        """Test handling of project with no subjects."""
        project = MockXnatProject()

        qc_instance.discover_xnat(project, interactive=False, parallel=False)

        assert len(qc_instance.patients) == 0

    def test_subject_with_no_experiments(self, qc_instance):
        """Test handling of subject with no experiments."""
        project = MockXnatProject()
        project._subjects['SUBJ001'] = MockXnatSubject(label='SUBJ001')

        qc_instance.discover_xnat(project, interactive=False, parallel=False)

        assert 'SUBJ001' in qc_instance.patients
        assert len(qc_instance.patients['SUBJ001'].studies) == 0

    def test_experiment_with_no_scans(self, qc_instance):
        """Test handling of experiment with no scans."""
        project = MockXnatProject()
        subject = MockXnatSubject(label='SUBJ001')
        subject._experiments['SESS001'] = MockXnatExperiment(label='SESS001')
        project._subjects['SUBJ001'] = subject

        qc_instance.discover_xnat(project, interactive=False, parallel=False)

        assert 'SUBJ001' in qc_instance.patients
        assert 'SESS001' in qc_instance.patients['SUBJ001'].studies
        assert len(qc_instance.patients['SUBJ001'].studies['SESS001'].series) == 0

    def test_scan_with_no_files(self, qc_instance):
        """Test handling of scan with no DICOM files."""
        project = create_mock_project(num_subjects=1, sessions_per_subject=1, scans_per_session=1, files_per_scan=0)

        qc_instance.discover_xnat(project, interactive=False, parallel=False)

        series = qc_instance.get_all_series()[0]
        assert len(series._file_uris) == 0
