"""Tests for QuickCheck and HTML report generation."""

import zipfile

import pytest

from pathlib import Path

from dicom_qc.quickcheck import (
    QuickCheck,
    SeriesInfo,
    StudyInfo,
    PatientInfo,
    _is_in_defaced_dir,
)
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
        scan_id="series_001",
        results=[
            QCResult(
                status="PASS",
                check_name="Geometry Metadata",
                message="All tags present",
            ),
            QCResult(status="PASS", check_name="Slice Ordering", message="Monotonic"),
        ],
        overall_status="PASS",
        orientation_labels={"row_positive": "L", "col_positive": "P"},
        primary_plane="AXIAL",
    )

    # Create mock series
    series1 = SeriesInfo(
        uid="1.2.3.4.5.001",
        series_number=1,
        description="T1 MPRAGE",
        modality="MR",
        qc_report=qc_report,
    )
    series2 = SeriesInfo(
        uid="1.2.3.4.5.002",
        series_number=2,
        description="T2 FLAIR",
        modality="MR",
        qc_report=qc_report,
    )
    series3 = SeriesInfo(
        uid="1.2.3.4.5.003",
        series_number=3,
        description="DWI",
        modality="MR",
        error="Failed to load",
    )

    # Create mock study
    study = StudyInfo(
        uid="1.2.3.4.100",
        date="2024-01-15",
        description="Brain MRI",
        series={
            series1.uid: series1,
            series2.uid: series2,
            series3.uid: series3,
        },
    )

    # Create mock patient
    patient = PatientInfo(
        patient_id="PAT001",
        patient_name="Test Patient",
        studies={"1.2.3.4.100": study},
    )

    qc.patients = {"PAT001": patient}
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
        QuickCheck(data_dir=temp_data_dir)
        storage_dir = temp_data_dir / "_dicom_qc"
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
        assert summary["PASS"] == 2
        assert summary["ERROR"] == 1

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

        storage_dir = temp_data_dir / "_dicom_qc"
        assert storage_dir.exists()

        qc.reset(delete_storage=True)
        # Storage should be recreated empty
        assert qc.has_save() is False


class TestHTMLReportGeneration:
    """Tests for HTML report generation."""

    def test_generate_embedded_report(self, qc_with_mock_data, temp_data_dir):
        """Test generating embedded (single-file) HTML report."""
        output_path = temp_data_dir / "report.html"

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
        assert "DICOM" in content
        assert "Test Patient" in content or "PAT001" in content
        assert "T1 MPRAGE" in content

    def test_generate_external_thumbnail_report(self, qc_with_mock_data, temp_data_dir):
        """Test generating HTML report with external thumbnails + zip."""
        output_path = temp_data_dir / "report.html"

        html_path, zip_path = qc_with_mock_data.generate_html_report(
            output_path,
            embed_thumbnails=False,
        )

        # Should return index.html and zip path
        assert html_path.name == "index.html"
        assert zip_path is not None
        assert zip_path.suffix == ".zip"
        assert html_path.exists()
        assert zip_path.exists()

        # Check report directory structure
        report_dir = html_path.parent
        assert report_dir.name == "report"
        assert (report_dir / "thumbnails").exists()

    def test_external_zip_contents(self, qc_with_mock_data, temp_data_dir):
        """Test that external thumbnail zip contains all required files."""
        output_path = temp_data_dir / "report.html"

        html_path, zip_path = qc_with_mock_data.generate_html_report(
            output_path,
            embed_thumbnails=False,
        )

        # Extract and verify zip contents
        extract_dir = temp_data_dir / "extracted"
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        # Should contain single index.html and thumbnails folder
        report_in_zip = extract_dir / "report"
        assert (report_in_zip / "index.html").exists()
        assert (report_in_zip / "thumbnails").exists()

    def test_auto_embed_small_dataset(self, qc_with_mock_data, temp_data_dir):
        """Test that small datasets auto-select embedded mode."""
        output_path = temp_data_dir / "report.html"

        # With only 3 series, should auto-select embedded
        html_path, zip_path = qc_with_mock_data.generate_html_report(output_path)

        assert zip_path is None  # No zip for small datasets
        assert html_path.exists()
        assert html_path == output_path

    def test_report_contains_status_summary(self, qc_with_mock_data, temp_data_dir):
        """Test that report contains status summary."""
        output_path = temp_data_dir / "report.html"

        html_path, _ = qc_with_mock_data.generate_html_report(
            output_path,
            embed_thumbnails=True,
        )

        content = html_path.read_text()
        # Should have summary cards
        assert "Pass" in content
        assert "Error" in content

    def test_report_contains_series_info(self, qc_with_mock_data, temp_data_dir):
        """Test that report contains series information."""
        output_path = temp_data_dir / "report.html"

        html_path, _ = qc_with_mock_data.generate_html_report(
            output_path,
            embed_thumbnails=True,
        )

        content = html_path.read_text()
        assert "T1 MPRAGE" in content
        assert "T2 FLAIR" in content
        assert "DWI" in content

    def test_external_report_contains_all_series(
        self, qc_with_mock_data, temp_data_dir
    ):
        """Test that external thumbnail report contains all series in single file."""
        output_path = temp_data_dir / "report.html"

        html_path, _ = qc_with_mock_data.generate_html_report(
            output_path,
            embed_thumbnails=False,
        )

        # Single HTML file should contain all series
        content = html_path.read_text()
        assert "T1 MPRAGE" in content
        assert "T2 FLAIR" in content
        assert "DWI" in content


class TestSeriesInfo:
    """Tests for SeriesInfo dataclass."""

    def test_qc_status_pending(self):
        """Test qc_status is PENDING when no report."""
        series = SeriesInfo(uid="1", series_number=1, description="Test", modality="MR")
        assert series.qc_status == "PENDING"

    def test_qc_status_error(self):
        """Test qc_status is ERROR when error set."""
        series = SeriesInfo(
            uid="1",
            series_number=1,
            description="Test",
            modality="MR",
            error="Failed",
        )
        assert series.qc_status == "ERROR"

    def test_qc_status_derived(self):
        """Test qc_status is DERIVED when is_derived set."""
        series = SeriesInfo(
            uid="1",
            series_number=1,
            description="Test",
            modality="MR",
            is_derived=True,
        )
        assert series.qc_status == "DERIVED"

    def test_qc_status_from_report(self):
        """Test qc_status comes from report when present."""
        qc_report = QCReport(
            scan_id="test",
            results=[QCResult(status="WARNING", check_name="Test", message="Warn")],
            overall_status="WARNING",
            orientation_labels={},
            primary_plane="AXIAL",
        )
        series = SeriesInfo(
            uid="1",
            series_number=1,
            description="Test",
            modality="MR",
            qc_report=qc_report,
        )
        assert series.qc_status == "WARNING"

    def test_label_format(self):
        """Test series label format."""
        series = SeriesInfo(
            uid="1", series_number=5, description="T1 MPRAGE", modality="MR"
        )
        assert series.label == "#5 MR: T1 MPRAGE"


class TestStudyInfo:
    """Tests for StudyInfo dataclass."""

    def test_label_with_date_and_description(self):
        """Test study label with both date and description."""
        study = StudyInfo(uid="1", date="2024-01-15", description="Brain MRI")
        assert study.label == "2024-01-15: Brain MRI"

    def test_label_with_date_only(self):
        """Test study label with date only."""
        study = StudyInfo(uid="1", date="2024-01-15", description="")
        assert study.label == "2024-01-15"

    def test_label_with_description_only(self):
        """Test study label with description only."""
        study = StudyInfo(uid="1", date="", description="Brain MRI")
        assert study.label == "Brain MRI"


class TestSaveBehavior:
    """Tests for save() clearing non-serializable objects."""

    def test_save_clears_all_xnat_series_files(self, qc_instance):
        """Test that save() clears file objects for all XNAT series.

        XnatFileHandle objects hold xnatpy session references that are
        not serializable. Workers hold their own local references, so
        clearing series.files is safe.
        """
        completed_series = SeriesInfo(
            uid="1.2.3.1",
            series_number=1,
            description="Completed Series",
            modality="MR",
        )
        completed_series.files = ["file1.dcm", "file2.dcm"]
        completed_series._xnat_files = True
        completed_series._file_uris = ["/data/file1.dcm", "/data/file2.dcm"]
        completed_series.thumbnail = "base64data"

        pending_series = SeriesInfo(
            uid="1.2.3.2",
            series_number=2,
            description="Pending Series",
            modality="MR",
        )
        pending_series.files = ["file3.dcm", "file4.dcm"]
        pending_series._xnat_files = True
        pending_series._file_uris = ["/data/file3.dcm", "/data/file4.dcm"]

        patient = PatientInfo("PAT001", "Test Patient")
        study = StudyInfo(uid="study1", date="2024-01-01", description="Test")
        study.series = {
            completed_series.uid: completed_series,
            pending_series.uid: pending_series,
        }
        patient.studies = {"study1": study}
        qc_instance.patients = {"PAT001": patient}

        qc_instance.save()

        # Both series: files cleared
        assert completed_series.files == []
        assert pending_series.files == []
        # URIs preserved for restoration
        assert completed_series._file_uris == ["/data/file1.dcm", "/data/file2.dcm"]
        assert pending_series._file_uris == ["/data/file3.dcm", "/data/file4.dcm"]

    def test_files_cleared_for_error_series(self, qc_instance):
        """Test that files are cleared for series with errors."""
        series = SeriesInfo(
            uid="1.2.3.1",
            series_number=1,
            description="Error Series",
            modality="MR",
        )
        series.files = ["file1.dcm"]
        series._xnat_files = True
        series._file_uris = ["/data/file1.dcm"]
        series.error = "Failed to load"

        patient = PatientInfo("PAT001", "Test Patient")
        study = StudyInfo(uid="study1", date="2024-01-01", description="Test")
        study.series = {series.uid: series}
        patient.studies = {"study1": study}
        qc_instance.patients = {"PAT001": patient}

        qc_instance.save()

        assert series.files == []
        assert series._file_uris == ["/data/file1.dcm"]

    def test_files_cleared_for_derived_series(self, qc_instance):
        """Test that files are cleared for derived series."""
        series = SeriesInfo(
            uid="1.2.3.1",
            series_number=1,
            description="Derived Series",
            modality="MR",
        )
        series.files = ["file1.dcm"]
        series._xnat_files = True
        series._file_uris = ["/data/file1.dcm"]
        series.is_derived = True

        patient = PatientInfo("PAT001", "Test Patient")
        study = StudyInfo(uid="study1", date="2024-01-01", description="Test")
        study.series = {series.uid: series}
        patient.studies = {"study1": study}
        qc_instance.patients = {"PAT001": patient}

        qc_instance.save()

        assert series.files == []
        assert series._file_uris == ["/data/file1.dcm"]


class TestIsInDefacedDir:
    """Tests for _is_in_defaced_dir helper."""

    def test_defaced_dir(self):
        assert _is_in_defaced_dir(Path("/data/scan/DEFACED/file.dcm")) is True

    def test_non_defaced_dir(self):
        assert _is_in_defaced_dir(Path("/data/scan/DICOM/file.dcm")) is False

    def test_case_insensitive(self):
        assert _is_in_defaced_dir(Path("/data/scan/defaced/file.dcm")) is True

    def test_partial_name_not_matched(self):
        assert _is_in_defaced_dir(Path("/data/scan/DICOM/defaced_file.dcm")) is False
