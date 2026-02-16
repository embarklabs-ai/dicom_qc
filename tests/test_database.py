"""Tests for QCDatabase SQLite storage."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List

import pytest

from dicom_qc.storage.database import QCDatabase


# ---------------------------------------------------------------------------
# Mock QC report objects (mimics QCResult / QCReport from core.geometry)
# ---------------------------------------------------------------------------

@dataclass
class MockQCResult:
    check_name: str
    status: str
    message: str
    details: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            'check_name': self.check_name,
            'status': self.status,
            'message': self.message,
            'details': self.details,
        }


@dataclass
class MockQCReport:
    results: List[MockQCResult]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    """Create a QCDatabase backed by a temporary file."""
    database = QCDatabase(tmp_path / "test.db")
    yield database
    database.close()


def _insert_full_hierarchy(db, patient_id="PAT001", study_uid="1.2.3",
                           series_uid="1.2.3.4", **series_kwargs):
    """Helper: insert patient -> study -> series and return all three db IDs."""
    pid = db.insert_patient(patient_id)
    stid = db.insert_study(pid, study_uid)
    defaults = dict(series_description="T1 MPRAGE", modality="MR",
                    qc_status="PASS", file_count=100)
    defaults.update(series_kwargs)
    sid = db.insert_series(stid, series_uid, **defaults)
    db.commit()
    return pid, stid, sid


# ---------------------------------------------------------------------------
# 1. Schema creation and version
# ---------------------------------------------------------------------------

class TestSchemaCreation:

    def test_schema_version_stored(self, db):
        version = db.get_session_value("schema_version")
        assert version == str(QCDatabase.SCHEMA_VERSION)

    def test_database_starts_empty(self, db):
        assert db.is_empty()

    def test_tables_exist(self, db):
        """Core tables are created during init."""
        cursor = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row["name"] for row in cursor.fetchall()}
        expected = {"patients", "studies", "series", "qc_results",
                    "series_files", "qc_session"}
        assert expected.issubset(tables)


# ---------------------------------------------------------------------------
# 2. Insert patient / study / series round-trip
# ---------------------------------------------------------------------------

class TestInsertRoundTrip:

    def test_insert_patient(self, db):
        pid = db.insert_patient("PAT001", "John Doe", "XNAT_S001")
        assert isinstance(pid, int)
        assert not db.is_empty()

    def test_insert_patient_upsert(self, db):
        """Re-inserting same patient_id returns the same db ID and updates fields."""
        pid1 = db.insert_patient("PAT001", "Old Name")
        pid2 = db.insert_patient("PAT001", "New Name")
        assert pid1 == pid2

    def test_insert_study(self, db):
        pid = db.insert_patient("PAT001")
        stid = db.insert_study(pid, "1.2.3", study_date="20240101",
                               study_description="Brain MRI",
                               xnat_session_label="MR001")
        assert isinstance(stid, int)

    def test_insert_series(self, db):
        pid = db.insert_patient("PAT001")
        stid = db.insert_study(pid, "1.2.3")
        sid = db.insert_series(stid, "1.2.3.4", series_number=5,
                               series_description="T1 MPRAGE", modality="MR",
                               qc_status="PASS", file_count=180)
        assert isinstance(sid, int)

    def test_full_hierarchy_round_trip(self, db):
        """Insert full hierarchy and retrieve via get_filtered_series."""
        pid = db.insert_patient("PAT001", "John Doe", "XNAT_S001")
        stid = db.insert_study(pid, "1.2.3", study_date="20240115",
                               study_description="Brain MRI",
                               xnat_session_label="Session1")
        sid = db.insert_series(stid, "1.2.3.4", series_number=3,
                               series_description="T1 MPRAGE", modality="MR",
                               qc_status="PASS", file_count=192,
                               thumbnail_path="ab/abc123.jpg")
        db.commit()

        rows = db.get_filtered_series()
        assert len(rows) == 1

        row = rows[0]
        assert row["patient_id"] == "PAT001"
        assert row["patient_name"] == "John Doe"
        assert row["study_date"] == "20240115"
        assert row["series_description"] == "T1 MPRAGE"
        assert row["modality"] == "MR"
        assert row["qc_status"] == "PASS"
        assert row["file_count"] == 192
        assert row["thumbnail_path"] == "ab/abc123.jpg"
        assert row["series_db_id"] == sid

    def test_insert_series_upsert(self, db):
        """Re-inserting same series updates fields, keeps same ID."""
        pid = db.insert_patient("PAT001")
        stid = db.insert_study(pid, "1.2.3")

        sid1 = db.insert_series(stid, "1.2.3.4", qc_status="PENDING")
        sid2 = db.insert_series(stid, "1.2.3.4", qc_status="PASS")
        assert sid1 == sid2

        db.commit()
        rows = db.get_filtered_series()
        assert rows[0]["qc_status"] == "PASS"


# ---------------------------------------------------------------------------
# 3. QC results from report
# ---------------------------------------------------------------------------

class TestQCResults:

    def test_insert_and_retrieve_qc_results(self, db):
        _, _, sid = _insert_full_hierarchy(db)

        db.insert_qc_result(sid, "Slice Ordering", "PASS", "Monotonic")
        db.insert_qc_result(sid, "Gap Detection", "WARNING", "Irregular spacing",
                            details={"max_gap_mm": 4.5})
        db.commit()

        results = db.get_qc_results(sid)
        assert len(results) == 2
        assert results[0]["check_name"] == "Slice Ordering"
        assert results[0]["status"] == "PASS"
        assert results[1]["details"] == {"max_gap_mm": 4.5}

    def test_insert_qc_results_from_report(self, db):
        _, _, sid = _insert_full_hierarchy(db)

        report = MockQCReport(results=[
            MockQCResult("Geometry Metadata", "PASS", "All tags present"),
            MockQCResult("Voxel Anisotropy", "WARNING", "3x ratio",
                         details={"ratio": 3.0}),
            MockQCResult("Gap Detection", "FAIL", "Missing slices"),
        ])

        db.insert_qc_results_from_report(sid, report)
        db.commit()

        results = db.get_qc_results(sid)
        assert len(results) == 3
        assert results[0]["check_name"] == "Geometry Metadata"
        assert results[1]["status"] == "WARNING"
        assert results[1]["details"] == {"ratio": 3.0}
        assert results[2]["status"] == "FAIL"

    def test_insert_qc_results_from_report_replaces_existing(self, db):
        """Calling insert_qc_results_from_report clears previous results."""
        _, _, sid = _insert_full_hierarchy(db)

        report_v1 = MockQCReport(results=[
            MockQCResult("Check A", "PASS", "OK"),
            MockQCResult("Check B", "PASS", "OK"),
        ])
        db.insert_qc_results_from_report(sid, report_v1)
        db.commit()
        assert len(db.get_qc_results(sid)) == 2

        report_v2 = MockQCReport(results=[
            MockQCResult("Check C", "FAIL", "Bad"),
        ])
        db.insert_qc_results_from_report(sid, report_v2)
        db.commit()

        results = db.get_qc_results(sid)
        assert len(results) == 1
        assert results[0]["check_name"] == "Check C"


# ---------------------------------------------------------------------------
# 4-5. Filtered series queries and order_by validation
# ---------------------------------------------------------------------------

class TestFilteredSeries:

    @pytest.fixture(autouse=True)
    def _populate(self, db):
        """Populate db with a variety of series for filtering tests."""
        # Patient 1, Study A
        p1 = db.insert_patient("PAT001", "Alice")
        s1 = db.insert_study(p1, "1.2.1", study_date="20240101",
                             xnat_session_label="SessionA")
        db.insert_series(s1, "1.2.1.1", series_number=1,
                         series_description="T1 MPRAGE", modality="MR",
                         qc_status="PASS")
        db.insert_series(s1, "1.2.1.2", series_number=2,
                         series_description="T2 FLAIR", modality="MR",
                         qc_status="FAIL")

        # Patient 2, Study B
        p2 = db.insert_patient("PAT002", "Bob")
        s2 = db.insert_study(p2, "1.2.2", study_date="20240201",
                             xnat_session_label="SessionB")
        db.insert_series(s2, "1.2.2.1", series_number=1,
                         series_description="CT Head", modality="CT",
                         qc_status="PASS")
        db.insert_series(s2, "1.2.2.2", series_number=3,
                         series_description="DWI b1000", modality="MR",
                         qc_status="WARNING")

        db.commit()

    def test_no_filter_returns_all(self, db):
        rows = db.get_filtered_series(limit=100)
        assert len(rows) == 4

    def test_filter_by_status(self, db):
        rows = db.get_filtered_series({"status": "FAIL"})
        assert len(rows) == 1
        assert rows[0]["series_description"] == "T2 FLAIR"

    def test_filter_by_patient_id(self, db):
        rows = db.get_filtered_series({"patient_id": "PAT002"})
        assert len(rows) == 2
        assert all(r["patient_id"] == "PAT002" for r in rows)

    def test_filter_by_study_key_uid(self, db):
        rows = db.get_filtered_series({"study_key": "1.2.1"})
        assert len(rows) == 2
        assert all(r["study_uid"] == "1.2.1" for r in rows)

    def test_filter_by_study_key_xnat_label(self, db):
        rows = db.get_filtered_series({"study_key": "SessionB"})
        assert len(rows) == 2
        assert all(r["xnat_session_label"] == "SessionB" for r in rows)

    def test_filter_by_description_like(self, db):
        rows = db.get_filtered_series({"description_like": "%DWI%"})
        assert len(rows) == 1
        assert rows[0]["series_description"] == "DWI b1000"

    def test_filter_combined(self, db):
        rows = db.get_filtered_series({"patient_id": "PAT001", "status": "PASS"})
        assert len(rows) == 1
        assert rows[0]["series_description"] == "T1 MPRAGE"

    def test_count_filtered_series(self, db):
        assert db.count_filtered_series() == 4
        assert db.count_filtered_series({"status": "PASS"}) == 2
        assert db.count_filtered_series({"patient_id": "PAT002"}) == 2

    def test_get_summary_counts(self, db):
        counts = db.get_summary_counts()
        assert counts["PASS"] == 2
        assert counts["FAIL"] == 1
        assert counts["WARNING"] == 1

    def test_order_by_valid_column(self, db):
        rows = db.get_filtered_series(order_by="series_description")
        descriptions = [r["series_description"] for r in rows]
        assert descriptions == sorted(descriptions)

    def test_order_by_with_direction(self, db):
        rows = db.get_filtered_series(order_by="study_date DESC")
        dates = [r["study_date"] for r in rows]
        assert dates == sorted(dates, reverse=True)

    def test_order_by_multiple_columns(self, db):
        rows = db.get_filtered_series(order_by="patient_id, series_number")
        assert len(rows) == 4

    def test_order_by_invalid_column_raises(self, db):
        with pytest.raises(ValueError, match="Invalid order_by column"):
            db.get_filtered_series(order_by="nonexistent_column")

    def test_order_by_sql_injection_rejected(self, db):
        with pytest.raises(ValueError, match="Invalid order_by column"):
            db.get_filtered_series(order_by="patient_id; DROP TABLE series")


# ---------------------------------------------------------------------------
# 6-7. Delete orphan series
# ---------------------------------------------------------------------------

class TestDeleteOrphanSeries:

    def test_delete_orphans_removes_unlisted(self, db):
        _, _, sid1 = _insert_full_hierarchy(db, "PAT001", "1.2.1", "1.2.1.1")
        _, _, sid2 = _insert_full_hierarchy(db, "PAT002", "1.2.2", "1.2.2.1")

        num_deleted, _ = db.delete_orphan_series({sid1})
        db.commit()

        assert num_deleted == 1
        rows = db.get_filtered_series()
        assert len(rows) == 1
        assert rows[0]["patient_id"] == "PAT001"

    def test_delete_orphans_cascades_to_studies_and_patients(self, db):
        """When all series for a patient are deleted, study and patient are cleaned up."""
        _insert_full_hierarchy(db, "PAT001", "1.2.1", "1.2.1.1")
        _, _, sid2 = _insert_full_hierarchy(db, "PAT002", "1.2.2", "1.2.2.1")

        db.delete_orphan_series({sid2})
        db.commit()

        # PAT001 should be cleaned up entirely
        assert db.get_patient_id_by_patient_id("PAT001") is None
        assert db.get_patient_id_by_patient_id("PAT002") is not None

    def test_delete_orphans_returns_thumbnail_paths(self, db):
        _, _, sid1 = _insert_full_hierarchy(
            db, "PAT001", "1.2.1", "1.2.1.1",
            thumbnail_path="ab/abc123.jpg",
        )
        _, _, sid2 = _insert_full_hierarchy(
            db, "PAT002", "1.2.2", "1.2.2.1",
            thumbnail_path="cd/cde456.jpg",
        )

        # Keep sid1, orphan sid2
        _, thumbnails = db.delete_orphan_series({sid1})
        assert thumbnails == ["cd/cde456.jpg"]

    def test_delete_orphans_empty_valid_set_is_noop(self, db):
        """Empty valid set means keep nothing, but the method returns early."""
        _insert_full_hierarchy(db)
        num_deleted, thumbnails = db.delete_orphan_series(set())
        assert num_deleted == 0
        assert thumbnails == []

    def test_delete_orphans_no_orphans(self, db):
        """When all series are in valid set, nothing is deleted."""
        _, _, sid = _insert_full_hierarchy(db)
        num_deleted, thumbnails = db.delete_orphan_series({sid})
        assert num_deleted == 0
        assert thumbnails == []


# ---------------------------------------------------------------------------
# 8. Pagination (limit / offset)
# ---------------------------------------------------------------------------

class TestPagination:

    @pytest.fixture(autouse=True)
    def _populate(self, db):
        pid = db.insert_patient("PAT001")
        stid = db.insert_study(pid, "1.2.3")
        for i in range(10):
            db.insert_series(stid, f"1.2.3.{i}", series_number=i,
                             series_description=f"Series {i:02d}", modality="MR",
                             qc_status="PASS")
        db.commit()

    def test_limit(self, db):
        rows = db.get_filtered_series(limit=3)
        assert len(rows) == 3

    def test_offset(self, db):
        all_rows = db.get_filtered_series(limit=100, order_by="series_number")
        page2 = db.get_filtered_series(limit=3, offset=3, order_by="series_number")
        assert len(page2) == 3
        assert page2[0]["series_number"] == all_rows[3]["series_number"]

    def test_offset_beyond_total(self, db):
        rows = db.get_filtered_series(limit=10, offset=100)
        assert rows == []

    def test_full_pagination_covers_all(self, db):
        """Paginating through all records yields 10 unique series."""
        all_ids = set()
        offset = 0
        page_size = 4
        while True:
            page = db.get_filtered_series(limit=page_size, offset=offset)
            if not page:
                break
            all_ids.update(r["series_db_id"] for r in page)
            offset += page_size
        assert len(all_ids) == 10


# ---------------------------------------------------------------------------
# 9. Thread safety (concurrent writes)
# ---------------------------------------------------------------------------

class TestThreadSafety:

    def test_concurrent_inserts(self, tmp_path):
        """Multiple threads can insert without corruption."""
        db = QCDatabase(tmp_path / "thread_test.db")
        num_workers = 4
        inserts_per_worker = 25

        def worker(worker_id):
            for i in range(inserts_per_worker):
                patient_id = f"W{worker_id}_P{i}"
                pid = db.insert_patient(patient_id)
                stid = db.insert_study(pid, f"study_{worker_id}_{i}")
                db.insert_series(stid, f"series_{worker_id}_{i}",
                                 qc_status="PASS", modality="MR")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker, w) for w in range(num_workers)]
            for f in futures:
                f.result()  # raises if any thread failed

        db.commit()

        total = db.count_filtered_series()
        assert total == num_workers * inserts_per_worker
        db.close()
