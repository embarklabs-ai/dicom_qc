"""SQLite-backed storage for QC metadata.

Provides persistent storage for large-scale QC workflows (10K-100K+ series)
with fast SQL-based filtering and pagination.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class QCDatabase:
    """SQLite database for QC metadata storage.

    Stores patient/study/series hierarchy with QC results and thumbnail paths.
    Optimized for fast filtering and pagination of large datasets.

    Example:
        db = QCDatabase(Path("/data/_dicom_qc/qc_database.sqlite3"))
        db.insert_patient("patient123", "John Doe")
        series = db.get_filtered_series({"status": "FAIL"}, limit=50)
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: Path):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file. Created if doesn't exist.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        """Create tables if they don't exist."""
        cursor = self._conn.cursor()

        # Check schema version
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS qc_session (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                value TEXT
            )
        """)

        # Check if migration needed
        cursor.execute("SELECT value FROM qc_session WHERE key = 'schema_version'")
        row = cursor.fetchone()
        if row is None:
            cursor.execute(
                "INSERT INTO qc_session (key, value) VALUES ('schema_version', ?)",
                (str(self.SCHEMA_VERSION),)
            )
        # Future: handle schema migrations here

        # Patients table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL UNIQUE,
                patient_name TEXT,
                xnat_subject_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_patients_patient_id ON patients(patient_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_patients_xnat_subject ON patients(xnat_subject_id)")

        # Studies table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS studies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_db_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
                study_uid TEXT NOT NULL,
                study_date TEXT,
                study_description TEXT,
                xnat_session_label TEXT,
                xnat_experiment_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(patient_db_id, study_uid)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_studies_patient ON studies(patient_db_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_studies_date ON studies(study_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_studies_xnat ON studies(xnat_experiment_id)")

        # Series table (main table for filtering)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS series (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                study_db_id INTEGER NOT NULL REFERENCES studies(id) ON DELETE CASCADE,
                series_uid TEXT NOT NULL,
                series_number TEXT,
                series_description TEXT,
                modality TEXT,

                -- QC status (denormalized for fast filtering)
                qc_status TEXT DEFAULT 'PENDING',
                is_derived INTEGER DEFAULT 0,
                derived_info TEXT,
                error_message TEXT,

                -- Thumbnail path (relative to thumbnails/ dir)
                thumbnail_path TEXT,

                -- XNAT-specific
                xnat_scan_id TEXT,
                transfer_syntax TEXT,

                -- File tracking
                file_count INTEGER DEFAULT 0,

                -- Timestamps
                processed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                UNIQUE(study_db_id, series_uid)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_series_study ON series(study_db_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_series_status ON series(qc_status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_series_modality ON series(modality)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_series_description ON series(series_description)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_series_number ON series(series_number)")

        # QC check results (normalized)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS qc_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                series_db_id INTEGER NOT NULL REFERENCES series(id) ON DELETE CASCADE,
                check_name TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT,
                details_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_qc_series ON qc_results(series_db_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_qc_status ON qc_results(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_qc_check ON qc_results(check_name)")

        # Series file paths (for XNAT mode lazy loading)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS series_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                series_db_id INTEGER NOT NULL REFERENCES series(id) ON DELETE CASCADE,
                file_uri TEXT,
                local_path TEXT,
                file_order INTEGER DEFAULT 0
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_files_series ON series_files(series_db_id)")

        self._conn.commit()

    def is_empty(self) -> bool:
        """Check if database has any patient data."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM patients")
        return cursor.fetchone()[0] == 0

    def clear_all(self):
        """Clear all data from the database (for re-migration)."""
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM qc_results")
        cursor.execute("DELETE FROM series_files")
        cursor.execute("DELETE FROM series")
        cursor.execute("DELETE FROM studies")
        cursor.execute("DELETE FROM patients")
        self._conn.commit()

    # =========================================================================
    # Patient CRUD
    # =========================================================================

    def insert_patient(
        self,
        patient_id: str,
        patient_name: str = None,
        xnat_subject_id: str = None,
    ) -> int:
        """Insert patient, return database ID. Updates if exists."""
        cursor = self._conn.cursor()
        cursor.execute("""
            INSERT INTO patients (patient_id, patient_name, xnat_subject_id)
            VALUES (?, ?, ?)
            ON CONFLICT(patient_id) DO UPDATE SET
                patient_name = excluded.patient_name,
                xnat_subject_id = excluded.xnat_subject_id,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
        """, (patient_id, patient_name, xnat_subject_id))
        row = cursor.fetchone()
        return row[0]

    def get_patient_id_by_patient_id(self, patient_id: str) -> Optional[int]:
        """Get database ID for a patient by their patient_id."""
        cursor = self._conn.execute(
            "SELECT id FROM patients WHERE patient_id = ?",
            (patient_id,)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    # =========================================================================
    # Study CRUD
    # =========================================================================

    def insert_study(
        self,
        patient_db_id: int,
        study_uid: str,
        study_date: str = None,
        study_description: str = None,
        xnat_session_label: str = None,
        xnat_experiment_id: str = None,
    ) -> int:
        """Insert study, return database ID. Updates if exists."""
        cursor = self._conn.cursor()
        cursor.execute("""
            INSERT INTO studies (
                patient_db_id, study_uid, study_date, study_description,
                xnat_session_label, xnat_experiment_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(patient_db_id, study_uid) DO UPDATE SET
                study_date = excluded.study_date,
                study_description = excluded.study_description,
                xnat_session_label = excluded.xnat_session_label,
                xnat_experiment_id = excluded.xnat_experiment_id
            RETURNING id
        """, (
            patient_db_id, study_uid, study_date, study_description,
            xnat_session_label, xnat_experiment_id
        ))
        row = cursor.fetchone()
        return row[0]

    def get_study_id(self, patient_db_id: int, study_uid: str) -> Optional[int]:
        """Get database ID for a study."""
        cursor = self._conn.execute(
            "SELECT id FROM studies WHERE patient_db_id = ? AND study_uid = ?",
            (patient_db_id, study_uid)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    # =========================================================================
    # Series CRUD
    # =========================================================================

    def insert_series(
        self,
        study_db_id: int,
        series_uid: str,
        series_number: Any = None,
        series_description: str = None,
        modality: str = None,
        qc_status: str = 'PENDING',
        is_derived: bool = False,
        derived_info: str = None,
        error_message: str = None,
        thumbnail_path: str = None,
        xnat_scan_id: str = None,
        transfer_syntax: str = None,
        file_count: int = 0,
    ) -> int:
        """Insert series, return database ID. Updates if exists."""
        cursor = self._conn.cursor()
        cursor.execute("""
            INSERT INTO series (
                study_db_id, series_uid, series_number, series_description,
                modality, qc_status, is_derived, derived_info, error_message,
                thumbnail_path, xnat_scan_id, transfer_syntax, file_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(study_db_id, series_uid) DO UPDATE SET
                series_number = excluded.series_number,
                series_description = excluded.series_description,
                modality = excluded.modality,
                qc_status = excluded.qc_status,
                is_derived = excluded.is_derived,
                derived_info = excluded.derived_info,
                error_message = excluded.error_message,
                thumbnail_path = excluded.thumbnail_path,
                xnat_scan_id = excluded.xnat_scan_id,
                transfer_syntax = excluded.transfer_syntax,
                file_count = excluded.file_count,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
        """, (
            study_db_id, series_uid, str(series_number) if series_number is not None else None,
            series_description, modality, qc_status, int(is_derived), derived_info,
            error_message, thumbnail_path, xnat_scan_id, transfer_syntax, file_count
        ))
        row = cursor.fetchone()
        return row[0]

    def get_series_id(self, study_db_id: int, series_uid: str) -> Optional[int]:
        """Get database ID for a series."""
        cursor = self._conn.execute(
            "SELECT id FROM series WHERE study_db_id = ? AND series_uid = ?",
            (study_db_id, series_uid)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def update_series_thumbnail(self, series_db_id: int, thumbnail_path: str):
        """Update thumbnail path for a series."""
        self._conn.execute(
            "UPDATE series SET thumbnail_path = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (thumbnail_path, series_db_id)
        )

    def update_series_qc_status(
        self,
        series_db_id: int,
        qc_status: str,
        error_message: str = None,
    ):
        """Update QC status for a series."""
        self._conn.execute("""
            UPDATE series SET
                qc_status = ?,
                error_message = ?,
                processed_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (qc_status, error_message, series_db_id))

    # =========================================================================
    # QC Results
    # =========================================================================

    def insert_qc_result(
        self,
        series_db_id: int,
        check_name: str,
        status: str,
        message: str = None,
        details: dict = None,
    ):
        """Insert a single QC check result."""
        details_json = json.dumps(details) if details else None
        self._conn.execute("""
            INSERT INTO qc_results (series_db_id, check_name, status, message, details_json)
            VALUES (?, ?, ?, ?, ?)
        """, (series_db_id, check_name, status, message, details_json))

    def insert_qc_results_from_report(self, series_db_id: int, qc_report):
        """Insert all QC results from a QCReport object."""
        # Clear existing results for this series
        self._conn.execute("DELETE FROM qc_results WHERE series_db_id = ?", (series_db_id,))

        for result in qc_report.results:
            details = None
            if hasattr(result, 'to_dict'):
                details = result.to_dict()
            elif hasattr(result, 'details'):
                details = result.details

            self.insert_qc_result(
                series_db_id,
                check_name=result.check_name,
                status=result.status,
                message=result.message,
                details=details,
            )

    def get_qc_results(self, series_db_id: int) -> List[dict]:
        """Get all QC results for a series."""
        cursor = self._conn.execute("""
            SELECT check_name, status, message, details_json
            FROM qc_results
            WHERE series_db_id = ?
            ORDER BY id
        """, (series_db_id,))

        results = []
        for row in cursor.fetchall():
            result = {
                'check_name': row['check_name'],
                'status': row['status'],
                'message': row['message'],
                'details': json.loads(row['details_json']) if row['details_json'] else None,
            }
            results.append(result)
        return results

    # =========================================================================
    # Series Files
    # =========================================================================

    def insert_series_files(
        self,
        series_db_id: int,
        file_uris: List[str] = None,
        local_paths: List[str] = None,
    ):
        """Insert file paths for a series."""
        # Clear existing
        self._conn.execute("DELETE FROM series_files WHERE series_db_id = ?", (series_db_id,))

        file_uris = file_uris or []
        local_paths = local_paths or []

        # Pad shorter list with None
        max_len = max(len(file_uris), len(local_paths))
        file_uris = file_uris + [None] * (max_len - len(file_uris))
        local_paths = local_paths + [None] * (max_len - len(local_paths))

        for i, (uri, path) in enumerate(zip(file_uris, local_paths)):
            self._conn.execute("""
                INSERT INTO series_files (series_db_id, file_uri, local_path, file_order)
                VALUES (?, ?, ?, ?)
            """, (series_db_id, uri, path, i))

    def get_series_files(self, series_db_id: int) -> List[Tuple[str, str]]:
        """Get file URIs and paths for a series."""
        cursor = self._conn.execute("""
            SELECT file_uri, local_path FROM series_files
            WHERE series_db_id = ?
            ORDER BY file_order
        """, (series_db_id,))
        return [(row['file_uri'], row['local_path']) for row in cursor.fetchall()]

    # =========================================================================
    # Filtered Queries (for paginated UI)
    # =========================================================================

    def get_filtered_series(
        self,
        filters: Dict[str, Any] = None,
        limit: int = 50,
        offset: int = 0,
        order_by: str = 'patient_id, study_date, series_number',
    ) -> List[dict]:
        """Get series matching filters with pagination.

        Args:
            filters: Dict with optional keys:
                - status: QC status to filter by (None = all)
                - patient_id: Patient ID to filter by
                - study_key: Study UID or XNAT session label
                - series_number: Series number to filter by
                - description_like: Substring match on series description
            limit: Max rows to return
            offset: Rows to skip (for pagination)
            order_by: SQL ORDER BY clause

        Returns:
            List of dicts with series info including patient/study context.
        """
        filters = filters or {}

        # Build WHERE clause
        conditions = []
        params = []

        if filters.get('status'):
            conditions.append("s.qc_status = ?")
            params.append(filters['status'])

        if filters.get('patient_id'):
            conditions.append("p.patient_id = ?")
            params.append(filters['patient_id'])

        if filters.get('study_key'):
            conditions.append("(st.study_uid = ? OR st.xnat_session_label = ?)")
            params.extend([filters['study_key'], filters['study_key']])

        if filters.get('series_number') is not None:
            conditions.append("s.series_number = ?")
            params.append(str(filters['series_number']))

        if filters.get('description_like'):
            conditions.append("s.series_description LIKE ?")
            params.append(filters['description_like'])

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
            SELECT
                s.id as series_db_id,
                s.series_uid,
                s.series_number,
                s.series_description,
                s.modality,
                s.qc_status,
                s.is_derived,
                s.derived_info,
                s.error_message,
                s.thumbnail_path,
                s.xnat_scan_id,
                s.transfer_syntax,
                s.file_count,
                s.processed_at,
                st.id as study_db_id,
                st.study_uid,
                st.study_date,
                st.study_description,
                st.xnat_session_label,
                st.xnat_experiment_id,
                p.id as patient_db_id,
                p.patient_id,
                p.patient_name,
                p.xnat_subject_id
            FROM series s
            JOIN studies st ON s.study_db_id = st.id
            JOIN patients p ON st.patient_db_id = p.id
            WHERE {where_clause}
            ORDER BY {order_by}
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        cursor = self._conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def count_filtered_series(self, filters: Dict[str, Any] = None) -> int:
        """Count series matching filters."""
        filters = filters or {}

        conditions = []
        params = []

        if filters.get('status'):
            conditions.append("s.qc_status = ?")
            params.append(filters['status'])

        if filters.get('patient_id'):
            conditions.append("p.patient_id = ?")
            params.append(filters['patient_id'])

        if filters.get('study_key'):
            conditions.append("(st.study_uid = ? OR st.xnat_session_label = ?)")
            params.extend([filters['study_key'], filters['study_key']])

        if filters.get('series_number') is not None:
            conditions.append("s.series_number = ?")
            params.append(str(filters['series_number']))

        if filters.get('description_like'):
            conditions.append("s.series_description LIKE ?")
            params.append(filters['description_like'])

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
            SELECT COUNT(*) FROM series s
            JOIN studies st ON s.study_db_id = st.id
            JOIN patients p ON st.patient_db_id = p.id
            WHERE {where_clause}
        """

        cursor = self._conn.execute(query, params)
        return cursor.fetchone()[0]

    def get_summary_counts(self) -> Dict[str, int]:
        """Get counts by QC status."""
        cursor = self._conn.execute(
            "SELECT qc_status, COUNT(*) FROM series GROUP BY qc_status"
        )
        return dict(cursor.fetchall())

    def get_total_series_count(self) -> int:
        """Get total number of series."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM series")
        return cursor.fetchone()[0]

    # =========================================================================
    # Filter Options (for UI dropdowns)
    # =========================================================================

    def get_patient_options(self) -> List[Tuple[str, str]]:
        """Get (patient_id, patient_name) pairs for dropdown."""
        cursor = self._conn.execute("""
            SELECT patient_id, patient_name FROM patients ORDER BY patient_id
        """)
        return [(row['patient_id'], row['patient_name']) for row in cursor.fetchall()]

    def get_study_options(self, patient_id: str = None) -> List[Tuple[str, str, str]]:
        """Get (study_uid, study_date, study_description) for dropdown."""
        if patient_id:
            cursor = self._conn.execute("""
                SELECT st.study_uid, st.study_date, st.study_description, st.xnat_session_label
                FROM studies st
                JOIN patients p ON st.patient_db_id = p.id
                WHERE p.patient_id = ?
                ORDER BY st.study_date
            """, (patient_id,))
        else:
            cursor = self._conn.execute("""
                SELECT study_uid, study_date, study_description, xnat_session_label
                FROM studies ORDER BY study_date
            """)
        return [
            (row['study_uid'], row['study_date'], row['study_description'], row['xnat_session_label'])
            for row in cursor.fetchall()
        ]

    def get_series_number_options(self) -> List[str]:
        """Get unique series numbers for dropdown."""
        cursor = self._conn.execute("""
            SELECT DISTINCT series_number FROM series
            WHERE series_number IS NOT NULL
            ORDER BY
                CASE WHEN series_number GLOB '[0-9]*' THEN 0 ELSE 1 END,
                CAST(series_number AS INTEGER),
                series_number
        """)
        return [row['series_number'] for row in cursor.fetchall()]

    # =========================================================================
    # Session metadata
    # =========================================================================

    def set_session_value(self, key: str, value: str):
        """Set a session metadata value."""
        self._conn.execute("""
            INSERT INTO qc_session (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """, (key, value))

    def get_session_value(self, key: str) -> Optional[str]:
        """Get a session metadata value."""
        cursor = self._conn.execute(
            "SELECT value FROM qc_session WHERE key = ?", (key,)
        )
        row = cursor.fetchone()
        return row['value'] if row else None

    # =========================================================================
    # Transaction control
    # =========================================================================

    def commit(self):
        """Commit pending changes."""
        self._conn.commit()

    def close(self):
        """Close database connection."""
        self._conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - commit and close."""
        if exc_type is None:
            self.commit()
        self.close()
