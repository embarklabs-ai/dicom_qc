"""Quickcheck module for efficient batch review of DICOM data."""

import hashlib
import html
import os
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pydicom


from dicom_qc.core.dicom_loader import DicomLoader
from dicom_qc.core.geometry import GeometryQC
from dicom_qc.visualization.snapshots import SnapshotGenerator
from dicom_qc.quickcheck_html import QuickCheckHTMLMixin
from dicom_qc.quickcheck_display import QuickCheckDisplayMixin


def _xnat_retry(func: Callable, max_retries: int = 3, base_delay: float = 2.0):
    """Execute function with retry logic for transient XNAT errors.

    Retries on 502, 503, 504 errors and connection issues with exponential backoff.

    Args:
        func: Callable to execute
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (doubles each retry)

    Returns:
        Result of func()

    Raises:
        Last exception if all retries fail
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            error_str = str(e).lower()
            # Check if it's a retryable error (503, 502, 504, connection issues)
            is_retryable = any(
                x in error_str
                for x in ["503", "502", "504", "connection", "timeout", "temporarily"]
            )

            if not is_retryable or attempt >= max_retries:
                raise

            delay = base_delay * (2**attempt)
            time.sleep(delay)


def _fetch_scan_files(scan) -> List:
    """Fetch all DICOM files from an XNAT scan, excluding SNAPSHOTS resource."""
    all_files = []
    for res in scan.resources.values():
        res_label = getattr(res, "label", "").upper()
        if res_label != "SNAPSHOTS":
            all_files.extend(res.files.values())
    return all_files


class XnatFileHandle:
    """Lightweight file handle that mimics xnatpy file objects.

    Provides the same interface as xnatpy file objects (uri, data_path, open())
    using stored paths. Prefers local data_path when available.
    """

    def __init__(self, session: Any, uri: str, local_path: str = None):
        """
        Args:
            session: xnatpy session object (for download fallback)
            uri: XNAT REST API URI for the file
            local_path: Local filesystem path if file is mounted
        """
        self._session = session
        self.uri = uri
        self._local_path = local_path

    @property
    def data_path(self) -> Optional[str]:
        """Return local path if file exists, else None."""
        if self._local_path and Path(self._local_path).exists():
            return self._local_path
        return None

    def open(self):
        """Open file - uses local path if available, downloads otherwise."""
        if self.data_path:
            return open(self.data_path, "rb")
        # Fallback to download
        response = self._session.get(self.uri)
        return BytesIO(response.content)


@dataclass
class SeriesInfo:
    """Information about a DICOM series."""

    uid: str  # SeriesInstanceUID (DICOM), or empty if unavailable
    series_number: Any  # int or str (e.g., XNAT scan ID)
    description: str
    modality: str
    files: List[Any] = field(default_factory=list)  # Path or xnat file objects
    volume: Any = None
    qc_report: Any = None
    thumbnail: Optional[str] = None  # Base64 thumbnail (legacy, for backward compat)
    _thumbnail_path: Optional[str] = None  # Relative path to disk-cached thumbnail
    error: Optional[str] = None
    transfer_syntax: Optional[str] = None
    implementation: Optional[str] = None
    _xnat_files: bool = False  # True if files are xnat file objects
    _scan_obj: Any = field(
        default=None, repr=False
    )  # xnat scan object for lazy file loading
    _file_uris: List[str] = field(
        default_factory=list
    )  # Picklable file URIs for restore
    _file_paths: List[str] = field(
        default_factory=list
    )  # Local file paths (if mounted)
    _scan_uri: Optional[str] = None  # Scan URI for restoring access
    xnat_scan_id: Optional[str] = None  # XNAT scan ID (only in XNAT mode)
    _xnat_subject_label: Optional[str] = (
        None  # XNAT subject label (for thumbnail naming)
    )
    _xnat_session_label: Optional[str] = (
        None  # XNAT session label (for thumbnail naming)
    )
    # Derived data fields (RTStruct, SEG, etc.)
    is_derived: bool = False
    referenced_series_uid: Optional[str] = (
        None  # SeriesInstanceUID of referenced images
    )
    derived_info: Optional[str] = None  # Human-readable info about the derived data
    _db_id: Optional[int] = None  # Database row ID (for scaled mode)

    @property
    def qc_status(self) -> str:
        if self.error:
            return "ERROR"
        if self.is_derived:
            return "DERIVED"
        if self.qc_report:
            return self.qc_report.overall_status
        return "PENDING"

    @property
    def label(self) -> str:
        return f"#{self.series_number} {self.modality}: {self.description}"


@dataclass
class StudyInfo:
    """Information about a DICOM study (timepoint)."""

    uid: str  # StudyInstanceUID (DICOM), or empty if unavailable
    date: str
    description: str
    series: Dict[str, SeriesInfo] = field(default_factory=dict)
    xnat_session_label: Optional[str] = None  # XNAT session label (only in XNAT mode)
    xnat_experiment_id: Optional[str] = (
        None  # XNAT internal experiment ID (e.g., XNAT_TEST02_E39302)
    )

    @property
    def label(self) -> str:
        if self.date and self.description:
            return f"{self.date}: {self.description}"
        return self.description or self.date or self.uid


@dataclass
class PatientInfo:
    """Information about a patient."""

    patient_id: str
    patient_name: str
    studies: Dict[str, StudyInfo] = field(default_factory=dict)
    xnat_subject_id: Optional[str] = (
        None  # XNAT internal subject ID (e.g., XNAT_TEST02_S24513)
    )

    @property
    def label(self) -> str:
        return (
            f"{self.patient_id}: {self.patient_name}"
            if self.patient_name
            else self.patient_id
        )


class QuickCheck(QuickCheckHTMLMixin, QuickCheckDisplayMixin):
    """Batch DICOM review with thumbnail grid visualization."""

    STATUS_COLORS = {
        "PASS": "#28a745",
        "WARNING": "#ffc107",
        "FAIL": "#dc3545",
        "ERROR": "#6c757d",
        "PENDING": "#17a2b8",
        "DERIVED": "#9c27b0",  # Purple for derived/non-image types
        "NOTE": "#17a2b8",  # Cyan/teal for informational notes (4D data, etc.)
    }

    # Modalities that are derived/non-image data (don't require geometry QC)
    DERIVED_MODALITIES = {
        "RTSTRUCT",  # Radiotherapy Structure Set (contours)
        "SEG",  # Segmentation
        "PR",  # Presentation State
        "KO",  # Key Object Selection
        "SR",  # Structured Report
        "DOC",  # Document
        "REG",  # Registration
        "FID",  # Fiducials
        "RTPLAN",  # Radiotherapy Plan
    }

    # SOP Classes that are NOT displayable images (no PixelData)
    NON_IMAGE_SOP_CLASSES = {
        "1.2.840.10008.5.1.4.1.1.66",  # Raw Data Storage
        "1.2.840.10008.5.1.4.1.1.66.1",  # Spatial Registration Storage
        "1.2.840.10008.5.1.4.1.1.66.2",  # Spatial Fiducials Storage
        "1.2.840.10008.5.1.4.1.1.66.3",  # Deformable Spatial Registration Storage
        "1.2.840.10008.5.1.4.1.1.66.4",  # Segmentation Storage (sometimes no pixels)
        "1.2.840.10008.5.1.4.1.1.67",  # Real World Value Mapping Storage
        "1.2.840.10008.5.1.4.1.1.88.11",  # Basic Text SR Storage
        "1.2.840.10008.5.1.4.1.1.88.22",  # Enhanced SR Storage
        "1.2.840.10008.5.1.4.1.1.88.33",  # Comprehensive SR Storage
        "1.2.840.10008.5.1.4.1.1.88.34",  # Comprehensive 3D SR Storage
    }

    def __init__(self, data_dir: Optional[Path] = None, use_db: bool = True):
        """Initialize with optional data directory.

        Args:
            data_dir: Directory containing DICOM files (local mode) or cache location (XNAT mode)
            use_db: If True, use SQLite database for scaled operations (100K+ series).
                   Database stored in {data_dir}/_dicom_qc/qc_database.sqlite3
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self.patients: Dict[str, PatientInfo] = {}
        self.loader = DicomLoader()
        self._xnat_mode = False
        self._xnat_session: Any = None  # XNAT session for restoring file access
        self._xnat_project_id: Optional[str] = None  # XNAT project ID for OHIF links
        self._xnat_base_url: Optional[str] = None  # XNAT base URL for OHIF links

        # Flag to skip orphan cleanup during discovery (partial data in memory)
        self._discovering = False

        # Scaled mode: SQLite database + thumbnail disk cache
        self._use_db = use_db
        self._db = None
        self._thumb_cache = None

        if use_db and data_dir:
            self._init_storage()

        # Register atexit handler to close DB on interpreter shutdown
        import atexit

        atexit.register(self.close)

    def close(self):
        """Close database connection and release resources.

        Safe to call multiple times. Called automatically on garbage collection
        and interpreter shutdown.
        """
        if self._db is not None:
            try:
                self._db.close()
            except Exception:
                pass
            self._db = None

    def __del__(self):
        """Safety net: close DB on garbage collection."""
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _init_storage(self):
        """Initialize database and thumbnail cache for scaled operations."""
        if not self.data_dir:
            return

        from dicom_qc.storage import QCDatabase, ThumbnailCache

        storage_dir = self.data_dir / "_dicom_qc"
        storage_dir.mkdir(parents=True, exist_ok=True)

        self._db = QCDatabase(storage_dir / "qc_database.sqlite3")
        self._thumb_cache = ThumbnailCache(storage_dir / "thumbnails")

    @staticmethod
    def _get_effective_uids(
        series: "SeriesInfo", study: "StudyInfo", study_key: str
    ) -> tuple:
        """Get stable UIDs for database keys.

        XNAT mode: ALWAYS use XNAT identifiers (scan_id/session_label) to avoid
        duplicates when DICOM UID gets populated later during processing.
        Local mode: use DICOM UIDs (always available from file headers).

        Returns:
            (effective_study_uid, effective_series_uid)
        """
        if series.xnat_scan_id:
            # XNAT mode - use stable XNAT identifiers
            return (study.xnat_session_label or study_key, series.xnat_scan_id)
        else:
            # Local mode - use DICOM UIDs
            return (study.uid or study_key, series.uid or str(series.series_number))

    def _sync_series_to_db(
        self,
        series: "SeriesInfo",
        patient_id: str,
        study_key: str,
        patient: "PatientInfo",
        study: "StudyInfo",
    ) -> int:
        """Sync a series to the database, return database ID."""
        if not self._db:
            return None

        # Insert/update patient
        patient_db_id = self._db.insert_patient(
            patient_id=patient_id,
            patient_name=patient.patient_name,
            xnat_subject_id=patient.xnat_subject_id,
        )

        effective_study_uid, effective_series_uid = self._get_effective_uids(
            series, study, study_key
        )

        # Insert/update study
        study_db_id = self._db.insert_study(
            patient_db_id=patient_db_id,
            study_uid=effective_study_uid,
            study_date=study.date,
            study_description=study.description,
            xnat_session_label=study.xnat_session_label,
            xnat_experiment_id=study.xnat_experiment_id,
        )

        # Insert/update series
        series_db_id = self._db.insert_series(
            study_db_id=study_db_id,
            series_uid=effective_series_uid,
            series_number=series.series_number,
            series_description=series.description,
            modality=series.modality,
            qc_status=series.qc_status,
            is_derived=series.is_derived,
            derived_info=series.derived_info,
            error_message=series.error,
            thumbnail_path=series._thumbnail_path,
            xnat_scan_id=series.xnat_scan_id,
            transfer_syntax=series.transfer_syntax,
            file_count=len(series.files) if series.files else len(series._file_uris),
        )

        # Store series files
        if series._file_uris or series._file_paths:
            self._db.insert_series_files(
                series_db_id,
                file_uris=series._file_uris,
                local_paths=series._file_paths,
            )

        # Store QC results if available
        if series.qc_report:
            self._db.insert_qc_results_from_report(series_db_id, series.qc_report)

        series._db_id = series_db_id
        return series_db_id

    def _sync_all_to_db(self):
        """Sync all series to database and clean up orphan records."""
        if not self._db:
            return

        # Collect all valid series database IDs
        valid_series_ids = set()

        for patient_id, patient in self.patients.items():
            for study_key, study in patient.studies.items():
                for series_key, series in study.series.items():
                    db_id = self._sync_series_to_db(
                        series, patient_id, study_key, patient, study
                    )
                    if db_id:
                        valid_series_ids.add(db_id)

        # Delete orphan records (series no longer in memory)
        # Skip during discovery — only a partial set of series is in memory
        if self._discovering:
            self._db.commit()
            return
        deleted, orphan_thumb_paths = self._db.delete_orphan_series(valid_series_ids)
        if deleted:
            print(f"Cleaned up {deleted} orphan series from database")
            # Clean up orphan thumbnail files from disk
            if self._thumb_cache and orphan_thumb_paths:
                for rel_path in orphan_thumb_paths:
                    full_path = self._thumb_cache.cache_dir / rel_path
                    if full_path.exists():
                        try:
                            full_path.unlink()
                        except OSError:
                            pass

        self._db.commit()

    def _migrate_thumbnails_to_disk(self):
        """Migrate base64 thumbnails to disk cache."""
        if not self._thumb_cache:
            return

        migrated = 0
        for series in self.get_all_series():
            if series.thumbnail and not series._thumbnail_path:
                # Migrate base64 to disk
                # Use XNAT identifiers when available (series.uid may be empty in XNAT mode)
                if (
                    series.xnat_scan_id
                    and series._xnat_subject_label
                    and series._xnat_session_label
                ):
                    thumb_path = self._thumb_cache.get_path_for_xnat(
                        series._xnat_subject_label,
                        series._xnat_session_label,
                        series.xnat_scan_id,
                    )
                    thumb_path.parent.mkdir(parents=True, exist_ok=True)
                    # Decode and save
                    import base64

                    try:
                        raw_data = base64.b64decode(series.thumbnail)
                        if self._thumb_cache._looks_like_jpeg(raw_data):
                            thumb_path.write_bytes(raw_data)
                            rel_path = self._thumb_cache.get_relative_path_xnat(
                                series._xnat_subject_label,
                                series._xnat_session_label,
                                series.xnat_scan_id,
                            )
                        else:
                            rel_path = None
                    except Exception:
                        rel_path = None
                else:
                    rel_path = self._thumb_cache.save_jpeg_thumbnail_from_base64(
                        series.uid, series.thumbnail
                    )
                if rel_path:
                    series._thumbnail_path = rel_path
                    series.thumbnail = None  # Free memory
                    migrated += 1

        if migrated:
            print(f"Migrated {migrated} thumbnails to disk cache")

        return migrated

    def _load_from_db(self):
        """Reconstruct self.patients hierarchy from the database."""
        from dicom_qc.core.geometry import QCResult, QCReport

        rows = self._db.load_all()
        all_qc_results = self._db.load_all_qc_results()
        all_files = self._db.load_all_series_files()

        # Restore session metadata
        xnat_mode_val = self._db.get_session_value("xnat_mode")
        if xnat_mode_val is not None:
            self._xnat_mode = xnat_mode_val == "1"
        data_dir_val = self._db.get_session_value("data_dir")
        if data_dir_val and not self.data_dir:
            self.data_dir = Path(data_dir_val)
        xnat_project_id = self._db.get_session_value("xnat_project_id")
        if xnat_project_id:
            self._xnat_project_id = xnat_project_id
        xnat_base_url = self._db.get_session_value("xnat_base_url")
        if xnat_base_url:
            self._xnat_base_url = xnat_base_url

        self.patients = {}

        for row in rows:
            patient_id = row["patient_id"]
            study_uid = row["study_uid"]
            series_db_id = row["series_db_id"]

            # Ensure patient exists
            if patient_id not in self.patients:
                self.patients[patient_id] = PatientInfo(
                    patient_id=patient_id,
                    patient_name=row["patient_name"] or "",
                    xnat_subject_id=row["xnat_subject_id"],
                )
            patient = self.patients[patient_id]

            # Ensure study exists
            if study_uid not in patient.studies:
                patient.studies[study_uid] = StudyInfo(
                    uid=study_uid,
                    date=row["study_date"] or "",
                    description=row["study_description"] or "",
                    xnat_session_label=row["xnat_session_label"],
                    xnat_experiment_id=row["xnat_experiment_id"],
                )
            study = patient.studies[study_uid]

            # Reconstruct QCReport if we have results
            qc_report = None
            db_results = all_qc_results.get(series_db_id, [])
            if db_results:
                qc_results = []
                orientation_labels = {}
                primary_plane = ""
                overall_status = row["qc_status"] or "PASS"

                for r in db_results:
                    qc_results.append(
                        QCResult(
                            status=r["status"],
                            check_name=r["check_name"],
                            message=r["message"] or "",
                            details=r["details"] or {},
                        )
                    )
                    # Extract orientation info from the Orientation Labels check
                    if r["check_name"] == "Orientation Labels" and r["details"]:
                        orientation_labels = r["details"].get("orientation_labels", {})
                        primary_plane = r["details"].get("primary_plane", "")

                qc_report = QCReport(
                    scan_id=row["series_uid"],
                    results=qc_results,
                    overall_status=overall_status,
                    orientation_labels=orientation_labels,
                    primary_plane=primary_plane,
                )

            # Reconstruct file paths
            file_uris = []
            file_paths = []
            for uri, path in all_files.get(series_db_id, []):
                file_uris.append(uri or "")
                file_paths.append(path or "")

            # Determine series key
            series_key = row["xnat_scan_id"] or row["series_uid"]

            is_xnat = bool(row["xnat_scan_id"])

            series = SeriesInfo(
                uid=row["series_uid"],
                series_number=row["series_number"],
                description=row["series_description"] or "",
                modality=row["modality"] or "",
                qc_report=qc_report,
                _thumbnail_path=row["thumbnail_path"],
                error=row["error_message"],
                transfer_syntax=row["transfer_syntax"],
                is_derived=bool(row["is_derived"]),
                derived_info=row["derived_info"],
                _xnat_files=is_xnat,
                xnat_scan_id=row["xnat_scan_id"],
                _file_uris=file_uris,
                _file_paths=file_paths,
                _db_id=series_db_id,
                _xnat_subject_label=patient_id if is_xnat else None,
                _xnat_session_label=row["xnat_session_label"] if is_xnat else None,
            )

            study.series[series_key] = series

    def save(self) -> None:
        """Save state to the SQLite database.

        Syncs all in-memory series data to the database. Migrates any
        base64 thumbnails to disk cache first.

        For XNAT series, stores file URIs/paths and clears non-serializable
        objects (scan objects, file handles). Workers hold their own local
        references, so this is safe during parallel processing.
        """
        if not self._db:
            if not self.data_dir:
                return
            self._init_storage()

        # Ensure file URIs and paths are stored; clear non-serializable objects
        for series in self.get_all_series():
            series.volume = None
            if series._xnat_files:
                if series.files and not series._file_uris:
                    try:
                        series._file_uris = [f.uri for f in series.files]
                        series._file_paths = [
                            getattr(f, "data_path", None) or "" for f in series.files
                        ]
                    except Exception:
                        pass
                series._scan_obj = None
                series.files = []
            else:
                if series.files and not series._file_paths:
                    series._file_paths = [str(f) for f in series.files]

        # Migrate thumbnails to disk cache if available (frees memory)
        if self._thumb_cache:
            self._migrate_thumbnails_to_disk()

        # Sync to database
        self._sync_all_to_db()

        # Store session metadata for load()
        self._db.set_session_value("xnat_mode", "1" if self._xnat_mode else "0")
        self._db.set_session_value(
            "data_dir", str(self.data_dir) if self.data_dir else ""
        )
        if self._xnat_project_id:
            self._db.set_session_value("xnat_project_id", self._xnat_project_id)
        if self._xnat_base_url:
            self._db.set_session_value("xnat_base_url", self._xnat_base_url)
        self._db.commit()

    def load(self) -> "QuickCheck":
        """Load state from the SQLite database.

        Reconstructs the full patient/study/series hierarchy from the database,
        including QC results, file paths, and thumbnail references.

        Returns:
            self for chaining
        """
        if not self._db:
            if not self.data_dir:
                raise FileNotFoundError("No data_dir set and no database available")
            db_path = self.data_dir / "_dicom_qc" / "qc_database.sqlite3"
            if not db_path.exists():
                raise FileNotFoundError(f"No database found at {db_path}")
            self._init_storage()

        self._load_from_db()

        # Count what was loaded
        all_series = self.get_all_series()
        total_series = len(all_series)
        with_thumbnail = sum(1 for s in all_series if s.thumbnail or s._thumbnail_path)
        derived = sum(1 for s in all_series if s.is_derived)
        errors = sum(1 for s in all_series if s.error)
        pending = total_series - with_thumbnail - derived - errors

        print("Loaded state from database")
        status_parts = [f"{len(self.patients)} subjects", f"{total_series} series"]
        if pending > 0:
            status_parts.append(f"{pending} pending")
        if errors > 0:
            status_parts.append(f"{errors} errors")
        if self._db:
            status_parts.append("(scaled mode)")
        print(f"  {', '.join(status_parts)}")

        return self

    def has_save(self) -> bool:
        """Check if a database with data exists at the default location."""
        if not self.data_dir:
            return False
        db_path = self.data_dir / "_dicom_qc" / "qc_database.sqlite3"
        if not db_path.exists():
            return False
        # Check if it actually has data (not just an empty schema)
        if self._db:
            return not self._db.is_empty()
        # Open temporarily to check
        from dicom_qc.storage import QCDatabase

        try:
            db = QCDatabase(db_path)
            has_data = not db.is_empty()
            db.close()
            return has_data
        except Exception:
            return False

    def load_if_exists(self) -> bool:
        """Load from database if it exists and has data.

        Returns:
            True if loaded, False if no saved state found
        """
        if self.has_save():
            self.load()
            return True
        return False

    @classmethod
    def from_save(cls, data_dir: Path) -> "QuickCheck":
        """Create a QuickCheck instance from a saved database.

        Args:
            data_dir: Directory containing _dicom_qc/ storage

        Returns:
            QuickCheck instance with loaded state
        """
        qc = cls(data_dir=data_dir)
        qc.load()
        return qc

    def reset(self, delete_storage: bool = True) -> "QuickCheck":
        """Reset to fresh state, optionally deleting all stored data.

        Use this to start over with a clean slate. Clears:
        - All in-memory patient/study/series data
        - SQLite database (if delete_storage=True)
        - Thumbnail cache (if delete_storage=True)

        Args:
            delete_storage: If True, delete database and thumbnails.
                          If False, only clear in-memory state.

        Returns:
            self for chaining

        Example:
            qc = QuickCheck(data_dir)
            qc.reset()      # Clear everything and start fresh
            qc.discover()   # Re-scan for DICOM files
        """
        import shutil

        # Clear in-memory state
        self.patients = {}

        # Close database connection BEFORE deleting storage directory
        if self._db is not None:
            try:
                self._db.close()
            except Exception:
                pass
            self._db = None
        self._thumb_cache = None

        if delete_storage and self.data_dir:
            storage_dir = Path(self.data_dir) / "_dicom_qc"
            if storage_dir.exists():
                shutil.rmtree(storage_dir, ignore_errors=True)
                # If directory still exists (stale NFS .nfs lock files from a
                # previous interrupted process), rename it out of the way so
                # we can create a fresh _dicom_qc directory.
                if storage_dir.exists():
                    import uuid

                    stale_name = f"_dicom_qc_stale_{uuid.uuid4().hex[:8]}"
                    stale_dir = storage_dir.parent / stale_name
                    try:
                        storage_dir.rename(stale_dir)
                        print(
                            f"Renamed stale storage to {stale_name} (NFS lock files present, safe to delete later)"
                        )
                    except OSError:
                        # Can't even rename — clear what we can
                        for item in storage_dir.iterdir():
                            try:
                                if item.is_dir():
                                    shutil.rmtree(item, ignore_errors=True)
                                else:
                                    item.unlink()
                            except OSError:
                                pass
                        print(
                            f"Cleared storage: {storage_dir} (some NFS lock files may remain)"
                        )
                else:
                    print(f"Deleted storage: {storage_dir}")

        # Re-initialize storage if using scaled mode
        if self._use_db and self.data_dir:
            self._init_storage()

        print("Reset complete. Ready for discover().")
        return self

    def connect_xnat(self, session: Any) -> None:
        """Connect an XNAT session for file access after loading from save.

        After loading from a save file, call this method with your XNAT session
        to enable file access using the stored file paths/URIs. This avoids
        re-iterating through all XNAT subjects/experiments/scans.

        Args:
            session: xnatpy session object (from xnat.connect())

        Example:
            qc = QuickCheck.from_save(data_dir)
            qc.connect_xnat(xnat.connect())
            qc.process_all()  # Files accessed via stored paths
        """
        self._xnat_session = session

        # Extract base URL from session for OHIF links (if not already set)
        if not self._xnat_base_url:
            if hasattr(session, "_original_uri"):
                self._xnat_base_url = session._original_uri.rstrip("/")
            elif hasattr(session, "host"):
                self._xnat_base_url = session.host.rstrip("/")

        # Count series with stored file info
        all_series = self.get_all_series()
        total = len(all_series)

        if total == 0:
            print("XNAT session connected (no stored data)")
        else:
            with_paths = sum(
                1 for s in all_series if s._file_paths and any(s._file_paths)
            )
            with_uris = sum(1 for s in all_series if s._file_uris)
            if with_paths > 0:
                print(
                    f"XNAT session connected ({with_paths}/{total} series have local paths, {with_uris} have URIs)"
                )
            else:
                print(
                    f"XNAT session connected ({with_uris}/{total} series have stored URIs)"
                )

    def discover(self, refresh: bool = False) -> Dict[str, PatientInfo]:
        """Scan DICOM files and build patient/study/series hierarchy.

        Uses os.walk with followlinks=True to traverse symbolic links.

        Args:
            refresh: If True, clear existing data and re-discover everything.
                     If False (default), preserve existing series that have thumbnails/results.
        """

        # Store existing processed series to preserve them
        existing_series = {}
        if not refresh:
            for series in self.get_all_series():
                if series.thumbnail or series.is_derived or series.qc_report:
                    existing_series[series.uid] = series

        self.patients = {}
        self._discovering = True

        # Use os.walk to follow symlinks (rglob does not follow symlinks by default)
        dcm_files = []
        for root, dirs, files in os.walk(self.data_dir, followlinks=True):
            for f in files:
                if f.lower().endswith(".dcm"):
                    dcm_files.append(Path(root) / f)

        for dcm_file in dcm_files:
            try:
                ds = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
                self._add_file_to_hierarchy(ds, dcm_file)
            except Exception:
                pass

        # Restore existing processed data
        if existing_series:
            restored = 0
            for series in self.get_all_series():
                if series.uid in existing_series:
                    old = existing_series[series.uid]
                    series.thumbnail = old.thumbnail
                    series._thumbnail_path = old._thumbnail_path
                    series.qc_report = old.qc_report
                    series.is_derived = old.is_derived
                    series.derived_info = old.derived_info
                    series.error = old.error
                    restored += 1
            if restored > 0:
                print(f"Restored {restored} previously processed series")

        self._discovering = False

        # Auto-save after discovery
        if self.data_dir:
            self.save()

        return self.patients

    def _ensure_xnat_patient(
        self, subject_label: str, xnat_subject_id: str = None
    ) -> "PatientInfo":
        """Ensure a patient exists in the hierarchy, creating if needed.

        Args:
            subject_label: XNAT subject label (used as patient_id)
            xnat_subject_id: Internal XNAT subject ID

        Returns:
            PatientInfo for the subject
        """
        if subject_label not in self.patients:
            self.patients[subject_label] = PatientInfo(
                subject_label, "", xnat_subject_id=xnat_subject_id
            )
        elif xnat_subject_id and self.patients[subject_label].xnat_subject_id is None:
            self.patients[subject_label].xnat_subject_id = xnat_subject_id
        return self.patients[subject_label]

    def _ensure_xnat_study(
        self, patient: "PatientInfo", session_label: str, xnat_experiment_id: str = None
    ) -> "StudyInfo":
        """Ensure a study exists in the patient hierarchy, creating if needed.

        Args:
            patient: PatientInfo to add study to
            session_label: XNAT session label (used as study key)
            xnat_experiment_id: Internal XNAT experiment ID

        Returns:
            StudyInfo for the session
        """
        if session_label not in patient.studies:
            patient.studies[session_label] = StudyInfo(
                uid="",
                date="",
                description=session_label,
                xnat_session_label=session_label,
                xnat_experiment_id=xnat_experiment_id,
            )
        elif (
            xnat_experiment_id
            and patient.studies[session_label].xnat_experiment_id is None
        ):
            patient.studies[session_label].xnat_experiment_id = xnat_experiment_id
        return patient.studies[session_label]

    def _create_series_from_xnat_scan(
        self, scan: Any, subject_label: str, session_label: str
    ) -> "SeriesInfo":
        """Create a SeriesInfo from an XNAT scan, fetching metadata and file URIs.

        Args:
            scan: XNAT scan object
            subject_label: Parent subject label
            session_label: Parent session label

        Returns:
            SeriesInfo populated with XNAT metadata and file URIs
        """
        scan_id = scan.id

        # Get metadata from XNAT (with retry for transient errors)
        try:
            series_desc = _xnat_retry(
                lambda: getattr(scan, "series_description", "") or ""
            )
            modality = _xnat_retry(lambda: getattr(scan, "modality", "??") or "??")
        except Exception:
            series_desc = ""
            modality = "??"

        series = SeriesInfo(
            uid="",
            series_number=scan_id,
            description=series_desc,
            modality=modality,
            xnat_scan_id=scan_id,
            _xnat_subject_label=subject_label,
            _xnat_session_label=session_label,
        )
        series._xnat_files = True

        # Fetch and store file URIs and local paths
        try:
            all_files = _xnat_retry(lambda: _fetch_scan_files(scan))
            series._file_uris = [f.uri for f in all_files]
            series._file_paths = [
                getattr(f, "data_path", None) or "" for f in all_files
            ]
        except Exception as e:
            series.error = f"Failed to fetch file list: {e}"

        return series

    def _refetch_missing_file_uris(self, series: "SeriesInfo", scan: Any) -> None:
        """Re-fetch file URIs for an existing series that's missing them.

        Args:
            series: Existing SeriesInfo missing file URIs
            scan: XNAT scan object to fetch from
        """
        try:
            all_files = _xnat_retry(lambda: _fetch_scan_files(scan))
            if all_files:
                series._file_uris = [f.uri for f in all_files]
                series._file_paths = [
                    getattr(f, "data_path", None) or "" for f in all_files
                ]
            else:
                series.error = "No DICOM files in XNAT resources"
        except Exception as e:
            series.error = f"Failed to fetch files: {e}"

    @staticmethod
    def _fetch_subject_hierarchy(subject: Any) -> tuple:
        """Fetch experiment/scan hierarchy for one subject from XNAT.

        Pure I/O operation with no state mutation, safe for parallel execution.

        Args:
            subject: XNAT subject object

        Returns:
            Tuple of (subject_label, xnat_subj_id, experiments_data) where
            experiments_data is list of (session_label, xnat_exp_id, scans_list)
        """
        subject_label = subject.label
        xnat_subj_id = getattr(subject, "id", None)
        experiments_data = []
        try:
            experiments = _xnat_retry(lambda: list(subject.experiments.values()))
        except Exception:
            return subject_label, xnat_subj_id, []
        for experiment in experiments:
            session_label = experiment.label
            xnat_exp_id = getattr(experiment, "id", None)
            try:
                scans = _xnat_retry(lambda: list(experiment.scans.values()))
            except Exception:
                scans = []
            experiments_data.append((session_label, xnat_exp_id, scans))
        return subject_label, xnat_subj_id, experiments_data

    def _collect_xnat_scan_tasks(
        self,
        subjects: List[Any],
        refresh: bool,
        progress_callback=None,
        max_workers: int = 8,
    ) -> tuple:
        """Collect scan tasks from XNAT hierarchy.

        Phase 1 of discovery: fetch subject/experiment/scan hierarchy from XNAT
        (parallelized across subjects), then build task list for processing.

        Args:
            subjects: List of XNAT subject objects
            refresh: If True, include all scans. If False, skip scans with URIs.
            progress_callback: Optional callback(subjects_done, total_subjects, subject_label)
            max_workers: Number of parallel workers for fetching

        Returns:
            Tuple of (scan_tasks, total_sessions, total_scans, skipped_scans)
            where scan_tasks is list of (subject_label, session_label, scan, study)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Step 1: Fetch all subject hierarchies in parallel (I/O-bound)
        subject_results = []
        use_parallel = max_workers > 1 and len(subjects) > 2
        subjects_done = 0

        if use_parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._fetch_subject_hierarchy, s): s
                    for s in subjects
                }
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        subject_results.append(result)
                    except Exception:
                        pass
                    subjects_done += 1
                    if progress_callback:
                        label = result[0] if result else ""
                        progress_callback(subjects_done, len(subjects), label)
        else:
            for subject in subjects:
                result = self._fetch_subject_hierarchy(subject)
                subject_results.append(result)
                subjects_done += 1
                if progress_callback:
                    progress_callback(subjects_done, len(subjects), result[0])

        # Step 2: Build hierarchy and collect scan tasks (sequential, fast)
        scan_tasks = []
        total_sessions = 0
        total_scans = 0
        skipped_scans = 0

        for subject_label, xnat_subj_id, experiments_data in subject_results:
            patient = self._ensure_xnat_patient(subject_label, xnat_subj_id)
            for session_label, xnat_exp_id, scans in experiments_data:
                total_sessions += 1
                study = self._ensure_xnat_study(patient, session_label, xnat_exp_id)
                for scan in scans:
                    scan_id = scan.id
                    total_scans += 1

                    # Skip existing scans in incremental mode
                    if not refresh and scan_id in study.series:
                        existing = study.series[scan_id]
                        if existing.error:
                            # Retry errored series as new scan tasks
                            existing.error = None
                            scan_tasks.append(
                                (subject_label, session_label, scan, study)
                            )
                            continue
                        if existing._file_uris:
                            skipped_scans += 1
                            continue
                        # Re-fetch missing file URIs
                        self._refetch_missing_file_uris(existing, scan)
                        skipped_scans += 1
                        continue

                    scan_tasks.append((subject_label, session_label, scan, study))

        return scan_tasks, total_sessions, total_scans, skipped_scans

    def discover_xnat(
        self,
        project: Any,
        interactive: bool = True,
        refresh: bool = None,
        read_dicom: bool = False,
        parallel: bool = None,
        max_workers: int = 8,
    ) -> None:
        """Discover DICOM series from an XNAT project.

        Uses XNAT hierarchy (subject/session/scan) instead of DICOM headers.

        Args:
            project: xnatpy project object
            interactive: If True, show progress in Jupyter
            refresh: If True, clear existing data and re-discover everything.
                     If False, only add new subjects/sessions/scans (incremental).
                     If None (default), auto-detect: refresh only if no data loaded.
            read_dicom: If True, read first DICOM file to get accurate metadata.
                       If False (default), use only XNAT metadata (much faster).
            parallel: If True, use parallel discovery for faster throughput.
                     If None, auto-enable for >10 subjects.
            max_workers: Number of parallel workers. Default: 8.

        Notes:
            Updates ``self.patients`` in place.
        """
        # Auto-detect refresh mode: skip full re-discovery if data already loaded
        if refresh is None:
            refresh = len(self.patients) == 0
            if not refresh:
                total = len(self.get_all_series())
                print(
                    f"Found {len(self.patients)} subjects, {total} series in saved state — running incremental update"
                )
        if refresh:
            self.patients = {}
        self._xnat_mode = True
        self._discovering = True

        # Store project info for OHIF links
        try:
            self._xnat_project_id = project.id
            if hasattr(project, "_session") and hasattr(
                project._session, "_original_uri"
            ):
                self._xnat_base_url = project._session._original_uri.rstrip("/")
            elif hasattr(project, "_session") and hasattr(project._session, "host"):
                self._xnat_base_url = project._session.host.rstrip("/")
        except Exception:
            pass

        # Store session reference for file access
        try:
            self._xnat_session = project._session
        except Exception:
            pass

        # Set up progress display early so user sees feedback immediately
        progress_widget = None
        status_widget = None
        error_widget = None
        discovery_errors = []
        initial_error_count = len(self.get_errors()) if not refresh else 0
        mode_str = "Discovering" if refresh else "Updating"

        def _render_discovery_errors() -> str:
            total_err = len(self.get_errors())
            new_err = len(discovery_errors)
            summary = (
                "<div class='qc-card' style='font-size:11px;color:#4f5f78;'>"
                f"<b>Total ERROR:</b> {total_err}"
                f"<span style='margin-left:10px;'><b>Pre-existing:</b> {initial_error_count}</span>"
                f"<span style='margin-left:10px;'><b>New in discovery:</b> {new_err}</span>"
            )
            if not discovery_errors:
                return summary + (
                    "<div style='margin-top:6px;color:#6a7890;'>Tracking discovery errors for this run...</div>"
                    "</div>"
                )

            rows = []
            for label, detail in discovery_errors[-6:]:
                rows.append(
                    "<div style='margin-top:6px;font-size:11px;color:#42526b;'>"
                    "<span style='display:inline-block;min-width:56px;text-align:center;padding:1px 6px;border-radius:999px;"
                    "background:#6c757d;color:#fff;font-weight:700;'>ERROR</span> "
                    f"<span style='font-weight:600;'>{html.escape(label[:52])}</span> "
                    f"<span style='color:#6a7890;'>- {html.escape(detail[:120])}</span>"
                    "</div>"
                )
            return summary + (
                f"<div style='margin-top:4px;max-height:140px;overflow-y:auto;padding-right:2px;'>{''.join(rows)}</div>"
                "</div>"
            )

        def _record_discovery_error(
            subject_label: str, session_label: str, scan_id: str, message: str
        ) -> None:
            label = f"{subject_label} / {session_label} / scan {scan_id}"
            discovery_errors.append((label, message or "Unknown error"))
            if error_widget is not None:
                error_widget.value = _render_discovery_errors()

        if interactive:
            try:
                import ipywidgets as widgets
                from IPython.display import display

                progress_widget = widgets.FloatProgress(
                    value=0,
                    min=0,
                    max=100,
                    description=f"{mode_str}:",
                    bar_style="info",
                    style={"bar_color": "#17a2b8", "description_width": "80px"},
                    layout=widgets.Layout(width="95%"),
                )
                status_widget = widgets.HTML(
                    '<div class="qc-card qc-mono">Fetching subject list from XNAT...</div>'
                )
                error_widget = widgets.HTML(
                    '<div class="qc-card" style="font-size:11px;color:#6a7890;">Tracking discovery errors for this run...</div>'
                )
                theme_widget = widgets.HTML("""
                    <style>
                        .qc-card {
                            background: linear-gradient(180deg, #fbfdff 0%, #f5f8fc 100%);
                            border: 1px solid #dce5f0;
                            border-radius: 8px;
                            padding: 8px 10px;
                            box-shadow: 0 1px 0 rgba(19, 35, 66, 0.03);
                        }
                        .qc-mono {
                            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
                        }
                        .qc-section-title {
                            margin: 10px 0 6px;
                            font-size: 12px;
                            font-weight: 700;
                            letter-spacing: 0.02em;
                            color: #2f3a4a;
                        }
                    </style>
                """)
                display(
                    widgets.VBox(
                        [
                            theme_widget,
                            progress_widget,
                            status_widget,
                            widgets.HTML(
                                '<div class="qc-section-title">Error Review</div>'
                            ),
                            error_widget,
                        ],
                        layout=widgets.Layout(width="95%"),
                    )
                )
            except Exception:
                interactive = False

        subjects = list(project.subjects.values())
        total_subjects = len(subjects)

        # Auto-enable parallelism for larger projects
        if parallel is None:
            parallel = total_subjects > 2

        # Phase 1: Collect all scan tasks (parallel fetch from XNAT)
        def _phase1_progress(subjects_done, subjects_total, subject_label):
            if progress_widget:
                progress_widget.value = (subjects_done / subjects_total) * 50
                status_widget.value = (
                    f'<div class="qc-card qc-mono">'
                    f"Collecting scan list... {subjects_done}/{subjects_total} subjects<br>"
                    f'<span style="color:#888">{subject_label}</span></div>'
                )

        scan_tasks, total_sessions, total_scans, skipped_scans = (
            self._collect_xnat_scan_tasks(
                subjects,
                refresh,
                progress_callback=_phase1_progress,
                max_workers=max_workers,
            )
        )
        total_scan_tasks = len(scan_tasks)
        new_scans = 0

        if progress_widget:
            progress_widget.value = 50
            status_widget.value = (
                f'<div class="qc-card qc-mono">'
                f"Processing 0/{total_scan_tasks} scans | {total_sessions} sessions | {total_subjects} subjects</div>"
            )
            if error_widget:
                error_widget.value = _render_discovery_errors()

        # Phase 2: Process scans (parallel or sequential)
        if parallel and max_workers > 1 and total_scan_tasks > 0:
            new_scans = self._process_xnat_scans_parallel(
                scan_tasks,
                total_scan_tasks,
                total_sessions,
                total_subjects,
                max_workers,
                progress_widget,
                status_widget,
                error_callback=_record_discovery_error,
            )
        elif total_scan_tasks > 0:
            new_scans = self._process_xnat_scans_sequential(
                scan_tasks,
                total_scan_tasks,
                total_sessions,
                total_subjects,
                progress_widget,
                status_widget,
                error_callback=_record_discovery_error,
            )

        # Final progress update
        if progress_widget:
            progress_widget.value = 100
            progress_widget.bar_style = "success"
            if refresh:
                status_widget.value = (
                    f'<div class="qc-card qc-mono">'
                    f"<b>✓ Discovery complete:</b> {total_subjects} subjects | "
                    f"{total_sessions} sessions | {total_scans} scans</div>"
                )
            else:
                status_widget.value = (
                    f'<div class="qc-card qc-mono">'
                    f"<b>✓ Update complete:</b> {new_scans} new scans | "
                    f"{skipped_scans} existing (skipped)</div>"
                )
            if error_widget:
                error_widget.value = _render_discovery_errors()
        else:
            print(
                f"Discovered {total_subjects} subjects, {total_sessions} sessions, {total_scans} scans"
            )

        # Flag series missing file URIs
        for series in self.get_all_series():
            if series._xnat_files and not series._file_uris and not series.error:
                series.error = (
                    "No file paths available (scan may have been removed from XNAT)"
                )

        # Discovery complete — allow orphan cleanup on next save
        self._discovering = False

        # Auto-save after discovery (this save will run orphan cleanup)
        if self.data_dir:
            self.save()

    def _process_xnat_scans_sequential(
        self,
        scan_tasks: List,
        total_tasks: int,
        total_sessions: int,
        total_subjects: int,
        progress_widget,
        status_widget,
        error_callback: Optional[Callable[[str, str, str, str], None]] = None,
    ) -> int:
        """Process XNAT scans sequentially.

        Args:
            scan_tasks: List of (subject_label, session_label, scan, study) tuples
            total_tasks: Total number of tasks for progress calculation
            total_sessions: Total sessions discovered (for display)
            total_subjects: Total subjects discovered (for display)
            progress_widget: Jupyter progress widget or None
            status_widget: Jupyter status HTML widget or None
            error_callback: Optional callback(subject_label, session_label, scan_id, error_message)

        Returns:
            Number of new scans processed
        """
        new_scans = 0

        for i, (subject_label, session_label, scan, study) in enumerate(scan_tasks):
            scan_id = scan.id

            # Update progress
            if progress_widget:
                progress_pct = 50 + ((i + 1) / total_tasks) * 50
                progress_widget.value = progress_pct
                status_widget.value = (
                    f'<div class="qc-card qc-mono">'
                    f"Processing {i + 1}/{total_tasks} scans | {total_sessions} sessions | {total_subjects} subjects<br>"
                    f'<span style="color:#888">{subject_label} / {session_label} / scan {scan_id}</span></div>'
                )

            # Create series and add to study
            series = self._create_series_from_xnat_scan(
                scan, subject_label, session_label
            )
            study.series[scan_id] = series
            new_scans += 1
            if series.error and error_callback:
                error_callback(subject_label, session_label, scan_id, series.error)

            # Periodic save (every 10 scans)
            if self.data_dir and new_scans % 10 == 0:
                self.save()

        # Final save
        if self.data_dir and new_scans > 0:
            self.save()

        return new_scans

    def _process_xnat_scans_parallel(
        self,
        scan_tasks: List,
        total_tasks: int,
        total_sessions: int,
        total_subjects: int,
        max_workers: int,
        progress_widget,
        status_widget,
        error_callback: Optional[Callable[[str, str, str, str], None]] = None,
    ) -> int:
        """Process XNAT scans in parallel using ThreadPoolExecutor.

        Args:
            scan_tasks: List of (subject_label, session_label, scan, study) tuples
            total_tasks: Total number of tasks for progress calculation
            total_sessions: Total sessions discovered (for display)
            total_subjects: Total subjects discovered (for display)
            max_workers: Number of parallel workers
            progress_widget: Jupyter progress widget or None
            status_widget: Jupyter status HTML widget or None
            error_callback: Optional callback(subject_label, session_label, scan_id, error_message)

        Returns:
            Number of new scans processed
        """
        from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

        new_scans = 0
        failed_scans = 0
        processed_count = [0]

        def process_one(task):
            """Process a single scan task."""
            subject_label, session_label, scan, study = task
            series = self._create_series_from_xnat_scan(
                scan, subject_label, session_label
            )
            return subject_label, session_label, scan.id, series, study

        def update_progress(subj_label, sess_label, scan_id):
            """Update progress widgets."""
            if progress_widget:
                progress_pct = 50 + (processed_count[0] / total_tasks) * 50
                progress_widget.value = progress_pct
                status_widget.value = (
                    f'<div class="qc-card qc-mono">'
                    f"Processing {processed_count[0]}/{total_tasks} scans | {total_sessions} sessions | "
                    f"{total_subjects} subjects<br>"
                    f'<span style="color:#888">{subj_label} / {sess_label} / scan {scan_id}</span></div>'
                )

        # Save interval scales with dataset size
        save_interval = 50 if total_tasks > 1000 else 10
        last_save_i = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_one, task): task for task in scan_tasks}
            pending = set(futures.keys())

            while pending:
                done, pending = wait(pending, timeout=0.3, return_when=FIRST_COMPLETED)

                for future in done:
                    task = futures[future]
                    subj_label, sess_label, scan, study = task
                    scan_id = scan.id
                    try:
                        subj_label, sess_label, scan_id, series, study = future.result()
                        study.series[scan_id] = series
                        new_scans += 1
                        processed_count[0] += 1
                        update_progress(subj_label, sess_label, scan_id)
                        if series.error and error_callback:
                            error_callback(
                                subj_label, sess_label, scan_id, series.error
                            )
                    except Exception as e:
                        # Preserve failed scans with an explicit error entry so
                        # discovery is complete and failures are visible/retryable.
                        failed_series = SeriesInfo(
                            uid="",
                            series_number=scan_id,
                            description=getattr(scan, "series_description", "") or "",
                            modality=getattr(scan, "modality", "??") or "??",
                            xnat_scan_id=scan_id,
                            _xnat_files=True,
                            _xnat_subject_label=subj_label,
                            _xnat_session_label=sess_label,
                            error=f"Failed to process scan: {e}",
                        )
                        study.series[scan_id] = failed_series
                        failed_scans += 1
                        processed_count[0] += 1
                        update_progress(subj_label, sess_label, scan_id)
                        if error_callback:
                            error_callback(
                                subj_label, sess_label, scan_id, failed_series.error
                            )

                # Periodic save
                i = processed_count[0]
                if self.data_dir and i >= last_save_i + save_interval:
                    try:
                        self.save()
                    except Exception as e:
                        print(f"\nWarning: save failed ({e}), continuing...")
                    last_save_i = i

        # Final save
        if self.data_dir and new_scans > 0:
            self.save()

        if failed_scans > 0:
            print(f"Warning: {failed_scans} scan(s) failed during parallel discovery")

        return new_scans

    def _add_file_to_hierarchy(self, ds: pydicom.Dataset, dcm_file: Path) -> None:
        """Add a DICOM file to the hierarchy."""
        patient_id = str(getattr(ds, "PatientID", "Unknown"))
        patient_name = str(getattr(ds, "PatientName", ""))
        study_uid = getattr(ds, "StudyInstanceUID", "unknown")
        study_date = getattr(ds, "StudyDate", "Unknown")
        study_desc = getattr(ds, "StudyDescription", "")
        series_uid = getattr(ds, "SeriesInstanceUID", "unknown")
        series_num = getattr(ds, "SeriesNumber", 0) or 0
        series_desc = getattr(ds, "SeriesDescription", "Unknown")
        modality = getattr(ds, "Modality", "??")

        if patient_id not in self.patients:
            self.patients[patient_id] = PatientInfo(patient_id, patient_name)
        patient = self.patients[patient_id]

        if study_uid not in patient.studies:
            patient.studies[study_uid] = StudyInfo(study_uid, study_date, study_desc)
        study = patient.studies[study_uid]

        if series_uid not in study.series:
            # Get transfer syntax and implementation from file_meta
            transfer_syntax = None
            implementation = None
            if hasattr(ds, "file_meta"):
                transfer_syntax = str(getattr(ds.file_meta, "TransferSyntaxUID", None))
                implementation = getattr(
                    ds.file_meta, "ImplementationVersionName", None
                )
            study.series[series_uid] = SeriesInfo(
                series_uid,
                series_num,
                series_desc,
                modality,
                transfer_syntax=transfer_syntax,
                implementation=implementation,
            )
        study.series[series_uid].files.append(dcm_file)

    def get_all_series(self) -> List[SeriesInfo]:
        """Get flat list of all series."""
        return [
            series
            for patient in self.patients.values()
            for study in patient.studies.values()
            for series in study.series.values()
        ]

    def get_summary(self) -> Dict[str, int]:
        """Get count summary by status."""
        counts = {
            "PASS": 0,
            "WARNING": 0,
            "FAIL": 0,
            "ERROR": 0,
            "PENDING": 0,
            "DERIVED": 0,
            "NOTE": 0,
        }
        for series in self.get_all_series():
            status = series.qc_status
            if status in counts:
                counts[status] += 1
        return counts

    def get_errors(self) -> List[SeriesInfo]:
        """Get list of series with errors."""
        return [s for s in self.get_all_series() if s.error]

    def show_errors(self):
        """Print details of all series with errors."""
        errors = self.get_errors()
        if not errors:
            print("No errors found")
            return

        print(f"{len(errors)} series with errors:\n")
        for s in errors:
            # Find patient/study for context
            for pid, patient in self.patients.items():
                for sid, study in patient.studies.items():
                    if s in study.series.values():
                        print(f"  {pid} / {study.label} / {s.label}")
                        print(f"    Error: {s.error}")
                        print(
                            f"    Has URIs: {bool(s._file_uris)}, Has paths: {bool(s._file_paths and any(s._file_paths))}"
                        )
                        print()

    def clear_errors(self, refetch_uris: bool = True):
        """Clear errors from all series so they can be retried.

        Args:
            refetch_uris: If True and XNAT session available, try to re-fetch file URIs

        Returns:
            Number of errors cleared
        """
        errors = self.get_errors()
        if not errors:
            print("No errors to clear")
            return 0

        cleared = 0
        for s in errors:
            s.error = None
            # Also clear file URIs so they get re-fetched
            if refetch_uris:
                s._file_uris = []
                s._file_paths = []
                s.files = []
            cleared += 1

        print(
            f"Cleared {cleared} errors. Run discover_xnat(refresh=False) to re-fetch file paths, then process_all_interactive()"
        )
        return cleared

    def reprocess_series(self, series: SeriesInfo, silent: bool = False) -> None:
        """Force reprocess a specific series, clearing any cached results.

        Args:
            series: SeriesInfo to reprocess
            silent: If True, don't print status messages

        Example:
            # Find and reprocess fMRI series
            fmri = [s for s in qc.get_all_series() if 'fMRI' in s.description][0]
            qc.reprocess_series(fmri)
            qc.display()  # Refresh display to see changes
        """
        # Clear ALL cached results
        series.thumbnail = None
        series.qc_report = None
        series.error = None
        series.volume = None
        series.files = []  # Force re-fetch so 4D filtering applies

        # Clear derived flags (in case incorrectly set)
        series.is_derived = False
        series.derived_info = None
        series.referenced_series_uid = None

        # Reprocess
        self.process_series(series, keep_volume=not silent)

        if not silent:
            # Show result
            status = series.qc_status
            if series.error:
                print(f"Error: {series.error}")
            else:
                print(f"Reprocessed: {series.description} -> {status}")
                if series.qc_report:
                    for r in series.qc_report.results:
                        if r.status != "PASS":
                            print(f"  {r.status}: {r.check_name} - {r.message}")
                # Show volume info
                if series.volume:
                    v = series.volume
                    print(
                        f"  Volume: {v.shape[0]} slices, {v.pixel_spacing[0]:.2f}x{v.pixel_spacing[1]:.2f}mm pixels, {v.slice_thickness:.2f}mm slice thickness"
                    )

            print("\nCall qc.display() to refresh the view")

    def reprocess_4d(self, save_interval: int = 5) -> int:
        """Reprocess all series that have 4D/NOTE status.

        Useful after code changes to 4D detection or display.

        Args:
            save_interval: Save progress every N series (0 to disable)

        Returns:
            Number of series reprocessed
        """
        # Find all 4D series (status NOTE, or 4D check returned NOTE)
        series_4d = []
        for s in self.get_all_series():
            is_4d = False
            if s.qc_status == "NOTE":
                is_4d = True
            elif s.qc_report:
                # Look for 4D Data check with NOTE status specifically
                for r in s.qc_report.results:
                    if r.check_name == "4D Data" and r.status == "NOTE":
                        is_4d = True
                        break
            if is_4d:
                series_4d.append(s)

        if not series_4d:
            print("No 4D series found to reprocess")
            return 0

        print(f"Reprocessing {len(series_4d)} 4D series...")

        for i, series in enumerate(series_4d):
            print(f"  [{i + 1}/{len(series_4d)}] {series.description[:50]}...", end=" ")
            self.reprocess_series(series, silent=True)
            print(f"-> {series.qc_status}")

            if save_interval > 0 and (i + 1) % save_interval == 0 and self.data_dir:
                self.save()

        # Final save
        if self.data_dir:
            self.save()

        print(f"\nDone! Reprocessed {len(series_4d)} series.")
        print("Call qc.display() to refresh the view")
        return len(series_4d)

    def reprocess_by_description(self, pattern: str, save_interval: int = 5) -> int:
        """Reprocess all series matching a description pattern.

        Args:
            pattern: Case-insensitive substring to match in series description
            save_interval: Save progress every N series (0 to disable)

        Returns:
            Number of series reprocessed

        Example:
            qc.reprocess_by_description('fMRI')
            qc.reprocess_by_description('DTI')
        """
        pattern_lower = pattern.lower()
        matching = [
            s
            for s in self.get_all_series()
            if pattern_lower in (s.description or "").lower()
        ]

        if not matching:
            print(f"No series found matching '{pattern}'")
            return 0

        print(f"Reprocessing {len(matching)} series matching '{pattern}'...")

        for i, series in enumerate(matching):
            print(f"  [{i + 1}/{len(matching)}] {series.description[:50]}...", end=" ")
            self.reprocess_series(series, silent=True)
            print(f"-> {series.qc_status}")

            if save_interval > 0 and (i + 1) % save_interval == 0 and self.data_dir:
                self.save()

        # Final save
        if self.data_dir:
            self.save()

        print(f"\nDone! Reprocessed {len(matching)} series.")
        print("Call qc.display() to refresh the view")
        return len(matching)

    # JPEG-2000 transfer syntaxes
    JPEG2000_UIDS = {
        "1.2.840.10008.1.2.4.90",  # JPEG 2000 Lossless
        "1.2.840.10008.1.2.4.91",  # JPEG 2000 Lossy
    }

    def _get_xnat_files(self, series: SeriesInfo) -> List[Any]:
        """Get XNAT files for a series.

        Loading priority:
        1. If files already loaded, return them
        2. If we have stored paths/URIs, restore from those (uses local paths if available)
        3. If we have a scan object, load files and store paths/URIs for next time
        """
        if series.files:
            return series.files

        # Try to restore from stored paths/URIs (after load from save file)
        if series._file_uris:
            try:
                # Pair URIs with local paths (if available)
                paths = (
                    series._file_paths
                    if series._file_paths
                    else [None] * len(series._file_uris)
                )
                series.files = [
                    XnatFileHandle(self._xnat_session, uri, local_path)
                    for uri, local_path in zip(series._file_uris, paths)
                ]
                return series.files
            except Exception:
                pass  # Fall through to scan object if restore fails

        # Load from scan object (fallback if file listing failed during discovery)
        if series._scan_obj is None:
            return []

        try:
            all_files = []
            for res in series._scan_obj.resources.values():
                res_label = getattr(res, "label", "").upper()
                if res_label == "SNAPSHOTS":
                    continue
                all_files.extend(res.files.values())
            series.files = all_files
            # Store URIs and local paths for save/restore
            series._file_uris = [f.uri for f in all_files]
            series._file_paths = [
                getattr(f, "data_path", None) or "" for f in all_files
            ]
        except Exception:
            series.files = []
        return series.files

    def process_series(self, series: SeriesInfo, keep_volume: bool = False) -> None:
        """Load volume, run QC, and generate thumbnail for a single series.

        Args:
            series: SeriesInfo to process
            keep_volume: If False (default), release volume after processing to save memory
        """
        # Handle derived modalities (RTStruct, SEG, etc.) - skip geometry QC
        if series.modality in self.DERIVED_MODALITIES:
            self._process_derived_series(series)
            return

        try:
            if series._xnat_files:
                files = self._get_xnat_files(series)
                if not files:
                    series.error = "No DICOM files found"
                    return

                # Check if this is a non-image SOP class (e.g., Raw Data Storage)
                try:
                    with files[0].open() as f:
                        ds = pydicom.dcmread(f, stop_before_pixels=True)
                        sop_class = getattr(
                            ds.file_meta, "MediaStorageSOPClassUID", None
                        )
                        if sop_class and str(sop_class) in self.NON_IMAGE_SOP_CLASSES:
                            series.is_derived = True
                            series.derived_info = f"Non-image DICOM: {sop_class.name if hasattr(sop_class, 'name') else 'Raw Data'}"
                            return
                except Exception:
                    pass  # Continue with normal loading attempt

                volume = self.loader.load_from_xnat(files)
            else:
                # For local files, use the parent directory of the series files
                # (SimpleITK's GetGDCMSeriesIDs only searches one directory level)
                if series.files:
                    series_dir = Path(series.files[0]).parent
                elif series._file_paths and any(series._file_paths):
                    # Use stored file paths (preserved during save/reprocess)
                    first_valid_path = next((p for p in series._file_paths if p), None)
                    series_dir = (
                        Path(first_valid_path).parent
                        if first_valid_path
                        else self.data_dir
                    )
                else:
                    series_dir = self.data_dir
                volume = self.loader.load_from_path_simpleitk(
                    series_dir, series_uid=series.uid
                )

            # Update series UID from volume if not set (XNAT mode initializes to '')
            if not series.uid:
                if volume.series_instance_uid:
                    series.uid = volume.series_instance_uid
                elif series._file_uris:
                    # Fallback: use first file URI as unique identifier (contains full XNAT path)
                    series.uid = hashlib.sha256(
                        series._file_uris[0].encode()
                    ).hexdigest()

            qc = GeometryQC(volume)
            series.qc_report = qc.run_all_checks(series.description)

            # Check for problematic JPEG-2000 encoding
            self._check_encoding(series)

            # Check for missing temporal metadata in dynamic series
            self._check_temporal_metadata(series)

            # Generate thumbnail - use disk cache if available (saves memory)
            snapshot_gen = SnapshotGenerator(volume)
            if self._thumb_cache:
                # Use human-readable paths for XNAT scans
                if (
                    series._xnat_subject_label
                    and series._xnat_session_label
                    and series.xnat_scan_id
                ):
                    thumb_path = self._thumb_cache.get_path_for_xnat(
                        series._xnat_subject_label,
                        series._xnat_session_label,
                        series.xnat_scan_id,
                    )
                    snapshot_gen.create_tripane_thumbnail_file(thumb_path)
                    series._thumbnail_path = self._thumb_cache.get_relative_path_xnat(
                        series._xnat_subject_label,
                        series._xnat_session_label,
                        series.xnat_scan_id,
                    )
                else:
                    # Fallback to hash-based path for local data
                    thumb_path = self._thumb_cache.get_path_for_series(series.uid)
                    snapshot_gen.create_tripane_thumbnail_file(thumb_path)
                    series._thumbnail_path = self._thumb_cache.get_relative_path(
                        series.uid
                    )
                series.thumbnail = None  # Don't store base64 in memory
            else:
                series.thumbnail = snapshot_gen.create_tripane_thumbnail()

            # Only keep volume if requested (saves memory for batch processing)
            if keep_volume:
                series.volume = volume
        except Exception as e:
            series.error = str(e)

    def _process_derived_series(self, series: SeriesInfo) -> None:
        """Process derived modalities (RTStruct, SEG, etc.) - extract references, skip geometry QC."""
        series.is_derived = True

        # Get files (lazy load for XNAT)
        files = self._get_xnat_files(series) if series._xnat_files else series.files
        if not files:
            return

        try:
            # Read first DICOM file to extract reference info
            if series._xnat_files:
                with files[0].open() as f:
                    ds = pydicom.dcmread(f, stop_before_pixels=True)
            else:
                ds = pydicom.dcmread(str(files[0]), stop_before_pixels=True)

            # Extract referenced series UID based on modality
            ref_uid = self._extract_referenced_series(ds, series.modality)
            if ref_uid:
                series.referenced_series_uid = ref_uid

            # Build informative description
            series.derived_info = self._build_derived_info(ds, series.modality)

        except Exception:
            pass  # Non-critical - just won't have reference info

    def _extract_referenced_series(
        self, ds: pydicom.Dataset, modality: str
    ) -> Optional[str]:
        """Extract the referenced series UID from a derived DICOM object."""
        try:
            # RTStruct: deeply nested reference structure
            if modality == "RTSTRUCT":
                for fof_ref in getattr(ds, "ReferencedFrameOfReferenceSequence", []):
                    for study_ref in getattr(fof_ref, "RTReferencedStudySequence", []):
                        for series_ref in getattr(
                            study_ref, "RTReferencedSeriesSequence", []
                        ):
                            return str(getattr(series_ref, "SeriesInstanceUID", ""))

            # SEG, PR, KO, SR: simpler ReferencedSeriesSequence
            for series_ref in getattr(ds, "ReferencedSeriesSequence", []):
                return str(getattr(series_ref, "SeriesInstanceUID", ""))

            # Alternative: CurrentRequestedProcedureEvidenceSequence (KO, SR)
            for evidence in getattr(
                ds, "CurrentRequestedProcedureEvidenceSequence", []
            ):
                for series_ref in getattr(evidence, "ReferencedSeriesSequence", []):
                    return str(getattr(series_ref, "SeriesInstanceUID", ""))

        except Exception:
            pass
        return None

    def _build_derived_info(self, ds: pydicom.Dataset, modality: str) -> str:
        """Build human-readable info about a derived DICOM object."""
        info_parts = []

        if modality == "RTSTRUCT":
            # Count ROIs
            roi_seq = getattr(ds, "StructureSetROISequence", [])
            if roi_seq:
                info_parts.append(f"{len(roi_seq)} ROIs")
                # List first few ROI names
                roi_names = [getattr(roi, "ROIName", "") for roi in roi_seq[:5]]
                roi_names = [n for n in roi_names if n]
                if roi_names:
                    info_parts.append(", ".join(roi_names[:3]))
                    if len(roi_names) > 3:
                        info_parts[-1] += "..."

        elif modality == "SEG":
            # Count segments
            seg_seq = getattr(ds, "SegmentSequence", [])
            if seg_seq:
                info_parts.append(f"{len(seg_seq)} segments")
                seg_labels = [getattr(seg, "SegmentLabel", "") for seg in seg_seq[:3]]
                seg_labels = [label for label in seg_labels if label]
                if seg_labels:
                    info_parts.append(", ".join(seg_labels))

        elif modality == "SR":
            # Content description
            content_desc = getattr(ds, "ContentDescription", "")
            if content_desc:
                info_parts.append(content_desc[:50])

        elif modality == "PR":
            info_parts.append("Presentation State")

        elif modality == "KO":
            info_parts.append("Key Object Selection")

        return " | ".join(info_parts) if info_parts else f"{modality} data"

    def _read_dicom_file(self, file_obj, stop_before_pixels=False):
        """Read a DICOM file, handling both local paths and XNAT file objects."""
        if hasattr(file_obj, "open"):
            with file_obj.open() as f:
                return pydicom.dcmread(f, stop_before_pixels=stop_before_pixels)
        else:
            return pydicom.dcmread(str(file_obj), stop_before_pixels=stop_before_pixels)

    def _check_encoding(self, series: SeriesInfo) -> None:
        """Flag JPEG-2000 transfer syntax as potentially problematic in OHIF."""
        from dicom_qc.core.geometry import QCResult

        if not series.qc_report or not series.transfer_syntax:
            return

        if series.transfer_syntax in self.JPEG2000_UIDS:
            result = QCResult(
                status="NOTE",
                check_name="JPEG-2000 Encoding",
                message="JPEG-2000 encoded — may render blurry in OHIF viewer",
                details={"transfer_syntax": series.transfer_syntax},
            )
            series.qc_report.results.append(result)

    def _check_temporal_metadata(self, series: SeriesInfo) -> None:
        """Check for missing temporal metadata in dynamic/perfusion series."""
        from dicom_qc.core.geometry import QCResult

        if not series.qc_report or not series.files:
            return

        # Only check series that look like dynamic/temporal acquisitions
        desc_lower = (series.description or "").lower()
        is_dynamic = any(
            p in desc_lower
            for p in ["perf", "dsc", "dce", "dynamic", "cine", "fmri", "bold"]
        )

        if not is_dynamic:
            return

        try:
            # Check first few files for temporal tags
            temporal_tags_found = {
                "TriggerTime": False,
                "AcquisitionTime": False,
                "TemporalPositionIdentifier": False,
                "NumberOfTemporalPositions": False,
            }

            files_to_check = series.files[: min(5, len(series.files))]
            for dcm_file in files_to_check:
                ds = self._read_dicom_file(dcm_file, stop_before_pixels=True)

                if hasattr(ds, "TriggerTime") and ds.TriggerTime is not None:
                    temporal_tags_found["TriggerTime"] = True
                if hasattr(ds, "AcquisitionTime") and ds.AcquisitionTime is not None:
                    temporal_tags_found["AcquisitionTime"] = True
                if (
                    hasattr(ds, "TemporalPositionIdentifier")
                    and ds.TemporalPositionIdentifier is not None
                ):
                    temporal_tags_found["TemporalPositionIdentifier"] = True
                if (
                    hasattr(ds, "NumberOfTemporalPositions")
                    and ds.NumberOfTemporalPositions is not None
                ):
                    temporal_tags_found["NumberOfTemporalPositions"] = True

            # Check if critical temporal tags are missing
            missing_tags = [
                tag for tag, found in temporal_tags_found.items() if not found
            ]

            if missing_tags:
                result = QCResult(
                    status="WARNING",
                    check_name="Temporal Metadata",
                    message=f"Dynamic series missing temporal tags: {', '.join(missing_tags)}",
                    details={
                        "series_description": series.description,
                        "missing_tags": missing_tags,
                        "found_tags": [
                            tag for tag, found in temporal_tags_found.items() if found
                        ],
                        "note": "Missing temporal metadata may cause viewer compatibility issues with ITK-SNAP, 3D Slicer",
                    },
                )
                series.qc_report.results.append(result)

                if series.qc_report.overall_status == "PASS":
                    series.qc_report.overall_status = "WARNING"

        except Exception:
            pass  # Skip check if we can't parse

    def _process_all_simple(
        self, progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[str, int]:
        """Process all series with optional progress callback (no UI).

        For non-Jupyter environments. Use process_all() for interactive use.

        Args:
            progress_callback: Optional callback(current, total, label) for progress updates

        Returns:
            Summary counts by status
        """
        all_series = self.get_all_series()
        total = len(all_series)

        for i, series in enumerate(all_series):
            if progress_callback:
                progress_callback(i, total, series.label[:40])
            self.process_series(series)

        if progress_callback:
            progress_callback(total, total, "Done!")

        return self.get_summary()

    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable time string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
        else:
            return f"{seconds // 3600:.0f}h {(seconds % 3600) // 60:.0f}m"

    def _format_status_counts(
        self, counts: Dict[str, int], include_optional: bool = True
    ) -> str:
        """Format status counts as colored HTML spans.

        Args:
            counts: Dict of status -> count
            include_optional: If True, include NOTE and DERIVED if non-zero

        Returns:
            HTML string with colored status counts
        """
        c = self.STATUS_COLORS
        parts = [
            f"<span style='color:{c['PASS']}'>✓{counts['PASS']}</span>",
            f"<span style='color:{c['WARNING']}'>⚠{counts['WARNING']}</span>",
            f"<span style='color:{c['FAIL']}'>✗{counts['FAIL']}</span>",
            f"<span style='color:{c['ERROR']}'>⊘{counts['ERROR']}</span>",
        ]
        if include_optional:
            if counts.get("NOTE", 0) > 0:
                parts.append(
                    f"<span style='color:{c['NOTE']}'>ℹ{counts['NOTE']}</span>"
                )
            if counts.get("DERIVED", 0) > 0:
                parts.append(
                    f"<span style='color:{c['DERIVED']}'>◇{counts['DERIVED']}</span>"
                )
        return " ".join(parts)

    def _track_status_sample(
        self,
        series: "SeriesInfo",
        status_samples: Dict[str, "SeriesInfo"],
        thumb_cache: Dict[str, str],
    ) -> bool:
        """Track one sample series per status type for representative display.

        Keeps one example of each status type (PASS, WARNING, FAIL, ERROR, DERIVED)
        to show users what each status looks like.

        Args:
            series: Just-processed series
            status_samples: Dict of status -> series (modified in place)
            thumb_cache: Cache of uid -> thumbnail base64 (modified in place)

        Returns:
            True if this is a new status type being added
        """
        status = series.qc_status
        is_new_status = status not in status_samples

        # Get thumbnail data
        thumb_b64 = None
        if series._thumbnail_path and self._thumb_cache:
            thumb_b64 = self._thumb_cache.get_thumbnail_base64(series._thumbnail_path)
        elif series.thumbnail:
            thumb_b64 = series.thumbnail

        # Always update the sample for this status (shows most recent example)
        # Remove old cache entry if replacing
        if status in status_samples:
            old_series = status_samples[status]
            thumb_cache.pop(old_series.uid, None)

        status_samples[status] = series
        if thumb_b64:
            thumb_cache[series.uid] = thumb_b64

        return is_new_status

    def _build_series_context_map(self) -> Dict:
        """Build a map from id(series) -> (patient_id, study_key, patient, study).

        Used by _checkpoint_series to sync a single series to the DB without
        iterating the full hierarchy.
        """
        ctx = {}
        for patient_id, patient in self.patients.items():
            for study_key, study in patient.studies.items():
                for series_key, series in study.series.items():
                    ctx[id(series)] = (patient_id, study_key, patient, study)
        return ctx

    def _checkpoint_series(self, series: "SeriesInfo", ctx: Dict) -> None:
        """Sync a single series to the database after processing.

        Args:
            series: The just-processed series
            ctx: Context map from _build_series_context_map()
        """
        if not self._db:
            return

        info = ctx.get(id(series))
        if not info:
            return

        patient_id, study_key, patient, study = info
        self._sync_series_to_db(series, patient_id, study_key, patient, study)
        self._db.commit()

        # Free memory - volume is no longer needed after checkpointing
        series.volume = None

    def _process_series_with_error_capture(self, series: "SeriesInfo") -> None:
        """Process a series, capturing any exceptions as series.error.

        Args:
            series: Series to process
        """
        # Clear stale errors before each attempt so successful reprocessing
        # cannot remain stuck in ERROR status.
        series.error = None
        try:
            self.process_series(series)
        except Exception as e:
            series.error = str(e)

    def process_all(
        self,
        reprocess: bool = False,
        retry_errors: bool = False,
        max_workers: int = 4,
    ):
        """Process all series with live Jupyter progress display and grid view.

        Uses parallel processing with per-series DB checkpointing. Each series
        is synced to the database immediately after processing (no pickle).

        Args:
            reprocess: If False, skip already-processed series (those with thumbnails or is_derived)
            retry_errors: If True, retry series that previously had errors
            max_workers: Number of parallel workers (default: 4)

        Returns:
            Summary counts by status
        """
        from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
        import time
        import warnings
        import ipywidgets as widgets
        from IPython.display import display

        warnings.filterwarnings("ignore")

        try:
            import SimpleITK as sitk

            sitk.ProcessObject_SetGlobalWarningDisplay(False)
        except Exception:
            pass

        all_series = self.get_all_series()

        # Filter series to process
        if reprocess:
            to_process = all_series
            skipped = 0
        else:
            to_process = [
                s
                for s in all_series
                if not s.thumbnail
                and not s._thumbnail_path
                and not s.is_derived
                and not s.error
            ]
            skipped = len(all_series) - len(to_process)

        if retry_errors and not reprocess:
            error_series = [s for s in all_series if s.error]
            if error_series:
                for s in error_series:
                    s.error = None
                to_process.extend(error_series)
                print(f"Retrying {len(error_series)} series with errors")

        total = len(to_process)

        if total == 0:
            if skipped > 0:
                print(
                    f"All {skipped} series already processed. Use reprocess=True to re-run, or retry_errors=True to retry errors."
                )
            else:
                print("No series to process")
            return self.get_summary()

        # Build context map for per-series DB checkpointing
        series_ctx = self._build_series_context_map()

        # Create progress widgets
        progress_bar = widgets.IntProgress(
            value=0,
            min=0,
            max=total,
            description="Processing:",
            bar_style="info",
            style={"bar_color": "#007bff", "description_width": "80px"},
            layout=widgets.Layout(width="95%"),
        )
        status_html = widgets.HTML("")
        run_health_html = widgets.HTML("")
        error_review_html = widgets.HTML("")
        status_dist_html = widgets.HTML("")
        recent_thumbs_html = widgets.HTML("")
        theme_html = widgets.HTML("""
            <style>
                .qc-section-title {
                    margin: 10px 0 6px;
                    font-size: 12px;
                    font-weight: 700;
                    letter-spacing: 0.02em;
                    color: #2f3a4a;
                }
                .qc-section-sub {
                    font-size: 11px;
                    font-weight: 500;
                    color: #6f7c91;
                }
                .qc-card {
                    background: linear-gradient(180deg, #fbfdff 0%, #f5f8fc 100%);
                    border: 1px solid #dce5f0;
                    border-radius: 8px;
                    padding: 8px 10px;
                    box-shadow: 0 1px 0 rgba(19, 35, 66, 0.03);
                }
                .qc-mono {
                    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
                }
            </style>
        """)

        container = widgets.VBox(
            [
                theme_html,
                progress_bar,
                status_html,
                widgets.HTML('<div class="qc-section-title">Run Health</div>'),
                run_health_html,
                widgets.HTML('<div class="qc-section-title">Error Review</div>'),
                error_review_html,
                widgets.HTML('<div class="qc-section-title">Status Distribution</div>'),
                status_dist_html,
                widgets.HTML(
                    '<div class="qc-section-title">Sample by Status <span class="qc-section-sub">(one example per status type)</span></div>'
                ),
                recent_thumbs_html,
            ],
            layout=widgets.Layout(width="95%"),
        )
        display(container)

        # Status-organized sample tracking (one example per status type)
        status_samples: Dict[
            str, SeriesInfo
        ] = {}  # status -> most recent series with that status
        status_thumb_cache: Dict[str, str] = {}  # series uid -> thumbnail base64

        # Timing state
        start_time = time.time()
        recent_times = []
        completion_timestamps = []

        # OPTIMIZATION: Incremental status counts (avoid iterating all series)
        status_counts = {
            "PASS": 0,
            "WARNING": 0,
            "FAIL": 0,
            "ERROR": 0,
            "PENDING": 0,
            "DERIVED": 0,
            "NOTE": 0,
        }
        # Initialize with already-processed series
        for s in all_series:
            if s not in to_process:
                status_counts[s.qc_status] = status_counts.get(s.qc_status, 0) + 1
        preexisting_error_count = status_counts.get("ERROR", 0)

        # Throttle expensive UI sections independently
        samples_update_interval = 1 if total < 200 else (5 if total < 2000 else 20)
        last_samples_update = [0]
        outcome_history = []
        consecutive_errors = [0]
        recent_issues = []
        new_run_error_count = [0]

        def render_status_distribution():
            """Render compact status distribution bar."""
            c = self.STATUS_COLORS
            statuses = [
                "PASS",
                "WARNING",
                "FAIL",
                "ERROR",
                "NOTE",
                "DERIVED",
                "PENDING",
            ]
            total_count = sum(status_counts.get(s, 0) for s in statuses) or 1
            parts = []
            for s in statuses:
                count = status_counts.get(s, 0)
                if count <= 0:
                    continue
                pct = (count / total_count) * 100
                parts.append(
                    f"<div style='height:12px;background:{c.get(s, '#777')};width:{pct:.2f}%;' title='{s}: {count}'></div>"
                )
            return (
                "<div class='qc-card'>"
                "<div style='border:1px solid #d5dfeb;border-radius:999px;overflow:hidden;background:#eef3f9;'>"
                f"<div style='display:flex;width:100%;height:14px;'>{''.join(parts)}</div>"
                "</div>"
                "</div>"
            )

        def render_run_health():
            """Render recent outcome strip and early-stop guidance."""
            if not outcome_history:
                return "<div style='color:#666;font-size:11px;'>Collecting run health...</div>"

            c = self.STATUS_COLORS
            tail = outcome_history[-80:]
            blocks = "".join(
                f"<div style='width:7px;height:10px;background:{c.get(status, '#777')};border-radius:1px;'></div>"
                for status in tail
            )
            recent = outcome_history[-30:]
            recent_error_rate = sum(1 for status in recent if status == "ERROR") / max(
                1, len(recent)
            )

            guidance = (
                "<span style='color:#2f6f44;'>Stable</span>"
                if recent_error_rate < 0.5
                else "<span style='color:#b36b00;'>Watch</span>"
            )
            if (
                len(recent) >= 20
                and recent_error_rate >= 0.9
                and consecutive_errors[0] >= 12
            ):
                guidance = (
                    "<span style='color:#b00020;'><b>High failure streak</b> - "
                    "consider interrupting this run now.</span>"
                )

            return (
                f"<div class='qc-card' style='display:flex;align-items:center;gap:10px;flex-wrap:wrap;'>"
                f"<div style='display:flex;gap:2px;background:#fff;padding:5px 6px;border:1px solid #dce4ef;border-radius:6px;'>{blocks}</div>"
                f"<div style='font-size:11px;color:#4f5f78;'>Recent errors: {recent_error_rate:.0%} | "
                f"Consecutive errors: {consecutive_errors[0]} | {guidance}</div>"
                "</div>"
            )

        def render_status_badges():
            """Render readable status counts as badges."""
            c = self.STATUS_COLORS
            statuses = ["PASS", "WARNING", "FAIL", "ERROR", "NOTE", "DERIVED"]
            badges = []
            for status in statuses:
                count = status_counts.get(status, 0)
                if count <= 0:
                    continue
                fg = "#1c1c1c" if status == "WARNING" else "#ffffff"
                badges.append(
                    f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;background:{c.get(status, '#777')};"
                    f"color:{fg};font-size:11px;font-weight:700;'>{status} {count}</span>"
                )
            return (
                " ".join(badges)
                if badges
                else "<span style='color:#7a869c;'>No completed series yet</span>"
            )

        def _extract_issue_detail(series: "SeriesInfo") -> str:
            """Return a short error/fail detail for review panel."""
            if series.error:
                return series.error
            if series.qc_report:
                for result in series.qc_report.results:
                    if result.status in ("FAIL", "ERROR", "WARNING"):
                        if result.message:
                            return f"{result.check_name}: {result.message}"
                        return result.check_name
            return "Issue detected"

        def render_error_review():
            """Render recent ERROR entries for quick triage."""
            total_err_count = status_counts.get("ERROR", 0)
            summary = (
                f"<div style='font-size:11px;color:#4f5f78;'>"
                f"<b>Total ERROR:</b> {total_err_count}"
                f"<span style='margin-left:10px;'><b>Pre-existing:</b> {preexisting_error_count}</span>"
                f"<span style='margin-left:10px;'><b>New this run:</b> {new_run_error_count[0]}</span>"
                "</div>"
            )
            if not recent_issues:
                return (
                    "<div class='qc-card'>"
                    f"{summary}"
                    "<div style='margin-top:6px;color:#6a7890;font-size:11px;'>No new errors recorded in this run yet.</div>"
                    "</div>"
                )

            rows = []
            for status, label, detail in recent_issues[-6:]:
                color = self.STATUS_COLORS.get(status, "#777")
                fg = "#1c1c1c" if status == "WARNING" else "#fff"
                rows.append(
                    f"<div style='margin-top:6px;font-size:11px;color:#42526b;'>"
                    f"<span style='display:inline-block;min-width:56px;text-align:center;padding:1px 6px;border-radius:999px;"
                    f"background:{color};color:{fg};font-weight:700;'>{status}</span> "
                    f"<span style='font-weight:600;'>{html.escape(label[:42])}</span> "
                    f"<span style='color:#6a7890;'>- {html.escape(detail[:120])}</span>"
                    f"</div>"
                )
            return (
                "<div class='qc-card'>"
                f"{summary}"
                f"<div style='margin-top:4px;max-height:140px;overflow-y:auto;padding-right:2px;'>{''.join(rows)}</div>"
                "</div>"
            )

        def render_status_samples():
            """Render one sample thumbnail per status type."""
            if not status_samples:
                return '<div style="color:#666;font-size:12px;">Processing...</div>'

            c = self.STATUS_COLORS
            # Order statuses logically: PASS first, then WARNING, FAIL, ERROR, DERIVED
            status_order = ["PASS", "WARNING", "FAIL", "ERROR", "DERIVED"]
            html_parts = ['<div style="display:flex;flex-wrap:wrap;gap:10px;">']

            for status in status_order:
                if status not in status_samples:
                    continue

                series = status_samples[status]
                color = c.get(status, "#ccc")
                thumb_b64 = status_thumb_cache.get(series.uid)

                if thumb_b64:
                    img = f'<img src="data:image/jpeg;base64,{thumb_b64}" style="width:100%;display:block;">'
                elif series.is_derived:
                    img = f'<div style="height:84px;background:linear-gradient(135deg,#efe5fb 0%,#e7f0ff 100%);color:#7b2ca6;display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:700;letter-spacing:0.03em;">{series.modality}</div>'
                elif series.error:
                    img = f'<div style="height:84px;background:linear-gradient(135deg,#ffe9ea 0%,#fff5e5 100%);color:#b00020;display:flex;align-items:center;justify-content:center;font-size:11px;padding:8px;text-align:center;">{html.escape(series.error[:48])}</div>'
                else:
                    img = '<div style="height:84px;background:#edf2f8;color:#607089;display:flex;align-items:center;justify-content:center;font-size:16px;">?</div>'

                # Status badge with name
                status_label = status
                html_parts.append(f"""<div style="width:220px;border:2px solid {color};border-radius:10px;overflow:hidden;background:#f7fafc;box-shadow:0 2px 6px rgba(19,35,66,0.08);">
                    <div style="background:{color};color:{"#1c1c1c" if status == "WARNING" else "#fff"};padding:4px 8px;font-size:11px;font-weight:700;letter-spacing:0.02em;">{status_label}</div>
                    {img}
                    <div style="padding:6px 8px;background:#fff;color:#1f2d3d;font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                        {html.escape(series.label[:35])}
                    </div>
                </div>""")

            html_parts.append("</div>")
            return "".join(html_parts)

        def update_progress(
            i: int,
            series: "SeriesInfo",
            eta_str: str = None,
            force: bool = False,
            pending_count: int = None,
        ):
            """Update all progress displays."""
            elapsed = time.time() - start_time

            if eta_str is None:
                if i > 0 and elapsed > 0:
                    # Throughput-based ETA is accurate for parallel runs.
                    global_rate = i / elapsed
                    recent_rate = None
                    if len(completion_timestamps) >= 6:
                        window = completion_timestamps[-50:]
                        dt = window[-1] - window[0]
                        if dt > 0:
                            recent_rate = (len(window) - 1) / dt
                    rate = (
                        global_rate
                        if recent_rate is None
                        else (0.3 * global_rate + 0.7 * recent_rate)
                    )
                    rate = max(rate, 1e-6)
                    remaining = total - i
                    eta = remaining / rate

                    warmup = max(24, 4 * max_workers)
                    if i < warmup:
                        spread = 0.50
                    elif i < 3 * warmup:
                        spread = 0.30
                    else:
                        spread = 0.20

                    eta_lo = remaining / (rate * (1 + spread))
                    eta_hi = remaining / max(rate * (1 - spread), 1e-6)
                    eta_str = (
                        f"ETA: {self._format_time(eta)} "
                        f"({self._format_time(eta_lo)}-{self._format_time(eta_hi)})"
                    )
                else:
                    eta_str = "ETA: calculating..."

            progress_bar.value = i
            pct = (i / total) * 100
            if pending_count is None:
                pending_count = max(total - i, 0)
            running = min(max_workers, pending_count)
            queued = max(pending_count - running, 0)
            rate_display = (i / elapsed) if elapsed > 0 else 0

            line1 = (
                f"<span style='font-weight:700;font-size:16px;'>{i}/{total}</span> "
                f"<span style='color:#5b6b82;'>({pct:.0f}%)</span> "
                f"<span style='margin-left:8px;'><b>Elapsed:</b> {self._format_time(elapsed)}</span> "
                f"<span style='margin-left:8px;'><b>{eta_str}</b></span>"
            )
            line2 = (
                f"<span><b>Queue:</b> {queued}</span> "
                f"<span style='margin-left:8px;'><b>Running:</b> {running}</span> "
                f"<span style='margin-left:8px;'><b>Done:</b> {i}</span> "
                f"<span style='margin-left:8px;'><b>Rate:</b> {rate_display:.2f}/s</span>"
            )
            status_html.value = (
                "<div class='qc-card qc-mono' style='padding:8px 10px;'>"
                f"<div>{line1}</div>"
                f"<div style='margin-top:6px;color:#4d5f78;'>{line2}</div>"
                f"<div style='margin-top:8px;'>{render_status_badges()}</div>"
                "</div>"
            )
            run_health_html.value = render_run_health()
            error_review_html.value = render_error_review()
            status_dist_html.value = render_status_distribution()

        def on_series_complete(series: "SeriesInfo", series_time: float):
            """Handle completion of a single series."""
            recent_times.append(series_time)
            completion_timestamps.append(time.time())
            # Update incremental counts
            status = series.qc_status
            status_counts[status] = status_counts.get(status, 0) + 1
            outcome_history.append(status)
            if status == "ERROR":
                consecutive_errors[0] += 1
                new_run_error_count[0] += 1
            else:
                consecutive_errors[0] = 0
            if status == "ERROR":
                recent_issues.append(
                    (status, series.label, _extract_issue_detail(series))
                )
            # Track sample per status type (always update to keep current example)
            self._track_status_sample(series, status_samples, status_thumb_cache)
            i = processed_count[0]
            if (i - last_samples_update[0]) >= samples_update_interval:
                recent_thumbs_html.value = render_status_samples()
                last_samples_update[0] = i
            # Checkpoint to DB immediately (no pickle, no race condition)
            self._checkpoint_series(series, series_ctx)

        # Initial render
        run_health_html.value = render_run_health()
        error_review_html.value = render_error_review()
        status_dist_html.value = render_status_distribution()

        processed_count = [0]

        def process_one(series):
            t0 = time.time()
            self._process_series_with_error_capture(series)
            return series, t0

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_one, s): s for s in to_process}
                pending = set(futures.keys())

                while pending:
                    # Short timeout allows widget updates to flush to frontend
                    done, pending = wait(
                        pending, timeout=0.1, return_when=FIRST_COMPLETED
                    )

                    for f in done:
                        series = futures[f]
                        try:
                            _, t0 = f.result()
                            series_time = time.time() - t0
                        except Exception as e:
                            series.error = str(e)
                            series_time = 0

                        processed_count[0] += 1

                        on_series_complete(series, series_time)
                        update_progress(
                            processed_count[0], series, pending_count=len(pending)
                        )

        except KeyboardInterrupt:
            # Close DB connection so NFS can release file handles.
            # Without this, kernel interrupt leaves .nfs lock files that
            # prevent directory deletion.
            if self._db:
                try:
                    self._db.commit()
                    self._db.close()
                except Exception:
                    pass
                self._db = None
            elapsed = time.time() - start_time
            counts = self.get_summary()
            progress_bar.bar_style = "warning"
            status_html.value = (
                "<div class='qc-card qc-mono' style='padding:8px 10px;'>"
                f"<b>Interrupted</b> after {self._format_time(elapsed)} | {self._format_status_counts(counts)}</div>"
            )
            print(
                f"\nProcessing interrupted. {sum(counts.values())} series processed so far."
            )
            print(
                "Call qc.process_all() to resume (already-processed series will be skipped)."
            )
            return counts

        # Final save (full sync to catch any stragglers)
        try:
            self.save()
        except Exception as e:
            print(
                f"\nWarning: final save failed ({e}). Call qc.save() manually to retry."
            )

        # Final update
        progress_bar.value = total
        progress_bar.bar_style = "success"
        elapsed = time.time() - start_time
        counts = self.get_summary()

        final_status = [
            "<b>✓ Complete!</b>",
            f"{total} processed in {self._format_time(elapsed)}",
        ]
        if skipped > 0:
            final_status.append(f"({skipped} skipped)")
        final_status.extend(["|", self._format_status_counts(counts)])

        status_html.value = (
            "<div class='qc-card qc-mono' style='padding:8px 10px;'>"
            f"{' '.join(final_status)}</div>"
        )
        recent_thumbs_html.value = render_status_samples()

        return counts

    # Backward compatibility alias
    process_all_interactive = process_all
