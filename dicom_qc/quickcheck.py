"""Quickcheck module for efficient batch review of DICOM data."""

import hashlib
import os
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pydicom


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
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            error_str = str(e).lower()
            # Check if it's a retryable error (503, 502, 504, connection issues)
            is_retryable = any(x in error_str for x in ['503', '502', '504', 'connection', 'timeout', 'temporarily'])

            if not is_retryable or attempt >= max_retries:
                raise

            last_error = e
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)

    raise last_error

from dicom_qc.core.dicom_loader import DicomLoader
from dicom_qc.core.geometry import GeometryQC
from dicom_qc.visualization.snapshots import SnapshotGenerator
from dicom_qc.quickcheck_html import QuickCheckHTMLMixin
from dicom_qc.quickcheck_display import QuickCheckDisplayMixin


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
            return open(self.data_path, 'rb')
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
    _scan_obj: Any = field(default=None, repr=False)  # xnat scan object for lazy file loading
    _file_uris: List[str] = field(default_factory=list)  # Picklable file URIs for restore
    _file_paths: List[str] = field(default_factory=list)  # Local file paths (if mounted)
    _scan_uri: Optional[str] = None  # Scan URI for restoring access
    xnat_scan_id: Optional[str] = None  # XNAT scan ID (only in XNAT mode)
    _xnat_subject_label: Optional[str] = None  # XNAT subject label (for thumbnail naming)
    _xnat_session_label: Optional[str] = None  # XNAT session label (for thumbnail naming)
    # Derived data fields (RTStruct, SEG, etc.)
    is_derived: bool = False
    referenced_series_uid: Optional[str] = None  # SeriesInstanceUID of referenced images
    derived_info: Optional[str] = None  # Human-readable info about the derived data
    _db_id: Optional[int] = None  # Database row ID (for scaled mode)

    @property
    def qc_status(self) -> str:
        if self.error:
            return 'ERROR'
        if self.is_derived:
            return 'DERIVED'
        if self.qc_report:
            return self.qc_report.overall_status
        return 'PENDING'

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
    xnat_experiment_id: Optional[str] = None  # XNAT internal experiment ID (e.g., XNAT_TEST02_E39302)

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
    xnat_subject_id: Optional[str] = None  # XNAT internal subject ID (e.g., XNAT_TEST02_S24513)

    @property
    def label(self) -> str:
        return f"{self.patient_id}: {self.patient_name}" if self.patient_name else self.patient_id


class QuickCheck(QuickCheckHTMLMixin, QuickCheckDisplayMixin):
    """Batch DICOM review with thumbnail grid visualization."""

    STATUS_COLORS = {
        'PASS': '#28a745',
        'WARNING': '#ffc107',
        'FAIL': '#dc3545',
        'ERROR': '#6c757d',
        'PENDING': '#17a2b8',
        'DERIVED': '#9c27b0',  # Purple for derived/non-image types
        'NOTE': '#17a2b8',     # Cyan/teal for informational notes (4D data, etc.)
    }

    # Modalities that are derived/non-image data (don't require geometry QC)
    DERIVED_MODALITIES = {
        'RTSTRUCT',  # Radiotherapy Structure Set (contours)
        'SEG',       # Segmentation
        'PR',        # Presentation State
        'KO',        # Key Object Selection
        'SR',        # Structured Report
        'DOC',       # Document
        'REG',       # Registration
        'FID',       # Fiducials
        'RTPLAN',    # Radiotherapy Plan
    }

    # SOP Classes that are NOT displayable images (no PixelData)
    NON_IMAGE_SOP_CLASSES = {
        '1.2.840.10008.5.1.4.1.1.66',     # Raw Data Storage
        '1.2.840.10008.5.1.4.1.1.66.1',   # Spatial Registration Storage
        '1.2.840.10008.5.1.4.1.1.66.2',   # Spatial Fiducials Storage
        '1.2.840.10008.5.1.4.1.1.66.3',   # Deformable Spatial Registration Storage
        '1.2.840.10008.5.1.4.1.1.66.4',   # Segmentation Storage (sometimes no pixels)
        '1.2.840.10008.5.1.4.1.1.67',     # Real World Value Mapping Storage
        '1.2.840.10008.5.1.4.1.1.88.11',  # Basic Text SR Storage
        '1.2.840.10008.5.1.4.1.1.88.22',  # Enhanced SR Storage
        '1.2.840.10008.5.1.4.1.1.88.33',  # Comprehensive SR Storage
        '1.2.840.10008.5.1.4.1.1.88.34',  # Comprehensive 3D SR Storage
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
        self._save_path: Optional[Path] = None
        self._xnat_session: Any = None  # XNAT session for restoring file access
        self._xnat_project_id: Optional[str] = None  # XNAT project ID for OHIF links
        self._xnat_base_url: Optional[str] = None  # XNAT base URL for OHIF links

        # Scaled mode: SQLite database + thumbnail disk cache
        self._use_db = use_db
        self._db = None
        self._thumb_cache = None

        if use_db and data_dir:
            self._init_storage()

    def _init_storage(self):
        """Initialize database and thumbnail cache for scaled operations."""
        if not self.data_dir:
            return

        from dicom_qc.storage import QCDatabase, ThumbnailCache

        storage_dir = self.data_dir / '_dicom_qc'
        storage_dir.mkdir(parents=True, exist_ok=True)

        self._db = QCDatabase(storage_dir / 'qc_database.sqlite3')
        self._thumb_cache = ThumbnailCache(storage_dir / 'thumbnails')

    def _sync_series_to_db(self, series: 'SeriesInfo', patient_id: str, study_key: str,
                           patient: 'PatientInfo', study: 'StudyInfo') -> int:
        """Sync a series to the database, return database ID."""
        if not self._db:
            return None

        # Insert/update patient
        patient_db_id = self._db.insert_patient(
            patient_id=patient_id,
            patient_name=patient.patient_name,
            xnat_subject_id=patient.xnat_subject_id,
        )

        # Insert/update study
        study_db_id = self._db.insert_study(
            patient_db_id=patient_db_id,
            study_uid=study.uid,
            study_date=study.date,
            study_description=study.description,
            xnat_session_label=study.xnat_session_label,
            xnat_experiment_id=study.xnat_experiment_id,
        )

        # Insert/update series
        series_db_id = self._db.insert_series(
            study_db_id=study_db_id,
            series_uid=series.uid,
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

        # Store series files if available
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
        """Sync all series to database."""
        if not self._db:
            return

        for patient_id, patient in self.patients.items():
            for study_key, study in patient.studies.items():
                for series_key, series in study.series.items():
                    self._sync_series_to_db(series, patient_id, study_key, patient, study)

        self._db.commit()

    def _migrate_thumbnails_to_disk(self):
        """Migrate base64 thumbnails to disk cache."""
        if not self._thumb_cache:
            return

        migrated = 0
        for series in self.get_all_series():
            if series.thumbnail and not series._thumbnail_path:
                # Migrate base64 to disk
                rel_path = self._thumb_cache.save_thumbnail_from_base64(
                    series.uid, series.thumbnail
                )
                if rel_path:
                    series._thumbnail_path = rel_path
                    series.thumbnail = None  # Free memory
                    migrated += 1

        if migrated:
            print(f"Migrated {migrated} thumbnails to disk cache")

        return migrated

    def save(self, path: Optional[Path] = None) -> Path:
        """Save discovery and processing state to a file.

        Preserves file URIs so files can be restored after load without
        re-iterating through XNAT. Call connect_xnat(session) after loading
        to enable file access.

        Args:
            path: Path to save file. If None, uses last save path or generates one.

        Returns:
            Path where data was saved
        """
        import pickle

        if path:
            self._save_path = Path(path)
        elif not self._save_path:
            # Default: save in _dicom_qc/ storage directory
            if self.data_dir:
                storage_dir = self.data_dir / '_dicom_qc'
                storage_dir.mkdir(parents=True, exist_ok=True)
                self._save_path = storage_dir / 'qc_state.pkl'
            else:
                self._save_path = Path('qc_state.pkl')

        # Ensure file URIs and paths are stored before clearing non-picklable objects
        missing_uris_count = 0
        for series in self.get_all_series():
            series.volume = None
            if series._xnat_files:
                # Store URIs and local paths if we have files but none stored yet
                if series.files and not series._file_uris:
                    try:
                        series._file_uris = [f.uri for f in series.files]
                        series._file_paths = [getattr(f, 'data_path', None) or '' for f in series.files]
                    except Exception:
                        pass  # Some file objects may not have uri
                # Track series missing file URIs (file listing failed during discovery)
                if not series._file_uris:
                    missing_uris_count += 1
                # Only clear files for COMPLETED series (has thumbnail, qc_report, error, or is_derived)
                # This allows parallel processing to continue using files for pending series
                is_complete = series.thumbnail or series._thumbnail_path or series.qc_report or series.error or series.is_derived
                if is_complete:
                    series.files = []
                    series._scan_obj = None

        # Migrate thumbnails to disk cache if available (frees memory)
        if self._thumb_cache:
            self._migrate_thumbnails_to_disk()

        # Sync to database if available
        if self._db:
            self._sync_all_to_db()

        # Prepare data for saving (pickle for backward compatibility)
        save_data = {
            'patients': self.patients,
            '_xnat_mode': self._xnat_mode,
            'data_dir': str(self.data_dir) if self.data_dir else None,
            '_xnat_project_id': self._xnat_project_id,
            '_xnat_base_url': self._xnat_base_url,
            '_use_db': self._use_db,  # Track if scaled mode was used
        }

        # Atomic save: write to temp file, then rename (prevents corruption on interrupt)
        import tempfile
        temp_fd, temp_path = tempfile.mkstemp(
            dir=self._save_path.parent,
            prefix='.qc_state_',
            suffix='.tmp'
        )
        try:
            with os.fdopen(temp_fd, 'wb') as f:
                pickle.dump(save_data, f)
            # Atomic rename (on POSIX systems)
            os.replace(temp_path, self._save_path)
        except Exception:
            # Clean up temp file on error
            try:
                os.unlink(temp_path)
            except Exception:
                pass
            raise

        return self._save_path

    def load(self, path: Path = None, use_db: bool = None) -> 'QuickCheck':
        """Load discovery and processing state from a file.

        Args:
            path: Path to saved state file. If None, uses default location
                  ({data_dir}/_dicom_qc/qc_state.pkl)
            use_db: If True, initialize database for scaled operations.
                   If None, auto-detect from save file.

        Returns:
            self for chaining
        """
        import pickle

        if path is None:
            if self.data_dir:
                path = self.data_dir / '_dicom_qc' / 'qc_state.pkl'
            else:
                path = Path('qc_state.pkl')
        else:
            path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Save file not found: {path}")

        with open(path, 'rb') as f:
            save_data = pickle.load(f)

        self.patients = save_data['patients']
        self._xnat_mode = save_data['_xnat_mode']
        if save_data.get('data_dir'):
            self.data_dir = Path(save_data['data_dir'])
        self._xnat_project_id = save_data.get('_xnat_project_id')
        self._xnat_base_url = save_data.get('_xnat_base_url')
        self._save_path = path

        # Determine if we should use scaled mode
        if use_db is None:
            # Auto-detect: use DB if save file used it, or if we have many series
            use_db = save_data.get('_use_db', False)
            all_series = self.get_all_series()
            if not use_db and len(all_series) > 500:
                use_db = True  # Auto-enable for large datasets

        self._use_db = use_db
        if use_db and self.data_dir:
            self._init_storage()
            # Migrate base64 thumbnails to disk if present
            self._migrate_thumbnails_to_disk()
            # Sync to database
            self._sync_all_to_db()

        # Count what was loaded
        all_series = self.get_all_series()
        total_series = len(all_series)
        with_thumbnail = sum(1 for s in all_series if s.thumbnail or s._thumbnail_path)
        derived = sum(1 for s in all_series if s.is_derived)
        errors = sum(1 for s in all_series if s.error)
        pending = total_series - with_thumbnail - derived - errors

        print(f"Loaded state from {path}")
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
        """Check if a save file exists at the default location."""
        if self.data_dir:
            path = self.data_dir / '_dicom_qc' / 'qc_state.pkl'
        else:
            path = Path('qc_state.pkl')
        return path.exists()

    def load_if_exists(self) -> bool:
        """Load from default save location if it exists.

        Returns:
            True if loaded, False if no save file found
        """
        if self.has_save():
            self.load()
            return True
        return False

    @classmethod
    def from_save(cls, path: Path) -> 'QuickCheck':
        """Create a QuickCheck instance from a saved state file.

        Args:
            path: Path to saved state file

        Returns:
            QuickCheck instance with loaded state
        """
        qc = cls()
        qc.load(path)
        return qc

    def reset(self, delete_storage: bool = True) -> 'QuickCheck':
        """Reset to fresh state, optionally deleting all stored data.

        Use this to start over with a clean slate. Clears:
        - All in-memory patient/study/series data
        - SQLite database (if delete_storage=True)
        - Thumbnail cache (if delete_storage=True)
        - Pickle save file (if delete_storage=True)

        Args:
            delete_storage: If True, delete database, thumbnails, and save file.
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

        if delete_storage and self.data_dir:
            storage_dir = Path(self.data_dir) / '_dicom_qc'
            if storage_dir.exists():
                # Use ignore_errors for NFS filesystems with lock files
                shutil.rmtree(storage_dir, ignore_errors=True)
                # If directory still exists (NFS locks), at least clear the contents we can
                if storage_dir.exists():
                    for item in storage_dir.iterdir():
                        if not item.name.startswith('.nfs'):
                            try:
                                if item.is_dir():
                                    shutil.rmtree(item, ignore_errors=True)
                                else:
                                    item.unlink()
                            except OSError:
                                pass
                    print(f"Cleared storage: {storage_dir} (some NFS lock files may remain)")
                else:
                    print(f"Deleted storage: {storage_dir}")

        if delete_storage and self._save_path and Path(self._save_path).exists():
            Path(self._save_path).unlink()
            print(f"Deleted save file: {self._save_path}")

        # Reset storage references
        self._db = None
        self._thumb_cache = None

        # Re-initialize storage if using scaled mode
        if self._use_db and self.data_dir:
            self._init_storage()

        print("Reset complete. Ready for discover().")
        return self

    def connect_xnat(self, session: Any) -> 'QuickCheck':
        """Connect an XNAT session for file access after loading from save.

        After loading from a save file, call this method with your XNAT session
        to enable file access using the stored file paths/URIs. This avoids
        re-iterating through all XNAT subjects/experiments/scans.

        Args:
            session: xnatpy session object (from xnat.connect())

        Returns:
            self for chaining

        Example:
            qc = QuickCheck.from_save('state.pkl')
            qc.connect_xnat(xnat.connect())
            qc.process_all_interactive()  # Files accessed via stored paths
        """
        self._xnat_session = session

        # Extract base URL from session for OHIF links (if not already set)
        if not self._xnat_base_url:
            if hasattr(session, '_original_uri'):
                self._xnat_base_url = session._original_uri.rstrip('/')
            elif hasattr(session, 'host'):
                self._xnat_base_url = session.host.rstrip('/')

        # Count series with stored file info
        all_series = self.get_all_series()
        total = len(all_series)

        if total == 0:
            print("XNAT session connected (no stored data)")
        else:
            with_paths = sum(1 for s in all_series if s._file_paths and any(s._file_paths))
            with_uris = sum(1 for s in all_series if s._file_uris)
            if with_paths > 0:
                print(f"XNAT session connected ({with_paths}/{total} series have local paths, {with_uris} have URIs)")
            else:
                print(f"XNAT session connected ({with_uris}/{total} series have stored URIs)")

    def discover(self, refresh: bool = False) -> Dict[str, PatientInfo]:
        """Scan DICOM files and build patient/study/series hierarchy.

        Uses os.walk with followlinks=True to traverse symbolic links.

        Args:
            refresh: If True, clear existing data and re-discover everything.
                     If False (default), preserve existing series that have thumbnails/results.
        """
        import os

        # Store existing processed series to preserve them
        existing_series = {}
        if not refresh:
            for series in self.get_all_series():
                if series.thumbnail or series.is_derived or series.qc_report:
                    existing_series[series.uid] = series

        self.patients = {}

        # Use os.walk to follow symlinks (rglob does not follow symlinks by default)
        dcm_files = []
        for root, dirs, files in os.walk(self.data_dir, followlinks=True):
            for f in files:
                if f.lower().endswith('.dcm'):
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

        # Auto-save after discovery
        if self.data_dir:
            self.save()

        return self.patients

    def discover_xnat(self, project: Any, interactive: bool = True, refresh: bool = True, read_dicom: bool = False, parallel: bool = None, max_workers: int = 8) -> Dict[str, PatientInfo]:
        """Discover DICOM series from an XNAT project.

        Uses XNAT hierarchy (subject/session/scan) instead of DICOM headers.

        Args:
            project: xnatpy project object
            interactive: If True, show progress in Jupyter
            refresh: If True, clear existing data and re-discover everything.
                     If False, only add new subjects/sessions/scans (incremental).
            read_dicom: If True, read first DICOM file to get accurate metadata.
                       If False (default), use only XNAT metadata (much faster).
            parallel: If True, use parallel discovery for faster throughput.
                     If None, auto-enable for >10 subjects.
            max_workers: Number of parallel workers. Default: 8.

        Returns:
            Dict of PatientInfo keyed by subject label
        """
        if refresh:
            self.patients = {}
        self._xnat_mode = True

        # Store project info for OHIF links
        try:
            self._xnat_project_id = project.id
            # Extract base URL from session
            if hasattr(project, '_session') and hasattr(project._session, '_original_uri'):
                self._xnat_base_url = project._session._original_uri.rstrip('/')
            elif hasattr(project, '_session') and hasattr(project._session, 'host'):
                self._xnat_base_url = project._session.host.rstrip('/')
        except Exception:
            pass

        # Store session reference for file access (via stored URIs after save/load)
        try:
            self._xnat_session = project._session
        except Exception:
            pass  # Session not available, URI restore won't work

        # Get subjects list first for progress tracking
        subjects = list(project.subjects.values())
        total_subjects = len(subjects)

        # Set up progress display if interactive
        progress_widget = None
        status_widget = None
        new_scans = 0
        skipped_scans = 0

        mode_str = "Discovering" if refresh else "Updating"
        if interactive:
            try:
                import ipywidgets as widgets
                from IPython.display import display

                # Use FloatProgress for smoother updates (estimate progress within subjects)
                progress_widget = widgets.FloatProgress(
                    value=0, min=0, max=100,
                    description=f'{mode_str}:',
                    bar_style='info',
                    style={'bar_color': '#17a2b8', 'description_width': '80px'},
                    layout=widgets.Layout(width='100%')
                )
                status_widget = widgets.HTML(f'<div style="font-family:monospace;">0/{total_subjects} subjects | 0 sessions | 0 scans</div>')
                display(widgets.VBox([progress_widget, status_widget]))
            except Exception:
                interactive = False

        total_sessions = 0
        total_scans = 0
        scans_per_subject = []  # Track for estimating progress

        # Auto-enable parallelism for projects with many subjects
        if parallel is None:
            parallel = total_subjects > 2

        # For parallel mode, take a snapshot of existing patients to avoid race conditions
        existing_patients_snapshot = dict(self.patients) if not refresh else {}

        # Helper function to discover a single subject
        def discover_subject(subject):
            """Discover all sessions and scans for a single subject."""
            subject_label = subject.label
            subject_sessions = 0
            subject_scans = 0
            subject_new_scans = 0
            subject_skipped = 0

            # Get or create patient (use snapshot for thread safety)
            if subject_label in existing_patients_snapshot:
                # Copy existing patient to avoid modifying shared state
                existing = existing_patients_snapshot[subject_label]
                patient = PatientInfo(
                    subject_label,
                    existing.patient_name,
                    xnat_subject_id=existing.xnat_subject_id or getattr(subject, 'id', None)
                )
                # Copy existing studies
                for study_key, study in existing.studies.items():
                    patient.studies[study_key] = StudyInfo(
                        uid=study.uid,
                        date=study.date,
                        description=study.description,
                        xnat_session_label=study.xnat_session_label,
                        xnat_experiment_id=study.xnat_experiment_id,
                    )
                    # Copy existing series
                    for series_key, series in study.series.items():
                        patient.studies[study_key].series[series_key] = series
            else:
                xnat_subj_id = getattr(subject, 'id', None)
                patient = PatientInfo(subject_label, '', xnat_subject_id=xnat_subj_id)

            # Get experiments with retry
            try:
                experiments = _xnat_retry(lambda: list(subject.experiments.values()))
            except Exception:
                return subject_label, patient, subject_sessions, subject_scans, subject_new_scans, subject_skipped

            for experiment in experiments:
                session_label = experiment.label
                subject_sessions += 1

                if session_label not in patient.studies:
                    xnat_exp_id = getattr(experiment, 'id', None)
                    patient.studies[session_label] = StudyInfo(
                        uid='',
                        date='',
                        description=session_label,
                        xnat_session_label=session_label,
                        xnat_experiment_id=xnat_exp_id,
                    )
                elif patient.studies[session_label].xnat_experiment_id is None:
                    patient.studies[session_label].xnat_experiment_id = getattr(experiment, 'id', None)

                study = patient.studies[session_label]

                # Get scans with retry
                try:
                    scans = _xnat_retry(lambda: list(experiment.scans.values()))
                except Exception:
                    continue

                for scan in scans:
                    scan_id = scan.id
                    subject_scans += 1

                    # Skip existing scans in incremental mode
                    if not refresh and scan_id in study.series:
                        existing = study.series[scan_id]
                        if existing._file_uris:
                            subject_skipped += 1
                            continue
                        # Re-fetch missing file URIs
                        try:
                            def fetch_existing_files():
                                all_files = []
                                for res in scan.resources.values():
                                    res_label = getattr(res, 'label', '').upper()
                                    if res_label != 'SNAPSHOTS':
                                        all_files.extend(res.files.values())
                                return all_files
                            all_files = _xnat_retry(fetch_existing_files)
                            if all_files:
                                existing._file_uris = [f.uri for f in all_files]
                                existing._file_paths = [getattr(f, 'data_path', None) or '' for f in all_files]
                            else:
                                existing.error = "No DICOM files in XNAT resources"
                        except Exception as e:
                            existing.error = f"Failed to fetch files: {e}"
                        subject_skipped += 1
                        continue

                    subject_new_scans += 1

                    # Get metadata from XNAT
                    try:
                        series_desc = _xnat_retry(lambda: getattr(scan, 'series_description', '') or '')
                        modality = _xnat_retry(lambda: getattr(scan, 'modality', '??') or '??')
                    except Exception:
                        series_desc = ''
                        modality = '??'

                    series = SeriesInfo(
                        uid='',
                        series_number=scan_id,
                        description=series_desc,
                        modality=modality,
                        xnat_scan_id=scan_id,
                        _xnat_subject_label=subject_label,
                        _xnat_session_label=session_label,
                    )
                    series._xnat_files = True

                    # Fetch file URIs and paths
                    try:
                        def fetch_files():
                            all_files = []
                            for res in scan.resources.values():
                                res_label = getattr(res, 'label', '').upper()
                                if res_label != 'SNAPSHOTS':
                                    all_files.extend(res.files.values())
                            return all_files
                        all_files = _xnat_retry(fetch_files)
                        series._file_uris = [f.uri for f in all_files]
                        series._file_paths = [getattr(f, 'data_path', None) or '' for f in all_files]
                    except Exception:
                        series._scan_obj = scan

                    study.series[scan_id] = series

            return subject_label, patient, subject_sessions, subject_scans, subject_new_scans, subject_skipped

        if parallel and max_workers > 1:
            # Parallel discovery with per-scan updates (mirrors process_all pattern)
            from concurrent.futures import ThreadPoolExecutor
            import threading
            import asyncio

            lock = threading.Lock()

            # Phase 1: Collect all scan tasks sequentially (fast - just XNAT hierarchy)
            scan_tasks = []  # List of (subject_label, session_label, scan, study_ref) tuples

            if progress_widget:
                status_widget.value = f'<div style="font-family:monospace;">Collecting scan list from {total_subjects} subjects...</div>'

            for subject in subjects:
                subject_label = subject.label

                # Ensure patient exists
                if subject_label not in self.patients:
                    xnat_subj_id = getattr(subject, 'id', None)
                    self.patients[subject_label] = PatientInfo(subject_label, '', xnat_subject_id=xnat_subj_id)
                elif self.patients[subject_label].xnat_subject_id is None:
                    self.patients[subject_label].xnat_subject_id = getattr(subject, 'id', None)

                # Get experiments
                try:
                    experiments = _xnat_retry(lambda: list(subject.experiments.values()))
                except Exception:
                    continue

                for experiment in experiments:
                    session_label = experiment.label
                    total_sessions += 1

                    # Ensure study exists
                    if session_label not in self.patients[subject_label].studies:
                        xnat_exp_id = getattr(experiment, 'id', None)
                        self.patients[subject_label].studies[session_label] = StudyInfo(
                            uid='',
                            date='',
                            description=session_label,
                            xnat_session_label=session_label,
                            xnat_experiment_id=xnat_exp_id,
                        )
                    elif self.patients[subject_label].studies[session_label].xnat_experiment_id is None:
                        self.patients[subject_label].studies[session_label].xnat_experiment_id = getattr(experiment, 'id', None)

                    study = self.patients[subject_label].studies[session_label]

                    # Get scans
                    try:
                        scans = _xnat_retry(lambda: list(experiment.scans.values()))
                    except Exception:
                        continue

                    for scan in scans:
                        scan_id = scan.id
                        total_scans += 1

                        # Skip existing scans in incremental mode
                        if not refresh and scan_id in study.series:
                            existing = study.series[scan_id]
                            if existing._file_uris:
                                skipped_scans += 1
                                continue

                        # Add to task list for parallel processing
                        scan_tasks.append((subject_label, session_label, scan, study))

            # Phase 2: Process each scan in parallel with per-scan UI updates
            total_scan_tasks = len(scan_tasks)
            processed_count = [0]

            if progress_widget:
                progress_widget.value = 0
                status_widget.value = f'<div style="font-family:monospace;">0/{total_scan_tasks} scans | {total_sessions} sessions | {total_subjects} subjects</div>'

            def process_scan(task):
                """Process a single scan (thread-safe)."""
                subject_label, session_label, scan, study = task
                scan_id = scan.id

                # Get metadata from XNAT
                try:
                    series_desc = _xnat_retry(lambda: getattr(scan, 'series_description', '') or '')
                    modality = _xnat_retry(lambda: getattr(scan, 'modality', '??') or '??')
                except Exception:
                    series_desc = ''
                    modality = '??'

                series = SeriesInfo(
                    uid='',
                    series_number=scan_id,
                    description=series_desc,
                    modality=modality,
                    xnat_scan_id=scan_id,
                    _xnat_subject_label=subject_label,
                    _xnat_session_label=session_label,
                )
                series._xnat_files = True

                # Fetch file URIs and paths (the slow part)
                try:
                    def fetch_files():
                        all_files = []
                        for res in scan.resources.values():
                            res_label = getattr(res, 'label', '').upper()
                            if res_label != 'SNAPSHOTS':
                                all_files.extend(res.files.values())
                        return all_files
                    all_files = _xnat_retry(fetch_files)
                    series._file_uris = [f.uri for f in all_files]
                    series._file_paths = [getattr(f, 'data_path', None) or '' for f in all_files]
                except Exception:
                    series._scan_obj = scan

                return subject_label, session_label, scan_id, series

            # Use asyncio pattern like process_all for responsive updates
            async def run_parallel():
                nonlocal new_scans
                loop = asyncio.get_running_loop()

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    pending_futures = {
                        loop.run_in_executor(executor, process_scan, task): task
                        for task in scan_tasks
                    }

                    while pending_futures:
                        done, pending = await asyncio.wait(
                            pending_futures.keys(),
                            timeout=0.3,
                            return_when=asyncio.FIRST_COMPLETED
                        )

                        for future in done:
                            task = pending_futures.pop(future)
                            subject_label, session_label, _, _ = task
                            try:
                                subj_label, sess_label, scan_id, series = await future

                                with lock:
                                    # Add series to study
                                    self.patients[subj_label].studies[sess_label].series[scan_id] = series
                                    new_scans += 1
                                    processed_count[0] += 1

                                    # Update progress
                                    if progress_widget:
                                        progress_pct = (processed_count[0] / total_scan_tasks) * 100
                                        progress_widget.value = progress_pct
                                        status_widget.value = (
                                            f'<div style="font-family:monospace;">'
                                            f'{processed_count[0]}/{total_scan_tasks} scans | {total_sessions} sessions | {total_subjects} subjects<br>'
                                            f'<span style="color:#888">{subj_label} / {sess_label} / scan {scan_id}</span></div>'
                                        )

                            except Exception:
                                with lock:
                                    processed_count[0] += 1

                        # Yield to event loop for widget updates
                        await asyncio.sleep(0.01)

            # Fallback for when asyncio doesn't work
            def run_with_fallback():
                nonlocal new_scans
                from concurrent.futures import as_completed

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(process_scan, task): task for task in scan_tasks}

                    for future in as_completed(futures):
                        task = futures[future]
                        subject_label, session_label, _, _ = task
                        try:
                            subj_label, sess_label, scan_id, series = future.result()

                            with lock:
                                self.patients[subj_label].studies[sess_label].series[scan_id] = series
                                new_scans += 1
                                processed_count[0] += 1

                                if progress_widget:
                                    progress_pct = (processed_count[0] / total_scan_tasks) * 100
                                    progress_widget.value = progress_pct
                                    status_widget.value = (
                                        f'<div style="font-family:monospace;">'
                                        f'{processed_count[0]}/{total_scan_tasks} scans | {total_sessions} sessions | {total_subjects} subjects<br>'
                                        f'<span style="color:#888">{subj_label} / {sess_label} / scan {scan_id}</span></div>'
                                    )

                        except Exception:
                            with lock:
                                processed_count[0] += 1

            # Run with asyncio if possible, fallback otherwise
            try:
                try:
                    loop = asyncio.get_running_loop()
                    import nest_asyncio
                    nest_asyncio.apply()
                    loop.run_until_complete(run_parallel())
                except RuntimeError:
                    asyncio.run(run_parallel())
            except ImportError:
                run_with_fallback()
            except Exception:
                import traceback
                traceback.print_exc()
                run_with_fallback()

            # Save once at end
            if self._save_path and new_scans > 0:
                self.save()

        else:
            # Sequential discovery (original behavior)
            for subj_idx, subject in enumerate(subjects):
                subject_label = subject.label
                subject_scan_count = 0
                subject_new_scans = 0

                if subject_label not in self.patients:
                    # Get internal XNAT subject ID
                    xnat_subj_id = getattr(subject, 'id', None)
                    self.patients[subject_label] = PatientInfo(subject_label, '', xnat_subject_id=xnat_subj_id)
                elif self.patients[subject_label].xnat_subject_id is None:
                    # Update missing XNAT subject ID (e.g., from old save file)
                    self.patients[subject_label].xnat_subject_id = getattr(subject, 'id', None)

                # Get experiments with retry for transient errors
                try:
                    experiments = _xnat_retry(lambda: list(subject.experiments.values()))
                except Exception as e:
                    # Skip subject if we can't access its experiments
                    continue

                for experiment in experiments:
                    session_label = experiment.label
                    total_sessions += 1

                    if session_label not in self.patients[subject_label].studies:
                        # Get internal XNAT experiment ID
                        xnat_exp_id = getattr(experiment, 'id', None)
                        self.patients[subject_label].studies[session_label] = StudyInfo(
                            uid='',  # Will be set from DICOM if available
                            date='',
                            description=session_label,
                            xnat_session_label=session_label,
                            xnat_experiment_id=xnat_exp_id,
                        )
                    elif self.patients[subject_label].studies[session_label].xnat_experiment_id is None:
                        # Update missing XNAT experiment ID (e.g., from old save file)
                        self.patients[subject_label].studies[session_label].xnat_experiment_id = getattr(experiment, 'id', None)

                    # Get scans with retry for transient errors
                    try:
                        scans = _xnat_retry(lambda: list(experiment.scans.values()))
                    except Exception as e:
                        # Skip session if we can't access its scans
                        continue

                    for scan in scans:
                        scan_id = scan.id
                        total_scans += 1
                        subject_scan_count += 1

                        # Update progress per scan - estimate progress within current subject
                        if progress_widget:
                            # Estimate: use avg scans/subject to calculate sub-progress
                            if scans_per_subject:
                                avg_scans = sum(scans_per_subject) / len(scans_per_subject)
                                # Progress = completed subjects + fraction of current subject
                                sub_progress = min(subject_scan_count / max(avg_scans, 1), 1.0)
                            else:
                                # First subject: assume we're partway through
                                sub_progress = min(subject_scan_count / 100, 0.9)  # Cap at 90% until we know more

                            progress_pct = ((subj_idx + sub_progress) / total_subjects) * 100
                            progress_widget.value = progress_pct
                            status_widget.value = f'<div style="font-family:monospace;">{subj_idx}/{total_subjects} subjects | {total_sessions} sessions | {total_scans} scans<br><span style="color:#888">{subject_label} / {session_label} / scan {scan_id}</span></div>'

                        # Skip existing scans in incremental mode (unless missing file URIs)
                        study = self.patients[subject_label].studies[session_label]
                        if not refresh and scan_id in study.series:
                            existing = study.series[scan_id]
                            if existing._file_uris:
                                skipped_scans += 1
                                continue
                            # Series exists but missing file URIs - re-fetch them (with retry)
                            try:
                                def fetch_existing_files():
                                    all_files = []
                                    for res in scan.resources.values():
                                        # Exclude SNAPSHOTS resource
                                        res_label = getattr(res, 'label', '').upper()
                                        if res_label != 'SNAPSHOTS':
                                            all_files.extend(res.files.values())
                                    return all_files

                                all_files = _xnat_retry(fetch_existing_files)
                                if all_files:
                                    existing._file_uris = [f.uri for f in all_files]
                                    existing._file_paths = [getattr(f, 'data_path', None) or '' for f in all_files]
                                else:
                                    existing.error = "No DICOM files in XNAT resources"
                            except Exception as e:
                                existing.error = f"Failed to fetch files: {e}"
                            skipped_scans += 1
                            continue

                        new_scans += 1
                        subject_new_scans += 1

                        # Get metadata from XNAT (with retry for transient errors)
                        try:
                            series_desc = _xnat_retry(lambda: getattr(scan, 'series_description', '') or '')
                            modality = _xnat_retry(lambda: getattr(scan, 'modality', '??') or '??')
                        except Exception:
                            # Fallback if metadata fetch fails entirely
                            series_desc = ''
                            modality = '??'

                        series = SeriesInfo(
                            uid='',  # Will be set during processing
                            series_number=scan_id,
                            description=series_desc,
                            modality=modality,
                            xnat_scan_id=scan_id,
                            _xnat_subject_label=subject_label,
                            _xnat_session_label=session_label,
                        )
                        series._xnat_files = True

                        # Fetch and store file URIs and local paths now (enables save/load without re-discovery)
                        try:
                            def fetch_files():
                                all_files = []
                                for res in scan.resources.values():
                                    # Exclude SNAPSHOTS resource
                                    res_label = getattr(res, 'label', '').upper()
                                    if res_label != 'SNAPSHOTS':
                                        all_files.extend(res.files.values())
                                return all_files

                            all_files = _xnat_retry(fetch_files)
                            series._file_uris = [f.uri for f in all_files]
                            series._file_paths = [getattr(f, 'data_path', None) or '' for f in all_files]
                        except Exception:
                            # Store scan object as fallback if file listing fails
                            series._scan_obj = scan

                        study.series[scan_id] = series

                # Track scans per subject for progress estimation
                scans_per_subject.append(subject_scan_count)

                # Save after each subject with new scans (file listing is slow, preserve progress)
                if self._save_path and subject_new_scans > 0:
                    self.save()

        # Final progress update
        if progress_widget:
            progress_widget.value = 100
            progress_widget.bar_style = 'success'
            if refresh:
                status_widget.value = f'<div style="font-family:monospace;"><b>✓ Discovery complete:</b> {total_subjects} subjects | {total_sessions} sessions | {total_scans} scans</div>'
            else:
                status_widget.value = f'<div style="font-family:monospace;"><b>✓ Update complete:</b> {new_scans} new scans | {skipped_scans} existing (skipped)</div>'
        else:
            # Non-interactive: print summary
            print(f"Discovered {total_subjects} subjects, {total_sessions} sessions, {total_scans} scans")

        # Flag any series still missing file URIs (e.g., deleted from XNAT but in saved state)
        missing_count = 0
        for series in self.get_all_series():
            if series._xnat_files and not series._file_uris and not series.error:
                series.error = "No file paths available (scan may have been removed from XNAT)"
                missing_count += 1

        # Auto-save after discovery
        if self.data_dir:
            self.save()

    def _add_file_to_hierarchy(self, ds: pydicom.Dataset, dcm_file: Path) -> None:
        """Add a DICOM file to the hierarchy."""
        patient_id = str(getattr(ds, 'PatientID', 'Unknown'))
        patient_name = str(getattr(ds, 'PatientName', ''))
        study_uid = getattr(ds, 'StudyInstanceUID', 'unknown')
        study_date = getattr(ds, 'StudyDate', 'Unknown')
        study_desc = getattr(ds, 'StudyDescription', '')
        series_uid = getattr(ds, 'SeriesInstanceUID', 'unknown')
        series_num = getattr(ds, 'SeriesNumber', 0) or 0
        series_desc = getattr(ds, 'SeriesDescription', 'Unknown')
        modality = getattr(ds, 'Modality', '??')

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
            if hasattr(ds, 'file_meta'):
                transfer_syntax = str(getattr(ds.file_meta, 'TransferSyntaxUID', None))
                implementation = getattr(ds.file_meta, 'ImplementationVersionName', None)
            study.series[series_uid] = SeriesInfo(
                series_uid, series_num, series_desc, modality,
                transfer_syntax=transfer_syntax, implementation=implementation
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
        counts = {'PASS': 0, 'WARNING': 0, 'FAIL': 0, 'ERROR': 0, 'PENDING': 0, 'DERIVED': 0, 'NOTE': 0}
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
                        print(f"    Has URIs: {bool(s._file_uris)}, Has paths: {bool(s._file_paths and any(s._file_paths))}")
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

        print(f"Cleared {cleared} errors. Run discover_xnat(refresh=False) to re-fetch file paths, then process_all_interactive()")
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
                        if r.status != 'PASS':
                            print(f"  {r.status}: {r.check_name} - {r.message}")
                # Show volume info
                if series.volume:
                    v = series.volume
                    print(f"  Volume: {v.shape[0]} slices, {v.pixel_spacing[0]:.2f}x{v.pixel_spacing[1]:.2f}mm pixels, {v.slice_thickness:.2f}mm slice thickness")

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
            if s.qc_status == 'NOTE':
                is_4d = True
            elif s.qc_report:
                # Look for 4D Data check with NOTE status specifically
                for r in s.qc_report.results:
                    if r.check_name == '4D Data' and r.status == 'NOTE':
                        is_4d = True
                        break
            if is_4d:
                series_4d.append(s)

        if not series_4d:
            print("No 4D series found to reprocess")
            return 0

        print(f"Reprocessing {len(series_4d)} 4D series...")

        for i, series in enumerate(series_4d):
            print(f"  [{i+1}/{len(series_4d)}] {series.description[:50]}...", end=" ")
            self.reprocess_series(series, silent=True)
            print(f"-> {series.qc_status}")

            if save_interval > 0 and (i + 1) % save_interval == 0 and self._save_path:
                self.save()

        # Final save
        if self._save_path:
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
        matching = [s for s in self.get_all_series()
                    if pattern_lower in (s.description or '').lower()]

        if not matching:
            print(f"No series found matching '{pattern}'")
            return 0

        print(f"Reprocessing {len(matching)} series matching '{pattern}'...")

        for i, series in enumerate(matching):
            print(f"  [{i+1}/{len(matching)}] {series.description[:50]}...", end=" ")
            self.reprocess_series(series, silent=True)
            print(f"-> {series.qc_status}")

            if save_interval > 0 and (i + 1) % save_interval == 0 and self._save_path:
                self.save()

        # Final save
        if self._save_path:
            self.save()

        print(f"\nDone! Reprocessed {len(matching)} series.")
        print("Call qc.display() to refresh the view")
        return len(matching)

    # JPEG-2000 transfer syntaxes
    JPEG2000_UIDS = {
        '1.2.840.10008.1.2.4.90',  # JPEG 2000 Lossless
        '1.2.840.10008.1.2.4.91',  # JPEG 2000 Lossy
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
                paths = series._file_paths if series._file_paths else [None] * len(series._file_uris)
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
                all_files.extend(res.files.values())
            series.files = all_files
            # Store URIs and local paths for save/restore
            series._file_uris = [f.uri for f in all_files]
            series._file_paths = [getattr(f, 'data_path', None) or '' for f in all_files]
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
                        sop_class = getattr(ds.file_meta, 'MediaStorageSOPClassUID', None)
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
                    series.uid = hashlib.sha256(series._file_uris[0].encode()).hexdigest()

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
                if series._xnat_subject_label and series._xnat_session_label and series.xnat_scan_id:
                    thumb_path = self._thumb_cache.get_path_for_xnat(
                        series._xnat_subject_label, series._xnat_session_label, series.xnat_scan_id
                    )
                    snapshot_gen.create_tripane_thumbnail_file(thumb_path)
                    series._thumbnail_path = self._thumb_cache.get_relative_path_xnat(
                        series._xnat_subject_label, series._xnat_session_label, series.xnat_scan_id
                    )
                else:
                    # Fallback to hash-based path for local data
                    thumb_path = self._thumb_cache.get_path_for_series(series.uid)
                    snapshot_gen.create_tripane_thumbnail_file(thumb_path)
                    series._thumbnail_path = self._thumb_cache.get_relative_path(series.uid)
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
                ds = pydicom.dcmread(files[0].open(), stop_before_pixels=True)
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

    def _extract_referenced_series(self, ds: pydicom.Dataset, modality: str) -> Optional[str]:
        """Extract the referenced series UID from a derived DICOM object."""
        try:
            # RTStruct: deeply nested reference structure
            if modality == 'RTSTRUCT':
                for fof_ref in getattr(ds, 'ReferencedFrameOfReferenceSequence', []):
                    for study_ref in getattr(fof_ref, 'RTReferencedStudySequence', []):
                        for series_ref in getattr(study_ref, 'RTReferencedSeriesSequence', []):
                            return str(getattr(series_ref, 'SeriesInstanceUID', ''))

            # SEG, PR, KO, SR: simpler ReferencedSeriesSequence
            for series_ref in getattr(ds, 'ReferencedSeriesSequence', []):
                return str(getattr(series_ref, 'SeriesInstanceUID', ''))

            # Alternative: CurrentRequestedProcedureEvidenceSequence (KO, SR)
            for evidence in getattr(ds, 'CurrentRequestedProcedureEvidenceSequence', []):
                for series_ref in getattr(evidence, 'ReferencedSeriesSequence', []):
                    return str(getattr(series_ref, 'SeriesInstanceUID', ''))

        except Exception:
            pass
        return None

    def _build_derived_info(self, ds: pydicom.Dataset, modality: str) -> str:
        """Build human-readable info about a derived DICOM object."""
        info_parts = []

        if modality == 'RTSTRUCT':
            # Count ROIs
            roi_seq = getattr(ds, 'StructureSetROISequence', [])
            if roi_seq:
                info_parts.append(f"{len(roi_seq)} ROIs")
                # List first few ROI names
                roi_names = [getattr(roi, 'ROIName', '') for roi in roi_seq[:5]]
                roi_names = [n for n in roi_names if n]
                if roi_names:
                    info_parts.append(', '.join(roi_names[:3]))
                    if len(roi_names) > 3:
                        info_parts[-1] += '...'

        elif modality == 'SEG':
            # Count segments
            seg_seq = getattr(ds, 'SegmentSequence', [])
            if seg_seq:
                info_parts.append(f"{len(seg_seq)} segments")
                seg_labels = [getattr(seg, 'SegmentLabel', '') for seg in seg_seq[:3]]
                seg_labels = [l for l in seg_labels if l]
                if seg_labels:
                    info_parts.append(', '.join(seg_labels))

        elif modality == 'SR':
            # Content description
            content_desc = getattr(ds, 'ContentDescription', '')
            if content_desc:
                info_parts.append(content_desc[:50])

        elif modality == 'PR':
            info_parts.append('Presentation State')

        elif modality == 'KO':
            info_parts.append('Key Object Selection')

        return ' | '.join(info_parts) if info_parts else f'{modality} data'

    def _check_encoding(self, series: SeriesInfo) -> None:
        """Check for problematic transfer syntax/encoding combinations."""
        from dicom_qc.core.geometry import QCResult
        import struct

        if not series.qc_report or not series.transfer_syntax:
            return

        # Check JPEG-2000 for multi-layer encoding (causes OHIF rendering issues)
        if series.transfer_syntax in self.JPEG2000_UIDS and series.files:
            try:
                import pydicom
                ds = pydicom.dcmread(str(series.files[0]))
                frame = list(pydicom.encaps.generate_pixel_data_frame(ds.PixelData))[0]

                # Find COD marker and extract number of layers
                cod_pos = frame.find(b'\xff\x52')
                if cod_pos != -1:
                    num_layers = struct.unpack('>H', frame[cod_pos+6:cod_pos+8])[0]

                    if num_layers > 1:
                        result = QCResult(
                            status='WARNING',
                            check_name='JPEG-2000 Encoding',
                            message=f'Multi-layer JPEG-2000 ({num_layers} layers) may render blurry in OHIF',
                            details={
                                'transfer_syntax': series.transfer_syntax,
                                'num_layers': num_layers,
                                'recommendation': 'Re-encode with single layer if viewer issues occur'
                            }
                        )
                        series.qc_report.results.append(result)

                        if series.qc_report.overall_status == 'PASS':
                            series.qc_report.overall_status = 'WARNING'
            except Exception:
                pass  # Skip encoding check if we can't parse

    def _check_temporal_metadata(self, series: SeriesInfo) -> None:
        """Check for missing temporal metadata in dynamic/perfusion series."""
        from dicom_qc.core.geometry import QCResult

        if not series.qc_report or not series.files:
            return

        # Only check series that look like dynamic/temporal acquisitions
        desc_lower = (series.description or "").lower()
        is_dynamic = any(p in desc_lower for p in ['perf', 'dsc', 'dce', 'dynamic', 'cine', 'fmri', 'bold'])

        if not is_dynamic:
            return

        try:
            import pydicom

            # Check first few files for temporal tags
            temporal_tags_found = {
                'TriggerTime': False,
                'AcquisitionTime': False,
                'TemporalPositionIdentifier': False,
                'NumberOfTemporalPositions': False,
            }

            files_to_check = series.files[:min(5, len(series.files))]
            for dcm_file in files_to_check:
                ds = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)

                if hasattr(ds, 'TriggerTime') and ds.TriggerTime is not None:
                    temporal_tags_found['TriggerTime'] = True
                if hasattr(ds, 'AcquisitionTime') and ds.AcquisitionTime is not None:
                    temporal_tags_found['AcquisitionTime'] = True
                if hasattr(ds, 'TemporalPositionIdentifier') and ds.TemporalPositionIdentifier is not None:
                    temporal_tags_found['TemporalPositionIdentifier'] = True
                if hasattr(ds, 'NumberOfTemporalPositions') and ds.NumberOfTemporalPositions is not None:
                    temporal_tags_found['NumberOfTemporalPositions'] = True

            # Check if critical temporal tags are missing
            missing_tags = [tag for tag, found in temporal_tags_found.items() if not found]

            if missing_tags:
                result = QCResult(
                    status='WARNING',
                    check_name='Temporal Metadata',
                    message=f'Dynamic series missing temporal tags: {", ".join(missing_tags)}',
                    details={
                        'series_description': series.description,
                        'missing_tags': missing_tags,
                        'found_tags': [tag for tag, found in temporal_tags_found.items() if found],
                        'note': 'Missing temporal metadata may cause viewer compatibility issues with ITK-SNAP, 3D Slicer'
                    }
                )
                series.qc_report.results.append(result)

                if series.qc_report.overall_status == 'PASS':
                    series.qc_report.overall_status = 'WARNING'

        except Exception:
            pass  # Skip check if we can't parse

    def _process_all_simple(self, progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Dict[str, int]:
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
            progress_callback(total, total, 'Done!')

        return self.get_summary()

    def process_all(
        self,
        reprocess: bool = False,
        retry_errors: bool = False,
        save_interval: int = 10,
        parallel: bool = None,
        max_workers: int = None,
    ):
        """Process all series with live Jupyter progress display and grid view.

        Shows progress bar, ETA, status counts, and a live-updating thumbnail grid.

        Args:
            reprocess: If False, skip already-processed series (those with thumbnails or is_derived)
            retry_errors: If True, retry series that previously had errors
            save_interval: Save progress every N series (0 to disable)
            parallel: If True, use parallel processing for faster throughput.
                     If None, auto-enable for >100 series.
            max_workers: Number of parallel workers. Default: 4 for parallel mode.

        Returns:
            Summary counts by status
        """
        import time
        import warnings
        import ipywidgets as widgets
        from IPython.display import display, clear_output

        # Suppress warnings during processing
        warnings.filterwarnings('ignore')

        # Suppress SimpleITK warnings (outputs to stderr, not Python warnings)
        try:
            import SimpleITK as sitk
            sitk.ProcessObject_SetGlobalWarningDisplay(False)
        except Exception:
            pass

        all_series = self.get_all_series()

        # Filter to only unprocessed series unless reprocess=True
        if reprocess:
            to_process = all_series
            skipped = 0
        else:
            to_process = [s for s in all_series
                         if not s.thumbnail and not s._thumbnail_path and not s.is_derived and not s.error]
            skipped = len(all_series) - len(to_process)

        # Add series with errors if retry_errors=True
        if retry_errors and not reprocess:
            error_series = [s for s in all_series if s.error]
            if error_series:
                # Clear errors so they can be retried
                for s in error_series:
                    s.error = None
                to_process.extend(error_series)
                print(f"Retrying {len(error_series)} series with errors")

        total = len(to_process)

        if total == 0:
            if skipped > 0:
                print(f"All {skipped} series already processed. Use reprocess=True to re-run, or retry_errors=True to retry errors.")
            else:
                print("No series to process")
            return self.get_summary()

        # Create progress widgets
        skip_msg = f" ({skipped} already processed)" if skipped > 0 else ""
        progress_bar = widgets.IntProgress(
            value=0, min=0, max=total,
            description='Processing:',
            bar_style='info',
            style={'bar_color': '#007bff', 'description_width': '80px'},
            layout=widgets.Layout(width='100%')
        )
        status_html = widgets.HTML(f'<div style="font-family:monospace;">0/{total} series{skip_msg}</div>')
        current_series_html = widgets.HTML('')

        # Compact status map (colored dots) + recent thumbnails
        status_map_html = widgets.HTML('')  # Compact overview of all series
        recent_thumbs_html = widgets.HTML('')  # Last N full thumbnails

        # Build display
        container = widgets.VBox([
            progress_bar, status_html, current_series_html,
            widgets.HTML('<div style="margin:10px 0 5px;font-size:11px;color:#888;"><b>Status Overview</b> (each square = 1 series)</div>'),
            status_map_html,
            widgets.HTML('<div style="margin:15px 0 5px;font-size:11px;color:#888;"><b>Recent Results</b></div>'),
            recent_thumbs_html,
        ], layout=widgets.Layout(width='100%'))
        display(container)

        # Track recent thumbnails - only show first few so user can verify, then stop
        # (no value in watching 100K thumbnails scroll by)
        max_recent = 4
        max_thumbs_to_show = 5  # Stop updating after this many
        recent_series = []
        recent_thumb_cache = {}
        thumbs_shown = [0]  # Use list for mutability in nested function

        # Track timing
        start_time = time.time()
        recent_times = []

        def format_time(seconds):
            if seconds < 60:
                return f"{seconds:.0f}s"
            elif seconds < 3600:
                return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
            else:
                return f"{seconds // 3600:.0f}h {(seconds % 3600) // 60:.0f}m"

        def update_status(i, series):
            elapsed = time.time() - start_time

            # Calculate ETA from recent processing times
            if recent_times:
                avg_time = sum(recent_times[-10:]) / len(recent_times[-10:])
                remaining = (total - i) * avg_time
                eta_str = f"ETA: {format_time(remaining)}"
            else:
                eta_str = "ETA: calculating..."

            # Update progress bar
            progress_bar.value = i
            pct = (i / total) * 100

            # Get current counts
            counts = self.get_summary()

            # Status HTML with colors
            c = self.STATUS_COLORS
            status_parts = [
                f"<b>{i}/{total}</b> ({pct:.0f}%)",
                f"Elapsed: {format_time(elapsed)}",
                eta_str,
                "|",
                f"<span style='color:{c['PASS']}'>✓{counts['PASS']}</span>",
                f"<span style='color:{c['WARNING']}'>⚠{counts['WARNING']}</span>",
                f"<span style='color:{c['FAIL']}'>✗{counts['FAIL']}</span>",
                f"<span style='color:{c['ERROR']}'>⊘{counts['ERROR']}</span>",
            ]
            if counts.get('NOTE', 0) > 0:
                status_parts.append(f"<span style='color:{c['NOTE']}'>ℹ{counts['NOTE']}</span>")
            if counts.get('DERIVED', 0) > 0:
                status_parts.append(f"<span style='color:{c['DERIVED']}'>◇{counts['DERIVED']}</span>")

            status_html.value = f"<div style='font-family:monospace;padding:5px 0;'>{' '.join(status_parts)}</div>"
            current_series_html.value = f"<div style='color:#666;font-size:12px;'>Current: {series.label[:60]}</div>"

        def render_status_map():
            """Render compact status map - one small square per series."""
            c = self.STATUS_COLORS
            squares = []
            for series in all_series:
                color = c.get(series.qc_status, '#444')
                squares.append(f'<div style="width:8px;height:8px;background:{color};border-radius:1px;" title="{series.label}"></div>')
            return f'<div style="display:flex;flex-wrap:wrap;gap:2px;max-height:150px;overflow-y:auto;">{"".join(squares)}</div>'

        def render_recent_thumbs():
            """Render recent thumbnails with full detail."""
            if not recent_series:
                return '<div style="color:#666;font-size:12px;">Processing...</div>'

            c = self.STATUS_COLORS
            html_parts = ['<div style="display:flex;flex-wrap:wrap;gap:6px;">']

            for series in recent_series:
                color = c.get(series.qc_status, '#ccc')

                # Get thumbnail from cache (loaded once when series added to recent)
                thumb_b64 = recent_thumb_cache.get(series.uid)

                if thumb_b64:
                    # Use jpeg for disk cache, png for legacy base64
                    mime = 'image/jpeg' if series._thumbnail_path else 'image/png'
                    img = f'<img src="data:{mime};base64,{thumb_b64}" style="width:100%;display:block;">'
                elif series.is_derived:
                    img = f'<div style="height:70px;background:#2d1f3d;color:#9c27b0;display:flex;align-items:center;justify-content:center;font-size:11px;">{series.modality}</div>'
                elif series.error:
                    img = f'<div style="height:70px;background:#2d1515;color:#dc3545;display:flex;align-items:center;justify-content:center;font-size:10px;padding:5px;text-align:center;">{series.error[:30]}</div>'
                else:
                    img = '<div style="height:70px;background:#333;color:#666;display:flex;align-items:center;justify-content:center;">?</div>'

                html_parts.append(f'''<div style="width:220px;border:3px solid {color};border-radius:4px;overflow:hidden;background:#000;">
                    {img}
                    <div style="padding:4px 6px;background:#222;color:white;font-size:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                        {series.label[:35]}
                    </div>
                </div>''')

            html_parts.append('</div>')

            # Show message when previews stop
            if thumbs_shown[0] >= max_thumbs_to_show:
                html_parts.append(
                    '<div style="color:#888;font-size:11px;margin-top:8px;font-style:italic;">'
                    'Preview stopped after first few series. See status map above for progress.'
                    '</div>'
                )

            return ''.join(html_parts)

        # Initial render
        status_map_html.value = render_status_map()

        # Auto-enable parallelism for large datasets
        if parallel is None:
            parallel = total > 100

        if max_workers is None:
            max_workers = 4 if parallel else 1

        if parallel and max_workers > 1:
            # Parallel processing with ThreadPoolExecutor
            from concurrent.futures import ThreadPoolExecutor, as_completed, FIRST_COMPLETED, wait
            import threading

            processed_count = [0]
            lock = threading.Lock()

            def process_one(series):
                """Process a single series (thread-safe)."""
                self.process_series(series)
                return series

            # Use asyncio for non-blocking parallel processing
            # This allows Jupyter's event loop to process widget updates
            import asyncio

            async def run_parallel():
                loop = asyncio.get_running_loop()
                nonlocal processed_count

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all jobs to thread pool, get futures
                    pending_futures = {
                        loop.run_in_executor(executor, process_one, s): s
                        for s in to_process
                    }
                    last_update = time.time()

                    while pending_futures:
                        # Wait for at least one to complete, with timeout for UI updates
                        done, pending = await asyncio.wait(
                            pending_futures.keys(),
                            timeout=0.5,
                            return_when=asyncio.FIRST_COMPLETED
                        )

                        for future in done:
                            series = pending_futures.pop(future)
                            try:
                                await future  # Get result/raise exception
                            except Exception as e:
                                series.error = str(e)

                            with lock:
                                processed_count[0] += 1
                                i = processed_count[0]

                                series_time = time.time() - start_time
                                recent_times.append(series_time / i if i > 0 else 0)

                                # Update displays
                                update_status(i, series)
                                status_map_html.value = render_status_map()

                                # Only show thumbnails for first few series
                                if thumbs_shown[0] < max_thumbs_to_show:
                                    if series._thumbnail_path and self._thumb_cache:
                                        recent_thumb_cache[series.uid] = self._thumb_cache.get_thumbnail_base64(series._thumbnail_path)
                                    elif series.thumbnail:
                                        recent_thumb_cache[series.uid] = series.thumbnail
                                    recent_series.append(series)
                                    if len(recent_series) > max_recent:
                                        old = recent_series.pop(0)
                                        recent_thumb_cache.pop(old.uid, None)
                                    thumbs_shown[0] += 1
                                    recent_thumbs_html.value = render_recent_thumbs()

                                # Periodic save
                                if save_interval > 0 and i % save_interval == 0 and self._save_path:
                                    self.save()

                            last_update = time.time()

                        # Heartbeat: update elapsed time even when no futures complete
                        now = time.time()
                        if now - last_update > 1.0 and pending_futures:
                            elapsed = now - start_time
                            i = processed_count[0]
                            remaining = len(pending_futures)
                            current_series_html.value = (
                                f"<div style='color:#666;font-size:12px;'>"
                                f"Processing ({remaining} remaining)...</div>"
                            )
                            progress_bar.value = i
                            pct = (i / total) * 100 if total > 0 else 0
                            counts = self.get_summary()
                            c = self.STATUS_COLORS
                            status_parts = [
                                f"<b>{i}/{total}</b> ({pct:.0f}%)",
                                f"Elapsed: {format_time(elapsed)}",
                                "Processing...",
                                "|",
                                f"<span style='color:{c['PASS']}'>✓{counts['PASS']}</span>",
                                f"<span style='color:{c['WARNING']}'>⚠{counts['WARNING']}</span>",
                                f"<span style='color:{c['FAIL']}'>✗{counts['FAIL']}</span>",
                                f"<span style='color:{c['ERROR']}'>⊘{counts['ERROR']}</span>",
                            ]
                            status_html.value = f"<div style='font-family:monospace;padding:5px 0;'>{' '.join(status_parts)}</div>"
                            last_update = now

                        # Yield to event loop to allow widget updates to flush
                        await asyncio.sleep(0.01)

            # Run the async function
            # In Jupyter, the event loop is already running, so we need nest_asyncio
            def run_with_fallback():
                from concurrent.futures import ThreadPoolExecutor as TPool, wait as sync_wait, FIRST_COMPLETED as FC

                last_series = to_process[0] if to_process else None
                with TPool(max_workers=max_workers) as executor:
                    futures = {executor.submit(process_one, s): s for s in to_process}
                    pending = set(futures.keys())
                    while pending:
                        done, pending = sync_wait(pending, timeout=1.0, return_when=FC)
                        for f in done:
                            series = futures[f]
                            try:
                                f.result()
                            except Exception as e:
                                series.error = str(e)
                            processed_count[0] += 1
                            last_series = series

                            # Track timing for ETA
                            series_time = time.time() - start_time
                            recent_times.append(series_time / processed_count[0] if processed_count[0] > 0 else 0)

                            # Update thumbnail preview for first few series
                            if thumbs_shown[0] < max_thumbs_to_show:
                                if series._thumbnail_path and self._thumb_cache:
                                    recent_thumb_cache[series.uid] = self._thumb_cache.get_thumbnail_base64(series._thumbnail_path)
                                elif series.thumbnail:
                                    recent_thumb_cache[series.uid] = series.thumbnail
                                recent_series.append(series)
                                if len(recent_series) > max_recent:
                                    old = recent_series.pop(0)
                                    recent_thumb_cache.pop(old.uid, None)
                                thumbs_shown[0] += 1
                                recent_thumbs_html.value = render_recent_thumbs()

                        # Update display periodically
                        i = processed_count[0]
                        if last_series:
                            update_status(i, last_series)
                        status_map_html.value = render_status_map()
                        if save_interval > 0 and i % save_interval == 0 and self._save_path:
                            self.save()

            try:
                # Try to get or create event loop
                try:
                    loop = asyncio.get_running_loop()
                    # Loop is running (Jupyter) - need nest_asyncio
                    import nest_asyncio
                    nest_asyncio.apply()
                    loop.run_until_complete(run_parallel())
                except RuntimeError:
                    # No running loop - create one and run
                    asyncio.run(run_parallel())
            except ImportError:
                # nest_asyncio not available - fall back to synchronous processing
                print("Note: For better progress display, install nest_asyncio: pip install nest_asyncio")
                run_with_fallback()
            except Exception:
                # Any other error - fall back to synchronous approach
                import traceback
                traceback.print_exc()
                run_with_fallback()
        else:
            # Sequential processing (original behavior)
            for i, series in enumerate(to_process):
                series_start = time.time()
                update_status(i, series)

                self.process_series(series)

                series_time = time.time() - series_start
                recent_times.append(series_time)

                # Update displays
                status_map_html.value = render_status_map()

                # Only show thumbnails for first few series (verify it's working)
                if thumbs_shown[0] < max_thumbs_to_show:
                    if series._thumbnail_path and self._thumb_cache:
                        recent_thumb_cache[series.uid] = self._thumb_cache.get_thumbnail_base64(series._thumbnail_path)
                    elif series.thumbnail:
                        recent_thumb_cache[series.uid] = series.thumbnail
                    recent_series.append(series)
                    if len(recent_series) > max_recent:
                        old = recent_series.pop(0)
                        recent_thumb_cache.pop(old.uid, None)
                    thumbs_shown[0] += 1
                    recent_thumbs_html.value = render_recent_thumbs()

                # Periodic save
                if save_interval > 0 and (i + 1) % save_interval == 0 and self._save_path:
                    self.save()

        # Final save
        if self._save_path:
            self.save()

        # Final update
        progress_bar.value = total
        progress_bar.bar_style = 'success'
        elapsed = time.time() - start_time
        counts = self.get_summary()

        c = self.STATUS_COLORS
        final_status = [
            f"<b>✓ Complete!</b>",
            f"{total} processed in {format_time(elapsed)}",
        ]
        if skipped > 0:
            final_status.append(f"({skipped} skipped)")
        final_status.extend([
            "|",
            f"<span style='color:{c['PASS']}'>✓{counts['PASS']}</span>",
            f"<span style='color:{c['WARNING']}'>⚠{counts['WARNING']}</span>",
            f"<span style='color:{c['FAIL']}'>✗{counts['FAIL']}</span>",
            f"<span style='color:{c['ERROR']}'>⊘{counts['ERROR']}</span>",
        ])
        if counts.get('NOTE', 0) > 0:
            final_status.append(f"<span style='color:{c['NOTE']}'>ℹ{counts['NOTE']}</span>")
        if counts.get('DERIVED', 0) > 0:
            final_status.append(f"<span style='color:{c['DERIVED']}'>◇{counts['DERIVED']}</span>")

        status_html.value = f"<div style='font-family:monospace;padding:5px 0;'>{' '.join(final_status)}</div>"
        current_series_html.value = ""

        return counts

    # Backward compatibility alias
    process_all_interactive = process_all

