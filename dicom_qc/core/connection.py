"""XNAT connection and navigation wrapper."""

import os
import logging
from typing import Optional, List, Any

from dicom_qc.core.volume import ScanInfo
from dicom_qc.utils.errors import XNATConnectionError

logger = logging.getLogger(__name__)


class XNATSession:
    """
    Manages XNAT connection and navigation.

    In XNAT Jupyter environment, credentials are auto-loaded from
    environment variables (XNAT_HOST, XNAT_USER, XNAT_PASS).
    """

    def __init__(self, connection: Optional[Any] = None):
        """
        Initialize with optional existing xnat connection.

        Args:
            connection: Existing xnat.Session object, or None to auto-connect
        """
        self._connection = connection
        self._connected = connection is not None

    def connect(
        self,
        server: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ) -> Any:
        """
        Connect to XNAT server.

        Args:
            server: XNAT server URL (uses XNAT_HOST env var if not provided)
            user: Username (uses XNAT_USER env var if not provided)
            password: Password (uses XNAT_PASS env var if not provided)

        Returns:
            xnat.Session object

        Raises:
            XNATConnectionError: If connection fails
        """
        try:
            import xnat
        except ImportError:
            raise XNATConnectionError(
                "xnat package not installed. Install with: pip install xnat"
            )

        try:
            if server or user or password:
                # Manual connection
                server = server or os.environ.get("XNAT_HOST")
                user = user or os.environ.get("XNAT_USER")
                password = password or os.environ.get("XNAT_PASS")

                if not server:
                    raise XNATConnectionError("No XNAT server URL provided")

                self._connection = xnat.connect(server, user=user, password=password)
            else:
                # Auto-connect (works in XNAT Jupyter environment)
                self._connection = xnat.connect()

            self._connected = True
            logger.info(f"Connected to XNAT: {self._connection.url}")
            return self._connection

        except Exception as e:
            raise XNATConnectionError(str(e), server)

    @property
    def connection(self) -> Any:
        """Get the underlying xnat connection."""
        if not self._connected or self._connection is None:
            self.connect()
        return self._connection

    @property
    def url(self) -> str:
        """Get the XNAT server URL."""
        return self.connection.url

    def get_projects(self) -> List[str]:
        """
        Get list of accessible project IDs.

        Returns:
            List of project IDs
        """
        return list(self.connection.projects.keys())

    def get_project(self, project_id: str) -> Any:
        """Get a specific project by ID."""
        return self.connection.projects[project_id]

    def get_subjects(self, project: str) -> List[str]:
        """
        Get list of subject labels in a project.

        Args:
            project: Project ID

        Returns:
            List of subject labels
        """
        proj = self.get_project(project)
        return [subj.label for subj in proj.subjects.values()]

    def get_experiments(self, project: str, subject: str) -> List[str]:
        """
        Get list of experiment/session IDs for a subject.

        Args:
            project: Project ID
            subject: Subject label

        Returns:
            List of experiment IDs
        """
        proj = self.get_project(project)

        # Find subject by label
        subj = None
        for s in proj.subjects.values():
            if s.label == subject:
                subj = s
                break

        if subj is None:
            return []

        return [exp.label for exp in subj.experiments.values()]

    def get_scans(self, project: str, subject: str, experiment: str) -> List[ScanInfo]:
        """
        Get list of scans in an experiment/session.

        Args:
            project: Project ID
            subject: Subject label
            experiment: Experiment/session label

        Returns:
            List of ScanInfo objects
        """
        proj = self.get_project(project)

        # Find subject
        subj = None
        for s in proj.subjects.values():
            if s.label == subject:
                subj = s
                break

        if subj is None:
            return []

        # Find experiment
        exp = None
        for e in subj.experiments.values():
            if e.label == experiment:
                exp = e
                break

        if exp is None:
            return []

        # Get scans
        scans = []
        for scan_id, scan in exp.scans.items():
            # Count DICOM files
            num_files = sum(
                1
                for f in scan.files.values()
                if f.uri.endswith(".dcm") or f.uri.endswith(".DCM")
            )

            scan_info = ScanInfo(
                id=scan_id,
                description=getattr(scan, "series_description", "") or scan_id,
                modality=getattr(scan, "modality", "Unknown"),
                num_files=num_files,
                series_description=getattr(scan, "series_description", None),
                project=project,
                subject=subject,
                experiment=experiment,
                _scan_obj=scan,
            )
            scans.append(scan_info)

        return scans

    def get_scan_files(self, scan_info: ScanInfo) -> List[Any]:
        """
        Get list of DICOM file objects for a scan.

        Args:
            scan_info: ScanInfo object with _scan_obj populated

        Returns:
            List of xnat file objects
        """
        if scan_info._scan_obj is None:
            raise ValueError("ScanInfo does not have scan object reference")

        scan = scan_info._scan_obj
        files = []

        for file_obj in scan.files.values():
            if file_obj.uri.endswith(".dcm") or file_obj.uri.endswith(".DCM"):
                files.append(file_obj)

        return files

    def close(self) -> None:
        """Close the XNAT connection."""
        if self._connection is not None:
            try:
                self._connection.disconnect()
            except Exception:
                pass
            self._connection = None
            self._connected = False

    def __enter__(self):
        """Context manager entry."""
        if not self._connected:
            self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
