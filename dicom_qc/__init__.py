"""
DICOM QC Review Tool

A Python package for reviewing de-identified DICOM data with focus on
geometry/orientation verification, interactive Jupyter widgets, and
exportable HTML reports.

Usage:
    from dicom_qc import DicomQCTool

    qc = DicomQCTool()
    qc.browse()  # or qc.load_session('Project', 'Subject', 'Session')
    qc.review()
    qc.generate_report('qc_report.html')
"""

from dicom_qc.core.volume import DicomVolume, ScanInfo
from dicom_qc.core.geometry import GeometryQC, QCResult, QCReport
from dicom_qc.utils.errors import DicomQCError

__version__ = "0.1.0"
__all__ = [
    "DicomQCTool",
    "QuickCheck",
    "DicomVolume",
    "ScanInfo",
    "GeometryQC",
    "QCResult",
    "QCReport",
    "DicomQCError",
]

# Lazy import of main tools to avoid loading all dependencies at import time
def __getattr__(name):
    if name == "DicomQCTool":
        from dicom_qc.tool import DicomQCTool
        return DicomQCTool
    if name == "QuickCheck":
        from dicom_qc.quickcheck import QuickCheck
        return QuickCheck
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
