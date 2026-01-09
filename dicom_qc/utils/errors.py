"""Custom exceptions for DICOM QC tool."""

from typing import Optional, List, Tuple


class DicomQCError(Exception):
    """Base exception for DICOM QC tool."""
    pass


class XNATConnectionError(DicomQCError):
    """Error connecting to XNAT server."""

    def __init__(self, message: str, server: Optional[str] = None):
        self.server = server
        super().__init__(f"XNAT connection failed: {message}")


class DicomLoadError(DicomQCError):
    """Error loading DICOM files."""

    def __init__(self, message: str, file_path: Optional[str] = None):
        self.file_path = file_path
        super().__init__(f"DICOM load error: {message}")


class IncompleteDicomError(DicomLoadError):
    """DICOM series is incomplete or has missing slices."""

    def __init__(self, expected: int, found: int):
        self.expected = expected
        self.found = found
        super().__init__(f"Expected {expected} slices, found {found}")


class CorruptDicomError(DicomLoadError):
    """DICOM file is corrupt or unreadable."""

    def __init__(self, message: str, file_path: Optional[str] = None):
        super().__init__(f"Corrupt DICOM: {message}", file_path)


class MissingTagError(DicomLoadError):
    """Required DICOM tag is missing."""

    def __init__(self, tag_name: str, tag_id: Optional[str] = None):
        self.tag_name = tag_name
        self.tag_id = tag_id
        tag_info = f"{tag_name} ({tag_id})" if tag_id else tag_name
        super().__init__(f"Missing required tag: {tag_info}")


class GeometryError(DicomQCError):
    """Error in geometry calculations."""

    def __init__(self, message: str):
        super().__init__(f"Geometry error: {message}")


class ReportGenerationError(DicomQCError):
    """Error generating HTML report."""

    def __init__(self, message: str):
        super().__init__(f"Report generation failed: {message}")


class ErrorDisplay:
    """Display errors in a user-friendly format within Jupyter."""

    @staticmethod
    def show_error(error: Exception, context: Optional[str] = None) -> None:
        """Display an error with styling."""
        from IPython.display import display, HTML

        error_type = type(error).__name__

        html = f'''
        <div style="background:#f8d7da; border:1px solid #f5c6cb;
                    border-radius:4px; padding:15px; margin:10px 0;">
            <h4 style="color:#721c24; margin-top:0;">
                Error: {error_type}
            </h4>
            <p style="color:#721c24;">{str(error)}</p>
            {f'<p style="color:#856404;"><strong>Context:</strong> {context}</p>' if context else ''}
        </div>
        '''
        display(HTML(html))

    @staticmethod
    def show_warning(message: str, details: Optional[str] = None) -> None:
        """Display a warning with styling."""
        from IPython.display import display, HTML

        html = f'''
        <div style="background:#fff3cd; border:1px solid #ffc107;
                    border-radius:4px; padding:15px; margin:10px 0;">
            <h4 style="color:#856404; margin-top:0;">
                Warning
            </h4>
            <p style="color:#856404;">{message}</p>
            {f'<p style="color:#856404;font-size:0.9em;">{details}</p>' if details else ''}
        </div>
        '''
        display(HTML(html))

    @staticmethod
    def show_loading_summary(
        total: int,
        loaded: int,
        errors: List[Tuple[str, Exception]],
        warnings: List[Tuple[str, str]]
    ) -> None:
        """Display summary after loading with errors/warnings."""
        from IPython.display import display, HTML

        if errors:
            error_list = '<ul>' + ''.join(
                f'<li><code>{fname}</code>: {str(err)}</li>'
                for fname, err in errors[:5]
            ) + '</ul>'
            if len(errors) > 5:
                error_list += f'<p>...and {len(errors)-5} more errors</p>'
        else:
            error_list = ''

        status_color = '#28a745' if not errors else '#ffc107'

        html = f'''
        <div style="background:#f8f9fa; border:1px solid #ddd;
                    border-radius:4px; padding:15px; margin:10px 0;">
            <h4 style="margin-top:0;">Loading Summary</h4>
            <p>
                <span style="color:{status_color};">
                    {loaded}/{total} files loaded successfully
                </span>
            </p>
            {f'<p style="color:#dc3545;"><strong>{len(errors)} errors:</strong></p>{error_list}' if errors else ''}
            {f'<p style="color:#856404;"><strong>{len(warnings)} warnings</strong> (see logs for details)</p>' if warnings else ''}
        </div>
        '''
        display(HTML(html))
