"""Generate deep links to OHIF viewer integrated with XNAT."""

from urllib.parse import urlencode


class OHIFLinkGenerator:
    """Generate deep links to OHIF viewer integrated with XNAT."""

    def __init__(self, xnat_base_url: str):
        """
        Initialize with XNAT server base URL.

        Args:
            xnat_base_url: Base URL of XNAT server (e.g., 'https://xnat.example.org')
        """
        self.base_url = xnat_base_url.rstrip('/')

    def generate_session_link(
        self,
        project: str,
        experiment_id: str
    ) -> str:
        """
        Generate OHIF link for an entire session.

        URL Pattern:
        {base}/VIEWER?project={project}&experimentId={experiment_id}

        Args:
            project: XNAT project ID
            experiment_id: XNAT experiment/session ID

        Returns:
            Full OHIF viewer URL
        """
        params = {
            'project': project,
            'experimentId': experiment_id
        }
        return f"{self.base_url}/VIEWER?{urlencode(params)}"

    def generate_scan_link(
        self,
        project: str,
        experiment_id: str,
        scan_id: str
    ) -> str:
        """
        Generate OHIF link for a specific scan.

        URL Pattern:
        {base}/VIEWER?project={project}&experimentId={experiment_id}&scanId={scan_id}

        Args:
            project: XNAT project ID
            experiment_id: XNAT experiment/session ID
            scan_id: XNAT scan ID within the experiment

        Returns:
            Full OHIF viewer URL with scan pre-selected
        """
        params = {
            'project': project,
            'experimentId': experiment_id,
            'scanId': scan_id
        }
        return f"{self.base_url}/VIEWER?{urlencode(params)}"

    def generate_dicomweb_link(
        self,
        study_instance_uid: str,
        series_instance_uid: str = None
    ) -> str:
        """
        Generate OHIF link using DICOM UIDs (for DICOMweb-native access).

        This is an alternative pattern if the XNAT instance supports
        native DICOMweb endpoints.

        URL Pattern:
        {base}/ohif/viewer?StudyInstanceUIDs={study_uid}

        Args:
            study_instance_uid: DICOM StudyInstanceUID
            series_instance_uid: Optional DICOM SeriesInstanceUID

        Returns:
            OHIF viewer URL
        """
        params = {'StudyInstanceUIDs': study_instance_uid}

        if series_instance_uid:
            params['SeriesInstanceUID'] = series_instance_uid

        return f"{self.base_url}/ohif/viewer?{urlencode(params)}"

    def generate_markdown_link(
        self,
        project: str,
        experiment_id: str,
        scan_id: str = None,
        link_text: str = "View in OHIF"
    ) -> str:
        """Generate markdown-formatted link for reports."""
        if scan_id:
            url = self.generate_scan_link(project, experiment_id, scan_id)
        else:
            url = self.generate_session_link(project, experiment_id)

        return f"[{link_text}]({url})"

    def generate_html_button(
        self,
        project: str,
        experiment_id: str,
        scan_id: str = None,
        button_text: str = "View in OHIF"
    ) -> str:
        """Generate HTML button for reports."""
        if scan_id:
            url = self.generate_scan_link(project, experiment_id, scan_id)
        else:
            url = self.generate_session_link(project, experiment_id)

        return f'''
            <a href="{url}" target="_blank"
               style="display: inline-block;
                      padding: 8px 16px;
                      background: #007bff;
                      color: white;
                      text-decoration: none;
                      border-radius: 4px;
                      font-weight: bold;">
                {button_text}
            </a>
        '''
