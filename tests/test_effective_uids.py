"""Tests for QuickCheck._get_effective_uids."""

from dicom_qc.quickcheck import QuickCheck, SeriesInfo, StudyInfo


def _make_series(**overrides) -> SeriesInfo:
    defaults = dict(uid="1.2.3.4", series_number=1, description="T1", modality="MR")
    defaults.update(overrides)
    return SeriesInfo(**defaults)


def _make_study(**overrides) -> StudyInfo:
    defaults = dict(uid="9.8.7.6", date="20250101", description="Brain MRI")
    defaults.update(overrides)
    return StudyInfo(**defaults)


class TestXnatMode:
    def test_returns_xnat_identifiers(self):
        series = _make_series(xnat_scan_id="5")
        study = _make_study(xnat_session_label="SESSION_01")

        result = QuickCheck._get_effective_uids(series, study, "study_key")

        assert result == ("SESSION_01", "5")

    def test_falls_back_to_study_key_when_session_label_is_none(self):
        series = _make_series(xnat_scan_id="5")
        study = _make_study(xnat_session_label=None)

        result = QuickCheck._get_effective_uids(series, study, "study_key")

        assert result == ("study_key", "5")


class TestLocalMode:
    def test_returns_dicom_uids(self):
        series = _make_series(uid="1.2.3.4")
        study = _make_study(uid="9.8.7.6")

        result = QuickCheck._get_effective_uids(series, study, "study_key")

        assert result == ("9.8.7.6", "1.2.3.4")

    def test_falls_back_to_series_number_when_series_uid_empty(self):
        series = _make_series(uid="", series_number=7)
        study = _make_study(uid="9.8.7.6")

        result = QuickCheck._get_effective_uids(series, study, "study_key")

        assert result == ("9.8.7.6", "7")

    def test_falls_back_to_study_key_when_study_uid_empty(self):
        series = _make_series(uid="1.2.3.4")
        study = _make_study(uid="")

        result = QuickCheck._get_effective_uids(series, study, "study_key")

        assert result == ("study_key", "1.2.3.4")
