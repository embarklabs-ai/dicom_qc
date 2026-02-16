"""Tests for disk-based thumbnail cache."""

import base64
import hashlib
from io import BytesIO
from unittest.mock import patch

import pytest

from dicom_qc.storage.thumbnail_cache import ThumbnailCache


@pytest.fixture
def cache(tmp_path) -> ThumbnailCache:
    """Create a ThumbnailCache in a temporary directory."""
    return ThumbnailCache(tmp_path / "thumbnails")


def _make_png_bytes() -> bytes:
    """Create a minimal valid 1x1 red PNG using PIL."""
    from PIL import Image

    img = Image.new("RGB", (1, 1), (255, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _hash_prefix(series_uid: str) -> str:
    """Return the 16-char sha256 hex prefix used for path hashing."""
    return hashlib.sha256(series_uid.encode()).hexdigest()[:16]


class TestSaveThumbnail:
    """Tests for save_thumbnail and get_thumbnail_bytes round-trip."""

    def test_round_trip(self, cache):
        """Saved bytes are returned unchanged by get_thumbnail_bytes."""
        uid = "1.2.3.4.5"
        data = b"\xff\xd8\xff\xe0fake-jpeg-data"

        rel_path = cache.save_thumbnail(uid, data)
        retrieved = cache.get_thumbnail_bytes(rel_path)

        assert retrieved == data

    def test_relative_path_format(self, cache):
        """Returned relative path matches hash-based ab/abcdef...jpg format."""
        uid = "1.2.3.4.5"
        expected_hash = _hash_prefix(uid)

        rel_path = cache.save_thumbnail(uid, b"data")

        assert rel_path == f"{expected_hash[:2]}/{expected_hash}.jpg"

    def test_file_created_on_disk(self, cache):
        """Thumbnail file is physically created at the expected path."""
        uid = "1.2.3.4.5"
        data = b"image-bytes"

        cache.save_thumbnail(uid, data)
        full_path = cache.get_path_for_series(uid)

        assert full_path.exists()
        assert full_path.read_bytes() == data


class TestSaveThumbnailFromBase64:
    """Tests for save_thumbnail_from_base64 with PNG-to-JPEG conversion."""

    def test_png_converted_to_jpeg(self, cache):
        """Valid PNG is converted to JPEG when PIL is available."""
        uid = "1.2.840.10008.1"
        png_bytes = _make_png_bytes()
        b64_data = base64.b64encode(png_bytes).decode()

        rel_path = cache.save_thumbnail_from_base64(uid, b64_data)

        assert rel_path.endswith(".jpg")
        saved_bytes = cache.get_thumbnail_bytes(rel_path)
        # JPEG files start with FF D8
        assert saved_bytes[:2] == b"\xff\xd8"

    def test_jpeg_passthrough(self, cache):
        """JPEG input is saved without conversion."""
        uid = "1.2.840.10008.2"
        jpeg_bytes = b"\xff\xd8\xff\xe0some-jpeg-content"
        b64_data = base64.b64encode(jpeg_bytes).decode()

        rel_path = cache.save_thumbnail_from_base64(uid, b64_data)

        assert rel_path.endswith(".jpg")
        assert cache.get_thumbnail_bytes(rel_path) == jpeg_bytes

    def test_invalid_base64_returns_none(self, cache):
        """Invalid base64 input returns None."""
        result = cache.save_thumbnail_from_base64("1.2.3", "!!!not-base64!!!")
        assert result is None

    def test_fallback_when_pil_unavailable(self, cache):
        """PNG saved with .png extension when PIL cannot be imported."""
        uid = "1.2.840.10008.3"
        png_bytes = _make_png_bytes()
        b64_data = base64.b64encode(png_bytes).decode()

        with patch.dict("sys.modules", {"PIL": None, "PIL.Image": None}):
            rel_path = cache.save_thumbnail_from_base64(uid, b64_data)

        assert rel_path.endswith(".png")
        saved_bytes = cache.get_thumbnail_bytes(rel_path)
        assert saved_bytes == png_bytes

    def test_fallback_when_conversion_fails(self, cache):
        """Corrupt PNG saved with .png extension when conversion raises."""
        uid = "1.2.840.10008.4"
        # Valid PNG header but corrupt body -- PIL will fail to open
        corrupt_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
        b64_data = base64.b64encode(corrupt_png).decode()

        rel_path = cache.save_thumbnail_from_base64(uid, b64_data)

        assert rel_path.endswith(".png")
        assert cache.get_thumbnail_bytes(rel_path) == corrupt_png


class TestGetThumbnailBase64:
    """Tests for get_thumbnail_base64."""

    def test_returns_base64_string(self, cache):
        """Stored bytes are returned as a valid base64 string."""
        uid = "1.2.3.4.5"
        data = b"hello-thumbnail"
        rel_path = cache.save_thumbnail(uid, data)

        b64 = cache.get_thumbnail_base64(rel_path)

        assert base64.b64decode(b64) == data

    def test_missing_file_returns_none(self, cache):
        """Non-existent path returns None."""
        assert cache.get_thumbnail_base64("no/such/file.jpg") is None


class TestXnatPaths:
    """Tests for XNAT-style path generation."""

    def test_path_structure(self, cache):
        """XNAT path follows subject/session/scan_id.jpg layout."""
        path = cache.get_path_for_xnat("SUBJ001", "SESSION01", "5")

        assert path == cache.cache_dir / "SUBJ001" / "SESSION01" / "5.jpg"

    def test_relative_path_structure(self, cache):
        """XNAT relative path follows subject/session/scan_id.jpg layout."""
        rel = cache.get_relative_path_xnat("SUBJ001", "SESSION01", "5")

        assert rel == "SUBJ001/SESSION01/5.jpg"

    def test_special_characters_sanitized(self, cache):
        """Characters unsafe for filenames are replaced with underscores."""
        path = cache.get_path_for_xnat("SUBJ/001", "SESSION:01", "scan 5")

        assert "SUBJ_001" in str(path)
        assert "SESSION_01" in str(path)
        assert "scan_5.jpg" in str(path)

    def test_xnat_thumbnail_round_trip(self, cache):
        """Thumbnails saved at XNAT paths can be read back."""
        xnat_path = cache.get_path_for_xnat("SUBJ001", "SESSION01", "5")
        xnat_path.parent.mkdir(parents=True, exist_ok=True)
        xnat_path.write_bytes(b"xnat-thumb")

        rel = cache.get_relative_path_xnat("SUBJ001", "SESSION01", "5")
        assert cache.get_thumbnail_bytes(rel) == b"xnat-thumb"


class TestCleanupOrphaned:
    """Tests for cleanup_orphaned."""

    def test_removes_orphaned_hash_based(self, cache):
        """Hash-based thumbnails not in valid_paths are deleted."""
        keep_path = cache.save_thumbnail("1.2.3.keep", b"keep")
        orphan_path = cache.save_thumbnail("1.2.3.orphan", b"orphan")

        removed = cache.cleanup_orphaned({keep_path})

        assert removed == 1
        assert cache.get_thumbnail_bytes(keep_path) == b"keep"
        assert cache.get_thumbnail_bytes(orphan_path) is None

    def test_removes_orphaned_xnat_style(self, cache):
        """XNAT-style thumbnails not in valid_paths are deleted."""
        xnat_path = cache.get_path_for_xnat("SUBJ", "SESS", "1")
        xnat_path.parent.mkdir(parents=True, exist_ok=True)
        xnat_path.write_bytes(b"xnat-data")
        cache.get_relative_path_xnat("SUBJ", "SESS", "1")

        removed = cache.cleanup_orphaned(set())

        assert removed == 1
        assert not xnat_path.exists()

    def test_keeps_valid_xnat_and_hash(self, cache):
        """Both hash-based and XNAT-style valid paths are preserved."""
        hash_rel = cache.save_thumbnail("1.2.3.keep", b"hash-data")

        xnat_path = cache.get_path_for_xnat("SUBJ", "SESS", "2")
        xnat_path.parent.mkdir(parents=True, exist_ok=True)
        xnat_path.write_bytes(b"xnat-data")
        xnat_rel = cache.get_relative_path_xnat("SUBJ", "SESS", "2")

        removed = cache.cleanup_orphaned({hash_rel, xnat_rel})

        assert removed == 0
        assert cache.get_thumbnail_bytes(hash_rel) == b"hash-data"
        assert cache.get_thumbnail_bytes(xnat_rel) == b"xnat-data"

    def test_removes_empty_directories(self, cache):
        """Empty subdirectories are cleaned up after orphan removal."""
        orphan_path = cache.save_thumbnail("1.2.3.orphan", b"orphan")
        orphan_dir = (cache.cache_dir / orphan_path).parent

        cache.cleanup_orphaned(set())

        assert not orphan_dir.exists()


class TestGetAllPaths:
    """Tests for get_all_paths."""

    def test_finds_hash_based_paths(self, cache):
        """Hash-based .jpg paths are included."""
        rel = cache.save_thumbnail("1.2.3.4.5", b"data")
        all_paths = cache.get_all_paths()

        assert rel in all_paths

    def test_finds_xnat_style_paths(self, cache):
        """XNAT-style subject/session/scan.jpg paths are included."""
        xnat_path = cache.get_path_for_xnat("SUBJ", "SESS", "3")
        xnat_path.parent.mkdir(parents=True, exist_ok=True)
        xnat_path.write_bytes(b"data")
        xnat_rel = cache.get_relative_path_xnat("SUBJ", "SESS", "3")

        all_paths = cache.get_all_paths()

        assert xnat_rel in all_paths

    def test_finds_png_files(self, cache):
        """PNG files created by fallback conversion are included."""
        uid = "1.2.840.10008.99"
        corrupt_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
        b64_data = base64.b64encode(corrupt_png).decode()

        rel_path = cache.save_thumbnail_from_base64(uid, b64_data)
        all_paths = cache.get_all_paths()

        assert rel_path in all_paths

    def test_finds_both_types_together(self, cache):
        """Hash-based and XNAT-style paths are returned together."""
        hash_rel = cache.save_thumbnail("1.2.3.hash", b"hash")

        xnat_path = cache.get_path_for_xnat("SUBJ", "SESS", "7")
        xnat_path.parent.mkdir(parents=True, exist_ok=True)
        xnat_path.write_bytes(b"xnat")
        xnat_rel = cache.get_relative_path_xnat("SUBJ", "SESS", "7")

        all_paths = cache.get_all_paths()

        assert hash_rel in all_paths
        assert xnat_rel in all_paths
        assert len(all_paths) == 2

    def test_empty_cache_returns_empty_set(self, cache):
        """Empty cache returns an empty set."""
        assert cache.get_all_paths() == set()
