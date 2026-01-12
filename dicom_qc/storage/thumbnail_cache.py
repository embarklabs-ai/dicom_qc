"""Disk-based thumbnail storage.

Stores thumbnails as JPEG files on disk instead of base64 strings in memory.
This reduces memory usage from ~1-5GB to ~50MB for 100K series.
"""

import base64
import hashlib
from io import BytesIO
from pathlib import Path
from typing import Optional, Set


class ThumbnailCache:
    """Manages thumbnail files on disk.

    Thumbnails are stored as JPEG files organized by hash prefix for
    filesystem performance with large numbers of files.

    Directory structure:
        cache_dir/
            ab/
                abc123def456.jpg
            cd/
                cde789ghi012.jpg
            ...

    Example:
        cache = ThumbnailCache(Path("/data/_dicom_qc/thumbnails"))
        path = cache.save_thumbnail_from_base64("1.2.3.4.5", png_base64)
        jpg_base64 = cache.get_thumbnail_base64(path)
    """

    def __init__(self, cache_dir: Path):
        """Initialize thumbnail cache.

        Args:
            cache_dir: Directory for storing thumbnail files.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_path_for_series(self, series_uid: str) -> Path:
        """Get full thumbnail path for a series UID.

        Args:
            series_uid: DICOM SeriesInstanceUID

        Returns:
            Path where thumbnail should be stored.
        """
        thumb_hash = hashlib.sha256(series_uid.encode()).hexdigest()[:16]
        return self.cache_dir / thumb_hash[:2] / f"{thumb_hash}.jpg"

    def get_relative_path(self, series_uid: str) -> str:
        """Get relative path for database storage.

        Args:
            series_uid: DICOM SeriesInstanceUID

        Returns:
            Relative path string (e.g., "ab/abc123def456.jpg")
        """
        thumb_hash = hashlib.sha256(series_uid.encode()).hexdigest()[:16]
        return f"{thumb_hash[:2]}/{thumb_hash}.jpg"

    def get_path_for_xnat(self, subject: str, session: str, scan_id: str) -> Path:
        """Get human-readable thumbnail path for XNAT scan.

        Args:
            subject: XNAT subject label
            session: XNAT session label
            scan_id: XNAT scan ID

        Returns:
            Path like cache_dir/subject/session/scan_id.jpg
        """
        # Sanitize names for filesystem safety
        def sanitize(name: str) -> str:
            # Replace problematic characters with underscore
            return "".join(c if c.isalnum() or c in '-_' else '_' for c in str(name))

        safe_subject = sanitize(subject)
        safe_session = sanitize(session)
        safe_scan = sanitize(scan_id)
        return self.cache_dir / safe_subject / safe_session / f"{safe_scan}.jpg"

    def get_relative_path_xnat(self, subject: str, session: str, scan_id: str) -> str:
        """Get human-readable relative path for XNAT scan.

        Args:
            subject: XNAT subject label
            session: XNAT session label
            scan_id: XNAT scan ID

        Returns:
            Relative path string (e.g., "SUBJ001/SESSION01/5.jpg")
        """
        def sanitize(name: str) -> str:
            return "".join(c if c.isalnum() or c in '-_' else '_' for c in str(name))

        safe_subject = sanitize(subject)
        safe_session = sanitize(session)
        safe_scan = sanitize(scan_id)
        return f"{safe_subject}/{safe_session}/{safe_scan}.jpg"

    def exists(self, series_uid: str) -> bool:
        """Check if thumbnail exists for a series.

        Args:
            series_uid: DICOM SeriesInstanceUID

        Returns:
            True if thumbnail file exists.
        """
        return self.get_path_for_series(series_uid).exists()

    def exists_by_path(self, relative_path: str) -> bool:
        """Check if thumbnail exists by relative path.

        Args:
            relative_path: Relative path from database (e.g., "ab/abc123.jpg")

        Returns:
            True if thumbnail file exists.
        """
        return (self.cache_dir / relative_path).exists()

    def save_thumbnail(self, series_uid: str, image_data: bytes) -> str:
        """Save thumbnail data, return relative path.

        Args:
            series_uid: DICOM SeriesInstanceUID
            image_data: Raw JPEG image bytes

        Returns:
            Relative path for database storage.
        """
        path = self.get_path_for_series(series_uid)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(image_data)
        return self.get_relative_path(series_uid)

    def save_thumbnail_from_base64(
        self,
        series_uid: str,
        base64_data: str,
        convert_png_to_jpeg: bool = True,
        jpeg_quality: int = 85,
    ) -> str:
        """Save thumbnail from base64 string, converting PNG to JPEG.

        Used for migrating existing base64 thumbnails to disk.

        Args:
            series_uid: DICOM SeriesInstanceUID
            base64_data: Base64-encoded image (PNG or JPEG)
            convert_png_to_jpeg: If True, convert PNG to JPEG
            jpeg_quality: JPEG quality (1-100)

        Returns:
            Relative path for database storage.
        """
        try:
            raw_data = base64.b64decode(base64_data)
        except Exception:
            # Invalid base64 - skip
            return None

        # Check if it's PNG (starts with PNG magic bytes)
        is_png = raw_data[:8] == b'\x89PNG\r\n\x1a\n'

        if is_png and convert_png_to_jpeg:
            try:
                from PIL import Image
                img = Image.open(BytesIO(raw_data))
                # Convert to RGB (JPEG doesn't support alpha)
                if img.mode in ('RGBA', 'LA', 'P'):
                    # Create black background for transparency
                    background = Image.new('RGB', img.size, (0, 0, 0))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')

                buf = BytesIO()
                img.save(buf, format='JPEG', quality=jpeg_quality)
                raw_data = buf.getvalue()
            except ImportError:
                # PIL not available - save as PNG
                pass
            except Exception:
                # Conversion failed - save as PNG
                pass

        return self.save_thumbnail(series_uid, raw_data)

    def get_thumbnail_bytes(self, relative_path: str) -> Optional[bytes]:
        """Load thumbnail bytes from disk.

        Args:
            relative_path: Relative path from database

        Returns:
            Raw image bytes or None if not found.
        """
        full_path = self.cache_dir / relative_path
        if full_path.exists():
            return full_path.read_bytes()
        return None

    def get_thumbnail_base64(self, relative_path: str) -> Optional[str]:
        """Load thumbnail and return as base64.

        Args:
            relative_path: Relative path from database

        Returns:
            Base64-encoded image string or None if not found.
        """
        data = self.get_thumbnail_bytes(relative_path)
        if data:
            return base64.b64encode(data).decode()
        return None

    def delete_thumbnail(self, series_uid: str) -> bool:
        """Delete thumbnail for a series.

        Args:
            series_uid: DICOM SeriesInstanceUID

        Returns:
            True if file was deleted, False if not found.
        """
        path = self.get_path_for_series(series_uid)
        if path.exists():
            path.unlink()
            return True
        return False

    def cleanup_orphaned(self, valid_paths: Set[str]) -> int:
        """Remove thumbnails not in valid_paths set.

        Useful for cleaning up thumbnails for deleted series.

        Args:
            valid_paths: Set of relative paths that should be kept

        Returns:
            Number of files deleted.
        """
        removed = 0
        for subdir in self.cache_dir.iterdir():
            if subdir.is_dir() and len(subdir.name) == 2:
                for thumb in subdir.iterdir():
                    if thumb.is_file():
                        rel_path = f"{subdir.name}/{thumb.name}"
                        if rel_path not in valid_paths:
                            thumb.unlink()
                            removed += 1
                # Remove empty subdirectory
                try:
                    subdir.rmdir()
                except OSError:
                    pass  # Not empty
        return removed

    def get_all_paths(self) -> Set[str]:
        """Get all thumbnail relative paths in cache.

        Returns:
            Set of relative paths.
        """
        paths = set()
        for subdir in self.cache_dir.iterdir():
            if subdir.is_dir() and len(subdir.name) == 2:
                for thumb in subdir.iterdir():
                    if thumb.is_file():
                        paths.add(f"{subdir.name}/{thumb.name}")
        return paths

    def get_cache_size_bytes(self) -> int:
        """Get total size of all thumbnails in bytes.

        Returns:
            Total bytes used by thumbnail files.
        """
        total = 0
        for subdir in self.cache_dir.iterdir():
            if subdir.is_dir():
                for thumb in subdir.iterdir():
                    if thumb.is_file():
                        total += thumb.stat().st_size
        return total

    def get_cache_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dict with count, total_bytes, avg_bytes.
        """
        count = 0
        total_bytes = 0
        for subdir in self.cache_dir.iterdir():
            if subdir.is_dir():
                for thumb in subdir.iterdir():
                    if thumb.is_file():
                        count += 1
                        total_bytes += thumb.stat().st_size

        return {
            'count': count,
            'total_bytes': total_bytes,
            'total_mb': total_bytes / (1024 * 1024),
            'avg_bytes': total_bytes // count if count > 0 else 0,
        }
