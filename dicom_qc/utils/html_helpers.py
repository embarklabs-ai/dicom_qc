"""Shared HTML helper functions."""


def _defaced_badge_html(extra_style: str = "") -> str:
    """Return the purple DEFACED badge HTML span.

    Args:
        extra_style: Additional inline CSS (padding, border-radius, positioning, etc.).
    """
    base = "background:#7c3aed;color:#fff;font-size:10px;font-weight:600;"
    style = base + extra_style
    return f'<span style="{style}">DEFACED</span>'
