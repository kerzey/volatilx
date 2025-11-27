"""Utilities for working with support/resistance level labels."""

from __future__ import annotations

from typing import Optional

_LEVEL_DISPLAY_OVERRIDES = {
    "S1": "Support 1",
    "S2": "Support 2",
    "R1": "Resistance 1",
    "R2": "Resistance 2",
}

_LEVEL_DISPLAY_PREFIXES = {
    "S": "Support",
    "R": "Resistance",
}


def humanize_level_label(label: Optional[str]) -> Optional[str]:
    """Convert compact level codes like S1/S2 into human-readable labels."""

    if label is None:
        return None

    text = str(label).strip()
    if not text:
        return text

    upper = text.upper()
    if upper == "NONE":
        return "None"

    if upper in _LEVEL_DISPLAY_OVERRIDES:
        return _LEVEL_DISPLAY_OVERRIDES[upper]

    prefix = upper[0]
    suffix = upper[1:]
    if prefix in _LEVEL_DISPLAY_PREFIXES and suffix.isdigit():
        return f"{_LEVEL_DISPLAY_PREFIXES[prefix]} {int(suffix)}"

    return text
