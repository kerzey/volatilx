"""Numeric helper utilities."""

from __future__ import annotations

from typing import Any, Optional

__all__ = ["safe_float"]


def safe_float(value: Any) -> Optional[float]:
    """Best-effort conversion to float returning None on failure."""

    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
