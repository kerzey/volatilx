"""Symbol sanitization and market lookup helpers."""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Set

from symbol_map import get_ticker_map

_SANITIZE_PATTERN = re.compile(r"[^A-Z0-9\.\-]")
_CANONICAL_PATTERN = re.compile(r"[^A-Z0-9]")


def sanitize_symbol(value: str) -> str:
    """Normalize raw ticker input into an uppercase alphanumeric token."""

    cleaned = (value or "").strip().upper()
    if not cleaned:
        return cleaned
    sanitized = _SANITIZE_PATTERN.sub("", cleaned)
    return sanitized[:24]


def report_symbol_candidates(symbol: str) -> List[str]:
    """Generate storage-friendly lookup candidates for a symbol."""

    base = (symbol or "").strip().upper()
    sanitized = sanitize_symbol(base)
    candidates: List[str] = []

    def _push(candidate: Optional[str]) -> None:
        if not candidate:
            return
        normalised = candidate.strip().upper()
        if not normalised:
            return
        candidates.append(normalised)

    _push(base.replace("/", "-").replace(" ", "_"))
    _push(sanitized)

    for market in ("equity", "crypto"):
        try:
            ticker_map = get_ticker_map(market)
        except ValueError:
            continue
        lookup = {sanitize_symbol(ticker): ticker for ticker in ticker_map.keys()}
        canonical = lookup.get(sanitized)
        if not canonical and base:
            canonical = lookup.get(sanitize_symbol(base))
        if canonical:
            azure_symbol = canonical.upper().replace("/", "-").replace(" ", "_")
            _push(azure_symbol)
            _push(sanitize_symbol(canonical))

    unique_candidates = list(dict.fromkeys(candidates))
    return unique_candidates


def filter_symbols_for_market(symbols: Iterable[str], market: str) -> Set[str]:
    """Return sanitized symbols that exist for the requested market."""

    try:
        ticker_map = get_ticker_map(market)
    except ValueError:
        return set()

    lookup = {sanitize_symbol(ticker): ticker for ticker in ticker_map.keys()}
    filtered: Set[str] = set()
    for raw_symbol in symbols:
        candidate = sanitize_symbol(raw_symbol)
        if candidate and candidate in lookup:
            filtered.add(candidate)
    return filtered


def canonicalize_symbol(value: Optional[str]) -> str:
    """Produce a comparison-friendly key for the provided symbol."""

    if not value:
        return ""

    sanitized = sanitize_symbol(value)
    if not sanitized:
        return ""

    return _CANONICAL_PATTERN.sub("", sanitized)
