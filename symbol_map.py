import csv
import logging
import os
import time
from pathlib import Path
from difflib import get_close_matches
from typing import Dict, Iterable, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

_FALLBACK_TICKERS: Dict[str, str] = {
    "AAPL": "Apple Inc.",
    "GOOG": "Google",
    "GOOGL": "Alphabet Class A",
    "MSFT": "Microsoft Corporation",
    "AMZN": "Amazon.com, Inc.",
    "TSLA": "Tesla, Inc.",
    "NVDA": "NVIDIA Corporation",
    "META": "Meta Platforms, Inc.",
    "NFLX": "Netflix, Inc.",
    "AMD": "Advanced Micro Devices, Inc.",
    "INTC": "Intel Corporation",
    "IBM": "International Business Machines Corporation",
    "ORCL": "Oracle Corporation",
    "ADBE": "Adobe Inc.",
    "CRM": "Salesforce, Inc.",
    "UBER": "Uber Technologies, Inc.",
    "LYFT": "Lyft, Inc.",
    "SHOP": "Shopify Inc.",
    "SQ": "Block, Inc.",
    "PYPL": "PayPal Holdings, Inc.",
    "SPOT": "Spotify Technology S.A.",
    "SNOW": "Snowflake Inc.",
    "PLTR": "Palantir Technologies Inc.",
    "BABA": "Alibaba Group Holding Limited",
    "BA": "The Boeing Company",
    "JPM": "JPMorgan Chase & Co.",
    "GS": "The Goldman Sachs Group, Inc.",
    "V": "Visa Inc.",
    "MA": "Mastercard Incorporated",
    "SPY": "SPDR S&P 500 ETF Trust",
    "QQQ": "Invesco QQQ Trust",
}

_ALPACA_CACHE_TTL_SECONDS = int(os.getenv("ALPACA_ASSET_CACHE_TTL_SECONDS", "21600"))
_FALLBACK_CACHE_TTL_SECONDS = int(os.getenv("ALPACA_ASSET_FALLBACK_TTL_SECONDS", "1800"))

_current_ticker_map: Dict[str, str] = {}
_ticker_catalog_expires_at: float = 0.0


def _resolve_csv_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "tickers.csv"


def _load_ticker_map_from_csv() -> Dict[str, str]:
    csv_path = _resolve_csv_path()
    mapping: Dict[str, str] = {}
    if not csv_path.exists():
        return mapping

    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                if not row or len(row) < 2:
                    continue
                ticker = row[0].strip().upper()
                company = row[1].strip()
                if not ticker or not company:
                    continue
                mapping.setdefault(ticker, company)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully on parse errors
        logger.warning("Failed to load ticker CSV at %s: %s", csv_path, exc)
        return {}

    return mapping


def _fetch_assets_from_alpaca() -> Dict[str, str]:
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    if not api_key or not secret_key:
        logger.warning("Alpaca credentials missing; symbol catalog falling back to static list")
        return {}

    url = f"{base_url.rstrip('/')}/v2/assets"
    params = {
        "status": "active",
        "asset_class": "us_equity",
    }
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=20)
        response.raise_for_status()
        assets = response.json()
    except Exception as exc:  # noqa: BLE001 - network/API errors are non-fatal
        logger.warning("Failed to fetch Alpaca asset catalog: %s", exc)
        return {}

    mapping: Dict[str, str] = {}
    for asset in assets or []:
        symbol = asset.get("symbol")
        if not symbol:
            continue
        status = str(asset.get("status", "")).lower()
        if status and status != "active":
            continue
        if asset.get("tradable") is False:
            continue
        company_name = (asset.get("name") or symbol).strip()
        mapping.setdefault(symbol.upper(), company_name)

    return mapping


def _load_ticker_map(force_refresh: bool = False) -> Dict[str, str]:
    global _current_ticker_map, _ticker_catalog_expires_at

    now = time.time()
    if not force_refresh and _current_ticker_map and now < _ticker_catalog_expires_at:
        return _current_ticker_map

    ttl = _ALPACA_CACHE_TTL_SECONDS
    mapping = _fetch_assets_from_alpaca()

    if not mapping:
        csv_mapping = _load_ticker_map_from_csv()
        if csv_mapping:
            mapping = csv_mapping
        else:
            mapping = {}
        ttl = _FALLBACK_CACHE_TTL_SECONDS
        logger.warning(
            "Using fallback symbol catalog with %s entries; check Alpaca connectivity",
            len(mapping) or len(_FALLBACK_TICKERS),
        )
    else:
        logger.info("Loaded %s symbols from Alpaca asset catalog", len(mapping))

    merged = _FALLBACK_TICKERS.copy()
    merged.update(mapping)

    _current_ticker_map = merged
    _ticker_catalog_expires_at = now + max(ttl, 60)
    return merged


def _build_name_lookup(source: Dict[str, str]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for ticker, company in source.items():
        lower_name = company.lower()
        lookup.setdefault(lower_name, ticker)
    return lookup


def refresh_symbol_catalog(force: bool = False) -> None:
    global TICKER_TO_COMPANY, NAME_TO_TICKER, ALL_LABELS

    mapping = _load_ticker_map(force_refresh=force)
    TICKER_TO_COMPANY = mapping
    NAME_TO_TICKER = _build_name_lookup(mapping)
    ALL_LABELS = list(TICKER_TO_COMPANY.keys()) + list(NAME_TO_TICKER.keys())


def _ensure_catalog_current() -> None:
    if time.time() >= _ticker_catalog_expires_at:
        refresh_symbol_catalog()


TICKER_TO_COMPANY: Dict[str, str] = {}
NAME_TO_TICKER: Dict[str, str] = {}
ALL_LABELS: Iterable[str] = []
refresh_symbol_catalog(force=True)


class SymbolNotFound(Exception):
    def __init__(self, raw_symbol: str, suggestion: Optional[str] = None) -> None:
        self.raw_symbol = raw_symbol
        self.suggestion = suggestion
        message = (
            f"Unknown symbol '{raw_symbol}'."
            + (f" Did you mean {suggestion}?" if suggestion else "")
        )
        super().__init__(message)


def normalize_symbol(raw_symbol: Optional[str]) -> Tuple[str, Optional[str]]:
    _ensure_catalog_current()

    if raw_symbol is None:
        raise SymbolNotFound("", None)

    stripped = raw_symbol.strip()
    if not stripped:
        raise SymbolNotFound(raw_symbol or "", None)

    upper_symbol = stripped.upper()
    if upper_symbol in TICKER_TO_COMPANY:
        return upper_symbol, None

    lower_symbol = stripped.lower()
    if lower_symbol in NAME_TO_TICKER:
        ticker = NAME_TO_TICKER[lower_symbol]
        company = TICKER_TO_COMPANY[ticker]
        return ticker, f"Using {company} ({ticker})."

    matches = get_close_matches(lower_symbol, ALL_LABELS, n=1, cutoff=0.6)
    if matches:
        match = matches[0]
        if match in TICKER_TO_COMPANY:
            ticker = match
        else:
            ticker = NAME_TO_TICKER[match]
        company = TICKER_TO_COMPANY[ticker]
        return ticker, f"Did you mean {company} ({ticker})? Running analysis with that ticker."

    ticker_matches = get_close_matches(upper_symbol, TICKER_TO_COMPANY.keys(), n=1, cutoff=0.4)
    suggestion = ticker_matches[0] if ticker_matches else None
    raise SymbolNotFound(raw_symbol, suggestion)


def get_symbol_catalog() -> List[Dict[str, str]]:
    _ensure_catalog_current()

    return [
        {
            "ticker": ticker,
            "company": company,
            "label": f"{ticker} - {company}",
        }
        for ticker, company in sorted(TICKER_TO_COMPANY.items())
    ]
