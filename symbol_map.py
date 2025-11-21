import csv
import logging
import os
import time
from pathlib import Path
from difflib import get_close_matches
from typing import Dict, Iterable, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

_EQUITY_FALLBACK_TICKERS: Dict[str, str] = {
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

_CRYPTO_FALLBACK_TICKERS: Dict[str, str] = {
    "BTC/USD": "Bitcoin",
    "ETH/USD": "Ethereum",
    "SOL/USD": "Solana",
    "ADA/USD": "Cardano",
    "DOGE/USD": "Dogecoin",
    "LTC/USD": "Litecoin",
    "XRP/USD": "XRP",
    "AVAX/USD": "Avalanche",
    "DOT/USD": "Polkadot",
    "LINK/USD": "Chainlink",
}

_ALPACA_CACHE_TTL_SECONDS = int(os.getenv("ALPACA_ASSET_CACHE_TTL_SECONDS", "21600"))
_FALLBACK_CACHE_TTL_SECONDS = int(os.getenv("ALPACA_ASSET_FALLBACK_TTL_SECONDS", "1800"))

_DEFAULT_MARKET = "equity"
_SUPPORTED_MARKETS: Tuple[str, ...] = ("equity", "crypto")
_ASSET_CLASS_BY_MARKET: Dict[str, str] = {
    "equity": "us_equity",
    "crypto": "crypto",
}
_FALLBACK_BY_MARKET: Dict[str, Dict[str, str]] = {
    "equity": _EQUITY_FALLBACK_TICKERS,
    "crypto": _CRYPTO_FALLBACK_TICKERS,
}

_market_ticker_map: Dict[str, Dict[str, str]] = {market: {} for market in _SUPPORTED_MARKETS}
_market_name_lookup: Dict[str, Dict[str, str]] = {market: {} for market in _SUPPORTED_MARKETS}
_market_labels: Dict[str, List[str]] = {market: [] for market in _SUPPORTED_MARKETS}
_market_expiry: Dict[str, float] = {market: 0.0 for market in _SUPPORTED_MARKETS}

# Public constants for other modules
DEFAULT_MARKET = _DEFAULT_MARKET
SUPPORTED_MARKETS = _SUPPORTED_MARKETS


def _normalize_market_key(market: Optional[str]) -> str:
    candidate = (market or _DEFAULT_MARKET).strip().lower()
    if candidate not in _SUPPORTED_MARKETS:
        raise ValueError(f"Unsupported market '{market}'. Expected one of {_SUPPORTED_MARKETS}.")
    return candidate


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


def _fetch_assets_from_alpaca(market: str) -> Dict[str, str]:
    market_key = _normalize_market_key(market)
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    if not api_key or not secret_key:
        logger.warning(
            "Alpaca credentials missing; symbol catalog falling back to static list for market '%s'",
            market_key,
        )
        return {}

    url = f"{base_url.rstrip('/')}/v2/assets"
    params = {
        "status": "active",
    }
    asset_class = _ASSET_CLASS_BY_MARKET.get(market_key)
    if asset_class:
        params["asset_class"] = asset_class
    if market_key == "crypto":
        params.pop("status", None)
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=20)
        response.raise_for_status()
        assets = response.json()
    except Exception as exc:  # noqa: BLE001 - network/API errors are non-fatal
        logger.warning("Failed to fetch Alpaca asset catalog for %s: %s", market_key, exc)
        return {}

    mapping: Dict[str, str] = {}
    for asset in assets or []:
        symbol = asset.get("symbol")
        if not symbol:
            continue
        status = str(asset.get("status", "")).lower()
        if market_key != "crypto" and status and status != "active":
            continue
        if asset.get("tradable") is False:
            continue
        company_name = (asset.get("name") or symbol).strip()
        mapping.setdefault(symbol.upper(), company_name)

    return mapping


def _load_ticker_map(market: str, force_refresh: bool = False) -> Dict[str, str]:
    market_key = _normalize_market_key(market)
    now = time.time()

    existing = _market_ticker_map.get(market_key, {})
    expires_at = _market_expiry.get(market_key, 0.0)
    if not force_refresh and existing and now < expires_at:
        return existing

    ttl = _ALPACA_CACHE_TTL_SECONDS
    mapping = _fetch_assets_from_alpaca(market_key)

    if not mapping:
        if market_key == "equity":
            csv_mapping = _load_ticker_map_from_csv()
            if csv_mapping:
                mapping = csv_mapping
        ttl = _FALLBACK_CACHE_TTL_SECONDS
        logger.warning(
            "Using fallback symbol catalog for %s with %s entries; check Alpaca connectivity",
            market_key,
            len(mapping) or len(_FALLBACK_BY_MARKET[market_key]),
        )
    else:
        logger.info(
            "Loaded %s symbols from Alpaca asset catalog for market '%s'",
            len(mapping),
            market_key,
        )

    merged = _FALLBACK_BY_MARKET[market_key].copy()
    merged.update(mapping)

    _market_ticker_map[market_key] = merged
    name_lookup = _build_name_lookup(merged)
    _market_name_lookup[market_key] = name_lookup
    _market_labels[market_key] = list(merged.keys()) + list(name_lookup.keys())
    _market_expiry[market_key] = now + max(ttl, 60)
    return merged


def _build_name_lookup(source: Dict[str, str]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for ticker, company in source.items():
        lower_name = company.lower()
        lookup.setdefault(lower_name, ticker)
    return lookup


def refresh_symbol_catalog(force: bool = False, market: Optional[str] = None) -> None:
    global TICKER_TO_COMPANY, NAME_TO_TICKER, ALL_LABELS

    targets = (
        [_normalize_market_key(market)] if market is not None else list(_SUPPORTED_MARKETS)
    )
    for market_key in targets:
        mapping = _load_ticker_map(market_key, force_refresh=force)
        # _load_ticker_map already updates lookup caches
        if market_key == _DEFAULT_MARKET:
            TICKER_TO_COMPANY = _market_ticker_map[market_key]
            NAME_TO_TICKER = _market_name_lookup[market_key]
            ALL_LABELS = _market_labels[market_key]


def _ensure_catalog_current(market: Optional[str] = None) -> None:
    market_key = _normalize_market_key(market)
    if time.time() >= _market_expiry.get(market_key, 0.0) or not _market_ticker_map[market_key]:
        refresh_symbol_catalog(market=market_key)


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


def normalize_symbol(raw_symbol: Optional[str], market: Optional[str] = None) -> Tuple[str, Optional[str]]:
    market_key = _normalize_market_key(market)
    _ensure_catalog_current(market_key)

    mapping = _market_ticker_map[market_key]
    name_lookup = _market_name_lookup[market_key]
    labels = _market_labels[market_key]

    if raw_symbol is None:
        raise SymbolNotFound("", None)

    stripped = raw_symbol.strip()
    if not stripped:
        raise SymbolNotFound(raw_symbol or "", None)

    upper_symbol = stripped.upper()
    if upper_symbol in mapping:
        return upper_symbol, None

    lower_symbol = stripped.lower()
    if lower_symbol in name_lookup:
        ticker = name_lookup[lower_symbol]
        company = mapping[ticker]
        return ticker, f"Using {company} ({ticker})."

    matches = get_close_matches(lower_symbol, labels, n=1, cutoff=0.6)
    if matches:
        match = matches[0]
        if match in mapping:
            ticker = match
        else:
            ticker = name_lookup[match]
        company = mapping[ticker]
        return ticker, f"Did you mean {company} ({ticker})? Running analysis with that ticker."

    ticker_matches = get_close_matches(upper_symbol, mapping.keys(), n=1, cutoff=0.4)
    suggestion = ticker_matches[0] if ticker_matches else None
    raise SymbolNotFound(raw_symbol, suggestion)


def get_symbol_catalog(market: Optional[str] = None) -> List[Dict[str, str]]:
    market_key = _normalize_market_key(market)
    _ensure_catalog_current(market_key)

    mapping = _market_ticker_map[market_key]
    return [
        {
            "ticker": ticker,
            "company": company,
            "label": f"{ticker} - {company}",
        }
        for ticker, company in sorted(mapping.items())
    ]


def get_all_symbol_catalogs() -> Dict[str, List[Dict[str, str]]]:
    return {market: get_symbol_catalog(market) for market in _SUPPORTED_MARKETS}


def get_ticker_map(market: Optional[str] = None) -> Dict[str, str]:
    market_key = _normalize_market_key(market)
    _ensure_catalog_current(market_key)
    return _market_ticker_map[market_key]
