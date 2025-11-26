import asyncio
import logging
import math
import os
import time
from functools import partial
import uuid
import re
import smtplib
from email.message import EmailMessage
import html

import aiohttp

try:
    from azure.communication.email import EmailClient
except ImportError:  # pragma: no cover - optional dependency
    EmailClient = None  # type: ignore[assignment]
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form, status, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from authlib.integrations.starlette_client import OAuth
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware
# from starlette.middleware.trustedhost import ProxyHeadersMiddleware
# from starlette.middleware.trustedhost import ForwardedMiddleware
from starlette.responses import RedirectResponse
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Set, Tuple

# Import user-related components - UPDATED to include get_user_manager
from user import (
    User,
    UserRead,
    UserCreate,
    UserUpdate,
    get_user_db,
    get_user_manager,
    init_db,
    get_current_user_sync,
)
from db import SessionLocal
from billing import (
    create_subscription_checkout_session,
    ensure_customer,
    get_publishable_key,
    sync_plan_catalogue,
    StripeWebhookConfig,
    enqueue_usage,
    handle_checkout_session_completed,
    handle_invoice_paid,
    handle_subscription_deleted,
    handle_subscription_updated,
    parse_event,
)
from models import SubscriptionPlan, UserSubscription, UserFavoriteSymbol
from sqlalchemy.orm import Session, joinedload
from pydantic import BaseModel, Field

# FastAPI Users imports
from fastapi_users import FastAPIUsers
from fastapi_users.authentication import CookieTransport, AuthenticationBackend
from fastapi_users.authentication.strategy import JWTStrategy
from fastapi_users.password import PasswordHelper

# Backend imports
from day_trading_agent import MultiSymbolDayTraderAgent
from indicator_fetcher import ComprehensiveMultiTimeframeAnalyzer
import json
import urllib.parse

#OpenAI imports
# from ai_agents.openai_service import OpenAIAnalysisService
import jwt
import hashlib
from ai_agents.openai_service import openai_service
from ai_agents.principal_agent import PrincipalAgent
from ai_agents.price_action import PriceActionAnalyzer
from symbol_map import (
    SymbolNotFound,
    SUPPORTED_MARKETS,
    get_all_symbol_catalogs,
    get_ticker_map,
    normalize_symbol,
)
from azure_storage import store_ai_report, fetch_reports_for_date
from market_data.live_price import LivePriceStream
from market_data.live_crypto import LiveCryptoStream

load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")


logger = logging.getLogger(__name__)

AI_ANALYSIS_TIMEOUT = float(os.getenv("AI_ANALYSIS_TIMEOUT_SECONDS", "45"))

ACTIVE_SUBSCRIPTION_STATUSES = {"active", "trialing", "past_due"}
TRIAL_PLAN_SLUG = os.getenv("TRIAL_PLAN_SLUG", "trial")
TRIAL_DURATION_DAYS = int(os.getenv("TRIAL_DURATION_DAYS", "30"))

STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
stripe_webhook_config = (
    StripeWebhookConfig(signing_secret=STRIPE_WEBHOOK_SECRET)
    if STRIPE_WEBHOOK_SECRET
    else None
)

# In-memory store for AI analysis results (for demo; use Redis/db for production)
# In-memory store for AI analysis results (for demo; use Redis/db for production)
ai_analysis_jobs = {}

REPORT_CENTER_ALLOWED_PLAN_SLUGS = {"sigma", "omega"}
DEFAULT_REPORT_CENTER_REPORT_LIMIT = int(
    os.getenv("REPORT_CENTER_MAX_REPORTS", os.getenv("DASHBOARD_MAX_REPORTS", "120"))
)
DEFAULT_ACTION_CENTER_SYMBOL = os.getenv("ACTION_CENTER_DEFAULT_SYMBOL", "SPY").upper()
ACTION_CENTER_LOOKBACK_DAYS = int(os.getenv("ACTION_CENTER_LOOKBACK_DAYS", "3"))
CONTACT_EMAIL_RECIPIENT = os.getenv("CONTACT_EMAIL_RECIPIENT")
ALPACA_DATA_REST_URL = os.getenv("ALPACA_DATA_REST_URL", "https://data.alpaca.markets").rstrip("/")
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").strip().lower() not in {"0", "false", "no"}
CONTACT_EMAIL_PREFIX = os.getenv("CONTACT_EMAIL_SUBJECT_PREFIX", "VolatilX Contact")
CONTACT_EMAIL_SENDER = os.getenv("CONTACT_EMAIL_SENDER") or SMTP_USERNAME or CONTACT_EMAIL_RECIPIENT
ACS_CONNECTION_STRING = os.getenv("ACS_CONNECTION_STRING")
ACS_EMAIL_SENDER = os.getenv("ACS_EMAIL_SENDER") or CONTACT_EMAIL_SENDER
ENABLE_LIVE_PRICE_STREAM = os.getenv("ENABLE_LIVE_PRICE_STREAM", "true").strip().lower() not in {"0", "false", "no"}
ENABLE_LIVE_CRYPTO_STREAM = os.getenv("ENABLE_LIVE_CRYPTO_STREAM", "true").strip().lower() not in {"0", "false", "no"}
DEFAULT_CRYPTO_STREAM_SYMBOL = os.getenv("CRYPTO_CENTER_DEFAULT_SYMBOL", "BTCUSD").strip().upper() or "BTCUSD"
ALPACA_CRYPTO_REST_MARKET = os.getenv("ALPACA_CRYPTO_REST_MARKET", os.getenv("ALPACA_CRYPTO_MARKET", "us")).strip().lower() or "us"
ALPACA_CRYPTO_REST_FEED = os.getenv("ALPACA_CRYPTO_REST_FEED", os.getenv("ALPACA_CRYPTO_FEED", "us")).strip().lower() or "us"


def _build_base_url(request: Request) -> str:
    """Best-effort reconstruction of the public base URL for redirects."""

    forwarded_host = request.headers.get("x-forwarded-host")
    forwarded_proto = request.headers.get("x-forwarded-proto")
    if forwarded_host:
        scheme = forwarded_proto or request.url.scheme
        return f"{scheme}://{forwarded_host}"

    host = request.headers.get("host") or request.url.netloc
    scheme = forwarded_proto or request.url.scheme
    return f"{scheme}://{host}"


def _serialize_plan(plan: SubscriptionPlan) -> Dict[str, Any]:
    return {
        "id": plan.id,
        "slug": plan.slug,
        "name": plan.name,
        "description": plan.description,
        "monthly_price_cents": plan.monthly_price_cents,
        "monthly_price_dollars": plan.monthly_price_cents / 100,
        "ai_runs_included": plan.ai_runs_included,
        "is_active": plan.is_active,
        "stripe_price_configured": bool(plan.stripe_price_id),
    }


def _serialize_subscription(subscription: Optional[UserSubscription]) -> Optional[Dict[str, Any]]:
    if subscription is None:
        return None

    plan_payload = _serialize_plan(subscription.plan) if subscription.plan else None
    return {
        "id": subscription.id,
        "status": subscription.status,
        "runs_remaining": subscription.runs_remaining,
        "auto_renew": subscription.auto_renew,
        "cancel_at_period_end": subscription.cancel_at_period_end,
        "current_period_start": subscription.current_period_start.isoformat()
        if subscription.current_period_start
        else None,
        "current_period_end": subscription.current_period_end.isoformat()
        if subscription.current_period_end
        else None,
        "plan": plan_payload,
    }


def _find_relevant_subscription(session: Session, user_id: int) -> Optional[UserSubscription]:
    query = (
        session.query(UserSubscription)
        .options(joinedload(UserSubscription.plan))
        .filter(UserSubscription.user_id == user_id)
        .order_by(UserSubscription.created_at.desc())
    )
    subscriptions = query.all()
    for sub in subscriptions:
        if sub.status in ACTIVE_SUBSCRIPTION_STATUSES:
            return sub
    return subscriptions[0] if subscriptions else None


def _get_subscription_for_user(user_id: int) -> Optional[UserSubscription]:
    with SessionLocal() as session:
        return _find_relevant_subscription(session, user_id)


def _check_report_center_access(user: User) -> Tuple[Optional[UserSubscription], bool]:
    subscription = _get_subscription_for_user(user.id)
    allowed = bool(
        subscription
        and subscription.plan
        and subscription.plan.slug
        and subscription.plan.slug.lower() in REPORT_CENTER_ALLOWED_PLAN_SLUGS
    )
    return subscription, allowed


def _sanitize_symbol(value: str) -> str:
    cleaned = (value or "").strip().upper()
    if not cleaned:
        return cleaned
    return re.sub(r"[^A-Z0-9\.\-]", "", cleaned)[:24]


def _is_symbol_favorited(session: Session, user_id: int, symbol: str) -> bool:
    if not symbol:
        return False
    return (
        session.query(UserFavoriteSymbol)
        .filter(UserFavoriteSymbol.user_id == user_id, UserFavoriteSymbol.symbol == symbol)
        .first()
        is not None
    )


def _filter_symbols_for_market(symbols: Iterable[str], market: str) -> Set[str]:
    try:
        ticker_map = get_ticker_map(market)
    except ValueError:
        return set()

    lookup = {_sanitize_symbol(ticker): ticker for ticker in ticker_map.keys()}
    filtered: Set[str] = set()
    for raw_symbol in symbols:
        candidate = _sanitize_symbol(raw_symbol)
        if candidate and candidate in lookup:
            filtered.add(candidate)
    return filtered


_STRATEGY_SUPPORT_KEYWORDS = (
    "support",
    "floor",
    "base",
    "bounce",
)
_STRATEGY_RESISTANCE_KEYWORDS = (
    "resistance",
    "ceiling",
    "cap",
    "lid",
)
_STRATEGY_SUPPORT_HINTS = (
    "drop",
    "below",
    "falls",
    "fell",
    "lose",
    "loss",
    "fails",
    "failure",
    "downside",
    "pullback",
    "slide",
    "slip",
    "pressure",
    "weakness",
)
_STRATEGY_RESISTANCE_HINTS = (
    "above",
    "break",
    "reclaim",
    "rally",
    "upside",
    "advance",
    "surge",
    "target",
    "push",
    "hold above",
    "close above",
    "stall under",
    "remains under",
    "capped",
)
_STRATEGY_RANGE_HINTS = (
    "between",
    "range",
    "band",
    "zone",
    "channel",
    "box",
    "chop",
    "balanced",
)

_PRICE_NUMBER_PATTERN = r"\d+(?:,\d{3})*(?:\.\d+)?"
_PRICE_RANGE_PATTERN = re.compile(
    rf"(?P<first>{_PRICE_NUMBER_PATTERN})\s*(?:–|-|to)\s*(?P<second>{_PRICE_NUMBER_PATTERN})",
    re.UNICODE,
)
_PRICE_TOKEN_PATTERN = re.compile(_PRICE_NUMBER_PATTERN)
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[\.\!\?])\s+")


def _clean_context_snippet(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _parse_price_value(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if not value:
        return None
    text = str(value)
    match = _PRICE_TOKEN_PATTERN.search(text)
    if not match:
        return None
    token = match.group(0).replace(",", "")
    try:
        return float(token)
    except ValueError:
        return None


def _split_strategy_sentences(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    fragments: List[str] = []
    for segment in re.split(r"[\n;]+", text):
        trimmed = segment.strip()
        if not trimmed:
            continue
        sentences = _SENTENCE_SPLIT_PATTERN.split(trimmed)
        if not sentences:
            fragments.append(trimmed)
            continue
        for sentence in sentences:
            cleaned = sentence.strip(" -")
            if cleaned:
                fragments.append(cleaned)
    return fragments


def _extract_price_tokens(chunk: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    if not chunk:
        return results
    spans: List[Tuple[int, int]] = []
    for match in _PRICE_RANGE_PATTERN.finditer(chunk):
        start, end = match.span()
        spans.append((start, end))
        first = _parse_price_value(match.group("first"))
        second = _parse_price_value(match.group("second"))
        if first is not None:
            results.append({
                "price": first,
                "start": match.start("first"),
                "end": match.end("first"),
                "raw": match.group("first"),
            })
        if second is not None:
            results.append({
                "price": second,
                "start": match.start("second"),
                "end": match.end("second"),
                "raw": match.group("second"),
            })
    for match in _PRICE_TOKEN_PATTERN.finditer(chunk):
        start, end = match.span()
        if any(start >= span_start and end <= span_end for span_start, span_end in spans):
            continue
        value = _parse_price_value(match.group(0))
        if value is None:
            continue
        results.append({
            "price": value,
            "start": start,
            "end": end,
            "raw": match.group(0),
        })
    return results


def _classify_price_context(
    chunk: str,
    token: Dict[str, Any],
) -> str:
    start = token.get("start", 0)
    end = token.get("end", 0)
    window_start = max(0, start - 60)
    window_end = min(len(chunk), end + 60)
    window = chunk[window_start:window_end].lower()

    support_hits = sum(1 for word in _STRATEGY_SUPPORT_KEYWORDS if word in window)
    resistance_hits = sum(1 for word in _STRATEGY_RESISTANCE_KEYWORDS if word in window)
    if support_hits and not resistance_hits:
        return "support"
    if resistance_hits and not support_hits:
        return "resistance"
    if support_hits and resistance_hits:
        return "support" if support_hits >= resistance_hits else "resistance"

    range_hint = any(word in window for word in _STRATEGY_RANGE_HINTS)
    if range_hint:
        return "range"

    support_score = sum(window.count(word) for word in _STRATEGY_SUPPORT_HINTS)
    resistance_score = sum(window.count(word) for word in _STRATEGY_RESISTANCE_HINTS)
    if support_score > resistance_score:
        return "support"
    if resistance_score > support_score:
        return "resistance"
    return ""


def _extract_strategy_levels(
    strategy: Dict[str, Any],
    *,
    strategy_key: str,
    strategy_label: str,
    latest_price: Optional[float],
) -> Dict[str, List[Dict[str, Any]]]:
    supports: List[Dict[str, Any]] = []
    resistances: List[Dict[str, Any]] = []
    support_seen: Set[float] = set()
    resistance_seen: Set[float] = set()
    ambiguous: List[Tuple[float, str]] = []

    def _register(
        target: List[Dict[str, Any]],
        seen: Set[float],
        price: Optional[float],
        context: str,
        *,
        source: str,
    ) -> None:
        if price is None:
            return
        key = round(price, 4)
        if key in seen:
            return
        entry = {
            "price": float(price),
            "distance_pct": None,
            "timeframe": strategy_label,
            "source": source,
            "context": _clean_context_snippet(context)[:280],
        }
        target.append(entry)
        seen.add(key)

    text_chunks: List[str] = []
    summary = strategy.get("summary")
    if isinstance(summary, str):
        text_chunks.extend(_split_strategy_sentences(summary))
    next_actions = strategy.get("next_actions") or []
    if isinstance(next_actions, (list, tuple)):
        for action in next_actions:
            text_chunks.extend(_split_strategy_sentences(str(action)))
    elif isinstance(next_actions, str):
        text_chunks.extend(_split_strategy_sentences(next_actions))

    for chunk in text_chunks:
        if not chunk:
            continue
        tokens = _extract_price_tokens(chunk)
        if not tokens:
            continue
        chunk_lower = chunk.lower()
        handled_indices: Set[int] = set()
        if any(word in chunk_lower for word in _STRATEGY_RANGE_HINTS) and len(tokens) >= 2:
            ordered = sorted(enumerate(tokens), key=lambda item: item[1].get("price", 0.0))
            min_index, min_token = ordered[0]
            max_index, max_token = ordered[-1]
            _register(
                supports,
                support_seen,
                min_token.get("price"),
                chunk,
                source="strategy-text",
            )
            _register(
                resistances,
                resistance_seen,
                max_token.get("price"),
                chunk,
                source="strategy-text",
            )
            handled_indices.update({min_index, max_index})
        for idx, token in enumerate(tokens):
            if idx in handled_indices:
                continue
            price = token.get("price")
            if price is None:
                continue
            classification = _classify_price_context(chunk, token)
            if classification == "support":
                _register(supports, support_seen, price, chunk, source="strategy-text")
            elif classification == "resistance":
                _register(resistances, resistance_seen, price, chunk, source="strategy-text")
            elif classification == "range" and len(tokens) >= 2:
                # for residual range detection, defer until ambiguity handling
                ambiguous.append((price, chunk))
            else:
                ambiguous.append((price, chunk))

    raw_levels = strategy.get("key_levels")
    if isinstance(raw_levels, dict):
        for key, value in raw_levels.items():
            context = f"{key}: {value}"
            classification_hint = "support" if isinstance(key, str) and "support" in key.lower() else "resistance" if isinstance(key, str) and "resist" in key.lower() else ""
            prices = _extract_price_tokens(str(value)) or [{"price": _parse_price_value(value), "start": 0, "end": 0, "raw": value}]
            for token in prices:
                price = token.get("price")
                if price is None:
                    continue
                if classification_hint == "support":
                    _register(supports, support_seen, price, context, source="strategy-key")
                elif classification_hint == "resistance":
                    _register(resistances, resistance_seen, price, context, source="strategy-key")
                else:
                    ambiguous.append((price, context))
    elif isinstance(raw_levels, (list, tuple)):
        for value in raw_levels:
            price = _parse_price_value(value)
            if price is None:
                continue
            ambiguous.append((price, str(value)))

    for price, context in ambiguous:
        if price is None:
            continue
        if isinstance(latest_price, (int, float)):
            if price <= latest_price:
                _register(supports, support_seen, price, context, source="strategy-heuristic")
            else:
                _register(resistances, resistance_seen, price, context, source="strategy-heuristic")

    supports.sort(
        key=lambda entry: (
            abs((latest_price or 0.0) - entry["price"]) if isinstance(latest_price, (int, float)) else entry["price"]
        )
    )
    resistances.sort(
        key=lambda entry: (
            abs((latest_price or 0.0) - entry["price"]) if isinstance(latest_price, (int, float)) else entry["price"]
        )
    )

    return {
        "supports": supports[:6],
        "resistances": resistances[:6],
    }


def _favorite_symbols(
    session: Session,
    user_id: int,
    limit: int = 25,
    *,
    market: Optional[str] = None,
) -> List[str]:
    rows = (
        session.query(UserFavoriteSymbol.symbol)
        .filter(UserFavoriteSymbol.user_id == user_id)
        .order_by(UserFavoriteSymbol.created_at.desc())
        .limit(max(limit, 1))
        .all()
    )
    sanitized: List[str] = []
    for row in rows:
        if not row:
            continue
        candidate = _sanitize_symbol(row[0])
        if candidate:
            sanitized.append(candidate)

    if market:
        allowed = _filter_symbols_for_market(sanitized, market)
        sanitized = [symbol for symbol in sanitized if symbol in allowed]

    return sanitized


def _all_favorite_symbols(session: Session, market: Optional[str] = None) -> Set[str]:
    rows = session.query(UserFavoriteSymbol.symbol).distinct().all()
    symbols = {_sanitize_symbol(row[0]) for row in rows if row and row[0]}
    symbols.discard("")
    if market:
        return _filter_symbols_for_market(symbols, market)
    return symbols


class _PooledSubscriptionRegistry:
    def __init__(
        self,
        market: str,
        stream_getter: Callable[[FastAPI], Optional[LivePriceStream]],
    ) -> None:
        self._market = market
        self._stream_getter = stream_getter
        self._symbols: Set[str] = set()
        self._lock = asyncio.Lock()

    def snapshot(self) -> Set[str]:
        return set(self._symbols)

    async def ensure(
        self,
        app: FastAPI,
        symbols: Iterable[str],
        stream: Optional[LivePriceStream] = None,
    ) -> None:
        normalized: Set[str] = set()
        for raw_symbol in symbols:
            sanitized = _sanitize_symbol(raw_symbol)
            if sanitized:
                normalized.add(sanitized)
        if not normalized:
            return

        additions: Set[str]
        async with self._lock:
            additions = normalized - self._symbols
            if not additions:
                return
            self._symbols.update(additions)

        target_stream = stream if stream is not None else self._stream_getter(app)
        if not target_stream:
            logger.debug(
                "Live %s stream unavailable; pooled symbols pending subscription: %s",
                self._market,
                sorted(additions),
            )
            return

        try:
            await target_stream.subscribe(additions)
        except Exception as exc:  # pragma: no cover - runtime safety
            logger.debug(
                "Failed to subscribe pooled %s symbols %s: %s",
                self._market,
                sorted(additions),
                exc,
            )


def _get_equity_registry(app: FastAPI) -> _PooledSubscriptionRegistry:
    registry = getattr(app.state, "equity_symbol_registry", None)
    if registry is None:
        registry = _PooledSubscriptionRegistry("equity", _get_live_stream)
        app.state.equity_symbol_registry = registry
    return registry


def _get_crypto_registry(app: FastAPI) -> _PooledSubscriptionRegistry:
    registry = getattr(app.state, "crypto_symbol_registry", None)
    if registry is None:
        registry = _PooledSubscriptionRegistry("crypto", _get_crypto_stream)
        app.state.crypto_symbol_registry = registry
    return registry


async def _ensure_equity_stream_symbols(app: FastAPI, symbols: Iterable[str]) -> None:
    registry = _get_equity_registry(app)
    await registry.ensure(app, symbols)


async def _ensure_crypto_stream_symbols(app: FastAPI, symbols: Iterable[str]) -> None:
    registry = _get_crypto_registry(app)
    await registry.ensure(app, symbols)


def _get_live_stream(app: FastAPI) -> Optional[LivePriceStream]:
    stream = getattr(app.state, "live_price_stream", None)
    if not stream:
        return None
    if hasattr(stream, "is_active") and not stream.is_active:
        reason = getattr(stream, "disabled_reason", None)
        if reason and not getattr(stream, "_disabled_notice_logged", False):
            logger.warning("Live price streaming disabled: %s", reason)
            setattr(stream, "_disabled_notice_logged", True)
        return None
    return stream


def _get_crypto_stream(app: FastAPI) -> Optional[LiveCryptoStream]:
    stream = getattr(app.state, "live_crypto_stream", None)
    if not stream:
        return None
    if hasattr(stream, "is_active") and not stream.is_active:
        reason = getattr(stream, "disabled_reason", None)
        if reason and not getattr(stream, "_disabled_notice_logged", False):
            logger.warning("Live crypto streaming disabled: %s", reason)
            setattr(stream, "_disabled_notice_logged", True)
        return None
    return stream


async def _fetch_rest_live_price(symbol: str) -> Optional[Dict[str, Any]]:
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not api_secret:
        return None

    url = f"{ALPACA_DATA_REST_URL}/v2/stocks/{symbol}/trades/latest"
    params = {"feed": os.getenv("ALPACA_DATA_FEED", "iex")}
    timeout = aiohttp.ClientTimeout(total=5)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            response = await session.get(
                url,
                headers={
                    "APCA-API-KEY-ID": api_key,
                    "APCA-API-SECRET-KEY": api_secret,
                },
                params=params,
            )
            if response.status == 200:
                payload = await response.json()
                trade = payload.get("trade") or {}
                price = trade.get("p")
                if price is None:
                    return None
                return {
                    "symbol": payload.get("symbol", symbol),
                    "price": price,
                    "bid": None,
                    "ask": None,
                    "timestamp": trade.get("t"),
                    "volume": trade.get("s"),
                    "source": "alpaca-rest",
                    "received_at": datetime.now(timezone.utc).isoformat(),
                }

            if response.status in {404, 422}:
                return None

            body = await response.text()
            logger.debug(
                "Alpaca REST live price fallback failed (%s): %s", response.status, body
            )
    except Exception as exc:  # pragma: no cover - diagnostic logging
        logger.debug("Alpaca REST live price fallback error for %s: %s", symbol, exc)

    return None


async def _fetch_rest_crypto_price(symbol: str) -> Optional[Dict[str, Any]]:
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not api_secret:
        return None

    url = f"{ALPACA_DATA_REST_URL}/v1beta3/crypto/{ALPACA_CRYPTO_REST_MARKET}/latest/trades"
    params = {"symbols": symbol.upper()}
    if ALPACA_CRYPTO_REST_FEED:
        params["feed"] = ALPACA_CRYPTO_REST_FEED
    timeout = aiohttp.ClientTimeout(total=5)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            response = await session.get(
                url,
                headers={
                    "APCA-API-KEY-ID": api_key,
                    "APCA-API-SECRET-KEY": api_secret,
                },
                params=params,
            )
            if response.status == 200:
                payload = await response.json()
                trades = (payload.get("trades") or {}) if isinstance(payload, dict) else {}
                trade = trades.get(symbol.upper()) or trades.get(symbol)
                if trade is None:
                    for key, candidate in trades.items():
                        if _sanitize_symbol(key) == symbol.upper():
                            trade = candidate
                            break
                if not trade:
                    return None
                price = trade.get("p") or trade.get("price")
                if price is None:
                    return None
                return {
                    "symbol": trade.get("S") or symbol.upper(),
                    "price": price,
                    "bid": None,
                    "ask": None,
                    "timestamp": trade.get("t") or trade.get("timestamp"),
                    "volume": trade.get("s") or trade.get("size"),
                    "source": "alpaca-crypto-rest",
                    "received_at": datetime.now(timezone.utc).isoformat(),
                }

            if response.status in {404, 422}:
                return None

            body = await response.text()
            logger.debug(
                "Alpaca REST crypto price fallback failed (%s): %s",
                response.status,
                body,
            )
    except Exception as exc:  # pragma: no cover - diagnostic logging
        logger.debug("Alpaca REST crypto price fallback error for %s: %s", symbol, exc)

    return None


async def _resolve_rest_prices(symbols: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    sanitized_symbols = {_sanitize_symbol(sym) for sym in symbols if _sanitize_symbol(sym)}
    if not sanitized_symbols:
        return {}

    results: Dict[str, Dict[str, Any]] = {}

    async def _collect(targets: Set[str], fetcher: Callable[[str], Awaitable[Optional[Dict[str, Any]]]]) -> None:
        for ticker in targets:
            try:
                payload = await fetcher(ticker)
            except Exception as exc:  # pragma: no cover - diagnostic logging
                logger.debug("REST price fallback batch fetch failed for %s: %s", ticker, exc)
                continue
            if isinstance(payload, dict) and payload.get("price") is not None:
                mapped_symbol = _sanitize_symbol(payload.get("symbol") or ticker)
                if mapped_symbol:
                    results[mapped_symbol] = payload

    equity_targets = _filter_symbols_for_market(sanitized_symbols, "equity")
    crypto_targets = _filter_symbols_for_market(sanitized_symbols, "crypto")

    await asyncio.gather(
        _collect(equity_targets, _fetch_rest_live_price),
        _collect(crypto_targets, _fetch_rest_crypto_price),
    )

    return results


def _ensure_trial_subscription(session: Session, user: User) -> None:
    """Provision a complimentary trial subscription when none exists."""

    active_subscription = (
        session.query(UserSubscription)
        .filter(UserSubscription.user_id == user.id)
        .filter(UserSubscription.status.in_(ACTIVE_SUBSCRIPTION_STATUSES))
        .order_by(UserSubscription.created_at.desc())
        .first()
    )
    if active_subscription is not None:
        logger.debug(
            "User %s already has active subscription %s",
            user.email,
            active_subscription.id,
        )
        return

    trial_plan = (
        session.query(SubscriptionPlan)
        .filter(SubscriptionPlan.slug == TRIAL_PLAN_SLUG)
        .one_or_none()
    )
    if trial_plan is None:
        logger.warning(
            "Trial plan '%s' not found; skipping trial provisioning for user %s",
            TRIAL_PLAN_SLUG,
            user.email,
        )
        return

    prior_trial = (
        session.query(UserSubscription)
        .filter(UserSubscription.user_id == user.id)
        .filter(UserSubscription.plan_id == trial_plan.id)
        .first()
    )
    if prior_trial is not None:
        logger.debug("User %s previously used trial plan", user.email)
        return

    now = datetime.now(timezone.utc)
    trial_subscription = UserSubscription(
        user_id=user.id,
        plan_id=trial_plan.id,
        stripe_customer_id=None,
        stripe_subscription_id=None,
        status="trialing",
        current_period_start=now,
        current_period_end=now + timedelta(days=TRIAL_DURATION_DAYS),
        runs_remaining=trial_plan.ai_runs_included or 0,
        auto_renew=False,
        cancel_at_period_end=True,
    )
    session.add(trial_subscription)
    user.tier = trial_plan.slug
    session.commit()
    logger.info(
        "Provisioned trial subscription %s for user %s with %s runs",
        trial_subscription.id,
        user.email,
        trial_subscription.runs_remaining,
    )


def _consume_subscription_units(
    user_id: int,
    *,
    units: int = 1,
    usage_type: str = "ai_run",
    notes: Optional[str] = None,
) -> int:
    """Deduct subscription allowance for metered AI features.

    Returns the remaining quota after consumption or raises an HTTPException
    if the user is not eligible to run the requested workload.
    """

    with SessionLocal() as session:
        subscription = _find_relevant_subscription(session, user_id)

        if subscription is None or subscription.plan is None:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail={
                    "error": "You need an active VolatilX subscription to run premium AI strategies.",
                    "code": "subscription_required",
                    "runs_remaining": 0,
                    "action_label": "View plans",
                    "action_url": "/subscribe?reason=subscription_required",
                },
            )

        if subscription.runs_remaining is None:
            subscription.runs_remaining = 0

        period_end = subscription.current_period_end
        now_utc = datetime.now(timezone.utc)
        if period_end is not None:
            if period_end.tzinfo is None:
                period_end = period_end.replace(tzinfo=timezone.utc)
            if period_end < now_utc:
                subscription.status = "expired"
                subscription.runs_remaining = 0
                session.commit()
                is_trial = bool(subscription.plan and subscription.plan.slug == TRIAL_PLAN_SLUG)
                raise HTTPException(
                    status_code=status.HTTP_402_PAYMENT_REQUIRED,
                    detail={
                        "error": (
                            "Your trial period has ended. Upgrade to continue using AI analysis."
                            if is_trial
                            else "Your subscription period has ended. Please renew to continue."
                        ),
                        "code": "trial_expired" if is_trial else "subscription_expired",
                        "runs_remaining": 0,
                        "action_label": "Upgrade plan",
                        "action_url": "/subscribe?reason=trial_expired" if is_trial else "/subscribe?reason=subscription_expired",
                    },
                )

        if subscription.runs_remaining < units:
            reset_at = (
                subscription.current_period_end.isoformat()
                if subscription.current_period_end
                else None
            )
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail={
                    "error": "You’ve used all AI runs included in your plan for this billing cycle.",
                    "code": "quota_exhausted",
                    "runs_remaining": max(subscription.runs_remaining, 0),
                    "action_label": "Manage subscription",
                    "action_url": "/subscribe?reason=quota_exhausted",
                    "renews_at": reset_at,
                },
            )

        subscription.runs_remaining -= units
        enqueue_usage(
            session,
            subscription,
            units=units,
            notes=notes,
            usage_type=usage_type,
        )

        session.commit()
        remaining = subscription.runs_remaining
        logger.info(
            "Consumed %s units (%s) for user %s subscription %s; remaining=%s",
            units,
            usage_type,
            user_id,
            subscription.id,
            remaining,
        )

    return remaining


@asynccontextmanager
async def lifespan(app: FastAPI):
    live_stream = None
    crypto_stream = None
    try:
        init_db()
        with SessionLocal() as session:
            sync_plan_catalogue(session)
        print("Database initialized successfully")
    except Exception as e:
        print(f"Database initialization failed: {e}")
        # Try to create tables anyway
        from db import Base, engine
        Base.metadata.create_all(bind=engine)
    if ENABLE_LIVE_PRICE_STREAM:
        try:
            live_stream = LivePriceStream()
            await live_stream.start()
            try:
                await live_stream.ready(timeout=5.0)
            except Exception as exc:  # pragma: no cover - startup resilience
                logger.debug("Live price stream ready wait skipped: %s", exc)
        except Exception as exc:  # pragma: no cover - startup resilience
            logger.warning("Live price streaming unavailable: %s", exc)
            live_stream = None
    app.state.live_price_stream = live_stream
    if live_stream:
        try:
            registry = _get_equity_registry(app)
            initial_symbols = {_sanitize_symbol(DEFAULT_ACTION_CENTER_SYMBOL)} if DEFAULT_ACTION_CENTER_SYMBOL else set()
            try:
                with SessionLocal() as session:
                    initial_symbols.update(_all_favorite_symbols(session, market="equity"))
            except Exception as exc:  # pragma: no cover - startup resilience
                logger.debug("Unable to load pooled favorite symbols: %s", exc)
            await registry.ensure(app, initial_symbols, stream=live_stream)
        except Exception as exc:  # pragma: no cover - startup resilience
            logger.debug("Unable to prime pooled equity subscriptions: %s", exc)
    if ENABLE_LIVE_CRYPTO_STREAM:
        try:
            crypto_stream = LiveCryptoStream()
            await crypto_stream.start()
            try:
                await crypto_stream.ready(timeout=5.0)
            except Exception as exc:  # pragma: no cover - startup resilience
                logger.debug("Live crypto stream ready wait skipped: %s", exc)
        except Exception as exc:  # pragma: no cover - startup resilience
            logger.warning("Live crypto streaming unavailable: %s", exc)
            crypto_stream = None
    app.state.live_crypto_stream = crypto_stream
    if crypto_stream:
        try:
            registry = _get_crypto_registry(app)
            initial_crypto = {_sanitize_symbol(DEFAULT_CRYPTO_STREAM_SYMBOL)} if DEFAULT_CRYPTO_STREAM_SYMBOL else set()
            try:
                with SessionLocal() as session:
                    initial_crypto.update(_all_favorite_symbols(session, market="crypto"))
            except Exception as exc:  # pragma: no cover - startup resilience
                logger.debug("Unable to load pooled crypto favorites: %s", exc)
            await registry.ensure(app, initial_crypto, stream=crypto_stream)
        except Exception as exc:  # pragma: no cover - startup resilience
            logger.debug("Unable to prime pooled crypto subscriptions: %s", exc)
    try:
        yield
    finally:
        if live_stream:
            try:
                await live_stream.stop()
            except Exception as exc:  # pragma: no cover - shutdown resilience
                logger.debug("Live price stream stop failed: %s", exc)
        if crypto_stream:
            try:
                await crypto_stream.stop()
            except Exception as exc:  # pragma: no cover - shutdown resilience
                logger.debug("Live crypto stream stop failed: %s", exc)
        app.state.live_price_stream = None
        app.state.live_crypto_stream = None

app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Session middleware for OAuth
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET_KEY", "super-secret-for-dev")
)
# app.add_middleware(ForwardedMiddleware)
# app.add_middleware(ProxyHeadersMiddleware) 

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://volatilx.com",
        "https://www.volatilx.com",
        "https://volatilx.ai",
        "https://www.volatilx.ai",
        "http://127.0.0.1:8000"  # keep this if you use local dev
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

##############################################################################################
# FastAPI Users Setup - CORRECTED VERSION
##############################################################################################
# JWT Strategy setup
SECRET = os.getenv("JWT_SECRET", "SECRET")

def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=SECRET, lifetime_seconds=3600)

# Cookie transport setup
cookie_transport = CookieTransport(cookie_name="volatilx_cookie", cookie_max_age=3600)

# Auth backend setup
auth_backend = AuthenticationBackend(
    name="jwt",
    transport=cookie_transport,
    get_strategy=get_jwt_strategy,
)

# FastAPI Users setup - FIXED: Now uses get_user_manager instead of get_user_db
fastapi_users = FastAPIUsers[User, int](get_user_manager, [auth_backend])

# Get current user dependency
current_active_user = fastapi_users.current_user(active=True)

# Password helper
password_helper = PasswordHelper()

##############################################################################################
# OAuth Setup - Azure and Google
##############################################################################################
# OAuth client setup
oauth = OAuth()
# Azure OAuth setup
oauth.register(
    name="azure",
    client_id=os.getenv("YOUR_ENTRA_CLIENT_ID"),
    client_secret=os.getenv("YOUR_ENTRA_CLIENT_SECRET"),
    server_metadata_url="https://login.microsoftonline.com/common/v2.0/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

# Google OAuth setup
oauth.register(
    name='google',
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"}
)
# OAuth Routes - Azure
##############################################################################################
@app.get("/auth/azure/login")
async def azure_login(request: Request):
    client_host = request.client.host if request.client else "unknown"
    logger.info("Azure login initiated from %s", client_host)
    redirect_uri = request.url_for("azure_callback")
    return await oauth.azure.authorize_redirect(request, redirect_uri)

@app.get("/auth/azure/callback")
async def azure_callback(request: Request):
    try:
        token = await oauth.azure.authorize_access_token(request)
        user_info = await oauth.azure.parse_id_token(request, token)
        email = user_info.get("email")
        oauth_sub = user_info.get("sub")
        
        if not email:
            return RedirectResponse(url="/signin?error=no_email")
        
        client_host = request.client.host if request.client else "unknown"
        logger.info("Azure OAuth callback for %s from %s", email, client_host)
        # Use sync database operations
        db = SessionLocal()
        try:
            # Query user directly with SQLAlchemy
            user = db.query(User).filter(User.email == email).first()
            
            if user is None:
                # Create new user
                logger.info("Creating Azure user record for %s", email)
                user = User(
                    email=email,
                    hashed_password=password_helper.hash(os.urandom(8).hex()),
                    is_active=True,
                    is_superuser=False,
                    is_verified=True,
                    oauth_provider="azure",
                    oauth_id=oauth_sub,
                )
                db.add(user)
                db.commit()
                db.refresh(user)
            
            _ensure_trial_subscription(db, user)

            # Issue JWT
            jwt_strategy = get_jwt_strategy()
            access_token = await jwt_strategy.write_token(user)
            response = RedirectResponse("/analyze")
            response.set_cookie(
                key="volatilx_cookie", 
                value=access_token,
                httponly=True,
                secure=False,  # Set to True in production
                samesite="lax"
            )
            return response
            
        except Exception as e:
            db.rollback()
            error_detail = urllib.parse.quote_plus(str(e))
            logger.exception("Database error in Azure callback for %s", email)
            print(f"Database error in Azure callback: {e}")
            return RedirectResponse(url=f"/signin?error=db_error&detail={error_detail}")
        finally:
            db.close()
            
    except Exception as e:
        logger.exception("Azure OAuth error")
        print(f"OAuth error in Azure callback: {e}")
        return RedirectResponse(url="/signin?error=oauth_failed")

##############################################################################################
# OAuth Routes - Google
##############################################################################################
@app.get("/auth/google/login")
async def google_login(request: Request):
    host = request.headers.get("host", "www.volatilx.com")
    redirect_uri = f"https://{host}/auth/google/callback"
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get("/auth/google/callback")
async def google_callback(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get("userinfo") or {}
        email = user_info.get("email")
        oauth_sub = user_info.get("sub") or user_info.get("id")

        if not email:
            return RedirectResponse(url="/signin?error=no_email")

        client_host = request.client.host if request.client else "unknown"
        logger.info("Google OAuth callback for %s from %s", email, client_host)

        db = SessionLocal()
        try:
            # Query user directly with SQLAlchemy
            user = db.query(User).filter(User.email == email).first()
            if user is None:
                print(f"Creating new user for {email}")
                logger.info("Creating Google user record for %s", email)
                user = User(
                    email=email,
                    hashed_password=password_helper.hash(os.urandom(8).hex()),
                    is_active=True,
                    is_superuser=False,
                    is_verified=True,
                    oauth_provider="google",
                    oauth_id=oauth_sub,
                )
                db.add(user)
                db.commit()
                db.refresh(user)
                print(f"User created successfully with ID: {user.id}")
            else:
                print(f"Existing user found: {user.id}")

            _ensure_trial_subscription(db, user)
            logger.info("Google user %s assigned to tier %s", user.email, user.tier)

            jwt_strategy = get_jwt_strategy()
            access_token = await jwt_strategy.write_token(user)

            response = RedirectResponse(url="/analyze")
            response.set_cookie(
                key="volatilx_cookie",
                value=access_token,
                httponly=True,
                secure=False,
                samesite="lax",
                max_age=3600,
                path="/",
            )
            print("OAuth successful, redirecting to /analyze")
            return response

        except Exception as e:
            db.rollback()
            error_detail = urllib.parse.quote_plus(str(e))
            logger.exception("Database error in Google callback for %s", email)
            print(f"Database error in Google callback: {e}")
            return RedirectResponse(url=f"/signin?error=db_error&detail={error_detail}")
        finally:
            db.close()

    except Exception as e:
        logger.exception("Google OAuth error")
        print(f"OAuth error in Google callback: {e}")
        return RedirectResponse(url="/signin?error=oauth_failed")

##############################################################################################
# FastAPI Users routers - Register the authentication endpoints
##############################################################################################
app.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/auth/jwt",
    tags=["auth"],
)

app.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/auth",
    tags=["auth"]
)

app.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/users",
    tags=["users"]
)

##############################################################################################
# Signin and Signup Page Routes
##############################################################################################
@app.get("/signin", response_class=HTMLResponse)
async def signin_page(request: Request):
    client_host = request.client.host if request.client else "unknown"
    logger.info("Signin page viewed from %s", client_host)
    return templates. TemplateResponse("signin.html", {"request": request})

@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    client_host = request.client.host if request.client else "unknown"
    logger.info("Signup page viewed from %s", client_host)
    return templates. TemplateResponse("signup.html", {"request": request})

# Basic demo handlers; customize for real authentication logic!
@app.post("/signin")
async def handle_signin(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
):
    client_host = request.client.host if request.client else "unknown"
    logger.info("Form signin attempt for %s from %s", email, client_host)
    # Demo: Redirect to home; integrate FastAPI Users for actual sign in
    # You need to implement credential check!
    response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    return response

@app.post("/signup")
async def handle_signup(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
):
    client_host = request.client.host if request.client else "unknown"
    logger.info("Form signup attempt for %s from %s", email, client_host)
    # Demo: Redirect to sign-in. Implement FastAPI Users 'register' logic here!
    response = RedirectResponse(url="/signin", status_code=status.HTTP_303_SEE_OTHER)
    return response

@app.get("/logout")
async def logout():
    logger.info("User initiated logout")
    response = RedirectResponse(url="/signin")
    response.delete_cookie("volatilx_cookie")
    return response

# Exception handler for unauthorized access
@app.exception_handler(401)
async def unauthorized_handler(request: Request, exc):
    if request.url.path.startswith("/api/") or request.headers.get("content-type") == "application/json":
        return JSONResponse(
            status_code=401,
            content={"detail": "Unauthorized"}
        )
    return RedirectResponse(url="/signin")

##############################################################################################
# Agent and Indicator Fetcher Initialization
##############################################################################################
indicator_fetcher = ComprehensiveMultiTimeframeAnalyzer()
price_action_analyzer = PriceActionAnalyzer(analyzer=indicator_fetcher)

principal_agent_instance: Optional[PrincipalAgent] = None


def get_principal_agent_instance() -> PrincipalAgent:
    """Initialise and cache the principal trading agent."""

    global principal_agent_instance
    if principal_agent_instance is None:
        try:
            principal_agent_instance = PrincipalAgent()
        except Exception as exc:  # noqa: BLE001 - surface initialization issues to caller
            logger.error("Failed to initialise PrincipalAgent: %s", exc)
            raise
    return principal_agent_instance

##############################################################################################
# Main Application Routes
##############################################################################################
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return RedirectResponse(url="/signin")

@app.get("/analyze", response_class=HTMLResponse)
async def analyze_page(request: Request, user: User = Depends(get_current_user_sync)):
    logger.info("User %s (%s) opened AI insights page", user.id, user.email)
    with SessionLocal() as session:
        subscription = _find_relevant_subscription(session, user.id)

    if subscription is None or subscription.plan is None:
        logger.info("Redirecting user %s to subscription page (no active plan)", user.id)
        return RedirectResponse(url="/subscribe?from=analyze", status_code=status.HTTP_303_SEE_OTHER)

    context = {
        "request": request,
        "user": user,
        "subscription": _serialize_subscription(subscription),
        "symbol_catalogs": get_all_symbol_catalogs(),
    }
    return templates.TemplateResponse("ai_insights.html", context)

@app.post("/analyze")
async def analyze(request: Request, background_tasks: BackgroundTasks, user: User = Depends(get_current_user_sync)):
    try:
        # Get JSON data from request
        data = await request.json()
        # print("=== TRADE ENDPOINT DEBUG ===")
        print("Printing data:", data)
        
        # Extract variables from the request data
        raw_stock_symbol = data.get('stock_symbol')
        stock_symbol = raw_stock_symbol
        market_raw = data.get('market')
        market = str(market_raw).strip().lower() if market_raw is not None else 'equity'
        if market not in SUPPORTED_MARKETS:
            logger.debug("Unsupported market '%s' requested; defaulting to equities", market)
            market = 'equity'
        use_ai_analysis = data.get('use_ai_analysis', False)
        use_principal_agent = data.get('use_principal_agent', use_ai_analysis)
        include_principal_raw = bool(data.get('include_principal_raw_results', False))
        price_action_timeframes_raw = data.get('price_action_timeframes')
        price_action_period_overrides_raw = data.get('price_action_period_overrides')
        # language = data.get('language', 'en')

        symbol_message: Optional[str] = None
        try:
            stock_symbol, message = normalize_symbol(stock_symbol, market=market)
            symbol_message = message
        except SymbolNotFound as exc:
            suggestion_company = None
            suggestion_symbol = exc.suggestion
            if suggestion_symbol:
                suggestion_company = get_ticker_map(market).get(suggestion_symbol)
            error_message = f"Unknown symbol '{raw_stock_symbol}'."
            if suggestion_symbol:
                if suggestion_company:
                    error_message += f" Did you mean {suggestion_company} ({suggestion_symbol})?"
                else:
                    error_message += f" Did you mean {suggestion_symbol}?"
            else:
                error_message += " Please choose a supported ticker."
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": error_message,
                    "code": "unknown_symbol",
                    "suggested_symbol": suggestion_symbol,
                    "suggested_company": suggestion_company,
                },
            )
        
        logger.info(
            "User %s requested AI insights for %s in %s market (ai=%s principal=%s)",
            user.id,
            stock_symbol,
            market,
            use_ai_analysis,
            use_principal_agent,
        )
        
        if not stock_symbol:
            return JSONResponse(
                content={'success': False, 'error': 'Stock symbol is required'},
                status_code=400
            )
        
        runs_remaining_after: Optional[int] = None
        should_meter = bool(use_ai_analysis or use_principal_agent)
        if should_meter:
            runs_remaining_after = _consume_subscription_units(
                user.id,
                usage_type="trade_analysis",
                notes=f"Trade analysis for {stock_symbol}",
            )

        # Initialize trading agent
        agent = MultiSymbolDayTraderAgent(
            symbols=stock_symbol,
            timeframes=['5m', '15m', '30m', '1h', '1d', '1wk','1mo']
        ,
            market=market,
        )
        
        # Set API credentials
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        if not api_key or not secret_key:
            logger.error("Alpaca credentials missing; cannot perform trade analysis")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "Alpaca credentials are not configured.",
                    "code": "alpaca_credentials_missing",
                },
            )

        agent.set_credentials(api_key=api_key, secret_key=secret_key, base_url=base_url)
        try:
            indicator_fetcher.set_credentials(api_key, secret_key, base_url=base_url)
        except Exception as cred_error:  # noqa: BLE001 - log but do not fail primary analysis
            logger.warning("Failed to configure shared indicator credentials: %s", cred_error)
        
        # Set trading parameters
        agent.set_trading_parameters(
            buy_threshold=70,
            sell_threshold=70,
            high_confidence_only=True
        )
        
        # Run the trading analysis
        print("=== RUNNING TRADING ANALYSIS ===")
        results = agent.run_sequential()
        result_data = None
        result_data_json = {}
        if results:
            result_data = agent.export_results(results)
            # If result_data[1] is a string, parse it:
            if isinstance(result_data[1], str):
                result_data_json = json.loads(result_data[1])  # ensures you have a dict
            else:
                result_data_json = result_data[1]


        price_action_timeframes: Optional[List[str]] = None
        if isinstance(price_action_timeframes_raw, str):
            parts = [item.strip() for item in price_action_timeframes_raw.split(',') if item and item.strip()]
            price_action_timeframes = parts or None
        elif isinstance(price_action_timeframes_raw, (list, tuple, set)):
            converted = [str(item).strip() for item in price_action_timeframes_raw if str(item).strip()]
            price_action_timeframes = converted or None

        price_action_period_overrides: Optional[Dict[str, str]] = None
        if isinstance(price_action_period_overrides_raw, dict):
            cleaned_overrides = {
                str(key): str(value)
                for key, value in price_action_period_overrides_raw.items()
                if value is not None and str(value).strip()
            }
            price_action_period_overrides = cleaned_overrides or None

        price_action_analysis: Optional[dict] = None
        try:
            price_action_analysis = price_action_analyzer.analyze(
                stock_symbol,
                timeframes=price_action_timeframes,
                period_overrides=price_action_period_overrides,
                market=market,
            )
        except Exception as exc:  # noqa: BLE001 - return structured error information
            logger.exception("Price action analysis failed for %s", stock_symbol)
            price_action_analysis = {
                "success": False,
                "symbol": stock_symbol,
                "error": str(exc),
            }

        # Prepare background job for AI / principal analysis if requested
        ai_job_id = None
        background_analysis_requested = (use_ai_analysis or use_principal_agent) and result_data is not None
        if background_analysis_requested:
            ai_job_id = str(uuid.uuid4())
            ai_analysis_jobs[ai_job_id] = {"status": "pending", "result": None, "user_id": user.id}

            def run_async_analysis(
                job_id: str,
                technical_data: Any,
                technical_snapshot: Dict[str, Any],
                price_action_snapshot: Optional[Dict[str, Any]],
                symbol: str,
                user_id: Any,
                include_ai: bool,
                include_principal: bool,
                include_principal_raw: bool,
            ) -> None:
                logger.info("Background analysis job %s started for %s", job_id, symbol)
                job_result: Dict[str, Any] = {
                    "ai_analysis": None,
                    "principal_plan": None,
                }
                price_value, price_tf, price_timestamp = _extract_latest_price(technical_snapshot, symbol)
                try:
                    if include_ai:
                        try:
                            logger.info("AI analysis started for %s (job %s)", symbol, job_id)
                            analysis = openai_service.analyze_trading_data(technical_data, symbol)
                            if isinstance(analysis, dict) and price_value is not None:
                                analysis["latest_price"] = price_value
                                if price_tf:
                                    analysis["latest_price_timeframe"] = price_tf
                                if price_timestamp:
                                    analysis["latest_price_timestamp"] = price_timestamp
                            job_result["ai_analysis"] = analysis
                            logger.info("AI analysis completed for %s (job %s)", symbol, job_id)
                        except Exception as ai_exc:  # noqa: BLE001 - store failure in job result
                            logger.exception("AI analysis failed for %s", symbol)
                            job_result["ai_analysis"] = {
                                "success": False,
                                "error": str(ai_exc),
                            }

                    if include_principal:
                        try:
                            logger.info("Principal agent started for %s (job %s)", symbol, job_id)
                            plan = get_principal_agent_instance().generate_trading_plan(
                                symbol,
                                technical_snapshot=technical_snapshot,
                                price_action_snapshot=price_action_snapshot,
                                include_raw_results=include_principal_raw,
                            )
                            if isinstance(plan, dict):
                                if price_value is not None:
                                    plan["latest_price"] = price_value
                                if price_tf:
                                    plan["latest_price_timeframe"] = price_tf
                                if price_timestamp:
                                    plan["latest_price_timestamp"] = price_timestamp
                            job_result["principal_plan"] = {
                                "success": True,
                                "data": plan,
                            }
                            logger.info("Principal agent completed for %s (job %s)", symbol, job_id)
                        except Exception as principal_exc:  # noqa: BLE001 - store failure in job result
                            logger.exception("Principal agent failed for %s", symbol)
                            job_result["principal_plan"] = {
                                "success": False,
                                "error": str(principal_exc),
                            }

                    ai_analysis_jobs[job_id] = {
                        "status": "done",
                        "result": job_result,
                        "user_id": user_id,
                    }
                    store_ai_report(
                        symbol,
                        user_id,
                        {
                            "ai_job_id": job_id,
                            "status": "done",
                            "ai_analysis": job_result.get("ai_analysis"),
                            "principal_plan": job_result.get("principal_plan"),
                            "technical_snapshot": technical_snapshot,
                            "price_action_snapshot": price_action_snapshot,
                        },
                    )
                    logger.info("Background analysis job %s finished", job_id)
                except Exception as exc:  # noqa: BLE001 - surface in job status
                    logger.exception("Background analysis job %s failed", job_id)
                    ai_analysis_jobs[job_id] = {
                        "status": "error",
                        "error": str(exc),
                        "user_id": user_id,
                    }
                    store_ai_report(
                        symbol,
                        user_id,
                        {
                            "ai_job_id": job_id,
                            "status": "error",
                            "error": str(exc),
                            "technical_snapshot": technical_snapshot,
                            "price_action_snapshot": price_action_snapshot,
                        },
                    )

            background_tasks.add_task(
                run_async_analysis,
                ai_job_id,
                result_data,
                result_data_json,
                price_action_analysis,
                stock_symbol,
                user.id,
                use_ai_analysis,
                use_principal_agent,
                include_principal_raw,
            )

        # ...existing code...


        response = {
            'success': True,
            'result': result_data_json,
            'price_action': price_action_analysis,
            'symbol': stock_symbol,
            'market': market,
            'timestamp': str(datetime.now()),
            'ai_job_id': ai_job_id,
            'symbol_message': symbol_message,
            'input_symbol': raw_stock_symbol,
        }
        if runs_remaining_after is not None:
            response['runs_remaining'] = runs_remaining_after
        logger.info(
            "Trade analysis completed for %s by user %s (runs_remaining=%s)",
            stock_symbol,
            user.id,
            response.get('runs_remaining'),
        )
        return JSONResponse(content=response, status_code=status.HTTP_200_OK)
    except HTTPException as exc:
        logger.warning(
            "Trade analysis blocked for user %s: %s",
            user.id,
            exc.detail,
        )
        error_payload: Dict[str, Any] = {"success": False}
        detail = exc.detail
        if isinstance(detail, dict):
            error_payload.update(detail)
            if "error" not in error_payload and "message" in error_payload:
                error_payload["error"] = error_payload["message"]
        elif isinstance(detail, str):
            error_payload["error"] = detail
        else:
            error_payload["error"] = "Subscription validation failed."

        if "runs_remaining" not in error_payload:
            error_payload["runs_remaining"] = None

        response = JSONResponse(
            content=error_payload,
            status_code=exc.status_code,
        )
        if exc.headers:
            for key, value in exc.headers.items():
                response.headers[key] = value
        return response
    except Exception as exc:
        logger.exception("Trade endpoint failed")
        response = {
            'success': False,
            'error': str(exc)
        }
        return JSONResponse(content=response, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Endpoint to poll for AI analysis result
@app.get("/api/ai-analysis-result/{job_id}")
async def get_ai_analysis_result(job_id: str, user: User = Depends(get_current_user_sync)):
    job = ai_analysis_jobs.get(job_id)
    if not job:
        return JSONResponse(content={"success": False, "error": "Job not found"}, status_code=404)
    # Only allow the user who started the job to see the result
    if job.get("user_id") != user.id:
        return JSONResponse(content={"success": False, "error": "Unauthorized"}, status_code=403)
    if job["status"] == "pending":
        return JSONResponse(content={"success": False, "status": "pending"}, status_code=200)
    if job["status"] == "done":
        return JSONResponse(content={"success": True, "status": "done", "result": job["result"]}, status_code=200)
    if job["status"] == "error":
        return JSONResponse(content={"success": False, "status": "error", "error": job.get("error")}, status_code=200)
@app.post("/api/principal/plan")
async def generate_principal_plan(request: Request, user: User = Depends(get_current_user_sync)):
    data = await request.json()
    symbol = data.get("symbol")
    include_raw = bool(data.get("include_raw_results", False))

    if not symbol:
        return JSONResponse(
            content={"success": False, "error": "Symbol is required"},
            status_code=400,
        )

    try:
        plan = get_principal_agent_instance().generate_trading_plan(
            symbol,
            include_raw_results=include_raw,
        )
    except ValueError as exc:
        logger.error("Principal agent rejected request for %s: %s", symbol, exc)
        return JSONResponse(
            content={"success": False, "error": str(exc)},
            status_code=400,
        )
    except Exception as exc:  # noqa: BLE001 - capture unexpected agent failures
        logger.exception("Principal agent failed for %s", symbol)
        return JSONResponse(
            content={
                "success": False,
                "error": "Principal agent failed to generate plan",
                "details": str(exc),
            },
            status_code=500,
        )

    return JSONResponse(
        content={
            "success": True,
            "plan": plan,
        },
        status_code=200,
    )


@app.post("/api/contact")
async def submit_contact_request(request: Request, background_tasks: BackgroundTasks):
    email_service_ready = bool(
        CONTACT_EMAIL_RECIPIENT
        and (ACS_CONNECTION_STRING or (CONTACT_EMAIL_SENDER and SMTP_HOST))
    )
    if not email_service_ready:
        logger.warning("Contact endpoint requested but email configuration is incomplete")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "Contact service is not configured. Please try again later."},
        )

    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001 - invalid JSON
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Invalid JSON payload."},
        ) from None

    try:
        user = get_current_user_sync(request)
    except HTTPException:
        user = None

    def _trim(value: Any, *, max_length: int) -> str:
        if value is None:
            return ""
        text_value = str(value).strip()
        if len(text_value) > max_length:
            return text_value[:max_length].strip()
        return text_value

    user_email_default = user.email if user and user.email else ""
    name = _trim(payload.get("name") or user_email_default or "Customer", max_length=120)
    email_address = _trim(payload.get("email") or user_email_default or "", max_length=255)
    message = payload.get("message")
    if message is None:
        message = ""
    else:
        message = str(message).strip()

    if not name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Please provide your name."},
        )

    if not email_address or "@" not in email_address:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Please provide a valid email address."},
        )

    if not message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Please include a brief message."},
        )

    if len(message) > 4000:
        message = message[:4000].rstrip()

    submission_payload: Dict[str, Any] = {
        "name": name,
        "email": email_address,
        "message": message,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }

    if user is not None:
        submission_payload["user_id"] = user.id
        submission_payload["user_email"] = user.email
        log_identity = f"user {user.id}"
    else:
        log_identity = "guest visitor"

    background_tasks.add_task(_send_contact_email, submission_payload)

    logger.info("Queued contact request from %s", log_identity)

    return JSONResponse(
        content={"success": True},
        status_code=status.HTTP_202_ACCEPTED,
    )


@app.get("/subscribe", response_class=HTMLResponse)
async def subscribe_page(request: Request, user: User = Depends(get_current_user_sync)):
    logger.info("User %s (%s) opened subscription page", user.id, user.email)
    with SessionLocal() as session:
        subscription = _find_relevant_subscription(session, user.id)

    context = {
        "request": request,
        "user": user,
        "current_subscription": _serialize_subscription(subscription) if subscription else None,
    }
    return templates.TemplateResponse("subscription.html", context)

def _resolve_report_center_date(raw_date: Optional[str]) -> Tuple[datetime, str, str]:
    """Resolve query date into a midnight UTC timestamp and display labels."""

    today_utc = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    if raw_date:
        candidate = str(raw_date).strip()
        if candidate:
            for fmt in ("%Y-%m-%d", "%Y%m%d"):
                try:
                    parsed = datetime.strptime(candidate, fmt)
                except ValueError:
                    continue
                resolved = datetime(parsed.year, parsed.month, parsed.day, tzinfo=timezone.utc)
                return resolved, resolved.strftime("%Y-%m-%d"), resolved.strftime("%b %d, %Y")

    return today_utc, today_utc.strftime("%Y-%m-%d"), today_utc.strftime("%b %d, %Y")


def _fetch_latest_action_report(symbol: str, *, lookback_days: int = ACTION_CENTER_LOOKBACK_DAYS) -> Optional[Dict[str, Any]]:
    """Fetch the most recent stored AI report for *symbol* within the lookback window."""

    safe_symbol = (symbol or DEFAULT_ACTION_CENTER_SYMBOL).strip().upper()
    now = datetime.now(timezone.utc)
    max_days = max(1, lookback_days)

    for offset in range(max_days):
        target_day = now - timedelta(days=offset)
        day_anchor = target_day.replace(hour=0, minute=0, second=0, microsecond=0)
        reports = fetch_reports_for_date(day_anchor, symbol=safe_symbol, max_reports=20)
        if not reports:
            continue

        def sort_key(item: Dict[str, Any]) -> datetime:
            timestamp = item.get("stored_at") or item.get("_blob_last_modified")
            parsed = _parse_iso_datetime(timestamp) if isinstance(timestamp, str) else None
            if parsed is None:
                return datetime.min.replace(tzinfo=timezone.utc)
            return parsed

        reports.sort(key=sort_key, reverse=True)
        return reports[0]

    return None


def _derive_action_center_view(
    report: Dict[str, Any],
    *,
    strategy_key: str,
    intent: str,
    price_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a presentation-friendly payload for the Action Center UI."""

    intent_slug = intent.lower()
    symbol = str(
        report.get("symbol")
        or report.get("principal_plan", {}).get("data", {}).get("symbol")
        or DEFAULT_ACTION_CENTER_SYMBOL
    ).upper()

    principal = (report.get("principal_plan") or {}).get("data") or {}
    strategies = principal.get("strategies") or {}
    strategy = strategies.get(strategy_key) or {}

    expert_outputs = report.get("expert_outputs") or {}
    technical_output = (expert_outputs.get("technical") or {}).get("agent_output") or {}
    price_action_output = (expert_outputs.get("price_action") or {}).get("agent_output") or {}

    indicator_signals = (technical_output.get("indicator_signals") or {})
    consensus = indicator_signals.get("consensus") or {}
    timeframes = indicator_signals.get("timeframes") or []

    latest_price = None
    latest_price_ts = None
    override_price_value = None
    override_price_timestamp = None
    override_price_source = None
    if isinstance(price_override, dict):
        override_price_value = _safe_float(price_override.get("price"))
        override_price_timestamp = price_override.get("timestamp")
        override_price_source = price_override.get("source")
    technical_snapshot = (report.get("technical_snapshot") or {})
    symbol_snapshot = None
    if isinstance(technical_snapshot, dict):
        symbol_snapshot = technical_snapshot.get(symbol)
        if not symbol_snapshot:
            for candidate in technical_snapshot.values():
                if isinstance(candidate, dict) and str(candidate.get("symbol")).upper() == symbol:
                    symbol_snapshot = candidate
                    break
    if isinstance(symbol_snapshot, dict):
        latest_price = symbol_snapshot.get("latest_price")
        latest_price_ts = symbol_snapshot.get("latest_price_timestamp")
        if not latest_price and isinstance(symbol_snapshot.get("price_by_timeframe"), dict):
            latest_price = symbol_snapshot["price_by_timeframe"].get("5m") or symbol_snapshot["price_by_timeframe"].get("15m")
    if latest_price is None:
        latest_price = report.get("latest_price") or report.get("principal_plan", {}).get("data", {}).get("latest_price")

    if not isinstance(latest_price, (int, float)):
        numeric_latest = _safe_float(latest_price)
        if numeric_latest is not None:
            latest_price = numeric_latest

    latest_price_value = latest_price if isinstance(latest_price, (int, float)) else None

    price_source_label = None
    if override_price_value is not None:
        latest_price = override_price_value
        latest_price_value = override_price_value
        if override_price_timestamp:
            latest_price_ts = override_price_timestamp
        price_source_label = override_price_source

    price_action_snapshot = report.get("price_action_snapshot") or {}
    overview = price_action_snapshot.get("overview") or {}
    key_levels = overview.get("key_levels") or []
    pa_supports: List[Dict[str, Any]] = []
    pa_resistances: List[Dict[str, Any]] = []
    for entry in key_levels:
        if not isinstance(entry, dict):
            continue
        price = _safe_float(entry.get("price"))
        if price is None:
            continue
        cleaned = {
            "price": price,
            "distance_pct": _safe_float(entry.get("distance_pct")),
            "timeframe": entry.get("timeframe"),
            "source": "price-action",
        }
        level_type = str(entry.get("type") or "").lower()
        if level_type == "support":
            pa_supports.append(cleaned)
        elif level_type == "resistance":
            pa_resistances.append(cleaned)

    strategy_label_map = {
        "day_trading": "Day Trading",
        "swing_trading": "Swing Trading",
        "longterm_trading": "Long-Term Trading",
    }
    strategy_label = (
        strategy.get("label")
        or strategy_label_map.get(strategy_key)
        or strategy.get("name")
        or strategy.get("strategy")
        or strategy_key.replace("_", " ").title()
    )

    strategy_levels = _extract_strategy_levels(
        strategy,
        strategy_key=strategy_key,
        strategy_label=strategy_label,
        latest_price=latest_price_value,
    )
    strategy_supports = strategy_levels.get("supports") or []
    strategy_resistances = strategy_levels.get("resistances") or []

    support_levels: List[Dict[str, Any]] = []
    resistance_levels: List[Dict[str, Any]] = []
    support_seen: Set[float] = set()
    resistance_seen: Set[float] = set()

    def _append_level(
        collection: List[Dict[str, Any]],
        seen: Set[float],
        level: Dict[str, Any],
        default_source: str,
    ) -> None:
        price_val = _safe_float(level.get("price"))
        if price_val is None:
            return
        key = round(price_val, 4)
        if key in seen:
            return
        clone = dict(level)
        if not clone.get("timeframe"):
            clone["timeframe"] = strategy_label
        clone.setdefault("source", default_source)
        collection.append(clone)
        seen.add(key)

    for level in strategy_supports:
        _append_level(support_levels, support_seen, level, level.get("source") or "strategy-text")
    for level in strategy_resistances:
        _append_level(resistance_levels, resistance_seen, level, level.get("source") or "strategy-text")

    for level in pa_supports:
        if len(support_levels) >= 4:
            break
        _append_level(support_levels, support_seen, level, level.get("source") or "price-action")
    for level in pa_resistances:
        if len(resistance_levels) >= 4:
            break
        _append_level(resistance_levels, resistance_seen, level, level.get("source") or "price-action")

    def _level_price(level: Optional[Dict[str, Any]]) -> Optional[float]:
        if not isinstance(level, dict):
            return None
        return _safe_float(level.get("price"))

    def _sort_by_price(levels: List[Dict[str, Any]]) -> None:
        def _key(lvl: Dict[str, Any]) -> float:
            price_val = _level_price(lvl)
            return price_val if price_val is not None else float("inf")

        levels.sort(key=_key)

    _sort_by_price(support_levels)
    _sort_by_price(resistance_levels)

    # Prefer levels on the expected side of the current price when anchoring the gauge.

    def _select_support_pair(
        levels: List[Dict[str, Any]],
        base_price: Optional[float],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        valid_levels = [lvl for lvl in levels if _level_price(lvl) is not None]
        if not valid_levels:
            return None, None
        if isinstance(base_price, (int, float)):
            below = [lvl for lvl in valid_levels if (_level_price(lvl) or base_price) <= base_price]
            below.sort(key=lambda lvl: _level_price(lvl) or 0.0, reverse=True)
            above = [lvl for lvl in valid_levels if (_level_price(lvl) or base_price) > base_price]
            above.sort(key=lambda lvl: _level_price(lvl) or 0.0)
            ordered: List[Dict[str, Any]] = below + above
        else:
            ordered = sorted(valid_levels, key=lambda lvl: _level_price(lvl) or 0.0, reverse=True)
        if not ordered:
            ordered = valid_levels
        primary = ordered[0]
        secondary = ordered[1] if len(ordered) > 1 else primary
        if primary and secondary:
            primary_price = _level_price(primary)
            secondary_price = _level_price(secondary)
            if (
                primary_price is not None
                and secondary_price is not None
                and secondary_price > primary_price
            ):
                primary, secondary = secondary, primary
        return primary, secondary

    def _select_resistance_pair(
        levels: List[Dict[str, Any]],
        base_price: Optional[float],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        valid_levels = [lvl for lvl in levels if _level_price(lvl) is not None]
        if not valid_levels:
            return None, None
        if isinstance(base_price, (int, float)):
            above = [lvl for lvl in valid_levels if (_level_price(lvl) or base_price) >= base_price]
            above.sort(key=lambda lvl: _level_price(lvl) or float("inf"))
            below = [lvl for lvl in valid_levels if (_level_price(lvl) or base_price) < base_price]
            below.sort(key=lambda lvl: _level_price(lvl) or 0.0, reverse=True)
            ordered = above + below
        else:
            ordered = sorted(valid_levels, key=lambda lvl: _level_price(lvl) or float("inf"))
        if not ordered:
            ordered = valid_levels
        primary = ordered[0]
        secondary = ordered[1] if len(ordered) > 1 else primary
        if primary and secondary:
            primary_price = _level_price(primary)
            secondary_price = _level_price(secondary)
            if (
                primary_price is not None
                and secondary_price is not None
                and secondary_price < primary_price
            ):
                primary, secondary = secondary, primary
        return primary, secondary

    s1_level, s2_level = _select_support_pair(support_levels, latest_price_value)
    r1_level, r2_level = _select_resistance_pair(resistance_levels, latest_price_value)

    def _distance_to(level: Dict[str, Any]) -> Optional[float]:
        if latest_price_value is None:
            return level.get("distance_pct")
        level_price = _safe_float(level.get("price"))
        if level_price in (None, 0):
            return level.get("distance_pct")
        return ((latest_price_value - level_price) / level_price) * 100

    radar_levels = {
        "s1": s1_level,
        "s2": s2_level,
        "r1": r1_level,
        "r2": r2_level,
    }

    for level_key in ("s1", "s2", "r1", "r2"):
        level = radar_levels.get(level_key)
        if isinstance(level, dict):
            distance_value = _distance_to(level)
            if distance_value is not None:
                level["distance_pct"] = distance_value

    def _format_level(level: Optional[Dict[str, Any]]) -> Optional[str]:
        if not isinstance(level, dict):
            return None
        price_val = _safe_float(level.get("price"))
        if price_val is None:
            return None
        return f"{price_val:,.2f}"

    def _formatted_distance(level: Optional[Dict[str, Any]]) -> Optional[str]:
        distance_val = _distance_to(level or {}) if isinstance(level, dict) else None
        if distance_val is None:
            return None
        return f"{distance_val:+.2f}%"

    s1_price = _safe_float((radar_levels.get("s1") or {}).get("price"))
    r1_price = _safe_float((radar_levels.get("r1") or {}).get("price"))

    zone = "wait"
    if latest_price is not None:
        if s1_price:
            proximity = abs((latest_price - s1_price) / s1_price) * 100
            if proximity <= 0.2:
                zone = "buy"
        if r1_price and zone == "wait":
            proximity = abs((r1_price - latest_price) / r1_price) * 100
            if proximity <= 0.2:
                zone = "sell"

    consensus_label = str(consensus.get("overall_recommendation") or "Neutral").upper()
    consensus_confidence = str(consensus.get("confidence") or "medium").upper()
    strength = _safe_float(consensus.get("strength"))

    def _confidence_to_score(label: str, fallback: float = 50.0) -> float:
        mapping = {
            "VERY HIGH": 95.0,
            "HIGH": 85.0,
            "MEDIUM": 65.0,
            "LOW": 45.0,
            "VERY LOW": 30.0,
        }
        return mapping.get(label.upper(), fallback)

    confidence_score = strength if strength is not None else _confidence_to_score(consensus_confidence)
    confidence_score = max(5.0, min(float(confidence_score), 100.0))

    def _title_case(text: str) -> str:
        return text[:1].upper() + text[1:].lower() if text else ""

    consensus_title = _title_case(consensus.get("overall_recommendation", "Neutral"))
    confidence_title = _title_case(consensus.get("confidence", "Medium"))

    def _classify_bias(text: str) -> Tuple[str, float]:
        normalized = text.lower()
        if "strong" in normalized and "bull" in normalized:
            return "Strongly Bullish", 90.0
        if "strong" in normalized and "bear" in normalized:
            return "Strongly Bearish", 10.0
        if "bear" in normalized:
            return "Bearish", 25.0
        if "bull" in normalized:
            return "Slightly Bullish", 75.0
        if "neutral" in normalized:
            return "Neutral", 50.0
        return text.title() if text else "Mixed", 55.0

    immediate_bias = price_action_output.get("immediate_bias") or overview.get("trend_alignment") or "Mixed"
    bias_label, bias_score = _classify_bias(str(immediate_bias))

    range_state = overview.get("range_state") or price_action_snapshot.get("per_timeframe", {}).get("30m", {}).get("structure", {}).get("range_state")
    if isinstance(range_state, str):
        range_state = range_state.lower()
    volatility_mapping = {
        "contracting": ("Low", 35.0),
        "sideways": ("Normal", 55.0),
        "expanding": ("Elevated", 75.0),
        "trending": ("Normal", 55.0),
        "volatile": ("High Risk", 85.0),
    }
    volatility_label, volatility_score = volatility_mapping.get(range_state, ("Normal", 55.0))

    signal_mapping = {
        "STRONG BUY": ("Strong Buy", 90.0),
        "BUY": ("Buy", 75.0),
        "HOLD": ("Neutral", 55.0),
        "SELL": ("Sell", 35.0),
        "STRONG SELL": ("Strong Sell", 15.0),
    }
    signal_label, signal_score = signal_mapping.get(consensus_label, (consensus_title or "Neutral", confidence_score))

    summary_text = str(strategy.get("summary") or principal.get("summary") or "")
    risk_note = str(price_action_output.get("structure", {}).get("risk_note") or "")

    def _first_sentence(text: str) -> str:
        if not text:
            return ""
        for delimiter in (". ", "\n", ".\n"):
            if delimiter in text:
                return text.split(delimiter)[0].strip().rstrip(".")
        return text.strip().rstrip(".")

    headline_sentence = _first_sentence(summary_text)
    risk_sentence = _first_sentence(risk_note)

    zone_copy = {
        "buy": {
            "title": "BUY",
            "subtitle": "Scale In",
            "icon": "buy",
        },
        "sell": {
            "title": "SELL",
            "subtitle": "Reduce",
            "icon": "sell",
        },
        "wait": {
            "title": "WAIT",
            "subtitle": "No-Trade Zone",
            "icon": "wait",
        },
    }

    primary_zone = zone_copy.get(zone, zone_copy["wait"]).copy()
    if intent_slug == "sell" and zone == "buy":
        primary_zone.update({"title": "HOLD", "subtitle": "Let Price Stabilise", "icon": "wait"})
    elif intent_slug == "buy" and zone == "sell":
        primary_zone.update({"title": "AVOID", "subtitle": "Risk Elevated", "icon": "sell"})

    explanation_lines: List[str] = []
    if headline_sentence:
        explanation_lines.append(headline_sentence)
    if risk_sentence and risk_sentence not in explanation_lines:
        explanation_lines.append(risk_sentence)
    if not explanation_lines:
        explanation_lines.append("Monitoring key levels before committing capital.")

    scenario_templates = {
        "bullish": {
            "title": "Bullish Scenario",
            "icon": "bull",
        },
        "bearish": {
            "title": "Bearish Scenario",
            "icon": "bear",
        },
        "range": {
            "title": "No-Trade Scenario",
            "icon": "range",
        },
    }

    scenarios: Dict[str, Dict[str, Any]] = {}
    for step in (strategy.get("next_actions") or []):
        if len(scenarios) == 3:
            break
        text = str(step).strip()
        lower = text.lower()
        if not text:
            continue
        if "above" in lower or "break" in lower and "above" in lower:
            key = "bullish"
        elif "below" in lower or "break" in lower and "below" in lower or "falls" in lower:
            key = "bearish"
        else:
            key = "range"
        if key in scenarios:
            continue
        scenarios[key] = {
            **scenario_templates[key],
            "body": text,
        }

    for missing_key in ("bullish", "bearish", "range"):
        scenarios.setdefault(
            missing_key,
            {
                **scenario_templates[missing_key],
                "body": "Awaiting additional AI guidance.",
            },
        )

    primary_entry = s1_price or latest_price
    add_entry = _safe_float((radar_levels.get("s2") or {}).get("price")) or primary_entry
    breakout_entry = r1_price

    target_prices = [lvl for lvl in (radar_levels.get("r1"), radar_levels.get("r2")) if isinstance(lvl, dict)]
    target_values = [f"{_safe_float(lvl.get('price')):,.2f}" for lvl in target_prices if _safe_float(lvl.get("price"))]
    stop_price = _safe_float((radar_levels.get("s2") or {}).get("price")) or primary_entry

    sizing_ranges = {
        "HIGH": "40–60%",
        "MEDIUM": "30–40%",
        "LOW": "20–30%",
    }
    position_size = sizing_ranges.get(consensus_confidence, "15–25%")

    risk_tips = (technical_output.get("risk_management") or {}).get("risk_tips") or []
    risk_notes: List[str] = []
    if risk_sentence:
        risk_notes.append(risk_sentence)
    for tip in risk_tips[:2]:
        if tip and tip not in risk_notes:
            risk_notes.append(str(tip))

    def _fmt(value: Optional[float]) -> Optional[str]:
        if value is None:
            return None
        return f"{value:,.2f}"

    alerts: List[str] = []
    if r1_price:
        alerts.append(f"Alert me if price breaks ABOVE {_fmt(r1_price)}")
    if s1_price:
        alerts.append(f"Alert me if price falls BELOW {_fmt(s1_price)}")
    if zone == "buy":
        alerts.append("Alert me if price enters BUY zone")
    if target_values:
        alerts.append(f"Alert me if price hits target {target_values[0]}")

    chart_range_min = _safe_float((radar_levels.get("s2") or {}).get("price")) or s1_price or latest_price or 0
    chart_range_max = _safe_float((radar_levels.get("r2") or {}).get("price")) or r1_price or latest_price or chart_range_min
    chart_position = None
    if latest_price is not None and chart_range_max not in (None, chart_range_min):
        span = chart_range_max - chart_range_min
        if span:
            chart_position = max(0.0, min(1.0, (latest_price - chart_range_min) / span))

    tags = []
    if bias_label:
        tags.append(f"Bias: {bias_label}")
    if overview.get("trend_alignment"):
        tags.append(f"Trend: {overview['trend_alignment'].title()}")
    if s1_price and r1_price:
        tags.append(f"Range: {s1_price:,.2f}–{r1_price:,.2f}")
    tags.append(f"Volatility: {volatility_label}")
    candlestick_note = price_action_output.get("candlestick_notes")
    if candlestick_note:
        first_note = _first_sentence(str(candlestick_note))
        if first_note:
            tags.append(first_note)

    plan_generated = principal.get("generated_display") or principal.get("generated")

    return {
        "symbol": symbol,
        "latest_price": f"{latest_price:,.2f}" if isinstance(latest_price, (int, float)) else latest_price,
        "latest_price_value": latest_price,
        "latest_price_timestamp": latest_price_ts,
        "latest_price_source": price_source_label,
        "generated_display": plan_generated,
        "primary_action": {
            **primary_zone,
            "confidence": confidence_title,
            "confidence_score": confidence_score,
            "explanation": explanation_lines[:2],
        },
        "gauges": {
            "bias": {"label": bias_label, "score": bias_score},
            "volatility": {"label": volatility_label, "score": volatility_score},
            "signal": {"label": signal_label, "score": signal_score},
        },
        "radar": {
            "s2": {
                "value": _format_level(radar_levels.get("s2")),
                "distance": _formatted_distance(radar_levels.get("s2")),
            },
            "s1": {
                "value": _format_level(radar_levels.get("s1")),
                "distance": _formatted_distance(radar_levels.get("s1")),
            },
            "price": {
                "value": f"{latest_price:,.2f}" if isinstance(latest_price, (int, float)) else None,
                "position": chart_position,
            },
            "r1": {
                "value": _format_level(radar_levels.get("r1")),
                "distance": _formatted_distance(radar_levels.get("r1")),
            },
            "r2": {
                "value": _format_level(radar_levels.get("r2")),
                "distance": _formatted_distance(radar_levels.get("r2")),
            },
        },
        "traffic_light": zone,
        "scenarios": [scenarios["bullish"], scenarios["bearish"], scenarios["range"]],
        "trade_plan": {
            "entry": {
                "primary": _fmt(primary_entry),
                "scale": _fmt(add_entry),
                "breakout": _fmt(breakout_entry),
            },
            "targets": target_values[:2],
            "stops": [_fmt(stop_price)],
            "position_size": position_size,
            "risk_notes": risk_notes,
        },
        "alerts": alerts,
        "chart": {
            "min": _fmt(chart_range_min),
            "max": _fmt(chart_range_max),
            "position": chart_position,
            "min_value": chart_range_min,
            "max_value": chart_range_max,
        },
        "tags": tags[:6],
        "timeframes": timeframes,
        "favorite": False,
    }

@app.get("/report-center", response_class=HTMLResponse)
async def report_center_page(
    request: Request,
    date: Optional[str] = None,
    symbol: Optional[str] = None,
    user: User = Depends(get_current_user_sync),
):
    subscription, allowed = _check_report_center_access(user)
    if not allowed:
        query = {"reason": "report_center_locked"}
        if subscription and subscription.plan and subscription.plan.slug:
            query["current_plan"] = subscription.plan.slug
        redirect_url = "/subscribe"
        if query:
            redirect_url = f"/subscribe?{urllib.parse.urlencode(query)}"
        return RedirectResponse(url=redirect_url, status_code=status.HTTP_303_SEE_OTHER)

    target_date, selected_date_iso, selected_date_label = _resolve_report_center_date(date)
    today_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    symbol_filter = None
    if symbol:
        symbol_filter_candidate = str(symbol).strip().upper()
        if symbol_filter_candidate:
            symbol_filter = symbol_filter_candidate

    raw_reports = fetch_reports_for_date(
        target_date,
        symbol=symbol_filter,
        max_reports=DEFAULT_REPORT_CENTER_REPORT_LIMIT,
    )

    prepared_reports: List[Dict[str, Any]] = []
    excluded_reports: List[Dict[str, Any]] = []
    for report in raw_reports:
        summary = _summarize_report_center_entry(report)
        if summary:
            prepared_reports.append(summary)
        else:
            excluded_reports.append(report)

    prepared_reports.sort(key=lambda item: item.get("generated_unix") or 0, reverse=True)

    available_symbols = sorted(
        {
            str(report.get("symbol") or "").upper()
            for report in raw_reports
            if report.get("symbol")
        }
    )

    logger.info(
        "User %s loaded report center date=%s symbol=%s reports=%s prepared=%s",
        user.id,
        selected_date_iso,
        symbol_filter,
        len(raw_reports),
        len(prepared_reports),
    )

    context = {
        "request": request,
        "user": user,
        "subscription": _serialize_subscription(subscription) if subscription else None,
        "selected_date": selected_date_iso,
        "selected_date_label": selected_date_label,
        "selected_symbol": symbol_filter,
        "selected_symbol_input": symbol or "",
        "today_date": today_iso,
        "reports": prepared_reports,
        "report_count": len(prepared_reports),
        "raw_report_count": len(raw_reports),
        "excluded_report_count": len(excluded_reports),
        "available_symbols": available_symbols,
        "max_reports": DEFAULT_REPORT_CENTER_REPORT_LIMIT,
    }
    return templates.TemplateResponse("report_center.html", context)


_STRATEGY_KEY_BY_TIMEFRAME = {
    "day": "day_trading",
    "swing": "swing_trading",
    "long": "longterm_trading",
}


def _normalise_timeframe(value: str) -> str:
    slug = (value or "day").strip().lower()
    return slug if slug in _STRATEGY_KEY_BY_TIMEFRAME else "day"


def _normalise_intent(value: str) -> str:
    slug = (value or "buy").strip().lower()
    return slug if slug in {"buy", "sell"} else "buy"


@app.get("/action-center", response_class=HTMLResponse)
async def action_center_page(
    request: Request,
    symbol: Optional[str] = None,
    timeframe: str = "day",
    intent: str = "buy",
    user: User = Depends(get_current_user_sync),
):
    subscription, allowed = _check_report_center_access(user)
    if not allowed:
        query = {"reason": "action_center_locked"}
        if subscription and subscription.plan and subscription.plan.slug:
            query["current_plan"] = subscription.plan.slug
        redirect_url = "/subscribe"
        if query:
            redirect_url = f"/subscribe?{urllib.parse.urlencode(query)}"
        return RedirectResponse(url=redirect_url, status_code=status.HTTP_303_SEE_OTHER)

    timeframe_slug = _normalise_timeframe(timeframe)
    intent_slug = _normalise_intent(intent)
    symbol_upper = (symbol or DEFAULT_ACTION_CENTER_SYMBOL).strip().upper() or DEFAULT_ACTION_CENTER_SYMBOL

    live_stream = _get_live_stream(request.app)
    live_price_available = bool(live_stream)

    with SessionLocal() as session:
        is_favorited = _is_symbol_favorited(session, user.id, symbol_upper)
        favorite_equity_symbols = _favorite_symbols(session, user.id, market="equity")
        favorite_crypto_symbols = _favorite_symbols(session, user.id, market="crypto")

    strategy_key = _STRATEGY_KEY_BY_TIMEFRAME[timeframe_slug]
    raw_report = _fetch_latest_action_report(symbol_upper)

    favorite_symbol_set = {sym.upper() for sym in favorite_equity_symbols}
    dashboards: List[Dict[str, Any]] = []
    price_overrides: Dict[str, Dict[str, Any]] = {}

    def _build_dashboard(symbol_key: str, report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        resolved_report = report or _fetch_latest_action_report(symbol_key)
        dashboard_payload: Optional[Dict[str, Any]] = None
        dashboard_summary: Optional[str] = None
        error_message: Optional[str] = None

        if resolved_report:
            price_override = price_overrides.get(symbol_key)
            dashboard_payload = _derive_action_center_view(
                resolved_report,
                strategy_key=strategy_key,
                intent=intent_slug,
                price_override=price_override,
            )
            dashboard_payload["favorite"] = symbol_key in favorite_symbol_set
            principal_data = (resolved_report.get("principal_plan") or {}).get("data") or {}
            strategies = principal_data.get("strategies") or {}
            active_strategy = strategies.get(strategy_key) or {}
            dashboard_summary = active_strategy.get("summary")
        else:
            error_message = "No recent AI analysis found for this symbol. Try running a new multi-agent analysis first."

        return {
            "symbol": symbol_key,
            "is_favorited": symbol_key in favorite_symbol_set,
            "strategy_summary": dashboard_summary,
            "action_data": dashboard_payload,
            "error_message": error_message,
            "report_blob": resolved_report.get("_blob_name") if resolved_report else None,
            "timeframe": timeframe_slug,
            "intent": intent_slug,
        }

    symbol_order: list[str] = []
    if symbol_upper not in symbol_order:
        symbol_order.append(symbol_upper)
    for fav_symbol in favorite_equity_symbols:
        fav_upper = fav_symbol.upper()
        if fav_upper not in symbol_order:
            symbol_order.append(fav_upper)

    price_symbols: Set[str] = set(symbol_order)
    price_symbols.update(favorite_crypto_symbols)
    price_overrides = await _resolve_rest_prices(price_symbols)

    for sym in symbol_order:
        if sym == symbol_upper:
            dashboards.append(_build_dashboard(sym, raw_report))
        else:
            dashboards.append(_build_dashboard(sym))

    await _ensure_equity_stream_symbols(request.app, symbol_order)

    primary_dashboard = dashboards[0] if dashboards else None

    dashboards_json = json.dumps(
        [
            {
                "symbol": dash["symbol"],
                "is_favorited": dash["is_favorited"],
                "strategy_summary": dash["strategy_summary"],
                "error_message": dash["error_message"],
                "action_data": dash["action_data"],
                "report_blob": dash["report_blob"],
                "timeframe": dash.get("timeframe", timeframe_slug),
                "intent": dash.get("intent", intent_slug),
            }
            for dash in dashboards
        ],
        separators=(",", ":"),
    )

    context = {
        "request": request,
        "user": user,
        "symbol": primary_dashboard["symbol"] if primary_dashboard else symbol_upper,
        "subscription": _serialize_subscription(subscription) if subscription else None,
        "timeframe": timeframe_slug,
        "intent": intent_slug,
        "dashboards": dashboards,
        "dashboards_json": dashboards_json,
        "primary_dashboard": primary_dashboard,
        "latest_report_blob": primary_dashboard["report_blob"] if primary_dashboard else None,
        "is_favorited": primary_dashboard["is_favorited"] if primary_dashboard else is_favorited,
        "favorite_symbols": favorite_equity_symbols,
        "favorite_crypto_symbols": favorite_crypto_symbols,
        "live_price_enabled": live_price_available,
        "rest_price_overrides": price_overrides,
    }

    return templates.TemplateResponse("action_center.html", context)


@app.get("/api/action-center")
async def action_center_api(
    symbol: str,
    timeframe: str = "day",
    intent: str = "buy",
    user: User = Depends(get_current_user_sync),
):
    subscription, allowed = _check_report_center_access(user)
    if not allowed:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Subscription upgrade required for Action Center")
    symbol_upper = (symbol or DEFAULT_ACTION_CENTER_SYMBOL).strip().upper() or DEFAULT_ACTION_CENTER_SYMBOL
    timeframe_slug = _normalise_timeframe(timeframe)
    intent_slug = _normalise_intent(intent)
    report = _fetch_latest_action_report(symbol_upper)
    if not report:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": "No recent analysis available for the requested symbol."},
        )

    strategy_key = _STRATEGY_KEY_BY_TIMEFRAME[timeframe_slug]
    price_overrides = await _resolve_rest_prices({symbol_upper})
    payload = _derive_action_center_view(
        report,
        strategy_key=strategy_key,
        intent=intent_slug,
        price_override=price_overrides.get(symbol_upper),
    )
    with SessionLocal() as session:
        is_favorited = _is_symbol_favorited(session, user.id, symbol_upper)
    payload["favorite"] = is_favorited
    return JSONResponse(
        content={
            "symbol": symbol_upper,
            "timeframe": timeframe_slug,
            "intent": intent_slug,
            "action": payload,
            "source_blob": report.get("_blob_name"),
            "favorite": is_favorited,
        }
    )


class FavoriteTogglePayload(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=24)
    follow: bool = True


@app.post("/api/action-center/favorites")
async def set_action_center_favorite(
    payload: FavoriteTogglePayload,
    request: Request,
    user: User = Depends(get_current_user_sync),
):
    symbol_clean = _sanitize_symbol(payload.symbol)
    if not symbol_clean:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="A valid symbol is required.")

    with SessionLocal() as session:
        existing = (
            session.query(UserFavoriteSymbol)
            .filter(UserFavoriteSymbol.user_id == user.id, UserFavoriteSymbol.symbol == symbol_clean)
            .first()
        )
        if payload.follow:
            if existing is None:
                session.add(UserFavoriteSymbol(user_id=user.id, symbol=symbol_clean))
                session.commit()
            else:
                session.commit()
        else:
            if existing is not None:
                session.delete(existing)
                session.commit()
            else:
                session.commit()
        current_state = _is_symbol_favorited(session, user.id, symbol_clean)
        favorites = _favorite_symbols(session, user.id, market="equity")

    if current_state:
        symbol_set = {symbol_clean}
        equity_symbols = _filter_symbols_for_market(symbol_set, "equity")
        if equity_symbols:
            await _ensure_equity_stream_symbols(request.app, equity_symbols)
        crypto_symbols = _filter_symbols_for_market(symbol_set, "crypto")
        if crypto_symbols:
            await _ensure_crypto_stream_symbols(request.app, crypto_symbols)

    return {
        "symbol": symbol_clean,
        "follow": current_state,
        "favorites": favorites,
    }


@app.get("/api/action-center/favorites")
async def list_action_center_favorites(user: User = Depends(get_current_user_sync)):
    with SessionLocal() as session:
        favorites = _favorite_symbols(session, user.id, limit=50, market="equity")
    return {"symbols": favorites}


@app.get("/api/live-price")
async def get_live_price(
    symbol: str,
    request: Request,
    market: Optional[str] = None,
    user: User = Depends(get_current_user_sync),
):
    symbol_clean = _sanitize_symbol(symbol)
    if not symbol_clean:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="A valid symbol is required.")

    market_slug = (market or "equity").strip().lower()
    if market_slug not in {"equity", "crypto"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported market requested.")

    logger.debug(
        "Live price request user=%s market=%s symbol=%s",
        getattr(user, "id", "?"),
        market_slug,
        symbol_clean,
    )

    if market_slug == "crypto":
        await _ensure_crypto_stream_symbols(request.app, {symbol_clean})
        service = _get_crypto_stream(request.app)
        fallback_fetcher = _fetch_rest_crypto_price
        unavailable_detail = "Live crypto stream unavailable."
    else:
        await _ensure_equity_stream_symbols(request.app, {symbol_clean})
        service = _get_live_stream(request.app)
        fallback_fetcher = _fetch_rest_live_price
        unavailable_detail = "Live price stream unavailable."

    if not service:
        fallback = await fallback_fetcher(symbol_clean)
        if fallback:
            fallback.setdefault("market", market_slug)
            return fallback
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": unavailable_detail},
        )

    try:
        await service.ready(timeout=5.0)
    except Exception:
        pass

    update = service.get_latest(symbol_clean)
    if update is None:
        fallback = await fallback_fetcher(symbol_clean)
        if fallback:
            fallback.setdefault("market", market_slug)
            return fallback
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={"symbol": symbol_clean, "status": "pending", "market": market_slug},
        )

    payload = update.to_dict()
    payload["market"] = market_slug
    return {
        "symbol": payload.get("symbol", symbol_clean),
        "price": payload.get("last_trade_price"),
        "bid": payload.get("bid_price"),
        "ask": payload.get("ask_price"),
        "timestamp": payload.get("last_trade_timestamp"),
        "volume": payload.get("volume"),
        "source": payload.get("source"),
        "received_at": payload.get("received_at"),
        "market": payload.get("market", market_slug),
    }


def _extract_latest_price(snapshot: Any, symbol: str) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """Best-effort extraction of latest price metadata from analysis snapshots."""

    if snapshot is None:
        return (None, None, None)

    material = snapshot
    if isinstance(material, str):
        try:
            material = json.loads(material)
        except (TypeError, ValueError):
            return (None, None, None)

    if not isinstance(material, dict):
        return (None, None, None)

    symbol_candidates = [symbol, symbol.upper(), symbol.lower()]
    symbol_data = None
    for candidate in symbol_candidates:
        if candidate in material:
            symbol_data = material.get(candidate)
            break

    if not isinstance(symbol_data, dict):
        return (None, None, None)

    price = symbol_data.get("latest_price")
    timeframe = symbol_data.get("latest_price_timeframe")
    timestamp = symbol_data.get("latest_price_timestamp")

    price_map = symbol_data.get("price_by_timeframe")
    if price is None and isinstance(price_map, dict):
        preferred_order = ("1m", "2m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo")
        for candidate_tf in preferred_order:
            candidate_price = price_map.get(candidate_tf)
            if candidate_price is not None:
                price = candidate_price
                if timeframe is None:
                    timeframe = candidate_tf
                break
        if price is None:
            for candidate_tf, candidate_price in price_map.items():
                if candidate_price is not None:
                    price = candidate_price
                    if timeframe is None:
                        timeframe = candidate_tf
                    break

    try:
        price_value = float(price) if price is not None else None
    except (TypeError, ValueError):
        price_value = None

    if timestamp is None:
        timestamps_map = symbol_data.get("price_timestamps")
        if isinstance(timestamps_map, dict) and timeframe:
            timestamp = timestamps_map.get(timeframe)

    if timestamp is not None:
        timestamp = str(timestamp)

    return (price_value, timeframe, timestamp)


def _send_contact_email(payload: Dict[str, Any]) -> bool:
    if not CONTACT_EMAIL_RECIPIENT:
        logger.warning("Contact email recipient not configured; skipping send")
        return False

    name = payload.get("name") or "Unknown"
    email_address = payload.get("email") or "unknown@example.com"
    message = payload.get("message") or "(no message provided)"
    submitted_at = payload.get("submitted_at") or datetime.now(timezone.utc).isoformat()

    subject = f"{CONTACT_EMAIL_PREFIX} - {name}" if CONTACT_EMAIL_PREFIX else f"Contact Request - {name}"

    text_body = (
        "New contact request from VolatilX Report Center.\n\n"
        f"Name: {name}\n"
        f"Email: {email_address}\n"
        f"Submitted At: {submitted_at}\n"
        "\nMessage:\n"
        f"{message}\n"
    )

    safe_name = html.escape(str(name))
    safe_email = html.escape(str(email_address))
    safe_message = "<br />".join(html.escape(str(message)).splitlines())
    safe_submitted = html.escape(str(submitted_at))

    html_body = f"""
        <html>
            <body>
                <h2>New VolatilX Contact Request</h2>
                <p><strong>Name:</strong> {safe_name}</p>
                <p><strong>Email:</strong> {safe_email}</p>
                <p><strong>Submitted:</strong> {safe_submitted}</p>
                <hr />
                <p>{safe_message or '<em>(no message provided)</em>'}</p>
            </body>
        </html>
    """

    if ACS_CONNECTION_STRING:
        if EmailClient is None:
            logger.warning(
                "Azure Communication Services email library missing. Install 'azure-communication-email' or disable ACS to use SMTP."
            )
        else:
            sender_address = ACS_EMAIL_SENDER or CONTACT_EMAIL_SENDER
            if not sender_address:
                logger.warning("ACS sender address not configured; skipping ACS send")
            else:
                try:
                    client = EmailClient.from_connection_string(ACS_CONNECTION_STRING)
                    email_message: Dict[str, Any] = {
                        "senderAddress": sender_address,
                        "recipients": {"to": [{"address": CONTACT_EMAIL_RECIPIENT}]},
                        "content": {
                            "subject": subject,
                            "plainText": text_body,
                            "html": html_body,
                        },
                    }
                    if email_address:
                        email_message["replyTo"] = [{"address": email_address}]

                    poller = client.begin_send(email_message)
                    result = poller.result()
                    status = getattr(result, "status", None)
                    if status and str(status).lower() not in {"queued", "accepted", "succeeded"}:
                        logger.warning("ACS email send completed with status %s", status)
                    else:
                        logger.info("Contact email queued via ACS for %s", email_address)
                        return True
                except Exception:  # noqa: BLE001 - logging for operational visibility
                    logger.exception("Failed to send contact email via ACS")

    if not CONTACT_EMAIL_SENDER or not SMTP_HOST:
        logger.warning("No working email transport configured (ACS failed and SMTP incomplete)")
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["To"] = CONTACT_EMAIL_RECIPIENT
    msg["From"] = CONTACT_EMAIL_SENDER
    if email_address:
        msg["Reply-To"] = email_address
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as smtp:
            if SMTP_USE_TLS:
                smtp.starttls()
            if SMTP_USERNAME and SMTP_PASSWORD:
                smtp.login(SMTP_USERNAME, SMTP_PASSWORD)
            smtp.send_message(msg)
        logger.info("Contact email sent via SMTP for %s", email_address)
        return True
    except Exception:  # noqa: BLE001 - log and surface failure gracefully
        logger.exception("Failed to send contact email via SMTP")
        return False


def _clean_text_fragment(value: Any, *, max_items: Optional[int] = None) -> str:
    """Convert heterogeneous plan fragments into compact plain-text strings."""

    if value is None:
        return ""

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, (list, tuple, set)):
        items = list(value)
        if max_items is not None:
            items = items[:max_items]
        fragments = [frag for frag in (_clean_text_fragment(item) for item in items) if frag]
        return "; ".join(fragments)

    if isinstance(value, dict):
        fragments = []
        for key, val in value.items():
            cleaned_val = _clean_text_fragment(val)
            if not cleaned_val:
                continue
            label = str(key).replace("_", " ").title()
            fragments.append(f"{label}: {cleaned_val}")
        return "; ".join(fragments)

    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _round_decimal(value: Any, places: int = 2) -> Any:
    """Best-effort rounding for numeric values while preserving non-numeric input."""

    if value is None:
        return None

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value

    if math.isfinite(numeric):
        return round(numeric, places)

    return value


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    cleaned = cleaned.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        try:
            parsed = datetime.strptime(cleaned, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _prepare_strategy_for_report_center(strategy: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(strategy, dict):
        return None

    summary = _clean_text_fragment(strategy.get("summary"))
    next_actions_raw = strategy.get("next_actions")
    actions: List[str] = []
    if isinstance(next_actions_raw, (list, tuple, set)):
        for item in next_actions_raw:
            cleaned = _clean_text_fragment(item)
            if cleaned:
                actions.append(cleaned)
            if len(actions) >= 4:
                break

    result: Dict[str, Any] = {}
    if summary:
        result["summary"] = summary
    if actions:
        result["next_actions"] = actions

    confidence = strategy.get("confidence") or strategy.get("strength")
    if confidence:
        result["confidence"] = _clean_text_fragment(confidence)

    return result or None


def _extract_consensus_snapshot(report: Dict[str, Any], symbol: str) -> Optional[Dict[str, Any]]:
    snapshot_map = report.get("technical_snapshot")
    if not isinstance(snapshot_map, dict):
        return None

    candidates = [symbol, symbol.upper(), symbol.lower()]
    snapshot = None
    for candidate in candidates:
        candidate_snapshot = snapshot_map.get(candidate)
        if candidate_snapshot:
            snapshot = candidate_snapshot
            break

    if not isinstance(snapshot, dict):
        return None

    consensus = snapshot.get("consensus")
    summary: Dict[str, Any] = {
        "status": snapshot.get("status"),
        "timestamp": snapshot.get("timestamp"),
    }

    if isinstance(consensus, dict):
        summary["recommendation"] = consensus.get("overall_recommendation")
        summary["confidence"] = consensus.get("confidence")
        strength_value = _round_decimal(consensus.get("strength"))
        if strength_value is not None:
            summary["strength"] = strength_value
        for field in ("buy_signals", "sell_signals", "hold_signals"):
            if field in consensus:
                summary[field] = consensus.get(field)
        reasoning = consensus.get("reasoning")
        if isinstance(reasoning, list):
            cleaned_reasoning: List[str] = []
            for item in reasoning[:3]:
                cleaned_item = _clean_text_fragment(item)
                if cleaned_item:
                    cleaned_reasoning.append(cleaned_item)
            if cleaned_reasoning:
                summary["reasoning"] = cleaned_reasoning

    decisions = snapshot.get("decisions")
    if isinstance(decisions, dict):
        for focus_tf in ("1d", "4h", "1h"):
            details = decisions.get(focus_tf)
            if isinstance(details, dict):
                focus_summary = {
                    "timeframe": focus_tf,
                    "recommendation": details.get("recommendation"),
                    "confidence": details.get("confidence"),
                }
                for field in ("entry_price", "stop_loss", "take_profit", "risk_reward_ratio"):
                    if details.get(field) is not None:
                        focus_summary[field] = _round_decimal(details.get(field))
                summary["focus"] = focus_summary
                break

    return summary or None


def _extract_price_details(report: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    price_action = report.get("price_action_snapshot")
    price_info: Dict[str, Any] = {}
    price_action_summary: Dict[str, Any] = {}

    if isinstance(price_action, dict) and price_action.get("success"):
        per_timeframe = price_action.get("per_timeframe")
        if isinstance(per_timeframe, dict):
            for timeframe in ("1d", "4h", "1h", "30m"):
                timeframe_data = per_timeframe.get(timeframe)
                if isinstance(timeframe_data, dict):
                    price_data = timeframe_data.get("price")
                    if isinstance(price_data, dict):
                        close_price = _round_decimal(price_data.get("close"))
                        change_pct = _round_decimal(price_data.get("close_change_pct"))
                        price_info = {
                            "timeframe": timeframe,
                            "close": close_price,
                            "change_pct": change_pct,
                            "volume": price_data.get("volume"),
                            "timestamp": price_data.get("timestamp"),
                        }
                        break

        overview = price_action.get("overview")
        if isinstance(overview, dict):
            price_action_summary["trend_alignment"] = overview.get("trend_alignment")
            key_levels = overview.get("key_levels")
            if isinstance(key_levels, list) and key_levels:
                sorted_levels = sorted(
                    (
                        level
                        for level in key_levels
                        if isinstance(level, dict) and level.get("price") is not None
                    ),
                    key=lambda item: abs(_safe_float(item.get("distance_pct")) or 0.0),
                )
                cleaned_levels: List[Dict[str, Any]] = []
                for level in sorted_levels[:3]:
                    cleaned_level = dict(level)
                    cleaned_level["price"] = _round_decimal(level.get("price"))
                    if "distance_pct" in level:
                        cleaned_level["distance_pct"] = _round_decimal(level.get("distance_pct"))
                    cleaned_levels.append(cleaned_level)
                price_action_summary["key_levels"] = cleaned_levels

            patterns = overview.get("recent_patterns")
            if isinstance(patterns, list) and patterns:
                formatted_patterns: List[str] = []
                for pattern in patterns:
                    if len(formatted_patterns) >= 3:
                        break

                    if isinstance(pattern, dict):
                        timeframe = pattern.get("timeframe") or pattern.get("period")
                        pattern_name = pattern.get("pattern") or pattern.get("name")
                        confidence_raw = pattern.get("confidence")

                        parts: List[str] = []
                        if timeframe:
                            parts.append(str(timeframe).upper())
                        if pattern_name:
                            parts.append(str(pattern_name))

                        confidence_text = None
                        if confidence_raw is not None:
                            try:
                                confidence_value = float(confidence_raw)
                            except (TypeError, ValueError):
                                confidence_value = None

                            if confidence_value is not None and math.isfinite(confidence_value):
                                if abs(confidence_value) <= 1:
                                    confidence_value *= 100
                                confidence_text = f"{int(round(confidence_value))}% confidence"

                        description = ": ".join(parts[:2]) if parts else None
                        if description and confidence_text:
                            formatted_patterns.append(f"{description} ({confidence_text})")
                        elif description:
                            formatted_patterns.append(description)
                        elif confidence_text:
                            formatted_patterns.append(confidence_text)
                        else:
                            formatted_patterns.append(str(pattern))
                    else:
                        formatted_patterns.append(str(pattern))

                if formatted_patterns:
                    price_action_summary["recent_patterns"] = formatted_patterns

    return (price_info or None, price_action_summary or None)


def _summarize_report_center_entry(report: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    plan_wrapper = report.get("principal_plan")
    if not isinstance(plan_wrapper, dict) or not plan_wrapper.get("success"):
        return None

    plan_data = plan_wrapper.get("data")
    if not isinstance(plan_data, dict):
        return None

    symbol = plan_data.get("symbol") or report.get("symbol")
    if not symbol:
        return None
    symbol_clean = str(symbol).upper()

    generated_dt = _parse_iso_datetime(
        plan_data.get("generated")
        or plan_data.get("generated_at")
        or report.get("stored_at")
        or report.get("_blob_last_modified")
    )

    generated_iso = None
    generated_display = plan_data.get("generated_display")
    generated_unix = None
    if generated_dt:
        generated_iso = generated_dt.isoformat().replace("+00:00", "Z")
        generated_unix = int(generated_dt.timestamp())
        if not generated_display:
            generated_display = generated_dt.strftime("%b %d, %Y %H:%M UTC")

    strategies = plan_data.get("strategies")
    prepared_strategies: Dict[str, Any] = {}
    if isinstance(strategies, dict):
        labels = {
            "day_trading": "Day Trading",
            "swing_trading": "Swing Trading",
            "longterm_trading": "Long-Term Trading",
        }
        for key, label in labels.items():
            prepared = _prepare_strategy_for_report_center(strategies.get(key))
            if prepared:
                prepared["label"] = label
                prepared_strategies[key] = prepared

    consensus = _extract_consensus_snapshot(report, symbol_clean)
    price_info, price_action = _extract_price_details(report)

    summary: Dict[str, Any] = {
        "symbol": symbol_clean,
        "generated_iso": generated_iso,
        "generated_display": generated_display,
        "generated_unix": generated_unix,
        "strategies": prepared_strategies,
        "consensus": consensus,
        "price": price_info,
        "price_action": price_action,
        "stored_at": report.get("stored_at"),
        "source": {
            "blob": report.get("_blob_name"),
            "user_id": report.get("user_id"),
            "ai_job_id": report.get("ai_job_id"),
        },
    }

    return summary

def structure_trading_data_for_ai(result_data: any, symbol: str) -> dict:
    """Structure the trading agent result data for AI analysis"""
    
    print(f"=== STRUCTURING DATA FOR AI ===")
    print(f"Input type: {type(result_data)}")
    
    # Initialize structured data with defaults
    structured = {
        "symbol": symbol,
        "timestamp": str(datetime.now()),
        "analysis_type": "day_trading_analysis",
        "current_price": "N/A",
        "price_change": "N/A",
        "volume": "N/A",
        "market_sentiment": "N/A",
        "signals": {},
        "recommendations": {},
        "technical_indicators": {},
        "trading_result": result_data,
        "data_source": "MultiSymbolDayTraderAgent"
    }
    
    try:
        if isinstance(result_data, dict):
            print("Processing dictionary result...")
            print("Available keys:", list(result_data.keys()))
            
            # Extract data based on common key patterns
            for key, value in result_data.items():
                key_lower = key.lower()
                print(f"Processing key: {key} = {str(value)[:100]}...")
                
                # Price-related data
                if any(price_key in key_lower for price_key in ['price', 'close', 'last']):
                    if isinstance(value, (int, float)):
                        structured["current_price"] = value
                    elif isinstance(value, dict) and 'close' in value:
                        structured["current_price"] = value['close']
                
                # Change/movement data
                elif any(change_key in key_lower for change_key in ['change', 'move', 'diff']):
                    structured["price_change"] = value
                
                # Volume data
                elif 'volume' in key_lower:
                    structured["volume"] = value
                
                # Signal data
                elif any(signal_key in key_lower for signal_key in ['signal', 'buy', 'sell', 'action']):
                    if isinstance(value, dict):
                        structured["signals"].update(value)
                    else:
                        structured["signals"][key] = value
                
                # Recommendation data
                elif any(rec_key in key_lower for rec_key in ['recommend', 'advice', 'suggest']):
                    if isinstance(value, dict):
                        structured["recommendations"].update(value)
                    else:
                        structured["recommendations"][key] = value
                
                # Technical indicator data
                elif any(tech_key in key_lower for tech_key in ['rsi', 'macd', 'sma', 'ema', 'indicator', 'technical']):
                    if isinstance(value, dict):
                        structured["technical_indicators"].update(value)
                    else:
                        structured["technical_indicators"][key] = value
            
            # Try to extract nested data
            if 'analysis' in result_data:
                analysis_data = result_data['analysis']
                if isinstance(analysis_data, dict):
                    structured["signals"].update(analysis_data.get('signals', {}))
                    structured["recommendations"].update(analysis_data.get('recommendations', {}))
            
            # Look for timeframe-specific data
            for timeframe in ['1m', '5m', '15m', '30m', '1h', '1d']:
                if timeframe in result_data:
                    tf_data = result_data[timeframe]
                    if isinstance(tf_data, dict):
                        structured["technical_indicators"][f"{timeframe}_data"] = tf_data
            
            # Determine market sentiment based on signals
            if structured["signals"]:
                buy_signals = sum(1 for k, v in structured["signals"].items() 
                                if 'buy' in str(v).lower() or 'bullish' in str(v).lower())
                sell_signals = sum(1 for k, v in structured["signals"].items() 
                                 if 'sell' in str(v).lower() or 'bearish' in str(v).lower())
                
                if buy_signals > sell_signals:
                    structured["market_sentiment"] = "Bullish"
                elif sell_signals > buy_signals:
                    structured["market_sentiment"] = "Bearish"
                else:
                    structured["market_sentiment"] = "Neutral"
        
        elif isinstance(result_data, list) and len(result_data) > 0:
            print("Processing list result...")
            # If it's a list, try to process the first item
            first_item = result_data[0]
            if isinstance(first_item, dict):
                return structure_trading_data_for_ai(first_item, symbol)
            else:
                structured["raw_analysis"] = str(result_data)
                structured["market_sentiment"] = "Data processed"
        
        else:
            print(f"Processing {type(result_data)} result...")
            structured["raw_analysis"] = str(result_data)
            structured["market_sentiment"] = "Analysis completed"
    
    except Exception as e:
        print(f"Error structuring data: {e}")
        structured["extraction_error"] = str(e)
        structured["market_sentiment"] = "Error in data processing"
    
    print("=== FINAL STRUCTURED DATA ===")
    print(f"Current Price: {structured['current_price']}")
    print(f"Signals: {len(structured['signals'])} items")
    print(f"Recommendations: {len(structured['recommendations'])} items")
    print(f"Technical Indicators: {len(structured['technical_indicators'])} items")
    print(f"Market Sentiment: {structured['market_sentiment']}")
    
    return structured

##############################################################################################
# Billing Endpoints
##############################################################################################


@app.get("/api/billing/plans")
async def list_subscription_plans(user: User = Depends(get_current_user_sync)):
    with SessionLocal() as session:
        plans = (
            session.query(SubscriptionPlan)
            .filter(SubscriptionPlan.is_active.is_(True))
            .order_by(SubscriptionPlan.monthly_price_cents.asc())
            .all()
        )
        current_subscription = _find_relevant_subscription(session, user.id)

    payload = {
        "plans": [_serialize_plan(plan) for plan in plans],
        "currency": "usd",
        "current_subscription": _serialize_subscription(current_subscription),
    }
    return JSONResponse(content=payload)


@app.get("/api/billing/subscription")
async def get_current_subscription(user: User = Depends(get_current_user_sync)):
    with SessionLocal() as session:
        subscription = _find_relevant_subscription(session, user.id)

    return JSONResponse(content={"subscription": _serialize_subscription(subscription)})


@app.post("/api/billing/checkout-session")
async def create_checkout_session(request: Request, user: User = Depends(get_current_user_sync)):
    payload = await request.json()
    plan_slug = payload.get("plan_slug")
    if not plan_slug:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": "plan_slug is required."},
        )

    with SessionLocal() as session:
        plan = (
            session.query(SubscriptionPlan)
            .filter(SubscriptionPlan.slug == plan_slug, SubscriptionPlan.is_active.is_(True))
            .one_or_none()
        )

        if plan is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"detail": "Selected plan is unavailable."},
            )

        if not plan.stripe_price_id:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Plan is missing Stripe price configuration."},
            )

        try:
            customer = ensure_customer(
                email=user.email,
                existing_customer_id=user.stripe_customer_id,
            )
        except Exception as exc:  # noqa: BLE001 - want full context in logs
            logger.exception("Failed to ensure Stripe customer for user %s", user.id)
            return JSONResponse(
                status_code=status.HTTP_502_BAD_GATEWAY,
                content={"detail": "Unable to contact billing provider."},
            )

        if not user.stripe_customer_id:
            db_user = session.query(User).filter(User.id == user.id).one()
            db_user.stripe_customer_id = customer.id
            session.commit()
            user.stripe_customer_id = customer.id

        base_url = _build_base_url(request)
        success_url = f"{base_url}/settings?checkout=success"
        cancel_url = f"{base_url}/settings?checkout=cancel"

        try:
            session_obj = create_subscription_checkout_session(
                price_id=plan.stripe_price_id,
                success_url=success_url,
                cancel_url=cancel_url,
                client_reference_id=str(user.id),
                customer=customer.id,
                metadata={
                    "plan_id": str(plan.id),
                    "plan_slug": plan.slug,
                    "user_id": str(user.id),
                },
            )
        except Exception as exc:  # noqa: BLE001 - propagate message via logs
            logger.exception(
                "Failed to create Stripe checkout session for user %s and plan %s",
                user.id,
                plan.slug,
            )
            return JSONResponse(
                status_code=status.HTTP_502_BAD_GATEWAY,
                content={"detail": "Unable to initiate checkout session."},
            )

    return JSONResponse(content={"checkout_url": session_obj.url, "session_id": session_obj.id})


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, user: User = Depends(get_current_user_sync)):
    try:
        publishable_key = get_publishable_key()
    except RuntimeError:
        publishable_key = None

    context = {
        "request": request,
        "user": user,
        "stripe_publishable_key": publishable_key,
    }
    return templates.TemplateResponse("settings.html", context)


@app.post("/api/billing/stripe/webhook")
async def stripe_webhook(request: Request):
    if stripe_webhook_config is None:
        raise HTTPException(status_code=503, detail="Stripe webhooks are not configured.")

    payload = await request.body()
    signature = request.headers.get("stripe-signature")
    if not signature:
        raise HTTPException(status_code=400, detail="Missing Stripe signature header.")

    event = parse_event(payload, signature, stripe_webhook_config.signing_secret)
    event_type = event.get("type")

    logger.info("Stripe webhook received: %s", event_type)

    with SessionLocal() as session:
        if event_type in {"customer.subscription.created", "customer.subscription.updated"}:
            handle_subscription_updated(session, event)
        elif event_type == "customer.subscription.deleted":
            handle_subscription_deleted(session, event)
        elif event_type == "invoice.payment_succeeded":
            handle_invoice_paid(session, event)
        elif event_type == "checkout.session.completed":
            handle_checkout_session_completed(session, event)
        else:
            logger.debug("Unhandled Stripe webhook event type: %s", event_type)

    return JSONResponse(content={"received": True})

@app.post("/api/analyze")
async def analyze_trading_data(
    request: Request,
    user: User = Depends(get_current_user_sync)
):
    """Analyze trading data using AI"""
    try:
        runs_remaining_after = _consume_subscription_units(
            user.id,
            usage_type="direct_ai_analysis",
            notes="Direct AI analysis request",
        )
        # Get the request data
        data = await request.json()
        symbol = data.get('symbol', 'UNKNOWN')
        trading_data = data.get('trading_data', {})
        
        print(f"AI analysis requested by {user.email} for {symbol}")
        print(f"Trading data keys: {list(trading_data.keys())}")
        
        # Enhance trading data with metadata
        enhanced_trading_data = {
            **trading_data,
            "analysis_requested_by": user.email,
            "analysis_timestamp": str(datetime.now()),
            "symbol": symbol
        }
        
        # Use the centralized OpenAI service
        analysis_result = openai_service.analyze_trading_data(enhanced_trading_data, symbol)
        
        if analysis_result["success"]:
            print(f"AI analysis completed for {symbol}, tokens used: {analysis_result.get('tokens_used', 0)}")
            
            # Add metadata to response
            analysis_result.update({
                "analyzed_by": user.email,
                "analysis_timestamp": str(datetime.now()),
                "symbol": symbol
            })
            analysis_result["runs_remaining"] = runs_remaining_after
            
            return JSONResponse(
                content=analysis_result,
                status_code=200
            )
        else:
            print(f"AI analysis failed for {symbol}: {analysis_result.get('error', 'Unknown error')}")
            analysis_result["runs_remaining"] = runs_remaining_after
            return JSONResponse(
                content=analysis_result,
                status_code=400
            )
            
    except HTTPException as exc:
        error_payload: Dict[str, Any] = {"success": False}
        detail = exc.detail
        if isinstance(detail, dict):
            error_payload.update(detail)
            if "error" not in error_payload and "message" in error_payload:
                error_payload["error"] = error_payload["message"]
        elif isinstance(detail, str):
            error_payload["error"] = detail
        else:
            error_payload["error"] = "Subscription validation failed."

        if "runs_remaining" not in error_payload:
            error_payload["runs_remaining"] = None

        response = JSONResponse(
            content=error_payload,
            status_code=exc.status_code,
        )
        if exc.headers:
            for key, value in exc.headers.items():
                response.headers[key] = value
        return response
    except Exception as e:
        print(f"Error in AI analysis: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": f"Analysis failed: {str(e)}",
                "timestamp": str(datetime.now()),
                "runs_remaining": runs_remaining_after if 'runs_remaining_after' in locals() else None,
            },
            status_code=500
        )
    
# Add a health check endpoint for OpenAI
@app.get("/api/test-openai")
async def test_openai_connection(user: User = Depends(get_current_user_sync)):
    """Test the centralized OpenAI API connection"""
    try:
        result = openai_service.test_connection()
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            content={
                "success": False, 
                "error": f"OpenAI service initialization error: {str(e)}",
                "error_type": "service_error"
            },
            status_code=500
        )
# Add OpenAI status endpoint for more detailed info
@app.get("/api/openai/status")
async def openai_detailed_status(user: User = Depends(get_current_user_sync)):
    """Get detailed OpenAI service status"""
    try:
        # Test connection
        connection_result = openai_service.test_connection()
        
        # Get API key status (masked for security)
        api_key = os.getenv("OPENAI_API_KEY", "")
        api_key_status = {
            "configured": bool(api_key),
            "key_preview": f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "Invalid key format",
            "key_length": len(api_key)
        }
        
        return JSONResponse(content={
            "connection_test": connection_result,
            "api_key_status": api_key_status,
            "service_version": "OpenAI v1.x",
            "timestamp": str(datetime.now())
        })
        
    except Exception as e:
        return JSONResponse(
            content={
                "success": False,
                "error": f"Status check failed: {str(e)}",
                "timestamp": str(datetime.now())
            },
            status_code=500
        )
# if __name__ == "__main__":
#     import uvicorn

#     # Run the FastAPI app with Uvicorn
#     uvicorn.run(
#         "app:app",  # "app" is the filename, and "app" is the FastAPI instance
#         host="127.0.0.1",  # Localhost
#         port=8000,         # Port number
#         reload=True        # Enable auto-reload for development
#     )