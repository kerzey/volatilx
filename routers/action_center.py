"""FastAPI router for Action Center endpoints."""

from __future__ import annotations

import json
import urllib.parse
from typing import Any, Dict, List, Optional, Set

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from user import User, get_current_user_sync
from models import UserFavoriteSymbol
from db import SessionLocal

from action_center import STRATEGY_KEY_BY_TIMEFRAME, normalize_intent, normalize_timeframe
from app import (
    DEFAULT_ACTION_CENTER_SYMBOL,
    _check_report_center_access,
    _derive_action_center_view,
    _ensure_crypto_stream_symbols,
    _ensure_equity_stream_symbols,
    _fetch_latest_action_report,
    _get_live_stream,
    _resolve_rest_prices,
    _serialize_subscription,
    templates,
)
from services.favorites import favorite_symbols, is_symbol_favorited
from utils.symbols import canonicalize_symbol, filter_symbols_for_market, sanitize_symbol

router = APIRouter()


@router.get("/action-center", response_class=HTMLResponse)
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

    timeframe_slug = normalize_timeframe(timeframe)
    intent_slug = normalize_intent(intent)

    raw_symbol = (symbol or "").strip()
    requested_symbol = raw_symbol or DEFAULT_ACTION_CENTER_SYMBOL
    requested_display_symbol = requested_symbol.upper() or DEFAULT_ACTION_CENTER_SYMBOL

    primary_symbol_sanitized = (
        sanitize_symbol(requested_symbol)
        or sanitize_symbol(DEFAULT_ACTION_CENTER_SYMBOL)
        or DEFAULT_ACTION_CENTER_SYMBOL
    )
    primary_symbol_canonical = (
        canonicalize_symbol(requested_symbol)
        or canonicalize_symbol(DEFAULT_ACTION_CENTER_SYMBOL)
        or DEFAULT_ACTION_CENTER_SYMBOL
    )

    symbol_display_map: Dict[str, str] = {}

    live_stream = _get_live_stream(request.app)
    live_price_available = bool(live_stream)

    with SessionLocal() as session:
        favorite_equity_symbols = favorite_symbols(session, user.id, market="equity")
        favorite_crypto_symbols = favorite_symbols(session, user.id, market="crypto")

    favorite_canonical_set: Set[str] = set()
    favorite_union: List[str] = []

    for favorite_symbol in list(favorite_equity_symbols) + list(favorite_crypto_symbols):
        favorite_clean = sanitize_symbol(favorite_symbol)
        if not favorite_clean:
            continue
        favorite_canonical = canonicalize_symbol(favorite_clean)
        if not favorite_canonical or favorite_canonical in favorite_canonical_set:
            continue
        favorite_canonical_set.add(favorite_canonical)
        favorite_union.append(favorite_clean)
        symbol_display_map.setdefault(favorite_clean, favorite_clean)

    matching_favorite = next(
        (fav for fav in favorite_union if canonicalize_symbol(fav) == primary_symbol_canonical),
        None,
    )

    if matching_favorite:
        primary_symbol_sanitized = matching_favorite
    elif favorite_union:
        primary_symbol_sanitized = favorite_union[0]

    primary_is_favorite = primary_symbol_canonical in favorite_canonical_set
    symbol_display_map[primary_symbol_sanitized] = requested_display_symbol or primary_symbol_sanitized

    strategy_key = STRATEGY_KEY_BY_TIMEFRAME[timeframe_slug]
    raw_report = _fetch_latest_action_report(primary_symbol_sanitized)

    dashboards: List[Dict[str, Any]] = []
    price_overrides: Dict[str, Dict[str, Any]] = {}

    def _build_dashboard(symbol_key: str, report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        resolved_report = report or _fetch_latest_action_report(symbol_key)
        symbol_display = symbol_display_map.get(symbol_key, symbol_key)
        symbol_canonical = canonicalize_symbol(symbol_key)
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
            dashboard_payload["favorite"] = bool(symbol_canonical and symbol_canonical in favorite_canonical_set)
            principal_data = (resolved_report.get("principal_plan") or {}).get("data") or {}
            strategies = principal_data.get("strategies") or {}
            active_strategy = strategies.get(strategy_key) or {}
            dashboard_summary = active_strategy.get("summary")
        else:
            error_message = "No recent AI analysis found for this symbol. Try running a new multi-agent analysis first."

        return {
            "symbol": symbol_key,
            "symbol_display": symbol_display,
            "symbol_canonical": symbol_canonical,
            "is_favorited": bool(symbol_canonical and symbol_canonical in favorite_canonical_set),
            "strategy_summary": dashboard_summary,
            "action_data": dashboard_payload,
            "error_message": error_message,
            "report_blob": resolved_report.get("_blob_name") if resolved_report else None,
            "timeframe": timeframe_slug,
            "intent": intent_slug,
        }

    if favorite_union:
        symbol_order: List[str] = [primary_symbol_sanitized] + [sym for sym in favorite_union if sym != primary_symbol_sanitized]
    else:
        symbol_order = [primary_symbol_sanitized]

    price_symbols: Set[str] = set(symbol_order)
    price_overrides = await _resolve_rest_prices(price_symbols)

    for sym in symbol_order:
        symbol_display_map.setdefault(sym, sym)
        if sym == primary_symbol_sanitized:
            dashboards.append(_build_dashboard(sym, raw_report))
        else:
            dashboards.append(_build_dashboard(sym))

    equity_symbol_set = {
        cleaned
        for fav in favorite_equity_symbols
        for cleaned in [sanitize_symbol(fav)]
        if cleaned
    }
    equity_stream_symbols = [sym for sym in symbol_order if sym in equity_symbol_set] if equity_symbol_set else [primary_symbol_sanitized]
    if equity_stream_symbols:
        await _ensure_equity_stream_symbols(request.app, equity_stream_symbols)

    primary_dashboard = dashboards[0] if dashboards else None

    dashboards_json = json.dumps(
        [
            {
                "symbol": dash["symbol"],
                "symbol_display": dash.get("symbol_display", dash["symbol"]),
                "symbol_canonical": dash.get("symbol_canonical"),
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
        "symbol": primary_dashboard["symbol"] if primary_dashboard else primary_symbol_sanitized,
        "symbol_display": primary_dashboard.get("symbol_display") if primary_dashboard else requested_display_symbol,
        "symbol_canonical": primary_dashboard.get("symbol_canonical") if primary_dashboard else primary_symbol_canonical,
        "requested_symbol": requested_symbol,
        "requested_symbol_display": requested_display_symbol,
        "subscription": _serialize_subscription(subscription) if subscription else None,
        "timeframe": timeframe_slug,
        "intent": intent_slug,
        "dashboards": dashboards,
        "dashboards_json": dashboards_json,
        "primary_dashboard": primary_dashboard,
        "latest_report_blob": primary_dashboard["report_blob"] if primary_dashboard else None,
        "is_favorited": primary_dashboard["is_favorited"] if primary_dashboard else primary_is_favorite,
        "favorite_symbols": favorite_union,
        "favorite_symbols_canonical": sorted(favorite_canonical_set),
        "favorite_crypto_symbols": favorite_crypto_symbols,
        "live_price_enabled": live_price_available,
        "rest_price_overrides": price_overrides,
    }

    return templates.TemplateResponse("action_center.html", context)


@router.get("/api/action-center")
async def action_center_api(
    symbol: str,
    timeframe: str = "day",
    intent: str = "buy",
    user: User = Depends(get_current_user_sync),
):
    subscription, allowed = _check_report_center_access(user)
    if not allowed:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Subscription upgrade required for Action Center")
    raw_symbol = (symbol or "").strip()
    requested_symbol = raw_symbol or DEFAULT_ACTION_CENTER_SYMBOL
    sanitized_symbol = (
        sanitize_symbol(requested_symbol)
        or sanitize_symbol(DEFAULT_ACTION_CENTER_SYMBOL)
        or DEFAULT_ACTION_CENTER_SYMBOL
    )
    canonical_symbol = (
        canonicalize_symbol(requested_symbol)
        or canonicalize_symbol(DEFAULT_ACTION_CENTER_SYMBOL)
        or DEFAULT_ACTION_CENTER_SYMBOL
    )
    display_symbol = requested_symbol.upper() or sanitized_symbol
    timeframe_slug = normalize_timeframe(timeframe)
    intent_slug = normalize_intent(intent)
    report = _fetch_latest_action_report(sanitized_symbol)
    if not report:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": "No recent analysis available for the requested symbol."},
        )

    strategy_key = STRATEGY_KEY_BY_TIMEFRAME[timeframe_slug]
    price_overrides = await _resolve_rest_prices({sanitized_symbol})
    payload = _derive_action_center_view(
        report,
        strategy_key=strategy_key,
        intent=intent_slug,
        price_override=price_overrides.get(sanitized_symbol),
    )
    with SessionLocal() as session:
        is_favorited = is_symbol_favorited(session, user.id, sanitized_symbol)
    payload["favorite"] = is_favorited
    return JSONResponse(
        content={
            "symbol": sanitized_symbol,
            "symbol_display": display_symbol,
            "symbol_canonical": canonical_symbol,
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


@router.post("/api/action-center/favorites")
async def set_action_center_favorite(
    payload: FavoriteTogglePayload,
    request: Request,
    user: User = Depends(get_current_user_sync),
):
    symbol_clean = sanitize_symbol(payload.symbol)
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
        current_state = is_symbol_favorited(session, user.id, symbol_clean)
        favorites = favorite_symbols(session, user.id, market="equity")

    if current_state:
        symbol_set = {symbol_clean}
        equity_symbols = filter_symbols_for_market(symbol_set, "equity")
        if equity_symbols:
            await _ensure_equity_stream_symbols(request.app, equity_symbols)
        crypto_symbols = filter_symbols_for_market(symbol_set, "crypto")
        if crypto_symbols:
            await _ensure_crypto_stream_symbols(request.app, crypto_symbols)

    return {
        "symbol": symbol_clean,
        "symbol_canonical": canonicalize_symbol(symbol_clean),
        "follow": current_state,
        "favorites": favorites,
    }


@router.get("/api/action-center/favorites")
async def list_action_center_favorites(user: User = Depends(get_current_user_sync)):
    with SessionLocal() as session:
        favorites = favorite_symbols(session, user.id, limit=50, market="equity")
    return {"symbols": favorites}
