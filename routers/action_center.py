"""FastAPI router for Action Center endpoints."""

from __future__ import annotations

import json
import urllib.parse
from typing import Any, Dict, List, Mapping, Optional, Set

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from user import User, get_current_user_sync
from models import UserFavoriteSymbol
from db import SessionLocal

from action_center import STRATEGY_KEY_BY_TIMEFRAME, normalize_intent, normalize_timeframe
from core.datetime_utils import parse_iso_datetime
from core.numeric_utils import safe_float
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


def _coerce_trade_setup(setup: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(setup, Mapping):
        return None

    entry_val = safe_float(setup.get("entry"))
    stop_val = safe_float(setup.get("stop"))

    targets: List[float] = []
    raw_targets = setup.get("targets")
    if isinstance(raw_targets, (list, tuple, set)):
        for target in raw_targets:
            target_val = safe_float(target)
            if target_val is not None:
                targets.append(target_val)

    if entry_val is None and stop_val is None and not targets:
        return None

    return {
        "entry": entry_val if entry_val is not None else 0.0,
        "stop": stop_val if stop_val is not None else 0.0,
        "targets": targets,
    }


def _coerce_no_trade_zones(zones: Any) -> List[Dict[str, float]]:
    prepared: List[Dict[str, float]] = []
    if not isinstance(zones, (list, tuple, set)):
        return prepared

    for zone in zones:
        if not isinstance(zone, Mapping):
            continue
        low_raw = zone.get("min") if zone.get("min") is not None else zone.get("low")
        high_raw = zone.get("max") if zone.get("max") is not None else zone.get("high")
        low_val = safe_float(low_raw)
        high_val = safe_float(high_raw)
        if low_val is None or high_val is None:
            continue
        if low_val > high_val:
            low_val, high_val = high_val, low_val
        prepared.append({"min": low_val, "max": high_val})

    return prepared


def _resolve_generated_display(plan_data: Mapping[str, Any], report: Mapping[str, Any]) -> Optional[str]:
    label = plan_data.get("generated_display")
    if isinstance(label, str) and label.strip():
        return label.strip()

    timestamp = (
        plan_data.get("generated")
        or plan_data.get("generated_at")
        or report.get("stored_at")
        or report.get("_blob_last_modified")
    )

    if not isinstance(timestamp, str):
        return None

    parsed = parse_iso_datetime(timestamp)
    if not parsed:
        return None

    return parsed.strftime("%b %d, %Y %H:%M UTC")


def _normalise_confidence_label(raw_confidence: Optional[str]) -> str:
    if not raw_confidence:
        return "MEDIUM"
    normalized = str(raw_confidence).strip().upper()
    mapping = {
        "LOW": "LOW",
        "MEDIUM": "MEDIUM",
        "MID": "MEDIUM",
        "MODERATE": "MEDIUM",
        "HIGH": "HIGH",
        "STRONG": "HIGH",
    }
    return mapping.get(normalized, "MEDIUM")


def _normalise_recommendation(raw_recommendation: Optional[str]) -> str:
    if not raw_recommendation:
        return "HOLD"
    normalized = str(raw_recommendation).strip().upper()
    mapping = {
        "BUY": "BUY",
        "STRONG BUY": "BUY",
        "SELL": "SELL",
        "STRONG SELL": "SELL",
        "HOLD": "HOLD",
        "NEUTRAL": "HOLD",
    }
    return mapping.get(normalized, "HOLD")


def _extract_technical_consensus(report: Mapping[str, Any], symbol: str) -> Optional[Dict[str, Any]]:
    snapshot_map = report.get("technical_snapshot")
    if not isinstance(snapshot_map, Mapping):
        return None

    snapshot = None
    for candidate in (symbol, symbol.upper(), symbol.lower()):
        candidate_snapshot = snapshot_map.get(candidate)
        if isinstance(candidate_snapshot, Mapping):
            snapshot = candidate_snapshot
            break

    if not isinstance(snapshot, Mapping):
        return None

    consensus = snapshot.get("consensus")
    if not isinstance(consensus, Mapping):
        return None

    recommendation = _normalise_recommendation(consensus.get("overall_recommendation"))
    confidence = _normalise_confidence_label(consensus.get("confidence"))
    strength = safe_float(consensus.get("strength"))

    payload: Dict[str, Any] = {
        "overall_recommendation": recommendation,
        "confidence": confidence,
    }
    if strength is not None:
        payload["strength"] = strength

    return payload


def _coerce_bias(bias: Any) -> Optional[Dict[str, float]]:
    if not isinstance(bias, Mapping):
        return None

    low_val = safe_float(bias.get("low"))
    high_val = safe_float(bias.get("high"))
    invalid_val = safe_float(bias.get("invalid"))

    if None in (low_val, high_val, invalid_val):
        return None

    return {
        "low": low_val,
        "high": high_val,
        "invalid": invalid_val,
    }


def _prepare_strategy(strategy: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(strategy, Mapping):
        return None

    summary = str(strategy.get("summary") or "").strip()
    buy_setup = _coerce_trade_setup(strategy.get("buy_setup"))
    sell_setup = _coerce_trade_setup(strategy.get("sell_setup"))
    no_trade_zone = _coerce_no_trade_zones(strategy.get("no_trade_zone"))

    if buy_setup is None:
        buy_setup = {"entry": 0.0, "stop": 0.0, "targets": []}
    if sell_setup is None:
        sell_setup = {"entry": 0.0, "stop": 0.0, "targets": []}

    payload: Dict[str, Any] = {
        "summary": summary,
        "buy_setup": buy_setup,
        "sell_setup": sell_setup,
        "no_trade_zone": no_trade_zone,
    }

    bias_payload = _coerce_bias(strategy.get("bias"))
    if bias_payload:
        payload["bias"] = bias_payload

    reward_risk = safe_float(
        strategy.get("rewardRisk")
        or strategy.get("reward_risk")
        or strategy.get("reward_to_risk")
        or strategy.get("reward_to_risk_ratio")
        or strategy.get("risk_reward_ratio")
    )
    if reward_risk is not None:
        payload["rewardRisk"] = reward_risk

    conviction = safe_float(
        strategy.get("conviction")
        or strategy.get("confidence_score")
        or strategy.get("confidence")
        or strategy.get("strength")
    )
    if conviction is not None:
        payload["conviction"] = conviction

    return payload


def _prepare_principal_plan(
    report: Mapping[str, Any],
    *,
    price_override: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    plan_wrapper = report.get("principal_plan")
    if not isinstance(plan_wrapper, Mapping):
        return None

    plan_data = plan_wrapper.get("data")
    if not isinstance(plan_data, Mapping):
        return None

    symbol_value = plan_data.get("symbol") or report.get("symbol")
    if not symbol_value:
        return None

    symbol_clean = str(symbol_value).strip().upper()

    generated_display = _resolve_generated_display(plan_data, report) or ""

    override_price = safe_float((price_override or {}).get("price"))
    plan_price = safe_float(plan_data.get("latest_price"))
    report_price = safe_float(report.get("latest_price"))

    latest_price = override_price
    if latest_price is None:
        latest_price = plan_price if plan_price is not None else report_price
    if latest_price is None:
        latest_price = 0.0

    strategies_raw = plan_data.get("strategies")
    if not isinstance(strategies_raw, Mapping):
        return None

    prepared_strategies: Dict[str, Dict[str, Any]] = {}
    for key in ("day_trading", "swing_trading", "longterm_trading"):
        prepared = _prepare_strategy(strategies_raw.get(key))
        if prepared:
            prepared_strategies[key] = prepared

    if not prepared_strategies:
        return None

    consensus_payload = _extract_technical_consensus(report, symbol_clean)

    plan_payload: Dict[str, Any] = {
        "symbol": symbol_clean,
        "generated_display": generated_display,
        "latest_price": latest_price,
        "strategies": prepared_strategies,
    }

    if consensus_payload:
        plan_payload["technical_consensus"] = consensus_payload

    return plan_payload


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
        (
            fav
            for fav in favorite_union
            if canonicalize_symbol(fav) == primary_symbol_canonical
        ),
        None,
    )

    if matching_favorite:
        primary_symbol_sanitized = matching_favorite
    elif not raw_symbol and favorite_union:
        primary_symbol_sanitized = favorite_union[0]

    primary_symbol_canonical = canonicalize_symbol(primary_symbol_sanitized)

    primary_is_favorite = primary_symbol_canonical in favorite_canonical_set
    symbol_display_map[primary_symbol_sanitized] = requested_display_symbol or primary_symbol_sanitized

    strategy_key = STRATEGY_KEY_BY_TIMEFRAME[timeframe_slug]
    dashboards: List[Dict[str, Any]] = []
    price_overrides: Dict[str, Dict[str, Any]] = {}
    principal_plan_map: Dict[str, Dict[str, Any]] = {}
    principal_plan_order: List[str] = []

    def _build_dashboard(symbol_key: str, report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        nonlocal principal_plan_map, principal_plan_order
        resolved_report = report or _fetch_latest_action_report(symbol_key)
        symbol_display = symbol_display_map.get(symbol_key, symbol_key)
        symbol_canonical = canonicalize_symbol(symbol_key)
        dashboard_payload: Optional[Dict[str, Any]] = None
        dashboard_summary: Optional[str] = None
        error_message: Optional[str] = None
        plan_payload: Optional[Dict[str, Any]] = None

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
            plan_payload = _prepare_principal_plan(
                resolved_report,
                price_override=price_override,
            )
            if plan_payload:
                plan_payload["symbol_display"] = symbol_display
                plan_payload["is_favorited"] = bool(
                    symbol_canonical and symbol_canonical in favorite_canonical_set
                )
                if symbol_key not in principal_plan_map:
                    principal_plan_map[symbol_key] = plan_payload
                    principal_plan_order.append(symbol_key)
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

    candidate_symbols: List[str] = []
    candidate_symbols.append(primary_symbol_sanitized)
    for fav_symbol in favorite_union:
        if fav_symbol not in candidate_symbols:
            candidate_symbols.append(fav_symbol)

    raw_report = _fetch_latest_action_report(primary_symbol_sanitized)

    if not raw_symbol:
        has_principal_plan = bool(
            raw_report
            and _prepare_principal_plan(raw_report, price_override=None)
        )
        if not has_principal_plan:
            for candidate_symbol in candidate_symbols:
                if candidate_symbol == primary_symbol_sanitized:
                    continue
                candidate_report = _fetch_latest_action_report(candidate_symbol)
                if not candidate_report:
                    continue
                candidate_plan = _prepare_principal_plan(candidate_report, price_override=None)
                if candidate_plan:
                    primary_symbol_sanitized = candidate_symbol
                    primary_symbol_canonical = canonicalize_symbol(candidate_symbol) or primary_symbol_canonical
                    raw_report = candidate_report
                    symbol_display_map.setdefault(primary_symbol_sanitized, candidate_symbol)
                    break

    if favorite_union:
        symbol_order: List[str] = [primary_symbol_sanitized] + [sym for sym in candidate_symbols if sym != primary_symbol_sanitized]
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

    principal_plan_payload = principal_plan_map.get(primary_symbol_sanitized)
    principal_plan_json = (
        json.dumps(principal_plan_payload, separators=(",", ":"))
        if principal_plan_payload is not None
        else None
    )
    principal_plan_map_json = (
        json.dumps(principal_plan_map, separators=(",", ":"))
        if principal_plan_map
        else None
    )
    principal_plan_order_json = (
        json.dumps(principal_plan_order, separators=(",", ":"))
        if principal_plan_order
        else None
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
        "primary_dashboard": primary_dashboard,
        "latest_report_blob": primary_dashboard["report_blob"] if primary_dashboard else None,
        "is_favorited": primary_dashboard["is_favorited"] if primary_dashboard else primary_is_favorite,
        "favorite_symbols": favorite_union,
        "favorite_symbols_canonical": sorted(favorite_canonical_set),
        "favorite_crypto_symbols": favorite_crypto_symbols,
        "live_price_enabled": live_price_available,
        "rest_price_overrides": price_overrides,
        "principal_plan": principal_plan_payload,
        "principal_plan_json": principal_plan_json,
        "principal_plan_map_json": principal_plan_map_json,
        "principal_plan_order_json": principal_plan_order_json,
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

    principal_plan_payload = _prepare_principal_plan(
        report,
        price_override=price_overrides.get(sanitized_symbol),
    )
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
            "principal_plan": principal_plan_payload,
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
