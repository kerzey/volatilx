"""FastAPI router for Report Center endpoints."""

from __future__ import annotations

import logging
import urllib.parse
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse

from azure_storage import fetch_reports_for_date
from db import SessionLocal
from report_center_backend.report_center_service import resolve_report_center_date, summarize_report_center_entry
from services.favorites import favorite_symbols
from utils.symbols import canonicalize_symbol
from user import User, get_current_user_sync

from app import (  # pylint: disable=cyclic-import
    DEFAULT_REPORT_CENTER_REPORT_LIMIT,
    _check_report_center_access,
    _serialize_subscription,
    templates,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/report-center", response_class=HTMLResponse)
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

    target_date, selected_date_iso, selected_date_label = resolve_report_center_date(date)
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
        summary = summarize_report_center_entry(report)
        if summary:
            if not summary.get("symbol_display") and summary.get("symbol"):
                summary["symbol_display"] = summary["symbol"].upper()
            if not summary.get("symbol_canonical") and summary.get("symbol"):
                summary["symbol_canonical"] = canonicalize_symbol(summary["symbol"])
            prepared_reports.append(summary)
        else:
            excluded_reports.append(report)

    prepared_reports.sort(key=lambda item: item.get("generated_unix") or 0, reverse=True)

    with SessionLocal() as session:
        favorite_equity_symbols = favorite_symbols(session, user.id, market="equity")
        favorite_crypto_symbols = favorite_symbols(session, user.id, market="crypto")

    favorite_union = list(dict.fromkeys(favorite_equity_symbols + favorite_crypto_symbols))
    favorite_canonical: List[str] = []
    for symbol_key in favorite_union:
        canonical_value = canonicalize_symbol(symbol_key)
        if canonical_value:
            favorite_canonical.append(canonical_value)
    favorite_canonical = list(dict.fromkeys(favorite_canonical))

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
        "favorite_symbols": favorite_union,
        "favorite_symbols_canonical": favorite_canonical,
        "favorite_equity_symbols": favorite_equity_symbols,
        "favorite_crypto_symbols": favorite_crypto_symbols,
    }
    return templates.TemplateResponse("report_center.html", context)
