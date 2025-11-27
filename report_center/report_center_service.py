"""Domain helpers for the Report Center router."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from core.datetime_utils import parse_iso_datetime
from core.numeric_utils import safe_float
from utils.symbols import canonicalize_symbol, sanitize_symbol


def resolve_report_center_date(raw_date: Optional[str]) -> Tuple[datetime, str, str]:
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


def summarize_report_center_entry(report: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a stored AI report into a template-friendly summary."""

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
    symbol_sanitized = sanitize_symbol(symbol_clean)
    symbol_canonical = canonicalize_symbol(symbol_clean)

    generated_dt = parse_iso_datetime(
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
        "symbol_display": symbol_clean,
        "symbol_sanitized": symbol_sanitized or None,
        "symbol_canonical": symbol_canonical or None,
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
                    key=lambda item: abs(safe_float(item.get("distance_pct")) or 0.0),
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


def _clean_text_fragment(value: Any, *, max_items: Optional[int] = None) -> str:
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
    if value is None:
        return None

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value

    if math.isfinite(numeric):
        return round(numeric, places)

    return value
