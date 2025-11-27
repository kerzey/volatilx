"""Business logic for Action Center guidance generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from core.numeric_utils import safe_float

__all__ = [
    "STRATEGY_KEY_BY_TIMEFRAME",
    "ActionCenterSettings",
    "generate_action_json",
    "normalize_intent",
    "normalize_timeframe",
]


STRATEGY_KEY_BY_TIMEFRAME: Dict[str, str] = {
    "day": "day_trading",
    "swing": "swing_trading",
    "long": "longterm_trading",
}


@dataclass(frozen=True)
class ActionCenterSettings:
    proximity_tolerance_pct: float
    level_break_partial: str
    second_level_partial: str
    second_level_approach_min: float
    second_level_approach_max: float
    second_level_near_progress: float
    second_level_near_level_pct: float


_TIMEFRAME_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "day": ("15m", "m15", "15-min", "15 minute", "15min", "quarter"),
    "swing": ("1h", "60m", "h1", "hour", "60min"),
    "long": ("1d", "d1", "day", "daily", "24h"),
}

_TIMEFRAME_LABELS: Dict[str, str] = {
    "day": "15m",
    "swing": "1h",
    "long": "1D",
}


def normalize_timeframe(value: str) -> str:
    slug = (value or "day").strip().lower()
    return slug if slug in STRATEGY_KEY_BY_TIMEFRAME else "day"


def normalize_intent(value: str) -> str:
    slug = (value or "buy").strip().lower()
    return slug if slug in {"buy", "sell"} else "buy"


def _extract_timeframe_close(
    price_context: Mapping[str, Any],
    timeframe_slug: str,
) -> Tuple[Optional[float], Optional[str]]:
    if not isinstance(price_context, Mapping):
        return (None, None)

    tokens = _TIMEFRAME_KEYWORDS.get(timeframe_slug, ())
    if not tokens:
        return (None, None)

    sources: List[Tuple[str, Mapping[str, Any]]] = []
    close_map = price_context.get("close_by_timeframe")
    if isinstance(close_map, Mapping):
        sources.append(("close_by_timeframe", close_map))
    price_map = price_context.get("price_by_timeframe")
    if isinstance(price_map, Mapping):
        sources.append(("price_by_timeframe", price_map))

    for source_name, mapping in sources:
        for key, value in mapping.items():
            key_norm = str(key).lower()
            if any(token in key_norm for token in tokens):
                candidate = safe_float(value)
                if candidate is not None:
                    return candidate, f"{source_name}:{key}"

    return (None, None)


def validate_level_break(
    settings: ActionCenterSettings,
    timeframe_slug: str,
    price_context: Mapping[str, Any],
    s1_price: Optional[float],
    r1_price: Optional[float],
    s2_price: Optional[float],
    r2_price: Optional[float],
) -> Dict[str, Any]:
    """Detect closes that invalidate proximity logic by breaching primary levels."""

    close_price, close_source = _extract_timeframe_close(price_context, timeframe_slug)
    latest_price = safe_float(price_context.get("latest_price"))
    debug = {
        "timeframe_slug": timeframe_slug,
        "close_price": close_price,
        "close_source": close_source,
        "latest_price": latest_price,
        "s1_price": s1_price,
        "r1_price": r1_price,
    }

    result = {
        "triggered": False,
        "path": None,
        "primary_level": None,
        "secondary_target": None,
        "message": "",
        "reason": "No level break detected.",
        "partial_action": "none",
        "needs_reanalysis": False,
        "debug": debug,
    }

    if close_price is None:
        debug["note"] = "Missing timeframe close data."
        return result

    timeframe_label = _TIMEFRAME_LABELS.get(timeframe_slug, timeframe_slug.upper())
    epsilon = 1e-6

    reclaim_tolerance = settings.proximity_tolerance_pct / 100.0

    if s1_price is not None and close_price < s1_price * (1 - epsilon):
        if latest_price is not None and latest_price >= s1_price * (1 - reclaim_tolerance):
            debug["reclaimed"] = {
                "latest_price": latest_price,
                "threshold": s1_price * (1 - reclaim_tolerance),
            }
            result["reason"] = "Primary support reclaimed after intraday breach."
            return result
        breach_pct = ((s1_price - close_price) / s1_price) * 100 if s1_price else None
        debug["breach_pct"] = breach_pct
        result.update(
            {
                "triggered": True,
                "path": "support_break",
                "primary_level": "S1",
                "secondary_target": "S2" if s2_price is not None else None,
                "message": "Support break - price likely moving toward Support 2",
                "reason": (
                    f"{timeframe_label} close {close_price:,.2f} printed below Support 1 {s1_price:,.2f}"
                    + (f" ({breach_pct:.2f}% breach)" if breach_pct is not None else "")
                ),
                "partial_action": settings.level_break_partial,
                "needs_reanalysis": True,
            }
        )
        return result

    if r1_price is not None and close_price > r1_price * (1 + epsilon):
        if latest_price is not None and latest_price <= r1_price * (1 + reclaim_tolerance):
            debug["rejected"] = {
                "latest_price": latest_price,
                "threshold": r1_price * (1 + reclaim_tolerance),
            }
            result["reason"] = "Primary resistance rejected post breakout attempt."
            return result
        breach_pct = ((close_price - r1_price) / r1_price) * 100 if r1_price else None
        debug["breach_pct"] = breach_pct
        result.update(
            {
                "triggered": True,
                "path": "resistance_break",
                "primary_level": "R1",
                "secondary_target": "R2" if r2_price is not None else None,
                "message": "Resistance breakout - price likely moving toward Resistance 2",
                "reason": (
                    f"{timeframe_label} close {close_price:,.2f} cleared Resistance 1 {r1_price:,.2f}"
                    + (f" ({breach_pct:.2f}% breakout)" if breach_pct is not None else "")
                ),
                "partial_action": settings.level_break_partial,
                "needs_reanalysis": True,
            }
        )
        return result

    debug["note"] = "Close remained inside primary band."
    return result


def compute_zone_proximity(
    latest_price: Optional[float],
    s1_price: Optional[float],
    r1_price: Optional[float],
    tolerance_pct: float,
) -> Dict[str, Any]:
    """Baseline proximity logic to primary support/resistance."""

    debug = {
        "latest_price": latest_price,
        "s1_price": s1_price,
        "r1_price": r1_price,
        "tolerance_pct": tolerance_pct,
    }

    base_response = {
        "raw_zone": "wait",
        "primary_level": "None",
        "secondary_target": None,
        "reason": "Primary price reference unavailable.",
        "message": "Waiting for price to approach a key level.",
        "path": None,
        "debug": debug,
    }

    if latest_price is None:
        return base_response

    candidates: List[Tuple[str, float, str]] = []
    if s1_price:
        diff_pct = abs(latest_price - s1_price) / s1_price * 100
        candidates.append(("buy", diff_pct, "S1"))
    if r1_price:
        diff_pct = abs(r1_price - latest_price) / r1_price * 100
        candidates.append(("sell", diff_pct, "R1"))

    debug["distances"] = [
        {"zone": zone, "level": level, "diff_pct": diff}
        for zone, diff, level in candidates
    ]

    if not candidates:
        base_response["reason"] = "No primary levels available for proximity computation."
        base_response["message"] = "Key support or resistance absent - monitor manually."
        return base_response

    candidates.sort(key=lambda item: item[1])
    best_zone, best_diff, best_level = candidates[0]
    debug["closest_level"] = {"zone": best_zone, "diff_pct": best_diff, "level": best_level}

    secondary_target = "S2" if best_level == "S1" else "R2"
    response = {
        "raw_zone": best_zone if best_diff <= tolerance_pct else "wait",
        "primary_level": best_level if best_diff <= tolerance_pct else "None",
        "secondary_target": secondary_target if best_diff <= tolerance_pct else None,
        "reason": (
            f"Price is {best_diff:.2f}% from {best_level}."
            if best_diff <= tolerance_pct
            else "Price is holding between primary support and resistance."
        ),
        "message": (
            "Primary support engaged - continue only with confirmation."
            if best_zone == "buy" and best_diff <= tolerance_pct
            else "Primary resistance engaged - consider defensive positioning."
            if best_zone == "sell" and best_diff <= tolerance_pct
            else "Price remains between primary levels - patience required."
        ),
        "path": "support_proximity" if best_zone == "buy" and best_diff <= tolerance_pct else "resistance_proximity" if best_zone == "sell" and best_diff <= tolerance_pct else None,
        "debug": debug,
    }

    return response


def _label_support_levels(levels: Sequence[Mapping[str, Any]]) -> List[Tuple[str, float]]:
    prices: List[float] = []
    seen: set[float] = set()
    for level in levels:
        price_val = safe_float(level.get("price")) if isinstance(level, Mapping) else None
        if price_val is None or price_val in seen:
            continue
        seen.add(price_val)
        prices.append(price_val)
    prices.sort(reverse=True)
    return [(f"S{index + 1}", price) for index, price in enumerate(prices)]


def _label_resistance_levels(levels: Sequence[Mapping[str, Any]]) -> List[Tuple[str, float]]:
    prices: List[float] = []
    seen: set[float] = set()
    for level in levels:
        price_val = safe_float(level.get("price")) if isinstance(level, Mapping) else None
        if price_val is None or price_val in seen:
            continue
        seen.add(price_val)
        prices.append(price_val)
    prices.sort()
    return [(f"R{index + 1}", price) for index, price in enumerate(prices)]


def compute_second_level_progress(
    settings: ActionCenterSettings,
    latest_price: Optional[float],
    radar_levels: Mapping[str, Any],
    support_levels: Sequence[Mapping[str, Any]],
    resistance_levels: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Evaluate progress toward secondary levels and recommend escalation handling."""

    result = {
        "engaged": False,
        "path": None,
        "message": None,
        "reason": "Secondary levels not engaged.",
        "needs_reanalysis": False,
        "primary_level": None,
        "secondary_target": None,
        "partial_action": "none",
        "progress": None,
        "debug": {
            "latest_price": latest_price,
        },
    }

    if latest_price is None:
        result["debug"]["note"] = "No latest price available for secondary analysis."
        return result

    s_series = _label_support_levels(support_levels)
    r_series = _label_resistance_levels(resistance_levels)
    s_lookup = {label: price for label, price in s_series}
    r_lookup = {label: price for label, price in r_series}

    s1_price = safe_float((radar_levels.get("s1") or {}).get("price")) or s_lookup.get("S1")
    s2_price = safe_float((radar_levels.get("s2") or {}).get("price")) or s_lookup.get("S2")
    r1_price = safe_float((radar_levels.get("r1") or {}).get("price")) or r_lookup.get("R1")
    r2_price = safe_float((radar_levels.get("r2") or {}).get("price")) or r_lookup.get("R2")

    result["debug"]["level_snapshot"] = {
        "S1": s1_price,
        "S2": s2_price,
        "R1": r1_price,
        "R2": r2_price,
    }

    candidates: List[Dict[str, Any]] = []

    if s1_price is not None and s2_price is not None and s2_price < s1_price and latest_price <= s1_price:
        span = abs(s1_price - s2_price)
        if span > 0:
            progress = min(max((s1_price - latest_price) / span, 0.0), 2.0)
            candidates.append(
                {
                    "path": "support_path",
                    "progress": progress,
                    "base_label": "S1",
                    "target_label": "S2",
                    "base_price": s1_price,
                    "target_price": s2_price,
                    "next_label": "S3" if "S3" in s_lookup else "S2",
                    "next_price": s_lookup.get("S3") if "S3" in s_lookup else s2_price,
                }
            )

    if r1_price is not None and r2_price is not None and r2_price > r1_price and latest_price >= r1_price:
        span = abs(r2_price - r1_price)
        if span > 0:
            progress = min(max((latest_price - r1_price) / span, 0.0), 2.0)
            candidates.append(
                {
                    "path": "resistance_path",
                    "progress": progress,
                    "base_label": "R1",
                    "target_label": "R2",
                    "base_price": r1_price,
                    "target_price": r2_price,
                    "next_label": "R3" if "R3" in r_lookup else "R2",
                    "next_price": r_lookup.get("R3") if "R3" in r_lookup else r2_price,
                }
            )

    if not candidates:
        result["debug"]["note"] = "Price not advancing toward secondary levels."
        return result

    selected = max(candidates, key=lambda item: item["progress"])
    progress = selected["progress"]
    path = selected["path"]
    target_label = selected["target_label"]
    target_price = selected["target_price"]
    next_label = selected["next_label"]
    next_price = selected["next_price"]

    result["debug"]["selected"] = {
        "path": path,
        "progress": progress,
        "target_label": target_label,
        "target_price": target_price,
    }

    if progress < settings.second_level_approach_min:
        result["reason"] = f"Progress toward {target_label} at {progress * 100:.1f}% still below escalation threshold."
        return result

    result["engaged"] = True
    result["path"] = path
    result["progress"] = progress
    result["needs_reanalysis"] = True

    corridor = f"{selected['base_label']}->{target_label}"
    reason = f"Price is {progress * 100:.1f}% through the {corridor} corridor."

    gap_pct = None
    if target_price:
        gap_pct = abs(target_price - latest_price) / abs(target_price) * 100 if target_price else None
    result["debug"]["gap_pct_to_target"] = gap_pct

    if settings.second_level_approach_min <= progress <= settings.second_level_approach_max:
        message = "Approaching next level - run new analysis."
        partial = settings.level_break_partial
    elif progress >= settings.second_level_near_progress or (
        gap_pct is not None and gap_pct <= settings.second_level_near_level_pct
    ):
        message = (
            "Secondary support in play - run new analysis."
            if path == "support_path"
            else "Secondary resistance in play - run new analysis."
        )
        partial = settings.second_level_partial
        result["primary_level"] = target_label
        result["secondary_target"] = next_label if next_label != target_label else None
    else:
        message = "Secondary momentum building - monitor closely."
        partial = settings.level_break_partial

    result["message"] = message
    result["partial_action"] = partial
    result["reason"] = reason

    return result


def resolve_intent_conflict(
    intent: str,
    path_state: Optional[str],
    current_zone: str,
    current_message: str,
) -> Dict[str, Any]:
    """Override zone when the trader's intent conflicts with price trajectory."""

    normalized_intent = (intent or "buy").strip().lower()
    path_state = (path_state or "").strip().lower() or None

    conflict = {
        "validated_zone": current_zone,
        "message": current_message,
        "reason": "Intent aligned with market trajectory.",
        "needs_reanalysis": False,
        "partial_action": "none",
        "overridden": False,
        "debug": {
            "intent": normalized_intent,
            "path_state": path_state,
        },
    }

    if normalized_intent == "buy" and path_state in {"resistance_break", "resistance_path"}:
        conflict.update(
            {
                "validated_zone": "avoid",
                "message": "Chasing strength - wait for pullback or Resistance 2 analysis.",
                "reason": "Resistance break conflicts with buy intent.",
                "needs_reanalysis": True,
                "overridden": True,
            }
        )
    elif normalized_intent == "sell" and path_state in {"support_break", "support_path"}:
        conflict.update(
            {
                "validated_zone": "hold",
                "message": "Acceleration down - wait for Support 2 or new analysis.",
                "reason": "Support break conflicts with sell intent.",
                "needs_reanalysis": True,
                "overridden": True,
            }
        )

    conflict["debug"]["overridden"] = conflict["overridden"]
    return conflict


def generate_action_json(
    *,
    intent: str,
    strategy_key: str,
    latest_price: Optional[float],
    radar_levels: Mapping[str, Any],
    price_context: Mapping[str, Any],
    support_levels: Sequence[Mapping[str, Any]],
    resistance_levels: Sequence[Mapping[str, Any]],
    settings: ActionCenterSettings,
) -> Dict[str, Any]:
    """State machine that synthesises actionable guidance for the Action Center."""

    s1_price = safe_float((radar_levels.get("s1") or {}).get("price"))
    s2_price = safe_float((radar_levels.get("s2") or {}).get("price"))
    r1_price = safe_float((radar_levels.get("r1") or {}).get("price"))
    r2_price = safe_float((radar_levels.get("r2") or {}).get("price"))

    proximity = compute_zone_proximity(
        latest_price,
        s1_price,
        r1_price,
        settings.proximity_tolerance_pct,
    )

    decision = {
        "raw_zone": proximity.get("raw_zone") or "wait",
        "validated_zone": proximity.get("raw_zone") or "wait",
        "reason": proximity.get("reason") or "Proximity baseline applied.",
        "primary_level": proximity.get("primary_level") or "None",
        "secondary_target": proximity.get("secondary_target") or None,
        "needs_reanalysis": False,
        "partial_action": "none",
        "message": proximity.get("message") or "Watching key levels for engagement.",
        "debug": {
            "proximity": proximity.get("debug") or {},
            "priority_flow": ["proximity"],
        },
    }

    path_state = proximity.get("path")
    priority_state = 5

    timeframe_slug = next(
        (slug for slug, key in STRATEGY_KEY_BY_TIMEFRAME.items() if key == strategy_key),
        "day",
    )

    level_break = validate_level_break(
        settings,
        timeframe_slug,
        price_context or {},
        s1_price,
        r1_price,
        s2_price,
        r2_price,
    )
    decision["debug"]["level_break"] = level_break["debug"]

    if level_break.get("triggered"):
        decision.update(
            {
                "validated_zone": "wait",
                "reason": level_break.get("reason") or decision["reason"],
                "message": level_break.get("message") or decision["message"],
                "needs_reanalysis": True,
            }
        )
        if level_break.get("primary_level"):
            decision["primary_level"] = level_break["primary_level"]
        if level_break.get("secondary_target"):
            decision["secondary_target"] = level_break["secondary_target"]
        if decision.get("partial_action") == "none" and level_break.get("partial_action") not in {None, "none"}:
            decision["partial_action"] = level_break["partial_action"]
        path_state = level_break.get("path") or path_state
        priority_state = 1
        decision["debug"]["priority_flow"].append("level_break")

    second_level = compute_second_level_progress(
        settings,
        latest_price,
        radar_levels,
        support_levels,
        resistance_levels,
    )
    decision["debug"]["second_level"] = second_level["debug"]

    if second_level.get("engaged") and priority_state > 4:
        decision.update(
            {
                "validated_zone": "wait",
                "reason": second_level.get("reason") or decision["reason"],
                "message": second_level.get("message") or decision["message"],
                "needs_reanalysis": True,
            }
        )
        if second_level.get("primary_level"):
            decision["primary_level"] = second_level["primary_level"]
        if second_level.get("secondary_target"):
            decision["secondary_target"] = second_level["secondary_target"]
        if decision.get("partial_action") == "none" and second_level.get("partial_action") not in {None, "none"}:
            decision["partial_action"] = second_level["partial_action"]
        path_state = second_level.get("path") or path_state
        priority_state = 4
        decision["debug"]["priority_flow"].append("second_level")

    intent_conflict = resolve_intent_conflict(
        intent,
        path_state,
        decision["validated_zone"],
        decision["message"],
    )
    decision["debug"]["intent_conflict"] = intent_conflict["debug"]

    if intent_conflict.get("overridden"):
        decision.update(
            {
                "validated_zone": intent_conflict.get("validated_zone", decision["validated_zone"]),
                "message": intent_conflict.get("message", decision["message"]),
                "reason": intent_conflict.get("reason", decision["reason"]),
            }
        )
        decision["needs_reanalysis"] = decision["needs_reanalysis"] or intent_conflict.get("needs_reanalysis", False)
        if decision.get("partial_action") == "none" and intent_conflict.get("partial_action") not in {None, "none"}:
            decision["partial_action"] = intent_conflict["partial_action"]
        decision["debug"]["priority_flow"].append("intent_conflict")

    decision["secondary_target"] = decision.get("secondary_target") or None
    decision["primary_level"] = decision.get("primary_level") or "None"
    decision["partial_action"] = decision.get("partial_action") or "none"
    decision["validated_zone"] = decision.get("validated_zone") or "wait"
    decision["raw_zone"] = decision.get("raw_zone") or "wait"
    decision["debug"]["final_state_priority"] = priority_state
    decision["debug"]["path_state"] = path_state

    return decision
