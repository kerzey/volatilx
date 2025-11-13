"""Deterministic price action analysis utilities.

This module provides a pure-python price action analyser that reuses the
existing :class:`ComprehensiveMultiTimeframeAnalyzer` data pipeline to pull
candles from Alpaca. It extracts structural insights (trend, support and
resistance, candlestick signals) for each timeframe so the UI can render a
"Price Action" tab without additional LLM calls.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from indicator_fetcher import ComprehensiveMultiTimeframeAnalyzer

logger = logging.getLogger(__name__)

DEFAULT_TIMEFRAMES: List[str] = ["30m", "1h", "1d","1wk"]

TIMEFRAME_DEFAULT_PERIOD: Dict[str, str] = {
    "1m": "5d",
    "2m": "5d",
    "5m": "10d",
    "15m": "1mo",
    "30m": "3mo",
    "60m": "6mo",
    "1h": "6mo",
    "1d": "2y",
    "1wk": "5y",
    "1mo": "10y",
}


def _safe_float(value: Any) -> Optional[float]:
    """Convert values to builtin float, ignoring objects that cannot be cast."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:  # noqa: BLE001
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iso_timestamp(value: Any) -> Optional[str]:
    """Return an ISO8601 timestamp for pandas / datetime inputs."""

    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        if value.tzinfo:
            return value.tz_convert("UTC").isoformat()
        return value.to_pydatetime().isoformat()
    try:
        return datetime.fromisoformat(str(value)).isoformat()
    except Exception:  # noqa: BLE001
        return None


def _percent_diff(base: Optional[float], target: Optional[float]) -> Optional[float]:
    """Return percentage difference between two prices."""

    if base in (None, 0) or target is None:
        return None
    try:
        return round((target - base) / base * 100, 2)
    except Exception:  # noqa: BLE001
        return None


class PriceActionAnalyzer:
    """Extract price action features for a symbol across multiple timeframes."""

    def __init__(
        self,
        *,
        analyzer: Optional[ComprehensiveMultiTimeframeAnalyzer] = None,
    ) -> None:
        self.analyzer = analyzer or self._build_analyzer()

    def analyze(
        self,
        symbol: str,
        *,
        timeframes: Optional[Iterable[str]] = None,
        period_overrides: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Return price action summary keyed by timeframe."""

        cleaned_symbol = symbol.upper()
        prepared_timeframes = self._prepare_timeframes(timeframes)
        period_map = dict(period_overrides or {})

        analyses: Dict[str, Any] = {}
        errors: List[str] = []

        for timeframe in prepared_timeframes:
            period = period_map.get(timeframe, TIMEFRAME_DEFAULT_PERIOD.get(timeframe, "6mo"))
            try:
                raw_df = self.analyzer.get_stock_data(cleaned_symbol, interval=timeframe, period=period)
            except Exception as exc:  # noqa: BLE001 - propagate network diagnostics via errors list
                message = f"Failed to fetch {timeframe} data for {cleaned_symbol}: {exc}"
                logger.error(message)
                errors.append(message)
                continue

            if raw_df is None or raw_df.empty:
                logger.debug("No candle data for %s on timeframe %s", cleaned_symbol, timeframe)
                continue

            df = self._prepare_dataframe(raw_df)
            if df.empty or len(df) < 15:
                logger.debug("Insufficient candles after cleaning for %s on %s", cleaned_symbol, timeframe)
                continue

            timeframe_result = self._analyze_timeframe(cleaned_symbol, timeframe, df)
            if timeframe_result:
                analyses[timeframe] = timeframe_result

        overview = self._build_overview(analyses)
        return {
            "success": bool(analyses),
            "symbol": cleaned_symbol,
            "generated_at": datetime.utcnow().isoformat(),
            "timeframes": prepared_timeframes,
            "overview": overview,
            "per_timeframe": analyses,
            "errors": errors,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.copy()
        cleaned = cleaned.dropna(subset=["Open", "High", "Low", "Close"])
        if cleaned.index.has_duplicates:
            cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
        cleaned = cleaned.sort_index()
        return cleaned

    def _analyze_timeframe(self, symbol: str, timeframe: str, df: pd.DataFrame) -> Dict[str, Any]:
        pivots = self._find_pivots(df)
        trend = self._calculate_trend(df)
        levels = self._identify_key_levels(df, pivots)
        structure = self._assess_structure(df, pivots)
        candlestick = self._detect_candlestick_patterns(df)

        latest = df.iloc[-1]
        previous_close = df["Close"].iloc[-2] if len(df) > 1 else None
        latest_time = _iso_timestamp(df.index[-1])

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "price": {
                "open": _safe_float(latest["Open"]),
                "high": _safe_float(latest["High"]),
                "low": _safe_float(latest["Low"]),
                "close": _safe_float(latest["Close"]),
                "close_change_pct": _percent_diff(previous_close, _safe_float(latest["Close"])),
                "volume": _safe_float(latest.get("Volume")),
                "timestamp": latest_time,
            },
            "trend": trend,
            "levels": levels,
            "structure": structure,
            "candlestick": candlestick,
        }

    def _find_pivots(self, df: pd.DataFrame, left: int = 3, right: int = 3) -> Dict[str, List[Tuple[pd.Timestamp, float]]]:
        highs: List[Tuple[pd.Timestamp, float]] = []
        lows: List[Tuple[pd.Timestamp, float]] = []
        if len(df) < left + right + 2:
            return {"highs": highs, "lows": lows}

        high_values = df["High"].to_numpy()
        low_values = df["Low"].to_numpy()
        index_values = df.index.to_list()
        length = len(df)

        for idx in range(left, length - right):
            high_window = high_values[idx - left : idx + right + 1]
            low_window = low_values[idx - left : idx + right + 1]
            high_price = high_values[idx]
            low_price = low_values[idx]

            if high_price == float(np.max(high_window)):
                highs.append((index_values[idx], float(high_price)))
            if low_price == float(np.min(low_window)):
                lows.append((index_values[idx], float(low_price)))

        # Only keep the most recent pivots for downstream summaries
        return {
            "highs": highs[-12:],
            "lows": lows[-12:],
        }

    def _identify_key_levels(
        self,
        df: pd.DataFrame,
        pivots: Dict[str, List[Tuple[pd.Timestamp, float]]],
        *,
        max_levels: int = 4,
        tolerance_ratio: float = 0.006,
    ) -> Dict[str, Any]:
        current_price = _safe_float(df["Close"].iloc[-1]) or 0.0
        tolerance = max(current_price * tolerance_ratio, 0.1)

        supports: List[Dict[str, Any]] = []
        resistances: List[Dict[str, Any]] = []

        for ts, price in sorted(pivots.get("lows", []), key=lambda item: abs(current_price - item[1])):
            if price > current_price * 1.01:
                continue
            if all(abs(price - entry["price"]) > tolerance for entry in supports):
                supports.append(
                    {
                        "price": round(price, 4),
                        "tested_at": _iso_timestamp(ts),
                        "distance_pct": _percent_diff(current_price, price),
                    }
                )
            if len(supports) >= max_levels:
                break

        for ts, price in sorted(pivots.get("highs", []), key=lambda item: abs(current_price - item[1])):
            if price < current_price * 0.99:
                continue
            if all(abs(price - entry["price"]) > tolerance for entry in resistances):
                resistances.append(
                    {
                        "price": round(price, 4),
                        "tested_at": _iso_timestamp(ts),
                        "distance_pct": _percent_diff(current_price, price),
                    }
                )
            if len(resistances) >= max_levels:
                break

        formatted_highs = [
            {
                "price": round(item[1], 4),
                "timestamp": _iso_timestamp(item[0]),
            }
            for item in sorted(pivots.get("highs", []), key=lambda pair: pair[0])[-5:]
        ]
        formatted_lows = [
            {
                "price": round(item[1], 4),
                "timestamp": _iso_timestamp(item[0]),
            }
            for item in sorted(pivots.get("lows", []), key=lambda pair: pair[0])[-5:]
        ]

        return {
            "supports": supports,
            "resistances": resistances,
            "recent_pivots": {
                "highs": formatted_highs,
                "lows": formatted_lows,
            },
        }

    def _calculate_trend(self, df: pd.DataFrame, window: int = 30) -> Dict[str, Any]:
        closes = df["Close"].tail(window)
        if len(closes) < max(8, window // 2):
            return {"direction": "unknown", "window": len(closes)}

        y = closes.to_numpy(dtype=float)
        x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)
        fitted = intercept + slope * x
        diff = y - fitted
        ss_res = float(np.sum(diff**2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) if len(y) > 1 else 0.0
        r_squared = 1 - ss_res / ss_tot if ss_tot else 0.0

        slope_pct = slope / np.mean(y) if np.mean(y) else 0.0
        direction: str
        strength: str

        if slope_pct > 0.001:
            direction = "uptrend"
        elif slope_pct < -0.001:
            direction = "downtrend"
        else:
            direction = "sideways"

        if abs(slope_pct) > 0.005 and r_squared > 0.4:
            strength = "strong"
        elif abs(slope_pct) > 0.0025 and r_squared > 0.25:
            strength = "moderate"
        else:
            strength = "weak"

        return {
            "direction": direction,
            "strength": strength,
            "slope": round(float(slope), 6),
            "slope_pct": round(float(slope_pct) * 100, 2),
            "r_squared": round(r_squared, 3),
            "window": len(closes),
        }

    def _assess_structure(
        self,
        df: pd.DataFrame,
        pivots: Dict[str, List[Tuple[pd.Timestamp, float]]],
    ) -> Dict[str, Any]:
        highs = sorted(pivots.get("highs", []), key=lambda item: item[0])[-3:]
        lows = sorted(pivots.get("lows", []), key=lambda item: item[0])[-3:]

        higher_highs = bool(highs) and all(highs[idx][1] >= highs[idx - 1][1] for idx in range(1, len(highs)))
        higher_lows = bool(lows) and all(lows[idx][1] >= lows[idx - 1][1] for idx in range(1, len(lows)))
        lower_highs = bool(highs) and all(highs[idx][1] <= highs[idx - 1][1] for idx in range(1, len(highs)))
        lower_lows = bool(lows) and all(lows[idx][1] <= lows[idx - 1][1] for idx in range(1, len(lows)))

        if higher_highs and higher_lows:
            bias = "bullish"
        elif lower_highs and lower_lows:
            bias = "bearish"
        else:
            bias = "mixed"

        recent = df.tail(20)
        prior = df.tail(40).head(20) if len(df) >= 40 else pd.DataFrame()
        recent_range = (recent["High"].max() - recent["Low"].min()) if not recent.empty else 0.0
        prior_range = (prior["High"].max() - prior["Low"].min()) if not prior.empty else None
        range_ratio: Optional[float] = None
        range_state = "unknown"

        if prior_range and prior_range > 0:
            range_ratio = float(recent_range / prior_range)
            if range_ratio < 0.7:
                range_state = "contracting"
            elif range_ratio > 1.3:
                range_state = "expanding"
            else:
                range_state = "stable"

        return {
            "bias": bias,
            "higher_highs": higher_highs,
            "higher_lows": higher_lows,
            "lower_highs": lower_highs,
            "lower_lows": lower_lows,
            "range_state": range_state,
            "range_ratio": round(range_ratio, 2) if range_ratio is not None else None,
        }

    def _detect_candlestick_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        signals: List[Dict[str, Any]] = []
        if len(df) < 2:
            return {"signals": signals}

        latest = df.iloc[-1]
        previous = df.iloc[-2]

        open_ = float(latest["Open"])
        close = float(latest["Close"])
        high = float(latest["High"])
        low = float(latest["Low"])

        body = abs(close - open_)
        candle_range = max(high - low, 1e-9)
        upper_shadow = high - max(open_, close)
        lower_shadow = min(open_, close) - low

        body_ratio = body / candle_range
        upper_ratio = upper_shadow / candle_range
        lower_ratio = lower_shadow / candle_range

        if body_ratio < 0.3 and lower_ratio > 0.5 and upper_ratio < 0.25 and close > open_:
            signals.append({"name": "bullish_hammer", "confidence": round(lower_ratio, 2)})
        if body_ratio < 0.3 and upper_ratio > 0.5 and lower_ratio < 0.25 and close < open_:
            signals.append({"name": "bearish_shooting_star", "confidence": round(upper_ratio, 2)})
        if body_ratio < 0.1:
            signals.append({"name": "doji", "confidence": round(1 - body_ratio, 2)})

        prev_open = float(previous["Open"])
        prev_close = float(previous["Close"])
        prev_body = abs(prev_close - prev_open) + 1e-9

        if close > open_ and prev_close < prev_open:
            if close >= prev_open and open_ <= prev_close:
                strength = min(body / prev_body, 4.0)
                signals.append({"name": "bullish_engulfing", "confidence": round(strength, 2)})
        if close < open_ and prev_close > prev_open:
            if close <= prev_open and open_ >= prev_close:
                strength = min(body / prev_body, 4.0)
                signals.append({"name": "bearish_engulfing", "confidence": round(strength, 2)})

        signals_sorted = sorted(signals, key=lambda item: item.get("confidence", 0), reverse=True)

        return {
            "signals": signals_sorted[:4],
            "latest_candle": {
                "open": round(open_, 4),
                "close": round(close, 4),
                "high": round(high, 4),
                "low": round(low, 4),
                "timestamp": _iso_timestamp(df.index[-1]),
            },
        }

    def _build_overview(self, analyses: Dict[str, Any]) -> Dict[str, Any]:
        if not analyses:
            return {
                "trend_alignment": "no_data",
                "key_levels": [],
                "recent_patterns": [],
            }

        directions = [
            analysis.get("trend", {}).get("direction", "unknown")
            for analysis in analyses.values()
        ]
        filtered = [direction for direction in directions if direction != "unknown"]
        if filtered:
            counts = Counter(filtered)
            dominant, dominant_count = counts.most_common(1)[0]
            alignment_ratio = dominant_count / max(len(filtered), 1)
            if alignment_ratio > 0.75:
                trend_alignment = f"{dominant} (high agreement)"
            elif alignment_ratio > 0.5:
                trend_alignment = f"{dominant} (mixed)"
            else:
                trend_alignment = "mixed"
        else:
            trend_alignment = "unknown"

        key_levels: List[Dict[str, Any]] = []
        for timeframe, analysis in analyses.items():
            levels = analysis.get("levels", {})
            for label in ("supports", "resistances"):
                for level in levels.get(label, [])[:2]:
                    key_levels.append(
                        {
                            "timeframe": timeframe,
                            "type": label[:-1],
                            "price": level.get("price"),
                            "distance_pct": level.get("distance_pct"),
                        }
                    )

        recent_patterns: List[Dict[str, Any]] = []
        for timeframe, analysis in analyses.items():
            for signal in analysis.get("candlestick", {}).get("signals", []):
                recent_patterns.append(
                    {
                        "timeframe": timeframe,
                        "pattern": signal.get("name"),
                        "confidence": signal.get("confidence"),
                    }
                )

        return {
            "trend_alignment": trend_alignment,
            "key_levels": key_levels,
            "recent_patterns": sorted(
                recent_patterns,
                key=lambda item: item.get("confidence", 0) or 0,
                reverse=True,
            )[:6],
        }

    def _prepare_timeframes(self, timeframes: Optional[Iterable[str]]) -> List[str]:
        if timeframes is None:
            return list(DEFAULT_TIMEFRAMES)
        cleaned: List[str] = []
        for tf in timeframes:
            if not tf:
                continue
            candidate = tf.strip()
            if candidate and candidate not in cleaned:
                cleaned.append(candidate)
        return cleaned or list(DEFAULT_TIMEFRAMES)

    def _build_analyzer(self) -> ComprehensiveMultiTimeframeAnalyzer:
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

        analyzer = ComprehensiveMultiTimeframeAnalyzer(
            api_key=api_key,
            secret_key=secret_key,
            base_url=base_url,
        )
        if analyzer.api is None:
            logger.warning(
                "PriceActionAnalyzer initialised without Alpaca credentials. "
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY to enable data fetches."
            )
        return analyzer


__all__ = ["PriceActionAnalyzer"]
