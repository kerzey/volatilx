"""OpenAI-powered expert agents that generate specialised market insights.

This module consolidates all expert agents (momentum, volatility, pattern,
trend) into a single API surface so higher-level trading agents can orchestrate
and reuse them. Each expert agent:

* pulls market data through :mod:`indicator_fetcher`
* extracts a domain-specific feature set
* feeds that context into OpenAI Agent Builder via LangGraph
* returns a structured JSON analysis that downstream agents can rely on
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, TypedDict

import pandas as pd
import pandas_ta as ta
from langgraph.graph import END, START, StateGraph
from openai import APIError, OpenAI

from indicator_fetcher import ComprehensiveMultiTimeframeAnalyzer

logger = logging.getLogger(__name__)


DEFAULT_TIMEFRAMES: List[str] = ["5m", "15m", "1h", "1d"]


try:
    _SUPPORTED_TIMEFRAMES = set(
        ComprehensiveMultiTimeframeAnalyzer().valid_intervals  # type: ignore[arg-type]
    )
except Exception:  # noqa: BLE001 - dependency initialisation may fail without Alpaca creds
    logger.warning(
        "Falling back to default timeframe list; unable to read analyzer intervals."
    )
    _SUPPORTED_TIMEFRAMES = set(DEFAULT_TIMEFRAMES)


TIMEFRAME_DEFAULT_PERIOD: Dict[str, str] = {
    "1m": "1d",
    "2m": "1d",
    "5m": "5d",
    "15m": "1mo",
    "30m": "3mo",
    "60m": "6mo",
    "90m": "6mo",  # LangGraph pipeline remaps 90m -> 60m
    "1h": "6mo",
    "1d": "2y",
    "1wk": "5y",
    "1mo": "10y",
    "3mo": "10y",
}


class AgentState(TypedDict, total=False):
    """State container shared across LangGraph nodes."""

    symbol: str
    timeframes: List[str]
    indicator_summary: Dict[str, Any]
    agent_input: str
    agent_result: Dict[str, Any]
    period_overrides: Dict[str, str]


@dataclass
class AgentMetadata:
    """Metadata describing the OpenAI agent configuration."""

    name: str
    description: str
    instructions: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.4


def _safe_numeric(value: Any) -> Optional[float]:
    """Best-effort conversion to a JSON-safe float."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:  # noqa: BLE001 - fall back to str below
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sanitize(value: Any) -> Any:
    """Recursively sanitise indicator payloads for JSON compatibility."""

    if isinstance(value, dict):
        return {key: _sanitize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize(item) for item in value]
    if isinstance(value, (str, bool)) or value is None:
        return value
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:  # noqa: BLE001 - fall back below
            pass

    numeric = _safe_numeric(value)
    if numeric is not None:
        return numeric

    return str(value)


class BaseExpertAgent:
    """Shared scaffolding for domain-specific OpenAI agents."""

    metadata: AgentMetadata
    default_timeframes: List[str] = DEFAULT_TIMEFRAMES

    def __init__(
        self,
        *,
        analyzer: Optional[ComprehensiveMultiTimeframeAnalyzer] = None,
        openai_client: Optional[OpenAI] = None,
    ) -> None:
        self.analyzer = analyzer or self._build_analyzer()
        self.client = openai_client or self._build_openai_client()
        self.assistant_id = self._ensure_assistant()
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------
    def run(
        self,
        symbol: str,
        *,
        timeframes: Optional[Iterable[str]] = None,
        period_overrides: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Execute the orchestration workflow and return structured analysis."""

        prepared_tfs = self._prepare_timeframes(timeframes)
        initial_state: AgentState = {
            "symbol": symbol.upper(),
            "timeframes": prepared_tfs,
            "indicator_summary": {},
        }
        if period_overrides:
            initial_state["period_overrides"] = dict(period_overrides)

        compiled_graph = self.graph
        final_state = compiled_graph.invoke(
            initial_state,
            config={
                "metadata": {
                    "agent": self.metadata.name,
                    "symbol": symbol.upper(),
                    "timeframes": prepared_tfs,
                }
            },
        )

        return {
            "success": True,
            "symbol": symbol.upper(),
            "agent": self.metadata.name,
            "collected_at": datetime.utcnow().isoformat(),
            "timeframes": prepared_tfs,
            "indicator_summary": final_state.get("indicator_summary", {}),
            "agent_input": final_state.get("agent_input"),
            "agent_output": final_state.get("agent_result"),
            "model": self.metadata.model,
        }

    # ------------------------------------------------------------------
    # LangGraph construction
    # ------------------------------------------------------------------
    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("collect_indicators", self._node_collect_indicators)
        builder.add_node("call_agent", self._node_call_agent)
        builder.add_edge(START, "collect_indicators")
        builder.add_edge("collect_indicators", "call_agent")
        builder.add_edge("call_agent", END)
        return builder.compile()

    def _node_collect_indicators(self, state: AgentState) -> AgentState:
        symbol = state["symbol"]
        timeframes = state.get("timeframes", self.default_timeframes)
        period_overrides: Dict[str, str] = state.get("period_overrides", {})

        summary = self._collect_indicator_summary(symbol, timeframes, period_overrides)
        agent_payload = self._format_agent_payload(symbol, summary)
        return {
            "indicator_summary": summary,
            "agent_input": agent_payload,
        }

    def _node_call_agent(self, state: AgentState) -> AgentState:
        payload = state.get("agent_input", "")
        if not payload:
            raise RuntimeError("Agent payload missing; indicator collection failed.")

        try:
            thread = self.client.beta.threads.create(
                messages=[{"role": "user", "content": payload}]
            )
            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant_id,
                temperature=self.metadata.temperature,
            )

            while run.status in {"queued", "in_progress"}:
                time.sleep(0.5)
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id,
                )

            if run.status == "requires_action":
                logger.error(
                    "Assistant run requires action but no tool handling is implemented."
                )
                raise RuntimeError("Assistant run requires tool action")

            if run.status == "failed":
                details = getattr(run, "last_error", None)
                message = getattr(details, "message", "unknown error")
                logger.error("Assistant run failed: %s", message)
                raise RuntimeError(f"Assistant run failed: {message}")

            if run.status != "completed":
                logger.error("Assistant run ended with unexpected status: %s", run.status)
                raise RuntimeError(f"Assistant run ended with status: {run.status}")

            messages = self.client.beta.threads.messages.list(
                thread_id=thread.id,
                run_id=run.id,
            )

            response_texts: List[str] = []
            for message in reversed(messages.data):
                if getattr(message, "role", "") != "assistant":
                    continue
                print()
                for content in getattr(message, "content", []) or []:
                    if getattr(content, "type", "") == "text":
                        response_texts.append(content.text.value)

            combined_response = "\n".join(chunk for chunk in response_texts if chunk).strip()
            if not combined_response:
                raise RuntimeError("Assistant returned no text content")
        except APIError as exc:  # noqa: BLE001
            logger.error("OpenAI assistant invocation failed: %s", exc)
            raise

        parsed = self._parse_agent_response(combined_response)
        return {"agent_result": parsed}

    # ------------------------------------------------------------------
    # Agent setup helpers
    # ------------------------------------------------------------------
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
                "ComprehensiveMultiTimeframeAnalyzer initialised without Alpaca "
                "credentials. Indicator collection will fail until credentials "
                "are configured."
            )
        return analyzer

    def _build_openai_client(self) -> OpenAI:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        return OpenAI(api_key=api_key)

    def _ensure_assistant(self) -> str:
        assistant = self.client.beta.assistants.create(
            name=self.metadata.name,
            description=self.metadata.description,
            instructions=self.metadata.instructions,
            model=self.metadata.model,
        )
        logger.debug(
            "Created OpenAI assistant '%s' with id %s",
            self.metadata.name,
            assistant.id,
        )
        return assistant.id

    # ------------------------------------------------------------------
    # Indicator collection helpers
    # ------------------------------------------------------------------
    def _prepare_timeframes(self, timeframes: Optional[Iterable[str]]) -> List[str]:
        if timeframes is None:
            base = self.default_timeframes
        else:
            base = [tf.strip() for tf in timeframes if tf and tf.strip()]

        cleaned: List[str] = []
        for tf in base:
            alias = "60m" if tf == "90m" else tf
            if alias not in _SUPPORTED_TIMEFRAMES:
                logger.debug("Skipping unsupported timeframe: %s", tf)
                continue
            if alias not in cleaned:
                cleaned.append(alias)
        return cleaned or list(self.default_timeframes)

    def _collect_indicator_summary(
        self,
        symbol: str,
        timeframes: Iterable[str],
        period_overrides: Dict[str, str],
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        for timeframe in timeframes:
            period = period_overrides.get(timeframe, TIMEFRAME_DEFAULT_PERIOD.get(timeframe, "6mo"))
            try:
                df = self.analyzer.get_stock_data(symbol, interval=timeframe, period=period)
            except Exception as exc:  # noqa: BLE001 - network / third-party errors
                logger.error("Failed to fetch %s data for %s: %s", timeframe, symbol, exc)
                continue

            if df is None or isinstance(df, pd.DataFrame) and df.empty:
                logger.debug("No candles returned for %s on %s", symbol, timeframe)
                continue

            indicators = self.analyzer.calculate_comprehensive_indicators(df, timeframe)
            features = self._extract_features(symbol, timeframe, df, indicators)
            if features:
                summary[timeframe] = _sanitize(features)

        return summary

    # ------------------------------------------------------------------
    # Methods to customise per agent
    # ------------------------------------------------------------------
    def _extract_features(self, symbol: str, timeframe: str, df: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
        """Extract domain-specific features (override in subclasses)."""

        raise NotImplementedError

    def _format_agent_payload(self, symbol: str, summary: Dict[str, Any]) -> str:
        context_json = json.dumps(summary, indent=2)
        return (
            f"Symbol: {symbol}\n"
            f"Indicator summary JSON:\n{context_json}\n"
            "Return a concise structured analysis."
        )

    def _parse_agent_response(self, response_text: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            parsed = {"raw_text": response_text}
        return parsed


class MomentumExpertAgent(BaseExpertAgent):
    """Interprets MACD, RSI, moving averages, and volume momentum."""

    metadata = AgentMetadata(
        name="Momentum Expert Agent",
        description="Generates momentum insights from MACD, RSI, moving averages, and volume cues.",
        instructions=(
            "You are a senior momentum analyst. Review the supplied indicator summary and return a JSON object with "
            "keys: 'momentum_state', 'signals', 'risk', 'actionable_setups'. Highlight divergences, crossovers, "
            "and timeframe alignment."
        ),
    )

    def _extract_features(self, symbol: str, timeframe: str, df: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        basic = indicators.get("basic", {})
        trading = indicators.get("trading_signals", {})
        macd = basic.get("MACD", {})
        rsi = basic.get("RSI", {})
        moving = basic.get("Moving_Averages", {})
        obv = basic.get("OBV", {})

        latest_price = _safe_numeric(df["Close"].iloc[-1])

        return {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "price": latest_price,
            "macd": {
                "value": macd.get("macd"),
                "signal": macd.get("signal"),
                "histogram": macd.get("histogram"),
                "crossover": macd.get("crossover"),
                "settings": macd.get("settings"),
            },
            "rsi": {
                "value": rsi.get("value"),
                "overbought": rsi.get("overbought"),
                "oversold": rsi.get("oversold"),
                "period": rsi.get("period"),
            },
            "moving_average_trend": moving.get("trend"),
            "moving_averages": moving,
            "obv": obv,
            "trading_bias": trading.get("overall_bias"),
            "trading_strength": trading.get("strength"),
            "bullish_signals": trading.get("entry_signals", []),
            "bearish_signals": trading.get("exit_signals", []),
        }


class VolatilityExpertAgent(BaseExpertAgent):
    """Assesses volatility regimes using Bollinger Bands, ATR, and realised volatility."""

    metadata = AgentMetadata(
        name="Volatility Expert Agent",
        description="Assesses volatility dynamics using Bollinger Bands, ATR, and realised volatility.",
        instructions=(
            "You are a volatility strategist. Review the indicator summary and return a JSON object with keys: "
            "'volatility_state', 'band_analysis', 'risk_alerts', 'position_sizing'. Include numeric context."
        ),
    )

    def _extract_features(self, symbol: str, timeframe: str, df: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        basic = indicators.get("basic", {})
        bands = basic.get("Bollinger_Bands", {})
        current_price = _safe_numeric(df["Close"].iloc[-1])

        length = min(14, max(len(df) - 1, 1))
        atr_series = ta.atr(df["High"], df["Low"], df["Close"], length=length)
        atr_value = _safe_numeric(atr_series.iloc[-1]) if atr_series is not None else None

        log_returns = df["Close"].pct_change().dropna()
        realised_vol = None
        if not log_returns.empty:
            realised_vol = float(log_returns.std() * math.sqrt(252))

        trading = indicators.get("trading_signals", {})

        return {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "price": current_price,
            "bollinger_bands": bands,
            "atr": atr_value,
            "atr_length": length,
            "realised_volatility": realised_vol,
            "key_levels": trading.get("key_levels", {}),
            "risk_reward": trading.get("risk_reward", {}),
            "bullish_signals": trading.get("entry_signals", []),
            "bearish_signals": trading.get("exit_signals", []),
        }


class PatternRecognitionExpertAgent(BaseExpertAgent):
    """Evaluates Elliott Wave structure and Fibonacci confluence."""

    default_timeframes = ["1h", "1d", "1wk"]

    metadata = AgentMetadata(
        name="Pattern Recognition Expert Agent",
        description="Interprets Elliott Wave structures and Fibonacci retracements for pattern confirmation.",
        instructions=(
            "You are a market structure specialist. Return a JSON object with keys: 'wave_diagnosis', 'fibonacci_levels', "
            "'confluence_zones', 'trade_plan'. Be explicit about invalidation criteria."
        ),
    )

    def _extract_features(self, symbol: str, timeframe: str, df: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        fibonacci = indicators.get("fibonacci", {})
        elliott = indicators.get("elliott_wave", {})
        trading = indicators.get("trading_signals", {})

        return {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "price": _safe_numeric(df["Close"].iloc[-1]),
            "fibonacci": fibonacci,
            "elliott_wave": elliott,
            "key_levels": trading.get("key_levels", {}),
            "overall_bias": trading.get("overall_bias"),
            "confidence": trading.get("confidence"),
            "bullish_signals": trading.get("entry_signals", []),
            "bearish_signals": trading.get("exit_signals", []),
        }


class TrendExpertAgent(BaseExpertAgent):
    """Summarises multi-timeframe trend alignment and directional bias."""

    default_timeframes = ["5m", "15m", "1h", "1d", "1wk"]

    metadata = AgentMetadata(
        name="Trend Expert Agent",
        description="Produces consolidated trend insights from moving averages, ADX, and sentiment cues.",
        instructions=(
            "You are responsible for multi-timeframe trend assessment. Return a JSON object with keys: "
            "'overall_trend', 'timeframe_breakdown', 'trade_ideas', 'risk_management', 'confidence'."
        ),
    )

    def _extract_features(self, symbol: str, timeframe: str, df: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        basic = indicators.get("basic", {})
        trading_signals = indicators.get("trading_signals", {})

        return {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "price": _safe_numeric(df["Close"].iloc[-1]),
            "bias": trading_signals.get("overall_bias"),
            "strength": _safe_numeric(trading_signals.get("strength")),
            "confidence": trading_signals.get("confidence"),
            "bullish_signals": trading_signals.get("entry_signals", []),
            "bearish_signals": trading_signals.get("exit_signals", []),
            "risk_reward": _sanitize(trading_signals.get("risk_reward", {})),
            "key_levels": _sanitize(trading_signals.get("key_levels", {})),
            "indicator_details": {
                "macd": _sanitize(basic.get("MACD")),
                "adx": _sanitize(basic.get("ADX")),
                "rsi": _sanitize(basic.get("RSI")),
                "obv": _sanitize(basic.get("OBV")),
                "moving_averages": _sanitize(basic.get("Moving_Averages")),
                "bollinger": _sanitize(basic.get("Bollinger_Bands")),
            },
        }

    def _format_agent_payload(self, symbol: str, summary: Dict[str, Any]) -> str:
        context_json = json.dumps(summary, indent=2)
        return (
            "You evaluate trend alignment across intraday, swing, and position timeframes. "
            "Provide consensus, conflicts, and recommended trade structures.\n"
            f"Symbol: {symbol.upper()}\n"
            f"Trend indicator summary (JSON):\n{context_json}\n"
            "Output strict JSON with the keys: overall_trend, timeframe_breakdown, trade_ideas, risk_management, confidence."
        )


__all__ = [
    "BaseExpertAgent",
    "MomentumExpertAgent",
    "VolatilityExpertAgent",
    "PatternRecognitionExpertAgent",
    "TrendExpertAgent",
]
