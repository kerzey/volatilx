"""OpenAI-powered expert agents that generate specialised market insights.

This module consolidates all expert agents (momentum, volatility, pattern,
trend) into a single API surface so higher-level trading agents can orchestrate
and reuse them. Each expert agent:

* pulls market data through :mod:`indicator_fetcher`
* extracts a domain-specific feature set
* delivers the context to OpenAI's Responses API via LangGraph orchestration
* returns a structured JSON analysis that downstream agents can rely on
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypedDict

import pandas as pd
import pandas_ta as ta
from langgraph.graph import END, START, StateGraph
from openai import APIError, OpenAI

from indicator_fetcher import ComprehensiveMultiTimeframeAnalyzer

try:  # noqa: SIM105 - optional dependency for HTTP fallback
    import requests
except ImportError:  # pragma: no cover - requests is optional
    requests = None  # type: ignore[assignment]

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


def _normalise_base_url(raw: Optional[Any]) -> str:
    """Ensure the OpenAI base URL ends with /v1 and has no trailing slash."""

    if raw is None:
        base = "https://api.openai.com/v1"
    else:
        base = str(raw).strip()
        if not base:
            base = "https://api.openai.com/v1"

    base = base.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


class OpenAIResponsesMixin:
    """Shared helpers to invoke the Responses API with SDK or HTTP fallback."""

    client: OpenAI
    _openai_api_key: Optional[str]
    _openai_organization: Optional[str]
    _openai_project: Optional[str]
    _openai_base_url: str
    _responses_timeout: float

    def _init_openai_client(self, openai_client: Optional[OpenAI]) -> OpenAI:
        organization_env = os.getenv("OPENAI_ORG") or os.getenv("OPENAI_ORGANIZATION")
        project_env = os.getenv("OPENAI_PROJECT")
        raw_base_url_env = os.getenv("OPENAI_BASE_URL")

        timeout_env = os.getenv("OPENAI_TIMEOUT")
        try:
            self._responses_timeout = float(timeout_env) if timeout_env else 60.0
        except ValueError:
            self._responses_timeout = 60.0

        if openai_client is not None:
            base_candidate = getattr(openai_client, "base_url", None) or raw_base_url_env
            self._openai_base_url = _normalise_base_url(base_candidate)
            client_key = getattr(openai_client, "api_key", None) or os.getenv("OPENAI_API_KEY")
            if not client_key:
                raise ValueError(
                    "OpenAI API key is required when providing a custom OpenAI client"
                )
            self._openai_api_key = client_key
            self._openai_organization = getattr(openai_client, "organization", organization_env)
            self._openai_project = getattr(openai_client, "project", project_env)
            return openai_client

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self._openai_api_key = api_key
        self._openai_organization = organization_env
        self._openai_project = project_env
        self._openai_base_url = _normalise_base_url(raw_base_url_env)

        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if organization_env:
            client_kwargs["organization"] = organization_env
        if project_env:
            client_kwargs["project"] = project_env
        if raw_base_url_env:
            client_kwargs["base_url"] = raw_base_url_env

        return OpenAI(**client_kwargs)

    def _create_responses_call(self, body: Dict[str, Any]) -> Any:
        payload = {key: value for key, value in body.items() if value is not None}
        responses_resource = getattr(self.client, "responses", None)
        if responses_resource is not None:
            return responses_resource.create(**payload)
        logger.debug("OpenAI client missing 'responses' resource; using HTTP fallback")
        return self._fallback_responses_http(payload)

    def _fallback_responses_http(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if requests is None:
            raise RuntimeError(
                "OpenAI SDK does not expose the Responses API and the 'requests' package is missing. "
                "Install openai>=1.40 or add requests as a dependency."
            )
        if not self._openai_api_key:
            raise RuntimeError("OpenAI API key unavailable for Responses API fallback")

        headers = {
            "Authorization": f"Bearer {self._openai_api_key}",
            "Content-Type": "application/json",
        }
        if self._openai_organization:
            headers["OpenAI-Organization"] = self._openai_organization
        if self._openai_project:
            headers["OpenAI-Project"] = self._openai_project

        url = f"{self._openai_base_url.rstrip('/')}/responses"
        try:
            response = requests.post(  # type: ignore[call-arg]
                url,
                headers=headers,
                json=payload,
                timeout=self._responses_timeout,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to call OpenAI Responses endpoint: {exc}") from exc

        if response.status_code >= 400:
            raise RuntimeError(
                f"OpenAI Responses API error {response.status_code}: {response.text}"
            )

        try:
            return response.json()
        except ValueError as exc:  # pragma: no cover - unlikely unless API misbehaves
            raise RuntimeError("Failed to parse OpenAI Responses API JSON payload") from exc


class AgentState(TypedDict, total=False):
    """State container shared across LangGraph nodes."""

    symbol: str
    timeframes: List[str]
    indicator_summary: Dict[str, Any]
    agent_input: str
    agent_result: Dict[str, Any]
    agent_raw_text: str
    agent_usage: Dict[str, Any]
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



def _extract_usage_details(response: Any) -> Dict[str, Any]:
    """Normalize token usage metadata from OpenAI responses."""

    usage: Dict[str, Any] = {}
    response_usage = getattr(response, "usage", None)
    if response_usage is None and isinstance(response, dict):
        response_usage = response.get("usage")

    if isinstance(response_usage, list):
        list_usage: Dict[str, Any] = {}
        for entry in response_usage:
            if not isinstance(entry, dict):
                continue
            metric = entry.get("metric") or entry.get("name")
            value = entry.get("value")
            if isinstance(metric, str):
                list_usage[metric] = value
        response_usage = list_usage or response_usage

    if response_usage is not None:
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            value = getattr(response_usage, key, None)
            if value is None and isinstance(response_usage, dict):
                value = response_usage.get(key)
            normalized = _coerce_token_value(value)
            if normalized is not None:
                usage[key] = normalized

    if isinstance(response_usage, dict):
        synonym_map = {
            "input_tokens": ("prompt_tokens", "prompt", "input", "accepted_tokens"),
            "output_tokens": ("completion_tokens", "completion", "output", "generated_tokens"),
            "total_tokens": ("total", "aggregate_tokens", "all_tokens"),
        }
        for target_key, synonyms in synonym_map.items():
            if target_key in usage:
                continue
            for synonym in synonyms:
                if synonym in response_usage:
                    normalized = _coerce_token_value(response_usage.get(synonym))
                    if normalized is not None:
                        usage[target_key] = normalized
                        break

    if "total_tokens" not in usage:
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        if input_tokens is not None or output_tokens is not None:
            usage["total_tokens"] = int((input_tokens or 0) + (output_tokens or 0))

    return usage


def _coerce_token_value(value: Any) -> Optional[int]:
    """Coerce varied token usage formats into integers."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return None
    if isinstance(value, dict):
        for candidate_key in ("total_tokens", "value", "tokens", "count", "total", "billable"):
            candidate_val = value.get(candidate_key)
            if isinstance(candidate_val, (int, float)):
                return int(candidate_val)
        numeric_parts = [int(v) for v in value.values() if isinstance(v, (int, float))]
        if numeric_parts:
            return int(sum(numeric_parts))
    if isinstance(value, list):
        numeric_parts = [int(v) for v in value if isinstance(v, (int, float))]
        if numeric_parts:
            return int(sum(numeric_parts))
    return None


def _extract_output_text(response: Any) -> str:
    """Best-effort extraction of model text from Responses payloads."""

    if response is None:
        return ""

    candidate = getattr(response, "output_text", None)
    if isinstance(candidate, str) and candidate.strip():
        return candidate

    if isinstance(response, dict):
        candidate = response.get("output_text")
        if isinstance(candidate, str) and candidate.strip():
            return candidate
        output_items = response.get("output")
        collected: List[str] = []
        if isinstance(output_items, list):
            for item in output_items:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                text_value = item.get("text")
                if item_type == "output_text" and isinstance(text_value, str):
                    collected.append(text_value)
                    continue
                content_list = item.get("content")
                if isinstance(content_list, list):
                    for content in content_list:
                        if not isinstance(content, dict):
                            continue
                        content_type = content.get("type")
                        if content_type == "output_text" and isinstance(content.get("text"), str):
                            collected.append(str(content["text"]))
                        elif content_type == "input_text" and item_type == "message" and content.get("role") == "assistant":
                            text_candidate = content.get("text")
                            if isinstance(text_candidate, str):
                                collected.append(text_candidate)
            if collected:
                return "\n".join(collected)

    output_attr = getattr(response, "output", None)
    if isinstance(output_attr, list):
        parts: List[str] = []
        for item in output_attr:
            if not isinstance(item, dict):
                continue
            text_value = getattr(item, "text", None)
            item_type = getattr(item, "type", None)
            if item_type == "output_text" and isinstance(text_value, str):
                parts.append(text_value)
            content_list = getattr(item, "content", None)
            if isinstance(content_list, list):
                for content in content_list:
                    text_candidate = getattr(content, "text", None)
                    content_type = getattr(content, "type", None)
                    if content_type == "output_text" and isinstance(text_candidate, str):
                        parts.append(text_candidate)
        if parts:
            return "\n".join(parts)

    return ""


class BaseExpertAgent(OpenAIResponsesMixin):
    """Shared scaffolding for domain-specific OpenAI agents."""

    metadata: AgentMetadata
    default_timeframes: List[str] = DEFAULT_TIMEFRAMES

    def __init__(
        self,
        *,
        analyzer: Optional[ComprehensiveMultiTimeframeAnalyzer] = None,
        openai_client: Optional[OpenAI] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> None:
        self.analyzer = analyzer or self._build_analyzer()
        self.client = self._init_openai_client(openai_client)
        metadata_template = getattr(self.__class__, "metadata")
        if model is not None or temperature is not None:
            metadata_template = replace(
                metadata_template,
                model=model or metadata_template.model,
                temperature=temperature if temperature is not None else metadata_template.temperature,
            )
        self.metadata = metadata_template
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

        final_state = self.graph.invoke(
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
            "agent_output_text": final_state.get("agent_raw_text"),
            "model_usage": final_state.get("agent_usage"),
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
            response_text, usage = self._invoke_responses(payload)
        except APIError as exc:  # noqa: BLE001
            logger.error("OpenAI responses API invocation failed: %s", exc)
            raise

        parsed = self._parse_agent_response(response_text)
        return {
            "agent_result": parsed,
            "agent_raw_text": response_text,
            "agent_usage": usage,
        }

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
    def _extract_features(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
    ) -> Dict[str, Any]:  # noqa: D401
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

    def _build_system_prompt(self) -> str:
        return (
            f"You are {self.metadata.name}. {self.metadata.description} "
            f"Follow these instructions:\n{self.metadata.instructions}"
        )

    def _build_responses_input(self, payload_text: str) -> List[Dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": self._build_system_prompt()}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": payload_text}],
            },
        ]

    def _invoke_responses(self, payload_text: str) -> Tuple[str, Dict[str, Any]]:
        request_payload: Dict[str, Any] = {
            "model": self.metadata.model,
            "input": self._build_responses_input(payload_text),
        }
        temperature = getattr(self.metadata, "temperature", None)
        if (
            temperature is not None
            and not str(self.metadata.model).lower().startswith("gpt-5")
        ):
            request_payload["temperature"] = temperature

        response = self._create_responses_call(request_payload)

        response_text = _extract_output_text(response)
        if not response_text:
            raise RuntimeError("Model returned no text content")

        usage = _extract_usage_details(response)
        return response_text, usage


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

    def _extract_features(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
    ) -> Dict[str, Any]:
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
class UIBackedExpertAgent(OpenAIResponsesMixin):
    """OpenAI Responses API wrapper that consumes UI-provided analysis payloads."""

    metadata: AgentMetadata

    def __init__(
        self,
        *,
        openai_client: Optional[OpenAI] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        include_payload: bool = False,
    ) -> None:
        self.client = self._init_openai_client(openai_client)
        metadata_template = getattr(self.__class__, "metadata")
        if model is not None or temperature is not None:
            metadata_template = replace(
                metadata_template,
                model=model or metadata_template.model,
                temperature=temperature if temperature is not None else metadata_template.temperature,
            )
        self.metadata = metadata_template
        self.include_payload = include_payload

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        symbol: str,
        payload: Any,
        *,
        include_payload: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if payload is None:
            return {
                "success": False,
                "agent": self.metadata.name,
                "symbol": symbol.upper(),
                "error": "No payload provided",
            }

        effective_include = self.include_payload if include_payload is None else include_payload
        sanitized_payload = _sanitize(payload)
        user_prompt = self._format_payload(symbol, sanitized_payload)

        try:
            response_text, usage = self._invoke_responses(user_prompt)
        except Exception as exc:  # noqa: BLE001 - bubble structured error upwards
            logger.exception("Expert '%s' failed for %s", self.metadata.name, symbol)
            return {
                "success": False,
                "agent": self.metadata.name,
                "symbol": symbol.upper(),
                "error": str(exc),
            }

        parsed = self._parse_agent_response(response_text)
        result: Dict[str, Any] = {
            "success": True,
            "agent": self.metadata.name,
            "symbol": symbol.upper(),
            "agent_output": parsed,
            "raw_text": response_text,
            "model": self.metadata.model,
            "model_usage": usage,
        }
        if effective_include:
            result["input_payload"] = sanitized_payload
        return result

    # ------------------------------------------------------------------
    # Responses helpers
    # ------------------------------------------------------------------
    def _build_system_prompt(self) -> str:
        return (
            f"You are {self.metadata.name}. {self.metadata.description} "
            f"Follow these instructions:\n{self.metadata.instructions}"
        )

    def _build_responses_input(self, payload_text: str) -> List[Dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": self._build_system_prompt()}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": payload_text}],
            },
        ]

    def _invoke_responses(self, payload_text: str) -> Tuple[str, Dict[str, Any]]:
        request_payload: Dict[str, Any] = {
            "model": self.metadata.model,
            "input": self._build_responses_input(payload_text),
        }
        temperature = getattr(self.metadata, "temperature", None)
        if (
            temperature is not None
            and not str(self.metadata.model).lower().startswith("gpt-5")
        ):
            request_payload["temperature"] = temperature

        response = self._create_responses_call(request_payload)

        response_text = _extract_output_text(response)
        if not response_text:
            raise RuntimeError("Model returned no text content")

        usage = _extract_usage_details(response)
        return response_text, usage

    def _format_payload(self, symbol: str, payload: Any) -> str:
        context_json = json.dumps(payload, indent=2)
        return (
            f"Symbol: {symbol.upper()}\n"
            "Input analysis JSON from UI:\n"
            f"{context_json}\n"
            "Return a JSON object with your insights."
        )

    def _parse_agent_response(self, response_text: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            parsed = {"raw_text": response_text}
        return parsed


class TechnicalAnalysisSummaryAgent(UIBackedExpertAgent):
    """Summarises indicator-based technical analysis produced in the UI."""

    metadata = AgentMetadata(
        name="Technical Analysis Expert",
        description="Transforms UI-generated technical indicator digests into structured trading insights.",
        instructions=(
            "You are the technical analysis expert. The user provides comprehensive indicator data already computed "
            "by upstream services (indicators, Elliott Wave, Fibonacci). Analyse the JSON and respond with a JSON "
            "object containing keys: 'market_structure', 'indicator_signals', 'risk_management', and 'trade_setups'. "
            "Each key should hold succinct, client-ready observations."
        ),
        model="gpt-5-nano",
        temperature=0.3,
    )

    def _format_payload(self, symbol: str, payload: Any) -> str:
        context_json = json.dumps(payload, indent=2)
        return (
            f"Symbol: {symbol.upper()}\n"
            "The following JSON is the raw technical analysis produced in the UI (indicators, Elliott Wave, Fibonacci).\n"
            f"{context_json}\n"
            "Return JSON with keys: 'market_structure', 'indicator_signals', 'risk_management', 'trade_setups'."
        )


class PriceActionSummaryAgent(UIBackedExpertAgent):
    """Summarises deterministic price action analysis from the UI."""

    metadata = AgentMetadata(
        name="Price Action Expert",
        description="Converts deterministic price action analysis into consumable insights for the principal agent.",
        instructions=(
            "You are the price action specialist. Review the supplied JSON, which includes support/resistance levels, "
            "market structure, and candlestick signals. Respond with a JSON object containing: 'structure', 'levels', "
            "'candlestick_notes', and 'immediate_bias'. Provide concise, actionable language."
        ),
        model="gpt-5-nano",
        temperature=0.3,
    )

    def _format_payload(self, symbol: str, payload: Any) -> str:
        context_json = json.dumps(payload, indent=2)
        return (
            f"Symbol: {symbol.upper()}\n"
            "The following JSON contains the deterministic price action analysis from the UI (trend, structure, levels, patterns).\n"
            f"{context_json}\n"
            "Return JSON with keys: 'structure', 'levels', 'candlestick_notes', 'immediate_bias'."
        )


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
    "OpenAIResponsesMixin",
    "BaseExpertAgent",
    "MomentumExpertAgent",
    "VolatilityExpertAgent",
    "PatternRecognitionExpertAgent",
    "TrendExpertAgent",
    "UIBackedExpertAgent",
    "TechnicalAnalysisSummaryAgent",
    "PriceActionSummaryAgent",
]
