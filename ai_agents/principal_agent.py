"""Principal agent that fuses trading strategies for the client."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from langgraph.graph import END, START, StateGraph
from openai import APIError, OpenAI
from indicator_fetcher import ComprehensiveMultiTimeframeAnalyzer

from .expert_agent import (
    MomentumExpertAgent,
    PatternRecognitionExpertAgent,
    TrendExpertAgent,
    VolatilityExpertAgent,
)
from .trading_agents import (
    BaseTradingAgent,
    DayTradingAgent,
    LongTermTradingAgent,
    SwingTradingAgent,
)


logger = logging.getLogger(__name__)


class PrincipalAgentState(TypedDict, total=False):
    """Intermediate state handed through the LangGraph pipeline."""

    symbol: str
    include_raw_results: bool
    trading_results: Dict[str, Any]
    principal_result: Dict[str, Any]


def _json_dump(data: Any) -> str:
    return json.dumps(data, indent=2, default=str)


class PrincipalAgent:
    """High-level orchestrator that aggregates trading strategies."""

    def __init__(
        self,
        *,
        openai_client: Optional[OpenAI] = None,
        openai_model: str = "gpt-4o-mini",#"gpt-5-nano",#
        temperature: float = 0.35,
        momentum_agent: Optional[MomentumExpertAgent] = None,
        volatility_agent: Optional[VolatilityExpertAgent] = None,
        pattern_agent: Optional[PatternRecognitionExpertAgent] = None,
        trend_agent: Optional[TrendExpertAgent] = None,
        day_trading_agent: Optional[DayTradingAgent] = None,
        swing_trading_agent: Optional[SwingTradingAgent] = None,
        longterm_trading_agent: Optional[LongTermTradingAgent] = None,
    ) -> None:
        self.client = openai_client or self._build_openai_client()
        self.model = openai_model
        self.temperature = temperature
        self._summary_token_limit = 1600

        analyzer = ComprehensiveMultiTimeframeAnalyzer(
            api_key=os.getenv("ALPACA_API_KEY"),
            secret_key=os.getenv("ALPACA_SECRET_KEY"),
            base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
        )

        # Instantiate expert agents once and share across trading agents
        self.experts = {
            "momentum": momentum_agent
            or MomentumExpertAgent(openai_client=self.client, analyzer=analyzer),
            "volatility": volatility_agent
            or VolatilityExpertAgent(openai_client=self.client, analyzer=analyzer),
            "pattern": pattern_agent
            or PatternRecognitionExpertAgent(openai_client=self.client, analyzer=analyzer),
            "trend": trend_agent
            or TrendExpertAgent(openai_client=self.client, analyzer=analyzer),
        }

        self.trading_agents = {
            "day_trading": day_trading_agent
            or DayTradingAgent(
                momentum_agent=self.experts["momentum"],
                volatility_agent=self.experts["volatility"],
                trend_agent=self.experts["trend"],
            ),
            "swing_trading": swing_trading_agent
            or SwingTradingAgent(
                momentum_agent=self.experts["momentum"],
                volatility_agent=self.experts["volatility"],
                pattern_agent=self.experts["pattern"],
                trend_agent=self.experts["trend"],
            ),
            "longterm_trading": longterm_trading_agent
            or LongTermTradingAgent(
                volatility_agent=self.experts["volatility"],
                pattern_agent=self.experts["pattern"],
                trend_agent=self.experts["trend"],
            ),
        }

        self._trading_sequence: list[tuple[str, BaseTradingAgent]] = []
        self.refresh_graph()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_trading_plan(
        self,
        symbol: str,
        *,
        include_raw_results: bool = True,
    ) -> Dict[str, Any]:
        """Collect strategy insights and return client-facing recommendations."""

        self.refresh_graph()
        initial_state: PrincipalAgentState = {
            "symbol": symbol.upper(),
            "include_raw_results": include_raw_results,
            "trading_results": {},
        }

        final_state = self.graph.invoke(
            initial_state,
            config={
                "metadata": {
                    "agent": "principal",
                    "symbol": symbol.upper(),
                }
            },
        )

        result = final_state.get("principal_result")
        if not isinstance(result, dict):
            raise RuntimeError("Principal agent did not produce a result")

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_openai_client(self) -> OpenAI:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        return OpenAI(api_key=api_key)

    def _summarise_for_client(
        self,
        symbol: str,
        trading_results: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        user_payload = _json_dump(trading_results)
        system_prompt = (
            "You are the principal trading strategist for an AI-driven desk. "
            "The model outputs from subordinate trading agents (day, swing, long-term) are provided in JSON. "
            "Return a strict JSON object with keys: 'day_trading', 'swing_trading', 'longterm_trading', "
            "'global_risks', and 'portfolio_guidance'. Each strategy key should include entries for 'summary', "
            "'key_levels', and 'next_actions'."
        )

        try:
            response = self._create_summary_completion(symbol, system_prompt, user_payload)
        except APIError as exc:  # noqa: BLE001
            raise RuntimeError(f"Principal agent failed to summarise strategies: {exc}") from exc

        content = response.choices[0].message.content or "{}"
        try:
            summary = json.loads(content)
        except json.JSONDecodeError:
            summary = {"raw_text": content}

        usage = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
            "completion_tokens": getattr(response.usage, "completion_tokens", None),
            "total_tokens": getattr(response.usage, "total_tokens", None),
        }

        return summary, usage

    def _create_summary_completion(self, symbol: str, system_prompt: str, user_payload: str):
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Symbol: {symbol.upper()}\n"
                    "Trading agent outputs (JSON):\n"
                    f"{user_payload}\n"
                    "Craft disciplined, risk-aware guidance for the client."
                ),
            },
        ]

        base_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }

        if self._model_requires_default_temperature():
            base_kwargs.pop("temperature", None)

        token_params = ("max_completion_tokens", "max_tokens")
        last_error: Optional[APIError] = None
        for param_name in token_params:
            try:
                response = self.client.chat.completions.create(
                    **base_kwargs,
                    **{param_name: self._summary_token_limit},
                )
                return response
            except APIError as exc:  # noqa: BLE001
                last_error = exc
                if not self._is_unsupported_parameter_error(exc):
                    raise
                continue

        if last_error is not None:
            raise last_error

        raise RuntimeError("Failed to create summary completion: no completion attempted")

    @staticmethod
    def _is_unsupported_parameter_error(exc: APIError) -> bool:
        message = str(getattr(exc, "message", ""))
        if "unsupported parameter" in message.lower():
            return True
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            error_data = body.get("error")
            if isinstance(error_data, dict):
                if error_data.get("code") == "unsupported_parameter":
                    return True
                message_text = error_data.get("message", "")
                if isinstance(message_text, str) and "unsupported parameter" in message_text.lower():
                    return True
        return False

    def _model_requires_default_temperature(self) -> bool:
        if not self.model:
            return False
        lowered = self.model.lower()
        return lowered.startswith("gpt-5-")

    def _build_graph(self):
        builder = StateGraph(PrincipalAgentState)
        builder.add_node("initialise", self._initialise_state)
        builder.add_edge(START, "initialise")

        previous = "initialise"
        for strategy, agent in self._trading_sequence:
            node_name = f"run_{strategy}"
            builder.add_node(node_name, self._make_trading_node(strategy, agent))
            builder.add_edge(previous, node_name)
            previous = node_name

        builder.add_node("summarise", self._summarise_node)
        builder.add_edge(previous, "summarise")
        builder.add_edge("summarise", END)
        return builder.compile()

    def refresh_graph(self) -> None:
        """Recompile the LangGraph pipeline after registry changes."""

        self._trading_sequence = list(self.trading_agents.items())
        self.graph = self._build_graph()

    def _initialise_state(self, state: PrincipalAgentState) -> PrincipalAgentState:
        symbol = state.get("symbol")
        if symbol is None:
            raise RuntimeError("Principal agent initial state missing symbol")
        return {
            "symbol": symbol,
            "trading_results": dict(state.get("trading_results", {})),
            "include_raw_results": state.get("include_raw_results", True),
        }

    def _make_trading_node(self, strategy: str, agent: BaseTradingAgent):
        def _node(state: PrincipalAgentState) -> PrincipalAgentState:
            symbol = state["symbol"]
            logger.debug("Running trading agent '%s' for symbol %s", strategy, symbol)
            result = agent.run(symbol)
            results = dict(state.get("trading_results", {}))
            results[strategy] = result
            return {"symbol": symbol, "trading_results": results}

        return _node

    def _summarise_node(self, state: PrincipalAgentState) -> PrincipalAgentState:
        symbol = state.get("symbol")
        if symbol is None:
            raise RuntimeError("Principal agent state missing symbol during summary")
        trading_results = state.get("trading_results", {})
        include_raw = state.get("include_raw_results", True)

        strategy_summary, usage = self._summarise_for_client(symbol, trading_results)
        strategies, supplemental = self._normalise_strategy_summary(strategy_summary)

        payload: Dict[str, Any] = {
            "symbol": symbol,
            "generated_at": datetime.utcnow().isoformat(),
            "strategies": strategies,
            "model": self.model,
            "usage": usage,
        }

        if supplemental:
            payload["context"] = supplemental
        if not strategies:
            payload["strategies"] = self._fallback_strategies_from_trading_results(trading_results)

        if include_raw:
            payload["trading_agent_outputs"] = trading_results

        return {"principal_result": payload}

    def _normalise_strategy_summary(self, summary: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        strategies: Dict[str, Any] = {}
        supplemental: Dict[str, Any] = {}
        if not isinstance(summary, dict):
            return strategies, supplemental

        working = summary.get("strategies") if isinstance(summary.get("strategies"), dict) else summary

        alias_map = {
            "day trading": "day_trading",
            "day_trading": "day_trading",
            "intraday": "day_trading",
            "swing trading": "swing_trading",
            "swing_trading": "swing_trading",
            "swing": "swing_trading",
            "longterm trading": "longterm_trading",
            "longterm_trading": "longterm_trading",
            "long term": "longterm_trading",
            "long-term": "longterm_trading",
        }

        for key, value in working.items():
            normalised_key = alias_map.get(key.lower()) if isinstance(key, str) else None
            if normalised_key:
                strategies[normalised_key] = value
            else:
                supplemental[key] = value

        return strategies, supplemental

    def _fallback_strategies_from_trading_results(self, trading_results: Dict[str, Any]) -> Dict[str, Any]:
        fallback: Dict[str, Any] = {}
        for strategy_key, result in trading_results.items():
            summary_lines: List[str] = []
            aggregated_levels: Dict[str, Any] = {}
            next_actions: List[str] = []

            if isinstance(result, dict):
                bias = result.get("strategy")
                if isinstance(bias, str) and bias:
                    summary_lines.append(f"Strategy focus: {self._humanise_key(bias)}")

                agent_data = result.get("experts")
                if isinstance(agent_data, dict):
                    for expert_key, expert_value in agent_data.items():
                        summary_text, key_levels, actions = self._summarise_expert_output(expert_key, expert_value)
                        if summary_text:
                            summary_lines.append(f"{self._humanise_key(expert_key)}: {summary_text}")
                        for level_key, level_value in key_levels.items():
                            human_key = self._humanise_key(level_key)
                            key_name = human_key if human_key not in aggregated_levels else f"{self._humanise_key(expert_key)} {human_key}"
                            aggregated_levels[key_name] = level_value
                        if actions:
                            next_actions.extend(actions)

            fallback[strategy_key] = {
                "summary": " ".join(summary_lines) if summary_lines else "No summary available.",
                "key_levels": aggregated_levels or None,
                "next_actions": next_actions or ["Review expert diagnostics for details."],
            }

        return fallback

    def _summarise_expert_output(
        self,
        expert_key: str,
        expert_value: Any,
    ) -> Tuple[str, Dict[str, Any], List[str]]:
        if not isinstance(expert_value, dict):
            return "", {}, []

        output = expert_value.get("agent_output") or expert_value.get("agent_result")
        if not isinstance(output, dict):
            return "", {}, []

        summary_candidates = (
            "summary",
            "overall_trend",
            "momentum_state",
            "volatility_state",
            "wave_diagnosis",
            "trade_bias",
            "market_position",
        )

        summary_text = ""
        for candidate in summary_candidates:
            value = output.get(candidate)
            summary_text = self._coerce_to_sentence(value)
            if summary_text:
                break

        if not summary_text:
            backup_sources = (
                output.get("signals"),
                output.get("timeframe_breakdown"),
                output.get("indicator_details"),
            )
            for source in backup_sources:
                summary_text = self._coerce_to_sentence(source)
                if summary_text:
                    break

        key_levels: Dict[str, Any] = {}
        levels_value = output.get("key_levels")
        if isinstance(levels_value, dict):
            key_levels = {self._humanise_key(k): v for k, v in levels_value.items()}
        elif isinstance(levels_value, list):
            key_levels = {str(idx + 1): item for idx, item in enumerate(levels_value)}

        next_actions: List[str] = []
        action_keys = (
            "next_actions",
            "actionable_setups",
            "trade_plan",
            "trade_ideas",
            "risk_management",
            "position_sizing",
        )
        for action_key in action_keys:
            action_value = output.get(action_key)
            if action_value:
                next_actions.extend(self._coerce_to_list_of_strings(action_value))

        if not summary_text and next_actions:
            summary_text = next_actions[0]
        if not summary_text and key_levels:
            level_key, level_value = next(iter(key_levels.items()))
            summary_text = f"{self._humanise_key(level_key)} at {self._coerce_to_sentence(level_value)}"

        return summary_text, key_levels, next_actions

    def _coerce_to_sentence(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        items = self._coerce_to_list_of_strings(value)
        return items[0] if items else ""

    def _coerce_to_list_of_strings(self, value: Any, limit: int = 6) -> List[str]:
        results: List[str] = []

        def _collect(val: Any) -> None:
            if len(results) >= limit or val is None:
                return
            if isinstance(val, str):
                text = val.strip()
                if text:
                    results.append(text)
                return
            if isinstance(val, (int, float)):
                results.append(str(val))
                return
            if isinstance(val, bool):
                results.append("Yes" if val else "No")
                return
            if isinstance(val, list):
                for item in val:
                    _collect(item)
                    if len(results) >= limit:
                        break
                return
            if isinstance(val, dict):
                for key, nested in val.items():
                    nested_results = self._coerce_to_list_of_strings(nested, limit)
                    if nested_results:
                        label = self._humanise_key(key)
                        if len(nested_results) == 1:
                            results.append(f"{label}: {nested_results[0]}")
                        else:
                            results.append(f"{label}: {', '.join(nested_results)}")
                    if len(results) >= limit:
                        break

        _collect(value)
        deduped = list(dict.fromkeys(results))
        return deduped[:limit]

    @staticmethod
    def _humanise_key(key: Any) -> str:
        if not isinstance(key, str):
            key = str(key)
        return key.replace("_", " ").strip().title()


__all__ = ["PrincipalAgent"]

