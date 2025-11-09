"""Principal agent that fuses trading strategies for the client."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from openai import APIError, OpenAI

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
        openai_model: str = "gpt-4o-mini",
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

        # Instantiate expert agents once and share across trading agents
        self.experts = {
            "momentum": momentum_agent or MomentumExpertAgent(openai_client=self.client),
            "volatility": volatility_agent or VolatilityExpertAgent(openai_client=self.client),
            "pattern": pattern_agent or PatternRecognitionExpertAgent(openai_client=self.client),
            "trend": trend_agent or TrendExpertAgent(openai_client=self.client),
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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
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
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"},
                max_tokens=1600,
            )
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

        payload: Dict[str, Any] = {
            "symbol": symbol,
            "generated_at": datetime.utcnow().isoformat(),
            "strategies": strategy_summary,
            "model": self.model,
            "usage": usage,
        }

        if include_raw:
            payload["trading_agent_outputs"] = trading_results

        return {"principal_result": payload}


__all__ = ["PrincipalAgent"]

