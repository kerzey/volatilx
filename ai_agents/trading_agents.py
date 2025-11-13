"""Trading strategy agents that orchestrate expert insights."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from .expert_agent import (
    BaseExpertAgent,
    MomentumExpertAgent,
    PatternRecognitionExpertAgent,
    TrendExpertAgent,
    VolatilityExpertAgent,
)


logger = logging.getLogger(__name__)


class TradingAgentState(TypedDict, total=False):
    """State container passed between LangGraph nodes."""

    symbol: str
    expert_outputs: Dict[str, Any]
    final_result: Dict[str, Any]


class BaseTradingAgent:
    """Collects domain expert outputs for a specific trading horizon."""

    strategy_name: str = "base"

    def __init__(
        self,
        *,
        momentum_agent: Optional[MomentumExpertAgent] = None,
        volatility_agent: Optional[VolatilityExpertAgent] = None,
        pattern_agent: Optional[PatternRecognitionExpertAgent] = None,
        trend_agent: Optional[TrendExpertAgent] = None,
    ) -> None:
        self.expert_registry: Dict[str, BaseExpertAgent] = {}
        if momentum_agent:
            self.expert_registry["momentum"] = momentum_agent
        if volatility_agent:
            self.expert_registry["volatility"] = volatility_agent
        if pattern_agent:
            self.expert_registry["pattern"] = pattern_agent
        if trend_agent:
            self.expert_registry["trend"] = trend_agent

        self._plan_template: List[Dict[str, Any]] = []
        self.refresh_graph()

    def run(self, symbol: str) -> Dict[str, Any]:
        self.refresh_graph()
        initial_state: TradingAgentState = {
            "symbol": symbol.upper(),
            "expert_outputs": {},
        }

        final_state = self.graph.invoke(
            initial_state,
            config={
                "metadata": {
                    "strategy": self.strategy_name,
                    "symbol": symbol.upper(),
                }
            },
        )

        result = final_state.get("final_result")
        if not isinstance(result, dict):
            raise RuntimeError(f"{self.strategy_name} agent did not produce a result")
        return result

    def refresh_graph(self) -> None:
        """Rebuild the LangGraph pipeline (call after modifying expert plan)."""

        plan = self._expert_plan() or []
        self._plan_template = [dict(task) for task in plan]
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Customisation points
    # ------------------------------------------------------------------
    def _expert_plan(self) -> List[Dict[str, Any]]:
        """Return plan describing which expert agents to consult."""

        raise NotImplementedError

    def _build_graph(self):
        builder = StateGraph(TradingAgentState)
        builder.add_node("initialise", self._initialise_state)
        builder.add_edge(START, "initialise")

        previous = "initialise"
        for index, task in enumerate(self._plan_template):
            node_name = f"expert_{index}_{task.get('key', 'unknown')}"
            builder.add_node(node_name, self._make_expert_node(task))
            builder.add_edge(previous, node_name)
            previous = node_name

        builder.add_node("assemble", self._assemble_result)
        builder.add_edge(previous, "assemble")
        builder.add_edge("assemble", END)
        return builder.compile()

    def _initialise_state(self, state: TradingAgentState) -> TradingAgentState:
        outputs = dict(state.get("expert_outputs", {}))
        symbol = state.get("symbol")
        if symbol is None:
            raise RuntimeError("Trading agent initial state missing symbol")
        return {"symbol": symbol, "expert_outputs": outputs}

    def _make_expert_node(self, task: Dict[str, Any]):
        key = task.get("key")
        timeframes = task.get("timeframes")
        period_overrides = task.get("period_overrides")

        def _node(state: TradingAgentState) -> TradingAgentState:
            if not key:
                return {}

            agent = self.expert_registry.get(key)
            if not agent:
                logger.warning(
                    "No expert agent registered for key '%s' in strategy '%s'",
                    key,
                    self.strategy_name,
                )
                symbol = state.get("symbol")
                return {"symbol": symbol} if symbol else {}

            kwargs: Dict[str, Any] = {}
            if timeframes:
                kwargs["timeframes"] = timeframes
            if period_overrides:
                kwargs["period_overrides"] = period_overrides

            symbol = state.get("symbol")
            if symbol is None:
                raise RuntimeError(
                    f"Trading agent '{self.strategy_name}' missing symbol before expert '{key}'"
                )
            expert_result = agent.run(symbol, **kwargs)
            outputs = dict(state.get("expert_outputs", {}))
            outputs[key] = expert_result
            return {"symbol": symbol, "expert_outputs": outputs}

        return _node

    def _assemble_result(self, state: TradingAgentState) -> TradingAgentState:
        symbol = state.get("symbol")
        if symbol is None:
            raise RuntimeError("Trading agent state missing symbol during assembly")
        outputs = state.get("expert_outputs", {})
        result = {
            "success": True,
            "strategy": self.strategy_name,
            "symbol": symbol,
            "collected_at": datetime.utcnow().isoformat(),
            "experts": outputs,
        }
        return {"final_result": result}


class DayTradingAgent(BaseTradingAgent):
    """Intraday-focused trading agent."""

    strategy_name = "day_trading"

    def _expert_plan(self) -> List[Dict[str, Any]]:
        short_term = ["5m", "15m", "30m"]
        return [
            {"key": "momentum", "timeframes": short_term, "period_overrides": {"5m": "1d"}},
            {"key": "volatility", "timeframes": ["1m", "5m", "15m", "1h"]},
            {"key": "trend", "timeframes": ["5m", "15m", "1h"]},
        ]


class SwingTradingAgent(BaseTradingAgent):
    """Swing horizon agent focusing on multi-day to multi-week moves."""

    strategy_name = "swing_trading"

    def _expert_plan(self) -> List[Dict[str, Any]]:
        return [
            {"key": "momentum", "timeframes": ["15m", "1h", "1d"]},
            {"key": "trend", "timeframes": ["1h", "1d", "1wk"]},
            {"key": "pattern", "timeframes": ["1h", "1d", "1wk"]},
            {"key": "volatility", "timeframes": ["1h", "1d"]},
        ]


class LongTermTradingAgent(BaseTradingAgent):
    """Position trading agent targeting higher timeframe plays."""

    strategy_name = "longterm_trading"

    def _expert_plan(self) -> List[Dict[str, Any]]:
        return [
            {"key": "trend", "timeframes": ["1d", "1wk", "1mo"]},
            {"key": "pattern", "timeframes": ["1d", "1wk", "1mo"]},
            {"key": "volatility", "timeframes": ["1d", "1wk"]},
        ]


__all__ = [
    "BaseTradingAgent",
    "DayTradingAgent",
    "SwingTradingAgent",
    "LongTermTradingAgent",
]
