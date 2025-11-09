"""Trading strategy agents that orchestrate expert insights."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from .expert_agent import (
    BaseExpertAgent,
    MomentumExpertAgent,
    PatternRecognitionExpertAgent,
    TrendExpertAgent,
    VolatilityExpertAgent,
)


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

    def run(self, symbol: str) -> Dict[str, Any]:
        plan = self._expert_plan()
        expert_payloads: Dict[str, Any] = {}
        for task in plan:
            key = task["key"]
            agent = self.expert_registry.get(key)
            if not agent:
                continue
            kwargs = {
                "timeframes": task.get("timeframes"),
                "period_overrides": task.get("period_overrides"),
            }
            expert_payloads[key] = agent.run(symbol, **kwargs)

        return {
            "success": True,
            "strategy": self.strategy_name,
            "symbol": symbol.upper(),
            "collected_at": datetime.utcnow().isoformat(),
            "experts": expert_payloads,
        }

    # ------------------------------------------------------------------
    # Customisation points
    # ------------------------------------------------------------------
    def _expert_plan(self) -> List[Dict[str, Any]]:
        """Return plan describing which expert agents to consult."""

        raise NotImplementedError


class DayTradingAgent(BaseTradingAgent):
    """Intraday-focused trading agent."""

    strategy_name = "day_trading"

    def _expert_plan(self) -> List[Dict[str, Any]]:
        short_term = ["1m", "5m", "15m"]
        return [
            {"key": "momentum", "timeframes": short_term, "period_overrides": {"1m": "1d"}},
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
