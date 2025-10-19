# """
# LangGraph DayTrader Agent — Production Skeleton (v1)
# ---------------------------------------------------
# This file gives you a production-grade skeleton for an intraday DayTrader agent built with LangGraph.
# It is intentionally self-contained with mock tools so you can run it immediately, then swap in your
# real market data, feature store, risk, and broker adapters.

# Key ideas:
# - Deterministic state machine (LangGraph) with explicit nodes for signal gen → proposal → risk → exec → monitor → exit
# - Clean Tool interfaces (MarketDataTool, FeatureStoreTool, RiskTool, BrokerTool, BacktestTool)
# - Config-driven (bar sizes, indicators, risk)
# - Paper execution out of the box; replace with your live broker adapter later
# - Supports 1, 3, 5 minute bars; ATR- and VWAP-aware entries; SuperTrend + ADX trend gating

# How to run (quickstart):
# ------------------------
# python daytrader_langgraph.py

# What you’ll see:
# - The graph will generate synthetic minute bars, compute features, propose a trade, pass risk checks,
#   execute paper orders, and monitor an ATR-based trailing stop until exit.

# Next steps to production:
# - Replace MockMarketDataTool with your data vendor adapter (e.g., Polygon/IBKR/Tradier/Alpaca)
# - Replace MockBrokerTool with your broker (paper/live)
# - Wire in Feast for FeatureStoreTool
# - Persist state & logs (e.g., Postgres/Redis + mlflow for model/thresholds)
# - Schedule runs with Airflow/Prefect
# """

# from __future__ import annotations

# import math
# import random
# from dataclasses import dataclass, field
# from datetime import datetime, timedelta, timezone
# from typing import Dict, List, Literal, Optional, Tuple, TypedDict

# import numpy as np
# import pandas as pd

# # LangGraph core
# from langgraph.graph import StateGraph, START, END

# # -----------------------------
# # 1) Domain config & state types
# # -----------------------------

# @dataclass
# class IndicatorSettings:
#     atr_len: int = 14
#     supertrend_atr_len: int = 10
#     supertrend_mult: float = 2.0
#     adx_len: int = 14

# @dataclass
# class RiskSettings:
#     max_daily_loss_pct: float = 0.02  # account equity per day
#     max_symbol_risk_pct: float = 0.005
#     max_positions: int = 5
#     sl_atr_mult: float = 1.5
#     tp_atr_mult: float = 3.0

# @dataclass
# class StrategySettings:
#     bar_sizes: Tuple[str, ...] = ("1min", "3min", "5min")
#     min_adx_trend: float = 18.0
#     vwap_filter: bool = True
#     opening_range_minutes: int = 15
#     min_rr: float = 1.8

# @dataclass
# class AgentConfig:
#     symbols: Tuple[str, ...] = ("AAPL", "NVDA")
#     indicator: IndicatorSettings = field(default_factory=IndicatorSettings)
#     risk: RiskSettings = field(default_factory=RiskSettings)
#     strat: StrategySettings = field(default_factory=StrategySettings)
#     account_equity: float = 100_000.0
#     slippage_bps: float = 1.0

# class TradeTicket(TypedDict):
#     symbol: str
#     side: Literal["BUY", "SELL"]
#     qty: int
#     entry: float
#     stop: float
#     take_profit: float
#     rationale: str

# class Position(TypedDict):
#     symbol: str
#     side: Literal["LONG", "SHORT"]
#     qty: int
#     avg_price: float
#     stop: float
#     take_profit: float
#     opened_at: datetime

# class AgentState(TypedDict, total=False):
#     symbol: str
#     now: datetime
#     bars: pd.DataFrame            # minute bars with features
#     features: pd.DataFrame        # same index; feature columns
#     signal: Optional[str]         # "LONG", "SHORT", or None
#     proposal: Optional[TradeTicket]
#     risk_ok: bool
#     position: Optional[Position]
#     exit_reason: Optional[str]
#     # NEW: allow passing your external JSON payload or file path
#     json_payload: Optional[dict]
#     json_path: Optional[str]
#     logs: List[str]

# # -----------------------------
# # 2) Tool Interfaces (mock impls)
# # -----------------------------

# class MarketDataTool:
#     def get_minute_bars(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
#         raise NotImplementedError

# class FeatureStoreTool:
#     def write_features(self, symbol: str, df: pd.DataFrame) -> None:
#         pass

# class JsonFeatureTool(FeatureStoreTool):
#     """Helper to ingest precomputed multi-timeframe JSON (no indicator calc needed).
#     Expected schema (simplified):
#     {
#       "symbol": "TICKER",
#       "analysis_time": "YYYY-mm-dd HH:MM:SS",
#       "timeframes": {
#         "2m": {"price": float, "summary": {"ADX": float, "RSI": float},
#                 "details": {"basic": {"ATR": {"value": float}},
#                               "trading_signals": {"risk_reward": {"stop_loss": float, "take_profit": float}}}},
#         ...
#       },
#       "consensus": {"timeframe_recommendations": {"2m": "SELL"|"BUY"|"HOLD"},
#                      "high_confidence_timeframes": ["2m","5m",...]}
#     }
#     Returns a single-row DataFrame at analysis_time with columns:
#     close, ATR, ADX, RSI, signal, confidence, SL, TP, timeframe
#     """
#     def read_json(self, payload: Optional[dict] = None, path: Optional[str] = None) -> pd.DataFrame:
#         import json, datetime as _dt
#         if payload is None and path is None:
#             return pd.DataFrame()
#         if payload is None:
#             with open(path, "r") as f:
#                 payload = json.load(f)
#         tfs = payload.get("timeframes", {})
#         if not tfs:
#             return pd.DataFrame()
#         # Preferred timeframe ordering for day trading
#         order = ["1m","2m","3m","5m","10m","15m","30m","45m","1h"]
#         conf_list = payload.get("consensus", {}).get("high_confidence_timeframes", []) or []
#         # pick first available in conf_list following order, else fallback to first available by order
#         candidates = [tf for tf in order if tf in conf_list and tf in tfs]
#         if not candidates:
#             candidates = [tf for tf in order if tf in tfs]
#         tf = candidates[0]
#         node = tfs[tf]
#         # Extract fields safely
#         price = node.get("price")
#         sum_ = node.get("summary", {})
#         det = node.get("details", {})
#         basic = det.get("basic", {})
#         atr_val = (basic.get("ATR", {}) or {}).get("value")
#         adx_val = sum_.get("ADX") or (basic.get("ADX", {}) or {}).get("adx")
#         rsi_val = sum_.get("RSI") or (basic.get("RSI", {}) or {}).get("value")
#         conf = sum_.get("confidence")
#         rr = (det.get("trading_signals", {}) or {}).get("risk_reward", {})
#         sl = rr.get("stop_loss")
#         tp = rr.get("take_profit")
#         # Determine signal from consensus timeframe recommendation; map to LONG/SHORT
#         tf_recs = (payload.get("consensus", {}) or {}).get("timeframe_recommendations", {})
#         rec_raw = tf_recs.get(tf)
#         sig_map = {"SELL": "SHORT", "BUY": "LONG", "HOLD": None, None: None}
#         sig = sig_map.get(rec_raw, None)
#         # analysis_time index
#         at = payload.get("analysis_time")
#         try:
#             ts = pd.to_datetime(at, utc=True)
#         except Exception:
#             ts = pd.Timestamp.utcnow()
#         df = pd.DataFrame({
#             "close": [price],
#             "ATR": [atr_val],
#             "ADX": [adx_val],
#             "RSI": [rsi_val],
#             "signal": [sig],
#             "confidence": [conf],
#             "SL": [sl],
#             "TP": [tp],
#             "timeframe": [tf],
#         }, index=[ts])
#         return df

# class RiskTool:
#     def check(self, cfg: AgentConfig, ticket: TradeTicket, day_pnl_pct: float, open_positions: int) -> Tuple[bool, str]:
#         # Basic, synchronous risk checks
#         if day_pnl_pct <= -cfg.risk.max_daily_loss_pct:
#             return False, "Daily loss limit hit"
#         if open_positions >= cfg.risk.max_positions:
#             return False, "Too many open positions"
#         # Per-symbol risk budget
#         risk_per_symbol = cfg.account_equity * cfg.risk.max_symbol_risk_pct
#         stop_risk = abs(ticket.entry - ticket.stop) * ticket.qty
#         if stop_risk > risk_per_symbol:
#             return False, f"Ticket risk {stop_risk:.2f} exceeds per-symbol limit {risk_per_symbol:.2f}"
#         return True, "OK"

# class BrokerTool:
#     def __init__(self):
#         self.positions: Dict[str, Position] = {}
#         self.realized_pnl: float = 0.0
#         self.today_start = datetime.now(timezone.utc).date()

#     def submit(self, ticket: TradeTicket) -> Position:
#         side = "LONG" if ticket["side"] == "BUY" else "SHORT"
#         pos: Position = {
#             "symbol": ticket["symbol"],
#             "side": side,
#             "qty": ticket["qty"],
#             "avg_price": ticket["entry"],
#             "stop": ticket["stop"],
#             "take_profit": ticket["take_profit"],
#             "opened_at": datetime.utcnow(),
#         }
#         self.positions[ticket["symbol"]] = pos
#         return pos

#     def mark_to_market(self, prices: Dict[str, float]) -> float:
#         # Returns today PnL (unrealized + realized)
#         pnl = self.realized_pnl
#         for sym, pos in self.positions.items():
#             if sym in prices:
#                 px = prices[sym]
#                 mult = 1 if pos["side"] == "LONG" else -1
#                 pnl += (px - pos["avg_price"]) * pos["qty"] * mult
#         return pnl

#     def flatten(self, symbol: str, exit_price: float) -> None:
#         if symbol not in self.positions:
#             return
#         pos = self.positions.pop(symbol)
#         mult = 1 if pos["side"] == "LONG" else -1
#         self.realized_pnl += (exit_price - pos["avg_price"]) * pos["qty"] * mult

# class BacktestTool:
#     pass

# # Mock data: geometric random walk with intraday drift & volatility
# class MockMarketDataTool(MarketDataTool):
#     def get_minute_bars(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
#         n = int((end - start).total_seconds() // 60)
#         if n <= 0:
#             return pd.DataFrame()
#         ts = pd.date_range(start=start, periods=n, freq="T")
#         px0 = 100 + hash(symbol) % 50
#         rets = np.random.normal(loc=0.0001, scale=0.0025, size=n)
#         prices = px0 * np.exp(np.cumsum(rets))
#         highs = prices * (1 + np.random.rand(n) * 0.0015)
#         lows = prices * (1 - np.random.rand(n) * 0.0015)
#         opens = prices * (1 + np.random.randn(n) * 0.0005)
#         closes = prices
#         vols = np.random.randint(1_000, 8_000, size=n)
#         df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols}, index=ts)
#         return df

# # -----------------------------
# # 3) Feature engineering helpers
# # -----------------------------

# def ema(series: pd.Series, n: int) -> pd.Series:
#     return series.ewm(span=n, adjust=False).mean()

# def true_range(df: pd.DataFrame) -> pd.Series:
#     prev_close = df["close"].shift(1)
#     tr = pd.concat([
#         df["high"] - df["low"],
#         (df["high"] - prev_close).abs(),
#         (df["low"] - prev_close).abs(),
#     ], axis=1).max(axis=1)
#     return tr

# def atr(df: pd.DataFrame, n: int) -> pd.Series:
#     return true_range(df).rolling(n).mean()

# def vwap(df: pd.DataFrame) -> pd.Series:
#     pv = (df["close"] * df["volume"]).cumsum()
#     vv = df["volume"].cumsum()
#     return pv / vv

# def adx(df: pd.DataFrame, n: int) -> pd.Series:
#     up_move = df["high"].diff()
#     down_move = -df["low"].diff()
#     plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
#     minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
#     tr = true_range(df)
#     plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(n).sum() / tr.rolling(n).sum()
#     minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(n).sum() / tr.rolling(n).sum()
#     dx = ( (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) ) * 100
#     return dx.rolling(n).mean()

# def supertrend(df: pd.DataFrame, n: int, m: float) -> pd.Series:
#     # Simple SuperTrend channel-based trend filter (not full implementation w/ flips)
#     mid = (df["high"] + df["low"]) / 2
#     band = atr(df, n) * m
#     st = pd.Series(np.where(df["close"] >= mid, mid - band, mid + band), index=df.index)
#     return st

# # -----------------------------
# # 4) Node implementations
# # -----------------------------

# # NEW: helper to merge external features into state
# def attach_external_features(state: AgentState, json_tool: Optional[JsonFeatureTool]) -> AgentState:
#     if json_tool is None:
#         return {}
#     payload = state.get("json_payload")
#     path = state.get("json_path")
#     df = json_tool.read_json(payload=payload, path=path)
#     if df.empty:
#         return {}
#     logs = state.get("logs", []) + [f"Loaded external features from {'payload' if payload else path}"]
#     # Create a minimal bars frame if not present (use close; synthesize OHL)
#     bars = state.get("bars")
#     if bars is None or bars.empty:
#         close = df["close"].fillna(method="ffill")
#         bars = pd.DataFrame({
#             "open": close.shift(1).fillna(close),
#             "high": close.rolling(3, min_periods=1).max(),
#             "low": close.rolling(3, min_periods=1).min(),
#             "close": close,
#             "volume": 0,
#         }, index=df.index)
#     # Prefer externally supplied features
#     features = df[[c for c in ["ATR", "ADX", "RSI", "SL", "TP"] if c in df.columns]].copy()
#     # Bring over external top-line signal if present
#     ext_signal = df.iloc[-1].get("signal") if not df.empty else None
#     out: AgentState = {"bars": bars, "features": features, "signal": ext_signal, "logs": logs}
#     return out

# class Nodes:
#     def __init__(self, cfg: AgentConfig, md: MarketDataTool, fs: FeatureStoreTool, risk: RiskTool, broker: BrokerTool):
#         self.cfg = cfg
#         self.md = md
#         self.fs = fs
#         self.risk = risk
#         self.broker = broker

#     # START → warmup
#     def warmup(self, state: AgentState) -> AgentState:
#         symbol = state.get("symbol") or self.cfg.symbols[0]
#         now = datetime.utcnow().replace(second=0, microsecond=0)
#         start = now - timedelta(minutes=240)
#         bars = self.md.get_minute_bars(symbol, start, now)
#         logs = state.get("logs", []) + [f"Warmup fetched {len(bars)} bars for {symbol} up to {now:%H:%M}"]
#         # If user passed external JSON, attach it now (overrides computed features later)
#         ext = attach_external_features(state, json_tool=self.fs if isinstance(self.fs, JsonFeatureTool) else None)
#         merged = {"symbol": symbol, "now": now, "bars": bars, "logs": logs}
#         merged.update(ext)
#         return merged

#     # warmup → features
#     def build_features(self, state: AgentState) -> AgentState:
#         # If external features already exist, skip computation
#         if state.get("features") is not None and not state["features"].empty:
#             self.fs.write_features(state["symbol"], state["features"])  # no-op for JsonFeatureTool
#             logs = state.get("logs", []) + ["Using external features from JSON; skipping computation"]
#             return {"logs": logs}
#         df = state["bars"].copy()
#         ind = self.cfg.indicator
#         df["ATR"] = atr(df, ind.atr_len)
#         df["VWAP"] = vwap(df)
#         df["ADX"] = adx(df, ind.adx_len)
#         df["ST"] = supertrend(df, ind.supertrend_atr_len, ind.supertrend_mult)
#         feats = df[["ATR", "VWAP", "ADX", "ST"]]
#         self.fs.write_features(state["symbol"], feats)
#         logs = state.get("logs", []) + ["Features built: ATR/VWAP/ADX/SuperTrend"]
#         return {"features": feats, "logs": logs}

#     # features → signal
#     def generate_signal(self, state: AgentState) -> AgentState:
#         df = state["bars"].join(state["features"]).dropna()
#         if df.empty:
#             return {"signal": None, "logs": state.get("logs", []) + ["No features to signal on"]}
#         row = df.iloc[-1]
#         # If external signal was supplied, honor it; else derive
#         sig = state.get("signal")
#         if sig is None:
#             trend_ok = row["ADX"] >= self.cfg.strat.min_adx_trend
#             vwap_ok = (row["close"] > row["VWAP"]) if self.cfg.strat.vwap_filter else True
#             if trend_ok and vwap_ok and row["close"] > row["ST"]:
#                 sig = "LONG"
#             elif trend_ok and (not vwap_ok) and row["close"] < row["ST"]:
#                 sig = "SHORT"
#         logs = state.get("logs", []) + [f"Signal: {sig} (derived={state.get('signal') is None})"]
#         return {"signal": sig, "logs": logs}

#     # signal → proposal
#     def propose_trade(self, state: AgentState) -> AgentState:
#         if not state.get("signal"):
#             return {"proposal": None, "logs": state.get("logs", []) + ["No signal, no proposal"]}
#         df = state["bars"].join(state["features"]).dropna(how="all")
#         row = df.iloc[-1]
#         sym = state["symbol"]
#         # Use provided price and SL/TP if available
#         entry = float(row.get("close", np.nan))
#         if math.isnan(entry):
#             return {"proposal": None, "logs": state.get("logs", []) + ["Entry price missing"]}
#         atr_val = row.get("ATR", np.nan)
#         sl_ext = row.get("SL", np.nan)
#         tp_ext = row.get("TP", np.nan)
#         side = "BUY" if state["signal"] == "LONG" else "SELL"
#         # Apply small slippage
#         entry = entry * (1 + (self.cfg.slippage_bps/1e4) * (1 if side=="BUY" else -1))
#         if not math.isnan(sl_ext) and not math.isnan(tp_ext):
#             stop = float(sl_ext)
#             take_profit = float(tp_ext)
#         else:
#             # Fallback to ATR-based stops if not provided
#             if math.isnan(atr_val) or atr_val <= 0:
#                 return {"proposal": None, "logs": state.get("logs", []) + ["No SL/TP and ATR invalid; cannot size"]}
#             stop = entry - self.cfg.risk.sl_atr_mult * atr_val if side == "BUY" else entry + self.cfg.risk.sl_atr_mult * atr_val
#             take_profit = entry + self.cfg.risk.tp_atr_mult * atr_val if side == "BUY" else entry - self.cfg.risk.tp_atr_mult * atr_val
#         # Position sizing: risk per symbol / (entry-stop)
#         risk_per_symbol = self.cfg.account_equity * self.cfg.risk.max_symbol_risk_pct
#         unit_risk = abs(entry - stop)
#         qty = max(0, int(risk_per_symbol / max(unit_risk, 1e-6)))
#         if qty == 0:
#             return {"proposal": None, "logs": state.get("logs", []) + ["Qty=0 by risk sizing, skip"]}
#         rr = abs(take_profit - entry) / max(abs(entry - stop), 1e-6)
#         if rr < self.cfg.strat.min_rr:
#             return {"proposal": None, "logs": state.get("logs", []) + [f"RR {rr:.2f} < min_rr {self.cfg.strat.min_rr}"]}
#         ticket: TradeTicket = {
#             "symbol": sym,
#             "side": side,
#             "qty": qty,
#             "entry": round(entry, 2),
#             "stop": round(stop, 2),
#             "take_profit": round(take_profit, 2),
#             "rationale": f"External consensus-driven {state['signal']} (tf={row.get('timeframe','?')}, conf={row.get('confidence')})",
#         }
#         logs = state.get("logs", []) + [f"Proposed {ticket}"]
#         return {"proposal": ticket, "logs": logs}
#     # proposal → risk
#     def risk_check(self, state: AgentState) -> AgentState:
#         if not state.get("proposal"):
#             return {"risk_ok": False, "logs": state.get("logs", []) + ["No proposal for risk check"]}
#         # Compute day PnL pct vs equity
#         last_close = float(state["bars"].iloc[-1]["close"]) if not state["bars"].empty else 0.0
#         prices = {state["symbol"]: last_close}
#         todays_pnl = self.broker.mark_to_market(prices)
#         day_pnl_pct = todays_pnl / self.cfg.account_equity
#         ok, reason = self.risk.check(self.cfg, state["proposal"], day_pnl_pct, open_positions=len(self.broker.positions))
#         logs = state.get("logs", []) + [f"Risk check: {ok} ({reason})"]
#         return {"risk_ok": ok, "logs": logs}

#     # risk → execute
#     def execute(self, state: AgentState) -> AgentState:
#         if not state.get("risk_ok"):
#             return {"position": None, "logs": state.get("logs", []) + ["Risk not approved; skip execution"]}
#         pos = self.broker.submit(state["proposal"])  # paper fill at entry
#         logs = state.get("logs", []) + [f"Executed: {pos}"]
#         return {"position": pos, "logs": logs}

#     # monitor → exit
#     def monitor_and_exit(self, state: AgentState) -> AgentState:
#         if not state.get("position"):
#             return {"exit_reason": "no_position", "logs": state.get("logs", []) + ["No position to monitor"]}
#         sym = state["symbol"]
#         pos = state["position"]
#         # Simulate next 30 minutes and exit on SL/TP or end-of-sim
#         start = state["bars"].index[-1] + timedelta(minutes=1)
#         end = start + timedelta(minutes=30)
#         future = self.md.get_minute_bars(sym, start, end)
#         if future.empty:
#             return {"exit_reason": "no_future_bars", "logs": state.get("logs", []) + ["No future bars"]}
#         exited = False
#         reason = "time"
#         for ts, row in future.iterrows():
#             px = float(row["close"]) * (1 + (self.cfg.slippage_bps/1e4))
#             if pos["side"] == "LONG":
#                 if px <= pos["stop"]:
#                     self.broker.flatten(sym, px)
#                     exited, reason = True, "stop"
#                     break
#                 if px >= pos["take_profit"]:
#                     self.broker.flatten(sym, px)
#                     exited, reason = True, "take_profit"
#                     break
#             else:  # SHORT
#                 if px >= pos["stop"]:
#                     self.broker.flatten(sym, px)
#                     exited, reason = True, "stop"
#                     break
#                 if px <= pos["take_profit"]:
#                     self.broker.flatten(sym, px)
#                     exited, reason = True, "take_profit"
#                     break
#         if not exited:
#             # flatten at last price
#             last_px = float(future.iloc[-1]["close"]) * (1 + (self.cfg.slippage_bps/1e4))
#             self.broker.flatten(sym, last_px)
#         logs = state.get("logs", []) + [f"Exited position on {reason}"]
#         return {"exit_reason": reason, "logs": logs}

# # -----------------------------
# # 5) Build the LangGraph
# # -----------------------------

# def build_graph(cfg: AgentConfig, tools: Tuple[MarketDataTool, FeatureStoreTool, RiskTool, BrokerTool]):
#     md, fs, risk, broker = tools
#     nodes = Nodes(cfg, md, fs, risk, broker)

#     graph = StateGraph(AgentState)

#     graph.add_node("warmup", nodes.warmup)
#     graph.add_node("features", nodes.build_features)
#     graph.add_node("signal", nodes.generate_signal)
#     graph.add_node("propose", nodes.propose_trade)
#     graph.add_node("risk", nodes.risk_check)
#     graph.add_node("execute", nodes.execute)
#     graph.add_node("monitor_exit", nodes.monitor_and_exit)

#     graph.add_edge(START, "warmup")
#     graph.add_edge("warmup", "features")
#     graph.add_edge("features", "signal")
#     graph.add_edge("signal", "propose")

#     # Conditional: if no proposal, end early
#     def has_proposal(state: AgentState) -> Literal["risk", "end"]:
#         return "risk" if state.get("proposal") else "end"

#     graph.add_conditional_edges("propose", has_proposal, {"risk": "risk", "end": END})

#     def risk_ok(state: AgentState) -> Literal["execute", "end"]:
#         return "execute" if state.get("risk_ok") else "end"

#     graph.add_conditional_edges("risk", risk_ok, {"execute": "execute", "end": END})

#     graph.add_edge("execute", "monitor_exit")
#     graph.add_edge("monitor_exit", END)

#     return graph.compile()

# # -----------------------------
# # 6) Demo main
# # -----------------------------

# class NoopFS(FeatureStoreTool):
#     def write_features(self, symbol: str, df: pd.DataFrame) -> None:
#         pass

# class JsonFS(JsonFeatureTool):
#     pass


# def main():
#     cfg = AgentConfig()
#     md = MockMarketDataTool()
#     fs = JsonFS()  # or NoopFS() if you prefer computed features
#     risk = RiskTool()
#     broker = BrokerTool()

#     app = build_graph(cfg, (md, fs, risk, broker))

#     # Seed state (symbol selectable)
#         # Example: pass your JSON payload (or set json_path to a file path)
#     example_payload = {
#         "symbol": "DUOL",
#         "analysis_time": "2025-10-18 16:39:42",
#         "timeframes": {
#             "2m": {
#             "price": 324.0199890136719,
#             "summary": {
#                 "bias": "strong_bearish",
#                 "strength": 84.61538461538461,
#                 "confidence": "high",
#                 "MACD": "bearish",
#                 "ADX": 35.80147327871291,
#                 "RSI": 25.98931422288567,
#                 "ATR": 1.0134880302520632,
#                 "Fib": "downtrend",
#                 "Wave": "impulse-60%"
#             },
#             "details": {
#                 "data_points": 182,
#                 "price_range": {
#                 "low": 322.0,
#                 "high": 334.2900085449219
#                 },
#                 "basic": {
#                 "MACD": {
#                     "settings": "(6,13,4)",
#                     "macd": -0.4627601973481319,
#                     "signal": -0.13449817307207088,
#                     "crossover": "bearish"
#                 },
#                 "ADX": {
#                     "period": 7,
#                     "adx": 35.80147327871291,
#                     "+DI": 16.083852243340473,
#                     "-DI": 45.095283793621235,
#                     "trend_strength": "strong"
#                 },
#                 "RSI": {
#                     "period": 7,
#                     "value": 25.98931422288567,
#                     "status": "Oversold"
#                 },
#                 "OBV": {
#                     "current": -207107.0,
#                     "trend": "bearish"
#                 },
#                 "ATR": {
#                     "period": 7,
#                     "value": 1.0134880302520632,
#                     "volatility_level": "high",
#                     "comment": "High volatility \u2013 signs of breakout potential"
#                 },
#                 "Moving_Averages": {
#                     "sma_20": 326.1007507324219,
#                     "sma_50": 325.9304626464844,
#                     "trend": "bullish"
#                 },
#                 "Bollinger_Bands": {
#                     "upper": 328.1954870364294,
#                     "middle": 326.1007507324219,
#                     "lower": 324.00601442841435,
#                     "position": {
#                     "category": "lower_half",
#                     "percentage": 0.33,
#                     "distance_from_middle": 2.0808
#                     }
#                 }
#                 },
#                 "fibonacci": {
#                 "trend_direction": "downtrend",
#                 "price_range": {
#                     "low": 322.69500732421875,
#                     "high": 329.20001220703125
#                 },
#                 "key_levels": {
#                     "38.2%": 325.17991918945313,
#                     "50%": 325.947509765625,
#                     "61.8%": 326.71510034179687
#                 },
#                 "nearest_support": {
#                     "level": "0%",
#                     "price": 322.69500732421875
#                 },
#                 "nearest_resistance": {
#                     "level": "23.6%",
#                     "price": 324.2301884765625
#                 }
#                 },
#                 "elliott_wave": {
#                 "pattern": "impulse",
#                 "trend": "bearish",
#                 "confidence": 60,
#                 "next_expectation": "Expect bounce to 445.96 - 522.12 range",
#                 "wave_points_count": 5
#                 },
#                 "trading_signals": {
#                 "overall_bias": "strong_bearish",
#                 "strength": 84.61538461538461,
#                 "confidence": "high",
#                 "signal_breakdown": {
#                     "basic_score": -6,
#                     "fibonacci_score": -10,
#                     "elliott_wave_score": -2,
#                     "total_score": -18
#                 },
#                 "entry_signals": [
#                     "RSI oversold (26.0)",
#                     "Price above moving averages"
#                 ],
#                 "exit_signals": [
#                     "MACD bearish crossover (6,13,4)",
#                     "Strong bearish trend (ADX: 35.8)",
#                     "Volume supporting downtrend (OBV)"
#                 ],
#                 "risk_reward": {
#                     "ratio": 0.1586432964046259,
#                     "risk_amount": 1.324981689453125,
#                     "reward_potential": 0.21019946289061409,
#                     "stop_loss": 322.69500732421875,
#                     "take_profit": 324.2301884765625
#                 }
#                 }
#             }
#             },
#             "5m": {
#             "price": 324.0199890136719,
#             "summary": {
#                 "bias": "strong_bearish",
#                 "strength": 100.0,
#                 "confidence": "high",
#                 "MACD": "bearish",
#                 "ADX": 27.975837991832897,
#                 "RSI": 35.210867957267205,
#                 "ATR": 1.4780040063374538,
#                 "Fib": "downtrend",
#                 "Wave": "corrective-70%"
#             },
#             "details": {
#                 "data_points": 389,
#                 "price_range": {
#                 "low": 320.2099914550781,
#                 "high": 343.99798583984375
#                 },
#                 "basic": {
#                 "MACD": {
#                     "settings": "(6,13,4)",
#                     "macd": -0.10069571640411823,
#                     "signal": 0.04488394158658615,
#                     "crossover": "bearish"
#                 },
#                 "ADX": {
#                     "period": 7,
#                     "adx": 27.975837991832897,
#                     "+DI": 24.018091599964453,
#                     "-DI": 38.072130922785306,
#                     "trend_strength": "strong"
#                 },
#                 "RSI": {
#                     "period": 7,
#                     "value": 35.210867957267205,
#                     "status": "Normal"
#                 },
#                 "OBV": {
#                     "current": -339302.0,
#                     "trend": "bearish"
#                 },
#                 "ATR": {
#                     "period": 7,
#                     "value": 1.4780040063374538,
#                     "volatility_level": "high",
#                     "comment": "High volatility \u2013 signs of breakout potential"
#                 },
#                 "Moving_Averages": {
#                     "sma_20": 325.90696868896487,
#                     "sma_50": 326.9585638427734,
#                     "trend": "bearish"
#                 },
#                 "Bollinger_Bands": {
#                     "upper": 329.1380210740416,
#                     "middle": 325.90696868896487,
#                     "lower": 322.67591630388813,
#                     "position": {
#                     "category": "lower_half",
#                     "percentage": 20.8,
#                     "distance_from_middle": 1.887
#                     }
#                 }
#                 },
#                 "fibonacci": {
#                 "trend_direction": "downtrend",
#                 "price_range": {
#                     "low": 322.69500732421875,
#                     "high": 331.80499267578125
#                 },
#                 "key_levels": {
#                     "38.2%": 326.1750217285156,
#                     "50%": 327.25,
#                     "61.8%": 328.3249782714844
#                 },
#                 "nearest_support": {
#                     "level": "0%",
#                     "price": 322.69500732421875
#                 },
#                 "nearest_resistance": {
#                     "level": "23.6%",
#                     "price": 324.8449638671875
#                 }
#                 },
#                 "elliott_wave": {
#                 "pattern": "corrective",
#                 "trend": "bullish",
#                 "confidence": 70,
#                 "next_expectation": "Corrective pattern - expect continuation of main trend after completion",
#                 "wave_points_count": 5
#                 },
#                 "trading_signals": {
#                 "overall_bias": "strong_bearish",
#                 "strength": 100.0,
#                 "confidence": "high",
#                 "signal_breakdown": {
#                     "basic_score": -12,
#                     "fibonacci_score": -10,
#                     "elliott_wave_score": -1,
#                     "total_score": -23
#                 },
#                 "entry_signals": [],
#                 "exit_signals": [
#                     "MACD bearish crossover (6,13,4)",
#                     "Strong bearish trend (ADX: 28.0)",
#                     "Volume supporting downtrend (OBV)"
#                 ],
#                 "risk_reward": {
#                     "ratio": 0.6226311352696072,
#                     "risk_amount": 1.324981689453125,
#                     "reward_potential": 0.8249748535156414,
#                     "stop_loss": 322.69500732421875,
#                     "take_profit": 324.8449638671875
#                 }
#                 }
#             }
#             },
#             "15m": {
#             "price": 324.0199890136719,
#             "summary": {
#                 "bias": "strong_bearish",
#                 "strength": 100.0,
#                 "confidence": "high",
#                 "MACD": "bearish",
#                 "ADX": 13.015420414228622,
#                 "RSI": 39.87119451416047,
#                 "ATR": 2.8334582001285207,
#                 "Fib": "downtrend",
#                 "Wave": "corrective-55%"
#             },
#             "details": {
#                 "data_points": 130,
#                 "price_range": {
#                 "low": 320.2099914550781,
#                 "high": 343.99798583984375
#                 },
#                 "basic": {
#                 "MACD": {
#                     "settings": "(8,17,6)",
#                     "macd": -0.815031271348289,
#                     "signal": -0.7024415333080436,
#                     "crossover": "bearish"
#                 },
#                 "ADX": {
#                     "period": 10,
#                     "adx": 13.015420414228622,
#                     "+DI": 18.828819289926397,
#                     "-DI": 26.578301328310815,
#                     "trend_strength": "weak"
#                 },
#                 "RSI": {
#                     "period": 10,
#                     "value": 39.87119451416047,
#                     "status": "Normal"
#                 },
#                 "OBV": {
#                     "current": -645073.0,
#                     "trend": "bearish"
#                 },
#                 "ATR": {
#                     "period": 10,
#                     "value": 2.8334582001285207,
#                     "volatility_level": "high",
#                     "comment": "High volatility \u2013 signs of breakout potential"
#                 },
#                 "Moving_Averages": {
#                     "sma_20": 326.9365676879883,
#                     "sma_50": 329.38978942871097,
#                     "trend": "bearish"
#                 },
#                 "Bollinger_Bands": {
#                     "upper": 329.7826823245799,
#                     "middle": 326.9365676879883,
#                     "lower": 324.0904530513967,
#                     "position": {
#                     "category": "below_lower",
#                     "percentage": -1.24,
#                     "distance_from_middle": 2.9166
#                     }
#                 }
#                 },
#                 "fibonacci": {
#                 "trend_direction": "downtrend",
#                 "price_range": {
#                     "low": 320.2099914550781,
#                     "high": 343.99798583984375
#                 },
#                 "key_levels": {
#                     "38.2%": 329.2970053100586,
#                     "50%": 332.10398864746094,
#                     "61.8%": 334.9109719848633
#                 },
#                 "nearest_support": {
#                     "level": "0%",
#                     "price": 320.2099914550781
#                 },
#                 "nearest_resistance": {
#                     "level": "23.6%",
#                     "price": 325.82395812988284
#                 }
#                 },
#                 "elliott_wave": {
#                 "pattern": "corrective",
#                 "trend": "bullish",
#                 "confidence": 55,
#                 "next_expectation": "Corrective pattern - expect continuation of main trend after completion",
#                 "wave_points_count": 5
#                 },
#                 "trading_signals": {
#                 "overall_bias": "strong_bearish",
#                 "strength": 100.0,
#                 "confidence": "high",
#                 "signal_breakdown": {
#                     "basic_score": -8,
#                     "fibonacci_score": -1,
#                     "elliott_wave_score": -1,
#                     "total_score": -10
#                 },
#                 "entry_signals": [],
#                 "exit_signals": [
#                     "MACD bearish crossover (8,17,6)",
#                     "Volume supporting downtrend (OBV)",
#                     "Price below moving averages"
#                 ],
#                 "risk_reward": {
#                     "ratio": 0.47348301106964497,
#                     "risk_amount": 3.80999755859375,
#                     "reward_potential": 1.8039691162109648,
#                     "stop_loss": 320.2099914550781,
#                     "take_profit": 325.82395812988284
#                 }
#                 }
#             }
#             },
#             "30m": {
#             "price": 324.0199890136719,
#             "summary": {
#                 "bias": "strong_bearish",
#                 "strength": 100.0,
#                 "confidence": "high",
#                 "MACD": "bearish",
#                 "ADX": 36.6900135136129,
#                 "RSI": 37.7232990298783,
#                 "ATR": 4.4042110712844735,
#                 "Fib": "downtrend",
#                 "Wave": None
#             },
#             "details": {
#                 "data_points": 65,
#                 "price_range": {
#                 "low": 320.2099914550781,
#                 "high": 343.99798583984375
#                 },
#                 "basic": {
#                 "MACD": {
#                     "settings": "(8,17,6)",
#                     "macd": -1.5914949497305315,
#                     "signal": -1.5442690528123681,
#                     "crossover": "bearish"
#                 },
#                 "ADX": {
#                     "period": 10,
#                     "adx": 36.6900135136129,
#                     "+DI": 10.833750966927957,
#                     "-DI": 20.883036526020604,
#                     "trend_strength": "strong"
#                 },
#                 "RSI": {
#                     "period": 10,
#                     "value": 37.7232990298783,
#                     "status": "Normal"
#                 },
#                 "OBV": {
#                     "current": -758137.0,
#                     "trend": "bearish"
#                 },
#                 "ATR": {
#                     "period": 10,
#                     "value": 4.4042110712844735,
#                     "volatility_level": "high",
#                     "comment": "High volatility \u2013 signs of breakout potential"
#                 },
#                 "Moving_Averages": {
#                     "sma_20": 327.4089981079102,
#                     "sma_50": 332.01359272003174,
#                     "trend": "bearish"
#                 },
#                 "Bollinger_Bands": {
#                     "upper": 332.3712134756202,
#                     "middle": 327.4089981079102,
#                     "lower": 322.44678274020015,
#                     "position": {
#                     "category": "lower_half",
#                     "percentage": 15.85,
#                     "distance_from_middle": 3.389
#                     }
#                 }
#                 },
#                 "fibonacci": {
#                 "trend_direction": "downtrend",
#                 "price_range": {
#                     "low": 320.2099914550781,
#                     "high": 343.99798583984375
#                 },
#                 "key_levels": {
#                     "38.2%": 329.2970053100586,
#                     "50%": 332.10398864746094,
#                     "61.8%": 334.9109719848633
#                 },
#                 "nearest_support": {
#                     "level": "0%",
#                     "price": 320.2099914550781
#                 },
#                 "nearest_resistance": {
#                     "level": "23.6%",
#                     "price": 325.82395812988284
#                 }
#                 },
#                 "elliott_wave": {
#                 "pattern": "insufficient_data",
#                 "trend": None,
#                 "confidence": 0,
#                 "next_expectation": None,
#                 "wave_points_count": 0
#                 },
#                 "trading_signals": {
#                 "overall_bias": "strong_bearish",
#                 "strength": 100.0,
#                 "confidence": "high",
#                 "signal_breakdown": {
#                     "basic_score": -12,
#                     "fibonacci_score": -1,
#                     "elliott_wave_score": 0,
#                     "total_score": -13
#                 },
#                 "entry_signals": [],
#                 "exit_signals": [
#                     "MACD bearish crossover (8,17,6)",
#                     "Strong bearish trend (ADX: 36.7)",
#                     "Volume supporting downtrend (OBV)"
#                 ],
#                 "risk_reward": {
#                     "ratio": 0.47348301106964497,
#                     "risk_amount": 3.80999755859375,
#                     "reward_potential": 1.8039691162109648,
#                     "stop_loss": 320.2099914550781,
#                     "take_profit": 325.82395812988284
#                 }
#                 }
#             }
#             },
#             "1h": {
#             "price": 324.0199890136719,
#             "summary": {
#                 "bias": "strong_bearish",
#                 "strength": 92.3076923076923,
#                 "confidence": "high",
#                 "MACD": "bearish",
#                 "ADX": 26.72987532768984,
#                 "RSI": 39.2281613813936,
#                 "ATR": 5.877106454213438,
#                 "Fib": "uptrend",
#                 "Wave": "corrective-30%"
#             },
#             "details": {
#                 "data_points": 154,
#                 "price_range": {
#                 "low": 281.8349914550781,
#                 "high": 353.0
#                 },
#                 "basic": {
#                 "MACD": {
#                     "settings": "(12,26,9)",
#                     "macd": -2.1457558159566474,
#                     "signal": -1.1928528213174427,
#                     "crossover": "bearish"
#                 },
#                 "ADX": {
#                     "period": 14,
#                     "adx": 26.72987532768984,
#                     "+DI": 9.825201688075943,
#                     "-DI": 24.191395775029825,
#                     "trend_strength": "strong"
#                 },
#                 "RSI": {
#                     "period": 14,
#                     "value": 39.2281613813936,
#                     "status": "Normal"
#                 },
#                 "OBV": {
#                     "current": 1406709.0,
#                     "trend": "bearish"
#                 },
#                 "ATR": {
#                     "period": 14,
#                     "value": 5.877106454213438,
#                     "volatility_level": "high",
#                     "comment": "High volatility \u2013 signs of breakout potential"
#                 },
#                 "Moving_Averages": {
#                     "sma_20": 332.8631240844727,
#                     "sma_50": 333.10045471191404,
#                     "trend": "bearish"
#                 },
#                 "Bollinger_Bands": {
#                     "upper": 346.2109013600337,
#                     "middle": 332.8631240844727,
#                     "lower": 319.5153468089117,
#                     "position": {
#                     "category": "lower_half",
#                     "percentage": 16.87,
#                     "distance_from_middle": 8.8431
#                     }
#                 }
#                 },
#                 "fibonacci": {
#                 "trend_direction": "uptrend",
#                 "price_range": {
#                     "low": 318.57501220703125,
#                     "high": 353.0
#                 },
#                 "key_levels": {
#                     "38.2%": 339.8496546630859,
#                     "50%": 335.7875061035156,
#                     "61.8%": 331.72535754394534
#                 },
#                 "nearest_support": {
#                     "level": "100%",
#                     "price": 318.57501220703125
#                 },
#                 "nearest_resistance": {
#                     "level": "78.6%",
#                     "price": 325.94195959472654
#                 }
#                 },
#                 "elliott_wave": {
#                 "pattern": "corrective",
#                 "trend": "bullish",
#                 "confidence": 30,
#                 "next_expectation": "Corrective pattern - expect continuation of main trend after completion",
#                 "wave_points_count": 5
#                 },
#                 "trading_signals": {
#                 "overall_bias": "strong_bearish",
#                 "strength": 92.3076923076923,
#                 "confidence": "high",
#                 "signal_breakdown": {
#                     "basic_score": -12,
#                     "fibonacci_score": 1,
#                     "elliott_wave_score": 0,
#                     "total_score": -11
#                 },
#                 "entry_signals": [
#                     "Fibonacci analysis shows uptrend structure"
#                 ],
#                 "exit_signals": [
#                     "MACD bearish crossover (12,26,9)",
#                     "Strong bearish trend (ADX: 26.7)",
#                     "Volume supporting downtrend (OBV)"
#                 ],
#                 "risk_reward": {
#                     "ratio": 0.3529804899647423,
#                     "risk_amount": 5.444976806640625,
#                     "reward_potential": 1.9219705810546657,
#                     "stop_loss": 318.57501220703125,
#                     "take_profit": 325.94195959472654
#                 }
#                 }
#             }
#             }
#         },
#         "consensus": {
#             "overall_recommendation": "SELL",
#             "consensus_confidence": "High",
#             "agreement_pct": 100.0,
#             "breakdown": {
#             "buy_count": 0,
#             "sell_count": 5,
#             "hold_count": 0,
#             "buy_pct": 0.0,
#             "sell_pct": 100.0,
#             "hold_pct": 0.0
#             },
#             "timeframe_recommendations": {
#             "2m": "SELL",
#             "5m": "SELL",
#             "15m": "SELL",
#             "30m": "SELL",
#             "1h": "SELL"
#             },
#             "avg_strength": 95.38461538461539,
#             "high_confidence_timeframes": [
#             "2m",
#             "5m",
#             "15m",
#             "30m",
#             "1h"
#             ]
#         }
#     }

#     init: AgentState = {"symbol": cfg.symbols[0], "logs": [], "json_payload": example_payload}
#     out: AgentState = app.invoke(init)

#     print("\n=== DayTrader Agent Run ===")
#     for line in out.get("logs", []):
#         print(line)
#     print("\nPositions left:", broker.positions)
#     print("Realized PnL:", round(broker.realized_pnl, 2))


# if __name__ == "__main__":
#     main()
 