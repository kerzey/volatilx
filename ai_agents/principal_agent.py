"""Principal agent orchestration built on GPT-5 Responses API."""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, TypedDict

from langgraph.graph import END, START, StateGraph
from openai import APIError, OpenAI

from core.numeric_utils import safe_float
from .expert_agent import (
    OpenAIResponsesMixin,
    PriceActionSummaryAgent,
    TechnicalAnalysisSummaryAgent,
    UIBackedExpertAgent,
    _extract_output_text,
    _extract_usage_details,
)


logger = logging.getLogger(__name__)


class PrincipalAgentState(TypedDict, total=False):
    """State container passed between LangGraph nodes."""

    symbol: str
    include_raw_results: bool
    raw_inputs: Dict[str, Any]
    expert_outputs: Dict[str, Any]
    principal_result: Dict[str, Any]


def _json_dump(data: Any) -> str:
    return json.dumps(data, indent=2, default=str)


def _json_default_encoder(value: Any) -> Any:
    if isinstance(value, (set, tuple)):
        return list(value)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:  # noqa: BLE001 - fall back to string conversion
            return str(value)
    return str(value)


def _safe_payload_copy(payload: Any) -> Any:
    try:
        return json.loads(json.dumps(payload, default=_json_default_encoder))
    except (TypeError, ValueError):
        return payload


class PrincipalAgent(OpenAIResponsesMixin):
    """High-level orchestrator that fuses expert signals into trading plans."""

    def __init__(
        self,
        *,
        openai_client: Optional[OpenAI] = None,
        openai_model: str = "gpt-5",
        reasoning_effort: str = "low",
        text_verbosity: str = "medium",
        max_output_tokens: int = 2000,
        expert_registry: Optional[Dict[str, UIBackedExpertAgent]] = None,
        technical_agent: Optional[TechnicalAnalysisSummaryAgent] = None,
        price_action_agent: Optional[PriceActionSummaryAgent] = None,
    ) -> None:
        self.client = self._init_openai_client(openai_client)
        self.model = openai_model
        self.reasoning_effort = reasoning_effort
        self.text_verbosity = text_verbosity
        self.max_output_tokens = max_output_tokens

        registry: Dict[str, UIBackedExpertAgent] = {}
        if technical_agent is None:
            technical_agent = TechnicalAnalysisSummaryAgent(
                openai_client=self.client,
                model="gpt-5-nano",
            )
        if price_action_agent is None:
            price_action_agent = PriceActionSummaryAgent(
                openai_client=self.client,
                model="gpt-5-nano",
            )

        registry["technical"] = technical_agent
        registry["price_action"] = price_action_agent

        if expert_registry:
            registry.update(expert_registry)

        self.experts: Dict[str, UIBackedExpertAgent] = registry
        self.refresh_graph()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_trading_plan(
        self,
        symbol: str,
        *,
        technical_snapshot: Optional[Dict[str, Any]] = None,
        price_action_snapshot: Optional[Dict[str, Any]] = None,
        extra_inputs: Optional[Dict[str, Any]] = None,
        include_raw_results: bool = True,
    ) -> Dict[str, Any]:
        if (
            technical_snapshot is None
            and price_action_snapshot is None
            and not extra_inputs
        ):
            raise ValueError(
                "At least one expert payload (technical, price action, or extra_inputs) must be provided."
            )

        raw_inputs: Dict[str, Any] = {}
        if technical_snapshot is not None:
            raw_inputs["technical"] = _safe_payload_copy(technical_snapshot)
        if price_action_snapshot is not None:
            raw_inputs["price_action"] = _safe_payload_copy(price_action_snapshot)
        if extra_inputs:
            for key, value in extra_inputs.items():
                if value is not None:
                    raw_inputs[key] = _safe_payload_copy(value)

        initial_state: PrincipalAgentState = {
            "symbol": symbol.upper(),
            "include_raw_results": include_raw_results,
            "raw_inputs": raw_inputs,
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
    def refresh_graph(self) -> None:
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(PrincipalAgentState)
        builder.add_node("initialise", self._initialise_state)
        builder.add_node("collect_experts", self._collect_experts_node)
        builder.add_node("summarise", self._summarise_node)
        builder.add_edge(START, "initialise")
        builder.add_edge("initialise", "collect_experts")
        builder.add_edge("collect_experts", "summarise")
        builder.add_edge("summarise", END)
        return builder.compile()

    def _initialise_state(self, state: PrincipalAgentState) -> PrincipalAgentState:
        symbol = state.get("symbol")
        if symbol is None:
            raise RuntimeError("Principal agent initial state missing symbol")
        return {
            "symbol": symbol,
            "include_raw_results": state.get("include_raw_results", True),
            "raw_inputs": dict(state.get("raw_inputs", {})),
        }

    def _collect_experts_node(self, state: PrincipalAgentState) -> PrincipalAgentState:
        symbol = state.get("symbol")
        if symbol is None:
            raise RuntimeError("Principal agent state missing symbol before expert collection")

        raw_inputs = dict(state.get("raw_inputs", {}))
        include_raw = state.get("include_raw_results", True)
        expert_outputs: Dict[str, Any] = {}

        if not self.experts:
            logger.warning("Principal agent registry is empty; no expert outputs available")
            return {
                "symbol": symbol,
                "include_raw_results": include_raw,
                "expert_outputs": expert_outputs,
                "raw_inputs": raw_inputs,
            }

        with ThreadPoolExecutor(max_workers=max(len(self.experts), 1)) as executor:
            future_map = {}
            for key, agent in self.experts.items():
                payload = raw_inputs.get(key)
                if payload is None:
                    expert_outputs[key] = {
                        "success": False,
                        "error": "No payload provided for expert",
                    }
                    continue
                future = executor.submit(agent.run, symbol, payload)
                future_map[future] = key

            for future in as_completed(future_map):
                expert_key = future_map[future]
                try:
                    expert_outputs[expert_key] = future.result()
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Expert '%s' failed for %s", expert_key, symbol)
                    expert_outputs[expert_key] = {
                        "success": False,
                        "error": str(exc),
                    }

        next_state: PrincipalAgentState = {
            "symbol": symbol,
            "include_raw_results": include_raw,
            "expert_outputs": expert_outputs,
        }
        if include_raw:
            next_state["raw_inputs"] = raw_inputs
        return next_state

    def _summarise_node(self, state: PrincipalAgentState) -> PrincipalAgentState:
        symbol = state.get("symbol")
        if symbol is None:
            raise RuntimeError("Principal agent state missing symbol during summary")

        expert_outputs = state.get("expert_outputs", {})
        raw_inputs = state.get("raw_inputs", {})
        include_raw = state.get("include_raw_results", True)
        print("include raw result", include_raw)
        summary_payload, usage, raw_principal_text = self._summarise_for_client(
            symbol,
            expert_outputs,
            raw_inputs,
        )
        strategies, supplemental = self._normalise_strategy_summary(summary_payload)

        required_keys = ("day_trading", "swing_trading", "longterm_trading")
        if not strategies:
            logger.warning(
                "Principal agent produced no structured strategies for %s; falling back to heuristic summary",
                symbol,
            )
            strategies = self._fallback_strategies_from_experts(expert_outputs)
        else:
            missing = [key for key in required_keys if key not in strategies]
            if missing:
                logger.debug(
                    "Principal agent missing strategies %s for %s; supplementing with fallback",
                    missing,
                    symbol,
                )
                fallback = self._fallback_strategies_from_experts(expert_outputs)
                for key in missing:
                    if key in fallback:
                        strategies[key] = fallback[key]

        generated_time = datetime.utcnow()
        total_tokens = usage.get("total_tokens") if usage else None

        plan: Dict[str, Any] = {
            "symbol": symbol,
            "generated": generated_time.isoformat(),
            "generated_display": generated_time.strftime("%m/%d/%Y, %I:%M:%S %p"),
            "model": self.model,
            "tokens": usage,
            "strategies": strategies,
            "global_risks": summary_payload.get("global_risks", []),
            "portfolio_guidance": summary_payload.get("portfolio_guidance", {}),
            "supplemental": supplemental,
        }

        plan["generated_at"] = plan["generated"]
        plan["usage"] = usage if usage else {}

        plan["formatted"] = self._render_human_plan(
            symbol,
            plan["generated_display"],
            total_tokens,
            strategies,
        )

        if include_raw:
            plan["expert_outputs"] = expert_outputs
            plan["raw_inputs"] = raw_inputs
            plan["trading_agent_outputs"] = expert_outputs
            plan["principal_raw_text"] = raw_principal_text

        return {"principal_result": plan}

    # ------------------------------------------------------------------
    # GPT-5 Responses API summarisation
    # ------------------------------------------------------------------
    def _summarise_for_client(
        self,
        symbol: str,
        expert_outputs: Dict[str, Any],
        raw_inputs: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
        payload = {
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "expert_outputs": expert_outputs,
        }
        if raw_inputs:
            payload["raw_inputs"] = raw_inputs

        system_prompt = (
            "You are the Principal Strategist of an AI-driven multi-agent trading desk. "
            "You receive analysis from two experts: one provides indicator-based insights, "
            "and the other focuses on price action patterns. "
            "Your job is to fuse both perspectives into a single, consistent trading plan "
            "for three strategies: day trading, swing trading, and long-term trading.\n\n"
            "Output requirements:\n"
            "- Respond ONLY with a single valid JSON object (no markdown, no commentary).\n"
            "- Use plain, simple language in summaries (no jargon, no indicator names like MACD, RSI, Fibonacci, Elliott wave).\n"
            "- For EACH strategy you MUST provide: a summary, a BUY setup, a SELL setup, and at least one NO-TRADE zone.\n"
            "- For BOTH BUY and SELL setups, you MUST provide:\n"
            "    - a single numeric entry level,\n"
            "    - a single numeric stop level,\n"
            "    - an array of at least TWO numeric target prices.\n"
            "- Use only numeric values for prices (no text in those fields).\n"
            "- Do NOT recommend that the user buy or sell; instead, describe conditions such as 'if price breaks above' or 'if price closes below'."
            # "You are the principal strategist of an advanced AI-driven trading desk. "
            # "You receive analysis from two experts: one provides indicator-based insights (including basic, advanced, Elliott wave, and Fibonacci), "
            # "and the other focuses on price action patterns. "
            # "Your job is to synthesize both perspectives and deliver a unified, clear, and actionable summary for three trading strategies: "
            # "day trading, swing trading, and long-term trading. "
            # "Always present information in plain, understandable language—avoid jargon or technical terms wherever possible. "
            # "For each strategy, highlight the key price levels, possible actions, and any reasons for caution based on conflicting signals. "
            # "Your summaries must help a user quickly see the situation, the price levels that matter, and practical next steps. "
            # "When signals disagree, clearly mention the risk or caution. "
            # "Never recommend trades directly; instead, describe the circumstances under which action could be taken. "
            # "Your response must follow the strict JSON template and use natural, concise, and informative wording."
        )

        template_instruction = (
            "Format your response as a STRICT JSON object with EXACTLY these top-level keys: "
            "\"day_trading\", \"swing_trading\", and \"longterm_trading\".\n"
            "Each key MUST contain this structure:\n"
            "{\n"
            " \"summary\": \"string\",\n"
            " \"buy_setup\": {\n"
            " \"entry\": number,\n"
            " \"stop\": number,\n"
            " \"targets\": [number, number]\n"
            " },\n"
            " \"sell_setup\": {\n"
            " \"entry\": number,\n"
            " \"stop\": number,\n"
            " \"targets\": [number, number]\n"
            " },\n"
            " \"no_trade_zone\": [ {\"min\": number, \"max\": number} ]\n"
            "}\n"
            "- Do NOT produce next_actions or key_levels anywhere.\n"
            "- Do NOT output anything except valid JSON.\n"
            # "Format your response as a strict JSON object with these top-level keys: \"day_trading\", \"swing_trading\", and \"longterm_trading\". "
            # "Each key must contain a dictionary with:\n"
            # "  - \"summary\": 1–2 sentences summarizing the overall outlook for that strategy (mention timeframe, direction, and degree of agreement or conflict between expert opinions).\n"
            # "  - \"next_actions\": a list of at least two actionable, specific steps or considerations (including key price levels). "
            # "Express these in plain language, showing what a user should watch for or consider doing at given price ranges, including caution if signals are mixed.\n"
            # "DO NOT use terms like \"MACD\", \"RSI\", \"Fibonacci\", or \"Elliott wave\" in the output. Instead, describe the situation in terms of \"momentum\", "
            # "\"upward trend\", \"downward trend\", \"potential reversal\", \"conflicting signals\", \"support level\", or \"resistance level\".\n"
            # "Example format:\n"
            # "{\n"
            # "  \"day_trading\": {\n"
            # "    \"summary\": \"Short-term analysis shows mixed momentum. Prices may rise if a certain level is broken, but caution is needed due to a possible pullback.\",\n"
            # "    \"next_actions\": [\n"
            # "      \"Consider opportunities if price rises above 150 with strong momentum.\",\n"
            # "      \"Be ready to exit quickly if price falls below 146.\",\n"
            # "      \"Monitor for sudden trend changes near 148-150.\"\n"
            # "    ]\n"
            # "  },\n"
            # "  \"swing_trading\": {\n"
            # "    \"summary\": \"...\",\n"
            # "    \"next_actions\": [ ... ]\n"
            # "  },\n"
            # "  \"longterm_trading\": {\n"
            # "    \"summary\": \"...\",\n"
            # "    \"next_actions\": [ ... ]\n"
            # "  }\n"
            # "}"
        )

        response_input = [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            f"Symbol: {symbol}\n"
                            "Expert agent outputs (JSON):\n"
                            f"{_json_dump(payload)}\n"
                            f"{template_instruction}"
                        ),
                    }
                ],
            },
        ]

        try:
            response = self._create_responses_call(
                {
                    "model": self.model,
                    "input": response_input,
                    "reasoning": {"effort": self.reasoning_effort},
                    "text": {"verbosity": self.text_verbosity},
                    "max_output_tokens": self.max_output_tokens,
                }
            )
        except APIError as exc:  # noqa: BLE001
            raise RuntimeError(f"Principal agent failed to summarise strategies: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Principal agent failed to summarise strategies: {exc}") from exc

        raw_text = _extract_output_text(response) or ""
        logger.info(
            "Principal raw_text length for %s: %s", symbol, len(raw_text),
        )
        logger.debug("Principal raw_text payload for %s:\n%s", symbol, raw_text)
        parsed_summary = self._parse_json_from_text(raw_text)
        if parsed_summary is None:
            logger.warning(
                "Principal agent returned unstructured summary for %s; storing raw text for inspection",
                symbol,
            )
            parsed_summary = {"raw_text": raw_text}
        else:
            logger.debug(
                "Principal parsed keys for %s: %s",
                symbol,
                list(parsed_summary.keys()),
            )

        usage = _extract_usage_details(response)

        return parsed_summary, usage, raw_text

    # ------------------------------------------------------------------
    # Output formatting utilities
    # ------------------------------------------------------------------
    def _render_human_plan(
        self,
        symbol: str,
        generated_display: str,
        tokens_used: Optional[int],
        strategies: Dict[str, Any],
    ) -> str:

        lines: List[str] = []
        lines.append(f"Symbol: {symbol}")
        lines.append(f"Generated: {generated_display}")
        if tokens_used is not None:
            lines.append(f"Tokens used: {tokens_used}")
        lines.append("")

        def _render_strategy(title: str, key: str) -> None:
            section = strategies.get(key)
            if not isinstance(section, dict):
                return

            summary = section.get("summary", "")

            buy = section.get("buy_setup", {})
            sell = section.get("sell_setup", {})
            no_trade = section.get("no_trade_zone", [])

            lines.append(f"{title} — ACTION PLAN")
            lines.append("")
            lines.append(f"Summary: {summary}")
            lines.append("")

            # BUY SETUP
            lines.append("🔵 BUY SETUP")
            lines.append(f" • Entry: {buy.get('entry')}")
            lines.append(f" • Stop: {buy.get('stop')}")
            targets = buy.get("targets", [])
            if targets:
                lines.append(" • Targets: " + " → ".join(map(str, targets)))
            lines.append("")

            # SELL SETUP
            lines.append("🔴 SELL SETUP")
            lines.append(f" • Entry: {sell.get('entry')}")
            lines.append(f" • Stop: {sell.get('stop')}")
            targets = sell.get("targets", [])
            if targets:
                lines.append(" • Targets: " + " → ".join(map(str, targets)))
            lines.append("")

            # NO-TRADE ZONE
            if no_trade:
                lines.append("⚠️ NO-TRADE ZONE")
                for zone in no_trade:
                    if isinstance(zone, dict):
                        lines.append(f" • {zone['min']} → {zone['max']}")
                lines.append("")

        _render_strategy("DAY TRADING", "day_trading")
        _render_strategy("SWING TRADING", "swing_trading")
        _render_strategy("LONG-TERM TRADING", "longterm_trading")

        return "\n".join(lines)
            
    #     self,
    #     symbol: str,
    #     generated_display: str,
    #     tokens_used: Optional[int],
    #     strategies: Dict[str, Any],
    # ) -> str:
    #     lines: List[str] = []
    #     lines.append(f"Symbol: {symbol}")
    #     lines.append(f"Generated: {generated_display}")
    #     if tokens_used is not None:
    #         lines.append(f"Tokens used: {tokens_used}")
    #     lines.append("")

    #     for heading, key in (
    #         ("Day Trading", "day_trading"),
    #         ("Swing Trading", "swing_trading"),
    #         ("Long-Term Trading", "longterm_trading"),
    #     ):
    #         section = strategies.get(key)
    #         if not isinstance(section, dict):
    #             continue
    #         lines.append(heading)
    #         summary = self._coerce_to_sentence(section.get("summary")) or "No summary provided."
    #         lines.append(f"Summary: {summary}")

    #         key_levels = section.get("key_levels")
    #         if isinstance(key_levels, dict) and key_levels:
    #             lines.append("Key Levels:")
    #             for level_key, level_value in key_levels.items():
    #                 label = self._humanise_key(level_key)
    #                 value_text = self._coerce_to_sentence(level_value) or str(level_value)
    #                 lines.append(f"{label}: {value_text}")
    #         elif isinstance(key_levels, list) and key_levels:
    #             lines.append("Key Levels: " + ", ".join(self._coerce_to_list_of_strings(key_levels)))

    #         actions = self._coerce_to_list_of_strings(section.get("next_actions"))
    #         if actions:
    #             lines.append("Next Actions: " + "; ".join(actions))
    #         lines.append("")

    #     return "\n".join(line.rstrip() for line in lines).strip()

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------
    def _normalise_strategy_summary(
        self,
        summary: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        strategies: Dict[str, Any] = {}
        supplemental: Dict[str, Any] = {}

        if not isinstance(summary, dict):
            return strategies, supplemental

        if isinstance(summary.get("principal_trading_plan"), dict):
            summary = summary["principal_trading_plan"]

        alias_map = {
            "day trading": "day_trading",
            "day_trading": "day_trading",
            "intraday": "day_trading",
            "swing trading": "swing_trading",
            "swing_trading": "swing_trading",
            "swing": "swing_trading",
            "long-term trading": "longterm_trading",
            "long term trading": "longterm_trading",
            "longterm trading": "longterm_trading",
            "longterm_trading": "longterm_trading",
            "position trading": "longterm_trading",
        }

        strategies_section = summary.get("strategies")
        working: Any
        if isinstance(strategies_section, dict):
            working = strategies_section
        else:
            working = summary

        if isinstance(working, list):
            for item in working:
                if not isinstance(item, dict):
                    continue
                key = item.get("name") or item.get("strategy") or item.get("label")
                if not isinstance(key, str):
                    continue
                mapped = alias_map.get(key.lower())
                if mapped:
                    strategies[mapped] = item
                else:
                    supplemental[key] = item
            return strategies, supplemental

        if not isinstance(working, dict):
            return strategies, supplemental

        for key, value in working.items():
            mapped = alias_map.get(key.lower()) if isinstance(key, str) else None
            if mapped:
                strategies[mapped] = value
            else:
                supplemental[key] = value

        return strategies, supplemental

    # def _fallback_strategies_from_experts(self, expert_outputs: Dict[str, Any]) -> Dict[str, Any]:
    #     fallback: Dict[str, Any] = {}
    #     summary_texts: List[str] = []
    #     key_levels: Dict[str, Any] = {}
    #     next_actions: List[str] = []

    #     for key in ("technical", "price_action"):
    #         summarised, levels, actions = self._summarise_expert_output(key, expert_outputs.get(key))
    #         if summarised:
    #             summary_texts.append(f"{self._humanise_key(key)}: {summarised}")
    #         if levels:
    #             key_levels.update(levels)
    #         if actions:
    #             next_actions.extend(actions)

    #     default_summary = summary_texts[0] if summary_texts else "Awaiting detailed guidance from expert agents."
    #     default_levels = key_levels or {"Reference": "Monitor recent swing highs and lows."}
    #     default_actions = next_actions or [
    #         "Track momentum alignment before committing capital.",
    #         "Place alerts near key support and resistance clusters.",
    #     ]

    #     template = {
    #         "summary": default_summary,
    #         "key_levels": default_levels,
    #         "next_actions": default_actions,
    #     }

    #     return {
    #         "day_trading": template,
    #         "swing_trading": template,
    #         "longterm_trading": template,
    #     }

    def _fallback_strategies_from_experts(self, expert_outputs: Dict[str, Any]) -> Dict[str, Any]:
        fallback = {}
        summary_texts, support_prices, resistance_prices = [], [], []

        for key in ("technical", "price_action"):
            summarised, levels, actions = self._summarise_expert_output(key, expert_outputs.get(key))
            if summarised:
                summary_texts.append(f"{self._humanise_key(key)}: {summarised}")
            for label, value in (levels or {}).items():
                for token in self._coerce_to_list_of_strings(value, limit=4):
                    price = safe_float(token.replace(",", ""))
                    if price is None:
                        continue
                    if "support" in label.lower():
                        support_prices.append(price)
                    elif "resistance" in label.lower():
                        resistance_prices.append(price)

        def default_setup(entry_hint: Optional[float], direction: str) -> Dict[str, Any]:
            entry = entry_hint or 0.0
            step = abs(entry * 0.005) if entry else 1.0
            if direction == "buy":
                targets = [entry + step, entry + step * 2]
                stop = entry - step
            else:
                targets = [entry - step, entry - step * 2]
                stop = entry + step
            return {
                "entry": round(entry, 2),
                "stop": round(stop, 2),
                "targets": [round(t, 2) for t in targets],
            }

        buy_entry = support_prices[0] if support_prices else (resistance_prices[0] if resistance_prices else None)
        sell_entry = resistance_prices[0] if resistance_prices else (support_prices[0] if support_prices else None)
        summary = summary_texts[0] if summary_texts else "No structured output generated. Re-run analysis."

        template = {
            "summary": summary,
            "buy_setup": default_setup(buy_entry, "buy"),
            "sell_setup": default_setup(sell_entry, "sell"),
            "no_trade_zone": [{
                "min": round(min(buy_entry or 0.0, sell_entry or 0.0), 2),
                "max": round(max(buy_entry or 0.0, sell_entry or 0.0), 2),
            }],
        }

        return {
            "day_trading": template,
            "swing_trading": template,
            "longterm_trading": template,
        }
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
            "market_structure",
            "overall_trend",
            "immediate_bias",
        )
        summary_text = ""
        for candidate in summary_candidates:
            value = output.get(candidate)
            summary_text = self._coerce_to_sentence(value)
            if summary_text:
                break

        if not summary_text:
            backup_sources = (
                output.get("indicator_signals"),
                output.get("structure"),
                output.get("trade_setups"),
            )
            for source in backup_sources:
                summary_text = self._coerce_to_sentence(source)
                if summary_text:
                    break

        key_levels: Dict[str, Any] = {}
        raw_levels = output.get("key_levels") or output.get("levels")
        if isinstance(raw_levels, dict):
            key_levels = {self._humanise_key(k): v for k, v in raw_levels.items()}
        elif isinstance(raw_levels, list):
            key_levels = {str(idx + 1): item for idx, item in enumerate(raw_levels)}

        next_actions = self._coerce_to_list_of_strings(
            output.get("next_actions")
            or output.get("trade_setups")
            or output.get("trade_plan")
            or output.get("actionable_setups")
        )

        if not summary_text and next_actions:
            summary_text = next_actions[0]
        if not summary_text and key_levels:
            level_key, level_value = next(iter(key_levels.items()))
            summary_text = f"{self._humanise_key(level_key)} at {self._coerce_to_sentence(level_value)}"

        return summary_text, key_levels, next_actions

    # ------------------------------------------------------------------
    # Parsing utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_json_from_text(raw_text: str) -> Optional[Dict[str, Any]]:
        if not isinstance(raw_text, str):
            return None

        text = raw_text.strip()
        if not text:
            return None

        if text.startswith("```"):
            text = "\n".join(line for line in text.splitlines() if not line.startswith("```"))
            text = text.strip()

        timestamp_prefix = re.compile(r"^\s*\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\s*")
        # Strip streaming timestamp prefixes injected by the Responses API logs.
        cleaned_lines: List[str] = []
        for line in text.splitlines():
            cleaned = timestamp_prefix.sub("", line).rstrip()
            if cleaned:
                cleaned_lines.append(cleaned)

        text = "\n".join(cleaned_lines).strip()
        if not text:
            return None

        first_positions = [idx for idx in (text.find("{"), text.find("[")) if idx != -1]
        if first_positions:
            text = text[min(first_positions) :].strip()
        if not text:
            return None

        def _balanced_slice(value: str) -> Optional[str]:
            # Walk the substring while tracking braces and quoted strings.
            stack: List[str] = []
            in_string = False
            escape = False
            start: Optional[int] = None
            for index, char in enumerate(value):
                if in_string:
                    if escape:
                        escape = False
                    elif char == "\\":
                        escape = True
                    elif char == '"':
                        in_string = False
                    continue

                if char == '"':
                    in_string = True
                    continue

                if char in "{[":
                    if not stack:
                        start = index
                    stack.append(char)
                elif char in "}]":
                    if not stack:
                        return None
                    opener = stack.pop()
                    if (opener, char) not in (("{", "}"), ("[", "]")):
                        return None
                    if not stack and start is not None:
                        return value[start : index + 1]
            return None

        def _extract_named_block(source: str, field: str) -> Optional[str]:
            needle = f'"{field}"'
            search_index = 0
            while True:
                key_index = source.find(needle, search_index)
                if key_index == -1:
                    return None
                colon_index = source.find(":", key_index + len(needle))
                if colon_index == -1:
                    return None
                brace_index = colon_index + 1
                while brace_index < len(source) and source[brace_index] in " \t\r\n":
                    brace_index += 1
                if brace_index >= len(source):
                    return None
                if source[brace_index] not in "{[":
                    search_index = key_index + len(needle)
                    continue
                block = _balanced_slice(source[brace_index:])
                if block:
                    return block
                search_index = key_index + len(needle)

        target_fields = ("day_trading", "swing_trading", "longterm_trading")
        assembled: Dict[str, Any] = {}
        for field in target_fields:
            block = _extract_named_block(text, field)
            if not block:
                continue
            try:
                assembled[field] = json.loads(block)
            except json.JSONDecodeError:
                continue
        if assembled:
            return assembled

        candidates: List[str] = []
        seen: Set[str] = set()
        for idx, char in enumerate(text):
            if char not in "{[":
                continue
            candidate = _balanced_slice(text[idx:])
            if not candidate:
                continue
            candidate = candidate.strip()
            if candidate and candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)

        if text and text not in seen:
            candidates.append(text)

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        return item
        return None

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
