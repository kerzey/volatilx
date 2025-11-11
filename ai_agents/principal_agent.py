"""Principal agent that fuses trading strategies for the client."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from langgraph.graph import END, START, StateGraph
from openai import APIError, BadRequestError, OpenAI
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


def _make_strategy_schema(key_levels_property: Dict[str, Any]) -> Dict[str, Any]:
    key_levels_spec = {"description": "Support and resistance levels or comparable technical thresholds."}
    key_levels_spec.update(key_levels_property)

    return {
        "type": "object",
        "required": ["summary", "key_levels", "next_actions"],
        "properties": {
            "summary": {
                "type": "string",
                "description": "Concise description of the recommended trading stance.",
            },
            "key_levels": key_levels_spec,
            "next_actions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Prioritised sequence of client-ready actions.",
            },
        },
        "additionalProperties": False,
    }


def _make_principal_plan_schema(strategy_schema: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "object",
        "required": [
            "day_trading",
            "swing_trading",
            "longterm_trading",
            "global_risks",
            "portfolio_guidance",
        ],
        "properties": {
            "day_trading": strategy_schema,
            "swing_trading": strategy_schema,
            "longterm_trading": strategy_schema,
            "global_risks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Cross-strategy risk factors the client must monitor.",
            },
            "portfolio_guidance": {
                "type": "object",
                "required": ["summary"],
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Top-level guidance synthesising the trading landscape.",
                    },
                    "asset_allocation": {
                        "type": "string",
                        "description": "Adjustments to position sizing and capital deployment.",
                    },
                    "risk_management": {
                        "type": "string",
                        "description": "Risk controls, hedges, or protective orders to consider.",
                    },
                    "positioning": {
                        "type": "string",
                        "description": "Suggested directional stance and timeframe alignment.",
                    },
                    "notes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Additional clarifications or implementation caveats.",
                    },
                },
                "additionalProperties": True,
            },
            "context": {
                "type": "object",
                "description": "Optional supplemental data keyed by label.",
                "additionalProperties": True,
            },
        },
        "additionalProperties": False,
    }


def _make_response_format(schema: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "principal_trading_plan",
            "schema": schema,
            "strict": True,
        },
    }


_KEY_LEVELS_PROPERTY_STANDARD: Dict[str, Any] = {
    "oneOf": [
        {
            "type": "object",
            "additionalProperties": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "number"},
                    {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                ]
            },
        },
        {
            "type": "array",
            "items": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "number"},
                    {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                ]
            },
        },
        {"type": "string"},
        {"type": "null"},
    ]
}


_KEY_LEVELS_PROPERTY_SIMPLE: Dict[str, Any] = {
    "type": "object",
    "default": {},
    "additionalProperties": {
        "type": "string",
        "description": "Key level value or rationale in free-form text.",
    },
}


_STRATEGY_SCHEMA: Dict[str, Any] = _make_strategy_schema(_KEY_LEVELS_PROPERTY_STANDARD)
_STRATEGY_SCHEMA_SIMPLE: Dict[str, Any] = _make_strategy_schema(_KEY_LEVELS_PROPERTY_SIMPLE)


_PRINCIPAL_PLAN_SCHEMA: Dict[str, Any] = _make_principal_plan_schema(_STRATEGY_SCHEMA)
_PRINCIPAL_PLAN_SCHEMA_SIMPLE: Dict[str, Any] = _make_principal_plan_schema(_STRATEGY_SCHEMA_SIMPLE)


_PRINCIPAL_PLAN_RESPONSE_FORMAT = _make_response_format(_PRINCIPAL_PLAN_SCHEMA)
_PRINCIPAL_PLAN_RESPONSE_FORMAT_SIMPLE = _make_response_format(_PRINCIPAL_PLAN_SCHEMA_SIMPLE)

_PRINCIPAL_PLAN_OUTPUT_TEMPLATE = json.dumps(  # Example shape for models without strict schema
    {
        "day_trading": {
            "summary": "Clear guidance for intraday positioning",
            "key_levels": {
                "example_level": "123.45 - rationale",
                "alternate_level": "Another key observation",
            },
            "next_actions": [
                "First concrete step",
                "Second concrete step",
            ],
        },
        "swing_trading": {
            "summary": "Swing outlook and bias",
            "key_levels": {
                "swing_support": "Support rationale",
                "swing_resistance": "Resistance rationale",
            },
            "next_actions": [
                "Swing action item",
                "Another swing action item",
            ],
        },
        "longterm_trading": {
            "summary": "Position trade framing",
            "key_levels": {
                "long_support": "Support rationale",
                "long_target": "Target rationale",
            },
            "next_actions": [
                "Long-term action",
                "Risk control action",
            ],
        },
        "global_risks": ["Cross-horizon risk"],
        "portfolio_guidance": {
            "summary": "Top-level synthesis",
            "asset_allocation": "Allocation advice",
            "risk_management": "Risk management advice",
            "positioning": "Directional stance",
            "notes": ["Additional note"],
        },
    },
    indent=2,
)


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
    openai_model: str = "gpt-5-nano",#"gpt-4o-mini"
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
            or MomentumExpertAgent(openai_client=self.client, analyzer=analyzer, model=self.model),
            "volatility": volatility_agent
            or VolatilityExpertAgent(openai_client=self.client, analyzer=analyzer, model=self.model),
            "pattern": pattern_agent
            or PatternRecognitionExpertAgent(openai_client=self.client, analyzer=analyzer, model=self.model),
            "trend": trend_agent
            or TrendExpertAgent(openai_client=self.client, analyzer=analyzer, model=self.model),
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
            "Adhere exactly to the provided response JSON schema and populate every required field. "
            "Each strategy must include a disciplined 'summary', the most actionable 'key_levels', and a sequenced "
            "list of 'next_actions'. "
            "Highlight risks that cut across horizons and conclude with portfolio-level guidance that balances "
            "opportunity with capital protection."
        )

        if self.model and self.model.lower().startswith("gpt-5-"):
            system_prompt += (
                " For each strategy's 'key_levels', return an object that maps concise level names to short"
                " descriptive strings (e.g., '402.5 - intraday resistance')."
                " Respond with a single JSON object containing the keys 'day_trading', 'swing_trading',"
                " 'longterm_trading', 'global_risks', and 'portfolio_guidance'. Each strategy key must include"
                " 'summary', 'key_levels', and 'next_actions'."
            )

        template_instructions = (
            "Return only JSON. Populate the following template with substantive values (replace all placeholder text):\n"
            f"{_PRINCIPAL_PLAN_OUTPUT_TEMPLATE}\n"
            "Use descriptive keys inside 'key_levels' and provide at least two actionable next steps per strategy."
        )

        try:
            response = self._create_summary_completion(
                symbol,
                system_prompt,
                user_payload,
                template_instructions,
            )
        except (APIError, BadRequestError) as exc:  # noqa: BLE001
            raise RuntimeError(f"Principal agent failed to summarise strategies: {exc}") from exc

        content = self._extract_response_content(response) or "{}"
        logger.debug("Principal summary raw response for %s: %s", symbol, content)
        parsed_summary = self._parse_json_from_text(content)
        if parsed_summary is None:
            snippet = content.strip()
            if len(snippet) > 1000:
                snippet = snippet[:1000] + "..."
            print(f"[PrincipalAgent] Unstructured summary for {symbol}: {snippet}")
        else:
            print(
                f"[PrincipalAgent] Parsed summary for {symbol}: {list(parsed_summary.keys())}"
            )
        summary = parsed_summary if parsed_summary is not None else {"raw_text": content}

        usage = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
            "completion_tokens": getattr(response.usage, "completion_tokens", None),
            "total_tokens": getattr(response.usage, "total_tokens", None),
        }

        return summary, usage

    def _create_summary_completion(
        self,
        symbol: str,
        system_prompt: str,
        user_payload: str,
        template_instructions: str,
    ):
        response_format = self._principal_plan_response_format()
        schema_enabled = bool(response_format and response_format.get("type") == "json_schema")
        responses_supported = self._responses_api_supported()

        if responses_supported:
            try:
                return self._create_summary_with_responses_api(
                    system_prompt,
                    user_payload,
                    symbol,
                    response_format=response_format,
                    template_instructions=template_instructions,
                )
            except AttributeError:  # pragma: no cover - defensive fallback for older clients
                logger.debug("Responses API unavailable at runtime, falling back to chat completions")
            except (APIError, BadRequestError) as exc:  # noqa: BLE001
                if self._is_invalid_schema_error(exc):
                    logger.warning(
                        "Responses API rejected schema; reverting to json_object for chat completion: %s",
                        exc,
                    )
                    schema_enabled = False
                    response_format = None
                else:
                    raise

        return self._create_summary_with_chat_api(
            system_prompt,
            user_payload,
            symbol,
            schema_enabled=schema_enabled,
            response_format=response_format,
            template_instructions=template_instructions,
        )

    def _create_summary_with_responses_api(
        self,
        system_prompt: str,
        user_payload: str,
        symbol: str,
        *,
        response_format: Optional[Dict[str, Any]],
        template_instructions: str,
    ):
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Symbol: {symbol.upper()}\n"
                            "Trading agent outputs (JSON):\n"
                            f"{user_payload}\n"
                            "Craft disciplined, risk-aware guidance for the client.\n"
                            f"{template_instructions}"
                        ),
                    }
                ],
            },
        ]

        base_kwargs = {
            "model": self.model,
            "input": messages,
        }

        if response_format is not None:
            base_kwargs["response_format"] = response_format

        if not self._model_requires_default_temperature():
            base_kwargs["temperature"] = self.temperature

        if self._model_supports_reasoning_controls():
            base_kwargs.setdefault("reasoning", {"effort": "medium"})
            base_kwargs.setdefault("text", {"verbosity": "medium"})
            base_kwargs.setdefault("max_output_tokens", self._summary_token_limit)
        else:
            base_kwargs.setdefault("max_output_tokens", self._summary_token_limit)

        try:
            return self.client.responses.create(**base_kwargs)
        except (APIError, BadRequestError) as exc:  # noqa: BLE001
            if not self._is_unsupported_parameter_error(exc):
                raise
            param_name_error = self._extract_error_param_name(exc)
            if param_name_error:
                base_kwargs.pop(param_name_error, None)
            if param_name_error == "response_format":
                base_kwargs.pop("response_format", None)
            return self.client.responses.create(**base_kwargs)

    def _create_summary_with_chat_api(
        self,
        system_prompt: str,
        user_payload: str,
        symbol: str,
        *,
        schema_enabled: bool = True,
        response_format: Optional[Dict[str, Any]] = None,
        template_instructions: str,
    ):
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Symbol: {symbol.upper()}\n"
                    "Trading agent outputs (JSON):\n"
                    f"{user_payload}\n"
                    "Craft disciplined, risk-aware guidance for the client.\n"
                    f"{template_instructions}"
                ),
            },
        ]

        if schema_enabled and response_format is None:
            response_format = _PRINCIPAL_PLAN_RESPONSE_FORMAT

        chat_response_format: Dict[str, Any]
        if response_format is not None:
            chat_response_format = response_format
        else:
            chat_response_format = {"type": "json_object"}

        base_kwargs = {
            "model": self.model,
            "messages": messages,
            "response_format": chat_response_format,
        }

        if not self._model_requires_default_temperature():
            base_kwargs["temperature"] = self.temperature

        token_params = list(self._token_param_candidates())
        last_error: Optional[BaseException] = None
        index = 0
        while index < len(token_params):
            param_name = token_params[index]
            try:
                return self.client.chat.completions.create(
                    **base_kwargs,
                    **{param_name: self._summary_token_limit},
                )
            except (APIError, BadRequestError) as exc:  # noqa: BLE001
                last_error = exc
                if self._is_invalid_schema_error(exc) and schema_enabled:
                    logger.warning(
                        "Chat completion rejected schema; retrying with json_object format: %s",
                        exc,
                    )
                    schema_enabled = False
                    base_kwargs["response_format"] = {"type": "json_object"}
                    continue
                if self._is_unsupported_parameter_error(exc):
                    param_name_error = self._extract_error_param_name(exc)
                    if param_name_error == "response_format":
                        base_kwargs.pop("response_format", None)
                        base_kwargs.setdefault("response_format", {"type": "json_object"})
                        continue
                    index += 1
                    continue
                raise
            index += 1

        if last_error is not None:
            raise last_error

        raise RuntimeError("Failed to create summary completion: no completion attempted")

    def _responses_api_supported(self) -> bool:
        responses_attr = getattr(self.client, "responses", None)
        return callable(getattr(responses_attr, "create", None))

    @staticmethod
    def _extract_response_content(response: Any) -> str:
        if response is None:
            return ""

        text = getattr(response, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text

        candidate_data: Any = None
        if hasattr(response, "model_dump"):
            try:
                candidate_data = response.model_dump()
            except Exception:  # noqa: BLE001
                candidate_data = None
        if candidate_data is None and hasattr(response, "to_dict"):
            try:
                candidate_data = response.to_dict()
            except Exception:  # noqa: BLE001
                candidate_data = None
        if candidate_data is None:
            candidate_data = response

        if isinstance(candidate_data, dict):
            output = candidate_data.get("output")
            if isinstance(output, list):
                fragments: List[str] = []
                for node in output:
                    if isinstance(node, dict):
                        content_list = node.get("content")
                    else:
                        content_list = getattr(node, "content", None)
                    if isinstance(content_list, list):
                        for content_item in content_list:
                            if isinstance(content_item, dict):
                                text_value = content_item.get("text")
                                if isinstance(text_value, str):
                                    fragments.append(text_value)
                                elif isinstance(text_value, dict):
                                    value = text_value.get("value")
                                    if isinstance(value, str):
                                        fragments.append(value)
                            else:
                                text_value = getattr(content_item, "text", None)
                                if isinstance(text_value, str):
                                    fragments.append(text_value)
                    if fragments:
                        break
                if fragments:
                    return "\n".join(fragments).strip()

            choices = candidate_data.get("choices")
            if isinstance(choices, list) and choices:
                message = choices[0].get("message") if isinstance(choices[0], dict) else None
                if isinstance(message, dict):
                    content_value = message.get("content")
                    if isinstance(content_value, str):
                        return content_value

        return ""

    @staticmethod
    def _parse_json_from_text(raw_text: str) -> Optional[Dict[str, Any]]:
        if not isinstance(raw_text, str):
            return None

        def _strip_code_fence(text: str) -> str:
            lines = text.strip().splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            return "\n".join(lines).strip()

        def _first_json_object(text: str) -> Optional[str]:
            start = text.find("{")
            if start == -1:
                return None
            depth = 0
            for index in range(start, len(text)):
                char = text[index]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : index + 1]
            return None

        candidates: List[str] = []
        stripped = raw_text.strip()
        if not stripped:
            return None

        fenced = _strip_code_fence(stripped)
        if fenced:
            candidates.append(fenced)

        object_slice = _first_json_object(fenced)
        if object_slice:
            candidates.insert(0, object_slice)

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

    @staticmethod
    def _is_unsupported_parameter_error(exc: BaseException) -> bool:
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

    @staticmethod
    def _extract_error_param_name(exc: BaseException) -> Optional[str]:
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            error_data = body.get("error")
            if isinstance(error_data, dict):
                param = error_data.get("param")
                if isinstance(param, str) and param:
                    return param
        message = str(getattr(exc, "message", exc))
        lowered = message.lower()
        for candidate in (
            "max_tokens",
            "max_completion_tokens",
            "max_output_tokens",
            "response_format",
            "reasoning",
            "text",
        ):
            if candidate in lowered:
                return candidate
        return None

    @staticmethod
    def _is_invalid_schema_error(exc: BaseException) -> bool:
        message = str(getattr(exc, "message", ""))
        if "invalid schema" in message.lower():
            return True
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            error_data = body.get("error")
            if isinstance(error_data, dict):
                message_text = error_data.get("message", "")
                if isinstance(message_text, str) and "invalid schema" in message_text.lower():
                    return True
                param = error_data.get("param")
                if param == "response_format":
                    return True
        return False

    def _model_requires_default_temperature(self) -> bool:
        if not self.model:
            return False
        lowered = self.model.lower()
        return lowered.startswith("gpt-5-")

    def _token_param_candidates(self) -> Tuple[str, ...]:
        if not self.model:
            return ("max_completion_tokens", "max_tokens")
        lowered = self.model.lower()
        if lowered.startswith("gpt-5-"):
            return ("max_completion_tokens",)
        return ("max_completion_tokens", "max_tokens")

    def _principal_plan_response_format(self) -> Optional[Dict[str, Any]]:
        if not self.model:
            return _PRINCIPAL_PLAN_RESPONSE_FORMAT
        lowered = self.model.lower()
        if lowered.startswith("gpt-5-"):
            return {"type": "json_object"}
        return _PRINCIPAL_PLAN_RESPONSE_FORMAT

    def _model_supports_reasoning_controls(self) -> bool:
        if not self.model:
            return False
        return self.model.lower().startswith("gpt-5-")

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
            print(f"[PrincipalAgent] Running trading agent '{strategy}' for {symbol}")
            result = agent.run(symbol)
            try:
                pretty_result = _json_dump(result)
            except Exception:  # noqa: BLE001 - defensive for non-serialisable diagnostics
                pretty_result = repr(result)
            print(f"[PrincipalAgent] Result for '{strategy}' on {symbol}: {pretty_result}")
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

        if isinstance(summary.get("principal_trading_plan"), dict):
            summary = summary["principal_trading_plan"]  # model may nest output under schema name

        alias_map = {
            "day trading": "day_trading",
            "day_trading": "day_trading",
            "daytrading": "day_trading",
            "daytrades": "day_trading",
            "intraday": "day_trading",
            "daytradingplan": "day_trading",
            "daytradingstrategy": "day_trading",
            "daytradinginsights": "day_trading",
            "swing trading": "swing_trading",
            "swing_trading": "swing_trading",
            "swingtrading": "swing_trading",
            "swing": "swing_trading",
            "swingplan": "swing_trading",
            "longterm trading": "longterm_trading",
            "longterm_trading": "longterm_trading",
            "long-term": "longterm_trading",
            "long term": "longterm_trading",
            "longterm": "longterm_trading",
            "position trading": "longterm_trading",
            "position": "longterm_trading",
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
                normalised_key = key.lower()
                mapped = alias_map.get(normalised_key)
                if mapped:
                    strategies[mapped] = item
                else:
                    supplemental[key] = item
            return strategies, supplemental

        if not isinstance(working, dict):
            return strategies, supplemental

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

