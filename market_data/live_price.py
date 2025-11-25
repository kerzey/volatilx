"""Alpaca live price streaming utilities.

This module exposes an asynchronous helper that manages a persistent
connection to the Alpaca stock data websocket feed. It keeps an
in-memory cache of the latest trade and quote for each subscribed
symbol and lets callers register callbacks for real-time updates.

Usage example (inside an async context):

    service = LivePriceStream()
    await service.start()
    await service.subscribe({"AAPL", "MSFT"})

    def on_update(update: PriceUpdate) -> None:
        print(update.symbol, update.last_trade_price)

    service.add_price_listener(on_update)
    await service.ready()  # wait for initial connection

    # ... later
    latest = service.get_latest("AAPL")
    await service.stop()

Future logic can import this module to fetch the latest cached price or
attach their own listeners for UI delivery without having to manage the
websocket details each time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Iterable, Literal, Optional, Set, cast

import aiohttp
import msgpack

logger = logging.getLogger(__name__)

StreamListener = Callable[[Any], Awaitable[None] | None]
PriceListener = Callable[["PriceUpdate"], Awaitable[None] | None]

_DEFAULT_FEED = os.getenv("ALPACA_DATA_FEED", "iex").lower()
_DEFAULT_VERSION = os.getenv("ALPACA_DATA_VERSION", "v2").lower()
_SANDBOX_HINT = "sandbox" in (os.getenv("APCA_API_BASE_URL", "") or "")


@dataclass(slots=True)
class PriceUpdate:
    """Represents the latest known trade and quote for a symbol."""

    symbol: str
    last_trade_price: Optional[float] = None
    last_trade_size: Optional[int] = None
    last_trade_timestamp: Optional[datetime] = None
    bid_price: Optional[float] = None
    bid_size: Optional[int] = None
    ask_price: Optional[float] = None
    ask_size: Optional[int] = None
    quote_timestamp: Optional[datetime] = None
    volume: Optional[int] = None
    source: str = "alpaca"
    received_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the update into primitives for JSON responses."""

        return {
            "symbol": self.symbol,
            "last_trade_price": self.last_trade_price,
            "last_trade_size": self.last_trade_size,
            "last_trade_timestamp": self._serialize_dt(self.last_trade_timestamp),
            "bid_price": self.bid_price,
            "bid_size": self.bid_size,
            "ask_price": self.ask_price,
            "ask_size": self.ask_size,
            "quote_timestamp": self._serialize_dt(self.quote_timestamp),
            "volume": self.volume,
            "source": self.source,
            "received_at": self._serialize_dt(self.received_at),
        }

    @staticmethod
    def _serialize_dt(value: Optional[datetime]) -> Optional[str]:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()


class LivePriceStream:
    """Manages a websocket stream for live Alpaca equity prices."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        feed: str = _DEFAULT_FEED,
        version: str = _DEFAULT_VERSION,
        sandbox: Optional[bool] = None,
        channels: Iterable[str] = ("trades", "quotes"),
        reconnect_min_delay: float = 1.0,
        reconnect_max_delay: float = 30.0,
        session: Optional[aiohttp.ClientSession] = None,
        message_format: Literal["json", "msgpack"] = "json",
    ) -> None:
        self._api_key = api_key or os.getenv("ALPACA_API_KEY")
        self._api_secret = api_secret or os.getenv("APCA_API_SECRET_KEY")
        if not self._api_key or not self._api_secret:
            raise RuntimeError("Alpaca API credentials are required for live streaming.")

        self._feed = feed.lower()
        self._version = version.lower()
        self._sandbox = _SANDBOX_HINT if sandbox is None else sandbox
        self._channels = tuple(dict.fromkeys(channels))
        self._reconnect_min = reconnect_min_delay
        self._reconnect_max = reconnect_max_delay
        fmt = (message_format or "json").lower()
        if fmt not in {"json", "msgpack"}:
            raise ValueError("message_format must be 'json' or 'msgpack'")
        self._message_format = fmt
        self._ws_headers: Dict[str, str] = {}
        if self._message_format == "msgpack":
            self._ws_headers = {
                "Content-Type": "application/msgpack",
                "Accept": "application/msgpack",
            }

        self._session = session
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._stop_event = asyncio.Event()
        self._ready_event = asyncio.Event()
        self._symbols: Set[str] = set()
        self._latest: Dict[str, PriceUpdate] = {}
        self._listeners: Set[StreamListener] = set()
        self._lock = asyncio.Lock()
        self._runner_task: Optional[asyncio.Task[None]] = None

    @property
    def url(self) -> str:
        base = "wss://stream.data.sandbox.alpaca.markets" if self._sandbox else "wss://stream.data.alpaca.markets"
        return f"{base}/{self._version}/{self._feed}"

    async def start(self) -> None:
        """Start the streaming task if it is not already running."""

        async with self._lock:
            if self._runner_task and not self._runner_task.done():
                return
            self._stop_event.clear()
            loop = asyncio.get_running_loop()
            self._runner_task = loop.create_task(self._run_loop(), name="live-price-stream")

    async def stop(self) -> None:
        """Stop the streaming task and close resources."""

        self._stop_event.set()
        async with self._lock:
            if self._ws and not self._ws.closed:
                await self._ws.close(code=aiohttp.WSCloseCode.GOING_AWAY)
            if self._runner_task:
                await asyncio.wait({self._runner_task}, return_when=asyncio.ALL_COMPLETED)
                self._runner_task = None
            if self._session:
                await self._session.close()
                self._session = None
            self._ready_event.clear()

    async def ready(self, timeout: Optional[float] = 10.0) -> None:
        """Wait until the websocket is authenticated and subscribed."""

        await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)

    async def subscribe(self, symbols: Iterable[str]) -> None:
        """Add the provided symbols to the subscription set."""

        normalized = {symbol.strip().upper() for symbol in symbols if symbol and symbol.strip()}
        if not normalized:
            return
        async with self._lock:
            added = normalized - self._symbols
            if not added:
                return
            self._symbols |= added
            if self._ws and not self._ws.closed:
                await self._send({"action": "subscribe", **self._channels_payload(added)})
            else:
                logger.debug("Queueing subscription for %s", ", ".join(sorted(added)))

    async def unsubscribe(self, symbols: Iterable[str]) -> None:
        """Remove symbols from the subscription set."""

        normalized = {symbol.strip().upper() for symbol in symbols if symbol and symbol.strip()}
        if not normalized:
            return
        async with self._lock:
            removed = normalized & self._symbols
            if not removed:
                return
            self._symbols -= removed
            if self._ws and not self._ws.closed:
                await self._send({"action": "unsubscribe", **self._channels_payload(removed)})

    def get_latest(self, symbol: str) -> Optional[PriceUpdate]:
        """Return the cached price for *symbol* if available."""

        return self._latest.get(symbol.upper())

    def add_price_listener(self, callback: PriceListener | StreamListener) -> Callable[[], None]:
        """Register a callback that fires for every price update."""

        listener = cast(StreamListener, callback)
        self._listeners.add(listener)

        def remove() -> None:
            self._listeners.discard(listener)

        return remove

    async def _run_loop(self) -> None:
        backoff = self._reconnect_min
        while not self._stop_event.is_set():
            try:
                await self._connect()
                backoff = self._reconnect_min
                await self._consume()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - runtime resilience
                logger.exception("Live price stream error: %s", exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._reconnect_max)
            finally:
                await self._cleanup_ws()
        await self._cleanup_ws()

    async def _connect(self) -> None:
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=None, sock_read=None, sock_connect=10)
            self._session = aiohttp.ClientSession(timeout=timeout)
        logger.info("Connecting to Alpaca stream at %s", self.url)
        headers = self._ws_headers or None
        self._ws = await self._session.ws_connect(self.url, heartbeat=20, autoping=True, headers=headers)
        await self._expect_success("connected")
        await self._send({"action": "auth", "key": self._api_key, "secret": self._api_secret})
        await self._expect_success("authenticated")
        if self._symbols:
            await self._send({"action": "subscribe", **self._channels_payload(self._symbols)})
        self._ready_event.set()

    async def _consume(self) -> None:
        assert self._ws is not None
        async for message in self._ws:
            if self._stop_event.is_set():
                break
            if message.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                logger.info("Alpaca websocket closed (%s)", message.extra)
                break
            if message.type == aiohttp.WSMsgType.ERROR:
                logger.warning("Websocket error: %s", message.data)
                break
            payload = self._decode_payload(message)
            if payload is None:
                continue
            await self._handle_payload(payload)

    async def _handle_payload(self, payload: Any) -> None:
        events = payload if isinstance(payload, list) else [payload]
        for event in events:
            if not isinstance(event, dict):
                continue
            kind = event.get("T")
            if kind == "success":
                continue
            if kind in {"subscription", "authorization"}:
                logger.debug("Stream control message: %s", event)
                continue
            if kind == "t":
                await self._handle_trade(event)
            elif kind == "q":
                await self._handle_quote(event)
            elif kind in {"b", "u", "d"}:
                await self._handle_bar(event)
            else:
                logger.debug("Unhandled stream event: %s", event)

    async def _handle_trade(self, event: Dict[str, Any]) -> None:
        symbol = event.get("S")
        if not symbol:
            return
        update = self._ensure_update(symbol)
        update.last_trade_price = self._safe_float(event.get("p"))
        update.last_trade_size = self._safe_int(event.get("s"))
        update.last_trade_timestamp = self._parse_timestamp(event.get("t"))
        update.received_at = datetime.now(timezone.utc)
        await self._dispatch(update)

    async def _handle_quote(self, event: Dict[str, Any]) -> None:
        symbol = event.get("S")
        if not symbol:
            return
        update = self._ensure_update(symbol)
        update.bid_price = self._safe_float(event.get("bp"))
        update.bid_size = self._safe_int(event.get("bs"))
        update.ask_price = self._safe_float(event.get("ap"))
        update.ask_size = self._safe_int(event.get("as"))
        update.quote_timestamp = self._parse_timestamp(event.get("t"))
        update.received_at = datetime.now(timezone.utc)
        await self._dispatch(update)

    async def _handle_bar(self, event: Dict[str, Any]) -> None:
        symbol = event.get("S")
        if not symbol:
            return
        update = self._ensure_update(symbol)
        update.last_trade_price = self._safe_float(event.get("c"), update.last_trade_price)
        update.volume = self._safe_int(event.get("v"), update.volume)
        update.received_at = datetime.now(timezone.utc)
        await self._dispatch(update)

    async def _dispatch(self, update: Any) -> None:
        if not self._listeners:
            return
        for callback in list(self._listeners):
            try:
                result = callback(update)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Price listener failed for %s", update.symbol)

    def _ensure_update(self, symbol: str) -> PriceUpdate:
        key = symbol.upper()
        existing = self._latest.get(key)
        if existing:
            return existing
        update = PriceUpdate(symbol=key)
        self._latest[key] = update
        return update

    async def _expect_success(self, message: str, timeout: float = 5.0) -> None:
        assert self._ws is not None
        while True:
            response = await asyncio.wait_for(self._ws.receive(), timeout=timeout)
            if response.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                raise RuntimeError("Websocket closed during handshake")
            if response.type == aiohttp.WSMsgType.ERROR:
                raise RuntimeError(f"Websocket error during handshake: {response.data}")
            payload = self._decode_payload(response)
            if payload is None:
                continue
            events = payload if isinstance(payload, list) else [payload]
            for event in events:
                if not isinstance(event, dict):
                    continue
                if event.get("T") == "success" and event.get("msg") == message:
                    return
                logger.debug("Handshake passthrough: %s", event)

    async def _send(self, payload: Dict[str, Any]) -> None:
        if not self._ws or self._ws.closed:
            raise RuntimeError("Cannot send on a closed Alpaca websocket.")
        if self._message_format == "msgpack":
            await self._ws.send_bytes(msgpack.dumps(payload, use_bin_type=True))
        else:
            await self._ws.send_json(payload)

    def _channels_payload(self, symbols: Iterable[str]) -> Dict[str, Any]:
        symbols_list = sorted({symbol.upper() for symbol in symbols})
        payload: Dict[str, Any] = {}
        for channel in self._channels:
            payload[channel] = symbols_list
        return payload

    async def _cleanup_ws(self) -> None:
        if self._ws is not None:
            if not self._ws.closed:
                await self._ws.close()
            self._ws = None
            self._ready_event.clear()

    def _decode_payload(self, message: aiohttp.WSMessage) -> Optional[Any]:
        if self._message_format == "json":
            if message.type != aiohttp.WSMsgType.TEXT:
                logger.debug("Ignoring non-text message in JSON mode: %s", message.type)
                return None
            try:
                return json.loads(message.data)
            except json.JSONDecodeError:
                logger.debug("Failed to decode JSON payload: %s", message.data)
                return None
        # msgpack mode
        if message.type == aiohttp.WSMsgType.BINARY:
            try:
                return msgpack.loads(message.data, raw=False)
            except Exception:
                logger.debug("Failed to decode msgpack payload", exc_info=True)
                return None
        logger.debug("Ignoring websocket message type %s for msgpack mode", message.type)
        return None

    @staticmethod
    def _parse_timestamp(raw: Any) -> Optional[datetime]:
        if not raw:
            return None
        if isinstance(raw, datetime):
            return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
        if isinstance(raw, (int, float)):
            return datetime.fromtimestamp(float(raw), tz=timezone.utc)
        if isinstance(raw, str):
            cleaned = raw.replace("Z", "+00:00")
            try:
                return datetime.fromisoformat(cleaned)
            except ValueError:
                logger.debug("Could not parse timestamp: %s", raw)
                return None
        return None

    @staticmethod
    def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default


__all__ = ["LivePriceStream", "PriceUpdate", "PriceListener", "StreamListener"]
