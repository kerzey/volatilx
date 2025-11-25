"""Real-time Alpaca news streaming utilities.

This module wraps Alpaca's live news websocket feed and exposes an
interface similar to the other market data helpers in this package. It
keeps the most recent articles cached per symbol and provides a
listener-based pub/sub hook so downstream logic can react to breaking
headlines without re-implementing websocket management.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Deque, Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple, cast

from .live_price import LivePriceStream, StreamListener

logger = logging.getLogger(__name__)

DEFAULT_NEWS_VERSION = os.getenv("ALPACA_NEWS_STREAM_VERSION", "v1beta1").strip().lower()
DEFAULT_NEWS_FEED = os.getenv("ALPACA_NEWS_STREAM_FEED", "news").strip().lower()

NewsListener = Callable[["NewsArticle"], Awaitable[None] | None]


@dataclass(slots=True)
class NewsArticle:
    """Represents a single Alpaca news item delivered via the stream."""

    id: str
    headline: str
    summary: Optional[str]
    url: Optional[str]
    author: Optional[str]
    source: Optional[str]
    importance: Optional[int]
    sentiment: Optional[str]
    symbols: Tuple[str, ...]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    images: Tuple[Dict[str, Any], ...]
    received_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the article into primitives for JSON responses."""

        return {
            "id": self.id,
            "headline": self.headline,
            "summary": self.summary,
            "url": self.url,
            "author": self.author,
            "source": self.source,
            "importance": self.importance,
            "sentiment": self.sentiment,
            "symbols": list(self.symbols),
            "created_at": self._serialize_dt(self.created_at),
            "updated_at": self._serialize_dt(self.updated_at),
            "images": [dict(image) for image in self.images],
            "received_at": self._serialize_dt(self.received_at),
        }

    @staticmethod
    def _serialize_dt(value: Optional[datetime]) -> Optional[str]:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()


class LiveNewsStream(LivePriceStream):
    """Manage a persistent connection to Alpaca's live news feed."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        feed: Optional[str] = None,
        version: Optional[str] = None,
        sandbox: Optional[bool] = None,
        channels: Iterable[str] = ("news",),
        message_format: Literal["json", "msgpack"] = "json",
        max_cached_per_symbol: int = 50,
    ) -> None:
        super().__init__(
            api_key=api_key,
            api_secret=api_secret,
            feed=(feed or DEFAULT_NEWS_FEED or "news").strip().lower(),
            version=(version or DEFAULT_NEWS_VERSION or "v1beta1").strip().lower(),
            sandbox=sandbox,
            channels=channels,
            message_format=message_format,
        )
        self._articles: Dict[str, NewsArticle] = {}
        self._symbol_index: Dict[str, Deque[str]] = defaultdict(lambda: deque(maxlen=max_cached_per_symbol))
        self._max_cached_per_symbol = max_cached_per_symbol

    def add_news_listener(self, callback: NewsListener | StreamListener) -> Callable[[], None]:
        """Register a callback that fires for every incoming article."""

        listener = cast(StreamListener, callback)
        self._listeners.add(listener)

        def remove() -> None:
            self._listeners.discard(listener)

        return remove

    def list_symbols(self) -> List[str]:
        """Return the currently subscribed symbols sorted alphabetically."""

        return sorted(self._symbols)

    def get_article(self, article_id: str) -> Optional[NewsArticle]:
        """Fetch a cached article by its Alpaca identifier."""

        return self._articles.get(article_id)

    def get_latest(self, symbol: str) -> Optional[NewsArticle]:  # type: ignore[override]
        """Return the most recent article cached for *symbol* if available."""

        symbol_key = symbol.strip().upper()
        if not symbol_key:
            return None
        ids = self._symbol_index.get(symbol_key)
        if not ids:
            return None
        for article_id in ids:
            article = self._articles.get(article_id)
            if article is not None:
                return article
        return None

    def get_recent(self, symbol: str, limit: int = 10) -> List[NewsArticle]:
        """Return up to *limit* cached articles for *symbol* ordered newest-first."""

        symbol_key = symbol.strip().upper()
        if not symbol_key or limit <= 0:
            return []
        ids = self._symbol_index.get(symbol_key)
        if not ids:
            return []
        articles: List[NewsArticle] = []
        for article_id in list(ids)[:limit]:
            article = self._articles.get(article_id)
            if article is not None:
                articles.append(article)
        return articles

    async def _handle_payload(self, payload: Any) -> None:  # type: ignore[override]
        events = payload if isinstance(payload, list) else [payload]
        for event in events:
            if not isinstance(event, dict):
                continue
            kind = event.get("T")
            if kind == "success":
                continue
            if kind in {"subscription", "authorization"}:
                logger.debug("News stream control message: %s", event)
                continue
            if kind != "n":
                logger.debug("Unhandled news event: %s", event)
                continue
            article = self._parse_article(event)
            if article is None:
                continue
            self._articles[article.id] = article
            for symbol in article.symbols:
                symbol_key = symbol.upper()
                bucket = self._symbol_index[symbol_key]
                bucket.appendleft(article.id)
                if len(bucket) > self._max_cached_per_symbol:
                    bucket.pop()
            await self._dispatch(article)

    async def _dispatch(self, update: Any) -> None:  # type: ignore[override]
        if not self._listeners:
            return
        for callback in list(self._listeners):
            try:
                result = callback(update)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:  # pragma: no cover - defensive logging
                article_id = getattr(update, "id", "<unknown>")
                logger.exception("News listener raised for article %s", article_id)

    def _parse_article(self, event: Dict[str, Any]) -> Optional[NewsArticle]:
        try:
            article_id = str(event.get("id") or event.get("i"))
        except Exception:
            logger.debug("News event missing identifier: %s", event)
            return None
        if not article_id:
            logger.debug("Skipping news event without id: %s", event)
            return None
        symbols_raw: Sequence[str] | None = event.get("symbols") or event.get("S")
        symbols: Tuple[str, ...] = tuple(sorted({s.upper() for s in symbols_raw or () if s}))
        created = self._parse_timestamp(event.get("created_at") or event.get("t"))
        updated = self._parse_timestamp(event.get("updated_at") or event.get("u"))
        images_raw = event.get("images")
        if isinstance(images_raw, dict):
            images_seq: Sequence[Dict[str, Any]] = (images_raw,)  # type: ignore[assignment]
        elif isinstance(images_raw, (list, tuple)):
            images_seq = [img for img in images_raw if isinstance(img, dict)]
        else:
            images_seq = []
        summary = event.get("summary") or event.get("s")
        article = NewsArticle(
            id=article_id,
            headline=str(event.get("headline") or event.get("H") or "").strip(),
            summary=str(summary or "").strip() or None,
            url=_coerce_optional_str(event.get("url") or event.get("l")),
            author=_coerce_optional_str(event.get("author") or event.get("a")),
            source=_coerce_optional_str(event.get("source") or event.get("r")),
            importance=_coerce_optional_int(event.get("importance") or event.get("p")),
            sentiment=_coerce_optional_str(event.get("sentiment") or event.get("d")),
            symbols=symbols,
            created_at=created,
            updated_at=updated,
            images=tuple(images_seq),
        )
        article.received_at = datetime.now(timezone.utc)
        return article


def _coerce_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_optional_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = ["LiveNewsStream", "NewsArticle", "NewsListener"]
