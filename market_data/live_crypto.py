"""Utilities for streaming live cryptocurrency quotes from Alpaca.

This module mirrors the equities streamer in ``live_price`` but targets
Alpaca's crypto websocket feed (v1beta3). It exposes a high-level
``LiveCryptoStream`` helper that can be started once and re-used by
future business logic to subscribe to symbols, retrieve cached
quotes/trades, or register listeners for push-style updates.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

from .live_price import LivePriceStream, PriceListener, PriceUpdate


_DEFAULT_CRYPTO_VERSION = os.getenv("ALPACA_CRYPTO_STREAM_VERSION", "v1beta3").lower()
_DEFAULT_CRYPTO_MARKET = os.getenv("ALPACA_CRYPTO_MARKET", "us").lower()
_DEFAULT_CRYPTO_FEED = os.getenv("ALPACA_CRYPTO_FEED", f"crypto/{_DEFAULT_CRYPTO_MARKET}").lower()


class LiveCryptoStream(LivePriceStream):
    """Stream live crypto trades/quotes with the same interface as equities."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        market: Optional[str] = None,
        feed: Optional[str] = None,
        version: Optional[str] = None,
        sandbox: Optional[bool] = None,
        channels: Iterable[str] = ("trades", "quotes", "bars"),
    ) -> None:
        market_slug = (market or _DEFAULT_CRYPTO_MARKET or "us").strip().lower()
        resolved_feed = (feed or _DEFAULT_CRYPTO_FEED or f"crypto/{market_slug}").strip().lower()
        if not resolved_feed.startswith("crypto"):
            resolved_feed = f"crypto/{market_slug}"
        super().__init__(
            api_key=api_key,
            api_secret=api_secret,
            feed=resolved_feed,
            version=(version or _DEFAULT_CRYPTO_VERSION or "v1beta3").strip().lower(),
            sandbox=sandbox,
            channels=channels,
        )


__all__ = ["LiveCryptoStream", "PriceUpdate", "PriceListener"]
