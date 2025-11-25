"""Real-time Alpaca option data streaming utilities.

The option market stream is exposed via Alpaca's v1beta1 websocket and
serves data exclusively in MsgPack format. This module wraps the generic
``LivePriceStream`` helper with defaults tailored for option feeds so
future logic can subscribe to contracts, access cached quotes/trades,
and register listeners in the same fashion as equity/crypto streams.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

from .live_price import LivePriceStream, PriceListener, PriceUpdate


_DEFAULT_OPTION_FEED = os.getenv("ALPACA_OPTION_FEED", "indicative").lower()
_DEFAULT_OPTION_VERSION = "v1beta1"


class LiveOptionStream(LivePriceStream):
    """Stream live option trades/quotes from Alpaca's MsgPack feed."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        feed: Optional[str] = None,
        version: Optional[str] = None,
        sandbox: Optional[bool] = None,
        channels: Iterable[str] = ("trades", "quotes"),
    ) -> None:
        super().__init__(
            api_key=api_key,
            api_secret=api_secret,
            feed=(feed or _DEFAULT_OPTION_FEED or "indicative").strip().lower(),
            version=(version or _DEFAULT_OPTION_VERSION),
            sandbox=sandbox,
            channels=channels,
            message_format="msgpack",
        )


__all__ = ["LiveOptionStream", "PriceUpdate", "PriceListener"]
