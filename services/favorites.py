"""Database helpers for user favorite symbols."""

from __future__ import annotations

from typing import Iterable, List, Optional, Set

from sqlalchemy.orm import Session

from models import UserFavoriteSymbol
from utils.symbols import filter_symbols_for_market, sanitize_symbol


def is_symbol_favorited(session: Session, user_id: int, symbol: str) -> bool:
    """Return True if the user has the symbol in their favorites."""

    if not symbol:
        return False

    return (
        session.query(UserFavoriteSymbol)
        .filter(UserFavoriteSymbol.user_id == user_id, UserFavoriteSymbol.symbol == symbol)
        .first()
        is not None
    )


def favorite_symbols(
    session: Session,
    user_id: int,
    limit: int = 25,
    *,
    market: Optional[str] = None,
) -> List[str]:
    """Fetch the user's favorite symbols with optional market filtering."""

    rows = (
        session.query(UserFavoriteSymbol.symbol)
        .filter(UserFavoriteSymbol.user_id == user_id)
        .order_by(UserFavoriteSymbol.created_at.desc())
        .limit(max(limit, 1))
        .all()
    )

    sanitized: List[str] = []
    for row in rows:
        if not row:
            continue
        candidate = sanitize_symbol(row[0])
        if candidate:
            sanitized.append(candidate)

    if market:
        allowed = filter_symbols_for_market(sanitized, market)
        sanitized = [symbol for symbol in sanitized if symbol in allowed]

    return sanitized


def all_favorite_symbols(session: Session, market: Optional[str] = None) -> Set[str]:
    """Return a distinct set of favorited symbols across all users."""

    rows = session.query(UserFavoriteSymbol.symbol).distinct().all()
    symbols = {sanitize_symbol(row[0]) for row in rows if row and row[0]}
    symbols.discard("")
    if market:
        return filter_symbols_for_market(symbols, market)
    return symbols
