"""Stripe integration helpers for subscription billing."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, Optional

import stripe

APP_NAME = "VolatilX"


def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is required for billing.")
    return value


@lru_cache(maxsize=1)
def _configure_stripe() -> None:
    api_key = _get_required_env("STRIPE_SECRET_KEY")
    stripe.api_key = api_key
    stripe.app_info = {
        "name": APP_NAME,
        "version": "1.0.0",
        "url": "https://www.volatilx.com",
    }


def get_publishable_key() -> str:
    """Return the Stripe publishable key for client-side SDK."""

    return _get_required_env("STRIPE_PUBLISHABLE_KEY")


def ensure_customer(*, email: str, existing_customer_id: Optional[str] = None) -> stripe.Customer:
    """Retrieve or create a Stripe customer."""

    _configure_stripe()

    if existing_customer_id:
        return stripe.Customer.retrieve(existing_customer_id)

    return stripe.Customer.create(email=email)


def create_subscription_checkout_session(
    *,
    price_id: str,
    success_url: str,
    cancel_url: str,
    client_reference_id: str,
    customer: Optional[str] = None,
    customer_email: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> stripe.checkout.Session:
    """Create a Stripe Checkout session for a subscription plan."""

    _configure_stripe()

    if not customer and not customer_email:
        raise ValueError("Either customer or customer_email must be provided.")

    params: Dict[str, Any] = {
        "mode": "subscription",
        "line_items": [
            {
                "price": price_id,
                "quantity": 1,
            }
        ],
        "success_url": success_url,
        "cancel_url": cancel_url,
        "client_reference_id": client_reference_id,
        "metadata": metadata or {},
        "allow_promotion_codes": True,
    }

    if customer:
        params["customer"] = customer
    else:
        params["customer_email"] = customer_email

    session = stripe.checkout.Session.create(**params)
    return session
