"""Stripe webhook handlers for subscription lifecycle management."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import stripe
from fastapi import HTTPException
from sqlalchemy.orm import Session

from models import SubscriptionPlan, SubscriptionUsage, UserSubscription

logger = logging.getLogger(__name__)


@dataclass
class StripeWebhookConfig:
    signing_secret: str


def parse_event(payload: bytes, signature_header: str, signing_secret: str) -> stripe.Event:
    """Validate and parse a Stripe webhook event."""

    try:
        event = stripe.Webhook.construct_event(
            payload=payload,
            sig_header=signature_header,
            secret=signing_secret,
        )
    except stripe.error.SignatureVerificationError as exc:  # pragma: no cover - external dependency
        logger.warning("Invalid Stripe signature: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid signature") from exc
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to parse Stripe webhook: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid payload") from exc

    return event


def _find_subscription_by_stripe_id(session: Session, stripe_subscription_id: str) -> Optional[UserSubscription]:
    return (
        session.query(UserSubscription)
        .filter(UserSubscription.stripe_subscription_id == stripe_subscription_id)
        .one_or_none()
    )


def _find_plan_by_stripe_price(session: Session, stripe_price_id: str) -> Optional[SubscriptionPlan]:
    return (
        session.query(SubscriptionPlan)
        .filter(SubscriptionPlan.stripe_price_id == stripe_price_id)
        .one_or_none()
    )


def _iso_to_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromtimestamp(int(value))
    except (TypeError, ValueError):
        return None


def _process_subscription_payload(session: Session, subscription_data: Dict[str, Any]) -> None:
    stripe_subscription_id = subscription_data.get("id")
    status = subscription_data.get("status")
    current_period_start = subscription_data.get("current_period_start")
    current_period_end = subscription_data.get("current_period_end")
    cancel_at_period_end = subscription_data.get("cancel_at_period_end", False)
    items = subscription_data.get("items", {}).get("data", [])

    plan_price_id = None
    if items:
        plan_price_id = items[0].get("price", {}).get("id")

    subscription = _find_subscription_by_stripe_id(session, stripe_subscription_id)

    if subscription is None:
        # Attempt to link using client_reference_id stored in metadata
        metadata = subscription_data.get("metadata", {})
        user_id = metadata.get("user_id")
        plan_slug = metadata.get("plan_slug")

        if not user_id:
            logger.warning(
                "Received subscription event without metadata user reference: %s",
                stripe_subscription_id,
            )
            return

        from user import User  # Local import to avoid circular dependency

        user = session.query(User).filter(User.id == int(user_id)).one_or_none()
        if user is None:
            logger.warning(
                "Stripe subscription references unknown user %s", user_id
            )
            return

        plan = None
        if plan_price_id:
            plan = _find_plan_by_stripe_price(session, plan_price_id)
        if plan is None and plan_slug:
            plan = (
                session.query(SubscriptionPlan)
                .filter(SubscriptionPlan.slug == plan_slug)
                .one_or_none()
            )

        if plan is None:
            logger.warning(
                "Unable to resolve plan for Stripe subscription %s (price=%s, slug=%s)",
                stripe_subscription_id,
                plan_price_id,
                plan_slug,
            )
            return

        subscription = UserSubscription(
            user_id=user.id,
            plan_id=plan.id,
            stripe_customer_id=subscription_data.get("customer"),
            stripe_subscription_id=stripe_subscription_id,
            status=status or "incomplete",
            current_period_start=_iso_to_datetime(current_period_start),
            current_period_end=_iso_to_datetime(current_period_end),
            runs_remaining=plan.ai_runs_included,
            cancel_at_period_end=bool(cancel_at_period_end),
        )
        session.add(subscription)
        session.commit()
        return

    # Update existing subscription
    if plan_price_id:
        plan = _find_plan_by_stripe_price(session, plan_price_id)
        if plan and subscription.plan_id != plan.id:
            subscription.plan_id = plan.id
            subscription.runs_remaining = plan.ai_runs_included

    subscription.status = status or subscription.status
    subscription.current_period_start = _iso_to_datetime(current_period_start)
    subscription.current_period_end = _iso_to_datetime(current_period_end)
    subscription.cancel_at_period_end = bool(cancel_at_period_end)
    subscription.stripe_customer_id = subscription_data.get("customer")

    session.commit()


def handle_subscription_updated(session: Session, event: stripe.Event) -> None:
    subscription_data: Dict[str, Any] = event["data"]["object"]
    _process_subscription_payload(session, subscription_data)


def handle_invoice_paid(session: Session, event: stripe.Event) -> None:
    invoice_data: Dict[str, Any] = event["data"]["object"]
    stripe_subscription_id = invoice_data.get("subscription")
    if not stripe_subscription_id:
        return

    subscription = _find_subscription_by_stripe_id(session, stripe_subscription_id)
    if subscription is None:
        logger.info("Invoice paid for unknown subscription %s", stripe_subscription_id)
        return

    # On successful payment, replenish run quota to plan allowance
    plan = subscription.plan
    if plan:
        subscription.runs_remaining = plan.ai_runs_included
        subscription.status = "active"
        session.commit()


def handle_subscription_deleted(session: Session, event: stripe.Event) -> None:
    subscription_data: Dict[str, Any] = event["data"]["object"]
    stripe_subscription_id = subscription_data.get("id")
    subscription = _find_subscription_by_stripe_id(session, stripe_subscription_id)
    if subscription is None:
        return

    subscription.status = subscription_data.get("status", "canceled")
    subscription.cancel_at_period_end = True
    subscription.runs_remaining = 0
    session.commit()


def enqueue_usage(
    session: Session,
    subscription: UserSubscription,
    *,
    units: int = 1,
    notes: Optional[str] = None,
    usage_type: str = "ai_run",
) -> None:
    usage = SubscriptionUsage(
        subscription_id=subscription.id,
        units_consumed=units,
        notes=notes,
        usage_type=usage_type,
    )
    session.add(usage)


def handle_checkout_session_completed(session: Session, event: stripe.Event) -> None:
    session_data: Dict[str, Any] = event["data"]["object"]
    subscription_id = session_data.get("subscription")
    if not subscription_id:
        logger.info("Checkout completed without subscription reference")
        return

    if not stripe.api_key:
        import os

        secret = os.getenv("STRIPE_SECRET_KEY")
        if not secret:
            logger.error("Stripe secret key not configured; cannot fetch subscription %s", subscription_id)
            return
        stripe.api_key = secret

    try:  # pragma: no cover - Stripe API call
        subscription_data = stripe.Subscription.retrieve(
            subscription_id,
            expand=["items"],
        )
    except Exception as exc:
        logger.error(
            "Unable to retrieve subscription %s after checkout: %s",
            subscription_id,
            exc,
        )
        return

    _process_subscription_payload(session, subscription_data)