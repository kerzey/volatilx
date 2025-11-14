"""Static subscription plan catalogue and sync helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from sqlalchemy.orm import Session

from models import SubscriptionPlan


@dataclass(frozen=True)
class PlanDefinition:
    slug: str
    name: str
    monthly_price_cents: int
    ai_runs_included: int
    description: str
    stripe_price_id: Optional[str] = None


PLAN_DEFINITIONS: List[PlanDefinition] = [
    PlanDefinition(
        slug="trial",
        name="Starter Trial",
        monthly_price_cents=0,
        ai_runs_included=10,
        stripe_price_id=None,
        description="Complimentary first-month access with 10 AI analyses for new users.",
    ),
    PlanDefinition(
        slug="alpha",
        name="Alpha",
        monthly_price_cents=1999,
        ai_runs_included=100,
        stripe_price_id="price_1SSjVGC0H4HP0kcQgw0gsdzK",
        description="Entry plan with 100 AI strategy runs per billing cycle.",
    ),
    PlanDefinition(
        slug="sigma",
        name="Sigma",
        monthly_price_cents=4499,
        ai_runs_included=240,
        stripe_price_id="price_1SSjWzC0H4HP0kcQmAYFjXwn",
        description="Balanced tier for active traders with 240 AI runs each month.",
    ),
    PlanDefinition(
        slug="omega",
        name="Omega",
        monthly_price_cents=8999,
        ai_runs_included=500,
        stripe_price_id="price_1SSjXfC0H4HP0kcQxtPM9WuD",
        description="Advanced plan unlocking 500 AI runs and priority processing.",
    ),
]


def get_plan_definitions() -> Iterable[PlanDefinition]:
    """Return the immutable list of supported plan definitions."""

    return tuple(PLAN_DEFINITIONS)


def sync_plan_catalogue(
    session: Session,
    *,
    allow_updates: bool = True,
    commit: bool = True,
) -> List[SubscriptionPlan]:
    """Ensure each plan definition exists in the database.

    Parameters
    ----------
    session:
        Open SQLAlchemy session.
    allow_updates:
        When True, existing plans will be updated to match definitions
        (excluding unset Stripe price identifiers).
    commit:
        Whether to commit the session before returning.

    Returns
    -------
    List[SubscriptionPlan]
        The persisted plan records, ordered like ``PLAN_DEFINITIONS``.
    """

    # Ensure dependent SQLAlchemy mappers (e.g., User) are registered before querying.
    try:  # pragma: no cover - defensive import for mapper configuration
        import user  # noqa: F401
    except Exception:
        pass

    persisted: List[SubscriptionPlan] = []

    for definition in PLAN_DEFINITIONS:
        plan = (
            session.query(SubscriptionPlan)
            .filter(SubscriptionPlan.slug == definition.slug)
            .one_or_none()
        )

        if plan is None:
            plan = SubscriptionPlan(
                slug=definition.slug,
                name=definition.name,
                description=definition.description,
                monthly_price_cents=definition.monthly_price_cents,
                ai_runs_included=definition.ai_runs_included,
                stripe_price_id=definition.stripe_price_id,
                is_active=True,
            )
            session.add(plan)
        elif allow_updates:
            plan.name = definition.name
            plan.description = definition.description
            plan.monthly_price_cents = definition.monthly_price_cents
            plan.ai_runs_included = definition.ai_runs_included
            if definition.stripe_price_id:
                plan.stripe_price_id = definition.stripe_price_id
            plan.is_active = True

        persisted.append(plan)

    if commit:
        session.commit()
    else:
        session.flush()

    return persisted
