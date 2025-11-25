from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func


class TimestampMixin:
	"""Reusable timestamp columns for created/updated tracking."""

	created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
	updated_at = Column(
		DateTime(timezone=True),
		nullable=False,
		server_default=func.now(),
		onupdate=func.now(),
	)

Base = declarative_base()


class SubscriptionPlan(TimestampMixin, Base):
	"""Static catalogue of available subscription plans."""

	__tablename__ = "subscription_plans"

	id = Column(Integer, primary_key=True, index=True)
	slug = Column(String(50), unique=True, nullable=False)
	name = Column(String(100), nullable=False)
	description = Column(Text, nullable=True)
	monthly_price_cents = Column(Integer, nullable=False)
	ai_runs_included = Column(Integer, nullable=False)
	stripe_price_id = Column(String(128), nullable=True)
	is_active = Column(Boolean, nullable=False, default=True)

	subscriptions = relationship(
		"UserSubscription",
		back_populates="plan",
		cascade="all, delete-orphan",
	)

	def __repr__(self) -> str:  # pragma: no cover - debug helper
		return (
			f"<SubscriptionPlan slug={self.slug!r} name={self.name!r} "
			f"price_cents={self.monthly_price_cents} runs={self.ai_runs_included}>"
		)


class UserSubscription(TimestampMixin, Base):
	"""Active subscription assigned to a user."""

	__tablename__ = "user_subscriptions"

	id = Column(Integer, primary_key=True, index=True)
	user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
	plan_id = Column(Integer, ForeignKey("subscription_plans.id"), nullable=False)
	stripe_customer_id = Column(String(255), nullable=True)
	stripe_subscription_id = Column(String(255), nullable=True, unique=True)
	status = Column(String(50), nullable=False, default="pending")
	current_period_start = Column(DateTime(timezone=True), nullable=True)
	current_period_end = Column(DateTime(timezone=True), nullable=True)
	runs_remaining = Column(Integer, nullable=False, default=0)
	auto_renew = Column(Boolean, nullable=False, default=True)
	cancel_at_period_end = Column(Boolean, nullable=False, default=False)

	plan = relationship("SubscriptionPlan", back_populates="subscriptions")
	user = relationship("User", backref="subscriptions")
	usage_entries = relationship(
		"SubscriptionUsage",
		back_populates="subscription",
		cascade="all, delete-orphan",
	)

	def __repr__(self) -> str:  # pragma: no cover - debug helper
		return (
			f"<UserSubscription user_id={self.user_id} plan={self.plan_id} "
			f"status={self.status!r} runs_remaining={self.runs_remaining}>"
		)


class SubscriptionUsage(TimestampMixin, Base):
	"""Ledger of subscription consumption events (e.g., AI analysis runs)."""

	__tablename__ = "subscription_usage"

	id = Column(Integer, primary_key=True, index=True)
	subscription_id = Column(
		Integer,
		ForeignKey("user_subscriptions.id"),
		nullable=False,
		index=True,
	)
	usage_type = Column(String(50), nullable=False, default="ai_run")
	units_consumed = Column(Integer, nullable=False, default=1)
	notes = Column(Text, nullable=True)

	subscription = relationship("UserSubscription", back_populates="usage_entries")

	def __repr__(self) -> str:  # pragma: no cover - debug helper
		return (
			f"<SubscriptionUsage subscription_id={self.subscription_id} "
			f"type={self.usage_type!r} units={self.units_consumed}>"
		)


class UserFavoriteSymbol(TimestampMixin, Base):
	"""Symbols a user has elected to follow within Action Center."""

	__tablename__ = "user_favorite_symbols"
	__table_args__ = (UniqueConstraint("user_id", "symbol", name="uq_user_symbol"),)

	id = Column(Integer, primary_key=True, index=True)
	user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
	symbol = Column(String(24), nullable=False)

	user = relationship("User", backref="favorite_symbols")

	def __repr__(self) -> str:  # pragma: no cover - debug helper
		return f"<UserFavoriteSymbol user_id={self.user_id} symbol={self.symbol!r}>"