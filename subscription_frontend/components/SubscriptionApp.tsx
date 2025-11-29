import { useCallback, useEffect, useMemo, useState } from "react";
import type { BillingPlan, CurrentSubscription, SubscriptionBootstrap, SubscriptionPayload } from "../types";

type MessageTone = "info" | "success" | "error";

type PlanMeta = {
  badge?: string;
  tagline?: string;
  ctaLabel?: string;
  highlight?: string;
  features?: string[];
  logo?: string;
  cardTone: string;
  accentTone: string;
  chipTone: string;
};

type MessageState = {
  text: string;
  tone: MessageTone;
};

type SubscriptionAppProps = {
  bootstrap: SubscriptionBootstrap;
};

type QueryContext = {
  fromAnalyze: boolean;
  outOfRuns: boolean;
  upgradePrompt: boolean;
};

const PLAN_META: Record<string, PlanMeta> = {
  alpha: {
    badge: "Starter",
    tagline: "Silver-tier intelligence for confident entries.",
    ctaLabel: "Activate Alpha",
    highlight: "Essential daily signals",
    features: [
      "Multi-timeframe technical breakdowns",
      "Principal agent trade plans",
      "Email support during market hours",
    ],
    logo: "/static/Alpha.PNG",
    cardTone: "border-sky-400/40 bg-slate-950/70",
    accentTone: "bg-sky-500/15 text-sky-100",
    chipTone: "border-sky-500/40 text-sky-100",
  },
  sigma: {
    badge: "Most Popular",
    tagline: "Gold coverage for active swing & day traders.",
    ctaLabel: "Upgrade to Sigma",
    highlight: "Expanded AI run capacity",
    features: [
      "Priority queueing near market opens",
      "Advanced price-action scenario modeling",
      "Expanded monthly AI run allowance",
    ],
    logo: "/static/Sigma.PNG",
    cardTone: "border-emerald-400/40 bg-slate-950/70",
    accentTone: "bg-emerald-500/15 text-emerald-100",
    chipTone: "border-emerald-500/40 text-emerald-100",
  },
  omega: {
    badge: "Elite",
    tagline: "Black titanium access for elite strategy teams.",
    ctaLabel: "Go Omega",
    highlight: "Concierge agent coverage",
    features: [
      "Maximum monthly AI collaborations included",
      "Early access to experimental agents",
      "Quarterly concierge research briefings",
    ],
    logo: "/static/Omega.PNG",
    cardTone: "border-amber-400/40 bg-slate-950/70",
    accentTone: "bg-amber-500/15 text-amber-100",
    chipTone: "border-amber-500/40 text-amber-100",
  },
  default: {
    ctaLabel: "Select Plan",
    cardTone: "border-slate-700 bg-slate-950/60",
    accentTone: "bg-slate-700/40 text-slate-100",
    chipTone: "border-slate-600 text-slate-100",
  },
};

const currencyFormatter = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const runsFormatter = new Intl.NumberFormat("en-US");

function resolveMessage(context: QueryContext, subscription: CurrentSubscription | null): MessageState {
  const hasSubscription = Boolean(subscription?.plan?.slug);

  if (!hasSubscription) {
    if (context.outOfRuns) {
      return {
        text: "You have exhausted this cycle’s AI runs. Upgrade to continue immediately.",
        tone: "info",
      };
    }
    if (context.fromAnalyze) {
      return {
        text: "Activate a plan to launch your first VolatilX AI strategy run.",
        tone: "info",
      };
    }
    return {
      text: "Select a plan to unlock your next AI strategy run.",
      tone: "info",
    };
  }

  if (context.outOfRuns) {
    return {
      text: "Need more capacity? Upgrade for additional monthly AI runs.",
      tone: "info",
    };
  }

  if (context.upgradePrompt) {
    return {
      text: "Your current tier is nearly capped. Upgrade whenever you need more throughput.",
      tone: "info",
    };
  }

  return {
    text: "You can upgrade or downgrade at any time.",
    tone: "success",
  };
}

function useQueryContext(): QueryContext {
  return useMemo(() => {
    const params = new URLSearchParams(window.location.search);
    const reason = params.get("reason");
    const fromAnalyze = params.get("from") === "analyze" || reason === "subscription_required";
    const outOfRuns = params.get("status") === "out_of_runs" || reason === "quota_exhausted";
    const upgradePrompt = reason === "upgrade_prompt";

    return { fromAnalyze, outOfRuns, upgradePrompt };
  }, []);
}

function normalizePlans(value: unknown): BillingPlan[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((plan): plan is BillingPlan => !!plan && typeof plan === "object");
}

function safeSlug(plan: BillingPlan): string {
  return (plan.slug ?? "").toString().toLowerCase();
}

function PlanCard({
  plan,
  meta,
  isCurrent,
  onCheckout,
  disabled,
  pending,
}: {
  plan: BillingPlan;
  meta: PlanMeta;
  isCurrent: boolean;
  onCheckout: () => void;
  disabled: boolean;
  pending: boolean;
}) {
  const priceCents = typeof plan.monthly_price_cents === "number" ? plan.monthly_price_cents : null;
  const priceLabel = priceCents !== null ? currencyFormatter.format(priceCents / 100) : "Contact";
  const runsIncluded = typeof plan.ai_runs_included === "number" ? plan.ai_runs_included : null;
  const runsLabel = runsIncluded !== null ? `${runsFormatter.format(runsIncluded)} AI agent runs / month` : "Custom runway";
  const description = plan.description?.trim();
  const features = meta.features ?? [];

  return (
    <article
      className={`group relative flex h-full flex-col gap-6 rounded-3xl border px-6 py-7 shadow-lg shadow-black/30 transition duration-300 hover:-translate-y-1 hover:shadow-2xl ${meta.cardTone}`}
    >
      {meta.badge ? (
        <span
          className={`absolute right-6 top-6 inline-flex items-center rounded-full border px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.35em] ${meta.chipTone}`}
        >
          {meta.badge}
        </span>
      ) : null}
      <div className="space-y-4">
        <div className="flex items-center gap-3">
          {meta.logo ? (
            <span className="inline-flex h-12 w-12 items-center justify-center overflow-hidden rounded-2xl bg-slate-900/80">
              <img src={meta.logo} alt="" className="h-full w-full object-contain" loading="lazy" />
            </span>
          ) : null}
          <div className="flex flex-col">
            <span className="text-sm font-semibold uppercase tracking-[0.25em] text-slate-400">{plan.name ?? "VolatilX Plan"}</span>
            {meta.tagline ? <p className="text-sm text-slate-300">{meta.tagline}</p> : null}
          </div>
        </div>
        <div className="space-y-2">
          <div className="text-3xl font-semibold text-white">
            {priceLabel}
            <span className="ml-1 text-sm font-normal text-slate-400">/ month</span>
          </div>
          <p className="text-xs uppercase tracking-[0.3em] text-slate-400">{runsLabel}</p>
          {meta.highlight ? (
            <span className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-medium ${meta.accentTone}`}>
              <span className="h-1.5 w-1.5 rounded-full bg-current" aria-hidden="true" />
              {meta.highlight}
            </span>
          ) : null}
        </div>
        {description ? <p className="text-sm leading-relaxed text-slate-300">{description}</p> : null}
        {features.length ? (
          <ul className="space-y-2 text-sm text-slate-200">
            {features.map((feature) => (
              <li key={feature} className="flex items-start gap-2">
                <span className="mt-1 h-1.5 w-1.5 rounded-full bg-slate-400" aria-hidden="true" />
                <span>{feature}</span>
              </li>
            ))}
          </ul>
        ) : null}
      </div>
      <div className="mt-auto pt-2">
        <button
          type="button"
          onClick={onCheckout}
          disabled={disabled}
          className={`inline-flex w-full items-center justify-center rounded-2xl px-5 py-3 text-sm font-semibold uppercase tracking-wide transition ${
            isCurrent
              ? "cursor-default border border-slate-700 bg-slate-800/80 text-slate-300"
              : disabled
              ? "cursor-wait border border-slate-700 bg-slate-800/80 text-slate-300"
              : "border border-sky-500/50 bg-sky-500/20 text-sky-100 hover:border-sky-400 hover:bg-sky-400/30"
          }`}
        >
          {isCurrent ? "Current Plan" : pending ? "Redirecting…" : meta.ctaLabel ?? "Select Plan"}
        </button>
      </div>
    </article>
  );
}

function LoadingSkeleton() {
  return (
    <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-3">
      {Array.from({ length: 3 }).map((_, index) => (
        <div
          key={index}
          className="h-full rounded-3xl border border-slate-800/80 bg-slate-900/60 p-6 shadow-inner shadow-black/20"
        >
          <div className="h-10 w-32 rounded-full bg-slate-800/60" />
          <div className="mt-6 h-8 w-3/4 rounded-full bg-slate-800/60" />
          <div className="mt-4 space-y-2">
            <div className="h-3 w-full rounded-full bg-slate-800/60" />
            <div className="h-3 w-5/6 rounded-full bg-slate-800/60" />
            <div className="h-3 w-2/3 rounded-full bg-slate-800/60" />
          </div>
          <div className="mt-8 h-10 w-full rounded-2xl bg-slate-800/60" />
        </div>
      ))}
    </div>
  );
}

export function SubscriptionApp({ bootstrap }: SubscriptionAppProps) {
  const queryContext = useQueryContext();
  const [plans, setPlans] = useState<BillingPlan[] | null>(null);
  const [currentSubscription, setCurrentSubscription] = useState<CurrentSubscription | null>(
    bootstrap.currentSubscription ?? null,
  );
  const [message, setMessage] = useState<MessageState>(() => resolveMessage(queryContext, bootstrap.currentSubscription ?? null));
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "error">("loading");
  const [pendingSlug, setPendingSlug] = useState<string | null>(null);

  const currentPlanSlug = (currentSubscription?.plan?.slug ?? "").toLowerCase();

  const loadPlans = useCallback(async () => {
    setStatus("loading");
    setPendingSlug(null);

    try {
      const response = await fetch("/api/billing/plans", { headers: { Accept: "application/json" } });
      if (!response.ok) {
        throw new Error("Failed to load subscription plans.");
      }
      const payload = (await response.json()) as SubscriptionPayload;
      const normalizedPlans = normalizePlans(payload?.plans);
      const nextCurrent = payload?.current_subscription ?? bootstrap.currentSubscription ?? null;

      setPlans(normalizedPlans);
      setCurrentSubscription(nextCurrent);
      setMessage(resolveMessage(queryContext, nextCurrent));
      setStatus("ready");
    } catch (error) {
      console.warn("[Subscription] Failed to load plans", error);
      setPlans([]);
      setStatus("error");
      setMessage({
        text: error instanceof Error ? error.message : "Unexpected error loading billing information.",
        tone: "error",
      });
    }
  }, [bootstrap.currentSubscription, queryContext]);

  useEffect(() => {
    loadPlans();
  }, [loadPlans]);

  const handleCheckout = useCallback(
    async (plan: BillingPlan) => {
      const slug = safeSlug(plan);
      if (!slug || slug === currentPlanSlug) {
        return;
      }

      setPendingSlug(slug);
      setMessage((prev) => (prev.tone === "error" ? resolveMessage(queryContext, currentSubscription) : prev));

      try {
        const response = await fetch("/api/billing/checkout-session", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ plan_slug: slug }),
        });

        if (!response.ok) {
          const detail = await response.json().catch(() => ({}));
          const messageText = typeof detail?.detail === "string" ? detail.detail : "Unable to start checkout. Please try again.";
          throw new Error(messageText);
        }

        const data = await response.json();
        if (data?.checkout_url) {
          window.location.href = data.checkout_url as string;
          return;
        }

        throw new Error("Checkout session did not return a redirect URL.");
      } catch (error) {
        console.warn("[Subscription] Checkout failed", error);
        setPendingSlug(null);
        setMessage({
          text: error instanceof Error ? error.message : "Billing error. Please try again later.",
          tone: "error",
        });
      }
    },
    [currentPlanSlug, currentSubscription, queryContext],
  );

  const renderPlans = () => {
    if (status === "loading") {
      return <LoadingSkeleton />;
    }

    if (!plans || plans.length === 0) {
      return (
        <div className="rounded-3xl border border-slate-800 bg-slate-950/60 p-10 text-center text-sm text-slate-300">
          No plans are available right now. Please check back later.
        </div>
      );
    }

    const order = ["alpha", "sigma", "omega"];
    const sortedPlans = [...plans].sort((a, b) => {
      const aSlug = safeSlug(a);
      const bSlug = safeSlug(b);
      const aIndex = order.indexOf(aSlug);
      const bIndex = order.indexOf(bSlug);
      if (aIndex === -1 && bIndex === -1) {
        const aPrice = typeof a.monthly_price_cents === "number" ? a.monthly_price_cents : Number.MAX_SAFE_INTEGER;
        const bPrice = typeof b.monthly_price_cents === "number" ? b.monthly_price_cents : Number.MAX_SAFE_INTEGER;
        return aPrice - bPrice;
      }
      if (aIndex === -1) return 1;
      if (bIndex === -1) return -1;
      return aIndex - bIndex;
    });

    return (
      <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-3">
        {sortedPlans.map((plan, index) => {
          const slug = safeSlug(plan);
          const meta = PLAN_META[slug] ?? PLAN_META.default;
          const isCurrent = slug === currentPlanSlug;
          const checkoutDisabled = isCurrent || pendingSlug === slug || plan.stripe_price_configured === false;

          return (
            <PlanCard
              key={plan.id ?? slug ?? plan.name ?? `plan-${index}`}
              plan={plan}
              meta={meta}
              isCurrent={isCurrent}
              onCheckout={() => handleCheckout(plan)}
              disabled={checkoutDisabled}
              pending={pendingSlug === slug}
            />
          );
        })}
      </div>
    );
  };

  return (
    <div className="mx-auto flex w-full max-w-7xl flex-col gap-10 px-6 py-10 lg:py-14">
      <section className="rounded-3xl border border-slate-800 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 px-8 py-12 shadow-2xl shadow-black/40">
        <div className="flex flex-col gap-6 text-center">
          <div className="inline-flex items-center gap-2 self-center rounded-full border border-slate-700/60 bg-slate-900/70 px-4 py-1 text-[11px] font-semibold uppercase tracking-[0.4em] text-slate-200/90">
            <span className="h-1.5 w-1.5 rounded-full bg-sky-400" aria-hidden="true" />
            Premium Intelligence
          </div>
          <h1 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">
            Unlock VolatilX premium intelligence tailored to your trading desk.
          </h1>
          <p className="mx-auto max-w-3xl text-base leading-relaxed text-slate-300">
            Every tier connects you with multi-agent strategy orchestration, prioritized AI workflows, and the same market
            playbooks used by elite desks. Scale up or down whenever your cadence shifts.
          </p>
        </div>
      </section>

      {message ? (
        <div
          className={`rounded-3xl border px-5 py-4 text-sm font-medium shadow-inner shadow-black/30 ${
            message.tone === "error"
              ? "border-rose-500/40 bg-rose-500/10 text-rose-100"
              : message.tone === "success"
              ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-100"
              : "border-sky-500/40 bg-sky-500/10 text-sky-100"
          }`}
          role="status"
          aria-live="polite"
        >
          {message.text}
        </div>
      ) : null}

      <section className="space-y-6">
        <header className="flex flex-col gap-2">
          <h2 className="text-xl font-semibold text-white">Plans engineered for every cadence</h2>
          <p className="text-sm text-slate-300">
            Compare allocation, agent throughput, and premium concierge options. You can switch tiers instantly.
          </p>
        </header>
        {renderPlans()}
      </section>

      <section className="grid gap-6 rounded-3xl border border-slate-800 bg-slate-950/70 p-8 shadow-inner shadow-black/40 lg:grid-cols-[1.1fr_0.9fr]">
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-white">Transparent billing, zero lock-in</h3>
          <p className="text-sm leading-relaxed text-slate-300">
            Your subscription renews monthly with prorated upgrades. Cancel anytime from your billing portal. Every checkout is
            secured through Stripe, and you will see the exact charge before confirming.
          </p>
          <ul className="space-y-3 text-sm text-slate-200">
            <li className="flex items-start gap-2">
              <span className="mt-1 h-1.5 w-1.5 rounded-full bg-emerald-400" aria-hidden="true" />
              PCI-compliant checkout backed by Stripe Billing.
            </li>
            <li className="flex items-start gap-2">
              <span className="mt-1 h-1.5 w-1.5 rounded-full bg-sky-400" aria-hidden="true" />
              Instant upgrades with prorated charges and cancellation credits.
            </li>
            <li className="flex items-start gap-2">
              <span className="mt-1 h-1.5 w-1.5 rounded-full bg-amber-400" aria-hidden="true" />
              Concierge support for enterprise desks on Omega.
            </li>
          </ul>
        </div>
        <div className="rounded-3xl border border-slate-800 bg-slate-950/60 p-6 text-sm text-slate-300">
          <h4 className="text-base font-semibold text-white">Need a bespoke package?</h4>
          <p className="mt-2 leading-relaxed">
            If your desk needs deeper integration, white-glove research, or team-wide access, ping us through the in-app contact
            widget. We can stage a custom enterprise rollout in under 48 hours.
          </p>
          <div className="mt-4 inline-flex items-center gap-2 rounded-full border border-slate-700/70 bg-slate-900/70 px-3 py-1 text-xs uppercase tracking-[0.25em] text-slate-400">
            <span className="h-1.5 w-1.5 rounded-full bg-rose-400" aria-hidden="true" />
            Concierge desk
          </div>
        </div>
      </section>
    </div>
  );
}
