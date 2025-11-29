export type BillingPlan = {
  id?: string;
  slug?: string | null;
  name?: string | null;
  description?: string | null;
  monthly_price_cents?: number | null;
  ai_runs_included?: number | null;
  stripe_price_configured?: boolean | null;
};

export type SubscriptionPlanDetails = {
  slug?: string | null;
};

export type CurrentSubscription = {
  plan?: SubscriptionPlanDetails | null;
};

export type SubscriptionPayload = {
  plans: BillingPlan[];
  current_subscription?: CurrentSubscription | null;
};

export type SubscriptionBootstrap = {
  currentSubscription?: CurrentSubscription | null;
};
