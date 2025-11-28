import type { SignInBootstrap } from "../types";

type IconProps = { className?: string };

const SparkIcon = ({ className = "h-5 w-5" }: IconProps) => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    stroke="currentColor"
    strokeWidth={1.5}
    className={className}
  >
    <circle cx={12} cy={12} r={3.75} />
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 3.75V6.5m0 11v2.75m8.25-8.25H17.5M6.5 12H3.75m14.116-6.366l-1.944 1.944M7.078 16.884l-1.944 1.944m12.25 0l-1.944-1.944M7.078 7.116 5.134 5.172" />
  </svg>
);

const NetworkIcon = ({ className = "h-5 w-5" }: IconProps) => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    stroke="currentColor"
    strokeWidth={1.5}
    className={className}
  >
    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12a3 3 0 100-6 3 3 0 000 6z" />
    <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 12a3 3 0 100-6 3 3 0 000 6z" />
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M18 19.5v-1.125A2.625 2.625 0 0015.375 15.75h-6.75A2.625 2.625 0 006 18.375V19.5"
    />
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M20.25 19.5v-1.125a2.625 2.625 0 00-2.625-2.625h-1.5"
    />
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M3.75 19.5v-1.125a2.625 2.625 0 012.625-2.625h1.5"
    />
  </svg>
);

const BlueprintIcon = ({ className = "h-5 w-5" }: IconProps) => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    stroke="currentColor"
    strokeWidth={1.5}
    className={className}
  >
    <rect x={4.5} y={4.5} width={6} height={6} rx={1.25} />
    <rect x={13.5} y={4.5} width={6} height={6} rx={1.25} />
    <rect x={4.5} y={13.5} width={6} height={6} rx={1.25} />
    <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 13.5v6m-3-3h6" />
  </svg>
);

const ShieldIcon = ({ className = "h-5 w-5" }: IconProps) => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    stroke="currentColor"
    strokeWidth={1.5}
    className={className}
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M12 3.75L4.5 6.75v4.5c0 4.97 3.134 9.674 7.5 11.25 4.366-1.576 7.5-6.28 7.5-11.25v-4.5L12 3.75z"
    />
    <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 12.75l1.875 1.875 2.625-3.75" />
  </svg>
);

const ChartIcon = ({ className = "h-5 w-5" }: IconProps) => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    stroke="currentColor"
    strokeWidth={1.5}
    className={className}
  >
    <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 19.5h16.5" />
    <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 15.75V12m4.5 3.75V9.75m4.5 6V6" />
  </svg>
);

type FeatureCard = {
  title: string;
  description: string;
  Icon: (props: IconProps) => JSX.Element;
  cardTone: string;
  iconTone: string;
};

const FEATURE_CARDS: FeatureCard[] = [
  {
    title: "Instant Insights",
    description: "Get AI-curated market context the moment you log in so you never trade in the dark.",
    Icon: SparkIcon,
    cardTone: "border-sky-500/35 bg-gradient-to-br from-sky-500/15 via-slate-950/70 to-slate-950/80",
    iconTone: "border-sky-500/40 bg-sky-500/15 text-sky-100",
  },
  {
    title: "Agent Collaboration",
    description: "Leverage multi-agent strategies that adapt to your intent, timeframe, and risk profile.",
    Icon: NetworkIcon,
    cardTone: "border-emerald-500/35 bg-gradient-to-br from-emerald-500/12 via-slate-950/70 to-slate-950/80",
    iconTone: "border-emerald-500/35 bg-emerald-500/15 text-emerald-100",
  },
  {
    title: "Strategy Library",
    description: "Browse playbooks for day, swing, and momentum flows, complete with ready-to-execute setups.",
    Icon: BlueprintIcon,
    cardTone: "border-amber-500/35 bg-gradient-to-br from-amber-500/12 via-slate-950/70 to-slate-950/80",
    iconTone: "border-amber-500/35 bg-amber-500/15 text-amber-100",
  },
  {
    title: "Secure Access",
    description: "OAuth-backed authentication keeps your workspace protected across sessions and devices.",
    Icon: ShieldIcon,
    cardTone: "border-rose-500/35 bg-gradient-to-br from-rose-500/12 via-slate-950/70 to-slate-950/80",
    iconTone: "border-rose-500/35 bg-rose-500/15 text-rose-100",
  },
  {
    title: "Data Clarity",
    description: "Track alerts, PnL, and risk in one command center so you can execute with confidence.",
    Icon: ChartIcon,
    cardTone: "border-indigo-500/35 bg-gradient-to-br from-indigo-500/12 via-slate-950/70 to-slate-950/80",
    iconTone: "border-indigo-500/35 bg-indigo-500/15 text-indigo-100",
  },
];

export type SignInAppProps = {
  bootstrap: SignInBootstrap;
};

export function SignInApp({ bootstrap }: SignInAppProps) {
  const loginUrl = bootstrap.loginUrl ?? "/auth/google/login";
  const videoUrl = bootstrap.videoEmbedUrl ?? "https://www.youtube.com/embed/HZUbGpUUXL4";

  return (
    <div className="mx-auto max-w-7xl px-6 pb-16 pt-10 lg:pb-24 lg:pt-14">
      <section className="relative overflow-hidden rounded-3xl border border-slate-800 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 px-8 py-12 shadow-lg shadow-black/30 lg:px-12 lg:py-16">
        <div className="pointer-events-none absolute inset-0">
          <div className="absolute -left-28 top-0 h-64 w-64 rounded-full bg-sky-500/20 blur-3xl" aria-hidden="true" />
          <div className="absolute -right-24 bottom-0 h-72 w-72 rounded-full bg-emerald-500/15 blur-3xl" aria-hidden="true" />
        </div>
        <div className="relative grid gap-10 lg:grid-cols-[1.1fr_0.9fr] lg:items-center">
          <div className="space-y-6">
            <span className="inline-flex items-center gap-2 rounded-full border border-slate-700/70 bg-slate-900/70 px-4 py-1 text-xs font-semibold uppercase tracking-[0.25em] text-slate-200/90">
              <span className="h-2 w-2 rounded-full bg-sky-400" aria-hidden="true" />
              Secure Entry
            </span>
            <h1 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">
              Rejoin your VolatilX cockpit and sync with the latest AI drops.
            </h1>
            <p className="max-w-2xl text-base leading-relaxed text-slate-200">
              Log in with Google OAuth to unlock your personalized report center, action plans, and live AI agents. We keep
              enterprise-grade security in play so you can focus on the next trade.
            </p>
            <div className="flex flex-wrap items-center gap-4">
              <a
                className="inline-flex items-center gap-3 rounded-2xl border border-sky-500/60 bg-sky-500/20 px-5 py-3 text-sm font-semibold uppercase tracking-wide text-sky-100 transition hover:border-sky-400 hover:bg-sky-400/30"
                href={loginUrl}
              >
                <span className="inline-flex h-8 w-8 items-center justify-center rounded-xl bg-sky-500/30 text-lg font-bold text-white">
                  G
                </span>
                Continue with Google
              </a>
              <span className="flex items-center gap-2 text-xs uppercase tracking-wide text-slate-400">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" aria-hidden="true" />
                SSO encrypted · SOC2 aligned
              </span>
            </div>
          </div>
          <div className="relative overflow-hidden rounded-3xl border border-slate-800 bg-slate-950/80 shadow-xl shadow-black/40">
            <iframe
              title="VolatilX overview"
              src={videoUrl}
              loading="lazy"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
              allowFullScreen
              className="h-64 w-full rounded-3xl bg-slate-900 sm:h-72 lg:h-[320px]"
            />
          </div>
        </div>
      </section>

      <section className="mt-16 grid gap-8 lg:grid-cols-[0.85fr_1.15fr]">
        <div className="flex flex-col gap-6 rounded-3xl border border-slate-800 bg-slate-950/70 p-8 shadow-inner shadow-black/40">
          <div className="space-y-3">
            <span className="text-xs uppercase tracking-[0.3em] text-slate-500">Inside your workspace</span>
            <h2 className="text-2xl font-semibold text-white">What unlocks when you sign in</h2>
            <p className="text-sm leading-relaxed text-slate-300">
              Your dashboard synchronizes with the latest AI agent output, favorite tickers, and strategy templates tailored to
              your trading cadence.
            </p>
          </div>
          <ul className="space-y-4 text-sm text-slate-200">
            <li className="flex items-start gap-3 rounded-2xl border border-slate-800 bg-slate-900/80 px-4 py-3">
              <span className="mt-1 h-2 w-2 flex-shrink-0 rounded-full bg-sky-400" aria-hidden="true" />
              Auto-refreshing action center with traffic light alerts, bias meters, and gauge visualizations.
            </li>
            <li className="flex items-start gap-3 rounded-2xl border border-slate-800 bg-slate-900/80 px-4 py-3">
              <span className="mt-1 h-2 w-2 flex-shrink-0 rounded-full bg-emerald-400" aria-hidden="true" />
              Report center drops prioritized around your favorite symbols, capped for clarity but instantly searchable.
            </li>
            <li className="flex items-start gap-3 rounded-2xl border border-slate-800 bg-slate-900/80 px-4 py-3">
              <span className="mt-1 h-2 w-2 flex-shrink-0 rounded-full bg-amber-400" aria-hidden="true" />
              Access to evolving plan switchers, consensus snapshots, and plan math utilities that mirror the pro tooling.
            </li>
          </ul>
        </div>
        <div className="grid gap-6 sm:grid-cols-2">
          {FEATURE_CARDS.map(({ title, description, Icon, cardTone, iconTone }) => (
            <article
              key={title}
              className={`flex flex-col gap-4 rounded-3xl border ${cardTone} p-6 shadow-inner shadow-black/30 transition hover:border-white/30`}
            >
              <span
                className={`inline-flex h-11 w-11 items-center justify-center rounded-2xl border ${iconTone} shadow-lg shadow-black/20`}
              >
                <Icon className="h-5 w-5" />
              </span>
              <h3 className="text-base font-semibold tracking-tight text-white">{title}</h3>
              <p className="text-sm leading-relaxed text-slate-300">{description}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="mt-16 rounded-3xl border border-slate-800 bg-slate-950/60 p-8 text-center shadow-inner shadow-black/40">
        <h2 className="text-xl font-semibold text-white">Need a quick walkthrough?</h2>
        <p className="mt-2 text-sm text-slate-300">
          Drop us a line via the in-app contact widget and we will beam over a tailored onboarding sequence.
        </p>
      </section>
    </div>
  );
}
