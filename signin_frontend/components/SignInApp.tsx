import type { SignInBootstrap } from "../types";

const FEATURE_CARDS = [
  {
    code: "IN",
    title: "Instant Insights",
    description: "Get AI-curated market context the moment you log in so you never trade in the dark.",
    accent: "bg-sky-500/10 border-sky-500/40 text-sky-100",
  },
  {
    code: "AI",
    title: "Agent Collaboration",
    description: "Leverage multi-agent strategies that adapt to your intent, timeframe, and risk profile.",
    accent: "bg-emerald-500/10 border-emerald-500/40 text-emerald-100",
  },
  {
    code: "ST",
    title: "Strategy Library",
    description: "Browse playbooks for day, swing, and momentum flows, complete with ready-to-execute setups.",
    accent: "bg-amber-500/10 border-amber-500/40 text-amber-100",
  },
  {
    code: "SEC",
    title: "Secure Access",
    description: "OAuth-backed authentication keeps your workspace protected across sessions and devices.",
    accent: "bg-rose-500/10 border-rose-500/40 text-rose-100",
  },
  {
    code: "DX",
    title: "Data Clarity",
    description: "Track alerts, PnL, and risk in one command center so you can execute with confidence.",
    accent: "bg-indigo-500/10 border-indigo-500/40 text-indigo-100",
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
          {FEATURE_CARDS.map((feature) => (
            <article
              key={feature.title}
              className={`flex flex-col gap-3 rounded-3xl border ${feature.accent} bg-slate-950/70 p-6 shadow-inner shadow-black/30`}
            >
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-xl border border-white/10 bg-white/10 text-sm font-semibold uppercase tracking-wide text-white/90">
                {feature.code}
              </span>
              <h3 className="text-lg font-semibold text-white">{feature.title}</h3>
              <p className="text-sm leading-relaxed text-slate-300">{feature.description}</p>
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
