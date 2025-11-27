import { StrategyPlan } from "../types";
import { formatPrice } from "../utils/planMath";

export type ScenarioCardsProps = {
  plan: StrategyPlan;
};

export function ScenarioCards({ plan }: ScenarioCardsProps) {
  const buyTargets = plan.buy_setup?.targets ?? [];
  const sellTargets = plan.sell_setup?.targets ?? [];
  const noTradeRanges = (plan.no_trade_zone ?? []).map((zone) => `${formatPrice(zone.min)} – ${formatPrice(zone.max)}`);

  const bullishCopy = `Buy only on strength through ${formatPrice(plan.buy_setup?.entry)} with risk parked at ${formatPrice(
    plan.buy_setup?.stop,
  )}. First targets sit at ${buyTargets.map((target) => formatPrice(target)).join(" then ") || "the posted levels"}.`;

  const bearishCopy = `Go short on a break under ${formatPrice(plan.sell_setup?.entry)} while defending ${formatPrice(
    plan.sell_setup?.stop,
  )}. Work covers into ${sellTargets.map((target) => formatPrice(target)).join(" then ") || "the marked downside"}.`;

  const noTradeCopy = noTradeRanges.length
    ? `Hold fire inside ${noTradeRanges.join(", ")}. Wait for price to leave the band with intent before committing risk.`
    : "No neutral band provided. Let price paint the next high-conviction level before stepping in.";

  return (
    <section className="rounded-3xl border border-slate-800 bg-slate-900 p-8 shadow-sm">
      <header className="mb-6 flex flex-col gap-1">
        <h2 className="text-lg font-semibold text-slate-50">Scenario Playbook</h2>
        <p className="text-sm text-slate-400">Trade paths to pre-plan before volatility hits</p>
      </header>
      <div className="grid gap-4 lg:grid-cols-3">
        <ScenarioCard tone="bullish" title="Bullish Path" highlight={`Buy trigger: ${formatPrice(plan.buy_setup?.entry)}`}
          body={bullishCopy} />
        <ScenarioCard tone="bearish" title="Bearish Path" highlight={`Sell trigger: ${formatPrice(plan.sell_setup?.entry)}`}
          body={bearishCopy} />
        <ScenarioCard tone="neutral" title="No-Trade" highlight={noTradeRanges[0] ?? "Stay patient"} body={noTradeCopy} />
      </div>
    </section>
  );
}

type Tone = "bullish" | "bearish" | "neutral";

type ScenarioCardProps = {
  tone: Tone;
  title: string;
  highlight: string;
  body: string;
};

const TONE_STYLES: Record<Tone, { badge: string; accent: string }> = {
  bullish: {
    badge: "bg-emerald-500/15 text-emerald-200 border border-emerald-400/40",
    accent: "text-emerald-300",
  },
  bearish: {
    badge: "bg-rose-500/15 text-rose-200 border border-rose-400/40",
    accent: "text-rose-300",
  },
  neutral: {
    badge: "bg-amber-500/15 text-amber-200 border border-amber-400/40",
    accent: "text-amber-300",
  },
};

function ScenarioCard({ tone, title, highlight, body }: ScenarioCardProps) {
  const style = TONE_STYLES[tone];
  return (
    <article className="flex h-full flex-col gap-4 rounded-2xl border border-slate-800/80 bg-slate-950/60 p-6 shadow-sm">
      <div className="flex items-center justify-between">
        <h3 className="text-base font-semibold text-slate-100">{title}</h3>
        <span className={`rounded-full px-3 py-1 text-xs font-semibold ${style.badge}`}>{tone.toUpperCase()}</span>
      </div>
      <p className={`text-sm font-semibold ${style.accent}`}>{highlight}</p>
      <p className="text-sm leading-relaxed text-slate-400">{body}</p>
    </article>
  );
}
