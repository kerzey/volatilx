import { StrategyPlan } from "../types";
import { clampToRange, formatPercent, formatPrice } from "../utils/planMath";

type PnLPreviewProps = {
  plan: StrategyPlan;
  latestPrice: number;
};

type LongStats = {
  entry: number;
  stop: number;
  target1: number;
  target2?: number;
  moveFromEntryPct: number;
  progressToTarget1: number;
  progressToTarget2?: number;
  progressToStop: number;
};

type ShortStats = {
  entry: number;
  stop: number;
  target1: number;
  target2?: number;
  moveFromEntryPct: number;
  progressToTarget1: number;
  progressToTarget2?: number;
  progressToStop: number;
};

export function PnLPreview({ plan, latestPrice }: PnLPreviewProps) {
  const longStats = computeLongStats(plan, latestPrice);
  const shortStats = computeShortStats(plan, latestPrice);

  if (!longStats && !shortStats) {
    return null;
  }

  return (
    <section className="rounded-3xl border border-slate-800 bg-slate-900 p-8 shadow-sm">
      <header className="mb-6 flex flex-col gap-1">
        <h2 className="text-lg font-semibold text-slate-50">P/L Preview</h2>
        <p className="text-sm text-slate-400">Quick sanity check on reward versus risk for both sides</p>
      </header>
      <div className="grid gap-6 lg:grid-cols-2">
        {longStats ? <SideCard tone="bullish" title="Long Path" stats={longStats} latestPrice={latestPrice} /> : null}
        {shortStats ? <SideCard tone="bearish" title="Short Path" stats={shortStats} latestPrice={latestPrice} /> : null}
      </div>
    </section>
  );
}

type Tone = "bullish" | "bearish";

type SideCardProps = {
  tone: Tone;
  title: string;
  stats: LongStats | ShortStats;
  latestPrice: number;
};

function SideCard({ tone, title, stats, latestPrice }: SideCardProps) {
  const accent = tone === "bullish" ? "text-emerald-300" : "text-rose-300";
  const barTone = tone === "bullish" ? "bg-emerald-500" : "bg-rose-500";
  const stopTone = "bg-amber-400";
  const signedMove = formatPercent(stats.moveFromEntryPct, 1);
  const target2Copy = typeof stats.progressToTarget2 === "number" ? `${Math.round(clampToRange(stats.progressToTarget2))}% of the way to Target 2, ` : "";
  const stopDistance = 100 - Math.round(clampToRange(stats.progressToStop));

  const entryPrice = formatPrice(stats.entry);
  const targetOne = formatPrice(stats.target1);
  const stopPrice = formatPrice(stats.stop);

  const narrative = tone === "bullish"
    ? `If long from ${entryPrice}, price is ${signedMove} versus entry, ${Math.round(clampToRange(stats.progressToTarget1))}% of the push toward Target 1, ${target2Copy}${stopDistance}% of the stop buffer still intact.`
    : `If short from ${entryPrice}, price is ${signedMove} versus entry, ${Math.round(clampToRange(stats.progressToTarget1))}% of the path toward Target 1, ${target2Copy}${stopDistance}% of the stop buffer still open.`;

  return (
    <article className="flex h-full flex-col gap-4 rounded-2xl border border-slate-800/80 bg-slate-950/60 p-6 shadow-sm">
      <header className="flex items-center justify-between">
        <h3 className="text-base font-semibold text-slate-100">{title}</h3>
        <span className={`rounded-full px-3 py-1 text-xs font-semibold ${accent}`}>{tone === "bullish" ? "LONG" : "SHORT"}</span>
      </header>
      <p className={`text-sm leading-relaxed ${accent}`}>{narrative}</p>
      <div className="space-y-4 text-xs font-medium text-slate-300">
        <ProgressBar label="Entry → Target 1" percent={clampToRange(stats.progressToTarget1)} toneClass={barTone} />
        {typeof stats.progressToTarget2 === "number" ? (
          <ProgressBar label="Entry → Target 2" percent={clampToRange(stats.progressToTarget2)} toneClass={barTone} subtle />
        ) : null}
        <ProgressBar label="Entry → Stop" percent={clampToRange(stats.progressToStop)} toneClass={stopTone} />
        <div className="flex items-center justify-between text-[11px] text-slate-400">
          <span>Entry {entryPrice}</span>
          <span>Last {formatPrice(latestPrice)}</span>
          <span>Stop {stopPrice}</span>
        </div>
      </div>
    </article>
  );
}

type ProgressBarProps = {
  label: string;
  percent: number;
  toneClass: string;
  subtle?: boolean;
};

function ProgressBar({ label, percent, toneClass, subtle }: ProgressBarProps) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-[11px] uppercase tracking-wide text-slate-500">
        <span>{label}</span>
        <span>{Math.round(clampToRange(percent))}%</span>
      </div>
      <div className={`h-2 w-full rounded-full ${subtle ? "bg-slate-900" : "bg-slate-800"}`}>
        <div
          className={`h-2 rounded-full ${toneClass}`}
          style={{ width: `${clampToRange(percent)}%` }}
        />
      </div>
    </div>
  );
}

function computeLongStats(plan: StrategyPlan, latestPrice: number): LongStats | undefined {
  const entry = safeNumber(plan.buy_setup?.entry);
  const stop = safeNumber(plan.buy_setup?.stop);
  const targets = plan.buy_setup?.targets ?? [];
  const target1 = safeNumber(targets[0]);
  const target2 = safeNumber(targets[1]);

  if (![entry, stop, target1].every(Number.isFinite) || entry === stop || target1 === entry) {
    return undefined;
  }

  const moveFromEntryPct = ((latestPrice - entry) / entry) * 100;
  const progressToTarget1 = ((latestPrice - entry) / (target1 - entry)) * 100;
  const progressToTarget2 = Number.isFinite(target2) && target2 !== entry ? ((latestPrice - entry) / (target2 - entry)) * 100 : undefined;
  const progressToStop = ((entry - latestPrice) / (entry - stop)) * 100;

  return {
    entry,
    stop,
    target1,
    target2: Number.isFinite(target2) ? target2 : undefined,
    moveFromEntryPct,
    progressToTarget1,
    progressToTarget2,
    progressToStop,
  };
}

function computeShortStats(plan: StrategyPlan, latestPrice: number): ShortStats | undefined {
  const entry = safeNumber(plan.sell_setup?.entry);
  const stop = safeNumber(plan.sell_setup?.stop);
  const targets = plan.sell_setup?.targets ?? [];
  const target1 = safeNumber(targets[0]);
  const target2 = safeNumber(targets[1]);

  if (![entry, stop, target1].every(Number.isFinite) || entry === stop || target1 === entry) {
    return undefined;
  }

  const moveFromEntryPct = ((entry - latestPrice) / entry) * 100;
  const progressToTarget1 = ((entry - latestPrice) / (entry - target1)) * 100;
  const progressToTarget2 = Number.isFinite(target2) && target2 !== entry ? ((entry - latestPrice) / (entry - target2)) * 100 : undefined;
  const progressToStop = ((latestPrice - entry) / (stop - entry)) * 100;

  return {
    entry,
    stop,
    target1,
    target2: Number.isFinite(target2) ? target2 : undefined,
    moveFromEntryPct,
    progressToTarget1,
    progressToTarget2,
    progressToStop,
  };
}

function safeNumber(value: unknown): number {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : NaN;
}
