import { useEffect, useRef } from "react";
import { StrategyPlan } from "../types";
import { clampToRange, deriveTradeState, formatPercent, formatPrice } from "../utils/planMath";

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
  entryTriggered: boolean;
  entryDistanceValue: number;
  entryDistancePct: number;
  entryProximityPct: number;
  rewardRisk?: number;
  rewardValue?: number;
  riskValue?: number;
  stopBufferValue: number;
  targetBufferValue: number;
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
  entryTriggered: boolean;
  entryDistanceValue: number;
  entryDistancePct: number;
  entryProximityPct: number;
  rewardRisk?: number;
  rewardValue?: number;
  riskValue?: number;
  stopBufferValue: number;
  targetBufferValue: number;
};

export function PnLPreview({ plan, latestPrice }: PnLPreviewProps) {
  const tradeState = deriveTradeState({ plan, latestPrice });
  const previousPriceRef = useRef<number | undefined>(undefined);
  const lastPrice = previousPriceRef.current;
  const delta = typeof lastPrice === "number" ? latestPrice - lastPrice : 0;
  const priceDirection = delta > 0 ? 1 : delta < 0 ? -1 : 0;

  useEffect(() => {
    previousPriceRef.current = latestPrice;
  }, [latestPrice]);

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
        {longStats ? (
          <SideCard
            tone="bullish"
            title="Long Path"
            stats={longStats}
            latestPrice={latestPrice}
            neutralMode={tradeState === "NO_TRADE"}
            priceDirection={priceDirection}
          />
        ) : null}
        {shortStats ? (
          <SideCard
            tone="bearish"
            title="Short Path"
            stats={shortStats}
            latestPrice={latestPrice}
            neutralMode={tradeState === "NO_TRADE"}
            priceDirection={priceDirection}
          />
        ) : null}
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
  neutralMode: boolean;
  priceDirection: number;
};

function SideCard({ tone, title, stats, latestPrice, neutralMode, priceDirection }: SideCardProps) {
  const accent = tone === "bullish" ? "text-emerald-300" : "text-rose-300";
  const barTone = tone === "bullish" ? "bg-emerald-500" : "bg-rose-500";
  const stopTone = "bg-amber-400";
  const signedMove = formatPercent(stats.moveFromEntryPct, 1);
  const showNeutral = neutralMode && !stats.entryTriggered;

  const entryPrice = formatPrice(stats.entry);
  const targetOne = formatPrice(stats.target1);
  const stopPrice = formatPrice(stats.stop);
  const targetGap = formatPrice(Math.max(0, stats.targetBufferValue));
  const stopGap = formatPrice(Math.max(0, stats.stopBufferValue));
  const planLine = `Plan: Entry ${entryPrice} · Target 1 ${targetOne} · Stop ${stopPrice}`;
  const rewardLine = Number.isFinite(stats.rewardRisk)
    && Number.isFinite(stats.rewardValue)
    && Number.isFinite(stats.riskValue)
    ? `R:R ${stats.rewardRisk?.toFixed(1)}x · +${formatPrice(stats.rewardValue ?? 0)} vs -${formatPrice(stats.riskValue ?? 0)}`
    : undefined;

  const distanceHelper = stats.entryTriggered
    ? tone === "bullish"
      ? "Entry filled. Manage the long toward your targets."
      : "Entry filled. Ride the short toward your targets."
    : tone === "bullish"
      ? `Need ${signedPercent(stats.entryDistancePct)} (${formatPrice(stats.entryDistanceValue)}) to trigger entry.`
      : `Need ${signedPercent(stats.entryDistancePct, "down")} (${formatPrice(stats.entryDistanceValue)}) to trigger entry.`;

  const headline = showNeutral
    ? tone === "bullish"
      ? `Need ${signedPercent(stats.entryDistancePct)} (${formatPrice(stats.entryDistanceValue)}) to activate the long plan.`
      : `Need ${signedPercent(stats.entryDistancePct, "down")} (${formatPrice(stats.entryDistanceValue)}) to light up the short plan.`
    : stats.entryTriggered
      ? tone === "bullish"
        ? `Long is live. Price is ${signedMove} versus entry.`
        : `Short is live. Price is ${signedMove} versus entry.`
      : tone === "bullish"
        ? `${formatPrice(stats.entryDistanceValue)} below your buy price. Let it come to you.`
        : `${formatPrice(stats.entryDistanceValue)} above your short trigger. Stay patient.`;

  const guidance = showNeutral
    ? tone === "bullish"
      ? `No trade yet. Set an alert at ${entryPrice}?`
      : `No trade yet. Set a breakdown alert at ${entryPrice}?`
    : stats.entryTriggered
      ? tone === "bullish"
        ? `Target 1 is ${targetGap} away. Still ${stopGap} before your safety stop.`
        : `Target 1 is ${targetGap} away. Stop is ${stopGap} overhead.`
      : tone === "bullish"
        ? `Next: Wait for a strong close over ${entryPrice}.`
        : `Next: Watch for momentum slipping under ${entryPrice}.`;

  return (
    <article className="flex h-full flex-col gap-4 rounded-2xl border border-slate-800/80 bg-slate-950/60 p-6 shadow-sm">
      <header className="flex items-center justify-between">
        <h3 className="text-base font-semibold text-slate-100">{title}</h3>
        <span className={`rounded-full px-3 py-1 text-xs font-semibold ${accent}`}>{tone === "bullish" ? "LONG" : "SHORT"}</span>
      </header>
      <p className={`text-sm leading-relaxed ${accent}`}>{headline}</p>
      <p className="text-xs font-semibold text-slate-200">{guidance}</p>
      <div className="rounded-xl bg-slate-900/40 p-3 text-[11px] text-slate-300">
        <p>{planLine}</p>
        {rewardLine ? <p>{rewardLine}</p> : null}
      </div>
      <div className="space-y-4 text-xs font-medium text-slate-300">
        <DistanceMeter
          label="Price → Entry"
          percent={stats.entryProximityPct}
          toneClass={barTone}
          entryTriggered={stats.entryTriggered}
          helper={distanceHelper}
          tone={tone}
          neutralMode={showNeutral}
          momentumDirection={priceDirection}
        />
        {showNeutral ? (
          <NeutralGuidance tone={tone} targetGap={targetGap} stopGap={stopGap} />
        ) : (
          <>
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
          </>
        )}
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

type DistanceMeterProps = {
  label: string;
  percent: number;
  toneClass: string;
  helper: string;
  entryTriggered: boolean;
  tone: Tone;
  neutralMode: boolean;
  momentumDirection: number;
};

function DistanceMeter({ label, percent, toneClass, helper, entryTriggered, tone, neutralMode, momentumDirection }: DistanceMeterProps) {
  const width = clampToRange(percent);
  const neutralTone = "bg-slate-700";
  const bullishMomentumTone = "bg-emerald-400";
  const bearishMomentumTone = "bg-rose-400";
  const activeMomentumTone = tone === "bullish"
    ? (momentumDirection > 0 ? bullishMomentumTone : neutralTone)
    : (momentumDirection < 0 ? bearishMomentumTone : neutralTone);
  const progressTone = entryTriggered
    ? `${toneClass} opacity-40`
    : neutralMode
      ? activeMomentumTone
      : momentumDirection === 0
        ? toneClass
        : activeMomentumTone;
  const trackTone = neutralMode ? "bg-slate-800" : "bg-slate-900";
  const barHeight = neutralMode ? "h-1.5" : "h-2";

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-[11px] uppercase tracking-wide text-slate-500">
        <span>{label}</span>
        <span>{Math.round(width)}%</span>
      </div>
      <div className={`${barHeight} w-full rounded-full ${trackTone}`}>
        <div
          className={`${barHeight} rounded-full ${progressTone}`}
          style={{ width: `${width}%` }}
        />
      </div>
      <p className="text-[11px] font-medium text-slate-400">{helper}</p>
    </div>
  );
}

type NeutralGuidanceProps = {
  tone: Tone;
  targetGap: string;
  stopGap: string;
};

function NeutralGuidance({ tone, targetGap, stopGap }: NeutralGuidanceProps) {
  return (
    <div className="rounded-xl border border-slate-800/70 bg-slate-950/50 p-3 text-[11px] text-slate-300">
      <p>Target 1 sits {targetGap} away.</p>
      <p>Stop buffer is {stopGap}.</p>
      <p className="mt-1 text-slate-400">Next: {tone === "bullish" ? "Wait for a breakout candle." : "Wait for momentum to roll over."}</p>
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
  const entryTriggered = latestPrice >= entry;
  const entryDistanceValue = entryTriggered ? 0 : Math.max(0, entry - latestPrice);
  const entryBase = Math.abs(entry);
  const rawEntryDistancePct = entryTriggered || !entryBase
    ? 0
    : ((entry - latestPrice) / entryBase) * 100;
  const safeEntryDistancePct = Number.isFinite(rawEntryDistancePct) ? Math.max(0, rawEntryDistancePct) : 0;
  const entryProximityPct = clampToRange(entryTriggered ? 100 : 100 - safeEntryDistancePct);
  const rewardValue = Math.abs(target1 - entry);
  const riskValue = Math.abs(entry - stop);
  const rewardRisk = riskValue > 0 ? rewardValue / riskValue : undefined;
  const stopBufferValue = Math.max(0, latestPrice - stop);
  const targetBufferValue = Math.max(0, target1 - latestPrice);

  return {
    entry,
    stop,
    target1,
    target2: Number.isFinite(target2) ? target2 : undefined,
    moveFromEntryPct,
    progressToTarget1,
    progressToTarget2,
    progressToStop,
    entryTriggered,
    entryDistanceValue,
    entryDistancePct: safeEntryDistancePct,
    entryProximityPct,
    rewardRisk,
    rewardValue,
    riskValue,
    stopBufferValue,
    targetBufferValue,
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
  const entryTriggered = latestPrice <= entry;
  const entryDistanceValue = entryTriggered ? 0 : Math.max(0, latestPrice - entry);
  const entryBase = Math.abs(entry);
  const rawEntryDistancePct = entryTriggered || !entryBase
    ? 0
    : ((latestPrice - entry) / entryBase) * 100;
  const safeEntryDistancePct = Number.isFinite(rawEntryDistancePct) ? Math.max(0, rawEntryDistancePct) : 0;
  const entryProximityPct = clampToRange(entryTriggered ? 100 : 100 - safeEntryDistancePct);
  const rewardValue = Math.abs(entry - target1);
  const riskValue = Math.abs(stop - entry);
  const rewardRisk = riskValue > 0 ? rewardValue / riskValue : undefined;
  const stopBufferValue = Math.max(0, stop - latestPrice);
  const targetBufferValue = Math.max(0, latestPrice - target1);

  return {
    entry,
    stop,
    target1,
    target2: Number.isFinite(target2) ? target2 : undefined,
    moveFromEntryPct,
    progressToTarget1,
    progressToTarget2,
    progressToStop,
    entryTriggered,
    entryDistanceValue,
    entryDistancePct: safeEntryDistancePct,
    entryProximityPct,
    rewardRisk,
    rewardValue,
    riskValue,
    stopBufferValue,
    targetBufferValue,
  };
}

function safeNumber(value: unknown): number {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : NaN;
}

function signedPercent(value: number, direction: "up" | "down" = "up"): string {
  const safeValue = Number.isFinite(value) ? Math.max(0, value) : 0;
  const formatted = safeValue.toFixed(1);
  return direction === "down" ? `-${formatted}%` : `+${formatted}%`;
}
