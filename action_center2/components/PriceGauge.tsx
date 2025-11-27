import { NoTradeZone, TradeSetup } from "../types";
import { clampToRange, formatPrice } from "../utils/planMath";

export type PriceGaugeProps = {
  latestPrice: number;
  buySetup: TradeSetup;
  sellSetup: TradeSetup;
  noTradeZones: NoTradeZone[];
};

type Segment = {
  key: string;
  label: string;
  value: string;
  className: string;
  width: number;
};

export function PriceGauge({ latestPrice, buySetup, sellSetup, noTradeZones }: PriceGaugeProps) {
  const toNumber = (value: number | string | null | undefined): number => {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : NaN;
  };

  const normalizedSellTargets = Array.isArray(sellSetup?.targets)
    ? [...sellSetup.targets].map(toNumber).filter((value) => Number.isFinite(value)).sort((a, b) => a - b)
    : [];
  const normalizedBuyTargets = Array.isArray(buySetup?.targets)
    ? [...buySetup.targets].map(toNumber).filter((value) => Number.isFinite(value)).sort((a, b) => a - b)
    : [];

  const shortEntry = toNumber(sellSetup?.entry);
  const longEntry = toNumber(buySetup?.entry);
  const lowestShortTarget = normalizedSellTargets[0];
  const highestLongTarget = normalizedBuyTargets[normalizedBuyTargets.length - 1];

  const normalizedZones = Array.isArray(noTradeZones)
    ? noTradeZones
      .map((zone) => {
        const lower = toNumber(zone?.min);
        const upper = toNumber(zone?.max);
        if (!Number.isFinite(lower) || !Number.isFinite(upper)) {
          return null;
        }
        return {
          lower: Math.min(lower, upper),
          upper: Math.max(lower, upper),
        };
      })
      .filter((value): value is { lower: number; upper: number } => Boolean(value))
    : [];

  const neutralLower = normalizedZones.length ? Math.min(...normalizedZones.map((zone) => zone.lower)) : NaN;
  const neutralUpper = normalizedZones.length ? Math.max(...normalizedZones.map((zone) => zone.upper)) : NaN;

  const boundaryCandidates = [
    lowestShortTarget,
    shortEntry,
    neutralLower,
    neutralUpper,
    longEntry,
    highestLongTarget,
    latestPrice,
  ].filter((value): value is number => Number.isFinite(value));

  let minBound = boundaryCandidates.length ? Math.min(...boundaryCandidates) : latestPrice;
  let maxBound = boundaryCandidates.length ? Math.max(...boundaryCandidates) : latestPrice;

  if (minBound === maxBound) {
    minBound -= 1;
    maxBound += 1;
  }

  const range = Math.max(maxBound - minBound, 1e-6);
  const pointer = clampToRange(((latestPrice - minBound) / range) * 100, 0, 100);

  const noTradeLabel = normalizedZones.length
    ? normalizedZones.map((zone) => `${formatPrice(zone.lower)} – ${formatPrice(zone.upper)}`).join(" | ")
    : "No neutral zone";

  const segments: Segment[] = [];
  let cursor = minBound;

  const pushSegment = (key: string, label: string, endValue: number, value: string, className: string) => {
    if (!Number.isFinite(endValue)) {
      return;
    }
    const clampedEnd = Math.max(endValue, cursor);
    if (clampedEnd <= cursor) {
      return;
    }
    const width = ((clampedEnd - cursor) / range) * 100;
    if (width <= 0) {
      return;
    }
    segments.push({ key, label, value, className, width });
    cursor = clampedEnd;
  };

  const nextAfterShortTargets = Number.isFinite(shortEntry)
    ? shortEntry
    : Number.isFinite(neutralLower)
      ? neutralLower
      : Number.isFinite(longEntry)
        ? longEntry
        : maxBound;

  if (nextAfterShortTargets > cursor) {
    const displayValue = Number.isFinite(lowestShortTarget) ? lowestShortTarget : cursor;
    pushSegment(
      "sellTarget",
      "Short Targets",
      nextAfterShortTargets,
      formatPrice(displayValue),
      "bg-rose-600/20 text-rose-200 border-r border-rose-500/20",
    );
  }

  const nextAfterShortEntry = Number.isFinite(neutralLower)
    ? neutralLower
    : Number.isFinite(longEntry)
      ? longEntry
      : maxBound;

  if (Number.isFinite(shortEntry) && nextAfterShortEntry > cursor) {
    pushSegment(
      "sellEntry",
      "Short Entry",
      nextAfterShortEntry,
      formatPrice(shortEntry),
      "bg-rose-500/15 text-rose-100 border-r border-rose-400/20",
    );
  }

  if (Number.isFinite(neutralLower) && Number.isFinite(neutralUpper) && neutralUpper > cursor) {
    pushSegment(
      "neutral",
      "No-Trade",
      neutralUpper,
      noTradeLabel,
      "bg-amber-500/15 text-amber-100 border-r border-amber-400/20",
    );
  }

  if (Number.isFinite(longEntry) && longEntry > cursor) {
    pushSegment(
      "buyEntry",
      "Long Entry",
      longEntry,
      formatPrice(longEntry),
      "bg-emerald-500/15 text-emerald-100 border-r border-emerald-400/20",
    );
  }

  if (cursor < maxBound) {
    const displayTarget = Number.isFinite(highestLongTarget)
      ? highestLongTarget
      : Number.isFinite(longEntry)
        ? longEntry
        : maxBound;
    pushSegment(
      "buyTarget",
      "Long Targets",
      maxBound,
      formatPrice(displayTarget),
      "bg-emerald-600/20 text-emerald-200",
    );
  }

  if (!segments.length) {
    segments.push({
      key: "range",
      label: "Price Range",
      value: formatPrice(latestPrice),
      className: "bg-slate-800/60 text-slate-200",
      width: 100,
    });
  }

  return (
    <section className="rounded-3xl border border-slate-800 bg-slate-900 p-8 shadow-sm">
      <header className="mb-6 flex items-start justify-between">
        <div>
          <h2 className="text-lg font-semibold text-slate-50">Price Gauge</h2>
          <p className="text-sm text-slate-400">Visualise key regions before committing risk</p>
        </div>
        <div className="text-right">
          <p className="text-xs uppercase tracking-wide text-slate-500">Last</p>
          <p className="text-lg font-semibold text-slate-100">{formatPrice(latestPrice)}</p>
        </div>
      </header>
      <div className="relative">
        <div className="relative flex overflow-hidden rounded-2xl border border-slate-800/70 bg-slate-950/80 text-[11px] font-semibold uppercase tracking-wide text-slate-200">
          {segments.map((segment, index) => (
            <div
              key={segment.key}
              className={`flex flex-col items-center justify-center gap-1 p-4 text-center ${segment.className} ${index === segments.length - 1 ? "border-r-0" : ""}`}
              style={{ flexBasis: `${segment.width}%`, flexGrow: 0, flexShrink: 0 }}
            >
              <span>{segment.label}</span>
              <span className="text-[10px] normal-case text-slate-300">{segment.value}</span>
            </div>
          ))}
        </div>
        <div
          className="pointer-events-none absolute -top-3 flex flex-col items-center text-xs text-slate-200"
          style={{ left: `calc(${pointer}% - 12px)` }}
        >
          <span className="h-3 w-[2px] rounded-full bg-indigo-400" />
          <span className="mt-1 rounded-full bg-indigo-500/20 px-3 py-1 text-[10px] font-semibold">
            {formatPrice(latestPrice)}
          </span>
        </div>
      </div>
    </section>
  );
}
