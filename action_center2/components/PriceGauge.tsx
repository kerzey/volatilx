import { NoTradeZone, TradeSetup } from "../types";
import { formatPrice } from "../utils/planMath";

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
  start: number;
  end: number;
  width: number;
};

export function PriceGauge({ latestPrice, buySetup, sellSetup, noTradeZones }: PriceGaugeProps) {
  const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

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

  const shortTargetFar = normalizedSellTargets.length > 1
    ? normalizedSellTargets[0]
    : normalizedSellTargets[0];
  const shortTargetNear = normalizedSellTargets.length
    ? normalizedSellTargets[normalizedSellTargets.length - 1]
    : NaN;

  const longTargetNear = normalizedBuyTargets.length
    ? normalizedBuyTargets[0]
    : NaN;
  const longTargetFar = normalizedBuyTargets.length > 1
    ? normalizedBuyTargets[normalizedBuyTargets.length - 1]
    : normalizedBuyTargets[normalizedBuyTargets.length - 1];

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

  const numericBounds = [
    shortTargetFar,
    shortTargetNear,
    shortEntry,
    neutralLower,
    neutralUpper,
    longEntry,
    longTargetNear,
    longTargetFar,
    latestPrice,
  ].filter((value): value is number => Number.isFinite(value));

  let minBound = numericBounds.length ? Math.min(...numericBounds) : latestPrice - 1;
  let maxBound = numericBounds.length ? Math.max(...numericBounds) : latestPrice + 1;

  if (minBound === maxBound) {
    minBound -= 1;
    maxBound += 1;
  }

  const segments: Segment[] = [];

  const addSegment = ({
    key,
    label,
    start,
    end,
    value,
    className,
  }: {
    key: string;
    label: string;
    start: number;
    end: number;
    value: string;
    className: string;
  }) => {
    if (!Number.isFinite(start) || !Number.isFinite(end)) {
      return;
    }
    if (end <= start) {
      return;
    }
    segments.push({ key, label, value, className, start, end, width: 0 });
  };

  if (Number.isFinite(shortTargetFar)) {
    addSegment({
      key: "shortTarget2",
      label: "Short Target 2",
      start: minBound,
      end: shortTargetFar,
      value: formatPrice(shortTargetFar),
      className: "bg-rose-950/60 text-rose-100 border-r border-rose-800/50",
    });
  }

  if (Number.isFinite(shortTargetNear)) {
    addSegment({
      key: "shortTarget1",
      label: "Short Target 1",
      start: Number.isFinite(shortTargetFar) ? shortTargetFar : minBound,
      end: shortTargetNear,
      value: formatPrice(shortTargetNear),
      className: "bg-rose-900/55 text-rose-100 border-r border-rose-700/45",
    });
  }

  if (Number.isFinite(shortEntry)) {
    addSegment({
      key: "shortEntry",
      label: "Short Entry",
      start: Number.isFinite(shortTargetNear)
        ? shortTargetNear
        : Number.isFinite(shortTargetFar)
          ? shortTargetFar
          : minBound,
      end: shortEntry,
      value: formatPrice(shortEntry),
      className: "bg-rose-700/45 text-rose-50 border-r border-rose-500/35",
    });
  }

  if (Number.isFinite(neutralLower) && Number.isFinite(neutralUpper)) {
    addSegment({
      key: "neutral",
      label: "No-Trade",
      start: neutralLower,
      end: neutralUpper,
      value: `${formatPrice(neutralLower)} – ${formatPrice(neutralUpper)}`,
      className: "bg-amber-400/25 text-amber-50 border-r border-amber-300/40",
    });
  }

  if (Number.isFinite(longEntry)) {
    addSegment({
      key: "longEntry",
      label: "Long Entry",
      start: Number.isFinite(neutralUpper)
        ? neutralUpper
        : Number.isFinite(shortEntry)
          ? shortEntry
          : Number.isFinite(shortTargetNear)
            ? shortTargetNear
            : minBound,
      end: longEntry,
      value: formatPrice(longEntry),
      className: "bg-emerald-700/40 text-emerald-50 border-r border-emerald-500/35",
    });
  }

  if (Number.isFinite(longTargetNear)) {
    addSegment({
      key: "longTarget1",
      label: "Long Target 1",
      start: Number.isFinite(longEntry) ? longEntry : Number.isFinite(neutralUpper) ? neutralUpper : minBound,
      end: longTargetNear,
      value: formatPrice(longTargetNear),
      className: "bg-emerald-800/45 text-emerald-100 border-r border-emerald-600/40",
    });
  }

  if (Number.isFinite(longTargetFar) && longTargetFar !== longTargetNear) {
    addSegment({
      key: "longTarget2",
      label: "Long Target 2",
      start: Number.isFinite(longTargetNear)
        ? longTargetNear
        : Number.isFinite(longEntry)
          ? longEntry
          : Number.isFinite(neutralUpper)
            ? neutralUpper
            : minBound,
      end: longTargetFar,
      value: formatPrice(longTargetFar),
      className: "bg-emerald-900/55 text-emerald-100",
    });
  }

  if (!segments.length) {
    segments.push({
      key: "range",
      label: "Price Range",
      value: formatPrice(latestPrice),
      className: "bg-slate-800/60 text-slate-200",
      start: minBound,
      end: maxBound,
      width: 0,
    });
  }

  const firstSegment = segments[0];
  if (firstSegment && firstSegment.start > minBound) {
    segments.unshift({
      key: "belowPlan",
      label: "Below Plan",
      value: formatPrice(firstSegment.start),
      className: "bg-slate-900/60 text-slate-300 border-r border-slate-700/40",
      start: minBound,
      end: firstSegment.start,
      width: 0,
    });
  }

  const lastSegment = segments[segments.length - 1];
  if (lastSegment && lastSegment.end < maxBound) {
    segments.push({
      key: "abovePlan",
      label: "Above Plan",
      value: formatPrice(maxBound),
      className: "bg-slate-900/60 text-slate-300",
      start: lastSegment.end,
      end: maxBound,
      width: 0,
    });
  }

  const segmentWidth = 100 / segments.length;
  segments.forEach((segment) => {
    segment.width = segmentWidth;
  });

  const resolvePointer = (): number => {
    const span = segments.length ? segmentWidth : 100;
    let index = segments.findIndex((segment) => latestPrice >= segment.start && latestPrice <= segment.end);
    let ratio = 0.5;

    if (index === -1) {
      if (latestPrice < segments[0].start) {
        index = 0;
        ratio = 0;
      } else {
        index = segments.length - 1;
        ratio = 1;
      }
    } else {
      const segment = segments[index];
      const range = segment.end - segment.start;
      ratio = Number.isFinite(range) && range > 0
        ? clamp((latestPrice - segment.start) / range, 0, 1)
        : 0.5;
    }

    const position = span * (index + ratio);
    return clamp(position, 0, 100);
  };

  const pointer = resolvePointer();

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
