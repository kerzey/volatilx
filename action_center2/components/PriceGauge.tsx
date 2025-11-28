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

  const uniqueShortTargets = Array.from(new Set(normalizedSellTargets));
  const uniqueLongTargets = Array.from(new Set(normalizedBuyTargets));

  const hasShortTarget2 = uniqueShortTargets.length >= 2;
  const hasLongTarget2 = uniqueLongTargets.length >= 2;

  const shortEntry = toNumber(sellSetup?.entry);
  const longEntry = toNumber(buySetup?.entry);

  const shortTargetFar = uniqueShortTargets.length ? uniqueShortTargets[0] : NaN;
  const shortTargetNear = uniqueShortTargets.length ? uniqueShortTargets[uniqueShortTargets.length - 1] : NaN;

  const longTargetNear = uniqueLongTargets.length ? uniqueLongTargets[0] : NaN;
  const longTargetFar = uniqueLongTargets.length ? uniqueLongTargets[uniqueLongTargets.length - 1] : NaN;

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

  const candidateValues = [
    ...uniqueShortTargets,
    shortEntry,
    neutralLower,
    neutralUpper,
    longEntry,
    ...uniqueLongTargets,
    latestPrice,
  ].filter((value): value is number => Number.isFinite(value));

  const candidateMin = candidateValues.length ? Math.min(...candidateValues) : latestPrice - 1;
  const candidateMax = candidateValues.length ? Math.max(...candidateValues) : latestPrice + 1;
  const baseSpan = Math.max(candidateMax - candidateMin, Math.abs(latestPrice) * 0.02, 1);
  const padding = baseSpan * 0.08;
  const minBound = candidateMin - padding;
  const maxBound = candidateMax + padding;
  const totalSpan = Math.max(maxBound - minBound, 1e-6);

  const breakpointsSet = new Set<number>([minBound, maxBound]);
  if (Number.isFinite(shortTargetFar)) breakpointsSet.add(shortTargetFar);
  if (Number.isFinite(shortTargetNear)) breakpointsSet.add(shortTargetNear);
  if (Number.isFinite(shortEntry)) breakpointsSet.add(shortEntry);
  if (Number.isFinite(neutralLower)) breakpointsSet.add(neutralLower);
  if (Number.isFinite(neutralUpper)) breakpointsSet.add(neutralUpper);
  if (Number.isFinite(longEntry)) breakpointsSet.add(longEntry);
  if (Number.isFinite(longTargetNear)) breakpointsSet.add(longTargetNear);
  if (Number.isFinite(longTargetFar)) breakpointsSet.add(longTargetFar);

  const breakpoints = Array.from(breakpointsSet)
    .filter((value) => Number.isFinite(value))
    .sort((a, b) => a - b);

  const resolveZoneKey = (start: number, end: number): string => {
    const mid = (start + end) / 2;

    if (hasShortTarget2 && Number.isFinite(shortTargetFar) && mid <= shortTargetFar) {
      return "shortTarget2";
    }

    if (Number.isFinite(shortTargetNear) && mid <= shortTargetNear) {
      return "shortTarget1";
    }

    if (Number.isFinite(shortEntry) && mid <= shortEntry) {
      return "shortEntry";
    }

    if (Number.isFinite(neutralLower) && Number.isFinite(neutralUpper) && mid >= neutralLower && mid <= neutralUpper) {
      return "neutral";
    }

    if (Number.isFinite(neutralUpper) && mid < neutralLower && Number.isFinite(shortEntry)) {
      return "shortEntry";
    }

    if (Number.isFinite(longEntry) && mid <= longEntry) {
      return "longEntry";
    }

    if (Number.isFinite(longTargetNear) && mid <= longTargetNear) {
      return "longTarget1";
    }

    if (hasLongTarget2 && Number.isFinite(longTargetFar) && mid <= longTargetFar) {
      return "longTarget2";
    }

    if (mid < shortEntry) {
      return "belowPlan";
    }

    return "abovePlan";
  };

  const numericSegments: Array<{ key: string; start: number; end: number }> = [];

  for (let index = 0; index < breakpoints.length - 1; index += 1) {
    const start = breakpoints[index];
    const end = breakpoints[index + 1];
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
      continue;
    }
    const key = resolveZoneKey(start, end);
    numericSegments.push({ key, start, end });
  }

  if (!numericSegments.length) {
    numericSegments.push({ key: "neutral", start: minBound, end: maxBound });
  }

  const mergedSegments: Array<{ key: string; start: number; end: number }> = [];
  numericSegments.forEach((segment) => {
    const last = mergedSegments[mergedSegments.length - 1];
    if (last && last.key === segment.key) {
      last.end = segment.end;
    } else {
      mergedSegments.push({ ...segment });
    }
  });

  const zoneMeta: Record<string, { label: string; className: string; value: () => string }> = {
    belowPlan: {
      label: "Below Plan",
      className: "bg-slate-900/60 text-slate-300 border-r border-slate-700/40",
      value: () => `< ${formatPrice(shortTargetFar || shortEntry || minBound)}`,
    },
    shortTarget2: {
      label: "Short Target 2",
      className: "bg-rose-950/70 text-rose-100 border-r border-rose-900/40",
      value: () => formatPrice(shortTargetFar),
    },
    shortTarget1: {
      label: "Short Target 1",
      className: "bg-rose-900/55 text-rose-100 border-r border-rose-700/45",
      value: () => formatPrice(shortTargetNear),
    },
    shortEntry: {
      label: "Short Entry",
      className: "bg-rose-700/45 text-rose-50 border-r border-rose-500/35",
      value: () => formatPrice(shortEntry),
    },
    neutral: {
      label: "No-Trade",
      className: "bg-amber-400/25 text-amber-50 border-r border-amber-300/40",
      value: () => `${formatPrice(neutralLower)} – ${formatPrice(neutralUpper)}`,
    },
    longEntry: {
      label: "Long Entry",
      className: "bg-emerald-700/40 text-emerald-50 border-r border-emerald-500/35",
      value: () => formatPrice(longEntry),
    },
    longTarget1: {
      label: "Long Target 1",
      className: "bg-emerald-800/45 text-emerald-100 border-r border-emerald-600/40",
      value: () => formatPrice(longTargetNear),
    },
    longTarget2: {
      label: "Long Target 2",
      className: "bg-emerald-950/65 text-emerald-100",
      value: () => formatPrice(longTargetFar),
    },
    abovePlan: {
      label: "Above Plan",
      className: "bg-slate-900/60 text-slate-300",
      value: () => `> ${formatPrice(longTargetFar || longEntry || maxBound)}`,
    },
  };

  const metaForKey = (key: string) => zoneMeta[key] ?? zoneMeta.neutral;

  const decoratedSegments = mergedSegments
    .map((segment) => {
    const meta = metaForKey(segment.key);
      const span = Math.max(segment.end - segment.start, 0);
      if (span <= 0) {
        return null;
      }
      return {
        key: segment.key,
        label: meta.label,
        value: meta.value(),
        className: meta.className,
        start: segment.start,
        end: segment.end,
        width: (span / totalSpan) * 100,
      } as Segment;
    })
    .filter((segment): segment is Segment => Boolean(segment));

  const pointer = clamp(((latestPrice - minBound) / totalSpan) * 100, 0, 100);

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
          {segmentsWithWidth.map((segment, index) => (
            <div
              key={segment.key}
              className={`flex flex-col items-center justify-center gap-1 p-4 text-center ${segment.className} ${index === segmentsWithWidth.length - 1 ? "border-r-0" : ""}`}
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
