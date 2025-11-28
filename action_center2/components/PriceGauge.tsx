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

  const levelOrder = [
    "shortTarget2",
    "shortTarget1",
    "shortEntry",
    "neutralLower",
    "neutralUpper",
    "longEntry",
    "longTarget1",
    "longTarget2",
  ] as const;

  const originalLevels: Record<(typeof levelOrder)[number], number | undefined> = {
    shortTarget2: Number.isFinite(shortTargetFar) ? shortTargetFar : undefined,
    shortTarget1: Number.isFinite(shortTargetNear) ? shortTargetNear : undefined,
    shortEntry: Number.isFinite(shortEntry) ? shortEntry : undefined,
    neutralLower: Number.isFinite(neutralLower) ? neutralLower : undefined,
    neutralUpper: Number.isFinite(neutralUpper) ? neutralUpper : undefined,
    longEntry: Number.isFinite(longEntry) ? longEntry : undefined,
    longTarget1: Number.isFinite(longTargetNear) ? longTargetNear : undefined,
    longTarget2: Number.isFinite(longTargetFar) ? longTargetFar : undefined,
  };

  const finiteLevelValues = levelOrder
    .map((key) => originalLevels[key])
    .filter((value): value is number => Number.isFinite(value));

  const derivedRange = finiteLevelValues.length >= 2 ? Math.max(...finiteLevelValues) - Math.min(...finiteLevelValues) : 0;
  let fallbackSpacing = derivedRange > 0 ? derivedRange / (levelOrder.length + 1) : 0;
  if (!Number.isFinite(fallbackSpacing) || fallbackSpacing <= 0) {
    const basis = candidateValues.length >= 2 ? Math.max(...candidateValues) - Math.min(...candidateValues) : Math.abs(latestPrice) * 0.05;
    fallbackSpacing = Math.max(basis, 1);
  }

  const firstLevel = originalLevels[levelOrder[0]];
  let minBound = Number.isFinite(firstLevel)
    ? (firstLevel as number) - fallbackSpacing
    : finiteLevelValues.length
      ? Math.min(...finiteLevelValues) - fallbackSpacing
      : latestPrice - fallbackSpacing;

  let lastValue = minBound;
  const resolvedLevels = {} as Record<(typeof levelOrder)[number], number>;

  levelOrder.forEach((key) => {
    let value = originalLevels[key];
    if (!Number.isFinite(value)) {
      value = lastValue + fallbackSpacing;
    }
    const minimumIncrement = Math.max(fallbackSpacing * 0.25, 0.01);
    if (value <= lastValue) {
      value = lastValue + minimumIncrement;
    }
    resolvedLevels[key] = value as number;
    lastValue = value as number;
  });

  const maxBound = lastValue + fallbackSpacing;
  const totalSpan = Math.max(maxBound - minBound, 1e-6);

  const zoneMeta: Record<string, { label: string; className: string; value: () => string }> = {
    belowPlan: {
      label: "Below Plan",
      className: "bg-slate-900/60 text-slate-300 border-r border-slate-700/40",
      value: () => `< ${formatPrice(originalLevels.shortTarget2 ?? resolvedLevels.shortTarget2)}`,
    },
    shortTarget2: {
      label: "Short Target 2",
      className: "bg-rose-950/70 text-rose-100 border-r border-rose-900/40",
      value: () => formatPrice(originalLevels.shortTarget2 ?? resolvedLevels.shortTarget2),
    },
    shortTarget1: {
      label: "Short Target 1",
      className: "bg-rose-900/55 text-rose-100 border-r border-rose-700/45",
      value: () => formatPrice(originalLevels.shortTarget1 ?? resolvedLevels.shortTarget1),
    },
    shortEntry: {
      label: "Short Entry",
      className: "bg-rose-700/45 text-rose-50 border-r border-rose-500/35",
      value: () => formatPrice(originalLevels.shortEntry ?? resolvedLevels.shortEntry),
    },
    neutral: {
      label: "No-Trade",
      className: "bg-amber-400/25 text-amber-50 border-r border-amber-300/40",
      value: () => `${formatPrice(originalLevels.neutralLower ?? resolvedLevels.neutralLower)} – ${formatPrice(originalLevels.neutralUpper ?? resolvedLevels.neutralUpper)}`,
    },
    longEntry: {
      label: "Long Entry",
      className: "bg-emerald-700/40 text-emerald-50 border-r border-emerald-500/35",
      value: () => formatPrice(originalLevels.longEntry ?? resolvedLevels.longEntry),
    },
    longTarget1: {
      label: "Long Target 1",
      className: "bg-emerald-800/45 text-emerald-100 border-r border-emerald-600/40",
      value: () => formatPrice(originalLevels.longTarget1 ?? resolvedLevels.longTarget1),
    },
    longTarget2: {
      label: "Long Target 2",
      className: "bg-emerald-950/65 text-emerald-100",
      value: () => formatPrice(originalLevels.longTarget2 ?? resolvedLevels.longTarget2),
    },
    abovePlan: {
      label: "Above Plan",
      className: "bg-slate-900/60 text-slate-300",
      value: () => `> ${formatPrice(originalLevels.longTarget2 ?? resolvedLevels.longTarget2)}`,
    },
  };

  const metaForKey = (key: string) => zoneMeta[key] ?? zoneMeta.neutral;

  const segmentsBlueprint: Array<{ key: Segment["key"]; start: number; end: number }> = [
    { key: "belowPlan", start: minBound, end: resolvedLevels.shortTarget2 },
    { key: "shortTarget2", start: resolvedLevels.shortTarget2, end: resolvedLevels.shortTarget1 },
    { key: "shortTarget1", start: resolvedLevels.shortTarget1, end: resolvedLevels.shortEntry },
    { key: "shortEntry", start: resolvedLevels.shortEntry, end: resolvedLevels.neutralLower },
    { key: "neutral", start: resolvedLevels.neutralLower, end: resolvedLevels.neutralUpper },
    { key: "longEntry", start: resolvedLevels.neutralUpper, end: resolvedLevels.longEntry },
    { key: "longTarget1", start: resolvedLevels.longEntry, end: resolvedLevels.longTarget1 },
    { key: "longTarget2", start: resolvedLevels.longTarget1, end: resolvedLevels.longTarget2 },
    { key: "abovePlan", start: resolvedLevels.longTarget2, end: maxBound },
  ];

  const segmentsWithWidth = segmentsBlueprint
    .map((segment) => {
      const span = segment.end - segment.start;
      if (!Number.isFinite(span) || span <= 0) {
        return null;
      }
      const meta = metaForKey(segment.key);
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
          style={{ left: `${pointer}%`, transform: "translateX(-50%)" }}
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
