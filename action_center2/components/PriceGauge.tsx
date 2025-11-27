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

  const shortTargetFar = normalizedSellTargets.length > 1
    ? normalizedSellTargets[0]
    : Number.isFinite(normalizedSellTargets[0])
      ? normalizedSellTargets[0]
      : NaN;
  const shortTargetNear = normalizedSellTargets.length
    ? normalizedSellTargets[normalizedSellTargets.length - 1]
    : NaN;

  const longTargetNear = normalizedBuyTargets.length
    ? normalizedBuyTargets[0]
    : NaN;
  const longTargetFar = normalizedBuyTargets.length > 1
    ? normalizedBuyTargets[normalizedBuyTargets.length - 1]
    : NaN;

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

  const neutralLabel = Number.isFinite(neutralLower) && Number.isFinite(neutralUpper)
    ? `${formatPrice(neutralLower)} – ${formatPrice(neutralUpper)}`
    : "No neutral zone";

  type SegmentSeed = Omit<Segment, "width">;
  const seeds: SegmentSeed[] = [];

  if (Number.isFinite(shortTargetFar) && Number.isFinite(shortTargetNear) && shortTargetFar !== shortTargetNear) {
    seeds.push({
      key: "shortTarget2",
      label: "Short Target 2",
      value: formatPrice(shortTargetFar),
      className: "bg-rose-950/60 text-rose-100 border-r border-rose-800/50",
    });
  }

  if (Number.isFinite(shortTargetNear)) {
    seeds.push({
      key: "shortTarget1",
      label: "Short Target 1",
      value: formatPrice(shortTargetNear),
      className: "bg-rose-900/55 text-rose-100 border-r border-rose-700/45",
    });
  }

  if (Number.isFinite(shortEntry)) {
    seeds.push({
      key: "shortEntry",
      label: "Short Entry",
      value: formatPrice(shortEntry),
      className: "bg-rose-700/45 text-rose-50 border-r border-rose-500/35",
    });
  }

  if (Number.isFinite(neutralLower) && Number.isFinite(neutralUpper)) {
    seeds.push({
      key: "neutral",
      label: "No-Trade",
      value: neutralLabel,
      className: "bg-amber-400/25 text-amber-50 border-r border-amber-300/40",
    });
  }

  if (Number.isFinite(longEntry)) {
    seeds.push({
      key: "longEntry",
      label: "Long Entry",
      value: formatPrice(longEntry),
      className: "bg-emerald-700/40 text-emerald-50 border-r border-emerald-500/35",
    });
  }

  if (Number.isFinite(longTargetNear)) {
    seeds.push({
      key: "longTarget1",
      label: "Long Target 1",
      value: formatPrice(longTargetNear),
      className: "bg-emerald-800/45 text-emerald-100 border-r border-emerald-600/40",
    });
  }

  if (Number.isFinite(longTargetFar) && longTargetFar !== longTargetNear) {
    seeds.push({
      key: "longTarget2",
      label: "Long Target 2",
      value: formatPrice(longTargetFar),
      className: "bg-emerald-900/55 text-emerald-100",
    });
  }

  if (!seeds.length) {
    seeds.push({
      key: "fallback",
      label: "Price Range",
      value: formatPrice(latestPrice),
      className: "bg-slate-800/60 text-slate-200",
    });
  }

  const segmentWidth = 100 / seeds.length;
  const segments: Segment[] = seeds.map((seed) => ({ ...seed, width: segmentWidth }));

  const segmentKeys = new Set(seeds.map((seed) => seed.key));

  const pointerKey = (() => {
    if (Number.isFinite(neutralLower) && Number.isFinite(neutralUpper) && latestPrice >= neutralLower && latestPrice <= neutralUpper) {
      return "neutral";
    }

    if (Number.isFinite(shortEntry) && latestPrice <= shortEntry) {
      if (Number.isFinite(shortTargetFar) && latestPrice <= shortTargetFar && segmentKeys.has("shortTarget2")) {
        return "shortTarget2";
      }
      if (Number.isFinite(shortTargetNear) && latestPrice <= shortTargetNear && segmentKeys.has("shortTarget1")) {
        return "shortTarget1";
      }
      return segmentKeys.has("shortEntry") ? "shortEntry" : segmentKeys.values().next().value;
    }

    if (Number.isFinite(neutralLower) && latestPrice < neutralLower && segmentKeys.has("shortEntry")) {
      return "shortEntry";
    }

    if (Number.isFinite(longEntry) && latestPrice >= longEntry) {
      if (Number.isFinite(longTargetNear) && latestPrice <= longTargetNear && segmentKeys.has("longTarget1")) {
        return "longTarget1";
      }
      if (Number.isFinite(longTargetFar) && latestPrice <= longTargetFar && segmentKeys.has("longTarget2")) {
        return "longTarget2";
      }
      return segmentKeys.has("longTarget2")
        ? "longTarget2"
        : segmentKeys.has("longTarget1")
          ? "longTarget1"
          : "longEntry";
    }

    if (Number.isFinite(neutralUpper) && latestPrice > neutralUpper && segmentKeys.has("longEntry")) {
      return "longEntry";
    }

    return segments[segments.length - 1]?.key ?? segments[0]?.key ?? "fallback";
  })();

  const pointerIndex = Math.max(0, segments.findIndex((segment) => segment.key === pointerKey));
  const pointer = ((pointerIndex + 0.5) / segments.length) * 100;

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
