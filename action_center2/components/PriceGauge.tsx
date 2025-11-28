import { NoTradeZone, TradeSetup } from "../types";
import { formatPrice } from "../utils/planMath";

export type PriceGaugeProps = {
  latestPrice: number;
  buySetup: TradeSetup;
  sellSetup: TradeSetup;
  noTradeZones: NoTradeZone[];
};

type LevelKey =
  | "shortTarget2"
  | "shortTarget1"
  | "shortTarget"
  | "shortEntry"
  | "neutralLower"
  | "neutralUpper"
  | "longEntry"
  | "longTarget1"
  | "longTarget2"
  | "longTarget";

type MarkerTone = "short" | "neutral" | "long";

type Marker = {
  key: LevelKey;
  label: string;
  value: number;
  tone: MarkerTone;
  align: "top" | "bottom";
};

type MarkerGroup = {
  value: number;
  percent: number;
  markers: Marker[];
};

type ToneStyleMap = Record<
  MarkerTone,
  {
    dot: string;
    line: string;
    swatch: string;
    legendText: string;
  }
>;

const toneStyles: ToneStyleMap = {
  short: {
    dot: "bg-rose-400 shadow-[0_0_0_3px_rgba(244,63,94,0.35)]",
    line: "bg-rose-400/70",
    swatch: "bg-gradient-to-r from-rose-500 to-rose-400",
    legendText: "text-rose-200",
  },
  neutral: {
    dot: "bg-amber-300 shadow-[0_0_0_3px_rgba(252,211,77,0.35)]",
    line: "bg-amber-300/70",
    swatch: "bg-gradient-to-r from-amber-400 to-amber-300",
    legendText: "text-amber-200",
  },
  long: {
    dot: "bg-emerald-400 shadow-[0_0_0_3px_rgba(16,185,129,0.35)]",
    line: "bg-emerald-400/70",
    swatch: "bg-gradient-to-r from-emerald-500 to-emerald-400",
    legendText: "text-emerald-200",
  },
};

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

const toNumber = (value: number | string | null | undefined): number => {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : NaN;
};

const buildMarkerGroups = (markers: Marker[], minBound: number, maxBound: number): MarkerGroup[] => {
  if (!markers.length) {
    return [];
  }

  const totalSpan = Math.max(maxBound - minBound, 1e-6);
  const percentForValue = (value: number) => clamp(((value - minBound) / totalSpan) * 100, 0, 100);
  const tolerance = Math.max(totalSpan * 0.0025, 0.01);

  const ordered = [...markers].sort((a, b) => a.value - b.value);
  const groups: MarkerGroup[] = [];

  ordered.forEach((marker) => {
    const lastGroup = groups[groups.length - 1];
    if (lastGroup && Math.abs(lastGroup.value - marker.value) <= tolerance) {
      lastGroup.markers.push(marker);
    } else {
      groups.push({
        value: marker.value,
        percent: percentForValue(marker.value),
        markers: [marker],
      });
    }
  });

  return groups;
};

const adjustMarkerGroupPercents = (groups: MarkerGroup[], minSpacingPercent: number): MarkerGroup[] => {
  if (groups.length <= 1) {
    return groups;
  }

  const adjusted = groups.map((group) => ({ ...group }));

  adjusted[0].percent = clamp(adjusted[0].percent, 0, 100);
  for (let i = 1; i < adjusted.length; i++) {
    adjusted[i].percent = clamp(adjusted[i].percent, 0, 100);
    if (adjusted[i].percent - adjusted[i - 1].percent < minSpacingPercent) {
      adjusted[i].percent = Math.min(100, adjusted[i - 1].percent + minSpacingPercent);
    }
  }

  for (let i = adjusted.length - 2; i >= 0; i--) {
    if (adjusted[i + 1].percent - adjusted[i].percent < minSpacingPercent) {
      adjusted[i].percent = Math.max(0, adjusted[i + 1].percent - minSpacingPercent);
    }
  }

  for (let i = 1; i < adjusted.length; i++) {
    if (adjusted[i].percent - adjusted[i - 1].percent < minSpacingPercent) {
      adjusted[i].percent = Math.min(100, adjusted[i - 1].percent + minSpacingPercent);
    }
  }

  return adjusted;
};

export function PriceGauge({ latestPrice, buySetup, sellSetup, noTradeZones }: PriceGaugeProps) {
  const normalizedSellTargets = Array.isArray(sellSetup?.targets)
    ? [...sellSetup.targets].map(toNumber).filter((value) => Number.isFinite(value)).sort((a, b) => a - b)
    : [];
  const normalizedBuyTargets = Array.isArray(buySetup?.targets)
    ? [...buySetup.targets].map(toNumber).filter((value) => Number.isFinite(value)).sort((a, b) => a - b)
    : [];

  const shortEntry = toNumber(sellSetup?.entry);
  const longEntry = toNumber(buySetup?.entry);

  const uniqueShortTargets = Array.from(new Set(normalizedSellTargets));
  const uniqueLongTargets = Array.from(new Set(normalizedBuyTargets));

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

  const markers: Marker[] = [];

  if (uniqueShortTargets.length === 1) {
    const value = uniqueShortTargets[0];
    markers.push({
      key: "shortTarget",
      label: "Short Target",
      value,
      tone: "short",
      align: "bottom",
    });
  } else if (uniqueShortTargets.length > 1) {
    const farTarget = uniqueShortTargets[0];
    const nearTarget = uniqueShortTargets[uniqueShortTargets.length - 1];
    if (Number.isFinite(farTarget)) {
      markers.push({
        key: "shortTarget2",
        label: "Short Target 2",
        value: farTarget,
        tone: "short",
        align: "bottom",
      });
    }
    if (Number.isFinite(nearTarget) && Math.abs(nearTarget - farTarget) > 0) {
      markers.push({
        key: "shortTarget1",
        label: "Short Target 1",
        value: nearTarget,
        tone: "short",
        align: "bottom",
      });
    }
  }

  if (Number.isFinite(shortEntry)) {
    markers.push({
      key: "shortEntry",
      label: "Short Entry",
      value: shortEntry,
      tone: "short",
      align: "bottom",
    });
  }

  if (Number.isFinite(neutralLower) && Number.isFinite(neutralUpper) && neutralLower <= neutralUpper) {
    markers.push({
      key: "neutralLower",
      label: "No-Trade Min",
      value: neutralLower,
      tone: "neutral",
      align: "bottom",
    });
    if (Math.abs(neutralUpper - neutralLower) > 0) {
      markers.push({
        key: "neutralUpper",
        label: "No-Trade Max",
        value: neutralUpper,
        tone: "neutral",
        align: "bottom",
      });
    }
  }

  if (Number.isFinite(longEntry)) {
    markers.push({
      key: "longEntry",
      label: "Long Entry",
      value: longEntry,
      tone: "long",
      align: "bottom",
    });
  }

  if (uniqueLongTargets.length === 1) {
    const value = uniqueLongTargets[0];
    markers.push({
      key: "longTarget",
      label: "Long Target",
      value,
      tone: "long",
      align: "bottom",
    });
  } else if (uniqueLongTargets.length > 1) {
    const nearTarget = uniqueLongTargets[0];
    const farTarget = uniqueLongTargets[uniqueLongTargets.length - 1];
    if (Number.isFinite(nearTarget)) {
      markers.push({
        key: "longTarget1",
        label: "Long Target 1",
        value: nearTarget,
        tone: "long",
        align: "bottom",
      });
    }
    if (Number.isFinite(farTarget) && Math.abs(farTarget - nearTarget) > 0) {
      markers.push({
        key: "longTarget2",
        label: "Long Target 2",
        value: farTarget,
        tone: "long",
        align: "bottom",
      });
    }
  }

  const valuesForBounds = markers.map((marker) => marker.value).concat(Number.isFinite(latestPrice) ? [latestPrice] : []);

  let minValue = valuesForBounds.length ? Math.min(...valuesForBounds) : latestPrice;
  let maxValue = valuesForBounds.length ? Math.max(...valuesForBounds) : latestPrice;

  if (!Number.isFinite(minValue)) {
    minValue = latestPrice;
  }
  if (!Number.isFinite(maxValue)) {
    maxValue = latestPrice;
  }

  let baseRange = maxValue - minValue;
  if (!Number.isFinite(baseRange) || baseRange <= 0) {
    baseRange = Math.max(Math.abs(latestPrice) * 0.1, 1);
  }

  const padding = Math.max(baseRange * 0.12, baseRange === 0 ? 1 : 0);
  const minBound = minValue - padding;
  const maxBound = maxValue + padding;
  const totalSpan = Math.max(maxBound - minBound, 1e-6);

  const pointerPercent = clamp(((latestPrice - minBound) / totalSpan) * 100, 0, 100);

  const markerGroups = buildMarkerGroups(markers, minBound, maxBound);
  const spacedMarkerGroups = adjustMarkerGroupPercents(markerGroups, 3);

  const priceChipClass =
    "rounded-lg border border-slate-700/70 bg-slate-950/90 px-2.5 py-1 text-xs font-semibold text-slate-100 shadow-inner shadow-black/20 backdrop-blur-sm";

  const legendOrder: LevelKey[] = [
    "shortTarget2",
    "shortTarget1",
    "shortTarget",
    "shortEntry",
    "neutralLower",
    "neutralUpper",
    "longEntry",
    "longTarget1",
    "longTarget2",
    "longTarget",
  ];

  const legendMap = new Map<LevelKey, Marker>();
  markers.forEach((marker) => {
    if (!legendMap.has(marker.key)) {
      legendMap.set(marker.key, marker);
    }
  });

  const legendItems = legendOrder
    .map((key) => legendMap.get(key))
    .filter((value): value is Marker => Boolean(value));

  const legendItemsSorted = [...legendItems].sort((a, b) => a.value - b.value);

  const hasNeutralZone = Number.isFinite(neutralLower) && Number.isFinite(neutralUpper) && neutralUpper > neutralLower;
  const neutralStartPercent = hasNeutralZone ? clamp(((neutralLower - minBound) / totalSpan) * 100, 0, 100) : 0;
  const neutralEndPercent = hasNeutralZone ? clamp(((neutralUpper - minBound) / totalSpan) * 100, 0, 100) : 0;
  const neutralWidthPercent = Math.max(neutralEndPercent - neutralStartPercent, 0);

  return (
    <section className="rounded-3xl border border-slate-800/70 bg-slate-950/80 p-8 shadow-lg shadow-indigo-500/5">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-slate-50">Price Gauge</h2>
          <p className="text-sm text-slate-400">Track plan levels and see where price is leaning right now.</p>
        </div>
        <div className="text-right">
          <p className="text-xs uppercase tracking-wide text-slate-500">Last Trade</p>
          <p className="text-lg font-semibold text-slate-100">{formatPrice(latestPrice)}</p>
        </div>
      </header>

      <div className="relative mt-10">
        <div className="relative h-3 w-full rounded-full bg-gradient-to-r from-rose-900/80 via-amber-500/25 to-emerald-500/60">
          <div className="absolute inset-0 rounded-full ring-1 ring-white/5" />
          {hasNeutralZone && (
            <div
              className="absolute top-0 bottom-0 rounded-full bg-amber-200/20 ring-1 ring-amber-200/40 backdrop-blur-sm"
              style={{ left: `${neutralStartPercent}%`, width: `${neutralWidthPercent}%` }}
            />
          )}
        </div>

        {spacedMarkerGroups.map((group) => {
          const bottomMarkers = group.markers.filter((marker) => marker.align === "bottom");
          const dominantTone = group.markers[0]?.tone ?? "neutral";
          const tone = toneStyles[dominantTone];

          return (
            <div
              key={`${group.value}-${dominantTone}`}
              className="absolute top-0 z-20 flex h-full w-0 -translate-x-1/2"
              style={{ left: `${group.percent}%` }}
            >
              <div className="flex flex-col items-center">
                <div className="flex flex-col items-center text-[10px]">
                  <span className={`mb-1 h-4 w-px ${tone.line}`} />
                  <span className={`h-2 w-2 rounded-full ${tone.dot}`} />
                  <span className={`mt-1 h-4 w-px ${tone.line}`} />
                </div>

                {bottomMarkers.length > 0 && (
                  <div className="flex flex-col items-center pt-1">
                    <span className={priceChipClass}>{formatPrice(group.value)}</span>
                  </div>
                )}
              </div>
            </div>
          );
        })}

        <div
          className="pointer-events-none absolute -top-12 z-30 flex -translate-x-1/2 flex-col items-center gap-2 text-xs text-indigo-100 transition-all duration-500"
          style={{ left: `${pointerPercent}%` }}
        >
          <div className="rounded-full bg-indigo-500 px-3 py-1 text-[11px] font-semibold text-indigo-50 shadow-lg shadow-indigo-500/30">
            {formatPrice(latestPrice)}
          </div>
          <span className="block h-9 w-[2px] rounded-full bg-indigo-400" />
          <span className="h-2 w-2 rounded-full border border-indigo-200/60 bg-indigo-500 shadow-[0_0_0_3px_rgba(99,102,241,0.15)]" />
        </div>

        {!markerGroups.length && (
          <p className="mt-6 text-center text-sm text-slate-400">
            Plan did not publish level targets for this symbol. The gauge will activate as soon as fresh levels arrive.
          </p>
        )}
      </div>

      {legendItemsSorted.length > 0 && (
        <div className="mt-35 flex flex-wrap items-center gap-x-3 gap-y-2 text-sm text-slate-300">
          {legendItemsSorted.map((item) => {
            const tone = toneStyles[item.tone];
            return (
              <div
                key={item.key}
                className="flex items-center gap-3 rounded-2xl border border-slate-800/60 bg-slate-950/70 px-3 py-2"
              >
                <span className={`h-2 w-10 rounded-full ${tone.swatch}`} />
                <div className="flex flex-col gap-[2px]">
                  <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">{item.label}</span>
                  <span className={`text-sm font-semibold ${tone.legendText}`}>{formatPrice(item.value)}</span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}
