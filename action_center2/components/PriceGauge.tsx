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
    labelChip: string;
    priceText: string;
  }
>;

const zoneGlowStyles: Record<MarkerTone, string> = {
  short: "bg-rose-500/12 ring-rose-400/30 shadow-[0_0_30px_rgba(244,63,94,0.25)]",
  neutral: "bg-amber-300/12 ring-amber-300/30 shadow-[0_0_30px_rgba(252,211,77,0.25)]",
  long: "bg-emerald-400/12 ring-emerald-400/30 shadow-[0_0_30px_rgba(16,185,129,0.25)]",
};

const pointerToneLines: Record<MarkerTone, string> = {
  short: "bg-rose-400/80",
  neutral: "bg-amber-300/80",
  long: "bg-emerald-400/80",
};

const toneStyles: ToneStyleMap = {
  short: {
    dot: "bg-rose-400 shadow-[0_0_0_3px_rgba(244,63,94,0.35)]",
    line: "bg-rose-400/70",
    labelChip: "border border-rose-400/60 bg-rose-500/15 text-rose-100",
    priceText: "text-rose-100",
  },
  neutral: {
    dot: "bg-amber-300 shadow-[0_0_0_3px_rgba(252,211,77,0.35)]",
    line: "bg-amber-300/70",
    labelChip: "border border-amber-300/60 bg-amber-400/15 text-amber-100",
    priceText: "text-amber-100",
  },
  long: {
    dot: "bg-emerald-400 shadow-[0_0_0_3px_rgba(16,185,129,0.35)]",
    line: "bg-emerald-400/70",
    labelChip: "border border-emerald-400/60 bg-emerald-500/15 text-emerald-100",
    priceText: "text-emerald-100",
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

const alignPointerPercentWithMarkers = (
  pointerPercent: number,
  groups: MarkerGroup[],
  latestPrice: number,
): number => {
  if (!Number.isFinite(pointerPercent) || !Number.isFinite(latestPrice) || !groups.length) {
    return clamp(pointerPercent, 0, 100);
  }

  const epsilon = 0.5; // small visual offset so the pointer stays clearly to one side
  let adjusted = clamp(pointerPercent, 0, 100);

  const leftGroups = groups.filter((group) => Number.isFinite(group.value) && latestPrice >= group.value);
  const rightGroups = groups.filter((group) => Number.isFinite(group.value) && latestPrice <= group.value);

  if (leftGroups.length) {
    const maxLeftPercent = Math.max(...leftGroups.map((group) => group.percent));
    if (adjusted <= maxLeftPercent) {
      adjusted = Math.min(100, maxLeftPercent + epsilon);
    }
  }

  if (rightGroups.length) {
    const minRightPercent = Math.min(...rightGroups.map((group) => group.percent));
    if (adjusted >= minRightPercent) {
      adjusted = Math.max(0, minRightPercent - epsilon);
    }
  }

  if (leftGroups.length && rightGroups.length) {
    const leftBound = Math.max(...leftGroups.map((group) => group.percent)) + epsilon;
    const rightBound = Math.min(...rightGroups.map((group) => group.percent)) - epsilon;
    if (leftBound > rightBound) {
      adjusted = clamp((leftBound + rightBound) / 2, 0, 100);
    } else {
      adjusted = clamp(Math.min(Math.max(adjusted, leftBound), rightBound), 0, 100);
    }
  }

  return clamp(adjusted, 0, 100);
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
      });
    }
    if (Number.isFinite(nearTarget) && Math.abs(nearTarget - farTarget) > 0) {
      markers.push({
        key: "shortTarget1",
        label: "Short Target 1",
        value: nearTarget,
        tone: "short",
      });
    }
  }

  if (Number.isFinite(shortEntry)) {
    markers.push({
      key: "shortEntry",
      label: "Short Entry",
      value: shortEntry,
      tone: "short",
    });
  }

  if (Number.isFinite(neutralLower) && Number.isFinite(neutralUpper) && neutralLower <= neutralUpper) {
    markers.push({
      key: "neutralLower",
      label: "No-Trade Min",
      value: neutralLower,
      tone: "neutral",
    });
    if (Math.abs(neutralUpper - neutralLower) > 0) {
      markers.push({
        key: "neutralUpper",
        label: "No-Trade Max",
        value: neutralUpper,
        tone: "neutral",
      });
    }
  }

  if (Number.isFinite(longEntry)) {
    markers.push({
      key: "longEntry",
      label: "Long Entry",
      value: longEntry,
      tone: "long",
    });
  }

  if (uniqueLongTargets.length === 1) {
    const value = uniqueLongTargets[0];
    markers.push({
      key: "longTarget",
      label: "Long Target",
      value,
      tone: "long",
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
      });
    }
    if (Number.isFinite(farTarget) && Math.abs(farTarget - nearTarget) > 0) {
      markers.push({
        key: "longTarget2",
        label: "Long Target 2",
        value: farTarget,
        tone: "long",
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

  const pointerPercentRaw = clamp(((latestPrice - minBound) / totalSpan) * 100, 0, 100);

  const markerGroups = buildMarkerGroups(markers, minBound, maxBound);
  const spacedMarkerGroups = adjustMarkerGroupPercents(markerGroups, 3);
  const pointerPercent = alignPointerPercentWithMarkers(pointerPercentRaw, spacedMarkerGroups, latestPrice);

  const hasNeutralZone = Number.isFinite(neutralLower) && Number.isFinite(neutralUpper) && neutralUpper > neutralLower;
  const neutralStartPercent = hasNeutralZone ? clamp(((neutralLower - minBound) / totalSpan) * 100, 0, 100) : 0;
  const neutralEndPercent = hasNeutralZone ? clamp(((neutralUpper - minBound) / totalSpan) * 100, 0, 100) : 0;
  const neutralWidthPercent = Math.max(neutralEndPercent - neutralStartPercent, 0);

  const layout = {
    containerHeight: 320,
    channelPadding: 80,
    barHeight: 12,
    extensionLength: 14,
    pointerClearance: 14,
    highlightGap: 8,
  } as const;

  const topBarY = layout.channelPadding;
  const bottomBarY = layout.containerHeight - layout.channelPadding - layout.barHeight;
  const pointerTop = topBarY + layout.barHeight + layout.pointerClearance;
  const pointerBottom = layout.containerHeight - (bottomBarY - layout.pointerClearance);
  const zoneTop = topBarY + layout.barHeight + layout.highlightGap;
  const zoneBottom = layout.containerHeight - (bottomBarY - layout.highlightGap);

  const determineToneForPrice = (): MarkerTone => {
    if (!Number.isFinite(latestPrice)) {
      return "neutral";
    }
    if (hasNeutralZone) {
      if (latestPrice < neutralLower) {
        return "short";
      }
      if (latestPrice > neutralUpper) {
        return "long";
      }
      return "neutral";
    }
    if (Number.isFinite(shortEntry) && latestPrice <= shortEntry) {
      return "short";
    }
    if (Number.isFinite(longEntry) && latestPrice >= longEntry) {
      return "long";
    }
    return "neutral";
  };

  const priceTone = determineToneForPrice();
  const pointerToneLine = pointerToneLines[priceTone];
  const zoneGlowClass = zoneGlowStyles[priceTone];

  const renderLabelChip = (marker: Marker) => {
    const tone = toneStyles[marker.tone];
    return (
      <span
        key={`${marker.key}-${marker.value}`}
        className={`rounded-full px-3 py-1 text-[10px] font-semibold uppercase tracking-wide ${tone.labelChip}`}
      >
        {marker.label}
      </span>
    );
  };

  const priceChipClass =
    "rounded-lg border border-slate-700/70 bg-slate-950/90 px-2.5 py-1 text-xs font-semibold shadow-inner shadow-black/20 backdrop-blur-sm";

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

      <div className="relative mt-10" style={{ height: `${layout.containerHeight}px` }}>
        <div
          className="absolute inset-x-0 z-10"
          style={{ top: `${zoneTop}px`, bottom: `${zoneBottom}px` }}
        >
          <div className={`h-full rounded-2xl ring-1 transition-colors duration-500 ${zoneGlowClass}`} />
        </div>

        <div
          className="absolute inset-x-0 z-20"
          style={{ top: `${topBarY}px`, height: `${layout.barHeight}px` }}
        >
          <div className="relative h-full rounded-full bg-gradient-to-r from-rose-900/80 via-amber-500/25 to-emerald-500/60">
            <div className="absolute inset-0 rounded-full ring-1 ring-white/5" />
            {hasNeutralZone && (
              <div
                className="absolute top-0 bottom-0 rounded-full bg-amber-200/20 ring-1 ring-amber-200/40 backdrop-blur-sm"
                style={{ left: `${neutralStartPercent}%`, width: `${neutralWidthPercent}%` }}
              />
            )}
            <div className="pointer-events-none absolute inset-0">
              {spacedMarkerGroups.map((group) => {
                const dominantTone = group.markers[0]?.tone ?? "neutral";
                const tone = toneStyles[dominantTone];
                return (
                  <div
                    key={`top-${group.value}-${dominantTone}`}
                    className="absolute -translate-x-1/2"
                    style={{ left: `${group.percent}%`, top: `${-layout.extensionLength - 32}px` }}
                  >
                    <div className="flex flex-col items-center gap-2">
                      <div className={`${priceChipClass} ${tone.priceText}`}>{formatPrice(group.value)}</div>
                      <span
                        className={`block w-[2px] rounded-full ${tone.line}`}
                        style={{ height: `${layout.extensionLength + layout.barHeight}px` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        <div
          className="absolute inset-x-0 z-20"
          style={{ top: `${bottomBarY}px`, height: `${layout.barHeight}px` }}
        >
          <div className="relative h-full rounded-full bg-gradient-to-r from-rose-900/80 via-amber-500/25 to-emerald-500/60">
            <div className="absolute inset-0 rounded-full ring-1 ring-white/5" />
            {hasNeutralZone && (
              <div
                className="absolute top-0 bottom-0 rounded-full bg-amber-200/20 ring-1 ring-amber-200/40 backdrop-blur-sm"
                style={{ left: `${neutralStartPercent}%`, width: `${neutralWidthPercent}%` }}
              />
            )}
            <div className="pointer-events-none absolute inset-0">
              {spacedMarkerGroups.map((group) => {
                const dominantTone = group.markers[0]?.tone ?? "neutral";
                const tone = toneStyles[dominantTone];
                return (
                  <div
                    key={`bottom-${group.value}-${dominantTone}`}
                    className="absolute -translate-x-1/2"
                    style={{ left: `${group.percent}%`, top: "0px" }}
                  >
                    <span
                      className={`block w-[2px] rounded-full ${tone.line}`}
                      style={{ height: `${layout.barHeight + layout.extensionLength}px` }}
                    />
                    <div className="mt-2 flex flex-col items-center gap-1">
                      {group.markers.map((marker) => renderLabelChip(marker))}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        <div
          className="pointer-events-none absolute z-30 flex w-0 -translate-x-1/2 flex-col items-center text-xs text-indigo-100 transition-all duration-500"
          style={{ left: `${pointerPercent}%`, top: `${pointerTop}px`, bottom: `${pointerBottom}px` }}
        >
          <span className={`flex-1 w-[2px] rounded-full ${pointerToneLine}`} />
          <div className="my-2 rounded-full bg-indigo-500 px-3 py-1 text-[11px] font-semibold text-indigo-50 shadow-lg shadow-indigo-500/30">
            {formatPrice(latestPrice)}
          </div>
          <span className={`flex-1 w-[2px] rounded-full ${pointerToneLine}`} />
        </div>

        {!markerGroups.length && (
          <p className="absolute inset-x-0 bottom-0 translate-y-full text-center text-sm text-slate-400">
            Plan did not publish level targets for this symbol. The gauge will activate as soon as fresh levels arrive.
          </p>
        )}
      </div>
    </section>
  );
}
