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

const spreadMarkerGroups = (groups: MarkerGroup[]): MarkerGroup[] => {
  if (groups.length <= 1) {
    return groups.map((group) => ({ ...group }));
  }

  const adjusted = groups.map((group) => ({ ...group }));
  const denseClusterThreshold = 4; // percent difference that triggers extra spacing
  const baseSpacing = Math.max(5, 90 / Math.max(groups.length, 1));
  const denseSpacing = baseSpacing + 4;

  for (let i = 1; i < adjusted.length; i++) {
    const prevOriginalPercent = groups[i - 1].percent;
    const currentOriginalPercent = groups[i].percent;
    const actualGap = currentOriginalPercent - prevOriginalPercent;
    const requiredSpacing = actualGap < denseClusterThreshold ? denseSpacing : baseSpacing;
    const previous = adjusted[i - 1];
    const current = adjusted[i];
    const diff = current.percent - previous.percent;
    if (diff < requiredSpacing) {
      const shift = requiredSpacing - diff;
      for (let j = i; j < adjusted.length; j++) {
        adjusted[j].percent = clamp(adjusted[j].percent + shift, 0, 100);
      }
    }
  }

  for (let i = adjusted.length - 2; i >= 0; i--) {
    const nextOriginalPercent = groups[i + 1].percent;
    const currentOriginalPercent = groups[i].percent;
    const actualGap = nextOriginalPercent - currentOriginalPercent;
    const requiredSpacing = actualGap < denseClusterThreshold ? denseSpacing : baseSpacing;
    const next = adjusted[i + 1];
    const current = adjusted[i];
    const diff = next.percent - current.percent;
    if (diff < requiredSpacing) {
      const shift = requiredSpacing - diff;
      for (let j = i; j >= 0; j--) {
        adjusted[j].percent = clamp(adjusted[j].percent - shift, 0, 100);
      }
    }
  }

  const firstPercent = adjusted[0].percent;
  const lastPercent = adjusted[adjusted.length - 1].percent;
  const span = Math.max(lastPercent - firstPercent, 1);

  return adjusted.map((group) => ({
    ...group,
    percent: ((group.percent - firstPercent) / span) * 96 + 2, // leave 2% gutters on each side
  }));
};

const mapPercentToDisplay = (percent: number, source: MarkerGroup[], target: MarkerGroup[]): number => {
  if (!source.length || source.length !== target.length) {
    return clamp(percent, 0, 100);
  }

  const clamped = clamp(percent, 0, 100);
  const sourcePercents = source.map((group) => group.percent);
  const targetPercents = target.map((group) => group.percent);

  if (clamped <= sourcePercents[0]) {
    const firstSource = Math.max(sourcePercents[0], 1e-6);
    const ratio = clamped / firstSource;
    return clamp(targetPercents[0] * ratio, 0, 100);
  }

  for (let i = 0; i < sourcePercents.length - 1; i++) {
    const start = sourcePercents[i];
    const end = sourcePercents[i + 1];
    if (clamped >= start && clamped <= end) {
      const span = Math.max(end - start, 1e-6);
      const t = (clamped - start) / span;
      const targetStart = targetPercents[i];
      const targetEnd = targetPercents[i + 1];
      return clamp(targetStart + (targetEnd - targetStart) * t, 0, 100);
    }
  }

  const lastIndex = sourcePercents.length - 1;
  const lastSource = sourcePercents[lastIndex];
  const remainder = Math.max(100 - lastSource, 1e-6);
  const ratio = (clamped - lastSource) / remainder;
  return clamp(targetPercents[lastIndex] + (100 - targetPercents[lastIndex]) * ratio, 0, 100);
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
  const displayMarkerGroups = spreadMarkerGroups(markerGroups);
  const pointerPercent = mapPercentToDisplay(pointerPercentRaw, markerGroups, displayMarkerGroups);

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

  const renderLabelChip = (marker: Marker, extraClass = "") => {
    const tone = toneStyles[marker.tone];
    const label = marker.label.trim();
    const lower = label.toLowerCase();
    const targetIndex = lower.indexOf("target");
    let lines: string[] = [label];
    if (targetIndex !== -1) {
      const firstLine = label.slice(0, targetIndex + "target".length).trim();
      const remainder = label.slice(targetIndex + "target".length).trim();
      lines = remainder ? [firstLine, remainder] : [label];
    }
    return (
      <span
        key={`${marker.key}-${marker.value}`}
        className={`block w-full rounded-full px-3 py-1 text-[10px] font-semibold uppercase tracking-wide text-center ${tone.labelChip} ${extraClass}`}
      >
        {lines.map((line, index) => (
          <span key={`${marker.key}-${marker.value}-line-${index}`} className="block leading-tight">
            {line}
          </span>
        ))}
      </span>
    );
  };

  const renderDefinitionStack = (markers: Marker[]) => {
    if (!markers.length) {
      return null;
    }
    const layoutClass =
      markers.length >= 4
        ? "mt-2 grid min-w-[200px] grid-cols-2 gap-1 text-center"
        : "mt-2 flex flex-col items-center gap-1";

    return <div className={layoutClass}>{markers.map((marker) => renderLabelChip(marker))}</div>;
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
          <div className={`h-full rounded-2xl transition-colors duration-500 ${zoneGlowClass}`} />
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
              {displayMarkerGroups.map((group) => {
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
              {displayMarkerGroups.map((group) => {
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
                    {renderDefinitionStack(group.markers)}
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
