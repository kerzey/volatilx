import { NoTradeZone, TradeSetup, LivePriceMeta } from "../types";
import { formatPrice } from "../utils/planMath";

export type PriceGaugeProps = {
  latestPrice: number;
  liveMeta?: LivePriceMeta | null;
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

const pointerNeedleClass = "bg-indigo-500/80";

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

const formatRelativeTimestamp = (timestamp?: string) => {
  if (!timestamp) {
    return null;
  }
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) {
    return null;
  }
  const now = Date.now();
  const diffMs = now - date.getTime();
  const diffSeconds = Math.abs(diffMs) / 1000;
  if (diffSeconds < 5) {
    return "just now";
  }
  if (diffSeconds < 60) {
    return `${Math.round(diffSeconds)}s ago`;
  }
  const diffMinutes = diffSeconds / 60;
  if (diffMinutes < 60) {
    return `${Math.round(diffMinutes)}m ago`;
  }
  const diffHours = diffMinutes / 60;
  if (diffHours < 24) {
    return `${Math.round(diffHours)}h ago`;
  }
  return date.toLocaleString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    month: "short",
    day: "numeric",
  });
};

const formatSourceLabel = (source?: string) => {
  if (!source) {
    return null;
  }
  const cleaned = source.replace(/[_-]+/g, " ").trim();
  if (!cleaned) {
    return null;
  }
  return cleaned
    .split(" ")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
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

export function PriceGauge({ latestPrice, liveMeta, buySetup, sellSetup, noTradeZones }: PriceGaugeProps) {
  const metaTimestamp = liveMeta?.timestamp || liveMeta?.received_at;
  const metaRelative = formatRelativeTimestamp(metaTimestamp || undefined) ?? null;
  const metaTitle = metaTimestamp || undefined;
  const sourceLabel = formatSourceLabel(liveMeta?.source);
  const marketLabel = liveMeta?.market ? liveMeta.market.toUpperCase() : null;

  let statusLabel = "Last Trade";
  let statusDetail: string | null = null;

  if (!liveMeta) {
    statusLabel = "Last Report Price";
    statusDetail = "Waiting for live feed";
  } else if (liveMeta.error) {
    statusLabel = "Live Feed Paused";
    statusDetail = "Retrying connection";
  } else {
    statusLabel = marketLabel ? `${marketLabel} Feed` : "Live Price";
    const detailParts = [sourceLabel, metaRelative].filter((value): value is string => Boolean(value));
    statusDetail = detailParts.join(" • ") || null;
  }

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
      label: "Target",
      value,
      tone: "short",
    });
  } else if (uniqueShortTargets.length > 1) {
    const farTarget = uniqueShortTargets[0];
    const nearTarget = uniqueShortTargets[uniqueShortTargets.length - 1];
    if (Number.isFinite(farTarget)) {
      markers.push({
        key: "shortTarget2",
        label: "Target 2",
        value: farTarget,
        tone: "short",
      });
    }
    if (Number.isFinite(nearTarget) && Math.abs(nearTarget - farTarget) > 0) {
      markers.push({
        key: "shortTarget1",
        label: "Target 1",
        value: nearTarget,
        tone: "short",
      });
    }
  }

  if (Number.isFinite(shortEntry)) {
    markers.push({
      key: "shortEntry",
      label: "Entry",
      value: shortEntry,
      tone: "short",
    });
  }

  if (Number.isFinite(neutralLower) && Number.isFinite(neutralUpper) && neutralLower <= neutralUpper) {
    markers.push({
      key: "neutralLower",
      label: "Min",
      value: neutralLower,
      tone: "neutral",
    });
    if (Math.abs(neutralUpper - neutralLower) > 0) {
      markers.push({
        key: "neutralUpper",
        label: "Max",
        value: neutralUpper,
        tone: "neutral",
      });
    }
  }

  if (Number.isFinite(longEntry)) {
    markers.push({
      key: "longEntry",
      label: "Entry",
      value: longEntry,
      tone: "long",
    });
  }

  if (uniqueLongTargets.length === 1) {
    const value = uniqueLongTargets[0];
    markers.push({
      key: "longTarget",
      label: "Target",
      value,
      tone: "long",
    });
  } else if (uniqueLongTargets.length > 1) {
    const nearTarget = uniqueLongTargets[0];
    const farTarget = uniqueLongTargets[uniqueLongTargets.length - 1];
    if (Number.isFinite(nearTarget)) {
      markers.push({
        key: "longTarget1",
        label: "Target 1",
        value: nearTarget,
        tone: "long",
      });
    }
    if (Number.isFinite(farTarget) && Math.abs(farTarget - nearTarget) > 0) {
      markers.push({
        key: "longTarget2",
        label: "Target 2",
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
  const mapToDisplayPercent = (percent: number) => mapPercentToDisplay(percent, markerGroups, displayMarkerGroups);
  const pointerPercent = mapToDisplayPercent(pointerPercentRaw);

  const hasNeutralZone = Number.isFinite(neutralLower) && Number.isFinite(neutralUpper) && neutralUpper > neutralLower;
  const neutralStartPercent = hasNeutralZone ? clamp(((neutralLower - minBound) / totalSpan) * 100, 0, 100) : 0;
  const neutralEndPercent = hasNeutralZone ? clamp(((neutralUpper - minBound) / totalSpan) * 100, 0, 100) : 0;

  const layout = {
    containerHeight: 240,
    barY: 100,
    barHeight: 14,
    pointerRise: 60,
  } as const;

  const barTop = layout.barY;
  const pointerTop = Math.max(barTop - layout.pointerRise, 0);
  const pointerHeight = layout.pointerRise;
  const stackTop = barTop + layout.barHeight + 6;

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

  const defaultShortBoundary = 45;
  const defaultNeutralBoundary = 55;
  const shortZoneBoundary = hasNeutralZone ? mapToDisplayPercent(neutralStartPercent) : defaultShortBoundary;
  const neutralZoneBoundary = hasNeutralZone ? mapToDisplayPercent(neutralEndPercent) : defaultNeutralBoundary;
  const shortBoundary = clamp(Math.min(shortZoneBoundary, neutralZoneBoundary), 0, 100);
  const neutralBoundary = clamp(Math.max(shortZoneBoundary, neutralZoneBoundary), 0, 100);

  const channelBackground = `linear-gradient(to right,
    rgba(244,63,94,0.18) 0%,
    rgba(244,63,94,0.18) ${shortBoundary}%,
    rgba(251,191,36,0.18) ${shortBoundary}%,
    rgba(251,191,36,0.18) ${neutralBoundary}%,
    rgba(16,185,129,0.18) ${neutralBoundary}%,
    rgba(16,185,129,0.18) 100%)`;

  const renderLabelChip = (marker: Marker, extraClass = "") => {
    const tone = toneStyles[marker.tone];
    return (
      <span
        key={`${marker.key}-${marker.value}`}
        className={`inline-flex min-w-[70px] items-center justify-center rounded-full px-3 py-1 text-[10px] font-semibold uppercase tracking-wide text-center leading-tight ${tone.labelChip} ${extraClass}`}
      >
        {marker.label}
      </span>
    );
  };

  const renderDefinitionStack = (markers: Marker[]) => {
    if (!markers.length) {
      return null;
    }
    return (
      <div className="mt-2 flex flex-wrap justify-center gap-1 text-center">
        {markers.map((marker) => renderLabelChip(marker))}
      </div>
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
          <p className="text-xs uppercase tracking-wide text-slate-500">{statusLabel}</p>
          <p className="text-lg font-semibold text-slate-100">{formatPrice(latestPrice)}</p>
          {statusDetail && (
            <p className="mt-1 text-[11px] font-medium text-slate-500" title={metaTitle}>
              {statusDetail}
            </p>
          )}
        </div>
      </header>

      <div className="relative mt-6" style={{ height: `${layout.containerHeight}px` }}>
        <div
          className="absolute inset-x-0 z-10"
          style={{ top: `${barTop}px`, height: `${layout.barHeight}px` }}
        >
          <div
            className="relative h-full rounded-full shadow-[0_10px_35px_rgba(2,6,23,0.55)]"
            style={{ background: channelBackground }}
          >
            <div className="absolute inset-0 rounded-full ring-1 ring-white/10" />
            {hasNeutralZone && (
              <div
                className="absolute inset-y-[2px] rounded-full bg-amber-200/25 ring-1 ring-amber-200/30"
                style={{ left: `${shortBoundary}%`, width: `${Math.max(neutralBoundary - shortBoundary, 0)}%` }}
              />
            )}
          </div>
        </div>

        {displayMarkerGroups.map((group) => {
          const dominantTone = group.markers[0]?.tone ?? "neutral";
          const tone = toneStyles[dominantTone];
          const stackPercent = clamp(group.percent, 4, 96);
          return (
            <div
              key={`stack-${group.value}-${dominantTone}`}
              className="pointer-events-none absolute z-20 flex -translate-x-1/2 flex-col items-center text-center"
              style={{ left: `${stackPercent}%`, top: `${stackTop}px` }}
            >
              <span
                className={`mb-1 block h-6 w-[2px] rounded-full ${tone.line}`}
              />
              <div className={`${priceChipClass} ${tone.priceText}`}>{formatPrice(group.value)}</div>
              {renderDefinitionStack(group.markers)}
            </div>
          );
        })}

        <div
          className="pointer-events-none absolute z-30 flex -translate-x-1/2 flex-col items-center text-xs text-indigo-100 transition-all duration-500"
          style={{ left: `${pointerPercent}%`, top: `${pointerTop}px` }}
        >
          <div className="rounded-full bg-indigo-500 px-3 py-1 text-[11px] font-semibold text-indigo-50 shadow-lg shadow-indigo-500/30">
            {formatPrice(latestPrice)}
          </div>
          <span
            className={`mt-2 block w-[3px] rounded-full ${pointerNeedleClass}`}
            style={{ height: `${pointerHeight}px` }}
          />
          <span className="mt-1 block h-2 w-2 rounded-full border border-white/50 bg-indigo-500" />
        </div>

        {!markerGroups.length && (
          <p className="absolute inset-x-0 bottom-0 translate-y-full text-center text-sm text-slate-400">
            Plan did not publish level targets for this symbol. The gauge will activate as soon as fresh levels arrive.
          </p>
        )}
      </div>

      <div className="mt-6 flex flex-wrap gap-3 text-xs font-semibold uppercase tracking-wide text-slate-300">
        <div className="flex items-center gap-2">
          <span className="h-2 w-10 rounded-full bg-rose-500/60" />
          <span>Short Scenario</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="h-2 w-10 rounded-full bg-amber-300/70" />
          <span>No-Trade Zone</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="h-2 w-10 rounded-full bg-emerald-400/70" />
          <span>Long Scenario</span>
        </div>
      </div>
    </section>
  );
}
