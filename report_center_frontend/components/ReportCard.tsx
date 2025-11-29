import { useMemo } from "react";
import { formatPrice } from "../../action_center2/utils/planMath";
import type {
  NoTradeZone,
  StrategyKey,
  StrategyPlan,
  TradeSetup,
} from "../../action_center2/types";
import type {
  ReportCenterConsensus,
  ReportCenterEntry,
  ReportCenterPriceAction,
  ReportCenterStrategySummary,
} from "../types";

const STRATEGY_ORDER: Array<{ key: StrategyKey; label: string }> = [
  { key: "day_trading", label: "Day Trading" },
  { key: "swing_trading", label: "Swing Trading" },
  { key: "longterm_trading", label: "Long-Term" },
];

const METRIC_BADGE_BASE =
  "inline-flex items-center gap-2 rounded-full border border-slate-700 bg-slate-900 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-slate-200";

const BULLISH_ACCENT = "text-emerald-300";
const BEARISH_ACCENT = "text-rose-300";

const PRICE_ACTION_CARD_LABEL = "Price action pulse";

const CANDLE_GLOSSARY: Array<{ pattern: RegExp; explanation: string }> = [
  {
    pattern: /bullish\s+hammer|hammer/i,
    explanation: "Long lower wick shows sellers lost control and buyers shoved price back up.",
  },
  {
    pattern: /shooting\s+star/i,
    explanation: "Tall upper wick means buyers were rejected quickly, so supply is lurking overhead.",
  },
  {
    pattern: /doji/i,
    explanation: "Open and close nearly match, signaling indecision after a tug-of-war between bulls and bears.",
  },
  {
    pattern: /bullish\s+engulfing/i,
    explanation: "Fresh green candle swallowed the prior red body, often hinting at upside momentum returning.",
  },
  {
    pattern: /bearish\s+engulfing/i,
    explanation: "New red candle ate the previous green body, showing sellers just overpowered buyers.",
  },
  {
    pattern: /inside\s+bar/i,
    explanation: "Smaller candle trapped inside the prior range, usually a pause before price breaks out.",
  },
];

const explainCandlestickNote = (note?: string): string | null => {
  if (!note) {
    return null;
  }
  for (const entry of CANDLE_GLOSSARY) {
    if (entry.pattern.test(note)) {
      return entry.explanation;
    }
  }
  return null;
};

const resolveBiasTone = (bias?: string): "bullish" | "bearish" | "neutral" => {
  if (!bias) {
    return "neutral";
  }
  const fingerprint = bias.toLowerCase();
  if (/(bull|long|accum)/.test(fingerprint)) {
    return "bullish";
  }
  if (/(bear|short|distrib)/.test(fingerprint)) {
    return "bearish";
  }
  return "neutral";
};

const biasChipClass = (bias?: string): string => {
  const tone = resolveBiasTone(bias);
  if (tone === "bullish") {
    return "border-emerald-500/50 bg-emerald-500/10 text-emerald-200";
  }
  if (tone === "bearish") {
    return "border-rose-500/50 bg-rose-500/10 text-rose-200";
  }
  return "border-slate-600 bg-slate-800/70 text-slate-200";
};

const coerceNumber = (value: unknown): number | null => {
  if (value === null || value === undefined) {
    return null;
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
};

const formatPriceValue = (value: unknown): string => {
  const numeric = coerceNumber(value);
  if (numeric === null) {
    return "—";
  }
  return formatPrice(numeric);
};

const formatPercent = (value: unknown, { allowScaling = true }: { allowScaling?: boolean } = {}): string => {
  const numeric = coerceNumber(value);
  if (numeric === null) {
    return "—";
  }
  const percentage = allowScaling && Math.abs(numeric) <= 1 ? numeric * 100 : numeric;
  const prefix = percentage > 0 ? "+" : "";
  return `${prefix}${percentage.toFixed(2)}%`;
};

const formatRewardRisk = (value: unknown): string | null => {
  const numeric = coerceNumber(value);
  if (numeric === null) {
    return null;
  }
  return numeric.toFixed(2);
};

const formatConviction = (value: unknown): string | null => {
  const numeric = coerceNumber(value);
  if (numeric === null) {
    return null;
  }
  const scaled = Math.abs(numeric) <= 1 ? numeric * 100 : numeric;
  return `${Math.round(scaled)}%`;
};

const formatNoTradeZones = (zones: NoTradeZone[] | undefined): string[] => {
  if (!zones?.length) {
    return [];
  }
  return zones.map((zone) => {
    const low = formatPriceValue(zone?.min ?? null);
    const high = formatPriceValue(zone?.max ?? null);
    return `${low} – ${high}`;
  });
};

const formatVolume = (value: unknown): string => {
  if (value === null || value === undefined) {
    return "—";
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      return "—";
    }
    if (Math.abs(value) >= 1_000_000_000) {
      return `${(value / 1_000_000_000).toFixed(1)}B`;
    }
    if (Math.abs(value) >= 1_000_000) {
      return `${(value / 1_000_000).toFixed(1)}M`;
    }
    if (Math.abs(value) >= 1_000) {
      return `${(value / 1_000).toFixed(1)}K`;
    }
    return value.toString();
  }
  return String(value);
};

const resolveStrategySummary = (
  summaries: Record<string, ReportCenterStrategySummary> | undefined,
  key: StrategyKey,
): ReportCenterStrategySummary | undefined => {
  if (!summaries) {
    return undefined;
  }
  return summaries[key];
};

const StrategyMetrics = ({ plan }: { plan: StrategyPlan }) => {
  const rewardRisk = formatRewardRisk(plan.rewardRisk);
  const conviction = formatConviction(plan.conviction);

  if (!rewardRisk && !conviction) {
    return null;
  }

  return (
    <div className="flex flex-wrap items-center gap-3 text-xs uppercase tracking-wide text-slate-400">
      {rewardRisk ? (
        <span className="rounded-full border border-slate-700 px-3 py-1 text-slate-300">R/R {rewardRisk}</span>
      ) : null}
      {conviction ? (
        <span className="rounded-full border border-slate-700 px-3 py-1 text-slate-300">Conviction {conviction}</span>
      ) : null}
    </div>
  );
};

const TradeSetupBlock = ({
  label,
  setup,
  tone,
}: {
  label: string;
  setup?: TradeSetup;
  tone: "bullish" | "bearish";
}) => {
  if (!setup) {
    return null;
  }

  const accent = tone === "bullish" ? BULLISH_ACCENT : BEARISH_ACCENT;
  const targets = Array.isArray(setup.targets) ? setup.targets : [];

  return (
    <div className="rounded-2xl border border-slate-800/70 bg-slate-900/50 p-4">
      <p className={`text-xs font-semibold uppercase tracking-widest ${accent}`}>{label}</p>
      <dl className="mt-3 space-y-2 text-sm text-slate-200">
        <div className="flex items-center justify-between gap-3 text-slate-300">
          <dt className="text-xs uppercase tracking-wide text-slate-500">Entry</dt>
          <dd className="text-base font-semibold text-slate-100">{formatPriceValue(setup.entry)}</dd>
        </div>
        <div className="flex items-center justify-between gap-3 text-slate-300">
          <dt className="text-xs uppercase tracking-wide text-slate-500">Stop</dt>
          <dd className="text-base font-semibold text-slate-100">{formatPriceValue(setup.stop)}</dd>
        </div>
        <div>
          <dt className="text-xs uppercase tracking-wide text-slate-500">Targets</dt>
          <dd className="mt-1 flex flex-wrap gap-2">
            {targets.length ? (
              targets.map((target, index) => (
                <span
                  key={`target-${index}`}
                  className="rounded-full border border-slate-800 px-2 py-0.5 text-xs font-medium text-slate-200"
                >
                  T{index + 1}: {formatPriceValue(target)}
                </span>
              ))
            ) : (
              <span className="text-slate-500">—</span>
            )}
          </dd>
        </div>
      </dl>
    </div>
  );
};

const NoTradeZoneBlock = ({ zones }: { zones?: NoTradeZone[] }) => {
  const ranges = formatNoTradeZones(zones);
  return (
    <div className="rounded-2xl border border-amber-500/30 bg-amber-500/5 p-4">
      <p className="text-xs font-semibold uppercase tracking-widest text-amber-300">No-Trade Zone</p>
      {ranges.length ? (
        <ul className="mt-3 space-y-2 text-sm text-amber-200">
          {ranges.map((range, index) => (
            <li key={`zone-${index}`}>{range}</li>
          ))}
        </ul>
      ) : (
        <p className="mt-3 text-sm text-amber-200/70">Safe range not defined for this strategy.</p>
      )}
    </div>
  );
};

const ConsensusPanel = ({ consensus }: { consensus?: ReportCenterConsensus }) => {
  if (!consensus) {
    return null;
  }

  const { recommendation, confidence, strength, reasoning, focus } = consensus;

  return (
    <section className="rounded-2xl border border-slate-800 bg-slate-900/40 p-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <span className="text-xs uppercase tracking-wide text-slate-500">Technical read</span>
        <div className="flex flex-wrap items-center gap-3 text-xs uppercase tracking-wide text-slate-400">
          {recommendation ? (
            <span className="rounded-full border border-slate-700 px-3 py-1 text-slate-200">{recommendation}</span>
          ) : null}
          {confidence ? (
            <span className="rounded-full border border-slate-700 px-3 py-1 text-slate-200">Confidence {confidence}</span>
          ) : null}
          {strength !== undefined && strength !== null ? (
            <span className="rounded-full border border-slate-700 px-3 py-1 text-slate-200">Strength {strength}</span>
          ) : null}
        </div>
      </div>
      {focus ? (
        <div className="mt-4 grid gap-4 text-sm text-slate-200 sm:grid-cols-2 lg:grid-cols-4">
          <div>
            <p className="text-slate-500">Focus</p>
            <p className="font-medium text-slate-100">{focus.timeframe ?? "—"}</p>
          </div>
          <div>
            <p className="text-slate-500">Recommendation</p>
            <p className="font-medium text-slate-100">{focus.recommendation ?? "—"}</p>
          </div>
          <div>
            <p className="text-slate-500">Entry</p>
            <p className="font-medium text-slate-100">{formatPriceValue(focus.entry_price)}</p>
          </div>
          <div>
            <p className="text-slate-500">Stop</p>
            <p className="font-medium text-slate-100">{formatPriceValue(focus.stop_loss)}</p>
          </div>
        </div>
      ) : null}
      {reasoning?.length ? (
        <ul className="mt-4 space-y-2 text-sm text-slate-300">
          {reasoning.map((item, index) => (
            <li key={`reason-${index}`} className="flex gap-3">
              <span className="mt-1 h-1.5 w-1.5 rounded-full bg-slate-500" aria-hidden="true" />
              <span className="text-slate-200">{item}</span>
            </li>
          ))}
        </ul>
      ) : null}
    </section>
  );
};

const PriceActionPanel = ({ priceAction }: { priceAction?: ReportCenterPriceAction }) => {
  if (!priceAction) {
    return null;
  }

  const {
    trend_alignment: trendAlignment,
    key_levels: keyLevels,
    recent_patterns: patterns,
    candlestick_notes: candlestickNotesRaw,
    immediate_bias: immediateBias,
  } = priceAction;

  const candlestickNotes = Array.isArray(candlestickNotesRaw)
    ? candlestickNotesRaw
    : candlestickNotesRaw
    ? [candlestickNotesRaw]
    : [];

  if (!trendAlignment && !keyLevels?.length && !patterns?.length && !candlestickNotes.length && !immediateBias) {
    return null;
  }

  return (
    <section className="rounded-2xl border border-slate-800 bg-slate-900/40 p-5">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <span className="text-xs uppercase tracking-wide text-slate-500">{PRICE_ACTION_CARD_LABEL}</span>
        {immediateBias ? (
          <span
            className={`inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-wide ${biasChipClass(immediateBias)}`}
          >
            {immediateBias}
          </span>
        ) : null}
      </div>
      {trendAlignment ? (
        <p className="mt-3 text-sm leading-relaxed text-slate-300">{trendAlignment}</p>
      ) : null}
      {candlestickNotes.length ? (
        <div className="mt-4 space-y-3">
          {candlestickNotes.map((note, index) => {
            const explainer = explainCandlestickNote(note);
            return (
              <div key={`candlestick-note-${index}`} className="rounded-2xl border border-slate-800/70 bg-slate-950/40 p-4">
                <p className="text-sm text-slate-200">{note}</p>
                {explainer ? <p className="mt-1 text-xs text-slate-400">Plain English: {explainer}</p> : null}
              </div>
            );
          })}
        </div>
      ) : null}
      {keyLevels?.length ? (
        <div className="mt-4 grid gap-4 text-sm text-slate-200 sm:grid-cols-2 lg:grid-cols-3">
          {keyLevels.map((level, index) => (
            <div key={`level-${index}`} className="rounded-xl border border-slate-800/70 bg-slate-950/40 p-4">
              <p className="text-xs uppercase tracking-wide text-slate-500">Level</p>
              <p className="mt-1 text-lg font-semibold text-slate-100">{formatPriceValue(level.price)}</p>
              {level.distance_pct !== undefined && level.distance_pct !== null ? (
                <p className="mt-1 text-sm text-slate-400">{formatPercent(level.distance_pct)} away</p>
              ) : null}
            </div>
          ))}
        </div>
      ) : null}
      {patterns?.length ? (
        <ul className="mt-4 space-y-2 text-sm text-slate-300">
          {patterns.map((pattern, index) => (
            <li key={`pattern-${index}`} className="flex gap-3">
              <span className="mt-1 h-1.5 w-1.5 rounded-full bg-slate-500" aria-hidden="true" />
              <span className="text-slate-200">{pattern}</span>
            </li>
          ))}
        </ul>
      ) : null}
    </section>
  );
};

const FavoriteButton = ({
  isFavorite,
  pending,
  onClick,
}: {
  isFavorite: boolean;
  pending: boolean;
  onClick: () => void;
}) => (
  <button
    type="button"
    onClick={onClick}
    disabled={pending}
    aria-pressed={isFavorite}
    className={`inline-flex items-center gap-2 rounded-full border px-4 py-2 text-xs font-semibold uppercase tracking-wide transition focus:outline-none focus:ring-2 focus:ring-slate-300/40 focus:ring-offset-0 ${
      isFavorite
        ? "border-amber-400 bg-amber-400/10 text-amber-200 hover:border-amber-300"
        : "border-slate-700 bg-slate-800/80 text-slate-200 hover:border-slate-600"
    } ${pending ? "opacity-60" : ""}`}
  >
    <span aria-hidden="true" className={isFavorite ? "text-amber-300" : "text-slate-300"}>
      {isFavorite ? "★" : "☆"}
    </span>
    <span>{isFavorite ? "Favorited" : "Favorite"}</span>
  </button>
);

const StrategySection = ({
  label,
  plan,
  summary,
}: {
  label: string;
  plan?: StrategyPlan;
  summary?: ReportCenterStrategySummary;
}) => {
  if (!plan) {
    return null;
  }

  return (
    <section className="rounded-3xl border border-slate-800 bg-slate-900/40 p-6">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <span className="text-xs uppercase tracking-widest text-slate-500">Strategy lane</span>
          <h3 className="mt-2 text-xl font-semibold text-slate-100">{label}</h3>
          {plan.summary ? (
            <p className="mt-2 text-sm leading-relaxed text-slate-300">{plan.summary}</p>
          ) : summary?.summary ? (
            <p className="mt-2 text-sm leading-relaxed text-slate-300">{summary.summary}</p>
          ) : null}
          {summary?.next_actions?.length ? (
            <ul className="mt-3 space-y-1 text-sm text-slate-300">
              {summary.next_actions.map((action, index) => (
                <li key={`action-${index}`} className="flex gap-2">
                  <span className="mt-1 h-1.5 w-1.5 rounded-full bg-slate-500" aria-hidden="true" />
                  <span>{action}</span>
                </li>
              ))}
            </ul>
          ) : null}
        </div>
        <StrategyMetrics plan={plan} />
      </div>
      <div className="mt-6 grid gap-4 lg:grid-cols-3">
        <TradeSetupBlock label="Buy Setup" setup={plan.buy_setup} tone="bullish" />
        <TradeSetupBlock label="Sell Setup" setup={plan.sell_setup} tone="bearish" />
        <NoTradeZoneBlock zones={plan.no_trade_zone} />
      </div>
    </section>
  );
};

export type ReportCardProps = {
  report: ReportCenterEntry;
  isFavorite: boolean;
  pending: boolean;
  onToggleFavorite: (report: ReportCenterEntry, follow: boolean) => void;
};

export function ReportCard({ report, isFavorite, pending, onToggleFavorite }: ReportCardProps) {
  const plan = report.plan ?? undefined;
  const latestPrice = plan?.latest_price;

  const strategies = useMemo(() => {
    if (!plan?.strategies) {
      return [] as Array<{ key: StrategyKey; label: string; plan?: StrategyPlan; summary?: ReportCenterStrategySummary }>;
    }
    return STRATEGY_ORDER.map(({ key, label }) => ({
      key,
      label,
      plan: plan.strategies[key],
      summary: resolveStrategySummary(report.strategies, key),
    })).filter((entry) => Boolean(entry.plan));
  }, [plan?.strategies, report.strategies]);

  const handleFavoriteToggle = () => {
    onToggleFavorite(report, !isFavorite);
  };

  const priceMeta = report.price ?? undefined;
  const closeNumeric = coerceNumber(priceMeta?.close);
  const changeNumeric = coerceNumber(priceMeta?.change_pct);
  const hasVolume = priceMeta?.volume !== undefined && priceMeta?.volume !== null;

  type MetricBadge = { key: string; color: string; label: string };
  const metricBadges: MetricBadge[] = [];

  if (latestPrice !== undefined && latestPrice !== null) {
    metricBadges.push({
      key: "last",
      color: "bg-sky-400",
      label: `Last ${formatPriceValue(latestPrice)}`,
    });
  }

  if (closeNumeric !== null) {
    metricBadges.push({
      key: "close",
      color: "bg-indigo-400",
      label: `Close ${formatPriceValue(priceMeta?.close ?? closeNumeric)}`,
    });
  }

  if (changeNumeric !== null) {
    metricBadges.push({
      key: "change",
      color: changeNumeric > 0 ? "bg-emerald-400" : changeNumeric < 0 ? "bg-rose-400" : "bg-slate-500",
      label: `Change ${formatPercent(changeNumeric, { allowScaling: false })}`,
    });
  }

  if (hasVolume) {
    metricBadges.push({
      key: "volume",
      color: "bg-violet-400",
      label: `Vol ${formatVolume(priceMeta?.volume)}`,
    });
  }

  return (
    <article className="rounded-3xl border border-slate-800 bg-slate-950/60 shadow-xl shadow-black/40 transition hover:border-slate-700">
      <header className="flex flex-col gap-4 border-b border-slate-800 p-6 md:flex-row md:items-center md:justify-between">
        <div className="flex flex-col gap-2">
          <span className="text-xs uppercase tracking-widest text-slate-500">Shared plan</span>
          <h2 className="text-3xl font-semibold tracking-tight text-slate-50">{report.symbol_display ?? report.symbol}</h2>
        </div>
        <div className="flex flex-wrap items-center justify-end gap-3">
          {metricBadges.map(({ key, color, label }) => (
            <span key={key} className={METRIC_BADGE_BASE}>
              <span className={`h-1.5 w-1.5 rounded-full ${color}`} aria-hidden="true" />
              {label}
            </span>
          ))}
          <FavoriteButton isFavorite={isFavorite} pending={pending} onClick={handleFavoriteToggle} />
        </div>
      </header>
      <div className="flex flex-col gap-6 p-6">
        {strategies.length ? (
          <div className="flex flex-col gap-6">
            {strategies.map(({ key, label, plan: strategyPlan, summary }) => (
              <StrategySection key={key} label={label} plan={strategyPlan} summary={summary} />
            ))}
          </div>
        ) : null}
        <ConsensusPanel consensus={report.consensus} />
        <PriceActionPanel priceAction={report.price_action} />
      </div>
      <footer className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-800 px-6 py-4 text-xs text-slate-500">
        {report.source?.blob ? <span>Blob: {report.source.blob}</span> : <span>Blob: unknown</span>}
        {report.source?.user_id ? <span>Author: {report.source.user_id}</span> : null}
      </footer>
    </article>
  );
}
