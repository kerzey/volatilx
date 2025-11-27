import { TechnicalConsensus } from "../types";

export type TrendHeatmapProps = {
const RECOMMENDATION_STYLE: Record<NonNullable<TechnicalConsensus["overall_recommendation"]>, string> = {
  BUY: "text-emerald-200 bg-emerald-500/20 border border-emerald-400/40",
  SELL: "text-rose-200 bg-rose-500/20 border border-rose-400/40",
  HOLD: "text-amber-200 bg-amber-500/20 border border-amber-400/40",
};

const CONFIDENCE_VALUE: Record<NonNullable<TechnicalConsensus["confidence"]>, number> = {
  LOW: 0.33,
  MEDIUM: 0.66,
  HIGH: 1,
export function TrendHeatmap({ consensus }: TrendHeatmapProps) {
  if (!consensus) {
    return (
      <section className="rounded-3xl border border-slate-800 bg-slate-900 p-8 shadow-sm">
        <header className="mb-4">
          <h2 className="text-lg font-semibold text-slate-50">Trend Heatmap</h2>
          <p className="text-sm text-slate-400">No consensus data yet</p>
        </header>
        <p className="text-sm text-slate-500">
          Refresh the plan when you have updated breadth and momentum inputs to light up this view.
        </p>
      </section>
    );
  }

  const strengthPct = clamp(consensus.strength * 100, 0, 100);
  const confidencePct = clamp((CONFIDENCE_VALUE[consensus.confidence] ?? 0) * 100, 0, 100);

  return (
    <section className="rounded-3xl border border-slate-800 bg-slate-900 p-8 shadow-sm">
      <header className="mb-6 flex flex-col gap-2 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h2 className="text-lg font-semibold text-slate-50">Trend Heatmap</h2>
          <p className="text-sm text-slate-400">Momentum, structure, and participation at a glance</p>
        </div>
        <span className={`inline-flex items-center gap-2 rounded-full px-4 py-2 text-xs font-semibold uppercase tracking-wide ${RECOMMENDATION_STYLE[consensus.overall_recommendation]}`}>
          <span className="h-2 w-2 rounded-full bg-current" />
          {consensus.overall_recommendation}
        </span>
      </header>
      <div className="grid gap-6 md:grid-cols-2">
        <MetricCard
          title="Confidence"
          description={`Desk confidence tagged ${consensus.confidence.toLowerCase()}.`}
          percent={confidencePct}
        />
        <MetricCard
          title="Strength"
          description="Composite read on breadth, momentum, and liquidity."
          percent={strengthPct}
        />
      </div>
    </section>
  );
}

type MetricCardProps = {
  title: string;
  description: string;
  percent: number;
};

function MetricCard({ title, description, percent }: MetricCardProps) {
  return (
    <article className="flex flex-col gap-3 rounded-2xl border border-slate-800/80 bg-slate-950/60 p-6 shadow-sm">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-100">{title}</h3>
        <span className="text-sm font-semibold text-slate-200">{Math.round(percent)}%</span>
      </div>
      <p className="text-sm text-slate-400">{description}</p>
      <div className="h-2 w-full rounded-full bg-slate-800">
        <div
          className="h-2 rounded-full bg-indigo-500"
          style={{ width: `${percent}%` }}
        />
      </div>
    </article>
  );
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}
type HeatmapPoint = {
  label: string;
  bias: "bullish" | "neutral" | "bearish";
};

type TrendHeatmapProps = {
  intraday: HeatmapPoint[];
  higherTimeframe: HeatmapPoint[];
};

const BIAS_COLOR: Record<HeatmapPoint["bias"], string> = {
  bullish: "bg-emerald-500/15 text-emerald-600 border-emerald-500/30",
  neutral: "bg-slate-500/10 text-slate-600 border-slate-500/20",
  bearish: "bg-rose-500/15 text-rose-500 border-rose-500/30",
};

export function TrendHeatmap({ intraday, higherTimeframe }: TrendHeatmapProps) {
  return (
    <section className="flex flex-col gap-4 rounded-3xl border border-slate-200/70 bg-white/70 p-6 shadow-sm backdrop-blur">
      <header className="flex items-start justify-between">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">Trend Heatmap</h2>
          <p className="text-sm text-slate-500">Context stack across timeframes</p>
        </div>
      </header>
      <div className="grid gap-4 lg:grid-cols-2">
        <HeatmapColumn title="Intraday" points={intraday} />
        <HeatmapColumn title="Higher Timeframe" points={higherTimeframe} />
      </div>
    </section>
  );
}

type HeatmapColumnProps = {
  title: string;
  points: HeatmapPoint[];
};

function HeatmapColumn({ title, points }: HeatmapColumnProps) {
  return (
    <article className="flex flex-col gap-3 rounded-2xl border border-slate-200/60 bg-white/60 p-4">
      <h3 className="text-sm font-semibold text-slate-900">{title}</h3>
      <div className="grid grid-cols-2 gap-2 text-xs">
        {points.map((point) => (
          <span
            key={`${title}-${point.label}`}
            className={`rounded-full border px-3 py-2 font-semibold ${BIAS_COLOR[point.bias]}`}
          >
            {point.label}
          </span>
        ))}
      </div>
    </article>
  );
}
