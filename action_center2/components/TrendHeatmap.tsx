import { TechnicalConsensus } from "../types";

export type TrendHeatmapProps = {
  consensus?: TechnicalConsensus;
};

type Recommendation = NonNullable<TechnicalConsensus["overall_recommendation"]>;
type Confidence = NonNullable<TechnicalConsensus["confidence"]>;

const RECOMMENDATION_STYLE: Record<Recommendation, string> = {
  BUY: "text-emerald-200 bg-emerald-500/20 border border-emerald-400/40",
  SELL: "text-rose-200 bg-rose-500/20 border border-rose-400/40",
  HOLD: "text-amber-200 bg-amber-500/20 border border-amber-400/40",
};

const CONFIDENCE_VALUE: Record<Confidence, number> = {
  LOW: 0.33,
  MEDIUM: 0.66,
  HIGH: 1,
};

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

  const confidencePct = clamp((CONFIDENCE_VALUE[consensus.confidence] ?? 0) * 100, 0, 100);
  const strengthRaw = typeof consensus.strength === "number" ? consensus.strength : 0;
  const strengthPct = clamp(strengthRaw * 100, 0, 100);

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
        <div className="h-2 rounded-full bg-indigo-500" style={{ width: `${percent}%` }} />
      </div>
    </article>
  );
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}
