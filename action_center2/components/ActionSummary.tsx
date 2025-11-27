import { ActionSummary } from "../types";

export type ActionSummaryProps = {
  summary: ActionSummary;
};

export function ActionSummaryPanel({ summary }: ActionSummaryProps) {
  const badgeClass = summary.status === "bullish"
    ? "bg-emerald-500/15 text-emerald-200 border border-emerald-400/40"
    : summary.status === "bearish"
      ? "bg-rose-500/15 text-rose-200 border border-rose-400/40"
      : "bg-slate-500/15 text-slate-200 border border-slate-400/40";

  return (
    <section className="rounded-3xl border border-slate-800 bg-slate-900/80 p-6 shadow-sm">
      <div className="flex flex-col gap-5 md:flex-row md:items-start md:justify-between">
        <div className="flex flex-col gap-3">
          <div className={`inline-flex w-max items-center gap-2 rounded-full px-3 py-1 text-xs font-semibold ${badgeClass}`}>
            <span className="h-2 w-2 rounded-full bg-current" />
            {summary.title}
          </div>
          <p className="max-w-3xl text-sm leading-relaxed text-slate-300">{summary.subtitle}</p>
          <ul className="flex flex-col gap-1 text-sm text-slate-200">
            {summary.narrative.map((line, index) => (
              <li key={index} className="flex items-center gap-2">
                <span className="text-slate-500">•</span>
                <span>{line}</span>
              </li>
            ))}
          </ul>
        </div>
        <div className="flex flex-col items-end gap-3 text-right">
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-500">Confidence</p>
            <p className="text-2xl font-semibold text-slate-50">{summary.confidenceLabel}</p>
          </div>
          <div className="h-2 w-48 rounded-full bg-slate-800">
            <span
              className="block h-full rounded-full bg-sky-400"
              style={{ width: `${summary.confidenceScore}%` }}
            />
          </div>
        </div>
      </div>
    </section>
  );
}
