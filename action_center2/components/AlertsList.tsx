import { AlertSuggestion } from "../utils/planMath";

export type AlertsListProps = {
  alerts: AlertSuggestion[];
};

export function AlertsList({ alerts }: AlertsListProps) {
  return (
    <section className="rounded-3xl border border-slate-800 bg-slate-900 p-8 shadow-sm">
      <header className="mb-6 flex flex-col gap-1">
        <h2 className="text-lg font-semibold text-slate-50">Alert Checklist</h2>
        <p className="text-sm text-slate-400">Codify reactions so execution stays systematic</p>
      </header>
      <ul className="flex flex-col gap-4">
        {alerts.map((alert) => (
          <li
            key={alert.id}
            className="flex flex-col gap-3 rounded-2xl border border-slate-800/80 bg-slate-950/60 p-5 shadow-sm md:flex-row md:items-center md:justify-between"
          >
            <div>
              <p className="text-sm font-semibold text-slate-100">{alert.label}</p>
              <p className="text-sm text-slate-400">{alert.description}</p>
            </div>
            <button
              type="button"
              className="inline-flex items-center justify-center rounded-full border border-indigo-500/40 bg-indigo-500/10 px-4 py-2 text-xs font-semibold uppercase tracking-wide text-indigo-300 transition hover:border-indigo-400 hover:bg-indigo-500/20"
            >
              Create alert
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
