import { useCallback, useMemo, useState } from "react";
import { AlertSuggestion } from "../utils/planMath";

export type AlertsListProps = {
  alerts: AlertSuggestion[];
  symbol?: string;
  latestPrice?: number;
};

type AlertSendState = "idle" | "sending" | "sent" | "error";

export function AlertsList({ alerts, symbol, latestPrice }: AlertsListProps) {
  const [statusMap, setStatusMap] = useState<Record<string, AlertSendState>>({});
  const [errorMap, setErrorMap] = useState<Record<string, string>>({});
  const [serviceError, setServiceError] = useState<string | null>(null);

  const uppercaseSymbol = useMemo(() => (symbol ? symbol.toUpperCase() : undefined), [symbol]);
  const payloadLatestPrice = useMemo(() => (Number.isFinite(latestPrice) ? Number(latestPrice) : undefined), [latestPrice]);

  const queueAlertEmail = useCallback(
    async (alert: AlertSuggestion) => {
      const currentStatus = statusMap[alert.id];
      if (currentStatus === "sending" || currentStatus === "sent") {
        return;
      }

      setServiceError(null);
      setStatusMap((prev) => ({ ...prev, [alert.id]: "sending" }));
      setErrorMap((prev) => {
        const { [alert.id]: _removed, ...rest } = prev;
        return rest;
      });

      try {
        const response = await fetch("/api/alerts/email", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "include",
          body: JSON.stringify({
            alertId: alert.id,
            label: alert.label,
            description: alert.description,
            symbol: uppercaseSymbol,
            latestPrice: payloadLatestPrice,
          }),
        });

        if (!response.ok) {
          let detail: string | undefined;
          try {
            const data = await response.json();
            if (data && typeof data.detail === "string") {
              detail = data.detail;
            }
          } catch (error) {
            // ignore json parse errors
          }

          if (response.status === 503) {
            setServiceError(detail ?? "Email alerts are currently unavailable.");
          }

          throw new Error(detail ?? `Failed to send alert email (status ${response.status}).`);
        }

        setStatusMap((prev) => ({ ...prev, [alert.id]: "sent" }));
      } catch (error) {
        const message = error instanceof Error ? error.message : "Failed to send alert email.";
        setStatusMap((prev) => ({ ...prev, [alert.id]: "error" }));
        setErrorMap((prev) => ({ ...prev, [alert.id]: message }));
      }
    },
    [payloadLatestPrice, statusMap, uppercaseSymbol],
  );

  const deactivateAlert = useCallback((alertId: string) => {
    setStatusMap((prev) => ({ ...prev, [alertId]: "idle" }));
    setErrorMap((prev) => {
      const { [alertId]: _removed, ...rest } = prev;
      return rest;
    });
  }, []);

  return (
    <section className="rounded-3xl border border-slate-800 bg-slate-900 p-8 shadow-sm">
      <header className="mb-6 flex flex-col gap-1">
        <h2 className="text-lg font-semibold text-slate-50">Alert Checklist</h2>
        <p className="text-sm text-slate-400">Codify reactions so execution stays systematic</p>
      </header>

      {serviceError ? (
        <div className="mb-4 rounded-2xl border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
          {serviceError}
        </div>
      ) : null}

      <ul className="flex flex-col gap-4">
        {alerts.map((alert) => {
          const state = statusMap[alert.id] ?? "idle";
          const isActive = state === "sent";
          const isSending = state === "sending";
          const cardTone = isActive
            ? "border-indigo-400/60 bg-indigo-900/40 shadow-lg shadow-indigo-500/10"
            : "border-slate-800/80 bg-slate-950/60";
          const errorMessage = errorMap[alert.id];

          return (
            <li
              key={alert.id}
              className={`flex flex-col gap-3 rounded-2xl p-5 shadow-sm transition-colors md:flex-row md:items-center md:justify-between ${cardTone}`}
            >
              <div>
                <p className="text-sm font-semibold text-slate-100">{alert.label}</p>
                <p className="text-sm text-slate-400">{alert.description}</p>
                {errorMessage ? <p className="mt-2 text-xs text-rose-300">{errorMessage}</p> : null}
              </div>
              <div className="flex flex-col items-stretch gap-2 md:flex-row md:items-center md:gap-3">
                <button
                  type="button"
                  className={`inline-flex items-center justify-center rounded-full px-4 py-2 text-xs font-semibold uppercase tracking-wide transition ${
                    isActive
                      ? "border border-emerald-400/50 bg-emerald-500/15 text-emerald-100 hover:border-emerald-300"
                      : isSending
                        ? "border border-indigo-400/40 bg-indigo-500/20 text-indigo-100"
                        : "border border-indigo-500/40 bg-indigo-500/10 text-indigo-300 hover:border-indigo-400 hover:bg-indigo-500/20"
                  }`}
                  onClick={() => {
                    if (isActive) {
                      deactivateAlert(alert.id);
                    } else if (!isSending) {
                      queueAlertEmail(alert);
                    }
                  }}
                  disabled={isSending}
                >
                  {isSending ? "Sending..." : isActive ? "Alert active" : "Create alert"}
                </button>
                {isActive ? (
                  <button
                    type="button"
                    className="text-xs font-semibold uppercase tracking-wide text-slate-400 transition hover:text-slate-200"
                    onClick={() => deactivateAlert(alert.id)}
                  >
                    Deselect
                  </button>
                ) : null}
                {state === "error" ? (
                  <button
                    type="button"
                    className="text-xs font-semibold uppercase tracking-wide text-rose-300 transition hover:text-rose-200"
                    onClick={() => queueAlertEmail(alert)}
                  >
                    Retry
                  </button>
                ) : null}
              </div>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
