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

  const handleCheckboxChange = useCallback(
    (alert: AlertSuggestion, checked: boolean) => {
      if (checked) {
        queueAlertEmail(alert);
      } else {
        setStatusMap((prev) => ({ ...prev, [alert.id]: "idle" }));
      }
    },
    [queueAlertEmail],
  );

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
          const isChecked = state === "sent" || state === "sending";
          const isDisabled = serviceError !== null || state === "sending" || state === "sent";
          const errorMessage = errorMap[alert.id];

          return (
            <li
              key={alert.id}
              className="flex flex-col gap-4 rounded-2xl border border-slate-800/80 bg-slate-950/60 p-5 shadow-sm md:flex-row md:items-center md:justify-between"
            >
              <label className="flex w-full cursor-pointer items-start gap-3 md:w-auto md:flex-1">
                <input
                  type="checkbox"
                  className="mt-1 h-4 w-4 rounded border-slate-600 bg-slate-950 text-indigo-400 focus:outline-none focus:ring-2 focus:ring-indigo-400 focus:ring-offset-0"
                  checked={isChecked}
                  disabled={isDisabled && !errorMessage}
                  onChange={(event) => handleCheckboxChange(alert, event.target.checked)}
                  aria-label={`Email me the alert ${alert.label}`}
                />
                <span>
                  <p className="text-sm font-semibold text-slate-100">{alert.label}</p>
                  <p className="text-sm text-slate-400">{alert.description}</p>
                </span>
              </label>

              <div className="text-xs font-semibold uppercase tracking-wide">
                {state === "sending" ? (
                  <span className="inline-flex items-center gap-2 rounded-full border border-indigo-500/40 bg-indigo-500/10 px-3 py-1 text-indigo-200">
                    Sending...
                  </span>
                ) : null}
                {state === "sent" ? (
                  <span className="inline-flex items-center gap-2 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1 text-emerald-200">
                    Email queued — watch your inbox
                  </span>
                ) : null}
                {state === "error" ? (
                  <div className="flex flex-col items-start gap-2 text-left text-rose-200">
                    <span>{errorMessage ?? "Could not send email."}</span>
                    <button
                      type="button"
                      className="rounded-full border border-rose-400/60 px-3 py-1 text-[11px] font-semibold uppercase tracking-wide text-rose-200 transition hover:border-rose-300"
                      onClick={() => queueAlertEmail(alert)}
                    >
                      Retry email
                    </button>
                  </div>
                ) : null}
                {state === "idle" && !errorMessage ? (
                  <span className="text-slate-500">Check to email yourself this alert</span>
                ) : null}
              </div>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
