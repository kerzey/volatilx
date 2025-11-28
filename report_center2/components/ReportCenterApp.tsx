import { useCallback, useEffect, useMemo, useState } from "react";
import type { ReportCenterBootstrapMeta, ReportCenterEntry } from "../types";
import { ReportCard } from "./ReportCard";

const normalizeSymbol = (value: unknown): string => {
  if (value === null || value === undefined) {
    return "";
  }
  return String(value).trim().toUpperCase();
};

const canonicalizeSymbol = (value: unknown): string => {
  const normalized = normalizeSymbol(value);
  if (!normalized) {
    return "";
  }
  return normalized.replace(/[^A-Z0-9]/g, "");
};

type FavoriteSet = Set<string>;

type ReportCenterAppProps = {
  reports: ReportCenterEntry[];
  favorites: string[];
  meta?: ReportCenterBootstrapMeta;
};

const buildSymbolHref = (symbol: string, meta?: ReportCenterBootstrapMeta): string => {
  const params = new URLSearchParams();
  if (meta?.selectedDate) {
    params.set("date", meta.selectedDate);
  }
  if (symbol) {
    params.set("symbol", symbol);
  }
  const query = params.toString();
  return query ? `/report-center?${query}` : "/report-center";
};

export function ReportCenterApp({ reports, favorites, meta }: ReportCenterAppProps) {
  const [favoriteSet, setFavoriteSet] = useState<FavoriteSet>(() => {
    const initial = new Set<string>();
    favorites.forEach((value) => {
      const canonical = canonicalizeSymbol(value);
      if (canonical) {
        initial.add(canonical);
      }
    });
    return initial;
  });

  useEffect(() => {
    setFavoriteSet(() => {
      const next = new Set<string>();
      favorites.forEach((value) => {
        const canonical = canonicalizeSymbol(value);
        if (canonical) {
          next.add(canonical);
        }
      });
      return next;
    });
  }, [favorites]);

  const [pending, setPending] = useState<Set<string>>(() => new Set());

  const markPending = useCallback((key: string, active: boolean) => {
    setPending((prev) => {
      const next = new Set(prev);
      if (active) {
        next.add(key);
      } else {
        next.delete(key);
      }
      return next;
    });
  }, []);

  const isFavorite = useCallback(
    (entry: ReportCenterEntry): boolean => {
      const canonical = entry.symbol_canonical || canonicalizeSymbol(entry.symbol);
      return canonical ? favoriteSet.has(canonical) : false;
    },
    [favoriteSet],
  );

  const isPending = useCallback(
    (entry: ReportCenterEntry): boolean => {
      const canonical = entry.symbol_canonical || canonicalizeSymbol(entry.symbol);
      if (!canonical) {
        return false;
      }
      return pending.has(canonical);
    },
    [pending],
  );

  const handleToggleFavorite = useCallback(
    async (entry: ReportCenterEntry, follow: boolean) => {
      const symbolDisplay = entry.symbol_display || entry.symbol;
      const normalized = normalizeSymbol(symbolDisplay);
      const canonical = entry.symbol_canonical || canonicalizeSymbol(symbolDisplay);
      if (!normalized) {
        return;
      }
      const key = canonical || normalized;

      markPending(key, true);

      try {
        const response = await fetch("/api/action-center/favorites", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          credentials: "same-origin",
          body: JSON.stringify({ symbol: normalized, follow }),
        });

        if (!response.ok) {
          throw new Error(`Favorite toggle failed with status ${response.status}`);
        }

        setFavoriteSet((prev) => {
          const next = new Set(prev);
          if (follow) {
            next.add(canonicalizeSymbol(key));
          } else {
            next.delete(canonicalizeSymbol(key));
          }
          return next;
        });
      } catch (error) {
        console.warn("[ReportCenter] Unable to update favorite", error);
        window.alert(
          follow
            ? "Unable to add symbol to favorites. Please try again."
            : "Unable to remove symbol from favorites. Please try again.",
        );
      } finally {
        markPending(key, false);
      }
    },
    [markPending],
  );

  const summaryText = useMemo(() => {
    const dateLabel = meta?.selectedDateLabel || "today";
    const focusSymbol = meta?.selectedSymbol ? meta.selectedSymbol.toUpperCase() : null;
    const count = meta?.reportCount ?? reports.length;
    const parts: string[] = [];
    parts.push(`Viewing ${count} curated report${count === 1 ? "" : "s"} for ${dateLabel}`);
    if (focusSymbol) {
      parts.push(`focused on ${focusSymbol}`);
    }
    parts.push(`(max ${meta?.maxReports ?? "?"})`);
    return parts.join(" ");
  }, [meta?.maxReports, meta?.reportCount, meta?.selectedDateLabel, meta?.selectedSymbol, reports.length]);

  const availableSymbols = meta?.availableSymbols ?? [];
  const activeSymbol = meta?.selectedSymbol ? meta.selectedSymbol.toUpperCase() : "";

  return (
    <div className="flex flex-col gap-8">
      <section className="rounded-3xl border border-slate-800 bg-slate-950/60 p-6 shadow-inner shadow-black/30">
        <p className="text-sm text-slate-300">{summaryText}</p>
        {meta?.excludedReportCount ? (
          <p className="mt-3 text-sm text-amber-300">
            Hidden {meta.excludedReportCount} report{meta.excludedReportCount === 1 ? "" : "s"} awaiting processing.
          </p>
        ) : null}
        {availableSymbols.length ? (
          <div className="mt-4 flex flex-wrap gap-2">
            {availableSymbols.map((symbol) => {
              const symbolLabel = symbol.toUpperCase();
              const href = buildSymbolHref(symbolLabel, meta);
              const isActive = symbolLabel === activeSymbol;
              return (
                <a
                  key={symbolLabel}
                  href={href}
                  className={`rounded-full border px-3 py-1 text-sm transition ${
                    isActive
                      ? "border-sky-400 bg-sky-400/10 text-sky-200"
                      : "border-slate-700 bg-slate-900/60 text-slate-200 hover:border-slate-600"
                  }`}
                >
                  {symbolLabel}
                </a>
              );
            })}
          </div>
        ) : null}
      </section>

      {reports.length === 0 ? (
        <div className="rounded-3xl border border-slate-800 bg-slate-950/60 p-12 text-center text-slate-400">
          <p className="text-lg font-medium">No AI reports found for this selection.</p>
          <p className="mt-2 text-sm">
            Try a different date{activeSymbol ? " or clear the symbol filter" : ""}.
          </p>
        </div>
      ) : (
        <div className="flex flex-col gap-8">
          {reports.map((report) => (
            <ReportCard
              key={`${report.symbol}-${report.generated_unix ?? report.generated_iso ?? report.stored_at ?? ""}`}
              report={report}
              isFavorite={isFavorite(report)}
              pending={isPending(report)}
              onToggleFavorite={handleToggleFavorite}
            />
          ))}
        </div>
      )}
    </div>
  );
}
