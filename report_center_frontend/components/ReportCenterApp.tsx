import { useCallback, useEffect, useMemo, useState } from "react";
import type { ReportCenterBootstrapMeta, ReportCenterEntry } from "../types";
import { ReportCard } from "./ReportCard";
import { ReportPlanSwitcher } from "./ReportPlanSwitcher";

const MAX_VISIBLE_REPORTS = 10;

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

  const favoritePriorityList = useMemo(() => Array.from(favoriteSet), [favoriteSet]);

  const { symbolMap, allOptions } = useMemo(() => {
    type SymbolBucket = {
      symbol: string;
      symbolDisplay: string;
      canonical: string;
      isFavorite: boolean;
      reports: ReportCenterEntry[];
      latestGenerated?: string;
    };

    const map = new Map<string, SymbolBucket>();

    reports.forEach((report) => {
      const symbolKey = normalizeSymbol(report.symbol || report.symbol_display || "");
      if (!symbolKey) {
        return;
      }
      const canonical = report.symbol_canonical || canonicalizeSymbol(symbolKey);
      let bucket = map.get(symbolKey);
      if (!bucket) {
        bucket = {
          symbol: symbolKey,
          symbolDisplay: report.symbol_display || symbolKey,
          canonical,
          isFavorite: canonical ? favoriteSet.has(canonical) : false,
          reports: [],
          latestGenerated: report.generated_display || undefined,
        };
        map.set(symbolKey, bucket);
      }

      bucket.reports.push(report);

      if (!bucket.isFavorite && canonical && favoriteSet.has(canonical)) {
        bucket.isFavorite = true;
      }

      if (!bucket.latestGenerated && report.generated_display) {
        bucket.latestGenerated = report.generated_display;
      }

      if (!bucket.symbolDisplay && report.symbol_display) {
        bucket.symbolDisplay = report.symbol_display;
      }
    });

    const list = Array.from(map.values()).sort((a, b) => a.symbol.localeCompare(b.symbol));
    return { symbolMap: map, allOptions: list };
  }, [reports, favoriteSet]);

  const [activeSymbol, setActiveSymbol] = useState<string>(() => {
    const selected = normalizeSymbol(meta?.selectedSymbol);
    if (selected && symbolMap.has(selected)) {
      return selected;
    }
    return allOptions[0]?.symbol ?? "";
  });

  useEffect(() => {
    const normalizedSelected = normalizeSymbol(meta?.selectedSymbol);
    if (normalizedSelected && symbolMap.has(normalizedSelected)) {
      setActiveSymbol((prev) => (prev === normalizedSelected ? prev : normalizedSelected));
      return;
    }

    setActiveSymbol((prev) => {
      if (prev && symbolMap.has(prev)) {
        return prev;
      }
      const fallback = allOptions[0]?.symbol ?? "";
      return fallback === prev ? prev : fallback;
    });
  }, [meta?.selectedSymbol, symbolMap, allOptions]);

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

  const prioritizedOptions = useMemo(() => {
    if (allOptions.length === 0) {
      return [] as typeof allOptions;
    }

    const totalReports = reports.length;

    if (totalReports < MAX_VISIBLE_REPORTS || allOptions.length <= MAX_VISIBLE_REPORTS) {
      return allOptions;
    }

    const favoriteIndex = new Map<string, number>();
    favoritePriorityList.forEach((canonical, index) => {
      favoriteIndex.set(canonical, index);
    });

    const favoritesOnly = allOptions
      .filter((option) => option.isFavorite)
      .sort((a, b) => {
        const aRank = favoriteIndex.has(a.canonical) ? favoriteIndex.get(a.canonical)! : Number.MAX_SAFE_INTEGER;
        const bRank = favoriteIndex.has(b.canonical) ? favoriteIndex.get(b.canonical)! : Number.MAX_SAFE_INTEGER;
        if (aRank !== bRank) {
          return aRank - bRank;
        }
        return a.symbol.localeCompare(b.symbol);
      });

    const others = allOptions
      .filter((option) => !option.isFavorite)
      .sort((a, b) => a.symbol.localeCompare(b.symbol));

    let combined = [...favoritesOnly, ...others];

    if (combined.length > MAX_VISIBLE_REPORTS) {
      combined = combined.slice(0, MAX_VISIBLE_REPORTS);
    }

    if (activeSymbol) {
      const hasActive = combined.some((option) => option.symbol === activeSymbol);
      if (!hasActive) {
        const activeOption = allOptions.find((option) => option.symbol === activeSymbol);
        if (activeOption) {
          combined = [activeOption, ...combined];
        }
      }
    }

    const seen = new Set<string>();
    const deduped: typeof combined = [];
    for (const option of combined) {
      if (seen.has(option.symbol)) {
        continue;
      }
      deduped.push(option);
      seen.add(option.symbol);
      if (deduped.length === MAX_VISIBLE_REPORTS) {
        break;
      }
    }

    return deduped;
  }, [activeSymbol, allOptions, favoritePriorityList, reports.length]);

  const planOptions = useMemo(
    () =>
      prioritizedOptions.map((option) => ({
        symbol: option.symbol,
        symbolDisplay: option.symbolDisplay || option.symbol,
        isFavorite: option.isFavorite,
        generatedDisplay: option.latestGenerated,
      })),
    [prioritizedOptions],
  );

  const visibleReports = useMemo(() => {
    if (!activeSymbol) {
      return reports;
    }
    const bucket = symbolMap.get(activeSymbol);
    if (bucket) {
      return bucket.reports;
    }
    return reports;
  }, [activeSymbol, symbolMap, reports]);

  const displayedReports = useMemo(() => {
    if (visibleReports.length <= MAX_VISIBLE_REPORTS) {
      return visibleReports;
    }
    return visibleReports.slice(0, MAX_VISIBLE_REPORTS);
  }, [visibleReports]);

  const extraReportCount = visibleReports.length - displayedReports.length;

  const handleSelectSymbol = useCallback(
    (symbol: string) => {
      const normalized = normalizeSymbol(symbol);
      if (!normalized || !symbolMap.has(normalized)) {
        return;
      }
      setActiveSymbol((prev) => (prev === normalized ? prev : normalized));
    },
    [symbolMap],
  );

  const summaryCopy = useMemo(() => {
    const dateLabel = meta?.selectedDateLabel || "today";
    const focusSymbol = activeSymbol || null;
    const visibleCount = visibleReports.length;
    const displayedCount = Math.min(visibleCount, MAX_VISIBLE_REPORTS);
    const maxReports = meta?.maxReports ?? "?";
    const hasMore = visibleCount > displayedCount;

    const headline = focusSymbol
      ? `${focusSymbol} game plan`
      : `${displayedCount || "No"} active coverage`;

    const subtitleParts: string[] = [];
    subtitleParts.push(`Generated ${dateLabel}`);
    if (focusSymbol) {
      subtitleParts.push("Tap another ticker to pivot instantly");
    } else {
      subtitleParts.push("Broad-read across tracked symbols");
    }

    const footer = hasMore
      ? `Showing ${displayedCount} of ${visibleCount} drops · window max ${maxReports}`
      : `${displayedCount} drop${displayedCount === 1 ? "" : "s"} locked · window max ${maxReports}`;

    return {
      headline,
      subtitle: subtitleParts.join(" · "),
      footer,
    };
  }, [activeSymbol, meta?.maxReports, meta?.selectedDateLabel, visibleReports.length]);

  const hasReports = displayedReports.length > 0;
  const activeSymbolLabel = activeSymbol || normalizeSymbol(meta?.selectedSymbol);

  return (
    <div className="flex flex-col gap-8">
      <section>
        <div className="flex flex-col gap-4 p-6 md:flex-row md:items-center md:justify-between">
          <div className="flex flex-col gap-2">
            <span className="text-xs uppercase tracking-widest text-slate-500">Shared intel</span>
            <h2 className="text-2xl font-semibold tracking-tight text-white">{summaryCopy.headline}</h2>
            <p className="text-sm leading-relaxed text-slate-300">{summaryCopy.subtitle}</p>
          </div>
          <div className="inline-flex items-center gap-2 rounded-full border border-slate-700 bg-slate-900/70 px-4 py-2 text-xs font-semibold uppercase tracking-wide text-slate-200">
            <span className="h-2 w-2 rounded-full bg-sky-400" aria-hidden="true" />
            {summaryCopy.footer}
          </div>
        </div>
        {meta?.excludedReportCount ? (
          <p className="mt-2 px-6 pb-6 text-sm text-amber-300">
            Holding {meta.excludedReportCount} report{meta.excludedReportCount === 1 ? "" : "s"} while they hydrate.
          </p>
        ) : null}
      </section>

      {planOptions.length ? (
        <ReportPlanSwitcher options={planOptions} activeSymbol={activeSymbol} onSelect={handleSelectSymbol} />
      ) : null}

      {!hasReports ? (
        <div className="rounded-3xl border border-slate-800 bg-slate-950/60 p-12 text-center text-slate-400">
          <span className="text-xs uppercase tracking-wide text-slate-500">Status check</span>
          <p className="mt-3 text-lg font-medium text-slate-200">
            {activeSymbolLabel ? `No AI drops matched ${activeSymbolLabel}.` : "No AI drops matched this slice."}
          </p>
          <p className="mt-2 text-sm">
            Nudge the date or adjust the symbol filter to rerun coverage.
          </p>
        </div>
      ) : (
        <div className="flex flex-col gap-8">
          {extraReportCount > 0 ? (
            <p className="text-xs uppercase tracking-wide text-slate-500">
              Showing top {MAX_VISIBLE_REPORTS} drops · {extraReportCount} more in reserve
            </p>
          ) : null}
          {displayedReports.map((report) => (
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
