import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { AiInsightsBootstrap, SymbolCatalogEntry } from "../types";

type Market = "equity" | "crypto";
type ActiveTab = "technical" | "priceAction" | "intelligence";

type AnalyzeState = {
  symbol: string;
  market: string;
  result?: Record<string, unknown>;
  priceAction?: unknown;
  principalPlan?: unknown;
  aiAnalysis?: unknown;
  runsRemaining?: number | null;
  raw: unknown;
};

type PendingJob = {
  jobId: string;
  includeAi: boolean;
  includePrincipal: boolean;
};

type AsyncStatus = {
  variant: "pending" | "error";
  message: string;
};

type TabDefinition = {
  id: ActiveTab;
  label: string;
};

type ErrorMeta = {
  runsRemaining?: number;
  renewsAt?: string;
  actionUrl?: string;
  actionLabel?: string;
};

const MAX_SUGGESTIONS = 9;
const LEGAL_NOTE = "This content is for informational purposes only and does not constitute financial advice.";

export function AiInsightsApp({ bootstrap }: { bootstrap: AiInsightsBootstrap }) {
  const symbolCatalogs = bootstrap.symbolCatalogs ?? {};
  const initialMarket: Market = symbolCatalogs.crypto && Array.isArray(symbolCatalogs.crypto) ? "equity" : "equity";

  const [market, setMarket] = useState<Market>(initialMarket);
  const [symbol, setSymbol] = useState("");
  const [useAiSummary, setUseAiSummary] = useState(false);
  const [usePrincipalPlan, setUsePrincipalPlan] = useState(false);
  const [includePrincipalRaw, setIncludePrincipalRaw] = useState(false);
  const [status, setStatus] = useState<"idle" | "loading" | "success" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const [errorMeta, setErrorMeta] = useState<ErrorMeta | null>(null);
  const [analysis, setAnalysis] = useState<AnalyzeState | null>(null);
  const [symbolMessage, setSymbolMessage] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<ActiveTab>("technical");
  const [pendingJob, setPendingJob] = useState<PendingJob | null>(null);
  const [asyncStatus, setAsyncStatus] = useState<AsyncStatus | null>(null);
  const [suggestionsOpen, setSuggestionsOpen] = useState(false);
  const [activeSuggestion, setActiveSuggestion] = useState(-1);

  const hideSuggestionsTimeout = useRef<number | null>(null);

  const suggestions = useMemo(() => {
    const catalog = Array.isArray(symbolCatalogs[market]) ? (symbolCatalogs[market] as SymbolCatalogEntry[]) : [];
    return filterSymbolCatalog(catalog, symbol);
  }, [symbolCatalogs, market, symbol]);

  const hasSuggestions = suggestionsOpen && suggestions.length > 0;

  useEffect(() => {
    if (!usePrincipalPlan) {
      setIncludePrincipalRaw(false);
    }
  }, [usePrincipalPlan]);

  useEffect(() => {
    return () => {
      if (hideSuggestionsTimeout.current !== null) {
        window.clearTimeout(hideSuggestionsTimeout.current);
      }
    };
  }, []);

  const availableTabs = useMemo<TabDefinition[]>(() => {
    const tabs: TabDefinition[] = [{ id: "technical", label: "Technical" }];
    if (analysis?.priceAction) {
      tabs.push({ id: "priceAction", label: "Price Action" });
    }
    const hasIntelligence = Boolean(
      usePrincipalPlan ||
        useAiSummary ||
        analysis?.principalPlan ||
        analysis?.aiAnalysis ||
        asyncStatus ||
        pendingJob,
    );
    if (hasIntelligence) {
      tabs.push({ id: "intelligence", label: "Intelligence" });
    }
    return tabs;
  }, [analysis?.priceAction, analysis?.principalPlan, analysis?.aiAnalysis, usePrincipalPlan, useAiSummary, asyncStatus, pendingJob]);

  useEffect(() => {
    if (!availableTabs.some((tab) => tab.id === activeTab)) {
      setActiveTab(availableTabs[0]?.id ?? "technical");
    }
  }, [availableTabs, activeTab]);

  useEffect(() => {
    if (!pendingJob) {
      return;
    }

    let attempts = 0;
    const maxAttempts = 40;
    let cancelled = false;
    let timeoutId: number | undefined;

    const poll = async () => {
      attempts += 1;
      try {
        const response = await fetch(`/api/ai-analysis-result/${pendingJob.jobId}`);
        if (response.ok) {
          const payload = await response.json();
          if (payload?.success && payload.status === "done") {
            const resultPayload = isRecord(payload.result) ? (payload.result as Record<string, unknown>) : {};
            setAnalysis((prev) => {
              if (!prev) {
                return prev;
              }
              return {
                ...prev,
                aiAnalysis: pendingJob.includeAi ? resultPayload.ai_analysis ?? prev.aiAnalysis : prev.aiAnalysis,
                principalPlan: pendingJob.includePrincipal ? resultPayload.principal_plan ?? prev.principalPlan : prev.principalPlan,
              };
            });
            setAsyncStatus(null);
            setPendingJob(null);
            return;
          }

          if (payload?.status === "error") {
            const rawError = typeof payload.error === "string" ? payload.error : "Analysis failed during finalization.";
            const friendly = rawError.toLowerCase().includes("timeout")
              ? "Our trading wizard timed out. Give it another go in a moment."
              : rawError;
            setAsyncStatus({ variant: "error", message: friendly });
            setPendingJob(null);
            return;
          }
        }
      } catch (pollError) {
        console.warn("[AiInsights] Polling failed", pollError);
      }

      if (attempts >= maxAttempts) {
        setAsyncStatus({
          variant: "error",
          message: "Our agents need more time than usual. Please retry shortly.",
        });
        setPendingJob(null);
        return;
      }

      if (!cancelled) {
        timeoutId = window.setTimeout(poll, 3000);
      }
    };

    poll();

    return () => {
      cancelled = true;
      if (timeoutId) {
        window.clearTimeout(timeoutId);
      }
    };
  }, [pendingJob]);

  const handleSymbolChange = useCallback((value: string) => {
    setSymbol(value.toUpperCase());
    setSuggestionsOpen(true);
    setActiveSuggestion(-1);
  }, []);

  const handleSuggestionSelect = useCallback((entry: SymbolCatalogEntry) => {
    setSymbol(entry.ticker.toUpperCase());
    setSuggestionsOpen(false);
    setActiveSuggestion(-1);
  }, []);

  const handleSymbolKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLInputElement>) => {
      if (!hasSuggestions) {
        if (event.key === "Escape") {
          setSuggestionsOpen(false);
        }
        return;
      }

      if (event.key === "ArrowDown") {
        event.preventDefault();
        setActiveSuggestion((prev) => {
          const next = prev + 1;
          if (next >= suggestions.length) {
            return 0;
          }
          return next;
        });
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        setActiveSuggestion((prev) => {
          const next = prev - 1;
          if (next < 0) {
            return suggestions.length - 1;
          }
          return next;
        });
      } else if (event.key === "Enter") {
        if (activeSuggestion >= 0 && suggestions[activeSuggestion]) {
          event.preventDefault();
          handleSuggestionSelect(suggestions[activeSuggestion]);
        }
      } else if (event.key === "Escape") {
        setSuggestionsOpen(false);
      }
    },
    [hasSuggestions, suggestions, activeSuggestion, handleSuggestionSelect],
  );

  const handleSubmit = useCallback(
    async (event: React.FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      const trimmed = symbol.trim().toUpperCase();
      if (!trimmed) {
        setError("Please enter a symbol to analyze.");
        setStatus("error");
        setErrorMeta(null);
        return;
      }

      setStatus("loading");
      setError(null);
      setErrorMeta(null);
      setAnalysis(null);
      setSymbolMessage(null);
      setAsyncStatus(null);
      setPendingJob(null);

      const payload = {
        stock_symbol: trimmed,
        market,
        use_ai_analysis: useAiSummary,
        use_principal_agent: usePrincipalPlan,
        include_principal_raw_results: includePrincipalRaw && usePrincipalPlan,
      };

      try {
        const response = await fetch("/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        const data = (await response.json().catch(() => null)) as Record<string, unknown> | null;
        if (!response.ok || !data || data.success === false) {
          const message = selectErrorMessage(data);
          setError(message);
          setStatus("error");
          setErrorMeta({
            runsRemaining: typeof data?.runs_remaining === "number" ? data.runs_remaining : undefined,
            renewsAt: typeof data?.renews_at === "string" ? data.renews_at : undefined,
            actionUrl: typeof data?.action_url === "string" ? data.action_url : undefined,
            actionLabel: typeof data?.action_label === "string" ? data.action_label : undefined,
          });
          return;
        }

        const resultRecord = isRecord(data.result) ? (data.result as Record<string, unknown>) : undefined;
        const normalizedSymbol = findMatchingSymbolKey(resultRecord, trimmed) ?? trimmed;

        setAnalysis({
          symbol: normalizedSymbol,
          market: typeof data.market === "string" ? data.market : market,
          result: resultRecord,
          priceAction: data.price_action,
          principalPlan: data.principal_plan,
          aiAnalysis: data.ai_analysis,
          runsRemaining: typeof data.runs_remaining === "number" ? data.runs_remaining : null,
          raw: data,
        });
        setSymbolMessage(typeof data.symbol_message === "string" ? data.symbol_message : null);
        setStatus("success");
        setActiveTab("technical");
        setError(null);
        setErrorMeta(null);

        if (typeof data.market === "string" && (data.market === "equity" || data.market === "crypto")) {
          setMarket(data.market);
        }

        if (typeof data.ai_job_id === "string" && data.ai_job_id && (useAiSummary || usePrincipalPlan)) {
          setPendingJob({
            jobId: data.ai_job_id,
            includeAi: useAiSummary,
            includePrincipal: usePrincipalPlan,
          });
          setAsyncStatus({ variant: "pending", message: "AI agents are finalizing deeper analysis." });
        }
      } catch (submitError) {
        console.error("[AiInsights] Analyze request failed", submitError);
        setError("Network error. Please try again.");
        setStatus("error");
        setErrorMeta(null);
      }
    },
    [symbol, market, useAiSummary, usePrincipalPlan, includePrincipalRaw],
  );

  const buttonLabel = useMemo(() => {
    if (status === "loading") {
      const tasks = [];
      if (useAiSummary) tasks.push("AI Summary");
      if (usePrincipalPlan) tasks.push("Strategy");
      return tasks.length ? `Analyzing (${tasks.join(" + ")})…` : "Analyzing…";
    }
    if (useAiSummary || usePrincipalPlan) {
      return "Analyze & Sync";
    }
    return "Analyze Symbol";
  }, [status, useAiSummary, usePrincipalPlan]);

  const disableSubmit = status === "loading";

  const renderPanel = () => {
    if (status === "idle") {
      return <IntroPanel />;
    }

    if (status === "loading") {
      return <LoadingPanel symbol={symbol} useAi={useAiSummary} usePrincipal={usePrincipalPlan} />;
    }

    if (status === "error") {
      return <ErrorPanel message={error ?? "Analysis failed."} meta={errorMeta} />;
    }

    if (status === "success" && analysis) {
      switch (activeTab) {
        case "technical":
          return <TechnicalPanel analysis={analysis} />;
        case "priceAction":
          return <PriceActionPanel analysis={analysis} />;
        case "intelligence":
          return (
            <IntelligencePanel
              analysis={analysis}
              asyncStatus={asyncStatus}
              useAiSummary={useAiSummary}
              usePrincipalPlan={usePrincipalPlan}
              includePrincipalRaw={includePrincipalRaw}
            />
          );
        default:
          return null;
      }
    }

    return null;
  };

  const runsMeta = analysis?.runsRemaining;

  return (
    <div className="mx-auto max-w-7xl px-6 py-10">
      <div className="grid gap-8 lg:grid-cols-[360px_minmax(0,1fr)]">
        <section className="rounded-3xl border border-slate-800 bg-slate-950/60 p-6 shadow-inner shadow-black/30 lg:sticky lg:top-20 lg:h-fit">
          <div className="flex flex-col gap-6">
            <header className="space-y-3">
              <span className="inline-flex items-center gap-2 rounded-full border border-slate-700/70 bg-slate-900/70 px-3 py-1 text-xs font-semibold uppercase tracking-[0.25em] text-slate-300">
                <span className="h-2 w-2 rounded-full bg-sky-400" aria-hidden="true" />
                Insight Console
              </span>
              <h2 className="text-2xl font-semibold tracking-tight text-white">Run AI-powered market diagnostics</h2>
              <p className="text-sm leading-relaxed text-slate-300">
                Select a market, pick your symbol, and decide which agents you want to activate. We will hydrate price action,
                consensus signals, and multi-agent strategies in one sweep.
              </p>
            </header>

            <form className="flex flex-col gap-5" onSubmit={handleSubmit}>
              <label className="flex flex-col gap-2">
                <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">Market</span>
                <select
                  className="rounded-2xl border border-slate-700 bg-slate-900/70 px-4 py-2 text-sm font-medium text-slate-100 focus:border-sky-400 focus:outline-none focus:ring-2 focus:ring-sky-500/50"
                  value={market}
                  onChange={(event) => {
                    const next = event.target.value;
                    setMarket(next === "crypto" ? "crypto" : "equity");
                    setSuggestionsOpen(false);
                  }}
                >
                  <option value="equity">Stocks (Equities)</option>
                  <option value="crypto">Crypto</option>
                </select>
              </label>

              <div className="flex flex-col gap-2">
                <label className="text-xs font-semibold uppercase tracking-wide text-slate-400" htmlFor="aiInsightsSymbol">
                  Symbol
                </label>
                <div className="relative">
                  <input
                    id="aiInsightsSymbol"
                    className="w-full rounded-2xl border border-slate-700 bg-slate-900/70 px-4 py-3 text-sm font-semibold uppercase tracking-[0.2em] text-slate-100 placeholder:tracking-normal placeholder:text-slate-500 focus:border-sky-400 focus:outline-none focus:ring-2 focus:ring-sky-500/50"
                    placeholder={market === "crypto" ? "Enter crypto pair (e.g., BTC/USD)" : "Enter stock symbol (e.g., AAPL)"}
                    autoComplete="off"
                    value={symbol}
                    onChange={(event) => handleSymbolChange(event.target.value)}
                    onFocus={() => {
                      if (hideSuggestionsTimeout.current) {
                        window.clearTimeout(hideSuggestionsTimeout.current);
                        hideSuggestionsTimeout.current = null;
                      }
                      setSuggestionsOpen(true);
                    }}
                    onBlur={() => {
                      hideSuggestionsTimeout.current = window.setTimeout(() => {
                        setSuggestionsOpen(false);
                      }, 120);
                    }}
                    onKeyDown={handleSymbolKeyDown}
                  />
                  <SymbolSuggestions
                    suggestions={suggestions}
                    visible={hasSuggestions}
                    activeIndex={activeSuggestion}
                    onSelect={(entry) => handleSuggestionSelect(entry)}
                    onHighlight={(index) => setActiveSuggestion(index)}
                  />
                </div>
              </div>

              <div className="space-y-3">
                <ToggleRow
                  label="AI Summary"
                  description="Generate a narrative brief from our generative assistant."
                  checked={useAiSummary}
                  onChange={(value) => setUseAiSummary(value)}
                />
                <ToggleRow
                  label="Multi-Agent Strategy"
                  description="Coordinate the principal agent for trade-ready playbooks."
                  checked={usePrincipalPlan}
                  onChange={(value) => setUsePrincipalPlan(value)}
                />
                <ToggleRow
                  label="Include Expert Diagnostics"
                  description="Append raw agent diagnostics for deeper inspection."
                  checked={includePrincipalRaw}
                  onChange={(value) => setIncludePrincipalRaw(value)}
                  disabled={!usePrincipalPlan}
                />
              </div>

              <button
                type="submit"
                className="inline-flex items-center justify-center gap-3 rounded-2xl bg-sky-500 px-5 py-3 text-sm font-semibold uppercase tracking-wide text-white transition hover:bg-sky-400 focus:outline-none focus:ring-2 focus:ring-sky-400/70 disabled:cursor-not-allowed disabled:opacity-60"
                disabled={disableSubmit}
              >
                {status === "loading" ? <Spinner /> : <span className="inline-flex h-2 w-2 rounded-full bg-white" aria-hidden="true" />}
                {buttonLabel}
              </button>

              {runsMeta !== undefined && runsMeta !== null ? (
                <div className="rounded-2xl border border-slate-800 bg-slate-900/70 px-4 py-3 text-xs font-semibold uppercase tracking-wide text-slate-400">
                  Runs remaining: <span className="text-slate-200">{runsMeta}</span>
                </div>
              ) : null}
            </form>
          </div>
        </section>

        <section className="rounded-3xl border border-slate-800 bg-slate-950/60 p-6 shadow-inner shadow-black/30">
          <div className="flex flex-col gap-6">
            {status === "success" && symbolMessage ? <SymbolMessageBanner message={symbolMessage} /> : null}
            {asyncStatus?.variant === "pending" ? (
              <AsyncStatusBanner variant="pending" message={asyncStatus.message} />
            ) : asyncStatus?.variant === "error" ? (
              <AsyncStatusBanner variant="error" message={asyncStatus.message} />
            ) : null}

            {availableTabs.length > 1 && status === "success" ? (
              <ResultsTabs tabs={availableTabs} activeTab={activeTab} onChange={setActiveTab} />
            ) : null}

            <div className="rounded-3xl border border-slate-800 bg-slate-950/70 p-6 shadow-inner shadow-black/40">
              {renderPanel()}
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

function SymbolSuggestions({
  suggestions,
  visible,
  activeIndex,
  onSelect,
  onHighlight,
}: {
  suggestions: SymbolCatalogEntry[];
  visible: boolean;
  activeIndex: number;
  onSelect: (entry: SymbolCatalogEntry) => void;
  onHighlight: (index: number) => void;
}) {
  if (!visible || suggestions.length === 0) {
    return null;
  }

  return (
    <div
      className="absolute z-20 mt-2 w-full rounded-2xl border border-slate-700/80 bg-slate-950/95 p-2 shadow-xl shadow-black/30"
      role="listbox"
      aria-label="Ticker suggestions"
    >
      {suggestions.map((entry, index) => {
        const isActive = index === activeIndex;
        return (
          <button
            type="button"
            key={`${entry.ticker}-${entry.company}`}
            className={
              "flex w-full items-center justify-between gap-3 rounded-xl border px-3 py-2 text-left text-sm transition " +
              (isActive
                ? "border-sky-500/40 bg-sky-500/15 text-slate-100"
                : "border-transparent text-slate-200 hover:border-slate-600 hover:bg-slate-800/60")
            }
            role="option"
            aria-selected={isActive}
            onMouseEnter={() => onHighlight(index)}
            onMouseDown={(event) => event.preventDefault()}
            onClick={() => onSelect(entry)}
          >
            <span className="font-semibold tracking-wide text-white">{entry.ticker.toUpperCase()}</span>
            <span className="text-xs text-slate-400">{entry.company}</span>
          </button>
        );
      })}
    </div>
  );
}

function ResultsTabs({ tabs, activeTab, onChange }: { tabs: TabDefinition[]; activeTab: ActiveTab; onChange: (tab: ActiveTab) => void }) {
  if (tabs.length <= 1) {
    return null;
  }

  return (
    <div className="flex flex-wrap items-center gap-2">
      {tabs.map((tab) => {
        const isActive = tab.id === activeTab;
        return (
          <button
            type="button"
            key={tab.id}
            className={
              "rounded-full border px-4 py-2 text-xs font-semibold uppercase tracking-wide transition " +
              (isActive
                ? "border-sky-500/60 bg-sky-500/20 text-sky-100 shadow-lg shadow-sky-900/40"
                : "border-slate-700 bg-slate-900/60 text-slate-300 hover:border-slate-500 hover:text-slate-100")
            }
            onClick={() => onChange(tab.id)}
          >
            {tab.label}
          </button>
        );
      })}
    </div>
  );
}

function IntroPanel() {
  return (
    <div className="space-y-4 text-sm leading-relaxed text-slate-300">
      <p>
        Choose a symbol and enable the agents you need. Technical consensus will run automatically. Price action and multi-agent
        strategies appear as soon as their pipelines finish hydrating.
      </p>
      <p className="text-xs text-slate-500">
        Tip: start with your favorite symbols or paste from the watchlist switcher on the Action Center.
      </p>
    </div>
  );
}

function LoadingPanel({ symbol, useAi, usePrincipal }: { symbol: string; useAi: boolean; usePrincipal: boolean }) {
  const tasks = [];
  if (useAi) tasks.push("AI summary");
  if (usePrincipal) tasks.push("multi-agent plan");
  const taskCopy = tasks.length ? ` (${tasks.join(" + ")})` : "";

  return (
    <div className="flex flex-col items-center gap-4 text-center">
      <Spinner size="lg" />
      <div className="space-y-2">
        <p className="text-sm font-semibold uppercase tracking-wide text-slate-200">Analyzing {symbol || "symbol"}{taskCopy}…</p>
        <p className="text-xs text-slate-500">This usually takes a few seconds. We will populate each tab as results arrive.</p>
      </div>
    </div>
  );
}

function ErrorPanel({ message, meta }: { message: string; meta: ErrorMeta | null }) {
  const renewal = meta?.renewsAt ? formatDateTime(meta.renewsAt) : null;
  return (
    <div className="space-y-4">
      <div className="rounded-2xl border border-rose-500/40 bg-rose-500/10 p-5 text-sm text-rose-100">
        <p className="font-semibold">Analysis unavailable</p>
        <p className="mt-2 leading-relaxed text-rose-100/90">{message}</p>
        {meta?.runsRemaining !== undefined ? (
          <p className="mt-3 text-xs uppercase tracking-wide text-rose-200/80">
            Runs remaining: <span className="font-semibold text-white">{meta.runsRemaining}</span>
          </p>
        ) : null}
        {renewal ? (
          <p className="mt-1 text-xs uppercase tracking-wide text-rose-200/80">Renews: {renewal}</p>
        ) : null}
        {meta?.actionUrl ? (
          <a
            href={meta.actionUrl}
            className="mt-4 inline-flex items-center gap-2 rounded-full border border-rose-400/60 bg-rose-500/20 px-4 py-2 text-xs font-semibold uppercase tracking-wide text-rose-100 transition hover:border-rose-300 hover:bg-rose-400/30"
          >
            {meta.actionLabel ?? "Manage subscription"}
          </a>
        ) : null}
      </div>
      <LegalNote />
    </div>
  );
}

function SymbolMessageBanner({ message }: { message: string }) {
  return (
    <div className="rounded-2xl border border-emerald-500/40 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-100">
      {message}
    </div>
  );
}

function AsyncStatusBanner({ variant, message }: { variant: "pending" | "error"; message: string }) {
  const baseClasses = "rounded-2xl border px-4 py-3 text-sm";
  const variantClasses =
    variant === "pending"
      ? "border-sky-500/40 bg-sky-500/10 text-sky-100"
      : "border-amber-500/40 bg-amber-500/10 text-amber-100";
  return <div className={`${baseClasses} ${variantClasses}`}>{message}</div>;
}

function TechnicalPanel({ analysis }: { analysis: AnalyzeState }) {
  const symbolData = findMatchingSymbolData(analysis.result, analysis.symbol);

  if (!symbolData) {
    return <EmptyState title="No technical data" description="Try another symbol or rerun the request." />;
  }

  const consensus = readRecord(symbolData, "consensus");
  const decisions = readRecord(symbolData, "decisions");
  const reasoning = readArray(consensus, "reasoning")?.slice(0, 4);

  const latestPrice = readNumber(consensus, "latest_price") ?? readNumber(symbolData, "latest_price");
  const overall = readString(consensus, "overall_recommendation") ?? "N/A";
  const confidence = readString(consensus, "confidence") ?? "N/A";
  const strength = readNumber(consensus, "strength");
  const buySignals = consensus?.buy_signals;
  const sellSignals = consensus?.sell_signals;
  const holdSignals = consensus?.hold_signals;

  const decisionRows = decisions
    ? Object.entries(decisions)
        .map(([timeframe, value]) => {
          if (!isRecord(value)) {
            return null;
          }
          const recommendation = readString(value, "recommendation") ?? "N/A";
          const confidenceValue = readString(value, "confidence") ?? readNumber(value, "confidence") ?? "N/A";
          const strengthValue = readNumber(value, "strength");
          const entryPrice = readNumber(value, "entry_price");
          const stop = readNumber(value, "stop_loss");
          const target = readNumber(value, "take_profit");
          const ratio = readNumber(value, "risk_reward_ratio");
          const reason = readArray(value, "reasoning");
          return {
            timeframe,
            recommendation,
            confidence: confidenceValue,
            strength: strengthValue,
            entryPrice,
            stop,
            target,
            ratio,
            reasoning: reason,
          };
        })
        .filter(Boolean)
    : [];

  return (
    <div className="space-y-6">
      <section className="rounded-3xl border border-slate-800 bg-slate-900/70 p-6 shadow-inner shadow-black/40">
        <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
          <div className="space-y-2">
            <h3 className="text-xl font-semibold text-white">{analysis.symbol} · Technical consensus</h3>
            <p className="text-sm text-slate-300">{overall}</p>
          </div>
          <div className="grid gap-3 text-sm text-slate-200 sm:grid-cols-3">
            <MetricTile label="Spot price" value={formatPrice(latestPrice)} variant="solid" />
            <MetricTile label="Confidence" value={confidence} variant="solid" />
            <MetricTile label="Strength" value={formatPercent(strength)} variant="solid" />
          </div>
        </div>
        <div className="mt-6 grid gap-3 text-xs font-semibold uppercase tracking-wide text-slate-300 sm:grid-cols-3">
          <MetricTile label="Buy signals" value={formatCount(buySignals)} compact />
          <MetricTile label="Sell signals" value={formatCount(sellSignals)} compact />
          <MetricTile label="Hold signals" value={formatCount(holdSignals)} compact />
        </div>
      </section>

      {reasoning && reasoning.length ? (
        <section className="rounded-3xl border border-slate-800 bg-slate-900/60 p-6">
          <h4 className="text-sm font-semibold uppercase tracking-wide text-slate-300">Key takeaways</h4>
          <ul className="mt-4 grid gap-3 text-sm leading-relaxed text-slate-200 sm:grid-cols-2">
            {reasoning.map((item, index) => (
              <li key={index} className="flex items-start gap-3 rounded-2xl border border-slate-800/80 bg-slate-950/60 p-4">
                <span className="mt-1 h-2 w-2 rounded-full bg-sky-400" aria-hidden="true" />
                <span>{String(item)}</span>
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <h4 className="text-sm font-semibold uppercase tracking-wide text-slate-300">Timeframe breakdown</h4>
          <span className="text-xs text-slate-500">Signals across horizon</span>
        </div>
        {decisionRows.length ? (
          <div className="grid gap-4 lg:grid-cols-2">
            {decisionRows.map((row) => (
              <article key={row?.timeframe as string} className="rounded-3xl border border-slate-800 bg-slate-900/60 p-5 shadow-inner shadow-black/30">
                <header className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-xs uppercase tracking-wide text-slate-400">{formatTimeframeLabel(row?.timeframe ?? "")}</p>
                    <p className="text-sm font-semibold text-white">{row?.recommendation ?? "N/A"}</p>
                  </div>
                  <div className="grid gap-2 text-xs text-slate-200">
                    <MetricChip label="Confidence" value={String(row?.confidence ?? "N/A")} />
                    <MetricChip label="Strength" value={formatPercent(row?.strength)} />
                  </div>
                </header>
                <div className="mt-4 grid gap-3 text-sm text-slate-200 sm:grid-cols-3">
                  <MetricTile label="Entry" value={formatPrice(row?.entryPrice)} compact />
                  <MetricTile label="Stop" value={formatPrice(row?.stop)} compact />
                  <MetricTile label="Target" value={formatPrice(row?.target)} compact />
                </div>
                <div className="mt-4 flex items-center justify-between text-xs text-slate-400">
                  <span>Risk / Reward</span>
                  <span className="font-semibold text-slate-200">{formatNumber(row?.ratio, 2)}</span>
                </div>
                {row?.reasoning?.length ? (
                  <ul className="mt-3 space-y-2 text-xs leading-relaxed text-slate-300">
                    {row.reasoning.slice(0, 3).map((reason, idx) => (
                      <li key={idx} className="flex gap-2">
                        <span className="mt-1 h-1.5 w-1.5 rounded-full bg-slate-500" aria-hidden="true" />
                        <span>{String(reason)}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="mt-3 text-xs text-slate-500">No supplemental notes.</p>
                )}
              </article>
            ))}
          </div>
        ) : (
          <EmptyState title="No timeframe diagnostics" description="Strategists did not return per-timeframe metrics." />
        )}
      </section>

      <LegalNote />
    </div>
  );
}

function PriceActionPanel({ analysis }: { analysis: AnalyzeState }) {
  const priceAction = isRecord(analysis.priceAction) ? (analysis.priceAction as Record<string, unknown>) : null;

  if (!priceAction) {
    return <EmptyState title="No price action" description="Enable the price action pipeline or retry later." />;
  }

  const success = priceAction.success !== false;
  const latestPrice = readNumber(priceAction, "latest_price");
  const generatedAt = priceAction.generated_at ?? priceAction.generatedAt;
  const summaryMessage = success
    ? "Price action insights generated."
    : typeof priceAction.error === "string"
      ? priceAction.error
      : "Price action analysis failed.";
  const overview = readRecord(priceAction, "overview");
  const perTimeframe = readRecord(priceAction, "per_timeframe");
  const errors = readArray(priceAction, "errors");

  return (
    <div className="space-y-6">
      <section className="rounded-3xl border border-slate-800 bg-slate-900/60 p-6">
        <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
          <div className="space-y-2">
            <h3 className="text-lg font-semibold text-white">Price action</h3>
            <p className="text-sm text-slate-300">{summaryMessage}</p>
          </div>
          <div className="grid gap-3 text-sm text-slate-200 sm:grid-cols-2">
            <MetricTile label="Spot price" value={formatPrice(latestPrice)} variant="solid" />
            <MetricTile label="Pipeline" value={success ? "Ready" : "Issue"} variant={success ? "solid" : "warning"} />
          </div>
        </div>
      </section>

      {overview ? (
        <section className="grid gap-4 rounded-3xl border border-slate-800 bg-slate-900/60 p-6 md:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
          <div className="space-y-4">
            <h4 className="text-sm font-semibold uppercase tracking-wide text-slate-300">Trend alignment</h4>
            <div className="rounded-2xl border border-slate-800/80 bg-slate-950/60 p-4 text-sm text-slate-200">
              {renderValue(overview.trend_alignment ?? "No dominant trend highlighted.")}
            </div>
            {Array.isArray(overview.recent_patterns) && overview.recent_patterns.length ? (
              <div>
                <h5 className="text-xs font-semibold uppercase tracking-wide text-slate-400">Recent patterns</h5>
                <ul className="mt-3 space-y-2 text-sm text-slate-200">
                  {overview.recent_patterns.slice(0, 4).map((pattern, index) => (
                    <li key={index} className="flex items-start gap-3 rounded-2xl border border-slate-800/70 bg-slate-950/50 p-3">
                      <span className="mt-1 inline-flex h-2 w-2 rounded-full bg-sky-400" aria-hidden="true" />
                      <span>{renderValue(pattern)}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}
          </div>
          <div className="space-y-4">
            <h5 className="text-xs font-semibold uppercase tracking-wide text-slate-400">Key levels</h5>
            <div className="grid gap-3 md:grid-cols-2">
              {Array.isArray(overview.key_levels) && overview.key_levels.length ? (
                overview.key_levels.slice(0, 6).map((level, index) => (
                  <div key={index} className="rounded-2xl border border-slate-800/80 bg-slate-950/60 p-4 text-sm text-slate-200">
                    <p className="text-xs uppercase tracking-wide text-slate-400">{formatTimeframeLabel(level?.timeframe)}</p>
                    <p className="mt-1 text-sm font-semibold text-white">{formatPrice(level?.price)}</p>
                    <p className="mt-1 text-xs text-slate-400">{capitalizeLabel(level?.type || "level")}</p>
                    <p className="mt-2 text-xs text-slate-500">{formatPercent(level?.distance_pct)} from spot</p>
                  </div>
                ))
              ) : (
                <div className="rounded-2xl border border-slate-800/80 bg-slate-950/60 p-4 text-sm text-slate-400">No shared support or resistance levels detected.</div>
              )}
            </div>
          </div>
        </section>
      ) : null}

      {perTimeframe ? (
        <section className="grid gap-4 lg:grid-cols-2">
          {Object.entries(perTimeframe).map(([timeframe, data]) => (
            <TimeframeCard key={timeframe} timeframe={timeframe} data={data} />
          ))}
        </section>
      ) : null}

      {errors && errors.length ? (
        <section className="rounded-3xl border border-amber-500/30 bg-amber-500/10 p-5 text-sm text-amber-100">
          <h4 className="font-semibold uppercase tracking-wide">Data warnings</h4>
          <ul className="mt-2 space-y-1 text-xs">
            {errors.map((item, index) => (
              <li key={index}>{String(item)}</li>
            ))}
          </ul>
        </section>
      ) : null}

      <LegalNote />
    </div>
  );
}

type TimeframeCardProps = { timeframe: string; data: unknown };

const TIMEFRAME_METADATA_KEYS = new Set<string>([
  "symbol",
  "tested_at",
  "testedAt",
  "timestamp",
  "generated_at",
  "generatedAt",
  "collected_at",
  "collectedAt",
  "retrieved_at",
  "retrievedAt",
]);

function TimeframeCard({ timeframe, data }: TimeframeCardProps) {
  const record = isRecord(data) ? (data as Record<string, unknown>) : null;
  const trend = record ? readRecord(record, "trend") : null;
  const direction = readString(trend, "direction") ?? "Mixed";
  const trendStrength = trend ? readNumber(trend, "strength") : undefined;
  const trendConfidence = trend ? readNumber(trend, "confidence") : undefined;
  const momentumScore = trend ? readNumber(trend, "momentum_score") ?? readNumber(trend, "score") : undefined;

  const metrics: { label: string; value: string | number | null | undefined }[] = [];
  if (direction && direction !== "Mixed") {
    metrics.push({ label: "Direction", value: direction });
  }
  if (trendStrength !== undefined) {
    metrics.push({ label: "Strength", value: formatPercent(trendStrength) });
  }
  if (trendConfidence !== undefined) {
    metrics.push({ label: "Confidence", value: formatPercent(trendConfidence) });
  }
  if (momentumScore !== undefined) {
    metrics.push({ label: "Momentum", value: formatNumber(momentumScore, 2) });
  }

  const consumedKeys = new Set<string>(["trend"]);
  const sanitizedEntries = record
    ? Object.entries(record).filter(([key]) => !TIMEFRAME_METADATA_KEYS.has(key) && key !== "trend")
    : [];

  for (const [key, value] of sanitizedEntries) {
    if (consumedKeys.has(key)) {
      continue;
    }
    if (typeof value === "number" && Number.isFinite(value)) {
      metrics.push({ label: humanizeKey(key), value: formatNumber(value, 2) });
      consumedKeys.add(key);
      continue;
    }
    if (typeof value === "string") {
      const trimmed = value.trim();
      if (trimmed && trimmed.length <= 32) {
        metrics.push({ label: humanizeKey(key), value: trimmed });
        consumedKeys.add(key);
      }
    }
    if (metrics.length >= 6) {
      break;
    }
  }

  const paragraphEntries = sanitizedEntries.filter(([key, value]) => {
    if (consumedKeys.has(key)) {
      return false;
    }
    if (typeof value === "string") {
      const trimmed = value.trim();
      if (!trimmed) {
        return false;
      }
      consumedKeys.add(key);
      return true;
    }
    return false;
  });

  const listEntries = sanitizedEntries.filter(([key, value]) => {
    if (consumedKeys.has(key)) {
      return false;
    }
    if (Array.isArray(value) && value.length) {
      consumedKeys.add(key);
      return true;
    }
    return false;
  });

  const nestedEntries = sanitizedEntries.filter(([key, value]) => {
    if (consumedKeys.has(key)) {
      return false;
    }
    if (isRecord(value)) {
      consumedKeys.add(key);
      return true;
    }
    return false;
  });

  const hasDetails = Boolean(paragraphEntries.length || listEntries.length || nestedEntries.length);

  return (
    <article className="rounded-3xl border border-slate-800 bg-slate-900/50 p-5 shadow-inner shadow-black/30">
      <header className="flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-wide text-slate-400">{formatTimeframeLabel(timeframe)}</p>
          <p className="text-sm font-semibold text-white">Timeframe signals</p>
        </div>
        <MetricChip label="Trend" value={direction} />
      </header>
      {metrics.length ? (
        <div className="mt-4 grid gap-3 sm:grid-cols-2">
          {metrics.slice(0, 6).map((metric) => (
            <MetricTile key={metric.label} label={metric.label} value={metric.value} compact />
          ))}
        </div>
      ) : null}
      {paragraphEntries.length ? (
        <div className="mt-4 space-y-3 text-sm leading-relaxed text-slate-200">
          {paragraphEntries.map(([key, value]) => (
            <section key={key} className="rounded-2xl border border-slate-800/70 bg-slate-950/50 p-3">
              <h6 className="text-xs font-semibold uppercase tracking-wide text-slate-400">{humanizeKey(key)}</h6>
              <p className="mt-1 text-sm text-slate-200">{String(value)}</p>
            </section>
          ))}
        </div>
      ) : null}
      {listEntries.length ? (
        <div className="mt-4 space-y-3">
          {listEntries.map(([key, value]) => (
            <section key={key} className="rounded-2xl border border-slate-800/70 bg-slate-950/50 p-3">
              <h6 className="text-xs font-semibold uppercase tracking-wide text-slate-400">{humanizeKey(key)}</h6>
              {renderListValue(value as unknown[])}
            </section>
          ))}
        </div>
      ) : null}
      {nestedEntries.length ? (
        <div className="mt-4 space-y-3 text-sm text-slate-200">
          {nestedEntries.map(([key, value]) => (
            <section key={key} className="rounded-2xl border border-slate-800/70 bg-slate-950/50 p-3">
              <h6 className="text-xs font-semibold uppercase tracking-wide text-slate-400">{humanizeKey(key)}</h6>
              <div className="mt-2 space-y-2 text-sm text-slate-200">{renderValue(sanitizeTimeframeRecord(value as Record<string, unknown>))}</div>
            </section>
          ))}
        </div>
      ) : null}
      {!hasDetails && !metrics.length ? (
        <p className="mt-4 text-sm text-slate-400">No additional diagnostics shared for this window.</p>
      ) : null}
    </article>
  );
}

function IntelligencePanel({
  analysis,
  asyncStatus,
  useAiSummary,
  usePrincipalPlan,
  includePrincipalRaw,
}: {
  analysis: AnalyzeState;
  asyncStatus: AsyncStatus | null;
  useAiSummary: boolean;
  usePrincipalPlan: boolean;
  includePrincipalRaw: boolean;
}) {
  const aiAnalysis = isRecord(analysis.aiAnalysis) ? (analysis.aiAnalysis as Record<string, unknown>) : null;
  const principalPlan = isRecord(analysis.principalPlan) ? (analysis.principalPlan as Record<string, unknown>) : null;

  const hasAi = Boolean(aiAnalysis);
  const hasPlan = Boolean(principalPlan);

  if (!hasAi && !hasPlan && asyncStatus?.variant === "pending") {
    return <LoadingPanel symbol={analysis.symbol} useAi={useAiSummary} usePrincipal={usePrincipalPlan} />;
  }

  if (!hasAi && !hasPlan) {
    return (
      <div className="space-y-6">
        <EmptyState title="No additional intelligence" description="Enable AI summary or the multi-agent plan to populate this tab." />
        <LegalNote />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {hasAi ? <AiSummaryCard aiAnalysis={aiAnalysis} /> : null}
      {hasPlan ? <PrincipalPlanCard plan={principalPlan} includeRaw={includePrincipalRaw} /> : null}
      <LegalNote />
    </div>
  );
}

function AiSummaryCard({ aiAnalysis }: { aiAnalysis: Record<string, unknown> }) {
  const success = aiAnalysis.success !== false;
  if (!success) {
    const errorMessage = typeof aiAnalysis.error === "string" ? aiAnalysis.error : "AI analysis failed.";
    return (
      <div className="rounded-3xl border border-rose-500/40 bg-rose-500/10 p-6 text-sm text-rose-100">
        <h4 className="text-base font-semibold text-white">AI analysis unavailable</h4>
        <p className="mt-2 text-sm text-rose-100/90">{errorMessage}</p>
      </div>
    );
  }

  const latestPrice = readNumber(aiAnalysis, "latest_price");
  const tokens = readNumber(aiAnalysis, "tokens_used");
  const analysisBody = readRecord(aiAnalysis, "analysis");

  const narrativeSections: { key: string; label: string }[] = [
    { key: "market_position", label: "Current market position" },
    { key: "short_term_strategy", label: "Short-term strategy" },
    { key: "long_term_strategy", label: "Long-term strategy" },
    { key: "risk_assessment", label: "Risk assessment" },
    { key: "action_items", label: "Action items" },
    { key: "summary", label: "Summary" },
  ];

  return (
    <section className="rounded-3xl border border-slate-800 bg-slate-900/60 p-6 shadow-inner shadow-black/30">
      <header className="flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
        <div className="space-y-2">
          <h4 className="text-lg font-semibold text-white">AI investment strategy</h4>
          <p className="text-sm text-slate-300">Generative briefing stitched from cross-agent telemetry.</p>
        </div>
        <div className="grid gap-3 text-xs font-semibold uppercase tracking-wide text-slate-300 sm:grid-cols-2">
          <MetricTile label="Spot price" value={formatPrice(latestPrice)} compact variant="solid" />
          <MetricTile label="Tokens" value={tokens !== undefined ? tokens.toString() : "N/A"} compact variant="solid" />
        </div>
      </header>

      {analysisBody ? (
        <div className="mt-6 divide-y divide-slate-800/80 rounded-3xl border border-slate-800/60 bg-slate-950/40">
          {narrativeSections
            .map(({ key, label }) => {
              const value = analysisBody[key];
              if (!value) {
                return null;
              }
              return (
                <section key={key} className="space-y-2 px-5 py-4 first:rounded-t-3xl last:rounded-b-3xl">
                  <h5 className="text-xs font-semibold uppercase tracking-wide text-slate-400">{label}</h5>
                  <div className="text-sm leading-relaxed text-slate-200">{renderValue(value)}</div>
                </section>
              );
            })
            .filter(Boolean)}
        </div>
      ) : null}

      {!analysisBody && aiAnalysis.raw_response ? (
        <div className="mt-6 rounded-3xl border border-slate-800/70 bg-slate-950/60 p-5 text-sm text-slate-200">
          {String(aiAnalysis.raw_response)}
        </div>
      ) : null}
    </section>
  );
}

function PrincipalPlanCard({ plan, includeRaw }: { plan: Record<string, unknown>; includeRaw: boolean }) {
  const success = plan.success !== false;
  if (!success) {
    const errorMessage = typeof plan.error === "string" ? plan.error : "Principal agent failed to generate a plan.";
    return (
      <div className="rounded-3xl border border-rose-500/40 bg-rose-500/10 p-6 text-sm text-rose-100">
        <h4 className="text-base font-semibold text-white">Strategy unavailable</h4>
        <p className="mt-2 text-sm text-rose-100/90">{errorMessage}</p>
      </div>
    );
  }

  const data = readRecord(plan, "data");
  if (!data) {
    return <EmptyState title="No strategy data" description="Principal agent did not return the expected payload." />;
  }

  const symbol = readString(data, "symbol") ?? "N/A";
  const generatedAt = formatDateTime(data.generated_at ?? data.generatedAt);
  const latestPrice = readNumber(data, "latest_price");
  const usage = readRecord(data, "usage");
  const tokens = usage ? readNumber(usage, "total_tokens") ?? readNumber(usage, "output_tokens") : undefined;
  const strategies = readRecord(data, "strategies");
  const context = data.context;
  const expertOutputs = readRecord(data, "trading_agent_outputs");

  const summaryCells = [
    { label: "Symbol", value: symbol },
    { label: "Generated", value: generatedAt },
    { label: "Current price", value: formatPrice(latestPrice) },
    { label: "Tokens", value: tokens !== undefined ? tokens.toString() : "N/A" },
  ];

  const strategyOrder = [
    { key: "day_trading", label: "Day trading" },
    { key: "swing_trading", label: "Swing trading" },
    { key: "longterm_trading", label: "Long-term" },
  ];

  return (
    <section className="space-y-6 rounded-3xl border border-slate-800 bg-slate-900/60 p-6 shadow-inner shadow-black/30">
      <header className="space-y-4">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <h4 className="text-lg font-semibold text-white">Multi-agent trading strategy</h4>
            <p className="text-sm text-slate-300">Principal agent synthesis tailored to {symbol}.</p>
          </div>
          <div className="grid gap-3 text-xs font-semibold uppercase tracking-wide text-slate-300 sm:grid-cols-2 lg:grid-cols-4">
            {summaryCells.map((cell) => (
              <MetricTile key={cell.label} label={cell.label} value={cell.value} compact variant="solid" />
            ))}
          </div>
        </div>
      </header>

      {strategies ? (
        <div className="space-y-4">
          {strategyOrder.map(({ key, label }) => {
            const strategy = readRecord(strategies, key);
            if (!strategy) {
              return null;
            }
            const summary = strategy.summary ?? strategy.overview;
            const buySetup = readRecord(strategy, "buy_setup");
            const sellSetup = readRecord(strategy, "sell_setup");
            const noTrade = readArray(strategy, "no_trade_zone");
            const keyLevels = strategy.key_levels;
            const nextActions = strategy.next_actions;
            if (!summary && !buySetup && !sellSetup && !noTrade && !keyLevels && !nextActions) {
              return null;
            }
            return (
              <article key={key} className="space-y-4 rounded-3xl border border-slate-800/80 bg-slate-950/50 p-5">
                <header className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                  <h5 className="text-sm font-semibold uppercase tracking-wide text-slate-300">{label}</h5>
                  {strategy.bias ? <MetricChip label="Bias" value={String(strategy.bias)} /> : null}
                </header>
                {summary ? <div className="text-sm leading-relaxed text-slate-200">{renderValue(summary)}</div> : null}
                <div className="grid gap-4 md:grid-cols-2">
                  {buySetup ? <PlanSubsection title="Buy setup" value={buySetup} /> : null}
                  {sellSetup ? <PlanSubsection title="Sell setup" value={sellSetup} /> : null}
                </div>
                <div className="grid gap-4 md:grid-cols-2">
                  {noTrade && noTrade.length ? <PlanSubsection title="No-trade zone" value={noTrade} /> : null}
                  {keyLevels ? <PlanSubsection title="Key levels" value={keyLevels} /> : null}
                  {nextActions ? <PlanSubsection title="Next actions" value={nextActions} /> : null}
                </div>
              </article>
            );
          })}
        </div>
      ) : null}

      {context ? (
        <section className="rounded-3xl border border-slate-800/80 bg-slate-950/50 p-5">
          <h5 className="text-sm font-semibold uppercase tracking-wide text-slate-300">Portfolio context</h5>
          <div className="mt-3 text-sm leading-relaxed text-slate-200">{renderValue(context)}</div>
        </section>
      ) : null}

      {expertOutputs ? <ExpertDiagnostics outputs={expertOutputs} includeRaw={includeRaw} /> : null}
    </section>
  );
}

function PlanSubsection({ title, value }: { title: string; value: unknown }) {
  const rendered = renderValue(value);
  if (!rendered) {
    return null;
  }
  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-4">
      <h6 className="text-xs font-semibold uppercase tracking-wide text-slate-300">{title}</h6>
      <div className="mt-2 text-sm leading-relaxed text-slate-200">{rendered}</div>
    </div>
  );
}

function ExpertDiagnostics({ outputs, includeRaw }: { outputs: Record<string, unknown>; includeRaw: boolean }) {
  const entries = Object.entries(outputs).filter(([, value]) => isRecord(value));
  if (!entries.length) {
    if (includeRaw) {
      return (
        <div className="rounded-3xl border border-slate-800 bg-slate-900/50 p-5 text-sm text-slate-200">
          <h5 className="text-sm font-semibold uppercase tracking-wide text-slate-300">Expert diagnostics</h5>
          <pre className="mt-3 overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/80 p-4 text-xs text-slate-400">
            {JSON.stringify(outputs, null, 2)}
          </pre>
        </div>
      );
    }
    return null;
  }

  return (
    <section className="space-y-4">
      <h5 className="text-sm font-semibold uppercase tracking-wide text-slate-300">Expert diagnostics</h5>
      <div className="space-y-4">
        {entries.map(([key, value]) => {
          const record = value as Record<string, unknown>;
          const agentName = readString(record, "agent") ?? humanizeKey(key);
          const isSuccess = record.success !== false;
          const statusLabel = isSuccess ? "Ready" : "Issue";
          const metaItems: string[] = [];
          const metaSymbol = readString(record, "symbol");
          if (metaSymbol) {
            metaItems.push(`Symbol ${metaSymbol.toUpperCase()}`);
          }
          const usage = readRecord(record, "model_usage");
          const tokens = usage ? readNumber(usage, "total_tokens") ?? readNumber(usage, "output_tokens") : undefined;
          if (tokens !== undefined) {
            metaItems.push(`Tokens ${tokens}`);
          }
          const generated = record.generated_at ?? record.collected_at ?? record.timestamp;
          if (generated) {
            metaItems.push(`Generated ${formatDateTime(generated)}`);
          }

          const output = record.agent_output ?? record.agent_result;
          const renderedOutput = renderValue(output);
          const fallback = includeRaw && record.raw_text ? (
            <pre className="mt-3 overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/80 p-4 text-xs text-slate-400">
              {String(record.raw_text)}
            </pre>
          ) : null;

          const errorMessage = !isSuccess && record.error ? (
            <p className="mt-2 text-xs text-rose-200">{String(record.error)}</p>
          ) : null;

          return (
            <article key={key} className="space-y-4 rounded-3xl border border-slate-800 bg-slate-900/50 p-5">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <p className="text-sm font-semibold text-white">{agentName}</p>
                  <p className="text-xs text-slate-400">Specialist diagnostic</p>
                </div>
                <MetricChip label="Status" value={statusLabel} variant={isSuccess ? "success" : "warning"} />
              </div>
              {metaItems.length ? (
                <div className="flex flex-wrap gap-2 text-xs text-slate-400">
                  {metaItems.map((item) => (
                    <span key={item} className="rounded-full border border-slate-700 bg-slate-900/70 px-3 py-1">
                      {item}
                    </span>
                  ))}
                </div>
              ) : null}
              {errorMessage}
              <div className="space-y-3 text-sm leading-relaxed text-slate-200">
                {renderedOutput}
                {fallback}
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}

function EmptyState({ title, description }: { title: string; description: string }) {
  return (
    <div className="rounded-3xl border border-slate-800 bg-slate-900/40 p-6 text-center text-sm text-slate-300">
      <p className="text-base font-semibold text-slate-100">{title}</p>
      <p className="mt-2 text-sm text-slate-400">{description}</p>
    </div>
  );
}

type MetricChipVariant = "default" | "success" | "warning";

function MetricChip({ label, value, variant = "default" }: { label: string; value: string | number | null | undefined; variant?: MetricChipVariant }) {
  const chipStyles: Record<MetricChipVariant, { container: string; label: string; value: string }> = {
    default: {
      container: "border border-slate-700 bg-slate-900/70",
      label: "text-slate-400",
      value: "text-slate-100",
    },
    success: {
      container: "border border-emerald-500/40 bg-emerald-500/10",
      label: "text-emerald-200",
      value: "text-emerald-100",
    },
    warning: {
      container: "border border-amber-500/40 bg-amber-500/10",
      label: "text-amber-200",
      value: "text-amber-100",
    },
  };

  const style = chipStyles[variant];
  const displayValue = value ?? "N/A";

  return (
    <div className={`inline-flex items-center gap-2 rounded-full px-3 py-1 ${style.container}`}>
      <span className={`text-[0.65rem] uppercase tracking-wide ${style.label}`}>{label}</span>
      <span className={`text-xs font-semibold ${style.value}`}>{displayValue}</span>
    </div>
  );
}

type MetricTileVariant = "default" | "solid" | "warning";

function MetricTile({ label, value, compact, variant = "default" }: { label: string; value: string | number | null | undefined; compact?: boolean; variant?: MetricTileVariant }) {
  const tileStyles: Record<MetricTileVariant, { container: string; label: string; value: string }> = {
    default: {
      container: "border border-slate-800 bg-slate-950/70",
      label: "text-slate-500",
      value: "text-slate-100",
    },
    solid: {
      container: "border border-slate-700 bg-slate-800/80",
      label: "text-slate-200",
      value: "text-white",
    },
    warning: {
      container: "border border-amber-500/40 bg-amber-500/10",
      label: "text-amber-200",
      value: "text-amber-100",
    },
  };

  const style = tileStyles[variant];
  const displayValue = value ?? "N/A";

  return (
    <div className={`flex flex-col gap-1 rounded-2xl px-4 py-3 ${style.container}`}>
      <span className={`text-xs uppercase tracking-wide ${style.label}`}>{label}</span>
      <span className={`text-sm font-semibold ${style.value} ${compact ? "" : "lg:text-base"}`}>{displayValue}</span>
    </div>
  );
}

function ToggleRow({
  label,
  description,
  checked,
  onChange,
  disabled,
}: {
  label: string;
  description: string;
  checked: boolean;
  onChange: (value: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <label className={`flex items-start justify-between gap-4 rounded-2xl border border-slate-800 bg-slate-900/60 px-4 py-3 ${disabled ? "opacity-60" : ""}`}>
      <span className="flex-1">
        <span className="block text-sm font-semibold text-white">{label}</span>
        <span className="mt-1 block text-xs leading-relaxed text-slate-400">{description}</span>
      </span>
      <input
        type="checkbox"
        className="mt-1 h-5 w-5 rounded border border-slate-600 bg-slate-900 text-sky-500 focus:outline-none focus:ring-2 focus:ring-sky-400"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        disabled={disabled}
      />
    </label>
  );
}

function Spinner({ size = "md" }: { size?: "md" | "lg" }) {
  const dimension = size === "lg" ? "h-10 w-10" : "h-4 w-4";
  return <span className={`inline-block ${dimension} animate-spin rounded-full border-2 border-slate-600 border-t-sky-400`} aria-hidden="true" />;
}

function LegalNote() {
  return <p className="text-xs italic text-slate-500">{LEGAL_NOTE}</p>;
}

function filterSymbolCatalog(catalog: SymbolCatalogEntry[], query: string): SymbolCatalogEntry[] {
  const normalized = query.trim().toLowerCase();
  if (!normalized) {
    return [];
  }
  const tickerMatches: SymbolCatalogEntry[] = [];
  const otherMatches: SymbolCatalogEntry[] = [];
  const seen = new Set<string>();
  for (const entry of catalog) {
    if (!entry || !entry.ticker) {
      continue;
    }
    const tickerLower = entry.ticker.toLowerCase();
    const companyLower = (entry.company ?? "").toLowerCase();
    if (tickerLower.startsWith(normalized)) {
      if (!seen.has(tickerLower)) {
        tickerMatches.push(entry);
        seen.add(tickerLower);
      }
      continue;
    }
    if (tickerLower.includes(normalized) || companyLower.includes(normalized)) {
      if (!seen.has(tickerLower)) {
        otherMatches.push(entry);
        seen.add(tickerLower);
      }
    }
  }
  const combined = tickerMatches.concat(otherMatches);
  return combined.slice(0, MAX_SUGGESTIONS);
}

function findMatchingSymbolData(result: Record<string, unknown> | undefined, symbol: string): Record<string, unknown> | null {
  if (!result) {
    return null;
  }
  const direct = result[symbol];
  if (isRecord(direct)) {
    return direct;
  }
  const upper = symbol.toUpperCase();
  const upperValue = result[upper];
  if (isRecord(upperValue)) {
    return upperValue;
  }
  for (const [key, value] of Object.entries(result)) {
    if (typeof key === "string" && key.toUpperCase() === upper && isRecord(value)) {
      return value;
    }
  }
  return null;
}

function findMatchingSymbolKey(result: Record<string, unknown> | undefined, symbol: string): string | null {
  if (!result) {
    return null;
  }
  if (result[symbol]) {
    return symbol;
  }
  const upper = symbol.toUpperCase();
  if (result[upper]) {
    return upper;
  }
  for (const key of Object.keys(result)) {
    if (key.toUpperCase() === upper) {
      return key;
    }
  }
  return null;
}

function readRecord(source: Record<string, unknown> | null, key: string): Record<string, unknown> | null {
  if (!source) {
    return null;
  }
  const value = source[key];
  return isRecord(value) ? (value as Record<string, unknown>) : null;
}

function readArray(source: Record<string, unknown> | null, key: string): unknown[] | null {
  if (!source) {
    return null;
  }
  const value = source[key];
  return Array.isArray(value) ? value : null;
}

function readString(source: Record<string, unknown> | null, key: string): string | undefined {
  if (!source) {
    return undefined;
  }
  const value = source[key];
  return typeof value === "string" ? value : undefined;
}

function readNumber(source: Record<string, unknown> | null | undefined, key: string): number | undefined {
  if (!source) {
    return undefined;
  }
  const value = source[key];
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return undefined;
}

function formatNumber(value: unknown, digits = 2): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "N/A";
  }
  return value.toFixed(digits);
}

function formatPrice(value: unknown): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "N/A";
  }
  return `$${value.toFixed(2)}`;
}

function formatPercent(value: unknown, digits = 2): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "N/A";
  }
  return `${value.toFixed(digits)}%`;
}

function formatCount(value: unknown): string {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value.toString();
  }
  if (typeof value === "string" && value.trim()) {
    return value;
  }
  return "0";
}

function formatDateTime(value: unknown): string {
  if (!value) {
    return "N/A";
  }
  const date = new Date(value as string);
  if (Number.isNaN(date.getTime())) {
    return typeof value === "string" ? value : "N/A";
  }
  return date.toLocaleString();
}

function humanizeKey(value: unknown): string {
  if (!value) {
    return "";
  }
  return String(value)
    .replace(/[_\-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatTimeframeLabel(value: unknown): string {
  if (!value) {
    return "";
  }
  const str = String(value).trim();
  const match = str.match(/^(\d+)([a-zA-Z]+)$/);
  if (!match) {
    return str.toUpperCase();
  }
  const quantity = Number(match[1]);
  const unitRaw = match[2].toLowerCase();
  const lookup: Record<string, string> = {
    m: "Minute",
    min: "Minute",
    h: "Hour",
    hr: "Hour",
    d: "Day",
    day: "Day",
    wk: "Week",
    w: "Week",
    mo: "Month",
    mon: "Month",
  };
  const normalized = lookup[unitRaw] || lookup[unitRaw.slice(0, 2)] || lookup[unitRaw.charAt(0)] || unitRaw.toUpperCase();
  const plural = Number.isFinite(quantity) && quantity !== 1;
  return `${quantity} ${plural ? `${normalized}s` : normalized}`;
}

function sanitizeTimeframeRecord(value: Record<string, unknown> | null): Record<string, unknown> | null {
  if (!value) {
    return null;
  }
  const entries = Object.entries(value).filter(([key]) => !TIMEFRAME_METADATA_KEYS.has(key));
  if (!entries.length) {
    return null;
  }
  return Object.fromEntries(entries);
}

function renderListValue(value: unknown[]): React.ReactNode {
  if (!value.length) {
    return <p className="text-sm text-slate-400">None provided.</p>;
  }
  const simpleItems = value.every((item) => typeof item === "string" || typeof item === "number" || typeof item === "boolean");
  if (simpleItems) {
    return (
      <ul className="space-y-2 text-sm text-slate-200">
        {value.slice(0, 8).map((item, index) => (
          <li key={index} className="flex gap-2">
            <span className="mt-1 h-1.5 w-1.5 rounded-full bg-sky-400" aria-hidden="true" />
            <span>{String(item)}</span>
          </li>
        ))}
      </ul>
    );
  }
  return (
    <div className="space-y-2 text-sm text-slate-200">
      {value.slice(0, 5).map((item, index) => (
        <div key={index} className="rounded-xl border border-slate-800/60 bg-slate-950/60 p-3">{renderValue(item)}</div>
      ))}
    </div>
  );
}

function capitalizeLabel(value: unknown): string {
  if (value === null || value === undefined) {
    return "";
  }
  const text = String(value).trim();
  if (!text) {
    return "";
  }
  const lower = text.toLowerCase();
  return lower.charAt(0).toUpperCase() + lower.slice(1);
}

function renderValue(value: unknown): React.ReactNode {
  if (value === null || value === undefined) {
    return <span className="text-sm text-slate-500">N/A</span>;
  }
  if (typeof value === "string") {
    return <span>{value}</span>;
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      return <span className="text-sm text-slate-500">N/A</span>;
    }
    return <span>{value}</span>;
  }
  if (typeof value === "boolean") {
    return <span>{value ? "Yes" : "No"}</span>;
  }
  if (Array.isArray(value)) {
    if (!value.length) {
      return <span className="text-sm text-slate-500">None</span>;
    }
    return (
      <ul className="list-disc space-y-1 pl-5">
        {value.map((item, index) => (
          <li key={index}>{renderValue(item)}</li>
        ))}
      </ul>
    );
  }
  if (isRecord(value)) {
    const entries = Object.entries(value);
    if (!entries.length) {
      return <span className="text-sm text-slate-500">None</span>;
    }
    return (
      <div className="space-y-2">
        {entries.map(([key, entryValue]) => (
          <div key={key} className="space-y-1">
            <div className="text-xs uppercase tracking-wide text-slate-400">{humanizeKey(key)}</div>
            <div className="text-sm text-slate-200">{renderValue(entryValue)}</div>
          </div>
        ))}
      </div>
    );
  }
  return <span>{String(value)}</span>;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function selectErrorMessage(data: Record<string, unknown> | null): string {
  if (!data) {
    return "Analysis failed.";
  }
  if (typeof data.error === "string" && data.error) {
    return data.error;
  }
  if (typeof data.detail === "string" && data.detail) {
    return data.detail;
  }
  if (typeof data.message === "string" && data.message) {
    return data.message;
  }
  return "Analysis failed.";
}
