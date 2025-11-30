import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  PrincipalPlan,
  PrincipalPlanOption,
  StrategyKey,
  StrategyPlan,
  TradeIntent,
  LivePriceSnapshot,
  LivePriceMeta,
} from "../types";
import { deriveTradeState, buildAlertSuggestions, deriveActionSummary } from "../utils/planMath";
import { TradeStateHeader } from "./TradeStateHeader";
import { StrategySelector } from "./StrategySelector";
import { PriceGauge } from "./PriceGauge";
import { ScenarioCards } from "./ScenarioCards";
import { PnLPreview } from "./PnLPreview";
import { AlertsList } from "./AlertsList";
import { TrendHeatmap } from "./TrendHeatmap";
import { PlanSwitcher } from "./PlanSwitcher";
import { ActionSummaryPanel } from "./ActionSummary";
import { IntentSelector } from "./IntentSelector";

export type ActionCenterProps = {
  plan: PrincipalPlan;
  initialStrategy?: StrategyKey;
  planOptions?: PrincipalPlanOption[];
};

const DEFAULT_STRATEGY: StrategyKey = "day_trading";
const STRATEGY_KEYS: StrategyKey[] = ["day_trading", "swing_trading", "longterm_trading"];

type PersistedSymbolPrefs = {
  strategy?: StrategyKey;
  intent?: TradeIntent;
  alerts?: string[];
};

type PersistedSelections = {
  activeSymbol?: string;
  perSymbol: Record<string, PersistedSymbolPrefs>;
};

type FavoriteRecord = {
  symbol: string;
  label: string;
};

declare global {
  interface Window {
    __ACTION_CENTER_PRICE_OVERRIDES__?: Record<string, LivePriceSnapshot>;
  }
}

const SANITIZE_PATTERN = /[^A-Z0-9.\-]/g;
const CANONICAL_PATTERN = /[^A-Z0-9]/g;

const sanitizeSymbolValue = (value: unknown): string => {
  if (typeof value !== "string") {
    return "";
  }
  const trimmed = value.trim().toUpperCase();
  if (!trimmed) {
    return "";
  }
  return trimmed.replace(SANITIZE_PATTERN, "").slice(0, 24);
};

const canonicalizeSymbolValue = (value: unknown): string => {
  const sanitized = sanitizeSymbolValue(value);
  if (!sanitized) {
    return "";
  }
  return sanitized.replace(CANONICAL_PATTERN, "");
};

const STORAGE_KEY = "volatilx:action-center:selections";
const LIVE_PRICE_POLL_MS = 5000;

const isStrategyKey = (value: unknown): value is StrategyKey =>
  typeof value === "string" && STRATEGY_KEYS.includes(value as StrategyKey);

const isTradeIntent = (value: unknown): value is TradeIntent => value === "buy" || value === "sell" || value === "both";

function readPersistedSelections(): PersistedSelections {
  if (typeof window === "undefined") {
    return { perSymbol: {} };
  }

  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return { perSymbol: {} };
    }
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") {
      return { perSymbol: {} };
    }

    const activeSymbol = typeof parsed.activeSymbol === "string" ? parsed.activeSymbol : undefined;
    const perSymbolInput = parsed.perSymbol && typeof parsed.perSymbol === "object" ? parsed.perSymbol : {};
    const perSymbol: Record<string, PersistedSymbolPrefs> = {};

    Object.entries(perSymbolInput).forEach(([symbol, prefs]) => {
      if (!prefs || typeof prefs !== "object") {
        return;
      }
      const strategyValue = (prefs as PersistedSymbolPrefs).strategy;
      const strategy = isStrategyKey(strategyValue) ? strategyValue : undefined;
      const intent = isTradeIntent((prefs as PersistedSymbolPrefs).intent) ? (prefs as PersistedSymbolPrefs).intent : undefined;
      const alertsRaw = Array.isArray((prefs as PersistedSymbolPrefs).alerts)
        ? (prefs as PersistedSymbolPrefs).alerts.filter((value): value is string => typeof value === "string")
        : [];
      const alerts = alertsRaw.length ? Array.from(new Set(alertsRaw)) : undefined;
      if (strategy || intent || alerts) {
        perSymbol[symbol] = {
          ...(strategy ? { strategy: strategy as StrategyKey } : {}),
          ...(intent ? { intent } : {}),
          ...(alerts ? { alerts } : {}),
        };
      }
    });

    return {
      ...(activeSymbol ? { activeSymbol } : {}),
      perSymbol,
    };
  } catch (error) {
    return { perSymbol: {} };
  }
}

function writePersistedSelections(selections: PersistedSelections): void {
  if (typeof window === "undefined") {
    return;
  }

  try {
    const payload: PersistedSelections = { perSymbol: {} };
    if (selections.activeSymbol && typeof selections.activeSymbol === "string") {
      payload.activeSymbol = selections.activeSymbol;
    }

    if (selections.perSymbol && typeof selections.perSymbol === "object") {
      const sanitized: Record<string, PersistedSymbolPrefs> = {};
      Object.entries(selections.perSymbol).forEach(([symbol, prefs]) => {
        if (!prefs || typeof prefs !== "object") {
          return;
        }
        const entry: PersistedSymbolPrefs = {};
        if (isStrategyKey(prefs.strategy)) {
          entry.strategy = prefs.strategy;
        }
        if (isTradeIntent(prefs.intent)) {
          entry.intent = prefs.intent;
        }
        if (Array.isArray(prefs.alerts)) {
          const alerts = prefs.alerts.filter((value): value is string => typeof value === "string");
          if (alerts.length) {
            entry.alerts = Array.from(new Set(alerts));
          }
        }
        if (entry.strategy || entry.intent || (entry.alerts && entry.alerts.length)) {
          sanitized[symbol] = entry;
        }
      });
      payload.perSymbol = sanitized;
    }

    if (!payload.activeSymbol && !Object.keys(payload.perSymbol).length) {
      window.localStorage.removeItem(STORAGE_KEY);
      return;
    }

    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  } catch (error) {
    // Swallow storage errors (e.g., private mode)
  }
}

export function ActionCenterPage({ plan, initialStrategy, planOptions = [] }: ActionCenterProps) {
  const normalizedOptions = useMemo(() => {
    const seen = new Set<string>();
    const list: PrincipalPlanOption[] = [];

    function pushOption(option: PrincipalPlanOption | null | undefined) {
      if (!option || !option.plan || !option.plan.symbol) {
        return;
      }
      const symbol = option.symbol || option.plan.symbol;
      if (seen.has(symbol)) {
        return;
      }
      seen.add(symbol);
      list.push({
        symbol,
        symbolDisplay: option.symbolDisplay || option.plan.symbol_display || symbol,
        isFavorite: option.isFavorite ?? Boolean(option.plan.is_favorited),
        plan: option.plan,
      });
    }

    pushOption({
      symbol: plan.symbol,
      symbolDisplay: plan.symbol_display || plan.symbol,
      isFavorite: Boolean(plan.is_favorited),
      plan,
    });

    if (Array.isArray(planOptions)) {
      planOptions.forEach((option) => pushOption(option));
    }

    return list;
  }, [plan, planOptions]);

  const initialSelectionsRef = useRef<PersistedSelections | null>(null);
  if (initialSelectionsRef.current === null) {
    initialSelectionsRef.current = readPersistedSelections();
  }
  const initialSelections = initialSelectionsRef.current ?? { perSymbol: {} };
  const priceSnapshotsRef = useRef<Record<string, LivePriceSnapshot>>(
    (typeof window !== "undefined" && window.__ACTION_CENTER_PRICE_OVERRIDES__) || {},
  );

  const [activeSymbol, setActiveSymbol] = useState<string>(initialSelections.activeSymbol || plan.symbol);
  const [symbolPrefs, setSymbolPrefs] = useState<Record<string, PersistedSymbolPrefs>>(initialSelections.perSymbol);
  const [favoriteMap, setFavoriteMap] = useState<Map<string, FavoriteRecord>>(() => {
    const initial = new Map<string, FavoriteRecord>();
    normalizedOptions.forEach((option) => {
      if (!option?.symbol || !option.isFavorite) {
        return;
      }
      const sanitized = sanitizeSymbolValue(option.symbol);
      const canonical = canonicalizeSymbolValue(option.symbol);
      if (!sanitized || !canonical) {
        return;
      }
      initial.set(canonical, {
        symbol: sanitized,
        label: option.symbolDisplay || sanitized,
      });
    });
    return initial;
  });
  const [favoritesLoading, setFavoritesLoading] = useState(false);
  const [favoritePending, setFavoritePending] = useState<Set<string>>(() => new Set());

  const handleAlertsChange = useCallback(
    (symbolKey: string, activeAlerts: string[]) => {
      const deduped = Array.from(new Set(activeAlerts.filter((id): id is string => typeof id === "string")));
      setSymbolPrefs((prev) => {
        const existing = prev[symbolKey] ?? {};
        const currentAlerts = existing.alerts ?? [];
        if (currentAlerts.length === deduped.length && currentAlerts.every((id, idx) => id === deduped[idx])) {
          return prev;
        }

        const nextPrefs: PersistedSymbolPrefs = { ...existing };
        if (deduped.length) {
          nextPrefs.alerts = deduped;
        } else {
          delete nextPrefs.alerts;
        }

        if (!nextPrefs.strategy && !nextPrefs.intent && (!nextPrefs.alerts || nextPrefs.alerts.length === 0)) {
          const { [symbolKey]: _removed, ...rest } = prev;
          return rest;
        }

        return {
          ...prev,
          [symbolKey]: nextPrefs,
        };
      });
    },
    [setSymbolPrefs],
  );

  const optionLabelLookup = useMemo(() => {
    const lookup = new Map<string, string>();
    normalizedOptions.forEach((option) => {
      const canonical = canonicalizeSymbolValue(option.symbol);
      if (!canonical) {
        return;
      }
      lookup.set(canonical, option.symbolDisplay || option.symbol);
    });
    return lookup;
  }, [normalizedOptions]);

  useEffect(() => {
    setFavoriteMap((prev) => {
      let changed = false;
      const next = new Map(prev);
      normalizedOptions.forEach((option) => {
        const canonical = canonicalizeSymbolValue(option.symbol);
        const sanitized = sanitizeSymbolValue(option.symbol);
        if (!canonical || !sanitized) {
          return;
        }
        const preferredLabel = option.symbolDisplay || sanitized;
        const existing = next.get(canonical);
        if (option.isFavorite && !existing) {
          next.set(canonical, { symbol: sanitized, label: preferredLabel });
          changed = true;
        } else if (existing && existing.label !== preferredLabel) {
          next.set(canonical, { ...existing, label: preferredLabel });
          changed = true;
        }
      });
      return changed ? next : prev;
    });
  }, [normalizedOptions]);

  const loadFavorites = useCallback(async () => {
    setFavoritesLoading(true);
    try {
      const response = await fetch("/api/action-center/favorites", { credentials: "same-origin" });
      if (!response.ok) {
        throw new Error(`Failed to load favorites (status ${response.status})`);
      }
      const data = await response.json();
      if (!data || !Array.isArray(data.symbols)) {
        return;
      }
      setFavoriteMap(() => {
        const next = new Map<string, FavoriteRecord>();
        data.symbols.forEach((entry: unknown) => {
          const sanitized = sanitizeSymbolValue(entry);
          const canonical = canonicalizeSymbolValue(entry);
          if (!sanitized || !canonical) {
            return;
          }
          const label = optionLabelLookup.get(canonical) ?? sanitized;
          next.set(canonical, { symbol: sanitized, label });
        });
        return next;
      });
    } catch (error) {
      console.warn("[ActionCenter] Failed to load favorites", error);
    } finally {
      setFavoritesLoading(false);
    }
  }, [optionLabelLookup]);

  useEffect(() => {
    loadFavorites();
  }, [loadFavorites]);

  const markFavoritePending = useCallback((canonical: string, active: boolean) => {
    if (!canonical) {
      return;
    }
    setFavoritePending((prev) => {
      const next = new Set(prev);
      if (active) {
        next.add(canonical);
      } else {
        next.delete(canonical);
      }
      return next;
    });
  }, []);

  const handleToggleFavorite = useCallback(
    async (symbol: string, follow: boolean) => {
      const sanitized = sanitizeSymbolValue(symbol);
      const canonical = canonicalizeSymbolValue(symbol);
      if (!sanitized || !canonical) {
        return;
      }

      markFavoritePending(canonical, true);
      try {
        const response = await fetch("/api/action-center/favorites", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          credentials: "same-origin",
          body: JSON.stringify({ symbol: sanitized, follow }),
        });
        if (!response.ok) {
          throw new Error(`Favorite toggle failed with status ${response.status}`);
        }
        setFavoriteMap((prev) => {
          const next = new Map(prev);
          if (follow) {
            const nextLabel = optionLabelLookup.get(canonical) ?? sanitized;
            next.set(canonical, { symbol: sanitized, label: nextLabel });
          } else {
            next.delete(canonical);
          }
          return next;
        });
      } catch (error) {
        console.warn("[ActionCenter] Unable to toggle favorite", error);
        window.alert(
          follow ? "Unable to add symbol to favorites. Please try again." : "Unable to remove symbol from favorites. Please try again.",
        );
      } finally {
        markFavoritePending(canonical, false);
      }
    },
    [markFavoritePending, optionLabelLookup],
  );

  const favoriteEntries = useMemo(() => {
    const entries = Array.from(favoriteMap.entries()).map(([canonical, record]) => ({
      canonical,
      symbol: record.symbol,
      label: optionLabelLookup.get(canonical) ?? record.label ?? record.symbol,
      isTracked: optionLabelLookup.has(canonical),
      isPending: favoritePending.has(canonical),
      isFavorite: true,
    }));
    return entries.sort((a, b) => a.label.localeCompare(b.label));
  }, [favoriteMap, favoritePending, optionLabelLookup]);

  const planSwitcherOptions = useMemo(
    () =>
      normalizedOptions.map((option) => {
        const canonical = canonicalizeSymbolValue(option.symbol);
        if (!canonical) {
          return option;
        }
        const isFavorite = favoriteMap.has(canonical);
        if ((option.isFavorite ?? false) === isFavorite) {
          return option;
        }
        return {
          ...option,
          isFavorite,
        };
      }),
    [favoriteMap, normalizedOptions],
  );

  const handleSelectSymbolFromSwitcher = useCallback(
    (symbol: string) => {
      if (!symbol) {
        return;
      }
      const matched = normalizedOptions.find((option) => option.symbol === symbol);
      if (!matched || matched.symbol === activeSymbol) {
        return;
      }
      setActiveSymbol(matched.symbol);
    },
    [activeSymbol, normalizedOptions],
  );

  useEffect(() => {
    const availableSymbols = normalizedOptions.map((option) => option.symbol);
    if (availableSymbols.includes(activeSymbol)) {
      return;
    }

    const persistedSymbol = initialSelectionsRef.current?.activeSymbol;
    if (persistedSymbol && availableSymbols.includes(persistedSymbol)) {
      setActiveSymbol(persistedSymbol);
      return;
    }

    const fallback = availableSymbols[0] ?? plan.symbol;
    if (fallback && fallback !== activeSymbol) {
      setActiveSymbol(fallback);
    }
  }, [activeSymbol, normalizedOptions, plan.symbol]);

  const activeOption = useMemo(
    () => normalizedOptions.find((option) => option.symbol === activeSymbol) ?? normalizedOptions[0],
    [activeSymbol, normalizedOptions],
  );

  const getMarketForSymbol = useCallback((symbol?: string) => {
    if (!symbol) {
      return undefined;
    }
    const key = sanitizeSymbolValue(symbol);
    if (!key) {
      return undefined;
    }
    return priceSnapshotsRef.current[key]?.market;
  }, []);

  const activePlan = activeOption?.plan ?? plan;
  const activePrefs = symbolPrefs[activeSymbol] ?? {};

  const baseLatestPrice = typeof activePlan?.latest_price === "number" ? activePlan.latest_price : null;
  const [livePrice, setLivePrice] = useState<number | null>(baseLatestPrice);
  const [livePriceMeta, setLivePriceMeta] = useState<LivePriceMeta | null>(null);

  useEffect(() => {
    setLivePrice(baseLatestPrice);
    setLivePriceMeta(null);
  }, [baseLatestPrice, activeOption?.symbol]);

  const effectiveLatestPrice = useMemo(() => {
    if (typeof livePrice === "number" && Number.isFinite(livePrice)) {
      return livePrice;
    }
    if (typeof baseLatestPrice === "number" && Number.isFinite(baseLatestPrice)) {
      return baseLatestPrice;
    }
    return 0;
  }, [baseLatestPrice, livePrice]);

  useEffect(() => {
    const symbol = activeOption?.symbol;
    if (!symbol) {
      return undefined;
    }

    let cancelled = false;
    let timerId: number | undefined;

    const poll = async () => {
      const params = new URLSearchParams({ symbol });
      const derivedMarket = getMarketForSymbol(symbol);
      if (derivedMarket) {
        params.set("market", derivedMarket);
      }

      try {
        const response = await fetch(`/api/live-price?${params.toString()}`, { credentials: "same-origin" });
        if (cancelled) {
          return;
        }

        if (response.status === 202) {
          setLivePriceMeta((prev) => ({ ...(prev ?? {}), market: derivedMarket ?? prev?.market }));
        } else if (response.ok) {
          const data = await response.json();
          if (cancelled) {
            return;
          }
          if (typeof data.price === "number") {
            setLivePrice(data.price);
            setLivePriceMeta({
              timestamp: data.timestamp || data.received_at,
              received_at: data.received_at,
              source: data.source || "live-feed",
              market: data.market || derivedMarket,
            });
            const key = sanitizeSymbolValue(symbol);
            if (key) {
              priceSnapshotsRef.current[key] = {
                ...priceSnapshotsRef.current[key],
                market: data.market || derivedMarket || priceSnapshotsRef.current[key]?.market,
              };
            }
          }
        }
      } catch (error) {
        if (!cancelled) {
          setLivePriceMeta((prev) => ({ ...(prev ?? {}), error: true }));
        }
      } finally {
        if (!cancelled) {
          timerId = window.setTimeout(poll, LIVE_PRICE_POLL_MS);
        }
      }
    };

    poll();

    return () => {
      cancelled = true;
      if (timerId) {
        window.clearTimeout(timerId);
      }
    };
  }, [activeOption?.symbol, getMarketForSymbol]);

  const [selectedStrategy, setSelectedStrategy] = useState<StrategyKey>(() => {
    if (activePrefs.strategy && activePlan.strategies[activePrefs.strategy]) {
      return activePrefs.strategy;
    }
    if (initialStrategy && activePlan.strategies[initialStrategy]) {
      return initialStrategy;
    }
    if (activePlan.strategies[DEFAULT_STRATEGY]) {
      return DEFAULT_STRATEGY;
    }
    const availableKeys = Object.keys(activePlan.strategies ?? {}) as StrategyKey[];
    return availableKeys[0] ?? DEFAULT_STRATEGY;
  });
  const [selectedIntent, setSelectedIntent] = useState<TradeIntent>(() => {
    return activePrefs.intent && isTradeIntent(activePrefs.intent) ? activePrefs.intent : "buy";
  });

  useEffect(() => {
    if (activePlan.strategies[selectedStrategy]) {
      return;
    }
    const available = activePlan.strategies ?? {};
    const fallback = (() => {
      if (initialStrategy && available[initialStrategy]) {
        return initialStrategy;
      }
      if (available[DEFAULT_STRATEGY]) {
        return DEFAULT_STRATEGY;
      }
      const keys = Object.keys(available) as StrategyKey[];
      return keys[0] ?? DEFAULT_STRATEGY;
    })();
    if (fallback !== selectedStrategy) {
      setSelectedStrategy(fallback);
    }
  }, [activePlan, initialStrategy, selectedStrategy]);

  const previousSymbolRef = useRef<string>(activeSymbol);

  useEffect(() => {
    const previousSymbol = previousSymbolRef.current;
    if (activeSymbol === previousSymbol) {
      return;
    }
    previousSymbolRef.current = activeSymbol;

    const prefs = symbolPrefs[activeSymbol] ?? {};
    const available = activePlan.strategies ?? {};

    let nextStrategy: StrategyKey | undefined;
    if (prefs.strategy && available[prefs.strategy]) {
      nextStrategy = prefs.strategy;
    } else if (initialStrategy && available[initialStrategy]) {
      nextStrategy = initialStrategy;
    } else if (available[DEFAULT_STRATEGY]) {
      nextStrategy = DEFAULT_STRATEGY;
    } else {
      const keys = Object.keys(available) as StrategyKey[];
      nextStrategy = keys[0] ?? selectedStrategy;
    }

    if (nextStrategy && nextStrategy !== selectedStrategy) {
      setSelectedStrategy(nextStrategy);
    }

    const nextIntent = prefs.intent && isTradeIntent(prefs.intent) ? prefs.intent : "buy";
    if (nextIntent !== selectedIntent) {
      setSelectedIntent(nextIntent);
    }
  }, [activePlan, activeSymbol, initialStrategy, selectedIntent, selectedStrategy, symbolPrefs]);

  useEffect(() => {
    if (!activeSymbol) {
      return;
    }
    setSymbolPrefs((prev) => {
      const existing = prev[activeSymbol] ?? {};
      if (existing.strategy === selectedStrategy && existing.intent === selectedIntent) {
        return prev;
      }
      return {
        ...prev,
        [activeSymbol]: {
          ...existing,
          strategy: selectedStrategy,
          intent: selectedIntent,
        },
      };
    });
  }, [activeSymbol, selectedIntent, selectedStrategy]);

  useEffect(() => {
    writePersistedSelections({ activeSymbol, perSymbol: symbolPrefs });
  }, [activeSymbol, symbolPrefs]);

  const strategyPlan: StrategyPlan = activePlan.strategies[selectedStrategy] ?? activePlan.strategies[DEFAULT_STRATEGY];

  const tradeState = useMemo(
    () =>
      deriveTradeState({
        plan: strategyPlan,
        latestPrice: effectiveLatestPrice,
      }),
    [effectiveLatestPrice, strategyPlan],
  );

  const alertSuggestions = useMemo(
    () => buildAlertSuggestions(effectiveLatestPrice, strategyPlan),
    [effectiveLatestPrice, strategyPlan],
  );

  const actionSummary = useMemo(
    () =>
      deriveActionSummary({
        strategy: strategyPlan,
        tradeState,
        latestPrice: effectiveLatestPrice,
        intent: selectedIntent,
      }),
    [effectiveLatestPrice, selectedIntent, strategyPlan, tradeState],
  );

  return (
    <div className="min-h-screen bg-transparent py-10 text-slate-50">
      <div className="mx-auto flex max-w-7xl flex-col gap-8 rounded-[32px] border border-white/5 bg-slate-950/60 px-6 py-10 shadow-[0_25px_80px_rgba(2,6,23,0.6)] backdrop-blur">
        <PlanSwitcher
          options={planSwitcherOptions}
          activeSymbol={activeOption?.symbol ?? plan.symbol}
          onSelect={handleSelectSymbolFromSwitcher}
          favorites={favoriteEntries}
          favoritesLoading={favoritesLoading}
          onToggleFavorite={handleToggleFavorite}
        />
        <TradeStateHeader
          symbol={activeOption?.symbolDisplay ?? activePlan.symbol}
          latestPrice={effectiveLatestPrice}
          generatedAt={activePlan.generated_display}
          tradeState={tradeState}
          strategy={selectedStrategy}
          summary={strategyPlan.summary}
        />
        <div className="flex flex-wrap items-center justify-between gap-4">
          <StrategySelector selected={selectedStrategy} onSelect={setSelectedStrategy} />
          <IntentSelector selected={selectedIntent} onSelect={setSelectedIntent} />
        </div>
        <ActionSummaryPanel summary={actionSummary} />
        <PriceGauge
          latestPrice={effectiveLatestPrice}
          liveMeta={livePriceMeta}
          buySetup={strategyPlan.buy_setup}
          sellSetup={strategyPlan.sell_setup}
          noTradeZones={strategyPlan.no_trade_zone ?? []}
        />
        <ScenarioCards plan={strategyPlan} />
        <PnLPreview plan={strategyPlan} latestPrice={effectiveLatestPrice} />
        <div className="grid gap-6 xl:grid-cols-2">
          <AlertsList
            alerts={alertSuggestions}
            symbol={activeSymbol}
            latestPrice={effectiveLatestPrice}
            initialActiveAlerts={activePrefs.alerts ?? []}
            onActiveAlertsChange={(ids) => handleAlertsChange(activeSymbol, ids)}
          />
          <TrendHeatmap consensus={activePlan.technical_consensus} />
        </div>
      </div>
    </div>
  );
}
