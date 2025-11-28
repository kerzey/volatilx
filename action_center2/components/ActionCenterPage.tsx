import { useEffect, useMemo, useRef, useState } from "react";
import { PrincipalPlan, PrincipalPlanOption, StrategyKey, StrategyPlan, TradeIntent } from "../types";
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
};

type PersistedSelections = {
  activeSymbol?: string;
  perSymbol?: Record<string, PersistedSymbolPrefs>;
};

const STORAGE_KEY = "volatilx:action-center:selections";

const isStrategyKey = (value: unknown): value is StrategyKey =>
  typeof value === "string" && STRATEGY_KEYS.includes(value as StrategyKey);

const isTradeIntent = (value: unknown): value is TradeIntent => value === "buy" || value === "sell" || value === "both";

function readPersistedSelections(): PersistedSelections {
  if (typeof window === "undefined") {
    return {};
  }

  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return {};
    }
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") {
      return {};
    }

    const activeSymbol = typeof parsed.activeSymbol === "string" ? parsed.activeSymbol : undefined;
    const perSymbolInput = parsed.perSymbol && typeof parsed.perSymbol === "object" ? parsed.perSymbol : undefined;
    const perSymbol: Record<string, PersistedSymbolPrefs> = {};

    if (perSymbolInput) {
      Object.entries(perSymbolInput).forEach(([symbol, prefs]) => {
        if (!prefs || typeof prefs !== "object") {
          return;
        }
        const strategyValue = (prefs as PersistedSymbolPrefs).strategy;
        const strategy = isStrategyKey(strategyValue) ? strategyValue : undefined;
        const intent = isTradeIntent((prefs as PersistedSymbolPrefs).intent) ? (prefs as PersistedSymbolPrefs).intent : undefined;
        if (strategy || intent) {
          perSymbol[symbol] = {
            ...(strategy ? { strategy: strategy as StrategyKey } : {}),
            ...(intent ? { intent } : {}),
          };
        }
      });
    }

    return {
      ...(activeSymbol ? { activeSymbol } : {}),
      ...(Object.keys(perSymbol).length ? { perSymbol } : {}),
    };
  } catch (error) {
    return {};
  }
}

function writePersistedSelections(selections: PersistedSelections): void {
  if (typeof window === "undefined") {
    return;
  }

  try {
    const payload: PersistedSelections = {};
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
        if (entry.strategy || entry.intent) {
          sanitized[symbol] = entry;
        }
      });
      if (Object.keys(sanitized).length) {
        payload.perSymbol = sanitized;
      }
    }

    if (!payload.activeSymbol && !payload.perSymbol) {
      window.localStorage.removeItem(STORAGE_KEY);
      return;
    }

    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  } catch (error) {
    // Swallow storage errors (e.g., private mode)
  }
}

function updatePersistedSelections(updater: (prev: PersistedSelections) => PersistedSelections): void {
  if (typeof window === "undefined") {
    return;
  }
  const previous = readPersistedSelections();
  const next = updater(previous);
  writePersistedSelections(next);
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
    const initialSelections = initialSelectionsRef.current ?? {};

    const initialActiveSymbol = initialSelections.activeSymbol || plan.symbol;

    const [activeSymbol, setActiveSymbol] = useState<string>(initialActiveSymbol);

  useEffect(() => {
    if (!normalizedOptions.some((option) => option.symbol === activeSymbol)) {
      setActiveSymbol(plan.symbol);
    }
  }, [activeSymbol, normalizedOptions, plan.symbol]);

  const activeOption = useMemo(
    () => normalizedOptions.find((option) => option.symbol === activeSymbol) ?? normalizedOptions[0],
    [activeSymbol, normalizedOptions],
  );

  const activePlan = activeOption?.plan ?? plan;

    const initialSymbolPrefs = (initialSelections.perSymbol ?? {})[initialActiveSymbol];

    const deriveInitialStrategy = (): StrategyKey => {
      const preferred = initialSymbolPrefs?.strategy;
      if (preferred && activePlan.strategies[preferred]) {
        return preferred;
      }
      if (initialStrategy && activePlan.strategies[initialStrategy]) {
        return initialStrategy;
      }
      if (activePlan.strategies[DEFAULT_STRATEGY]) {
        return DEFAULT_STRATEGY;
      }
      const availableKeys = Object.keys(activePlan.strategies ?? {}) as StrategyKey[];
      return availableKeys[0] ?? DEFAULT_STRATEGY;
    };

    const [selectedStrategy, setSelectedStrategy] = useState<StrategyKey>(deriveInitialStrategy);
    const [selectedIntent, setSelectedIntent] = useState<TradeIntent>(() => {
      const preferred = initialSymbolPrefs?.intent;
      return isTradeIntent(preferred) ? preferred : "buy";
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
    const previous = previousSymbolRef.current;
    if (activeSymbol === previous) {
      return;
    }
    previousSymbolRef.current = activeSymbol;

    const persisted = readPersistedSelections();
    const prefs = (persisted.perSymbol ?? {})[activeSymbol];
    const available = activePlan.strategies ?? {};

    let nextStrategy: StrategyKey | undefined;
    if (prefs?.strategy && available[prefs.strategy]) {
      nextStrategy = prefs.strategy;
    } else if (available[selectedStrategy]) {
      nextStrategy = selectedStrategy;
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

    const nextIntent = prefs?.intent && isTradeIntent(prefs.intent) ? prefs.intent : selectedIntent;
    if (nextIntent !== selectedIntent) {
      setSelectedIntent(nextIntent);
    }
  }, [activePlan, activeSymbol, initialStrategy, selectedIntent, selectedStrategy]);

  useEffect(() => {
    if (!activeSymbol) {
      return;
    }
    updatePersistedSelections((prev) => ({
      ...prev,
      activeSymbol,
    }));
  }, [activeSymbol]);

  useEffect(() => {
    if (!activeSymbol || !activePlan.strategies[selectedStrategy]) {
      return;
    }
    updatePersistedSelections((prev) => {
      const perSymbol = { ...(prev.perSymbol ?? {}) };
      perSymbol[activeSymbol] = {
        strategy: selectedStrategy,
        intent: selectedIntent,
      };
      return {
        ...prev,
        activeSymbol,
        perSymbol,
      };
    });
  }, [activePlan.strategies, activeSymbol, selectedIntent, selectedStrategy]);

  const strategyPlan: StrategyPlan = activePlan.strategies[selectedStrategy] ?? activePlan.strategies[DEFAULT_STRATEGY];

  const tradeState = useMemo(
    () =>
      deriveTradeState({
        plan: strategyPlan,
        latestPrice: activePlan.latest_price,
      }),
    [activePlan.latest_price, strategyPlan],
  );

  const alertSuggestions = useMemo(
    () => buildAlertSuggestions(activePlan.latest_price, strategyPlan),
    [activePlan.latest_price, strategyPlan],
  );

  const actionSummary = useMemo(
    () =>
      deriveActionSummary({
        strategy: strategyPlan,
        tradeState,
        latestPrice: activePlan.latest_price,
        intent: selectedIntent,
      }),
    [activePlan.latest_price, selectedIntent, strategyPlan, tradeState],
  );

  return (
    <div className="min-h-screen bg-slate-950 py-10 text-slate-50">
      <div className="mx-auto flex max-w-7xl flex-col gap-8 px-6">
        <PlanSwitcher
          options={normalizedOptions}
          activeSymbol={activeOption?.symbol ?? plan.symbol}
          onSelect={setActiveSymbol}
        />
        <TradeStateHeader
          symbol={activeOption?.symbolDisplay ?? activePlan.symbol}
          latestPrice={activePlan.latest_price}
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
          latestPrice={activePlan.latest_price}
          buySetup={strategyPlan.buy_setup}
          sellSetup={strategyPlan.sell_setup}
          noTradeZones={strategyPlan.no_trade_zone ?? []}
        />
        <ScenarioCards plan={strategyPlan} />
        <PnLPreview plan={strategyPlan} latestPrice={activePlan.latest_price} />
        <div className="grid gap-6 xl:grid-cols-2">
          <AlertsList
            alerts={alertSuggestions}
            symbol={activeOption?.symbol ?? activePlan.symbol}
            latestPrice={activePlan.latest_price}
          />
          <TrendHeatmap consensus={activePlan.technical_consensus} />
        </div>
      </div>
    </div>
  );
}
