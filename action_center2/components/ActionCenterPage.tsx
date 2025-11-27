import { useEffect, useMemo, useState } from "react";
import { PrincipalPlan, PrincipalPlanOption, StrategyKey, StrategyPlan } from "../types";
import { deriveTradeState, buildAlertSuggestions } from "../utils/planMath";
import { TradeStateHeader } from "./TradeStateHeader";
import { StrategySelector } from "./StrategySelector";
import { PriceGauge } from "./PriceGauge";
import { ScenarioCards } from "./ScenarioCards";
import { PnLPreview } from "./PnLPreview";
import { AlertsList } from "./AlertsList";
import { TrendHeatmap } from "./TrendHeatmap";
import { PlanSwitcher } from "./PlanSwitcher";

export type ActionCenterProps = {
  plan: PrincipalPlan;
  initialStrategy?: StrategyKey;
  planOptions?: PrincipalPlanOption[];
};

const DEFAULT_STRATEGY: StrategyKey = "day_trading";

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

  const [activeSymbol, setActiveSymbol] = useState<string>(plan.symbol);

  useEffect(() => {
    setActiveSymbol(plan.symbol);
  }, [plan.symbol]);

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

  const [selectedStrategy, setSelectedStrategy] = useState<StrategyKey>(initialStrategy ?? DEFAULT_STRATEGY);

  useEffect(() => {
    if (!activePlan.strategies[selectedStrategy]) {
      setSelectedStrategy(DEFAULT_STRATEGY);
    }
  }, [activePlan, selectedStrategy]);

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
        </div>
        <PriceGauge
          latestPrice={activePlan.latest_price}
          buySetup={strategyPlan.buy_setup}
          sellSetup={strategyPlan.sell_setup}
          noTradeZones={strategyPlan.no_trade_zone ?? []}
        />
        <ScenarioCards plan={strategyPlan} />
        <PnLPreview plan={strategyPlan} latestPrice={activePlan.latest_price} />
        <div className="grid gap-6 xl:grid-cols-2">
          <AlertsList alerts={alertSuggestions} />
          <TrendHeatmap consensus={activePlan.technical_consensus} />
        </div>
      </div>
    </div>
  );
}
