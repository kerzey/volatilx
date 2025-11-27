import { useMemo, useState } from "react";
import { PrincipalPlan, StrategyKey, StrategyPlan } from "../types";
import { deriveTradeState, buildAlertSuggestions } from "../utils/planMath";
import { TradeStateHeader } from "./TradeStateHeader";
import { StrategySelector } from "./StrategySelector";
import { PriceGauge } from "./PriceGauge";
import { ScenarioCards } from "./ScenarioCards";
import { PnLPreview } from "./PnLPreview";
import { AlertsList } from "./AlertsList";
import { TrendHeatmap } from "./TrendHeatmap";

export type ActionCenterProps = {
  plan: PrincipalPlan;
  initialStrategy?: StrategyKey;
};

const DEFAULT_STRATEGY: StrategyKey = "day_trading";

export function ActionCenterPage({ plan, initialStrategy }: ActionCenterProps) {
  const [selectedStrategy, setSelectedStrategy] = useState<StrategyKey>(initialStrategy ?? DEFAULT_STRATEGY);

  const strategyPlan: StrategyPlan = plan.strategies[selectedStrategy] ?? plan.strategies[DEFAULT_STRATEGY];

  const tradeState = useMemo(
    () =>
      deriveTradeState({
        plan: strategyPlan,
        latestPrice: plan.latest_price,
      }),
    [plan.latest_price, strategyPlan],
  );

  const alertSuggestions = useMemo(
    () => buildAlertSuggestions(plan.latest_price, strategyPlan),
    [plan.latest_price, strategyPlan],
  );

  return (
    <div className="min-h-screen bg-slate-950 py-10 text-slate-50">
      <div className="mx-auto flex max-w-7xl flex-col gap-8 px-6">
        <TradeStateHeader
          symbol={plan.symbol}
          latestPrice={plan.latest_price}
          generatedAt={plan.generated_display}
          tradeState={tradeState}
          strategy={selectedStrategy}
          summary={strategyPlan.summary}
        />
        <div className="flex flex-wrap items-center justify-between gap-4">
          <StrategySelector selected={selectedStrategy} onSelect={setSelectedStrategy} />
        </div>
        <PriceGauge
          latestPrice={plan.latest_price}
          buySetup={strategyPlan.buy_setup}
          sellSetup={strategyPlan.sell_setup}
          noTradeZones={strategyPlan.no_trade_zone ?? []}
        />
        <ScenarioCards plan={strategyPlan} />
        <PnLPreview plan={strategyPlan} latestPrice={plan.latest_price} />
        <div className="grid gap-6 xl:grid-cols-2">
          <AlertsList alerts={alertSuggestions} />
          <TrendHeatmap consensus={plan.technical_consensus} />
        </div>
      </div>
    </div>
  );
}
