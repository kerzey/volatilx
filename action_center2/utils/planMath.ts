import { NoTradeZone, StrategyPlan, TradeState, TradeSetup } from "../types";

const NEAR_TOLERANCE = 0.002; // 0.2%

export type TradeStateInput = {
  plan: StrategyPlan;
  latestPrice: number;
  tolerancePct?: number;
};

export function getTradeState(latestPrice: number, strategy: StrategyPlan): TradeState {
  if (!Number.isFinite(latestPrice) || !strategy) {
    return "WAIT";
  }

  const buy = strategy.buy_setup;
  const sell = strategy.sell_setup;
  const zones = Array.isArray(strategy.no_trade_zone) ? strategy.no_trade_zone : [];

  if (isWithinNoTrade(latestPrice, zones)) {
    return "NO_TRADE";
  }

  if (isBuyActive(latestPrice, buy)) {
    return "BUY_ACTIVE";
  }

  if (isSellActive(latestPrice, sell)) {
    return "SELL_ACTIVE";
  }

  if (near(latestPrice, buy?.entry)) {
    return "BUY_TRIGGERING";
  }

  if (near(latestPrice, sell?.entry)) {
    return "SELL_TRIGGERING";
  }

  return "WAIT";
}

export function deriveTradeState({
  plan,
  latestPrice,
}: TradeStateInput): TradeState {
  return getTradeState(latestPrice, plan);
}

function isWithinNoTrade(price: number, zones: NoTradeZone[]): boolean {
  return zones.some((zone) => {
    const lower = safeNumber(zone?.min);
    const upper = safeNumber(zone?.max);
    if (!isFinite(lower) || !isFinite(upper)) {
      return false;
    }
    const [min, max] = lower <= upper ? [lower, upper] : [upper, lower];
    return price >= min && price <= max;
  });
}

function isBuyActive(price: number, setup?: TradeSetup): boolean {
  if (!setup) return false;
  const entry = safeNumber(setup.entry);
  const stop = safeNumber(setup.stop);
  if (!isFinite(entry) || !isFinite(stop)) {
    return false;
  }
  return price >= entry && price > stop;
}

function isSellActive(price: number, setup?: TradeSetup): boolean {
  if (!setup) return false;
  const entry = safeNumber(setup.entry);
  const stop = safeNumber(setup.stop);
  if (!isFinite(entry) || !isFinite(stop)) {
    return false;
  }
  return price <= entry && price < stop;
}

function near(value: number, target: number | undefined): boolean {
  const numericTarget = safeNumber(target);
  if (!Number.isFinite(value) || !Number.isFinite(numericTarget) || numericTarget === 0) {
    return false;
  }
  return Math.abs(value - numericTarget) / Math.abs(numericTarget) <= NEAR_TOLERANCE;
}

function safeNumber(value: unknown): number {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : NaN;
}

export function formatPrice(value: number | undefined, digits = 2): string {
  if (!isFinite(Number(value))) {
    return "N/A";
  }
  return Number(value).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function formatPercent(change: number | undefined, digits = 2): string {
  if (!isFinite(Number(change))) {
    return "";
  }
  const prefix = change > 0 ? "+" : "";
  return `${prefix}${change.toFixed(digits)}%`;
}

export type AlertSuggestion = {
  id: string;
  label: string;
  description: string;
};

export function buildAlertSuggestions(latestPrice: number, strategy: StrategyPlan): AlertSuggestion[] {
  const suggestions: AlertSuggestion[] = [];
  const buy = strategy.buy_setup;
  const sell = strategy.sell_setup;
  const zones = Array.isArray(strategy.no_trade_zone) ? strategy.no_trade_zone : [];

  const add = (id: string, label: string, description: string) => {
    if (!suggestions.some((item) => item.id === id)) {
      suggestions.push({ id, label, description });
    }
  };

  if (isFinite(safeNumber(buy?.entry))) {
    add("long-breakout", "Breakout buy trigger", `Ping me when price clears ${formatPrice(buy.entry)} so I can lean long.`);
  }

  if (isFinite(safeNumber(buy?.stop))) {
    add("long-stop", "Long protective stop", `Warn me if ${formatPrice(buy.stop)} prints so I can flatten longs fast.`);
  }

  if (Array.isArray(buy?.targets) && buy.targets.length) {
    add("long-target", "Long take profit", `Alert on ${formatPrice(buy.targets[0])} to start scaling the long.`);
  }

  if (isFinite(safeNumber(sell?.entry))) {
    add("short-breakdown", "Breakdown sell trigger", `Notify me when price loses ${formatPrice(sell.entry)} to lean short.`);
  }

  if (isFinite(safeNumber(sell?.stop))) {
    add("short-stop", "Short protective stop", `Flag ${formatPrice(sell.stop)} so I can cover shorts before squeeze risk.`);
  }

  if (Array.isArray(sell?.targets) && sell.targets.length) {
    add("short-target", "Short take profit", `Alert on ${formatPrice(sell.targets[0])} to lock in short side gains.`);
  }

  const outsideNoTrade = zones.length
    ? !isWithinNoTrade(latestPrice, zones)
    : false;

  zones.forEach((zone, index) => {
    if (!outsideNoTrade) {
      return;
    }
    const range = `${formatPrice(zone.min)} – ${formatPrice(zone.max)}`;
    add(
      `neutral-${index}`,
      "No-trade re-entry",
      `Let me know if price trades back inside ${range} so I can stand down.`,
    );
  });

  return suggestions;
}

export function clampToRange(value: number, min = 0, max = 100): number {
  if (!Number.isFinite(value)) {
    return min;
  }
  return Math.min(Math.max(value, min), max);
}
