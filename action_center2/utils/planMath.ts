import {
  ActionSummary,
  NoTradeZone,
  StrategyPlan,
  TradeIntent,
  TradeState,
  TradeSetup,
} from "../types";

type PriceLike = number | null | undefined;

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
  if (!Number.isFinite(entry) || !Number.isFinite(stop)) {
    return false;
  }
  return price >= entry && price > stop;
}

function isSellActive(price: number, setup?: TradeSetup): boolean {
  if (!setup) return false;
  const entry = safeNumber(setup.entry);
  const stop = safeNumber(setup.stop);
  if (!Number.isFinite(entry) || !Number.isFinite(stop)) {
    return false;
  }
  return price <= entry && price < stop;
}

function near(value: number, target: PriceLike): boolean {
  const numericTarget = safeNumber(target);
  if (!Number.isFinite(value) || !Number.isFinite(numericTarget) || numericTarget === 0) {
    return false;
  }
  return Math.abs(value - numericTarget) / Math.abs(numericTarget) <= NEAR_TOLERANCE;
}

function safeNumber(value: PriceLike): number {
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

function computeConfidence(score: number | undefined): { label: string; score: number } {
  if (!Number.isFinite(score)) {
    return { label: "Medium", score: 50 };
  }

  const normalized = score <= 1 ? score * 100 : score;
  const clamped = clampToRange(normalized, 0, 100);
  if (clamped >= 70) {
    return { label: "High", score: clamped };
  }
  if (clamped >= 45) {
    return { label: "Medium", score: clamped };
  }
  return { label: "Low", score: clamped };
}

function formatDelta(current: number | undefined, reference: number | undefined): string | null {
  const currentVal = safeNumber(current);
  const refVal = safeNumber(reference);
  if (!Number.isFinite(currentVal) || !Number.isFinite(refVal) || refVal === 0) {
    return null;
  }
  const changePct = ((currentVal - refVal) / Math.abs(refVal)) * 100;
  const direction = changePct >= 0 ? "above" : "below";
  return `${formatPrice(currentVal)} (${formatPercent(changePct)} ${direction} entry)`;
}

export type ActionSummaryInput = {
  strategy: StrategyPlan;
  tradeState: TradeState;
  latestPrice: number;
  intent: TradeIntent;
};

export function deriveActionSummary({
  strategy,
  tradeState,
  latestPrice,
  intent,
}: ActionSummaryInput): ActionSummary {
  const setup = intent === "sell" ? strategy.sell_setup : strategy.buy_setup;
  const opposingSetup = intent === "sell" ? strategy.buy_setup : strategy.sell_setup;

  const entry = safeNumber(setup?.entry);
  const stop = safeNumber(setup?.stop);
  const targets = Array.isArray(setup?.targets) ? setup.targets.map((target) => safeNumber(target)).filter(Number.isFinite) : [];

  const riskLabel = strategy.rewardRisk && Number.isFinite(strategy.rewardRisk)
    ? `Reward/Risk ${strategy.rewardRisk.toFixed(2)}`
    : null;

  const deltaLabel = formatDelta(latestPrice, entry);

  const narrative: string[] = [];
  if (Number.isFinite(entry)) {
    narrative.push(`Entry ${formatPrice(entry)}`);
  }
  if (Number.isFinite(stop)) {
    narrative.push(`Stop ${formatPrice(stop)}`);
  }
  if (targets.length) {
    const formattedTargets = targets.map((value) => formatPrice(value)).join(" · ");
    narrative.push(`Targets ${formattedTargets}`);
  }
  if (riskLabel) {
    narrative.push(riskLabel);
  }
  if (deltaLabel) {
    narrative.push(deltaLabel);
  }

  const { label: confidenceLabel, score: confidenceScore } = computeConfidence(strategy.conviction);

  const isBuyIntent = intent === "buy";
  let title = isBuyIntent ? "Bullish setup standing by" : "Bearish setup standing by";
  let subtitle = isBuyIntent
    ? "Price is nearing the long trigger."
    : "Price is nearing the short trigger.";
  let status: ActionSummary["status"] = isBuyIntent ? "bullish" : "bearish";

  if (tradeState === "NO_TRADE") {
    title = "Inside no-trade zone";
    subtitle = "Respect the neutral range until price exits decisively.";
    status = "neutral";
  } else if (tradeState === "WAIT") {
    subtitle = isBuyIntent
      ? "Let price confirm above the entry level before leaning long."
      : "Let price break under the entry level before leaning short.";
  } else if (tradeState === "BUY_TRIGGERING" && isBuyIntent) {
    title = "Long trigger arming";
    subtitle = "Momentum is challenging resistance — prep the long plan.";
  } else if (tradeState === "SELL_TRIGGERING" && !isBuyIntent) {
    title = "Short trigger arming";
    subtitle = "Momentum is testing support — prep the short plan.";
  } else if (tradeState === "BUY_ACTIVE") {
    if (isBuyIntent) {
      title = "Long setup active";
      subtitle = "Price is engaged with the long trigger — manage risk around stops.";
    } else {
      title = "Long bias dominating";
      subtitle = "Opposing long momentum is in control; avoid aggressive shorts.";
      status = "neutral";
    }
  } else if (tradeState === "SELL_ACTIVE") {
    if (!isBuyIntent) {
      title = "Short setup active";
      subtitle = "Price is following through under resistance — manage short risk.";
    } else {
      title = "Short momentum dominant";
      subtitle = "Opposing short pressure in play; wait for strength before buying.";
      status = "neutral";
    }
  }

  if (opposingSetup && Number.isFinite(safeNumber(opposingSetup.entry))) {
    const opposingLabel = isBuyIntent ? "short" : "long";
    narrative.push(`Opposing ${opposingLabel} entry ${formatPrice(opposingSetup.entry)}`);
  }

  return {
    title,
    subtitle,
    narrative,
    status,
    confidenceLabel,
    confidenceScore,
  };
}
