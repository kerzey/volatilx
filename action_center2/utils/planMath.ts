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

const TRIGGER_TOLERANCE = 0.003; // 0.3%
const STOP_PROXIMITY_TOLERANCE = 0.003;

type PreparedBand = { min: number; max: number };

type PriceZone =
  | "LONG_INVALIDATED"
  | "SHORT_INVALIDATED"
  | "BREAKDOWN"
  | "SHORT_LADDER"
  | "PRE_SHORT"
  | "NEUTRAL"
  | "PRE_LONG"
  | "LONG_TRIGGER"
  | "LONG_LADDER"
  | "BREAKOUT";

type ZoneClassification = {
  zone: PriceZone;
  longLadderIndex?: number;
  shortLadderIndex?: number;
  ladderBounds?: { lower: number; upper: number };
};

type ZoneInputs = {
  latestPrice: number;
  longEntry: PriceLike;
  longStop: PriceLike;
  longTargets: number[];
  shortEntry: PriceLike;
  shortStop: PriceLike;
  shortTargets: number[];
  neutralBands: PreparedBand[];
};

type SummaryContext = ZoneInputs & {
  classification: ZoneClassification;
  tradeState: TradeState;
};

type ZoneResponse = {
  title: string;
  subtitle: string;
  bullets: string[];
  status: ActionSummary["status"];
};

function normalizeTargets(values: PriceLike[], order: "asc" | "desc"): number[] {
  const filtered = (values || [])
    .map((value) => safeNumber(value))
    .filter((value): value is number => Number.isFinite(value));

  filtered.sort((a, b) => (order === "asc" ? a - b : b - a));
  return filtered;
}

function prepareNeutralBands(bands: NoTradeZone[] | undefined): PreparedBand[] {
  if (!Array.isArray(bands)) {
    return [];
  }

  return bands
    .map((band) => {
      const min = safeNumber(band?.min);
      const max = safeNumber(band?.max);
      if (!Number.isFinite(min) || !Number.isFinite(max)) {
        return null;
      }
      const lower = Math.min(min, max);
      const upper = Math.max(min, max);
      return { min: lower, max: upper };
    })
    .filter((band): band is PreparedBand => Boolean(band));
}

function classifyPriceZone({
  latestPrice,
  longEntry,
  longStop,
  longTargets,
  shortEntry,
  shortStop,
  shortTargets,
  neutralBands,
}: ZoneInputs): ZoneClassification {
  const price = safeNumber(latestPrice);
  if (!Number.isFinite(price)) {
    return { zone: "NEUTRAL" };
  }

  const longStopValue = safeNumber(longStop);
  if (Number.isFinite(longStopValue) && price <= longStopValue) {
    return { zone: "LONG_INVALIDATED" };
  }

  const shortStopValue = safeNumber(shortStop);
  if (Number.isFinite(shortStopValue) && price >= shortStopValue) {
    return { zone: "SHORT_INVALIDATED" };
  }

  const shortTargetsDesc = [...shortTargets];
  if (shortTargetsDesc.length) {
    const lowestShort = shortTargetsDesc[shortTargetsDesc.length - 1];
    if (Number.isFinite(lowestShort) && price < lowestShort) {
      return { zone: "BREAKDOWN" };
    }

    for (let idx = 0; idx < shortTargetsDesc.length - 1; idx += 1) {
      const upper = shortTargetsDesc[idx];
      const lower = shortTargetsDesc[idx + 1];
      if (!Number.isFinite(upper) || !Number.isFinite(lower)) {
        continue;
      }
      if (price < upper && price >= lower) {
        return {
          zone: "SHORT_LADDER",
          shortLadderIndex: idx,
          ladderBounds: { lower, upper },
        };
      }
    }

    const shortEntryValue = safeNumber(shortEntry);
    const highestShort = shortTargetsDesc[0];
    if (
      Number.isFinite(shortEntryValue)
      && Number.isFinite(highestShort)
      && price < shortEntryValue
      && price >= highestShort
    ) {
      return {
        zone: "PRE_SHORT",
        ladderBounds: { lower: highestShort, upper: shortEntryValue },
      };
    }

    if (Number.isFinite(shortEntryValue) && !Number.isFinite(highestShort) && price < shortEntryValue) {
      return { zone: "PRE_SHORT" };
    }
  } else {
    const shortEntryValue = safeNumber(shortEntry);
    if (Number.isFinite(shortEntryValue) && price < shortEntryValue) {
      return { zone: "PRE_SHORT" };
    }
  }

  for (const band of neutralBands) {
    if (price >= band.min && price <= band.max) {
      return { zone: "NEUTRAL" };
    }
  }

  const longEntryValue = safeNumber(longEntry);
  const neutralTop = neutralBands.length
    ? Math.max(...neutralBands.map((band) => band.max))
    : safeNumber(shortEntry);

  if (
    Number.isFinite(longEntryValue)
    && Number.isFinite(neutralTop)
    && price >= neutralTop
    && price < longEntryValue
  ) {
    return {
      zone: "PRE_LONG",
      ladderBounds: { lower: neutralTop, upper: longEntryValue },
    };
  }

  if (
    Number.isFinite(longEntryValue)
    && Math.abs(price - longEntryValue) / Math.abs(longEntryValue) <= TRIGGER_TOLERANCE
  ) {
    return { zone: "LONG_TRIGGER" };
  }

  const ladderLevels = [longEntryValue, ...longTargets].filter((value): value is number => Number.isFinite(value));
  ladderLevels.sort((a, b) => a - b);

  for (let idx = 0; idx < ladderLevels.length - 1; idx += 1) {
    const lower = ladderLevels[idx];
    const upper = ladderLevels[idx + 1];
    if (price >= lower && price < upper) {
      return {
        zone: "LONG_LADDER",
        longLadderIndex: idx,
        ladderBounds: { lower, upper },
      };
    }
  }

  const highestLong = ladderLevels[ladderLevels.length - 1];
  if (Number.isFinite(highestLong) && price > highestLong) {
    return { zone: "BREAKOUT" };
  }

  if (Number.isFinite(longEntryValue) && price >= longEntryValue) {
    return { zone: "LONG_LADDER" };
  }

  if (Number.isFinite(longEntryValue) && price < longEntryValue) {
    return { zone: "PRE_LONG" };
  }

  return { zone: "NEUTRAL" };
}

function formatRange(lower?: number, upper?: number): string | null {
  if (!Number.isFinite(lower) || !Number.isFinite(upper)) {
    return null;
  }
  return `${formatPrice(lower)} – ${formatPrice(upper)}`;
}

function resolveBuySummary(context: SummaryContext): ZoneResponse {
  const {
    classification,
    longEntry,
    longStop,
    longTargets,
    shortEntry,
    shortTargets,
    neutralBands,
  } = context;

  const lowestShort = shortTargets[shortTargets.length - 1];
  const highestShort = shortTargets[0];
  const firstLongTarget = longTargets[0];
  const secondLongTarget = longTargets[1];
  const finalLongTarget = longTargets[longTargets.length - 1];
  const primaryNeutral = neutralBands[0];
  const ladderBounds = classification.ladderBounds;

  const longEntryLabel = Number.isFinite(longEntry) ? formatPrice(longEntry) : null;
  const longStopLabel = Number.isFinite(longStop) ? formatPrice(longStop) : null;
  const shortEntryLabel = Number.isFinite(shortEntry) ? formatPrice(shortEntry) : null;
  const lowestShortLabel = Number.isFinite(lowestShort) ? formatPrice(lowestShort) : null;
  const highestShortLabel = Number.isFinite(highestShort) ? formatPrice(highestShort) : null;
  const targetOneLabel = Number.isFinite(firstLongTarget) ? formatPrice(firstLongTarget) : null;
  const targetTwoLabel = Number.isFinite(secondLongTarget) ? formatPrice(secondLongTarget) : null;
  const finalLongTargetLabel = Number.isFinite(finalLongTarget) ? formatPrice(finalLongTarget) : null;
  const neutralRangeLabel = primaryNeutral ? formatRange(primaryNeutral.min, primaryNeutral.max) : null;
  const ladderRangeLabel = ladderBounds ? formatRange(ladderBounds.lower, ladderBounds.upper) : null;

  switch (classification.zone) {
    case "LONG_INVALIDATED":
      return {
        title: "Long plan broken",
        subtitle: longStopLabel
          ? `Price has moved below the long stop (${longStopLabel}).`
          : "Price has moved below the long stop.",
        bullets: [
          "Close or avoid long positions.",
          "Allow volatility to cool and wait for a fresh setup from the new analysis.",
        ],
        status: "bearish",
      };
    case "SHORT_INVALIDATED":
      return {
        title: "Short setup failed",
        subtitle: shortEntryLabel
          ? `Price is squeezing above the bearish level (${shortEntryLabel}).`
          : "Price is squeezing above the bearish level.",
        bullets: [
          "Bias shifts more positive for longs.",
          "Expect acceleration if price holds above that level and prepare a revised long plan.",
        ],
        status: "bullish",
      };
    case "BREAKDOWN":
      return {
        title: "Stand aside",
        subtitle: "Strong downward move in progress.",
        bullets: [
          "Wait until price stops making new lows.",
          lowestShortLabel
            ? `Look for a clear bounce back above the lowest short target (${lowestShortLabel}) before considering longs.`
            : "Look for a clear bounce back above the lowest short target before considering longs.",
        ],
        status: "bearish",
      };
    case "SHORT_LADDER":
      return {
        title: "Let sellers finish their move",
        subtitle: "Price is still moving through downside targets.",
        bullets: [
          ladderRangeLabel
            ? `Watch how price reacts between ${ladderRangeLabel}.`
            : "Watch how price reacts near each short target.",
          "Only prepare long ideas once the drops start getting smaller and momentum slows.",
        ],
        status: "bearish",
      };
    case "PRE_SHORT":
      return {
        title: "Still below key resistance",
        subtitle: "Long setup needs price back above the bearish pivot.",
        bullets: [
          shortEntryLabel
            ? `Set an alert at the short trigger level (${shortEntryLabel}).`
            : "Set an alert at the short trigger level.",
          "Stay patient until price can reclaim resistance and hold.",
        ],
        status: "neutral",
      };
    case "NEUTRAL":
      return {
        title: "Inside neutral band",
        subtitle: "No clear edge for new longs yet.",
        bullets: [
          neutralRangeLabel
            ? `Avoid forcing trades while price is stuck in the range (${neutralRangeLabel}).`
            : "Avoid forcing trades while price is stuck in the range.",
          longEntryLabel
            ? `Wait for a clean break above the band and toward the long entry (${longEntryLabel}).`
            : "Wait for a clean break above the band and toward the long entry.",
        ],
        status: "neutral",
      };
    case "PRE_LONG":
      return {
        title: "Prepare the long setup",
        subtitle: "Price is moving toward your long entry.",
        bullets: [
          longStopLabel
            ? `Plan order size and risk using the long stop (${longStopLabel}).`
            : "Plan order size and risk using the long stop.",
          longEntryLabel
            ? `If momentum looks weak, let the candle close above ${longEntryLabel} before acting.`
            : "If momentum looks weak, let the candle close above the long entry before acting.",
        ],
        status: "bullish",
      };
    case "LONG_TRIGGER":
      return {
        title: "Starter long becomes valid",
        subtitle: "Long trigger level is now in play.",
        bullets: [
          longEntryLabel && longStopLabel
            ? `Open an initial long near ${longEntryLabel} with a stop at ${longStopLabel}.`
            : "Open an initial long near the entry with a stop at the long stop.",
          targetOneLabel && targetTwoLabel
            ? `Plan to take first profits at ${targetOneLabel} and reassess toward ${targetTwoLabel}.`
            : targetOneLabel
              ? `Plan to take first profits at ${targetOneLabel} and reassess toward the next target.`
              : "Plan to take first profits at Target 1 and reassess toward Target 2.",
        ],
        status: "bullish",
      };
    case "LONG_LADDER":
      return {
        title: "Manage the winning long",
        subtitle: "Price is moving through your profit targets.",
        bullets: [
          ladderRangeLabel
            ? `Take partial profits as each target is reached between ${ladderRangeLabel}.`
            : "Take partial profits as each target is reached.",
          "Trail your stop up toward the previous target to lock in gains.",
        ],
        status: "bullish",
      };
    case "BREAKOUT":
      return {
        title: "Upside extension",
        subtitle: "Price is running beyond planned targets.",
        bullets: [
          finalLongTargetLabel
            ? `Only add with very tight stops or wait for a pullback into the last target zone near ${finalLongTargetLabel}.`
            : "Only add with very tight stops or wait for a pullback into the last target zone.",
          "Consider re-running analysis to update upside levels.",
        ],
        status: "bullish",
      };
    default:
      return {
        title: "Long setup on watch",
        subtitle: "Follow the plan as levels evolve.",
        bullets: [
          highestShortLabel && longEntryLabel
            ? `Respect the path from ${highestShortLabel} to ${longEntryLabel} before leaning long.`
            : "Stay patient while the plan develops.",
        ],
        status: "neutral",
      };
  }
}

function resolveSellSummary(context: SummaryContext): ZoneResponse {
  const {
    classification,
    longEntry,
    longStop,
    longTargets,
    shortEntry,
    neutralBands,
  } = context;

  const firstLongTarget = longTargets[0];
  const finalLongTarget = longTargets[longTargets.length - 1];
  const primaryNeutral = neutralBands ? neutralBands[0] : undefined;
  const ladderBounds = classification.ladderBounds;

  const longEntryLabel = Number.isFinite(longEntry) ? formatPrice(longEntry) : null;
  const longStopLabel = Number.isFinite(longStop) ? formatPrice(longStop) : null;
  const shortEntryLabel = Number.isFinite(shortEntry) ? formatPrice(shortEntry) : null;
  const targetOneLabel = Number.isFinite(firstLongTarget) ? formatPrice(firstLongTarget) : null;
  const finalLongTargetLabel = Number.isFinite(finalLongTarget) ? formatPrice(finalLongTarget) : null;
  const neutralRangeLabel = primaryNeutral ? formatRange(primaryNeutral.min, primaryNeutral.max) : null;
  const ladderRangeLabel = ladderBounds ? formatRange(ladderBounds.lower, ladderBounds.upper) : null;

  switch (classification.zone) {
    case "LONG_INVALIDATED":
      return {
        title: "Exit the long idea",
        subtitle: longStopLabel
          ? `Price has broken the long stop level (${longStopLabel}).`
          : "Price has broken the long stop level.",
        bullets: [
          "Close remaining long exposure if not already flat.",
          "Review the trade and risk management before re-entering.",
        ],
        status: "bearish",
      };
    case "SHORT_INVALIDATED":
      return {
        title: "Relief rally underway",
        subtitle: shortEntryLabel
          ? `Short stop above the bearish level (${shortEntryLabel}) has triggered.`
          : "Short stop above the bearish level has triggered.",
        bullets: [
          "Use the bounce to exit any remaining longs on your terms.",
          "Run a new analysis if you are considering flipping to a fresh long setup.",
        ],
        status: "bullish",
      };
    case "BREAKDOWN":
      return {
        title: "Emergency exit zone",
        subtitle: "Long idea has failed; sellers are in control.",
        bullets: [
          "Close remaining long exposure as price breaks down.",
          "Reassess the chart once price stops making new lows.",
        ],
        status: "bearish",
      };
    case "SHORT_LADDER":
      return {
        title: "Sell into strength",
        subtitle: "Price is moving through short-side targets.",
        bullets: [
          ladderRangeLabel
            ? `Use intraday bounces to reduce or close long size between ${ladderRangeLabel}.`
            : "Use intraday bounces to reduce or close long size.",
          "Tighten a trailing stop above recent minor highs.",
        ],
        status: "bearish",
      };
    case "PRE_SHORT":
      return {
        title: "Risk is increasing",
        subtitle: "Price is trading under the bearish pivot.",
        bullets: [
          shortEntryLabel
            ? `Trim longs if price cannot reclaim the short trigger (${shortEntryLabel}).`
            : "Trim longs if price cannot reclaim the short trigger.",
          "Have a plan ready in case a full breakdown starts.",
        ],
        status: "bearish",
      };
    case "NEUTRAL":
      return {
        title: "Hold steady in the range",
        subtitle: "Price is back inside the neutral band.",
        bullets: [
          neutralRangeLabel
            ? `Avoid panic selling in the middle of the range (${neutralRangeLabel}).`
            : "Avoid panic selling in the middle of the range.",
          longEntryLabel
            ? `Plan staggered exits closer to resistance or your long entry area near ${longEntryLabel}.`
            : "Plan staggered exits closer to resistance or your long entry area.",
        ],
        status: "neutral",
      };
    case "PRE_LONG":
      return {
        title: "Strength rebuilding",
        subtitle: "Price is moving back toward the long trigger.",
        bullets: [
          longStopLabel
            ? `Tighten your stop below key support or the long stop area (${longStopLabel}).`
            : "Tighten your stop below key support or the long stop area.",
          "Consider taking partial profits if conviction is low.",
        ],
        status: "bullish",
      };
    case "LONG_TRIGGER":
      return {
        title: "Stay confidently long",
        subtitle: "Support is forming around the trigger level.",
        bullets: [
          longEntryLabel
            ? `Keep a core position while ${longEntryLabel} acts as support.`
            : "Keep a core position while entry acts as support.",
          targetOneLabel
            ? `Take partial profits at Target 1 (${targetOneLabel}) and follow the trade-state cue for further scaling.`
            : "Take partial profits at Target 1 and follow the trade-state cue for further scaling.",
        ],
        status: "bullish",
      };
    case "LONG_LADDER":
      return {
        title: "Scale out of the winner",
        subtitle: "Price is moving through upside targets.",
        bullets: [
          ladderRangeLabel
            ? `Sell portions of your position into strength between ${ladderRangeLabel}.`
            : "Sell portions of your position into strength at each target.",
          "Move stops higher to protect open profits.",
        ],
        status: "bullish",
      };
    case "BREAKOUT":
      return {
        title: "Final push higher",
        subtitle: "Use the strong spike to exit well.",
        bullets: [
          finalLongTargetLabel
            ? `Finish taking profits beyond the last target (${finalLongTargetLabel}).`
            : "Finish taking profits beyond the last target.",
          "Keep only a small runner with a very tight stop if you want extra upside optionality.",
        ],
        status: "neutral",
      };
    default:
      return {
        title: "Manage long exposure",
        subtitle: "Follow the exit plan level by level.",
        bullets: [
          longStopLabel && targetOneLabel
            ? `Stop ${longStopLabel} · Next trim ${targetOneLabel}.`
            : "Stay systematic with stops and trims.",
        ],
        status: "neutral",
      };
  }
}

function resolveBothSummary(context: SummaryContext): ZoneResponse {
  const {
    classification,
    longEntry,
    longStop,
    longTargets,
    shortEntry,
    shortTargets,
    neutralBands,
  } = context;

  const firstLongTarget = longTargets[0];
  const finalLongTarget = longTargets[longTargets.length - 1];
  const lowestShort = shortTargets[shortTargets.length - 1];
  const highestShort = shortTargets[0];
  const primaryNeutral = neutralBands ? neutralBands[0] : undefined;
  const ladderBounds = classification.ladderBounds;

  const longEntryLabel = Number.isFinite(longEntry) ? formatPrice(longEntry) : null;
  const longStopLabel = Number.isFinite(longStop) ? formatPrice(longStop) : null;
  const shortEntryLabel = Number.isFinite(shortEntry) ? formatPrice(shortEntry) : null;
  const lowestShortLabel = Number.isFinite(lowestShort) ? formatPrice(lowestShort) : null;
  const highestShortLabel = Number.isFinite(highestShort) ? formatPrice(highestShort) : null;
  const targetOneLabel = Number.isFinite(firstLongTarget) ? formatPrice(firstLongTarget) : null;
  const finalLongTargetLabel = Number.isFinite(finalLongTarget) ? formatPrice(finalLongTarget) : null;
  const neutralRangeLabel = primaryNeutral ? formatRange(primaryNeutral.min, primaryNeutral.max) : null;
  const ladderRangeLabel = ladderBounds ? formatRange(ladderBounds.lower, ladderBounds.upper) : null;

  switch (classification.zone) {
    case "LONG_INVALIDATED":
      return {
        title: "Bias turns short",
        subtitle: longStopLabel
          ? `Long stop has broken (${longStopLabel}).`
          : "Long stop has broken.",
        bullets: [
          "Look for fresh short entries on any retest of the broken support.",
          "Rebuild the plan once the volatility spike settles.",
        ],
        status: "bearish",
      };
    case "SHORT_INVALIDATED":
      return {
        title: "Bias turns long",
        subtitle: shortEntryLabel
          ? `Short idea failed above the bearish level (${shortEntryLabel}).`
          : "Short idea failed above the bearish level.",
        bullets: [
          "Shift focus to the long setup and upside targets.",
          "If entering new longs, use tighter stops due to recent volatility.",
        ],
        status: "bullish",
      };
    case "BREAKDOWN":
      return {
        title: "Short continuation zone",
        subtitle: "Momentum is breaking through supports.",
        bullets: [
          lowestShortLabel
            ? `Favor short trades toward the lowest short target (${lowestShortLabel}) with defined stops.`
            : "Favor short trades toward the lowest short target with defined stops.",
          "Mark this area as an extreme level to watch for a later bounce.",
        ],
        status: "bearish",
      };
    case "SHORT_LADDER":
      return {
        title: "Trade the short path",
        subtitle: "Price is stepping through downside targets.",
        bullets: [
          ladderRangeLabel
            ? `Take profits on shorts between ${ladderRangeLabel}.`
            : "Take profits on shorts between the ladder bounds.",
          "Follow the trade-state cue for when to reduce size or look for reversal signs.",
        ],
        status: "bearish",
      };
    case "PRE_SHORT":
      return {
        title: "Guard the short trigger",
        subtitle: "Price is testing the bearish pivot area.",
        bullets: [
          shortEntryLabel
            ? `Open or add to shorts only once price is clearly below the short trigger (${shortEntryLabel}).`
            : "Open or add to shorts only once price is clearly below the short trigger.",
          highestShortLabel
            ? `Aggressive traders may try a small position near the highest short target (${highestShortLabel}) with tight risk.`
            : "Aggressive traders may try a small position near the highest short target with tight risk.",
        ],
        status: "neutral",
      };
    case "NEUTRAL":
      return {
        title: "Low-conviction zone",
        subtitle: "Keep risk small or stay flat.",
        bullets: [
          neutralRangeLabel
            ? `Avoid heavy positioning while price is inside the neutral band (${neutralRangeLabel}).`
            : "Avoid heavy positioning while price is inside the neutral band.",
          longEntryLabel && shortEntryLabel
            ? `Use alerts at the top and bottom of the band to catch the next directional move (${longEntryLabel} / ${shortEntryLabel}).`
            : "Use alerts at the top and bottom of the band to catch the next directional move.",
        ],
        status: "neutral",
      };
    case "PRE_LONG":
      return {
        title: "Long trigger setting up",
        subtitle: "Price is approaching the long entry.",
        bullets: [
          longEntryLabel && longStopLabel
            ? `Prepare for a breakout long with stop at the long stop (${longStopLabel}).`
            : "Prepare for a breakout long with stop at the long stop.",
          "If you are still short, consider reducing risk as price nears the long trigger.",
        ],
        status: "bullish",
      };
    case "LONG_TRIGGER":
      return {
        title: "Run the long playbook",
        subtitle: "Long trigger level is firing.",
        bullets: [
          longEntryLabel && longStopLabel
            ? `Enter long with stop at the long stop (${longStopLabel}).`
            : "Enter long with stop at the long stop.",
          targetOneLabel
            ? `Scale at Target 1 (${targetOneLabel}) or tighten stops quickly if volatility is high.`
            : "Scale at Target 1 or tighten stops quickly if volatility is high.",
        ],
        status: "bullish",
      };
    case "LONG_LADDER":
      return {
        title: "Manage longs, watch for turn",
        subtitle: "Price is moving through upside targets.",
        bullets: [
          ladderRangeLabel
            ? `Trail long profits as targets are reached between ${ladderRangeLabel}.`
            : "Trail long profits as targets are reached.",
          "Watch for exhaustion near the upper targets if you plan to look for a new short later.",
        ],
        status: "bullish",
      };
    case "BREAKOUT":
      return {
        title: "Decide: ride or fade",
        subtitle: "Price is stretching above planned targets.",
        bullets: [
          "Either continue with the trend using tight risk.",
          finalLongTargetLabel
            ? `Or wait for a controlled pullback to look for a new short near the final long target (${finalLongTargetLabel}) and previous resistance.`
            : "Or wait for a controlled pullback to look for a new short near the final long target and previous resistance.",
        ],
        status: "neutral",
      };
    default:
      return {
        title: "Follow the dominant setup",
        subtitle: "Let price dictate the side to trade.",
        bullets: [
          longEntryLabel && shortEntryLabel
            ? `Key levels — Long ${longEntryLabel} | Short ${shortEntryLabel}.`
            : "Keep both plans in view and react to price.",
        ],
        status: "neutral",
      };
  }
}

function buildStopNotes({
  latestPrice,
  longStop,
  shortStop,
  intent,
}: {
  latestPrice: number;
  longStop: PriceLike;
  shortStop: PriceLike;
  intent: TradeIntent;
}): string[] {
  const notes = new Set<string>();
  const price = safeNumber(latestPrice);
  const longStopValue = safeNumber(longStop);
  const shortStopValue = safeNumber(shortStop);

  const nearLongStop = Number.isFinite(price)
    && Number.isFinite(longStopValue)
    && Math.abs(price - longStopValue) / Math.abs(longStopValue) <= STOP_PROXIMITY_TOLERANCE;
  const nearShortStop = Number.isFinite(price)
    && Number.isFinite(shortStopValue)
    && Math.abs(price - shortStopValue) / Math.abs(shortStopValue) <= STOP_PROXIMITY_TOLERANCE;

  if (nearLongStop) {
    notes.add("Price is very close to the stop level — expect fast moves.");
    notes.add("If this level breaks, the setup is no longer valid.");
  }

  if (nearShortStop) {
    notes.add("Price is very close to the stop level — expect fast moves.");
    if (intent === "both") {
      notes.add("If this level breaks, the setup is no longer valid.");
    }
    notes.add("The opposing idea is about to fail; a sharp squeeze is possible if this level breaks.");
  }

  return Array.from(notes);
}

function buildExtremeNotes({
  classification,
  latestPrice,
  longTargets,
  shortTargets,
}: {
  classification: ZoneClassification;
  latestPrice: number;
  longTargets: number[];
  shortTargets: number[];
}): string[] {
  const notes = new Set<string>();
  const price = safeNumber(latestPrice);

  const lowestShort = shortTargets[shortTargets.length - 1];
  const highestLong = longTargets[longTargets.length - 1];
  const extremeMessage = "Price has moved well beyond the planned range. Run a fresh analysis to update targets and risk.";
  const outdatedMessage = "This plan is now outdated. Wait for a new analysis before building a fresh position.";

  const isWithinBand = (first?: number, second?: number): boolean => {
    if (!Number.isFinite(price) || !Number.isFinite(first) || !Number.isFinite(second)) {
      return false;
    }
    const lower = Math.min(first as number, second as number);
    const upper = Math.max(first as number, second as number);
    return (price as number) >= lower && (price as number) <= upper;
  };

  const inLongScalingZone = isWithinBand(longTargets[0], longTargets[1]);
  const inShortScalingZone = isWithinBand(shortTargets[0], shortTargets[1]);

  if (
    Number.isFinite(price)
    && Number.isFinite(highestLong)
    && price >= (highestLong as number) * 1.01
  ) {
    notes.add(extremeMessage);
  }

  if (
    Number.isFinite(price)
    && Number.isFinite(lowestShort)
    && price <= (lowestShort as number) * 0.99
  ) {
    notes.add(extremeMessage);
  }

  if (
    classification.zone === "LONG_INVALIDATED"
    || classification.zone === "SHORT_INVALIDATED"
  ) {
    notes.add(outdatedMessage);
  }

  if (inLongScalingZone || inShortScalingZone) {
    notes.add(outdatedMessage);
  }

  return Array.from(notes);
}

function buildPlanSnapshot({
  intent,
  longEntry,
  longStop,
  longTargets,
  shortEntry,
  shortStop,
  shortTargets,
}: {
  intent: TradeIntent;
  longEntry: PriceLike;
  longStop: PriceLike;
  longTargets: number[];
  shortEntry: PriceLike;
  shortStop: PriceLike;
  shortTargets: number[];
}): string[] {
  const snapshots: string[] = [];

  const longTargetsLabel = longTargets.length
    ? longTargets.map((value) => formatPrice(value)).join(" / ")
    : null;
  const shortTargetsLabel = shortTargets.length
    ? shortTargets.map((value) => formatPrice(value)).join(" / ")
    : null;

  const longSnapshotParts = [
    `Long plan: Entry ${formatPrice(longEntry)}`,
    `Stop ${formatPrice(longStop)}`,
    longTargetsLabel ? `Targets ${longTargetsLabel}` : null,
  ].filter(Boolean) as string[];

  const shortSnapshotParts = [
    `Short plan: Entry ${formatPrice(shortEntry)}`,
    `Stop ${formatPrice(shortStop)}`,
    shortTargetsLabel ? `Targets ${shortTargetsLabel}` : null,
  ].filter(Boolean) as string[];

  const longSnapshot = longSnapshotParts.join(" • ");
  const shortSnapshot = shortSnapshotParts.join(" • ");

  const entries: Array<{ kind: "long" | "short"; text: string }> = [];

  if (longSnapshotParts.length) {
    entries.push({ kind: "long", text: longSnapshot });
  }
  if (shortSnapshotParts.length) {
    entries.push({ kind: "short", text: shortSnapshot });
  }

  const priority = (entry: { kind: "long" | "short" }): number => {
    if (intent === "buy") {
      return entry.kind === "long" ? 0 : 1;
    }
    if (intent === "sell") {
      return entry.kind === "short" ? 0 : 1;
    }
    return 0;
  };

  const ordered = entries.sort((a, b) => priority(a) - priority(b));

  return ordered.map((entry) => entry.text);
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
  const longEntry = safeNumber(strategy.buy_setup?.entry);
  const longStop = safeNumber(strategy.buy_setup?.stop);
  const longTargets = normalizeTargets(
    Array.isArray(strategy.buy_setup?.targets) ? strategy.buy_setup?.targets : [],
    "asc",
  );

  const shortEntry = safeNumber(strategy.sell_setup?.entry);
  const shortStop = safeNumber(strategy.sell_setup?.stop);
  const shortTargets = normalizeTargets(
    Array.isArray(strategy.sell_setup?.targets) ? strategy.sell_setup?.targets : [],
    "desc",
  );

  const neutralBands = prepareNeutralBands(strategy.no_trade_zone);

  const classification = classifyPriceZone({
    latestPrice,
    longEntry,
    longStop,
    longTargets,
    shortEntry,
    shortStop,
    shortTargets,
    neutralBands,
  });

  const context: SummaryContext = {
    classification,
    tradeState,
    latestPrice,
    longEntry,
    longStop,
    longTargets,
    shortEntry,
    shortStop,
    shortTargets,
    neutralBands,
  };

  const response: ZoneResponse = (() => {
    if (intent === "sell") {
      return resolveSellSummary(context);
    }
    if (intent === "both") {
      return resolveBothSummary(context);
    }
    return resolveBuySummary(context);
  })();

  const narrative: string[] = [
    ...response.bullets,
    ...buildStopNotes({ latestPrice, longStop, shortStop, intent }),
    ...buildExtremeNotes({ classification, latestPrice, longTargets, shortTargets }),
    ...buildPlanSnapshot({ intent, longEntry, longStop, longTargets, shortEntry, shortStop, shortTargets }),
  ].filter(Boolean);

  if (strategy.rewardRisk && Number.isFinite(strategy.rewardRisk)) {
    narrative.push(`Reward/Risk: ${strategy.rewardRisk.toFixed(2)} (potential gain vs planned loss).`);
  }

  const { label: confidenceLabel, score: confidenceScore } = computeConfidence(strategy.conviction);

  return {
    title: response.title,
    subtitle: response.subtitle,
    narrative,
    status: response.status,
    confidenceLabel,
    confidenceScore,
  };
}
