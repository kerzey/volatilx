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

function describeTradeState(tradeState: TradeState, focus: "long" | "short" | "neutral"): string | null {
  switch (tradeState) {
    case "BUY_TRIGGERING":
      return focus === "long" ? "Momentum is arming the long trigger." : null;
    case "SELL_TRIGGERING":
      return focus === "short" ? "Momentum is arming the short trigger." : null;
    case "BUY_ACTIVE":
      return focus === "long" ? "Long setup is active — manage position sizing." : null;
    case "SELL_ACTIVE":
      return focus === "short" ? "Short setup is active — expect pressure." : null;
    case "NO_TRADE":
      return focus === "neutral" ? "Respect the neutral range until price exits decisively." : null;
    default:
      return null;
  }
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
    tradeState,
  } = context;

  const firstLongTarget = longTargets[0];
  const finalLongTarget = longTargets[longTargets.length - 1];
  const lowestShort = shortTargets[shortTargets.length - 1];
  const highestShort = shortTargets[0];
  const primaryNeutral = neutralBands[0];
  const ladderBounds = classification.ladderBounds;

  switch (classification.zone) {
    case "LONG_INVALIDATED":
      return {
        title: "Long thesis invalidated",
        subtitle: `Price slipped below ${formatPrice(longStop)}`,
        bullets: [
          "Abort long entries until a fresh analysis rebuilds the plan.",
          "Wait for volatility to settle before re-engaging.",
        ],
        status: "neutral",
      };
    case "SHORT_INVALIDATED":
      return {
        title: "Short setup failed",
        subtitle: `Momentum is squeezing above ${formatPrice(shortEntry)}`,
        bullets: [
          "Prepare to execute the long plan — sellers are capitulating.",
          `Expect acceleration if ${formatPrice(shortEntry)} now holds as support.`,
        ],
        status: "bullish",
      };
    case "BREAKDOWN":
      return {
        title: "Stand aside",
        subtitle: "Selling pressure dominates the tape.",
        bullets: [
          lowestShort
            ? `Wait for price to reclaim ${formatPrice(lowestShort)} before planning longs.`
            : "Stay patient until price stabilises above the last short target.",
          "Monitor momentum for a higher low before deploying capital.",
        ],
        status: "bearish",
      };
    case "SHORT_LADDER":
      return {
        title: "Let shorts exhaust",
        subtitle: "Price is still working through downside targets.",
        bullets: [
          ladderBounds
            ? `Watch reactions between ${formatPrice(ladderBounds.lower)} and ${formatPrice(ladderBounds.upper)}.`
            : "Track short targets for signs of exhaustion.",
          "Only prepare longs once momentum slows and a higher low forms.",
        ],
        status: "bearish",
      };
    case "PRE_SHORT":
      return {
        title: "Still below bearish pivot",
        subtitle: "Long setup needs price back above resistance.",
        bullets: [
          `Set an alert at ${formatPrice(shortEntry)} to know when selling pressure fades.`,
          highestShort ? `Watch ${formatPrice(highestShort)} for basing attempts.` : "Wait for the first higher low before committing capital.",
        ],
        status: "neutral",
      };
    case "NEUTRAL": {
      const rangeLabel = primaryNeutral ? formatRange(primaryNeutral.min, primaryNeutral.max) : null;
      return {
        title: "Inside neutral band",
        subtitle: "No directional edge yet.",
        bullets: [
          rangeLabel ? `Sit out until price exits ${rangeLabel}.` : "Stand aside until price leaves the neutral range.",
          `Long bias requires a breakout above ${formatPrice(longEntry)}.`,
        ],
        status: "neutral",
      };
    }
    case "PRE_LONG": {
      const tradeCue = describeTradeState(tradeState, "long");
      return {
        title: "Prep the long trigger",
        subtitle: `Price is approaching ${formatPrice(longEntry)}.`,
        bullets: [
          `Queue entry orders near ${formatPrice(longEntry)} with stop at ${formatPrice(longStop)}.`,
          tradeCue || "Let the breakout confirm before deploying full size.",
        ],
        status: "bullish",
      };
    }
    case "LONG_TRIGGER": {
      const tradeCue = describeTradeState(tradeState, "long");
      return {
        title: "Execute starter long",
        subtitle: "Trigger level is engaging.",
        bullets: [
          `Enter near ${formatPrice(longEntry)} and honour ${formatPrice(longStop)} as the fail level.`,
          firstLongTarget
            ? `Map first scale-out at ${formatPrice(firstLongTarget)}.`
            : "Use trail stops to protect gains while price confirms.",
          tradeCue || "Manage size responsibly as volatility lifts.",
        ],
        status: "bullish",
      };
    }
    case "LONG_LADDER": {
      const tradeCue = describeTradeState(tradeState, "long");
      return {
        title: "Manage the long ladder",
        subtitle: "Price is advancing through profit targets.",
        bullets: [
          ladderBounds
            ? `Scale partial profits between ${formatPrice(ladderBounds.lower)} and ${formatPrice(ladderBounds.upper)}.`
            : `Respect the profit ladder above ${formatPrice(longEntry)}.`,
          `Trail stops to the prior rung to protect gains.`,
          tradeCue || "Stay alert for momentum shifts near targets.",
        ],
        status: "bullish",
      };
    }
    case "BREAKOUT":
      return {
        title: "Momentum breakout",
        subtitle: "Price is extending beyond planned upside targets.",
        bullets: [
          finalLongTarget ? `Use ${formatPrice(finalLongTarget)} as the new risk pivot.` : "Define a fresh risk pivot for continuation trades.",
          "Either add with tight stops or wait for a pullback before adding size.",
        ],
        status: "bullish",
      };
    default:
      return {
        title: "Long setup on watch",
        subtitle: "Follow the plan as levels evolve.",
        bullets: [
          `Entry ${formatPrice(longEntry)} · Stop ${formatPrice(longStop)} · Target ${formatPrice(firstLongTarget)}`,
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
    shortTargets,
    tradeState,
  } = context;

  const firstLongTarget = longTargets[0];
  const finalLongTarget = longTargets[longTargets.length - 1];
  const lowestShort = shortTargets[shortTargets.length - 1];
  const ladderBounds = classification.ladderBounds;

  switch (classification.zone) {
    case "LONG_INVALIDATED":
      return {
        title: "Exit the long",
        subtitle: `Stop level ${formatPrice(longStop)} just broke.`,
        bullets: [
          "Flatten remaining shares immediately to protect capital.",
          "Review the trade and wait for a new plan before re-entering.",
        ],
        status: "bearish",
      };
    case "SHORT_INVALIDATED":
      return {
        title: "Relief rally in progress",
        subtitle: `Short stop near ${formatPrice(shortEntry)} triggered.`,
        bullets: [
          "Use the squeeze to scale out remaining size into strength.",
          `Consider re-running analysis if you plan to flip long above ${formatPrice(shortEntry)}.`,
        ],
        status: "bullish",
      };
    case "BREAKDOWN":
      return {
        title: "Emergency exit",
        subtitle: "Long thesis failed — sellers are in control.",
        bullets: [
          lowestShort
            ? `Close remaining exposure before ${formatPrice(lowestShort)} fails.`
            : "Prioritise capital preservation over targets.",
          "Re-assess once price stabilises.",
        ],
        status: "bearish",
      };
    case "SHORT_LADDER":
      return {
        title: "Sell into bounces",
        subtitle: "Price is marching through short targets.",
        bullets: [
          ladderBounds
            ? `Trim into pops toward ${formatPrice(ladderBounds.upper)}.`
            : "Fade strength while the short ladder is in control.",
          "Tighten trailing stops above recent micro highs.",
        ],
        status: "bearish",
      };
    case "PRE_SHORT":
      return {
        title: "Risk is rising",
        subtitle: "Below the bearish pivot.",
        bullets: [
          `If ${formatPrice(shortEntry)} breaks, trim more size immediately.`,
          "Prepare contingency orders for a full breakdown.",
        ],
        status: "bearish",
      };
    case "NEUTRAL":
      return {
        title: "Hold steady",
        subtitle: "Price is back inside the neutral range.",
        bullets: [
          "Avoid panic selling mid-range; let levels dictate the next move.",
          `Stage exits near ${formatPrice(longEntry)} or the upper band.`,
        ],
        status: "neutral",
      };
    case "PRE_LONG":
      return {
        title: "Strength rebuilding",
        subtitle: `Price approaching ${formatPrice(longEntry)} trigger.`,
        bullets: [
          `Keep remaining shares with stop tightened under ${formatPrice(longStop)}.`,
          "Trim weak hands into strength if conviction is low.",
        ],
        status: "bullish",
      };
    case "LONG_TRIGGER":
      return {
        title: "Stay long with conviction",
        subtitle: "Trigger level is engaging.",
        bullets: [
          `Hold core while ${formatPrice(longEntry)} acts as support.`,
          firstLongTarget
            ? `Book partial profits into ${formatPrice(firstLongTarget)}.`
            : "Use a trailing stop to lock in progress.",
          describeTradeState(tradeState, "long") || "Let the plan dictate scale-out points.",
        ],
        status: "bullish",
      };
    case "LONG_LADDER":
      return {
        title: "Scale profits",
        subtitle: "Price working through upside targets.",
        bullets: [
          ladderBounds
            ? `Sell into strength between ${formatPrice(ladderBounds.lower)} and ${formatPrice(ladderBounds.upper)}.`
            : `Use the target ladder to guide remaining exits above ${formatPrice(longEntry)}.`,
          "Trail stops tighter with each rung cleared.",
        ],
        status: "bullish",
      };
    case "BREAKOUT":
      return {
        title: "Parabolic push",
        subtitle: "Use the final spike to exit gracefully.",
        bullets: [
          finalLongTarget
            ? `Finish scaling out into strength beyond ${formatPrice(finalLongTarget)}.`
            : "Close remaining size as momentum extends.",
          "Keep only a token runner with a very tight stop.",
        ],
        status: "neutral",
      };
    default:
      return {
        title: "Manage long exposure",
        subtitle: "Follow the exit plan level by level.",
        bullets: [
          `Stop ${formatPrice(longStop)} · Next trim ${formatPrice(firstLongTarget)}.`,
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
    tradeState,
  } = context;

  const firstLongTarget = longTargets[0];
  const finalLongTarget = longTargets[longTargets.length - 1];
  const lowestShort = shortTargets[shortTargets.length - 1];
  const highestShort = shortTargets[0];
  const ladderBounds = classification.ladderBounds;

  switch (classification.zone) {
    case "LONG_INVALIDATED":
      return {
        title: "Flip short bias",
        subtitle: `Long stop at ${formatPrice(longStop)} just failed.`,
        bullets: [
          "Look for short entries on weak retests of the broken level.",
          "Rebuild the plan once volatility cools.",
        ],
        status: "bearish",
      };
    case "SHORT_INVALIDATED":
      return {
        title: "Favour longs",
        subtitle: `Short invalidation above ${formatPrice(shortEntry)} triggered.`,
        bullets: [
          "Shift focus to the long ladder while the squeeze runs.",
          "Set tight stops in case the move reverses.",
        ],
        status: "bullish",
      };
    case "BREAKDOWN":
      return {
        title: "Short continuation",
        subtitle: "Momentum is breaking supports.",
        bullets: [
          lowestShort
            ? `Press shorts toward ${formatPrice(lowestShort)} with stop above the prior rung.`
            : "Lean short but trail stops aggressively.",
          "Mark the extreme for potential bounce opportunities.",
        ],
        status: "bearish",
      };
    case "SHORT_LADDER":
      return {
        title: "Trade the short ladder",
        subtitle: "Price stepping through downside targets.",
        bullets: [
          ladderBounds
            ? `Take profits between ${formatPrice(ladderBounds.lower)} and ${formatPrice(ladderBounds.upper)}.`
            : "Respect each short target for profit taking.",
          describeTradeState(tradeState, "short") || "Watch for reversal triggers as targets complete.",
        ],
        status: "bearish",
      };
    case "PRE_SHORT":
      return {
        title: "Guard the short trigger",
        subtitle: "Price is testing the bearish pivot.",
        bullets: [
          `Initiate shorts on a breakdown below ${formatPrice(shortEntry)}.`,
          highestShort ? `Fade into ${formatPrice(highestShort)} if momentum rolls over.` : "Keep long scalps small until buyers prove control.",
        ],
        status: "neutral",
      };
    case "NEUTRAL":
      return {
        title: "Low conviction zone",
        subtitle: "Keep position sizes light.",
        bullets: [
          "Stay flat or scalp the edges of the neutral band.",
          `Arm alerts above ${formatPrice(longEntry)} and below ${formatPrice(shortEntry)}.`,
        ],
        status: "neutral",
      };
    case "PRE_LONG":
      return {
        title: "Long trigger loading",
        subtitle: `Price approaching ${formatPrice(longEntry)}.`,
        bullets: [
          `Stalk the breakout for long entries with stop at ${formatPrice(longStop)}.`,
          describeTradeState(tradeState, "long") || "Keep short risk tight while buyers test the level.",
        ],
        status: "bullish",
      };
    case "LONG_TRIGGER":
      return {
        title: "Execute the long plan",
        subtitle: "Trigger level is firing.",
        bullets: [
          `Go long near ${formatPrice(longEntry)} with stop ${formatPrice(longStop)}.`,
          firstLongTarget
            ? `Scale out at ${formatPrice(firstLongTarget)} while monitoring for reversal to re-enter short.`
            : "Trail stops quickly; reassess if momentum stalls.",
        ],
        status: "bullish",
      };
    case "LONG_LADDER":
      return {
        title: "Manage both sides",
        subtitle: "Price is moving through upside targets.",
        bullets: [
          ladderBounds
            ? `Trail longs and look for exhaustion near ${formatPrice(ladderBounds.upper)} for potential fades.`
            : `Use the ladder above ${formatPrice(longEntry)} for both profit taking and fade setups.`,
          describeTradeState(tradeState, "long") || "Stay nimble — trend traders hold, mean-reverters wait for signals.",
        ],
        status: "bullish",
      };
    case "BREAKOUT":
      return {
        title: "Choose continuation or fade",
        subtitle: "Price is stretching above planned targets.",
        bullets: [
          finalLongTarget
            ? `Continuation: ride longs while ${formatPrice(finalLongTarget)} holds.`
            : "Continuation: only chase with tight risk.",
          lowestShort
            ? `Mean reversion: scout fades back toward ${formatPrice(lowestShort)} once momentum cracks.`
            : "Mean reversion: wait for a failed high before leaning short.",
        ],
        status: "neutral",
      };
    default:
      return {
        title: "Follow the dominant setup",
        subtitle: "Let price dictate the side to trade.",
        bullets: [
          `Key levels — Long ${formatPrice(longEntry)} | Short ${formatPrice(shortEntry)}.`,
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
  const notes: string[] = [];
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
    if (intent === "buy" || intent === "sell" || intent === "both") {
      notes.push("Long stop is under pressure — tighten risk or exit immediately.");
    }
  }

  if (nearShortStop && Number.isFinite(shortStopValue)) {
    if (intent === "buy") {
      notes.push("Short crowd is close to invalidation — expect a squeeze higher.");
    } else if (intent === "sell") {
      notes.push("Short stop is in play — use the bounce to finish scaling out.");
    } else {
      notes.push("Short stop nearly triggered — be ready to flip long on confirmation.");
    }
  }

  return notes;
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
  const notes: string[] = [];
  const price = safeNumber(latestPrice);

  const lowestShort = shortTargets[shortTargets.length - 1];
  if (
    classification.zone === "BREAKDOWN"
    && Number.isFinite(price)
    && Number.isFinite(lowestShort)
    && price <= (lowestShort as number) * 0.99
  ) {
    notes.push("Price broke beyond the last downside target — run a fresh analysis for new support.");
  }

  const highestLong = longTargets[longTargets.length - 1];
  if (
    classification.zone === "BREAKOUT"
    && Number.isFinite(price)
    && Number.isFinite(highestLong)
    && price >= (highestLong as number) * 1.01
  ) {
    notes.push("Price is extending more than 1% beyond the final upside target — refresh the plan.");
  }

  if (classification.zone === "LONG_INVALIDATED") {
    notes.push("Plan invalidated — schedule another analysis once conditions reset.");
  }
  if (classification.zone === "SHORT_INVALIDATED") {
    notes.push("Short playbook failed — rerun analysis for updated upside levels.");
  }

  return notes;
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

  const longSummary = [
    `Entry ${formatPrice(longEntry)}`,
    `Stop ${formatPrice(longStop)}`,
    longTargets.length ? `Targets ${longTargets.map((value) => formatPrice(value)).join(" · ")}` : null,
  ]
    .filter(Boolean)
    .join(" | ");

  const shortSummary = [
    `Entry ${formatPrice(shortEntry)}`,
    `Stop ${formatPrice(shortStop)}`,
    shortTargets.length ? `Targets ${shortTargets.map((value) => formatPrice(value)).join(" · ")}` : null,
  ]
    .filter(Boolean)
    .join(" | ");

  if (intent === "buy") {
    if (longSummary) {
      snapshots.push(`Long plan — ${longSummary}`);
    }
  } else if (intent === "sell") {
    if (longSummary) {
      snapshots.push(`Exit map — ${longSummary}`);
    }
  } else {
    if (longSummary) {
      snapshots.push(`Long plan — ${longSummary}`);
    }
    if (shortSummary) {
      snapshots.push(`Short plan — ${shortSummary}`);
    }
  }

  return snapshots;
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
    narrative.push(`Reward/Risk ${strategy.rewardRisk.toFixed(2)}`);
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
