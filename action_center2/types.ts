export type TradeSetup = {
  entry: number;
  stop: number;
  targets: number[];
};

export type NoTradeZone = {
  min: number;
  max: number;
};

export type StrategyPlan = {
  summary: string;
  buy_setup: TradeSetup;
  sell_setup: TradeSetup;
  no_trade_zone: NoTradeZone[];
  bias?: {
    low: number;
    high: number;
    invalid: number;
  };
  rewardRisk?: number;
  conviction?: number;
};

export type Strategies = {
  day_trading: StrategyPlan;
  swing_trading: StrategyPlan;
  longterm_trading: StrategyPlan;
};

export type TechnicalConsensus = {
  overall_recommendation: "BUY" | "SELL" | "HOLD";
  confidence: "LOW" | "MEDIUM" | "HIGH";
  strength: number;
};

export type PrincipalPlan = {
  symbol: string;
  symbol_display?: string;
  is_favorited?: boolean;
  generated_display: string;
  latest_price: number;
  strategies: Strategies;
  technical_consensus?: TechnicalConsensus;
};

export type StrategyKey = keyof Strategies;

export type TradeIntent = "buy" | "sell";

export type PrincipalPlanOption = {
  symbol: string;
  symbolDisplay: string;
  isFavorite: boolean;
  plan: PrincipalPlan;
};

export type TradeState =
  | "NO_TRADE"
  | "BUY_ACTIVE"
  | "SELL_ACTIVE"
  | "BUY_TRIGGERING"
  | "SELL_TRIGGERING"
  | "WAIT";

export type Scenario = {
  title: string;
  narrative: string;
  highlight: string;
  tone: "bullish" | "bearish" | "neutral";
};

export type ScenarioCollection = {
  bullish: Scenario;
  bearish: Scenario;
  neutrality: Scenario;
};

export type GaugePoint = {
  key: string;
  label: string;
  value: number;
  tone: "bearish" | "neutral" | "bullish";
};

export type GaugeBand = {
  from: number;
  to: number;
  tone: "bearish" | "neutral" | "bullish";
};

export type GaugeModel = {
  min: number;
  max: number;
  points: GaugePoint[];
  noTradeBands: GaugeBand[];
};

export type ActionSummary = {
  title: string;
  subtitle: string;
  narrative: string[];
  status: "bullish" | "bearish" | "neutral";
  confidenceLabel: string;
  confidenceScore: number;
};
