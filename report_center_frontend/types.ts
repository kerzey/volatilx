import type { PrincipalPlan } from "../action_center2/types";

export type ReportCenterStrategySummary = {
  label?: string;
  summary?: string;
  next_actions?: string[];
  confidence?: string;
};

export type ReportCenterConsensusFocus = {
  timeframe?: string;
  recommendation?: string;
  confidence?: string;
  entry_price?: number;
  stop_loss?: number;
  take_profit?: number;
  risk_reward_ratio?: number;
};

export type ReportCenterConsensus = {
  status?: string;
  timestamp?: string;
  recommendation?: string;
  confidence?: string;
  strength?: number;
  buy_signals?: number;
  sell_signals?: number;
  hold_signals?: number;
  reasoning?: string[];
  focus?: ReportCenterConsensusFocus;
};

export type ReportCenterPrice = {
  timeframe?: string;
  close?: number | string;
  change_pct?: number | string;
  volume?: number | string;
  timestamp?: string;
};

export type ReportCenterKeyLevel = {
  price?: number | string;
  distance_pct?: number | string;
  label?: string;
};

export type ReportCenterPriceAction = {
  trend_alignment?: string;
  key_levels?: ReportCenterKeyLevel[];
  recent_patterns?: string[];
  immediate_bias?: string;
  candlestick_notes?: string[];
};

export type ReportCenterEntry = {
  symbol: string;
  symbol_display?: string;
  symbol_sanitized?: string | null;
  symbol_canonical?: string | null;
  generated_iso?: string | null;
  generated_display?: string | null;
  generated_unix?: number | null;
  strategies?: Record<string, ReportCenterStrategySummary>;
  consensus?: ReportCenterConsensus;
  price?: ReportCenterPrice;
  price_action?: ReportCenterPriceAction;
  stored_at?: string | null;
  source?: Record<string, unknown> | null;
  plan?: PrincipalPlan | null;
};

export type ReportCenterBootstrapMeta = {
  reportCount?: number;
  excludedReportCount?: number;
  maxReports?: number;
  selectedDate?: string;
  selectedDateLabel?: string;
  selectedSymbol?: string | null;
  availableSymbols?: string[];
};

export type ReportCenterBootstrap = {
  reports?: ReportCenterEntry[];
  favorites?: string[];
  meta?: ReportCenterBootstrapMeta;
};
