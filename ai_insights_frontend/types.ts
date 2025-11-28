export type SymbolCatalogEntry = {
  ticker: string;
  company: string;
};

export type SymbolCatalogMap = Record<string, SymbolCatalogEntry[]>;

export type AiInsightsBootstrap = {
  symbolCatalogs?: SymbolCatalogMap;
};

export type AnalyzeResponse = {
  success: boolean;
  error?: string;
  detail?: string;
  status?: number;
  runs_remaining?: number;
  renews_at?: string;
  action_url?: string;
  action_label?: string;
  code?: string;
  symbol_message?: string;
  market?: string;
  ai_job_id?: string;
  result?: Record<string, unknown>;
  price_action?: unknown;
  principal_plan?: unknown;
  ai_analysis?: unknown;
};
