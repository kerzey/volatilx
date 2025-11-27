import { StrategyKey, TradeState } from "../types";
import { formatPrice } from "../utils/planMath";

export type TradeStateHeaderProps = {
  symbol: string;
  latestPrice: number;
  generatedAt: string;
  tradeState: TradeState;
  strategy: StrategyKey;
  summary: string;
};

const STATE_META: Record<TradeState, { label: string; badge: string; message: string }> = {
  BUY_ACTIVE: {
    label: "Buy zone live",
    badge: "bg-emerald-500/15 text-emerald-200 border border-emerald-400/40",
    message: "In buy zone – manage risk versus nearby targets.",
  },
  BUY_TRIGGERING: {
    label: "Buy setup arming",
    badge: "bg-emerald-400/15 text-emerald-100 border border-emerald-300/40",
    message: "Buy setup arming – watch for a clean breakout.",
  },
  SELL_ACTIVE: {
    label: "Sell zone live",
    badge: "bg-rose-500/15 text-rose-200 border border-rose-400/40",
    message: "In sell zone – press shorts while levels hold.",
  },
  SELL_TRIGGERING: {
    label: "Sell setup arming",
    badge: "bg-rose-400/15 text-rose-100 border border-rose-300/40",
    message: "Sell setup arming – be ready for breakdown follow-through.",
  },
  NO_TRADE: {
    label: "Inside no-trade band",
    badge: "bg-amber-500/15 text-amber-100 border border-amber-400/40",
    message: "Inside no-trade band – best edge is patience.",
  },
  WAIT: {
    label: "Standing by",
    badge: "bg-slate-500/15 text-slate-200 border border-slate-400/40",
    message: "Between levels – wait for a decisive break.",
  },
};

const STRATEGY_LABELS: Record<StrategyKey, string> = {
  day_trading: "Day Trading",
  swing_trading: "Swing Trading",
  longterm_trading: "Long-Term Trading",
};

export function TradeStateHeader({
  symbol,
  latestPrice,
  generatedAt,
  tradeState,
  strategy,
  summary,
}: TradeStateHeaderProps) {
  const meta = STATE_META[tradeState];

  return (
    <section className="rounded-3xl border border-slate-800 bg-slate-900 p-8 shadow-sm">
      <div className="flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
        <div className="flex flex-col gap-3">
          <div className="flex flex-wrap items-center gap-3 text-sm text-slate-400">
            <span className="uppercase tracking-widest text-xs text-slate-500">What should I do?</span>
            <span className="rounded-full border border-slate-800 bg-slate-950 px-3 py-1 text-xs font-semibold text-slate-300">
              {STRATEGY_LABELS[strategy]}
            </span>
            <span className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-semibold ${meta.badge}`}>
              <span className="h-2 w-2 rounded-full bg-current" />
              {meta.label}
            </span>
          </div>
          <h1 className="text-3xl font-semibold tracking-tight text-slate-50">{symbol}</h1>
          <p className="max-w-3xl text-sm leading-relaxed text-slate-400">{summary}</p>
          <p className="text-xs uppercase tracking-wide text-slate-500">Generated {generatedAt}</p>
        </div>
        <div className="flex flex-col items-end gap-3 text-right">
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-500">Last Price</p>
            <p className="text-3xl font-semibold text-slate-50">{formatPrice(latestPrice)}</p>
          </div>
          <p className="max-w-xs text-sm text-slate-300">{meta.message}</p>
        </div>
      </div>
    </section>
  );
}
