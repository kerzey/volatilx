import { NoTradeZone, TradeSetup } from "../types";
import { clampToRange, formatPrice } from "../utils/planMath";

export type PriceGaugeProps = {
  latestPrice: number;
  buySetup: TradeSetup;
  sellSetup: TradeSetup;
  noTradeZones: NoTradeZone[];
};

type Segment = {
  key: string;
  label: string;
  value: string;
  className: string;
};

export function PriceGauge({ latestPrice, buySetup, sellSetup, noTradeZones }: PriceGaugeProps) {
  const sellTargets = sellSetup?.targets ?? [];
  const buyTargets = buySetup?.targets ?? [];
  const lowest = Math.min(
    sellTargets[0] ?? sellSetup?.entry ?? latestPrice,
    ...noTradeZones.map((z) => z.min),
    buySetup?.entry ?? latestPrice,
    buyTargets[0] ?? latestPrice,
  );
  const highest = Math.max(
    sellTargets[sellTargets.length - 1] ?? sellSetup?.entry ?? latestPrice,
    ...noTradeZones.map((z) => z.max),
    buySetup?.entry ?? latestPrice,
    buyTargets[buyTargets.length - 1] ?? latestPrice,
  );

  const pointer = highest === lowest ? 50 : clampToRange(((latestPrice - lowest) / (highest - lowest)) * 100, 0, 100);

  const noTradeLabel = noTradeZones.length
    ? noTradeZones.map((zone) => `${formatPrice(zone.min)} – ${formatPrice(zone.max)}`).join(" | ")
    : "No neutral zone";

  const segments: Segment[] = [
    {
      key: "sellTarget",
      label: "Short Targets",
      value: formatPrice(sellTargets[sellTargets.length - 1] ?? sellSetup?.entry),
      className: "bg-rose-600/20 text-rose-200 border-r border-rose-500/20",
    },
    {
      key: "sellEntry",
      label: "Short Entry",
      value: formatPrice(sellSetup?.entry),
      className: "bg-rose-500/15 text-rose-100 border-r border-rose-400/20",
    },
    {
      key: "neutral",
      label: "No-Trade",
      value: noTradeLabel,
      className: "bg-amber-500/15 text-amber-100 border-r border-amber-400/20",
    },
    {
      key: "buyEntry",
      label: "Long Entry",
      value: formatPrice(buySetup?.entry),
      className: "bg-emerald-500/15 text-emerald-100 border-r border-emerald-400/20",
    },
    {
      key: "buyTarget",
      label: "Long Targets",
      value: formatPrice(buyTargets[buyTargets.length - 1] ?? buySetup?.entry),
      className: "bg-emerald-600/20 text-emerald-200",
    },
  ];

  return (
    <section className="rounded-3xl border border-slate-800 bg-slate-900 p-8 shadow-sm">
      <header className="mb-6 flex items-start justify-between">
        <div>
          <h2 className="text-lg font-semibold text-slate-50">Price Gauge</h2>
          <p className="text-sm text-slate-400">Visualise key regions before committing risk</p>
        </div>
        <div className="text-right">
          <p className="text-xs uppercase tracking-wide text-slate-500">Last</p>
          <p className="text-lg font-semibold text-slate-100">{formatPrice(latestPrice)}</p>
        </div>
      </header>
      <div className="relative">
        <div className="relative flex overflow-hidden rounded-2xl border border-slate-800/70 bg-slate-950/80 text-[11px] font-semibold uppercase tracking-wide text-slate-200">
          {segments.map((segment, index) => (
            <div
              key={segment.key}
              className={`flex flex-1 flex-col items-center justify-center gap-1 p-4 text-center ${segment.className} ${index === segments.length - 1 ? "border-r-0" : ""}`}
            >
              <span>{segment.label}</span>
              <span className="text-[10px] normal-case text-slate-300">{segment.value}</span>
            </div>
          ))}
        </div>
        <div
          className="pointer-events-none absolute -top-3 flex flex-col items-center text-xs text-slate-200"
          style={{ left: `calc(${pointer}% - 12px)` }}
        >
          <span className="h-3 w-[2px] rounded-full bg-indigo-400" />
          <span className="mt-1 rounded-full bg-indigo-500/20 px-3 py-1 text-[10px] font-semibold">
            {formatPrice(latestPrice)}
          </span>
        </div>
      </div>
    </section>
  );
}
