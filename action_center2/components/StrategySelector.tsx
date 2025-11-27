import { StrategyKey } from "../types";

export type StrategySelectorProps = {
  selected: StrategyKey;
  onSelect: (strategy: StrategyKey) => void;
};

const STRATEGY_LABELS: Record<StrategyKey, string> = {
  day_trading: "Day Trading",
  swing_trading: "Swing Trading",
  longterm_trading: "Long-Term Trading",
};

export function StrategySelector({ selected, onSelect }: StrategySelectorProps) {
  return (
    <div className="inline-flex rounded-full border border-slate-800 bg-slate-900 p-1 text-sm font-semibold shadow-sm">
      {(Object.keys(STRATEGY_LABELS) as StrategyKey[]).map((key) => {
        const active = key === selected;
        return (
          <button
            key={key}
            type="button"
            onClick={() => onSelect(key)}
            className={[
              "rounded-full px-4 py-2 transition-all focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500",
              active
                ? "bg-indigo-500 text-white shadow"
                : "text-slate-400 hover:text-slate-200",
            ].join(" ")}
          >
            {STRATEGY_LABELS[key]}
          </button>
        );
      })}
    </div>
  );
}
