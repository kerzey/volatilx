import { TradeIntent } from "../types";

export type IntentSelectorProps = {
  selected: TradeIntent;
  onSelect: (intent: TradeIntent) => void;
};

const OPTIONS: Array<{ value: TradeIntent; label: string }> = [
  { value: "buy", label: "Buy Intent" },
  { value: "sell", label: "Sell Intent" },
];

export function IntentSelector({ selected, onSelect }: IntentSelectorProps) {
  return (
    <div className="flex gap-2 rounded-full bg-slate-900/60 p-1 text-xs font-semibold text-slate-300">
      {OPTIONS.map((option) => {
        const isActive = option.value === selected;
        const baseClasses =
          "flex-1 rounded-full px-4 py-2 transition border border-transparent text-center cursor-pointer";
        const stateClasses = isActive
          ? " bg-sky-500/20 text-sky-100 border-sky-400/40"
          : " hover:bg-slate-800/80 hover:text-slate-100";
        return (
          <button
            key={option.value}
            type="button"
            onClick={() => onSelect(option.value)}
            className={baseClasses + stateClasses}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}
