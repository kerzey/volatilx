import { PrincipalPlanOption } from "../types";

export type PlanSwitcherProps = {
  options: PrincipalPlanOption[];
  activeSymbol: string;
  onSelect: (symbol: string) => void;
};

export function PlanSwitcher({ options, activeSymbol, onSelect }: PlanSwitcherProps) {
  if (!options || options.length <= 1) {
    return null;
  }

  return (
    <div className="rounded-3xl border border-slate-800 bg-slate-900/80 px-4 py-3 shadow-sm">
      <div className="flex flex-wrap items-center gap-2 text-sm text-slate-300">
        <span className="text-xs uppercase tracking-widest text-slate-500">Tracked Symbols</span>
        {options.map((option) => {
          const isActive = option.symbol === activeSymbol;
          const baseClasses = "inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-semibold transition";
          const stateClasses = option.symbol === activeSymbol
            ? " border-sky-500/60 bg-sky-500/10 text-sky-100"
            : " border-slate-700 bg-slate-900 text-slate-400 hover:border-slate-600 hover:text-slate-200";

          return (
            <button
              key={option.symbol}
              type="button"
              onClick={() => onSelect(option.symbol)}
              title={`Generated ${option.plan.generated_display}`}
              className={baseClasses + stateClasses}
            >
              <span>{option.symbolDisplay}</span>
              {option.isFavorite ? (
                <span aria-hidden="true" className="text-amber-300" title="Favorited">
                  *
                </span>
              ) : null}
            </button>
          );
        })}
      </div>
    </div>
  );
}
