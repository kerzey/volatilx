import { useEffect, useMemo, useRef, useState } from "react";
import { PrincipalPlanOption } from "../types";

type FavoriteMenuEntry = {
  symbol: string;
  canonical: string;
  label: string;
  isTracked: boolean;
  isPending: boolean;
};

export type PlanSwitcherProps = {
  options?: PrincipalPlanOption[];
  activeSymbol: string;
  onSelect: (symbol: string) => void;
  favorites?: FavoriteMenuEntry[];
  favoritesLoading?: boolean;
  onRemoveFavorite?: (symbol: string) => void;
};

export function PlanSwitcher({
  options = [],
  activeSymbol,
  onSelect,
  favorites = [],
  favoritesLoading = false,
  onRemoveFavorite,
}: PlanSwitcherProps) {
  const hasTrackedOptions = options.length > 1;
  const hasFavoritesContext = favorites.length > 0 || favoritesLoading || Boolean(onRemoveFavorite);
  if (!hasTrackedOptions && !hasFavoritesContext) {
    return null;
  }

  return (
    <div className="rounded-3xl border border-slate-800 bg-slate-900/80 px-4 py-3 shadow-sm">
      <div className="flex flex-col gap-3 text-sm text-slate-300 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs uppercase tracking-widest text-slate-500">Tracked Symbols</span>
          {hasTrackedOptions ? (
            options.map((option) => {
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
            })
          ) : (
            <span className="rounded-full border border-slate-700 bg-slate-900 px-3 py-1 text-xs text-slate-500">
              Single symbol hydrated
            </span>
          )}
        </div>
        {hasFavoritesContext ? (
          <FavoritesDropdown
            favorites={favorites}
            loading={favoritesLoading}
            onSelect={onSelect}
            onRemoveFavorite={onRemoveFavorite}
          />
        ) : null}
      </div>
    </div>
  );
}

type FavoritesDropdownProps = {
  favorites: FavoriteMenuEntry[];
  loading: boolean;
  onSelect: (symbol: string) => void;
  onRemoveFavorite?: (symbol: string) => void;
};

function FavoritesDropdown({ favorites, loading, onSelect, onRemoveFavorite }: FavoritesDropdownProps) {
  const [open, setOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open || typeof document === "undefined") {
      return;
    }

    const handlePointer = (event: MouseEvent) => {
      if (!menuRef.current) {
        return;
      }
      if (!menuRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpen(false);
      }
    };

    document.addEventListener("mousedown", handlePointer);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handlePointer);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [open]);

  const list = useMemo(() => favorites.slice().sort((a, b) => a.label.localeCompare(b.label)), [favorites]);
  const buttonLabel = `*Favorites${list.length ? `(${list.length})` : ""}`;

  return (
    <div className="relative self-start sm:self-auto" ref={menuRef}>
      <button
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        className="inline-flex items-center rounded-full border border-amber-400/40 bg-transparent px-3 py-1 text-xs font-semibold text-amber-200 transition hover:text-amber-100"
        aria-expanded={open}
      >
        {buttonLabel}
      </button>
      {open ? (
        <div className="absolute right-0 z-20 mt-2 w-64 rounded-2xl border border-slate-800 bg-slate-950/95 p-2 shadow-2xl shadow-black/70">
          {loading ? (
            <div className="px-3 py-2 text-xs text-slate-400">Loading favorites…</div>
          ) : list.length ? (
            <ul className="space-y-2">
              {list.map((entry) => {
                const baseClasses = "flex w-full items-center gap-3 rounded-xl border px-3 py-2 text-left text-sm transition";
                const stateClasses = entry.isTracked
                  ? "border-slate-800/60 bg-slate-900/80 text-slate-100 hover:border-slate-700 hover:bg-slate-900"
                  : "cursor-not-allowed border-slate-900 bg-slate-950 text-slate-500";

                return (
                  <li key={entry.canonical} className="flex items-center gap-2">
                    <button
                      type="button"
                      className={baseClasses + " " + stateClasses}
                      onClick={() => {
                        if (!entry.isTracked) {
                          return;
                        }
                        onSelect(entry.symbol);
                        setOpen(false);
                      }}
                    >
                      <span className="font-semibold">{entry.label}</span>
                      {!entry.isTracked ? (
                        <span className="text-[10px] uppercase tracking-wide text-rose-300">Not loaded</span>
                      ) : null}
                    </button>
                    {onRemoveFavorite ? (
                      <button
                        type="button"
                        className="text-[11px] font-semibold uppercase tracking-wide text-rose-200 transition hover:text-rose-100 disabled:opacity-50"
                        onClick={(event) => {
                          event.stopPropagation();
                          onRemoveFavorite(entry.symbol);
                        }}
                        disabled={entry.isPending}
                      >
                        {entry.isPending ? "…" : "Unfavorite"}
                      </button>
                    ) : null}
                  </li>
                );
              })}
            </ul>
          ) : (
            <div className="px-3 py-2 text-xs text-slate-400">No favorites saved yet.</div>
          )}
        </div>
      ) : null}
    </div>
  );
}
