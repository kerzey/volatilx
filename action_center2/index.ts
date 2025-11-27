import React from "react";
import { createRoot, type Root } from "react-dom/client";

import { ActionCenterPage } from "./components/ActionCenterPage";
import type { ActionCenterProps } from "./components/ActionCenterPage";
import type { PrincipalPlan, PrincipalPlanOption, StrategyKey } from "./types";

export { ActionCenterPage };
export type { ActionCenterProps, PrincipalPlan };

const STRATEGY_KEY_BY_TIMEFRAME: Record<string, StrategyKey> = {
	day: "day_trading",
	swing: "swing_trading",
	long: "longterm_trading",
};

type BootstrapOptions = {
	plan?: PrincipalPlan | null;
	rootId?: string;
	initialStrategy?: StrategyKey;
};

declare global {
	interface Window {
		__ACTION_CENTER_PRINCIPAL_PLAN__?: PrincipalPlan | null;
		__ACTION_CENTER_PRINCIPAL_PLAN_OPTIONS__?: PrincipalPlanOption[] | null;
		bootstrapActionCenterPage?: (options?: BootstrapOptions) => Root | null;
	}
}

let root: Root | null = null;

function inferInitialStrategy(): StrategyKey | undefined {
	if (typeof document === "undefined") {
		return undefined;
	}

	const timeframeSlug = document.body?.dataset?.timeframe;
	if (!timeframeSlug) {
		return undefined;
	}

	return STRATEGY_KEY_BY_TIMEFRAME[timeframeSlug] ?? undefined;
}

export function bootstrapActionCenterPage(options: BootstrapOptions = {}): Root | null {
	if (typeof document === "undefined") {
		console.warn("[ActionCenter] Document is not available; skipping bootstrap.");
		return null;
	}

	const { plan, rootId = "action-center2-root", initialStrategy } = options;
	const mountNode = document.getElementById(rootId);

	if (!mountNode) {
		console.warn(`[ActionCenter] Mount node with id "${rootId}" is missing.`);
		return null;
	}

	const resolvedPlan = plan ?? window.__ACTION_CENTER_PRINCIPAL_PLAN__ ?? null;
	const availablePlans = (() => {
		const raw = window.__ACTION_CENTER_PRINCIPAL_PLAN_OPTIONS__;
		if (!Array.isArray(raw)) {
			return undefined;
		}
		return raw.filter((entry): entry is PrincipalPlanOption => Boolean(entry && entry.plan && entry.plan.symbol));
	})();
	if (!resolvedPlan) {
		console.warn("[ActionCenter] Principal plan payload missing; rendering aborted.");
		return null;
	}

	const strategy = initialStrategy ?? inferInitialStrategy();

	if (!root) {
		root = createRoot(mountNode);
	}

	root.render(
		React.createElement(ActionCenterPage, {
			plan: resolvedPlan,
			initialStrategy: strategy,
			planOptions: availablePlans,
		}),
	);
	return root;
}

function autoBootstrap(): void {
	if (typeof document === "undefined") {
		return;
	}
	const mountNode = document.getElementById("action-center2-root");
	if (!mountNode) {
		return;
	}
	bootstrapActionCenterPage();
}

if (typeof window !== "undefined") {
	window.bootstrapActionCenterPage = bootstrapActionCenterPage;

	if (document.readyState === "loading") {
		document.addEventListener("DOMContentLoaded", autoBootstrap, { once: true });
	} else {
		autoBootstrap();
	}
}
