import { createRoot } from "react-dom/client";
import { AiInsightsApp } from "./components/AiInsightsApp";
import type { AiInsightsBootstrap } from "./types";

function readBootstrap(): AiInsightsBootstrap {
  const node = document.getElementById("aiInsightsBootstrap");
  if (!node) {
    return {};
  }

  try {
    const raw = node.textContent || node.innerHTML || "{}";
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object") {
      return parsed as AiInsightsBootstrap;
    }
  } catch (error) {
    console.warn("[AiInsights] Failed to parse bootstrap payload", error);
  }

  return {};
}

function main() {
  const container = document.getElementById("aiInsightsRoot");
  if (!container) {
    return;
  }

  const bootstrap = readBootstrap();
  const root = createRoot(container);
  root.render(<AiInsightsApp bootstrap={bootstrap} />);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", main, { once: true });
} else {
  main();
}
