import { createRoot } from "react-dom/client";
import { SubscriptionApp } from "./components/SubscriptionApp";
import type { SubscriptionBootstrap } from "./types";

function readBootstrapPayload(): SubscriptionBootstrap {
  const node = document.getElementById("subscriptionBootstrap");
  if (!node) {
    return {};
  }

  try {
    const raw = node.textContent || node.innerHTML || "{}";
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object") {
      return parsed as SubscriptionBootstrap;
    }
  } catch (error) {
    console.warn("[Subscription] Unable to parse bootstrap payload", error);
  }

  return {};
}

function main() {
  const container = document.getElementById("subscriptionRoot");
  if (!container) {
    return;
  }

  const bootstrap = readBootstrapPayload();
  const root = createRoot(container);
  root.render(<SubscriptionApp bootstrap={bootstrap} />);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", main, { once: true });
} else {
  main();
}
