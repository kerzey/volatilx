import { createRoot } from "react-dom/client";
import { ReportCenterApp } from "./components/ReportCenterApp";
import type { ReportCenterBootstrap } from "./types";

function readBootstrapPayload(): ReportCenterBootstrap {
  const node = document.getElementById("reportCenterBootstrap");
  if (!node) {
    return {};
  }

  try {
    const raw = node.textContent || node.innerHTML || "{}";
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object") {
      return parsed as ReportCenterBootstrap;
    }
  } catch (error) {
    console.warn("[ReportCenter] Unable to parse bootstrap payload", error);
  }

  return {};
}

function main() {
  const container = document.getElementById("reportCenterRoot");
  if (!container) {
    return;
  }

  const bootstrap = readBootstrapPayload();
  const reports = Array.isArray(bootstrap.reports) ? bootstrap.reports : [];
  const favorites = Array.isArray(bootstrap.favorites) ? bootstrap.favorites : [];
  const meta = bootstrap.meta ?? {};

  const root = createRoot(container);
  root.render(<ReportCenterApp reports={reports} favorites={favorites} meta={meta} />);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", main, { once: true });
} else {
  main();
}
