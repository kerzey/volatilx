import { createRoot } from "react-dom/client";
import { SignInApp } from "./components/SignInApp";
import type { SignInBootstrap } from "./types";

function readBootstrapPayload(): SignInBootstrap {
  const node = document.getElementById("signInBootstrap");
  if (!node) {
    return {};
  }

  try {
    const raw = node.textContent || node.innerHTML || "{}";
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object") {
      return parsed as SignInBootstrap;
    }
  } catch (error) {
    console.warn("[SignIn] Unable to parse bootstrap payload", error);
  }

  return {};
}

function main() {
  const container = document.getElementById("signInRoot");
  if (!container) {
    return;
  }

  const bootstrap = readBootstrapPayload();
  const root = createRoot(container);
  root.render(<SignInApp bootstrap={bootstrap} />);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", main, { once: true });
} else {
  main();
}
