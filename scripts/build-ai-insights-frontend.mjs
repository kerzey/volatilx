import { buildClientBundle } from "./build-utils.mjs";

await buildClientBundle({
  entryPath: "ai_insights_frontend/index.tsx",
  entryName: "ai-insights",
  manifestKey: "ai-insights.js",
  logLabel: "AiInsights",
});
