import { buildClientBundle } from "./build-utils.mjs";

await buildClientBundle({
  entryPath: "subscription_frontend/index.tsx",
  entryName: "subscription",
  manifestKey: "subscription.js",
  logLabel: "Subscription",
});
