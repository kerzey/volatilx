import { buildClientBundle } from "./build-utils.mjs";

await buildClientBundle({
  entryPath: "action_center2/index.ts",
  entryName: "action-center2",
  manifestKey: "action-center2.js",
  logLabel: "ActionCenter2",
});
