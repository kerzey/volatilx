import { buildClientBundle } from "./build-utils.mjs";

await buildClientBundle({
  entryPath: "report_center_frontend/index.tsx",
  entryName: "report-center",
  manifestKey: "report-center.js",
  logLabel: "ReportCenter",
});
