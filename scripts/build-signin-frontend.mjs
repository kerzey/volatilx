import { buildClientBundle } from "./build-utils.mjs";

await buildClientBundle({
  entryPath: "signin_frontend/index.tsx",
  entryName: "signin",
  manifestKey: "signin.js",
  logLabel: "SignIn",
});
