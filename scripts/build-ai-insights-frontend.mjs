import { build } from "esbuild";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, "..");
const entryFile = path.resolve(projectRoot, "ai_insights_frontend/index.tsx");
const outputDir = path.resolve(projectRoot, "static/js");
const outputFile = path.resolve(outputDir, "ai-insights.js");

async function ensureOutputDir() {
  await fs.mkdir(outputDir, { recursive: true });
}

async function main() {
  await ensureOutputDir();

  try {
    const result = await build({
      entryPoints: [entryFile],
      bundle: true,
      format: "esm",
      platform: "browser",
      target: ["es2020"],
      outfile: outputFile,
      sourcemap: true,
      minify: false,
    });

    if (result.errors?.length) {
      console.error("[AiInsights] Build finished with errors:");
      result.errors.forEach((err) => console.error(err));
      process.exitCode = 1;
      return;
    }

    const relativeOutFile = path.relative(projectRoot, outputFile);
    console.log(`[AiInsights] Bundle written to ${relativeOutFile}`);
  } catch (error) {
    console.error("[AiInsights] Build failed:", error);
    process.exitCode = 1;
  }
}

main();
