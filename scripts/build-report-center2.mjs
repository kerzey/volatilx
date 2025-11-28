import { build } from "esbuild";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, "..");
const entryFile = path.resolve(projectRoot, "report_center2/index.tsx");
const outputDir = path.resolve(projectRoot, "static/js");
const outputFile = path.resolve(outputDir, "report-center2.js");

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
      console.error("[ReportCenter2] Build finished with errors:");
      result.errors.forEach((err) => console.error(err));
      process.exitCode = 1;
      return;
    }

    const relativeOutFile = path.relative(projectRoot, outputFile);
    console.log(`[ReportCenter2] Bundle written to ${relativeOutFile}`);
  } catch (error) {
    console.error("[ReportCenter2] Build failed:", error);
    process.exitCode = 1;
  }
}

main();
