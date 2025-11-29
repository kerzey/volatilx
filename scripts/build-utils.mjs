import { build } from "esbuild";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, "..");
const outputDir = path.resolve(projectRoot, "static/js");
const manifestPath = path.resolve(outputDir, "manifest.json");

const POSIX_SEP = "/";

function toPosixPath(filePath) {
  return filePath.split(path.sep).join(POSIX_SEP);
}

async function ensureOutputDir() {
  await fs.mkdir(outputDir, { recursive: true });
}

async function readManifest() {
  try {
    const raw = await fs.readFile(manifestPath, "utf-8");
    const data = JSON.parse(raw);
    return typeof data === "object" && data !== null ? data : {};
  } catch (error) {
    if (error.code === "ENOENT") {
      return {};
    }
    throw error;
  }
}

async function writeManifest(manifest) {
  const payload = `${JSON.stringify(manifest, null, 2)}\n`;
  await fs.writeFile(manifestPath, payload, "utf-8");
}

async function updateManifest(key, value) {
  const manifest = await readManifest();
  manifest[key] = value;
  await writeManifest(manifest);
}

export async function buildClientBundle({ entryPath, entryName, manifestKey, logLabel }) {
  if (!entryPath || !entryName || !manifestKey) {
    throw new Error("buildClientBundle requires entryPath, entryName, and manifestKey");
  }

  const label = logLabel || entryName;
  const absoluteEntry = path.resolve(projectRoot, entryPath);
  await ensureOutputDir();

  try {
    const result = await build({
      entryPoints: { [entryName]: absoluteEntry },
      outdir: outputDir,
      bundle: true,
      format: "esm",
      platform: "browser",
      target: ["es2020"],
      entryNames: "[name]-[hash]",
      chunkNames: "chunks/[name]-[hash]",
      assetNames: "assets/[name]-[hash]",
      sourcemap: true,
      splitting: true,
      minify: true,
      treeShaking: true,
      metafile: true,
      define: {
        "process.env.NODE_ENV": '"production"',
      },
      logLevel: "info",
    });

    if (result.errors?.length) {
      console.error(`[${label}] Build finished with errors:`);
      result.errors.forEach((err) => console.error(err));
      process.exitCode = 1;
      return;
    }

    const outputs = Object.entries(result.metafile.outputs || {});
    const entryOutput = outputs.find(([outputPath, meta]) => {
      if (!outputPath.endsWith(".js")) {
        return false;
      }
      if (meta.entryPoint) {
        const resolvedEntry = path.resolve(projectRoot, meta.entryPoint);
        if (resolvedEntry === absoluteEntry) {
          return true;
        }
      }
      const filename = path.basename(outputPath);
      return filename.startsWith(`${entryName}-`);
    });

    if (!entryOutput) {
      throw new Error(`Unable to locate output chunk for entry ${entryName}`);
    }

    const [outputPath] = entryOutput;
    const absoluteOutput = path.resolve(projectRoot, outputPath);
    const relativeToOutputDir = path.relative(outputDir, absoluteOutput);
    const manifestValue = toPosixPath(relativeToOutputDir);

    await updateManifest(manifestKey, manifestValue);
    console.log(`[${label}] Bundle written to static/js/${manifestValue}`);
  } catch (error) {
    console.error(`[${label}] Build failed:`, error);
    process.exitCode = 1;
  }
}

export { projectRoot, outputDir, manifestPath };
