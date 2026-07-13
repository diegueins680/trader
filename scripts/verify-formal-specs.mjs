#!/usr/bin/env node

import { readFile, readdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const REPO_ROOT = path.resolve(HERE, "..");
export const DEFAULT_MANIFEST = path.join(REPO_ROOT, "formal", "specifications.json");

const VALID_CRITICALITY = new Set(["safety", "correctness", "security", "operational", "presentation"]);
const VALID_EVIDENCE_LEVELS = new Set(["bounded-exhaustive", "property", "integration", "regression", "static", "operational"]);

function normalizePath(value) {
  return value.split(path.sep).join("/").replace(/^\.\//, "");
}

export function globToRegExp(glob) {
  let source = "";
  for (let index = 0; index < glob.length; index += 1) {
    const char = glob[index];
    if (char === "*") {
      if (glob[index + 1] === "*") {
        source += ".*";
        index += 1;
      } else {
        source += "[^/]*";
      }
    } else if (char === "?") {
      source += "[^/]";
    } else {
      source += char.replace(/[\\^$+?.()|{}\[\]]/g, "\\$&");
    }
  }
  return new RegExp(`^${source}$`);
}

export function matchesGlob(file, glob) {
  return globToRegExp(normalizePath(glob)).test(normalizePath(file));
}

async function walkFiles(root, relative = "") {
  const directory = path.join(root, relative);
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const child = path.join(relative, entry.name);
    if (entry.isDirectory()) files.push(...(await walkFiles(root, child)));
    else if (entry.isFile()) files.push(normalizePath(child));
  }
  return files;
}

async function filesInCoverage(repoRoot, coverage) {
  const files = new Set(coverage.explicit ?? []);
  for (const area of coverage.areas ?? []) {
    const absoluteRoot = path.join(repoRoot, area.root);
    if (!existsSync(absoluteRoot)) continue;
    const candidates = await walkFiles(absoluteRoot);
    for (const candidate of candidates) {
      const full = normalizePath(path.posix.join(area.root, candidate));
      const included = (area.include ?? []).some((glob) => matchesGlob(full, glob));
      const excluded = (area.exclude ?? []).some((glob) => matchesGlob(full, glob));
      if (included && !excluded) files.add(full);
    }
  }
  return [...files].sort();
}

function findDuplicates(values) {
  const seen = new Set();
  const duplicates = new Set();
  for (const value of values) {
    if (seen.has(value)) duplicates.add(value);
    seen.add(value);
  }
  return [...duplicates].sort();
}

function validateDependencyGraph(specifications, errors) {
  const byId = new Map(specifications.map((spec) => [spec.id, spec]));
  const visiting = new Set();
  const visited = new Set();
  function visit(id, trail) {
    if (visiting.has(id)) {
      errors.push(`spec dependency cycle: ${[...trail, id].join(" -> ")}`);
      return;
    }
    if (visited.has(id)) return;
    visiting.add(id);
    const spec = byId.get(id);
    for (const dependency of spec?.dependsOn ?? []) {
      if (!byId.has(dependency)) errors.push(`${id}: unknown dependency ${dependency}`);
      else visit(dependency, [...trail, id]);
    }
    visiting.delete(id);
    visited.add(id);
  }
  for (const id of byId.keys()) visit(id, []);
}

export async function verifyManifest(manifest, { repoRoot = REPO_ROOT } = {}) {
  const errors = [];
  const specifications = manifest.specifications ?? [];
  const globalIds = (manifest.globalInvariants ?? []).map((item) => item.id);
  const specIds = specifications.map((item) => item.id);
  const contractIds = specifications.flatMap((spec) =>
    ["requires", "ensures", "invariants", "failures"].flatMap((kind) =>
      (spec[kind] ?? []).map((clause) => clause.id),
    ),
  );

  if (manifest.schemaVersion !== 1) errors.push("schemaVersion must be 1");
  if (!manifest.semantics?.transitionSystem) errors.push("semantics.transitionSystem is required");
  for (const [label, values] of [
    ["global invariant", globalIds],
    ["spec", specIds],
    ["contract clause", contractIds],
  ]) {
    const duplicates = findDuplicates(values);
    if (duplicates.length > 0) errors.push(`duplicate ${label} IDs: ${duplicates.join(", ")}`);
  }

  const globalIdSet = new Set(globalIds);
  for (const global of manifest.globalInvariants ?? []) {
    if (!global.statement?.trim()) errors.push(`${global.id}: empty global invariant`);
  }

  for (const spec of specifications) {
    if (!spec.id || !spec.title || !(spec.features?.length > 0)) errors.push(`${spec.id ?? "<unknown>"}: id, title, and features are required`);
    if (!VALID_CRITICALITY.has(spec.criticality)) errors.push(`${spec.id}: invalid criticality ${spec.criticality}`);
    if (!(spec.implementation?.length > 0)) errors.push(`${spec.id}: implementation scope is empty`);
    for (const kind of ["requires", "ensures", "invariants", "failures"]) {
      if (!(spec[kind]?.length > 0)) errors.push(`${spec.id}: ${kind} must contain at least one formal clause`);
      for (const clause of spec[kind] ?? []) {
        if (!clause.id || !clause.statement?.trim()) errors.push(`${spec.id}: malformed ${kind} clause`);
      }
    }
    for (const globalId of spec.uses ?? []) {
      if (!globalIdSet.has(globalId)) errors.push(`${spec.id}: unknown global invariant ${globalId}`);
    }
    if (!(spec.evidence?.length > 0)) errors.push(`${spec.id}: no verification evidence`);
    for (const evidence of spec.evidence ?? []) {
      if (!VALID_EVIDENCE_LEVELS.has(evidence.level)) errors.push(`${spec.id}: invalid evidence level ${evidence.level}`);
      if (!evidence.path) {
        errors.push(`${spec.id}: evidence path missing`);
        continue;
      }
      const evidencePath = path.join(repoRoot, evidence.path);
      if (!existsSync(evidencePath)) {
        errors.push(`${spec.id}: missing evidence ${evidence.path}`);
        continue;
      }
      if (evidence.contains) {
        const contents = await readFile(evidencePath, "utf8");
        if (!contents.includes(evidence.contains)) errors.push(`${spec.id}: evidence marker ${JSON.stringify(evidence.contains)} missing from ${evidence.path}`);
      }
    }
    if (spec.criticality === "safety" && !(spec.evidence ?? []).some((item) => ["bounded-exhaustive", "property", "integration", "regression"].includes(item.level))) {
      errors.push(`${spec.id}: safety-critical feature lacks executable evidence`);
    }
  }
  validateDependencyGraph(specifications, errors);

  const coveredFiles = await filesInCoverage(repoRoot, manifest.coverage ?? {});
  const uncoveredFiles = coveredFiles.filter(
    (file) => !specifications.some((spec) => (spec.implementation ?? []).some((glob) => matchesGlob(file, glob))),
  );
  if (uncoveredFiles.length > 0) errors.push(`uncovered implementation files:\n${uncoveredFiles.map((file) => `  - ${file}`).join("\n")}`);

  for (const spec of specifications) {
    for (const glob of spec.implementation ?? []) {
      if (!coveredFiles.some((file) => matchesGlob(file, glob))) errors.push(`${spec.id}: implementation glob matches no covered file: ${glob}`);
    }
  }

  return {
    ok: errors.length === 0,
    errors,
    statistics: {
      specifications: specifications.length,
      features: specifications.reduce((sum, spec) => sum + (spec.features?.length ?? 0), 0),
      clauses: contractIds.length + globalIds.length,
      implementationFiles: coveredFiles.length,
      evidence: specifications.reduce((sum, spec) => sum + (spec.evidence?.length ?? 0), 0),
    },
  };
}

export async function loadAndVerify(manifestPath = DEFAULT_MANIFEST, options = {}) {
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  return verifyManifest(manifest, options);
}

async function main() {
  const manifestPath = process.argv[2] ? path.resolve(process.argv[2]) : DEFAULT_MANIFEST;
  const result = await loadAndVerify(manifestPath);
  if (!result.ok) {
    for (const error of result.errors) process.stderr.write(`formal-spec error: ${error}\n`);
    process.exitCode = 1;
    return;
  }
  const stats = result.statistics;
  process.stdout.write(
    `Formal specification registry valid: ${stats.specifications} specs, ${stats.features} named features, ${stats.clauses} clauses, ${stats.implementationFiles} implementation files, ${stats.evidence} evidence links.\n`,
  );
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  await main();
}
