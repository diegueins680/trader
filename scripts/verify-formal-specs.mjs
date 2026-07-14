#!/usr/bin/env node

import { readFile, readdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const REPO_ROOT = path.resolve(HERE, "..");
export const DEFAULT_MANIFEST = path.join(REPO_ROOT, "formal", "specifications.json");
export const DEFAULT_RISK_REGISTER = path.join(REPO_ROOT, "formal", "risk-register.json");

const VALID_CRITICALITY = new Set(["safety", "correctness", "security", "operational", "presentation"]);
const VALID_EVIDENCE_LEVELS = new Set(["bounded-exhaustive", "property", "integration", "regression", "static", "operational"]);
const CANONICAL_RISK_SEVERITIES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"];
const CANONICAL_RISK_STATUSES = ["OPEN", "MITIGATED", "CLOSED"];
const CANONICAL_RISK_PROJECTIONS = {
  markdown: "haskell/RISK_REGISTER.md",
  haskell: "haskell/app/Trader/Formal/RiskRegister.hs",
};
const RISK_ID_PATTERN = /^[A-Z0-9]+(?:-[A-Z0-9]+)+$/;

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

function sameSequence(left, right) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function stripMarkdown(value) {
  return value.replace(/\*\*/g, "").replace(/`/g, "").trim();
}

export function parseMarkdownRiskProjection(source) {
  const entries = [];
  for (const line of source.split(/\r?\n/)) {
    if (!line.trimStart().startsWith("|")) continue;
    const cells = line
      .trim()
      .slice(1, -1)
      .split("|")
      .map(stripMarkdown);
    if (cells.length < 6 || cells[0] === "ID" || cells.every((cell) => /^-+$/.test(cell))) continue;
    entries.push({ id: cells[0], severity: cells[2], status: cells[4] });
  }
  return entries;
}

export function parseHaskellRiskProjection(source) {
  const idMappings = [];
  const mappingPattern = /^\s*([A-Z][A-Z0-9_]*)\s*->\s*"([A-Z0-9-]+)"\s*$/gm;
  for (const match of source.matchAll(mappingPattern)) {
    idMappings.push({ constructor: match[1], id: match[2] });
  }

  const constructors = [];
  const declaration = source.match(/data\s+RiskID([\s\S]*?)\n\s+deriving\b/);
  if (declaration) {
    for (const line of declaration[1].split(/\r?\n/)) {
      const match = line.match(/^\s*(?:\||=)\s*([A-Z][A-Z0-9_]*)\s*$/);
      if (match) constructors.push(match[1]);
    }
  }

  const entries = [];
  const entryPattern = /\briskEntry\s+([A-Z][A-Z0-9_]*)\s+(LOW|MEDIUM|HIGH|CRITICAL)\s+(OPEN|MITIGATED|CLOSED)\b/g;
  const idByConstructor = new Map(idMappings.map((mapping) => [mapping.constructor, mapping.id]));
  for (const match of source.matchAll(entryPattern)) {
    entries.push({
      constructor: match[1],
      id: idByConstructor.get(match[1]),
      severity: match[2],
      status: match[3],
    });
  }
  return { constructors, idMappings, entries };
}

function compareRiskProjection(label, canonicalEntries, projectionEntries, errors) {
  const canonicalIds = canonicalEntries.map((entry) => entry.id);
  const projectionIds = projectionEntries.map((entry) => entry.id);
  const canonicalIdSet = new Set(canonicalIds);
  const projectionIdSet = new Set(projectionIds);
  const duplicates = findDuplicates(projectionIds.filter(Boolean));
  if (duplicates.length > 0) errors.push(`${label}: duplicate risk IDs: ${duplicates.join(", ")}`);

  const missing = canonicalIds.filter((id) => !projectionIdSet.has(id));
  const unexpected = projectionIds.filter((id) => id && !canonicalIdSet.has(id));
  if (missing.length > 0) errors.push(`${label}: missing canonical risk IDs: ${missing.join(", ")}`);
  if (unexpected.length > 0) errors.push(`${label}: unexpected risk IDs: ${unexpected.join(", ")}`);
  if (!sameSequence(canonicalIds, projectionIds)) errors.push(`${label}: risk IDs are not in canonical order`);

  const canonicalById = new Map(canonicalEntries.map((entry) => [entry.id, entry]));
  for (const projection of projectionEntries) {
    const canonical = canonicalById.get(projection.id);
    if (!canonical) continue;
    if (projection.severity !== canonical.severity) {
      errors.push(`${label}: ${projection.id} severity ${projection.severity} does not match canonical ${canonical.severity}`);
    }
    if (projection.status !== canonical.status) {
      errors.push(`${label}: ${projection.id} status ${projection.status} does not match canonical ${canonical.status}`);
    }
  }
}

export function verifyRiskRegisterSources(register, { markdownSource, haskellSource }) {
  const errors = [];
  const entries = register.entries ?? [];
  const ids = entries.map((entry) => entry.id);

  if (register.schemaVersion !== 1) errors.push("risk register schemaVersion must be 1");
  if (!sameSequence(register.severities ?? [], CANONICAL_RISK_SEVERITIES)) {
    errors.push(`risk register severities must be ${CANONICAL_RISK_SEVERITIES.join(", ")}`);
  }
  if (!sameSequence(register.statuses ?? [], CANONICAL_RISK_STATUSES)) {
    errors.push(`risk register statuses must be ${CANONICAL_RISK_STATUSES.join(", ")}`);
  }
  if (entries.length === 0) errors.push("risk register must contain at least one entry");
  const duplicates = findDuplicates(ids.filter(Boolean));
  if (duplicates.length > 0) errors.push(`duplicate canonical risk IDs: ${duplicates.join(", ")}`);
  if (!sameSequence(ids, [...ids].sort())) errors.push("canonical risk entries must be sorted by ID");
  for (const entry of entries) {
    if (!RISK_ID_PATTERN.test(entry.id ?? "")) errors.push(`invalid canonical risk ID: ${entry.id ?? "<missing>"}`);
    if (!CANONICAL_RISK_SEVERITIES.includes(entry.severity)) errors.push(`${entry.id ?? "<missing>"}: invalid severity ${entry.severity}`);
    if (!CANONICAL_RISK_STATUSES.includes(entry.status)) errors.push(`${entry.id ?? "<missing>"}: invalid status ${entry.status}`);
  }

  const markdownEntries = parseMarkdownRiskProjection(markdownSource);
  compareRiskProjection("Markdown risk register", entries, markdownEntries, errors);

  const haskell = parseHaskellRiskProjection(haskellSource);
  const mappedConstructors = haskell.idMappings.map((mapping) => mapping.constructor);
  const mappedIds = haskell.idMappings.map((mapping) => mapping.id);
  const entryConstructors = haskell.entries.map((entry) => entry.constructor);
  for (const [label, values] of [
    ["Haskell RiskID constructor mapping", mappedConstructors],
    ["Haskell canonical risk ID mapping", mappedIds],
    ["Haskell risk entry constructor", entryConstructors],
  ]) {
    const repeated = findDuplicates(values);
    if (repeated.length > 0) errors.push(`${label}: duplicates: ${repeated.join(", ")}`);
  }
  if (!sameSequence(haskell.constructors, mappedConstructors)) {
    errors.push("Haskell RiskID constructors and riskIdText mappings differ or are out of order");
  }
  if (!sameSequence(haskell.constructors, entryConstructors)) {
    errors.push("Haskell RiskID constructors and riskRegister entries differ or are out of order");
  }
  for (const entry of haskell.entries) {
    if (!entry.id) errors.push(`Haskell risk entry ${entry.constructor} has no canonical riskIdText mapping`);
  }
  compareRiskProjection("Haskell risk register", entries, haskell.entries, errors);

  return {
    ok: errors.length === 0,
    errors,
    statistics: { entries: entries.length },
  };
}

export async function verifyRiskRegister(register, { repoRoot = REPO_ROOT } = {}) {
  const markdownPath = register.projections?.markdown;
  const haskellPath = register.projections?.haskell;
  const errors = [];
  if (!markdownPath) errors.push("risk register projections.markdown is required");
  if (!haskellPath) errors.push("risk register projections.haskell is required");
  if (markdownPath && markdownPath !== CANONICAL_RISK_PROJECTIONS.markdown) {
    errors.push(`risk register Markdown projection must be ${CANONICAL_RISK_PROJECTIONS.markdown}`);
  }
  if (haskellPath && haskellPath !== CANONICAL_RISK_PROJECTIONS.haskell) {
    errors.push(`risk register Haskell projection must be ${CANONICAL_RISK_PROJECTIONS.haskell}`);
  }
  if (errors.length > 0) return { ok: false, errors, statistics: { entries: register.entries?.length ?? 0 } };

  let markdownSource;
  let haskellSource;
  try {
    [markdownSource, haskellSource] = await Promise.all([
      readFile(path.join(repoRoot, markdownPath), "utf8"),
      readFile(path.join(repoRoot, haskellPath), "utf8"),
    ]);
  } catch {
    return {
      ok: false,
      errors: ["canonical risk-register projection is missing or unreadable"],
      statistics: { entries: register.entries?.length ?? 0 },
    };
  }
  return verifyRiskRegisterSources(register, { markdownSource, haskellSource });
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

export async function loadAndVerifyRiskRegister(registerPath = DEFAULT_RISK_REGISTER, options = {}) {
  const register = JSON.parse(await readFile(registerPath, "utf8"));
  return verifyRiskRegister(register, options);
}

async function main() {
  const manifestPath = process.argv[2] ? path.resolve(process.argv[2]) : DEFAULT_MANIFEST;
  const [result, riskResult] = await Promise.all([loadAndVerify(manifestPath), loadAndVerifyRiskRegister()]);
  if (!result.ok || !riskResult.ok) {
    for (const error of result.errors) process.stderr.write(`formal-spec error: ${error}\n`);
    for (const error of riskResult.errors) process.stderr.write(`risk-register error: ${error}\n`);
    process.exitCode = 1;
    return;
  }
  const stats = result.statistics;
  process.stdout.write(
    `Formal specification registry valid: ${stats.specifications} specs, ${stats.features} named features, ${stats.clauses} clauses, ${stats.implementationFiles} implementation files, ${stats.evidence} evidence links, ${riskResult.statistics.entries} canonical risks.\n`,
  );
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  await main();
}
