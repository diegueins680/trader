import fs from "node:fs/promises";
import { randomUUID } from "node:crypto";
import path from "node:path";

export function stripMarkdownFences(raw) {
  const text = String(raw ?? "").trim();
  const fenced = text.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i);
  return fenced ? fenced[1].trim() : text;
}

export function extractResponseText(response) {
  const output = Array.isArray(response?.output) ? response.output : [];
  const parts = [];
  for (const item of output) {
    if (item?.type !== "message") continue;
    const content = Array.isArray(item.content) ? item.content : [];
    for (const part of content) {
      if (part?.type === "output_text" && typeof part.text === "string") parts.push(part.text);
    }
  }
  return parts.join("\n").trim();
}

export function extractAnthropicResponseText(response) {
  const content = Array.isArray(response?.content) ? response.content : [];
  const parts = [];
  for (const part of content) {
    if (part?.type === "text" && typeof part.text === "string") parts.push(part.text);
  }
  return parts.join("\n").trim();
}

export function parseJsonResponse(raw) {
  const text = stripMarkdownFences(raw);
  if (!text) throw new Error("Model returned empty text.");
  try {
    return JSON.parse(text);
  } catch (err) {
    // Anthropic without assistant prefill can emit a stray leading/trailing
    // line. Fall back to extracting the first balanced JSON object.
    const extracted = extractFirstJsonObject(text);
    if (extracted) {
      try {
        return JSON.parse(extracted);
      } catch {
        // fall through to original error
      }
    }
    throw new Error(`Model returned invalid JSON: ${err instanceof Error ? err.message : String(err)}`);
  }
}

function extractFirstJsonObject(text) {
  const start = text.indexOf("{");
  if (start === -1) return "";
  let depth = 0;
  let inString = false;
  let escape = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (escape) {
        escape = false;
      } else if (ch === "\\") {
        escape = true;
      } else if (ch === '"') {
        inString = false;
      }
      continue;
    }
    if (ch === '"') {
      inString = true;
      continue;
    }
    if (ch === "{") depth++;
    else if (ch === "}") {
      depth--;
      if (depth === 0) return text.slice(start, i + 1);
    }
  }
  return "";
}

function extractCodexEventMessageText(item) {
  if (!item || typeof item !== "object") return "";
  if (typeof item.text === "string" && item.text.trim()) return item.text.trim();
  const content = Array.isArray(item.content) ? item.content : [];
  return content
    .map((part) => {
      if (!part || typeof part !== "object") return "";
      if (typeof part.text === "string") return part.text;
      return "";
    })
    .filter(Boolean)
    .join("\n")
    .trim();
}

export function extractCodexExecLastMessage(raw) {
  const lines = String(raw ?? "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  let lastMessage = "";
  const seenEventTypes = [];

  for (const line of lines) {
    if (!line.startsWith("{")) continue;
    let event;
    try {
      event = JSON.parse(line);
    } catch (err) {
      throw new Error(`Codex exec returned invalid JSONL: ${err instanceof Error ? err.message : String(err)}`);
    }
    if (typeof event?.type === "string") seenEventTypes.push(event.type);
    if (event?.type !== "item.completed") continue;
    const itemType = typeof event?.item?.type === "string" ? event.item.type : "";
    if (!["agent_message", "assistant_message", "message"].includes(itemType)) continue;
    const text = extractCodexEventMessageText(event.item);
    if (text) lastMessage = text;
  }

  if (!lastMessage) {
    const eventSummary = uniqueStrings(seenEventTypes).join(", ");
    throw new Error(
      `Codex exec returned no completed agent message${eventSummary ? ` (events: ${eventSummary})` : ""}.`,
    );
  }
  return lastMessage;
}

export function clampText(raw, maxChars) {
  const text = String(raw ?? "");
  const limit = Math.max(0, Math.trunc(maxChars));
  if (text.length <= limit) return text;
  if (limit === 0) return "";

  const renderTruncation = (suffixBuilder) => {
    let removed = text.length;
    for (let i = 0; i < 4; i += 1) {
      const suffix = suffixBuilder(removed);
      const keep = limit - suffix.length;
      if (keep < 0) return null;
      const nextRemoved = text.length - keep;
      if (nextRemoved === removed) return `${text.slice(0, keep)}${suffix}`;
      removed = nextRemoved;
    }
    const suffix = suffixBuilder(removed);
    const keep = limit - suffix.length;
    return keep < 0 ? null : `${text.slice(0, keep)}${suffix}`;
  };

  return (
    renderTruncation((removed) => `\n...[truncated ${removed} chars]`) ??
    renderTruncation((removed) => `...[+${removed}]`) ??
    `${text.slice(0, Math.max(0, limit - 3))}${".".repeat(Math.min(3, limit))}`
  );
}

export function sanitizeRelativePath(raw) {
  const value = String(raw ?? "").trim().replace(/\\/g, "/");
  if (!value) throw new Error("Path is empty.");
  if (value.includes("\0")) throw new Error(`Path contains NUL byte: ${value}`);
  if (value.startsWith("/")) throw new Error(`Absolute path is not allowed: ${value}`);
  if (/^[A-Za-z]:\//.test(value)) throw new Error(`Absolute path is not allowed: ${value}`);
  if (value.split("/").some((part) => part === "..")) throw new Error(`Path traversal is not allowed: ${value}`);
  const normalized = path.posix.normalize(value).replace(/^\.\/+/, "");
  if (normalized.startsWith("/")) throw new Error(`Absolute path is not allowed: ${value}`);
  if (/^[A-Za-z]:\//.test(normalized)) throw new Error(`Absolute path is not allowed: ${value}`);
  if (normalized === ".." || normalized.startsWith("../") || normalized.split("/").some((part) => part === "..")) {
    throw new Error(`Path traversal is not allowed: ${value}`);
  }
  if (!normalized || normalized === ".") throw new Error(`Path resolves to empty: ${value}`);
  return normalized;
}

function readString(raw, field) {
  if (typeof raw !== "string") throw new Error(`${field} must be a string.`);
  const value = raw.trim();
  if (!value) throw new Error(`${field} must not be empty.`);
  return value;
}

function coerceNarrativeString(raw) {
  if (typeof raw === "string") return raw.trim();
  if (raw === undefined || raw === null) return "";
  if (Array.isArray(raw)) {
    return raw.map(coerceNarrativeString).filter(Boolean).join("\n");
  }
  if (typeof raw === "object") {
    for (const key of ["text", "summary", "message", "title", "value", "content"]) {
      const value = coerceNarrativeString(raw[key]);
      if (value) return value;
    }
    return Object.entries(raw)
      .map(([key, value]) => {
        const text = coerceNarrativeString(value);
        return text ? `${key}: ${text}` : "";
      })
      .filter(Boolean)
      .join("\n");
  }
  return String(raw).trim();
}

function readNarrativeString(raw, field) {
  const value = coerceNarrativeString(raw);
  if (!value) throw new Error(`${field} must not be empty.`);
  return value;
}

function readStringArray(raw, field, maxItems = 12) {
  if (!Array.isArray(raw)) throw new Error(`${field} must be an array.`);
  if (raw.length === 0) throw new Error(`${field} must not be empty.`);
  if (raw.length > maxItems) throw new Error(`${field} exceeds max items (${maxItems}).`);
  return raw.map((item, idx) => readString(item, `${field}[${idx}]`));
}

function matchesScopedPathPrefix(value, prefix) {
  const normalizedPrefix = sanitizeRelativePath(prefix);
  // Exact file scopes must not match sibling lookalikes such as
  // `FORMAL_METHODS.md.bak`; directory scopes end with `/`.
  if (!normalizedPrefix.endsWith("/")) return value === normalizedPrefix;
  return value.startsWith(normalizedPrefix);
}

function readScopedPath(raw, field, allowedPrefixes) {
  const value = sanitizeRelativePath(readString(raw, field));
  if (!allowedPrefixes.some((prefix) => matchesScopedPathPrefix(value, prefix))) {
    throw new Error(`${field} must be within: ${allowedPrefixes.join(", ")}`);
  }
  return value;
}

const ALGORITHM_REVIEW_PREFIXES = ["haskell/app/"];
const FORMAL_METHODS_REVIEW_PREFIXES = ["FORMAL_METHODS.md", "haskell/app/Trader/Formal/", "test/", "haskell/test/"];

export function normalizeIdeaSelection(raw, options = {}) {
  const algorithmReviewPrefixes = options.algorithmReviewPrefixes || ALGORITHM_REVIEW_PREFIXES;
  const formalMethodsReviewPrefixes = options.formalMethodsReviewPrefixes || FORMAL_METHODS_REVIEW_PREFIXES;
  const obj = raw && typeof raw === "object" ? raw : {};
  const noChange = obj.noChange === true;
  const filesNeeded = noChange
    ? []
    : readStringArray(obj.filesNeeded, "filesNeeded", 10).map(sanitizeRelativePath);
  const algorithmReviewPath = noChange
    ? String(obj.algorithmReviewPath ?? "").trim()
    : readScopedPath(obj.algorithmReviewPath, "algorithmReviewPath", algorithmReviewPrefixes);
  const formalMethodsPath = noChange
    ? String(obj.formalMethodsPath ?? "").trim()
    : readScopedPath(obj.formalMethodsPath, "formalMethodsPath", formalMethodsReviewPrefixes);
  if (!noChange) {
    if (!filesNeeded.includes(algorithmReviewPath)) throw new Error("filesNeeded must include algorithmReviewPath.");
    if (!filesNeeded.includes(formalMethodsPath)) throw new Error("filesNeeded must include formalMethodsPath.");
  }
  return {
    noChange,
    title: noChange ? String(obj.title ?? "").trim() : readString(obj.title, "title"),
    rationale: noChange ? String(obj.rationale ?? "").trim() : readString(obj.rationale, "rationale"),
    algorithmReviewPath,
    algorithmReviewFocus: noChange
      ? String(obj.algorithmReviewFocus ?? "").trim()
      : readString(obj.algorithmReviewFocus, "algorithmReviewFocus"),
    formalMethodsPath,
    formalMethodsFocus: noChange
      ? String(obj.formalMethodsFocus ?? "").trim()
      : readString(obj.formalMethodsFocus, "formalMethodsFocus"),
    filesNeeded,
    verificationCommands: Array.isArray(obj.verificationCommands)
      ? obj.verificationCommands.map((item, idx) => readString(item, `verificationCommands[${idx}]`))
      : [],
  };
}

function readOptionalString(raw, field) {
  if (raw === undefined || raw === null) return "";
  if (typeof raw !== "string") throw new Error(`${field} must be a string.`);
  return raw;
}

function normalizeReplacement(raw, idx, changePath) {
  if (!raw || typeof raw !== "object") throw new Error(`changes[${idx}].replacements[] must be an object.`);
  const find = readString(raw.find, `changes[${idx}].replacements[].find`);
  const replace = readOptionalString(raw.replace ?? "", `changes[${idx}].replacements[].replace`);
  const expectedCount = raw.expectedCount === undefined ? 1 : Number(raw.expectedCount);
  if (!Number.isInteger(expectedCount) || expectedCount <= 0) {
    throw new Error(`changes[${idx}].replacements[].expectedCount must be a positive integer.`);
  }
  if (looksLikePatchPayload(find) || looksLikePatchPayload(replace)) {
    throw new Error(`changes[${idx}].replacements[] for ${changePath} must not contain patch/diff payloads.`);
  }
  return {
    find,
    replace,
    expectedCount,
    reason: typeof raw.reason === "string" ? raw.reason.trim() : "",
  };
}

function normalizeFileChange(raw, idx) {
  if (!raw || typeof raw !== "object") throw new Error(`changes[${idx}] must be an object.`);
  const path = sanitizeRelativePath(raw.path);
  const deleteFile = raw.delete === true;
  const hasContent = Object.prototype.hasOwnProperty.call(raw, "content");
  const replacements = Array.isArray(raw.replacements)
    ? raw.replacements.map((replacement) => normalizeReplacement(replacement, idx, path))
    : [];
  if (!deleteFile && hasContent && replacements.length > 0) {
    throw new Error(`changes[${idx}] for ${path} must use either content or replacements, not both.`);
  }
  if (!deleteFile && !hasContent && replacements.length === 0) {
    throw new Error(`changes[${idx}] for ${path} must include content or replacements.`);
  }
  const content = deleteFile || !hasContent ? "" : readString(raw.content ?? "", `changes[${idx}].content`);
  if (!deleteFile && hasContent && looksLikePatchPayload(content)) {
    throw new Error(
      `changes[${idx}].content for ${path} looks like a patch/diff payload; provide complete replacement file content instead.`,
    );
  }
  if (!deleteFile && hasContent && looksLikeInstructionPayload(content, path)) {
    throw new Error(
      `changes[${idx}].content for ${path} looks like edit instructions; provide complete replacement file content instead.`,
    );
  }
  return {
    path,
    delete: deleteFile,
    replacements,
    content,
    reason: typeof raw.reason === "string" ? raw.reason.trim() : "",
  };
}

function looksLikePatchPayload(content) {
  const trimmed = String(content ?? "").trimStart();
  return trimmed.startsWith("*** Begin Patch") || trimmed.startsWith("diff --git ");
}

function looksLikeInstructionPayload(content, filePath) {
  const firstLine = String(content ?? "").trimStart().split(/\r?\n/, 1)[0]?.trim() ?? "";
  if (!firstLine) return false;

  const verb = "(?:replace|add|update|modify|change|insert|delete|remove|append)";
  const pathPattern = escapeRegExp(filePath);
  const genericPathPattern = "[A-Za-z0-9_./-]+";

  return (
    new RegExp(`^In\\s+${pathPattern}\\s*,\\s*${verb}\\b`, "i").test(firstLine) ||
    new RegExp(`^In\\s+${genericPathPattern}\\s*,\\s*${verb}\\b`, "i").test(firstLine) ||
    new RegExp(`^For\\s+${genericPathPattern}\\s*,\\s*${verb}\\b`, "i").test(firstLine) ||
    new RegExp(`^${verb}\\s+(?:this|the)\\b`, "i").test(firstLine)
  );
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function normalizePatchPlan(raw) {
  const obj = raw && typeof raw === "object" ? raw : {};
  const noChange = obj.noChange === true;
  const changes = Array.isArray(obj.changes) ? obj.changes.map(normalizeFileChange) : [];
  if (!noChange && changes.length === 0) throw new Error("changes must not be empty.");
  if (changes.length > 10) throw new Error("changes exceeds max items (10).");
  const seen = new Set();
  for (const change of changes) {
    if (seen.has(change.path)) throw new Error(`changes contains duplicate path: ${change.path}`);
    seen.add(change.path);
  }
  return {
    noChange,
    title: noChange ? String(obj.title ?? "").trim() : readNarrativeString(obj.title, "title"),
    summary: noChange ? String(obj.summary ?? "").trim() : readNarrativeString(obj.summary, "summary"),
    commitMessage: noChange ? String(obj.commitMessage ?? "").trim() : readNarrativeString(obj.commitMessage, "commitMessage"),
    algorithmReviewSummary: noChange
      ? String(obj.algorithmReviewSummary ?? "").trim()
      : readNarrativeString(obj.algorithmReviewSummary, "algorithmReviewSummary"),
    formalMethodsSummary: noChange
      ? String(obj.formalMethodsSummary ?? "").trim()
      : readNarrativeString(obj.formalMethodsSummary, "formalMethodsSummary"),
    changes,
    verificationCommands: Array.isArray(obj.verificationCommands)
      ? obj.verificationCommands.map((item, idx) => readString(item, `verificationCommands[${idx}]`))
      : [],
  };
}

export function uniqueStrings(values) {
  return Array.from(new Set(values.map((value) => String(value))));
}

export function parseGitStatusPaths(rawStatus) {
  const lines = String(rawStatus ?? "")
    .split(/\r?\n/)
    .map((line) => line.trimEnd())
    .filter(Boolean);
  const paths = [];
  for (const line of lines) {
    const match = line.match(/^[ A-Z?!]{1,2}\s+(.*)$/);
    const entry = match?.[1]?.trim() ?? "";
    if (!entry) continue;
    const nextPath = entry.includes(" -> ") ? entry.split(" -> ").pop() : entry;
    if (!nextPath) continue;
    paths.push(nextPath.replace(/^"(.*)"$/, "$1"));
  }
  return uniqueStrings(paths);
}

export function normalizeGitBranchShortName(rawBranch) {
  const raw = String(rawBranch ?? "").trim();
  if (!raw || raw === "HEAD" || raw === "origin" || raw.endsWith("/HEAD")) return "";
  return raw
    .replace(/^refs\/heads\//, "")
    .replace(/^refs\/remotes\/origin\//, "")
    .replace(/^origin\//, "")
    .trim();
}

export function isAutoloopRecoveryBranch(rawBranch) {
  const shortName = normalizeGitBranchShortName(rawBranch);
  return shortName.startsWith("autoloop/recovery/") || shortName.startsWith("autoloop/checkpoint/");
}

export function isAutoloopMergeBranch(rawBranch) {
  const shortName = normalizeGitBranchShortName(rawBranch);
  return shortName.startsWith("autoloop/") && !isAutoloopRecoveryBranch(shortName);
}

export function buildBranchMergeCandidates(
  { localBranches = [], remoteBranches = [], baseBranch = "main", includeRecoveryBranches = false } = {},
) {
  const base = normalizeGitBranchShortName(baseBranch || "main");
  const localByShortName = new Map();
  const remoteByShortName = new Map();

  for (const branch of localBranches) {
    const shortName = normalizeGitBranchShortName(branch);
    if (!shortName || shortName === base) continue;
    if (!includeRecoveryBranches && isAutoloopRecoveryBranch(shortName)) continue;
    localByShortName.set(shortName, String(branch).trim());
  }

  for (const branch of remoteBranches) {
    const shortName = normalizeGitBranchShortName(branch);
    if (!shortName || shortName === base) continue;
    if (!includeRecoveryBranches && isAutoloopRecoveryBranch(shortName)) continue;
    remoteByShortName.set(shortName, String(branch).trim());
  }

  return uniqueStrings([...localByShortName.keys(), ...remoteByShortName.keys()])
    .sort((left, right) => left.localeCompare(right))
    .map((shortName) => ({
      shortName,
      ref: localByShortName.get(shortName) || remoteByShortName.get(shortName) || shortName,
      localRef: localByShortName.get(shortName) || "",
      remoteRef: remoteByShortName.get(shortName) || "",
    }));
}

export function buildAutoloopScratchBranchCandidates({ localBranches = [], remoteBranches = [], baseBranch = "main" } = {}) {
  const base = normalizeGitBranchShortName(baseBranch || "main");
  const localByShortName = new Map();
  const remoteByShortName = new Map();

  for (const branch of localBranches) {
    const shortName = normalizeGitBranchShortName(branch);
    if (!shortName || shortName === base || !isAutoloopRecoveryBranch(shortName)) continue;
    localByShortName.set(shortName, String(branch).trim());
  }

  for (const branch of remoteBranches) {
    const shortName = normalizeGitBranchShortName(branch);
    if (!shortName || shortName === base || !isAutoloopRecoveryBranch(shortName)) continue;
    remoteByShortName.set(shortName, String(branch).trim());
  }

  return uniqueStrings([...localByShortName.keys(), ...remoteByShortName.keys()])
    .sort((left, right) => left.localeCompare(right))
    .map((shortName) => ({
      shortName,
      ref: localByShortName.get(shortName) || remoteByShortName.get(shortName) || shortName,
      localRef: localByShortName.get(shortName) || "",
      remoteRef: remoteByShortName.get(shortName) || "",
    }));
}

export function selectMergeVerificationTarget(changedPaths = []) {
  const paths = uniqueStrings(changedPaths.map((filePath) => sanitizeRelativePath(filePath)));
  if (paths.length === 0) return null;

  const documentationOnly = (filePath) =>
    filePath === "README.md" ||
    filePath === "CHANGELOG.md" ||
    filePath === "FORMAL_METHODS.md" ||
    filePath.startsWith("docs/") ||
    filePath.startsWith("artifacts/") ||
    filePath.startsWith("research-notes/");
  const codePaths = paths.filter((filePath) => !documentationOnly(filePath));
  if (codePaths.length === 0) return "full";

  const isHaskellPath = (filePath) => filePath.startsWith("haskell/") && !filePath.startsWith("haskell/web/");
  const isWebPath = (filePath) => filePath.startsWith("haskell/web/");
  const isAutomationPath = (filePath) =>
    filePath === "test/autoloop.test.mjs" ||
    filePath === "scripts/codex-logical-correctness-loop.sh" ||
    filePath === "scripts/restart-local-stack.sh" ||
    filePath.startsWith("scripts/autoloop") ||
    filePath.startsWith("scripts/install-autoloop");

  const hasHaskell = codePaths.some(isHaskellPath);
  const hasWeb = codePaths.some(isWebPath);
  const hasAutomation = codePaths.some(isAutomationPath);
  const hasUnknown = codePaths.some(
    (filePath) => !isHaskellPath(filePath) && !isWebPath(filePath) && !isAutomationPath(filePath),
  );

  if (hasHaskell && !hasWeb && !hasAutomation && !hasUnknown) return null;
  if (hasWeb && !hasHaskell && !hasAutomation && !hasUnknown) return "web";
  if (hasAutomation && !hasHaskell && !hasWeb && !hasUnknown) return "automation";
  return "full";
}

export function prepareShellCommand(command) {
  const normalized = String(command ?? "").trim();
  const needsGhcup =
    /\bcabal\b/.test(normalized) || normalized === "cd haskell && bash scripts/ci_smoke.sh";
  if (!normalized || !needsGhcup || normalized.includes(".ghcup/env")) return normalized;
  return `source "$HOME/.ghcup/env" 2>/dev/null || true; ${normalized}`;
}

export function resolveAutoloopBackend(rawBackend, { hasAnthropicKey, hasOpenAiKey, hasCodex }) {
  const requested = String(rawBackend ?? "").trim().toLowerCase();
  if (!requested || requested === "auto") {
    if (hasAnthropicKey) return "anthropic";
    if (hasOpenAiKey) return "openai";
    if (hasCodex) return "codex";
    return "";
  }
  if (requested === "anthropic" || requested === "claude") {
    return hasAnthropicKey ? "anthropic" : "";
  }
  if (requested === "openai" || requested === "responses") {
    return hasOpenAiKey ? "openai" : "";
  }
  if (requested === "codex") {
    return hasCodex ? "codex" : "";
  }
  throw new Error(`Unknown autoloop backend: ${rawBackend}`);
}

export function parseLsRemoteBranchHead(raw, branchName = "") {
  const text = String(raw ?? "").trim();
  if (!text) return "";
  const expectedRef = typeof branchName === "string" && branchName.trim() ? `refs/heads/${branchName.trim()}` : "";
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const [oid = "", ref = ""] = trimmed.split(/\s+/, 2);
    if (!/^[0-9a-f]{40}$/i.test(oid)) continue;
    if (!expectedRef || ref === expectedRef) return oid;
  }
  return "";
}

export function buildForceWithLeaseFlag(branchName, expectedOid) {
  const branch = readString(branchName, "branchName");
  const ref = branch.startsWith("refs/heads/") ? branch : `refs/heads/${branch}`;
  const expected = String(expectedOid ?? "").trim();
  if (expected && !/^[0-9a-f]{40}$/i.test(expected)) {
    throw new Error("expectedOid must be a 40-character hex object id.");
  }
  return `--force-with-lease=${ref}:${expected}`;
}

export function buildRemoteTrackingRefspec(branchName) {
  const branch = readString(branchName, "branchName");
  const headRef = branch.startsWith("refs/heads/") ? branch : `refs/heads/${branch}`;
  const shortName = headRef.replace(/^refs\/heads\//, "");
  return `${headRef}:refs/remotes/origin/${shortName}`;
}

export function buildActionsRunsApiPath(headSha, branchName = "", perPage = 50) {
  const sha = readString(headSha, "headSha");
  const params = new URLSearchParams();
  params.set("head_sha", sha);
  params.set("per_page", String(Math.min(100, Math.max(1, Math.trunc(perPage) || 1))));
  if (typeof branchName === "string" && branchName.trim()) {
    params.set("branch", branchName.replace(/^refs\/heads\//, "").trim());
  }
  return `repos/:owner/:repo/actions/runs?${params.toString()}`;
}

export function buildOpenAiApiError(status, payload) {
  const errorObj = payload?.error && typeof payload.error === "object" ? payload.error : {};
  const code = typeof errorObj.code === "string" ? errorObj.code : "";
  const type = typeof errorObj.type === "string" ? errorObj.type : "";
  const err = new Error(`OpenAI API request failed (${status}): ${JSON.stringify(payload)}`);
  err.openAiStatus = Number(status) || 0;
  err.openAiCode = code;
  err.openAiType = type;
  const authOrPermissionDenied =
    err.openAiStatus === 401 ||
    err.openAiStatus === 403 ||
    code === "invalid_api_key" ||
    code === "insufficient_permissions" ||
    type === "authentication_error" ||
    type === "permission_error";
  err.skipAutoloop =
    code === "insufficient_quota" ||
    type === "insufficient_quota" ||
    authOrPermissionDenied;
  return err;
}

export function buildAnthropicApiError(status, payload) {
  const errorObj = payload?.error && typeof payload.error === "object" ? payload.error : {};
  const type = typeof errorObj.type === "string" ? errorObj.type : "";
  const message = typeof errorObj.message === "string" ? errorObj.message : "";
  const err = new Error(`Anthropic API request failed (${status}): ${JSON.stringify(payload)}`);
  err.anthropicStatus = Number(status) || 0;
  err.anthropicType = type;
  err.anthropicMessage = message;
  const authOrPermissionDenied =
    err.anthropicStatus === 401 ||
    err.anthropicStatus === 403 ||
    type === "authentication_error" ||
    type === "permission_error";
  const billingExhausted =
    err.anthropicStatus === 402 || /credit balance|billing|quota|insufficient/i.test(message);
  err.skipAutoloop = authOrPermissionDenied || billingExhausted;
  return err;
}

function isNonNegativeSafeInteger(value) {
  return Number.isSafeInteger(value) && value >= 0;
}

function isPositiveSafeInteger(value) {
  return Number.isSafeInteger(value) && value > 0;
}

function runnerPidError(message, code) {
  const error = new Error(message);
  error.code = code;
  return error;
}

export function parseRunnerPidRecord(raw) {
  const text = String(raw ?? "").trim();
  if (!text) return null;

  if (/^[1-9][0-9]*$/.test(text)) {
    const pid = Number(text);
    return isPositiveSafeInteger(pid) ? { schemaVersion: 0, pid, token: null, acquiredAt: null } : null;
  }

  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch {
    return null;
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return null;
  if (parsed.schemaVersion !== 1 || !isPositiveSafeInteger(parsed.pid)) return null;
  if (typeof parsed.token !== "string" || !/^[A-Za-z0-9._-]{8,128}$/.test(parsed.token)) return null;
  if (typeof parsed.acquiredAt !== "string" || !Number.isFinite(Date.parse(parsed.acquiredAt))) return null;
  return {
    schemaVersion: 1,
    pid: parsed.pid,
    token: parsed.token,
    acquiredAt: parsed.acquiredAt,
  };
}

export function runnerProcessExists(pid) {
  if (!isPositiveSafeInteger(pid)) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    if (error?.code === "ESRCH") return false;
    if (error?.code === "EPERM") return true;
    throw error;
  }
}

async function quarantineStalePidFile(filePath, pid, attempt, nowMs) {
  const stalePath = `${filePath}.stale-${nowMs}-${pid}-${attempt}-${randomUUID()}`;
  try {
    await fs.rename(filePath, stalePath);
    return stalePath;
  } catch (error) {
    if (error?.code === "ENOENT") return null;
    throw error;
  }
}

export async function acquireRunnerPidFile({
  filePath,
  pid = process.pid,
  token = randomUUID(),
  nowMs = Date.now(),
  initializationGraceMs = 30_000,
  probePid = runnerProcessExists,
} = {}) {
  const target = String(filePath ?? "").trim();
  if (!target) throw new Error("filePath must not be empty.");
  if (!isPositiveSafeInteger(pid)) throw new Error("pid must be a positive safe integer.");
  if (typeof token !== "string" || !/^[A-Za-z0-9._-]{8,128}$/.test(token)) {
    throw new Error("token must contain 8-128 safe identifier characters.");
  }
  if (!Number.isFinite(nowMs)) throw new Error("nowMs must be finite.");
  if (!Number.isFinite(initializationGraceMs) || initializationGraceMs < 0) {
    throw new Error("initializationGraceMs must be finite and non-negative.");
  }
  if (typeof probePid !== "function") throw new Error("probePid must be a function.");

  const owner = {
    schemaVersion: 1,
    pid,
    token,
    acquiredAt: new Date(nowMs).toISOString(),
  };
  const recoveredStalePaths = [];

  for (let attempt = 0; attempt < 8; attempt += 1) {
    let handle;
    try {
      handle = await fs.open(target, "wx", 0o600);
    } catch (error) {
      if (error?.code !== "EEXIST") throw error;

      let raw;
      let stat;
      try {
        [raw, stat] = await Promise.all([fs.readFile(target, "utf8"), fs.stat(target)]);
      } catch (readError) {
        if (readError?.code === "ENOENT") continue;
        throw readError;
      }

      const currentOwner = parseRunnerPidRecord(raw);
      if (currentOwner && (await probePid(currentOwner.pid))) {
        throw runnerPidError(`Autoloop runner is already active with PID ${currentOwner.pid}.`, "EALREADY");
      }
      if (!currentOwner && raw.trim()) {
        throw runnerPidError("Autoloop runner ownership record is malformed.", "EBADMSG");
      }
      if (!currentOwner && Math.max(0, nowMs - stat.mtimeMs) < initializationGraceMs) {
        throw runnerPidError("Autoloop runner ownership is still initializing.", "EBUSY");
      }

      const stalePath = await quarantineStalePidFile(target, pid, attempt, nowMs);
      if (stalePath) recoveredStalePaths.push(stalePath);
      continue;
    }

    try {
      await handle.writeFile(`${JSON.stringify(owner)}\n`, "utf8");
      await handle.sync();
    } catch (error) {
      await handle.close().catch(() => {});
      handle = null;
      await fs.unlink(target).catch(() => {});
      throw error;
    } finally {
      await handle?.close().catch(() => {});
    }

    return { owner, recoveredStalePaths };
  }

  throw runnerPidError("Autoloop runner ownership changed repeatedly during acquisition.", "EBUSY");
}

export async function releaseRunnerPidFile({ filePath, pid = process.pid, token } = {}) {
  const target = String(filePath ?? "").trim();
  if (!target || !isPositiveSafeInteger(pid) || typeof token !== "string") return false;
  const raw = await fs.readFile(target, "utf8").catch((error) => {
    if (error?.code === "ENOENT") return "";
    throw error;
  });
  const owner = parseRunnerPidRecord(raw);
  if (!owner || owner.schemaVersion !== 1 || owner.pid !== pid || owner.token !== token) return false;
  await fs.unlink(target);
  return true;
}

export function maxCycleCountFromMetrics(raw) {
  let max = 0;
  for (const line of String(raw ?? "").split(/\r?\n/)) {
    if (!line.trim()) continue;
    try {
      const value = JSON.parse(line)?.cycleCount;
      if (isNonNegativeSafeInteger(value) && value > max) max = value;
    } catch {
      // Malformed historical metric lines do not erase valid later witnesses.
    }
  }
  return max;
}

export function parseCycleSequence(raw) {
  const text = String(raw ?? "").trim();
  if (!text) throw new Error("Cycle sequence file is empty.");
  let sequence;
  try {
    sequence = JSON.parse(text);
  } catch (error) {
    throw new Error(`Cycle sequence file is invalid JSON: ${error instanceof Error ? error.message : String(error)}`);
  }
  if (!sequence || typeof sequence !== "object" || Array.isArray(sequence)) {
    throw new Error("Cycle sequence must be an object.");
  }
  if (sequence.schemaVersion !== 1 || !isNonNegativeSafeInteger(sequence.lastIssued)) {
    throw new Error("Cycle sequence has an unsupported schema or invalid lastIssued value.");
  }
  if (typeof sequence.issuedAt !== "string" || !Number.isFinite(Date.parse(sequence.issuedAt))) {
    throw new Error("Cycle sequence issuedAt must be a valid timestamp.");
  }
  return {
    schemaVersion: 1,
    lastIssued: sequence.lastIssued,
    issuedAt: sequence.issuedAt,
  };
}

export function resolveResumedCycleCount({ metricsRaw = "", status = null, currentCycle = null, sequence = null } = {}) {
  const candidates = [maxCycleCountFromMetrics(metricsRaw)];
  if (isNonNegativeSafeInteger(status?.cycleCount)) candidates.push(status.cycleCount);
  const runIdMatch = String(currentCycle?.runId ?? "").match(/^cycle-([0-9]+)$/);
  if (runIdMatch) {
    const current = Number(runIdMatch[1]);
    if (isNonNegativeSafeInteger(current)) candidates.push(current);
  }
  if (sequence !== null) {
    if (
      sequence?.schemaVersion !== 1 ||
      !isNonNegativeSafeInteger(sequence?.lastIssued) ||
      typeof sequence?.issuedAt !== "string" ||
      !Number.isFinite(Date.parse(sequence.issuedAt))
    ) {
      throw new Error("Cycle sequence is invalid.");
    }
    candidates.push(sequence.lastIssued);
  }
  return Math.max(...candidates);
}

export function nextCycleSequence(lastIssued, issuedAt = new Date().toISOString()) {
  if (!isNonNegativeSafeInteger(lastIssued) || lastIssued >= Number.MAX_SAFE_INTEGER) {
    throw new Error("lastIssued must leave room for another safe cycle identifier.");
  }
  if (typeof issuedAt !== "string" || !Number.isFinite(Date.parse(issuedAt))) {
    throw new Error("issuedAt must be a valid timestamp.");
  }
  return {
    schemaVersion: 1,
    lastIssued: lastIssued + 1,
    issuedAt,
  };
}

export async function writeJsonFileAtomic(filePath, value) {
  const target = String(filePath ?? "").trim();
  if (!target) throw new Error("filePath must not be empty.");
  const dir = path.dirname(target);
  await fs.mkdir(dir, { recursive: true });
  const temp = path.join(
    dir,
    `.tmp-${process.pid}-${Date.now()}-${Math.random().toString(16).slice(2)}.json`,
  );
  await fs.writeFile(temp, `${JSON.stringify(value, null, 2)}\n`, "utf8");
  await fs.rename(temp, target);
}
