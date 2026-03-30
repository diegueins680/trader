import fs from "node:fs/promises";
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

export function parseJsonResponse(raw) {
  const text = stripMarkdownFences(raw);
  if (!text) throw new Error("Model returned empty text.");
  try {
    return JSON.parse(text);
  } catch (err) {
    throw new Error(`Model returned invalid JSON: ${err instanceof Error ? err.message : String(err)}`);
  }
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

export function normalizeIdeaSelection(raw) {
  const obj = raw && typeof raw === "object" ? raw : {};
  const noChange = obj.noChange === true;
  const filesNeeded = noChange
    ? []
    : readStringArray(obj.filesNeeded, "filesNeeded", 10).map(sanitizeRelativePath);
  const algorithmReviewPath = noChange
    ? String(obj.algorithmReviewPath ?? "").trim()
    : readScopedPath(obj.algorithmReviewPath, "algorithmReviewPath", ALGORITHM_REVIEW_PREFIXES);
  const formalMethodsPath = noChange
    ? String(obj.formalMethodsPath ?? "").trim()
    : readScopedPath(obj.formalMethodsPath, "formalMethodsPath", FORMAL_METHODS_REVIEW_PREFIXES);
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

function normalizeFileChange(raw, idx) {
  if (!raw || typeof raw !== "object") throw new Error(`changes[${idx}] must be an object.`);
  const path = sanitizeRelativePath(raw.path);
  const deleteFile = raw.delete === true;
  const content = deleteFile ? "" : readString(raw.content ?? "", `changes[${idx}].content`);
  return {
    path,
    delete: deleteFile,
    content,
    reason: typeof raw.reason === "string" ? raw.reason.trim() : "",
  };
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
    title: noChange ? String(obj.title ?? "").trim() : readString(obj.title, "title"),
    summary: noChange ? String(obj.summary ?? "").trim() : readString(obj.summary, "summary"),
    commitMessage: noChange ? String(obj.commitMessage ?? "").trim() : readString(obj.commitMessage, "commitMessage"),
    algorithmReviewSummary: noChange
      ? String(obj.algorithmReviewSummary ?? "").trim()
      : readString(obj.algorithmReviewSummary, "algorithmReviewSummary"),
    formalMethodsSummary: noChange
      ? String(obj.formalMethodsSummary ?? "").trim()
      : readString(obj.formalMethodsSummary, "formalMethodsSummary"),
    changes,
    verificationCommands: Array.isArray(obj.verificationCommands)
      ? obj.verificationCommands.map((item, idx) => readString(item, `verificationCommands[${idx}]`))
      : [],
  };
}

export function uniqueStrings(values) {
  return Array.from(new Set(values.map((value) => String(value))));
}

export function prepareShellCommand(command) {
  const normalized = String(command ?? "").trim();
  const needsGhcup =
    /\bcabal\b/.test(normalized) || normalized === "cd haskell && bash scripts/ci_smoke.sh";
  if (!normalized || !needsGhcup || normalized.includes(".ghcup/env")) return normalized;
  return `source "$HOME/.ghcup/env" 2>/dev/null || true; ${normalized}`;
}

export function resolveAutoloopBackend(rawBackend, { hasOpenAiKey, hasCodex }) {
  const requested = String(rawBackend ?? "").trim().toLowerCase();
  if (!requested || requested === "auto") {
    if (hasOpenAiKey) return "openai";
    if (hasCodex) return "codex";
    return "";
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
