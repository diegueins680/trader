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
  if (value.startsWith("/")) throw new Error(`Absolute path is not allowed: ${value}`);
  if (/^[A-Za-z]:\//.test(value)) throw new Error(`Absolute path is not allowed: ${value}`);
  if (value.split("/").some((part) => part === "..")) throw new Error(`Path traversal is not allowed: ${value}`);
  if (value.includes("\0")) throw new Error(`Path contains NUL byte: ${value}`);
  const normalized = value.replace(/^\.\/+/, "");
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

function readScopedPath(raw, field, allowedPrefixes) {
  const value = sanitizeRelativePath(readString(raw, field));
  if (!allowedPrefixes.some((prefix) => value === prefix || value.startsWith(prefix))) {
    throw new Error(`${field} must be within: ${allowedPrefixes.join(", ")}`);
  }
  return value;
}

export function normalizeIdeaSelection(raw) {
  const obj = raw && typeof raw === "object" ? raw : {};
  const noChange = obj.noChange === true;
  const filesNeeded = noChange
    ? []
    : readStringArray(obj.filesNeeded, "filesNeeded", 10).map(sanitizeRelativePath);
  const uiReviewPath = noChange
    ? String(obj.uiReviewPath ?? "").trim()
    : readScopedPath(obj.uiReviewPath, "uiReviewPath", ["haskell/web/src/"]);
  const correctnessPath = noChange
    ? String(obj.correctnessPath ?? "").trim()
    : readScopedPath(obj.correctnessPath, "correctnessPath", [
        "FORMAL_METHODS.md",
        "test/",
        "haskell/test/",
        "haskell/web/test/",
      ]);
  if (!noChange) {
    if (!filesNeeded.includes(uiReviewPath)) throw new Error("filesNeeded must include uiReviewPath.");
    if (!filesNeeded.includes(correctnessPath)) throw new Error("filesNeeded must include correctnessPath.");
  }
  return {
    noChange,
    title: noChange ? String(obj.title ?? "").trim() : readString(obj.title, "title"),
    rationale: noChange ? String(obj.rationale ?? "").trim() : readString(obj.rationale, "rationale"),
    uiReviewPath,
    uiReviewFocus: noChange ? String(obj.uiReviewFocus ?? "").trim() : readString(obj.uiReviewFocus, "uiReviewFocus"),
    correctnessPath,
    correctnessFocus: noChange
      ? String(obj.correctnessFocus ?? "").trim()
      : readString(obj.correctnessFocus, "correctnessFocus"),
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
    uiReviewSummary: noChange
      ? String(obj.uiReviewSummary ?? "").trim()
      : readString(obj.uiReviewSummary, "uiReviewSummary"),
    correctnessSummary: noChange
      ? String(obj.correctnessSummary ?? "").trim()
      : readString(obj.correctnessSummary, "correctnessSummary"),
    changes,
    verificationCommands: Array.isArray(obj.verificationCommands)
      ? obj.verificationCommands.map((item, idx) => readString(item, `verificationCommands[${idx}]`))
      : [],
  };
}

export function uniqueStrings(values) {
  return Array.from(new Set(values.map((value) => String(value))));
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
