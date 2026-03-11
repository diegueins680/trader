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
  const keep = Math.max(0, limit - 32);
  return `${text.slice(0, keep)}\n...[truncated ${text.length - keep} chars]`;
}

export function sanitizeRelativePath(raw) {
  const value = String(raw ?? "").trim().replace(/\\/g, "/");
  if (!value) throw new Error("Path is empty.");
  if (value.startsWith("/")) throw new Error(`Absolute path is not allowed: ${value}`);
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

export function normalizeIdeaSelection(raw) {
  const obj = raw && typeof raw === "object" ? raw : {};
  const noChange = obj.noChange === true;
  return {
    noChange,
    title: noChange ? String(obj.title ?? "").trim() : readString(obj.title, "title"),
    rationale: noChange ? String(obj.rationale ?? "").trim() : readString(obj.rationale, "rationale"),
    filesNeeded: noChange ? [] : readStringArray(obj.filesNeeded, "filesNeeded", 8).map(sanitizeRelativePath),
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
  if (changes.length > 8) throw new Error("changes exceeds max items (8).");
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
    changes,
    verificationCommands: Array.isArray(obj.verificationCommands)
      ? obj.verificationCommands.map((item, idx) => readString(item, `verificationCommands[${idx}]`))
      : [],
  };
}

export function uniqueStrings(values) {
  return Array.from(new Set(values.map((value) => String(value))));
}

export function buildOpenAiApiError(status, payload) {
  const errorObj = payload?.error && typeof payload.error === "object" ? payload.error : {};
  const code = typeof errorObj.code === "string" ? errorObj.code : "";
  const type = typeof errorObj.type === "string" ? errorObj.type : "";
  const err = new Error(`OpenAI API request failed (${status}): ${JSON.stringify(payload)}`);
  err.openAiStatus = Number(status) || 0;
  err.openAiCode = code;
  err.openAiType = type;
  err.skipAutoloop =
    code === "insufficient_quota" ||
    code === "invalid_api_key" ||
    type === "insufficient_quota" ||
    err.openAiStatus === 401;
  return err;
}
