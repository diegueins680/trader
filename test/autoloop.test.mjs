import assert from "node:assert/strict";
import test from "node:test";
import {
  buildOpenAiApiError,
  clampText,
  extractResponseText,
  normalizeIdeaSelection,
  normalizePatchPlan,
  parseJsonResponse,
  sanitizeRelativePath,
  stripMarkdownFences,
  uniqueStrings,
} from "../scripts/autoloop-lib.mjs";

test("stripMarkdownFences unwraps fenced JSON", () => {
  assert.equal(stripMarkdownFences("```json\n{\"ok\":true}\n```"), "{\"ok\":true}");
  assert.equal(stripMarkdownFences("{\"ok\":true}"), "{\"ok\":true}");
});

test("extractResponseText concatenates output_text parts", () => {
  const response = {
    output: [
      {
        type: "message",
        content: [
          { type: "output_text", text: "one" },
          { type: "ignored", text: "skip" },
          { type: "output_text", text: "two" },
        ],
      },
    ],
  };
  assert.equal(extractResponseText(response), "one\ntwo");
});

test("parseJsonResponse rejects invalid JSON", () => {
  assert.throws(() => parseJsonResponse("not-json"), /invalid JSON/);
});

test("sanitizeRelativePath rejects absolute and traversal paths", () => {
  assert.equal(sanitizeRelativePath("./haskell/web/src/App.tsx"), "haskell/web/src/App.tsx");
  assert.throws(() => sanitizeRelativePath("./"), /resolves to empty/);
  assert.throws(() => sanitizeRelativePath("/tmp/nope"), /Absolute path/);
  assert.throws(() => sanitizeRelativePath("../nope"), /Path traversal/);
});

test("normalizeIdeaSelection validates required fields", () => {
  const idea = normalizeIdeaSelection({
    noChange: false,
    title: "Improve docs",
    rationale: "Tighten user guidance",
    filesNeeded: ["README.md", "CHANGELOG.md"],
    verificationCommands: ["cd haskell && cabal build"],
  });
  assert.deepEqual(idea.filesNeeded, ["README.md", "CHANGELOG.md"]);
  assert.throws(
    () =>
      normalizeIdeaSelection({
        noChange: false,
        title: "",
        rationale: "missing title",
        filesNeeded: ["README.md"],
      }),
    /title must not be empty/,
  );
});

test("normalizePatchPlan validates change entries", () => {
  const plan = normalizePatchPlan({
    noChange: false,
    title: "Patch docs",
    summary: "Explain setup",
    commitMessage: "Explain setup",
    changes: [{ path: "README.md", content: "# hi" }],
    verificationCommands: [],
  });
  assert.equal(plan.changes[0]?.path, "README.md");
  assert.throws(
    () =>
      normalizePatchPlan({
        noChange: false,
        title: "Bad patch",
        summary: "Bad patch",
        commitMessage: "Bad patch",
        changes: [{ path: "../oops", content: "x" }],
      }),
    /Path traversal/,
  );
  assert.throws(
    () =>
      normalizePatchPlan({
        noChange: false,
        title: "Duplicate patch",
        summary: "Duplicate patch",
        commitMessage: "Duplicate patch",
        changes: [
          { path: "README.md", content: "# one" },
          { path: "README.md", content: "# two" },
        ],
      }),
    /duplicate path/,
  );
});

test("clampText preserves short text and truncates long text", () => {
  assert.equal(clampText("short", 20), "short");
  assert.match(clampText("abcdefghijklmnopqrstuvwxyz", 12), /\[truncated/);
});

test("uniqueStrings preserves first occurrence order", () => {
  assert.deepEqual(uniqueStrings(["a", "b", "a", "c"]), ["a", "b", "c"]);
});

test("buildOpenAiApiError marks quota and auth failures as skippable", () => {
  const quotaErr = buildOpenAiApiError(429, {
    error: { code: "insufficient_quota", type: "insufficient_quota" },
  });
  const authErr = buildOpenAiApiError(401, {
    error: { code: "invalid_api_key", type: "invalid_request_error" },
  });
  const serverErr = buildOpenAiApiError(500, {
    error: { code: "server_error", type: "server_error" },
  });
  assert.equal(quotaErr.skipAutoloop, true);
  assert.equal(authErr.skipAutoloop, true);
  assert.equal(serverErr.skipAutoloop, false);
});
