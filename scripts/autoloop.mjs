#!/usr/bin/env node

import fs from "node:fs/promises";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { execFileSync } from "node:child_process";
import {
  buildOpenAiApiError,
  clampText,
  extractResponseText,
  normalizeIdeaSelection,
  normalizePatchPlan,
  sanitizeRelativePath,
  parseJsonResponse,
  uniqueStrings,
} from "./autoloop-lib.mjs";

const ROOT = process.cwd();
const OPENAI_BASE_URL = (process.env.OPENAI_BASE_URL || "https://api.openai.com/v1").replace(/\/$/, "");
const OPENAI_MODEL = process.env.AUTOLOOP_MODEL || process.env.OPENAI_MODEL || "gpt-5.4";
const BASE_BRANCH =
  process.env.AUTOLOOP_BASE_BRANCH || process.env.GITHUB_BASE_REF || process.env.GITHUB_REF_NAME || "main";
const LOOP_BRANCH = process.env.AUTOLOOP_BRANCH || `autoloop/${BASE_BRANCH}`;
const MAX_ITERATIONS = clampInt(process.env.AUTOLOOP_MAX_ITERATIONS, 2, 1, 5);
const MAX_EDITABLE_FILE_BYTES = clampInt(process.env.AUTOLOOP_MAX_FILE_BYTES, 40000, 4000, 100000);
const MAX_EDITABLE_FILES = clampInt(process.env.AUTOLOOP_MAX_FILES, 120, 20, 300);
const DRY_RUN = process.argv.includes("--dry-run");

const ALLOWED_EDIT_PREFIXES = [
  "README.md",
  "CHANGELOG.md",
  "FORMAL_METHODS.md",
  "haskell/app/",
  "haskell/test/",
  "haskell/web/src/",
  "haskell/web/test/",
  "haskell/scripts/",
];

const BLOCKED_EDIT_PREFIXES = [
  ".env",
  ".github/",
  ".git/",
  "deploy/",
  "dist/",
  "haskell/web/dist/",
  "node_modules/",
  "scripts/autoloop",
  "package-lock.json",
  "haskell/web/package-lock.json",
  "package.json",
  "haskell/web/package.json",
  "haskell/trader.cabal",
];

const SAFE_VERIFICATION_COMMANDS = new Set([
  "cd haskell && cabal build",
  "cd haskell && cabal test",
  "cd haskell && bash scripts/ci_smoke.sh",
  "cd haskell/web && npm --workspaces=false run typecheck",
  "cd haskell/web && npm --workspaces=false run test",
  "cd haskell/web && npm --workspaces=false run build",
  "node --test test/autoloop.test.mjs",
]);

async function main() {
  if (!process.env.OPENAI_API_KEY) {
    const message = "OPENAI_API_KEY is not set; autoloop is skipping.";
    if (DRY_RUN) throw new Error(message);
    console.log(message);
    return;
  }

  assertCleanWorktree();
  runGit(["fetch", "origin", BASE_BRANCH, "--prune"]);
  await checkoutLoopBranch();

  let failureContext = null;
  for (let iteration = 1; iteration <= MAX_ITERATIONS; iteration += 1) {
    hardResetToCurrentHead();
    const repoContext = await buildRepoContext();
    const idea = failureContext
      ? await requestFixIdea(repoContext, failureContext)
      : await requestIdeaSelection(repoContext);

    if (idea.noChange) {
      console.log(`No safe change proposed${idea.rationale ? `: ${idea.rationale}` : "."}`);
      return;
    }

    const editableFiles = await readEditableFiles(idea.filesNeeded);
    const plan = await requestPatchPlan(repoContext, idea, editableFiles, failureContext);
    if (plan.noChange) {
      console.log(`No patch plan returned${plan.summary ? `: ${plan.summary}` : "."}`);
      return;
    }

    assertPlanMatchesEditableFiles(plan.changes, editableFiles);
    const plannedPaths = uniqueStrings(plan.changes.map((change) => sanitizeRelativePath(change.path)));
    applyFileChanges(plan.changes);
    let changedPaths = collectChangedPlanPaths(plannedPaths);
    if (changedPaths.length === 0) {
      console.log("Model produced no file changes after normalization; stopping.");
      return;
    }

    const verificationCommands = planVerificationCommands(changedPaths, [
      ...idea.verificationCommands,
      ...plan.verificationCommands,
    ]);
    await runVerificationCommands(verificationCommands);
    changedPaths = collectChangedPlanPaths(plannedPaths);
    if (changedPaths.length === 0) {
      console.log("Verification returned the worktree to baseline; stopping.");
      return;
    }
    const unexpectedChanges = collectUnexpectedWorktreeChanges(plannedPaths);
    if (unexpectedChanges.length > 0) {
      console.warn(`Skipping unexpected worktree paths outside the plan: ${unexpectedChanges.join(", ")}`);
    }

    const commitMessage = plan.commitMessage;
    if (DRY_RUN) {
      console.log(JSON.stringify({ commitMessage, changedPaths, verificationCommands }, null, 2));
      return;
    }

    commitBranch(commitMessage, changedPaths);
    pushBranch();

    const pr = ensurePullRequest(plan.title, plan.summary);
    const ci = waitForPullRequestCi(pr.number);
    if (ci.ok) {
      mergePullRequest(pr.number);
      console.log(`Merged PR #${pr.number}.`);
      return;
    }

    failureContext = {
      iteration,
      prNumber: pr.number,
      runId: ci.runId,
      runUrl: ci.runUrl,
      failedLog: ci.failedLog,
      changedPaths,
    };
  }

  throw new Error(`Autoloop exhausted ${MAX_ITERATIONS} iteration(s) without a green merge.`);
}

function clampInt(raw, fallback, min, max) {
  const n = Number(raw);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, Math.trunc(n)));
}

function runCommand(command, args, opts = {}) {
  const capture = opts.capture !== false;
  const cwd = opts.cwd || ROOT;
  const env = { ...process.env, ...(opts.env || {}) };
  try {
    const out = execFileSync(command, args, {
      cwd,
      env,
      encoding: "utf8",
      stdio: capture ? ["ignore", "pipe", "pipe"] : "inherit",
    });
    return capture ? out.trim() : "";
  } catch (err) {
    const stdout = err?.stdout ? String(err.stdout) : "";
    const stderr = err?.stderr ? String(err.stderr) : "";
    throw new Error(`${command} ${args.join(" ")} failed.\n${stdout}${stderr}`.trim());
  }
}

function runGit(args, opts) {
  return runCommand("git", args, opts);
}

function runGh(args, opts = {}) {
  return runCommand("gh", args, {
    ...opts,
    env: {
      ...(opts.env || {}),
      GH_TOKEN: process.env.GITHUB_TOKEN || process.env.GH_TOKEN || "",
    },
  });
}

function runBash(command, opts = {}) {
  return runCommand("/bin/bash", ["-lc", command], opts);
}

function assertCleanWorktree() {
  const status = runGit(["status", "--porcelain"]);
  if (status) throw new Error(`Worktree must be clean before autoloop starts.\n${status}`);
}

async function checkoutLoopBranch() {
  const remoteBranch = runGit(["ls-remote", "--heads", "origin", LOOP_BRANCH]);
  if (remoteBranch.trim()) {
    runGit(["checkout", "-B", LOOP_BRANCH, `origin/${LOOP_BRANCH}`], { capture: false });
  } else {
    runGit(["checkout", "-B", LOOP_BRANCH, `origin/${BASE_BRANCH}`], { capture: false });
  }
}

function hardResetToCurrentHead() {
  runGit(["reset", "--hard", "HEAD"], { capture: false });
  runGit(["clean", "-fd", "-e", "node_modules/", "-e", "haskell/web/node_modules/"], { capture: false });
}

function allowedEditPath(filePath) {
  const rel = sanitizeRelativePath(filePath);
  if (BLOCKED_EDIT_PREFIXES.some((prefix) => rel === prefix || rel.startsWith(prefix))) return false;
  return ALLOWED_EDIT_PREFIXES.some((prefix) => rel === prefix || rel.startsWith(prefix));
}

async function listEditableFiles() {
  const files = runGit(["ls-files"]).split(/\r?\n/).filter(Boolean);
  const result = [];
  for (const rel of files) {
    if (!allowedEditPath(rel)) continue;
    const abs = path.join(ROOT, rel);
    const stat = await fs.stat(abs);
    if (!stat.isFile()) continue;
    if (stat.size > MAX_EDITABLE_FILE_BYTES) continue;
    result.push({ path: rel, size: stat.size });
    if (result.length >= MAX_EDITABLE_FILES) break;
  }
  return result;
}

async function buildRepoContext() {
  const editableFiles = await listEditableFiles();
  const agents = await fs.readFile(path.join(ROOT, "AGENTS.md"), "utf8");
  const ciWorkflow = await fs.readFile(path.join(ROOT, ".github/workflows/ci.yml"), "utf8");
  const readme = clampText(await fs.readFile(path.join(ROOT, "README.md"), "utf8"), 14000);
  const changelog = clampText(await fs.readFile(path.join(ROOT, "CHANGELOG.md"), "utf8"), 10000);
  const formal = clampText(await fs.readFile(path.join(ROOT, "FORMAL_METHODS.md"), "utf8"), 8000);
  const packageJson = await fs.readFile(path.join(ROOT, "package.json"), "utf8");
  const webPackageJson = await fs.readFile(path.join(ROOT, "haskell/web/package.json"), "utf8");
  const editableList = editableFiles.map((file) => `${file.path} (${file.size} bytes)`).join("\n");
  const recentCommits = runGit(["log", "--oneline", "-5"]);

  return {
    agents,
    ciWorkflow,
    readme,
    changelog,
    formal,
    packageJson,
    webPackageJson,
    editableList,
    recentCommits,
  };
}

function repoContextText(repoContext) {
  return [
    "AGENTS.md:",
    repoContext.agents,
    "",
    "Recent commits:",
    repoContext.recentCommits,
    "",
    "CI workflow:",
    repoContext.ciWorkflow,
    "",
    "Root package.json:",
    repoContext.packageJson,
    "",
    "Web package.json:",
    repoContext.webPackageJson,
    "",
    "README excerpt:",
    repoContext.readme,
    "",
    "CHANGELOG excerpt:",
    repoContext.changelog,
    "",
    "FORMAL_METHODS excerpt:",
    repoContext.formal,
    "",
    `Editable files (limited to ${MAX_EDITABLE_FILE_BYTES} bytes each):`,
    repoContext.editableList,
  ].join("\n");
}

async function callModelJson({ prompt, maxOutputTokens = 4000 }) {
  const response = await fetch(`${OPENAI_BASE_URL}/responses`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
    },
    body: JSON.stringify({
      model: OPENAI_MODEL,
      instructions:
        "Return JSON only. The word JSON must appear in the output context, and the final output must be a single valid JSON object with no markdown fences.",
      input: prompt,
      text: { format: { type: "json_object" } },
      temperature: 0.2,
      max_output_tokens: maxOutputTokens,
      store: false,
    }),
  });

  const json = await response.json();
  if (!response.ok) {
    throw buildOpenAiApiError(response.status, json);
  }
  return parseJsonResponse(extractResponseText(json));
}

async function requestIdeaSelection(repoContext) {
  const prompt = [
    "You are selecting exactly one safe autonomous improvement for this repository.",
    "Respond in JSON with keys: noChange, title, rationale, filesNeeded, verificationCommands.",
    "Constraints:",
    "- Choose one small, high-confidence change with measurable value.",
    "- Touch only files from the editable file list.",
    "- Do not propose dependency updates, workflow changes, deploy changes, or secrets.",
    "- Prefer changes that can be verified by local tests/build commands.",
    "- If the change is user-visible, include README.md and CHANGELOG.md in filesNeeded.",
    `- Verification commands must be chosen from this allowlist: ${Array.from(SAFE_VERIFICATION_COMMANDS).join(" | ")}`,
    "- If no safe change is apparent, return {\"noChange\":true,...}.",
    "",
    repoContextText(repoContext),
  ].join("\n");

  return normalizeIdeaSelection(await callModelJson({ prompt, maxOutputTokens: 2000 }));
}

async function requestFixIdea(repoContext, failureContext) {
  const prompt = [
    "You are selecting a repair for a failed autonomous PR CI run.",
    "Respond in JSON with keys: noChange, title, rationale, filesNeeded, verificationCommands.",
    "Constraints:",
    "- Focus on fixing the reported failure with the smallest safe change.",
    "- Touch only files from the editable file list.",
    `- Verification commands must be chosen from this allowlist: ${Array.from(SAFE_VERIFICATION_COMMANDS).join(" | ")}`,
    "",
    `Failed run: ${failureContext.runUrl}`,
    `Changed paths on the current branch: ${failureContext.changedPaths.join(", ")}`,
    "Failed log excerpt:",
    clampText(failureContext.failedLog, 20000),
    "",
    repoContextText(repoContext),
  ].join("\n");

  return normalizeIdeaSelection(await callModelJson({ prompt, maxOutputTokens: 2200 }));
}

async function readEditableFiles(paths) {
  const out = [];
  for (const rawPath of paths) {
    const rel = sanitizeRelativePath(rawPath);
    if (!allowedEditPath(rel)) throw new Error(`Path is outside the autoloop allowlist: ${rel}`);
    const abs = path.join(ROOT, rel);
    const content = await fs.readFile(abs, "utf8");
    out.push({ path: rel, content });
  }
  return out;
}

async function requestPatchPlan(repoContext, idea, editableFiles, failureContext) {
  const fileSections = editableFiles
    .map((file) => `FILE: ${file.path}\n<<<FILE\n${file.content}\nFILE;`)
    .join("\n\n");

  const prompt = [
    "You are implementing a single repository change.",
    "Respond in JSON with keys: noChange, title, summary, commitMessage, verificationCommands, changes.",
    "Each entry in changes must be an object with path, content, and optional reason.",
    "The content field must contain the complete replacement file content for that path.",
    "Do not include markdown fences or prose outside JSON.",
    "Constraints:",
    "- Only modify the provided files.",
    "- Preserve unrelated content.",
    "- Keep the change minimal and focused.",
    "- Use ASCII unless the file already requires Unicode.",
    `- Verification commands must be chosen from this allowlist: ${Array.from(SAFE_VERIFICATION_COMMANDS).join(" | ")}`,
    idea.title ? `Selected idea: ${idea.title}` : "",
    idea.rationale ? `Rationale: ${idea.rationale}` : "",
    failureContext ? `Failed CI log excerpt:\n${clampText(failureContext.failedLog, 18000)}` : "",
    "",
    "Repository context:",
    clampText(repoContextText(repoContext), 30000),
    "",
    "Editable file contents:",
    fileSections,
  ]
    .filter(Boolean)
    .join("\n");

  return normalizePatchPlan(await callModelJson({ prompt, maxOutputTokens: 12000 }));
}

function applyFileChanges(changes) {
  for (const change of changes) {
    if (!allowedEditPath(change.path)) throw new Error(`Autoloop cannot edit ${change.path}`);
    const abs = path.join(ROOT, change.path);
    if (change.delete) throw new Error(`Autoloop does not allow deletes: ${change.path}`);
    mkdirSync(path.dirname(abs), { recursive: true });
    writeFileSync(abs, change.content, "utf8");
  }
}

function assertPlanMatchesEditableFiles(changes, editableFiles) {
  const allowed = new Set(editableFiles.map((file) => sanitizeRelativePath(file.path)));
  const unexpected = uniqueStrings(
    changes
      .map((change) => sanitizeRelativePath(change.path))
      .filter((filePath) => !allowed.has(filePath)),
  );
  if (unexpected.length > 0) {
    throw new Error(`Patch plan touched files outside the requested context: ${unexpected.join(", ")}`);
  }
}

function listTrackedWorktreeChanges() {
  return uniqueStrings(runGit(["diff", "--name-only"]).split(/\r?\n/).filter(Boolean).map(sanitizeRelativePath));
}

function listUntrackedWorktreeChanges() {
  return uniqueStrings(
    runGit(["ls-files", "--others", "--exclude-standard"])
      .split(/\r?\n/)
      .filter(Boolean)
      .map(sanitizeRelativePath),
  );
}

function collectChangedPlanPaths(plannedPaths) {
  const planned = new Set(plannedPaths.map(sanitizeRelativePath));
  return uniqueStrings([...listTrackedWorktreeChanges(), ...listUntrackedWorktreeChanges()]).filter((filePath) =>
    planned.has(filePath),
  );
}

function collectUnexpectedWorktreeChanges(plannedPaths) {
  const planned = new Set(plannedPaths.map(sanitizeRelativePath));
  return uniqueStrings([...listTrackedWorktreeChanges(), ...listUntrackedWorktreeChanges()]).filter(
    (filePath) => !planned.has(filePath),
  );
}

function planVerificationCommands(changedPaths, suggestedCommands) {
  const planned = new Set();
  const needsWeb = changedPaths.some((file) => file.startsWith("haskell/web/"));
  const needsHsBuild = changedPaths.some((file) => file.startsWith("haskell/app/") || file.startsWith("haskell/test/") || file.startsWith("haskell/scripts/"));
  const needsHsTest = changedPaths.some((file) => file.startsWith("haskell/app/") || file.startsWith("haskell/test/"));

  if (needsWeb) {
    planned.add("cd haskell/web && npm --workspaces=false run typecheck");
    planned.add("cd haskell/web && npm --workspaces=false run test");
    planned.add("cd haskell/web && npm --workspaces=false run build");
  }
  if (needsHsBuild) planned.add("cd haskell && cabal build");
  if (needsHsTest) planned.add("cd haskell && cabal test");

  for (const command of suggestedCommands) {
    const normalized = String(command || "").trim();
    if (SAFE_VERIFICATION_COMMANDS.has(normalized)) planned.add(normalized);
  }
  return Array.from(planned);
}

async function runVerificationCommands(commands) {
  for (const command of commands) {
    console.log(`Running verification: ${command}`);
    runBash(command, { capture: false });
  }
}

function commitBranch(message, changedPaths) {
  runGit(["config", "user.name", "autoloop[bot]"], { capture: false });
  runGit(["config", "user.email", "autoloop[bot]@users.noreply.github.com"], { capture: false });
  runGit(["add", "--", ...changedPaths], { capture: false });
  runGit(["commit", "-m", message], { capture: false });
}

function pushBranch() {
  runGit(["push", "--force-with-lease", "-u", "origin", LOOP_BRANCH], { capture: false });
}

function ensurePullRequest(title, summary) {
  const listJson = runGh([
    "pr",
    "list",
    "--head",
    LOOP_BRANCH,
    "--base",
    BASE_BRANCH,
    "--state",
    "open",
    "--json",
    "number,url,title",
  ]);
  const existing = JSON.parse(listJson);
  const body = [
    "Autonomous improvement loop.",
    "",
    `Summary: ${summary}`,
    "",
    `Branch: \`${LOOP_BRANCH}\``,
    `Base: \`${BASE_BRANCH}\``,
  ].join("\n");

  if (existing.length > 0) {
    const pr = existing[0];
    runGh(["pr", "edit", String(pr.number), "--title", title, "--body", body], { capture: false });
    return { number: pr.number, url: pr.url };
  }

  const out = runGh([
    "pr",
    "create",
    "--base",
    BASE_BRANCH,
    "--head",
    LOOP_BRANCH,
    "--title",
    title,
    "--body",
    body,
  ]);
  const prNumber = Number(out.match(/\/pull\/(\d+)/)?.[1]);
  if (!Number.isFinite(prNumber)) throw new Error(`Unable to parse PR number from: ${out}`);
  return { number: prNumber, url: out.trim() };
}

function waitForPullRequestCi(prNumber) {
  const headSha = runGit(["rev-parse", "HEAD"]);
  let runId = null;
  let runUrl = null;

  for (let i = 0; i < 12; i += 1) {
    const runs = JSON.parse(
      runGh([
        "run",
        "list",
        "--branch",
        LOOP_BRANCH,
        "--event",
        "pull_request",
        "--limit",
        "10",
        "--json",
        "databaseId,headSha,status,conclusion,url,workflowName",
      ]),
    );
    const match = runs.find((run) => run.headSha === headSha && run.workflowName === "CI");
    if (match) {
      runId = match.databaseId;
      runUrl = match.url;
      break;
    }
    runCommand("sleep", ["10"], { capture: false });
  }

  if (!runId) throw new Error(`No CI run found for PR #${prNumber} and head ${headSha}.`);

  try {
    runGh(["run", "watch", String(runId), "--interval", "30", "--exit-status"], { capture: false });
    return { ok: true, runId, runUrl };
  } catch {
    const failedLog = runGh(["run", "view", String(runId), "--log-failed"]);
    return { ok: false, runId, runUrl, failedLog };
  }
}

function mergePullRequest(prNumber) {
  runGh(["pr", "merge", String(prNumber), "--squash", "--delete-branch"], { capture: false });
}

main().catch((err) => {
  if (err?.skipAutoloop) {
    const message = err instanceof Error ? err.message : String(err);
    console.warn(`Autoloop skipped: ${message}`);
    return;
  }
  const message = err instanceof Error ? err.stack || err.message : String(err);
  console.error(message);
  process.exitCode = 1;
});
