import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, symlinkSync, truncateSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import test from "node:test";

const lockPath = "formal/research/toolchain.json";
const capabilityPath = "formal/research/capability-source-contract.json";
const sourcePath = "haskell/app/Main.hs";
const hash = (value) => createHash("sha256").update(value).digest("hex");

function fixture(t) {
  const root = mkdtempSync(path.join(tmpdir(), "trader-source-lock-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  mkdirSync(path.join(root, "formal/research"), { recursive: true });
  mkdirSync(path.join(root, "haskell/app"), { recursive: true });
  writeFileSync(path.join(root, sourcePath), "module Main where\nmain = pure ()\n");
  writeFileSync(path.join(root, capabilityPath), JSON.stringify({ schemaVersion: 1, sourceInventory: [sourcePath] }));
  const lock = {
    schemaVersion: 1,
    sourceHashes: Object.fromEntries([sourcePath, capabilityPath].map((file) => [file, hash(readFileSync(path.join(root, file)))])),
  };
  const save = () => writeFileSync(path.join(root, lockPath), JSON.stringify(lock));
  save();
  return { root, lock, save };
}

function check(root) {
  return spawnSync("python3", ["scripts/check-formal-source-lock.py", "--root", root], {
    encoding: "utf8",
    timeout: 30_000,
  });
}

test("source preflight accepts matching pins and detects source drift without importing the solver", (t) => {
  const { root } = fixture(t);
  const valid = check(root);
  assert.equal(valid.status, 0, valid.stderr);
  assert.deepEqual(JSON.parse(valid.stdout), { valid: true, scope: "source-lock-and-inventory", pinnedFiles: 2, applicationModules: 1 });
  writeFileSync(path.join(root, sourcePath), "changed source");
  const drift = check(root);
  assert.equal(drift.status, 1);
  assert.match(drift.stdout, /source hash mismatch: haskell\/app\/Main.hs/);
});

test("source preflight detects newly added modules and missing inventory pins", (t) => {
  const { root, lock, save } = fixture(t);
  writeFileSync(path.join(root, "haskell/app/Added.hs"), "module Added where\n");
  assert.match(check(root).stdout, /module inventory mismatch/);
  delete lock.sourceHashes[sourcePath];
  save();
  assert.equal(check(root).status, 1);
  assert.match(check(root).stdout, /module has no source pin/);
});

test("source preflight catches stale reviewed bindings even when enclosing and global hashes are refreshed", (t) => {
  const { root, lock, save } = fixture(t);
  const capability = { schemaVersion: 1, sourceInventory: [sourcePath], supportSourceHashes: { [sourcePath]: lock.sourceHashes[sourcePath] } };
  writeFileSync(path.join(root, sourcePath), "module Main where\nmain = print ()\n");
  writeFileSync(path.join(root, capabilityPath), JSON.stringify(capability));
  for (const file of [sourcePath, capabilityPath]) lock.sourceHashes[file] = hash(readFileSync(path.join(root, file)));
  save();
  const stale = check(root);
  assert.equal(stale.status, 1);
  assert.match(stale.stdout, /reviewed source binding mismatch: haskell\/app\/Main.hs/);
});

test("source preflight refuses traversal, absolute paths, bad hashes and malformed pin maps", (t) => {
  const { root, lock, save } = fixture(t);
  const original = structuredClone(lock.sourceHashes);
  for (const key of ["../outside", "/etc/passwd", "haskell//app/Main.hs", "./haskell/app/Main.hs", "C:\\outside"]) {
    lock.sourceHashes = { ...original, [key]: "0".repeat(64) };
    save();
    const result = check(root);
    assert.equal(result.status, 1);
    assert.match(result.stdout, /unsafe source path/);
  }
  for (const pins of [null, [], {}, { ...original, [sourcePath]: true }, { ...original, [sourcePath]: "invalid" }]) {
    lock.sourceHashes = pins;
    save();
    assert.equal(check(root).status, 1);
  }
});

test("source preflight rejects duplicate JSON metadata and unsupported schemas", (t) => {
  const { root, lock, save } = fixture(t);
  writeFileSync(path.join(root, lockPath), '{"schemaVersion":1,"schemaVersion":1,"sourceHashes":{}}');
  assert.match(check(root).stdout, /duplicate metadata key/);
  for (const schema of [true, "1", 2]) {
    lock.schemaVersion = schema;
    save();
    assert.equal(check(root).status, 1);
    assert.match(check(root).stdout, /unsupported metadata schema/);
  }
});

test("source preflight rejects symlinked source files and directories", (t) => {
  const { root, lock, save } = fixture(t);
  symlinkSync(path.join(root, sourcePath), path.join(root, "haskell/app/Alias.hs"));
  lock.sourceHashes["haskell/app/Alias.hs"] = lock.sourceHashes[sourcePath];
  save();
  assert.match(check(root).stdout, /symlinked source path/);
  delete lock.sourceHashes["haskell/app/Alias.hs"];
  symlinkSync(path.join(root, "haskell/app"), path.join(root, "alias"), "dir");
  lock.sourceHashes["alias/Main.hs"] = lock.sourceHashes[sourcePath];
  save();
  assert.match(check(root).stdout, /symlinked source path/);
});

test("source preflight bounds metadata, pin counts and source reads", (t) => {
  const { root, lock, save } = fixture(t);
  writeFileSync(path.join(root, lockPath), " ".repeat(2 * 1024 * 1024 + 1));
  assert.equal(check(root).status, 1);
  assert.match(check(root).stdout, /metadata size limit/);
  const original = structuredClone(lock.sourceHashes);
  for (let i = 0; i < 4096; i++) lock.sourceHashes[`extra-${i}`] = "0".repeat(64);
  save();
  assert.equal(check(root).status, 1);
  assert.match(check(root).stdout, /invalid source pins/);
  lock.sourceHashes = original;
  save();
  truncateSync(path.join(root, sourcePath), 64 * 1024 * 1024 + 1);
  assert.equal(check(root).status, 1);
  assert.match(check(root).stdout, /source size limit/);
});
