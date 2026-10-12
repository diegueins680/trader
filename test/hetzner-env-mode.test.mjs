import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const scriptPath = path.join(root, "deploy/hetzner/deploy-remote.sh");
const script = await fs.readFile(scriptPath, "utf8");
const contract = script.split("# BEGIN environment contract")[1]?.split("\n").slice(1).join("\n")
  .split("# END environment contract.")[0];
assert.ok(contract?.includes("assert_environment_preserved()"));

async function fixture(t) {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), "trader-deploy-env-"));
  t.after(() => fs.rm(directory, { recursive: true, force: true }));
  const envFile = path.join(directory, "runtime.env");
  const original = "# Synthetic fixture only\nTRADER_BINANCE_LIVE=false\nTRADER_BOT_TRADE=false\nSETTING=old\nUNMANAGED=retained\n";
  await fs.writeFile(envFile, original, { mode: 0o600 });
  const run = (mode, managed = "", overrides = "", suffix = "assert_environment_preserved") =>
    spawnSync("bash", ["-c", `set -euo pipefail\n${contract}\n${suffix}`], {
      encoding: "utf8", env: { PATH: process.env.PATH, ENV_MODE: mode, ENV_FILE: envFile,
        MANAGED_ENV_FILE: managed, ENV_OVERRIDES_FILE: overrides },
    });
  return { directory, envFile, original, run };
}

test("preserve deploy contract leaves runtime bytes and permissions unchanged", async (t) => {
  const f = await fixture(t);
  const result = f.run("preserve");
  assert.equal(result.status, 0, result.stderr);
  assert.equal(await fs.readFile(f.envFile, "utf8"), f.original);
  assert.equal((await fs.stat(f.envFile)).mode & 0o777, 0o600);
  assert.doesNotMatch(result.stdout + result.stderr, /TRADER_BINANCE_LIVE|TRADER_BOT_TRADE|SETTING=/);
  assert.ok(script.indexOf('--exclude "/${env_file#${repo_dir%/}/}"') < script.indexOf("--include '*.env.managed'"));
});

test("preserve rejects overlays and invalid modes without touching runtime bytes", async (t) => {
  const f = await fixture(t);
  const overlay = path.join(f.directory, "overlay");
  await fs.writeFile(overlay, "SETTING=new\n");
  for (const args of [["preserve", overlay], ["preserve", "", overlay], ["typo"]]) {
    assert.notEqual(f.run(...args).status, 0);
    assert.equal(await fs.readFile(f.envFile, "utf8"), f.original);
  }
  for (const env of [{ TRADER_HETZNER_ENV_MODE: "typo" }, { TRADER_HETZNER_MANAGED_ENV_FILE: overlay }]) {
    const result = spawnSync("bash", [scriptPath, "unused.invalid"], {
      encoding: "utf8", env: { PATH: process.env.PATH, TRADER_GIT_COMMIT: "abcdef1", ...env },
    });
    assert.equal(result.status, 1);
    assert.match(result.stderr, /environment mode|preserve mode refuses/);
    assert.equal(result.stdout, ""); // Rejected before rsync/SSH or frontend checks.
  }
});

test("rsync protects the configured env path even with a non-env extension", async (t) => {
  const f = await fixture(t);
  const source = path.join(f.directory, "source");
  const destination = path.join(f.directory, "destination");
  await fs.mkdir(source); await fs.mkdir(destination);
  await fs.writeFile(path.join(source, "settings.conf"), "replacement\n");
  await fs.writeFile(path.join(source, "code.txt"), "new code\n");
  const expression = script.match(/rsync -az --human-readable \\\n  --exclude (.+) \\\n/)[1];
  for (const envPath of ["settings.conf", "/opt/trader/settings.conf"]) {
    await fs.writeFile(path.join(destination, "settings.conf"), f.original);
    const result = spawnSync("bash", ["-c", `set -euo pipefail\nenv_file="$ENV_PATH"\nrepo_dir=/opt/trader\nexclude=${expression}\nrsync -a --exclude "$exclude" "$SOURCE/" "$DESTINATION/"`], {
      encoding: "utf8", env: { PATH: process.env.PATH, ENV_PATH: envPath, SOURCE: source, DESTINATION: destination },
    });
    assert.equal(result.status, 0, result.stderr);
    assert.equal(await fs.readFile(path.join(destination, "settings.conf"), "utf8"), f.original);
    assert.equal(await fs.readFile(path.join(destination, "code.txt"), "utf8"), "new code\n");
  }
});

test("preserve detects runtime mutation before successful attestation", async (t) => {
  const f = await fixture(t);
  const result = f.run("preserve", "", "", 'printf "SETTING=changed\\n" >> "$ENV_FILE"\nassert_environment_preserved');
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /environment changed/);
  assert.match(script, /if ! assert_environment_preserved; then\n  trap - ERR\n  exit 1\nfi\n"\$\{compose\[@\]\}" up/);
  assert.match(script, /assert_environment_preserved\necho "==> Deploy healthy and commit-attested/);
});

test("explicit apply retains overlay precedence and unrelated settings", async (t) => {
  const f = await fixture(t);
  const managed = path.join(f.directory, "managed");
  const overrides = path.join(f.directory, "overrides");
  await fs.writeFile(managed, "SETTING=managed\nADDED=value\n");
  await fs.writeFile(overrides, "SETTING=override\n");
  const result = f.run("apply", managed, overrides);
  assert.equal(result.status, 0, result.stderr);
  const actual = await fs.readFile(f.envFile, "utf8");
  assert.match(actual, /SETTING=override\n/);
  assert.match(actual, /ADDED=value\n/);
  assert.match(actual, /UNMANAGED=retained\n/);
  assert.match(actual, /TRADER_BOT_TRADE=false\n/);
});

test("preserve is an update of an existing release, not an implicit bootstrap", async (t) => {
  const f = await fixture(t);
  for (const values of [["", "image", "commit"], ["container", "", "commit"], ["container", "image", ""]]) {
    const assignments = ["previous_container", "previous_image_id", "previous_commit"]
      .map((key, i) => `${key}=${values[i]}`).join("\n");
    const result = f.run("preserve", "", "", `${assignments}\nassert_existing_release`);
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /existing running, commit-attested API/);
  }
  assert.equal(f.run("preserve", "", "", "previous_container=c\nprevious_image_id=i\nprevious_commit=s\nassert_existing_release").status, 0);
  assert.ok(script.indexOf("assert_existing_release\nif [[") < script.indexOf('"${compose[@]}" build api'));
});

test("automatic workflow preserves env and withholds exchange credentials", async () => {
  const workflow = await fs.readFile(path.join(root, ".github/workflows/deploy-hetzner.yml"), "utf8");
  assert.match(workflow, /ENVIRONMENT_MODE: \$\{\{ inputs\.environment_mode \|\| 'preserve' \}\}/);
  assert.match(workflow, /export TRADER_HETZNER_ENV_MODE="\$ENVIRONMENT_MODE"/);
  assert.match(workflow, /export TRADER_HETZNER_MANAGED_ENV_FILE=""/);
  assert.match(workflow, /export TRADER_HETZNER_ENV_OVERRIDES_FILE=""/);
  for (const key of ["HETZNER_BINANCE_API_KEY", "HETZNER_BINANCE_API_SECRET", "HETZNER_TRADING_BINANCE_API_KEY", "HETZNER_TRADING_BINANCE_API_SECRET"]) {
    assert.ok(workflow.includes(`${key}: \${{ inputs.environment_mode == 'apply' && secrets.${key} || '' }}`));
  }
});
