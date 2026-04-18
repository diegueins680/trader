# AI Development Workflow

Use the repo-root verification wrapper instead of reconstructing CI by hand.

## Canonical verification commands

- `npm run verify:haskell`
- `npm run verify:web`
- `npm run verify:automation`
- `npm run verify`

All four commands route through `bash scripts/verify.sh ...`.

`verify:haskell` runs format, lint, build, smoke, and tests.

`verify:web` runs the web typecheck, test suite, and production build.

`verify:automation` runs the root autoloop regression test at `test/autoloop.test.mjs`.

`verify` runs the full local verification stack in the same order.

## Bootstrap

Haskell:

```bash
cd haskell
cabal update
cabal build
```

Web:

```bash
npm --workspaces=false --prefix haskell/web ci
```

## Pinned local toolchain

The repo now pins the local toolchain with `.tool-versions` and `.nvmrc`.

- GHC: `9.4.8`
- Cabal: `3.12.1.0`
- Node: `20.19.0`

CI also uses `fourmolu 0.15.0.0` and `hlint 3.8`.

## Local AI automation environment

Optional local `.env` settings:

- `OPENAI_API_KEY`
- `AUTOLOOP_BACKEND`
- `AUTOLOOP_MODEL`
- `AUTOLOOP_MAX_ITERATIONS`
- `AUTOLOOP_FOREVER_INTERVAL_SECONDS`
- `CODEX_LOOP_MODEL`
- `CODEX_LOOP_ALLOW_DIRTY`

These are documented in `.env.example`.

GitHub-hosted autoloop still requires repository secrets:

- `OPENAI_API_KEY`
- `AUTOLOOP_PUSH_TOKEN`

`AUTOLOOP_PUSH_TOKEN` is intentionally a GitHub Actions secret, not a checked-in local default.

## Local automation commands

Bounded autoloop:

```bash
npm run autoloop -- --dry-run
```

Persistent autoloop runner:

```bash
scripts/autoloop-forever.sh start
scripts/autoloop-forever.sh status
scripts/autoloop-forever.sh stop
```

Logical-correctness loop:

```bash
scripts/codex-logical-correctness-loop.sh
```

Each cycle starts with a backend Haskell trading-algorithm audit before it asks Codex to implement the selected logic/correctness improvement.

## Scope guidance

- If you touch `haskell/app`, `haskell/test`, or `haskell/scripts`, run `npm run verify:haskell`.
- If you touch `haskell/web`, run `npm run verify:web`.
- If you touch `scripts/autoloop*.mjs`, `scripts/codex-logical-correctness-loop.sh`, or `test/autoloop.test.mjs`, run `npm run verify:automation`.
- Before a PR, run `npm run verify`.
