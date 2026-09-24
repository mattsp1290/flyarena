# FlyArena

A client-only 3D Connectome Arena proof of concept: a reproducible 90-second arena experiment comparing a pinned, measured MaleCNS-derived connectome against seeded degree-preserving and disconnected controls, with a closed-loop simulation/render pipeline, an inspectable model ledger, and browser/CI/performance gates.

## Requirements

- Node.js 22.22.2 or newer within the Node 22 release line (developed with 22.22.3)
- npm 10.x (developed with 10.9.8)

## Development

```bash
npm ci
npm run dev
```

Vite prints the local development URL. No backend, credentials, database, or runtime service is required.

## Verification

```bash
npm test -- --run
npm run check
npm run build
```

The production build is written to `dist/`.

### End-to-end browser tests

`tests/e2e/arena.spec.ts` (Playwright) drives a real production build end to
end: asset load and hash verification, Start/Pause/Reset control states,
fixed-seed replay determinism, topology switching (including the
disconnected negative control), the model-ledger vocabulary and CC BY
attribution, a WebGL-unavailable fallback, a tampered-artifact hash-mismatch
error path, and the performance gates (interactive load time under a
throttled network profile, median neural step latency, and main-thread long
tasks during a run).

```bash
npx playwright install --with-deps chromium   # once, to fetch the browser
npm run test:e2e
```

`playwright.config.ts`'s `webServer` runs `npm run build && npm run preview`
automatically and serves on `http://127.0.0.1:4173`, so no separate server
needs to be started first.

### Multi-seed descriptive study

`scripts/experiments/seed-sweep.ts` runs >= 20 fixed seeds through the real
checked-in MaleCNS-derived artifact for every experimental arm (biological,
rewired, disconnected), headlessly (no browser) via the same
`ExperimentRunner` the app uses:

```bash
npm run experiment:seed-sweep
```

Writes raw per-seed results to the gitignored
`scripts/experiments/out/seed-sweep-results.json`; see
[docs/seed-sweep.md](docs/seed-sweep.md) for the committed descriptive
summary (mean/median/sample-standard-deviation only — no superiority claim).

### CI

`.github/workflows/ci.yml` runs `npm run check`, the Vitest suite, the
production build, the Playwright suite (against a fresh Chromium install),
`uv run pytest tests_python` (fixture data only, no downloads), an `npm
audit` report, and a dependency license report, on every push to `main` and
every pull request.

## Deployment

Use `./scripts/deploy.sh` for deployment. See [.agents/deployment.md](.agents/deployment.md)
for configuration, hosting requirements, verification, and rollback. Actual hostnames,
addresses, and SSH destinations belong only in the ignored `.env` file.

## Architecture and model claims

- [Architecture and closed-loop contract](docs/architecture.md)
- [Model-ledger vocabulary](docs/model-ledger.md)

FlyArena is an inspectable experiment POC, not a claim of whole-animal or brain emulation.
