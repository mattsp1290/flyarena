# FlyArena

A client-only 3D Connectome Arena proof of concept. The current milestone provides the tested Svelte application shell and the contracts that later simulation, connectome, rendering, and data work will implement.

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

The production build is written to `dist/`. `npm run test:e2e` is reserved for the later Playwright browser suite and currently succeeds with no tests.

## Deployment

Use `./scripts/deploy.sh` for deployment. See [.agents/deployment.md](.agents/deployment.md)
for configuration, hosting requirements, verification, and rollback. Actual hostnames,
addresses, and SSH destinations belong only in the ignored `.env` file.

## Architecture and model claims

- [Architecture and closed-loop contract](docs/architecture.md)
- [Model-ledger vocabulary](docs/model-ledger.md)

FlyArena is an inspectable experiment POC, not a claim of whole-animal or brain emulation.
