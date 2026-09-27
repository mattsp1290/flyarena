# Private real-graph lab

This document uses placeholders only: `<TAILNET_ORIGIN>`, `<TAILNET_ADDRESS>`,
and `<PORT>` stand in for real values. Never put a real hostname, origin, or
IP address in this file, in any other tracked documentation, or in a script
(the rule enforced by `.agents/deployment.md`).

## Purpose

The graph lab is a private backend service (`backend/graph_lab/`) that runs
jobs against the **real MaleCNS connectome**, for private, tailnet-only use.
It is a separate service from the existing synthetic lab
(`backend/flyarena_lab/`, `docs/counterfactual-lab.md`): different data,
different provenance, and its own token/origin/bind configuration
(`GRAPH_LAB_TOKEN`, `GRAPH_LAB_ORIGINS`, `GRAPH_LAB_BIND`). The public static
site keeps working with no backend running; the front-end route is labeled
private, and every rendered result is labeled **"Computed on DGX (private,
not published)"**.

There are no public endpoints, no accounts, and no persistence beyond a
small in-memory job store.

## Job types and bounds

All three job types resolve a `graph` the same way: `biological` (the
image's checked, sha-verified `malecns-arena-v1.bin.gz`), `rewired:<seed>`
(seed 0-499, regenerated on demand and sha-checked against
`rewiring-null-v1.json`), or `disconnected`.

| Job (`kind`) | Bounds | Wall-clock ceiling |
| --- | --- | --- |
| **Lesion sweep** (`lesion`) | 1-32 neuron sets, each 1-64 unique neuron indices; 4-100 consecutive seeds; ticks 300-1800 | 20 min |
| **Atlas search** (`atlas`) | population 4-64; generations 1-48; ticks 300-900. Refuses to start (reports "GPU busy") if free GPU memory is under 2 GiB | 15 min |
| **Swap-set intervention** (`swapset`) | 1-50 swaps (each a valid degree-preserving 2-swap); 0-100 class-matched random controls; seeds/ticks follow the lesion bounds | 20 min |

Exact request schemas and validation live in `backend/graph_lab/models.py`;
see `.agents/plans/graph-lab/02-job-engines.md` for the full engine design.

## Provenance and labels

Every result carries `{ graph, graphSha256, bundleSha256, host: {arch, node,
torch}, label: "Computed on DGX (private, not published)" }`. `graphSha256`
identifies the exact graph data used; `bundleSha256` identifies the exact
JS code (see "Authority" below). `GET /api/graph/v1/health` (unauthenticated,
and deliberately limited to these non-sensitive fields) reports the model
version, `bundleSha256`, `graphSha256`, and whether a GPU is available -- no
free-memory figure, no address, no token.

## Authority: the TypeScript bundle is the source of truth

The service runs job logic as esbuild-bundled Node scripts
(`scripts/graph-lab/bundle.mjs`, producing `backend/graph_lab/js/*.mjs`,
gitignored and rebuilt by `npm run graph-lab:bundle`) built from the exact
same TypeScript modules that power the rest of this repo's offline analysis
(`scripts/training/episode.ts`, the null-worker sharding pattern,
`scripts/atlas/*`). There is no separate Python re-implementation of the
evaluator: the same source code runs everywhere, so results cannot drift
from what the offline pipeline would compute. `bundle.json` records every
bundled file's sha256, plus a single `bundleSha256` covering the whole
bundle; that value is reported by `/health` and by every job result, so a
result can always be traced back to the exact code that produced it.

`scripts/graph-lab.sh --build` re-runs `backend/graph_lab/tests/test_reproduction.py`'s
reproduction checks against the fresh bundle before building the image (see
"Reproduction checks" below).

## Privacy

- **Tailnet only.** The container is bound only to the Spark's Tailscale
  address (`GRAPH_LAB_BIND`, which must equal an address printed by
  `tailscale ip -4`) on a dedicated Docker bridge network whose subnet has
  no outbound network access (a `DOCKER-USER` iptables rule; see
  "Egress deny" below). It is never reachable from the public internet or
  from the host's LAN interfaces.
- **The token lives in browser memory only.** `GRAPH_LAB_TOKEN` is entered
  by hand in the front-end route and held only in page memory; it is never
  persisted (no `localStorage`, no cookie) and never logged by the service
  (the service never logs `Authorization` headers).
- **CORS is limited to configured origins.** `GRAPH_LAB_ORIGINS` is an
  explicit, comma-separated allowlist; wildcard (`*`) origins are rejected
  at service startup.

## Lifetime

Job results live only in the service's in-memory job store (at most 4
retained). **A container restart loses every job result.** There is no
database and no on-disk job history. Export or record anything you need to
keep before stopping the container.

## HTTP/mixed-content constraint

Both the static site and the tailnet backend are served over plain `http`
today, so there is no mixed-content issue. **If the site ever moves to
`https`**, the graph-lab backend will need TLS too (a browser refuses to let
an `https` page make plain-`http` requests to a private tailnet address).
This is a known constraint, not yet a requirement -- there is no near-term
plan to add TLS termination to the tailnet-only backend.

## Egress deny

The container's bridge network subnet (`GRAPH_LAB_SUBNET`, default
`172.31.250.0/24`) is denied all new outbound connections by a
`DOCKER-USER` iptables rule that `scripts/graph-lab.sh --start`/`--status`
verify are present (see `.agents/deployment.md`'s "Real-graph lab" runbook
for the full procedure, including the one-time reboot/iptables-persistence
caveat). `--start` refuses to run the container at all if the rule cannot
be confirmed present.

## Reproduction checks (WP2 gate, re-run on every build)

`backend/graph_lab/tests/test_reproduction.py` reproduces each job type
against the real, checked-in MaleCNS data and the published artifacts:

- **Lesion** and **swap-set** are exact-match checks (deterministic on CPU)
  -- `scripts/graph-lab.sh --build` treats a failure here as a hard gate and
  refuses to tag `flyarena-graph-lab:local`.
- **Atlas** is deliberately **tolerance-based**
  (`ATLAS_MIN_MATCHING_CELLS`, at least 24 of 30 cells must match), because
  `torch.sparse.mm`'s CUDA kernel has no deterministic implementation --
  confirmed intrinsic to the kernel, not to GPU contention. `--build`
  retries this specific check (up to 3 attempts) and treats a persistent
  failure as **a signal to re-run and investigate later, not a release
  gate**: it still builds and tags the image, with a printed warning.
  This is a deliberate deviation from `02-job-engines.md`'s more literal
  wording ("refuses to tag ... on failure") for this one check, per
  `test_reproduction.py`'s own module docstring, which asks the
  implementer of `--build` to follow this exact guidance.

Set `GRAPH_LAB_SKIP_REPRO=1` to skip this step in `--build` entirely --
useful while another GPU job is already running on the Spark, since the
atlas check needs the GPU. Re-run it manually once the GPU is free:
`cd backend/graph_lab && uv run pytest -m spark -v`. Missing `uv` on `PATH`
is treated as a hard failure, not a silent skip -- only the explicit
`GRAPH_LAB_SKIP_REPRO=1` opt-out may bypass the gate.

A second, smaller deviation from `02-job-engines.md`'s wording: `--build`
runs these checks against the freshly bundled JS **before** `docker build`,
not "against the fresh image" as that plan literally says.
`test_reproduction.py`'s checks invoke the engines directly (not the built
container), so there is nothing image-specific left to test post-build, and
failing fast here skips an otherwise-wasted image build on a hard-gate
failure.

## Verification (owner-run, real launch)

See `.agents/deployment.md`'s "Real-graph lab" section for the full launch
and verification runbook (build, start, the CORS/PNA/auth/egress checks,
tailscaled-restart behavior, and token rotation). Every command there uses
placeholders in the same way this document does.

## WP5 finding: Playwright/PNA preflight (recorded later)

`.agents/plans/graph-lab/04-launch-and-runbook.md`'s WP5 records, once the
site is redeployed and round-tripped against the real tailnet backend,
whether the pinned Playwright Chromium sends an
`Access-Control-Request-Private-Network` preflight for a
`100.64.0.0/10` (tailnet) target from the site's origin. That finding is not
yet recorded -- WP5 is a later phase than this document's own WP4 scope. The
Private Network Access middleware (`backend/graph_lab/service.py`) stays
either way; the gate is the real end-to-end round trip, not the presence of
that header.
