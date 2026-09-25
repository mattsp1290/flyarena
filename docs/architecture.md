# Architecture

FlyArena builds a static Svelte application into `dist/`. The default arena and counterfactual workbench run entirely in the browser. An optional DGX sandbox uses a separate bounded Python training service for a distinct synthetic model. There is no account system, multiplayer service, database, or claim of full biomechanical fly simulation.

## Component boundaries

- **Navigation shell (`src/Shell.svelte`)** selects hash-based arena/workbench/sandbox views and lazy-loads optional components. It disposes arena and counterfactual computation on navigation; after its first visit the sandbox stays mounted but hidden so it can track an active GPU job.
- **Arena view (`src/App.svelte`)** owns layout, the accessible regions for the arena, experiment controls, telemetry, and model ledger, asset loading/hash verification, Worker lifecycle (one dedicated Worker per arm), and the render loop. It contains no simulation or neural logic of its own — it wires `src/lib/experiment/runner.ts` to the renderer and the UI panels.
- **Arena model (`src/lib/arena/`)** owns deterministic plain-data world state, observations, decoded actions, and replay snapshots. It does not depend on Three.js or DOM APIs.
- **Connectome model (`src/lib/connectome/`)** owns graph parsing, the synchronous CPU oracle, the disconnected-topology transform (`format.ts#createDisconnectedGraph`), and compact neural telemetry.
- **Worker boundary (`src/lib/worker/`)** runs neural steps away from the main thread through a versioned message protocol (`protocol.ts`, `neural.worker.ts`) and a Promise-based RPC client (`client.ts`) that matches responses to callers by `requestId`.
- **Experiment orchestration (`src/lib/experiment/`)** owns the closed-loop tick pipeline (`runner.ts`), the explicit state machine (`state.ts`), artifact fetch/gunzip/hash verification (`assets.ts`), and the CPU-oracle/Worker-client adapters that satisfy the runner's step-function interface (`bindings.ts`). The causal engine and headless evaluators also call the same `stepWorld`; the renderer never does.
- **Renderer (`src/lib/render/`)** treats interpolated arena snapshots (`ArenaSnapshot`) as read-only presentation data — `ArenaScene.update()` never mutates its input and holds no reference back into `arena/world.ts` state beyond the snapshot it was last handed. Camera/`OrbitControls` state and frame timing are local to the renderer and have no path back into observations or the physics step. Visual-only food-pickup/hazard-contact effects are derived by diffing consecutive snapshots (e.g. `FoodState.respawns`) purely for decoration; they are never scored and may not line up frame-for-frame with the simulation's own scoring. The two agents' fixed shape/label identity (`BIO` = left, icosahedron; `REWIRED` = right, octahedron+wireframe) is a renderer-level visual identifier, independent of which topology is actually selected for that arm — see `docs/model-ledger.md`.
- **UI panels (`src/lib/ui/`)** — `ExperimentPanel.svelte` (Start/Pause/Reset, seed, per-agent topology selectors, replay download), `TelemetryPanel.svelte` (read-only live telemetry), and `LedgerPanel.svelte` (the model ledger vocabulary plus provenance links) — issue explicit experiment commands and present telemetry/provenance without bypassing the declared model contracts.

### Opt-in full-neuron rate streaming

`step` responses normally carry only compact telemetry (`connectome/telemetry.ts#computeTelemetry`) and action features — the Worker never returns full per-neuron state by default. A `set-activity` request (`{ type: 'set-activity'; enabled: boolean }`) toggles a per-Worker-instance flag that starts `false` and resets to `false` on every `init`, so this is opt-in per arm, per runtime instance, not a standing capability. While enabled, every `step` response additionally carries `rates: Float32Array` (length `neuronCount`, ~4 KB for the 1,008-neuron MaleCNS arm) — a fresh copy (`state.rate.slice()`) taken outside the allocation-free substep loop, whose `ArrayBuffer` is transferred (not structured-cloned) in the same `postMessage` call. With activity disabled — the default, and the state with the activity view closed — a `step` response has no `rates` key at all (`!('rates' in response)`), so the closed-view path is byte-for-byte the same response shape as before this feature existed. At 30 Hz with both arms streaming, this is about 2 x 1,008 x 4 B x 30 ~= 240 KB/s over `postMessage`.

`neural.worker.ts#handleWorkerRequest` stays a pure function; it returns `{ response, transfer? }` rather than calling `postMessage` itself, and the one production caller (the dedicated-worker-scope wiring at the bottom of that module) is what actually calls `postMessage(response, transfer ?? [])`.

`ExperimentRunner#setActivityStreaming(enabled)` (`experiment/runner.ts`) calls `AgentBinding#setActivity` on each arm that implements it (the oracle/CPU binding does not, and is a no-op) and records the flag immediately, independent of a topology switch's own generation bump — `reset()` deliberately keeps the current streaming setting. `ExperimentController#changeTopology` (`experiment/controller.ts`) re-issues `setActivity(true)` on a rebuilt arm's fresh binding, before the arm is used again, if streaming was on — a fresh `init` always starts a Worker back at activity-off. `ExperimentRunner#getLatestRates(agentId)` exposes the latest per-tick vector, replaced (not accumulated) each tick, `undefined` until streaming is enabled and a tick has actually run, and cleared back to `undefined` on every `setActivityStreaming` call (enable or disable alike, so a reopened view never briefly paints a frame left over from before it closed) and on `dispose()`. An `activityEpoch` counter, bumped by the same calls, also guards `runOneTick` itself: a tick whose Worker round trip straddles a toggle carries rates computed under the *old* setting, and is dropped rather than written back.

WP3's lesion-effect color mode (`src/lib/ui/ActivityPanel.svelte#switchColorMode`) adds a third streaming state beyond "view open"/"view closed": the view open *and* streaming deliberately off, because the mode paints static colors from an offline-computed atlas instead. `frame()` skips its whole rates-polling block while that mode is active, and calls `setActivityStreaming(false)`/`(true)` on entry/exit the same way collapsing/expanding the view does.

## Closed-loop contract

Each fixed world tick follows one conceptual order:

**observe → encode → K neural substeps → aggregate → decode → world step → telemetry**

`observe → encode` is `src/lib/arena/sensors.ts#observeAgent`: it returns the already-normalized 8-channel `Observation` directly usable as a Worker `step` request's `channelValues`, so there is no separate encode function. `K neural substeps → aggregate` is one `step` request per arm, run concurrently, each performing `NEURAL_SUBSTEPS_PER_TICK` (`src/lib/experiment/runner.ts`, currently 4, bounded by `MAX_SUBSTEPS_PER_TICK`) calls to `connectome/model.ts#stepModel` against the same held-constant observation before aggregating into action features. `decode` is `arena/actions.ts#decodeAction`. `world step` is exactly one `stepWorld` call per tick, using both arms' decoded actions together. `telemetry` is `ExperimentRunner#getTelemetry()`.

The sensory encoder and `decodeAction` are shared by all experimental arms in both decoder modes. In **Authored mode** (the default) the whole encode → decode path, including `aggregateOutputs`' three population sums, is identical across arms. In **Trained mode** each arm's readout weights are its own — trained independently per arm on the DGX Spark — while every arm shares one architecture, parameter count, and training procedure; see `docs/model-ledger.md`'s "Trained readout" row for the exact scoping and `docs/trained-readout-report.md` for the held-out evaluation.

### Trained decoder branch

Selecting the Trained decoder (`SetDecoderWorkerRequest`, `src/lib/worker/protocol.ts`) changes the "aggregate → decode" step of the loop above: instead of `aggregateOutputs`' three population sums, the Worker gathers the per-neuron rates of every output-assigned neuron (`connectome/readout.ts#outputNeuronIndices`, length `D`) and runs them through that arm's trained readout (`readoutForward`, `D → H → 3`, tanh/tanh/sigmoid) before the same shared `decodeAction`. A Worker's readout weights are set once, at `init` (`InitWorkerRequest.readout`, validated against that Worker's own parsed graph — a shape/`NaN` mismatch fails `init` itself with `invalid-request`, not a corrupt-graph error), and never touched again per step; `set-decoder` only flips which precomputed path a later `step` call takes, so a decoder switch never requires re-initializing (dispose/`init`) the Worker. `ExperimentController#setDecoder` always switches both arms' Workers together and is only accepted between runs (mirroring the topology-switch contract two paragraphs below), and it always resets the run to tick 0 with the current seed. If the trained-readout artifact is missing, fails its sha256 check, or fails `validateReadoutWeights` against the loaded graph, the Trained option is simply unavailable (surfaced honestly in the ledger panel and the decoder control) and the experiment keeps running in Authored mode — this mirrors the existing arena-artifact hash-verification contract (`assets.ts#loadArenaArtifacts`) rather than introducing a new failure mode.

No renderer, camera, animation state, hidden pathfinder, or privileged map coordinate may bypass the observation contract. Rendering is decoupled from the fixed simulation timestep: `src/App.svelte`'s `requestAnimationFrame` loop only calls `ExperimentRunner#getSnapshot(nowMs)`, which interpolates between the world's last two completed ticks (`arena/world.ts#createSnapshot`) — it never steps the simulation itself.

### One neural control interval per tick, with backpressure instead of catch-up

`ExperimentRunner` (`src/lib/experiment/runner.ts`) drives the loop as a strict, single-flight async pipeline: observe both arms → send one `step` request per arm, concurrently → `await` both → decode → `stepWorld` once. There is no queue and no unbounded catch-up `while` loop: at most one tick is ever in flight, so world tick *N* always uses the actions computed from tick *N*'s own observations, regardless of how long the Worker round trip took. If a step call is slow, the next tick simply starts later — the sim runs slower than 30 Hz and telemetry reports `behindRealtime`, rather than skipping a tick or reusing a stale action. This makes determinism independent of Worker latency and of render-frame timing (see `tests/unit/experiment-runner.test.ts`'s timing-invariance test, which asserts an identical final replay hash across two runs with randomized async step delays).

Each arm runs in its own dedicated Worker (one Worker per arm, not one Worker for both): the two arms can carry independent topologies (biological, rewired, or disconnected), and re-initializing one arm's Worker for a topology switch never touches the other arm's Worker. A topology switch is only accepted between runs (`ExperimentRunner#setAgentBinding` requires `ready`/`finished`, never `running`/`paused`) and always implies a fresh `reset()`, so a run never mixes two topologies for the same arm; `src/App.svelte` additionally serializes switches per arm and locks the relevant controls while one is in flight, since the underlying Worker protocol only accepts one `init` per `dispose`.

## Determinism scope

The arena/rate-model closed loop is bit-exact *within* a platform: `stepModel`/`aggregateOutputs` (`src/lib/connectome/model.ts`) use only `+`/`-`/`*`, which IEEE-754 guarantees identical results on any conforming implementation. It is not bit-exact *across* CPU architectures: the observation path (`src/lib/arena/sensors.ts`, plus a few call sites in `world.ts`) uses `Math.atan2`/`Math.sin`/`Math.cos`/`Math.hypot`, and `src/lib/connectome/readout.ts`'s trained-readout forward pass uses `Math.tanh`/`Math.exp` — all implementation-defined rounding per the ECMAScript spec, so V8's transcendental-function results can differ in the last bit or two between x86_64 and arm64.

Measured (`fix/golden-cross-arch` PR review, GitHub's `ubuntu-latest` x86_64 runner against the arm64-generated committed golden fixtures): one `observations` value out of 4 seeds x 60 ticks x 8 channels differed by ~2.8e-17 absolute / ~3.1e-16 relative (about one float64 ULP), and it did not propagate in that run — downstream columns round-trip through `Float32Array` (~1.19e-7 relative precision), which absorbed it. `episode.test.ts`'s golden-vs-fresh score comparison, which runs entirely through `stepWorld`'s float64 physics (not protected by `Float32Array` rounding), also matched exactly for every committed seed. Every other committed fixture regenerated byte-identical on x86_64. This is an *observation* for these four seeds, not a structural guarantee — a future fixture refresh could see the divergence propagate further through `world.ts`'s float64 state. A follow-up measurement confirmed exactly that: sweeping a synthetic 1-ULP perturbation across every individual transcendental call in one `buildGoldenFiles` run found that `world.ts`'s float64 feedback loop (`Math.sin` in `stepWorld`'s heading integration specifically) can spread noise across as many as 28 leaves in one seed file; the noisiest leaf found (a cancellation in a small-magnitude observation) was ~5.7e-16 absolute / ~2.9e-14 relative — still pure noise, not a behavior difference, but denser and relatively larger than the single-leaf case above.

The authoritative golden-trace fixtures (`tests/fixtures/golden/`) and evaluator numbers were produced on `linux-arm64` with Node `v22.22.3` (`cross-arch-tolerance.ts`'s `GOLDEN_GENERATING_NODE`; a Node/V8 upgrade on the generating machine could shift `Math.*` results independent of architecture). `tests/unit/golden-traces.test.ts` enforces a strict byte-for-byte match only when `process.arch` is `arm64`; on any other architecture it falls back to a tolerance-based structural comparison (`tests/fixtures/cross-arch-tolerance.ts`: integers/strings/counts exact, floats within a tolerance derived from the measurement above, *and* a cap (`MAX_INEXACT_LEAVES`, 8) on how many numeric leaves may differ *above a float64 noise floor* (`LEAF_NOISE_FLOOR_REL`/`LEAF_NOISE_FLOOR_ABS`, 1e-13, both must be exceeded — the AND matters: the worst measured noise leaf's *relative* difference, ~2.9e-14, has only ~3-4x headroom under a relative-only floor, but its *absolute* difference has ~150x headroom, which is what actually excludes it) at all. Every measured float32-ULP-scale regression (a single graph edge weight nudged by one float32 ULP, or a `globalGain` change) left at least one committed file with dozens to thousands of above-floor leaves — comfortably over budget; the single weakest measured (edge, seed-file) pair left only 7 above-floor leaves in one file, but that same edge change still failed the check overall because its other three seed files landed at 304-907 above-floor leaves each (the check fails if *any* committed file exceeds the budget). `tests/unit/episode.test.ts`'s golden-vs-fresh score comparison follows the tolerance rule (not the leaf budget — an `AgentScore` has only 2 float leaves, too few for a count-based budget to add protection). Regenerate committed fixtures (`npm run training:traces`) on `linux-arm64` to keep the byte-exact path meaningful.

## Current scope

Work packages 1–7 of the 3D Connectome Arena POC plan are implemented: the static shell, the deterministic arena simulation core, the sparse neural oracle and Worker runtime, the pinned MaleCNS artifact and rewired control, the read-only Three.js renderer, the closed-loop experiment described above, and browser/CI/performance gates (`tests/e2e/arena.spec.ts`, `.github/workflows/ci.yml`, `scripts/experiments/seed-sweep.ts`, `docs/seed-sweep.md`). `App.svelte`'s previous scripted placeholder motion loop has been removed; the closed-loop `ExperimentRunner` now drives every run. Vercel deployment (work package 8) belongs to a later Beans work package.

## Offline trained-readout pipeline

A separate offline pipeline trains and evaluates the trained readout described
above: a PyTorch `uv` project at `training/` (run on a DGX Spark, batched
cross-entropy method (CEM), never authoritative for published numbers) and
Node `tsx` scripts at `scripts/training/` (`export-arms.ts`, `episode.ts`,
`evaluate.ts`, …) that run the same TypeScript arena/sensor/rate-model/readout
code the browser ships and are the *authoritative* scorer. This pipeline
publishes the pinned, hashed static artifacts the browser loads
(`public/data/trained-readout-v1.{json,manifest.json,report.json}`) and the
human-readable `docs/trained-readout-report.md`. Nothing in this pipeline runs
from the browser or in response to a user action: the shipped product remains
a static, client-only build with no live or product-facing training service,
the same as before this feature — this is a build-time/offline concern only,
distinct from the DGX sandbox's own separate, live PyTorch backend for the
counterfactual workbench described next. See
`.agents/plans/trained-readout/` for the full pipeline design.

## Counterfactual workbench and DGX sandbox

`src/lib/counterfactual/engine.ts` forks complete deterministic world/neural state
and runs baseline, sham and persistent silencing using the canonical arena and
connectome functions. Its dedicated Worker owns graph loading and simulation;
the client owns deadlines and termination. No live neural Worker protocol or
training artifact schema is changed. The evidence schema is distinct from arena
replay and synthetic lab exports. See [contract](counterfactual-workbench.md).

`src/lib/lab/` talks only to an explicitly configured sandbox backend.
`backend/flyarena_lab/` owns its separate PyTorch model and bounded, authenticated
single-job API. Its scoped styles cannot change arena presentation. The backend
launcher binds to loopback by default; no backend request or credential is needed
by the default arena or causal workbench. A hidden sandbox retains polling and
result identity; full component destruction best-effort cancels a known active
job before aborting polling. Browser termination cannot guarantee cancellation.
