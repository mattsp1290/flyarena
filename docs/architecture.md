# Architecture

FlyArena is a client-only static Svelte application. Vite builds the application into `dist/`; there is no server, account system, multiplayer service, training service, database, or claim of full biomechanical fly simulation.

## Component boundaries

- **Application shell (`src/App.svelte`)** owns layout, the accessible regions for the arena, experiment controls, telemetry, and model ledger, asset loading/hash verification, Worker lifecycle (one dedicated Worker per arm), and the render loop. It contains no simulation or neural logic of its own — it wires `src/lib/experiment/runner.ts` to the renderer and the UI panels.
- **Arena model (`src/lib/arena/`)** owns deterministic plain-data world state, observations, decoded actions, and replay snapshots. It does not depend on Three.js or DOM APIs.
- **Connectome model (`src/lib/connectome/`)** owns graph parsing, the synchronous CPU oracle, the disconnected-topology transform (`format.ts#createDisconnectedGraph`), and compact neural telemetry.
- **Worker boundary (`src/lib/worker/`)** runs neural steps away from the main thread through a versioned message protocol (`protocol.ts`, `neural.worker.ts`) and a Promise-based RPC client (`client.ts`) that matches responses to callers by `requestId`.
- **Experiment orchestration (`src/lib/experiment/`)** owns the closed-loop tick pipeline (`runner.ts`), the explicit state machine (`state.ts`), artifact fetch/gunzip/hash verification (`assets.ts`), and the CPU-oracle/Worker-client adapters that satisfy the runner's step-function interface (`bindings.ts`). This is the only layer that calls `stepWorld`; the renderer never does.
- **Renderer (`src/lib/render/`)** treats interpolated arena snapshots (`ArenaSnapshot`) as read-only presentation data — `ArenaScene.update()` never mutates its input and holds no reference back into `arena/world.ts` state beyond the snapshot it was last handed. Camera/`OrbitControls` state and frame timing are local to the renderer and have no path back into observations or the physics step. Visual-only food-pickup/hazard-contact effects are derived by diffing consecutive snapshots (e.g. `FoodState.respawns`) purely for decoration; they are never scored and may not line up frame-for-frame with the simulation's own scoring. The two agents' fixed shape/label identity (`BIO` = left, icosahedron; `REWIRED` = right, octahedron+wireframe) is a renderer-level visual identifier, independent of which topology is actually selected for that arm — see `docs/model-ledger.md`.
- **UI panels (`src/lib/ui/`)** — `ExperimentPanel.svelte` (Start/Pause/Reset, seed, per-agent topology selectors, replay download), `TelemetryPanel.svelte` (read-only live telemetry), and `LedgerPanel.svelte` (the model ledger vocabulary plus provenance links) — issue explicit experiment commands and present telemetry/provenance without bypassing the declared model contracts.

## Closed-loop contract

Each fixed world tick follows one conceptual order:

**observe → encode → K neural substeps → aggregate → decode → world step → telemetry**

`observe → encode` is `src/lib/arena/sensors.ts#observeAgent`: it returns the already-normalized 8-channel `Observation` directly usable as a Worker `step` request's `channelValues`, so there is no separate encode function. `K neural substeps → aggregate` is one `step` request per arm, run concurrently, each performing `NEURAL_SUBSTEPS_PER_TICK` (`src/lib/experiment/runner.ts`, currently 4, bounded by `MAX_SUBSTEPS_PER_TICK`) calls to `connectome/model.ts#stepModel` against the same held-constant observation before aggregating into action features. `decode` is `arena/actions.ts#decodeAction`. `world step` is exactly one `stepWorld` call per tick, using both arms' decoded actions together. `telemetry` is `ExperimentRunner#getTelemetry()`.

The sensory encoder and action decoder are shared by all experimental arms. No renderer, camera, animation state, hidden pathfinder, or privileged map coordinate may bypass the observation contract. Rendering is decoupled from the fixed simulation timestep: `src/App.svelte`'s `requestAnimationFrame` loop only calls `ExperimentRunner#getSnapshot(nowMs)`, which interpolates between the world's last two completed ticks (`arena/world.ts#createSnapshot`) — it never steps the simulation itself.

### One neural control interval per tick, with backpressure instead of catch-up

`ExperimentRunner` (`src/lib/experiment/runner.ts`) drives the loop as a strict, single-flight async pipeline: observe both arms → send one `step` request per arm, concurrently → `await` both → decode → `stepWorld` once. There is no queue and no unbounded catch-up `while` loop: at most one tick is ever in flight, so world tick *N* always uses the actions computed from tick *N*'s own observations, regardless of how long the Worker round trip took. If a step call is slow, the next tick simply starts later — the sim runs slower than 30 Hz and telemetry reports `behindRealtime`, rather than skipping a tick or reusing a stale action. This makes determinism independent of Worker latency and of render-frame timing (see `tests/unit/experiment-runner.test.ts`'s timing-invariance test, which asserts an identical final replay hash across two runs with randomized async step delays).

Each arm runs in its own dedicated Worker (one Worker per arm, not one Worker for both): the two arms can carry independent topologies (biological, rewired, or disconnected), and re-initializing one arm's Worker for a topology switch never touches the other arm's Worker. A topology switch is only accepted between runs (`ExperimentRunner#setAgentBinding` requires `ready`/`finished`, never `running`/`paused`) and always implies a fresh `reset()`, so a run never mixes two topologies for the same arm; `src/App.svelte` additionally serializes switches per arm and locks the relevant controls while one is in flight, since the underlying Worker protocol only accepts one `init` per `dispose`.

## Current scope

Work packages 1–6 of the 3D Connectome Arena POC plan are implemented: the static shell, the deterministic arena simulation core, the sparse neural oracle and Worker runtime, the pinned MaleCNS artifact and rewired control, the read-only Three.js renderer, and the closed-loop experiment described above. `App.svelte`'s previous scripted placeholder motion loop has been removed; the closed-loop `ExperimentRunner` now drives every run. Browser performance/CI gates (work package 7) and Vercel deployment (work package 8) belong to later Beans work packages.
