# Architecture

FlyArena is a client-only static Svelte application. Vite builds the application into `dist/`; there is no server, account system, multiplayer service, training service, database, or claim of full biomechanical fly simulation.

## Component boundaries

- **Application shell (`src/App.svelte`)** owns layout and the accessible regions for the arena, experiment controls, telemetry, and model ledger.
- **Arena model (`src/lib/arena/`, proposed)** will own deterministic plain-data world state, observations, decoded actions, and replay snapshots. It must not depend on Three.js or DOM APIs.
- **Connectome model (`src/lib/connectome/`, proposed)** will own graph parsing, the synchronous CPU oracle, topology controls, and compact neural telemetry.
- **Worker boundary (`src/lib/worker/`, proposed)** will run neural steps away from the main thread through a versioned message protocol.
- **Renderer (`src/lib/render/`)** treats interpolated arena snapshots (`ArenaSnapshot`) as read-only presentation data — `ArenaScene.update()` never mutates its input and holds no reference back into `arena/world.ts` state beyond the snapshot it was last handed. Camera/`OrbitControls` state and frame timing are local to the renderer and have no path back into observations or the physics step. Visual-only food-pickup/hazard-contact effects are derived by diffing consecutive snapshots (e.g. `FoodState.respawns`) purely for decoration; they are never scored and may not line up frame-for-frame with the simulation's own scoring.
- **UI panels (`src/lib/ui/`, proposed)** will issue explicit experiment commands and present telemetry and provenance without bypassing the declared model contracts.

## Closed-loop contract

Each fixed world tick follows one conceptual order:

**observe → encode → K neural substeps → aggregate → decode → world step → telemetry**

The sensory encoder and action decoder are shared by all experimental arms. No renderer, camera, animation state, hidden pathfinder, or privileged map coordinate may bypass the observation contract. Rendering is decoupled from the fixed simulation timestep.

## Current scope

The static shell, test harness, architecture/model vocabulary, the deterministic arena simulation core, and the read-only Three.js renderer (work package 5) are implemented. `App.svelte` currently drives the renderer with a scripted, tick-only placeholder motion loop (clearly commented as such) so the scene has real snapshots to draw before the neural Worker exists; it reads no sensors and does no pathfinding. Worker execution, graph artifacts, closed-loop integration (replacing the placeholder loop), browser performance tests, and deployment belong to later Beans work packages.
