# Architecture

FlyArena is a client-only static Svelte application. Vite builds the application into `dist/`; there is no server, account system, multiplayer service, training service, database, or claim of full biomechanical fly simulation.

## Component boundaries

- **Application shell (`src/App.svelte`)** owns layout and the accessible regions for the arena, experiment controls, telemetry, and model ledger.
- **Arena model (`src/lib/arena/`, proposed)** will own deterministic plain-data world state, observations, decoded actions, and replay snapshots. It must not depend on Three.js or DOM APIs.
- **Connectome model (`src/lib/connectome/`, proposed)** will own graph parsing, the synchronous CPU oracle, topology controls, and compact neural telemetry.
- **Worker boundary (`src/lib/worker/`, proposed)** will run neural steps away from the main thread through a versioned message protocol.
- **Renderer (`src/lib/render/`, proposed)** will treat interpolated arena snapshots as read-only presentation data. Camera and frame timing must not alter simulation outcomes.
- **UI panels (`src/lib/ui/`, proposed)** will issue explicit experiment commands and present telemetry and provenance without bypassing the declared model contracts.

## Closed-loop contract

Each fixed world tick follows one conceptual order:

**observe → encode → K neural substeps → aggregate → decode → world step → telemetry**

The sensory encoder and action decoder are shared by all experimental arms. No renderer, camera, animation state, hidden pathfinder, or privileged map coordinate may bypass the observation contract. Rendering is decoupled from the fixed simulation timestep.

## Current scope

This foundation bean implements only the static shell, test harness, and architecture/model vocabulary. Simulation, Worker execution, graph artifacts, Three.js rendering, closed-loop integration, browser performance tests, and deployment belong to later Beans work packages.
