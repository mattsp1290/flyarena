import type { AgentId } from '../arena/types';
import type { GraphMode } from '../connectome/format';
import { createWorkerClient, type WorkerClient } from '../worker/client';
import { loadArenaArtifacts, type ArenaManifest, type LoadedArenaArtifacts } from './assets';
import { buildGraphBufferForMode, createWorkerAgentBinding } from './bindings';
import { ExperimentRunner, isNotInitializedRejection, type ExperimentTelemetry } from './runner';
import { transition, type ExperimentStatus } from './state';

/**
 * Plain (framework-agnostic) orchestrator for everything that happens
 * *before* an `ExperimentRunner` exists, plus the one runner-adjacent
 * concurrency primitive that isn't really UI logic: switching an arm's
 * topology mid-experiment. Extracted out of `App.svelte` (WP6 item 6's
 * original home for this) so it is unit-testable directly — no mounted
 * component, no jsdom, no mocked `ArenaScene` render module — the same way
 * `ExperimentRunner` itself is tested (see `tests/unit/experiment-controller.test.ts`).
 *
 * Owns:
 * - `initialize()`: fetch + hash-verify both graph artifacts
 *   (`./assets.ts`), construct one dedicated Worker per arm, build both
 *   `AgentBinding`s, and construct the `ExperimentRunner`.
 * - `changeTopology()`: per-arm serialized topology switch (dispose ->
 *   rebuild the Worker binding -> `runner.setAgentBinding`), including the
 *   busy-counter bookkeeping a caller needs to lock its own UI controls
 *   while a switch is in flight.
 *
 * Does *not* own: rendering, DOM/canvas lifecycle, or the thin
 * Start/Pause/Reset/seed-input wiring — those stay in `App.svelte`, which
 * calls straight through to the constructed `ExperimentRunner` (see
 * `getRunner()`).
 */

export interface ExperimentControllerCallbacks {
  onStatusChange: (status: ExperimentStatus) => void;
  onTelemetry: (telemetry: ExperimentTelemetry) => void;
  onError: (message: string) => void;
  /** Fired once the manifest is fetched, independent of whether the subsequent Worker/binding construction below it succeeds — mirrors the ledger panel being able to show manifest info even if the run itself never reaches `ready`. */
  onManifest: (manifest: ArenaManifest) => void;
  /**
   * Fired once per agent right after `initialize()` successfully constructs
   * the runner (with each arm's *initial* topology), and again after every
   * topology switch that has *already* succeeded — never speculatively
   * before a switch is confirmed. The host's hook for keeping a
   * presentation layer (e.g. `ArenaScene#setAgentTopology`) honestly synced
   * to the arm's real topology; see that method's doc comment for why this
   * matters.
   */
  onTopologyApplied: (agentId: AgentId, mode: GraphMode) => void;
  /** Mirrors the controller's internal per-arm switch-in-flight counters back to the host's own reactive state after every change, so the host can lock its controls for the duration. */
  onTopologySwitchCountChange: (counts: Readonly<Record<AgentId, number>>) => void;
}

export interface ExperimentControllerOptions {
  seed: number;
  totalTicks: number;
  initialTopology: Record<AgentId, GraphMode>;
  /** Constructs one dedicated neural Worker; called twice (once per arm) by `initialize()`. Injectable so tests can supply a fake `Worker`-shaped stand-in without a real `Worker` global. */
  createWorker: () => Worker;
  callbacks: ExperimentControllerCallbacks;
  /** Injectable for tests; defaults to `./assets.ts#loadArenaArtifacts`. */
  loadArtifacts?: typeof loadArenaArtifacts;
}

/** See `AgentRunnerInfo.graphBinarySha256` (`./runner.ts`): the manifest-verified hash for a mode with a real compiled artifact of its own; `undefined` for the runtime-derived 'disconnected' control, which has none. */
const graphBinarySha256ForMode = (manifest: ArenaManifest, mode: GraphMode): string | undefined => {
  if (mode === 'biological') return manifest.binarySha256;
  if (mode === 'rewired') return manifest.rewiredArms.seed0?.binarySha256;
  return undefined;
};

export class ExperimentController {
  private readonly options: ExperimentControllerOptions;
  private readonly topology: Record<AgentId, GraphMode>;
  private readonly topologySwitchCount: Record<AgentId, number> = { left: 0, right: 0 };
  /**
   * Per-arm serialization for topology switches: each switch is chained
   * behind any earlier one for the *same* arm, so a Worker's `dispose()`/
   * `init()` calls can never interleave (the Worker protocol only allows
   * one `init` per `dispose` — see `neural.worker.ts`) — two overlapping
   * switches previously raced this straight into the terminal `error`
   * state.
   */
  private topologySwitchChains: Record<AgentId, Promise<unknown>> = { left: Promise.resolve(), right: Promise.resolve() };
  private destroyed = false;
  private runner: ExperimentRunner | undefined;
  private workerClients: Record<AgentId, WorkerClient> | undefined;
  private manifest: ArenaManifest | undefined;
  private biologicalGraphBuffer: ArrayBuffer | undefined;
  private rewiredGraphBuffer: ArrayBuffer | undefined;

  constructor(options: ExperimentControllerOptions) {
    this.options = options;
    this.topology = { ...options.initialTopology };
  }

  getRunner(): ExperimentRunner | undefined {
    return this.runner;
  }

  getManifest(): ArenaManifest | undefined {
    return this.manifest;
  }

  getTopology(): Readonly<Record<AgentId, GraphMode>> {
    return this.topology;
  }

  /**
   * WP6 item 2: fetch both graph artifacts, gunzip, and sha256-verify them
   * before anything is allowed to start. WP6 item 3: one dedicated Worker
   * per arm, initialized with the default biological (left) vs rewired
   * (right) topology.
   *
   * Calls `transition('loading', ...)` directly (rather than through
   * `ExperimentRunner#fail()`) because no runner exists yet during this
   * phase — there is nothing for a runner to own until the graphs are
   * loaded and both Worker bindings exist. Every failure path *after* the
   * runner is constructed goes through `runner.fail()` instead (see
   * `changeTopology`), so the host and the runner's own status can never
   * disagree once a run exists. See `./state.ts`'s doc comment for why this
   * one caller is a deliberate exception to "`ExperimentRunner` is the only
   * caller".
   */
  async initialize(): Promise<void> {
    const load = this.options.loadArtifacts ?? loadArenaArtifacts;
    let artifacts: LoadedArenaArtifacts;
    try {
      artifacts = await load(`${import.meta.env.BASE_URL}data`);
    } catch (error) {
      if (this.destroyed) return;
      this.options.callbacks.onError(error instanceof Error ? error.message : String(error));
      this.options.callbacks.onStatusChange(transition('loading', { type: 'assetsFailed' }));
      return;
    }
    if (this.destroyed) return;

    this.manifest = artifacts.manifest;
    this.biologicalGraphBuffer = artifacts.biological;
    this.rewiredGraphBuffer = artifacts.rewired;
    this.options.callbacks.onManifest(artifacts.manifest);

    try {
      const left = this.options.createWorker();
      const right = this.options.createWorker();
      this.workerClients = { left: createWorkerClient(left), right: createWorkerClient(right) };

      const [leftBinding, rightBinding] = await Promise.all([
        createWorkerAgentBinding(
          this.workerClients.left,
          buildGraphBufferForMode(this.biologicalGraphBuffer, this.rewiredGraphBuffer, this.topology.left),
          this.topology.left,
          graphBinarySha256ForMode(this.manifest, this.topology.left)
        ),
        createWorkerAgentBinding(
          this.workerClients.right,
          buildGraphBufferForMode(this.biologicalGraphBuffer, this.rewiredGraphBuffer, this.topology.right),
          this.topology.right,
          graphBinarySha256ForMode(this.manifest, this.topology.right)
        )
      ]);
      if (this.destroyed) return;

      this.runner = new ExperimentRunner({
        seed: this.options.seed,
        totalTicks: this.options.totalTicks,
        agents: { left: leftBinding, right: rightBinding },
        onStatusChange: (next) => {
          if (!this.destroyed) this.options.callbacks.onStatusChange(next);
        },
        onTelemetry: (next) => {
          if (!this.destroyed) this.options.callbacks.onTelemetry(next);
        },
        onError: (error) => {
          if (!this.destroyed) this.options.callbacks.onError(error.message);
        }
      });
      this.options.callbacks.onTelemetry(this.runner.getTelemetry());
      this.options.callbacks.onStatusChange(this.runner.getStatus());
      this.options.callbacks.onTopologyApplied('left', this.topology.left);
      this.options.callbacks.onTopologyApplied('right', this.topology.right);
    } catch (error) {
      if (this.destroyed) return;
      this.options.callbacks.onError(error instanceof Error ? error.message : String(error));
      this.options.callbacks.onStatusChange(transition('loading', { type: 'assetsFailed' }));
    }
  }

  /**
   * Re-initializes just one arm's Worker with a freshly derived graph
   * buffer for the chosen topology; only valid from `ready`/`finished`
   * (enforced here, by `ExperimentRunner#setAgentBinding`, and by the
   * host's own disabled-control wiring). Always implies a reset — see
   * `setAgentBinding`'s doc comment. A no-op (silently ignored) if called
   * before `initialize()` has produced a runner, or while the run is
   * active/paused.
   */
  changeTopology(agentId: AgentId, mode: GraphMode): void {
    if (!this.runner || !this.workerClients || !this.biologicalGraphBuffer || !this.rewiredGraphBuffer || !this.manifest) {
      return;
    }
    const currentStatus = this.runner.getStatus();
    if (currentStatus !== 'ready' && currentStatus !== 'finished') return;

    this.topology[agentId] = mode;
    const client = this.workerClients[agentId];
    const buffer = buildGraphBufferForMode(this.biologicalGraphBuffer, this.rewiredGraphBuffer, mode);
    const graphBinarySha256 = graphBinarySha256ForMode(this.manifest, mode);
    this.topologySwitchCount[agentId] += 1;
    this.options.callbacks.onTopologySwitchCountChange({ ...this.topologySwitchCount });

    this.topologySwitchChains[agentId] = this.topologySwitchChains[agentId]
      .then(async () => {
        try {
          await client.dispose();
          if (this.destroyed || !this.runner) return;
          const binding = await createWorkerAgentBinding(client, buffer, mode, graphBinarySha256);
          if (this.destroyed || !this.runner) return;
          this.runner.setAgentBinding(agentId, binding);
          // The rebuilt binding's Worker starts with activity streaming off
          // (a fresh `init`, per `neural.worker.ts`'s "always false" note),
          // even though this arm may have been streaming right before the
          // switch. Re-issue it on the *new* binding directly — not via
          // `runner.setActivityStreaming`, which would redundantly re-toggle
          // the other, untouched arm too — so the activity view's stream
          // never silently drops for this arm across a topology switch.
          //
          // Deliberately not awaited (dual review flagged the earlier
          // awaited version): `binding.setActivity` -> `client.setActivity`
          // -> `send` posts the `set-activity` message synchronously, inside
          // the Promise executor (`client.ts#send`), before this line even
          // returns — so it is already FIFO-ordered ahead of any later
          // `step` on this Worker regardless of whether its own round trip
          // is awaited. Awaiting it here bought no ordering guarantee, only
          // delayed `onTelemetry`/`onTopologyApplied` and, on a rejection,
          // routed a per-arm streaming-toggle failure into `runner.fail()`
          // — contradicting `ExperimentRunner#setActivityStreaming`'s own
          // documented policy that such a failure is not a run failure, and
          // leaving `onTopologyApplied` unfired even though `setAgentBinding`
          // above had already committed the new topology (a presentation
          // desync: the renderer would keep the old topology label).
          if (this.runner.isActivityStreaming()) {
            binding.setActivity?.(true).catch((error: unknown) => {
              if (this.destroyed) return;
              if (isNotInitializedRejection(error)) {
                // Expected, self-healing race (thermo review S2, applies
                // here for the same reason as `ExperimentRunner
                // #setActivityStreaming`'s own catch): a second switch on
                // this same arm can begin (and dispose this arm's Worker
                // again) before this fire-and-forget re-apply's own round
                // trip has settled, since neither this call nor the rest of
                // this `.then()` block awaits it. `console.debug`, not
                // `console.error`, so this routine race doesn't drown out a
                // genuine re-apply failure.
                console.debug(
                  `ExperimentController: re-applying activity streaming for ${agentId} rejected (expected: not-initialized during a topology switch)`,
                  error
                );
                return;
              }
              console.error(`ExperimentController: re-applying activity streaming for ${agentId} failed`, error);
            });
          }
          this.options.callbacks.onTelemetry(this.runner.getTelemetry());
          this.options.callbacks.onTopologyApplied(agentId, mode);
        } catch (error) {
          if (this.destroyed) return;
          // Route through the runner so the host and the runner's own
          // status can never disagree (see docs/architecture.md's
          // state-machine note) — never report a status change directly
          // from here.
          this.runner?.fail(error);
        }
      })
      .finally(() => {
        if (!this.destroyed) {
          this.topologySwitchCount[agentId] -= 1;
          this.options.callbacks.onTopologySwitchCountChange({ ...this.topologySwitchCount });
        }
      });
  }

  /** Stop accepting further work and tear down the runner/Workers. Idempotent; safe to call whether or not `initialize()` ever completed. */
  dispose(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.runner?.dispose();
    this.workerClients?.left.terminate();
    this.workerClients?.right.terminate();
  }
}
