import type { AgentId } from '../arena/types';
import { createDisconnectedGraph, parseGraphBinary, type ConnectomeGraph, type GraphMode } from '../connectome/format';
import { validateReadoutWeights, type ReadoutWeights } from '../connectome/readout';
import { createWorkerClient, type WorkerClient } from '../worker/client';
import type { DecoderKind } from '../worker/protocol';
import {
  loadArenaArtifacts,
  loadTrainedReadoutArtifact,
  type ArenaManifest,
  type LoadedArenaArtifacts,
  type TrainedReadoutLoadResult
} from './assets';
import { loadRewiringNull, type RewiringNullLoadResult } from './rewiringNull';
import { loadNullExplanation, type NullExplanationLoadResult } from './nullExplanation';
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
  /**
   * Fired once the manifest is fetched, independent of whether the
   * subsequent Worker/binding construction below it succeeds — mirrors the
   * ledger panel being able to show manifest info even if the run itself
   * never reaches `ready`. `biologicalGraph` is the already hash-verified,
   * parsed biological graph `loadArenaArtifacts` produced for this manifest
   * (`LoadedArenaArtifacts.parsedBiological`) — the host threads it into
   * `assets.ts#loadPositions` instead of letting that function re-fetch and
   * re-parse the same artifact a second time (thermo-architecture I1 fix).
   */
  onManifest: (manifest: ArenaManifest, biologicalGraph: ConnectomeGraph) => void;
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
  /**
   * Fired once the trained-readout artifact has been fetched, hash-verified,
   * and validated (or has failed any of those steps) — independent of
   * whether the rest of `initialize()` succeeds, mirroring `onManifest`'s
   * own independence from graph-artifact/Worker construction. The host's
   * hook for the ledger panel's "Trained (offline)" row and for gating the
   * decoder toggle's Trained option (WP6).
   */
  onTrainedReadoutStatus: (status: TrainedReadoutLoadResult) => void;
  /**
   * Fired once `loadRewiringNull` resolves for this manifest — independent
   * of, and never awaited before, Worker/binding construction (WP4's
   * "loading must not block Start"). Started right after `onManifest` fires,
   * in parallel with the trained-readout load and the Worker construction
   * below it; like `onTrainedReadoutStatus`, this never blocks reaching
   * `ready`. The host's hook for the ledger panel's "Topology null
   * distribution" section (`LedgerPanel.svelte`/`NullHistogram.svelte`).
   */
  onRewiringNull: (result: RewiringNullLoadResult) => void;
  /**
   * Fired once `loadNullExplanation` resolves (WP4 of
   * `.agents/plans/null-explanation`) — sequenced after the rewiring-null
   * load settles (`initialize()`'s own `nullLoad` doc comment explains why:
   * never races ahead of the histogram's own load, and this note's
   * cross-check needs `manifest`, not the resolved rewiring-null data), but
   * fired independently of `onRewiringNull` itself (a throwing
   * `onRewiringNull` host callback must never also skip this one). Like
   * `onRewiringNull`, this never blocks reaching `ready`. The host's hook
   * for the ledger panel's finding note, rendered next to
   * `NullHistogram.svelte` inside the "Topology null distribution" section.
   */
  onNullExplanation: (result: NullExplanationLoadResult) => void;
  /**
   * Fired once per agent right after `setDecoder()` has successfully applied
   * a decoder switch to both arms' Workers and reset the run to tick 0 —
   * mirrors `onTopologyApplied`'s "never speculatively before a switch is
   * confirmed" contract. Reports the single decoder now shared by both arms
   * (WP6's "toggle switches both agents together" non-negotiable), not a
   * per-agent value.
   */
  onDecoderApplied: (decoder: DecoderKind) => void;
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
  /** Injectable for tests; defaults to `./assets.ts#loadTrainedReadoutArtifact`. */
  loadTrainedReadout?: typeof loadTrainedReadoutArtifact;
  /**
   * Injectable for tests; defaults to `./rewiringNull.ts#loadRewiringNull`.
   * Without this seam, a test could never observe the `destroyed` guard on
   * this load's own `.then` (dual review, Important) — every other loader
   * here is injectable for exactly the same reason.
   */
  loadRewiringNull?: typeof loadRewiringNull;
  /**
   * Injectable for tests; defaults to `./nullExplanation.ts#loadNullExplanation`.
   * Same seam-for-testability reasoning as `loadRewiringNull` above.
   */
  loadNullExplanation?: typeof loadNullExplanation;
  /** Passed straight through to the constructed `ExperimentRunner` (see `ExperimentRunnerOptions.targetTickIntervalMs`); `0` disables real-time pacing entirely, which unit tests use to run a many-tick determinism check without waiting out real seconds. Omitted in production, matching the runner's own real-time default. */
  targetTickIntervalMs?: number;
}

/** See `AgentRunnerInfo.graphBinarySha256` (`./runner.ts`): the manifest-verified hash for a mode with a real compiled artifact of its own; `undefined` for the runtime-derived 'disconnected' control, which has none. */
const graphBinarySha256ForMode = (manifest: ArenaManifest, mode: GraphMode): string | undefined => {
  if (mode === 'biological') return manifest.binarySha256;
  if (mode === 'rewired') return manifest.rewiredArms.seed0?.binarySha256;
  return undefined;
};

/**
 * Owns the mutual-exclusion invariant between `ExperimentController#setDecoder`
 * and `ExperimentController#changeTopology`: the two must never interleave
 * on the same arm's Worker. A rebuilt binding's re-apply branch
 * (`changeTopology`) reads the controller's current decoder to decide
 * whether to re-issue `set-decoder: 'trained'` on the rebuilt Worker, but
 * `setDecoder` only writes that field *after* its own Worker round trip
 * resolves — so an interleaved topology switch could otherwise silently
 * leave an arm on `'authored'` while the controller went on to report
 * `'trained'` for both arms (the round-1 dual-review race). Previously this
 * invariant was enforced by two independently-invented primitives
 * (`decoderSwitchInFlight`, a plain boolean, and `topologySwitchCount`, a
 * per-arm busy counter) that a human had to keep in sync by reading two
 * separate doc comments (thermo-maintainability I1) — this class is the
 * single place that now owns the cross-check both directions read.
 *
 * `topologySwitchCount` is a *live reference* to `ExperimentController`'s
 * own per-arm busy counters (read here, not copied — `changeTopology`
 * mutates the same object in place via its bump/decrement bookkeeping).
 * This class does not own that counter's per-arm queuing semantics (a
 * second topology switch on the same arm still queues behind the first via
 * `topologySwitchChains`, unrelated to the decoder/topology invariant) —
 * only the read used to decide whether a decoder switch may start.
 */
class DecoderSwitch {
  private inFlight = false;

  constructor(private readonly topologySwitchCount: Readonly<Record<AgentId, number>>) {}

  /**
   * True for the synchronous-to-Promise-resolution duration of a
   * `setDecoder` call. `changeTopology`'s own guard bails out while this is
   * `true` — see this class's doc comment for the race that closes.
   */
  get isInFlight(): boolean {
    return this.inFlight;
  }

  /**
   * `setDecoder`'s own guard: true when a decoder switch may start right
   * now — no other decoder switch already in flight (including a second,
   * overlapping call to `setDecoder` itself — round-2 dual review), and no
   * topology switch in flight on either arm (avoids racing `setDecoder`'s
   * own Worker messages against a topology switch's dispose/init window).
   */
  canStartDecoderSwitch(): boolean {
    return !this.inFlight && this.topologySwitchCount.left === 0 && this.topologySwitchCount.right === 0;
  }

  /**
   * Claims exclusive ownership. Must be called synchronously, before
   * `setDecoder`'s first `await` — this is what lets `changeTopology`'s
   * `isInFlight` guard actually exclude a topology switch starting anywhere
   * during `setDecoder`'s Worker round trip, not just after the decoder
   * field is written at the end (see this class's doc comment).
   */
  acquire(): void {
    this.inFlight = true;
  }

  /** Releases ownership; called from `setDecoder`'s `finally` block. */
  release(): void {
    this.inFlight = false;
  }
}

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
  /**
   * `undefined` until `initialize()`'s trained-readout load/validate step
   * finishes (or never runs, e.g. `dispose()` raced it). `'ok'` carries the
   * per-arm decoded, graph-validated weights `readoutWeightsForMode` reads
   * from; `'unavailable'` carries the honest reason the ledger panel and the
   * decoder toggle's Trained option both surface — see `validateTrainedReadout`.
   */
  private trainedReadout: TrainedReadoutLoadResult | undefined;
  /** The decoder currently shared by both arms' Workers; always `'authored'` until a successful `setDecoder('trained')` call. */
  private decoder: DecoderKind = 'authored';
  /**
   * Owns the `setDecoder`-vs-`changeTopology` mutual-exclusion invariant —
   * see `DecoderSwitch`'s doc comment above for the race this closes.
   * Constructed with a live reference to `topologySwitchCount` (declared
   * above, so it is already initialized by the time this field's own
   * initializer runs).
   */
  private readonly decoderSwitch = new DecoderSwitch(this.topologySwitchCount);

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

  getDecoder(): DecoderKind {
    return this.decoder;
  }

  /** `undefined` before `initialize()`'s trained-readout step has resolved; see `trainedReadout`'s doc comment. */
  getTrainedReadoutStatus(): TrainedReadoutLoadResult | undefined {
    return this.trainedReadout;
  }

  /** This arm's trained-readout weights, or `undefined` when the artifact is unavailable/failed validation. Always passed to `createWorkerAgentBinding` (init-time), independent of which decoder is currently selected — see that function's doc comment. */
  private readoutWeightsForMode(mode: GraphMode): ReadoutWeights | undefined {
    return this.trainedReadout?.status === 'ok' ? this.trainedReadout.weightsByMode[mode] : undefined;
  }

  /**
   * Validate the fetched/hash-verified trained-readout artifact
   * (`loadTrainedReadoutArtifact`'s result) against the three arms' actual
   * parsed graphs — `validateReadoutWeights` checks `inputSize`/`hiddenSize`/
   * array lengths/finiteness per arm (WP6's "D/H/paramCount must match the
   * loaded graph" invariant). All-or-nothing: if any one arm fails, the
   * whole artifact is treated as unavailable, matching the product
   * invariant that every arm's readout shares one architecture/parameter
   * count — a per-arm-mismatched artifact is not a case "some arms get
   * Trained and others don't" is meant to handle. Never throws: an
   * `'ok'` input can still come back `'unavailable'` here; an `'unavailable'`
   * input passes through unchanged.
   */
  private validateTrainedReadout(
    result: TrainedReadoutLoadResult,
    parsedBiological: Readonly<ConnectomeGraph>,
    rewiredGraphBuffer: ArrayBuffer
  ): TrainedReadoutLoadResult {
    if (result.status !== 'ok') return result;
    try {
      const parsedRewired = parseGraphBinary(rewiredGraphBuffer.slice(0));
      const parsedDisconnected = createDisconnectedGraph(parsedBiological);
      validateReadoutWeights(result.weightsByMode.biological, parsedBiological);
      validateReadoutWeights(result.weightsByMode.rewired, parsedRewired);
      validateReadoutWeights(result.weightsByMode.disconnected, parsedDisconnected);
      return result;
    } catch (error) {
      return {
        status: 'unavailable',
        reason: `trained-readout-v1.json failed validation against the loaded graph: ${error instanceof Error ? error.message : String(error)}`
      };
    }
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
    const loadReadout = this.options.loadTrainedReadout ?? loadTrainedReadoutArtifact;
    const loadNull = this.options.loadRewiringNull ?? loadRewiringNull;
    const loadExplanation = this.options.loadNullExplanation ?? loadNullExplanation;
    const dataBaseUrl = `${import.meta.env.BASE_URL}data`;
    let artifacts: LoadedArenaArtifacts;
    try {
      artifacts = await load(dataBaseUrl);
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
    this.options.callbacks.onManifest(artifacts.manifest, artifacts.parsedBiological);

    // WP4: fire-and-forget, deliberately not awaited here (unlike the
    // trained-readout load just below) — "loading must not block Start"
    // means this must not sit in this method's own `await` chain ahead of
    // Worker construction. `loadRewiringNull` documents itself as "never
    // throws", but the leading `.catch` enforces that contract at the call
    // site too (dual review, Important — mirrors `App.svelte`'s own
    // `onManifest` handler, which added the equivalent `.catch` around
    // `loadPositions` for the same reason): without it, an unexpected throw
    // anywhere in the loader's chain would become an unhandled rejection and
    // leave `rewiringNullStatus` `undefined` forever (the ledger row stuck
    // on "Loading…"). The trailing `.catch` guards the *callback* instead —
    // `onRewiringNull` is host code (`App.svelte`), and a throw there would
    // otherwise also become an unhandled rejection with no error reported
    // anywhere.
    // `Promise.resolve().then(...)` rather than calling `loadNull` directly:
    // the production `loadRewiringNull` is `async` and can never throw
    // synchronously, but `loadNull` here can also be a test-injected
    // `ExperimentControllerOptions.loadRewiringNull` double, which is only
    // typed as returning a `Promise` — nothing stops a non-async double from
    // throwing before it ever produces one. Without this wrapper, that throw
    // would propagate out of `initialize()` synchronously, after `onManifest`
    // has already fired, bypassing both `.catch`es below entirely (round-2
    // dual review, Suggestion).
    // Never rejects (`loadNull` failures are converted to a resolved
    // `'unavailable'` status right here) — both `nullLoad.then(...)` chains
    // below fork off this *settled* promise independently, rather than one
    // chaining onto the other's own `.then((result) => onRewiringNull(...))`
    // step. That independence matters (round-2 dual review, Important): an
    // earlier version chained the null-explanation load directly after the
    // `onRewiringNull(result)` callback call, so a throwing `onRewiringNull`
    // host callback (host code, e.g. `App.svelte`) silently skipped the
    // null-explanation load too, contradicting this method's own "attempted
    // unconditionally" comment below. Forking both chains off `nullLoad`
    // instead means the two host callbacks (`onRewiringNull`,
    // `onNullExplanation`) can never take each other down.
    const nullLoad = Promise.resolve()
      .then(() => loadNull(artifacts.manifest, dataBaseUrl))
      .catch(
        // `'unavailable'`, not `'invalid'` (thermo review, Suggestion): this
        // is a genuine runtime/JS error — a throw somewhere in the loader's
        // chain, not a hash/shape/cross-check failure — so it must not be
        // described to a visitor as "failed verification" (`LedgerPanel.svelte`).
        (error: unknown): RewiringNullLoadResult => ({
          status: 'unavailable',
          reason: `unexpected error while loading the rewiring null: ${error instanceof Error ? error.message : String(error)}`
        })
      );

    void nullLoad
      .then((result) => {
        if (this.destroyed) return;
        this.options.callbacks.onRewiringNull(result);
      })
      .catch((error: unknown) => {
        if (this.destroyed) return;
        this.options.callbacks.onError(error instanceof Error ? error.message : String(error));
      });

    // WP4 of `.agents/plans/null-explanation` (`04-ledger-note.md`): "Load
    // after the null result" — sequenced after `nullLoad` *settles* (so it
    // never races ahead of the null histogram's own load and never issues a
    // duplicate fetch for the rewiring-null artifact), but forked off that
    // same promise rather than chained after the `onRewiringNull` callback
    // above (see the comment on `nullLoad` for why). `loadExplanation` only
    // needs `manifest`/`dataBaseUrl` (its own cross-check re-reads
    // `manifest.rewiringNull.sha256` directly, not the resolved
    // `RewiringNullLoadResult`), so it is attempted here unconditionally,
    // independent of whichever status the null load itself resolved to —
    // and independent of whether `onRewiringNull` throws.
    void nullLoad
      .then(() =>
        this.destroyed
          ? undefined
          : loadExplanation(artifacts.manifest, dataBaseUrl).catch(
              // `'unavailable'`, not `'invalid'` (mirrors `nullLoad`'s own
              // catch just above, and `NullExplanationLoadResult`'s doc
              // comment): a genuine runtime/JS error is not a verification
              // failure, and `LedgerPanel.svelte` renders `'invalid'` as
              // "Explanation failed verification".
              (error: unknown): NullExplanationLoadResult => ({
                status: 'unavailable',
                reason: `unexpected error while loading the null explanation: ${error instanceof Error ? error.message : String(error)}`
              })
            )
      )
      .then((result) => {
        if (this.destroyed || result === undefined) return;
        this.options.callbacks.onNullExplanation(result);
      })
      .catch((error: unknown) => {
        if (this.destroyed) return;
        this.options.callbacks.onError(error instanceof Error ? error.message : String(error));
      });

    // Trained-readout artifact: optional relative to the required arena
    // graph artifacts above — `loadTrainedReadoutArtifact` never throws, and
    // a missing/corrupt/mismatched artifact only ever disables the Trained
    // decoder (see `TrainedReadoutLoadResult`'s and `validateTrainedReadout`'s
    // doc comments), never `initialize()` itself.
    this.trainedReadout = this.validateTrainedReadout(
      await loadReadout(dataBaseUrl),
      artifacts.parsedBiological,
      artifacts.rewired
    );
    if (this.destroyed) return;
    this.options.callbacks.onTrainedReadoutStatus(this.trainedReadout);

    try {
      const left = this.options.createWorker();
      const right = this.options.createWorker();
      this.workerClients = { left: createWorkerClient(left), right: createWorkerClient(right) };

      const [leftBinding, rightBinding] = await Promise.all([
        createWorkerAgentBinding(
          this.workerClients.left,
          buildGraphBufferForMode(this.biologicalGraphBuffer, this.rewiredGraphBuffer, this.topology.left),
          this.topology.left,
          graphBinarySha256ForMode(this.manifest, this.topology.left),
          this.readoutWeightsForMode(this.topology.left)
        ),
        createWorkerAgentBinding(
          this.workerClients.right,
          buildGraphBufferForMode(this.biologicalGraphBuffer, this.rewiredGraphBuffer, this.topology.right),
          this.topology.right,
          graphBinarySha256ForMode(this.manifest, this.topology.right),
          this.readoutWeightsForMode(this.topology.right)
        )
      ]);
      if (this.destroyed) return;

      this.runner = new ExperimentRunner({
        seed: this.options.seed,
        totalTicks: this.options.totalTicks,
        agents: { left: leftBinding, right: rightBinding },
        targetTickIntervalMs: this.options.targetTickIntervalMs,
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
    // A no-op while a decoder switch is in flight — see `DecoderSwitch`'s
    // doc comment for the race this closes (dual review, round 1).
    if (this.decoderSwitch.isInFlight) return;
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
          const binding = await createWorkerAgentBinding(
            client,
            buffer,
            mode,
            graphBinarySha256,
            this.readoutWeightsForMode(mode)
          );
          if (this.destroyed || !this.runner) return;
          this.runner.setAgentBinding(agentId, binding);
          // A fresh `init` always starts a Worker back at `decoder:
          // 'authored'` (`neural.worker.ts`'s "authored-by-default" note),
          // even though `readout` was just passed again above and even
          // though the app may currently be in Trained mode. Unlike the
          // activity re-apply below, this can safely be awaited: it runs
          // strictly after this same chain's own `createWorkerAgentBinding`
          // (and therefore this arm's `init`) has already resolved, so there
          // is no init-window race to stay ahead of the way the activity
          // re-apply has to. `this.decoder === 'trained'` only when
          // `this.trainedReadout.status === 'ok'` (the only path
          // `setDecoder` allows it), so `readoutWeightsForMode(mode)` above
          // is guaranteed defined whenever this branch runs.
          if (this.decoder === 'trained') {
            await client.setDecoder('trained');
            if (this.destroyed || !this.runner) return;
          }
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

  /**
   * Switch both arms' decoder together and reset the run to tick 0 at the
   * current seed (`.agents/plans/trained-readout/06-browser-integration.md`'s
   * "Behavior and invariants": both agents always share one decoder;
   * switching is only allowed while the run is not `running`, and it resets
   * to tick 0). A no-op — silently, like `changeTopology`'s own guards — when:
   * no runner exists yet; `decoder` is already selected; the run is
   * currently `running`; a topology switch is in flight for either arm
   * (avoids racing this call's `set-decoder` against that switch's own
   * dispose/init window on the same Worker); or `'trained'` is requested
   * while the trained-readout artifact is unavailable or failed validation
   * (`getTrainedReadoutStatus()`) — the host is expected to have already
   * disabled the Trained control in that case, this is defense-in-depth.
   *
   * Sends `set-decoder` to both arms' Workers *before* resetting: the
   * Worker's own `set-decoder` handler does not touch neural state (see
   * `SetDecoderWorkerRequest`'s doc comment), so applying it first and
   * resetting second means a failure on either arm's `set-decoder` leaves
   * the run in `runner.fail()`'s `error` state rather than silently having
   * reset to a decoder selection that only landed on one arm.
   */
  async setDecoder(decoder: DecoderKind): Promise<void> {
    if (!this.runner || !this.workerClients) return;
    // `canStartDecoderSwitch()` covers both halves of `DecoderSwitch`'s
    // invariant in one call: no other decoder switch already in flight
    // (including a second, overlapping call to `setDecoder` itself —
    // round-2 dual review; `decoder === this.decoder` alone would not
    // exclude a same-target overlap, since the first call has not written
    // `this.decoder` yet when the second call's guards run), and no
    // topology switch in flight on either arm (avoids racing this call's
    // `set-decoder` against that switch's own dispose/init window on the
    // same Worker).
    if (!this.decoderSwitch.canStartDecoderSwitch()) return;
    if (decoder === this.decoder) return;
    if (this.runner.getStatus() === 'running') return;
    if (decoder === 'trained' && this.trainedReadout?.status !== 'ok') return;

    // Acquired synchronously, before the first `await` below — this is what
    // lets `changeTopology`'s own `decoderSwitch.isInFlight` guard actually
    // exclude a topology switch starting anywhere during this call's Worker
    // round trip, not just after `this.decoder` is written at the end. See
    // `DecoderSwitch`'s doc comment for the race this closes.
    this.decoderSwitch.acquire();
    try {
      await Promise.all([this.workerClients.left.setDecoder(decoder), this.workerClients.right.setDecoder(decoder)]);
      if (this.destroyed || !this.runner) return;
      this.decoder = decoder;
      this.runner.reset();
      this.options.callbacks.onDecoderApplied(decoder);
      this.options.callbacks.onTelemetry(this.runner.getTelemetry());
    } catch (error) {
      if (this.destroyed) return;
      // Route through the runner so the host and the runner's own status can
      // never disagree, matching `changeTopology`'s own failure handling.
      this.runner?.fail(error);
    } finally {
      this.decoderSwitch.release();
    }
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
