import { decodeAction } from '../arena/actions';
import { ARENA_CONFIG, type ArenaConfig } from '../arena/config';
import { createExperimentReplayExport, type ExperimentReplayExport, type ExperimentTraceEntry } from '../arena/replay';
import { observeAgent } from '../arena/sensors';
import type { ActionsByAgent, AgentId, ArenaSnapshot, WorldState } from '../arena/types';
import { createSnapshot, createWorld, stepWorld } from '../arena/world';
import { NEURAL_SUBSTEPS_PER_TICK } from '../connectome/constants';
import type { GraphMode } from '../connectome/format';
import type { NeuralTelemetry } from '../connectome/telemetry';
import { MAX_SUBSTEPS_PER_TICK } from '../worker/protocol';
import { canPause, canReset, canResume, canStart, transition, type ExperimentStatus } from './state';

/** Re-exported so existing call sites (this module, its tests) keep working; see `connectome/constants.ts` for why the value itself lives there. */
export { NEURAL_SUBSTEPS_PER_TICK };

/**
 * The tick-driven async closed-loop orchestrator (WP6 item 3): observe both
 * arms -> send one neural step per arm -> await both -> decode -> `stepWorld`
 * exactly once. See `docs/architecture.md`'s closed-loop contract.
 *
 * This module is intentionally step-function-agnostic: `AgentStepFn` is
 * satisfied identically by a synchronous CPU-oracle wrapper (see
 * `tests/unit/experiment-runner.test.ts`) and by `src/lib/worker/client.ts`'s
 * `WorkerClient.step`, which is how `src/App.svelte` actually drives it in
 * the browser. Determinism does not depend on which one is plugged in, or on
 * how long either call takes — see the "in-flight tick" discussion below.
 */

export interface AgentStepInput {
  channelValues: readonly number[];
  substeps: number;
}

export interface AgentStepResult {
  actionFeatures: readonly number[];
  telemetry: NeuralTelemetry;
  /**
   * The full per-neuron rate vector for this step, present only while
   * activity streaming is enabled for this arm (see
   * `ExperimentRunner#setActivityStreaming`). A `WorkerClient`-backed
   * binding passes `StepWorkerSuccess.rates` straight through; the
   * oracle/CPU binding has no analogous opt-in and never sets this.
   */
  rates?: Float32Array;
}

/** One neural control step for one arm. Must resolve with the outputs for exactly the observation it was given — never a stale/reused action. */
export type AgentStepFn = (input: AgentStepInput) => Promise<AgentStepResult>;

export interface AgentRunnerInfo {
  topology: GraphMode;
  neuronCount: number;
  edgeCount: number;
  /**
   * sha256 of the manifest-verified compiled graph artifact that produced
   * this arm's current binding, when one exists. Absent for the
   * runtime-derived 'disconnected' control, which is built on the fly from
   * an already-loaded arm rather than shipped as its own verified artifact
   * (see `connectome/format.ts#createDisconnectedGraph`) — there is no
   * separate manifest entry for it to cite. Threaded into
   * `ExperimentReplayExport` so a downloaded replay can self-attest to
   * exactly which compiled artifact produced it, without embedding the
   * artifact itself.
   */
  graphBinarySha256?: string;
}

export interface AgentBinding {
  step: AgentStepFn;
  /**
   * Zero the arm's neural state; called on `reset()`. Must apply strictly
   * after any previously-issued `step` has settled (FIFO) and must leave
   * the arm's neural state exactly as freshly initialized — a `step` that
   * is still in flight when `reset` is called must not run (or must not
   * observably affect state) after `reset` takes effect. A real Worker
   * satisfies this for free via `postMessage`'s FIFO delivery order; an
   * in-thread binding must serialize its own calls to satisfy it (see
   * `bindings.ts#createOracleAgentBinding`).
   */
  reset: () => Promise<void>;
  info: AgentRunnerInfo;
  /**
   * Toggle whether this arm's `step` results include `AgentStepResult.rates`
   * for the anatomical activity view. Optional: the oracle/CPU binding
   * (`bindings.ts#createOracleAgentBinding`) omits it entirely (the "not
   * supported" contract — `ExperimentRunner#setActivityStreaming` treats a
   * missing `setActivity` as a no-op for that arm, not an error). A
   * `WorkerClient`-backed binding implements it with `client.setActivity`.
   */
  setActivity?: (enabled: boolean) => Promise<void>;
}

export interface AgentTelemetry extends AgentRunnerInfo {
  foodPickups: number;
  hazardContacts: number;
  distanceTravelled: number;
  movementScore: number;
  activeFraction: number;
  meanRate: number;
  lastStepLatencyMs: number;
  medianStepLatencyMs: number;
}

export interface ExperimentTelemetry {
  status: ExperimentStatus;
  tick: number;
  totalTicks: number;
  elapsedSimulatedSeconds: number;
  seed: number;
  /** True when the most recently completed tick took longer than the real-time pacing budget: the sim is running slower than 30Hz, not skipping or reusing actions. */
  behindRealtime: boolean;
  agents: Record<AgentId, AgentTelemetry>;
}

const LATENCY_SAMPLE_CAP = 300;

/** Sorted-copy median. `[]` -> 0 rather than `NaN`, so a fresh telemetry read before the first tick is a plain, displayable number. */
export const computeMedian = (values: readonly number[]): number => {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

export interface ExperimentRunnerOptions {
  seed: number;
  /** 2,700 for the canonical 90-second run at 30 Hz; any positive integer is accepted for tests. */
  totalTicks: number;
  agents: Record<AgentId, AgentBinding>;
  arenaConfig?: Readonly<ArenaConfig>;
  substepsPerTick?: number;
  /**
   * Real-time pacing budget per tick in ms. Defaults to the arena's own
   * fixed step (`1000/30`): when a tick's neural round trip finishes faster
   * than that, the runner waits out the remainder so a compute-fast browser
   * doesn't replay the whole 90-second experiment in a fraction of a
   * second. `0` disables pacing entirely (ticks run back-to-back as fast as
   * `AgentStepFn` resolves) — used by unit tests so a 2,700-tick run
   * finishes in milliseconds instead of 90 real seconds.
   */
  targetTickIntervalMs?: number;
  onStatusChange?: (status: ExperimentStatus) => void;
  onTelemetry?: (telemetry: ExperimentTelemetry) => void;
  onError?: (error: Error) => void;
}

/**
 * Owns one closed-loop run: the world, both arms' step functions, the
 * explicit state machine (`./state.ts`), and the tick loop itself.
 *
 * Backlog/backpressure: `runLoop` is a single `while` loop that always
 * `await`s one tick's `Promise.all` before even *building* the next tick's
 * observation — there is no queue, so at most one tick is ever in flight.
 * If a step call is slow, the next tick simply starts later; nothing is
 * skipped and no action is ever reused across ticks (`behindRealtime`
 * reports this in telemetry instead).
 *
 * `reset()`/`dispose()` bump `generation`; an in-flight tick that resolves
 * after a reset/dispose raced it discards its own result instead of
 * clobbering the freshly-reset `world` — see `runOneTick`.
 */
export class ExperimentRunner {
  private readonly arenaConfig: Readonly<ArenaConfig>;
  private readonly substepsPerTick: number;
  private readonly targetTickIntervalMs: number;
  private readonly options: ExperimentRunnerOptions;

  private agents: Record<AgentId, AgentBinding>;
  private seed: number;
  private world: WorldState;
  private status: ExperimentStatus = 'ready';
  private generation = 0;
  private disposed = false;
  private loopActive = false;
  private behindRealtime = false;
  private lastTickWallClockMs: number | undefined;
  private lastNeuralTelemetry: Record<AgentId, NeuralTelemetry> | undefined;
  private latencies: Record<AgentId, number[]> = { left: [], right: [] };
  private lastLatencyMs: Record<AgentId, number> = { left: 0, right: 0 };
  private trace: ExperimentTraceEntry[] = [];
  private pendingDelayFinish: (() => void) | undefined;
  /**
   * Whether the anatomical activity view has asked for full per-neuron
   * rates. Not generation-scoped like `latestRates`/`lastNeuralTelemetry`
   * below: per the plan, `reset()` keeps the current streaming setting —
   * resetting the world/neural state is orthogonal to whether the view is
   * currently open — so this field is untouched by `reset()`/`dispose()`.
   */
  private activityStreaming = false;
  /**
   * Bumped by every `setActivityStreaming` call (enable or disable alike),
   * and captured by `runOneTick` at the start of each tick. A `step`
   * request is posted to the Worker (and its round trip can straddle a
   * later `setActivityStreaming` call, since the two are otherwise
   * independent async operations) before this generation-style guard is
   * consulted again when the result comes back — see `runOneTick`'s
   * `activityEpochAtStart` check. Without this, a tick issued just before a
   * disable but resolving just after it would still carry `rates` (the
   * Worker computed it while activity was still on) and get written into
   * `latestRates` even though the toggle already cleared it — a real,
   * reproduced race: dual review both independently confirmed a paused run
   * still showing a stale rates snapshot up to a full tick's Worker latency
   * after `setActivityStreaming(false)` resolved. Distinct from
   * `generation` (bumped only by `reset()`/`dispose()`/`setAgentBinding`):
   * a streaming toggle alone must invalidate in-flight rates without also
   * discarding the tick's `actionFeatures`/world-step effects.
   */
  private activityEpoch = 0;
  /**
   * Latest per-arm full rate vector, replaced (not accumulated) each tick.
   * `undefined` until `setActivityStreaming(true)` has both taken effect and
   * a subsequent tick has actually run; cleared back to `undefined` on
   * every `setActivityStreaming` call (enable or disable — an enable must
   * never briefly show a frame left over from before the view was closed)
   * and on `reset()` (a stale pre-reset snapshot must never be mistaken for
   * current state).
   */
  private latestRates: Record<AgentId, Float32Array | undefined> = { left: undefined, right: undefined };

  constructor(options: ExperimentRunnerOptions) {
    if (!Number.isInteger(options.totalTicks) || options.totalTicks <= 0) {
      throw new Error('ExperimentRunner: totalTicks must be a positive integer');
    }
    const substeps = options.substepsPerTick ?? NEURAL_SUBSTEPS_PER_TICK;
    if (!Number.isInteger(substeps) || substeps <= 0 || substeps > MAX_SUBSTEPS_PER_TICK) {
      throw new Error(`ExperimentRunner: substepsPerTick must be in [1, ${MAX_SUBSTEPS_PER_TICK}]`);
    }
    this.options = options;
    this.arenaConfig = options.arenaConfig ?? ARENA_CONFIG;
    this.substepsPerTick = substeps;
    this.targetTickIntervalMs = options.targetTickIntervalMs ?? this.arenaConfig.fixedDeltaSeconds * 1000;
    this.agents = options.agents;
    this.seed = options.seed;
    this.world = createWorld(this.seed, this.arenaConfig);
  }

  getStatus(): ExperimentStatus {
    return this.status;
  }

  getWorld(): Readonly<WorldState> {
    return this.world;
  }

  /** Interpolated presentation data for the renderer, decoupled from tick cadence — see `arena/world.ts#createSnapshot`. */
  getSnapshot(nowMs: number): ArenaSnapshot {
    const interval = this.targetTickIntervalMs > 0 ? this.targetTickIntervalMs : this.arenaConfig.fixedDeltaSeconds * 1000;
    const alpha =
      this.lastTickWallClockMs === undefined ? 1 : Math.max(0, Math.min(1, (nowMs - this.lastTickWallClockMs) / interval));
    return createSnapshot(this.world, alpha, this.arenaConfig);
  }

  getTelemetry(): ExperimentTelemetry {
    const agentTelemetry = (agentId: AgentId): AgentTelemetry => {
      const agent = this.world.agents.find((candidate) => candidate.id === agentId);
      const neural = this.lastNeuralTelemetry?.[agentId];
      return {
        ...this.agents[agentId].info,
        foodPickups: agent?.score.foodPickups ?? 0,
        hazardContacts: agent?.score.hazardContacts ?? 0,
        distanceTravelled: agent?.score.distanceTravelled ?? 0,
        movementScore: agent?.score.movementScore ?? 0,
        activeFraction: neural?.activeFraction ?? 0,
        meanRate: neural?.meanRate ?? 0,
        lastStepLatencyMs: this.lastLatencyMs[agentId],
        medianStepLatencyMs: computeMedian(this.latencies[agentId])
      };
    };
    return {
      status: this.status,
      tick: this.world.tick,
      totalTicks: this.options.totalTicks,
      elapsedSimulatedSeconds: this.world.timeSeconds,
      seed: this.seed,
      behindRealtime: this.behindRealtime,
      agents: { left: agentTelemetry('left'), right: agentTelemetry('right') }
    };
  }

  getReplayExport(): ExperimentReplayExport {
    return createExperimentReplayExport(this.world, {
      topology: { left: this.agents.left.info.topology, right: this.agents.right.info.topology },
      graphBinarySha256: {
        left: this.agents.left.info.graphBinarySha256,
        right: this.agents.right.info.graphBinarySha256
      },
      substepsPerTick: this.substepsPerTick,
      totalTicks: this.options.totalTicks,
      trace: this.trace
    });
  }

  isActivityStreaming(): boolean {
    return this.activityStreaming;
  }

  /**
   * Latest full per-neuron rate vector for `agentId`, or `undefined` until
   * streaming is enabled and a tick has run since. Replaced (a new
   * `Float32Array` reference every tick — `!==` against a previously-read
   * value is a cheap "is this fresh" check), never accumulated or mutated
   * in place. The caller must treat the returned array as read-only: it may
   * be the same object a later call still returns unchanged (between
   * ticks), so mutating it would corrupt what a subsequent read reports as
   * "this tick's" data.
   */
  getLatestRates(agentId: AgentId): Float32Array | undefined {
    return this.latestRates[agentId];
  }

  /**
   * Enable/disable full per-neuron rate streaming for the anatomical
   * activity view, for both arms. Allowed in any runner state (unlike
   * `setAgentBinding`) — the view can open/close at any point in a run.
   *
   * Sets `this.activityStreaming` synchronously, before any `await`, so
   * `isActivityStreaming()` reflects the caller's intent immediately —
   * `ExperimentController#changeTopology` (`controller.ts`) reads it right
   * after rebuilding an arm's binding to decide whether to re-issue
   * `setActivity(true)` on the fresh binding, and must see the *requested*
   * state even if this call's own `AgentBinding#setActivity` round trips are
   * still in flight elsewhere. The flag is deliberately never reverted by a
   * failed round trip below: a binding's `setActivity` can legitimately
   * reject with `not-initialized` while its Worker is mid topology-switch
   * (dispose -> init — see `controller.ts#changeTopology`), and the
   * controller's own re-apply step is what recovers that arm; this method
   * failing quietly here must not un-set what the caller asked for.
   *
   * Never rejects to the caller: each binding's `setActivity` is awaited and
   * caught independently, so one arm's failure never affects the other's,
   * and this method's own returned `Promise<void>` always resolves.
   */
  async setActivityStreaming(enabled: boolean): Promise<void> {
    if (this.disposed) return;
    const generationAtStart = this.generation;
    this.activityStreaming = enabled;
    // Bumped (and `latestRates` cleared) on *every* call, not just a
    // disable: an in-flight tick issued before this call can still resolve
    // with rates computed under the old setting — see `activityEpoch`'s doc
    // comment and `runOneTick`'s `activityEpochAtStart` check, which is what
    // actually keeps that stale result out of `latestRates`. Clearing on
    // enable too means a reopened view never briefly paints a frame left
    // over from before it was closed.
    this.activityEpoch += 1;
    this.latestRates = { left: undefined, right: undefined };
    const bindings = this.agents;
    await Promise.all(
      (['left', 'right'] as const).map(async (agentId) => {
        const setActivity = bindings[agentId].setActivity;
        if (!setActivity) return;
        try {
          await setActivity(enabled);
        } catch (error) {
          // A stale generation (a reset()/topology switch raced this call)
          // makes a rejection here expected and uninteresting — the bindings
          // this call was issued against may already be superseded. Only
          // log against the generation that actually issued this call, the
          // same discipline `reset()`'s own binding-reset catch uses
          // (`runner.ts`'s reset()); never call `this.fail()` here, since a
          // per-arm streaming-toggle failure is not a run failure.
          if (this.disposed || generationAtStart !== this.generation) return;
          console.error(`ExperimentRunner: setActivity(${enabled}) failed for agent ${agentId}`, error);
        }
      })
    );
  }

  /**
   * Swap an arm's binding (e.g. a topology change) between runs. Only valid
   * from `ready`/`finished` — not `running` or `paused` — and always
   * implies a `reset()`: a topology change mid-run would otherwise leave a
   * hybrid run (part of it under the old topology, part under the new one)
   * whose replay export and telemetry could only report one topology for
   * the whole run. Throws if the run is currently active or paused, or if
   * the runner has been disposed.
   */
  setAgentBinding(agentId: AgentId, binding: AgentBinding): void {
    if (this.disposed) throw new Error('ExperimentRunner: disposed');
    if (this.status !== 'ready' && this.status !== 'finished') {
      throw new Error(
        `ExperimentRunner: agent bindings can only change from ready/finished (was ${this.status}); pause and reset first`
      );
    }
    this.agents = { ...this.agents, [agentId]: binding };
    this.reset(this.seed);
  }

  /** Starts a fresh run from `ready`, or resumes from `paused`. No-op otherwise, or once disposed. */
  start(): void {
    if (this.disposed) return;
    if (canStart(this.status)) {
      this.setStatus(transition(this.status, { type: 'start' }));
    } else if (canResume(this.status)) {
      this.setStatus(transition(this.status, { type: 'resume' }));
    } else {
      return;
    }
    if (!this.loopActive) void this.runLoop();
  }

  pause(): void {
    if (this.disposed) return;
    if (!canPause(this.status)) return;
    this.setStatus(transition(this.status, { type: 'pause' }));
  }

  /** Fresh world (optionally a new seed) and zeroed neural state for both arms. Discards any in-flight tick's result. No-op once disposed. */
  reset(seed?: number): void {
    if (this.disposed) return;
    if (!canReset(this.status)) return;
    this.generation += 1;
    const resetGeneration = this.generation;
    this.pendingDelayFinish?.();
    this.seed = seed ?? this.seed;
    this.world = createWorld(this.seed, this.arenaConfig);
    this.lastTickWallClockMs = undefined;
    this.lastNeuralTelemetry = undefined;
    this.latestRates = { left: undefined, right: undefined };
    this.latencies = { left: [], right: [] };
    this.lastLatencyMs = { left: 0, right: 0 };
    this.trace = [];
    this.behindRealtime = false;
    this.setStatus(transition(this.status, { type: 'reset' }));
    const bindings = this.agents;
    void Promise.all([bindings.left.reset(), bindings.right.reset()]).catch((error: unknown) => {
      // A stale reset (superseded by a later reset(), or the runner was
      // disposed while this was in flight) must not fail a run that has
      // already moved on — only report against the generation that issued
      // this particular reset.
      if (this.disposed || resetGeneration !== this.generation) return;
      this.fail(error);
    });
  }

  /**
   * Report a runtime failure and move to `error`. The one path every
   * failure — a bad tick, a failed reset, a failed topology switch — is
   * expected to go through, so the state machine (`./state.ts`) never gets
   * bypassed and `onError`/`onStatusChange` always agree. No-op once
   * disposed: a disposed runner has nothing left to report to.
   */
  fail(error: unknown): void {
    if (this.disposed) return;
    this.setStatus(transition(this.status, { type: 'runtimeError' }));
    this.options.onError?.(error instanceof Error ? error : new Error(String(error)));
  }

  /** Stop the loop and stop accepting further transitions. Idempotent; does not touch agent Workers (the caller owns their lifecycle). */
  dispose(): void {
    this.generation += 1;
    this.disposed = true;
    this.pendingDelayFinish?.();
    // Nothing reads `getLatestRates` on a disposed runner in production, but
    // clearing it (rather than leaving a disposed runner's last-known array
    // reachable) avoids a caller ever mistaking it for live data.
    this.latestRates = { left: undefined, right: undefined };
  }

  private setStatus(next: ExperimentStatus): void {
    if (next === this.status) return;
    this.status = next;
    this.options.onStatusChange?.(next);
  }

  private recordLatency(agentId: AgentId, latencyMs: number): void {
    this.lastLatencyMs[agentId] = latencyMs;
    const samples = this.latencies[agentId];
    samples.push(latencyMs);
    if (samples.length > LATENCY_SAMPLE_CAP) samples.shift();
  }

  private pushTrace(): void {
    const byId = new Map(this.world.agents.map((agent) => [agent.id, agent]));
    this.trace.push({
      tick: this.world.tick,
      timeSeconds: this.world.timeSeconds,
      agents: {
        left: { ...byId.get('left')!.score },
        right: { ...byId.get('right')!.score }
      }
    });
  }

  private delay(ms: number): Promise<void> {
    return new Promise<void>((resolve) => {
      let finished = false;
      const finish = (): void => {
        if (finished) return;
        finished = true;
        this.pendingDelayFinish = undefined;
        resolve();
      };
      const timerId = setTimeout(finish, ms);
      this.pendingDelayFinish = (): void => {
        clearTimeout(timerId);
        finish();
      };
    });
  }

  /** Times one arm's `step` call independently, so per-arm latency telemetry reflects that arm's own round trip rather than whichever arm happened to settle last. */
  private async timedStep(
    agentId: AgentId,
    input: AgentStepInput
  ): Promise<{ result: AgentStepResult; latencyMs: number }> {
    const start = performance.now();
    const result = await this.agents[agentId].step(input);
    return { result, latencyMs: performance.now() - start };
  }

  /**
   * One observe -> step (both arms, concurrently, timed independently) ->
   * decode -> `stepWorld` round trip. Returns `discarded: true` (and leaves
   * `this.world` alone) if a `reset()`/`dispose()` happened while this
   * call's step requests were in flight — whether they ultimately resolved
   * or rejected — see the class doc comment. A rejection from the *current*
   * generation is rethrown so `runLoop` can report it.
   */
  private async runOneTick(): Promise<{ tick: number; discarded: boolean }> {
    const generationAtStart = this.generation;
    const activityEpochAtStart = this.activityEpoch;
    const world = this.world;
    const channelValues: Record<AgentId, readonly number[]> = {
      left: observeAgent(world, 'left', this.arenaConfig),
      right: observeAgent(world, 'right', this.arenaConfig)
    };

    let left: { result: AgentStepResult; latencyMs: number };
    let right: { result: AgentStepResult; latencyMs: number };
    try {
      [left, right] = await Promise.all([
        this.timedStep('left', { channelValues: channelValues.left, substeps: this.substepsPerTick }),
        this.timedStep('right', { channelValues: channelValues.right, substeps: this.substepsPerTick })
      ]);
    } catch (error) {
      // A step can reject instead of resolving (e.g. the Worker was
      // disposed mid-switch, or terminated). If a reset()/dispose() already
      // superseded this tick, that rejection is expected and not a failure
      // of the *current* run — swallow it the same way a late resolution
      // would be discarded. Otherwise it is a real failure; propagate it.
      if (generationAtStart !== this.generation || this.disposed) {
        return { tick: this.world.tick, discarded: true };
      }
      throw error;
    }
    if (generationAtStart !== this.generation || this.disposed) {
      return { tick: this.world.tick, discarded: true };
    }
    this.recordLatency('left', left.latencyMs);
    this.recordLatency('right', right.latencyMs);
    // Replaced, not accumulated, and only when a result actually carries
    // rates. Also gated on `activityEpochAtStart === this.activityEpoch`
    // (dual review, confirmed by reproduction): this tick's `step` requests
    // were posted while streaming was on, but `setActivityStreaming` can
    // toggle (and clear `latestRates`) while this same tick's round trip is
    // still in flight — the Worker already computed `rates` for this result
    // under the *old* setting, so writing it back here would resurrect a
    // snapshot the toggle just cleared (or, on a fast disable-then-enable,
    // one the current enable never asked for). A binding with streaming
    // disabled the whole tick, or mid topology-switch before the
    // controller's re-apply lands, simply carries no `rates` at all, so
    // there is nothing to (incorrectly) skip.
    if (activityEpochAtStart === this.activityEpoch) {
      if (left.result.rates) this.latestRates.left = left.result.rates;
      if (right.result.rates) this.latestRates.right = right.result.rates;
    }

    const actions: ActionsByAgent = {
      left: decodeAction(left.result.actionFeatures),
      right: decodeAction(right.result.actionFeatures)
    };
    this.world = stepWorld(world, actions, this.arenaConfig);
    this.lastNeuralTelemetry = { left: left.result.telemetry, right: right.result.telemetry };
    this.lastTickWallClockMs = performance.now();
    this.pushTrace();
    return { tick: this.world.tick, discarded: false };
  }

  private async runLoop(): Promise<void> {
    if (this.loopActive) return;
    this.loopActive = true;
    while (this.status === 'running') {
      const tickStart = performance.now();
      let result: { tick: number; discarded: boolean };
      try {
        result = await this.runOneTick();
      } catch (error) {
        if (!this.disposed) this.fail(error);
        break;
      }
      if (this.disposed) break;
      if (result.discarded) {
        // A reset()/dispose() already changed `status`/`world`; re-check the
        // loop condition rather than emitting telemetry for a tick that
        // never actually applied.
        continue;
      }
      const elapsed = performance.now() - tickStart;
      this.behindRealtime = this.targetTickIntervalMs > 0 && elapsed > this.targetTickIntervalMs;
      this.options.onTelemetry?.(this.getTelemetry());

      const nextStatus = transition(this.status, {
        type: 'tickCompleted',
        tick: result.tick,
        totalTicks: this.options.totalTicks
      });
      this.setStatus(nextStatus);
      if (this.status !== 'running') break;

      if (this.targetTickIntervalMs > 0 && elapsed < this.targetTickIntervalMs) {
        await this.delay(this.targetTickIntervalMs - elapsed);
        if (this.disposed || this.status !== 'running') break;
      }
    }
    this.loopActive = false;
  }
}
