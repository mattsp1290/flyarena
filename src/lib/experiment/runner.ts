import { decodeAction } from '../arena/actions';
import { ARENA_CONFIG, type ArenaConfig } from '../arena/config';
import { createExperimentReplayExport, type ExperimentReplayExport, type ExperimentTraceEntry } from '../arena/replay';
import { observeAgent } from '../arena/sensors';
import type { ActionsByAgent, AgentId, ArenaSnapshot, WorldState } from '../arena/types';
import { createSnapshot, createWorld, stepWorld } from '../arena/world';
import type { GraphMode } from '../connectome/format';
import type { NeuralTelemetry } from '../connectome/telemetry';
import { MAX_SUBSTEPS_PER_TICK } from '../worker/protocol';
import { canPause, canReset, canResume, canStart, transition, type ExperimentStatus } from './state';

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

/**
 * Neural substeps run per world tick, for every arm. A world tick is
 * `1/30`s of simulated time (`ARENA_CONFIG.fixedDeltaSeconds`); this many
 * `stepModel` calls run against the *same* held-constant observation before
 * the outputs are decoded and the world advances — see
 * `src/lib/connectome/model.ts#runSubsteps`. Chosen as a small constant well
 * under `MAX_SUBSTEPS_PER_TICK` (64): frequent enough re-observation (every
 * 1/30s) matters more for this arena than deep intra-tick integration, and 4
 * keeps worst-case per-tick Worker latency low. Exported so a later training
 * readout bean imports this exact value rather than restating it (see the
 * bean's own note in `implementation.md`).
 */
export const NEURAL_SUBSTEPS_PER_TICK = 4;

if (NEURAL_SUBSTEPS_PER_TICK > MAX_SUBSTEPS_PER_TICK) {
  throw new Error(
    `NEURAL_SUBSTEPS_PER_TICK (${NEURAL_SUBSTEPS_PER_TICK}) exceeds MAX_SUBSTEPS_PER_TICK (${MAX_SUBSTEPS_PER_TICK})`
  );
}

export interface AgentStepInput {
  channelValues: readonly number[];
  substeps: number;
}

export interface AgentStepResult {
  actionFeatures: readonly number[];
  telemetry: NeuralTelemetry;
}

/** One neural control step for one arm. Must resolve with the outputs for exactly the observation it was given — never a stale/reused action. */
export type AgentStepFn = (input: AgentStepInput) => Promise<AgentStepResult>;

export interface AgentRunnerInfo {
  topology: GraphMode;
  neuronCount: number;
  edgeCount: number;
}

export interface AgentBinding {
  step: AgentStepFn;
  /** Zero the arm's neural state; called on `reset()`. */
  reset: () => Promise<void>;
  info: AgentRunnerInfo;
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
      substepsPerTick: this.substepsPerTick,
      totalTicks: this.options.totalTicks,
      trace: this.trace
    });
  }

  /** Swap an arm's binding (e.g. a topology change) while idle. Throws if the run is currently active. */
  setAgentBinding(agentId: AgentId, binding: AgentBinding): void {
    if (this.status === 'running') {
      throw new Error('ExperimentRunner: cannot change an agent binding while running; pause or reset first');
    }
    this.agents = { ...this.agents, [agentId]: binding };
  }

  /** Starts a fresh run from `ready`, or resumes from `paused`. No-op otherwise. */
  start(): void {
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
    if (!canPause(this.status)) return;
    this.setStatus(transition(this.status, { type: 'pause' }));
  }

  /** Fresh world (optionally a new seed) and zeroed neural state for both arms. Discards any in-flight tick's result. */
  reset(seed?: number): void {
    if (!canReset(this.status)) return;
    this.generation += 1;
    this.pendingDelayFinish?.();
    this.seed = seed ?? this.seed;
    this.world = createWorld(this.seed, this.arenaConfig);
    this.lastTickWallClockMs = undefined;
    this.lastNeuralTelemetry = undefined;
    this.latencies = { left: [], right: [] };
    this.lastLatencyMs = { left: 0, right: 0 };
    this.trace = [];
    this.behindRealtime = false;
    this.setStatus(transition(this.status, { type: 'reset' }));
    const bindings = this.agents;
    void Promise.all([bindings.left.reset(), bindings.right.reset()]).catch((error: unknown) => {
      this.setStatus('error');
      this.options.onError?.(error instanceof Error ? error : new Error(String(error)));
    });
  }

  /** Stop the loop and stop accepting further transitions. Idempotent; does not touch agent Workers (the caller owns their lifecycle). */
  dispose(): void {
    this.generation += 1;
    this.disposed = true;
    this.pendingDelayFinish?.();
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

  /**
   * One observe -> step (both arms, concurrently) -> decode -> `stepWorld`
   * round trip. Returns `discarded: true` (and leaves `this.world` alone) if
   * a `reset()`/`dispose()` happened while this call's `Promise.all` was in
   * flight — see the class doc comment.
   */
  private async runOneTick(): Promise<{ tick: number; discarded: boolean }> {
    const generationAtStart = this.generation;
    const world = this.world;
    const channelValues: Record<AgentId, readonly number[]> = {
      left: observeAgent(world, 'left', this.arenaConfig),
      right: observeAgent(world, 'right', this.arenaConfig)
    };
    const stepStart: Record<AgentId, number> = { left: performance.now(), right: performance.now() };
    const [leftResult, rightResult] = await Promise.all([
      this.agents.left.step({ channelValues: channelValues.left, substeps: this.substepsPerTick }),
      this.agents.right.step({ channelValues: channelValues.right, substeps: this.substepsPerTick })
    ]);
    if (generationAtStart !== this.generation || this.disposed) {
      return { tick: this.world.tick, discarded: true };
    }
    const nowMs = performance.now();
    this.recordLatency('left', nowMs - stepStart.left);
    this.recordLatency('right', nowMs - stepStart.right);

    const actions: ActionsByAgent = {
      left: decodeAction(leftResult.actionFeatures),
      right: decodeAction(rightResult.actionFeatures)
    };
    this.world = stepWorld(world, actions, this.arenaConfig);
    this.lastNeuralTelemetry = { left: leftResult.telemetry, right: rightResult.telemetry };
    this.lastTickWallClockMs = nowMs;
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
        this.setStatus(transition(this.status, { type: 'runtimeError' }));
        this.options.onError?.(error instanceof Error ? error : new Error(String(error)));
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
