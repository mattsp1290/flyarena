import type { GraphMode } from '../connectome/format';
import {
  ARENA_CONFIG,
  retainArenaConfig,
  validateRetainedArenaConfig,
  type ArenaConfig
} from './config';
import type { ActionsByAgent, AgentId, ReadonlyWorldState, WorldState } from './types';
import { createWorld, stepWorld } from './world';

export interface ReplayAgentSummary {
  id: string;
  position: { x: number; z: number };
  velocity: { x: number; z: number };
  heading: number;
  activeHazardIds: string[];
  score: {
    foodPickups: number;
    hazardContacts: number;
    distanceTravelled: number;
    movementScore: number;
  };
}

export interface ReplaySummary {
  schemaVersion: 2;
  configFingerprint: string;
  seed: number;
  rngState: number;
  ticks: number;
  timeSeconds: number;
  agents: ReplayAgentSummary[];
  foods: Array<{
    id: string;
    position: { x: number; z: number };
    radius: number;
    respawns: number;
  }>;
  hazards: Array<{
    id: string;
    position: { x: number; z: number };
    velocity: { x: number; z: number };
    radius: number;
  }>;
}

export interface ReplayResult {
  summary: ReplaySummary;
  serialized: string;
  /** Lowercase, zero-padded 64-bit FNV-1a digest of `serialized`. */
  hash: string;
}

/**
 * Extract the deterministic simulation result. Interpolation-only previous
 * transforms are intentionally excluded so rendering cannot affect a replay.
 */
export const createReplaySummary = (world: Readonly<WorldState>): ReplaySummary => {
  validateRetainedArenaConfig(world.config, world.configFingerprint);
  return {
    schemaVersion: 2,
    configFingerprint: world.configFingerprint,
    seed: world.seed,
    rngState: world.rngState,
    ticks: world.tick,
    timeSeconds: world.timeSeconds,
    agents: world.agents.map((agent) => ({
      id: agent.id,
      position: { ...agent.position },
      velocity: { ...agent.velocity },
      heading: agent.heading,
      activeHazardIds: [...agent.activeHazardIds].sort(),
      score: { ...agent.score }
    })),
    foods: world.foods.map((food) => ({
      id: food.id,
      position: { ...food.position },
      radius: food.radius,
      respawns: food.respawns
    })),
    hazards: world.hazards.map((hazard) => ({
      id: hazard.id,
      position: { ...hazard.position },
      velocity: { ...hazard.velocity },
      radius: hazard.radius
    }))
  };
};

const canonicalize = (value: unknown): unknown => {
  if (Array.isArray(value)) return value.map(canonicalize);
  if (value !== null && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value)
        .sort(([left], [right]) => (left < right ? -1 : left > right ? 1 : 0))
        .map(([key, entry]) => [key, canonicalize(entry)])
    );
  }
  return value;
};

/** Canonical JSON with recursively lexicographic object keys and stable array order. */
export const serializeReplaySummary = (summary: Readonly<ReplaySummary>): string =>
  JSON.stringify(canonicalize(summary));

/**
 * Auditable 64-bit FNV-1a over UTF-8 bytes. This is a deterministic regression
 * identifier within the project's supported Node/toolchain, not a
 * cryptographic digest or a promise of cross-engine numeric identity.
 */
export const hashReplaySummary = (summary: Readonly<ReplaySummary>): string => {
  const bytes = new TextEncoder().encode(serializeReplaySummary(summary));
  let hash = 0xcbf29ce484222325n;
  for (const byte of bytes) {
    hash ^= BigInt(byte);
    hash = BigInt.asUintN(64, hash * 0x100000001b3n);
  }
  return hash.toString(16).padStart(16, '0');
};

/**
 * Run an integer number of fixed 30 Hz ticks. Invalid or negative tick counts
 * run zero ticks; fractional counts are truncated toward zero.
 */
export const runReplay = (
  seed: number,
  ticks: number,
  policy: (tick: number, world: ReadonlyWorldState) => ActionsByAgent = () => ({}),
  config: Readonly<ArenaConfig> = ARENA_CONFIG
): ReplayResult => {
  const count = Number.isFinite(ticks) ? Math.max(0, Math.trunc(ticks)) : 0;
  let world = createWorld(seed, config);
  for (let tick = 0; tick < count; tick += 1) {
    // Policies receive a detached view: readonly typing prevents mistakes and
    // the runtime copy prevents cast/JS-based mutation from touching authority.
    const policyWorld = {
      ...world,
      config: retainArenaConfig(world.config),
      agents: world.agents.map((agent) => ({
        ...agent,
        position: { ...agent.position },
        previousPosition: { ...agent.previousPosition },
        velocity: { ...agent.velocity },
        activeHazardIds: [...agent.activeHazardIds],
        score: { ...agent.score }
      })),
      foods: world.foods.map((food) => ({ ...food, position: { ...food.position } })),
      hazards: world.hazards.map((hazard) => ({
        ...hazard,
        position: { ...hazard.position },
        previousPosition: { ...hazard.previousPosition },
        velocity: { ...hazard.velocity }
      }))
    } satisfies WorldState;
    world = stepWorld(world, policy(tick, policyWorld));
  }
  const summary = createReplaySummary(world);
  const serialized = serializeReplaySummary(summary);
  return { summary, serialized, hash: hashReplaySummary(summary) };
};

/** One tick's worth of per-agent score-trace data, recorded by `src/lib/experiment/runner.ts`. */
export interface ExperimentTraceEntry {
  tick: number;
  timeSeconds: number;
  agents: Record<
    AgentId,
    {
      foodPickups: number;
      hazardContacts: number;
      distanceTravelled: number;
      movementScore: number;
    }
  >;
}

/**
 * The deterministic, downloadable record of one closed-loop experiment run:
 * configuration (seed, config fingerprint, per-agent topology) and score
 * traces only. This deliberately contains nothing shaped like a
 * `ConnectomeGraph` (no `biologicalIds`/CSR arrays/weights) — the point of a
 * replay download is to let someone reproduce or audit a run's *outcome*
 * from its declared inputs, never to re-distribute the connectome asset
 * itself. `tests/unit/replay-export.test.ts` asserts this shape directly.
 */
export interface ExperimentReplayExport {
  schemaVersion: 1;
  seed: number;
  configFingerprint: string;
  topology: Record<AgentId, GraphMode>;
  /**
   * sha256 of each arm's manifest-verified compiled graph artifact, when one
   * exists (see `src/lib/experiment/runner.ts`'s `AgentRunnerInfo`). A key
   * is simply absent for an arm with no separate verified artifact of its
   * own (the runtime-derived 'disconnected' control) — never a placeholder
   * value. Lets a downloaded replay self-attest to exactly which compiled
   * artifact produced it (pinning it against future re-compiles of
   * `public/data/*`) without embedding the connectome itself.
   */
  graphBinarySha256: Partial<Record<AgentId, string>>;
  substepsPerTick: number;
  totalTicks: number;
  finalSummary: ReplaySummary;
  finalHash: string;
  trace: readonly ExperimentTraceEntry[];
}

export const createExperimentReplayExport = (
  world: Readonly<WorldState>,
  options: {
    topology: Record<AgentId, GraphMode>;
    graphBinarySha256?: Partial<Record<AgentId, string>>;
    substepsPerTick: number;
    totalTicks: number;
    trace: readonly ExperimentTraceEntry[];
  }
): ExperimentReplayExport => {
  const finalSummary = createReplaySummary(world);
  return {
    schemaVersion: 1,
    seed: world.seed,
    configFingerprint: world.configFingerprint,
    topology: { ...options.topology },
    // Undefined-valued keys (an arm with no verified artifact) are dropped
    // by JSON.stringify automatically, so a downloaded/serialized replay
    // never carries a misleading "graphBinarySha256": null/"" for an arm
    // that simply has none.
    graphBinarySha256: { left: options.graphBinarySha256?.left, right: options.graphBinarySha256?.right },
    substepsPerTick: options.substepsPerTick,
    totalTicks: options.totalTicks,
    finalSummary,
    finalHash: hashReplaySummary(finalSummary),
    // A defensive copy: the caller's `trace` array (e.g. ExperimentRunner's
    // own live buffer) keeps growing after this export is built if the run
    // resumes, and this export must stay a frozen snapshot rather than
    // alias state that changes out from under whoever holds it.
    trace: options.trace.slice()
  };
};
