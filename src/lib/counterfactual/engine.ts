import { AUTHORED, createDecoder, type Decoder } from './decoder';
import { ARENA_CONFIG, createArenaConfigFingerprint } from '../arena/config';
import { observeAgent } from '../arena/sensors';
import { decodeAction } from '../arena/actions';
import { createWorld, createSnapshot, stepWorld } from '../arena/world';
import type { AgentScore, WorldState, ReadonlyWorldState } from '../arena/types';
import { NEURAL_SUBSTEPS_PER_TICK } from '../connectome/constants';
import { aggregateOutputs, createModelState, createOutputBuffer, createStepScratch, stepModel } from '../connectome/model';
import type { ConnectomeGraph } from '../connectome/format';
import type { PreparedGraph } from './targets';
import {
  MODEL_VERSION, SCORE_KEYS, seedsFor, validateRequest,
  type Evidence, type EvidenceHeader, type Frame, type Interval, type Request, type SeedResult, type BranchResult
} from './types';

/** Every branch owns a full world, neural state and reusable scratch. */
export function createBranch(graph: ConnectomeGraph, world: WorldState, rates?: Float32Array, decoder: Decoder = AUTHORED) {
  const state = createModelState(graph);
  if (rates) state.rate.set(rates);
  return { world: structuredClone(world), state, scratch: createStepScratch(graph), outputs: createOutputBuffer(graph), decode: createDecoder(graph, decoder) };
}
export type SimulationBranch = ReturnType<typeof createBranch>;

export function stepBranch(graph: ConnectomeGraph, branch: SimulationBranch, target: readonly number[] = []): void {
  const observation = observeAgent(branch.world, 'left');
  // The first scatter must not propagate the targeted checkpoint activity.
  for (const i of target) branch.state.rate[i] = 0;
  for (let k = 0; k < NEURAL_SUBSTEPS_PER_TICK; k++) {
    stepModel(graph, branch.state, branch.scratch, observation);
    for (const i of target) branch.state.rate[i] = 0;
  }
  aggregateOutputs(graph, branch.state, branch.outputs);
  const action = decodeAction(branch.decode(branch.state.rate, branch.outputs));
  branch.world = stepWorld(branch.world, { left: [action.thrust, action.yaw, action.brake], right: [0, 0, 0] });
}

export function captureFrame(world: ReadonlyWorldState): Frame {
  return {
    snapshot: createSnapshot(world, 1),
    scores: { left: { ...world.agents.find(a => a.id === 'left')!.score }, right: { ...world.agents.find(a => a.id === 'right')!.score } }
  };
}
export const scoreDifference = (a: AgentScore, b: AgentScore): AgentScore => ({
  foodPickups: a.foodPickups - b.foodPickups, hazardContacts: a.hazardContacts - b.hazardContacts,
  distanceTravelled: a.distanceTravelled - b.distanceTravelled, movementScore: a.movementScore - b.movementScore
});
export function interval(values: number[]): Interval {
  if (values.length < 2 || values.some(v => !Number.isFinite(v))) throw new Error('Invalid paired sample');
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / (values.length - 1);
  const margin = 1.96 * Math.sqrt(variance / values.length);
  return { mean, low: mean - margin, high: mean + margin };
}
const meanScore = (values: AgentScore[]): AgentScore => {
  const mean: AgentScore = { foodPickups: 0, hazardContacts: 0, distanceTravelled: 0, movementScore: 0 };
  for (const key of SCORE_KEYS) mean[key] = values.reduce((sum, value) => sum + value[key], 0) / values.length;
  return mean;
};

export function runSeed(prepared: PreparedGraph, request: Request, seed: number, decoder: Decoder = AUTHORED): SeedResult {
  const target = prepared.targets.find(t => t.id === request.target);
  if (!target?.indices.length) throw new Error('Selected target has no neurons');
  const warm = createBranch(prepared.graph, createWorld(seed), undefined, decoder);
  for (let i = 0; i < request.warmup; i++) stepBranch(prepared.graph, warm);
  const fork = captureFrame(warm.world);
  const count = Math.min(60, request.horizon);
  const sampleTicks = new Set(Array.from({ length: count + 1 }, (_, i) => Math.round(i * request.horizon / count)));
  function future(indices: readonly number[]): BranchResult {
    const branch = createBranch(prepared.graph, warm.world, warm.state.rate, decoder);
    const frames = [captureFrame(branch.world)];
    for (let tick = 1; tick <= request.horizon; tick++) {
      stepBranch(prepared.graph, branch, indices);
      if (sampleTicks.has(tick)) frames.push(captureFrame(branch.world));
    }
    return {
      outcome: scoreDifference(captureFrame(branch.world).scores.left, fork.scores.left), frames
    };
  }
  const branches = { baseline: future([]), sham: future([]), lesion: future(target.indices) };
  if (JSON.stringify(branches.baseline) !== JSON.stringify(branches.sham)) throw new Error('Sham control failed');
  return {
    seed, checkpoint: { tick: warm.world.tick, rngState: warm.world.rngState, scores: fork.scores }, branches,
    difference: scoreDifference(branches.lesion.outcome, branches.baseline.outcome),
    shamDifference: scoreDifference(branches.sham.outcome, branches.baseline.outcome)
  };
}

export function evidenceHeader(prepared: PreparedGraph, input: unknown, decoder: Decoder = AUTHORED): EvidenceHeader {
  const request = validateRequest(input);
  if (request.topology !== prepared.identity.topology) throw new Error('Prepared topology does not match request');
  const target = prepared.targets.find(t => t.id === request.target);
  if (!target?.indices.length) throw new Error('Selected target has no neurons');
  const seeds = seedsFor(request);
  return {
    schemaVersion: 1, modelVersion: MODEL_VERSION, request, seeds, graph: { ...prepared.identity },
    config: { ...ARENA_CONFIG }, configFingerprint: createArenaConfigFingerprint(ARENA_CONFIG),
    substeps: NEURAL_SUBSTEPS_PER_TICK, ...structuredClone(decoder), opponent: 'zero-action', target: structuredClone(target),
    provenance: {
      topology: request.topology === 'biological' ? 'Measured connectivity' : 'Authored control derived from measured connectivity',
      grouping: 'Authored input/output mappings, not anatomical regions',
      dynamics: 'Authored rate model and synthetic arena',
      interpretation: 'Descriptive paired model intervention; not biological causal inference'
    }
  };
}

/** Generator allows Worker progress between seeds without browser dependencies. */
export function* experiment(prepared: PreparedGraph, input: unknown, decoder: Decoder = AUTHORED): Generator<number, Evidence> {
  const header = evidenceHeader(prepared, input, decoder);
  const results: SeedResult[] = [];
  for (const seed of header.seeds) {
    results.push(runSeed(prepared, header.request, seed, decoder));
    yield results.length;
  }
  return {
    ...header,
    results,
    summary: {
      means: { baseline: meanScore(results.map(r => r.branches.baseline.outcome)), sham: meanScore(results.map(r => r.branches.sham.outcome)), lesion: meanScore(results.map(r => r.branches.lesion.outcome)) },
      effect: interval(results.map(r => r.difference.movementScore)),
      shamEffect: interval(results.map(r => r.shamDifference.movementScore))
    }
  };
}
export function runExperiment(prepared: PreparedGraph, request: unknown, decoder: Decoder = AUTHORED): Evidence {
  const iterator = experiment(prepared, request, decoder);
  let next = iterator.next();
  while (!next.done) next = iterator.next();
  return next.value;
}
