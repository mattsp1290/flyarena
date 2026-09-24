import { beforeAll, describe, expect, it, vi } from 'vitest';
import { prepareGraph, type PreparedGraph } from '../../src/lib/counterfactual/targets';
import { createBranch, captureFrame, interval, runExperiment, scoreDifference, stepBranch } from '../../src/lib/counterfactual/engine';
import { DEFAULT_REQUEST, seedsFor, validateRequest } from '../../src/lib/counterfactual/types';
import { createWorld } from '../../src/lib/arena/world';
import { NEURAL_SUBSTEPS_PER_TICK } from '../../src/lib/connectome/constants';
import { runEpisode } from '../../scripts/training/episode';
import { createOracleAgentBinding } from '../../src/lib/experiment/bindings';
import { ExperimentRunner, type AgentBinding } from '../../src/lib/experiment/runner';
import { encodeGraphBinary } from '../../src/lib/connectome/format';
import { loadArenaArtifacts } from '../../src/lib/experiment/assets';
import { createPublicDataFetch } from '../helpers/fake-worker';
import { counterfactualAssets } from '../fixtures/counterfactual';
import { createTinyGraph } from '../fixtures/tiny-graph';

const request = { ...DEFAULT_REQUEST, seed:17, seedCount:4, warmup:30, horizon:30 };
let prepared: PreparedGraph;
beforeAll(async () => { prepared = await prepareGraph(await counterfactualAssets(), 'biological'); });

describe('causal experiment invariants', () => {
  it('retains every checkpoint field without aliasing the world, contact lists, or neural buffers', () => {
    const world = createWorld(11);
    world.agents[0].activeHazardIds.push('checkpoint-contact');
    const rates = Float32Array.from({length:prepared.graph.metadata.neuronCount}, () => 0.25);
    const a = createBranch(prepared.graph, world, rates);
    const b = createBranch(prepared.graph, world, rates);
    expect(a.world).toEqual(world); expect(a.state.rate).toEqual(rates);
    a.world.agents[0].position.x = 99; a.world.agents[0].score.movementScore = 10;
    a.world.agents[0].activeHazardIds.push('new-contact'); a.world.rngState++;
    a.state.rate.fill(0);
    expect(b.world).toEqual(world); expect(b.state.rate).toEqual(rates);
    expect(b.world.agents[0].activeHazardIds).toEqual(['checkpoint-contact']);
    expect(b.world.rngState).not.toBe(a.world.rngState);
  });

  it('silences before the first scatter AND every substep, preventing freshly injected target activity from propagating', () => {
    const graph = createTinyGraph({ inputChannelCount:8, outputPopulationCount:3 });
    const active = createBranch(graph, createWorld(17));
    active.state.rate[0] = 1;
    const silenced = createBranch(graph, active.world, active.state.rate);
    stepBranch(graph, active);
    stepBranch(graph, silenced, [0]);
    expect(active.state.rate[2]).not.toBe(0);
    expect(silenced.state.rate[0]).toBe(0);
    expect(silenced.state.rate[2]).toBe(0);
    expect(silenced.outputs[0]).toBe(0);
    expect(active.outputs[0]).not.toBe(0);
  });

  it('matches sham exactly, repeats exactly, captures real intermediate scores and agrees with the authoritative evaluator', () => {
    const result = runExperiment(prepared, request);
    expect(runExperiment(prepared, request)).toEqual(result);
    for (const seedResult of result.results) {
      expect(seedResult.branches.sham).toEqual(seedResult.branches.baseline);
      const reference = (ticks:number) => runEpisode({ seed:seedResult.seed, ticks, substeps:NEURAL_SUBSTEPS_PER_TICK, left:{ decoder:'authored',graph:prepared.graph }, right:{decoder:'parked'} });
      const fork = reference(request.warmup);
      expect(seedResult.checkpoint.scores.left).toEqual(fork.left);
      expect(seedResult.branches.baseline.outcome).toEqual(scoreDifference(reference(request.warmup + request.horizon).left, fork.left));
      for (const index of [0, 15, 30]) {
        const frame = seedResult.branches.baseline.frames[index];
        expect(frame.scores.left).toEqual(reference(frame.snapshot.tick).left);
      }
      expect(seedResult.branches.baseline.frames[0].scores.left).not.toEqual(seedResult.branches.baseline.frames.at(-1)!.scores.left);
    }
    expect(result.summary.shamEffect).toEqual({mean:0,low:0,high:0});
    const differences = result.results.map(r => r.branches.lesion.outcome.movementScore - r.branches.baseline.outcome.movementScore);
    expect(result.summary.effect).toEqual(interval(differences));
    expect(interval([1,2,3,4])).toEqual({mean:2.5, low:2.5-1.96*Math.sqrt(5/12), high:2.5+1.96*Math.sqrt(5/12)});
  });

  it('matches the product runner with an actual oracle binding and zero-action opponent', async () => {
    const left = createOracleAgentBinding({graphBuffer:encodeGraphBinary(prepared.graph),mode:'biological'});
    const right: AgentBinding = {
      info:{topology:'disconnected',neuronCount:0,edgeCount:0}, reset:async () => {},
      step:async () => ({ actionFeatures:[0,0,0],telemetry:{meanRate:0,activeFraction:0,minRate:0,maxRate:0} })
    };
    const runner = new ExperimentRunner({seed:request.seed,totalTicks:request.warmup+request.horizon,agents:{left,right},targetTickIntervalMs:0});
    runner.start();
    await vi.waitFor(() => expect(runner.getStatus()).toBe('finished'));
    expect(captureFrame(runner.getWorld() as ReturnType<typeof createWorld>)).toEqual(runExperiment(prepared,request).results[0].branches.baseline.frames.at(-1));
    runner.dispose();
  });

  it('bounds sampled replay and retains both endpoints, and rejects empty targets', () => {
    const result = runExperiment(prepared,{...request,horizon:300,warmup:0});
    for (const seed of result.results) for (const branch of Object.values(seed.branches)) {
      expect(branch.frames).toHaveLength(61);
      expect(branch.frames[0].snapshot.tick).toBe(0);
      expect(branch.frames.at(-1)!.snapshot.tick).toBe(300);
    }
    const empty = {...prepared,targets:prepared.targets.map(t=>({...t,indices:[]}))};
    expect(()=>runExperiment(empty,request)).toThrow('no neurons');
  });

  it('uses verified real artifacts for all topologies, preserving body IDs and exact sham parity', async () => {
    vi.stubGlobal('fetch',createPublicDataFetch());
    try {
      const assets = await loadArenaArtifacts('/data');
      for (const topology of ['biological','rewired','disconnected'] as const) {
        const real = await prepareGraph(assets,topology);
        expect(real.targets.find(t=>t.id==='bridge')!.indices.length).toBe(800);
        const result = runExperiment(real,{...request,topology});
        expect(result.graph.neuronCount).toBe(1008);
        if (topology==='disconnected') expect(result.graph.edgeCount).toBe(0);
        for (const seed of result.results) expect(seed.branches.baseline).toEqual(seed.branches.sham);
        expect(result.target.bodyIds).toEqual(result.target.indices.map(i=>real.graph.biologicalIds[i].toString()));
      }
    } finally { vi.unstubAllGlobals(); }
  });

  it('validates bounds and portable seed wraparound before allocating', () => {
    for (const change of [{seed:true},{seed:-1},{seed:2**32},{seedCount:3},{seedCount:17},{warmup:-1},{warmup:301},{horizon:0},{horizon:301},{horizon:30.5},{target:'__proto__'},{topology:'auto'},{extra:1}]) {
      expect(()=>validateRequest({...request,...change})).toThrow();
    }
    expect(seedsFor({...request,seed:0xfffffffe})).toEqual([0xfffffffe,0xffffffff,0,1]);
    expect(()=>runExperiment(prepared,{...request,topology:'rewired'})).toThrow('topology');
  });
});
