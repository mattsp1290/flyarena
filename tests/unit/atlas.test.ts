import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { loadLocalAssets } from '../../scripts/experiments/local-assets';
import { loadLocalAtlas } from '../../scripts/atlas/files';
import { publishAtlas } from '../../scripts/atlas/publish';
import { evaluateBehavior } from '../../scripts/atlas/evaluate';
import { prepareGraph } from '../../src/lib/counterfactual/targets';
import {
  createBranch,
  stepBranch,
  captureFrame,
  runExperiment
} from '../../src/lib/counterfactual/engine';
import { createWorld } from '../../src/lib/arena/world';
import { createArenaConfigFingerprint, ARENA_CONFIG } from '../../src/lib/arena/config';
import { resolveController } from '../../src/lib/atlas/controller';
import { readAtlasManifest, verifyAtlasBytes } from '../../src/lib/atlas/assets';
import { validateAtlas } from '../../src/lib/atlas/validation';
import { COVERAGE_EDGES, binIndex } from '../../src/lib/atlas/types';
import { readoutFromFlat } from '../../src/lib/connectome/readout-serialization';
import { runEpisode } from '../../scripts/training/episode';
import { sha256Hex } from '../../scripts/training/fsio';
import { DEFAULT_REQUEST } from '../../src/lib/counterfactual/types';
import type { Decoder } from '../../src/lib/counterfactual/decoder';

const data = 'public/data';
const loaded = await loadLocalAtlas(data);
const prepared = await prepareGraph(await loadLocalAssets(data), 'biological');

describe('published behavior atlas', () => {
  it('contains real CUDA discovery and canonical diversity', () => {
    const atlas = loaded.atlas;
    expect(atlas.source.runtime.device).toBe('cuda');
    expect(atlas.cells.length).toBeGreaterThanOrEqual(6);
    expect(new Set(atlas.cells.map((c) => c.cell % 6)).size).toBeGreaterThanOrEqual(2);
    expect(new Set(atlas.cells.map((c) => Math.floor(c.cell / 6))).size).toBeGreaterThanOrEqual(3);
    expect(atlas.graphBinarySha256).toBe(prepared.identity.binarySha256);
    expect(atlas.configFingerprint).toBe(createArenaConfigFingerprint(ARENA_CONFIG));
    expect(
      loaded.atlas.source.discoverySeeds.some((s) => loaded.atlas.source.heldoutSeeds.includes(s))
    ).toBe(false);
  });
  it('bins boundaries and rejects corruption before simulation', async () => {
    expect([0, 0.05, 0.1, 1].map((v) => binIndex(v, COVERAGE_EDGES))).toEqual([0, 1, 2, 5]);
    const altered = structuredClone(loaded.atlas);
    altered.cells[0].cell = 100;
    expect(() => validateAtlas(altered)).toThrow();
    altered.cells[0].cell = loaded.atlas.cells[0].cell;
    altered.source.candidates[0].theta[0] = Infinity;
    expect(() => validateAtlas(altered)).toThrow();
    const manifest = readAtlasManifest(
      JSON.parse(readFileSync(`${data}/behavior-atlas-v1.manifest.json`, 'utf8'))
    );
    expect(() => readAtlasManifest(manifest, 'a'.repeat(64))).toThrow('Atlas changed');
    await expect(verifyAtlasBytes(new TextEncoder().encode('{}').buffer, manifest)).rejects.toThrow(
      'integrity'
    );
  });
  it('reconstructs a real published cell and distinguishes rehashed score corruption', async () => {
    const selected = loaded.atlas.cells[0];
    const source = {
      ...loaded.atlas.source,
      candidates: loaded.atlas.source.candidates.filter((c) => c.id === selected.id)
    };
    const fresh = await publishAtlas(source, data, false);
    expect(fresh.cells[0]).toEqual(selected);
    const changed = structuredClone(loaded.atlas);
    changed.cells[0].heldout.biological[0].movementScore++;
    const bytes = new TextEncoder().encode(JSON.stringify(changed));
    const rehashed = await verifyAtlasBytes(bytes.buffer, {
      schemaVersion: 1,
      artifact: 'behavior-atlas-v1.json',
      sha256: sha256Hex(bytes),
      bytes: bytes.length
    });
    expect(fresh.cells[0]).not.toEqual(rehashed.atlas.cells[0]);
  }, 120_000);
});

describe('selected-controller causal engine', () => {
  for (const cell of [loaded.atlas.cells[0], loaded.atlas.cells.at(-1)!]) {
    it(`matches authoritative episode before and after fork for controller ${cell.id}`, async () => {
      const controller = await resolveController(loaded, prepared, cell.id);
      const spec: Decoder = { decoder: 'atlas-trained', controller };
      const seed = 62001,
        warmup = 30,
        horizon = 60;
      const branch = createBranch(prepared.graph, createWorld(seed), undefined, spec);
      const weights = readoutFromFlat(
        controller.theta,
        controller.inputSize,
        controller.hiddenSize
      );
      const snapshots: ReturnType<typeof captureFrame>[] = [];
      const episode = runEpisode({
        seed,
        ticks: warmup + horizon,
        left: { decoder: 'trained', graph: prepared.graph, weights },
        right: { decoder: 'parked' },
        onTick: (_tick, _actions, world) => snapshots.push(captureFrame(world))
      });
      for (const expected of snapshots) {
        stepBranch(prepared.graph, branch);
        expect(captureFrame(branch.world)).toEqual(expected);
      }
      const result = runExperiment(
        prepared,
        { ...DEFAULT_REQUEST, seed, seedCount: 4, warmup, horizon, target: 'output' },
        spec
      );
      expect(result.results[0].branches.baseline.frames.at(-1)).toEqual(snapshots.at(-1));
      expect(result.results[0].checkpoint.scores.left).toEqual(snapshots[warmup - 1].scores.left);
      expect(result.results[0].branches.baseline.frames.at(-1)!.scores.left).toEqual(episode.left);
      for (const row of result.results) expect(row.branches.sham).toEqual(row.branches.baseline);
      expect(result.decoder).toBe('atlas-trained');
      expect(result.summary.shamEffect.mean).toBe(0);
    });
  }
  it('clamps output rates after every substep while preserving learned biases', async () => {
    const controller = await resolveController(loaded, prepared, loaded.atlas.cells[0].id);
    const branch = createBranch(prepared.graph, createWorld(62001), undefined, {
      decoder: 'atlas-trained',
      controller
    });
    const indices = Array.from({ length: prepared.graph.metadata.neuronCount }, (_, i) => i);
    const weights = readoutFromFlat(controller.theta, controller.inputSize, controller.hiddenSize);
    const episode = evaluateBehavior(
      { decoder: 'silenced', graph: prepared.graph, weights },
      62001,
      30
    );
    for (let i = 0; i < 30; i++) {
      stepBranch(prepared.graph, branch, indices);
      expect(branch.state.rate.every((v) => v === 0)).toBe(true);
    }
    expect(captureFrame(branch.world).scores.left.movementScore).toBe(
      episode.metrics.movementScore
    );
    await expect(resolveController(loaded, prepared, 99999)).rejects.toThrow();
    const wrong = { ...loaded, atlas: { ...loaded.atlas, graphBinarySha256: 'a'.repeat(64) } };
    await expect(resolveController(wrong, prepared, controller.id)).rejects.toThrow('identities');
  });
});
