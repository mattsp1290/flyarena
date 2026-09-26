import { describe, it, expect, afterAll, beforeAll } from 'vitest';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { gunzipSync } from 'node:zlib';
import { loadLocalAssets } from '../../scripts/experiments/local-assets';
import { loadLocalAtlas } from '../../scripts/atlas/files';
import { publishAtlas, evaluateSearchOnGraph } from '../../scripts/atlas/publish';
import { evaluateSearchForGraph } from '../../scripts/atlas/verify-search-graph';
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
import {
  ATLAS_VERSION,
  COVERAGE_EDGES,
  TURN_EDGES,
  DISCOVERY_SEEDS,
  HELDOUT_SEEDS,
  binIndex,
  type SearchArtifact
} from '../../src/lib/atlas/types';
import { readoutFromFlat } from '../../src/lib/connectome/readout-serialization';
import { readoutParameterCount } from '../../src/lib/connectome/readout';
import { encodeGraphBinary } from '../../src/lib/connectome/format';
import {
  computeArmBundleSha256,
  deserializeArmBundle,
  runExportArms,
  type SerializedArmBundle
} from '../../scripts/training/export-arms';
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

describe('evaluateSearchOnGraph / evaluateSearchForGraph (generalized to any verified graph)', () => {
  // Trace-graph fixture arms (biological/rewired/disconnected), exported the
  // same way `training:export-arms --fixture-rewire` does, so these tests
  // exercise real bundle JSON, never a hand-rolled shape
  // (`.agents/plans/repertoire-null/01-generalize-atlas-pipeline.md` WP1).
  let root: string;
  let biological: SerializedArmBundle;
  let rewired: SerializedArmBundle;
  let disconnected: SerializedArmBundle;

  beforeAll(() => {
    root = mkdtempSync(join(tmpdir(), 'atlas-verify-search-graph-'));
    const exportResult = runExportArms({ outDir: root, fixtureRewire: true, fixtureRewireSeed: 3 });
    biological = JSON.parse(
      readFileSync(join(exportResult.outDir, 'biological.json'), 'utf8')
    ) as SerializedArmBundle;
    rewired = JSON.parse(
      readFileSync(join(exportResult.outDir, 'rewired.json'), 'utf8')
    ) as SerializedArmBundle;
    disconnected = JSON.parse(
      readFileSync(join(exportResult.outDir, 'disconnected.json'), 'utf8')
    ) as SerializedArmBundle;
  });

  afterAll(() => {
    if (root) rmSync(root, { recursive: true, force: true });
  });

  const runtimeConfig = Object.fromEntries(
    Object.entries(ARENA_CONFIG).map(([key, value]) => [
      key.replace(/[A-Z]/g, (c) => '_' + c.toLowerCase()),
      value
    ])
  );

  // A minimal, `validateSearch`-passing one-candidate search artifact for
  // `bundle` (same D across every trace-graph arm, by `export-arms.ts`'s own
  // "D equal across arms" gate). ticks/population/generations are kept tiny
  // -- this exercises the rebinning/held-out plumbing, not a real search.
  const buildSearchArtifact = (bundle: SerializedArmBundle): SearchArtifact => {
    const d = bundle.D;
    const hiddenSize = 8;
    const parameterCount = readoutParameterCount(d, hiddenSize);
    const theta = Array.from({ length: parameterCount }, (_, i) => (((i * 37) % 101) / 101 - 0.5) * 0.2);
    return {
      schemaVersion: 1,
      modelVersion: ATLAS_VERSION,
      options: { seed: 1, population: 4, generations: 1, ticks: 30 },
      inputSize: d,
      hiddenSize,
      substeps: 4,
      discoverySeeds: DISCOVERY_SEEDS,
      heldoutSeeds: HELDOUT_SEEDS,
      coverageEdges: COVERAGE_EDGES,
      turnEdges: TURN_EDGES,
      graphArtifactSha256: bundle.graphArtifactSha256,
      bundleSha256: bundle.sha256,
      bundle: bundle as unknown as Record<string, unknown>,
      candidates: [{ id: 0, theta, quality: 0, coverage: 0, turning: 0 }],
      history: [{ generation: 1, occupied: 1, bestQuality: 0 }],
      searchPolicy: {
        initialStd: 0.5,
        mutationScales: [0.05, 0.15, 0.4],
        freshFraction: 0.25,
        weightBound: 8,
        ties: 'earlier candidate',
        rng: 'torch CPU Generator'
      },
      runtime: {
        device: 'cpu',
        deviceName: 'vitest',
        torch: '0.0.0',
        cuda: null,
        seconds: 1,
        peakTensorBytes: 0,
        config: runtimeConfig
      }
    };
  };

  it('evaluateSearchOnGraph with the diversity gate off and heldout "own" returns one occupied cell for a single candidate', async () => {
    const source = buildSearchArtifact(biological);
    const graph = deserializeArmBundle(biological);
    const result = await evaluateSearchOnGraph(source, graph, {
      requireDiversity: false,
      heldout: 'own'
    });
    expect(result.gpuArchiveSize).toBe(1);
    expect(result.collisions).toBe(0);
    expect(result.cells.length).toBe(1);
    expect(result.cells[0].heldout.biological).toHaveLength(12);
    expect(result.cells[0].heldout.disconnected).toBeUndefined();
    expect(result.cells[0].heldout.silenced).toBeUndefined();
  });

  it('evaluateSearchOnGraph with heldout "all" populates every CONTROL_NAMES control', async () => {
    const source = buildSearchArtifact(biological);
    const graph = deserializeArmBundle(biological);
    const result = await evaluateSearchOnGraph(source, graph, {
      requireDiversity: false,
      heldout: 'all'
    });
    expect(result.cells[0].heldout.biological).toHaveLength(12);
    expect(result.cells[0].heldout.disconnected).toHaveLength(12);
    expect(result.cells[0].heldout.silenced).toHaveLength(12);
  });

  it('accepts a rewired bundle whose re-encoded binary matches the expected seed sha', async () => {
    const searchPath = join(root, 'rewired-search.json');
    writeFileSync(searchPath, JSON.stringify(buildSearchArtifact(rewired)));
    const binarySha256 = sha256Hex(new Uint8Array(encodeGraphBinary(deserializeArmBundle(rewired))));
    const result = await evaluateSearchForGraph(searchPath, {
      arm: 'rewired',
      binarySha256,
      parentGzipSha256: rewired.graphArtifactSha256
    });
    expect(result.cells.length).toBe(1);
    expect(result.gpuArchiveSize).toBe(1);
  });

  it('rejects a rewired bundle whose re-encoded binary does not match the expected seed sha, even when its graphArtifactSha256 (parent) matches', async () => {
    const searchPath = join(root, 'rewired-search-wrong-binary.json');
    writeFileSync(searchPath, JSON.stringify(buildSearchArtifact(rewired)));
    await expect(
      evaluateSearchForGraph(searchPath, {
        arm: 'rewired',
        binarySha256: 'a'.repeat(64),
        parentGzipSha256: rewired.graphArtifactSha256
      })
    ).rejects.toThrow('binary identity mismatch');
  });

  it('rejects a graph whose graphArtifactSha256 (parent) does not match, even with a correct binary sha', async () => {
    const searchPath = join(root, 'rewired-search-wrong-parent.json');
    writeFileSync(searchPath, JSON.stringify(buildSearchArtifact(rewired)));
    const binarySha256 = sha256Hex(new Uint8Array(encodeGraphBinary(deserializeArmBundle(rewired))));
    await expect(
      evaluateSearchForGraph(searchPath, {
        arm: 'rewired',
        binarySha256,
        parentGzipSha256: 'a'.repeat(64)
      })
    ).rejects.toThrow('parent identity mismatch');
  });

  it('rejects a disconnected bundle with edgeCount > 0', async () => {
    // A well-formed, internally-consistent (non-disconnected) graph
    // mislabeled as arm "disconnected" -- the tamper scenario this check
    // exists for, not a bundle whose metadata disagrees with its own arrays
    // (which `deserializeArmBundle`'s `validateGraph` would already reject).
    // Its own sha256 is recomputed for the new `arm` value so this genuinely
    // tests "a self-consistent bundle that lies about its arm", not "an
    // internally-inconsistent bundle" (which the self-consistency check
    // above would reject first, with a less specific error).
    const mislabeledWithoutHash: SerializedArmBundle = { ...biological, arm: 'disconnected' };
    const mislabeled: SerializedArmBundle = {
      ...mislabeledWithoutHash,
      sha256: computeArmBundleSha256(mislabeledWithoutHash)
    };
    const searchPath = join(root, 'disconnected-search-nonzero-edges.json');
    writeFileSync(searchPath, JSON.stringify(buildSearchArtifact(mislabeled)));
    await expect(
      evaluateSearchForGraph(searchPath, {
        arm: 'disconnected',
        parentGzipSha256: mislabeled.graphArtifactSha256
      })
    ).rejects.toThrow('edgeCount 0');
  });

  it('accepts a disconnected bundle with edgeCount 0 and requires no binarySha256', async () => {
    const searchPath = join(root, 'disconnected-search.json');
    writeFileSync(searchPath, JSON.stringify(buildSearchArtifact(disconnected)));
    const result = await evaluateSearchForGraph(searchPath, {
      arm: 'disconnected',
      parentGzipSha256: disconnected.graphArtifactSha256
    });
    expect(result.cells.length).toBe(1);
  });

  it('verifies a disconnected bundle\'s binarySha256 when one is supplied, rather than ignoring it', async () => {
    const searchPath = join(root, 'disconnected-search-with-binary-sha.json');
    writeFileSync(searchPath, JSON.stringify(buildSearchArtifact(disconnected)));
    const binarySha256 = sha256Hex(new Uint8Array(encodeGraphBinary(deserializeArmBundle(disconnected))));
    const result = await evaluateSearchForGraph(searchPath, {
      arm: 'disconnected',
      binarySha256,
      parentGzipSha256: disconnected.graphArtifactSha256
    });
    expect(result.cells.length).toBe(1);

    const searchPathWrong = join(root, 'disconnected-search-wrong-binary-sha.json');
    writeFileSync(searchPathWrong, JSON.stringify(buildSearchArtifact(disconnected)));
    await expect(
      evaluateSearchForGraph(searchPathWrong, {
        arm: 'disconnected',
        binarySha256: 'f'.repeat(64),
        parentGzipSha256: disconnected.graphArtifactSha256
      })
    ).rejects.toThrow('binary identity mismatch');
  });

  it('accepts a biological bundle the same way as a rewired one', async () => {
    const searchPath = join(root, 'biological-search.json');
    writeFileSync(searchPath, JSON.stringify(buildSearchArtifact(biological)));
    const binarySha256 = sha256Hex(new Uint8Array(encodeGraphBinary(deserializeArmBundle(biological))));
    const result = await evaluateSearchForGraph(searchPath, {
      arm: 'biological',
      binarySha256,
      parentGzipSha256: biological.graphArtifactSha256
    });
    expect(result.cells.length).toBe(1);
  });

  it('requires binarySha256 for a rewired arm (never silently skipped)', async () => {
    const searchPath = join(root, 'rewired-search-missing-binary-sha.json');
    writeFileSync(searchPath, JSON.stringify(buildSearchArtifact(rewired)));
    await expect(
      evaluateSearchForGraph(searchPath, {
        arm: 'rewired',
        parentGzipSha256: rewired.graphArtifactSha256
      })
    ).rejects.toThrow('requires binarySha256');
  });

  it('rejects an arm mismatch between the bundle and what was expected', async () => {
    const searchPath = join(root, 'arm-mismatch-search.json');
    writeFileSync(searchPath, JSON.stringify(buildSearchArtifact(rewired)));
    await expect(
      evaluateSearchForGraph(searchPath, {
        arm: 'biological',
        binarySha256: sha256Hex(new Uint8Array(encodeGraphBinary(deserializeArmBundle(rewired)))),
        parentGzipSha256: rewired.graphArtifactSha256
      })
    ).rejects.toThrow('arm mismatch');
  });

  it('rejects a search artifact whose top-level identity disagrees with its own bundle', async () => {
    const corrupted = { ...buildSearchArtifact(rewired), bundleSha256: 'a'.repeat(64) };
    const searchPath = join(root, 'self-inconsistent-search.json');
    writeFileSync(searchPath, JSON.stringify(corrupted));
    const binarySha256 = sha256Hex(new Uint8Array(encodeGraphBinary(deserializeArmBundle(rewired))));
    await expect(
      evaluateSearchForGraph(searchPath, {
        arm: 'rewired',
        binarySha256,
        parentGzipSha256: rewired.graphArtifactSha256
      })
    ).rejects.toThrow('not self-consistent');
  });

  it('rejects a search artifact whose top-level arm disagrees with its own bundle', async () => {
    const corrupted = { ...buildSearchArtifact(rewired), arm: 'biological' };
    const searchPath = join(root, 'arm-field-mismatch-search.json');
    writeFileSync(searchPath, JSON.stringify(corrupted));
    const binarySha256 = sha256Hex(new Uint8Array(encodeGraphBinary(deserializeArmBundle(rewired))));
    await expect(
      evaluateSearchForGraph(searchPath, {
        arm: 'rewired',
        binarySha256,
        parentGzipSha256: rewired.graphArtifactSha256
      })
    ).rejects.toThrow('disagrees with its bundle arm');
  });

  it('counts collisions and keeps the earlier candidate on a quality tie', async () => {
    const base = buildSearchArtifact(biological);
    const [c0] = base.candidates;
    // Identical theta => identical discovery metrics => the same TS cell,
    // equal quality -- every candidate after the first collides.
    const source: SearchArtifact = {
      ...base,
      candidates: [c0, { ...c0, id: 1 }, { ...c0, id: 2 }]
    };
    const graph = deserializeArmBundle(biological);
    const result = await evaluateSearchOnGraph(source, graph, {
      requireDiversity: false,
      heldout: 'own'
    });
    expect(result.gpuArchiveSize).toBe(3);
    expect(result.evaluations.map((e) => e.id)).toEqual([0, 1, 2]);
    expect(result.cells).toHaveLength(1);
    expect(result.collisions).toBe(2);
    expect(result.cells[0].id).toBe(0); // strict '>' means a tie keeps the earlier id
  });

  it('throws the canonical diversity gate before any held-out episode', async () => {
    const source = buildSearchArtifact(biological);
    const graph = deserializeArmBundle(biological);
    await expect(
      evaluateSearchOnGraph(source, graph, { requireDiversity: true, heldout: 'all' })
    ).rejects.toThrow('Canonical diversity gate failed');
  });

  it('re-encodes a real rewired artifact to the same sha rewire_batch.py would call binarySha256', async () => {
    // Unlike the synthetic-fixture tests above (which compute their own
    // expected sha with the same TS expression the implementation uses,
    // and so cannot catch TS re-encoding drifting away from Python), this
    // uses the two real artifacts already committed to public/data and an
    // independently-computed expected value: the decompressed sha of the
    // real seed-0 rewired .bin.gz, which is exactly what rewire_batch.py's
    // index.json calls binarySha256 for that seed.
    const exportResult = runExportArms({
      graphPath: 'public/data/malecns-arena-v1.bin.gz',
      rewiredPath: 'public/data/malecns-arena-v1-rewired-seed0.bin.gz',
      fixtureRewire: false,
      fixtureRewireSeed: 0,
      outDir: join(root, 'real-artifact-export')
    });
    const realRewired = JSON.parse(
      readFileSync(join(exportResult.outDir, 'rewired.json'), 'utf8')
    ) as SerializedArmBundle;
    const expectedBinarySha256 = sha256Hex(
      gunzipSync(readFileSync('public/data/malecns-arena-v1-rewired-seed0.bin.gz'))
    );
    const actual = sha256Hex(new Uint8Array(encodeGraphBinary(deserializeArmBundle(realRewired))));
    expect(actual).toBe(expectedBinarySha256);
  });
});
