import { ARENA_CONFIG, createArenaConfigFingerprint } from '../../src/lib/arena/config';
import { createDisconnectedGraph, encodeGraphBinary } from '../../src/lib/connectome/format';
import { outputNeuronIndices, validateReadoutWeights } from '../../src/lib/connectome/readout';
import { readoutFromFlat } from '../../src/lib/connectome/readout-serialization';
import {
  ATLAS_VERSION,
  DISCOVERY_SEEDS,
  HELDOUT_SEEDS,
  CONTROL_NAMES,
  average,
  cellFor,
  type Atlas,
  type Cell,
  type Metrics
} from '../../src/lib/atlas/types';
import { validateSearch, validateAtlas } from '../../src/lib/atlas/validation';
import {
  computeArmBundleSha256,
  deserializeArmBundle,
  type SerializedArmBundle
} from '../training/export-arms';
import { sha256Hex } from '../training/fsio';
import { loadLocalAssets } from '../experiments/local-assets';
import { evaluateBehavior } from './evaluate';

export async function publishAtlas(
  input: unknown,
  data = 'public/data',
  requireDiversity = true
): Promise<Atlas> {
  const source = validateSearch(input),
    assets = await loadLocalAssets(data),
    graph = assets.parsedBiological;
  const bundle = source.bundle as unknown as SerializedArmBundle;
  if (
    source.graphArtifactSha256 !== assets.manifest.gzipSha256 ||
    source.bundleSha256 !== computeArmBundleSha256(bundle) ||
    bundle.sha256 !== source.bundleSha256 ||
    bundle.graphArtifactSha256 !== source.graphArtifactSha256 ||
    bundle.arm !== 'biological' ||
    bundle.graphSource !== 'artifact' ||
    sha256Hex(new Uint8Array(encodeGraphBinary(deserializeArmBundle(bundle)))) !==
      assets.manifest.binarySha256
  ) {
    throw new Error('Search graph identity mismatch');
  }
  const indices = Array.from(outputNeuronIndices(graph));
  if (JSON.stringify(indices) !== JSON.stringify(bundle.outputNeuronIndices))
    throw new Error('Search output neuron mismatch');
  const weights = new Map(
    source.candidates.map((candidate) => [
      candidate.id,
      validateReadoutWeights(
        readoutFromFlat(candidate.theta, source.inputSize, source.hiddenSize),
        graph
      )
    ])
  );
  const evaluations = source.candidates
    .slice()
    .sort((a, b) => a.id - b.id)
    .map((candidate) => ({
      id: candidate.id,
      discovery: DISCOVERY_SEEDS.map(
        (seed) =>
          evaluateBehavior(
            { decoder: 'trained', graph, weights: weights.get(candidate.id)! },
            seed,
            source.options.ticks
          ).metrics
      )
    }));
  const selected = new Map<
    number,
    {
      id: number;
      cell: number;
      quality: number;
      coverage: number;
      turning: number;
      discovery: Metrics[];
    }
  >();
  for (const evaluation of evaluations) {
    const coverage = average(evaluation.discovery.map((m) => m.coverage)),
      turning = average(evaluation.discovery.map((m) => m.turning));
    const quality = average(evaluation.discovery.map((m) => m.movementScore)),
      cell = cellFor(coverage, turning);
    if (!selected.has(cell) || quality > selected.get(cell)!.quality)
      selected.set(cell, { ...evaluation, cell, quality, coverage, turning });
  }
  // This gate runs before any held-out episode; extending discovery cannot peek at its outcomes.
  if (
    requireDiversity &&
    (selected.size < 6 ||
      new Set([...selected.keys()].map((c) => c % 6)).size < 2 ||
      new Set([...selected.keys()].map((c) => Math.floor(c / 6))).size < 3)
  )
    throw new Error('Canonical diversity gate failed; extend discovery before held-out evaluation');
  const disconnected = createDisconnectedGraph(graph);
  const cells: Cell[] = [...selected.values()]
    .sort((a, b) => a.cell - b.cell)
    .map((cell) => {
      const heldout = Object.fromEntries(
        CONTROL_NAMES.map((control) => [
          control,
          HELDOUT_SEEDS.map(
            (seed) =>
              evaluateBehavior(
                {
                  decoder: control === 'silenced' ? 'silenced' : 'trained',
                  graph: control === 'disconnected' ? disconnected : graph,
                  weights: weights.get(cell.id)!
                },
                seed,
                source.options.ticks
              ).metrics
          )
        ])
      ) as Cell['heldout'];
      return {
        ...cell,
        heldout,
        replay: evaluateBehavior(
          { decoder: 'trained', graph, weights: weights.get(cell.id)! },
          HELDOUT_SEEDS[0],
          source.options.ticks,
          true
        ).frames
      };
    });
  return validateAtlas({
    schemaVersion: 1,
    modelVersion: ATLAS_VERSION,
    source,
    graphBinarySha256: assets.manifest.binarySha256,
    configFingerprint: createArenaConfigFingerprint(ARENA_CONFIG),
    outputNeuronIndices: indices,
    evaluations,
    cells,
    authored: HELDOUT_SEEDS.map(
      (seed) => evaluateBehavior({ decoder: 'authored', graph }, seed, source.options.ticks).metrics
    )
  });
}
