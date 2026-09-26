import { ARENA_CONFIG, createArenaConfigFingerprint } from '../../src/lib/arena/config';
import {
  createDisconnectedGraph,
  encodeGraphBinary,
  type ConnectomeGraph
} from '../../src/lib/connectome/format';
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
  type Control,
  type Metrics,
  type SearchArtifact
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

/**
 * `evaluateSearchOnGraph`'s options. `heldout: 'all'` reproduces the
 * shipped atlas's three `CONTROL_NAMES` controls (`'biological'` here means
 * "trained decoder on the searched graph", not literally the biological
 * connectome -- for the shipped path the searched graph IS biological, so
 * this is unchanged behavior). `heldout: 'own'` evaluates only that one
 * control, for a null graph (rewired/disconnected) where "disconnected" and
 * "silenced" controls would either be redundant or meaningless
 * (`.agents/plans/repertoire-null/01-generalize-atlas-pipeline.md` WP1).
 */
export interface GraphEvaluationOptions {
  readonly requireDiversity: boolean;
  readonly heldout: 'all' | 'own';
}

/** Same shape as `Cell`, except `heldout` only carries the controls
 * `GraphEvaluationOptions.heldout` actually evaluated -- `'own'` mode
 * populates just the `'biological'`-named control (see this module's own
 * doc comment on `GraphEvaluationOptions`). */
export interface GraphCell {
  readonly id: number;
  readonly cell: number;
  readonly quality: number;
  readonly coverage: number;
  readonly turning: number;
  readonly discovery: Metrics[];
  readonly heldout: Partial<Record<Control, Metrics[]>>;
  readonly replay: Cell['replay'];
}

export interface GraphEvaluationResult {
  readonly evaluations: { id: number; discovery: Metrics[] }[];
  readonly cells: GraphCell[];
  /** `source.candidates.length` -- the GPU search's own per-cell archive size (an audit field, not every evaluated candidate; see `docs/behavior-atlas-validation.md`). */
  readonly gpuArchiveSize: number;
  /** Candidates whose TS-rebinned cell was already occupied by an earlier (by id) candidate -- the same rebinning/collision rule `occupied` (36 minus empty cells) is defined against. */
  readonly collisions: number;
}

/**
 * The graph-parameterized core of the atlas pipeline: re-evaluate a GPU
 * search's candidates on `graph` (discovery), rebin into cells with
 * collision resolution, optionally gate on canonical diversity, then
 * evaluate held-out controls per cell. Extracted out of `publishAtlas` so
 * the same evaluation code re-runs for rewired/disconnected null graphs
 * (WP2) with TS staying authoritative for every graph -- never a parallel
 * copy that could drift (`01-generalize-atlas-pipeline.md`'s key decision).
 * `publishAtlas` keeps its own biological-specific identity checks and
 * calls this with `heldout: 'all'`, producing byte-identical output to
 * before this extraction.
 */
export async function evaluateSearchOnGraph(
  source: SearchArtifact,
  graph: Readonly<ConnectomeGraph>,
  { requireDiversity, heldout }: GraphEvaluationOptions
): Promise<GraphEvaluationResult> {
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
  let collisions = 0;
  for (const evaluation of evaluations) {
    const coverage = average(evaluation.discovery.map((m) => m.coverage)),
      turning = average(evaluation.discovery.map((m) => m.turning));
    const quality = average(evaluation.discovery.map((m) => m.movementScore)),
      cell = cellFor(coverage, turning);
    if (!selected.has(cell)) {
      selected.set(cell, { ...evaluation, cell, quality, coverage, turning });
    } else {
      collisions += 1;
      if (quality > selected.get(cell)!.quality)
        selected.set(cell, { ...evaluation, cell, quality, coverage, turning });
    }
  }
  // This gate runs before any held-out episode; extending discovery cannot peek at its outcomes.
  if (
    requireDiversity &&
    (selected.size < 6 ||
      new Set([...selected.keys()].map((c) => c % 6)).size < 2 ||
      new Set([...selected.keys()].map((c) => Math.floor(c / 6))).size < 3)
  )
    throw new Error('Canonical diversity gate failed; extend discovery before held-out evaluation');
  const controls: readonly Control[] = heldout === 'all' ? CONTROL_NAMES : ['biological'];
  const disconnected = heldout === 'all' ? createDisconnectedGraph(graph) : undefined;
  const cells: GraphCell[] = [...selected.values()]
    .sort((a, b) => a.cell - b.cell)
    .map((cell) => {
      const heldoutMetrics = Object.fromEntries(
        controls.map((control) => [
          control,
          HELDOUT_SEEDS.map(
            (seed) =>
              evaluateBehavior(
                {
                  decoder: control === 'silenced' ? 'silenced' : 'trained',
                  graph: control === 'disconnected' ? disconnected! : graph,
                  weights: weights.get(cell.id)!
                },
                seed,
                source.options.ticks
              ).metrics
          )
        ])
      ) as Partial<Record<Control, Metrics[]>>;
      return {
        ...cell,
        heldout: heldoutMetrics,
        replay: evaluateBehavior(
          { decoder: 'trained', graph, weights: weights.get(cell.id)! },
          HELDOUT_SEEDS[0],
          source.options.ticks,
          true
        ).frames
      };
    });
  return { evaluations, cells, gpuArchiveSize: source.candidates.length, collisions };
}

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
  const { evaluations, cells } = await evaluateSearchOnGraph(source, graph, {
    requireDiversity,
    heldout: 'all'
  });
  return validateAtlas({
    schemaVersion: 1,
    modelVersion: ATLAS_VERSION,
    source,
    graphBinarySha256: assets.manifest.binarySha256,
    configFingerprint: createArenaConfigFingerprint(ARENA_CONFIG),
    outputNeuronIndices: indices,
    evaluations,
    cells: cells as unknown as Cell[],
    authored: HELDOUT_SEEDS.map(
      (seed) => evaluateBehavior({ decoder: 'authored', graph }, seed, source.options.ticks).metrics
    )
  });
}
