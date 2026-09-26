import { ARENA_CONFIG, createArenaConfigFingerprint } from '../../src/lib/arena/config';
import {
  createDisconnectedGraph,
  encodeGraphBinary,
  type ConnectomeGraph
} from '../../src/lib/connectome/format';
import {
  outputNeuronIndices,
  validateReadoutWeights,
  type ReadoutWeights
} from '../../src/lib/connectome/readout';
import { readoutFromFlat } from '../../src/lib/connectome/readout-serialization';
import {
  ATLAS_VERSION,
  DISCOVERY_SEEDS,
  HELDOUT_SEEDS,
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
 *
 * This module exposes `evaluateSearchOnGraph` as two overloads keyed on the
 * literal `heldout` value, so a caller that passes `heldout: 'all'` gets
 * back `GraphEvaluationResultAll` (`cells[].heldout` fully populated, the
 * same shape as `Cell['heldout']`) and a caller that passes `heldout: 'own'`
 * gets `GraphEvaluationResultOwn` (`cells[].heldout` only ever has the
 * `'biological'`-named key) -- `publishAtlas` (always `'all'`) therefore
 * needs no cast to satisfy `Atlas.cells: Cell[]` (a thermo-maintainability
 * review finding: the previous single non-discriminated return type forced
 * an `as unknown as Cell[]` cast at that call site).
 */
export interface GraphEvaluationOptions {
  readonly requireDiversity: boolean;
  readonly heldout: 'all' | 'own';
}

/** Every `Cell` field except `heldout`, which the two `GraphEvaluationResult*`
 * variants below type differently. */
export interface GraphCellBase {
  readonly id: number;
  readonly cell: number;
  readonly quality: number;
  readonly coverage: number;
  readonly turning: number;
  readonly discovery: Metrics[];
  readonly replay: Cell['replay'];
}

export interface GraphEvaluationResultAll {
  readonly evaluations: { id: number; discovery: Metrics[] }[];
  /** Fully populated -- structurally identical to `Cell` (`GraphCellBase & { heldout: Cell['heldout'] }`). */
  readonly cells: (GraphCellBase & { readonly heldout: Record<Control, Metrics[]> })[];
  /** `source.candidates.length` -- the GPU search's own per-cell archive size (an audit field, not every evaluated candidate; see `docs/behavior-atlas-validation.md`). */
  readonly gpuArchiveSize: number;
  /** Candidates whose TS-rebinned cell was already occupied by an earlier (by id) candidate -- the same rebinning/collision rule `occupied` (36 minus empty cells) is defined against. */
  readonly collisions: number;
}

export interface GraphEvaluationResultOwn {
  readonly evaluations: { id: number; discovery: Metrics[] }[];
  /** Only the `'biological'`-named control (see `GraphEvaluationOptions`'s doc comment). */
  readonly cells: (GraphCellBase & { readonly heldout: Partial<Record<Control, Metrics[]>> })[];
  readonly gpuArchiveSize: number;
  readonly collisions: number;
}

export type GraphEvaluationResult = GraphEvaluationResultAll | GraphEvaluationResultOwn;

/**
 * The search JSON's own self-consistency with its embedded bundle: the
 * bundle's self-certifying sha256, and that the search JSON's top-level
 * `graphArtifactSha256` agrees with the bundle's own. Shared by
 * `publishAtlas` (checked alongside the shipped biological manifest) and
 * `verify-search-graph.ts`'s `verifyAndEvaluateSearchGraph` (checked against
 * an arbitrary verified graph, with no external manifest to lean on) --
 * extracted per a thermo-maintainability review finding that the two
 * three-line, order-sensitive copies were drift-prone duplication.
 */
export const isBundleSelfConsistent = (
  source: Pick<SearchArtifact, 'bundleSha256' | 'graphArtifactSha256'>,
  bundle: Readonly<SerializedArmBundle>
): boolean =>
  source.bundleSha256 === computeArmBundleSha256(bundle) &&
  bundle.sha256 === source.bundleSha256 &&
  bundle.graphArtifactSha256 === source.graphArtifactSha256;

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
export function evaluateSearchOnGraph(
  source: SearchArtifact,
  graph: Readonly<ConnectomeGraph>,
  options: { readonly requireDiversity: boolean; readonly heldout: 'all' }
): Promise<GraphEvaluationResultAll>;
export function evaluateSearchOnGraph(
  source: SearchArtifact,
  graph: Readonly<ConnectomeGraph>,
  options: { readonly requireDiversity: boolean; readonly heldout: 'own' }
): Promise<GraphEvaluationResultOwn>;
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

  // One held-out control episode, shared by both branches below.
  const evalControl = (
    control: Control,
    controlGraph: Readonly<ConnectomeGraph>,
    controlWeights: Readonly<ReadoutWeights>
  ): Metrics[] =>
    HELDOUT_SEEDS.map(
      (seed) =>
        evaluateBehavior(
          { decoder: control === 'silenced' ? 'silenced' : 'trained', graph: controlGraph, weights: controlWeights },
          seed,
          source.options.ticks
        ).metrics
    );
  const replayFor = (controlWeights: Readonly<ReadoutWeights>) =>
    evaluateBehavior({ decoder: 'trained', graph, weights: controlWeights }, HELDOUT_SEEDS[0], source.options.ticks, true)
      .frames;
  const ordered = [...selected.values()].sort((a, b) => a.cell - b.cell);

  if (heldout === 'all') {
    // Computed once (not per cell): a pure function of `graph` alone, same
    // as the pre-refactor code's single top-level `createDisconnectedGraph`
    // call. Declaring it here, rather than as a top-level `| undefined`
    // dereferenced with `!` inside the per-cell map, means every reference
    // to it below is unconditionally defined -- nothing for the type
    // checker (or a future reader) to take on faith.
    const disconnectedGraph = createDisconnectedGraph(graph);
    const cells: GraphEvaluationResultAll['cells'] = ordered.map((cell) => {
      const cellWeights = weights.get(cell.id)!;
      const heldoutMetrics: Record<Control, Metrics[]> = {
        biological: evalControl('biological', graph, cellWeights),
        disconnected: evalControl('disconnected', disconnectedGraph, cellWeights),
        silenced: evalControl('silenced', graph, cellWeights)
      };
      return { ...cell, heldout: heldoutMetrics, replay: replayFor(cellWeights) };
    });
    return { evaluations, cells, gpuArchiveSize: source.candidates.length, collisions };
  }

  const cells: GraphEvaluationResultOwn['cells'] = ordered.map((cell) => {
    const cellWeights = weights.get(cell.id)!;
    const heldoutMetrics: Partial<Record<Control, Metrics[]>> = {
      biological: evalControl('biological', graph, cellWeights)
    };
    return { ...cell, heldout: heldoutMetrics, replay: replayFor(cellWeights) };
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
    !isBundleSelfConsistent(source, bundle) ||
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
    cells,
    authored: HELDOUT_SEEDS.map(
      (seed) => evaluateBehavior({ decoder: 'authored', graph }, seed, source.options.ticks).metrics
    )
  });
}
