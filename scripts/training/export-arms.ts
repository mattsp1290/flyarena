import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { basename, extname, resolve } from 'node:path';

import {
  parseGraphBinary,
  validateGraph,
  type ConnectomeGraph,
  type GraphMetadata
} from '../../src/lib/connectome/format';
import { outputNeuronIndices } from '../../src/lib/connectome/readout';
import { createTraceGraph } from '../../tests/fixtures/trace-graph';
import { createFixtureRewiredTraceGraph } from '../../tests/fixtures/trace-graph-rewire';
import { DEFAULT_GRAPH_ID, loadGraphArtifact } from './export-traces';

/**
 * Exports each experimental arm's exact graph arrays from TypeScript, using
 * the same parser the app/evaluator uses (`src/lib/connectome/format.ts`),
 * so the offline PyTorch trainer (`training/`, WP2/WP3) scores byte-identical
 * graphs to the ones `scripts/training/evaluate.ts` and the browser use —
 * never a graph re-derived in Python
 * (`.agents/plans/trained-readout/00-overview.md`'s "Arm CSR arrays
 * exported from TypeScript/compiled artifacts, not re-derived in Python"
 * key decision).
 *
 * Arms:
 * - `biological`: the artifact (or, in trace-graph dev mode, the fixture)
 *   as parsed, unmodified.
 * - `rewired`: for a real graph, loaded from the offline-compiled rewired
 *   artifact (`--rewired`, e.g. `public/data/malecns-arena-v1-rewired-seed0.bin.gz`,
 *   produced by `scripts/data/rewire.py` — see `docs/data-provenance.md`).
 *   There is no TypeScript rewiring function and this script does not
 *   attempt to reimplement one. For trace-graph dev mode only, `--fixture-rewire`
 *   uses a small, clearly-labeled, test-fixture-only degree-preserving
 *   swap (`tests/fixtures/trace-graph-rewire.ts`) — never usable with
 *   `--graph`, and always recorded with
 *   `provenance.kind === 'rewired-fixture-only-swap'` so it can never be
 *   mistaken for the product's rewired arm.
 * - `disconnected`: the same node/IO arrays as `biological` with every edge
 *   removed (`edgeCount: 0`), matching the closed-loop bean's runtime
 *   zero-edge-graph convention (`flyarena-bb45`, in parallel development) —
 *   recorded with `provenance.kind === 'disconnected-runtime-zero-edge'`.
 *
 * Gate: `D` (the output-neuron count, `outputNeuronIndices(graph).length`)
 * must be equal across every arm actually exported, or this script exits
 * non-zero. Rewiring/disconnection preserve the node set by construction,
 * so this is expected to hold trivially; the gate exists to catch a
 * provenance mismatch (e.g. a `--rewired` artifact from a different graph)
 * rather than to guard against an expected failure mode.
 */

const sha256Hex = (data: Uint8Array | string): string => createHash('sha256').update(data).digest('hex');

export type ArmName = 'biological' | 'rewired' | 'disconnected';

export type ArmProvenance =
  | { readonly kind: 'biological-artifact'; readonly artifactPath: string; readonly artifactSha256: string }
  | { readonly kind: 'biological-trace-graph-fixture' }
  | { readonly kind: 'rewired-artifact'; readonly artifactPath: string; readonly artifactSha256: string }
  | { readonly kind: 'rewired-fixture-only-swap'; readonly seed: number }
  | { readonly kind: 'disconnected-runtime-zero-edge' };

/**
 * One arm's exported CSR bundle: everything `training/`'s (not yet built)
 * PyTorch port needs to reconstruct the exact graph
 * `scripts/training/evaluate.ts` will later rescore against, plus enough
 * provenance to prove it. `sha256` is computed over this object's own
 * canonical JSON with `sha256` itself set to `""` first (see
 * `buildArmBundle`), so it self-certifies the bundle's bytes.
 */
export interface SerializedArmBundle {
  readonly formatVersion: 1;
  readonly arm: ArmName;
  readonly graphId: string;
  readonly graphSource: 'artifact' | 'trace-graph-fixture';
  /**
   * sha256 of the exact biological source bytes (the `--graph` artifact
   * file) or, in trace-graph dev mode, of the trace graph's own canonical
   * serialized arrays. Shared by every arm bundle written in the same
   * `export-arms.ts` invocation — it is what ties them together as "the
   * same graph-sha", and is the output directory name itself
   * (`training/runs/arms/<graphArtifactSha256>/<arm>.json`).
   */
  readonly graphArtifactSha256: string;
  readonly provenance: ArmProvenance;
  readonly metadata: GraphMetadata;
  /** Output-neuron count: the readout's `inputSize`. */
  readonly D: number;
  /** `outputNeuronIndices(graph)`, ascending. */
  readonly outputNeuronIndices: readonly number[];
  readonly biologicalIds: readonly string[];
  readonly presynapticOffsets: readonly number[];
  readonly postsynapticIndices: readonly number[];
  readonly contactMagnitudes: readonly number[];
  readonly presynapticSigns: readonly number[];
  readonly inputChannelIndex: readonly number[];
  readonly inputWeight: readonly number[];
  readonly outputPopulationIndex: readonly number[];
  readonly outputWeight: readonly number[];
  readonly sha256: string;
}

interface PlainGraphArrays {
  biologicalIds: string[];
  presynapticOffsets: number[];
  postsynapticIndices: number[];
  contactMagnitudes: number[];
  presynapticSigns: number[];
  inputChannelIndex: number[];
  inputWeight: number[];
  outputPopulationIndex: number[];
  outputWeight: number[];
}

const toPlainArrays = (graph: Readonly<ConnectomeGraph>): PlainGraphArrays => ({
  biologicalIds: Array.from(graph.biologicalIds, (id) => id.toString()),
  presynapticOffsets: Array.from(graph.presynapticOffsets),
  postsynapticIndices: Array.from(graph.postsynapticIndices),
  contactMagnitudes: Array.from(graph.contactMagnitudes),
  presynapticSigns: Array.from(graph.presynapticSigns),
  inputChannelIndex: Array.from(graph.inputChannelIndex),
  inputWeight: Array.from(graph.inputWeight),
  outputPopulationIndex: Array.from(graph.outputPopulationIndex),
  outputWeight: Array.from(graph.outputWeight)
});

/** Reverse of `toPlainArrays` + the bundle's own `metadata`: rebuild a `ConnectomeGraph` and validate it. */
export const deserializeArmBundle = (bundle: Readonly<SerializedArmBundle>): ConnectomeGraph => {
  const graph: ConnectomeGraph = {
    metadata: bundle.metadata,
    biologicalIds: BigUint64Array.from(bundle.biologicalIds.map((id) => BigInt(id))),
    presynapticOffsets: Uint32Array.from(bundle.presynapticOffsets),
    postsynapticIndices: Uint32Array.from(bundle.postsynapticIndices),
    contactMagnitudes: Float32Array.from(bundle.contactMagnitudes),
    presynapticSigns: Int8Array.from(bundle.presynapticSigns),
    inputChannelIndex: Int32Array.from(bundle.inputChannelIndex),
    inputWeight: Float32Array.from(bundle.inputWeight),
    outputPopulationIndex: Int32Array.from(bundle.outputPopulationIndex),
    outputWeight: Float32Array.from(bundle.outputWeight)
  };
  return validateGraph(graph);
};

/**
 * A graph with the same node set and I/O maps as `graph` but `edgeCount: 0`
 * (empty CSR rows), matching the closed-loop bean's runtime disconnected
 * mode: "the same graph with edgeCount 0"
 * (`.agents/plans/trained-readout/04-authoritative-evaluation-and-artifacts.md`).
 */
export const toDisconnectedGraph = (graph: Readonly<ConnectomeGraph>): ConnectomeGraph => ({
  metadata: { ...graph.metadata, edgeCount: 0 },
  biologicalIds: graph.biologicalIds.slice(),
  presynapticOffsets: new Uint32Array(graph.metadata.neuronCount + 1),
  postsynapticIndices: new Uint32Array(0),
  contactMagnitudes: new Float32Array(0),
  presynapticSigns: graph.presynapticSigns.slice(),
  inputChannelIndex: graph.inputChannelIndex.slice(),
  inputWeight: graph.inputWeight.slice(),
  outputPopulationIndex: graph.outputPopulationIndex.slice(),
  outputWeight: graph.outputWeight.slice()
});

interface GraphIdentity {
  readonly graphId: string;
  readonly graphSource: 'artifact' | 'trace-graph-fixture';
  readonly graphArtifactPath: string | null;
  readonly graphArtifactSha256: string;
  readonly graph: ConnectomeGraph;
}

const canonicalGraphHashInput = (graph: Readonly<ConnectomeGraph>): string =>
  JSON.stringify({ metadata: graph.metadata, ...toPlainArrays(graph) });

/**
 * Resolve which graph `export-arms.ts` (and, by the same computation,
 * `evaluate.ts`'s default `--arms-dir`) is working with. `--graph` omitted
 * means trace-graph dev mode (`tests/fixtures/trace-graph.ts`); its
 * `graphArtifactSha256` is a content hash of its own canonical arrays
 * (deterministic: `createTraceGraph()` always returns numerically
 * identical values), not a file hash, since there is no file.
 */
export const computeGraphIdentity = (graphPath?: string): GraphIdentity => {
  if (graphPath) {
    const raw = readFileSync(graphPath);
    const graph = loadGraphArtifact(graphPath);
    validateGraph(graph);
    return {
      graphId: basename(graphPath, extname(graphPath)),
      graphSource: 'artifact',
      graphArtifactPath: graphPath,
      graphArtifactSha256: sha256Hex(raw),
      graph
    };
  }
  const graph = createTraceGraph();
  validateGraph(graph);
  return {
    graphId: DEFAULT_GRAPH_ID,
    graphSource: 'trace-graph-fixture',
    graphArtifactPath: null,
    graphArtifactSha256: sha256Hex(canonicalGraphHashInput(graph)),
    graph
  };
};

const buildArmBundle = (
  arm: ArmName,
  graph: Readonly<ConnectomeGraph>,
  provenance: ArmProvenance,
  identity: Pick<GraphIdentity, 'graphId' | 'graphSource' | 'graphArtifactSha256'>
): SerializedArmBundle => {
  const indices = Array.from(outputNeuronIndices(graph));
  const withoutHash: Omit<SerializedArmBundle, 'sha256'> = {
    formatVersion: 1,
    arm,
    graphId: identity.graphId,
    graphSource: identity.graphSource,
    graphArtifactSha256: identity.graphArtifactSha256,
    provenance,
    metadata: graph.metadata,
    D: indices.length,
    outputNeuronIndices: indices,
    ...toPlainArrays(graph)
  };
  const sha256 = sha256Hex(JSON.stringify({ ...withoutHash, sha256: '' }));
  return { ...withoutHash, sha256 };
};

export class ExportArmsGateError extends Error {}

export const DEFAULT_ARMS_OUT_DIR = 'training/runs/arms';

export interface ExportArmsArgs {
  readonly graphPath?: string;
  readonly rewiredPath?: string;
  readonly fixtureRewire: boolean;
  readonly fixtureRewireSeed: number;
  readonly outDir: string;
}

const requireValue = (flag: string, value: string | undefined): string => {
  if (value === undefined || value.startsWith('--')) {
    throw new Error(`${flag} requires a value`);
  }
  return value;
};

export const parseExportArmsArgs = (argv: readonly string[]): ExportArmsArgs => {
  let graphPath: string | undefined;
  let rewiredPath: string | undefined;
  let fixtureRewire = false;
  let fixtureRewireSeed = 0;
  let outDir = DEFAULT_ARMS_OUT_DIR;

  for (let index = 0; index < argv.length; index += 1) {
    const flag = argv[index];
    if (flag === '--graph') {
      graphPath = requireValue(flag, argv[index + 1]);
      index += 1;
    } else if (flag === '--rewired') {
      rewiredPath = requireValue(flag, argv[index + 1]);
      index += 1;
    } else if (flag === '--fixture-rewire') {
      fixtureRewire = true;
    } else if (flag === '--fixture-rewire-seed') {
      const value = requireValue(flag, argv[index + 1]);
      const parsed = Number(value);
      if (!Number.isInteger(parsed)) throw new Error(`--fixture-rewire-seed must be an integer, got "${value}"`);
      fixtureRewireSeed = parsed;
      index += 1;
    } else if (flag === '--out') {
      outDir = requireValue(flag, argv[index + 1]);
      index += 1;
    } else {
      throw new Error(`Unknown argument: ${flag}`);
    }
  }

  if (fixtureRewire && graphPath) {
    throw new Error(
      '--fixture-rewire is a trace-graph-only development aid and cannot be combined with ' +
        "--graph: a real graph's rewired arm must come from --rewired (the offline compiler's " +
        'output), never a fixture swap.'
    );
  }
  if (rewiredPath && !graphPath) {
    throw new Error(
      '--rewired requires --graph: a rewired artifact is only meaningful paired with the ' +
        'biological graph it was rewired from.'
    );
  }

  return { graphPath, rewiredPath, fixtureRewire, fixtureRewireSeed, outDir };
};

export interface ExportArmsResult {
  readonly outDir: string;
  readonly written: readonly ArmName[];
  readonly d: number;
}

/** Core export logic, separated from CLI parsing/`main` so tests can call it in-process. */
export const runExportArms = (args: Readonly<ExportArmsArgs>): ExportArmsResult => {
  const identity = computeGraphIdentity(args.graphPath);
  const bundles: Partial<Record<ArmName, SerializedArmBundle>> = {};

  bundles.biological = buildArmBundle(
    'biological',
    identity.graph,
    identity.graphSource === 'artifact'
      ? {
          kind: 'biological-artifact',
          artifactPath: identity.graphArtifactPath as string,
          artifactSha256: identity.graphArtifactSha256
        }
      : { kind: 'biological-trace-graph-fixture' },
    identity
  );

  bundles.disconnected = buildArmBundle(
    'disconnected',
    toDisconnectedGraph(identity.graph),
    { kind: 'disconnected-runtime-zero-edge' },
    identity
  );

  if (args.rewiredPath) {
    const raw = readFileSync(args.rewiredPath);
    const rewiredGraph = loadGraphArtifact(args.rewiredPath);
    validateGraph(rewiredGraph);
    bundles.rewired = buildArmBundle(
      'rewired',
      rewiredGraph,
      { kind: 'rewired-artifact', artifactPath: args.rewiredPath, artifactSha256: sha256Hex(raw) },
      identity
    );
  } else if (args.fixtureRewire) {
    const rewiredGraph = createFixtureRewiredTraceGraph(identity.graph, args.fixtureRewireSeed);
    validateGraph(rewiredGraph);
    bundles.rewired = buildArmBundle(
      'rewired',
      rewiredGraph,
      { kind: 'rewired-fixture-only-swap', seed: args.fixtureRewireSeed },
      identity
    );
  } else {
    // eslint-disable-next-line no-console -- CLI tool: user-facing diagnostic.
    console.warn(
      'export-arms: no --rewired artifact and --fixture-rewire not set; exporting biological + ' +
        'disconnected only. Pass --rewired <path> for a real graph (required), or ' +
        '--fixture-rewire for trace-graph dev/testing only.'
    );
  }

  const entries = Object.entries(bundles) as Array<[ArmName, SerializedArmBundle]>;
  const distinctD = new Set(entries.map(([, bundle]) => bundle.D));
  if (distinctD.size > 1) {
    const detail = entries.map(([arm, bundle]) => `${arm}=${bundle.D}`).join(', ');
    throw new ExportArmsGateError(`D (output-neuron count) differs across arms: ${detail}`);
  }

  const outDir = resolve(process.cwd(), args.outDir, identity.graphArtifactSha256);
  mkdirSync(outDir, { recursive: true });
  const written: ArmName[] = [];
  for (const [arm, bundle] of entries) {
    writeFileSync(resolve(outDir, `${arm}.json`), JSON.stringify(bundle));
    written.push(arm);
  }

  return { outDir, written, d: entries[0][1].D };
};

const main = (): void => {
  const args = parseExportArmsArgs(process.argv.slice(2));
  try {
    const { outDir, written, d } = runExportArms(args);
    // eslint-disable-next-line no-console -- CLI tool: this is its user-facing output.
    console.log(`export-arms: wrote ${written.length} bundle(s) to ${outDir} (D=${d}): ${written.join(', ')}`);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    // eslint-disable-next-line no-console -- CLI tool: this is its user-facing error output.
    console.error(`export-arms failed: ${message}`);
    process.exit(1);
  }
};

if (process.argv[1] === fileURLToPath(import.meta.url)) main();
