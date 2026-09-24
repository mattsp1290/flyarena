import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { existsSync, mkdirSync, readFileSync, unlinkSync, writeFileSync } from 'node:fs';
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
 * must be equal across every arm actually exported, AND every non-biological
 * arm's node set / I/O maps / dynamics metadata (`biologicalIds`,
 * `outputNeuronIndices`, `outputPopulationIndex`, `inputChannelIndex`,
 * `inputWeight`, `outputWeight`, `presynapticSigns`, and every
 * `GraphMetadata` field except `edgeCount`) must exactly match the
 * biological arm's (`assertMatchingNodeSet`), or this script exits
 * non-zero. Rewiring/disconnection preserve the node set by construction,
 * so this is expected to hold trivially; the gate exists to catch a
 * provenance mismatch (e.g. a `--rewired` artifact from a different graph,
 * or one whose output neurons landed in a different order) rather than to
 * guard against an expected failure mode. D-equality alone cannot catch
 * either of those — see `assertMatchingNodeSet`'s doc comment.
 */

const sha256Hex = (data: Uint8Array | string): string => createHash('sha256').update(data).digest('hex');

export type ArmName = 'biological' | 'rewired' | 'disconnected';

export type ArmProvenance =
  | { readonly kind: 'biological-artifact'; readonly artifactPath: string; readonly artifactSha256: string }
  | { readonly kind: 'biological-trace-graph-fixture' }
  | {
      readonly kind: 'rewired-artifact';
      readonly artifactPath: string;
      readonly artifactSha256: string;
      /**
       * The rewiring seed recorded in the biological graph's sibling
       * `<graph>.manifest.json` (`rewiredArms.<key>.swapStats.seed`,
       * matched by this artifact's own gzip sha256 — see
       * `docs/data-provenance.md`'s "Rewired control arm" section), when
       * that manifest is present and has a matching entry. `null` when no
       * matching entry was found (e.g. a rewired artifact not produced by
       * `scripts/data/rewire.py`, or no manifest alongside `--graph`) —
       * best-effort provenance, not a requirement.
       */
      readonly rewiringSeed: number | null;
    }
  | { readonly kind: 'rewired-fixture-only-swap'; readonly seed: number }
  | { readonly kind: 'disconnected-runtime-zero-edge' };

/** The exact `ArmProvenance.kind` a valid bundle must have for a given arm + graph source. */
export const expectedProvenanceKind = (
  arm: ArmName,
  graphSource: 'artifact' | 'trace-graph-fixture'
): ArmProvenance['kind'] => {
  if (arm === 'disconnected') return 'disconnected-runtime-zero-edge';
  if (arm === 'biological') {
    return graphSource === 'artifact' ? 'biological-artifact' : 'biological-trace-graph-fixture';
  }
  return graphSource === 'artifact' ? 'rewired-artifact' : 'rewired-fixture-only-swap';
};

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

/**
 * The bundle's self-certifying hash: sha256 over its own canonical JSON
 * with `sha256` itself set to `""`. Shared by `buildArmBundle` (which
 * writes it) and `evaluate.ts`'s `loadArmGraphs` (which recomputes it to
 * verify a bundle was not hand-edited or swapped after export — a bundle's
 * `sha256` field is otherwise a self-declared claim, not a check).
 */
export const computeArmBundleSha256 = (bundle: Omit<SerializedArmBundle, 'sha256'>): string =>
  sha256Hex(JSON.stringify({ ...bundle, sha256: '' }));

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
  return { ...withoutHash, sha256: computeArmBundleSha256(withoutHash) };
};

export class ExportArmsGateError extends Error {}

/**
 * Best-effort lookup of the rewiring seed for `--rewired <rewiredPath>`,
 * paired with the biological graph at `graphPath`. Reads
 * `<graphPath-without-its-.bin[.gz]-extension>.manifest.json` (the
 * convention `scripts/data/compile.py`/`rewire.py` use — see
 * `public/data/malecns-arena-v1.manifest.json`'s `rewiredArms` block) and
 * matches its entries by `gzipSha256` against `rewiredPath`'s own raw
 * bytes. Returns `null` on any miss (no manifest, no matching entry, or a
 * malformed one) — this is provenance enrichment, not a validity gate; the
 * real pairing proof is `assertMatchingNodeSet` below, which is exact.
 */
const lookupRewiringSeed = (graphPath: string, rewiredArtifactSha256: string): number | null => {
  try {
    const base = graphPath.endsWith('.bin.gz')
      ? graphPath.slice(0, -'.bin.gz'.length)
      : graphPath.replace(extname(graphPath), '');
    const manifestPath = `${base}.manifest.json`;
    if (!existsSync(manifestPath)) return null;
    const manifest = JSON.parse(readFileSync(manifestPath, 'utf8')) as {
      rewiredArms?: Record<string, { gzipSha256?: string; swapStats?: { seed?: number } }>;
    };
    for (const entry of Object.values(manifest.rewiredArms ?? {})) {
      if (entry.gzipSha256 === rewiredArtifactSha256 && typeof entry.swapStats?.seed === 'number') {
        return entry.swapStats.seed;
      }
    }
    return null;
  } catch {
    return null;
  }
};

const sameNumericArray = (a: ArrayLike<number>, b: ArrayLike<number>): boolean =>
  a.length === b.length && Array.prototype.every.call(a, (value: number, index: number) => value === b[index]);
const sameStringArray = (a: ArrayLike<string>, b: ArrayLike<string>): boolean =>
  a.length === b.length && Array.prototype.every.call(a, (value: string, index: number) => value === b[index]);

/**
 * Exact node-set / I/O-map / dynamics-parameter equality against the
 * biological bundle. `D` equality alone (this script's older gate) cannot
 * catch a `--rewired` artifact compiled from a different graph (or a
 * different calibration of the same graph) that happens to have the same
 * output-neuron count — degree-preserving rewiring preserves the node set,
 * every per-neuron I/O mapping, sign ownership, and every dynamics
 * parameter exactly (`docs/data-provenance.md`'s "Rewired control arm"
 * section: only `edgeCount` and the CSR edge arrays themselves may differ,
 * and `edgeCount` is excluded here only because `disconnected`'s
 * `edgeCount: 0` is an intentional, expected difference for that one arm —
 * see `toDisconnectedGraph`), so every other comparison here is expected
 * to hold, not merely likely to. Verified directly against the real
 * MaleCNS biological/rewired-seed0 artifacts: identical on every field
 * this checks.
 */
const assertMatchingNodeSet = (biological: SerializedArmBundle, other: SerializedArmBundle): void => {
  const mismatches: string[] = [];
  // Every metadata field except edgeCount (disconnected's one expected,
  // intentional difference): timestepSeconds, leakRate, rateMin/rateMax,
  // inputClampMin/inputClampMax, globalGain, neuronCount, etc.
  const { edgeCount: _bioEdgeCount, ...biologicalMetadataSansEdgeCount } = biological.metadata;
  const { edgeCount: _otherEdgeCount, ...otherMetadataSansEdgeCount } = other.metadata;
  if (JSON.stringify(otherMetadataSansEdgeCount) !== JSON.stringify(biologicalMetadataSansEdgeCount)) {
    mismatches.push('metadata (excluding edgeCount)');
  }
  if (!sameStringArray(other.biologicalIds, biological.biologicalIds)) mismatches.push('biologicalIds');
  if (!sameNumericArray(other.outputNeuronIndices, biological.outputNeuronIndices)) {
    mismatches.push('outputNeuronIndices');
  }
  if (!sameNumericArray(other.outputPopulationIndex, biological.outputPopulationIndex)) {
    mismatches.push('outputPopulationIndex');
  }
  if (!sameNumericArray(other.inputChannelIndex, biological.inputChannelIndex)) mismatches.push('inputChannelIndex');
  if (!sameNumericArray(other.inputWeight, biological.inputWeight)) mismatches.push('inputWeight');
  if (!sameNumericArray(other.outputWeight, biological.outputWeight)) mismatches.push('outputWeight');
  if (!sameNumericArray(other.presynapticSigns, biological.presynapticSigns)) mismatches.push('presynapticSigns');
  if (mismatches.length > 0) {
    throw new ExportArmsGateError(
      `arm "${other.arm}" node set / I/O map / dynamics differs from "biological" (${mismatches.join(', ')}); ` +
        `is --rewired paired with the wrong biological --graph?`
    );
  }
};

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
    const artifactSha256 = sha256Hex(raw);
    const rewiringSeed = args.graphPath ? lookupRewiringSeed(args.graphPath, artifactSha256) : null;
    bundles.rewired = buildArmBundle(
      'rewired',
      rewiredGraph,
      { kind: 'rewired-artifact', artifactPath: args.rewiredPath, artifactSha256, rewiringSeed },
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
  // D equality alone cannot catch a --rewired artifact compiled from a
  // different graph with the same output-neuron count (or with its output
  // neurons in a different order) — see assertMatchingNodeSet's doc
  // comment. This is exact, not a count heuristic.
  for (const [, bundle] of entries) {
    if (bundle.arm !== 'biological') assertMatchingNodeSet(bundles.biological!, bundle);
  }

  const outDir = resolve(process.cwd(), args.outDir, identity.graphArtifactSha256);
  mkdirSync(outDir, { recursive: true });
  const written: ArmName[] = [];
  for (const [arm, bundle] of entries) {
    writeFileSync(resolve(outDir, `${arm}.json`), JSON.stringify(bundle));
    written.push(arm);
  }
  // Stale-bundle cleanup: an arm not written this invocation (most
  // commonly "rewired", when re-exporting without --rewired/--fixture-rewire)
  // must not leave a bundle from a previous invocation behind for
  // evaluate.ts to silently pick up.
  for (const arm of ['biological', 'rewired', 'disconnected'] as const) {
    if (written.includes(arm)) continue;
    const stalePath = resolve(outDir, `${arm}.json`);
    if (existsSync(stalePath)) {
      unlinkSync(stalePath);
      // eslint-disable-next-line no-console -- CLI tool: user-facing diagnostic.
      console.warn(`export-arms: removed stale ${stalePath} (not exported this run)`);
    }
  }

  return { outDir, written, d: entries[0][1].D };
};

const main = (): void => {
  try {
    const args = parseExportArmsArgs(process.argv.slice(2));
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
