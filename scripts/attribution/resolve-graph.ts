import { dirname, resolve } from 'node:path';
import { readFileSync } from 'node:fs';

import type { ConnectomeGraph } from '../../src/lib/connectome/format';
import { sha256Hex } from '../training/fsio';
import { graphFromTaskMode, loadVerifiedGraphBinary } from '../null/null-worker-shared';
import type { ArchivedReadout } from './archive-readouts';
import { resolveBigqGraphIdentity, resolveInterventionGraphIdentity, type GraphIdentity } from './graph-identity';
import { readAndValidateInterventionIndex } from './intervention-index';

/**
 * `.agents/plans/readout-attribution/02-analyses.md`'s WP2: maps one
 * archived readout to its actual `ConnectomeGraph`, sha-verified end to end
 * -- never against `armBundleSha256` (path-dependent, doesn't reproduce for
 * the intervention bundles -- see `graph-identity.ts`'s module doc comment
 * and the jfgv provenance review, 23c9557), always against the archive
 * entry's own path-independent `graphGzipSha256`/`graphBinarySha256`.
 *
 * `biological`/`rewired-seed0` come from `public/data/malecns-arena-v1*`
 * (sha-verified against the manifest, `graph-identity.ts`'s
 * `resolveBigqGraphIdentity`); intervention ids (`P`, `C000`-`C004`,
 * `M1000`-`M1004`) come from a **physical** intervention graph list -- a
 * directory holding both an `index.json` (`scripts/analysis/interventions.py`'s
 * output format) and the `graphs/` it points into, either freshly
 * regenerated with `interventions.py` or a byte-identical copy reused from
 * elsewhere (the task's own instructions: "reuse a sha-verified copy...the
 * index sha must equal training/archive/intervention-index-v1.json"). This
 * module enforces that equality itself (`assertPhysicalIndexMatchesArchive`)
 * rather than trusting the caller to have checked it, then verifies the
 * requested `graphId`'s entry against the **archive's own**
 * `graphGzipSha256`/`graphBinarySha256` (not merely the physical index's own
 * claimed values -- `resolveInterventionGraphIdentity` already re-derives
 * those from the actual bytes on disk, but a physical index that was itself
 * tampered with could still claim consistent-with-itself, wrong shas; this
 * module's own cross-check against the committed archive is what catches
 * that).
 *
 * `disconnected` has no separate artifact (`archive-readouts.ts`'s
 * `graphGzipSha256`/`graphBinarySha256` are `null` for it) -- derived here
 * from the sha-verified biological graph via `graphFromTaskMode('disconnected',
 * ...)`, which internally re-encodes `createDisconnectedGraph(biological)`
 * through the wire format and re-parses it, the same round trip
 * `scripts/atlas/verify-search-graph.ts` and the browser's own Worker `init`
 * path both take (`null-worker-shared.ts`'s `graphFromTaskMode` doc
 * comment).
 *
 * Any other `graphId` throws. Nothing here ever falls back to the shipped
 * rewired graph (`public/data/malecns-arena-v1-rewired-seed0.bin.gz`) for an
 * id it doesn't recognize -- a plan requirement (02-analyses.md), guarded
 * structurally: there is no default branch, only an explicit throw.
 */

export interface ResolveGraphConfig {
  /** `public/data/malecns-arena-v1.manifest.json`. */
  readonly manifestPath: string;
  /**
   * A physical `scripts/analysis/interventions.py` output directory's
   * `index.json` -- the one with a real `graphs/` subdirectory alongside it
   * (a fresh `interventions.py` regeneration, or a byte-identical reused
   * copy). Required only when resolving an intervention `graphId`.
   */
  readonly interventionIndexPath?: string;
  /**
   * The committed `training/archive/intervention-index-v1.json` --
   * `interventionIndexPath`'s bytes must equal this file's bytes exactly
   * (`assertPhysicalIndexMatchesArchive`). Required only when resolving an
   * intervention `graphId`.
   */
  readonly archivedInterventionIndexPath?: string;
}

const assertIdentityMatchesArchive = (
  graphId: string,
  identity: Readonly<GraphIdentity>,
  entry: Pick<ArchivedReadout, 'graphGzipSha256' | 'graphBinarySha256'>
): void => {
  if (entry.graphGzipSha256 === null || entry.graphBinarySha256 === null) {
    throw new Error(
      `resolve-graph: archive entry for graphId "${graphId}" has no graphGzipSha256/graphBinarySha256 to verify against`
    );
  }
  if (identity.graphGzipSha256 !== entry.graphGzipSha256 || identity.graphBinarySha256 !== entry.graphBinarySha256) {
    throw new Error(
      `resolve-graph: the regenerated graph for "${graphId}" (gzipSha256 ${identity.graphGzipSha256}, ` +
        `binarySha256 ${identity.graphBinarySha256}) does not match the archived readout's own ` +
        `graphGzipSha256 (${entry.graphGzipSha256}) / graphBinarySha256 (${entry.graphBinarySha256})`
    );
  }
};

/**
 * Refuses to resolve any intervention graph unless the physical index this
 * call was pointed at is byte-identical to the committed archive copy --
 * the task's own "the index sha must equal training/archive/
 * intervention-index-v1.json" requirement, enforced here rather than left
 * as an operator convention.
 */
const assertPhysicalIndexMatchesArchive = (physicalIndexPath: string, archivedIndexPath: string): void => {
  const physicalBytes = readFileSync(physicalIndexPath);
  const archivedBytes = readFileSync(archivedIndexPath);
  const physicalSha = sha256Hex(physicalBytes);
  const archivedSha = sha256Hex(archivedBytes);
  if (physicalSha !== archivedSha) {
    throw new Error(
      `resolve-graph: physical intervention index ${physicalIndexPath} (sha256 ${physicalSha}) does not match ` +
        `the committed archive copy ${archivedIndexPath} (sha256 ${archivedSha})`
    );
  }
};

const loadBiologicalGraphBinary = (manifestPath: string): { readonly graphBinary: ArrayBuffer; readonly identity: GraphIdentity } => {
  const identity = resolveBigqGraphIdentity('biological', manifestPath);
  if (!identity) throw new Error('resolve-graph: "biological" unexpectedly has no graph identity');
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8')) as { readonly artifact: string };
  const artifactPath = resolve(dirname(manifestPath), manifest.artifact);
  const graphBinary = loadVerifiedGraphBinary('resolve-graph', artifactPath, identity.graphBinarySha256);
  return { graphBinary, identity };
};

/**
 * Resolve one archived readout's `graphId` to its verified `ConnectomeGraph`.
 * `entry` need only carry the three fields this function actually reads.
 */
export const resolveReadoutGraph = (
  entry: Pick<ArchivedReadout, 'graphId' | 'graphGzipSha256' | 'graphBinarySha256'>,
  config: Readonly<ResolveGraphConfig>
): ConnectomeGraph => {
  const { graphId } = entry;

  if (graphId === 'biological' || graphId === 'rewired-seed0') {
    const identity = resolveBigqGraphIdentity(graphId, config.manifestPath);
    if (!identity) throw new Error(`resolve-graph: "${graphId}" unexpectedly has no graph identity`);
    assertIdentityMatchesArchive(graphId, identity, entry);
    const manifest = JSON.parse(readFileSync(config.manifestPath, 'utf8')) as {
      readonly artifact: string;
      readonly rewiredArms?: Readonly<Record<string, { readonly artifact: string }>>;
    };
    const manifestDir = dirname(config.manifestPath);
    const artifactPath =
      graphId === 'biological'
        ? resolve(manifestDir, manifest.artifact)
        : resolve(manifestDir, requireSeed0Artifact(manifest, config.manifestPath));
    const graphBinary = loadVerifiedGraphBinary('resolve-graph', artifactPath, identity.graphBinarySha256);
    return graphFromTaskMode(graphId === 'biological' ? 'biological' : 'rewired', graphBinary);
  }

  if (graphId === 'disconnected') {
    if (entry.graphGzipSha256 !== null || entry.graphBinarySha256 !== null) {
      throw new Error('resolve-graph: "disconnected" archive entries must have null graphGzipSha256/graphBinarySha256');
    }
    const { graphBinary } = loadBiologicalGraphBinary(config.manifestPath);
    // `graphFromTaskMode('disconnected', biologicalBinary)` derives
    // `createDisconnectedGraph(biological)`, re-encodes it to the wire
    // format, and re-parses -- see this module's doc comment.
    return graphFromTaskMode('disconnected', graphBinary);
  }

  // Every other graphId is treated as an intervention id (`P`, `C000`-`C004`,
  // `M1000`-`M1004`) -- validated against the physical/archived intervention
  // index below, which throws on an id it has no entry for. There is no
  // further fallback branch: an id neither bigq nor present in the
  // intervention index throws, never silently resolving to the shipped
  // rewired graph.
  if (!config.interventionIndexPath || !config.archivedInterventionIndexPath) {
    throw new Error(
      `resolve-graph: graphId "${graphId}" looks like an intervention id, but interventionIndexPath/` +
        'archivedInterventionIndexPath were not provided'
    );
  }
  assertPhysicalIndexMatchesArchive(config.interventionIndexPath, config.archivedInterventionIndexPath);
  const { index, indexDir } = readAndValidateInterventionIndex(config.interventionIndexPath);
  const identity = resolveInterventionGraphIdentity(graphId, index, indexDir);
  assertIdentityMatchesArchive(graphId, identity, entry);
  const listEntry = index.entries.find((candidate) => candidate.id === graphId);
  if (!listEntry) throw new Error(`resolve-graph: the intervention index has no entry for id "${graphId}"`);
  const graphPath = resolve(indexDir, listEntry.path);
  const graphBinary = loadVerifiedGraphBinary('resolve-graph', graphPath, identity.graphBinarySha256);
  // Every graph-list entry (P/Q/C.../M...) is a full, already-swapped CSR
  // binary -- `mode: 'rewired'` is `null-evaluate.ts`'s own established
  // convention for these (`buildGraphListTasks`'s doc comment: "Every
  // graph-list entry becomes a `mode: 'rewired'` task"), not a claim about
  // topology relative to the shipped rewired-seed0 graph.
  return graphFromTaskMode('rewired', graphBinary);
};

const requireSeed0Artifact = (
  manifest: { readonly rewiredArms?: Readonly<Record<string, { readonly artifact: string }>> },
  manifestPath: string
): string => {
  const seed0 = manifest.rewiredArms?.seed0;
  if (!seed0) throw new Error(`resolve-graph: ${manifestPath} has no "rewiredArms.seed0" entry`);
  return seed0.artifact;
};
