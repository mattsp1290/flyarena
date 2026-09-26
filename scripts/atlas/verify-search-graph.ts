import { readFile } from 'node:fs/promises';

import { encodeGraphBinary } from '../../src/lib/connectome/format';
import { outputNeuronIndices } from '../../src/lib/connectome/readout';
import { validateSearch } from '../../src/lib/atlas/validation';
import {
  computeArmBundleSha256,
  deserializeArmBundle,
  type ArmName,
  type SerializedArmBundle
} from '../training/export-arms';
import { sha256Hex } from '../training/fsio';
import { evaluateSearchOnGraph, type GraphEvaluationResult } from './publish';

/**
 * What a search JSON's bundle is expected to identify as, for
 * `evaluateSearchForGraph`'s identity checks
 * (`.agents/plans/repertoire-null/01-generalize-atlas-pipeline.md` WP1).
 *
 * `graphArtifactSha256` is the biological parent's gzip sha256 on every arm
 * bundle from one `export-arms.ts` invocation (it never distinguishes one
 * rewiring from another, and verifies no graph content by itself), so
 * `parentGzipSha256` is a weak, label-only, all-arms check -- it guards
 * against mixing export batches, not against a wrong or tampered graph.
 *
 * `binarySha256` is the *specific* seed's re-encoded binary sha -- the check
 * that actually proves which rewiring this is -- and must be the
 * **decompressed** sha `rewire_batch.py`'s `index.json` calls `binarySha256`
 * (never its `gzipSha256`: that's the *compressed* file's sha, which
 * `scripts/null/rewire-index.ts`'s `verifyRewiredFiles` checks against a raw
 * gzip file on disk -- a different check on different bytes than this
 * function's re-encoded-bundle comparison). Required for `biological`/
 * `rewired`. Optional for `disconnected` (no standalone disconnected
 * artifact exists to pin against -- see `metadata.edgeCount === 0` below),
 * but verified when a caller does supply one (e.g. derived from the
 * verified biological parent via
 * `sha256Hex(encodeGraphBinary(createDisconnectedGraph(biologicalGraph)))`)
 * rather than silently ignored.
 */
export interface ExpectedGraphIdentity {
  readonly arm: ArmName;
  readonly binarySha256?: string;
  readonly parentGzipSha256: string;
}

/**
 * Re-evaluate one GPU search JSON's candidates on the graph its own bundle
 * declares, after verifying that bundle really is the graph `expected`
 * names. Unlike `publishAtlas` (bound to the shipped biological artifact),
 * this accepts any verified rewired/disconnected/biological bundle --
 * WP2's per-graph null-search driver calls this once per searched graph.
 *
 * Checks, in order: the bundle's declared `arm` and parent label match
 * `expected`; the search JSON is self-consistent with its own embedded
 * bundle (the same bundle-sha/parent-sha agreement `publishAtlas` checks
 * against the shipped biological asset, applied here against the bundle
 * itself, since there is no external manifest for an arbitrary graph); a
 * search-JSON-level `arm` field (written by `atlas_cli.py` alongside
 * `graphArtifactSha256`), when present, agrees with the bundle's own `arm`;
 * then, after decoding the graph, that its output-neuron indices match the
 * bundle's declared ones; and finally the per-graph identity described on
 * `ExpectedGraphIdentity` (`binarySha256` for biological/rewired,
 * `metadata.edgeCount === 0` for disconnected).
 *
 * `requireDiversity: false` (a narrow repertoire is a valid null-graph
 * result, not an error) and `heldout: 'own'` (only the searched graph's own
 * trained-decoder control -- `disconnected`/`silenced` controls don't add
 * information for a null graph) -- both predeclared
 * (`.agents/plans/repertoire-null/00-overview.md`'s key decisions).
 */
export async function evaluateSearchForGraph(
  searchPath: string,
  expected: ExpectedGraphIdentity
): Promise<GraphEvaluationResult> {
  const input: unknown = JSON.parse(await readFile(searchPath, 'utf8'));
  const source = validateSearch(input);
  const bundle = source.bundle as unknown as SerializedArmBundle;

  if (bundle.arm !== expected.arm)
    throw new Error(`Search graph arm mismatch: expected ${expected.arm}, bundle says ${bundle.arm}`);
  if (bundle.graphArtifactSha256 !== expected.parentGzipSha256)
    throw new Error('Search graph parent identity mismatch');

  // The search JSON's own self-consistency with its embedded bundle --
  // mirrors publishAtlas's checks against the shipped biological asset
  // (scripts/atlas/publish.ts), applied here to the bundle itself, since a
  // generalized entry point has no external manifest to check against.
  if (
    source.bundleSha256 !== computeArmBundleSha256(bundle) ||
    bundle.sha256 !== source.bundleSha256 ||
    bundle.graphArtifactSha256 !== source.graphArtifactSha256
  ) {
    throw new Error('Search artifact is not self-consistent with its bundle');
  }
  // atlas_cli.py records `arm` at the search JSON's top level alongside
  // graphArtifactSha256 (an audit field, redundant with bundle.arm by
  // design); cross-check it when present. Optional because search JSON
  // written before this field existed (the shipped behavior-atlas-v1.json's
  // own source) has none.
  const topLevelArm = (source as unknown as { readonly arm?: unknown }).arm;
  if (topLevelArm !== undefined && topLevelArm !== bundle.arm) {
    throw new Error(
      `Search artifact arm ${String(topLevelArm)} disagrees with its bundle arm ${bundle.arm}`
    );
  }

  const graph = deserializeArmBundle(bundle);

  if (JSON.stringify(Array.from(outputNeuronIndices(graph))) !== JSON.stringify(bundle.outputNeuronIndices))
    throw new Error('Search output neuron mismatch');

  const actualBinarySha256 = sha256Hex(new Uint8Array(encodeGraphBinary(graph)));
  if (bundle.arm === 'disconnected') {
    if (bundle.metadata.edgeCount !== 0)
      throw new Error('Disconnected search graph must have edgeCount 0');
    if (expected.binarySha256 !== undefined && actualBinarySha256 !== expected.binarySha256)
      throw new Error('Search graph binary identity mismatch');
  } else {
    if (!expected.binarySha256) throw new Error(`Search graph identity requires binarySha256 for arm ${bundle.arm}`);
    if (actualBinarySha256 !== expected.binarySha256) throw new Error('Search graph binary identity mismatch');
  }

  return evaluateSearchOnGraph(source, graph, { requireDiversity: false, heldout: 'own' });
}
