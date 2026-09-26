import { readFile } from 'node:fs/promises';

import { createDisconnectedGraph, encodeGraphBinary, type ConnectomeGraph } from '../../src/lib/connectome/format';
import { outputNeuronIndices } from '../../src/lib/connectome/readout';
import { validateSearch } from '../../src/lib/atlas/validation';
import {
  deserializeArmBundle,
  type ArmName,
  type SerializedArmBundle
} from '../training/export-arms';
import { sha256Hex } from '../training/fsio';
import { evaluateSearchOnGraph, isBundleSelfConsistent, type GraphEvaluationResult } from './publish';

/**
 * What a search JSON's bundle is expected to identify as, for
 * `verifyAndEvaluateSearchGraph`'s identity checks
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
 * `rewired`. Optional for `disconnected` -- see
 * `verifyAndEvaluateSearchGraph`'s doc comment for what covers it when it's
 * omitted.
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
 * bundle (`isBundleSelfConsistent`, shared with `publishAtlas` -- there is
 * no external manifest for an arbitrary graph to check against instead); a
 * search-JSON-level `arm` field (written by `atlas_cli.py` alongside
 * `graphArtifactSha256`), when present, agrees with the bundle's own `arm`;
 * then, after decoding the graph, that its output-neuron indices match the
 * bundle's declared ones; and finally the per-graph identity: for
 * `biological`/`rewired`, the bundle's own arrays must re-encode to exactly
 * `expected.binarySha256`. For `disconnected`, `metadata.edgeCount === 0` is
 * mandatory, and `expected.binarySha256` -- if supplied -- is checked the
 * same way as biological/rewired.
 *
 * **Beyond the plan's minimum:** `01-generalize-atlas-pipeline.md:16` scopes
 * `disconnected`'s identity to `edgeCount === 0` alone, "because no
 * disconnected artifact exists" to pin a `binarySha256` against. Taken
 * literally, a `disconnected` search JSON with a self-consistent hash,
 * `edgeCount === 0`, and the shared/predictable `parentGzipSha256` label
 * would pass even if its neuron identities, dynamics constants
 * (`leakRate`/`rateMin`/`rateMax`/`globalGain`/etc.), or `biologicalIds`
 * were tampered with or came from an entirely different biological parent
 * -- a fail-open gap a dual review demonstrated directly (a probe bundle
 * with negated `presynapticSigns` and scaled dynamics constants passed).
 * "No artifact exists" is true for a *file*, but a disconnected graph's
 * expected binary IS computable, at zero extra I/O, from a graph the caller
 * already has: `createDisconnectedGraph(verifiedBiologicalGraph)` is
 * exactly how `export-arms.ts`'s own `toDisconnectedGraph` derives the real
 * disconnected bundle in the first place. So when `verifiedBiologicalGraph`
 * is supplied and `expected.binarySha256` is not, this function derives the
 * expected disconnected identity from it instead of skipping the check --
 * closing the gap here rather than deferring it to WP2's not-yet-written
 * driver, which the plan's own wording could otherwise be read to invite.
 * Omitting *both* `expected.binarySha256` and `verifiedBiologicalGraph` for
 * a `disconnected` bundle is a hard error, not a silent pass.
 */
export async function verifyAndEvaluateSearchGraph(
  searchPath: string,
  expected: ExpectedGraphIdentity,
  verifiedBiologicalGraph?: Readonly<ConnectomeGraph>
): Promise<GraphEvaluationResult> {
  const input: unknown = JSON.parse(await readFile(searchPath, 'utf8'));
  const source = validateSearch(input);
  const bundle = source.bundle as unknown as SerializedArmBundle;

  if (bundle.arm !== expected.arm)
    throw new Error(`Search graph arm mismatch: expected ${expected.arm}, bundle says ${bundle.arm}`);
  if (bundle.graphArtifactSha256 !== expected.parentGzipSha256)
    throw new Error('Search graph parent identity mismatch');
  if (!isBundleSelfConsistent(source, bundle))
    throw new Error('Search artifact is not self-consistent with its bundle');
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
    let expectedBinarySha256 = expected.binarySha256;
    if (expectedBinarySha256 === undefined) {
      if (!verifiedBiologicalGraph) {
        throw new Error(
          'Disconnected search graph identity requires expected.binarySha256 or verifiedBiologicalGraph'
        );
      }
      expectedBinarySha256 = sha256Hex(
        new Uint8Array(encodeGraphBinary(createDisconnectedGraph(verifiedBiologicalGraph)))
      );
    }
    if (actualBinarySha256 !== expectedBinarySha256) throw new Error('Search graph binary identity mismatch');
  } else {
    if (!expected.binarySha256) throw new Error(`Search graph identity requires binarySha256 for arm ${bundle.arm}`);
    if (actualBinarySha256 !== expected.binarySha256) throw new Error('Search graph binary identity mismatch');
  }

  return evaluateSearchOnGraph(source, graph, { requireDiversity: false, heldout: 'own' });
}
