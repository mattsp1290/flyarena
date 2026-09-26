import { readFile } from 'node:fs/promises';

import { encodeGraphBinary } from '../../src/lib/connectome/format';
import { validateSearch } from '../../src/lib/atlas/validation';
import {
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
 * rewiring from another), so `parentGzipSha256` is a weak, all-arms check.
 * `binarySha256` is the *specific* seed's re-encoded binary sha from the
 * regenerated `rewire_batch.py` `index.json` -- the check that actually
 * proves which rewiring this is -- and is required for `biological`/
 * `rewired`, never checked for `disconnected` (no per-graph disconnected
 * artifact exists to check against; its identity is `edgeCount === 0`
 * instead).
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
 * Per-graph identity: for `biological`/`rewired`, the bundle's own arrays
 * must re-encode (`encodeGraphBinary`) to exactly `expected.binarySha256`
 * -- the same file-identity discipline
 * `scripts/null/null-evaluate.ts`/`rewire-index.ts`'s `verifyRewiredFiles`
 * applies against `rewire_batch.py`'s `index.json`, just against the
 * bundle's re-encoded bytes rather than a gzip file already on disk (a
 * bundle has no standalone gzip artifact to check against). For
 * `disconnected`, there is no such artifact; the bundle's own
 * `arm === 'disconnected'` and `metadata.edgeCount === 0` are the identity
 * instead.
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

  const graph = deserializeArmBundle(bundle);

  if (bundle.arm === 'disconnected') {
    if (bundle.metadata.edgeCount !== 0)
      throw new Error('Disconnected search graph must have edgeCount 0');
  } else {
    if (!expected.binarySha256) throw new Error(`Search graph identity requires binarySha256 for arm ${bundle.arm}`);
    const actual = sha256Hex(new Uint8Array(encodeGraphBinary(graph)));
    if (actual !== expected.binarySha256) throw new Error('Search graph binary identity mismatch');
  }

  return evaluateSearchOnGraph(source, graph, { requireDiversity: false, heldout: 'own' });
}
