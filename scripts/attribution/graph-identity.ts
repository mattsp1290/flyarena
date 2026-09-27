import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { gunzipSync } from 'node:zlib';

import { sha256Hex } from '../training/fsio';
import type { GraphListIndex } from '../null/graph-list-index';

/**
 * A thermo-provenance review finding on WP1 (`.agents/plans/readout-attribution/01-archive-and-types.md`):
 * `ArchivedReadout.armBundleSha256` (`archive-readouts.ts`) is NOT a durable,
 * path-independent graph identity -- `export-arms.ts`'s `computeArmBundleSha256`
 * hashes the whole serialized arm bundle, including `provenance.artifactPath`,
 * the literal (often absolute, checkout-specific) CLI argument string
 * `export-arms.ts --rewired`/`--graph` was invoked with. Confirmed empirically:
 * the bigq bundles happen to reproduce (the original export used the
 * repo-relative `public/data/malecns-arena-v1*.bin.gz`), but the intervention
 * bundles do not (`train-sample.sh`/`lookup-intervention-graph.ts` always
 * resolve an ABSOLUTE path before calling `export-arms.ts --rewired`).
 *
 * `graphGzipSha256`/`graphBinarySha256` are the fix: the graph ARTIFACT's own
 * content hashes (gzip bytes, and the decompressed CSR binary bytes),
 * independent of any export-time CLI argument or working directory. WP2's
 * `resolve-graph.ts` should verify a regenerated graph against these, never
 * against `armBundleSha256`.
 *
 * Every value here is independently RECOMPUTED from the actual artifact
 * bytes and checked against the claimed source (the manifest, or the
 * intervention graph-list index) before being trusted -- never copied
 * blind.
 */
export interface GraphIdentity {
  readonly graphGzipSha256: string;
  readonly graphBinarySha256: string;
}

interface MalecnsManifest {
  readonly artifact: string;
  readonly gzipSha256: string;
  readonly binarySha256: string;
  readonly rewiredArms?: Readonly<
    Record<string, { readonly artifact: string; readonly gzipSha256: string; readonly binarySha256: string }>
  >;
}

/** Recompute an artifact file's gzip and decompressed-binary sha256, and refuse to return either unless both match the claimed values -- the "validate, don't just copy" requirement (a thermo review finding). */
const verifyArtifactIdentity = (
  label: string,
  artifactPath: string,
  claimedGzipSha256: string,
  claimedBinarySha256: string
): GraphIdentity => {
  let gzipBytes: Buffer;
  try {
    gzipBytes = readFileSync(artifactPath);
  } catch (error) {
    throw new Error(
      `graph-identity: ${label} names artifact "${artifactPath}", which could not be read (${
        error instanceof Error ? error.message : String(error)
      })`
    );
  }
  const actualGzipSha256 = sha256Hex(gzipBytes);
  if (actualGzipSha256 !== claimedGzipSha256) {
    throw new Error(
      `graph-identity: ${artifactPath}'s gzip sha256 ${actualGzipSha256} does not match ${label}'s claimed ` +
        `${claimedGzipSha256}`
    );
  }
  const actualBinarySha256 = sha256Hex(gunzipSync(gzipBytes));
  if (actualBinarySha256 !== claimedBinarySha256) {
    throw new Error(
      `graph-identity: ${artifactPath}'s decompressed sha256 ${actualBinarySha256} does not match ${label}'s ` +
        `claimed ${claimedBinarySha256}`
    );
  }
  return { graphGzipSha256: actualGzipSha256, graphBinarySha256: actualBinarySha256 };
};

/**
 * Resolve a bigq `graphId`'s path-independent identity from
 * `public/data/malecns-arena-v1.manifest.json`, validated against the actual
 * artifact bytes committed alongside it (`public/data/malecns-arena-v1.bin.gz`/
 * `malecns-arena-v1-rewired-seed0.bin.gz`), not merely copied from the
 * manifest's own claims.
 *
 * `"disconnected"` has no separate artifact at all -- `export-arms.ts`'s
 * `toDisconnectedGraph` derives it at runtime from the biological graph
 * (same node set, `edgeCount: 0`), matching `ArmProvenance`'s own
 * `disconnected-runtime-zero-edge` kind (no `artifactPath`/`artifactSha256`
 * either) -- so this returns `null` for it rather than fabricating a value.
 */
export const resolveBigqGraphIdentity = (graphId: string, manifestPath: string): GraphIdentity | null => {
  if (graphId === 'disconnected') return null;

  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8')) as MalecnsManifest;
  const manifestDir = dirname(manifestPath);

  if (graphId === 'biological') {
    return verifyArtifactIdentity(
      `${manifestPath}'s top-level artifact`,
      resolve(manifestDir, manifest.artifact),
      manifest.gzipSha256,
      manifest.binarySha256
    );
  }
  if (graphId === 'rewired-seed0') {
    const seed0 = manifest.rewiredArms?.seed0;
    if (!seed0) throw new Error(`graph-identity: ${manifestPath} has no "rewiredArms.seed0" entry`);
    return verifyArtifactIdentity(
      `${manifestPath}'s rewiredArms.seed0`,
      resolve(manifestDir, seed0.artifact),
      seed0.gzipSha256,
      seed0.binarySha256
    );
  }
  throw new Error(`graph-identity: "${graphId}" is not a known bigq graphId (biological/rewired-seed0/disconnected)`);
};

/**
 * Resolve an intervention `graphId`'s path-independent identity from an
 * already-parsed, already-structurally-validated `GraphListIndex`
 * (`scripts/null/graph-list-index.ts`'s `readGraphListIndex` -- entry shape,
 * reserved ids, duplicate ids, and path traversal are all checked by that
 * function before this one ever runs), re-verified here against the actual
 * graph artifact bytes on disk (`indexDir` + the entry's own `path`), not
 * merely copied from the index's own claimed `gzipSha256`/`binarySha256`.
 */
export const resolveInterventionGraphIdentity = (
  graphId: string,
  index: Readonly<GraphListIndex>,
  indexDir: string
): GraphIdentity => {
  const entry = index.entries.find((candidate) => candidate.id === graphId);
  if (!entry) {
    throw new Error(`graph-identity: the intervention graph-list index has no entry for id "${graphId}"`);
  }
  return verifyArtifactIdentity(
    `the intervention graph-list index's "${graphId}" entry`,
    resolve(indexDir, entry.path),
    entry.gzipSha256,
    entry.binarySha256
  );
};
