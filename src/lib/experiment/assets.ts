/**
 * Fetch, gunzip, and hash-verify the two checked-in MaleCNS-derived graph
 * artifacts (WP6 item 2) before the experiment is allowed to start. Nothing
 * here is Svelte-specific; `src/lib/experiment/controller.ts` is the only
 * caller.
 *
 * Every function below is a plain async function over `ArrayBuffer`s (no
 * dependency on a real network) so `tests/unit/experiment-assets.test.ts`
 * can exercise the real integrity-check logic against the real committed
 * `public/data/` files (read via `node:fs`) without a server.
 */

import { parseGraphBinary, type ConnectomeGraph, type GraphMode } from '../connectome/format';
import { decodeReadoutArtifact, type ReadoutWeights, type TrainedReadoutArtifactJson } from '../connectome/readout';
import { sha256 } from '@noble/hashes/sha2.js';
import { bytesToHex } from '@noble/hashes/utils.js';

export class ArtifactIntegrityError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ArtifactIntegrityError';
  }
}

export class DecompressionUnsupportedError extends Error {
  constructor() {
    // DecompressionStream ships in every evergreen browser this POC targets
    // (Chrome/Edge 80+, Firefox 113+, Safari 16.4+ — all Baseline-widely-
    // available well before this bean). Implementing a hand-rolled gzip
    // inflate as a fallback would be substantial extra surface for a
    // POC that already declares "no WASM" as a non-goal, purely to support
    // browsers this deployment does not target. The graceful "fallback" is
    // this typed error: it drives the experiment straight to the `error`
    // state with an actionable message instead of an unhandled
    // ReferenceError, per docs/architecture.md's error-state contract.
    super('This browser does not support DecompressionStream; use a current Chrome, Firefox, or Safari.');
    this.name = 'DecompressionUnsupportedError';
  }
}

// Static HTTP deployments do not expose crypto.subtle. Use the same SHA-256
// implementation in browser Workers and Node without weakening verification.
export const sha256Hex = async (data: ArrayBuffer): Promise<string> =>
  bytesToHex(sha256(new Uint8Array(data)));

/** Decompress one gzip member via the streaming Web Compression API. Throws `DecompressionUnsupportedError` if unavailable. */
export const decompressGzip = async (data: ArrayBuffer): Promise<ArrayBuffer> => {
  if (typeof DecompressionStream === 'undefined') throw new DecompressionUnsupportedError();
  const stream = new Response(data).body?.pipeThrough(new DecompressionStream('gzip'));
  if (!stream) throw new Error('Failed to open a decompression stream for the fetched artifact');
  return new Response(stream).arrayBuffer();
};

export interface ArtifactManifestEntry {
  gzipSha256: string;
  gzipBytes: number;
  binarySha256: string;
  binaryBytes: number;
}

/** The two leading bytes of every gzip member (RFC 1952 ID1/ID2). */
const GZIP_MAGIC_0 = 0x1f;
const GZIP_MAGIC_1 = 0x8b;

/**
 * True when `bytes` starts with the gzip magic number, i.e. this is a raw,
 * still-compressed gzip stream rather than something a transport layer
 * already decoded.
 */
const looksLikeGzip = (bytes: ArrayBuffer): boolean => {
  if (bytes.byteLength < 2) return false;
  const view = new Uint8Array(bytes, 0, 2);
  return view[0] === GZIP_MAGIC_0 && view[1] === GZIP_MAGIC_1;
};

/**
 * Verify a fetched artifact against its manifest entry and return the
 * decompressed graph bytes `parseGraphBinary` will consume.
 *
 * `fetchedBytes` is *not* guaranteed to still be gzip-compressed: static
 * file servers (Vite's own `preview`/`dev` among them — confirmed directly
 * against a real build, see the bean's verification notes) commonly serve a
 * `.gz` file with a `Content-Encoding: gzip` response header, which the
 * Fetch API's `response.arrayBuffer()` transparently decompresses before
 * this code ever sees the bytes. There is no portable way to ask `fetch`
 * whether that happened (`Content-Encoding` is stripped from the exposed
 * `Response` once decoded), so this detects it structurally: real gzip data
 * always starts with the fixed 2-byte magic number (RFC 1952); the
 * decompressed graph binary never does (`format.ts`'s "FANG" magic differs).
 * Whichever case applies, the corresponding hash/length from the manifest
 * is verified before anything is trusted.
 */
export const verifyAndDecompressArtifact = async (
  fetchedBytes: ArrayBuffer,
  entry: Readonly<ArtifactManifestEntry>
): Promise<ArrayBuffer> => {
  if (!looksLikeGzip(fetchedBytes)) {
    // Already decompressed by the transport layer: verify directly against
    // the decompressed-bytes manifest entry.
    if (fetchedBytes.byteLength !== entry.binaryBytes) {
      throw new ArtifactIntegrityError(
        `decompressed byte length ${fetchedBytes.byteLength} does not match manifest (${entry.binaryBytes})`
      );
    }
    const binaryHash = await sha256Hex(fetchedBytes);
    if (binaryHash !== entry.binarySha256) {
      throw new ArtifactIntegrityError(
        `decompressed sha256 ${binaryHash} does not match manifest (${entry.binarySha256})`
      );
    }
    return fetchedBytes;
  }

  // Still gzip-compressed: verify the compressed bytes first (fails fast on
  // transport corruption without spending time inflating garbage), then
  // decompress and verify again against the decompressed-bytes entry.
  if (fetchedBytes.byteLength !== entry.gzipBytes) {
    throw new ArtifactIntegrityError(
      `gzip byte length ${fetchedBytes.byteLength} does not match manifest (${entry.gzipBytes})`
    );
  }
  const gzipHash = await sha256Hex(fetchedBytes);
  if (gzipHash !== entry.gzipSha256) {
    throw new ArtifactIntegrityError(`gzip sha256 ${gzipHash} does not match manifest (${entry.gzipSha256})`);
  }

  const binary = await decompressGzip(fetchedBytes);
  if (binary.byteLength !== entry.binaryBytes) {
    throw new ArtifactIntegrityError(
      `decompressed byte length ${binary.byteLength} does not match manifest (${entry.binaryBytes})`
    );
  }
  const binaryHash = await sha256Hex(binary);
  if (binaryHash !== entry.binarySha256) {
    throw new ArtifactIntegrityError(`decompressed sha256 ${binaryHash} does not match manifest (${entry.binarySha256})`);
  }
  return binary;
};

/**
 * The manifest's `positions` entry (WP1), describing the soma-position
 * sidecar artifact (`malecns-arena-v1.positions.json`) built from the pinned
 * MaleCNS annotations — never from the compiler, so the graph artifacts'
 * bytes/hashes stay untouched. Optional on `ArenaManifest`: a manifest
 * produced before WP1 (or a hand-built test fixture) simply has no
 * anatomical activity view to offer — `loadPositions` below reports that as
 * `status: 'missing'` rather than throwing.
 */
export interface PositionsManifestEntry {
  artifact: string;
  sha256: string;
  coverage: { soma: number; tosoma: number; none: number };
}

export interface ArenaManifest {
  artifact: string;
  binaryBytes: number;
  binarySha256: string;
  edgeCount: number;
  formatVersion: number;
  gzipBytes: number;
  gzipSha256: string;
  inputChannelCount: number;
  license: string;
  neuronCount: number;
  outputPopulationCount: number;
  /** WP1's soma-position sidecar entry; see `PositionsManifestEntry`. */
  positions?: PositionsManifestEntry;
  sourceDataset: string;
  rewiredArms: Record<
    string,
    {
      artifact: string;
      binaryBytes: number;
      binarySha256: string;
      gzipBytes: number;
      gzipSha256: string;
      /**
       * The rewiring compiler's own accounting for this arm (see
       * `scripts/data/rewire.py`/`compile.py`). Degree-preserving rewiring
       * never changes the node set, so a rewired arm has no `neuronCount` of
       * its own to cross-check — it always shares the top-level manifest's
       * `neuronCount`. `swapStats.edgeCount` is the one rewired-arm-specific
       * count `loadArenaArtifacts` below cross-checks the parsed rewired
       * artifact against (bb45 follow-up: the biological cross-check below
       * previously had no rewired-arm counterpart).
       *
       * Required, not optional: `rewire.py`'s compiler always writes this
       * field in the same pass that writes this arm's hashes (see
       * `docs/data-provenance.md`), so no real producer ever omits it. A
       * review pass caught that an earlier, optional-typed version of this
       * field let `loadArenaArtifacts` below silently skip the whole
       * rewired-arm cross-check for a manifest that happened to be missing
       * it — exactly the "stale/hand-edited manifest" case that check exists
       * to catch. `loadArenaArtifacts` fails closed (throws
       * `ArtifactIntegrityError`) if it is ever absent or malformed.
       */
      swapStats: { edgeCount: number };
    }
  >;
}

const fetchJson = async <T>(url: string): Promise<T> => {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
  return (await response.json()) as T;
};

/** Exported for `loadPositions` below (fetching its own positions sidecar artifact) and for tests. */
export const fetchArrayBuffer = async (url: string): Promise<ArrayBuffer> => {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
  return response.arrayBuffer();
};

export interface LoadedArenaArtifacts {
  manifest: ArenaManifest;
  /** Hash-verified, decompressed binary graph buffer for the biological arm. */
  biological: ArrayBuffer;
  /** Hash-verified, decompressed binary graph buffer for the rewired-seed0 control arm. */
  rewired: ArrayBuffer;
  /**
   * The already-parsed biological graph (thermo-architecture I1 fix):
   * `loadArenaArtifacts` parses this exact graph below purely to cross-check
   * `neuronCount`/`edgeCount` against the manifest — returning it here lets
   * `loadPositions` reuse it (`biologicalIds`, `metadata.rateMin`/`rateMax`)
   * instead of re-fetching, re-verifying, and re-parsing the same artifact a
   * second time. Callers must not mutate any of its typed-array views.
   */
  parsedBiological: ConnectomeGraph;
}

/**
 * Fetch the manifest and both graph artifacts declared by it, hash-verify
 * every one, and return their decompressed buffers. Throws
 * `ArtifactIntegrityError`/`DecompressionUnsupportedError` (or a plain
 * `Error` for a fetch failure) on any problem; the caller (`App.svelte`)
 * maps any throw here to the `error` experiment state and must not enable
 * Start.
 */
export const loadArenaArtifacts = async (
  dataBaseUrl = '/data',
  manifestFilename = 'malecns-arena-v1.manifest.json'
): Promise<LoadedArenaArtifacts> => {
  const manifest = await fetchJson<ArenaManifest>(`${dataBaseUrl}/${manifestFilename}`);
  const rewiredEntry = manifest.rewiredArms.seed0;
  if (!rewiredEntry) throw new Error('Manifest is missing the seed0 rewired control arm');

  const [biologicalGzip, rewiredGzip] = await Promise.all([
    fetchArrayBuffer(`${dataBaseUrl}/${manifest.artifact}`),
    fetchArrayBuffer(`${dataBaseUrl}/${rewiredEntry.artifact}`)
  ]);

  const [biological, rewired] = await Promise.all([
    verifyAndDecompressArtifact(biologicalGzip, manifest),
    verifyAndDecompressArtifact(rewiredGzip, rewiredEntry)
  ]);

  // The hash/length checks above only prove `biological`'s bytes match what
  // the manifest declares byte-for-byte — they say nothing about whether
  // the manifest's own *descriptive* `neuronCount`/`edgeCount` fields (read
  // by the ledger panel, never re-derived from the parsed graph) are still
  // accurate. A manifest that is internally consistent (hashes/lengths
  // correct) but was hand-edited or left stale after a forgotten compiler
  // re-run would pass every check above and only surface as a silently
  // wrong number in the UI. Parse a throwaway copy (never the buffer
  // returned to the caller — `parseGraphBinary` only reads it, but a
  // defensive copy keeps this check from ever being able to alias/mutate
  // what the caller receives) and cross-check its actual counts.
  const parsedBiological = parseGraphBinary(biological.slice(0));
  if (
    parsedBiological.metadata.neuronCount !== manifest.neuronCount ||
    parsedBiological.metadata.edgeCount !== manifest.edgeCount
  ) {
    throw new ArtifactIntegrityError(
      `manifest neuronCount/edgeCount (${manifest.neuronCount}/${manifest.edgeCount}) does not match the ` +
        `parsed biological artifact (${parsedBiological.metadata.neuronCount}/${parsedBiological.metadata.edgeCount})`
    );
  }

  // bb45 follow-up: the cross-check above only ever covered the biological
  // artifact. This is defense-in-depth for manifest/artifact consistency —
  // not, as an earlier version of this comment claimed, something that
  // "surfaces in the UI": `swapStats.edgeCount` itself is never displayed
  // anywhere (the ledger panel only shows the biological manifest counts;
  // telemetry's `edgeCount` comes from the Worker's own `init` response over
  // the parsed graph, independent of this field). What this check actually
  // protects is the manifest/ledger JSON's own internal consistency, which
  // Python-side tooling and any future consumer of `public/data/*.json` can
  // rely on without re-parsing the binary artifact themselves. Degree-
  // preserving rewiring never changes the node set, so the rewired arm has
  // no `neuronCount` of its own; it shares the top-level manifest's
  // `neuronCount`. Its edge count lives at `rewiredEntry.swapStats.edgeCount`
  // (see `scripts/data/rewire.py`'s ledger output) rather than a top-level
  // field.
  const parsedRewired = parseGraphBinary(rewired.slice(0));
  const rewiredEdgeCount = rewiredEntry.swapStats?.edgeCount;
  // Fails closed: a manifest missing `swapStats.edgeCount` entirely (or
  // with a non-numeric value) is exactly as suspect as one with a wrong
  // value — `rewire.py` always writes this field, so its absence means the
  // manifest itself is malformed or stale, not that there is nothing to
  // check. An earlier, optional-typed version of this field let that case
  // silently skip the check instead (a review-caught gap).
  if (typeof rewiredEdgeCount !== 'number') {
    throw new ArtifactIntegrityError(
      "Manifest's seed0 rewired arm is missing swapStats.edgeCount (required for the rewired-artifact cross-check)"
    );
  }
  if (parsedRewired.metadata.neuronCount !== manifest.neuronCount || parsedRewired.metadata.edgeCount !== rewiredEdgeCount) {
    throw new ArtifactIntegrityError(
      `manifest neuronCount/rewired swapStats.edgeCount (${manifest.neuronCount}/${rewiredEdgeCount}) does not ` +
        `match the parsed rewired artifact (${parsedRewired.metadata.neuronCount}/${parsedRewired.metadata.edgeCount})`
    );
  }

  return { manifest, biological, rewired, parsedBiological };
};

/**
 * Per-neuron soma-position sidecar (WP1's `scripts/data/positions.py`
 * output, `malecns-arena-v1.positions.json`). `xyz[i]` is `null` exactly
 * when `positionSource[i] === 'none'` (no annotated coordinate for that
 * neuron — never fabricated); `bodyIds` is the MaleCNS body ID for neuron
 * `i` as a decimal string, in the same order as the compiled graph's
 * `biologicalIds` (verified by `loadPositions` below, not merely assumed).
 */
export interface PositionsArtifact {
  version: number;
  sourceFile: string;
  sourceSha256: string;
  /** sha256 of the compiled biological graph's *gzip* bytes this artifact was built against — matches `ArenaManifest.gzipSha256` when the two are in sync (see `loadPositions`'s cross-check). */
  graphSha256: string;
  /** Always the literal string "dataset voxel units (unverified)" as of WP1 — the view must never claim a verified physical unit. */
  units: string;
  bodyIds: readonly string[];
  positionSource: readonly ('soma' | 'tosoma' | 'none')[];
  role: readonly ('sensory' | 'bridge' | 'descending')[];
  xyz: ReadonlyArray<readonly [number, number, number] | null>;
  coverage: { soma: number; tosoma: number; none: number };
  roleCounts: { sensory: number; bridge: number; descending: number };
}

export type PositionsLoadResult =
  | { status: 'ok'; positions: PositionsArtifact; rateMin: number; rateMax: number }
  | { status: 'missing'; reason: string }
  | { status: 'invalid'; reason: string };

const isFiniteTriple = (value: unknown): value is readonly [number, number, number] =>
  Array.isArray(value) &&
  value.length === 3 &&
  value.every((component) => typeof component === 'number' && Number.isFinite(component));

/**
 * Fetch, hash-verify, and structurally validate the soma-position sidecar
 * artifact the manifest declares (`manifest.positions`), then cross-check it
 * against the already-parsed biological graph. Never throws: every failure
 * mode is a returned `status`, so a caller (`App.svelte`) can disable the
 * activity view's toggle with an honest reason instead of failing the whole
 * experiment (positions are optional presentation, unlike the graph
 * artifacts `loadArenaArtifacts` gates Start on).
 *
 * `dataBaseUrl` must be the same value the caller passes to
 * `loadArenaArtifacts` (`${import.meta.env.BASE_URL}data` in production) so
 * the positions artifact itself resolves under the app's real deployment
 * base path (e.g. `/fly/`), not a hardcoded `/data`.
 *
 * `biologicalGraph` must be the parsed graph `loadArenaArtifacts` already
 * fetched, hash-verified, and parsed for this same manifest (its
 * `LoadedArenaArtifacts.parsedBiological`, threaded through
 * `ExperimentController.initialize()`'s `onManifest` callback to the caller —
 * see `controller.ts`). Reusing it here (thermo-architecture I1 fix) means
 * this function no longer re-fetches, re-verifies, or re-parses the graph
 * artifact itself: it only fetches its own positions sidecar file.
 *
 * Three layers of integrity, all required for `status: 'ok'`:
 * 1. `manifest.positions.sha256` against the fetched bytes' own sha256 —
 *    proves this is exactly the file the manifest names, byte for byte.
 * 2. `positions.graphSha256` against `manifest.gzipSha256` — proves this
 *    positions file was built against *this* compiled graph artifact, not a
 *    stale one left over from an earlier compiler run.
 * 3. `positions.bodyIds` (as decimal strings) against `biologicalGraph`'s own
 *    `biologicalIds`, element-by-element — the plan's own required check
 *    that this artifact's per-index neuron identity actually lines up with
 *    the graph's. `biologicalGraph.metadata.rateMin`/`rateMax` (the declared
 *    dynamics bounds the activity view's colormap is scaled against — never a
 *    per-frame auto-normalized range, see the plan's "Color" decision) are
 *    returned alongside `positions` on success.
 */
export const loadPositions = async (
  manifest: ArenaManifest,
  dataBaseUrl: string,
  biologicalGraph: ConnectomeGraph
): Promise<PositionsLoadResult> => {
  const entry = manifest.positions;
  if (!entry) {
    return { status: 'missing', reason: 'The manifest has no positions artifact entry.' };
  }

  let rawBytes: ArrayBuffer;
  try {
    rawBytes = await fetchArrayBuffer(`${dataBaseUrl}/${entry.artifact}`);
  } catch (error) {
    return { status: 'missing', reason: error instanceof Error ? error.message : String(error) };
  }

  const rawHash = await sha256Hex(rawBytes);
  if (rawHash !== entry.sha256) {
    return {
      status: 'invalid',
      reason: `positions artifact sha256 ${rawHash} does not match the manifest (${entry.sha256})`
    };
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(new TextDecoder().decode(rawBytes));
  } catch (error) {
    return {
      status: 'invalid',
      reason: `positions artifact is not valid JSON: ${error instanceof Error ? error.message : String(error)}`
    };
  }
  // A hash-matched file can still be valid JSON that isn't an object at all
  // (e.g. `null`, a bare number, or an array) — dereferencing `.bodyIds`
  // below on a non-object throws instead of returning a status, breaking
  // this function's own "never throws" contract (dual review finding).
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    return { status: 'invalid', reason: 'positions artifact is not a JSON object' };
  }

  const positions = parsed as Partial<PositionsArtifact>;
  if (
    !Array.isArray(positions.bodyIds) ||
    !Array.isArray(positions.xyz) ||
    !Array.isArray(positions.positionSource) ||
    !Array.isArray(positions.role) ||
    typeof positions.graphSha256 !== 'string'
  ) {
    return { status: 'invalid', reason: 'positions artifact is missing one or more required fields' };
  }

  const { neuronCount } = manifest;
  if (
    positions.bodyIds.length !== neuronCount ||
    positions.xyz.length !== neuronCount ||
    positions.positionSource.length !== neuronCount ||
    positions.role.length !== neuronCount
  ) {
    return {
      status: 'invalid',
      reason: `positions arrays do not all have length ${neuronCount} (the manifest's neuronCount)`
    };
  }

  // Counted alongside the per-index validation below (free — the loop
  // already visits every index) so the coverage/roleCounts numbers the UI
  // displays (`ActivityPanel.svelte`'s "Positioned: N soma, ..." line) can
  // be checked against what the arrays actually contain, not merely taken
  // on the producer's word (dual review finding: an unvalidated `coverage`
  // could lie relative to the real data, and a missing one would throw
  // inside the panel's template instead of failing this loader honestly).
  const countedCoverage = { soma: 0, tosoma: 0, none: 0 };
  const countedRoles = { sensory: 0, bridge: 0, descending: 0 };
  for (let index = 0; index < neuronCount; index += 1) {
    const source: 'soma' | 'tosoma' | 'none' = positions.positionSource[index];
    const point = positions.xyz[index];
    if (source !== 'soma' && source !== 'tosoma' && source !== 'none') {
      return { status: 'invalid', reason: `positionSource[${index}] "${String(source)}" is not soma/tosoma/none` };
    }
    // Never invented: a "none" entry must carry no coordinate, and every
    // other entry must carry a real, finite one — this is the one runtime
    // check standing directly between a build-time bug and the view
    // silently plotting a fabricated position.
    if (source === 'none') {
      if (point !== null) {
        return { status: 'invalid', reason: `xyz[${index}] is non-null but positionSource is "none"` };
      }
    } else if (!isFiniteTriple(point)) {
      return { status: 'invalid', reason: `xyz[${index}] is missing/malformed for positionSource "${source}"` };
    }
    countedCoverage[source] += 1;
    const role: 'sensory' | 'bridge' | 'descending' = positions.role[index];
    if (role !== 'sensory' && role !== 'bridge' && role !== 'descending') {
      return { status: 'invalid', reason: `role[${index}] "${String(role)}" is not sensory/bridge/descending` };
    }
    countedRoles[role] += 1;
  }

  const coverageMatches = (
    value: unknown
  ): value is { soma: number; tosoma: number; none: number } =>
    typeof value === 'object' &&
    value !== null &&
    (value as Record<string, unknown>).soma === countedCoverage.soma &&
    (value as Record<string, unknown>).tosoma === countedCoverage.tosoma &&
    (value as Record<string, unknown>).none === countedCoverage.none;
  if (!coverageMatches(positions.coverage)) {
    return {
      status: 'invalid',
      reason: `positions.coverage does not match the actual positionSource counts (${JSON.stringify(countedCoverage)})`
    };
  }
  // The manifest carries its own copy of the same counts (WP1's ledger
  // output) — cross-check it too, so a generator bug that updates one copy
  // and not the other is caught here rather than only showing up as two
  // disagreeing numbers somewhere in the product.
  if (entry.coverage && !coverageMatches(entry.coverage)) {
    return {
      status: 'invalid',
      reason: `manifest positions.coverage does not match the actual positionSource counts (${JSON.stringify(countedCoverage)})`
    };
  }
  const roleCountsMatch =
    typeof positions.roleCounts === 'object' &&
    positions.roleCounts !== null &&
    (positions.roleCounts as Record<string, unknown>).sensory === countedRoles.sensory &&
    (positions.roleCounts as Record<string, unknown>).bridge === countedRoles.bridge &&
    (positions.roleCounts as Record<string, unknown>).descending === countedRoles.descending;
  if (!roleCountsMatch) {
    return {
      status: 'invalid',
      reason: `positions.roleCounts does not match the actual role counts (${JSON.stringify(countedRoles)})`
    };
  }
  // The "unverified units" disclosure is a non-negotiable honesty label
  // (see this file's own doc comment on `PositionsArtifact.units`) — a
  // missing/malformed `units` string must fail closed, not silently render
  // "(units: undefined)".
  if (typeof positions.units !== 'string' || !/unverified/i.test(positions.units)) {
    return { status: 'invalid', reason: 'positions.units is missing or does not disclose that units are unverified' };
  }

  if (positions.graphSha256 !== manifest.gzipSha256) {
    return {
      status: 'invalid',
      reason: 'positions.graphSha256 does not match the manifest’s compiled graph gzip sha256 (stale positions artifact)'
    };
  }

  if (biologicalGraph.biologicalIds.length !== neuronCount) {
    return {
      status: 'invalid',
      reason: `graph biologicalIds length ${biologicalGraph.biologicalIds.length} does not match neuronCount ${neuronCount}`
    };
  }
  for (let index = 0; index < neuronCount; index += 1) {
    if (positions.bodyIds[index] !== biologicalGraph.biologicalIds[index].toString()) {
      return {
        status: 'invalid',
        reason: `bodyIds[${index}] "${positions.bodyIds[index]}" does not match the graph's biologicalIds[${index}] "${biologicalGraph.biologicalIds[index].toString()}"`
      };
    }
  }

  return {
    status: 'ok',
    positions: positions as PositionsArtifact,
    rateMin: biologicalGraph.metadata.rateMin,
    rateMax: biologicalGraph.metadata.rateMax
  };
};

/**
 * The subset of `public/data/trained-readout-v1.manifest.json` the app
 * actually reads: enough to verify the artifact JSON's own bytes and to
 * label its provenance in the ledger panel (WP6). The manifest also carries
 * training/evaluator/env provenance the ledger does not surface — those
 * fields simply pass through untyped here rather than being restated field
 * by field.
 */
export interface TrainedReadoutManifest {
  version: number;
  /** sha256 of `trained-readout-v1.json`'s exact bytes — checked against the fetched artifact below before it is trusted. */
  artifactSha256: string;
  /** Output-neuron count (readout input size), identical across arms by construction (`export-arms.ts`'s node-set gate). */
  D: number;
  /** Hidden layer width, identical across arms (one shared training config). */
  H: number;
  /** Total trainable scalars per arm (`readoutParameterCount(D, H)`), identical across arms. */
  parameterCount: number;
}

/**
 * WP6's trained-readout counterpart to `PositionsLoadResult`: never throws,
 * and every failure is a returned `status` so `ExperimentController` can
 * keep the Authored path working and disable just the Trained decoder with
 * an honest reason (the "artifact failed verification" ledger message) —
 * never fail the whole experiment over a missing/corrupt trained-readout
 * artifact, which is optional relative to the required arena graph
 * artifacts `loadArenaArtifacts` gates Start on.
 */
export type TrainedReadoutLoadResult =
  | { status: 'ok'; manifest: TrainedReadoutManifest; weightsByMode: Readonly<Record<GraphMode, ReadoutWeights>> }
  | { status: 'unavailable'; reason: string };

/** Every `GraphMode` the trained-readout artifact carries a per-arm entry for (`TrainedReadoutArtifactJson.arms`'s required keys). */
const READOUT_ARMS: readonly GraphMode[] = ['biological', 'rewired', 'disconnected'];

/**
 * Fetch, sha256-verify, and base64-decode `trained-readout-v1.{json,manifest.json}`
 * (WP6 item 2's trained-readout counterpart to `loadArenaArtifacts`). Unlike
 * that function, this one never throws — see `TrainedReadoutLoadResult`'s doc
 * comment — and does not itself call `validateReadoutWeights` against a
 * graph: the caller (`ExperimentController`, which has the per-arm parsed
 * graphs) does that once it has something to validate each arm's weights
 * against.
 */
export const loadTrainedReadoutArtifact = async (dataBaseUrl = '/data'): Promise<TrainedReadoutLoadResult> => {
  try {
    const manifest = await fetchJson<TrainedReadoutManifest>(`${dataBaseUrl}/trained-readout-v1.manifest.json`);
    const artifactBytes = await fetchArrayBuffer(`${dataBaseUrl}/trained-readout-v1.json`);

    const artifactHash = await sha256Hex(artifactBytes);
    if (artifactHash !== manifest.artifactSha256) {
      return {
        status: 'unavailable',
        reason: `trained-readout-v1.json sha256 ${artifactHash} does not match its manifest (${manifest.artifactSha256})`
      };
    }

    let json: TrainedReadoutArtifactJson;
    try {
      json = JSON.parse(new TextDecoder().decode(artifactBytes)) as TrainedReadoutArtifactJson;
    } catch (error) {
      return {
        status: 'unavailable',
        reason: `trained-readout-v1.json is not valid JSON: ${error instanceof Error ? error.message : String(error)}`
      };
    }

    const weightsByMode: Partial<Record<GraphMode, ReadoutWeights>> = {};
    for (const arm of READOUT_ARMS) {
      weightsByMode[arm] = decodeReadoutArtifact(json, arm);
    }

    return {
      status: 'ok',
      manifest,
      weightsByMode: weightsByMode as Record<GraphMode, ReadoutWeights>
    };
  } catch (error) {
    // Fetch failure (e.g. a 404 because WP5's production artifact has not
    // shipped yet), a malformed manifest, or a `decodeReadoutArtifact`
    // failure (a missing arm key or a non-multiple-of-4 base64 payload) all
    // land here as the same honest "unavailable" outcome — the Trained
    // decoder is simply not offered, and Authored keeps working.
    return { status: 'unavailable', reason: error instanceof Error ? error.message : String(error) };
  }
};
