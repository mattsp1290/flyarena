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
import {
  decodeReadoutArtifact,
  readoutParameterCount,
  type ReadoutWeights,
  type TrainedReadoutArtifactJson
} from '../connectome/readout';
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
  /**
   * WP4's rewiring-null-distribution ledger entry: `null-report.ts`
   * (`.agents/plans/rewiring-null/02-authored-null-evaluation.md`) writes
   * this once `rewiring-null-v1.json` exists. Optional: a manifest produced
   * before that WP (or a hand-built test fixture) simply has no null
   * distribution to show — `loadRewiringNull` below reports that as
   * `status: 'missing'` rather than throwing.
   */
  rewiringNull?: { artifact: string; sha256: string };
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

/**
 * Every `GraphMode` the trained-readout artifact carries a per-arm entry
 * for (`TrainedReadoutArtifactJson.arms`'s required keys). Built from an
 * exhaustiveness-checked map, not a bare array literal: `READOUT_ARMS_MAP`'s
 * type (`Record<GraphMode, true>`) has no structural link a plain
 * `readonly GraphMode[]` array would — a future `GraphMode` variant added
 * to `format.ts` without a matching entry here fails to compile on this
 * object literal, instead of silently type-checking while
 * `readoutWeightsForMode()` (`controller.ts`) would return `undefined` for
 * the new arm at runtime with no error anywhere (round-1 dual review,
 * carried over as an open suggestion into the thermo-maintainability
 * review's S2).
 */
const READOUT_ARMS_MAP: Record<GraphMode, true> = { biological: true, rewired: true, disconnected: true };
const READOUT_ARMS: readonly GraphMode[] = Object.keys(READOUT_ARMS_MAP) as GraphMode[];

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

    // Dual review (round 1, Important): the sha256 check above only proves
    // the artifact's own bytes are untampered — it says nothing about
    // whether the manifest's separately-authored `D`/`H`/`parameterCount`
    // fields (what the ledger panel displays verbatim, `LedgerPanel.svelte`)
    // still describe *this* artifact. A manifest that is internally
    // consistent (its own hash check passes) but was hand-edited or left
    // stale after a forgotten regeneration would otherwise pass every check
    // above and only surface as a silently wrong architecture/parameter-count
    // claim in the UI — the same risk class `loadArenaArtifacts` above
    // already guards against for the arena manifest's `neuronCount`/
    // `edgeCount`. The MLP itself is unaffected either way (it always uses
    // `json.inputSize`/`json.hiddenSize` directly, separately checked by
    // `validateReadoutWeights` against the loaded graph); this is a
    // display-honesty gate, not a numerical-correctness one.
    const expectedParameterCount = readoutParameterCount(json.inputSize, json.hiddenSize);
    // Named per-field, not just a blanket mismatch (thermo-maintainability
    // review S3): the manifest and the artifact can diverge on any subset
    // of D/H/parameterCount independently (e.g. only `parameterCount` stale
    // after a hand-edit, with `D`/`H` both still correct) — naming exactly
    // which field(s) diverged makes the ledger's failure reason faster to
    // diagnose from a bug report than always restating every field.
    const divergingFields: string[] = [];
    if (manifest.D !== json.inputSize) divergingFields.push('D');
    if (manifest.H !== json.hiddenSize) divergingFields.push('H');
    if (manifest.parameterCount !== expectedParameterCount) divergingFields.push('parameterCount');
    if (divergingFields.length > 0) {
      return {
        status: 'unavailable',
        reason:
          `trained-readout-v1.manifest.json diverges from the artifact on ${divergingFields.join(', ')}: ` +
          `manifest (D=${manifest.D}, H=${manifest.H}, parameterCount=${manifest.parameterCount}) ` +
          `does not match the artifact's own inputSize=${json.inputSize}/hiddenSize=${json.hiddenSize} ` +
          `(expected parameterCount ${expectedParameterCount})`
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

/**
 * `public/data/rewiring-null-v1.json` (WP4,
 * `.agents/plans/rewiring-null/02-authored-null-evaluation.md`'s "Artifact
 * shape (conceptual)"): every graph's per-seed-averaged authored-decoder
 * score, plus the 500-graph null set's summary statistics and a pre-binned
 * histogram (`src/lib/ui/NullHistogram.svelte` draws bars straight from
 * `bins`, never rebinning `rewired` itself). `trained` is WP3's follow-on
 * addition (the CEM-retrained-readout sample on 20 rewirings) and may not
 * exist yet — deliberately typed `unknown` rather than a guessed shape:
 * `null-report.ts`'s `trained` section had not landed as of this WP, so any
 * hand-authored field list here would be unverified against a real
 * producer. `NullHistogram.svelte` does not render anything from it at all
 * in this WP (round-2 dual review: an earlier version guessed at its shape
 * and rendered a strip from it, which was removed) — this artifact's own
 * validation below never depends on it either. Rendering `trained` belongs
 * to WP3, against `null-report.ts`'s real output.
 */
export interface RewiringNullScoreStats {
  score: number;
  median: number;
  std: number;
  ci: readonly [number, number];
}

export interface RewiringNullRewiredEntry extends RewiringNullScoreStats {
  seed: number;
  gzipSha256: string;
  acceptedSwaps: number;
  attempts: number;
}

export interface RewiringNullSummary {
  n: number;
  mean: number;
  median: number;
  std: number;
  p2_5: number;
  p97_5: number;
  iqr: number;
  degenerate: boolean;
}

export interface RewiringNullBins {
  /** `counts.length + 1` sorted (non-decreasing) bin boundaries. */
  edges: readonly number[];
  /** One non-negative integer count per bin; `counts.length === edges.length - 1`. */
  counts: readonly number[];
}

export interface RewiringNullArtifact {
  version: number;
  condition: string;
  seeds: { start: number; count: number };
  ticks: number;
  substeps: number;
  sourceGraphSha256: string;
  rewireSourceSha256: string;
  shards: number;
  biological: RewiringNullScoreStats;
  disconnected: RewiringNullScoreStats;
  /** Sorted by seed (0…499); the 500-graph null set `NullHistogram.svelte` bins bars from. */
  rewired: readonly RewiringNullRewiredEntry[];
  null: RewiringNullSummary;
  /** Biological's empirical percentile in the null set, in `[0, 1]` (e.g. `0` means biological scored below every rewired graph). */
  bioPercentile: number;
  pLow: number;
  pHigh: number;
  bins: RewiringNullBins;
  /** WP3's trained-sample section; see this interface's own doc comment. */
  trained?: unknown;
}

export type RewiringNullLoadResult =
  | { status: 'ok'; data: RewiringNullArtifact }
  | { status: 'missing'; reason: string }
  | { status: 'invalid'; reason: string };

/** Exported as a small, reusable guard for any future caller that needs the same one-line predicate (round-2 dual review: the doc comment previously claimed `NullHistogram.svelte` imports this, but its own local copy — used for a since-removed `trained`-section narrowing — was deleted outright rather than replaced with this import). */
export const isFiniteNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);

const isPositiveInteger = (value: unknown): value is number => Number.isInteger(value) && (value as number) > 0;

const isUnitInterval = (value: unknown): value is number => isFiniteNumber(value) && value >= 0 && value <= 1;

/** `ci[0] <= ci[1]` — a confidence interval whose bounds are swapped is itself a sign of a producer bug, not a real interval (dual review). */
const isFiniteCiPair = (value: unknown): value is readonly [number, number] =>
  Array.isArray(value) &&
  value.length === 2 &&
  isFiniteNumber(value[0]) &&
  isFiniteNumber(value[1]) &&
  (value[0] as number) <= (value[1] as number);

const isScoreStats = (value: unknown): value is RewiringNullScoreStats => {
  if (typeof value !== 'object' || value === null) return false;
  const v = value as Record<string, unknown>;
  // `std` is a standard deviation: never negative for real data (dual review).
  return (
    isFiniteNumber(v.score) && isFiniteNumber(v.median) && isFiniteNumber(v.std) && (v.std as number) >= 0 && isFiniteCiPair(v.ci)
  );
};

const isRewiredEntry = (value: unknown): value is RewiringNullRewiredEntry => {
  if (!isScoreStats(value)) return false;
  const v = value as unknown as Record<string, unknown>;
  return (
    isFiniteNumber(v.seed) &&
    typeof v.gzipSha256 === 'string' &&
    isFiniteNumber(v.acceptedSwaps) &&
    isFiniteNumber(v.attempts)
  );
};

const isSummary = (value: unknown): value is RewiringNullSummary => {
  if (typeof value !== 'object' || value === null) return false;
  const v = value as Record<string, unknown>;
  return (
    isPositiveInteger(v.n) &&
    isFiniteNumber(v.mean) &&
    isFiniteNumber(v.median) &&
    isFiniteNumber(v.std) &&
    (v.std as number) >= 0 &&
    isFiniteNumber(v.p2_5) &&
    isFiniteNumber(v.p97_5) &&
    (v.p2_5 as number) <= (v.p97_5 as number) &&
    isFiniteNumber(v.iqr) &&
    (v.iqr as number) >= 0 &&
    typeof v.degenerate === 'boolean'
  );
};

/**
 * `edges` must be sorted (non-decreasing — `null-report.ts` writes strictly
 * increasing equal-width edges, but non-decreasing is the weakest check that
 * still catches a shuffled/corrupted array without rejecting a legitimate
 * degenerate bin) and `counts` must have exactly one fewer entry than
 * `edges`, every one a finite, non-negative integer count (dual review:
 * "finite" alone let a fractional count like `0.5` through).
 */
const isBins = (value: unknown): value is RewiringNullBins => {
  if (typeof value !== 'object' || value === null) return false;
  const v = value as Record<string, unknown>;
  if (!Array.isArray(v.edges) || !Array.isArray(v.counts)) return false;
  if (v.edges.length < 2 || v.counts.length !== v.edges.length - 1) return false;
  for (let i = 0; i < v.edges.length; i += 1) {
    if (!isFiniteNumber(v.edges[i])) return false;
    if (i > 0 && (v.edges[i] as number) < (v.edges[i - 1] as number)) return false;
  }
  for (const count of v.counts) {
    if (!Number.isInteger(count) || (count as number) < 0) return false;
  }
  return true;
};

/**
 * Structural validation (version 1, sorted bins, finite numbers) *plus* the
 * cross-field consistency the UI relies on but a per-field check alone
 * cannot catch (dual review, both reviewers, Important): a hash-valid
 * artifact can still contain internally-contradictory numbers (a
 * `bioPercentile` outside `[0, 1]`, a bin-count total that disagrees with
 * `null.n`/`rewired.length`, or a marker score that falls outside the
 * histogram's own domain and would render off-canvas while its legend entry
 * still claims it is shown). Mirrors `loadPositions`'s "never throws,
 * always return a reasoned status" contract. `trained` is deliberately left
 * unchecked (see `RewiringNullArtifact.trained`'s doc comment) — a
 * malformed `trained` section never fails the whole artifact.
 */
const validateRewiringNullShape = (value: unknown): { ok: true; data: RewiringNullArtifact } | { ok: false; reason: string } => {
  if (typeof value !== 'object' || value === null) {
    return { ok: false, reason: 'rewiring-null artifact is not a JSON object' };
  }
  const v = value as Record<string, unknown>;
  if (v.version !== 1) return { ok: false, reason: `rewiring-null artifact has unsupported version ${String(v.version)}` };
  if (typeof v.condition !== 'string') return { ok: false, reason: 'rewiring-null artifact is missing "condition"' };
  if (
    typeof v.seeds !== 'object' ||
    v.seeds === null ||
    !isFiniteNumber((v.seeds as Record<string, unknown>).start) ||
    !isPositiveInteger((v.seeds as Record<string, unknown>).count)
  ) {
    return { ok: false, reason: 'rewiring-null artifact has a malformed "seeds" field' };
  }
  if (!isFiniteNumber(v.ticks) || !isFiniteNumber(v.substeps) || !isFiniteNumber(v.shards)) {
    return { ok: false, reason: 'rewiring-null artifact is missing ticks/substeps/shards' };
  }
  if (typeof v.sourceGraphSha256 !== 'string' || typeof v.rewireSourceSha256 !== 'string') {
    return { ok: false, reason: 'rewiring-null artifact is missing sourceGraphSha256/rewireSourceSha256' };
  }
  if (!isScoreStats(v.biological)) return { ok: false, reason: 'rewiring-null artifact has a malformed "biological" field' };
  if (!isScoreStats(v.disconnected)) return { ok: false, reason: 'rewiring-null artifact has a malformed "disconnected" field' };
  if (!Array.isArray(v.rewired) || v.rewired.length === 0 || !v.rewired.every(isRewiredEntry)) {
    return { ok: false, reason: 'rewiring-null artifact has a malformed "rewired" array' };
  }
  const rewired = v.rewired as RewiringNullRewiredEntry[];
  // The bars claim to be "the N rewired graphs only" (`NullHistogram.svelte`'s
  // own non-negotiable) — a duplicated seed would double-count one graph
  // and silently misrepresent the null set.
  if (new Set(rewired.map((entry) => entry.seed)).size !== rewired.length) {
    return { ok: false, reason: 'rewiring-null artifact has duplicate seeds in "rewired"' };
  }
  if (!isSummary(v.null)) return { ok: false, reason: 'rewiring-null artifact has a malformed "null" summary field' };
  const summary = v.null as RewiringNullSummary;
  if (!isUnitInterval(v.bioPercentile) || !isUnitInterval(v.pLow) || !isUnitInterval(v.pHigh)) {
    return { ok: false, reason: 'rewiring-null artifact has bioPercentile/pLow/pHigh outside [0, 1]' };
  }
  if (!isBins(v.bins)) return { ok: false, reason: 'rewiring-null artifact has a malformed or unsorted "bins" field' };
  const bins = v.bins as RewiringNullBins;

  // The bars are drawn straight from `bins.counts` and captioned as "the N
  // rewired graphs" (`data.null.n`) — these three counts must agree, or the
  // chart and its own caption would each tell a different story.
  const binTotal = bins.counts.reduce((sum, count) => sum + count, 0);
  if (summary.n !== rewired.length || binTotal !== rewired.length) {
    return {
      ok: false,
      reason: `rewiring-null counts disagree (null.n=${summary.n}, rewired.length=${rewired.length}, sum(bins.counts)=${binTotal})`
    };
  }

  // Every marker `NullHistogram.svelte` draws (biological, disconnected,
  // and the shipped rewired-seed-0 control, when present) must fall inside
  // the histogram's own domain — `null-report.ts` widens the bin edges to
  // guarantee exactly this, so a marker outside `[edges[0], edges[last]]`
  // means the artifact is internally inconsistent, not merely that this
  // loader forgot to clamp it.
  const domainLow = bins.edges[0];
  const domainHigh = bins.edges[bins.edges.length - 1];
  const inDomain = (score: number): boolean => score >= domainLow && score <= domainHigh;
  const biological = v.biological as RewiringNullScoreStats;
  const disconnected = v.disconnected as RewiringNullScoreStats;
  const seed0 = rewired.find((entry) => entry.seed === 0);
  if (!inDomain(biological.score) || !inDomain(disconnected.score) || (seed0 && !inDomain(seed0.score))) {
    return {
      ok: false,
      reason: 'rewiring-null artifact has a marker (biological/disconnected/rewired-seed-0) outside the histogram bin domain'
    };
  }

  return { ok: true, data: value as RewiringNullArtifact };
};

/**
 * Fetch, sha256-verify, and structurally validate `rewiring-null-v1.json`
 * (WP4's counterpart to `loadTrainedReadoutArtifact` above). Never throws —
 * every failure mode is a returned `status`, matching `TrainedReadoutLoadResult`'s
 * and `PositionsLoadResult`'s "optional presentation, not a Start gate"
 * contract: a missing/tampered/malformed null-distribution artifact only
 * ever hides or degrades the ledger panel's "Topology null distribution"
 * section (`LedgerPanel.svelte`), never the experiment itself.
 *
 * `dataBaseUrl` must be the same value the caller passes to
 * `loadArenaArtifacts`/`loadTrainedReadoutArtifact` (`ExperimentController#initialize`
 * passes `${import.meta.env.BASE_URL}data`) so this artifact resolves under
 * the app's real deployment base path too.
 */
export const loadRewiringNull = async (
  manifest: ArenaManifest,
  dataBaseUrl: string
): Promise<RewiringNullLoadResult> => {
  const entry = manifest.rewiringNull;
  if (!entry) {
    return { status: 'missing', reason: 'The manifest has no rewiringNull artifact entry.' };
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
      reason: `rewiring-null artifact sha256 ${rawHash} does not match the manifest (${entry.sha256})`
    };
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(new TextDecoder().decode(rawBytes));
  } catch (error) {
    return {
      status: 'invalid',
      reason: `rewiring-null artifact is not valid JSON: ${error instanceof Error ? error.message : String(error)}`
    };
  }

  const validated = validateRewiringNullShape(parsed);
  if (!validated.ok) return { status: 'invalid', reason: validated.reason };
  const data = validated.data;

  // The sha256 check above only proves these bytes are the ones the
  // manifest's `rewiringNull` entry pins — it says nothing about whether
  // this artifact actually describes *this* manifest's graphs (a
  // hand-edited or merge-conflicted manifest could re-pin `rewiringNull` to
  // a null distribution computed against a different biological/rewired
  // graph, including a "shipped" seed-0 marker that is not really the
  // shipped control arm). `loadPositions` above already runs the equivalent
  // staleness check for its own sidecar artifact ("positions.graphSha256
  // does not match the manifest's compiled graph gzip sha256") — this
  // mirrors that precedent (dual review, Important).
  if (data.sourceGraphSha256 !== manifest.binarySha256) {
    return {
      status: 'invalid',
      reason: `rewiring-null sourceGraphSha256 ${data.sourceGraphSha256} does not match the manifest's biological graph (${manifest.binarySha256}) — stale artifact`
    };
  }
  const seed0 = data.rewired.find((entry) => entry.seed === 0);
  const shippedSeed0GzipSha256 = manifest.rewiredArms.seed0?.gzipSha256;
  if (seed0 && shippedSeed0GzipSha256 && seed0.gzipSha256 !== shippedSeed0GzipSha256) {
    return {
      status: 'invalid',
      reason: 'rewiring-null seed 0 does not match the shipped rewired control arm (rewiredArms.seed0)'
    };
  }

  return { status: 'ok', data };
};
