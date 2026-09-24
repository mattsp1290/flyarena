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

import { parseGraphBinary } from '../connectome/format';

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

const bytesToHex = (bytes: ArrayBuffer): string =>
  Array.from(new Uint8Array(bytes), (byte) => byte.toString(16).padStart(2, '0')).join('');

export const sha256Hex = async (data: ArrayBuffer): Promise<string> =>
  bytesToHex(await crypto.subtle.digest('SHA-256', data));

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

const fetchArrayBuffer = async (url: string): Promise<ArrayBuffer> => {
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

  return { manifest, biological, rewired };
};
