/**
 * Fetch, gunzip, and hash-verify the two checked-in MaleCNS-derived graph
 * artifacts (WP6 item 2) before the experiment is allowed to start. Nothing
 * here is Svelte-specific; `src/App.svelte` is the only caller.
 *
 * Every function below is a plain async function over `ArrayBuffer`s (no
 * dependency on a real network) so `tests/unit/experiment-assets.test.ts`
 * can exercise the real integrity-check logic against the real committed
 * `public/data/` files (read via `node:fs`) without a server.
 */

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

  return { manifest, biological, rewired };
};
