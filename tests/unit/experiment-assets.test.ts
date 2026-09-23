import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  ArtifactIntegrityError,
  DecompressionUnsupportedError,
  decompressGzip,
  loadArenaArtifacts,
  sha256Hex,
  verifyAndDecompressArtifact
} from '../../src/lib/experiment/assets';
import { parseGraphBinary, validateGraph } from '../../src/lib/connectome/format';
import { createPublicDataFetch } from '../helpers/fake-worker';

const here = dirname(fileURLToPath(import.meta.url));
const publicDataDir = resolve(here, '../../public/data');

const toArrayBuffer = (buffer: Buffer): ArrayBuffer =>
  buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength) as ArrayBuffer;

interface Manifest {
  artifact: string;
  binarySha256: string;
  binaryBytes: number;
  gzipSha256: string;
  gzipBytes: number;
  neuronCount: number;
  rewiredArms: Record<string, { artifact: string; binarySha256: string; binaryBytes: number; gzipSha256: string; gzipBytes: number }>;
}

const manifest = JSON.parse(
  readFileSync(resolve(publicDataDir, 'malecns-arena-v1.manifest.json'), 'utf-8')
) as Manifest;

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('sha256Hex / decompressGzip (real artifact bytes)', () => {
  it('sha256Hex matches the manifest for both the gzip and decompressed bytes', async () => {
    const gzipBytes = toArrayBuffer(readFileSync(resolve(publicDataDir, 'malecns-arena-v1.bin.gz')));
    expect(await sha256Hex(gzipBytes)).toBe(manifest.gzipSha256);

    const binary = await decompressGzip(gzipBytes);
    expect(await sha256Hex(binary)).toBe(manifest.binarySha256);
  });

  it('decompressGzip throws DecompressionUnsupportedError when the global is unavailable', async () => {
    const gzipBytes = toArrayBuffer(readFileSync(resolve(publicDataDir, 'malecns-arena-v1.bin.gz')));
    vi.stubGlobal('DecompressionStream', undefined);
    await expect(decompressGzip(gzipBytes)).rejects.toBeInstanceOf(DecompressionUnsupportedError);
  });
});

describe('verifyAndDecompressArtifact', () => {
  it('succeeds and returns the exact decompressed graph bytes for the real biological artifact', async () => {
    const gzipBytes = toArrayBuffer(readFileSync(resolve(publicDataDir, 'malecns-arena-v1.bin.gz')));
    const binary = await verifyAndDecompressArtifact(gzipBytes, manifest);
    expect(binary.byteLength).toBe(manifest.binaryBytes);

    const graph = parseGraphBinary(binary);
    expect(() => validateGraph(graph)).not.toThrow();
    expect(graph.metadata.neuronCount).toBeGreaterThan(0);
  });

  it('throws ArtifactIntegrityError when a fetched byte is corrupted (gzip hash check fails first)', async () => {
    const gzipBytes = Buffer.from(readFileSync(resolve(publicDataDir, 'malecns-arena-v1.bin.gz')));
    gzipBytes[0] ^= 0xff;
    await expect(verifyAndDecompressArtifact(toArrayBuffer(gzipBytes), manifest)).rejects.toBeInstanceOf(
      ArtifactIntegrityError
    );
  });

  it('throws ArtifactIntegrityError on a gzip byte-length mismatch before hashing anything', async () => {
    const truncated = toArrayBuffer(readFileSync(resolve(publicDataDir, 'malecns-arena-v1.bin.gz'))).slice(0, 10);
    await expect(verifyAndDecompressArtifact(truncated, manifest)).rejects.toBeInstanceOf(ArtifactIntegrityError);
  });

  it('verifies successfully even when the transport already decompressed the bytes (e.g. a static server sending Content-Encoding: gzip)', async () => {
    // Real-world case (confirmed against `vite preview`, which serves a
    // `.gz` file with `Content-Encoding: gzip` and lets the Fetch API
    // transparently decode it): `verifyAndDecompressArtifact` must detect
    // this structurally (no gzip magic number) and verify against the
    // decompressed-bytes manifest entry instead of failing or silently
    // mis-verifying against the wrong hash.
    const gzipBytes = readFileSync(resolve(publicDataDir, 'malecns-arena-v1.bin.gz'));
    const alreadyDecompressed = toArrayBuffer(Buffer.from(await decompressGzip(toArrayBuffer(gzipBytes))));
    const result = await verifyAndDecompressArtifact(alreadyDecompressed, manifest);
    expect(result.byteLength).toBe(manifest.binaryBytes);
    expect(await sha256Hex(result)).toBe(manifest.binarySha256);
  });

  it('also verifies the rewired-seed0 control arm against its own manifest entry', async () => {
    const entry = manifest.rewiredArms.seed0;
    const gzipBytes = toArrayBuffer(readFileSync(resolve(publicDataDir, entry.artifact)));
    const binary = await verifyAndDecompressArtifact(gzipBytes, entry);
    expect(binary.byteLength).toBe(entry.binaryBytes);
  });
});

describe('loadArenaArtifacts (fetch -> gunzip -> hash-verify, against real committed files)', () => {
  it('loads and verifies both arms via a fetch stand-in that serves the real public/data files', async () => {
    vi.stubGlobal('fetch', createPublicDataFetch());
    const artifacts = await loadArenaArtifacts('/data', 'malecns-arena-v1.manifest.json');
    expect(artifacts.manifest.neuronCount).toBe(manifest.neuronCount);
    expect(artifacts.biological.byteLength).toBe(manifest.binaryBytes);
    expect(artifacts.rewired.byteLength).toBe(manifest.rewiredArms.seed0.binaryBytes);

    const biologicalGraph = parseGraphBinary(artifacts.biological);
    const rewiredGraph = parseGraphBinary(artifacts.rewired);
    expect(() => validateGraph(biologicalGraph)).not.toThrow();
    expect(() => validateGraph(rewiredGraph)).not.toThrow();
  });

  it('rejects (never resolves to usable graph bytes) when the biological artifact is corrupted in transit', async () => {
    vi.stubGlobal('fetch', createPublicDataFetch({ corrupt: 'malecns-arena-v1.bin.gz' }));
    await expect(loadArenaArtifacts('/data', 'malecns-arena-v1.manifest.json')).rejects.toBeInstanceOf(
      ArtifactIntegrityError
    );
  });

  it('loads and verifies successfully when the fetch transport transparently decompressed the .gz files (the real vite preview behavior)', async () => {
    vi.stubGlobal('fetch', createPublicDataFetch({ transparentGzipDecode: true }));
    const artifacts = await loadArenaArtifacts('/data', 'malecns-arena-v1.manifest.json');
    expect(artifacts.biological.byteLength).toBe(manifest.binaryBytes);
    expect(artifacts.rewired.byteLength).toBe(manifest.rewiredArms.seed0.binaryBytes);
  });

  it('rejects when the manifest has no seed0 rewired entry', async () => {
    vi.stubGlobal('fetch', async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.endsWith('.manifest.json')) {
        return new Response(JSON.stringify({ ...manifest, rewiredArms: {} }), {
          status: 200,
          headers: { 'content-type': 'application/json' }
        });
      }
      return createPublicDataFetch()(input);
    });
    await expect(loadArenaArtifacts('/data', 'malecns-arena-v1.manifest.json')).rejects.toThrow(/seed0/);
  });
});
