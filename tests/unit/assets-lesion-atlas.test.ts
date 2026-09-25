import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gunzipSync } from 'node:zlib';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { lesionAtlasGraphKeyForTopology, loadLesionAtlas } from '../../src/lib/experiment/lesionAtlas';
import type { ArenaManifest } from '../../src/lib/experiment/assets';
import { parseGraphBinary, type ConnectomeGraph } from '../../src/lib/connectome/format';
import { createPublicDataFetch } from '../helpers/fake-worker';

/**
 * `loadLesionAtlas` (WP3, `.agents/plans/lesion-atlas/03-activity-lesion-mode.md`)
 * against the real, committed `public/data/lesion-atlas-v1.json` and its
 * manifest entry — the same "test against real committed files" discipline
 * `tests/unit/assets-positions.test.ts` uses for `loadPositions`.
 */

const here = dirname(fileURLToPath(import.meta.url));
const publicDataDir = resolve(here, '../../public/data');

const manifest = JSON.parse(
  readFileSync(resolve(publicDataDir, 'malecns-arena-v1.manifest.json'), 'utf-8')
) as ArenaManifest;

const biologicalGraph: ConnectomeGraph = (() => {
  const gzipBytes = readFileSync(resolve(publicDataDir, manifest.artifact));
  const binary = gunzipSync(gzipBytes);
  const arrayBuffer = binary.buffer.slice(binary.byteOffset, binary.byteOffset + binary.byteLength);
  return parseGraphBinary(arrayBuffer);
})();

const realLesionAtlas = JSON.parse(readFileSync(resolve(publicDataDir, 'lesion-atlas-v1.json'), 'utf-8')) as {
  neuronCount: number;
  bodyIds: string[];
  graphs: {
    biological: { graphSha256: string; effect: number[]; fdrSignificant: boolean[] };
    rewiredSeed0: { graphSha256: string; effect: number[]; fdrSignificant: boolean[] };
  };
  [key: string]: unknown;
};

afterEach(() => {
  vi.unstubAllGlobals();
});

/** Matches `tests/helpers/fake-worker.ts`'s own `toArrayBuffer` — `new Response(someBuffer, …)` where `someBuffer` is a Node `Buffer` hits a `BodyInit` overload-resolution quirk under this repo's TS setup; passing a plain `ArrayBuffer` avoids it. */
const toArrayBuffer = (buffer: Buffer): ArrayBuffer =>
  buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength) as ArrayBuffer;

const routeTamperedLesionAtlas = (tamperedBytes: Buffer): void => {
  vi.stubGlobal('fetch', async (input: RequestInfo | URL): Promise<Response> => {
    const url = typeof input === 'string' ? input : input.toString();
    if (url.endsWith('lesion-atlas-v1.json')) {
      return new Response(toArrayBuffer(tamperedBytes), { status: 200, headers: { 'content-type': 'application/json' } });
    }
    return createPublicDataFetch()(input);
  });
};

const tamperedManifestWithSha = (tamperedBytes: Buffer): ArenaManifest => ({
  ...manifest,
  lesionAtlas: { ...manifest.lesionAtlas!, sha256: createHash('sha256').update(tamperedBytes).digest('hex') }
});

describe('lesionAtlasGraphKeyForTopology', () => {
  it('maps biological -> biological and rewired -> rewiredSeed0 (the one shipped rewired seed)', () => {
    expect(lesionAtlasGraphKeyForTopology('biological')).toBe('biological');
    expect(lesionAtlasGraphKeyForTopology('rewired')).toBe('rewiredSeed0');
  });

  it('maps disconnected to undefined — no atlas coverage, never a fabricated key', () => {
    expect(lesionAtlasGraphKeyForTopology('disconnected')).toBeUndefined();
  });
});

describe('loadLesionAtlas', () => {
  it('loads and verifies the real lesion atlas, cross-checked against the real compiled graphs, with a shared absMax', () => {
    vi.stubGlobal('fetch', createPublicDataFetch());

    return loadLesionAtlas(manifest, '/data', biologicalGraph).then((result) => {
      expect(result.status).toBe('ok');
      if (result.status !== 'ok') throw new Error(`expected ok, got ${result.status}`);
      expect(result.data.neuronCount).toBe(manifest.neuronCount);
      expect(result.data.graphs.biological.effect).toHaveLength(manifest.neuronCount);
      expect(result.data.graphs.rewiredSeed0.effect).toHaveLength(manifest.neuronCount);
      // absMax is the max(|effect|) over BOTH graphs together, so it must be
      // at least as large as either graph's own max — and it must actually
      // be attained by one of them (not some unrelated third number).
      const bioMax = Math.max(...result.data.graphs.biological.effect.map(Math.abs));
      const rewiredMax = Math.max(...result.data.graphs.rewiredSeed0.effect.map(Math.abs));
      expect(result.absMax).toBeCloseTo(Math.max(bioMax, rewiredMax), 6);
      expect(result.absMax).toBeGreaterThan(0);
    });
  });

  it('returns "missing" (never throws) when the manifest has no lesionAtlas entry', async () => {
    const { lesionAtlas: _omitted, ...withoutLesionAtlas } = manifest;
    const result = await loadLesionAtlas(withoutLesionAtlas as ArenaManifest, '/data', biologicalGraph);
    expect(result.status).toBe('missing');
  });

  it('returns "invalid" when the fetched lesion atlas fails its sha256 check', async () => {
    vi.stubGlobal('fetch', createPublicDataFetch({ corrupt: 'lesion-atlas-v1.json' }));

    const result = await loadLesionAtlas(manifest, '/data', biologicalGraph);

    expect(result.status).toBe('invalid');
    if (result.status !== 'invalid') throw new Error('expected invalid');
    expect(result.reason).toMatch(/sha256/i);
  });

  it('returns "invalid" when graphs.biological.graphSha256 does not match the manifest\'s biological binarySha256 (stale artifact)', async () => {
    const tampered = { ...realLesionAtlas, graphs: { ...realLesionAtlas.graphs, biological: { ...realLesionAtlas.graphs.biological, graphSha256: 'f'.repeat(64) } } };
    const tamperedBytes = Buffer.from(JSON.stringify(tampered));
    routeTamperedLesionAtlas(tamperedBytes);

    const result = await loadLesionAtlas(tamperedManifestWithSha(tamperedBytes), '/data', biologicalGraph);

    expect(result.status).toBe('invalid');
    if (result.status !== 'invalid') throw new Error('expected invalid');
    expect(result.reason).toMatch(/biological graphSha256/);
  });

  it('returns "invalid" when graphs.rewiredSeed0.graphSha256 does not match the manifest\'s shipped seed-0 binarySha256 (stale artifact)', async () => {
    const tampered = {
      ...realLesionAtlas,
      graphs: { ...realLesionAtlas.graphs, rewiredSeed0: { ...realLesionAtlas.graphs.rewiredSeed0, graphSha256: 'f'.repeat(64) } }
    };
    const tamperedBytes = Buffer.from(JSON.stringify(tampered));
    routeTamperedLesionAtlas(tamperedBytes);

    const result = await loadLesionAtlas(tamperedManifestWithSha(tamperedBytes), '/data', biologicalGraph);

    expect(result.status).toBe('invalid');
    if (result.status !== 'invalid') throw new Error('expected invalid');
    expect(result.reason).toMatch(/rewiredSeed0 graphSha256/);
  });

  it('returns "invalid" when bodyIds order does not match the real graph\'s biologicalIds', async () => {
    const tamperedBodyIds = [...realLesionAtlas.bodyIds];
    [tamperedBodyIds[0], tamperedBodyIds[1]] = [tamperedBodyIds[1], tamperedBodyIds[0]];
    const tampered = { ...realLesionAtlas, bodyIds: tamperedBodyIds };
    const tamperedBytes = Buffer.from(JSON.stringify(tampered));
    routeTamperedLesionAtlas(tamperedBytes);

    const result = await loadLesionAtlas(tamperedManifestWithSha(tamperedBytes), '/data', biologicalGraph);

    expect(result.status).toBe('invalid');
    if (result.status !== 'invalid') throw new Error('expected invalid');
    expect(result.reason).toMatch(/bodyIds/);
  });

  it('returns "invalid" when a graph\'s effect array has the wrong length', async () => {
    const tampered = {
      ...realLesionAtlas,
      graphs: { ...realLesionAtlas.graphs, biological: { ...realLesionAtlas.graphs.biological, effect: realLesionAtlas.graphs.biological.effect.slice(0, 10) } }
    };
    const tamperedBytes = Buffer.from(JSON.stringify(tampered));
    routeTamperedLesionAtlas(tamperedBytes);

    const result = await loadLesionAtlas(tamperedManifestWithSha(tamperedBytes), '/data', biologicalGraph);

    expect(result.status).toBe('invalid');
    if (result.status !== 'invalid') throw new Error('expected invalid');
    expect(result.reason).toMatch(/graphs\.biological/);
  });

  it('returns "invalid" when a ciLow bound exceeds its ciHigh bound (producer-bug fail-closed check)', async () => {
    const tampered = JSON.parse(JSON.stringify(realLesionAtlas)) as typeof realLesionAtlas & {
      graphs: { biological: { ciLow: number[]; ciHigh: number[] } };
    };
    tampered.graphs.biological.ciLow[0] = tampered.graphs.biological.ciHigh[0] + 1;
    const tamperedBytes = Buffer.from(JSON.stringify(tampered));
    routeTamperedLesionAtlas(tamperedBytes);

    const result = await loadLesionAtlas(tamperedManifestWithSha(tamperedBytes), '/data', biologicalGraph);

    expect(result.status).toBe('invalid');
    if (result.status !== 'invalid') throw new Error('expected invalid');
    expect(result.reason).toMatch(/graphs\.biological/);
  });

  it('returns "invalid" (never throws) when the hash-matched artifact is valid JSON but not an object (e.g. null)', async () => {
    const tamperedBytes = Buffer.from('null');
    routeTamperedLesionAtlas(tamperedBytes);

    const result = await loadLesionAtlas(tamperedManifestWithSha(tamperedBytes), '/data', biologicalGraph);

    expect(result.status).toBe('invalid');
    if (result.status !== 'invalid') throw new Error('expected invalid');
    expect(result.reason).toMatch(/not a json object/i);
  });

  it('returns "invalid" when the artifact\'s own neuronCount does not match the manifest\'s (self-consistent but short artifact)', async () => {
    // Internally self-consistent (every array truncated to the same shorter
    // length, so `validateLesionAtlasShape`'s own per-array length checks
    // pass) but disagrees with the real manifest's `neuronCount` — the
    // specific cross-check this test targets, not the shape validator's own
    // length checks (already covered above).
    const shortCount = realLesionAtlas.neuronCount - 1;
    const truncateGraph = (graph: { effect: number[]; ciLow: number[]; ciHigh: number[]; fdrSignificant: boolean[]; [k: string]: unknown }) => ({
      ...graph,
      effect: graph.effect.slice(0, shortCount),
      ciLow: (graph.ciLow as number[]).slice(0, shortCount),
      ciHigh: (graph.ciHigh as number[]).slice(0, shortCount),
      fdrSignificant: (graph.fdrSignificant as boolean[]).slice(0, shortCount)
    });
    const tampered = {
      ...realLesionAtlas,
      neuronCount: shortCount,
      bodyIds: realLesionAtlas.bodyIds.slice(0, shortCount),
      graphs: {
        biological: truncateGraph(realLesionAtlas.graphs.biological as never),
        rewiredSeed0: truncateGraph(realLesionAtlas.graphs.rewiredSeed0 as never)
      }
    };
    const tamperedBytes = Buffer.from(JSON.stringify(tampered));
    routeTamperedLesionAtlas(tamperedBytes);

    const result = await loadLesionAtlas(tamperedManifestWithSha(tamperedBytes), '/data', biologicalGraph);

    expect(result.status).toBe('invalid');
    if (result.status !== 'invalid') throw new Error('expected invalid');
    expect(result.reason).toMatch(/neuronCount/);
  });
});
