import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gunzipSync } from 'node:zlib';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { loadPositions, type ArenaManifest } from '../../src/lib/experiment/assets';
import { parseGraphBinary, type ConnectomeGraph } from '../../src/lib/connectome/format';
import { createPublicDataFetch } from '../helpers/fake-worker';

/**
 * `loadPositions` (WP3) against the real, committed
 * `public/data/malecns-arena-v1.positions.json` and its manifest entry — the
 * same "test against real committed files" discipline
 * `tests/unit/experiment-assets.test.ts` uses for the graph artifacts.
 *
 * `loadPositions` no longer fetches/parses the biological graph itself
 * (thermo-architecture I1 fix) — callers thread in the already-parsed graph.
 * `biologicalGraph` below is built directly from the real committed
 * `.bin.gz` artifact (no `fetch` involved), mirroring what
 * `ExperimentController.initialize()` -> `App.svelte`'s `onManifest` really
 * hands `loadPositions` in production.
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

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('loadPositions', () => {
  it('loads and verifies the real positions artifact, cross-checked against the real compiled graph', async () => {
    vi.stubGlobal('fetch', createPublicDataFetch());

    const result = await loadPositions(manifest, '/data', biologicalGraph);

    expect(result.status).toBe('ok');
    if (result.status !== 'ok') throw new Error(`expected ok, got ${result.status}`);
    expect(result.positions.bodyIds.length).toBe(manifest.neuronCount);
    expect(result.positions.coverage).toEqual(manifest.positions?.coverage);
    expect(result.positions.coverage.soma + result.positions.coverage.tosoma + result.positions.coverage.none).toBe(
      manifest.neuronCount
    );
    expect(result.rateMin).toBeLessThanOrEqual(result.rateMax);
  });

  it('returns "missing" (never throws) when the manifest has no positions entry', async () => {
    const { positions: _omitted, ...withoutPositions } = manifest;
    const result = await loadPositions(withoutPositions as ArenaManifest, '/data', biologicalGraph);
    expect(result.status).toBe('missing');
  });

  it('returns "invalid" when the fetched positions artifact fails its sha256 check', async () => {
    vi.stubGlobal('fetch', createPublicDataFetch({ corrupt: 'malecns-arena-v1.positions.json' }));

    const result = await loadPositions(manifest, '/data', biologicalGraph);

    expect(result.status).toBe('invalid');
    if (result.status !== 'invalid') throw new Error('expected invalid');
    expect(result.reason).toMatch(/sha256/i);
  });

  it('returns "invalid" when positions.bodyIds order does not match the real graph’s biologicalIds', async () => {
    const real = JSON.parse(readFileSync(resolve(publicDataDir, 'malecns-arena-v1.positions.json'), 'utf-8')) as {
      bodyIds: string[];
      [key: string]: unknown;
    };
    const tampered = { ...real, bodyIds: [...real.bodyIds] };
    [tampered.bodyIds[0], tampered.bodyIds[1]] = [tampered.bodyIds[1], tampered.bodyIds[0]];
    const tamperedBytes = Buffer.from(JSON.stringify(tampered));
    const tamperedSha256 = createHash('sha256').update(tamperedBytes).digest('hex');

    vi.stubGlobal('fetch', async (input: RequestInfo | URL): Promise<Response> => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.endsWith('malecns-arena-v1.positions.json')) {
        return new Response(tamperedBytes, { status: 200, headers: { 'content-type': 'application/json' } });
      }
      return createPublicDataFetch()(input);
    });

    // Update just the positions manifest entry's sha256 so layer 1 (raw
    // bytes vs. manifest.positions.sha256) still passes and this test
    // exercises the bodyIds-vs-graph cross-check specifically, not an
    // incidental hash mismatch.
    const tamperedManifest: ArenaManifest = {
      ...manifest,
      positions: { ...manifest.positions!, sha256: tamperedSha256 }
    };

    const result = await loadPositions(tamperedManifest, '/data', biologicalGraph);

    expect(result.status).toBe('invalid');
    if (result.status !== 'invalid') throw new Error('expected invalid');
    expect(result.reason).toMatch(/bodyIds/);
  });

  it('returns "invalid" when a positionSource "none" entry carries a non-null xyz (would silently invent a position)', async () => {
    const real = JSON.parse(readFileSync(resolve(publicDataDir, 'malecns-arena-v1.positions.json'), 'utf-8')) as {
      positionSource: string[];
      xyz: (readonly [number, number, number] | null)[];
      [key: string]: unknown;
    };
    const noneIndex = real.positionSource.findIndex((source) => source === 'none');
    expect(noneIndex).toBeGreaterThanOrEqual(0);
    const tampered = { ...real, xyz: [...real.xyz] };
    tampered.xyz[noneIndex] = [1, 2, 3];
    const tamperedBytes = Buffer.from(JSON.stringify(tampered));
    const tamperedSha256 = createHash('sha256').update(tamperedBytes).digest('hex');

    vi.stubGlobal('fetch', async (input: RequestInfo | URL): Promise<Response> => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.endsWith('malecns-arena-v1.positions.json')) {
        return new Response(tamperedBytes, { status: 200, headers: { 'content-type': 'application/json' } });
      }
      return createPublicDataFetch()(input);
    });

    const tamperedManifest: ArenaManifest = {
      ...manifest,
      positions: { ...manifest.positions!, sha256: tamperedSha256 }
    };

    const result = await loadPositions(tamperedManifest, '/data', biologicalGraph);

    expect(result.status).toBe('invalid');
    if (result.status !== 'invalid') throw new Error('expected invalid');
    expect(result.reason).toMatch(/xyz/);
  });

  it('returns "invalid" when the declared coverage disagrees with the actual positionSource counts (dual review finding)', async () => {
    const real = JSON.parse(readFileSync(resolve(publicDataDir, 'malecns-arena-v1.positions.json'), 'utf-8')) as {
      coverage: { soma: number; tosoma: number; none: number };
      [key: string]: unknown;
    };
    const tampered = { ...real, coverage: { ...real.coverage, soma: real.coverage.soma + 1, none: real.coverage.none - 1 } };
    const tamperedBytes = Buffer.from(JSON.stringify(tampered));
    const tamperedSha256 = createHash('sha256').update(tamperedBytes).digest('hex');

    vi.stubGlobal('fetch', async (input: RequestInfo | URL): Promise<Response> => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.endsWith('malecns-arena-v1.positions.json')) {
        return new Response(tamperedBytes, { status: 200, headers: { 'content-type': 'application/json' } });
      }
      return createPublicDataFetch()(input);
    });

    const tamperedManifest: ArenaManifest = { ...manifest, positions: { ...manifest.positions!, sha256: tamperedSha256 } };
    const result = await loadPositions(tamperedManifest, '/data', biologicalGraph);

    expect(result.status).toBe('invalid');
    if (result.status !== 'invalid') throw new Error('expected invalid');
    expect(result.reason).toMatch(/coverage/i);
  });

  it('returns "invalid" when units does not disclose that they are unverified', async () => {
    const real = JSON.parse(readFileSync(resolve(publicDataDir, 'malecns-arena-v1.positions.json'), 'utf-8')) as {
      [key: string]: unknown;
    };
    const tampered = { ...real, units: 'meters' };
    const tamperedBytes = Buffer.from(JSON.stringify(tampered));
    const tamperedSha256 = createHash('sha256').update(tamperedBytes).digest('hex');

    vi.stubGlobal('fetch', async (input: RequestInfo | URL): Promise<Response> => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.endsWith('malecns-arena-v1.positions.json')) {
        return new Response(tamperedBytes, { status: 200, headers: { 'content-type': 'application/json' } });
      }
      return createPublicDataFetch()(input);
    });

    const tamperedManifest: ArenaManifest = { ...manifest, positions: { ...manifest.positions!, sha256: tamperedSha256 } };
    const result = await loadPositions(tamperedManifest, '/data', biologicalGraph);

    expect(result.status).toBe('invalid');
    if (result.status !== 'invalid') throw new Error('expected invalid');
    expect(result.reason).toMatch(/units/i);
  });

  it('returns "invalid" (never throws) when the hash-matched positions file is valid JSON but not an object (e.g. null)', async () => {
    const tamperedBytes = Buffer.from('null');
    const tamperedSha256 = createHash('sha256').update(tamperedBytes).digest('hex');

    vi.stubGlobal('fetch', async (input: RequestInfo | URL): Promise<Response> => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.endsWith('malecns-arena-v1.positions.json')) {
        return new Response(tamperedBytes, { status: 200, headers: { 'content-type': 'application/json' } });
      }
      return createPublicDataFetch()(input);
    });

    const tamperedManifest: ArenaManifest = { ...manifest, positions: { ...manifest.positions!, sha256: tamperedSha256 } };
    const result = await loadPositions(tamperedManifest, '/data', biologicalGraph);

    expect(result.status).toBe('invalid');
    if (result.status !== 'invalid') throw new Error('expected invalid');
    expect(result.reason).toMatch(/not a json object/i);
  });

  it('never fetches the biological graph artifact itself (thermo-architecture I1 fix: the parsed graph is passed in, not re-fetched)', async () => {
    const fetchedUrls: string[] = [];
    vi.stubGlobal('fetch', async (input: RequestInfo | URL): Promise<Response> => {
      const url = typeof input === 'string' ? input : input.toString();
      fetchedUrls.push(url);
      return createPublicDataFetch()(input);
    });

    const result = await loadPositions(manifest, '/data', biologicalGraph);

    expect(result.status).toBe('ok');
    // Only the positions sidecar is fetched — the graph artifact (whose
    // manifest entry is `manifest.artifact`) is never touched by `fetch`,
    // because `loadPositions` now uses the already-parsed `biologicalGraph`
    // passed in by the caller instead of re-fetching/re-verifying/
    // re-parsing it a second time.
    const graphFetches = fetchedUrls.filter((url) => url.endsWith(manifest.artifact));
    expect(graphFetches).toHaveLength(0);
    expect(fetchedUrls.filter((url) => url.endsWith('malecns-arena-v1.positions.json'))).toHaveLength(1);
  });

  it('returns "invalid" when the declared roleCounts disagrees with the actual role counts (round-2 review gap)', async () => {
    const real = JSON.parse(readFileSync(resolve(publicDataDir, 'malecns-arena-v1.positions.json'), 'utf-8')) as {
      roleCounts: { sensory: number; bridge: number; descending: number };
      [key: string]: unknown;
    };
    const tampered = {
      ...real,
      roleCounts: { ...real.roleCounts, sensory: real.roleCounts.sensory + 1, bridge: real.roleCounts.bridge - 1 }
    };
    const tamperedBytes = Buffer.from(JSON.stringify(tampered));
    const tamperedSha256 = createHash('sha256').update(tamperedBytes).digest('hex');

    vi.stubGlobal('fetch', async (input: RequestInfo | URL): Promise<Response> => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.endsWith('malecns-arena-v1.positions.json')) {
        return new Response(tamperedBytes, { status: 200, headers: { 'content-type': 'application/json' } });
      }
      return createPublicDataFetch()(input);
    });

    const tamperedManifest: ArenaManifest = { ...manifest, positions: { ...manifest.positions!, sha256: tamperedSha256 } };
    const result = await loadPositions(tamperedManifest, '/data', biologicalGraph);

    expect(result.status).toBe('invalid');
    if (result.status !== 'invalid') throw new Error('expected invalid');
    expect(result.reason).toMatch(/roleCounts/);
  });

  it('returns "invalid" (never throws) when the positions artifact is missing its coverage field entirely (round-2 review gap)', async () => {
    const real = JSON.parse(readFileSync(resolve(publicDataDir, 'malecns-arena-v1.positions.json'), 'utf-8')) as {
      [key: string]: unknown;
    };
    const { coverage: _omitted, ...withoutCoverage } = real;
    const tamperedBytes = Buffer.from(JSON.stringify(withoutCoverage));
    const tamperedSha256 = createHash('sha256').update(tamperedBytes).digest('hex');

    vi.stubGlobal('fetch', async (input: RequestInfo | URL): Promise<Response> => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.endsWith('malecns-arena-v1.positions.json')) {
        return new Response(tamperedBytes, { status: 200, headers: { 'content-type': 'application/json' } });
      }
      return createPublicDataFetch()(input);
    });

    const tamperedManifest: ArenaManifest = { ...manifest, positions: { ...manifest.positions!, sha256: tamperedSha256 } };
    const result = await loadPositions(tamperedManifest, '/data', biologicalGraph);

    expect(result.status).toBe('invalid');
    if (result.status !== 'invalid') throw new Error('expected invalid');
    expect(result.reason).toMatch(/coverage/i);
  });
});
