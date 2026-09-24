import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { loadPositions, type ArenaManifest } from '../../src/lib/experiment/assets';
import { createPublicDataFetch } from '../helpers/fake-worker';

/**
 * `loadPositions` (WP3) against the real, committed
 * `public/data/malecns-arena-v1.positions.json` and its manifest entry — the
 * same "test against real committed files" discipline
 * `tests/unit/experiment-assets.test.ts` uses for the graph artifacts.
 */

const here = dirname(fileURLToPath(import.meta.url));
const publicDataDir = resolve(here, '../../public/data');

const manifest = JSON.parse(
  readFileSync(resolve(publicDataDir, 'malecns-arena-v1.manifest.json'), 'utf-8')
) as ArenaManifest;

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('loadPositions', () => {
  it('loads and verifies the real positions artifact, cross-checked against the real compiled graph', async () => {
    vi.stubGlobal('fetch', createPublicDataFetch());

    const result = await loadPositions(manifest, '/data');

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
    const result = await loadPositions(withoutPositions as ArenaManifest, '/data');
    expect(result.status).toBe('missing');
  });

  it('returns "invalid" when the fetched positions artifact fails its sha256 check', async () => {
    vi.stubGlobal('fetch', createPublicDataFetch({ corrupt: 'malecns-arena-v1.positions.json' }));

    const result = await loadPositions(manifest, '/data');

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

    const result = await loadPositions(tamperedManifest, '/data');

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

    const result = await loadPositions(tamperedManifest, '/data');

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
    const result = await loadPositions(tamperedManifest, '/data');

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
    const result = await loadPositions(tamperedManifest, '/data');

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
    const result = await loadPositions(tamperedManifest, '/data');

    expect(result.status).toBe('invalid');
    if (result.status !== 'invalid') throw new Error('expected invalid');
    expect(result.reason).toMatch(/not a json object/i);
  });

  it('returns "missing" (not "invalid") when re-fetching the biological graph for the cross-check fails for a transient reason (dual review finding)', async () => {
    vi.stubGlobal('fetch', async (input: RequestInfo | URL): Promise<Response> => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.endsWith('malecns-arena-v1.bin.gz')) {
        throw new Error('simulated network failure');
      }
      return createPublicDataFetch()(input);
    });

    const result = await loadPositions(manifest, '/data');

    expect(result.status).toBe('missing');
    if (result.status !== 'missing') throw new Error('expected missing');
    expect(result.reason).toMatch(/re-fetch/i);
  });
});
