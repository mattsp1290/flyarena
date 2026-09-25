import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { ArenaManifest } from '../../src/lib/experiment/assets';
import { loadNullExplanation } from '../../src/lib/experiment/nullExplanation';
import { createPublicDataFetch } from '../helpers/fake-worker';

/**
 * WP4 of `.agents/plans/null-explanation` (`04-ledger-note.md`) unit
 * coverage for `nullExplanation.ts#loadNullExplanation`. Mirrors
 * `tests/unit/assets-null.test.ts` (its closest precedent, `loadRewiringNull`)
 * structurally: a first describe block against the real, committed
 * `public/data/null-explanation-v1.json` and manifest entry, and a second
 * against a synthetic manifest/artifact pair built to exercise the shape
 * validator and the cross-checks directly.
 */

const here = dirname(fileURLToPath(import.meta.url));
const publicDataDir = resolve(here, '../../public/data');

const manifest = JSON.parse(
  readFileSync(resolve(publicDataDir, 'malecns-arena-v1.manifest.json'), 'utf-8')
) as ArenaManifest;

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('loadNullExplanation (against the real committed WP3 artifact)', () => {
  it('loads and hash-verifies the real null-explanation-v1.json, returning its parsed contents', async () => {
    vi.stubGlobal('fetch', createPublicDataFetch());
    const result = await loadNullExplanation(manifest, '/data');
    expect(result.status).toBe('ok');
    if (result.status !== 'ok') return;
    expect(result.data.version).toBe(1);
    expect(result.data.finding.summarySentence.length).toBeGreaterThan(0);
    expect(result.data.finding.qualifyingMetrics.length).toBeGreaterThan(0);
    // The real artifact's own cross-check target: its `sources.rewiringNullSha256`
    // must equal the manifest's pinned rewiring-null artifact sha256.
    expect(result.data.sources.rewiringNullSha256).toBe(manifest.rewiringNull?.sha256);
  });

  it('is "missing" with an honest reason when the manifest has no nullExplanation entry', async () => {
    vi.stubGlobal('fetch', createPublicDataFetch());
    const manifestWithoutEntry: ArenaManifest = { ...manifest, nullExplanation: undefined };
    const result = await loadNullExplanation(manifestWithoutEntry, '/data');
    expect(result.status).toBe('missing');
  });

  // `'unavailable'`, not `'missing'` (round-2 dual review, Important): a
  // 404/network failure is a genuine fetch failure, distinct from "the
  // manifest has no entry at all" — mirrors `loadRewiringNull`'s own
  // `'absent'`/`'unavailable'` split.
  it('is "unavailable" (never throws) when the artifact 404s', async () => {
    vi.stubGlobal('fetch', async () => new Response(null, { status: 404, statusText: 'Not Found' }));
    const result = await loadNullExplanation(manifest, '/data');
    expect(result.status).toBe('unavailable');
  });

  it('is "invalid" with a sha256 reason when null-explanation-v1.json is tampered', async () => {
    vi.stubGlobal('fetch', createPublicDataFetch({ corrupt: 'null-explanation-v1.json' }));
    const result = await loadNullExplanation(manifest, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/sha256/i);
  });

  it('is "invalid" (sha256 mismatch, checked before parsing) when the served bytes are not even valid JSON', async () => {
    vi.stubGlobal('fetch', async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.endsWith('null-explanation-v1.json')) {
        return new Response('not json', { status: 200, headers: { 'content-type': 'application/json' } });
      }
      return createPublicDataFetch()(input);
    });
    const result = await loadNullExplanation(manifest, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/sha256/i);
  });
});

describe('loadNullExplanation (shape validation, with a synthetic manifest sha256 that matches the served bytes)', () => {
  const sha256Hex = (data: string): string => createHash('sha256').update(data).digest('hex');

  const validArtifact = {
    version: 1,
    sources: { rewiringNullSha256: manifest.rewiringNull?.sha256 ?? 'a'.repeat(64) },
    variants: { flipBoth: { bioPercentile: 0 } },
    regime: { gatePassed: true },
    finding: {
      categories: ['linearPathway'],
      definitionSensitive: false,
      qualifyingMetrics: [{ kind: 'transfer', name: 'T:rightClearance->thrust', spearman: 0.467 }],
      regimeInvalid: false,
      summarySentence: 'Biological is associated with one qualifying metric.'
    }
  };

  /**
   * Stubs `fetch` to serve `body` (JSON-stringified) at any URL and builds a
   * manifest whose `nullExplanation.sha256` actually matches those served
   * bytes — so `loadNullExplanation`'s sha256 check passes and its shape
   * validator/cross-checks are what each test below actually exercises.
   */
  const manifestServing = (body: unknown): { manifest: ArenaManifest } => {
    const raw = JSON.stringify(body);
    const hash = sha256Hex(raw);
    vi.stubGlobal('fetch', async () => new Response(raw, { status: 200, headers: { 'content-type': 'application/json' } }));
    return {
      manifest: { ...manifest, nullExplanation: { artifact: 'null-explanation-v1.json', sha256: hash } }
    };
  };

  it('is "ok" for a well-formed artifact whose sources.rewiringNullSha256 matches the manifest', async () => {
    const { manifest: manifestForBody } = manifestServing(validArtifact);
    const result = await loadNullExplanation(manifestForBody, '/data');
    expect(result.status).toBe('ok');
  });

  it('is "invalid" for the wrong version', async () => {
    const { manifest: manifestForBody } = manifestServing({ ...validArtifact, version: 2 });
    const result = await loadNullExplanation(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/version/i);
  });

  it('is "invalid" when sources.rewiringNullSha256 is missing', async () => {
    const { rewiringNullSha256: _omit, ...sourcesWithoutSha } = validArtifact.sources;
    const { manifest: manifestForBody } = manifestServing({ ...validArtifact, sources: sourcesWithoutSha });
    const result = await loadNullExplanation(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/sources\.rewiringNullSha256/);
  });

  it('is "invalid" when variants.flipBoth.bioPercentile is missing/out of [0, 1]', async () => {
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      variants: { flipBoth: { bioPercentile: 1.5 } }
    });
    const result = await loadNullExplanation(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/bioPercentile/);
  });

  it('is "invalid" when regime.gatePassed is not a boolean', async () => {
    const { manifest: manifestForBody } = manifestServing({ ...validArtifact, regime: {} });
    const result = await loadNullExplanation(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/gatePassed/);
  });

  it('is "invalid" when finding.summarySentence is empty', async () => {
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      finding: { ...validArtifact.finding, summarySentence: '   ' }
    });
    const result = await loadNullExplanation(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/finding/);
  });

  it('is "invalid" when a qualifying metric has an unknown kind', async () => {
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      finding: {
        ...validArtifact.finding,
        qualifyingMetrics: [{ kind: 'bogus', name: 'x', spearman: 0.5 }]
      }
    });
    const result = await loadNullExplanation(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/finding/);
  });

  // Cross-check, mirroring `loadRewiringNull`'s own
  // `sourceGraphSha256`/`manifest.binarySha256` staleness check.
  it('is "invalid" when sources.rewiringNullSha256 does not match the manifest\'s rewiring-null artifact (a stale/re-pinned artifact)', async () => {
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      sources: { rewiringNullSha256: 'f'.repeat(64) }
    });
    const result = await loadNullExplanation(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/sources\.rewiringNullSha256 does not match/);
  });

  it('is "invalid" when the manifest itself has no rewiringNull.sha256 to cross-check against', async () => {
    const raw = JSON.stringify(validArtifact);
    const hash = sha256Hex(raw);
    vi.stubGlobal('fetch', async () => new Response(raw, { status: 200, headers: { 'content-type': 'application/json' } }));
    const manifestForBody: ArenaManifest = {
      ...manifest,
      rewiringNull: undefined,
      nullExplanation: { artifact: 'null-explanation-v1.json', sha256: hash }
    };
    const result = await loadNullExplanation(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/missing rewiringNull\.sha256/);
  });

  // Mirrors `loadRewiringNull`'s own recomputed-rank-statistics precedent:
  // a hash-valid artifact can still ship internally-contradictory numbers.
  it('is "invalid" when finding.regimeInvalid disagrees with regime.gatePassed', async () => {
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      regime: { gatePassed: true },
      finding: { ...validArtifact.finding, regimeInvalid: true }
    });
    const result = await loadNullExplanation(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/regimeInvalid disagrees with regime\.gatePassed/);
  });

  // Exercises the true `parse-error` path (sha256-matching bytes that are
  // not valid JSON) directly — distinct from the "real committed artifact"
  // describe block's tampered-bytes test above, whose served bytes never
  // match the real manifest's sha256 in the first place, so it only ever
  // exercises the hash-mismatch branch.
  it('is "invalid" when the sha256-matching bytes are not valid JSON (exercises the JSON.parse failure path specifically)', async () => {
    const raw = 'not json';
    const hash = sha256Hex(raw);
    vi.stubGlobal('fetch', async () => new Response(raw, { status: 200, headers: { 'content-type': 'application/json' } }));
    const manifestForBody: ArenaManifest = {
      ...manifest,
      nullExplanation: { artifact: 'null-explanation-v1.json', sha256: hash }
    };
    const result = await loadNullExplanation(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/not valid JSON/i);
  });
});
