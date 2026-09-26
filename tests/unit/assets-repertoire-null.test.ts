import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { ArenaManifest } from '../../src/lib/experiment/assets';
import { loadRepertoireNull } from '../../src/lib/experiment/repertoireNull';
import { createPublicDataFetch } from '../helpers/fake-worker';

/**
 * WP3 of `.agents/plans/repertoire-null` unit coverage for
 * `repertoire.ts#loadRepertoireNull`. Mirrors
 * `tests/unit/assets-pathway-interventions.test.ts`'s own structure: a
 * first describe block against the real, committed
 * `public/data/behavior-repertoire-null-v1.json` and manifest entry, and a
 * second against a synthetic manifest/artifact pair built to exercise the
 * shape validator and cross-checks directly.
 */

const here = dirname(fileURLToPath(import.meta.url));
const publicDataDir = resolve(here, '../../public/data');

const manifest = JSON.parse(
  readFileSync(resolve(publicDataDir, 'malecns-arena-v1.manifest.json'), 'utf-8')
) as ArenaManifest;

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('loadRepertoireNull (against the real committed WP3 artifact)', () => {
  it('loads and hash-verifies the real behavior-repertoire-null-v1.json, returning its parsed contents', async () => {
    vi.stubGlobal('fetch', createPublicDataFetch());
    const result = await loadRepertoireNull(manifest, '/data');
    expect(result.status).toBe('ok');
    if (result.status !== 'ok') return;
    expect(result.data.version).toBe(1);
    // The real shipped result (independently verified, `.agents/plans/repertoire-null/00-overview.md`'s worked example).
    expect(result.data.primary.bio.occupied).toBe(28);
    expect(result.data.primary.category).toBe('narrower');
    expect(result.data.robustness.robust).toBe(false);
    expect(result.data.robustness.perSeed[1729]).toBe('narrower');
    expect(result.data.robustness.perSeed[1730]).toBe('narrower');
    expect(result.data.robustness.perSeed[1731]).toBe('typical');
    expect(result.data.search.rewiredCount).toBe(20);
    expect(result.data.search.rewiredSeedMatchedCount).toBe(5);
    expect(result.data.disconnected.occupied).toBe(24);
    expect(result.data.sources.biologicalSha).toBe(manifest.binarySha256);
    expect(result.data.sources.rewiringNullSha).toBe(manifest.rewiringNull?.sha256);
  });

  it('is "missing" with an honest reason when the manifest has no behaviorRepertoireNull entry', async () => {
    vi.stubGlobal('fetch', createPublicDataFetch());
    const manifestWithoutEntry: ArenaManifest = { ...manifest, behaviorRepertoireNull: undefined };
    const result = await loadRepertoireNull(manifestWithoutEntry, '/data');
    expect(result.status).toBe('missing');
  });

  it('is "unavailable" (never throws) when the artifact 404s', async () => {
    vi.stubGlobal('fetch', async () => new Response(null, { status: 404, statusText: 'Not Found' }));
    const result = await loadRepertoireNull(manifest, '/data');
    expect(result.status).toBe('unavailable');
  });

  it('is "invalid" with a sha256 reason when behavior-repertoire-null-v1.json is tampered', async () => {
    vi.stubGlobal('fetch', createPublicDataFetch({ corrupt: 'behavior-repertoire-null-v1.json' }));
    const result = await loadRepertoireNull(manifest, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/sha256/i);
  });
});

describe('loadRepertoireNull (shape validation, with a synthetic manifest sha256 that matches the served bytes)', () => {
  const sha256Hex = (data: string): string => createHash('sha256').update(data).digest('hex');

  // Ascending, length-20 (`search.rewiredCount`), with p25 <= p50 <= p75
  // bounded by the values themselves -- every structural property the
  // tightened `isRewiredDistribution` now requires (a maintainability
  // review, Important: an earlier fixture's 3-element arrays no longer
  // satisfy the "distribution size must equal search.rewiredCount" check).
  const distribution = (values: readonly number[]) => {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return { n: sorted.length, p25: sorted[Math.floor(sorted.length / 4)], p50: sorted[mid], p75: sorted[sorted.length - 1 - Math.floor(sorted.length / 4)], values: sorted };
  };
  const OCCUPIED_20 = [26, 26, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 29, 30, 30, 31, 32];
  const QD_20 = [616.6, 688.3, 695.1, 702.2, 725.3, 728.2, 733.1, 743.9, 749.8, 759.8, 763.0, 763.4, 786.1, 808.5, 828.4, 845.3, 878.1, 893.0, 896.6, 922.4];

  const validArtifact = {
    version: 1,
    sources: {
      biologicalSha: manifest.binarySha256,
      rewiringNullSha: manifest.rewiringNull?.sha256 ?? 'a'.repeat(64),
      // Not cross-checked by `loadRepertoireNull` itself (see that
      // function's own doc comment: `Atlas.svelte` is where the atlas
      // staleness check happens, once it has also loaded the atlas) --
      // any well-formed hex string is fine here.
      atlasSha256: 'c'.repeat(64)
    },
    search: {
      population: 64,
      generations: 24,
      ticks: 900,
      primarySearchSeed: 1729,
      extraSearchSeeds: [1730, 1731, 1732, 1733],
      rewiredCount: 20,
      rewiredSeedMatchedCount: 5
    },
    primary: {
      bio: { occupied: 28, qd: 637.49, span: 11, heldoutOwnMedian: 16.55 },
      rewiredDistribution: { occupied: distribution(OCCUPIED_20), qd: distribution(QD_20) },
      category: 'narrower',
      tie: false
    },
    robustness: {
      perSeed: { '1729': 'narrower', '1730': 'narrower', '1731': 'typical', '1732': 'typical', '1733': 'typical' },
      robust: false,
      seedMatchedCounts: { '1729': 20, '1730': 5, '1731': 5, '1732': 5, '1733': 5 }
    },
    disconnected: { occupied: 24, qd: 343.9 }
  };

  const manifestServing = (body: unknown): { manifest: ArenaManifest } => {
    const raw = JSON.stringify(body);
    const hash = sha256Hex(raw);
    vi.stubGlobal('fetch', async () => new Response(raw, { status: 200, headers: { 'content-type': 'application/json' } }));
    return { manifest: { ...manifest, behaviorRepertoireNull: { artifact: 'behavior-repertoire-null-v1.json', sha256: hash } } };
  };

  it('is "ok" for a well-formed artifact whose sources match the manifest', async () => {
    const { manifest: manifestForBody } = manifestServing(validArtifact);
    const result = await loadRepertoireNull(manifestForBody, '/data');
    expect(result.status).toBe('ok');
    if (result.status === 'ok') {
      expect(result.data.primary.category).toBe('narrower');
      expect(result.data.robustness.perSeed[1730]).toBe('narrower');
    }
  });

  it('is "invalid" for the wrong version', async () => {
    const { manifest: manifestForBody } = manifestServing({ ...validArtifact, version: 2 });
    const result = await loadRepertoireNull(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/version/i);
  });

  it('is "invalid" when primary.category is an unrecognized value', async () => {
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      primary: { ...validArtifact.primary, category: 'bogus' }
    });
    const result = await loadRepertoireNull(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/"primary"/);
  });

  it('is "invalid" when a rewired distribution has an empty values array (never renders "range undefined–undefined")', async () => {
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      primary: {
        ...validArtifact.primary,
        rewiredDistribution: {
          ...validArtifact.primary.rewiredDistribution,
          occupied: { n: 0, p25: 0, p50: 0, p75: 0, values: [] }
        }
      }
    });
    const result = await loadRepertoireNull(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/malformed "primary"/);
  });

  it('is "invalid" when a rewired distribution\'s values are unsorted', async () => {
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      primary: {
        ...validArtifact.primary,
        rewiredDistribution: {
          ...validArtifact.primary.rewiredDistribution,
          occupied: { ...distribution(OCCUPIED_20), values: [...OCCUPIED_20].reverse() }
        }
      }
    });
    const result = await loadRepertoireNull(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/malformed "primary"/);
  });

  it('is "invalid" when the primary distribution size disagrees with search.rewiredCount', async () => {
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      search: { ...validArtifact.search, rewiredCount: 19 }
    });
    const result = await loadRepertoireNull(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/disagrees with search\.rewiredCount/);
  });

  it('is "invalid" when primary.category disagrees with robustness.perSeed at the primary seed', async () => {
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      primary: { ...validArtifact.primary, category: 'typical' } // robustness.perSeed[1729] is "narrower"
    });
    const result = await loadRepertoireNull(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/primary\.category disagrees with robustness\.perSeed/);
  });

  it('is "invalid" when robustness.perSeed is missing a required seed', async () => {
    const { '1733': _omit, ...perSeedWithoutSeed1733 } = validArtifact.robustness.perSeed;
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      robustness: { ...validArtifact.robustness, perSeed: perSeedWithoutSeed1733 }
    });
    const result = await loadRepertoireNull(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/robustness entry for seed 1733/);
  });

  // Cross-field consistency (mirrors `pathwayInterventions.ts`'s own
  // `trainedRobust`-vs-`perSeedCategory` check): a hash-valid artifact can
  // still ship a `robustness.robust` that disagrees with its own per-seed
  // categories.
  it('is "invalid" when robustness.robust disagrees with its own per-seed categories', async () => {
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      robustness: {
        ...validArtifact.robustness,
        robust: true // real per-seed categories disagree (narrower/typical mixed)
      }
    });
    const result = await loadRepertoireNull(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/robustness\.robust disagrees/);
  });

  it('is "invalid" when sources.biologicalSha does not match the manifest\'s compiled biological graph (a stale/re-pinned artifact)', async () => {
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      sources: { ...validArtifact.sources, biologicalSha: 'f'.repeat(64) }
    });
    const result = await loadRepertoireNull(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/sources\.biologicalSha does not match/);
  });

  it('is "invalid" when sources.rewiringNullSha does not match the manifest\'s rewiring-null artifact (a stale/re-pinned artifact)', async () => {
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      sources: { ...validArtifact.sources, rewiringNullSha: 'f'.repeat(64) }
    });
    const result = await loadRepertoireNull(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/sources\.rewiringNullSha does not match/);
  });

  it('is "invalid" when the manifest itself has no rewiringNull.sha256 to cross-check against', async () => {
    const raw = JSON.stringify(validArtifact);
    const hash = sha256Hex(raw);
    vi.stubGlobal('fetch', async () => new Response(raw, { status: 200, headers: { 'content-type': 'application/json' } }));
    const manifestForBody: ArenaManifest = {
      ...manifest,
      rewiringNull: undefined,
      behaviorRepertoireNull: { artifact: 'behavior-repertoire-null-v1.json', sha256: hash }
    };
    const result = await loadRepertoireNull(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/missing rewiringNull\.sha256/);
  });

  it('is "invalid" when the sha256-matching bytes are not valid JSON', async () => {
    const raw = 'not json';
    const hash = sha256Hex(raw);
    vi.stubGlobal('fetch', async () => new Response(raw, { status: 200, headers: { 'content-type': 'application/json' } }));
    const manifestForBody: ArenaManifest = {
      ...manifest,
      behaviorRepertoireNull: { artifact: 'behavior-repertoire-null-v1.json', sha256: hash }
    };
    const result = await loadRepertoireNull(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/not valid JSON/i);
  });
});
