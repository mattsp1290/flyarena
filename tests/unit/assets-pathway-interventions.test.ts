import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { ArenaManifest } from '../../src/lib/experiment/assets';
import { loadPathwayInterventions } from '../../src/lib/experiment/pathwayInterventions';
import { createPublicDataFetch } from '../helpers/fake-worker';

/**
 * WP4 of `.agents/plans/pathway-interventions` unit coverage for
 * `pathwayInterventions.ts#loadPathwayInterventions`. Mirrors
 * `tests/unit/assets-null-explanation.test.ts`'s own structure: a first
 * describe block against the real, committed `public/data/pathway-interventions-v1.json`
 * and manifest entry, and a second against a synthetic manifest/artifact
 * pair built to exercise the shape validator and cross-checks directly.
 */

const here = dirname(fileURLToPath(import.meta.url));
const publicDataDir = resolve(here, '../../public/data');

const manifest = JSON.parse(
  readFileSync(resolve(publicDataDir, 'malecns-arena-v1.manifest.json'), 'utf-8')
) as ArenaManifest;

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('loadPathwayInterventions (against the real committed WP4 artifact)', () => {
  it('loads and hash-verifies the real pathway-interventions-v1.json, returning its parsed contents', async () => {
    vi.stubGlobal('fetch', createPublicDataFetch());
    const result = await loadPathwayInterventions(manifest, '/data');
    expect(result.status).toBe('ok');
    if (result.status !== 'ok') return;
    expect(result.data.version).toBe(1);
    expect(result.data.authored.category).toBe('pathway-supported');
    expect(result.data.authored.channelSpecific).toBe(true);
    expect(result.data.trained.trainedRobust).toBe(true);
    expect(result.data.trained.perSeedCategory['101']).toBe('no-specific-effect');
    expect(result.data.sources.rewiringNullSha).toBe(manifest.rewiringNull?.sha256);
    expect(result.data.sources.nullExplanationSha).toBe(manifest.nullExplanation?.sha256);
  });

  it('is "missing" with an honest reason when the manifest has no pathwayInterventions entry', async () => {
    vi.stubGlobal('fetch', createPublicDataFetch());
    const manifestWithoutEntry: ArenaManifest = { ...manifest, pathwayInterventions: undefined };
    const result = await loadPathwayInterventions(manifestWithoutEntry, '/data');
    expect(result.status).toBe('missing');
  });

  it('is "unavailable" (never throws) when the artifact 404s', async () => {
    vi.stubGlobal('fetch', async () => new Response(null, { status: 404, statusText: 'Not Found' }));
    const result = await loadPathwayInterventions(manifest, '/data');
    expect(result.status).toBe('unavailable');
  });

  it('is "invalid" with a sha256 reason when pathway-interventions-v1.json is tampered', async () => {
    vi.stubGlobal('fetch', createPublicDataFetch({ corrupt: 'pathway-interventions-v1.json' }));
    const result = await loadPathwayInterventions(manifest, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/sha256/i);
  });
});

describe('loadPathwayInterventions (shape validation, with a synthetic manifest sha256 that matches the served bytes)', () => {
  const sha256Hex = (data: string): string => createHash('sha256').update(data).digest('hex');

  const validArtifact = {
    version: 1,
    sources: {
      rewiringNullSha: manifest.rewiringNull?.sha256 ?? 'a'.repeat(64),
      nullExplanationSha: manifest.nullExplanation?.sha256 ?? 'b'.repeat(64)
    },
    authored: { category: 'pathway-supported', channelSpecific: true },
    trained: {
      trainedRobust: true,
      perSeed: {
        '101': { category: 'no-specific-effect' },
        '202': { category: 'no-specific-effect' },
        '303': { category: 'no-specific-effect' }
      }
    }
  };

  const manifestServing = (body: unknown): { manifest: ArenaManifest } => {
    const raw = JSON.stringify(body);
    const hash = sha256Hex(raw);
    vi.stubGlobal('fetch', async () => new Response(raw, { status: 200, headers: { 'content-type': 'application/json' } }));
    return { manifest: { ...manifest, pathwayInterventions: { artifact: 'pathway-interventions-v1.json', sha256: hash } } };
  };

  it('is "ok" for a well-formed artifact whose sources match the manifest', async () => {
    const { manifest: manifestForBody } = manifestServing(validArtifact);
    const result = await loadPathwayInterventions(manifestForBody, '/data');
    expect(result.status).toBe('ok');
  });

  it('is "invalid" for the wrong version', async () => {
    const { manifest: manifestForBody } = manifestServing({ ...validArtifact, version: 2 });
    const result = await loadPathwayInterventions(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/version/i);
  });

  it('is "invalid" when authored.category is an unrecognized value', async () => {
    const { manifest: manifestForBody } = manifestServing({ ...validArtifact, authored: { category: 'bogus', channelSpecific: true } });
    const result = await loadPathwayInterventions(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/"authored"/);
  });

  it('is "invalid" when trained.perSeed is missing a required seed', async () => {
    const { '303': _omit, ...perSeedWithoutSeed303 } = validArtifact.trained.perSeed;
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      trained: { ...validArtifact.trained, perSeed: perSeedWithoutSeed303 }
    });
    const result = await loadPathwayInterventions(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/perSeed\["303"\]\.category/);
  });

  // Cross-field consistency (mirrors `rewiringNull.ts`'s recomputed-rank-
  // statistics precedent): a hash-valid artifact can still ship a
  // `trainedRobust` that disagrees with its own per-seed categories.
  it('is "invalid" when trainedRobust disagrees with its own per-seed categories', async () => {
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      trained: {
        trainedRobust: true,
        perSeed: {
          '101': { category: 'no-specific-effect' },
          '202': { category: 'pathway-supported' },
          '303': { category: 'no-specific-effect' }
        }
      }
    });
    const result = await loadPathwayInterventions(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/trainedRobust disagrees/);
  });

  it('is "invalid" when sources.rewiringNullSha does not match the manifest\'s rewiring-null artifact (a stale/re-pinned artifact)', async () => {
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      sources: { ...validArtifact.sources, rewiringNullSha: 'f'.repeat(64) }
    });
    const result = await loadPathwayInterventions(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/sources\.rewiringNullSha does not match/);
  });

  it('is "invalid" when sources.nullExplanationSha does not match the manifest\'s null-explanation artifact (a stale/re-pinned artifact)', async () => {
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      sources: { ...validArtifact.sources, nullExplanationSha: 'f'.repeat(64) }
    });
    const result = await loadPathwayInterventions(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/sources\.nullExplanationSha does not match/);
  });

  it('is "invalid" when the manifest itself has no rewiringNull.sha256 to cross-check against', async () => {
    const raw = JSON.stringify(validArtifact);
    const hash = sha256Hex(raw);
    vi.stubGlobal('fetch', async () => new Response(raw, { status: 200, headers: { 'content-type': 'application/json' } }));
    const manifestForBody: ArenaManifest = {
      ...manifest,
      rewiringNull: undefined,
      pathwayInterventions: { artifact: 'pathway-interventions-v1.json', sha256: hash }
    };
    const result = await loadPathwayInterventions(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/missing rewiringNull\.sha256/);
  });

  it('is "invalid" when the manifest itself has no nullExplanation.sha256 to cross-check against', async () => {
    const raw = JSON.stringify(validArtifact);
    const hash = sha256Hex(raw);
    vi.stubGlobal('fetch', async () => new Response(raw, { status: 200, headers: { 'content-type': 'application/json' } }));
    const manifestForBody: ArenaManifest = {
      ...manifest,
      nullExplanation: undefined,
      pathwayInterventions: { artifact: 'pathway-interventions-v1.json', sha256: hash }
    };
    const result = await loadPathwayInterventions(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/missing nullExplanation\.sha256/);
  });

  it('is "invalid" when the sha256-matching bytes are not valid JSON', async () => {
    const raw = 'not json';
    const hash = sha256Hex(raw);
    vi.stubGlobal('fetch', async () => new Response(raw, { status: 200, headers: { 'content-type': 'application/json' } }));
    const manifestForBody: ArenaManifest = { ...manifest, pathwayInterventions: { artifact: 'pathway-interventions-v1.json', sha256: hash } };
    const result = await loadPathwayInterventions(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/not valid JSON/i);
  });
});
