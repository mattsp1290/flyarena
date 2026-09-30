import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { ArenaManifest } from '../../src/lib/experiment/assets';
import { loadReadoutAttribution } from '../../src/lib/experiment/readoutAttribution';
import { createPublicDataFetch } from '../helpers/fake-worker';

/**
 * WP3 of `.agents/plans/readout-attribution` unit coverage for
 * `readoutAttribution.ts#loadReadoutAttribution`. Mirrors
 * `tests/unit/assets-selection-robustness.test.ts`'s own structure: a
 * first describe block against the real, committed
 * `public/data/readout-attribution-v1.json` and manifest entry, and a
 * second against a synthetic manifest/artifact pair built to exercise the
 * shape validator and cross-checks directly.
 */

const here = dirname(fileURLToPath(import.meta.url));
const publicDataDir = resolve(here, '../../public/data');

const manifest = JSON.parse(readFileSync(resolve(publicDataDir, 'malecns-arena-v1.manifest.json'), 'utf-8')) as ArenaManifest;

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('loadReadoutAttribution (against the real committed WP3 artifact)', () => {
  it('loads and hash-verifies the real readout-attribution-v1.json, returning its parsed contents', async () => {
    vi.stubGlobal('fetch', createPublicDataFetch());
    const result = await loadReadoutAttribution(manifest, '/data');
    expect(result.status).toBe('ok');
    if (result.status !== 'ok') return;
    expect(result.data.version).toBe(1);
    // The real WP2 results (`docs/readout-attribution-report.md`): every
    // hypothesis came back inconclusive on this model's actual data.
    expect(result.data.hypotheses.H1.outcome).toBe('inconclusive');
    expect(result.data.hypotheses.H2.outcome).toBe('inconclusive');
    expect(result.data.hypotheses.H3.outcome).toBe('inconclusive');
    expect(result.data.hypotheses.hypothesisCount).toBe(3);
    expect(result.data.coverage.ids).toHaveLength(23);
    expect(result.data.coverage.perTaskIncluded).toBe(false);
    expect(result.data.sources.pathwayInterventionsSha).toBe(manifest.pathwayInterventions?.sha256);
    expect(result.data.sources.descendingTypesSha).toBe(manifest.descendingTypes?.sha256);
  });

  it('is "missing" with an honest reason when the manifest has no readoutAttribution entry', async () => {
    vi.stubGlobal('fetch', createPublicDataFetch());
    const manifestWithoutEntry: ArenaManifest = { ...manifest, readoutAttribution: undefined };
    const result = await loadReadoutAttribution(manifestWithoutEntry, '/data');
    expect(result.status).toBe('missing');
  });

  it('is "unavailable" (never throws) when the artifact 404s', async () => {
    vi.stubGlobal('fetch', async () => new Response(null, { status: 404, statusText: 'Not Found' }));
    const result = await loadReadoutAttribution(manifest, '/data');
    expect(result.status).toBe('unavailable');
  });

  it('is "invalid" with a sha256 reason when readout-attribution-v1.json is tampered', async () => {
    vi.stubGlobal('fetch', createPublicDataFetch({ corrupt: 'readout-attribution-v1.json' }));
    const result = await loadReadoutAttribution(manifest, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/sha256/i);
  });
});

describe('loadReadoutAttribution (shape validation, with a synthetic manifest sha256 that matches the served bytes)', () => {
  const sha256Hex = (data: string): string => createHash('sha256').update(data).digest('hex');

  const validArtifact = () => ({
    version: 1,
    sources: {
      archiveSha: 'a'.repeat(64),
      descendingTypesSha: manifest.descendingTypes?.sha256 ?? 'b'.repeat(64),
      trainedReadoutSha: 'PLACEHOLDER', // filled in per-test against the real trained-readout-v1.manifest.json
      pathwayInterventionsSha: manifest.pathwayInterventions?.sha256 ?? 'c'.repeat(64),
      producer: { script: 'scripts/attribution/attribution-report.ts', sourceSha256: 'd'.repeat(64), dependencies: [] }
    },
    coverage: { ids: ['biological-seed101'], perTaskIncluded: false },
    hypotheses: {
      hypothesisCount: 3,
      multipleComparisonCorrection: 'none',
      H1: { outcome: 'inconclusive', reason: 'threshold-not-met-in-all-seeds', evidence: {} },
      H2: { outcome: 'inconclusive', evidence: {} },
      H3: { outcome: 'inconclusive', evidence: {} }
    },
    host: { arch: 'arm64', node: 'v22.22.3' }
  });

  const realTrainedReadoutManifest = JSON.parse(
    readFileSync(resolve(publicDataDir, 'trained-readout-v1.manifest.json'), 'utf-8')
  ) as { artifactSha256: string };

  /**
   * Stubs `fetch` to serve `readout-attribution-v1.json` (the given `body`,
   * sha256-verified against the manifest entry this returns) and
   * `trained-readout-v1.manifest.json` (the real committed one, so
   * `sources.trainedReadoutSha` cross-checks against a real value) --
   * every other request falls through to the real `public/data/` fixture
   * fetch, mirroring how the real app would resolve both URLs against the
   * same `dataBaseUrl`.
   */
  const manifestServing = (body: unknown): { manifest: ArenaManifest } => {
    const raw = JSON.stringify(body);
    const hash = sha256Hex(raw);
    const publicDataFetch = createPublicDataFetch();
    vi.stubGlobal('fetch', async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.endsWith('/readout-attribution-v1.json')) {
        return new Response(raw, { status: 200, headers: { 'content-type': 'application/json' } });
      }
      return publicDataFetch(input);
    });
    return { manifest: { ...manifest, readoutAttribution: { artifact: 'readout-attribution-v1.json', sha256: hash } } };
  };

  it('is "ok" for a well-formed artifact whose sources match the manifest and the real trained-readout manifest', async () => {
    const artifact = { ...validArtifact(), sources: { ...validArtifact().sources, trainedReadoutSha: realTrainedReadoutManifest.artifactSha256 } };
    const { manifest: manifestForBody } = manifestServing(artifact);
    const result = await loadReadoutAttribution(manifestForBody, '/data');
    expect(result.status).toBe('ok');
    if (result.status === 'ok') {
      expect(result.data.hypotheses.H1.outcome).toBe('inconclusive');
      expect(result.data.coverage.ids).toEqual(['biological-seed101']);
    }
  });

  it('is "invalid" for the wrong version', async () => {
    const artifact = { ...validArtifact(), version: 2, sources: { ...validArtifact().sources, trainedReadoutSha: realTrainedReadoutManifest.artifactSha256 } };
    const { manifest: manifestForBody } = manifestServing(artifact);
    const result = await loadReadoutAttribution(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/version/i);
  });

  it('is "invalid" for an unrecognized hypothesis outcome', async () => {
    const base = validArtifact();
    const artifact = {
      ...base,
      sources: { ...base.sources, trainedReadoutSha: realTrainedReadoutManifest.artifactSha256 },
      hypotheses: { ...base.hypotheses, H1: { outcome: 'maybe', evidence: {} } }
    };
    const { manifest: manifestForBody } = manifestServing(artifact);
    const result = await loadReadoutAttribution(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/hypotheses\.H1/);
  });

  it('is "invalid" when sources.pathwayInterventionsSha does not match the manifest\'s pathwayInterventions artifact', async () => {
    const base = validArtifact();
    const artifact = {
      ...base,
      sources: { ...base.sources, trainedReadoutSha: realTrainedReadoutManifest.artifactSha256, pathwayInterventionsSha: 'f'.repeat(64) }
    };
    const { manifest: manifestForBody } = manifestServing(artifact);
    const result = await loadReadoutAttribution(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/sources\.pathwayInterventionsSha does not match/);
  });

  it('is "invalid" when sources.descendingTypesSha does not match the manifest\'s descendingTypes artifact', async () => {
    const base = validArtifact();
    const artifact = {
      ...base,
      sources: { ...base.sources, trainedReadoutSha: realTrainedReadoutManifest.artifactSha256, descendingTypesSha: 'f'.repeat(64) }
    };
    const { manifest: manifestForBody } = manifestServing(artifact);
    const result = await loadReadoutAttribution(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/sources\.descendingTypesSha does not match/);
  });

  it('is "invalid" when sources.trainedReadoutSha does not match trained-readout-v1.manifest.json\'s artifactSha256 (a stale/re-pinned artifact)', async () => {
    const base = validArtifact();
    const artifact = { ...base, sources: { ...base.sources, trainedReadoutSha: 'e'.repeat(64) } };
    const { manifest: manifestForBody } = manifestServing(artifact);
    const result = await loadReadoutAttribution(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/sources\.trainedReadoutSha does not match/);
  });

  it('is "invalid" when the sha256-matching bytes are not valid JSON', async () => {
    const raw = 'not json';
    const hash = sha256Hex(raw);
    vi.stubGlobal('fetch', async () => new Response(raw, { status: 200, headers: { 'content-type': 'application/json' } }));
    const manifestForBody: ArenaManifest = {
      ...manifest,
      readoutAttribution: { artifact: 'readout-attribution-v1.json', sha256: hash }
    };
    const result = await loadReadoutAttribution(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/not valid JSON/i);
  });
});
