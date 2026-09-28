import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { ArenaManifest } from '../../src/lib/experiment/assets';
import { loadTaskGenerality } from '../../src/lib/experiment/taskGenerality';
import { createPublicDataFetch } from '../helpers/fake-worker';

/**
 * WP4 of `.agents/plans/task-generality` unit coverage for
 * `taskGenerality.ts#loadTaskGenerality`. Mirrors
 * `tests/unit/assets-repertoire-null.test.ts`'s own structure: a first
 * describe block against the real, committed `public/data/task-generality-v1.json`
 * and manifest entry, and a second against a synthetic manifest/artifact
 * pair built to exercise the shape validator and cross-checks directly.
 */

const here = dirname(fileURLToPath(import.meta.url));
const publicDataDir = resolve(here, '../../public/data');

const manifest = JSON.parse(readFileSync(resolve(publicDataDir, 'malecns-arena-v1.manifest.json'), 'utf-8')) as ArenaManifest;

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('loadTaskGenerality (against the real committed WP4 artifact)', () => {
  it('loads and hash-verifies the real task-generality-v1.json, returning its parsed contents', async () => {
    vi.stubGlobal('fetch', createPublicDataFetch());
    const result = await loadTaskGenerality(manifest, '/data');
    expect(result.status).toBe('ok');
    if (result.status !== 'ok') return;
    expect(result.data.version).toBe(1);
    expect(result.data.tasks).toHaveLength(4);
    // The real shipped result: authored generalizes overall (3 of 4
    // non-degenerate tasks hold/generalize); trained does not (no task
    // reaches a robust pathway-supported category).
    expect(result.data.overall.authored.verdict).toBe('general');
    expect(result.data.overall.trained.verdict).toBe('task-dependent');
    const noMovement = result.data.tasks.find((t) => t.id === 'no-movement');
    expect(noMovement?.categorized).toBe(false);
    expect(noMovement?.pathway.category).toBe('degenerate');
    expect(result.data.sources.rewiringNullSha).toBe(manifest.rewiringNull?.sha256);
    expect(result.data.sources.pathwayInterventionsSha).toBe(manifest.pathwayInterventions?.sha256);
  });

  it('is "missing" with an honest reason when the manifest has no taskGenerality entry', async () => {
    vi.stubGlobal('fetch', createPublicDataFetch());
    const manifestWithoutEntry: ArenaManifest = { ...manifest, taskGenerality: undefined };
    const result = await loadTaskGenerality(manifestWithoutEntry, '/data');
    expect(result.status).toBe('missing');
  });

  it('is "unavailable" (never throws) when the artifact 404s', async () => {
    vi.stubGlobal('fetch', async () => new Response(null, { status: 404, statusText: 'Not Found' }));
    const result = await loadTaskGenerality(manifest, '/data');
    expect(result.status).toBe('unavailable');
  });

  it('is "invalid" with a sha256 reason when task-generality-v1.json is tampered', async () => {
    vi.stubGlobal('fetch', createPublicDataFetch({ corrupt: 'task-generality-v1.json' }));
    const result = await loadTaskGenerality(manifest, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/sha256/i);
  });
});

describe('loadTaskGenerality (shape validation, with a synthetic manifest sha256 that matches the served bytes)', () => {
  const sha256Hex = (data: string): string => createHash('sha256').update(data).digest('hex');

  const generalizingTask = (id: string) => ({
    id,
    categorized: true,
    null: { nullHolds: true, degenerate: false },
    pathway: { category: 'pathway-supported', generalizes: true },
    trained: {
      degenerate: false,
      trainedRobust: true,
      category: 'pathway-supported',
      perSeed: { '101': 'pathway-supported', '202': 'pathway-supported', '303': 'pathway-supported' }
    }
  });

  const validArtifact = {
    version: 1,
    sources: {
      rewiringNullSha: manifest.rewiringNull?.sha256 ?? 'a'.repeat(64),
      pathwayInterventionsSha: manifest.pathwayInterventions?.sha256 ?? 'b'.repeat(64)
    },
    tasks: ['hazard-heavy', 'sparse-food', 'no-movement', 'crowded'].map(generalizingTask),
    overall: {
      authored: { verdict: 'general', nonDegenerateCount: 4, totalCount: 4 },
      trained: { verdict: 'general', nonDegenerateCount: 4, totalCount: 4 }
    }
  };

  const manifestServing = (body: unknown): { manifest: ArenaManifest } => {
    const raw = JSON.stringify(body);
    const hash = sha256Hex(raw);
    vi.stubGlobal('fetch', async () => new Response(raw, { status: 200, headers: { 'content-type': 'application/json' } }));
    return { manifest: { ...manifest, taskGenerality: { artifact: 'task-generality-v1.json', sha256: hash } } };
  };

  it('is "ok" for a well-formed artifact whose sources match the manifest', async () => {
    const { manifest: manifestForBody } = manifestServing(validArtifact);
    const result = await loadTaskGenerality(manifestForBody, '/data');
    expect(result.status).toBe('ok');
    if (result.status === 'ok') {
      expect(result.data.overall.authored.verdict).toBe('general');
      expect(result.data.tasks).toHaveLength(4);
    }
  });

  it('is "invalid" for the wrong version', async () => {
    const { manifest: manifestForBody } = manifestServing({ ...validArtifact, version: 2 });
    const result = await loadTaskGenerality(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/version/i);
  });

  it('is "invalid" when a task has an unrecognized pathway category', async () => {
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      tasks: [{ ...validArtifact.tasks[0], pathway: { category: 'bogus', generalizes: true } }, ...validArtifact.tasks.slice(1)]
    });
    const result = await loadTaskGenerality(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/malformed "tasks"/);
  });

  it('is "invalid" when a non-degenerate trained task is missing a required trainer seed', async () => {
    const { '303': _omit, ...perSeedWithoutSeed303 } = validArtifact.tasks[0].trained.perSeed;
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      tasks: [
        { ...validArtifact.tasks[0], trained: { ...validArtifact.tasks[0].trained, perSeed: perSeedWithoutSeed303 } },
        ...validArtifact.tasks.slice(1)
      ]
    });
    const result = await loadTaskGenerality(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/malformed "tasks"/);
  });

  it('is "invalid" when overall.authored disagrees with its own per-task data (a hash-valid but internally inconsistent artifact)', async () => {
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      overall: { ...validArtifact.overall, authored: { verdict: 'task-dependent', nonDegenerateCount: 4, totalCount: 4 } }
    });
    const result = await loadTaskGenerality(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/overall\.authored disagrees/);
  });

  it('is "invalid" when overall.trained disagrees with its own per-task data', async () => {
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      overall: { ...validArtifact.overall, trained: { verdict: 'task-dependent', nonDegenerateCount: 4, totalCount: 4 } }
    });
    const result = await loadTaskGenerality(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/overall\.trained disagrees/);
  });

  it('is "invalid" when sources.rewiringNullSha does not match the manifest\'s rewiring-null artifact (a stale/re-pinned artifact)', async () => {
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      sources: { ...validArtifact.sources, rewiringNullSha: 'f'.repeat(64) }
    });
    const result = await loadTaskGenerality(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/sources\.rewiringNullSha does not match/);
  });

  it('is "invalid" when sources.pathwayInterventionsSha does not match the manifest\'s pathway-interventions artifact (a stale/re-pinned artifact)', async () => {
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      sources: { ...validArtifact.sources, pathwayInterventionsSha: 'f'.repeat(64) }
    });
    const result = await loadTaskGenerality(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/sources\.pathwayInterventionsSha does not match/);
  });

  it('is "invalid" when the manifest itself has no pathwayInterventions.sha256 to cross-check against', async () => {
    const raw = JSON.stringify(validArtifact);
    const hash = sha256Hex(raw);
    vi.stubGlobal('fetch', async () => new Response(raw, { status: 200, headers: { 'content-type': 'application/json' } }));
    const manifestForBody: ArenaManifest = {
      ...manifest,
      pathwayInterventions: undefined,
      taskGenerality: { artifact: 'task-generality-v1.json', sha256: hash }
    };
    const result = await loadTaskGenerality(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/missing pathwayInterventions\.sha256/);
  });

  it('is "invalid" when the sha256-matching bytes are not valid JSON', async () => {
    const raw = 'not json';
    const hash = sha256Hex(raw);
    vi.stubGlobal('fetch', async () => new Response(raw, { status: 200, headers: { 'content-type': 'application/json' } }));
    const manifestForBody: ArenaManifest = { ...manifest, taskGenerality: { artifact: 'task-generality-v1.json', sha256: hash } };
    const result = await loadTaskGenerality(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/not valid JSON/i);
  });
});
