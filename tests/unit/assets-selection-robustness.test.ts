import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { ArenaManifest } from '../../src/lib/experiment/assets';
import { loadSelectionRobustness, SELECTION_IDS, type SelectionId } from '../../src/lib/experiment/selectionRobustness';
import { createPublicDataFetch } from '../helpers/fake-worker';

/**
 * WP3 of `.agents/plans/selection-robustness` unit coverage for
 * `selectionRobustness.ts#loadSelectionRobustness`. Mirrors
 * `tests/unit/assets-task-generality.test.ts`'s own structure: a first
 * describe block against the real, committed `public/data/selection-robustness-v1.json`
 * and manifest entry, and a second against a synthetic manifest/artifact
 * pair built to exercise the shape validator and cross-checks directly.
 */

const here = dirname(fileURLToPath(import.meta.url));
const publicDataDir = resolve(here, '../../public/data');

const manifest = JSON.parse(readFileSync(resolve(publicDataDir, 'malecns-arena-v1.manifest.json'), 'utf-8')) as ArenaManifest;

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('loadSelectionRobustness (against the real committed WP3 artifact)', () => {
  it('loads and hash-verifies the real selection-robustness-v1.json, returning its parsed contents', async () => {
    vi.stubGlobal('fetch', createPublicDataFetch());
    const result = await loadSelectionRobustness(manifest, '/data');
    expect(result.status).toBe('ok');
    if (result.status !== 'ok') return;
    expect(result.data.version).toBe(1);
    expect(result.data.selections).toHaveLength(4);
    // The real shipped result: robust to size (smaller + larger), not
    // robust to method (random-bridge's explanation does not replicate),
    // and the channel-mapping result fails (alt-sensory-mapping's pathway
    // is not-supported) -- see `docs/selection-robustness-report.md`.
    expect(result.data.overall.robustToSize.verdict).toBe(true);
    expect(result.data.overall.robustToMethod.verdict).toBe(false);
    expect(result.data.overall.mapping.verdict).toBe(false);
    const randomBridge = result.data.selections.find((s) => s.id === 'random-bridge');
    expect(randomBridge?.categorized).toBe(false);
    expect(randomBridge?.pathway.cDegenerate).toBe(true);
    expect(randomBridge?.null.holds).toBe(true);
    const altSensoryMapping = result.data.selections.find((s) => s.id === 'alt-sensory-mapping');
    expect(altSensoryMapping?.pathway.supported).toBe(false);
    expect(result.data.sources.defaultGraphSha).toBe(manifest.binarySha256);
  });

  it('is "missing" with an honest reason when the manifest has no selectionRobustness entry', async () => {
    vi.stubGlobal('fetch', createPublicDataFetch());
    const manifestWithoutEntry: ArenaManifest = { ...manifest, selectionRobustness: undefined };
    const result = await loadSelectionRobustness(manifestWithoutEntry, '/data');
    expect(result.status).toBe('missing');
  });

  it('is "unavailable" (never throws) when the artifact 404s', async () => {
    vi.stubGlobal('fetch', async () => new Response(null, { status: 404, statusText: 'Not Found' }));
    const result = await loadSelectionRobustness(manifest, '/data');
    expect(result.status).toBe('unavailable');
  });

  it('is "invalid" with a sha256 reason when selection-robustness-v1.json is tampered', async () => {
    vi.stubGlobal('fetch', createPublicDataFetch({ corrupt: 'selection-robustness-v1.json' }));
    const result = await loadSelectionRobustness(manifest, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/sha256/i);
  });
});

describe('loadSelectionRobustness (shape validation, with a synthetic manifest sha256 that matches the served bytes)', () => {
  const sha256Hex = (data: string): string => createHash('sha256').update(data).digest('hex');

  const baseSelection = (id: SelectionId) => ({
    id,
    params: {},
    counts: {},
    globalGain: 0.0001,
    graphSha: 'a'.repeat(64),
    compiledFromGitRevision: 'b'.repeat(40),
    compilerSourceSha256: 'c'.repeat(64),
    null: { bioScore: -1, bioPercentile: 0, p25: 1, degenerate: false, holds: true },
    explanation: {
      passing: [{ name: 'T:rightClearance->thrust', spearman: 0.5, bioPercentile: 0.01 }],
      replicates: true,
      structuralReplicates: false,
      mirroredBioPercentile: 0,
      singleAxis: null,
      exploratory: null,
      exploratoryOmittedReason: 'selection-mode has no exploratory-unrestricted counterpart'
    },
    pathway: {
      pScore: 5,
      cP95: 1,
      mP95: 1,
      cDegenerate: false,
      mDegenerate: false,
      k: 2,
      targetReached: true,
      bridgePoolSize: 2557,
      maxSwaps: 200,
      searchLimited: false,
      supported: true,
      qScore: 5,
      qK: 2,
      qTargetReached: true,
      mqDegenerate: false,
      channelSpecific: true
    },
    coverage: { perChannel: [], perPopulation: [], flagged: false },
    categorized: true
  });

  const validArtifact = {
    version: 1,
    sources: {
      defaultGraphSha: manifest.binarySha256,
      rawFileShas: ['d'.repeat(64), 'e'.repeat(64), 'f'.repeat(64)],
      compilerSourceSha: 'g'.repeat(64),
      producer: { script: 'scripts/selections/selection-report.ts', sourceSha256: 'h'.repeat(64), dependencies: [] }
    },
    selections: SELECTION_IDS.map(baseSelection),
    overall: {
      robustToSize: { verdict: true },
      robustToMethod: { verdict: true },
      mapping: { verdict: true }
    },
    host: { arch: 'arm64', node: 'v22.22.3' }
  };

  const manifestServing = (body: unknown): { manifest: ArenaManifest } => {
    const raw = JSON.stringify(body);
    const hash = sha256Hex(raw);
    vi.stubGlobal('fetch', async () => new Response(raw, { status: 200, headers: { 'content-type': 'application/json' } }));
    return { manifest: { ...manifest, selectionRobustness: { artifact: 'selection-robustness-v1.json', sha256: hash } } };
  };

  it('is "ok" for a well-formed artifact whose sources match the manifest', async () => {
    const { manifest: manifestForBody } = manifestServing(validArtifact);
    const result = await loadSelectionRobustness(manifestForBody, '/data');
    expect(result.status).toBe('ok');
    if (result.status === 'ok') {
      expect(result.data.overall.robustToSize.verdict).toBe(true);
      expect(result.data.selections).toHaveLength(4);
    }
  });

  it('accepts a categorized, search-limited selection\'s "indeterminate" overall verdict -- never requires it to read as a "not-supported" failure', async () => {
    // A thermo review of this WP3 change (methodology I1) caught that both
    // `aggregateVerdict` (producer) and this loader's own `recomputeVerdict`
    // folded a search-limited, categorized pathway into a `false` "not
    // supported" failure at the aggregate level -- contradicting the
    // per-selection label's own predeclared rule. This is the loader-side
    // regression test for that fix: a shipped artifact whose `larger`
    // selection is categorized, `searchLimited: true`, and
    // `pathway.supported: false` (the search simply never finished) must be
    // accepted with `robustToSize: 'indeterminate'`, not rejected as
    // internally inconsistent, and must never be forced to `false`.
    const searchLimitedLarger = {
      ...baseSelection('larger'),
      pathway: {
        ...baseSelection('larger').pathway,
        supported: false,
        searchLimited: true,
        targetReached: false,
        k: 200
      }
    };
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      selections: [searchLimitedLarger, ...validArtifact.selections.filter((s) => s.id !== 'larger')],
      overall: {
        ...validArtifact.overall,
        robustToSize: { verdict: 'indeterminate', reason: 'larger: pathway is search-limited (the swap search reached its cap)' }
      }
    });
    const result = await loadSelectionRobustness(manifestForBody, '/data');
    expect(result.status).toBe('ok');
    if (result.status === 'ok') {
      expect(result.data.overall.robustToSize.verdict).toBe('indeterminate');
      expect(result.data.overall.robustToSize.reason).toContain('search-limited');
      expect(result.data.overall.robustToSize.reason).not.toContain('not-supported');
    }
  });

  it('is "invalid" when a categorized, search-limited selection\'s overall verdict is forced to "false" (the exact bug the fix above closes)', async () => {
    const searchLimitedLarger = {
      ...baseSelection('larger'),
      pathway: {
        ...baseSelection('larger').pathway,
        supported: false,
        searchLimited: true,
        targetReached: false,
        k: 200
      }
    };
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      selections: [searchLimitedLarger, ...validArtifact.selections.filter((s) => s.id !== 'larger')],
      overall: {
        ...validArtifact.overall,
        // The bug this fix closes: folding search-limited into a plain
        // "not-supported" failure would have produced this exact shape.
        robustToSize: { verdict: false, reason: 'larger: pathway is not-supported' }
      }
    });
    const result = await loadSelectionRobustness(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/overall\.robustToSize disagrees/);
  });

  it('is "invalid" for the wrong version', async () => {
    const { manifest: manifestForBody } = manifestServing({ ...validArtifact, version: 2 });
    const result = await loadSelectionRobustness(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/version/i);
  });

  it('is "invalid" when a selection is missing (fewer than the four predeclared ids)', async () => {
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      selections: validArtifact.selections.slice(0, 3)
    });
    const result = await loadSelectionRobustness(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/exactly 4 entries/);
  });

  it('is "invalid" when null.holds disagrees with bioScore < p25', async () => {
    const [first, ...rest] = validArtifact.selections;
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      selections: [{ ...first, null: { ...first.null, holds: false } }, ...rest]
    });
    const result = await loadSelectionRobustness(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/holds disagrees with bioScore < p25/);
  });

  it('is "invalid" when explanation.replicates disagrees with its own "passing" list', async () => {
    const [first, ...rest] = validArtifact.selections;
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      selections: [{ ...first, explanation: { ...first.explanation, replicates: false } }, ...rest]
    });
    const result = await loadSelectionRobustness(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/replicates disagrees with its own "passing" list/);
  });

  it('is "invalid" when a selection is cDegenerate/mDegenerate but has no degenerateMechanism string', async () => {
    const [first, ...rest] = validArtifact.selections;
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      selections: [{ ...first, pathway: { ...first.pathway, cDegenerate: true, mDegenerate: true } }, ...rest]
    });
    const result = await loadSelectionRobustness(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/has no "degenerateMechanism" string/);
  });

  it('is "invalid" when searchLimited disagrees with k/maxSwaps/targetReached', async () => {
    const [first, ...rest] = validArtifact.selections;
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      selections: [{ ...first, pathway: { ...first.pathway, searchLimited: true } }, ...rest]
    });
    const result = await loadSelectionRobustness(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/searchLimited disagrees with k\/maxSwaps\/targetReached/);
  });

  it('is "invalid" when categorized disagrees with null/pathway/coverage degeneracy', async () => {
    const [first, ...rest] = validArtifact.selections;
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      selections: [
        {
          ...first,
          categorized: true,
          pathway: { ...first.pathway, cDegenerate: true, mDegenerate: true, degenerateMechanism: 'k=0' }
        },
        ...rest
      ]
    });
    const result = await loadSelectionRobustness(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/categorized disagrees with null\/pathway\/coverage degeneracy/);
  });

  it('is "invalid" when overall.robustToSize disagrees with its own per-selection data (a hash-valid but internally inconsistent artifact)', async () => {
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      overall: { ...validArtifact.overall, robustToSize: { verdict: false, reason: 'bogus' } }
    });
    const result = await loadSelectionRobustness(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/overall\.robustToSize disagrees/);
  });

  it('is "invalid" when overall.robustToMethod disagrees with its own per-selection data', async () => {
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      overall: { ...validArtifact.overall, robustToMethod: { verdict: false, reason: 'bogus' } }
    });
    const result = await loadSelectionRobustness(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/overall\.robustToMethod disagrees/);
  });

  it('is "invalid" when overall.mapping disagrees with its own per-selection data', async () => {
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      overall: { ...validArtifact.overall, mapping: { verdict: false, reason: 'bogus' } }
    });
    const result = await loadSelectionRobustness(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/overall\.mapping disagrees/);
  });

  it('is "invalid" when sources.defaultGraphSha does not match the manifest\'s compiled biological graph (a stale/re-pinned artifact)', async () => {
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      sources: { ...validArtifact.sources, defaultGraphSha: 'f'.repeat(64) }
    });
    const result = await loadSelectionRobustness(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/sources\.defaultGraphSha does not match/);
  });

  it('is "invalid" when the sha256-matching bytes are not valid JSON', async () => {
    const raw = 'not json';
    const hash = sha256Hex(raw);
    vi.stubGlobal('fetch', async () => new Response(raw, { status: 200, headers: { 'content-type': 'application/json' } }));
    const manifestForBody: ArenaManifest = {
      ...manifest,
      selectionRobustness: { artifact: 'selection-robustness-v1.json', sha256: hash }
    };
    const result = await loadSelectionRobustness(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/not valid JSON/i);
  });
});
