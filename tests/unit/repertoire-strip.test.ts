import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it, vi } from 'vitest';
import type { ArenaManifest } from '../../src/lib/experiment/assets';
import { loadRepertoireNull, type RepertoireNullArtifact } from '../../src/lib/experiment/repertoireNull';
import { buildRepertoireStripText, isRepertoireStale, metricVerdictLabel } from '../../src/lib/atlas/repertoireStrip';
import { createPublicDataFetch } from '../helpers/fake-worker';

/**
 * Thermo-methodology review (Important, I-2): `Atlas.svelte`'s
 * `repertoireStale` cross-check and its `repertoireStripText` template had
 * zero test coverage at any level -- `Atlas.svelte` is never imported by
 * any unit/component test in this repo (only `Shell.svelte`'s lazy
 * `import()` references it), so a regression here (an inverted comparison,
 * a typo'd field name, or the whole check being dropped in a future
 * refactor) would previously have gone uncaught by `vitest run tests/unit`.
 *
 * Both functions under test were extracted out of the component
 * (`src/lib/atlas/repertoireStrip.ts`) specifically so this suite can
 * exercise them directly, in milliseconds, with no DOM/mount required --
 * `tests/e2e/atlas-repertoire.spec.ts` separately covers the full
 * render-time behavior (including the stale-atlas render path) end to end.
 */

const here = dirname(fileURLToPath(import.meta.url));
const publicDataDir = resolve(here, '../../public/data');
const manifest = JSON.parse(readFileSync(resolve(publicDataDir, 'malecns-arena-v1.manifest.json'), 'utf-8')) as ArenaManifest;

let realArtifact: RepertoireNullArtifact;

describe('repertoireStrip (against the real committed WP3 artifact)', () => {
  it('builds the real strip text, stating both occupied and qd next to the category', async () => {
    vi.stubGlobal('fetch', createPublicDataFetch());
    const result = await loadRepertoireNull(manifest, '/data');
    vi.unstubAllGlobals();
    expect(result.status).toBe('ok');
    if (result.status !== 'ok') return;
    realArtifact = result.data;

    const text = buildRepertoireStripText(realArtifact);
    // The real shipped numbers (independently verified,
    // `.agents/plans/repertoire-null/00-overview.md`'s worked example).
    expect(text).toContain('Repertoire vs 20 rewirings: biological occupies 28 of 36 cells');
    expect(text).toContain('rewired median 29, range 26–32');
    expect(text).toContain('qd 637 vs rewired median 763');
    // narrower on both metrics: occupied is 28 (<= p25=28, < p75=29) and qd
    // is 637.49 (<= p25=728.15, < p75=828.44) -- both independently narrower.
    expect(text).toContain('narrower on both metrics at search seed 1729');
    expect(text).toContain('not robust: seeds give 1729: narrower, 1730: narrower, 1731: typical, 1732: typical, 1733: typical');
    expect(text).toContain('Seeds 1730–1733 compare against only 5 seed-matched rewirings');
    expect(text).not.toContain('(tie)');
  });

  it('isRepertoireStale is false when the recorded atlas sha matches, true otherwise', () => {
    expect(isRepertoireStale(realArtifact.sources.atlasSha256, realArtifact.sources.atlasSha256)).toBe(false);
    expect(isRepertoireStale(realArtifact.sources.atlasSha256, 'f'.repeat(64))).toBe(true);
  });
});

describe('metricVerdictLabel', () => {
  it('is "wider" only when bio is at or above p75 and strictly above p25', () => {
    expect(metricVerdictLabel(30, { p25: 20, p75: 30 })).toBe('wider');
    expect(metricVerdictLabel(31, { p25: 20, p75: 30 })).toBe('wider');
  });

  it('is "narrower" only when bio is at or below p25 and strictly below p75', () => {
    expect(metricVerdictLabel(20, { p25: 20, p75: 30 })).toBe('narrower');
    expect(metricVerdictLabel(19, { p25: 20, p75: 30 })).toBe('narrower');
  });

  it('is "typical" otherwise, including the tie rule\'s degenerate p25 === p75 === bio case', () => {
    expect(metricVerdictLabel(25, { p25: 20, p75: 30 })).toBe('typical');
    expect(metricVerdictLabel(20, { p25: 20, p75: 20 })).toBe('typical'); // degenerate: neither wider nor narrower can hold
  });
});

describe('buildRepertoireStripText (synthetic fixtures)', () => {
  const distribution = (p25: number, p50: number, p75: number, values: readonly number[]) => ({ n: values.length, p25, p50, p75, values });

  const fixture = (overrides: Partial<RepertoireNullArtifact> = {}): RepertoireNullArtifact => ({
    version: 1,
    sources: { biologicalSha: 'a'.repeat(64), rewiringNullSha: 'b'.repeat(64), atlasSha256: 'c'.repeat(64) },
    search: { population: 64, generations: 24, ticks: 900, primarySearchSeed: 1729, extraSearchSeeds: [], rewiredCount: 5, rewiredSeedMatchedCount: 5 },
    primary: {
      bio: { occupied: 25, qd: 500, span: 10, heldoutOwnMedian: 10 },
      rewiredDistribution: {
        occupied: distribution(20, 25, 30, [18, 20, 25, 30, 32]),
        qd: distribution(400, 500, 600, [350, 400, 500, 600, 650])
      },
      category: 'typical',
      tie: false
    },
    robustness: { perSeed: { 1729: 'typical' }, robust: true, seedMatchedCounts: { 1729: 5 } },
    disconnected: { occupied: 10, qd: 100 },
    ...overrides
  });

  it('omits the "Seeds ... compare against" clause when there are no extra search seeds', () => {
    const text = buildRepertoireStripText(fixture());
    expect(text).not.toContain('Seeds');
    expect(text).not.toContain('compare against');
  });

  it('shows "robust across all N search seeds" when robust', () => {
    const text = buildRepertoireStripText(fixture());
    expect(text).toContain('robust across all 1 search seeds');
  });

  it('states each metric\'s own verdict when the category is "typical" (the two metrics can disagree)', () => {
    // occupied=25 sits strictly between p25=20 and p75=30 -> typical.
    // qd=500 sits strictly between p25=400 and p75=600 -> typical.
    const text = buildRepertoireStripText(fixture());
    expect(text).toContain('occupied typical, qd typical');
    expect(text).not.toContain('on both metrics');
  });

  it('states each metric\'s own verdict, even when they disagree, for a "typical" category', () => {
    const disagreeing = fixture({
      primary: {
        bio: { occupied: 32, qd: 500, span: 10, heldoutOwnMedian: 10 }, // occupied=32 is wider (>= p75=30, > p25=20)
        rewiredDistribution: {
          occupied: distribution(20, 25, 30, [18, 20, 25, 30, 32]),
          qd: distribution(400, 500, 600, [350, 400, 500, 600, 650]) // qd=500 is typical
        },
        category: 'typical', // categorize() requires BOTH metrics to agree for wider/narrower; here only one does
        tie: false
      }
    });
    const text = buildRepertoireStripText(disagreeing);
    expect(text).toContain('occupied wider, qd typical');
  });

  it('appends "(tie)" when primary.tie is true', () => {
    const text = buildRepertoireStripText(fixture({ primary: { ...fixture().primary, tie: true } }));
    expect(text).toContain('typical (tie) occupied typical, qd typical at search seed 1729');
  });

  it('derives the range from the sorted min/max of the occupied distribution\'s values', () => {
    const text = buildRepertoireStripText(fixture());
    expect(text).toContain('range 18–32');
  });
});
