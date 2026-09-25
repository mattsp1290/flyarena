import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { ArenaManifest } from '../../src/lib/experiment/assets';
import { loadRewiringNull } from '../../src/lib/experiment/rewiringNull';
import { createPublicDataFetch } from '../helpers/fake-worker';

/**
 * WP4 (`.agents/plans/rewiring-null/04-ledger-histogram.md`) unit coverage
 * for `rewiringNull.ts#loadRewiringNull` (extracted out of `assets.ts` by a
 * later thermo-nuclear review pass — see that file's own doc comment),
 * split into its own file (rather than folded into
 * `tests/unit/experiment-assets.test.ts`) since this WP's plan names it as
 * its own file (`tests/unit/assets-null.test.ts`). Exercises the real,
 * committed `public/data/rewiring-null-v1.json` and manifest entry —
 * `loadTrainedReadoutArtifact`'s existing tests in `experiment-assets.test.ts`
 * are this function's closest precedent and this file mirrors their
 * structure/naming.
 */

const here = dirname(fileURLToPath(import.meta.url));
const publicDataDir = resolve(here, '../../public/data');

const manifest = JSON.parse(
  readFileSync(resolve(publicDataDir, 'malecns-arena-v1.manifest.json'), 'utf-8')
) as ArenaManifest;

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('loadRewiringNull (against the real committed WP2 artifact)', () => {
  it('loads and hash-verifies the real rewiring-null-v1.json, returning its parsed contents', async () => {
    vi.stubGlobal('fetch', createPublicDataFetch());
    const result = await loadRewiringNull(manifest, '/data');
    expect(result.status).toBe('ok');
    if (result.status !== 'ok') return;
    expect(result.data.version).toBe(1);
    expect(result.data.rewired.length).toBeGreaterThan(0);
    expect(result.data.bins.counts.length).toBe(result.data.bins.edges.length - 1);
    expect(result.data.rewired.some((entry) => entry.seed === 0)).toBe(true);
  });

  it('is "absent" with an honest reason when the manifest has no rewiringNull entry', async () => {
    vi.stubGlobal('fetch', createPublicDataFetch());
    const manifestWithoutEntry: ArenaManifest = { ...manifest, rewiringNull: undefined };
    const result = await loadRewiringNull(manifestWithoutEntry, '/data');
    expect(result.status).toBe('absent');
  });

  // Thermo review, Suggestion: distinct from "absent" above — the manifest
  // *does* pin an entry, but fetching it failed, which is not the same
  // claim as "nothing was ever shipped".
  it('is "unavailable" (never throws) when the artifact 404s', async () => {
    vi.stubGlobal('fetch', async () => new Response(null, { status: 404, statusText: 'Not Found' }));
    const result = await loadRewiringNull(manifest, '/data');
    expect(result.status).toBe('unavailable');
  });

  it('is "invalid" with a sha256 reason when rewiring-null-v1.json is tampered', async () => {
    vi.stubGlobal('fetch', createPublicDataFetch({ corrupt: 'rewiring-null-v1.json' }));
    const result = await loadRewiringNull(manifest, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/sha256/i);
  });

  it('is "invalid" (sha256 mismatch, checked before parsing) when the served bytes are not even valid JSON', async () => {
    vi.stubGlobal('fetch', async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.endsWith('rewiring-null-v1.json')) {
        return new Response('not json', { status: 200, headers: { 'content-type': 'application/json' } });
      }
      return createPublicDataFetch()(input);
    });
    const result = await loadRewiringNull(manifest, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/sha256/i);
  });
});

describe('loadRewiringNull (shape validation, with a synthetic manifest sha256 that matches the served bytes)', () => {
  const sha256Hex = (data: string): string => createHash('sha256').update(data).digest('hex');

  /**
   * `sourceGraphSha256`/`rewired[0].gzipSha256` are the real committed
   * manifest's `binarySha256`/`rewiredArms.seed0.gzipSha256` — not
   * arbitrary placeholder hashes — because `loadRewiringNull` now
   * cross-checks both against `manifest` (dual review, Important: catches a
   * stale/re-pinned manifest that describes a different graph). Every test
   * below that expects `'ok'` depends on this matching; tests that expect a
   * shape failure fail before that cross-check ever runs, so they are
   * unaffected by these specific values.
   */
  const validArtifact = {
    version: 1,
    condition: 'authored, opponent parked',
    seeds: { start: 30001, count: 100 },
    ticks: 1800,
    substeps: 4,
    sourceGraphSha256: manifest.binarySha256,
    rewireSourceSha256: 'b'.repeat(64),
    shards: 18,
    biological: { score: -0.22, median: -1.41, std: 3.28, ci: [-0.85, 0.45] },
    disconnected: { score: -1.86, median: -2, std: 1.79, ci: [-2.22, -1.52] },
    rewired: [
      {
        seed: 0,
        gzipSha256: manifest.rewiredArms.seed0.gzipSha256,
        score: 1.02,
        median: 0.62,
        std: 3.73,
        ci: [0.3, 1.77],
        acceptedSwaps: 1,
        attempts: 2
      }
    ],
    null: { n: 1, mean: 1.02, median: 1.02, std: 0, p2_5: 1.02, p97_5: 1.02, iqr: 0, degenerate: true },
    bioPercentile: 0,
    pLow: 0.5,
    pHigh: 1,
    bins: { edges: [-2, 0, 2], counts: [0, 1] }
  };

  /**
   * Stubs `fetch` to serve `body` (JSON-stringified) at any URL and builds a
   * manifest whose `rewiringNull.sha256` actually matches those served
   * bytes — so `loadRewiringNull`'s sha256 check passes and its shape
   * validator is what each test below actually exercises.
   */
  const manifestServing = (body: unknown): { manifest: ArenaManifest; raw: string } => {
    const raw = JSON.stringify(body);
    const hash = sha256Hex(raw);
    vi.stubGlobal(
      'fetch',
      async () => new Response(raw, { status: 200, headers: { 'content-type': 'application/json' } })
    );
    return {
      manifest: { ...manifest, rewiringNull: { artifact: 'rewiring-null-v1.json', sha256: hash } },
      raw
    };
  };

  it('is "invalid" when the sha256-matching bytes are not valid JSON (exercises the JSON.parse failure path specifically)', async () => {
    const raw = 'not json';
    const hash = sha256Hex(raw);
    vi.stubGlobal('fetch', async () => new Response(raw, { status: 200, headers: { 'content-type': 'application/json' } }));
    const manifestForBody: ArenaManifest = {
      ...manifest,
      rewiringNull: { artifact: 'rewiring-null-v1.json', sha256: hash }
    };
    const result = await loadRewiringNull(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/not valid JSON/i);
  });

  it('is "ok" for a well-formed artifact', async () => {
    const { manifest: manifestForBody } = manifestServing(validArtifact);
    const result = await loadRewiringNull(manifestForBody, '/data');
    expect(result.status).toBe('ok');
  });

  it('is "invalid" when bins.counts.length does not equal edges.length - 1 (malformed bins)', async () => {
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      bins: { edges: [-2, 0, 2], counts: [0, 1, 5] }
    });
    const result = await loadRewiringNull(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/bins/i);
  });

  it('is "invalid" when bins.edges is not sorted', async () => {
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      bins: { edges: [2, 0, -2], counts: [0, 1] }
    });
    const result = await loadRewiringNull(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/bins/i);
  });

  it('is "invalid" when a required field is missing (a stale/hand-edited artifact)', async () => {
    const { biological: _biological, ...withoutBiological } = validArtifact;
    const { manifest: manifestForBody } = manifestServing(withoutBiological);
    const result = await loadRewiringNull(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/biological/i);
  });

  it('is "invalid" for the wrong version', async () => {
    const { manifest: manifestForBody } = manifestServing({ ...validArtifact, version: 2 });
    const result = await loadRewiringNull(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/version/i);
  });

  // Dual review, Important: cross-field consistency checks the renderer
  // relies on but a per-field check alone cannot catch.

  it('is "invalid" when bioPercentile is outside [0, 1] (e.g. a producer bug writing a percent, not a fraction)', async () => {
    const { manifest: manifestForBody } = manifestServing({ ...validArtifact, bioPercentile: 99.8 });
    const result = await loadRewiringNull(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/bioPercentile.*\[0, 1\]/);
  });

  it('is "invalid" when pLow/pHigh are outside [0, 1]', async () => {
    const { manifest: manifestForBody } = manifestServing({ ...validArtifact, pLow: -0.1 });
    const result = await loadRewiringNull(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/\[0, 1\]/);
  });

  it('is "invalid" when null.n disagrees with rewired.length', async () => {
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      null: { ...validArtifact.null, n: 5 }
    });
    const result = await loadRewiringNull(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/counts disagree/);
  });

  // Round-2 dual review (guardian S1, confirmed by mutation testing): the
  // test above only varies `null.n`, so it alone left `sum(bins.counts) !==
  // rewired.length` unexercised — deleting that half of the `||` still
  // passed every other test. `bins.counts` here sums to 2 while `null.n`
  // and `rewired.length` still agree with each other at 1, isolating this
  // specific clause. `null-report.ts`'s own history records exactly this
  // failure mode (a bin total that drifted from the graph count).
  it('is "invalid" when sum(bins.counts) disagrees with null.n/rewired.length even though those two agree with each other', async () => {
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      bins: { edges: [-2, 0, 2], counts: [1, 1] }
    });
    const result = await loadRewiringNull(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/sum\(bins\.counts\)=2/);
  });

  it('is "invalid" when a marker score (biological) falls outside the histogram bin domain', async () => {
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      biological: { ...validArtifact.biological, score: 99 }
    });
    const result = await loadRewiringNull(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/outside the histogram bin domain/);
  });

  it('is "invalid" when the shipped rewired-seed-0 marker falls outside the bin domain', async () => {
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      rewired: [{ ...validArtifact.rewired[0], score: -99 }]
    });
    const result = await loadRewiringNull(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/outside the histogram bin domain/);
  });

  it('is "invalid" when two rewired entries share the same seed (would double-count one graph)', async () => {
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      rewired: [validArtifact.rewired[0], { ...validArtifact.rewired[0] }],
      null: { ...validArtifact.null, n: 2 },
      bins: { edges: [-2, 0, 2], counts: [0, 2] }
    });
    const result = await loadRewiringNull(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/duplicate seeds/);
  });

  it('is "invalid" when a confidence interval is swapped (ci[0] > ci[1])', async () => {
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      biological: { ...validArtifact.biological, ci: [0.45, -0.85] }
    });
    const result = await loadRewiringNull(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/biological/i);
  });

  it('is "invalid" when a bin count is not an integer', async () => {
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      bins: { edges: [-2, 0, 2], counts: [0, 0.5] }
    });
    const result = await loadRewiringNull(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/bins/i);
  });

  it('is "invalid" when sourceGraphSha256 does not match the manifest\'s biological graph (a stale/re-pinned artifact)', async () => {
    const { manifest: manifestForBody } = manifestServing({ ...validArtifact, sourceGraphSha256: 'f'.repeat(64) });
    const result = await loadRewiringNull(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/sourceGraphSha256/);
  });

  it('is "invalid" when the seed-0 rewired entry\'s gzipSha256 does not match the shipped control arm', async () => {
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      rewired: [{ ...validArtifact.rewired[0], gzipSha256: 'e'.repeat(64) }]
    });
    const result = await loadRewiringNull(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/shipped rewired control arm/);
  });

  // Thermo review S3: the producer (`null-report.ts`) refuses to write a
  // report at all without a seed-0 entry — an artifact that hash/shape-
  // verifies but omits it must be rejected too, not silently rendered with
  // one fewer marker.
  it('is "invalid" when the rewired array has no seed-0 entry (the shipped control arm is missing entirely)', async () => {
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      rewired: [{ ...validArtifact.rewired[0], seed: 1 }]
    });
    const result = await loadRewiringNull(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/missing the shipped rewired-seed-0 control arm/);
  });

  // Thermo review S1: recomputes `(kBelow + 0.5*kEqual)/n` (and the paired
  // pLow/pHigh) from the artifact's own `rewired[].score`/`biological.score`
  // and rejects a hash/shape-valid artifact whose separately-authored
  // bioPercentile/pLow/pHigh disagree with that recomputation — the same
  // class of producer bug `null-report.ts`'s own comments document
  // happening once before for the bin totals, but here in the headline
  // percentile instead.
  it('is "invalid" when bioPercentile/pLow/pHigh disagree with the rank statistics recomputed from the artifact\'s own scores', async () => {
    const { manifest: manifestForBody } = manifestServing({
      ...validArtifact,
      // A second rewired graph scoring below biological (-0.22) shifts the
      // true bioPercentile to 0.5 — `bioPercentile`/`pLow`/`pHigh` below are
      // left at validArtifact's original (now-stale) values for n=1.
      rewired: [validArtifact.rewired[0], { ...validArtifact.rewired[0], seed: 1, score: -1 }],
      null: { ...validArtifact.null, n: 2 },
      bins: { edges: [-2, 0, 2], counts: [1, 1] }
    });
    const result = await loadRewiringNull(manifestForBody, '/data');
    expect(result.status).toBe('invalid');
    if (result.status === 'invalid') expect(result.reason).toMatch(/rank statistics recomputed/);
  });
});
