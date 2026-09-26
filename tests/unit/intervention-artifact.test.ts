// @vitest-environment node
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import {
  buildPathwayInterventionsArtifact,
  parseIntoArtifactArgs,
  renderPathwayInterventionsReportMarkdown,
  runPathwayInterventionsArtifact,
  type BuildArtifactInputs,
  type IntoArtifactArgs
} from '../../scripts/null/intervention-artifact';
import { TRANSFER_INPUT_CHANNELS, TRANSFER_OUTPUT_POPULATIONS } from '../../scripts/null/intervention-attribution';
import { sha256Hex } from '../../scripts/training/fsio';
import type { GraphKind, InterventionStatistics } from '../../scripts/null/intervention-report';
import type { NullTrainedInterventionEvaluationRaw, NullTrainedInterventionGraphRaw } from '../../scripts/null/null-trained-evaluate-graph-list';

/**
 * Coverage for `scripts/null/intervention-artifact.ts` — WP4's top-level
 * artifact/report/manifest producer. `buildPathwayInterventionsArtifact` is
 * exercised directly against synthetic in-memory inputs (no filesystem);
 * the CLI layer (`runPathwayInterventionsArtifact`) is exercised against a
 * real scratch directory, mirroring `tests/unit/intervention-report.test.ts`'s
 * own "CLI layer" describe block.
 */

const SHA = (label: string): string => label.padEnd(64, '0');

const transfer3x8 = () => TRANSFER_OUTPUT_POPULATIONS.map(() => TRANSFER_INPUT_CHANNELS.map(() => 0.001));

const attributionFixture = {
  version: 1,
  P: {
    kind: 'P',
    swaps: 6,
    targetReached: true,
    stopReason: 'target_reached',
    finalRightClearanceThrust: 0.09,
    finalForwardClearanceThrust: 0.02,
    candidateSampleSeed: 1,
    steps: [{}, {}, {}, {}, {}, {}]
  },
  Q: {
    kind: 'Q',
    swaps: 5,
    targetReached: true,
    stopReason: 'target_reached',
    finalRightClearanceThrust: 0.08,
    finalForwardClearanceThrust: 0.019,
    candidateSampleSeed: 1,
    steps: [{}, {}, {}, {}, {}]
  },
  R: { applicable: false, edgeCount: 0, reason: 'no input-labeled -> thrust edges in the biological graph' },
  biological: {
    transfer: { full3x8: transfer3x8(), rightClearanceThrust: 0.003, forwardClearanceThrust: 0.002 },
    sourceSha256: SHA('bio'),
    explanationCrossCheck: {}
  },
  nullRegeneration: { rightClearanceThrust: { p25: 0.01 }, forwardClearanceThrust: { p25: 0.009 } },
  producer: { script: 'scripts/analysis/interventions.py', sourceSha256: SHA('prod'), dependencies: [], host: { arch: 'aarch64', python: '3.12.3' } }
};

const indexFixture = (kinds: Record<string, string> = {}) => ({
  sourceArtifact: 'src.bin.gz',
  sourceSha256: SHA('bio'),
  controlCount: 100,
  entries: [
    { id: 'P', kind: 'P', gzipSha256: SHA('P'), binarySha256: SHA('Pb'), path: 'graphs/P.bin.gz', swaps: 6, transfer: { full3x8: transfer3x8() } },
    { id: 'Q', kind: 'Q', gzipSha256: SHA('Q'), binarySha256: SHA('Qb'), path: 'graphs/Q.bin.gz', swaps: 5, transfer: { full3x8: transfer3x8() } },
    ...Object.entries(kinds).map(([id, kind]) => ({ id, kind, gzipSha256: SHA(id), binarySha256: SHA(`${id}b`), path: `graphs/${id}.bin.gz`, swaps: 6, transfer: { full3x8: transfer3x8() } }))
  ]
});

const graphOutcome = (id: string, kind: GraphKind, mean: number) => ({
  id,
  kind,
  n: 3,
  mean,
  median: mean,
  std: 0,
  ci95: [mean, mean] as [number, number],
  publishedNullRank: { kBelow: 0, kEqual: 0, bioPercentile: 0.5, pLow: 0.5, pHigh: 0.5 }
});

const armDist = (scores: readonly number[]) => {
  const sorted = [...scores].sort((a, b) => a - b);
  return { n: sorted.length, scores: sorted, p5: sorted[0], p50: sorted[Math.floor(sorted.length / 2)], p95: sorted[sorted.length - 1] };
};

/**
 * `buildPathwayInterventionsArtifact` cross-checks `statistics.inputs.*Sha256`
 * against the actual bytes it hashes -- computed lazily (called only once
 * `indexWithControlKinds`/`rewiringNullFixture` are defined further down
 * this file) so a fixture using arbitrary placeholder hashes never fails
 * that check regardless of what a given test means to exercise.
 */
const indexShaFixture = (): string => sha256Hex(Buffer.from(JSON.stringify(indexWithControlKinds)));
const publishedNullShaFixture = (): string => sha256Hex(Buffer.from(JSON.stringify(rewiringNullFixture())));

const statisticsFixture = (): InterventionStatistics => ({
  version: 1,
  decoder: 'authored',
  seeds: { start: 30001, count: 3 },
  ticks: 1800,
  bootstrap: { seed: 42, resamples: 200 },
  inputs: { authoredSha256: SHA('auth'), indexSha256: indexShaFixture(), publishedNullSha256: publishedNullShaFixture() },
  biologicalReproduction: { computedScore: 0, publishedScore: 0, matches: true },
  graphs: [graphOutcome('P', 'P', 3.6), graphOutcome('Q', 'Q', 3.4)],
  controls: { C: armDist([1, 2, 3]), M: armDist([1, 2, 3]), MQ: armDist([1, 2, 3]) },
  publishedNullFloor: 1.5,
  p: { id: 'P', score: 3.6, percentileInPublishedNull: 0.856, pRankAmongC: 0.01, pRankAmongM: 0.01, category: 'pathway-supported' },
  q: { id: 'Q', score: 3.4, percentileInPublishedNull: 0.816, qRankAmongMQ: 0.01, channelSpecific: true },
  host: { arch: 'arm64', node: 'v22.0.0' }
});

const trainedGraphRaw = (id: string, trainerSeed: number, score: number): NullTrainedInterventionGraphRaw => ({
  id,
  trainerSeed,
  gzipSha256: SHA(id),
  armBundleSha256: SHA(`arm-${id}`),
  heldOutSeeds: [30001, 30002],
  movementScore: [score, score],
  foodPickups: [0, 0],
  hazardContacts: [0, 0]
});

const trainedFixture = (): NullTrainedInterventionEvaluationRaw => ({
  version: 1,
  seeds: { start: 30001, count: 2 },
  ticks: 20,
  substeps: 4,
  // Must equal the real index fixture's own sha256 -- `buildPathwayInterventionsArtifact`
  // now cross-checks `trainedRaw.graphListSha256` against it (a maintainability-review finding).
  graphListSha256: indexShaFixture(),
  host: { arch: 'arm64', node: 'v22.0.0' },
  d: 48,
  evaluatorGitRev: 'deadbeef',
  cemConfig: null,
  cemConfigWarnings: [],
  runs: [
    // All three P seeds stay at/below the C arm's max (66) -- mirrors this
    // study's real result (`no-specific-effect` at every seed, robust) --
    // see `intervention-report-trained.test.ts` for the category-boundary
    // cases (pathway-supported/edge-class-effect/non-robust) this fixture
    // deliberately does not exercise.
    trainedGraphRaw('P', 101, 60),
    trainedGraphRaw('P', 202, 61),
    trainedGraphRaw('P', 303, 62),
    ...['C000', 'C001', 'C002', 'C003', 'C004'].map((id, i) => trainedGraphRaw(id, 101, 62 + i)),
    ...['M1000', 'M1001', 'M1002', 'M1003', 'M1004'].map((id, i) => trainedGraphRaw(id, 101, 72 + i))
  ]
});

const rewiringNullFixture = () => ({
  sourceGraphSha256: SHA('bio'),
  trained: { rewired: Array.from({ length: 20 }, (_, i) => ({ score: 60 + i })) }
});

const indexWithControlKinds = indexFixture(
  Object.fromEntries([
    ...['C000', 'C001', 'C002', 'C003', 'C004'].map((id) => [id, 'C']),
    ...['M1000', 'M1001', 'M1002', 'M1003', 'M1004'].map((id) => [id, 'M'])
  ])
);

const buildInputs = (overrides: Partial<BuildArtifactInputs> = {}): BuildArtifactInputs => {
  const indexBytes = Buffer.from(JSON.stringify(indexWithControlKinds));
  const rewiringNullBytes = Buffer.from(JSON.stringify(rewiringNullFixture()));
  const statistics = statisticsFixture();
  return {
    statistics,
    attributionText: JSON.stringify(attributionFixture),
    attributionLabel: 'attribution.json',
    trainedRaw: trainedFixture(),
    indexText: indexBytes.toString('utf8'),
    indexLabel: 'index.json',
    indexBytes,
    rewiringNullBytes,
    rewiringNullParsed: rewiringNullFixture(),
    manifestBiologicalSha: SHA('bio'),
    manifestRewiringNullSha: statistics.inputs.publishedNullSha256,
    manifestNullExplanationSha: SHA('explain'),
    bootstrapSeed: 42,
    bootstrapResamples: 200,
    ...overrides
  };
};

describe('buildPathwayInterventionsArtifact', () => {
  it('combines statistics/attribution/trained/index into the full artifact', () => {
    const artifact = buildPathwayInterventionsArtifact(buildInputs());
    expect(artifact.version).toBe(1);
    expect(artifact.k).toBe(6);
    expect(artifact.kQ).toBe(5);
    expect(artifact.authored.category).toBe('pathway-supported');
    expect(artifact.authored.channelSpecific).toBe(true);
    expect(artifact.trained.trainedRobust).toBe(true);
    expect(artifact.trained.perSeed[101].category).toBe('no-specific-effect');
    expect(artifact.interventions.R.applicable).toBe(false);
    expect(artifact.sources.biologicalSha).toBe(SHA('bio'));
    expect(artifact.transferBeforeAfter.biological).toEqual(transfer3x8());
  });

  it('running twice on the same input is byte-identical (JSON.stringify)', () => {
    const inputs = buildInputs();
    const first = buildPathwayInterventionsArtifact(inputs);
    const second = buildPathwayInterventionsArtifact(inputs);
    expect(JSON.stringify(second)).toBe(JSON.stringify(first));
  });

  it('refuses to build from a diagnosticOnly statistics.json', () => {
    const statistics = { ...statisticsFixture(), diagnosticOnly: true as const };
    expect(() => buildPathwayInterventionsArtifact(buildInputs({ statistics }))).toThrow(/diagnosticOnly/);
  });

  it('throws when index.json sha256 does not match statistics.inputs.indexSha256 (stale statistics.json)', () => {
    const statistics = { ...statisticsFixture(), inputs: { ...statisticsFixture().inputs, indexSha256: SHA('wrong') } };
    expect(() => buildPathwayInterventionsArtifact(buildInputs({ statistics }))).toThrow(/does not match statistics\.json's recorded indexSha256/);
  });

  it('throws when the published null sha256 does not match statistics.inputs.publishedNullSha256', () => {
    const statistics = { ...statisticsFixture(), inputs: { ...statisticsFixture().inputs, publishedNullSha256: SHA('wrong') } };
    expect(() => buildPathwayInterventionsArtifact(buildInputs({ statistics }))).toThrow(/does not match statistics\.json's recorded publishedNullSha256/);
  });

  it('throws when the published null sha256 does not match the manifest rewiringNull.sha256 (stale manifest)', () => {
    expect(() => buildPathwayInterventionsArtifact(buildInputs({ manifestRewiringNullSha: SHA('wrong') }))).toThrow(
      /does not match the manifest's rewiringNull\.sha256/
    );
  });

  it("throws when manifest.binarySha256 does not match the published null's sourceGraphSha256", () => {
    expect(() => buildPathwayInterventionsArtifact(buildInputs({ manifestBiologicalSha: SHA('other-bio') }))).toThrow(
      /does not match the published null's sourceGraphSha256/
    );
  });

  it("throws when attribution.json's biological.sourceSha256 does not match manifest.binarySha256", () => {
    const badAttribution = { ...attributionFixture, biological: { ...attributionFixture.biological, sourceSha256: SHA('other') } };
    expect(() => buildPathwayInterventionsArtifact(buildInputs({ attributionText: JSON.stringify(badAttribution) }))).toThrow(
      /biological\.sourceSha256.*does not match manifest\.binarySha256/
    );
  });

  it("throws when P's swap count disagrees between index.json and attribution.json", () => {
    const badIndex = {
      ...indexWithControlKinds,
      entries: indexWithControlKinds.entries.map((e) => (e.id === 'P' ? { ...e, swaps: 99 } : e))
    };
    expect(() => buildPathwayInterventionsArtifact(buildInputs({ indexText: JSON.stringify(badIndex) }))).toThrow(
      /P's swap count disagrees/
    );
  });

  it("throws when Q's swap count disagrees between index.json and attribution.json", () => {
    const badIndex = {
      ...indexWithControlKinds,
      entries: indexWithControlKinds.entries.map((e) => (e.id === 'Q' ? { ...e, swaps: 99 } : e))
    };
    expect(() => buildPathwayInterventionsArtifact(buildInputs({ indexText: JSON.stringify(badIndex) }))).toThrow(
      /Q's swap count disagrees/
    );
  });

  it('throws when statistics.json has an unsupported version', () => {
    const statistics = { ...statisticsFixture(), version: 2 as unknown as 1 };
    expect(() => buildPathwayInterventionsArtifact(buildInputs({ statistics }))).toThrow(/statistics\.json has unsupported version/);
  });

  // Maintainability-review finding: `diagnosticOnly` is only ever set by
  // `intervention-report.ts` itself when `--allow-reproduction-mismatch`
  // was passed -- a hand-edited/older `statistics.json` with a failed
  // reproduction check but no `diagnosticOnly` flag must still be refused.
  it('refuses to build when biologicalReproduction.matches is false, even without the diagnosticOnly flag', () => {
    const statistics = { ...statisticsFixture(), biologicalReproduction: { ...statisticsFixture().biologicalReproduction, matches: false } };
    expect(() => buildPathwayInterventionsArtifact(buildInputs({ statistics }))).toThrow(/diagnosticOnly, or its biologicalReproduction\.matches is not true/);
  });

  it("throws when trained.json's graphListSha256 does not match index.json's sha256 (stale trained.json)", () => {
    const trainedRaw = { ...trainedFixture(), graphListSha256: SHA('stale') };
    expect(() => buildPathwayInterventionsArtifact(buildInputs({ trainedRaw }))).toThrow(
      /trained\.json graphListSha256 .* does not match .* sha256/
    );
  });

  it('throws when the published null has no trained.rewired scores', () => {
    expect(() => buildPathwayInterventionsArtifact(buildInputs({ rewiringNullParsed: { ...rewiringNullFixture(), trained: { rewired: [] } } }))).toThrow(
      /the published null has no trained\.rewired scores/
    );
  });

  it('throws when the published null has no trained section at all', () => {
    const { trained: _omit, ...withoutTrained } = rewiringNullFixture();
    expect(() => buildPathwayInterventionsArtifact(buildInputs({ rewiringNullParsed: withoutTrained }))).toThrow(
      /the published null has no trained\.rewired scores/
    );
  });

  it('throws when a trained.json run has a gzipSha256 that disagrees with index.json (stale trained.json)', () => {
    const trainedRaw = {
      ...trainedFixture(),
      runs: trainedFixture().runs.map((run) => (run.id === 'C000' ? { ...run, gzipSha256: SHA('stale-c000') } : run))
    };
    expect(() => buildPathwayInterventionsArtifact(buildInputs({ trainedRaw }))).toThrow(
      /"C000" was scored from a different graph file than index\.json currently lists/
    );
  });

  /**
   * Both reviewers flagged this as the headline finding: an earlier version
   * of `renderPathwayInterventionsReportMarkdown` hard-coded prose that
   * happened to be true only for this study's real (`no-specific-effect`,
   * robust) trained result. Rendering a synthetic `pathway-supported`,
   * non-robust trained result must change every trained-result sentence to
   * match -- never print the fixed "P shows no advantage ..." text
   * regardless of the actual category.
   */
  it('the rendered report reflects a non-current trained category and robustness, not hard-coded prose', () => {
    const trainedRaw: NullTrainedInterventionEvaluationRaw = {
      ...trainedFixture(),
      runs: trainedFixture().runs.map((run) =>
        run.id === 'P' ? { ...run, movementScore: [200, 200] } : run
      )
    };
    const artifact = buildPathwayInterventionsArtifact(buildInputs({ trainedRaw }));
    expect(artifact.trained.perSeed[101].category).toBe('pathway-supported');
    expect(artifact.trained.trainedRobust).toBe(true);

    const markdown = renderPathwayInterventionsReportMarkdown(artifact);
    expect(markdown).not.toMatch(/P shows no advantage over either freshly-trained control arm/);
    expect(markdown).toMatch(/P outperforms both freshly-trained control arms at all 3 trainer seeds tested/);
    expect(markdown).toMatch(/\*\*pathway-supported\*\*, robust: true/);
  });

  it('the rendered report states a non-robust trained result as such, per seed, without forcing a single category', () => {
    const trainedRaw: NullTrainedInterventionEvaluationRaw = {
      ...trainedFixture(),
      runs: trainedFixture().runs.map((run) => (run.id === 'P' && run.trainerSeed === 303 ? { ...run, movementScore: [200, 200] } : run))
    };
    const artifact = buildPathwayInterventionsArtifact(buildInputs({ trainedRaw }));
    expect(artifact.trained.trainedRobust).toBe(false);

    const markdown = renderPathwayInterventionsReportMarkdown(artifact);
    expect(markdown).toMatch(/the 3 trainer seeds disagree on the category/);
    expect(markdown).toMatch(/seed 101: no-specific-effect/);
    expect(markdown).toMatch(/seed 303: pathway-supported/);
    expect(markdown).not.toMatch(/P shows no advantage over either freshly-trained control arm at all 3 trainer seeds tested/);
  });
});

describe('renderPathwayInterventionsReportMarkdown', () => {
  it('states the predeclared categories verbatim and every mandated limitation', () => {
    const artifact = buildPathwayInterventionsArtifact(buildInputs());
    const markdown = renderPathwayInterventionsReportMarkdown(artifact);
    expect(markdown).toMatch(/Pathway supported/);
    expect(markdown).toMatch(/Edge-class effect/);
    expect(markdown).toMatch(/Generic rewiring effect/);
    expect(markdown).toMatch(/Not supported/);
    expect(markdown).toMatch(/Channel-specific \(modifier, authored decoder only\)/);
    expect(markdown).toMatch(/net effect of this accepted swap set/);
    expect(markdown).toMatch(/no\s+biological claim/);
    expect(markdown).toMatch(/hand-written, not trained and not biology/);
    expect(markdown).toMatch(/coarse resolution/);
    expect(markdown).toMatch(/authored-decoder only/);
    expect(markdown).toMatch(/generic-vs-not-supported split is undetermined/);
    expect(markdown).toMatch(/Multiple comparisons/);
    expect(markdown).toContain('| thrust |');
    expect(markdown).toContain('| yaw |');
    expect(markdown).toContain('| brake |');
  });
});

describe('parseIntoArtifactArgs', () => {
  it('applies defaults', () => {
    const args = parseIntoArtifactArgs([]);
    expect(args.bootstrapResamples).toBe(10000);
  });

  it('rejects --out overwriting an input file', () => {
    expect(() => parseIntoArtifactArgs(['--statistics', 'shared.json', '--out', 'shared.json'])).toThrow(/--out must not overwrite an input file/);
  });

  it('rejects an unknown flag', () => {
    expect(() => parseIntoArtifactArgs(['--bogus'])).toThrow(/Unknown argument/);
  });
});

describe('runPathwayInterventionsArtifact (CLI layer)', () => {
  let root: string;
  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'intervention-artifact-cli-'));
  });
  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  const writeFixtures = (): IntoArtifactArgs => {
    const statistics = statisticsFixture();
    writeFileSync(join(root, 'statistics.json'), JSON.stringify(statistics));
    writeFileSync(join(root, 'attribution.json'), JSON.stringify(attributionFixture));
    writeFileSync(join(root, 'trained.json'), JSON.stringify(trainedFixture()));
    writeFileSync(join(root, 'index.json'), JSON.stringify(indexWithControlKinds));
    writeFileSync(join(root, 'rewiring-null.json'), JSON.stringify(rewiringNullFixture()));
    // `verifyManifestRoundTrips` (reused from `null-report.ts`) refuses to
    // write unless re-serializing the manifest with sorted keys/2-space
    // indent reproduces the exact bytes on disk -- so this fixture must
    // already be written in that exact canonical form (mirrors the real
    // manifest's own Python `json.dumps(..., indent=2, sort_keys=True)`
    // convention), not compact/arbitrary-key-order JSON.
    writeFileSync(
      join(root, 'manifest.json'),
      `${JSON.stringify(
        {
          binarySha256: SHA('bio'),
          nullExplanation: { sha256: SHA('explain') },
          rewiringNull: { sha256: statistics.inputs.publishedNullSha256 }
        },
        null,
        2
      )}\n`
    );
    return {
      statistics: join(root, 'statistics.json'),
      attribution: join(root, 'attribution.json'),
      trained: join(root, 'trained.json'),
      index: join(root, 'index.json'),
      rewiringNull: join(root, 'rewiring-null.json'),
      manifest: join(root, 'manifest.json'),
      out: join(root, 'nested', 'pathway-interventions-v1.json'),
      reportMd: join(root, 'nested', 'pathway-interventions-report.md'),
      bootstrapSeed: 42,
      bootstrapResamples: 200
    };
  };

  it('writes the artifact, the manifest key, and the report; creates output directories', () => {
    const args = writeFixtures();
    const result = runPathwayInterventionsArtifact(args);
    expect(existsSync(result.out)).toBe(true);
    expect(existsSync(result.reportMdPath)).toBe(true);
    const manifest = JSON.parse(readFileSync(args.manifest, 'utf8'));
    expect(manifest.pathwayInterventions).toEqual({ artifact: 'pathway-interventions-v1.json', sha256: result.artifactSha256 });
  });

  it('regenerating the artifact twice gives byte-identical output', () => {
    const args = writeFixtures();
    const first = runPathwayInterventionsArtifact(args);
    const firstBytes = readFileSync(first.out);
    const second = runPathwayInterventionsArtifact(args);
    const secondBytes = readFileSync(second.out);
    expect(secondBytes.equals(firstBytes)).toBe(true);
    expect(second.artifactSha256).toBe(first.artifactSha256);
  });

  it('throws when the manifest is missing nullExplanation.sha256', () => {
    const args = writeFixtures();
    const statistics = statisticsFixture();
    writeFileSync(args.manifest, JSON.stringify({ binarySha256: SHA('bio'), rewiringNull: { sha256: statistics.inputs.publishedNullSha256 } }));
    expect(() => runPathwayInterventionsArtifact(args)).toThrow(/missing nullExplanation\.sha256/);
  });
});
