// @vitest-environment node
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import {
  buildReadoutAttributionArtifact,
  DEFAULT_DESCENDING_TYPES_PATH,
  DEFAULT_MANIFEST_PATH,
  DEFAULT_PATHWAY_INTERVENTIONS_PATH,
  DEFAULT_TRAINED_READOUT_ARTIFACT_PATH,
  DEFAULT_TRAINED_READOUT_MANIFEST_PATH,
  loadDescendingTypeNames,
  parseAttributionReportArgs,
  readoutAttributionProducer,
  runAttributionReport,
  type AblationJsonEntry,
  type BuildReadoutAttributionArtifactInputs,
  type HypothesisResult,
  type IndependenceJsonEntry,
  type LinkageJsonEntry,
  type RegimeJsonEntry,
  type SaliencyJsonEntry
} from '../../scripts/attribution/attribution-report';
import { sha256Hex } from '../../scripts/training/fsio';

/**
 * WP3 of `.agents/plans/readout-attribution`: unit coverage for
 * `scripts/attribution/attribution-report.ts`.
 *
 * Dual thermo review (both reviewers, CRITICAL): an earlier version of this
 * file read the real WP2 outputs directly from `training/runs/attribution/
 * *.json` -- gitignored scratch, absent in CI and any fresh checkout, which
 * made 8 of these tests fail with ENOENT outside this one already-populated
 * worktree. `training/archive/trained-readouts-v1.json` (the archive
 * itself) IS committed, but this file no longer even depends on that --
 * every `buildReadoutAttributionArtifact`/`runAttributionReport` test below
 * builds its own small, fully synthetic archive + WP2-output fixture world
 * in a temp directory, the same "pure builder against a synthetic fixture
 * world" discipline `tests/unit/selection-report.test.ts`/
 * `tests/unit/task-generality-report.test.ts`/`tests/unit/repertoire-report.test.ts`
 * already establish for their own sibling producers (see
 * `repertoire-report.test.ts`'s own top doc comment). Fixtures are typed
 * against `attribution-report.ts`'s own exported `SaliencyJsonEntry`/
 * `IndependenceJsonEntry`/`LinkageJsonEntry`/`RegimeJsonEntry`/
 * `AblationJsonEntry`/`HypothesisResult` shapes, so a fixture that compiles
 * is guaranteed to match what the real WP2 CLIs actually write.
 *
 * `descending-types-v1.json`/`trained-readout-v1.json`/`trained-readout-v1.manifest.json`/
 * `pathway-interventions-v1.json`/`malecns-arena-v1.manifest.json` are all
 * genuinely committed (not gitignored) `public/data/` artifacts, always
 * present in a fresh checkout -- those four cross-checks are exercised
 * against the real files (doctored COPIES for the negative-path tests,
 * never the originals), exactly as before.
 *
 * A separate describe block below ("the published artifact") reads
 * `public/data/readout-attribution-v1.json` directly (a real, committed
 * file) to keep one honest assertion about this study's actual real-data
 * outcome (all three hypotheses inconclusive, 23 default-task readouts) --
 * grounded in a committed file, never `training/runs/`.
 */

const FIXTURE_IDS = ['fixture-bio-seed1', 'fixture-p-seed2'] as const;

interface ArchiveFixtureEntry {
  readonly id: string;
  readonly arm: string;
  readonly kind: string;
  readonly graphId: string;
  readonly trainerSeed: number;
  readonly arenaTask: 'default';
}

const archiveEntry = (id: string, arm: string, graphId: string, trainerSeed: number): ArchiveFixtureEntry => ({
  id,
  arm,
  kind: 'bigq',
  graphId,
  trainerSeed,
  arenaTask: 'default'
});

const saliencyEntry = (id: string, scale: number): SaliencyJsonEntry => ({
  id,
  thrust: [scale * 0.1, scale * 0.2, scale * 0.3],
  yaw: [scale * 0.05, scale * 0.1, scale * 0.15],
  thrustVarWeighted: [scale * 0.01, scale * 0.02, scale * 0.03],
  yawVarWeighted: [scale * 0.005, scale * 0.01, scale * 0.015],
  inputMean: [0.1, 0.2, 0.3],
  inputStd: [0.01, 0.02, 0.03]
});

const independenceEntry = (id: string, trainedMean: number, silencedMean: number): IndependenceJsonEntry => ({
  id,
  trainedMean,
  silencedMean,
  defined: trainedMean > 1,
  ratio: trainedMean > 1 ? silencedMean / trainedMean : null
});

const linkageEntry = (id: string, rho: number): LinkageJsonEntry => ({
  id,
  degenerate: false,
  rhoThrust: rho,
  rhoYaw: rho * 0.5,
  ciCluster: [rho - 0.2, rho + 0.2],
  ciNeuron: [rho - 0.25, rho + 0.25],
  clusterCount: 3,
  clusterSizes: [1, 1, 1]
});

const regimeEntry = (id: string): RegimeJsonEntry => ({
  id,
  clampFraction: 0.02,
  steadyStateDistance: 0.4,
  valid: true
});

const ablationEntry = (id: string, baselineMean: number): AblationJsonEntry => ({
  id,
  n: 100,
  baselineMean,
  ablations: [
    { input: 0, rank: 'top', saliencyScore: 1.5, inputMean: 0.1, inputStd: 0.01, effect: { n: 100, meanDifference: -2, ci95: [-4, 0] } },
    { input: 1, rank: 'bottom', saliencyScore: 0.1, inputMean: 0.01, inputStd: 0.001, effect: { n: 100, meanDifference: 0.1, ci95: [-1, 1] } }
  ]
});

interface HypothesesFixture {
  readonly version: 1;
  readonly archiveSha256: string;
  readonly hypothesisCount: number;
  readonly multipleComparisonCorrection: string;
  readonly H1: HypothesisResult;
  readonly H2: HypothesisResult;
  readonly H3: HypothesisResult;
}

let tmpDir: string;

beforeEach(() => {
  tmpDir = mkdtempSync(join(tmpdir(), 'attribution-report-test-'));
});

afterEach(() => {
  rmSync(tmpDir, { recursive: true, force: true });
});

/**
 * Writes a small, self-contained, internally-consistent archive + WP2
 * output fixture world (two default-task readouts) to `tmpDir`, returning
 * the paths `buildReadoutAttributionArtifact`/`runAttributionReport` need
 * plus the archive's own real (computed, not hand-picked) sha256. The
 * archive itself needs only `id`/`arm`/`kind`/`graphId`/`trainerSeed`/
 * `arenaTask` -- `scripts/attribution/shared.ts#loadArchive` (which
 * `buildReadoutAttributionArtifact` calls) reads only those fields; it
 * never decodes `theta` or resolves a graph, so no weights/graph-identity
 * fields are needed for this producer's own tests (unlike WP2's own
 * per-analysis CLIs, which do).
 */
const writeFixtureWorld = (
  overrides?: Partial<{
    saliency: readonly SaliencyJsonEntry[];
    independence: readonly IndependenceJsonEntry[];
    linkage: readonly LinkageJsonEntry[];
    regime: readonly RegimeJsonEntry[];
    ablation: readonly AblationJsonEntry[];
  }>
): { readonly archivePath: string; readonly archiveSha256: string } & Record<
  'saliencyPath' | 'independencePath' | 'linkagePath' | 'regimePath' | 'ablationPath' | 'hypothesesPath',
  string
> => {
  const archiveObj = {
    version: 1,
    readouts: [archiveEntry(FIXTURE_IDS[0], 'biological', 'biological', 101), archiveEntry(FIXTURE_IDS[1], 'rewired', 'P', 202)]
  };
  const archivePath = join(tmpDir, 'trained-readouts-v1.json');
  const archiveBytes = JSON.stringify(archiveObj);
  writeFileSync(archivePath, archiveBytes);
  const archiveSha256 = sha256Hex(archiveBytes);

  const writeJson = (filename: string, body: unknown): string => {
    const path = join(tmpDir, filename);
    writeFileSync(path, JSON.stringify(body));
    return path;
  };

  const saliency = overrides?.saliency ?? [saliencyEntry(FIXTURE_IDS[0], 1), saliencyEntry(FIXTURE_IDS[1], 2)];
  const independence = overrides?.independence ?? [
    independenceEntry(FIXTURE_IDS[0], 70, 20),
    independenceEntry(FIXTURE_IDS[1], 65, 18)
  ];
  const linkage = overrides?.linkage ?? [linkageEntry(FIXTURE_IDS[0], 0.1), linkageEntry(FIXTURE_IDS[1], -0.1)];
  const regime = overrides?.regime ?? [regimeEntry(FIXTURE_IDS[0]), regimeEntry(FIXTURE_IDS[1])];
  const ablation = overrides?.ablation ?? [ablationEntry(FIXTURE_IDS[0], 70), ablationEntry(FIXTURE_IDS[1], 65)];

  const saliencyPath = writeJson('saliency.json', { version: 1, archiveSha256, entries: saliency });
  const independencePath = writeJson('independence.json', { version: 1, archiveSha256, entries: independence });
  const linkagePath = writeJson('linkage.json', { version: 1, archiveSha256, readouts: linkage });
  const regimePath = writeJson('regime.json', { version: 1, archiveSha256, entries: regime });
  const ablationPath = writeJson('ablation.json', { version: 1, archiveSha256, entries: ablation });
  // `evidence` mirrors the real shape `attribution-report-markdown.ts`'s
  // `renderH1`/`renderH2`/`renderH3` read (H1Evidence/H2Evidence/H3Evidence,
  // not exported by that module) -- the end-to-end `runAttributionReport`
  // test below exercises the real markdown renderer, which reads into
  // these fields, so an empty `evidence: {}` would throw inside the
  // renderer rather than exercising it.
  const hypotheses: HypothesesFixture = {
    version: 1,
    archiveSha256,
    hypothesisCount: 3,
    multipleComparisonCorrection: 'none',
    H1: {
      outcome: 'inconclusive',
      reason: 'threshold-not-met-in-all-seeds',
      evidence: {
        rhoThreshold: 0.3,
        seeds: [
          { id: FIXTURE_IDS[0], regimeValid: true, degenerate: false, rho: 0.1, ciCluster: [-0.1, 0.3], ciNeuron: [-0.1, 0.3], meetsThreshold: false }
        ]
      }
    },
    H2: {
      outcome: 'supported',
      evidence: { equivalenceBound: 0.1, differences: [0.01, -0.02, 0.01], ci: [-0.05, 0.05], n: 3 }
    },
    H3: {
      outcome: 'not-supported',
      evidence: {
        ratioBound: 1.25,
        ratios: [{ seed: 101, pMean: 2, bioMean: 1, ratio: 2 }],
        ci: [1.5, 2.5],
        newlyConnectedThrustDIndices: [0]
      }
    }
  };
  const hypothesesPath = writeJson('hypotheses.json', hypotheses);

  return { archivePath, archiveSha256, saliencyPath, independencePath, linkagePath, regimePath, ablationPath, hypothesesPath };
};

/** Combines a fixture world's synthetic paths with the real, committed `public/data/` defaults for the four manifest-cross-checked inputs. */
const inputsFor = (world: ReturnType<typeof writeFixtureWorld>): BuildReadoutAttributionArtifactInputs => ({
  archivePath: world.archivePath,
  saliencyPath: world.saliencyPath,
  independencePath: world.independencePath,
  linkagePath: world.linkagePath,
  regimePath: world.regimePath,
  ablationPath: world.ablationPath,
  hypothesesPath: world.hypothesesPath,
  descendingTypesPath: DEFAULT_DESCENDING_TYPES_PATH,
  trainedReadoutArtifactPath: DEFAULT_TRAINED_READOUT_ARTIFACT_PATH,
  trainedReadoutManifestPath: DEFAULT_TRAINED_READOUT_MANIFEST_PATH,
  pathwayInterventionsPath: DEFAULT_PATHWAY_INTERVENTIONS_PATH,
  manifestPath: DEFAULT_MANIFEST_PATH
});

/** Writes a doctored copy of one real, committed `public/data/` JSON file to `tmpDir`, applying `mutate` to the parsed object first -- never mutates the committed original. */
const doctoredCopy = (sourcePath: string, filename: string, mutate: (parsed: any) => any): string => {
  const parsed = JSON.parse(readFileSync(sourcePath, 'utf8'));
  const mutated = mutate(parsed);
  const outPath = join(tmpDir, filename);
  writeFileSync(outPath, JSON.stringify(mutated));
  return outPath;
};

describe('buildReadoutAttributionArtifact (pure builder, against a synthetic fixture world)', () => {
  it('builds a two-readout artifact from the synthetic fixture world, sha-verified against the manifest\'s real committed cross-check fields', () => {
    const world = writeFixtureWorld();
    const artifact = buildReadoutAttributionArtifact(inputsFor(world));
    expect(artifact.version).toBe(1);
    expect(artifact.coverage.ids).toEqual([...FIXTURE_IDS].sort());
    expect(artifact.coverage.perTaskIncluded).toBe(false);
    expect(artifact.readouts).toHaveLength(2);
    expect(artifact.sources.archiveSha).toBe(world.archiveSha256);
    expect(artifact.hypotheses.H1.outcome).toBe('inconclusive');
    expect(artifact.hypotheses.H2.outcome).toBe('supported');
    expect(artifact.hypotheses.H3.outcome).toBe('not-supported');
    const bio = artifact.readouts.find((r) => r.id === FIXTURE_IDS[0]);
    expect(bio?.arm).toBe('biological');
    expect(bio?.graphId).toBe('biological');
    expect(bio?.trainerSeed).toBe(101);
    expect(bio?.saliency.thrust).toEqual([0.1, 0.2, 0.3]);
    expect(bio?.ablation).toHaveLength(2);
    expect(bio?.independence.trainedMean).toBe(70);
    expect(bio?.linkage.rhoThrust).toBe(0.1);
    expect(bio?.regime.valid).toBe(true);
  });

  it('is byte-identical (via JSON.stringify) across two independent builds against the same synthetic inputs', () => {
    const world = writeFixtureWorld();
    const first = JSON.stringify(buildReadoutAttributionArtifact(inputsFor(world)));
    const second = JSON.stringify(buildReadoutAttributionArtifact(inputsFor(world)));
    expect(first).toBe(second);
  });

  it('readoutAttributionProducer() reports its own real source identity, dependencies including the markdown renderer', () => {
    const producer = readoutAttributionProducer();
    expect(producer.script).toBe('scripts/attribution/attribution-report.ts');
    expect(producer.sourceSha256).toMatch(/^[0-9a-f]{64}$/);
    expect(producer.dependencies).toContain('scripts/attribution/attribution-report-markdown.ts');
    expect(producer.dependencies).toContain('scripts/attribution/archive-readouts.ts');
  });

  it('loadDescendingTypeNames sorts the real 48 descending neurons by ascending graph-node index (D-space order), names them from the pinned annotations', () => {
    const types = loadDescendingTypeNames(DEFAULT_DESCENDING_TYPES_PATH);
    expect(types).toHaveLength(48);
    // Strictly ascending by the raw graph-node `index` field -- the same
    // order `outputNeuronIndices(graph)` produces, so array position `d`
    // is directly the D-space lookup (see `attribution-report-markdown.ts#nameForIndex`'s
    // own doc comment for why this must NOT be looked up by matching
    // `index === d`; the raw graph-node index is not 0..47).
    for (let i = 1; i < types.length; i += 1) {
      expect(types[i].index).toBeGreaterThan(types[i - 1].index);
    }
    expect(types.every((t) => typeof t.bodyId === 'string')).toBe(true);
  });

  it('throws when a WP2 output was produced from a different archive (archiveSha256 mismatch)', () => {
    const world = writeFixtureWorld();
    const doctoredSaliency = doctoredCopy(world.saliencyPath, 'doctored-saliency.json', (parsed) => ({
      ...parsed,
      archiveSha256: 'f'.repeat(64)
    }));
    expect(() => buildReadoutAttributionArtifact({ ...inputsFor(world), saliencyPath: doctoredSaliency })).toThrow(
      /was produced from a different archive/
    );
  });

  it('throws when a WP2 output is missing an entry for one of the archived readouts', () => {
    const world = writeFixtureWorld();
    const doctoredIndependence = doctoredCopy(world.independencePath, 'doctored-independence.json', (parsed) => ({
      ...parsed,
      entries: parsed.entries.filter((e: { id: string }) => e.id !== FIXTURE_IDS[0])
    }));
    expect(() =>
      buildReadoutAttributionArtifact({ ...inputsFor(world), independencePath: doctoredIndependence })
    ).toThrow(new RegExp(`independence\\.json has no entry for archived readout "${FIXTURE_IDS[0]}"`));
  });

  it('throws when descending-types-v1.json does not match the manifest\'s recorded sha256', () => {
    const world = writeFixtureWorld();
    const doctoredManifest = doctoredCopy(DEFAULT_MANIFEST_PATH, 'manifest.json', (parsed) => ({
      ...parsed,
      descendingTypes: { ...parsed.descendingTypes, sha256: 'e'.repeat(64) }
    }));
    expect(() => buildReadoutAttributionArtifact({ ...inputsFor(world), manifestPath: doctoredManifest })).toThrow(
      /descending-types-v1\.json sha256 .* does not match/
    );
  });

  it('throws when trained-readout-v1.json does not match trained-readout-v1.manifest.json\'s artifactSha256', () => {
    const world = writeFixtureWorld();
    const doctoredTrainedManifest = doctoredCopy(
      DEFAULT_TRAINED_READOUT_MANIFEST_PATH,
      'trained-readout-v1.manifest.json',
      (parsed) => ({ ...parsed, artifactSha256: 'd'.repeat(64) })
    );
    expect(() =>
      buildReadoutAttributionArtifact({ ...inputsFor(world), trainedReadoutManifestPath: doctoredTrainedManifest })
    ).toThrow(/does not match trained-readout-v1\.manifest\.json's artifactSha256/);
  });

  it('throws when pathway-interventions-v1.json does not match the manifest\'s recorded sha256', () => {
    const world = writeFixtureWorld();
    const doctoredManifest = doctoredCopy(DEFAULT_MANIFEST_PATH, 'manifest.json', (parsed) => ({
      ...parsed,
      pathwayInterventions: { ...parsed.pathwayInterventions, sha256: 'c'.repeat(64) }
    }));
    expect(() => buildReadoutAttributionArtifact({ ...inputsFor(world), manifestPath: doctoredManifest })).toThrow(
      /pathway-interventions-v1\.json sha256 .* does not match/
    );
  });
});

describe('runAttributionReport (CLI layer, end to end against the synthetic fixture world)', () => {
  it('writes a byte-identical artifact + a report on two independent runs', () => {
    const world = writeFixtureWorld();
    const out1 = join(tmpDir, 'run1', 'readout-attribution-v1.json');
    const reportMd1 = join(tmpDir, 'run1', 'readout-attribution-report.md');
    const out2 = join(tmpDir, 'run2', 'readout-attribution-v1.json');
    const reportMd2 = join(tmpDir, 'run2', 'readout-attribution-report.md');

    const result1 = runAttributionReport({ ...inputsFor(world), out: out1, reportMd: reportMd1, skipManifestUpdate: true });
    const result2 = runAttributionReport({ ...inputsFor(world), out: out2, reportMd: reportMd2, skipManifestUpdate: true });

    expect(result1.artifactSha256).toBe(result2.artifactSha256);
    expect(readFileSync(out1)).toEqual(readFileSync(out2));
    expect(readFileSync(reportMd1, 'utf8')).toBe(readFileSync(reportMd2, 'utf8'));

    const report = readFileSync(reportMd1, 'utf8');
    expect(report).toContain('# Readout attribution (under this model)');
    expect(report).toContain('## Predeclared hypotheses and outcome rules');
    expect(report).toContain('## What remains unexplained');
    expect(report).toContain('## Limitations');
    expect(report).toContain('**Outcome: inconclusive**');
    expect(report).toContain('**Outcome: supported**');
    expect(report).toContain('**Outcome: not-supported**');
  });
});

describe('parseAttributionReportArgs', () => {
  it('refuses --out and --report-md pointing at the same path', () => {
    expect(() => parseAttributionReportArgs(['--out', '/tmp/same.json', '--report-md', '/tmp/same.json'])).toThrow(
      /--out and --report-md must not be the same path/
    );
  });

  it('throws on an unrecognized flag', () => {
    expect(() => parseAttributionReportArgs(['--bogus', 'x'])).toThrow(/Unknown argument/);
  });
});

describe('the published readout-attribution-v1.json artifact (real, committed data)', () => {
  /**
   * Reads `public/data/readout-attribution-v1.json` directly -- a real,
   * committed file (never `training/runs/`) -- to keep one honest
   * assertion about this study's actual real-data outcome: all three
   * predeclared hypotheses came back inconclusive on this model, the
   * analysis covers exactly the 23 default-task archive entries, and the
   * manifest's pinned sha256 equals the artifact's own bytes.
   */
  it('states all three hypotheses inconclusive, covers 23 default-task readouts, and its sha matches the manifest', () => {
    const manifest = JSON.parse(readFileSync(DEFAULT_MANIFEST_PATH, 'utf8')) as {
      readonly readoutAttribution?: { readonly artifact: string; readonly sha256: string };
    };
    expect(manifest.readoutAttribution).toBeDefined();
    const artifactPath = join(DEFAULT_MANIFEST_PATH, '..', manifest.readoutAttribution!.artifact);
    const bytes = readFileSync(artifactPath);
    expect(sha256Hex(bytes)).toBe(manifest.readoutAttribution!.sha256);

    const artifact = JSON.parse(bytes.toString('utf8'));
    expect(artifact.version).toBe(1);
    expect(artifact.coverage.ids).toHaveLength(23);
    expect(artifact.coverage.perTaskIncluded).toBe(false);
    expect(artifact.hypotheses.H1.outcome).toBe('inconclusive');
    expect(artifact.hypotheses.H2.outcome).toBe('inconclusive');
    expect(artifact.hypotheses.H3.outcome).toBe('inconclusive');
    expect(artifact.hypotheses.hypothesisCount).toBe(3);
  });
});
