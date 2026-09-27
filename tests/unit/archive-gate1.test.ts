import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

import { sha256Hex } from '../../scripts/training/fsio';
import type { ArchivedReadout, TrainedReadoutArchive } from '../../scripts/attribution/archive-readouts';

/**
 * `.agents/plans/readout-attribution/01-archive-and-types.md`'s WP1 Gate 1,
 * encoded as a permanent, committed-data-only regression test (a dual-review
 * finding: the full rescore half of Gate 1 -- decoder `trained`, held-out
 * seeds 30001-30100 -- was verified once by hand when this archive was built
 * and is too expensive to re-run as a unit test, but this static half (every
 * archived identity/score matches what was already published) costs nothing
 * and would have caught a hand-edit or a future regression in
 * `archive-readouts.ts` immediately). Reads only files this repository
 * already commits -- no worktree, no episode simulation.
 */

const repoRoot = resolve(__dirname, '../..');
const readJson = <T>(relPath: string): T => JSON.parse(readFileSync(resolve(repoRoot, relPath), 'utf8')) as T;
const readBytes = (relPath: string): Buffer => readFileSync(resolve(repoRoot, relPath));

const archive = readJson<TrainedReadoutArchive>('training/archive/trained-readouts-v1.json');

interface RawInterventionRun {
  readonly id: string;
  readonly trainerSeed: number;
  readonly gzipSha256: string;
  readonly armBundleSha256: string;
  readonly movementScore: readonly number[];
}
const rawScores = readJson<{ readonly runs: readonly RawInterventionRun[] }>(
  'training/archive/intervention-trained-raw-v1.json'
);

interface PathwayInterventionsArtifact {
  readonly sources: { readonly indexSha: string };
  readonly trained: {
    readonly perSeed: Readonly<Record<string, { readonly score: number }>>;
    readonly controls: {
      readonly C: { readonly scores: readonly number[] };
      readonly M: { readonly scores: readonly number[] };
    };
  };
}
const pathwayInterventions = readJson<PathwayInterventionsArtifact>('public/data/pathway-interventions-v1.json');

interface MalecnsManifest {
  readonly gzipSha256: string;
  readonly binarySha256: string;
  readonly rewiredArms: Readonly<Record<string, { readonly gzipSha256: string; readonly binarySha256: string }>>;
}
const manifest = readJson<MalecnsManifest>('public/data/malecns-arena-v1.manifest.json');

interface InterventionIndexEntry {
  readonly id: string;
  readonly gzipSha256: string;
  readonly binarySha256: string;
}
const interventionIndex = readJson<{ readonly entries: readonly InterventionIndexEntry[] }>(
  'training/archive/intervention-index-v1.json'
);

interface TrainedReadoutReport {
  readonly arms: Readonly<
    Record<
      string,
      {
        readonly armBundleSha256: string;
        readonly replicas: Readonly<Record<string, { readonly weightsSha256: string }>>;
      }
    >
  >;
  readonly gpuRerun: { readonly arm: string; readonly trainerSeed: number; readonly heldOutMean: number };
}
const trainedReadoutReport = readJson<TrainedReadoutReport>('public/data/trained-readout-v1.report.json');

const mean = (values: readonly number[]): number => values.reduce((sum, v) => sum + v, 0) / values.length;

describe('WP1 archive vs published sources (Gate 1, identity/score half)', () => {
  it('intervention-index-v1.json is byte-identical to what pathway-interventions-v1.json was published against', () => {
    const actual = sha256Hex(readBytes('training/archive/intervention-index-v1.json'));
    expect(actual).toBe(pathwayInterventions.sources.indexSha);
  });

  it('has exactly 9 bigq entries (+ the GPU rerun) and 13 intervention entries', () => {
    const bigq = archive.readouts.filter((r) => r.kind === 'bigq');
    const intervention = archive.readouts.filter((r) => r.kind === 'intervention');
    expect(bigq).toHaveLength(10); // 3 arms x 3 seeds + 1 GPU rerun
    expect(intervention).toHaveLength(13); // P x 3 seeds + C000-004 + M1000-004
  });

  it('every intervention entry\'s armBundleSha256 matches the raw scores file\'s per-id record', () => {
    const rawByKey = new Map(rawScores.runs.map((r) => [`${r.id}\u0000${r.trainerSeed}`, r]));
    const interventionEntries = archive.readouts.filter((r) => r.kind === 'intervention');
    expect(interventionEntries.length).toBeGreaterThan(0);
    for (const entry of interventionEntries) {
      const raw = rawByKey.get(`${entry.graphId}\u0000${entry.trainerSeed}`);
      expect(raw, `no raw entry for ${entry.id}`).toBeDefined();
      expect(raw?.armBundleSha256).toBe(entry.armBundleSha256);
    }
  });

  it('every archived entry\'s weightsSha256 is unique (no run archived twice under two different ids)', () => {
    const byWeights = new Map<string, string>();
    for (const entry of archive.readouts) {
      const priorId = byWeights.get(entry.weightsSha256);
      expect(priorId, `${entry.id} shares weightsSha256 with ${String(priorId)}`).toBeUndefined();
      byWeights.set(entry.weightsSha256, entry.id);
    }
    expect(byWeights.size).toBe(archive.readouts.length);
  });

  it('every archived entry\'s theta decodes to bytes matching its own weightsSha256 (self-consistency, all 23 entries)', () => {
    expect(archive.readouts.length).toBeGreaterThan(0);
    for (const entry of archive.readouts) {
      const decoded = Buffer.from(entry.theta, 'base64');
      expect(sha256Hex(decoded), `${entry.id}'s theta does not hash to its own weightsSha256`).toBe(entry.weightsSha256);
    }
  });

  it('the GPU-rerun entry\'s weights actually differ from the CPU run\'s (it is a real rerun, not a copy)', () => {
    const cpu = archive.readouts.find((r) => r.id === 'biological-seed101');
    const gpu = archive.readouts.find((r) => r.id === 'biological-seed101-gpurerun');
    expect(cpu, 'no biological-seed101 entry').toBeDefined();
    expect(gpu, 'no biological-seed101-gpurerun entry').toBeDefined();
    expect(gpu?.weightsSha256).not.toBe(cpu?.weightsSha256);
  });

  it('P\'s per-seed raw scores reproduce the published pathway-interventions-v1.json perSeed scores', () => {
    const pRuns = rawScores.runs.filter((r) => r.id === 'P');
    expect(pRuns).toHaveLength(3);
    for (const run of pRuns) {
      const published = pathwayInterventions.trained.perSeed[String(run.trainerSeed)];
      expect(published, `no published perSeed for trainerSeed ${run.trainerSeed}`).toBeDefined();
      expect(mean(run.movementScore)).toBe(published.score);
    }
  });

  it('C000-C004\'s raw sorted means reproduce the published controls.C.scores array', () => {
    const cMeans = rawScores.runs
      .filter((r) => /^C\d{3}$/.test(r.id))
      .map((r) => mean(r.movementScore))
      .sort((a, b) => a - b);
    expect(cMeans).toHaveLength(5);
    expect(cMeans).toEqual(pathwayInterventions.trained.controls.C.scores);
  });

  it('M1000-M1004\'s raw sorted means reproduce the published controls.M.scores array', () => {
    const mMeans = rawScores.runs
      .filter((r) => /^M1\d{3}$/.test(r.id))
      .map((r) => mean(r.movementScore))
      .sort((a, b) => a - b);
    expect(mMeans).toHaveLength(5);
    expect(mMeans).toEqual(pathwayInterventions.trained.controls.M.scores);
  });

  it('every non-rerun bigq entry\'s weightsSha256 and armBundleSha256 match trained-readout-v1.report.json', () => {
    const bigqEntries = archive.readouts.filter((r) => r.kind === 'bigq' && !r.id.endsWith('-gpurerun'));
    expect(bigqEntries).toHaveLength(9);
    // graphId -> report.json's arm key ("rewired-seed0" archives under the shipped "rewired" arm).
    const reportArmFor = (entry: ArchivedReadout): string => (entry.graphId === 'rewired-seed0' ? 'rewired' : entry.graphId);
    for (const entry of bigqEntries) {
      const armReport = trainedReadoutReport.arms[reportArmFor(entry)];
      expect(armReport, `no report arm for ${entry.graphId}`).toBeDefined();
      expect(armReport.armBundleSha256).toBe(entry.armBundleSha256);
      const replica = armReport.replicas[String(entry.trainerSeed)];
      expect(replica, `no report replica for ${entry.graphId} seed ${entry.trainerSeed}`).toBeDefined();
      expect(replica.weightsSha256).toBe(entry.weightsSha256);
      // weightsSha256 IS recomputable from the archive's own theta, unlike thetaSha256.
      const decoded = Buffer.from(entry.theta, 'base64');
      expect(sha256Hex(decoded)).toBe(entry.weightsSha256);
    }
  });

  it('every bigq entry\'s graphGzipSha256/graphBinarySha256 match malecns-arena-v1.manifest.json (thermo-provenance: path-independent graph identity)', () => {
    const bigqEntries = archive.readouts.filter((r) => r.kind === 'bigq');
    expect(bigqEntries.length).toBeGreaterThan(0);
    for (const entry of bigqEntries) {
      if (entry.graphId === 'biological') {
        expect(entry.graphGzipSha256).toBe(manifest.gzipSha256);
        expect(entry.graphBinarySha256).toBe(manifest.binarySha256);
      } else if (entry.graphId === 'rewired-seed0') {
        expect(entry.graphGzipSha256).toBe(manifest.rewiredArms.seed0.gzipSha256);
        expect(entry.graphBinarySha256).toBe(manifest.rewiredArms.seed0.binarySha256);
      } else if (entry.graphId === 'disconnected') {
        // No separate artifact -- derived at runtime from biological with edgeCount 0.
        expect(entry.graphGzipSha256).toBeNull();
        expect(entry.graphBinarySha256).toBeNull();
      } else {
        throw new Error(`unexpected bigq graphId "${entry.graphId}"`);
      }
    }
  });

  it('every intervention entry\'s graphGzipSha256/graphBinarySha256 match intervention-index-v1.json\'s entry for that graphId', () => {
    const interventionEntries = archive.readouts.filter((r) => r.kind === 'intervention');
    expect(interventionEntries.length).toBeGreaterThan(0);
    const byId = new Map(interventionIndex.entries.map((e) => [e.id, e]));
    for (const entry of interventionEntries) {
      const indexEntry = byId.get(entry.graphId);
      expect(indexEntry, `no intervention-index-v1.json entry for ${entry.graphId}`).toBeDefined();
      expect(entry.graphGzipSha256).toBe(indexEntry?.gzipSha256);
      expect(entry.graphBinarySha256).toBe(indexEntry?.binarySha256);
    }
  });

  it('the GPU-rerun entry is labeled distinctly and matches trained-readout-v1.report.json\'s gpuRerun block', () => {
    const gpuRerun = archive.readouts.find((r) => r.id.endsWith('-gpurerun'));
    expect(gpuRerun).toBeDefined();
    expect(gpuRerun?.graphId).toBe(trainedReadoutReport.gpuRerun.arm);
    expect(gpuRerun?.trainerSeed).toBe(trainedReadoutReport.gpuRerun.trainerSeed);
  });
});
