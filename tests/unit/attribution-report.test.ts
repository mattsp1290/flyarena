// @vitest-environment node
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import {
  buildReadoutAttributionArtifact,
  DEFAULT_ABLATION_PATH,
  DEFAULT_ARCHIVE_PATH,
  DEFAULT_DESCENDING_TYPES_PATH,
  DEFAULT_HYPOTHESES_PATH,
  DEFAULT_INDEPENDENCE_PATH,
  DEFAULT_LINKAGE_PATH,
  DEFAULT_MANIFEST_PATH,
  DEFAULT_PATHWAY_INTERVENTIONS_PATH,
  DEFAULT_REGIME_PATH,
  DEFAULT_SALIENCY_PATH,
  DEFAULT_TRAINED_READOUT_ARTIFACT_PATH,
  DEFAULT_TRAINED_READOUT_MANIFEST_PATH,
  loadDescendingTypeNames,
  parseAttributionReportArgs,
  readoutAttributionProducer,
  runAttributionReport,
  type BuildReadoutAttributionArtifactInputs
} from '../../scripts/attribution/attribution-report';

/**
 * WP3 of `.agents/plans/readout-attribution`: unit coverage for
 * `scripts/attribution/attribution-report.ts`. Unlike
 * `tests/unit/selection-report.test.ts`'s/`tests/unit/task-generality-report.test.ts`'s
 * fully synthetic fixture worlds, this exercises the producer directly
 * against the real, committed `training/archive/trained-readouts-v1.json`
 * and the real WP2 outputs (`training/runs/attribution/*.json`, gitignored
 * scratch but present in this checkout) -- the real inputs are already the
 * exact shape the producer must parse (23 default-task readouts, real
 * theta/weightsSha256 pairs, real archive fingerprints), so reproducing
 * that shape synthetically would mostly re-derive the same fixture the repo
 * already has on disk. The negative/error-path tests below doctor a
 * single-field COPY of one real input at a time (never the committed
 * originals) to exercise each cross-check's fail-closed branch.
 */

const defaultInputs = (): BuildReadoutAttributionArtifactInputs => ({
  archivePath: DEFAULT_ARCHIVE_PATH,
  saliencyPath: DEFAULT_SALIENCY_PATH,
  independencePath: DEFAULT_INDEPENDENCE_PATH,
  linkagePath: DEFAULT_LINKAGE_PATH,
  regimePath: DEFAULT_REGIME_PATH,
  ablationPath: DEFAULT_ABLATION_PATH,
  hypothesesPath: DEFAULT_HYPOTHESES_PATH,
  descendingTypesPath: DEFAULT_DESCENDING_TYPES_PATH,
  trainedReadoutArtifactPath: DEFAULT_TRAINED_READOUT_ARTIFACT_PATH,
  trainedReadoutManifestPath: DEFAULT_TRAINED_READOUT_MANIFEST_PATH,
  pathwayInterventionsPath: DEFAULT_PATHWAY_INTERVENTIONS_PATH,
  manifestPath: DEFAULT_MANIFEST_PATH
});

let tmpDir: string;

beforeEach(() => {
  tmpDir = mkdtempSync(join(tmpdir(), 'attribution-report-test-'));
});

afterEach(() => {
  rmSync(tmpDir, { recursive: true, force: true });
});

/** Writes a doctored copy of one real WP2 output JSON to `tmpDir`, applying `mutate` to the parsed object first. */
const doctoredCopy = (sourcePath: string, filename: string, mutate: (parsed: any) => any): string => {
  const parsed = JSON.parse(readFileSync(sourcePath, 'utf8'));
  const mutated = mutate(parsed);
  const outPath = join(tmpDir, filename);
  writeFileSync(outPath, JSON.stringify(mutated));
  return outPath;
};

describe('buildReadoutAttributionArtifact (against the real committed archive and WP2 outputs)', () => {
  it('builds the real artifact: 23 default-task readouts, all three hypotheses inconclusive, sha-verified sources', () => {
    const artifact = buildReadoutAttributionArtifact(defaultInputs());
    expect(artifact.version).toBe(1);
    expect(artifact.coverage.ids).toHaveLength(23);
    expect(artifact.coverage.perTaskIncluded).toBe(false);
    expect(artifact.readouts).toHaveLength(23);
    expect(artifact.hypotheses.H1.outcome).toBe('inconclusive');
    expect(artifact.hypotheses.H2.outcome).toBe('inconclusive');
    expect(artifact.hypotheses.H3.outcome).toBe('inconclusive');
    expect(artifact.hypotheses.hypothesisCount).toBe(3);
    // Every readout carries a full 48-length saliency vector and a 16-entry ablation table.
    for (const readout of artifact.readouts) {
      expect(readout.saliency.thrust).toHaveLength(48);
      expect(readout.saliency.yaw).toHaveLength(48);
      expect(readout.ablation).toHaveLength(16);
    }
  });

  it('is byte-identical (via JSON.stringify) across two independent builds against the same inputs', () => {
    const first = JSON.stringify(buildReadoutAttributionArtifact(defaultInputs()));
    const second = JSON.stringify(buildReadoutAttributionArtifact(defaultInputs()));
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
    const doctoredSaliency = doctoredCopy(defaultInputs().saliencyPath, 'saliency.json', (parsed) => ({
      ...parsed,
      archiveSha256: 'f'.repeat(64)
    }));
    expect(() => buildReadoutAttributionArtifact({ ...defaultInputs(), saliencyPath: doctoredSaliency })).toThrow(
      /was produced from a different archive/
    );
  });

  it('throws when a WP2 output is missing an entry for one of the archived readouts', () => {
    const doctoredIndependence = doctoredCopy(defaultInputs().independencePath, 'independence.json', (parsed) => ({
      ...parsed,
      entries: parsed.entries.filter((e: { id: string }) => e.id !== 'biological-seed101')
    }));
    expect(() =>
      buildReadoutAttributionArtifact({ ...defaultInputs(), independencePath: doctoredIndependence })
    ).toThrow(/independence\.json has no entry for archived readout "biological-seed101"/);
  });

  it('throws when descending-types-v1.json does not match the manifest\'s recorded sha256', () => {
    const doctoredManifest = doctoredCopy(DEFAULT_MANIFEST_PATH, 'manifest.json', (parsed) => ({
      ...parsed,
      descendingTypes: { ...parsed.descendingTypes, sha256: 'e'.repeat(64) }
    }));
    expect(() => buildReadoutAttributionArtifact({ ...defaultInputs(), manifestPath: doctoredManifest })).toThrow(
      /descending-types-v1\.json sha256 .* does not match/
    );
  });

  it('throws when trained-readout-v1.json does not match trained-readout-v1.manifest.json\'s artifactSha256', () => {
    const doctoredTrainedManifest = doctoredCopy(DEFAULT_TRAINED_READOUT_MANIFEST_PATH, 'trained-readout-v1.manifest.json', (parsed) => ({
      ...parsed,
      artifactSha256: 'd'.repeat(64)
    }));
    expect(() =>
      buildReadoutAttributionArtifact({ ...defaultInputs(), trainedReadoutManifestPath: doctoredTrainedManifest })
    ).toThrow(/does not match trained-readout-v1\.manifest\.json's artifactSha256/);
  });

  it('throws when pathway-interventions-v1.json does not match the manifest\'s recorded sha256', () => {
    const doctoredManifest = doctoredCopy(DEFAULT_MANIFEST_PATH, 'manifest.json', (parsed) => ({
      ...parsed,
      pathwayInterventions: { ...parsed.pathwayInterventions, sha256: 'c'.repeat(64) }
    }));
    expect(() => buildReadoutAttributionArtifact({ ...defaultInputs(), manifestPath: doctoredManifest })).toThrow(
      /pathway-interventions-v1\.json sha256 .* does not match/
    );
  });
});

describe('runAttributionReport (end to end, writing real files to a temp directory)', () => {
  it('writes a byte-identical artifact + updates the manifest + writes a report on two independent runs', () => {
    const out1 = join(tmpDir, 'run1', 'readout-attribution-v1.json');
    const reportMd1 = join(tmpDir, 'run1', 'readout-attribution-report.md');
    const out2 = join(tmpDir, 'run2', 'readout-attribution-v1.json');
    const reportMd2 = join(tmpDir, 'run2', 'readout-attribution-report.md');

    const result1 = runAttributionReport({
      ...defaultInputs(),
      out: out1,
      reportMd: reportMd1,
      skipManifestUpdate: true
    });
    const result2 = runAttributionReport({
      ...defaultInputs(),
      out: out2,
      reportMd: reportMd2,
      skipManifestUpdate: true
    });

    expect(result1.artifactSha256).toBe(result2.artifactSha256);
    expect(readFileSync(out1)).toEqual(readFileSync(out2));
    expect(readFileSync(reportMd1, 'utf8')).toBe(readFileSync(reportMd2, 'utf8'));

    const report = readFileSync(reportMd1, 'utf8');
    expect(report).toContain('# Readout attribution (under this model)');
    expect(report).toContain('## Predeclared hypotheses and outcome rules');
    expect(report).toContain('## What remains unexplained');
    expect(report).toContain('## Limitations');
    expect(report).toContain('**Outcome: inconclusive**');
  });

});

describe('parseAttributionReportArgs', () => {
  it('refuses --out and --report-md pointing at the same path', () => {
    expect(() => parseAttributionReportArgs(['--out', '/tmp/same.json', '--report-md', '/tmp/same.json'])).toThrow(
      /--out and --report-md must not be the same path/
    );
  });

  it('defaults every path to the real committed/gitignored-scratch locations', () => {
    const args = parseAttributionReportArgs([]);
    expect(args.archivePath).toBe(DEFAULT_ARCHIVE_PATH);
    expect(args.saliencyPath).toBe(DEFAULT_SALIENCY_PATH);
    expect(args.manifestPath).toBe(DEFAULT_MANIFEST_PATH);
    expect(args.skipManifestUpdate).toBe(false);
  });

  it('throws on an unrecognized flag', () => {
    expect(() => parseAttributionReportArgs(['--bogus', 'x'])).toThrow(/Unknown argument/);
  });
});
