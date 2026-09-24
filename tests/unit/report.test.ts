import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

import { outputNeuronIndices } from '../../src/lib/connectome/readout';
import { TRACE_SUBSTEPS } from '../../scripts/training/export-traces';
import { runExportArms } from '../../scripts/training/export-arms';
import { resolveReportMdPath, runEvaluate, type EvaluateArgs } from '../../scripts/training/evaluate';
import {
  nearInputIndependentPolicyArms,
  OPPONENT_PARKED_DISCLOSURE,
  renderReportMarkdown,
  structurallyZeroReadoutInputArms,
  type ArmReplicaReport,
  type ArmReport,
  type EvaluationReport
} from '../../scripts/training/report';
import { createTraceGraph } from '../fixtures/trace-graph';
import { writeTinyRunDir } from '../fixtures/trained-readout-run';

/**
 * `docs/trained-readout-report.md` generation (`scripts/training/report.ts`,
 * wired into `runEvaluate`). Uses the same tiny synthetic run-dir fixture
 * as `tests/unit/evaluate.test.ts`'s `buildFixture` (trace graph, one
 * fixture-rewired arm, one replica per arm) rather than a committed golden
 * markdown file: the exact numeric values are an implementation detail of
 * the bootstrap CI computation, not something this test should pin byte-for-
 * byte. Instead this checks structure (every required section heading is
 * present, every table has the expected header row and a data row per
 * arm/replica) and the one place exact wording matters: the "what this does
 * not show" disclosure sentence.
 */

const REQUIRED_HEADINGS = [
  '# Trained-Readout Evaluation Report',
  '## Method',
  '## Parameter accounting',
  '## Results',
  '## Paired differences',
  '## Side-by-side',
  '## Limitations',
  '## What this does not show'
];

const buildFixture = (): { root: string; armsDir: string; runDirs: string[] } => {
  const root = mkdtempSync(join(tmpdir(), 'report-fixture-'));
  const armsRoot = join(root, 'arms');
  const exportResult = runExportArms({ outDir: armsRoot, fixtureRewire: true, fixtureRewireSeed: 3 });

  const graph = createTraceGraph();
  const D = outputNeuronIndices(graph).length;
  const H = 4;

  const runDirs: string[] = [];
  let weightSeed = 1;
  for (const arm of ['biological', 'rewired', 'disconnected'] as const) {
    const dir = join(root, 'runs', `${arm}-101`);
    writeTinyRunDir({ dir, arm, trainerSeed: 101, D, H, substeps: TRACE_SUBSTEPS, weightSeed, includeEnv: true });
    runDirs.push(dir);
    weightSeed += 17;
  }

  return { root, armsDir: exportResult.outDir, runDirs };
};

const baseArgs = (armsDir: string, runDirs: readonly string[], outDir: string): EvaluateArgs => ({
  armsDir,
  runDirs,
  outDir,
  outDirExplicit: true,
  ticks: 40,
  substeps: TRACE_SUBSTEPS,
  heldOutStart: 30001,
  heldOutCount: 4,
  bootstrapResamples: 50,
  bootstrapSeed: 12345
});

describe('runEvaluate: docs/trained-readout-report.md generation', () => {
  it('writes a report markdown file alongside a non-default --out, with every required section', () => {
    const { root, armsDir, runDirs } = buildFixture();
    try {
      const outDir = join(root, 'out');
      const result = runEvaluate(baseArgs(armsDir, runDirs, outDir));
      expect(result.reportMdPath).toBe(join(outDir, 'trained-readout-report.md'));

      const markdown = readFileSync(result.reportMdPath, 'utf8');
      for (const heading of REQUIRED_HEADINGS) {
        expect(markdown, `missing heading: ${heading}`).toContain(`${heading}\n`);
      }
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('states the opponent-parked disclosure verbatim, under "What this does not show"', () => {
    const { root, armsDir, runDirs } = buildFixture();
    try {
      const outDir = join(root, 'out');
      const result = runEvaluate(baseArgs(armsDir, runDirs, outDir));
      const markdown = readFileSync(result.reportMdPath, 'utf8');

      // Verbatim, word for word — this is the exact sentence the plan
      // requires (`.agents/plans/trained-readout/04-authoritative-evaluation-and-artifacts.md`):
      // "must state verbatim that the headline per-arm numbers were
      // measured single-agent with the opponent parked, which differs from
      // the shipped two-agent side-by-side default, and point to the
      // side-by-side table."
      expect(markdown).toContain(
        'The headline per-arm numbers above were measured single-agent with the opponent parked, ' +
          'which differs from the shipped two-agent side-by-side default; see the Side-by-side section ' +
          'of this report for the shipped two-agent condition.'
      );
      // Hardcoded above (not imported) so a future edit to the exported
      // constant can't silently "fix" this test along with the wording;
      // cross-check the exported constant still matches what evaluate.ts
      // actually wrote, too.
      expect(markdown).toContain(OPPONENT_PARKED_DISCLOSURE);

      const disclosureSection = markdown.slice(markdown.indexOf('## What this does not show'));
      expect(disclosureSection).toContain('single-agent');
      expect(disclosureSection).toContain('opponent parked');
      expect(disclosureSection).toContain('side-by-side');

      // The disclosure deliberately says "the Side-by-side section of this
      // report" rather than "above"/"below" (a correctness review round
      // caught the prior wording claiming "below" while Side-by-side
      // actually rendered above this section) — but the section still must
      // actually exist somewhere in the document for the pointer to mean
      // anything.
      expect(markdown.indexOf('## Side-by-side')).toBeGreaterThan(-1);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('has a parameter-accounting table with one row per arm, D/H/parameterCount equal across arms', () => {
    const { root, armsDir, runDirs } = buildFixture();
    try {
      const outDir = join(root, 'out');
      const result = runEvaluate(baseArgs(armsDir, runDirs, outDir));
      const markdown = readFileSync(result.reportMdPath, 'utf8');

      const section = markdown.slice(
        markdown.indexOf('## Parameter accounting'),
        markdown.indexOf('## Results')
      );
      const rows = section
        .split('\n')
        .filter((line) => line.startsWith('| ') && !line.includes('---') && !line.includes('Arm | D | H'));
      expect(rows).toHaveLength(3); // biological, rewired, disconnected

      const parsed = rows.map((row) =>
        row
          .split('|')
          .map((cell) => cell.trim())
          .filter((cell) => cell.length > 0)
      );
      expect(parsed.map((cells) => cells[0]).sort()).toEqual(['biological', 'disconnected', 'rewired']);
      const dValues = new Set(parsed.map((cells) => cells[1]));
      const hValues = new Set(parsed.map((cells) => cells[2]));
      const paramCountValues = new Set(parsed.map((cells) => cells[3]));
      expect(dValues.size).toBe(1);
      expect(hValues.size).toBe(1);
      expect(paramCountValues.size).toBe(1);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('has one results table per arm with authored + trained + silenced rows', () => {
    const { root, armsDir, runDirs } = buildFixture();
    try {
      const outDir = join(root, 'out');
      const result = runEvaluate(baseArgs(armsDir, runDirs, outDir));
      const markdown = readFileSync(result.reportMdPath, 'utf8');

      for (const arm of ['biological', 'rewired', 'disconnected']) {
        expect(markdown).toContain(`### ${arm}`);
      }
      const resultsSection = markdown.slice(markdown.indexOf('## Results'), markdown.indexOf('## Paired differences'));
      expect((resultsSection.match(/\| authored \|/g) ?? []).length).toBe(3);
      expect((resultsSection.match(/\| trained \|/g) ?? []).length).toBe(3);
      expect((resultsSection.match(/\| silenced \|/g) ?? []).length).toBe(3);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('has both trained-side-by-side and authored-side-by-side tables when biological + rewired are both evaluated', () => {
    const { root, armsDir, runDirs } = buildFixture();
    try {
      const outDir = join(root, 'out');
      const result = runEvaluate(baseArgs(armsDir, runDirs, outDir));
      const markdown = readFileSync(result.reportMdPath, 'utf8');

      expect(markdown).toContain('### trained-side-by-side');
      expect(markdown).toContain('### authored-side-by-side');
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('running twice produces byte-identical report markdown', () => {
    const { root, armsDir, runDirs } = buildFixture();
    try {
      const outA = join(root, 'out-a');
      const outB = join(root, 'out-b');
      const resultA = runEvaluate(baseArgs(armsDir, runDirs, outA));
      const resultB = runEvaluate(baseArgs(armsDir, runDirs, outB));

      const mdA = readFileSync(resultA.reportMdPath);
      const mdB = readFileSync(resultB.reportMdPath);
      expect(mdB.equals(mdA)).toBe(true);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('writes alongside a non-default --out (not docs/), so this suite never touches docs/', () => {
    const { root, armsDir, runDirs } = buildFixture();
    try {
      const outDir = join(root, 'custom-out');
      const result = runEvaluate(baseArgs(armsDir, runDirs, outDir));
      expect(result.reportMdPath).not.toContain('docs/trained-readout-report.md');
      expect(result.reportMdPath).toBe(join(outDir, 'trained-readout-report.md'));
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  // resolveReportMdPath is exercised directly (no file writes) so both
  // branches — including the docs/ default, which no test may ever
  // actually write to — are covered without touching the real docs/ tree.
  describe('resolveReportMdPath (pure; no file I/O)', () => {
    const args = (overrides: Partial<EvaluateArgs>): EvaluateArgs => ({
      runDirs: ['irrelevant'],
      outDir: 'public/data',
      outDirExplicit: true,
      ticks: 1,
      substeps: 1,
      heldOutStart: 1,
      heldOutCount: 1,
      bootstrapResamples: 1,
      bootstrapSeed: 1,
      ...overrides
    });

    it('defaults to docs/trained-readout-report.md when --out resolves to the default public/data', () => {
      expect(resolveReportMdPath(args({ outDir: 'public/data' }))).toBe(
        join(process.cwd(), 'docs', 'trained-readout-report.md')
      );
    });

    it('defaults to alongside --out when --out is not the default public/data', () => {
      expect(resolveReportMdPath(args({ outDir: 'training/runs/scratch' }))).toBe(
        join(process.cwd(), 'training', 'runs', 'scratch', 'trained-readout-report.md')
      );
    });

    it('--report-md overrides unconditionally, even when --out is public/data', () => {
      expect(resolveReportMdPath(args({ outDir: 'public/data', reportMdPath: 'somewhere/else.md' }))).toBe(
        join(process.cwd(), 'somewhere', 'else.md')
      );
    });
  });

  it('--report-md overrides the default path unconditionally', () => {
    const { root, armsDir, runDirs } = buildFixture();
    try {
      const outDir = join(root, 'out');
      const explicitPath = join(root, 'custom-report.md');
      const result = runEvaluate({ ...baseArgs(armsDir, runDirs, outDir), reportMdPath: explicitPath });
      expect(result.reportMdPath).toBe(explicitPath);
      expect(readFileSync(explicitPath, 'utf8')).toContain('# Trained-Readout Evaluation Report');
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});

/**
 * `00-overview.md`'s risk / `05-production-run.md`'s WP5 acceptance risk:
 * "If every trained arm scores within the authored decoder's CI and the
 * silenced control matches trained, the report must state that the readout
 * learned a near-input-independent policy." Exercised directly against
 * constructed report fixtures (not a real CEM run, whose trained/authored/
 * silenced scores can't be steered to a specific relationship) so both the
 * "true" and "false" cases are deterministic.
 */
describe('nearInputIndependentPolicyArms / the near-input-independent finding section', () => {
  const stats = (mean: number, ci95: readonly [number, number]) => ({ n: 4, mean, median: mean, std: 0.1, ci95 });
  const paired = (meanDifference: number, ci95: readonly [number, number]) => ({ n: 4, meanDifference, ci95 });

  const replica = (overrides: Partial<ArmReplicaReport> = {}): ArmReplicaReport => ({
    H: 4,
    parameterCount: 100,
    weightsSha256: 'deadbeef',
    env: null,
    trained: stats(10, [8, 12]),
    silenced: stats(10, [8, 12]),
    pairedTrainedVsAuthored: paired(0, [-1, 1]),
    pairedTrainedVsSilenced: paired(0, [-0.5, 0.5]), // includes 0: "silenced matches trained"
    ...overrides
  });

  const armReport = (
    replicas: Readonly<Record<string, ArmReplicaReport>>,
    authoredCi: readonly [number, number] = [8, 12],
    structurallyZeroInput = false
  ): ArmReport => ({
    D: 6,
    provenance: undefined,
    armBundleSha256: undefined,
    authored: stats(10, authoredCi), // trained's mean (10) falls inside [8, 12]
    replicas,
    structurallyZeroInput
  });

  const baseEvaluation: EvaluationReport['evaluation'] = {
    ticks: 40,
    substeps: 4,
    heldOutSeeds: { start: 30001, count: 4 },
    bootstrap: { resamples: 50, seed: 1 },
    opponentParked: true
  };

  const report = (arms: Readonly<Record<string, ArmReport>>): EvaluationReport => ({
    formatVersion: 1,
    graph: { source: 'trace-graph-fixture', path: null, sha256: 'x' },
    evaluation: baseEvaluation,
    arms,
    armPairs: [],
    sideBySide: [],
    warnings: []
  });

  it('flags an arm as near-input-independent when trained is within authored CI and silenced matches trained', () => {
    const r = report({ biological: armReport({ '101': replica() }) });
    expect(nearInputIndependentPolicyArms(r)).toEqual(['biological']);
  });

  it('does not flag an arm when trained falls outside the authored CI', () => {
    const r = report({
      biological: armReport({ '101': replica({ trained: stats(50, [45, 55]) }) }, [8, 12])
    });
    expect(nearInputIndependentPolicyArms(r)).toEqual([]);
  });

  it('does not flag an arm when the trained-vs-silenced CI excludes zero (a real difference)', () => {
    const r = report({
      biological: armReport({ '101': replica({ pairedTrainedVsSilenced: paired(5, [3, 7]) }) })
    });
    expect(nearInputIndependentPolicyArms(r)).toEqual([]);
  });

  it('requires every replica of an arm to qualify, not just one', () => {
    const r = report({
      biological: armReport({
        '101': replica(),
        '202': replica({ pairedTrainedVsSilenced: paired(5, [3, 7]) })
      })
    });
    expect(nearInputIndependentPolicyArms(r)).toEqual([]);
  });

  it('renders the "every arm" finding statement when all evaluated arms qualify', () => {
    const r = report({
      biological: armReport({ '101': replica() }),
      rewired: armReport({ '101': replica() })
    });
    const markdown = renderReportMarkdown(r);
    expect(markdown).toContain('## Finding: near-input-independent policy');
    expect(markdown).toContain('learned a near-input-independent policy for every arm');
    expect(markdown).not.toMatch(/\bbetter\b|\bsuperior\b|\boutperform/i);
  });

  it('renders the partial-arm finding statement when only some evaluated arms qualify', () => {
    const r = report({
      biological: armReport({ '101': replica() }),
      rewired: armReport({ '101': replica({ trained: stats(50, [45, 55]) }) }, [8, 12])
    });
    const markdown = renderReportMarkdown(r);
    expect(markdown).toContain('## Finding: near-input-independent policy');
    expect(markdown).toContain('but not every evaluated arm');
    expect(markdown).toContain('biological');
  });

  it('omits the finding section entirely when no arm qualifies', () => {
    const r = report({
      biological: armReport({ '101': replica({ trained: stats(50, [45, 55]) }) }, [8, 12])
    });
    const markdown = renderReportMarkdown(r);
    expect(markdown).not.toContain('## Finding: near-input-independent policy');
  });

  // structurallyZeroReadoutInputArms: distinct from nearInputIndependentPolicyArms
  // above — it fires purely on an *exact* trained-vs-silenced identity
  // (paired difference 0, CI exactly [0, 0]), regardless of how `trained`
  // compares to `authored`. The real disconnected-arm production run hit
  // exactly this case: trained clearly beat authored (no CI overlap), so
  // nearInputIndependentPolicyArms does NOT flag it, but trained/silenced
  // were bit-identical because the zero-edge graph never moves the
  // output-assigned neurons' rates away from zero.
  it('flags an arm whose trained/silenced paired difference is exactly [0, 0] AND the graph structurally guarantees it, even when trained clearly beats authored', () => {
    const r = report({
      disconnected: armReport(
        { '101': replica({ trained: stats(35, [33, 37]), pairedTrainedVsSilenced: paired(0, [0, 0]) }) },
        [-2, -1], // authored's CI: nowhere near trained's 35 — nearInputIndependentPolicyArms must NOT fire
        true // graphGuaranteesZeroReadoutInput: true for this arm's graph
      )
    });
    expect(nearInputIndependentPolicyArms(r)).toEqual([]);
    expect(structurallyZeroReadoutInputArms(r)).toEqual(['disconnected']);

    const markdown = renderReportMarkdown(r);
    expect(markdown).not.toContain('## Finding: near-input-independent policy');
    expect(markdown).toContain('## Finding: readout input was structurally zero');
    expect(markdown).toContain('disconnected');

    // Scoped to the new finding's own section, not the whole document: the
    // boilerplate "What this does not show" disclosure legitimately
    // contains the word "above" ("the headline per-arm numbers above"),
    // which is a document-position reference, not a superiority claim —
    // checking the whole markdown would false-positive on that.
    const findingSection = markdown.slice(
      markdown.indexOf('## Finding: readout input was structurally zero'),
      markdown.indexOf('## What this does not show')
    );
    expect(findingSection).not.toMatch(/\bbetter\b|\bsuperior\b|\boutperform|\bbeats\b|\babove\b/i);
  });

  it('does NOT flag an arm whose trained/silenced scores are empirically identical but the graph does not structurally guarantee a zero input (e.g. saturated units)', () => {
    // structurallyZeroInput defaults to false: the topological check did not
    // hold (real edges, or an output neuron directly input-channel-mapped),
    // so an empirical [0, 0] paired-difference CI alone must not be enough
    // to claim the input was structurally zero.
    const r = report({
      biological: armReport({ '101': replica({ pairedTrainedVsSilenced: paired(0, [0, 0]) }) })
    });
    expect(structurallyZeroReadoutInputArms(r)).toEqual([]);
    const markdown = renderReportMarkdown(r);
    expect(markdown).not.toContain('## Finding: readout input was structurally zero');
  });

  it('does not flag an arm whose trained/silenced CI is merely close to zero (not exactly [0, 0]), even if structurally guaranteed', () => {
    const r = report({
      disconnected: armReport(
        { '101': replica({ pairedTrainedVsSilenced: paired(0.001, [-0.5, 0.5]) }) },
        [8, 12],
        true
      )
    });
    expect(structurallyZeroReadoutInputArms(r)).toEqual([]);
  });

  it('requires every replica of an arm to be exactly [0, 0], not just one', () => {
    const r = report({
      disconnected: armReport(
        {
          '101': replica({ pairedTrainedVsSilenced: paired(0, [0, 0]) }),
          '202': replica({ pairedTrainedVsSilenced: paired(3, [1, 5]) })
        },
        [8, 12],
        true
      )
    });
    expect(structurallyZeroReadoutInputArms(r)).toEqual([]);
  });
});
