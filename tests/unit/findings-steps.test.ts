import { readFileSync, readdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { buildFindingSteps, type BuildFindingStepsInputs, type FindingStep } from '../../src/lib/findings/steps';
import type { ArenaManifest } from '../../src/lib/experiment/assets';
import { loadRewiringNull, type RewiringNullArtifact, type RewiringNullLoadResult } from '../../src/lib/experiment/rewiringNull';
import { loadNullExplanation, type NullExplanationArtifact, type NullExplanationLoadResult } from '../../src/lib/experiment/nullExplanation';
import {
  loadPathwayInterventions,
  type PathwayInterventionsArtifact,
  type PathwayInterventionsLoadResult
} from '../../src/lib/experiment/pathwayInterventions';
import { loadRepertoireNull, type RepertoireNullArtifact, type RepertoireNullLoadResult } from '../../src/lib/experiment/repertoireNull';
import { loadTaskGenerality, type TaskGeneralityArtifact, type TaskGeneralityLoadResult } from '../../src/lib/experiment/taskGenerality';
import {
  loadSelectionRobustness,
  type SelectionRobustnessArtifact,
  type SelectionRobustnessLoadResult
} from '../../src/lib/experiment/selectionRobustness';
import {
  loadReadoutAttribution,
  type ReadoutAttributionArtifact,
  type ReadoutAttributionLoadResult
} from '../../src/lib/experiment/readoutAttribution';
import { createPublicDataFetch } from '../helpers/fake-worker';

/**
 * WP1 of `.agents/plans/findings-tour`: unit coverage for
 * `src/lib/findings/steps.ts#buildFindingSteps`. Every step's template is
 * checked against the real, committed artifacts (`public/data/rewiring-null-v1.json`,
 * `null-explanation-v1.json`, `pathway-interventions-v1.json`), loaded and
 * validated through their own real loaders (`loadRewiringNull`/
 * `loadNullExplanation`/`loadPathwayInterventions`) rather than a bare
 * `JSON.parse` — `pathwayInterventions.ts`'s loader transforms the raw
 * `trained.perSeed` JSON field into the validated `trained.perSeedCategory`
 * shape this module's types declare, so a raw-parsed fixture would silently
 * disagree with the real loaded shape. This mirrors
 * `tests/unit/assets-pathway-interventions.test.ts`'s own "real committed
 * artifact" discipline. A missing/unavailable/invalid field makes only that
 * step degrade; every rendered number equals the shared formatter applied
 * to its named source field; a template-lint test fails if a template
 * string contains a bare numeric literal.
 */

const here = dirname(fileURLToPath(import.meta.url));
const publicDataDir = resolve(here, '../../public/data');

const manifest = JSON.parse(readFileSync(resolve(publicDataDir, 'malecns-arena-v1.manifest.json'), 'utf-8')) as ArenaManifest;

let realRewiringNull: RewiringNullArtifact;
let realNullExplanation: NullExplanationArtifact;
let realPathwayInterventions: PathwayInterventionsArtifact;
let realRepertoireNull: RepertoireNullArtifact;
let realTaskGenerality: TaskGeneralityArtifact;
let realSelectionRobustness: SelectionRobustnessArtifact;
let realReadoutAttribution: ReadoutAttributionArtifact;

beforeAll(async () => {
  vi.stubGlobal('fetch', createPublicDataFetch());
  const rewiringNullResult = await loadRewiringNull(manifest, '/data');
  const nullExplanationResult = await loadNullExplanation(manifest, '/data');
  const pathwayInterventionsResult = await loadPathwayInterventions(manifest, '/data');
  const repertoireNullResult = await loadRepertoireNull(manifest, '/data');
  const taskGeneralityResult = await loadTaskGenerality(manifest, '/data');
  const selectionRobustnessResult = await loadSelectionRobustness(manifest, '/data');
  const readoutAttributionResult = await loadReadoutAttribution(manifest, '/data');
  if (rewiringNullResult.status !== 'ok') throw new Error(`Fixture setup: rewiringNull is "${rewiringNullResult.status}"`);
  if (nullExplanationResult.status !== 'ok') throw new Error(`Fixture setup: nullExplanation is "${nullExplanationResult.status}"`);
  if (pathwayInterventionsResult.status !== 'ok') {
    throw new Error(`Fixture setup: pathwayInterventions is "${pathwayInterventionsResult.status}"`);
  }
  if (repertoireNullResult.status !== 'ok') throw new Error(`Fixture setup: repertoireNull is "${repertoireNullResult.status}"`);
  if (taskGeneralityResult.status !== 'ok') throw new Error(`Fixture setup: taskGenerality is "${taskGeneralityResult.status}"`);
  if (selectionRobustnessResult.status !== 'ok') {
    throw new Error(`Fixture setup: selectionRobustness is "${selectionRobustnessResult.status}"`);
  }
  if (readoutAttributionResult.status !== 'ok') {
    throw new Error(`Fixture setup: readoutAttribution is "${readoutAttributionResult.status}"`);
  }
  realRewiringNull = rewiringNullResult.data;
  realNullExplanation = nullExplanationResult.data;
  realPathwayInterventions = pathwayInterventionsResult.data;
  realRepertoireNull = repertoireNullResult.data;
  realTaskGenerality = taskGeneralityResult.data;
  realSelectionRobustness = selectionRobustnessResult.data;
  realReadoutAttribution = readoutAttributionResult.data;
  vi.unstubAllGlobals();
});

afterEach(() => vi.unstubAllGlobals());

const rewiringNullOk = (data?: RewiringNullArtifact): RewiringNullLoadResult => ({ status: 'ok', data: data ?? realRewiringNull });
const nullExplanationOk = (data: NullExplanationArtifact = realNullExplanation): NullExplanationLoadResult => ({
  status: 'ok',
  data
});
const pathwayInterventionsOk = (data: PathwayInterventionsArtifact = realPathwayInterventions): PathwayInterventionsLoadResult => ({
  status: 'ok',
  data
});
const repertoireNullOk = (data: RepertoireNullArtifact = realRepertoireNull): RepertoireNullLoadResult => ({
  status: 'ok',
  data
});
const taskGeneralityOk = (data: TaskGeneralityArtifact = realTaskGenerality): TaskGeneralityLoadResult => ({
  status: 'ok',
  data
});
const selectionRobustnessOk = (data: SelectionRobustnessArtifact = realSelectionRobustness): SelectionRobustnessLoadResult => ({
  status: 'ok',
  data
});
const readoutAttributionOk = (data: ReadoutAttributionArtifact = realReadoutAttribution): ReadoutAttributionLoadResult => ({
  status: 'ok',
  data
});

const baseInputs = (): BuildFindingStepsInputs => ({
  manifest,
  dataBaseUrl: '/data',
  rewiringNull: rewiringNullOk(),
  nullExplanation: nullExplanationOk(),
  pathwayInterventions: pathwayInterventionsOk(),
  repertoireNull: repertoireNullOk(),
  taskGenerality: taskGeneralityOk(),
  selectionRobustness: selectionRobustnessOk(),
  readoutAttribution: readoutAttributionOk()
});

const findStep = (steps: readonly FindingStep[], id: string): FindingStep => {
  const step = steps.find((candidate) => candidate.id === id);
  if (!step) throw new Error(`No step with id "${id}"`);
  return step;
};

describe('buildFindingSteps (against the real committed WP1 artifacts)', () => {
  it('builds exactly ten steps, in evidence-chain order', () => {
    const steps = buildFindingSteps(baseInputs());
    expect(steps.map((step) => step.id)).toEqual([
      'rewiring-null',
      'mirrored-decoder',
      'explanation',
      'intervention',
      'trained-null',
      'trained-interventions',
      'readout-attribution',
      'task-generality',
      'behavior-repertoire',
      'selection-robustness'
    ]);
  });

  it('step 1 (rewiring null) states the authored decoder and the real bioPercentile/null.n', () => {
    const steps = buildFindingSteps(baseInputs());
    const step = findStep(steps, 'rewiring-null');
    expect(step.status).toBe('ok');
    expect(step.condition).toBe('authored');
    expect(step.sentence).toContain('authored');
    expect(step.sentence).toContain(`${realRewiringNull.null.n} degree-preserving rewirings`);
    expect(step.sentence).toContain('0.0th percentile');
    expect(step.sentence).toMatch(/under this model\.$/);
    expect(step.provenance).toHaveLength(1);
    expect(step.provenance[0].sha256Prefix).toBe(manifest.rewiringNull?.sha256.slice(0, 12));
    expect(step.provenance[0].artifactPath).toBe(`/data/${manifest.rewiringNull?.artifact}`);
  });

  it('step 2 (mirrored decoder) states the mirrored-decoder result relative to the un-mirrored baseline, with one provenance entry per source artifact', () => {
    const steps = buildFindingSteps(baseInputs());
    const step = findStep(steps, 'mirrored-decoder');
    expect(step.status).toBe('ok');
    expect(step.sentence).toContain('mirroring');
    // Real data: both baseline and mirrored bioPercentile are 0 -- "still leaves ... at the bottom".
    expect(step.sentence).toContain('still leaves biological at the bottom');
    expect(step.sentence).toMatch(/under this model\.$/);
    // The sentence draws on two artifacts (the rewiring-null baseline and
    // the null-explanation mirrored variant) -- both must be citable.
    expect(step.provenance).toHaveLength(2);
    expect(step.provenance.some((entry) => entry.sha256Prefix === manifest.rewiringNull?.sha256.slice(0, 12))).toBe(true);
    expect(step.provenance.some((entry) => entry.sha256Prefix === manifest.nullExplanation?.sha256.slice(0, 12))).toBe(true);
  });

  it('step 2 shows whichever of its two source artifacts is worse, not always the rewiring-null one', () => {
    // rewiringNull merely "absent" (nothing shipped) vs. nullExplanation
    // actually "invalid" (a real verification failure) -- the failure must
    // not be hidden behind the more benign "absent" status.
    const absentPlusInvalid = buildFindingSteps({
      ...baseInputs(),
      rewiringNull: { status: 'absent', reason: 'The manifest has no rewiringNull artifact entry.' },
      nullExplanation: { status: 'invalid', reason: 'sha256 mismatch' }
    });
    const step1 = findStep(absentPlusInvalid, 'mirrored-decoder');
    expect(step1.status).toBe('invalid');
    expect(step1.reason).toBe('sha256 mismatch');

    // rewiringNull actually "invalid" vs. nullExplanation "ok" -- reachable
    // because loadNullExplanation cross-checks against the manifest's own
    // pinned rewiringNull.sha256, not against the live rewiring-null load's
    // own success.
    const invalidPlusOk = buildFindingSteps({
      ...baseInputs(),
      rewiringNull: { status: 'invalid', reason: 'bins do not sum to null.n' }
    });
    const step2 = findStep(invalidPlusOk, 'mirrored-decoder');
    expect(step2.status).toBe('invalid');
    expect(step2.reason).toBe('bins do not sum to null.n');
    expect(step2.provenance).toHaveLength(2);
  });

  it('step 3 (explanation) lists every qualifying metric with its own formatted rho', () => {
    const steps = buildFindingSteps(baseInputs());
    const step = findStep(steps, 'explanation');
    expect(step.status).toBe('ok');
    for (const metric of realNullExplanation.finding.qualifyingMetrics) {
      expect(step.sentence).toContain(metric.name);
      expect(step.sentence).toContain(`ρ = ${metric.spearman.toFixed(3)}`);
    }
    expect(step.sentence).toMatch(/under this model\.$/);
  });

  it('step 3 appends a regime-invalid clause before "under this model.", for both the has-metrics and zero-metrics templates', () => {
    const regimeInvalidWithMetrics: NullExplanationArtifact = {
      ...realNullExplanation,
      regime: { gatePassed: false },
      finding: { ...realNullExplanation.finding, regimeInvalid: true }
    };
    const withMetrics = findStep(
      buildFindingSteps({ ...baseInputs(), nullExplanation: nullExplanationOk(regimeInvalidWithMetrics) }),
      'explanation'
    );
    expect(withMetrics.sentence).toContain('regime-invalid (inconclusive)');
    expect(withMetrics.sentence).toMatch(/under this model\.$/);

    const regimeInvalidNoMetrics: NullExplanationArtifact = {
      ...realNullExplanation,
      regime: { gatePassed: false },
      finding: { ...realNullExplanation.finding, regimeInvalid: true, qualifyingMetrics: [] }
    };
    const noMetrics = findStep(
      buildFindingSteps({ ...baseInputs(), nullExplanation: nullExplanationOk(regimeInvalidNoMetrics) }),
      'explanation'
    );
    expect(noMetrics.sentence).toContain('regime-invalid (inconclusive)');
    expect(noMetrics.sentence).toMatch(/under this model\.$/);

    // The real shipped artifact's regime gate passed -- the clause must be absent.
    const valid = findStep(buildFindingSteps(baseInputs()), 'explanation');
    expect(valid.sentence).not.toContain('regime-invalid');
    expect(valid.sentence).toMatch(/under this model\.$/);
  });

  it('step 4 (intervention, authored) states the real authored category and channel-specific modifier', () => {
    const steps = buildFindingSteps(baseInputs());
    const step = findStep(steps, 'intervention');
    expect(step.status).toBe('ok');
    expect(step.sentence).toContain(realPathwayInterventions.authored.category);
    expect(step.sentence).toContain('channel-specific modifier holding');
    expect(step.sentence).toMatch(/under this model\.$/);
  });

  it('step 5 (trained null) states the trained decoder and every trainer seed\'s own percentile spread', () => {
    const steps = buildFindingSteps(baseInputs());
    const step = findStep(steps, 'trained-null');
    expect(step.status).toBe('ok');
    expect(step.condition).toBe('trained');
    // Real trained.bioReplicaPercentiles: seeds 101/303 at 0, seed 202 at 0.4 -- a real spread, not a single value.
    expect(step.sentence).toContain('0.0th percentile');
    expect(step.sentence).toContain('40.0th percentile');
    expect(step.sentence).toMatch(/under this model\.$/);
  });

  it('step 6 (trained interventions) follows the fixed sentence pattern and states "does not reproduce" for the real no-specific-effect vs. pathway-supported mismatch', () => {
    const steps = buildFindingSteps(baseInputs());
    const step = findStep(steps, 'trained-interventions');
    expect(step.status).toBe('ok');
    expect(step.condition).toBe('both');
    expect(step.sentence).toContain('all 3 seeds agree:');
    // (thermo review, methodology I1/I2) Rendered through the shared
    // `describeTrainedCategory` helper -- never the raw "no-specific-effect"
    // enum slug -- and, since the real shipped artifact's `trained.note`
    // field discloses the reporting-convention caveat, that caveat text too.
    expect(step.sentence).not.toContain('no-specific-effect');
    expect(step.sentence).toContain('no specific effect (neither pathway-supported nor edge-class');
    expect(step.sentence).toContain('a reporting convention adopted after the trained scores were known, not a predeclared category');
    expect(step.sentence).toContain('does not reproduce');
    expect(step.sentence).not.toContain('this matches');
    expect(step.sentence).toContain(`authored decoder's ${realPathwayInterventions.authored.category} result`);
    expect(step.sentence).toMatch(/under this model\.$/);
  });

  it('step 6 states "matches" when the trained category equals the authored category (robust)', () => {
    const matching: PathwayInterventionsArtifact = {
      ...realPathwayInterventions,
      authored: { ...realPathwayInterventions.authored, category: 'edge-class-effect' },
      trained: {
        trainedRobust: true,
        perSeedCategory: { '101': 'edge-class-effect', '202': 'edge-class-effect', '303': 'edge-class-effect' }
      }
    };
    const steps = buildFindingSteps({ ...baseInputs(), pathwayInterventions: pathwayInterventionsOk(matching) });
    const step = findStep(steps, 'trained-interventions');
    expect(step.sentence).toContain('this matches');
    expect(step.sentence).not.toContain('does not reproduce');
  });

  it('step 6 never claims "matches" when the seeds disagree, even if one seed happens to equal the authored category', () => {
    const disagreeing: PathwayInterventionsArtifact = {
      ...realPathwayInterventions,
      authored: { ...realPathwayInterventions.authored, category: 'pathway-supported' },
      trained: {
        trainedRobust: false,
        perSeedCategory: { '101': 'pathway-supported', '202': 'edge-class-effect', '303': 'no-specific-effect' }
      }
    };
    const steps = buildFindingSteps({ ...baseInputs(), pathwayInterventions: pathwayInterventionsOk(disagreeing) });
    const step = findStep(steps, 'trained-interventions');
    expect(step.sentence).toContain('seeds disagree:');
    expect(step.sentence).toContain('seed 101: pathway-supported');
    expect(step.sentence).toContain('seed 202: edge-class-effect');
    // The dissenting no-specific-effect seed reads through the same
    // human-readable helper, without the reporting-convention caveat (which
    // the per-seed "seeds disagree" listing deliberately omits, so it isn't
    // repeated once per dissenting seed -- see `describeTrainedCategory`'s
    // own doc comment).
    expect(step.sentence).not.toContain('seed 303: no-specific-effect');
    expect(step.sentence).toContain('seed 303: no specific effect (neither pathway-supported nor edge-class)');
    expect(step.sentence).not.toContain('reporting convention');
    expect(step.sentence).toContain('does not reproduce');
  });

  it('step 6 says "is consistent with", never "this matches" or "does not reproduce", when a robust no-specific-effect merges the authored generic-rewiring-effect/not-supported split', () => {
    for (const authoredCategory of ['generic-rewiring-effect', 'not-supported'] as const) {
      const consistent: PathwayInterventionsArtifact = {
        ...realPathwayInterventions,
        authored: { ...realPathwayInterventions.authored, category: authoredCategory },
        trained: {
          trainedRobust: true,
          perSeedCategory: { '101': 'no-specific-effect', '202': 'no-specific-effect', '303': 'no-specific-effect' }
        }
      };
      const step = findStep(
        buildFindingSteps({ ...baseInputs(), pathwayInterventions: pathwayInterventionsOk(consistent) }),
        'trained-interventions'
      );
      expect(step.sentence, authoredCategory).toContain('is consistent with');
      expect(step.sentence, authoredCategory).not.toContain('this matches');
      expect(step.sentence, authoredCategory).not.toContain('does not reproduce');
      expect(step.sentence, authoredCategory).toMatch(/under this model\.$/);
    }
  });

  it('step 6 never appends the reporting-convention caveat when the artifact carries no trained.note field (never unconditional)', () => {
    const noNote: PathwayInterventionsArtifact = {
      ...realPathwayInterventions,
      trained: {
        trainedRobust: true,
        perSeedCategory: { '101': 'no-specific-effect', '202': 'no-specific-effect', '303': 'no-specific-effect' }
        // no `note` field
      }
    };
    const step = findStep(
      buildFindingSteps({ ...baseInputs(), pathwayInterventions: pathwayInterventionsOk(noNote) }),
      'trained-interventions'
    );
    expect(step.sentence).toContain('no specific effect (neither pathway-supported nor edge-class)');
    expect(step.sentence).not.toContain('reporting convention');
  });

  it('the task-generality step states the real authored/trained overall verdicts as a short summary, with per-task detail in details', () => {
    const steps = buildFindingSteps(baseInputs());
    const step = findStep(steps, 'task-generality');
    expect(step.status).toBe('ok');
    expect(step.condition).toBe('both');
    // Real shipped data: authored is general (3 of 4 non-degenerate tasks
    // hold and generalize), trained is task-dependent (no task reaches a
    // robust pathway-supported result).
    expect(realTaskGenerality.overall.authored.verdict).toBe('general');
    expect(realTaskGenerality.overall.trained.verdict).toBe('task-dependent');
    expect(step.sentence).toContain(`${realTaskGenerality.tasks.length} task variants`);
    expect(step.sentence).toContain('authored: general');
    expect(step.sentence).toContain('trained: task-dependent');
    expect(step.sentence).toMatch(/under this model\.$/);
    // Thermo review (Important, both reviewers): the sentence itself must
    // stay short -- no per-task/per-seed enumeration folded in.
    expect(step.sentence?.length).toBeLessThan(400);
    expect(step.sentence).not.toMatch(/seed \d+ (pathway-supported|no specific effect)/);
    // "No specific effect" is defined once, in the sentence, not per task.
    expect(step.sentence).toContain('"no specific effect" means neither pathway-supported nor edge-class holds');
    expect(step.provenance).toHaveLength(1);
    expect(step.provenance[0].sha256Prefix).toBe(manifest.taskGenerality?.sha256.slice(0, 12));

    // Per-task detail: one entry per task, "not robust" and the dissenting
    // seed(s) stated inline, never all three seeds when only one dissents.
    // (A thermo-maintainability review, Important, I1: `perTask`'s separate
    // `{authored, trained}` fields were merged into one `text` field shared
    // with the selection-robustness step's `details` -- the exact same
    // strings are asserted here via `toContain`, just against the merged
    // `authored: X; trained: Y` text instead of two separate fields.)
    expect(step.details).toHaveLength(4);
    const hazardHeavy = step.details?.find((t) => t.id === 'hazard-heavy');
    expect(hazardHeavy?.text).toContain('authored: pathway-supported');
    expect(hazardHeavy?.text).toContain('trained: no specific effect, not robust (seed 202 pathway-supported)');
    const crowded = step.details?.find((t) => t.id === 'crowded');
    expect(crowded?.text).toContain('trained: no specific effect, not robust (seed 303 pathway-supported)');
    const sparseFood = step.details?.find((t) => t.id === 'sparse-food');
    expect(sparseFood?.text).toContain('trained: no specific effect (robust)');
    const noMovement = step.details?.find((t) => t.id === 'no-movement');
    // no-movement's degenerate authored (P/C/M) result stays separate from
    // its independently-valid Q-vs-MQ result -- the step never mentions
    // Q/MQ at all (that detail lives only in the full report), so there is
    // nothing here to conflate.
    expect(noMovement?.text).toContain('authored: degenerate');
  });

  it('the task-generality step states "general" for both authored and trained when every non-degenerate task generalizes/reaches a robust pathway-supported category', () => {
    const general: TaskGeneralityArtifact = {
      version: 1,
      sources: realTaskGenerality.sources,
      tasks: (['hazard-heavy', 'sparse-food', 'no-movement', 'crowded'] as const).map((id) => ({
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
      })),
      overall: {
        authored: { verdict: 'general', nonDegenerateCount: 4, totalCount: 4 },
        trained: { verdict: 'general', nonDegenerateCount: 4, totalCount: 4 }
      }
    };
    const step = findStep(buildFindingSteps({ ...baseInputs(), taskGenerality: taskGeneralityOk(general) }), 'task-generality');
    expect(step.sentence).toContain('authored: general');
    expect(step.sentence).toContain('trained: general');
    expect(step.sentence).not.toContain('not robust');
    // No task uses no-specific-effect here, so the definition clause is absent.
    expect(step.sentence).not.toContain('"no specific effect" means');
    expect(step.details?.every((t) => t.text.includes('authored: pathway-supported'))).toBe(true);
    expect(step.details?.every((t) => t.text.includes('trained: pathway-supported (robust)'))).toBe(true);
  });

  it('the task-generality step lists per-task detail, including a degenerate task, when the authored side is task-dependent', () => {
    const taskDependent: TaskGeneralityArtifact = {
      ...realTaskGenerality,
      tasks: [
        {
          id: 'no-movement',
          categorized: false,
          null: { nullHolds: true, degenerate: false },
          pathway: { category: 'degenerate', generalizes: false },
          trained: { degenerate: true }
        },
        {
          id: 'hazard-heavy',
          categorized: true,
          null: { nullHolds: true, degenerate: false },
          pathway: { category: 'not-supported', generalizes: false },
          trained: {
            degenerate: false,
            trainedRobust: true,
            category: 'no-specific-effect',
            perSeed: { '101': 'no-specific-effect', '202': 'no-specific-effect', '303': 'no-specific-effect' }
          }
        }
      ],
      overall: {
        authored: { verdict: 'task-dependent', nonDegenerateCount: 1, totalCount: 2 },
        trained: { verdict: 'task-dependent', nonDegenerateCount: 1, totalCount: 2 }
      }
    };
    const step = findStep(buildFindingSteps({ ...baseInputs(), taskGenerality: taskGeneralityOk(taskDependent) }), 'task-generality');
    expect(step.sentence).toContain('authored: task-dependent');
    expect(step.sentence).toContain('trained: task-dependent');
    expect(step.sentence).toContain('see per-task detail below');
    const noMovement = step.details?.find((t) => t.id === 'no-movement');
    expect(noMovement?.text).toContain('authored: degenerate');
    const hazardHeavy = step.details?.find((t) => t.id === 'hazard-heavy');
    expect(hazardHeavy?.text).toContain('authored: not-supported');
    expect(hazardHeavy?.text).toContain('trained: no specific effect (robust)');
  });

  it('the task-generality step is "missing" with no sentence when the task-generality artifact has not been published', () => {
    const steps = buildFindingSteps({
      ...baseInputs(),
      taskGenerality: { status: 'missing', reason: 'The manifest has no taskGenerality artifact entry.' }
    });
    const step = findStep(steps, 'task-generality');
    expect(step.status).toBe('missing');
    expect(step.sentence).toBeUndefined();
  });

  it('the task-generality step is "invalid" with the honest reason when the task-generality artifact fails verification, without affecting other steps', () => {
    const steps = buildFindingSteps({
      ...baseInputs(),
      taskGenerality: { status: 'invalid', reason: 'sha256 mismatch' }
    });
    const step = findStep(steps, 'task-generality');
    expect(step.status).toBe('invalid');
    expect(step.reason).toBe('sha256 mismatch');
    expect(findStep(steps, 'rewiring-null').status).toBe('ok');
    expect(findStep(steps, 'behavior-repertoire').status).toBe('ok');
  });

  it('the behavior-repertoire step states the real primary category, occupied count, rewired median, and search-seed robustness', () => {
    const steps = buildFindingSteps(baseInputs());
    const step = findStep(steps, 'behavior-repertoire');
    expect(step.status).toBe('ok');
    expect(step.condition).toBe('both');
    expect(step.sentence).toContain(`biological occupies ${realRepertoireNull.primary.bio.occupied} of 36`);
    expect(step.sentence).toContain(`rewired median of ${realRepertoireNull.primary.rewiredDistribution.occupied.p50}`);
    expect(step.sentence).toContain(realRepertoireNull.primary.category);
    expect(step.sentence).toContain(`search seed ${realRepertoireNull.search.primarySearchSeed}`);
    // Real shipped data (`00-overview.md`'s worked example): the category is
    // not robust across the 5 biological search seeds.
    expect(realRepertoireNull.robustness.robust).toBe(false);
    expect(step.sentence).toContain('not robust across search seeds');
    expect(step.sentence).toMatch(/under this model\.$/);
    expect(step.provenance).toHaveLength(1);
    expect(step.provenance[0].sha256Prefix).toBe(manifest.behaviorRepertoireNull?.sha256.slice(0, 12));
  });

  it('the behavior-repertoire step states "robust across all 5 search seeds" when every seed agrees', () => {
    const robust: RepertoireNullArtifact = {
      ...realRepertoireNull,
      robustness: {
        ...realRepertoireNull.robustness,
        robust: true,
        perSeed: Object.fromEntries(Object.keys(realRepertoireNull.robustness.perSeed).map((seed) => [seed, 'typical']))
      }
    };
    const step = findStep(buildFindingSteps({ ...baseInputs(), repertoireNull: repertoireNullOk(robust) }), 'behavior-repertoire');
    expect(step.sentence).toContain('robust across all 5 search seeds');
    expect(step.sentence).not.toContain('not robust');
  });

  it('the behavior-repertoire step is "missing" with no sentence when the repertoire-null artifact has not been published', () => {
    const steps = buildFindingSteps({
      ...baseInputs(),
      repertoireNull: { status: 'missing', reason: 'The manifest has no behaviorRepertoireNull artifact entry.' }
    });
    const step = findStep(steps, 'behavior-repertoire');
    expect(step.status).toBe('missing');
    expect(step.sentence).toBeUndefined();
  });

  describe('selection-robustness step (against the real committed WP3 artifact)', () => {
    it('states the authored decoder and the real overall verdicts, with per-selection detail listed separately', () => {
      const step = findStep(buildFindingSteps(baseInputs()), 'selection-robustness');
      expect(step.status).toBe('ok');
      expect(step.condition).toBe('authored');
      expect(step.sentence).toContain('authored');
      expect(step.sentence).toMatch(/under this model\.$/);
      expect(step.sentence).toContain(`robust to size is ${realSelectionRobustness.overall.robustToSize.verdict}`);
      expect(step.sentence).toContain(`robust to method is ${realSelectionRobustness.overall.robustToMethod.verdict}`);
      expect(step.sentence).toContain(`the channel-mapping result is ${realSelectionRobustness.overall.mapping.verdict}`);
      expect(step.details).toHaveLength(realSelectionRobustness.selections.length);
      expect(step.provenance).toHaveLength(1);
      expect(step.provenance[0].artifactPath).toContain('selection-robustness-v1.json');
      expect(step.provenance[0].reportPath).toContain('selection-robustness-report.md');
    });

    it('states the null-holds mechanism and pathway degenerate mechanism plainly for every selection, never a bare tag', () => {
      const step = findStep(buildFindingSteps(baseInputs()), 'selection-robustness');
      for (const selection of realSelectionRobustness.selections) {
        const entry = step.details?.find((item) => item.id === selection.id);
        expect(entry).toBeDefined();
        expect(entry?.text).toContain(`null ${selection.null.holds ? 'holds' : 'does not hold'}`);
        expect(entry?.text).toContain(`explanation ${selection.explanation.replicates ? 'replicates' : 'does not replicate'}`);
        if (selection.pathway.cDegenerate || selection.pathway.mDegenerate) {
          // Never a bare "degenerate" tag -- the mechanism is always stated.
          expect(entry?.text).toContain('pathway P/Q degenerate, not categorized');
          expect(entry?.text).toContain(selection.pathway.degenerateMechanism as string);
        }
      }
    });

    it("random-bridge's summary states the null holds narrowly and the explanation does not replicate", () => {
      const step = findStep(buildFindingSteps(baseInputs()), 'selection-robustness');
      const randomBridge = step.details?.find((item) => item.id === 'random-bridge');
      expect(randomBridge?.text).toContain('null holds');
      expect(randomBridge?.text).toContain('explanation does not replicate');
      expect(randomBridge?.text).toContain('pathway P/Q degenerate');
    });

    it("alt-sensory-mapping's summary states the pathway is not-supported despite the null holding and the explanation replicating", () => {
      const step = findStep(buildFindingSteps(baseInputs()), 'selection-robustness');
      const altSensoryMapping = step.details?.find((item) => item.id === 'alt-sensory-mapping');
      expect(altSensoryMapping?.text).toContain('null holds');
      expect(altSensoryMapping?.text).toContain('explanation replicates');
      expect(altSensoryMapping?.text).toContain('pathway not-supported');
    });

    it('is "missing" with no sentence when the selection-robustness artifact has not been published', () => {
      const steps = buildFindingSteps({
        ...baseInputs(),
        selectionRobustness: { status: 'missing', reason: 'The manifest has no selectionRobustness artifact entry.' }
      });
      const step = findStep(steps, 'selection-robustness');
      expect(step.status).toBe('missing');
      expect(step.sentence).toBeUndefined();
      expect(step.details).toBeUndefined();
    });

    /**
     * A thermo-maintainability review (Important, I3) found `verdictWord`'s
     * `'indeterminate'` branch and `selectionSummary`'s `searchLimited`
     * clause were never exercised by any test with synthetic data -- every
     * existing test here runs against the real committed artifact, whose
     * shipped verdicts happen to be `true`/`false`/`false` (never
     * `'indeterminate'`) and whose only degenerate selection (random-bridge)
     * hits the `cDegenerate` branch, never `searchLimited`. This closes that
     * gap on the client-rendering side, mirroring what
     * `tests/unit/selection-report.test.ts` already does on the producer
     * side.
     */
    it('renders the literal word "indeterminate" for an indeterminate overall verdict, and the "pathway search-limited" clause for a search-limited, categorized selection -- never "not-supported"', () => {
      const searchLimitedLarger: SelectionRobustnessArtifact['selections'][number] = {
        ...realSelectionRobustness.selections.find((s) => s.id === 'larger')!,
        pathway: {
          ...realSelectionRobustness.selections.find((s) => s.id === 'larger')!.pathway,
          cDegenerate: false,
          mDegenerate: false,
          searchLimited: true,
          targetReached: false,
          supported: false
        },
        categorized: true,
        categorizedReason: undefined
      };
      const synthetic: SelectionRobustnessArtifact = {
        ...realSelectionRobustness,
        selections: realSelectionRobustness.selections.map((s) => (s.id === 'larger' ? searchLimitedLarger : s)),
        overall: {
          ...realSelectionRobustness.overall,
          robustToSize: {
            verdict: 'indeterminate',
            reason: 'larger: pathway is search-limited (the swap search reached its cap before the transfer target)'
          }
        }
      };
      const step = findStep(buildFindingSteps({ ...baseInputs(), selectionRobustness: selectionRobustnessOk(synthetic) }), 'selection-robustness');
      expect(step.sentence).toContain('robust to size is indeterminate');
      const larger = step.details?.find((item) => item.id === 'larger');
      expect(larger?.text).toContain('pathway search-limited (the swap search reached its cap before the transfer target)');
      expect(larger?.text).not.toContain('pathway not-supported');
      expect(larger?.text).not.toContain('pathway supported');
    });

    it('is "invalid" with the loader\'s own reason when the artifact fails verification', () => {
      const steps = buildFindingSteps({
        ...baseInputs(),
        selectionRobustness: { status: 'invalid', reason: 'sha256 mismatch' }
      });
      const step = findStep(steps, 'selection-robustness');
      expect(step.status).toBe('invalid');
      expect(step.reason).toBe('sha256 mismatch');
    });
  });

  describe('readout-attribution step (against the real committed WP3 artifact)', () => {
    it('states the trained decoder, "under this model", and the real H1-H3 outcomes, hedged, with per-hypothesis detail', () => {
      const step = findStep(buildFindingSteps(baseInputs()), 'readout-attribution');
      expect(step.status).toBe('ok');
      expect(step.condition).toBe('trained');
      expect(step.sentence).toContain('trained decoder');
      expect(step.sentence).toMatch(/under this model\.$/);
      expect(step.sentence).toContain(`(H1: ${realReadoutAttribution.hypotheses.H1.outcome})`);
      expect(step.sentence).toContain(`(H2: ${realReadoutAttribution.hypotheses.H2.outcome})`);
      expect(step.sentence).toContain(`(H3: ${realReadoutAttribution.hypotheses.H3.outcome})`);
      expect(step.details).toHaveLength(4);
      expect(step.details?.map((d) => d.id)).toEqual(['h1', 'h2', 'h3', 'coverage']);
      expect(step.provenance).toHaveLength(1);
      expect(step.provenance[0].artifactPath).toContain('readout-attribution-v1.json');
      expect(step.provenance[0].reportPath).toContain('readout-attribution-report.md');
    });

    /**
     * The real committed artifact's H1-H3 are all `'inconclusive'` (WP2's
     * actual results — `00-overview.md`'s H1/H2/H3 rules never came back
     * `'supported'`/`'not-supported'` on this model's real data). This test
     * asserts that honestly, and — the task's own explicit requirement —
     * that an inconclusive outcome always reads as "inconclusive", never as
     * "not consistent"/"different"/"used more than" (a stronger, false-
     * shaped claim the predeclared rules never license from an inconclusive
     * result).
     */
    it('renders the real committed artifact\'s inconclusive H1-H3 as "inconclusive", never as "not consistent"/"different"/"used more than"', () => {
      expect(realReadoutAttribution.hypotheses.H1.outcome).toBe('inconclusive');
      expect(realReadoutAttribution.hypotheses.H2.outcome).toBe('inconclusive');
      expect(realReadoutAttribution.hypotheses.H3.outcome).toBe('inconclusive');
      const step = findStep(buildFindingSteps(baseInputs()), 'readout-attribution');
      expect(step.sentence).toContain('saliency is inconclusive on whether it is consistent with routing around the missing wiring');
      expect(step.sentence).toContain('the constant-policy share is inconclusive');
      expect(step.sentence).toContain("P's new wiring is inconclusive");
      expect(step.sentence).not.toContain('not consistent with routing around');
      expect(step.sentence).not.toContain('is different between');
      expect(step.sentence).not.toContain("is used more than biological's");
    });

    /**
     * Synthetic `'supported'`/`'not-supported'` fixtures for every
     * hypothesis, so the clause-mapping table itself is exercised end to
     * end, not merely the real (always-inconclusive) data.
     */
    it('renders "supported"/"not-supported" clauses correctly when synthetically forced', () => {
      const synthetic: ReadoutAttributionArtifact = {
        ...realReadoutAttribution,
        hypotheses: {
          ...realReadoutAttribution.hypotheses,
          H1: { outcome: 'supported', evidence: {} },
          H2: { outcome: 'not-supported', evidence: {} },
          H3: { outcome: 'supported', evidence: {} }
        }
      };
      const step = findStep(buildFindingSteps({ ...baseInputs(), readoutAttribution: readoutAttributionOk(synthetic) }), 'readout-attribution');
      expect(step.sentence).toContain('saliency is consistent with routing around the missing wiring (H1: supported)');
      expect(step.sentence).toContain('the constant-policy share is different between biological and rewired (H2: not-supported)');
      expect(step.sentence).toContain("P's new wiring is not used more than biological's (H3: supported)");
    });

    it('includes the reason in the per-hypothesis detail line when the outcome carries one', () => {
      const synthetic: ReadoutAttributionArtifact = {
        ...realReadoutAttribution,
        hypotheses: {
          ...realReadoutAttribution.hypotheses,
          H1: { outcome: 'inconclusive', reason: 'regime-invalid', evidence: {} }
        }
      };
      const step = findStep(buildFindingSteps({ ...baseInputs(), readoutAttribution: readoutAttributionOk(synthetic) }), 'readout-attribution');
      const h1 = step.details?.find((item) => item.id === 'h1');
      expect(h1?.text).toContain('regime-invalid');
    });

    it('is "missing" with no sentence when the readout-attribution artifact has not been published', () => {
      const steps = buildFindingSteps({
        ...baseInputs(),
        readoutAttribution: { status: 'missing', reason: 'The manifest has no readoutAttribution artifact entry.' }
      });
      const step = findStep(steps, 'readout-attribution');
      expect(step.status).toBe('missing');
      expect(step.sentence).toBeUndefined();
    });

    it('is "invalid" with the loader\'s own reason when the artifact fails verification', () => {
      const steps = buildFindingSteps({
        ...baseInputs(),
        readoutAttribution: { status: 'invalid', reason: 'sha256 mismatch' }
      });
      const step = findStep(steps, 'readout-attribution');
      expect(step.status).toBe('invalid');
      expect(step.reason).toBe('sha256 mismatch');
    });
  });

  it('every step is "loading" when its inputs are undefined (the controller has not resolved yet)', () => {
    const steps = buildFindingSteps({
      manifest: undefined,
      dataBaseUrl: '/data',
      rewiringNull: undefined,
      nullExplanation: undefined,
      pathwayInterventions: undefined,
      repertoireNull: undefined,
      taskGenerality: undefined,
      selectionRobustness: undefined,
      readoutAttribution: undefined
    });
    expect(findStep(steps, 'rewiring-null').status).toBe('loading');
    expect(findStep(steps, 'mirrored-decoder').status).toBe('loading');
    expect(findStep(steps, 'explanation').status).toBe('loading');
    expect(findStep(steps, 'intervention').status).toBe('loading');
    expect(findStep(steps, 'trained-null').status).toBe('loading');
    expect(findStep(steps, 'trained-interventions').status).toBe('loading');
    expect(findStep(steps, 'readout-attribution').status).toBe('loading');
    expect(findStep(steps, 'task-generality').status).toBe('loading');
    expect(findStep(steps, 'behavior-repertoire').status).toBe('loading');
    expect(findStep(steps, 'selection-robustness').status).toBe('loading');
  });

  it('maps rewiringNull "absent" onto the step vocabulary\'s "missing", with no sentence', () => {
    const inputs: BuildFindingStepsInputs = {
      ...baseInputs(),
      rewiringNull: { status: 'absent', reason: 'The manifest has no rewiringNull artifact entry.' }
    };
    const steps = buildFindingSteps(inputs);
    expect(findStep(steps, 'rewiring-null').status).toBe('missing');
    expect(findStep(steps, 'rewiring-null').sentence).toBeUndefined();
    // Steps 2 and 5 also depend on rewiringNull being 'ok'.
    expect(findStep(steps, 'mirrored-decoder').status).toBe('missing');
    expect(findStep(steps, 'trained-null').status).toBe('missing');
  });

  it('a rewiringNull "unavailable" only degrades the steps that depend on it, carrying the honest reason', () => {
    const inputs: BuildFindingStepsInputs = {
      ...baseInputs(),
      rewiringNull: { status: 'unavailable', reason: 'network hiccup' }
    };
    const steps = buildFindingSteps(inputs);
    expect(findStep(steps, 'rewiring-null').status).toBe('unavailable');
    expect(findStep(steps, 'rewiring-null').reason).toBe('network hiccup');
    expect(findStep(steps, 'trained-null').status).toBe('unavailable');
    // Steps 3, 4, 6 do not depend on rewiringNull and stay 'ok'.
    expect(findStep(steps, 'explanation').status).toBe('ok');
    expect(findStep(steps, 'intervention').status).toBe('ok');
    expect(findStep(steps, 'trained-interventions').status).toBe('ok');
  });

  it('a nullExplanation "invalid" degrades steps 2 and 3 only, carrying the honest reason', () => {
    const inputs: BuildFindingStepsInputs = {
      ...baseInputs(),
      nullExplanation: { status: 'invalid', reason: 'sha256 mismatch' }
    };
    const steps = buildFindingSteps(inputs);
    expect(findStep(steps, 'mirrored-decoder').status).toBe('invalid');
    expect(findStep(steps, 'mirrored-decoder').reason).toBe('sha256 mismatch');
    expect(findStep(steps, 'explanation').status).toBe('invalid');
    expect(findStep(steps, 'rewiring-null').status).toBe('ok');
    expect(findStep(steps, 'intervention').status).toBe('ok');
  });

  it('a pathwayInterventions "missing" degrades steps 4 and 6 only', () => {
    const inputs: BuildFindingStepsInputs = {
      ...baseInputs(),
      pathwayInterventions: { status: 'missing', reason: 'The manifest has no pathwayInterventions artifact entry.' }
    };
    const steps = buildFindingSteps(inputs);
    expect(findStep(steps, 'intervention').status).toBe('missing');
    expect(findStep(steps, 'trained-interventions').status).toBe('missing');
    expect(findStep(steps, 'rewiring-null').status).toBe('ok');
    expect(findStep(steps, 'explanation').status).toBe('ok');
  });

  it('step 5 is "invalid" when rewiringNull.data.trained fails validateRewiringNullTrained, without failing the rest of the step chain', () => {
    const malformed: RewiringNullArtifact = { ...realRewiringNull, trained: { nonsense: true } };
    const steps = buildFindingSteps({ ...baseInputs(), rewiringNull: rewiringNullOk(malformed) });
    const step = findStep(steps, 'trained-null');
    expect(step.status).toBe('invalid');
    expect(step.reason).toMatch(/trained-sample section/i);
    // Step 1 still renders fine from the same (otherwise valid) artifact.
    expect(findStep(steps, 'rewiring-null').status).toBe('ok');
  });

  it('step 5 is "invalid" when the artifact has no trained section at all', () => {
    const noTrained: RewiringNullArtifact = { ...realRewiringNull, trained: undefined };
    const steps = buildFindingSteps({ ...baseInputs(), rewiringNull: rewiringNullOk(noTrained) });
    expect(findStep(steps, 'trained-null').status).toBe('invalid');
  });

  it('template-lint: no step sentence contains a bare numeric literal not produced by a shared formatter', () => {
    const steps = buildFindingSteps(baseInputs());
    for (const step of steps) {
      if (step.status !== 'ok' || !step.sentence) continue;
      const percentileMentions = step.sentence.match(/\d+(\.\d+)?%/g) ?? [];
      // A bare "NN%" (not "th percentile") would indicate a formatter was
      // bypassed -- the only legitimate "%" text is the static "(0% = lowest
      // score, 100% = highest)" parenthetical, which does not appear in any
      // Findings step sentence (that phrasing is `NullHistogram.svelte`'s
      // own caption, not a Findings step template).
      expect(percentileMentions).toEqual([]);
    }
  });

  /**
   * (dual review, Important) The output-scanning test above only rejects a
   * bare "NN%" — it would pass a template that hard-coded, say, "among 500
   * degree-preserving rewirings" or "rho = 0.412" directly, since neither
   * contains a "%". This lints the *source* of `steps.ts` itself: every
   * string/template literal, with `${...}` interpolations and comments
   * stripped, must contain no digit at all, except the version number
   * inside a provenance label's own artifact filename (e.g.
   * "rewiring-null-v1.json") -- the one place a literal digit is legitimate
   * and unavoidable. This makes `format.ts`'s own doc-comment claim ("a
   * template-lint unit test fails on a numeric literal appearing directly
   * in a template string") actually true.
   */
  /**
   * Extracts every backtick/single-/double-quoted string literal from
   * `source` and returns the ones whose *static* text (interpolations
   * stripped) contains a digit outside an artifact filename's own version
   * number. A single alternation regex, not three independent passes over
   * progressively-blanked text (round-2 dual review, Important — an
   * earlier version ran a separate blank-then-rescan pass per quote style,
   * which silently skipped double-quoted literals entirely and, worse, let
   * an apostrophe inside one un-blanked double-quoted literal desync the
   * single-quote pass for the rest of the file, hiding every literal after
   * it). Matching all three quote styles in one linear scan means no quote
   * style can ever desync another, and none is silently skipped.
   */
  const findNumericLiteralOffenders = (source: string): string[] => {
    const withoutComments = source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '');
    const literalPattern = /`((?:[^`\\]|\\.)*)`|'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)"/g;
    const literals = [...withoutComments.matchAll(literalPattern)].map((match) => match[1] ?? match[2] ?? match[3] ?? '');
    return literals.filter((literal) => {
      const staticText = literal
        .replace(/\$\{[^}]*\}/g, '') // drop interpolations -- their computed values are asserted by the perturbation test below
        .replace(/[a-z][a-z0-9-]*-v\d+\.(json|md)/gi, ''); // allow a provenance label's own "<name>-v1.json"/".md" filename
      return /\d/.test(staticText);
    });
  };

  /**
   * Self-test for the lint helper itself (round-2 dual review, Important):
   * proves it actually catches a hard-coded number in each of the three
   * quote styles `steps.ts` uses, including past an unrelated apostrophe in
   * an earlier double-quoted literal -- exactly the shape of bug the round-2
   * review found in the prior version of this lint.
   */
  it('template-lint helper: catches a hard-coded number in backtick, single-, and double-quoted literals, including past an apostrophe', () => {
    const synthetic = [
      `const a = "the rewired null's range";`,
      `const b = \`among 500 things\`;`,
      `const c = 'also 42 things';`,
      `const d = "and 7 more things";`
    ].join('\n');
    const offenders = findNumericLiteralOffenders(synthetic);
    expect(offenders).toEqual(['among 500 things', 'also 42 things', 'and 7 more things']);
  });

  it('template-lint: steps.ts and every src/lib/findings/steps/*.ts file contain no hard-coded numeric literal outside an artifact filename', () => {
    // A thermo-maintainability review (Important, I2) split the former
    // single `steps.ts` (808 lines) into one orchestrator plus one
    // `src/lib/findings/steps/<study>.ts` file per study -- the actual
    // step-sentence template literals this lint exists to police now live
    // in those per-study files, not in the orchestrator itself, so the
    // scan must follow them there or it would pass trivially forever.
    const stepsDir = resolve(here, '../../src/lib/findings/steps');
    const stepsOrchestratorPath = resolve(here, '../../src/lib/findings/steps.ts');
    const perStudyPaths = readdirSync(stepsDir)
      .filter((name) => name.endsWith('.ts'))
      .map((name) => resolve(stepsDir, name));
    for (const path of [stepsOrchestratorPath, ...perStudyPaths]) {
      const src = readFileSync(path, 'utf-8');
      expect(findNumericLiteralOffenders(src)).toEqual([]);
    }
  });

  /**
   * (dual review, Important) Closes the gap the source lint above cannot:
   * confirms every number actually *tracks* the artifact it is read from,
   * rather than merely being absent as a literal. Perturbs the real
   * fixtures' numeric fields and asserts the new formatted values (not the
   * old ones) appear in the rebuilt sentences.
   */
  it('perturbation: step sentences track perturbed artifact values, not the original shipped numbers', () => {
    const perturbedRewiringNull: RewiringNullArtifact = {
      ...realRewiringNull,
      bioPercentile: 0.256,
      null: { ...realRewiringNull.null, n: 777 }
    };
    const step1 = findStep(
      buildFindingSteps({ ...baseInputs(), rewiringNull: rewiringNullOk(perturbedRewiringNull) }),
      'rewiring-null'
    );
    expect(step1.sentence).toContain('25.6th percentile');
    expect(step1.sentence).toContain('777 degree-preserving rewirings');
    expect(step1.sentence).not.toContain('0.0th percentile');
    expect(step1.sentence).not.toContain('500 degree-preserving rewirings');

    const perturbedTrained = {
      ...(realRewiringNull.trained as Record<string, unknown>),
      bioPercentile: 0.1,
      bioReplicaPercentiles: [
        { trainerSeed: 101, percentile: 0.1 },
        { trainerSeed: 202, percentile: 0.9 },
        { trainerSeed: 303, percentile: 0.1 }
      ]
    };
    const step5 = findStep(
      buildFindingSteps({
        ...baseInputs(),
        rewiringNull: rewiringNullOk({ ...realRewiringNull, trained: perturbedTrained })
      }),
      'trained-null'
    );
    expect(step5.sentence).toContain('10.0th percentile to 90.0th percentile');
    expect(step5.sentence).not.toContain('0.0th percentile to 40.0th percentile');
  });
});
