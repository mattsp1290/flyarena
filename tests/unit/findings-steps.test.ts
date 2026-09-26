import { readFileSync } from 'node:fs';
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

beforeAll(async () => {
  vi.stubGlobal('fetch', createPublicDataFetch());
  const rewiringNullResult = await loadRewiringNull(manifest, '/data');
  const nullExplanationResult = await loadNullExplanation(manifest, '/data');
  const pathwayInterventionsResult = await loadPathwayInterventions(manifest, '/data');
  if (rewiringNullResult.status !== 'ok') throw new Error(`Fixture setup: rewiringNull is "${rewiringNullResult.status}"`);
  if (nullExplanationResult.status !== 'ok') throw new Error(`Fixture setup: nullExplanation is "${nullExplanationResult.status}"`);
  if (pathwayInterventionsResult.status !== 'ok') {
    throw new Error(`Fixture setup: pathwayInterventions is "${pathwayInterventionsResult.status}"`);
  }
  realRewiringNull = rewiringNullResult.data;
  realNullExplanation = nullExplanationResult.data;
  realPathwayInterventions = pathwayInterventionsResult.data;
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

const baseInputs = (): BuildFindingStepsInputs => ({
  manifest,
  dataBaseUrl: '/data',
  rewiringNull: rewiringNullOk(),
  nullExplanation: nullExplanationOk(),
  pathwayInterventions: pathwayInterventionsOk()
});

const findStep = (steps: readonly FindingStep[], id: string): FindingStep => {
  const step = steps.find((candidate) => candidate.id === id);
  if (!step) throw new Error(`No step with id "${id}"`);
  return step;
};

describe('buildFindingSteps (against the real committed WP1 artifacts)', () => {
  it('builds exactly seven steps, in evidence-chain order', () => {
    const steps = buildFindingSteps(baseInputs());
    expect(steps.map((step) => step.id)).toEqual([
      'rewiring-null',
      'mirrored-decoder',
      'explanation',
      'intervention',
      'trained-null',
      'trained-interventions',
      'behavior-repertoire'
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
    expect(step.sentence).toContain('no-specific-effect');
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
    expect(step.sentence).toContain('seed 303: no-specific-effect');
    expect(step.sentence).toContain('does not reproduce');
  });

  it('step 7 (behavior repertoire) is always "missing" with "Not yet published" in this WP (repertoire-null has not landed)', () => {
    const steps = buildFindingSteps(baseInputs());
    const step = findStep(steps, 'behavior-repertoire');
    expect(step.status).toBe('missing');
    expect(step.sentence).toBeUndefined();
    expect(step.provenance).toHaveLength(0);
  });

  it('every step is "loading" when its inputs are undefined (the controller has not resolved yet)', () => {
    const steps = buildFindingSteps({
      manifest: undefined,
      dataBaseUrl: '/data',
      rewiringNull: undefined,
      nullExplanation: undefined,
      pathwayInterventions: undefined
    });
    expect(findStep(steps, 'rewiring-null').status).toBe('loading');
    expect(findStep(steps, 'mirrored-decoder').status).toBe('loading');
    expect(findStep(steps, 'explanation').status).toBe('loading');
    expect(findStep(steps, 'intervention').status).toBe('loading');
    expect(findStep(steps, 'trained-null').status).toBe('loading');
    expect(findStep(steps, 'trained-interventions').status).toBe('loading');
    // Step 7 is always "missing" regardless of loading state -- there is no loader for it in this WP.
    expect(findStep(steps, 'behavior-repertoire').status).toBe('missing');
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
  it('template-lint: steps.ts source contains no hard-coded numeric literal outside an artifact filename', () => {
    const stepsSourcePath = resolve(here, '../../src/lib/findings/steps.ts');
    const src = readFileSync(stepsSourcePath, 'utf-8')
      .replace(/\/\*[\s\S]*?\*\//g, '') // strip block comments (doc prose like "WP1"/"00-overview.md")
      .replace(/\/\/.*$/gm, ''); // strip line comments
    // Backtick literals are matched first, and directly (an apostrophe
    // inside one, e.g. "the rewired null's range", is not a string
    // delimiter) -- backtick literals are then blanked out of `src` before
    // the single-quote pass runs, so that same apostrophe can never be
    // misread as the start of a single-quoted string spanning into
    // surrounding real code.
    const backtickLiterals = [...src.matchAll(/`((?:[^`\\]|\\.)*)`/g)].map((match) => match[1]);
    const srcWithoutBackticks = src.replace(/`(?:[^`\\]|\\.)*`/g, '``');
    const singleQuoteLiterals = [...srcWithoutBackticks.matchAll(/'((?:[^'\\]|\\.)*)'/g)].map((match) => match[1]);
    const literals = [...backtickLiterals, ...singleQuoteLiterals];
    const offenders = literals.filter((literal) => {
      const staticText = literal
        .replace(/\$\{[^}]*\}/g, '') // drop interpolations -- their computed values are asserted by the perturbation test below
        .replace(/[a-z][a-z0-9-]*-v\d+\.(json|md)/gi, ''); // allow a provenance label's own "<name>-v1.json"/".md" filename
      return /\d/.test(staticText);
    });
    expect(offenders).toEqual([]);
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
