import { P_TRAINER_SEEDS, type PathwayInterventionsTrainedCategory } from '../../experiment/pathwayInterventions';
import { describeTrainedCategory } from '../format';
import { provenanceFor, reasonFor, sidecarStepStatus, type BuildFindingStepsInputs, type FindingStep } from './shared';

// ---------------------------------------------------------------------------
// Step 6: Trained interventions
// ---------------------------------------------------------------------------

export const buildTrainedInterventionsStep = (inputs: BuildFindingStepsInputs): FindingStep => {
  const provenance = provenanceFor(
    inputs.manifest,
    inputs.manifest?.pathwayInterventions,
    'Pathway-interventions result, trained section (pathway-interventions-v1.json)',
    'pathway-interventions-report.md',
    inputs.dataBaseUrl
  );
  const base = {
    id: 'trained-interventions',
    title: 'Trained interventions',
    condition: 'both' as const,
    provenance: provenance ? [provenance] : []
  };
  const status = sidecarStepStatus(inputs.pathwayInterventions);
  if (status !== 'ok' || inputs.pathwayInterventions?.status !== 'ok') {
    return { ...base, status, reason: reasonFor(status, inputs.pathwayInterventions) };
  }
  const { authored, trained } = inputs.pathwayInterventions.data;

  // Fixed sentence pattern (`01-findings-panel.md`): "With trained readouts,
  // {all N seeds agree: | seeds disagree:} {trained category} — {this does
  // not reproduce | this matches | this is consistent with} the authored
  // decoder's {authored category} result, under this model." The category
  // *values* are always shown (never only `trainedRobust`), so a robust
  // *absence* of effect (e.g. `no-specific-effect`) can never read as
  // confirmation — and when the seeds disagree, there is no single
  // robustly-reproduced category, so neither "matches" nor "is consistent
  // with" is ever claimed in that branch.
  const agreementPrefix = trained.trainedRobust ? `all ${P_TRAINER_SEEDS.length} seeds agree:` : 'seeds disagree:';
  // (thermo review, methodology I1/I2) Rendered through `describeTrainedCategory`
  // (`../format.ts`), never the raw enum slug — `'no-specific-effect'` reads
  // as "no effect" to an unprimed reader, when it actually means the
  // trained side's predeclared rules cannot decide between
  // `generic-rewiring-effect`/`not-supported`. The robust branch also
  // carries the short "reporting convention" caveat, but only when the
  // artifact's own `trained.note` field actually discloses it (never
  // unconditionally) — the per-seed "seeds disagree" listing omits it so a
  // dissenting no-specific-effect seed doesn't repeat the same caveat.
  const categoryText = trained.trainedRobust
    ? describeTrainedCategory(trained.perSeedCategory[P_TRAINER_SEEDS[0]], { withCaveat: Boolean(trained.note) })
    : P_TRAINER_SEEDS.map((seed) => `seed ${seed}: ${describeTrainedCategory(trained.perSeedCategory[seed])}`).join(', ');

  // (dual review, Important) The authored and trained categories are two
  // *different* vocabularies, not the same one — `no-specific-effect` is
  // `pathwayInterventions.ts`'s own documented merge of the authored
  // `generic-rewiring-effect`/`not-supported` split (the trained side's
  // predeclared rules cannot decide that finer split). Comparing the raw
  // strings would wrongly read an authored `not-supported` result next to a
  // robust trained `no-specific-effect` as "does not reproduce", when the
  // trained side is actually consistent with it — it just cannot confirm
  // which of the two merged authored categories held. Map the authored
  // category into the trained vocabulary before comparing, and only claim
  // "matches" when both sides literally agree (never merely map-equal).
  const authoredAsTrainedCategory: PathwayInterventionsTrainedCategory =
    authored.category === 'generic-rewiring-effect' || authored.category === 'not-supported'
      ? 'no-specific-effect'
      : (authored.category as PathwayInterventionsTrainedCategory);
  const trainedCategory: PathwayInterventionsTrainedCategory | undefined = trained.trainedRobust
    ? trained.perSeedCategory[P_TRAINER_SEEDS[0]]
    : undefined;
  const reproductionClause =
    trainedCategory === undefined || trainedCategory !== authoredAsTrainedCategory
      ? 'this does not reproduce'
      : (trainedCategory as string) === (authored.category as string)
        ? 'this matches'
        : 'this is consistent with (the trained rules cannot split generic-rewiring-effect from not-supported)';
  const sentence =
    `With trained readouts, ${agreementPrefix} ${categoryText} — ${reproductionClause} the authored decoder's ` +
    `${authored.category} result, under this model.`;
  return { ...base, status: 'ok', sentence };
};
