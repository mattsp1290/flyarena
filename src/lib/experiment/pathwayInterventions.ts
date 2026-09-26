/**
 * `public/data/pathway-interventions-v1.json` (WP4 of
 * `.agents/plans/pathway-interventions`, `04-report-and-ledger.md`): the
 * offline-computed clearance->thrust pathway intervention study — whether a
 * predeclared, degree-preserving swap set targeting the input->thrust
 * pathway the null-explanation study flagged actually moves the model's
 * behavior score, under both the authored decoder and CEM-retrained
 * readouts, against matched controls (`docs/pathway-interventions-report.md`
 * has the full method and every number below). The real producer's full
 * type is `scripts/null/intervention-artifact.ts`'s
 * `PathwayInterventionsArtifact` — that module is a Node-only pipeline (not
 * part of the browser bundle), so this file independently authors and
 * validates just the subset the ledger note actually renders, the same
 * "reimplemented, not imported" discipline `./nullExplanation.ts`/
 * `./rewiringNull.ts` already document for their own artifacts.
 *
 * Split into its own module rather than added to `./assets.ts` (thermo
 * review precedent this codebase already established for
 * `./rewiringNull.ts`/`./nullExplanation.ts`/`./lesionAtlas.ts`) — depends
 * on nothing from `assets.ts` except the shared `ArenaManifest` type and the
 * `fetchAndVerifySidecarJson` fetch->sha256-verify->JSON.parse helper.
 */

import { fetchAndVerifySidecarJson, type ArenaManifest } from './assets';
import type { SidecarLoadResult } from './sidecarResult';

/** `00-overview.md`'s predeclared authored-decoder categories, verbatim vocabulary. */
export type PathwayInterventionsAuthoredCategory = 'pathway-supported' | 'edge-class-effect' | 'generic-rewiring-effect' | 'not-supported';

/**
 * The trained decoder's own category vocabulary — deliberately not the
 * authored one above. `'no-specific-effect'` stands for "neither
 * pathway-supported nor edge-class; the generic-vs-not-supported split is
 * undetermined under this study's predeclared rules" (see
 * `scripts/null/intervention-report-trained.ts`'s own doc comment for why
 * that split cannot be mechanically decided on the trained side).
 */
export type PathwayInterventionsTrainedCategory = 'pathway-supported' | 'edge-class-effect' | 'no-specific-effect';

const isAuthoredCategory = (value: unknown): value is PathwayInterventionsAuthoredCategory =>
  value === 'pathway-supported' || value === 'edge-class-effect' || value === 'generic-rewiring-effect' || value === 'not-supported';

const isTrainedCategory = (value: unknown): value is PathwayInterventionsTrainedCategory =>
  value === 'pathway-supported' || value === 'edge-class-effect' || value === 'no-specific-effect';

/**
 * This study's fixed trainer seeds — `scripts/null/intervention-report-trained.ts`'s
 * `P_TRAINER_SEEDS`, restated here as strings (a Node-only module, not
 * importable into the browser bundle). Exported (thermo-maintainability
 * review, Suggestion) so `NullExplanationNote.svelte` — already a consumer
 * of this module's types — can import this constant too, instead of
 * hand-copying it a third time.
 */
export const P_TRAINER_SEEDS = ['101', '202', '303'] as const;

export interface PathwayInterventionsArtifact {
  readonly version: number;
  readonly sources: { readonly biologicalSha: string; readonly rewiringNullSha: string; readonly nullExplanationSha: string };
  readonly authored: { readonly category: PathwayInterventionsAuthoredCategory; readonly channelSpecific: boolean };
  readonly trained: {
    readonly trainedRobust: boolean;
    /**
     * Every trainer seed's own category, keyed by seed. When `trainedRobust`
     * is true, every entry agrees, and `NullExplanationNote.svelte` reads
     * `perSeedCategory['101']` as the representative value; when it is
     * false, the component renders every seed's own category so a reader
     * can see how they actually diverged, rather than only a bare
     * "do not agree" boolean.
     */
    readonly perSeedCategory: Readonly<Record<(typeof P_TRAINER_SEEDS)[number], PathwayInterventionsTrainedCategory>>;
    /**
     * `scripts/null/intervention-artifact.ts`'s own methodological
     * disclosure for why `'no-specific-effect'` exists as a category name
     * (thermo review, methodology I2): it is a reporting convention this
     * study's coordinator adopted after the trained scores were known, not
     * itself a predeclared category. Optional — an older or hand-built
     * fixture may simply not carry it; `src/lib/findings/steps.ts`'s step 6
     * only shows its short caveat when this field is actually present,
     * never unconditionally.
     */
    readonly note?: string;
  };
}

export type PathwayInterventionsLoadResult = SidecarLoadResult<PathwayInterventionsArtifact>;

const validateShape = (value: unknown): { ok: true; data: PathwayInterventionsArtifact } | { ok: false; reason: string } => {
  if (typeof value !== 'object' || value === null) {
    return { ok: false, reason: 'pathway-interventions artifact is not a JSON object' };
  }
  const v = value as Record<string, unknown>;
  if (v.version !== 1) return { ok: false, reason: `pathway-interventions artifact has unsupported version ${String(v.version)}` };

  const sources = v.sources as Record<string, unknown> | undefined;
  if (
    !sources ||
    typeof sources.biologicalSha !== 'string' ||
    typeof sources.rewiringNullSha !== 'string' ||
    typeof sources.nullExplanationSha !== 'string'
  ) {
    return { ok: false, reason: 'pathway-interventions artifact is missing sources.biologicalSha/sources.rewiringNullSha/sources.nullExplanationSha' };
  }

  const authored = v.authored as Record<string, unknown> | undefined;
  if (!authored || !isAuthoredCategory(authored.category) || typeof authored.channelSpecific !== 'boolean') {
    return { ok: false, reason: 'pathway-interventions artifact has a malformed "authored" field' };
  }

  const trained = v.trained as Record<string, unknown> | undefined;
  const perSeed = trained?.perSeed as Record<string, unknown> | undefined;
  if (!trained || typeof trained.trainedRobust !== 'boolean' || !perSeed) {
    return { ok: false, reason: 'pathway-interventions artifact has a malformed "trained" field' };
  }
  const perSeedCategory: Partial<Record<(typeof P_TRAINER_SEEDS)[number], PathwayInterventionsTrainedCategory>> = {};
  for (const seed of P_TRAINER_SEEDS) {
    const entry = perSeed[seed] as Record<string, unknown> | undefined;
    if (!entry || !isTrainedCategory(entry.category)) {
      return { ok: false, reason: `pathway-interventions artifact is missing a valid trained.perSeed["${seed}"].category` };
    }
    perSeedCategory[seed] = entry.category;
  }
  const categories = P_TRAINER_SEEDS.map((seed) => perSeedCategory[seed] as PathwayInterventionsTrainedCategory);

  // Cross-field consistency (mirrors `rewiringNull.ts`'s recomputed-rank-
  // statistics precedent, and `nullExplanation.ts`'s
  // `finding.regimeInvalid`-vs-`regime.gatePassed` check): a hash-valid
  // artifact could still ship a `trainedRobust` that disagrees with its own
  // per-seed categories.
  const recomputedRobust = categories.every((category) => category === categories[0]);
  if (recomputedRobust !== trained.trainedRobust) {
    return { ok: false, reason: 'pathway-interventions trained.trainedRobust disagrees with its own per-seed categories' };
  }

  return {
    ok: true,
    data: {
      version: 1,
      sources: { biologicalSha: sources.biologicalSha, rewiringNullSha: sources.rewiringNullSha, nullExplanationSha: sources.nullExplanationSha },
      authored: { category: authored.category, channelSpecific: authored.channelSpecific },
      trained: {
        trainedRobust: trained.trainedRobust,
        perSeedCategory: perSeedCategory as Readonly<Record<(typeof P_TRAINER_SEEDS)[number], PathwayInterventionsTrainedCategory>>,
        ...(typeof trained.note === 'string' ? { note: trained.note } : {})
      }
    }
  };
};

/**
 * Fetch, sha256-verify, and structurally validate `pathway-interventions-v1.json`
 * (`manifest.pathwayInterventions`), then cross-check it against the
 * manifest's own pinned rewiring-null and null-explanation artifacts —
 * never throws, matching `loadNullExplanation`/`loadRewiringNull`'s "never
 * throws, always return a reasoned status" contract, so a missing/
 * tampered/malformed artifact only ever hides or degrades the ledger's
 * tested-outcome sentence, never the rest of the experiment.
 *
 * `dataBaseUrl` must be the same value the caller passes to
 * `loadArenaArtifacts`/`loadRewiringNull`/`loadNullExplanation`
 * (`${import.meta.env.BASE_URL}data` in production) so this artifact
 * resolves under the app's real deployment base path too.
 */
export const loadPathwayInterventions = async (
  manifest: ArenaManifest,
  dataBaseUrl: string
): Promise<PathwayInterventionsLoadResult> => {
  const fetched = await fetchAndVerifySidecarJson(manifest.pathwayInterventions, dataBaseUrl, 'pathway-interventions artifact');
  if (fetched.status === 'no-entry') {
    return { status: 'missing', reason: 'The manifest has no pathwayInterventions artifact entry.' };
  }
  if (fetched.status === 'fetch-error') {
    return { status: 'unavailable', reason: fetched.reason };
  }
  if (fetched.status === 'hash-mismatch' || fetched.status === 'parse-error') {
    return { status: 'invalid', reason: fetched.reason };
  }

  const validated = validateShape(fetched.parsed);
  if (!validated.ok) return { status: 'invalid', reason: validated.reason };
  const data = validated.data;

  // The sha256 check above only proves these bytes are the ones the
  // manifest's `pathwayInterventions` entry pins — it says nothing about
  // whether this study was actually computed against *this* manifest's
  // biological/rewiring-null/null-explanation artifacts (a hand-edited or
  // merge-conflicted manifest could re-pin `pathwayInterventions` to a
  // study computed against a different biological graph, null distribution,
  // or explanation). Mirrors `loadNullExplanation`'s/`loadRewiringNull`'s own
  // staleness checks (methodology-review suggestion: an earlier version
  // cross-checked `rewiringNullSha`/`nullExplanationSha` but not
  // `biologicalSha` against `manifest.binarySha256`, even though the
  // producer itself already checks it).
  if (data.sources.biologicalSha !== manifest.binarySha256) {
    return {
      status: 'invalid',
      reason: `pathway-interventions sources.biologicalSha does not match the manifest's compiled biological graph (${manifest.binarySha256}) — stale artifact`
    };
  }
  const shippedRewiringNullSha = manifest.rewiringNull?.sha256;
  if (!shippedRewiringNullSha) {
    return { status: 'invalid', reason: 'manifest is missing rewiringNull.sha256, needed to cross-check the pathway-interventions artifact' };
  }
  if (data.sources.rewiringNullSha !== shippedRewiringNullSha) {
    return {
      status: 'invalid',
      reason: `pathway-interventions sources.rewiringNullSha does not match the manifest's rewiring-null artifact (${shippedRewiringNullSha}) — stale artifact`
    };
  }
  const shippedNullExplanationSha = manifest.nullExplanation?.sha256;
  if (!shippedNullExplanationSha) {
    return { status: 'invalid', reason: 'manifest is missing nullExplanation.sha256, needed to cross-check the pathway-interventions artifact' };
  }
  if (data.sources.nullExplanationSha !== shippedNullExplanationSha) {
    return {
      status: 'invalid',
      reason: `pathway-interventions sources.nullExplanationSha does not match the manifest's null-explanation artifact (${shippedNullExplanationSha}) — stale artifact`
    };
  }

  return { status: 'ok', data };
};
