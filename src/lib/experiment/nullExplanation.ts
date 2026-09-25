/**
 * `public/data/null-explanation-v1.json` (WP4 of
 * `.agents/plans/null-explanation`, `04-ledger-note.md`): the offline-computed
 * finding note that explains, under this model only, why the biological
 * MaleCNS graph scored below every one of its 500 degree-preserving
 * rewirings (`docs/rewiring-null-report.md`; the full method and every
 * number below are documented in `docs/null-explanation-report.md`). The
 * real producer's full type is `scripts/analysis/explain.py`'s output — that
 * module is a Node-only pipeline (not part of the browser bundle), so this
 * file independently authors and validates just the subset the ledger note
 * actually renders, the same "reimplemented, not imported" discipline
 * `./rewiringNull.ts`/`./lesionAtlas.ts` already document for their own
 * artifacts.
 *
 * Split into its own module rather than added to `./assets.ts` (thermo
 * review precedent: `./rewiringNull.ts` and `./lesionAtlas.ts` were both
 * split out for the same "god module" reason) — depends on nothing from
 * `assets.ts` except the shared `ArenaManifest` type and the
 * `fetchAndVerifySidecarJson` fetch->sha256-verify->JSON.parse helper it
 * shares with `loadPositions`/`loadRewiringNull`/`loadLesionAtlas`.
 */

import { fetchAndVerifySidecarJson, type ArenaManifest } from './assets';
import { isFiniteNumber } from './rewiringNull';
import type { SidecarLoadResult } from './sidecarResult';

/** The three metric families `docs/null-explanation-report.md` tests (transfer entries, derived predictors, and structural features) — matches `explain.py`'s own `kind` vocabulary. */
export type NullExplanationMetricKind = 'transfer' | 'derived' | 'feature';

/** One metric that independently passed both predeclared gates (outside the null's 2.5-97.5% range, and `|rho| >= 0.3`) — `finding.qualifyingMetrics[]`. */
export interface NullExplanationQualifyingMetric {
  readonly kind: NullExplanationMetricKind;
  readonly name: string;
  readonly spearman: number;
}

/**
 * The subset of `explain.py`'s `finding` object the ledger note renders:
 * the mechanically-evaluated outcome categories, the metrics that triggered
 * them, whether the structural-feature reading is definition-sensitive (the
 * feature-6/`weightedInDegree` disclosure — see
 * `docs/null-explanation-report.md`'s "Structural features" section), the
 * regime-gate outcome, and the templated one-sentence summary.
 */
export interface NullExplanationFinding {
  /** Validated as a cheap structural guard (every entry is a real string) but never itself rendered — `summarySentence` already states which categories held in prose, and `qualifyingMetrics` names the individual metrics behind them. */
  readonly categories: readonly string[];
  readonly definitionSensitive: boolean;
  readonly qualifyingMetrics: readonly NullExplanationQualifyingMetric[];
  readonly regimeInvalid: boolean;
  readonly summarySentence: string;
}

/** The mirrored decoder-convention check's result (`variants.flipBoth`) — the one required re-scoring run (thrust and yaw signs both flipped). Single-axis variants are conditional on this and are never rendered here. */
export interface NullExplanationFlipBothVariant {
  readonly bioPercentile: number;
}

export interface NullExplanationArtifact {
  readonly version: number;
  /** `sources.rewiringNullSha256` pins which `rewiring-null-v1.json` this explanation was computed against — cross-checked by `loadNullExplanation` below against `manifest.rewiringNull.sha256`. */
  readonly sources: { readonly rewiringNullSha256: string };
  readonly variants: { readonly flipBoth: NullExplanationFlipBothVariant };
  /** The aggregate linear-regime validity gate (`docs/null-explanation-report.md`'s "Regime check" section); `finding.regimeInvalid` is this same outcome folded into the finding. */
  readonly regime: { readonly gatePassed: boolean };
  readonly finding: NullExplanationFinding;
}

/**
 * Four states, mirroring `RewiringNullLoadResult`'s/`LesionAtlasLoadResult`'s
 * own split (round-2 dual review, Important — an earlier version folded a
 * fetch/network failure and an unexpected runtime error into `'missing'`/
 * `'invalid'` respectively, which meant a dropped connection silently hid
 * the note with no trace, and a `crypto.subtle`-unavailable throw was shown
 * to a visitor as "Explanation failed verification" even though nothing was
 * ever verified):
 * - `'missing'`: the manifest has no `nullExplanation` entry at all — nothing
 *   was ever shipped. The bean's own contract: the paragraph is hidden with
 *   no message.
 * - `'unavailable'`: a fetch/network failure, or an unexpected exception
 *   anywhere in the load chain (`controller.ts`'s leading `.catch`) — not a
 *   claim about the artifact's integrity. Shown as "Explanation could not be
 *   loaded: …", the same wording `LedgerPanel.svelte` already uses for
 *   `RewiringNullLoadResult`'s own `'unavailable'`.
 * - `'invalid'`: the artifact was actually fetched and failed a real
 *   verification step (sha256, shape, or the cross-check against the shipped
 *   rewiring-null artifact). Shown as "Explanation failed verification: …".
 *
 * A type alias for the shared `SidecarLoadResult` (thermo-maintainability
 * review I3) — zero behavior change, the discriminated union shape below is
 * identical to what this type used to define inline, so every existing
 * `status ===` narrowing site keeps working unchanged.
 */
export type NullExplanationLoadResult = SidecarLoadResult<NullExplanationArtifact>;

const isMetricKind = (value: unknown): value is NullExplanationMetricKind =>
  value === 'transfer' || value === 'derived' || value === 'feature';

const isUnitInterval = (value: unknown): value is number => isFiniteNumber(value) && value >= 0 && value <= 1;

const isQualifyingMetric = (value: unknown): value is NullExplanationQualifyingMetric => {
  if (typeof value !== 'object' || value === null) return false;
  const v = value as Record<string, unknown>;
  return isMetricKind(v.kind) && typeof v.name === 'string' && v.name.length > 0 && isFiniteNumber(v.spearman);
};

const isFinding = (value: unknown): value is NullExplanationFinding => {
  if (typeof value !== 'object' || value === null) return false;
  const v = value as Record<string, unknown>;
  return (
    Array.isArray(v.categories) &&
    v.categories.every((category) => typeof category === 'string') &&
    typeof v.definitionSensitive === 'boolean' &&
    Array.isArray(v.qualifyingMetrics) &&
    v.qualifyingMetrics.every(isQualifyingMetric) &&
    typeof v.regimeInvalid === 'boolean' &&
    typeof v.summarySentence === 'string' &&
    v.summarySentence.trim().length > 0
  );
};

const validateNullExplanationShape = (
  value: unknown
): { ok: true; data: NullExplanationArtifact } | { ok: false; reason: string } => {
  if (typeof value !== 'object' || value === null) {
    return { ok: false, reason: 'null-explanation artifact is not a JSON object' };
  }
  const v = value as Record<string, unknown>;
  if (v.version !== 1) {
    return { ok: false, reason: `null-explanation artifact has unsupported version ${String(v.version)}` };
  }

  const sources = v.sources;
  if (typeof sources !== 'object' || sources === null || typeof (sources as Record<string, unknown>).rewiringNullSha256 !== 'string') {
    return { ok: false, reason: 'null-explanation artifact is missing sources.rewiringNullSha256' };
  }

  const variants = v.variants;
  const flipBoth = typeof variants === 'object' && variants !== null ? (variants as Record<string, unknown>).flipBoth : undefined;
  if (typeof flipBoth !== 'object' || flipBoth === null || !isUnitInterval((flipBoth as Record<string, unknown>).bioPercentile)) {
    return { ok: false, reason: 'null-explanation artifact has a malformed variants.flipBoth.bioPercentile' };
  }

  const regime = v.regime;
  if (typeof regime !== 'object' || regime === null || typeof (regime as Record<string, unknown>).gatePassed !== 'boolean') {
    return { ok: false, reason: 'null-explanation artifact has a malformed regime.gatePassed' };
  }

  if (!isFinding(v.finding)) {
    return { ok: false, reason: 'null-explanation artifact has a malformed "finding" field' };
  }

  // Cross-field consistency (round-2 dual review, Important — mirrors
  // `rewiringNull.ts`'s own recomputed-rank-statistics precedent): the real
  // producer sets `finding.regimeInvalid = not regime.gatePassed` always
  // (`scripts/analysis/explain.py`'s `evaluate_categories`), and
  // `NullExplanationArtifact`'s own doc comment already claims this. A
  // hash-valid artifact whose two fields disagree would otherwise pass
  // validation and let the regime clause below state the wrong outcome —
  // `regime.gatePassed` would then be a validated field with no consumer
  // and no payoff.
  const regimeGatePassed = (regime as Record<string, unknown>).gatePassed as boolean;
  const finding = v.finding as NullExplanationFinding;
  if (finding.regimeInvalid !== !regimeGatePassed) {
    return { ok: false, reason: 'null-explanation finding.regimeInvalid disagrees with regime.gatePassed' };
  }

  return { ok: true, data: value as NullExplanationArtifact };
};

/**
 * Fetch, sha256-verify, and structurally validate `null-explanation-v1.json`
 * (`manifest.nullExplanation`), then cross-check it against the manifest's
 * own pinned rewiring-null artifact — never throws, matching
 * `loadPositions`/`loadRewiringNull`/`loadLesionAtlas`'s "never throws,
 * always return a reasoned status" contract, so a missing/tampered/malformed
 * explanation artifact only ever hides or degrades the ledger note, never
 * the rest of the experiment or the null histogram above it.
 *
 * `dataBaseUrl` must be the same value the caller passes to
 * `loadArenaArtifacts`/`loadRewiringNull` (`${import.meta.env.BASE_URL}data`
 * in production) so this artifact resolves under the app's real deployment
 * base path too.
 */
export const loadNullExplanation = async (
  manifest: ArenaManifest,
  dataBaseUrl: string
): Promise<NullExplanationLoadResult> => {
  const fetched = await fetchAndVerifySidecarJson(manifest.nullExplanation, dataBaseUrl, 'null-explanation artifact');
  if (fetched.status === 'no-entry') {
    return { status: 'missing', reason: 'The manifest has no nullExplanation artifact entry.' };
  }
  if (fetched.status === 'fetch-error') {
    return { status: 'unavailable', reason: fetched.reason };
  }
  if (fetched.status === 'hash-mismatch' || fetched.status === 'parse-error') {
    return { status: 'invalid', reason: fetched.reason };
  }

  const validated = validateNullExplanationShape(fetched.parsed);
  if (!validated.ok) return { status: 'invalid', reason: validated.reason };
  const data = validated.data;

  // The sha256 check above only proves these bytes are the ones the
  // manifest's `nullExplanation` entry pins — it says nothing about whether
  // this explanation was actually computed against *this* manifest's
  // rewiring-null artifact (a hand-edited or merge-conflicted manifest could
  // re-pin `nullExplanation` to a study computed against a different null
  // distribution). Mirrors `loadRewiringNull`'s own
  // `sourceGraphSha256`/`manifest.binarySha256` staleness check.
  const shippedRewiringNullSha256 = manifest.rewiringNull?.sha256;
  if (!shippedRewiringNullSha256) {
    return {
      status: 'invalid',
      reason: 'manifest is missing rewiringNull.sha256, needed to cross-check the null-explanation artifact'
    };
  }
  if (data.sources.rewiringNullSha256 !== shippedRewiringNullSha256) {
    return {
      status: 'invalid',
      reason:
        `null-explanation sources.rewiringNullSha256 does not match the manifest's rewiring-null artifact ` +
        `(${shippedRewiringNullSha256}) — stale artifact`
    };
  }

  return { status: 'ok', data };
};
