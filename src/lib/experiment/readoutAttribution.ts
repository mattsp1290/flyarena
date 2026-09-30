/**
 * `public/data/readout-attribution-v1.json` (WP3 of
 * `.agents/plans/readout-attribution`, `03-artifact-and-findings.md`):
 * whether saliency, readout-input ablation, input-independence share, and
 * pathway linkage explain why CEM-retrained readouts erase the biological
 * graph's deficit against its degree-preserving rewirings, and why the
 * pathway intervention P gains nothing under trained readouts
 * (`docs/readout-attribution-report.md` has the full method, per-readout
 * results, and limitations). The real producer's full type is
 * `scripts/attribution/attribution-report.ts`'s `ReadoutAttributionArtifact`
 * — that module is a Node-only pipeline (not part of the browser bundle),
 * so this file independently authors and validates just the subset the
 * Findings panel's step actually renders (`sources`, `coverage`,
 * `hypotheses`, `host`), the same "reimplemented, not imported" discipline
 * `./taskGenerality.ts`/`./selectionRobustness.ts` already document for
 * their own artifacts. The full per-readout `readouts[]` array (48-length
 * saliency vectors, 16-entry ablation tables per readout) is fetched and
 * sha256-verified along with everything else, but never deeply parsed here
 * — the Findings step's one sentence is templated entirely from the H1-H3
 * outcomes, not from any individual readout's own numbers.
 */

import { fetchAndVerifySidecarJson, fetchJson, type ArenaManifest } from './assets';
import type { SidecarLoadResult } from './sidecarResult';

export type HypothesisOutcome = 'supported' | 'not-supported' | 'inconclusive';

const isHypothesisOutcome = (value: unknown): value is HypothesisOutcome =>
  value === 'supported' || value === 'not-supported' || value === 'inconclusive';

export interface HypothesisResult {
  readonly outcome: HypothesisOutcome;
  readonly reason?: string;
  /** Per-hypothesis evidence shape (`H1`'s per-seed rho/CI table, `H2`'s paired differences, `H3`'s paired ratios) — deliberately untyped here, mirroring `scripts/attribution/hypotheses.ts`'s own `HypothesisResult.evidence: unknown`. The Findings step's sentence never reads into this; it exists on the loaded artifact only so a future consumer (or a test) can inspect it without a second fetch. */
  readonly evidence: unknown;
}

export interface ReadoutAttributionHypotheses {
  readonly hypothesisCount: number;
  readonly multipleComparisonCorrection: string;
  readonly H1: HypothesisResult;
  readonly H2: HypothesisResult;
  readonly H3: HypothesisResult;
}

export interface ReadoutAttributionProducer {
  readonly script: string;
  readonly sourceSha256: string;
  readonly dependencies: readonly string[];
}

export interface ReadoutAttributionCoverage {
  readonly ids: readonly string[];
  readonly perTaskIncluded: boolean;
}

export interface ReadoutAttributionArtifact {
  readonly version: 1;
  readonly sources: {
    readonly archiveSha: string;
    readonly descendingTypesSha: string;
    readonly trainedReadoutSha: string;
    readonly pathwayInterventionsSha: string;
    readonly producer: ReadoutAttributionProducer;
  };
  readonly coverage: ReadoutAttributionCoverage;
  readonly hypotheses: ReadoutAttributionHypotheses;
  readonly host: { readonly arch: string; readonly node: string };
}

export type ReadoutAttributionLoadResult = SidecarLoadResult<ReadoutAttributionArtifact>;

// ---------------------------------------------------------------------------
// Shape validation
// ---------------------------------------------------------------------------

const hypothesisReason = (value: unknown): string | undefined => {
  if (typeof value !== 'object' || value === null) return 'is not an object';
  const v = value as Record<string, unknown>;
  if (!isHypothesisOutcome(v.outcome)) return `has an invalid "outcome" (${String(v.outcome)})`;
  if (v.reason !== undefined && typeof v.reason !== 'string') return 'has a non-string "reason"';
  if (!('evidence' in v)) return 'is missing "evidence"';
  return undefined;
};

const producerReason = (value: unknown): string | undefined => {
  if (typeof value !== 'object' || value === null) return 'is not an object';
  const v = value as Record<string, unknown>;
  if (typeof v.script !== 'string') return 'is missing a string "script"';
  if (typeof v.sourceSha256 !== 'string') return 'is missing a string "sourceSha256"';
  if (!Array.isArray(v.dependencies) || !v.dependencies.every((d) => typeof d === 'string')) {
    return 'has a malformed "dependencies" array';
  }
  return undefined;
};

const validateShape = (value: unknown): { ok: true; data: ReadoutAttributionArtifact } | { ok: false; reason: string } => {
  if (typeof value !== 'object' || value === null) {
    return { ok: false, reason: 'readout-attribution artifact is not a JSON object' };
  }
  const v = value as Record<string, unknown>;
  if (v.version !== 1) return { ok: false, reason: `readout-attribution artifact has unsupported version ${String(v.version)}` };

  const sources = v.sources as Record<string, unknown> | undefined;
  if (
    !sources ||
    typeof sources.archiveSha !== 'string' ||
    typeof sources.descendingTypesSha !== 'string' ||
    typeof sources.trainedReadoutSha !== 'string' ||
    typeof sources.pathwayInterventionsSha !== 'string'
  ) {
    return {
      ok: false,
      reason:
        'readout-attribution artifact has a malformed "sources" object (missing archiveSha/descendingTypesSha/trainedReadoutSha/pathwayInterventionsSha)'
    };
  }
  const producerIssue = producerReason(sources.producer);
  if (producerIssue) return { ok: false, reason: `readout-attribution artifact sources.producer ${producerIssue}` };

  const coverage = v.coverage as Record<string, unknown> | undefined;
  if (
    !coverage ||
    !Array.isArray(coverage.ids) ||
    !coverage.ids.every((id) => typeof id === 'string') ||
    typeof coverage.perTaskIncluded !== 'boolean'
  ) {
    return { ok: false, reason: 'readout-attribution artifact has a malformed "coverage" object' };
  }

  const hypotheses = v.hypotheses as Record<string, unknown> | undefined;
  if (!hypotheses || typeof hypotheses.hypothesisCount !== 'number' || typeof hypotheses.multipleComparisonCorrection !== 'string') {
    return { ok: false, reason: 'readout-attribution artifact has a malformed "hypotheses" object' };
  }
  for (const key of ['H1', 'H2', 'H3'] as const) {
    const issue = hypothesisReason(hypotheses[key]);
    if (issue) return { ok: false, reason: `readout-attribution artifact hypotheses.${key} ${issue}` };
  }

  const host = v.host as Record<string, unknown> | undefined;
  if (!host || typeof host.arch !== 'string' || typeof host.node !== 'string') {
    return { ok: false, reason: 'readout-attribution artifact has a malformed "host" object' };
  }

  return {
    ok: true,
    data: {
      version: 1,
      sources: sources as ReadoutAttributionArtifact['sources'],
      coverage: coverage as unknown as ReadoutAttributionCoverage,
      hypotheses: hypotheses as unknown as ReadoutAttributionHypotheses,
      host: host as ReadoutAttributionArtifact['host']
    }
  };
};

// ---------------------------------------------------------------------------
// trained-readout-v1.manifest.json (a separate, non-`ArenaManifest` sidecar —
// see `assets.ts#loadTrainedReadoutArtifact`)
// ---------------------------------------------------------------------------

interface TrainedReadoutManifestSha {
  readonly artifactSha256: string;
}

/**
 * Fetch, sha256-verify, and structurally validate `readout-attribution-v1.json`
 * (`manifest.readoutAttribution`), then cross-check it against the
 * manifest's own pinned `descendingTypes`/`pathwayInterventions` entries and
 * the separate `trained-readout-v1.manifest.json`'s `artifactSha256` — never
 * throws, matching every sibling loader's "never throws, always return a
 * reasoned status" contract, so a missing/tampered/malformed artifact only
 * ever hides or degrades the Findings panel's readout-attribution step,
 * never the rest of the experiment.
 *
 * `dataBaseUrl` must be the same value the caller passes to the other
 * `loadX` functions (`${import.meta.env.BASE_URL}data` in production) so
 * this artifact resolves under the app's real deployment base path too.
 */
export const loadReadoutAttribution = async (
  manifest: ArenaManifest,
  dataBaseUrl: string
): Promise<ReadoutAttributionLoadResult> => {
  const fetched = await fetchAndVerifySidecarJson(manifest.readoutAttribution, dataBaseUrl, 'readout-attribution artifact');
  if (fetched.status === 'no-entry') {
    return { status: 'missing', reason: 'The manifest has no readoutAttribution artifact entry.' };
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

  const shippedDescendingTypesSha = manifest.descendingTypes?.sha256;
  if (!shippedDescendingTypesSha) {
    return { status: 'invalid', reason: 'manifest is missing descendingTypes.sha256, needed to cross-check the readout-attribution artifact' };
  }
  if (data.sources.descendingTypesSha !== shippedDescendingTypesSha) {
    return {
      status: 'invalid',
      reason: `readout-attribution sources.descendingTypesSha does not match the manifest's descendingTypes artifact (${shippedDescendingTypesSha}) — stale artifact`
    };
  }

  const shippedPathwayInterventionsSha = manifest.pathwayInterventions?.sha256;
  if (!shippedPathwayInterventionsSha) {
    return { status: 'invalid', reason: 'manifest is missing pathwayInterventions.sha256, needed to cross-check the readout-attribution artifact' };
  }
  if (data.sources.pathwayInterventionsSha !== shippedPathwayInterventionsSha) {
    return {
      status: 'invalid',
      reason: `readout-attribution sources.pathwayInterventionsSha does not match the manifest's pathwayInterventions artifact (${shippedPathwayInterventionsSha}) — stale artifact`
    };
  }

  let trainedReadoutManifest: TrainedReadoutManifestSha;
  try {
    trainedReadoutManifest = await fetchJson<TrainedReadoutManifestSha>(`${dataBaseUrl}/trained-readout-v1.manifest.json`);
  } catch (error) {
    return {
      status: 'unavailable',
      reason: `could not fetch trained-readout-v1.manifest.json to cross-check the readout-attribution artifact: ${error instanceof Error ? error.message : String(error)}`
    };
  }
  if (typeof trainedReadoutManifest.artifactSha256 !== 'string') {
    return { status: 'invalid', reason: 'trained-readout-v1.manifest.json is missing artifactSha256, needed to cross-check the readout-attribution artifact' };
  }
  if (data.sources.trainedReadoutSha !== trainedReadoutManifest.artifactSha256) {
    return {
      status: 'invalid',
      reason: `readout-attribution sources.trainedReadoutSha does not match trained-readout-v1.manifest.json's artifactSha256 (${trainedReadoutManifest.artifactSha256}) — stale artifact`
    };
  }

  return { status: 'ok', data };
};
