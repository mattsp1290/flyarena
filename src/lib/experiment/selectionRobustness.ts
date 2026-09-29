/**
 * `public/data/selection-robustness-v1.json` (WP3 of
 * `.agents/plans/selection-robustness`, `03-artifact-and-findings.md`):
 * whether the headline authored-decoder findings (rewiring-null holds,
 * clearance->thrust explains the gap, pathway P is pathway-supported) hold
 * across four predeclared alternative subgraph selections
 * (`docs/selection-robustness-report.md` has the full method, per-selection
 * results, and limitations). The real producer's full type is
 * `scripts/selections/selection-report.ts`'s `SelectionRobustnessArtifact`
 * -- that module is a Node-only pipeline (not part of the browser bundle),
 * so this file independently authors and validates just the subset the
 * Findings panel's step actually renders, the same "reimplemented, not
 * imported" discipline `./taskGenerality.ts`/`./rewiringNull.ts`/
 * `./pathwayInterventions.ts`/`./repertoireNull.ts` already document for
 * their own artifacts. Field *names* and nesting mirror the producer's real
 * JSON one-to-one so the two can never silently diverge on what a field is
 * called.
 */

import { fetchAndVerifySidecarJson, type ArenaManifest } from './assets';
import type { SidecarLoadResult } from './sidecarResult';

export const SELECTION_IDS = ['larger', 'smaller', 'random-bridge', 'alt-sensory-mapping'] as const;
export type SelectionId = (typeof SELECTION_IDS)[number];

const isSelectionId = (value: unknown): value is SelectionId => (SELECTION_IDS as readonly string[]).includes(value as string);

export interface SelectionNullResult {
  readonly bioScore: number;
  readonly bioPercentile: number;
  readonly p25: number;
  readonly degenerate: boolean;
  readonly holds: boolean;
}

export interface SelectionExplanationMetric {
  readonly name: string;
  readonly spearman: number;
  readonly bioPercentile: number;
}

export interface SelectionExplanationResult {
  readonly passing: readonly SelectionExplanationMetric[];
  readonly replicates: boolean;
  readonly structuralReplicates: boolean;
  readonly mirroredBioPercentile: number;
  readonly singleAxis: { readonly flipThrust: number; readonly flipYaw: number } | null;
  readonly exploratory: null;
  readonly exploratoryOmittedReason: string;
}

export interface SelectionPathwayResult {
  readonly pScore: number;
  readonly cP95: number;
  readonly mP95: number;
  readonly cDegenerate: boolean;
  readonly mDegenerate: boolean;
  readonly k: number;
  readonly targetReached: boolean;
  readonly bridgePoolSize: number;
  readonly maxSwaps: number;
  readonly searchLimited: boolean;
  readonly supported: boolean;
  readonly qScore: number;
  readonly qK: number;
  readonly qTargetReached: boolean;
  readonly mqDegenerate: boolean;
  readonly channelSpecific: boolean | 'degenerate';
  readonly degenerateMechanism?: string;
}

export interface SelectionCoverageEntry {
  readonly channel?: string;
  readonly population?: string;
  readonly coveredBridgeCount: number;
  readonly flagged: boolean;
}

export interface SelectionCoverageResult {
  readonly perChannel: readonly SelectionCoverageEntry[];
  readonly perPopulation: readonly SelectionCoverageEntry[];
  readonly flagged: boolean;
}

export interface SelectionResult {
  readonly id: SelectionId;
  readonly params: Readonly<Record<string, string | number>>;
  readonly counts: Readonly<Record<string, number>>;
  readonly globalGain: number;
  readonly graphSha: string;
  readonly compiledFromGitRevision: string;
  readonly compilerSourceSha256: string;
  readonly null: SelectionNullResult;
  readonly explanation: SelectionExplanationResult;
  readonly pathway: SelectionPathwayResult;
  readonly coverage: SelectionCoverageResult;
  readonly categorized: boolean;
  readonly categorizedReason?: string;
}

export interface RobustnessVerdict {
  readonly verdict: true | false | 'indeterminate';
  readonly reason?: string;
}

export interface SelectionRobustnessOverall {
  readonly robustToSize: RobustnessVerdict;
  readonly robustToMethod: RobustnessVerdict;
  readonly mapping: RobustnessVerdict;
}

export interface SelectionRobustnessArtifact {
  readonly version: 1;
  readonly sources: {
    readonly defaultGraphSha: string;
    readonly rawFileShas: readonly string[];
    readonly compilerSourceSha: string;
    readonly producer: { readonly script: string; readonly sourceSha256: string; readonly dependencies: readonly string[] };
  };
  readonly selections: readonly SelectionResult[];
  readonly overall: SelectionRobustnessOverall;
  readonly host: { readonly arch: string; readonly node: string };
}

export type SelectionRobustnessLoadResult = SidecarLoadResult<SelectionRobustnessArtifact>;

// ---------------------------------------------------------------------------
// Shape + cross-field validation
// ---------------------------------------------------------------------------

const isMetric = (value: unknown): value is SelectionExplanationMetric => {
  if (typeof value !== 'object' || value === null) return false;
  const v = value as Record<string, unknown>;
  return typeof v.name === 'string' && typeof v.spearman === 'number' && typeof v.bioPercentile === 'number';
};

const nullReason = (value: unknown): string | undefined => {
  if (typeof value !== 'object' || value === null) return 'is not an object';
  const v = value as Record<string, unknown>;
  if (typeof v.bioScore !== 'number') return 'is missing a number "bioScore"';
  if (typeof v.bioPercentile !== 'number') return 'is missing a number "bioPercentile"';
  if (typeof v.p25 !== 'number') return 'is missing a number "p25"';
  if (typeof v.degenerate !== 'boolean') return 'is missing a boolean "degenerate"';
  if (typeof v.holds !== 'boolean') return 'is missing a boolean "holds"';
  // Recomputed, not trusted: `holds` is supposed to be the plain restatement
  // of the value comparison `bioScore < p25` (`scripts/selections/selection-report.ts`'s
  // own doc comment explains why this is the canonical convention here,
  // over the rank-based `bioPercentile < 0.25`, which can disagree with it
  // at a close boundary).
  if (v.holds !== (v.bioScore as number) < (v.p25 as number)) return 'holds disagrees with bioScore < p25';
  return undefined;
};

const explanationReason = (value: unknown): string | undefined => {
  if (typeof value !== 'object' || value === null) return 'is not an object';
  const v = value as Record<string, unknown>;
  if (!Array.isArray(v.passing) || !v.passing.every(isMetric)) return 'has a malformed "passing" array';
  const passing = v.passing as SelectionExplanationMetric[];
  if (typeof v.replicates !== 'boolean') return 'is missing a boolean "replicates"';
  if (typeof v.structuralReplicates !== 'boolean') return 'is missing a boolean "structuralReplicates"';
  const names = new Set(passing.map((metric) => metric.name));
  // Recomputed, not trusted: mirrors `selection-report.ts`'s own
  // `replicates`/`structuralReplicates` rule -- `weightedInDegree:thrust`
  // never decides `replicates` (`00-overview.md`'s predeclared rule).
  const recomputedReplicates = names.has('T:rightClearance->thrust') || names.has('T:forwardClearance->thrust');
  if (v.replicates !== recomputedReplicates) return 'replicates disagrees with its own "passing" list';
  if (v.structuralReplicates !== names.has('weightedInDegree:thrust')) return 'structuralReplicates disagrees with its own "passing" list';
  if (typeof v.mirroredBioPercentile !== 'number') return 'is missing a number "mirroredBioPercentile"';
  if (v.singleAxis !== null) {
    const axis = v.singleAxis as Record<string, unknown> | undefined;
    if (!axis || typeof axis.flipThrust !== 'number' || typeof axis.flipYaw !== 'number') {
      return 'has a malformed non-null "singleAxis"';
    }
  }
  if (v.exploratory !== null) return 'has a non-null "exploratory" (selection-mode always omits it)';
  if (typeof v.exploratoryOmittedReason !== 'string') return 'is missing a string "exploratoryOmittedReason"';
  return undefined;
};

const pathwayReason = (value: unknown): string | undefined => {
  if (typeof value !== 'object' || value === null) return 'is not an object';
  const v = value as Record<string, unknown>;
  const numberFields = ['pScore', 'cP95', 'mP95', 'k', 'bridgePoolSize', 'maxSwaps', 'qScore', 'qK'] as const;
  for (const field of numberFields) {
    if (typeof v[field] !== 'number') return `is missing a number "${field}"`;
  }
  const boolFields = ['cDegenerate', 'mDegenerate', 'targetReached', 'searchLimited', 'supported', 'qTargetReached', 'mqDegenerate'] as const;
  for (const field of boolFields) {
    if (typeof v[field] !== 'boolean') return `is missing a boolean "${field}"`;
  }
  if (v.channelSpecific !== true && v.channelSpecific !== false && v.channelSpecific !== 'degenerate') {
    return 'has an invalid "channelSpecific"';
  }
  // Recomputed, not trusted: `searchLimited` is supposed to be the plain
  // restatement of `k >= maxSwaps && !targetReached`.
  const recomputedSearchLimited = (v.k as number) >= (v.maxSwaps as number) && !(v.targetReached as boolean);
  if (v.searchLimited !== recomputedSearchLimited) return 'searchLimited disagrees with k/maxSwaps/targetReached';
  if ((v.cDegenerate as boolean) || (v.mDegenerate as boolean)) {
    if (typeof v.degenerateMechanism !== 'string' || v.degenerateMechanism.length === 0) {
      return 'is cDegenerate/mDegenerate but has no "degenerateMechanism" string';
    }
  }
  return undefined;
};

const coverageEntryReason = (value: unknown): string | undefined => {
  if (typeof value !== 'object' || value === null) return 'is not an object';
  const v = value as Record<string, unknown>;
  if (typeof v.coveredBridgeCount !== 'number') return 'is missing a number "coveredBridgeCount"';
  if (typeof v.flagged !== 'boolean') return 'is missing a boolean "flagged"';
  return undefined;
};

const coverageReason = (value: unknown): string | undefined => {
  if (typeof value !== 'object' || value === null) return 'is not an object';
  const v = value as Record<string, unknown>;
  if (!Array.isArray(v.perChannel) || v.perChannel.some((entry) => coverageEntryReason(entry) !== undefined)) {
    return 'has a malformed "perChannel" array';
  }
  if (!Array.isArray(v.perPopulation) || v.perPopulation.some((entry) => coverageEntryReason(entry) !== undefined)) {
    return 'has a malformed "perPopulation" array';
  }
  if (typeof v.flagged !== 'boolean') return 'is missing a boolean "flagged"';
  return undefined;
};

const selectionReason = (value: unknown): string | undefined => {
  if (typeof value !== 'object' || value === null) return 'is not an object';
  const v = value as Record<string, unknown>;
  if (!isSelectionId(v.id)) return `has an invalid "id" (${String(v.id)})`;
  if (typeof v.params !== 'object' || v.params === null) return `id "${String(v.id)}": is missing a "params" object`;
  if (typeof v.counts !== 'object' || v.counts === null) return `id "${String(v.id)}": is missing a "counts" object`;
  if (typeof v.globalGain !== 'number') return `id "${String(v.id)}": is missing a number "globalGain"`;
  if (typeof v.graphSha !== 'string') return `id "${String(v.id)}": is missing a string "graphSha"`;
  if (typeof v.compiledFromGitRevision !== 'string') return `id "${String(v.id)}": is missing a string "compiledFromGitRevision"`;
  if (typeof v.compilerSourceSha256 !== 'string') return `id "${String(v.id)}": is missing a string "compilerSourceSha256"`;

  const nullIssue = nullReason(v.null);
  if (nullIssue) return `id "${String(v.id)}": null ${nullIssue}`;
  const explanationIssue = explanationReason(v.explanation);
  if (explanationIssue) return `id "${String(v.id)}": explanation ${explanationIssue}`;
  const pathwayIssue = pathwayReason(v.pathway);
  if (pathwayIssue) return `id "${String(v.id)}": pathway ${pathwayIssue}`;
  const coverageIssue = coverageReason(v.coverage);
  if (coverageIssue) return `id "${String(v.id)}": coverage ${coverageIssue}`;

  if (typeof v.categorized !== 'boolean') return `id "${String(v.id)}": is missing a boolean "categorized"`;
  // Recomputed, not trusted: mirrors `selection-report.ts`'s own guard --
  // `00-overview.md`'s "a selection whose null, C, or M is degenerate ...
  // is not categorized" (plus this producer's own coverage.flagged guard).
  const nullField = v.null as SelectionNullResult;
  const pathwayField = v.pathway as SelectionPathwayResult;
  const coverageField = v.coverage as SelectionCoverageResult;
  const recomputedCategorized = !nullField.degenerate && !pathwayField.cDegenerate && !pathwayField.mDegenerate && !coverageField.flagged;
  if (v.categorized !== recomputedCategorized) return `id "${String(v.id)}": categorized disagrees with null/pathway/coverage degeneracy`;
  if (!v.categorized && (typeof v.categorizedReason !== 'string' || v.categorizedReason.length === 0)) {
    return `id "${String(v.id)}": categorized is false but has no "categorizedReason" string`;
  }
  return undefined;
};

const isSelection = (value: unknown): value is SelectionResult => selectionReason(value) === undefined;

/** `scripts/selections/selection-report.ts`'s own `aggregateVerdict` -- reimplemented (not imported), same per-finding-across-selections rule: a categorized finding that fails makes the whole verdict `false` regardless of any other finding being uncategorized on the same or another selection; `'indeterminate'` only when nothing failed and at least one finding is uncategorized. */
const recomputeVerdict = (selections: readonly SelectionResult[]): RobustnessVerdict => {
  let anyGap = false;
  for (const selection of selections) {
    if (!selection.null.holds) return { verdict: false };
    if (!selection.explanation.replicates) return { verdict: false };
    if (!selection.categorized) {
      anyGap = true;
    } else if (!selection.pathway.supported) {
      return { verdict: false };
    }
  }
  return anyGap ? { verdict: 'indeterminate' } : { verdict: true };
};

const verdictReason = (value: unknown): string | undefined => {
  if (typeof value !== 'object' || value === null) return 'is not an object';
  const v = value as Record<string, unknown>;
  if (v.verdict !== true && v.verdict !== false && v.verdict !== 'indeterminate') return 'has an invalid "verdict"';
  if (v.verdict !== true && typeof v.reason !== 'string') return 'is not true but has no string "reason"';
  return undefined;
};

const validateShape = (value: unknown): { ok: true; data: SelectionRobustnessArtifact } | { ok: false; reason: string } => {
  if (typeof value !== 'object' || value === null) return { ok: false, reason: 'selection-robustness artifact is not a JSON object' };
  const v = value as Record<string, unknown>;
  if (v.version !== 1) return { ok: false, reason: `selection-robustness artifact has unsupported version ${String(v.version)}` };

  const sources = v.sources as Record<string, unknown> | undefined;
  if (
    !sources ||
    typeof sources.defaultGraphSha !== 'string' ||
    !Array.isArray(sources.rawFileShas) ||
    typeof sources.compilerSourceSha !== 'string'
  ) {
    return { ok: false, reason: 'selection-robustness artifact has a malformed "sources" object' };
  }

  if (!Array.isArray(v.selections) || v.selections.length !== SELECTION_IDS.length) {
    return { ok: false, reason: `selection-robustness artifact "selections" must have exactly ${SELECTION_IDS.length} entries` };
  }
  for (const selection of v.selections) {
    const reason = selectionReason(selection);
    if (reason) return { ok: false, reason: `selection-robustness artifact selections[]: ${reason}` };
  }
  const selections = v.selections as SelectionResult[];
  const ids = new Set(selections.map((selection) => selection.id));
  for (const id of SELECTION_IDS) {
    if (!ids.has(id)) return { ok: false, reason: `selection-robustness artifact "selections" is missing "${id}"` };
  }

  const overall = v.overall as Record<string, unknown> | undefined;
  if (!overall) return { ok: false, reason: 'selection-robustness artifact is missing "overall"' };
  for (const key of ['robustToSize', 'robustToMethod', 'mapping'] as const) {
    const reason = verdictReason(overall[key]);
    if (reason) return { ok: false, reason: `selection-robustness artifact overall.${key} ${reason}` };
  }

  const byId = new Map(selections.map((selection) => [selection.id, selection] as const));
  const smaller = byId.get('smaller') as SelectionResult;
  const larger = byId.get('larger') as SelectionResult;
  const randomBridge = byId.get('random-bridge') as SelectionResult;
  const altSensoryMapping = byId.get('alt-sensory-mapping') as SelectionResult;

  const recomputedRobustToSize = recomputeVerdict([smaller, larger]);
  const recomputedRobustToMethod = recomputeVerdict([randomBridge]);
  const recomputedMapping = recomputeVerdict([altSensoryMapping]);
  const shippedOverall = overall as unknown as SelectionRobustnessOverall;
  if (shippedOverall.robustToSize.verdict !== recomputedRobustToSize.verdict) {
    return { ok: false, reason: 'selection-robustness artifact overall.robustToSize disagrees with its own per-selection data' };
  }
  if (shippedOverall.robustToMethod.verdict !== recomputedRobustToMethod.verdict) {
    return { ok: false, reason: 'selection-robustness artifact overall.robustToMethod disagrees with its own per-selection data' };
  }
  if (shippedOverall.mapping.verdict !== recomputedMapping.verdict) {
    return { ok: false, reason: 'selection-robustness artifact overall.mapping disagrees with its own per-selection data' };
  }

  return {
    ok: true,
    data: {
      version: 1,
      sources: sources as SelectionRobustnessArtifact['sources'],
      selections,
      overall: shippedOverall,
      host: v.host as SelectionRobustnessArtifact['host']
    }
  };
};

/**
 * Fetch, sha256-verify, and structurally validate `selection-robustness-v1.json`
 * (`manifest.selectionRobustness`), then cross-check it against the
 * manifest's own pinned default graph sha -- never throws, matching every
 * sibling loader's "never throws, always return a reasoned status"
 * contract, so a missing/tampered/malformed artifact only ever hides or
 * degrades the Findings panel's selection-robustness step, never the rest
 * of the experiment.
 *
 * `dataBaseUrl` must be the same value the caller passes to the other
 * `loadX` functions (`${import.meta.env.BASE_URL}data` in production) so
 * this artifact resolves under the app's real deployment base path too.
 */
export const loadSelectionRobustness = async (
  manifest: ArenaManifest,
  dataBaseUrl: string
): Promise<SelectionRobustnessLoadResult> => {
  const fetched = await fetchAndVerifySidecarJson(manifest.selectionRobustness, dataBaseUrl, 'selection-robustness artifact');
  if (fetched.status === 'no-entry') {
    return { status: 'missing', reason: 'The manifest has no selectionRobustness artifact entry.' };
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

  if (data.sources.defaultGraphSha !== manifest.binarySha256) {
    return {
      status: 'invalid',
      reason: `selection-robustness sources.defaultGraphSha does not match the manifest's compiled biological graph (${manifest.binarySha256}) — stale artifact`
    };
  }

  return { status: 'ok', data };
};
