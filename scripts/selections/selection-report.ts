import { mkdirSync, readFileSync } from 'node:fs';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { requireValue } from '../training/cli';
import { atomicWriteFileSync, sha256Hex } from '../training/fsio';
import { collectRepoRelativeDependencies, computeSourceIdentitySha256 } from '../lib/import-graph';
import { sortKeysDeep, verifyManifestRoundTrips } from '../null/null-report';
import { quantileIndex } from '../null/null-stats';
import { renderSelectionRobustnessReportMarkdown } from './selection-report-markdown';

/**
 * WP3 of `.agents/plans/selection-robustness` (`03-artifact-and-findings.md`):
 * combines WP1's four alternative-selection graph ledgers and WP2's
 * per-selection null/explanation/coverage/intervention chain outputs
 * (`training/runs/selections/<id>/*.json`, gitignored scratch -- see
 * `02-per-selection-chain.md`) into `public/data/selection-robustness-v1.json`
 * (manifest key `selectionRobustness`) and `docs/selection-robustness-report.md`.
 * Same "one producer script combining upstream per-item JSON into one
 * artifact + report" shape `scripts/null/task-generality-report.ts` and
 * `scripts/atlas/repertoire-report.ts` already establish -- this file holds
 * the types, the pure builder, and the CLI; `./selection-report-markdown.ts`
 * (a separate file, following `repertoire-report.ts`/`repertoire-report-markdown.ts`'s
 * own split) holds the report renderer, so this file stays well under the
 * repo's 1000-line-per-file limit.
 *
 * Every number in the artifact is read once from an already-computed WP2
 * output (never re-simulated) and several summary flags are independently
 * *recomputed* here from raw per-item data rather than trusted from the
 * upstream JSON -- the same "recompute, don't trust another producer's own
 * summary field" discipline `taskGenerality.ts`'s loader and
 * `task-generality-report.ts`'s builder already apply, extended to two
 * places `00-overview.md` calls out by name:
 *
 * 1. **Null holds.** `00-overview.md`'s "Predeclared outcome per selection"
 *    defines it as "biological is below that selection's rewired-null 25th
 *    percentile" -- a *value* comparison, not `rewiring-null.json`'s own
 *    rank-based `bioPercentile < 0.25`. The two conventions can disagree at
 *    a close boundary (a thermo-methodology review of the WP2 chain results,
 *    item C1, found exactly this for `random-bridge`: `bioPercentile` is
 *    24.8% -- just under the 25% rank cutoff -- while the *value* convention
 *    gives the same answer only because `bioScore` (2.9527) also happens to
 *    sit just under `p25` (2.9802), a margin of a single rewired graph's
 *    rank). This module recomputes `p25` itself from the raw 500 rewired
 *    scores (`rewiring-null.json`'s own `rewired[].score` array) using
 *    `quantileIndex(500, 0.25)` -- `scripts/null/null-stats.ts`'s own
 *    low-tail-floor convention, the same one `intervention-report.ts`'s
 *    `publishedNullFloorValue` and `interventions.py`'s
 *    `_regenerate_null_targets` already use for the identical "25th
 *    percentile" language elsewhere in this study -- and cross-checks the
 *    result against `intervention-stats.json`'s own `publishedNullFloor`
 *    (computed by that same Python-side convention, over the same raw
 *    scores) as a belt-and-suspenders consistency check: the two must agree
 *    bit-for-bit, or this producer refuses to publish. `holds` is then the
 *    literal value comparison `bioScore < p25`, and the artifact always
 *    carries `bioScore`/`bioPercentile`/`p25` alongside the boolean so a
 *    reader can see exactly how close a call it was, per that same review's
 *    recommendation.
 * 2. **Explanation replicates / structuralReplicates.** Recomputed from each
 *    metric's raw `bioPercentile`/`spearman` against the published
 *    `thresholds.spearmanRho` and the null's own 2.5-97.5% band (i.e.
 *    `bioPercentile <= 0.025 || bioPercentile >= 0.975`), not trusted from
 *    `explain.py`'s own `metrics[].qualifiesBothGates` flag -- `00-overview.md`'s
 *    exact words: "`selection-report.ts` recomputes both flags **in
 *    TypeScript** from the raw `spearman` and `bioPercentile` values and the
 *    published thresholds. It does not trust the flags from `explain.py`."
 *    `replicates` is true only if `T:rightClearance->thrust` or
 *    `T:forwardClearance->thrust` (the two original gate-passing transfer
 *    entries `00-overview.md` names) independently qualifies;
 *    `structuralReplicates` is `weightedInDegree:thrust` alone and never
 *    decides `replicates` (a thermo-methodology review of the WP2 chain
 *    results, item I2: this is exactly what makes `random-bridge`'s
 *    "explanation replicates" a clean, non-degenerate `false` -- neither
 *    named transfer entry qualifies there, even though the structural
 *    feature does).
 *
 * Two fields extend the plan's own worked artifact shape beyond
 * `03-artifact-and-findings.md`'s literal field list, both because the plan
 * itself calls for the underlying disclosure and because a real chain run
 * surfaced a case that needs them to be legible rather than a bare tag:
 *
 * - `pathway.q*`/`channelSpecific`/`mqDegenerate`/`degenerateMechanism`:
 *   `00-overview.md`'s pathway rule and the WP2 chain's own
 *   `intervention-stats.json` both carry a Q (channel-specific) leg
 *   alongside P, and `random-bridge`'s own P/Q are `degenerate` for a
 *   specific, mechanical reason (its search reaches the transfer target
 *   with **zero** swaps, so the size-matched C/M/MQ controls are literally
 *   unperturbed copies of biological -- confirmed by that review's item
 *   I1). Surfacing *why* a selection is degenerate, not just tagging it
 *   `"degenerate"`, is what the task brief for this WP explicitly asks for.
 * - `explanation.mirroredBioPercentile`/`singleAxis`: `00-overview.md`'s
 *   "null-explanation" method runs a mirrored-decoder check on every
 *   selection, and conditionally runs two single-axis variants only when
 *   the mirrored run clears the 25% threshold (`.agents/plans/null-explanation/00-overview.md:35`).
 *   `random-bridge` is the one selection where that condition fired
 *   (mirrored 29.4%), so its single-axis results exist and are reported;
 *   the other three selections' `singleAxis` is `null` because the
 *   condition never fired for them, not because the data is missing.
 *
 * The plan's own `search-limited` category and bridge-candidate-pool
 * disclosure (`00-overview.md`'s "Search-budget disclosure") were **not**
 * implemented anywhere in WP2's merged code (confirmed by the same thermo
 * review, item I4 -- `intervention-stats.json` carries no `k`/`targetReached`/
 * pool-size field). This producer closes that gap itself, at the WP3 layer,
 * rather than disclosing it as an unfixed gap: `k`/`targetReached` come from
 * `interventions/index.json`'s own `kP`/entries (which *do* carry per-graph
 * `swaps`/`targetReached`, just not surfaced into `intervention-stats.json`),
 * and `bridgePoolSize` comes from the selection's own ledger
 * `selectionCounts.bridgeCandidateCount` (already recorded there for every
 * selection). `searchLimited` is `k >= maxSwaps && !targetReached`; none of
 * the four selections in this study's own run trips it (every P/Q search
 * reached its target well under the 200-swap cap), but the field is real
 * and load-bearing, not a placeholder -- a future selection that does hit
 * the cap will be reported `search-limited`, never silently folded into
 * `not-supported`.
 *
 * Deterministic and re-run-safe: every input is read once from disk, this
 * module never reads the clock or a random source, and `sortKeysDeep`+
 * `JSON.stringify` never iterate a `Map`/`Set` -- running this CLI twice
 * against the same inputs produces byte-identical `selection-robustness-v1.json`
 * bytes (`03-artifact-and-findings.md`'s "Regeneration on the Spark is
 * byte-identical" acceptance criterion).
 */

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '../..');

export const DEFAULT_SELECTIONS_DIR = resolve(repoRoot, 'training/runs/selections');
export const DEFAULT_MANIFEST = resolve(repoRoot, 'public/data/malecns-arena-v1.manifest.json');
export const DEFAULT_OUT = resolve(repoRoot, 'public/data/selection-robustness-v1.json');
export const DEFAULT_REPORT_MD = resolve(repoRoot, 'docs/selection-robustness-report.md');

/** `scripts/data/selections.py`'s `SELECTIONS` keys, minus `"default"` (the shipped baseline, never a WP3 row) -- the four predeclared alternative selections `00-overview.md`'s table names. */
export const SELECTION_IDS = ['larger', 'smaller', 'random-bridge', 'alt-sensory-mapping'] as const;
export type SelectionId = (typeof SELECTION_IDS)[number];

/**
 * `761r`'s reported variant binary shas (`.agents/plans/selection-robustness/00-overview.md`'s
 * repository-findings table cites them; this WP's own task brief restates
 * them), asserted against each selection's own ledger `binarySha256` at
 * build time -- the task brief's explicit "confirm the graph binary shas
 * are unchanged" requirement, enforced mechanically rather than left to a
 * human diff. `smaller`/`larger` were recompiled at this repo's own HEAD
 * (to pick up the I5 ledger fix below); this assertion is exactly what
 * proves that recompile reproduced the same graph bytes as the original WP1
 * compile, not a new graph.
 */
const EXPECTED_GRAPH_SHA_PREFIX: Readonly<Record<SelectionId, string>> = {
  larger: '6a6304aa',
  smaller: '47992d46',
  'random-bridge': '34076335',
  'alt-sensory-mapping': '25d48a35'
};

// ---------------------------------------------------------------------------
// Producer code identity
// ---------------------------------------------------------------------------

export interface SelectionRobustnessProducer {
  readonly script: string;
  readonly sourceSha256: string;
  readonly dependencies: readonly string[];
}

/** Same real-import-graph code-identity scheme as `task-generality-report.ts`'s `taskGeneralityProducer`/`repertoire-report.ts`'s producer -- `scripts/lib/import-graph.ts`'s `collectRepoRelativeDependencies`, not a hand-maintained filename list. Its dependency closure includes `./selection-report-markdown.ts` and `../null/null-stats.ts` (the `quantileIndex` recomputation above), since both are statically imported from this file. */
export const selectionRobustnessProducer = (): SelectionRobustnessProducer => {
  const dependencies = collectRepoRelativeDependencies(fileURLToPath(import.meta.url), repoRoot);
  return {
    script: 'scripts/selections/selection-report.ts',
    sourceSha256: computeSourceIdentitySha256(repoRoot, dependencies),
    dependencies
  };
};

// ---------------------------------------------------------------------------
// Artifact types
// ---------------------------------------------------------------------------

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
  /** Non-`null` only for the one selection whose mirrored run cleared the conditional 25% threshold (`00-overview.md`'s top doc comment above); see `.agents/plans/null-explanation/00-overview.md:35`. */
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
  /** `pScore >= null.p25 && pScore > cP95 && pScore > mP95` -- the mechanical rule, computed even when `cDegenerate`/`mDegenerate` is true (see this file's top doc comment: for `random-bridge`, `pScore === cP95 === mP95` exactly, so `pScore > cP95` is honestly `false` without any special-casing). A `false` here alongside `cDegenerate`/`mDegenerate: true` is a degenerate result, not a real "not-supported" categorization -- see `SelectionResult.categorized`. */
  readonly supported: boolean;
  readonly qScore: number;
  readonly qK: number;
  readonly qTargetReached: boolean;
  readonly mqDegenerate: boolean;
  /** `false | true | 'degenerate'` -- `intervention-stats.json`'s own `q.channelSpecific` field carries the same three-state shape for exactly the same reason (the MQ control arm can itself be degenerate). */
  readonly channelSpecific: boolean | 'degenerate';
  /** Present only when `cDegenerate || mDegenerate` -- states the mechanism (never a bare "degenerate" tag), per this file's top doc comment. */
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
  /** `false` when this selection's null, C, or M is degenerate (the IQR guard), or a coverage channel/population is flagged -- `00-overview.md`'s "A selection whose null, C, or M is degenerate ... is not categorized" and "a selection with a flagged channel is not categorized for the pathway finding." */
  readonly categorized: boolean;
  /** Present only when `categorized` is `false`; states which guard(s) fired and why (never a bare tag). */
  readonly categorizedReason?: string;
}

export interface RobustnessVerdict {
  readonly verdict: true | false | 'indeterminate';
  /** Present unless `verdict === true`. */
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
    /** The current `compile.py`+`selections.py` source sha, taken from the freshly-recompiled `smaller`/`larger` ledgers (regenerated at this repo's own HEAD to pick up the I5 fix below). `random-bridge`/`alt-sensory-mapping`'s own ledgers may show an older `compilerSourceSha256`, since their graphs were compiled at an earlier revision and reused unchanged (their `binarySha256` is confirmed byte-identical, see `EXPECTED_GRAPH_SHA_PREFIX`) -- each selection's own `compiledFromGitRevision`/`compilerSourceSha256` below is its real, individual production provenance. */
    readonly compilerSourceSha: string;
    readonly producer: SelectionRobustnessProducer;
  };
  readonly selections: readonly SelectionResult[];
  readonly overall: SelectionRobustnessOverall;
  readonly host: { readonly arch: string; readonly node: string };
}

// ---------------------------------------------------------------------------
// Raw-JSON reading helpers
// ---------------------------------------------------------------------------

const readJson = (path: string): Record<string, any> => JSON.parse(readFileSync(path, 'utf8'));

const requireField = (obj: Record<string, any>, key: string, label: string): any => {
  if (obj[key] === undefined) throw new Error(`selection-report: ${label} is missing "${key}"`);
  return obj[key];
};

interface SelectionInputPaths {
  readonly ledger: string;
  readonly rewiringNull: string;
  readonly nullExplanation: string;
  readonly coverage: string;
  readonly interventionStats: string;
  readonly interventionsIndex: string;
}

const selectionInputPaths = (selectionsDir: string, id: SelectionId): SelectionInputPaths => {
  const dir = join(selectionsDir, id);
  return {
    ledger: join(dir, `malecns-arena-${id}.ledger.json`),
    rewiringNull: join(dir, 'rewiring-null.json'),
    nullExplanation: join(dir, 'null-explanation.json'),
    coverage: join(dir, 'coverage.json'),
    interventionStats: join(dir, 'intervention-stats.json'),
    interventionsIndex: join(dir, 'interventions', 'index.json')
  };
};

// ---------------------------------------------------------------------------
// Per-selection builder
// ---------------------------------------------------------------------------

const buildNullResult = (id: SelectionId, rewiringNull: Record<string, any>, publishedNullFloor: number): SelectionNullResult => {
  const rewired = requireField(rewiringNull, 'rewired', `${id} rewiring-null.json`) as readonly { readonly score: number }[];
  const scores = rewired.map((entry) => requireField(entry, 'score', `${id} rewiring-null.json rewired[]`) as number);
  const sorted = [...scores].sort((a, b) => a - b);
  const p25 = sorted[quantileIndex(sorted.length, 0.25)];
  if (Math.abs(p25 - publishedNullFloor) > 1e-9) {
    throw new Error(
      `selection-report: ${id} p25 recomputed from rewiring-null.json's raw scores (${p25}) disagrees with ` +
        `intervention-stats.json's publishedNullFloor (${publishedNullFloor}) -- refusing to publish an ` +
        'internally-inconsistent null-holds boundary'
    );
  }
  const nullSummary = requireField(rewiringNull, 'null', `${id} rewiring-null.json`);
  const bioScore = requireField(requireField(rewiringNull, 'biological', `${id} rewiring-null.json`), 'score', `${id} rewiring-null.json biological`);
  const bioPercentile = requireField(rewiringNull, 'bioPercentile', `${id} rewiring-null.json`) as number;
  return {
    bioScore,
    bioPercentile,
    p25,
    degenerate: Boolean(nullSummary.degenerate),
    // `00-overview.md`'s literal rule is a *value* comparison against p25 --
    // see this file's top doc comment for why this, not the rank-based
    // `bioPercentile < 0.25`, is the canonical convention here.
    holds: bioScore < p25
  };
};

const buildExplanationResult = (id: SelectionId, nullExplanation: Record<string, any>): SelectionExplanationResult => {
  const metrics = requireField(nullExplanation, 'metrics', `${id} null-explanation.json`) as readonly Record<string, any>[];
  const thresholds = requireField(nullExplanation, 'thresholds', `${id} null-explanation.json`);
  const spearmanRho = requireField(thresholds, 'spearmanRho', `${id} null-explanation.json thresholds`) as number;
  // Recomputed, not trusted from `metrics[].qualifiesBothGates` -- see this
  // file's top doc comment, item 2. `requireField` (not a bare property
  // read) for both inputs: a metric missing either field must throw loudly,
  // matching this file's own "throw on missing field" discipline elsewhere
  // -- a silent `undefined` here would otherwise just evaluate every
  // comparison to `false`, quietly dropping a real qualifying metric.
  const qualifiesBothGates = (metric: Record<string, any>): boolean => {
    const bioPercentile = requireField(metric, 'bioPercentile', `${id} null-explanation.json metrics[${metric.name as string}]`) as number;
    const spearman = requireField(metric, 'spearman', `${id} null-explanation.json metrics[${metric.name as string}]`) as number;
    const outsideBand = bioPercentile <= 0.025 || bioPercentile >= 0.975;
    return outsideBand && Math.abs(spearman) >= spearmanRho;
  };
  const passing = metrics
    .filter(qualifiesBothGates)
    .map((metric) => ({ name: metric.name as string, spearman: metric.spearman as number, bioPercentile: metric.bioPercentile as number }));
  const passingNames = new Set(passing.map((metric) => metric.name));
  const replicates = passingNames.has('T:rightClearance->thrust') || passingNames.has('T:forwardClearance->thrust');
  const structuralReplicates = passingNames.has('weightedInDegree:thrust');

  const variants = requireField(nullExplanation, 'variants', `${id} null-explanation.json`);
  const flipBoth = requireField(variants, 'flipBoth', `${id} null-explanation.json variants`);
  const mirroredBioPercentile = requireField(flipBoth, 'bioPercentile', `${id} null-explanation.json variants.flipBoth`) as number;
  const singleAxis =
    variants.flipThrust && variants.flipYaw
      ? { flipThrust: variants.flipThrust.bioPercentile as number, flipYaw: variants.flipYaw.bioPercentile as number }
      : null;

  return {
    passing,
    replicates,
    structuralReplicates,
    mirroredBioPercentile,
    singleAxis,
    exploratory: null,
    exploratoryOmittedReason: requireField(nullExplanation, 'exploratoryOmittedReason', `${id} null-explanation.json`) as string
  };
};

const degenerateMechanism = (k: number): string =>
  `P's search reached its transfer target with ${k} swap${k === 1 ? '' : 's'}` +
  (k === 0
    ? ", so the size-matched C/M/MQ controls are unperturbed copies of biological and cannot be distinguished from it"
    : '');

const buildPathwayResult = (
  id: SelectionId,
  interventionStats: Record<string, any>,
  interventionsIndex: Record<string, any>,
  bridgePoolSize: number,
  nullP25: number
): SelectionPathwayResult => {
  const p = requireField(interventionStats, 'p', `${id} intervention-stats.json`);
  const q = requireField(interventionStats, 'q', `${id} intervention-stats.json`);
  const controls = requireField(interventionStats, 'controls', `${id} intervention-stats.json`);
  const armDegeneracy = requireField(interventionStats, 'armDegeneracy', `${id} intervention-stats.json`);
  const cP95 = requireField(requireField(controls, 'C', `${id} intervention-stats.json controls`), 'p95', `${id} controls.C`) as number;
  const mP95 = requireField(requireField(controls, 'M', `${id} intervention-stats.json controls`), 'p95', `${id} controls.M`) as number;
  const cDegenerate = Boolean(requireField(armDegeneracy, 'cArm', `${id} armDegeneracy`).degenerate);
  const mDegenerate = Boolean(requireField(armDegeneracy, 'mArm', `${id} armDegeneracy`).degenerate);
  const mqDegenerate = Boolean(requireField(armDegeneracy, 'mqArm', `${id} armDegeneracy`).degenerate);

  const maxSwaps = requireField(interventionsIndex, 'maxSwaps', `${id} interventions/index.json`) as number;
  const kP = requireField(interventionsIndex, 'kP', `${id} interventions/index.json`) as number;
  const kQ = requireField(interventionsIndex, 'kQ', `${id} interventions/index.json`) as number;
  const entries = requireField(interventionsIndex, 'entries', `${id} interventions/index.json`) as readonly Record<string, any>[];
  const pEntry = entries.find((entry) => entry.id === 'P');
  const qEntry = entries.find((entry) => entry.id === 'Q');
  if (!pEntry || !qEntry) throw new Error(`selection-report: ${id} interventions/index.json is missing its "P"/"Q" entries`);

  const pScore = requireField(p, 'score', `${id} intervention-stats.json p`) as number;
  const supported = pScore >= nullP25 && pScore > cP95 && pScore > mP95;

  return {
    pScore,
    cP95,
    mP95,
    cDegenerate,
    mDegenerate,
    k: kP,
    targetReached: Boolean(pEntry.targetReached),
    bridgePoolSize,
    maxSwaps,
    searchLimited: kP >= maxSwaps && !pEntry.targetReached,
    supported,
    qScore: requireField(q, 'score', `${id} intervention-stats.json q`) as number,
    qK: kQ,
    qTargetReached: Boolean(qEntry.targetReached),
    mqDegenerate,
    channelSpecific: q.channelSpecific,
    ...(cDegenerate || mDegenerate ? { degenerateMechanism: degenerateMechanism(kP) } : {})
  };
};

const buildCoverageResult = (coverage: Record<string, any>): SelectionCoverageResult => ({
  perChannel: coverage.perChannel,
  perPopulation: coverage.perPopulation,
  flagged: Boolean(coverage.flagged)
});

const buildSelectionResult = (selectionsDir: string, id: SelectionId): SelectionResult => {
  const paths = selectionInputPaths(selectionsDir, id);
  const ledger = readJson(paths.ledger);
  const rewiringNull = readJson(paths.rewiringNull);
  const nullExplanation = readJson(paths.nullExplanation);
  const coverage = readJson(paths.coverage);
  const interventionStats = readJson(paths.interventionStats);
  const interventionsIndex = readJson(paths.interventionsIndex);

  const graphSha = requireField(ledger, 'binarySha256', `${id} ledger`) as string;
  const expectedPrefix = EXPECTED_GRAPH_SHA_PREFIX[id];
  if (!graphSha.startsWith(expectedPrefix)) {
    throw new Error(
      `selection-report: ${id}'s ledger binarySha256 (${graphSha}) does not start with the expected ` +
        `${expectedPrefix} -- the compiled graph has changed since WP1; investigate before publishing`
    );
  }

  const selectionCounts = requireField(ledger, 'selectionCounts', `${id} ledger`);
  const bridgePoolSize = requireField(selectionCounts, 'bridgeCandidateCount', `${id} ledger selectionCounts`) as number;
  const publishedNullFloor = requireField(interventionStats, 'publishedNullFloor', `${id} intervention-stats.json`) as number;

  const nullResult = buildNullResult(id, rewiringNull, publishedNullFloor);
  const explanation = buildExplanationResult(id, nullExplanation);
  const pathway = buildPathwayResult(id, interventionStats, interventionsIndex, bridgePoolSize, nullResult.p25);
  const coverageResult = buildCoverageResult(coverage);

  const categorizedReasons: string[] = [];
  if (nullResult.degenerate) categorizedReasons.push("the rewired-null distribution is degenerate (IQR guard)");
  if (pathway.cDegenerate || pathway.mDegenerate) categorizedReasons.push(degenerateMechanism(pathway.k) + ' (IQR guard on the C/M control arms)');
  if (coverageResult.flagged) categorizedReasons.push('a sensory channel or descending population has zero bridge coverage');
  const categorized = categorizedReasons.length === 0;

  return {
    id,
    params: requireField(requireField(ledger, 'selection', `${id} ledger`), 'params', `${id} ledger.selection`),
    counts: selectionCounts,
    globalGain: requireField(requireField(ledger, 'dynamics', `${id} ledger`), 'globalGain', `${id} ledger.dynamics`) as number,
    graphSha,
    compiledFromGitRevision: requireField(ledger, 'compiledFromGitRevision', `${id} ledger`) as string,
    compilerSourceSha256: requireField(ledger, 'compilerSourceSha256', `${id} ledger`) as string,
    null: nullResult,
    explanation,
    pathway,
    coverage: coverageResult,
    categorized,
    ...(categorized ? {} : { categorizedReason: categorizedReasons.join('; ') })
  };
};

// ---------------------------------------------------------------------------
// Overall robustness aggregation
// ---------------------------------------------------------------------------

/**
 * `00-overview.md`'s aggregation rule, applied per finding across the given
 * selection(s): a categorized (non-degenerate) finding that fails makes the
 * whole verdict `false`, *regardless* of whether another finding on the same
 * selection is uncategorized -- a thermo-methodology review of the WP2 chain
 * results (item I2) found this is exactly what decides `random-bridge`'s
 * "robust to method" = `false`: its explanation leg is a clean, categorized
 * failure even though its pathway leg is separately degenerate/uncategorized.
 * Only when *no* finding fails, and at least one is uncategorized, is the
 * result `"indeterminate"`.
 */
const aggregateVerdict = (selections: readonly SelectionResult[]): RobustnessVerdict => {
  const failures: string[] = [];
  const gaps: string[] = [];
  for (const selection of selections) {
    if (!selection.null.holds) failures.push(`${selection.id}: null does not hold`);
    if (!selection.explanation.replicates) failures.push(`${selection.id}: explanation does not replicate`);
    if (!selection.categorized) {
      gaps.push(`${selection.id}: pathway is not categorized (${selection.categorizedReason})`);
    } else if (!selection.pathway.supported) {
      failures.push(`${selection.id}: pathway is not-supported`);
    }
  }
  if (failures.length > 0) return { verdict: false, reason: failures.join('; ') };
  if (gaps.length > 0) return { verdict: 'indeterminate', reason: gaps.join('; ') };
  return { verdict: true };
};

const buildOverall = (selections: readonly SelectionResult[]): SelectionRobustnessOverall => {
  const byId = new Map(selections.map((selection) => [selection.id, selection] as const));
  const require = (id: SelectionId): SelectionResult => {
    const selection = byId.get(id);
    if (!selection) throw new Error(`selection-report: missing selection "${id}"`);
    return selection;
  };
  return {
    // "larger, default, and smaller are nested cuts of one degree ranking,
    // so together they test bridge-population size" (`00-overview.md`) --
    // "default" is the shipped baseline (always at the extreme 0th
    // percentile, never re-run here), so this checks the two alternative
    // sizes against the same rule "default" already satisfies.
    robustToSize: aggregateVerdict([require('smaller'), require('larger')]),
    robustToMethod: aggregateVerdict([require('random-bridge')]),
    mapping: aggregateVerdict([require('alt-sensory-mapping')])
  };
};

// ---------------------------------------------------------------------------
// Top-level artifact builder
// ---------------------------------------------------------------------------

export interface BuildSelectionRobustnessArtifactInputs {
  readonly selectionsDir: string;
  readonly defaultGraphSha: string;
}

export const buildSelectionRobustnessArtifact = (inputs: BuildSelectionRobustnessArtifactInputs): SelectionRobustnessArtifact => {
  const selections = SELECTION_IDS.map((id) => buildSelectionResult(inputs.selectionsDir, id));

  const smallerLedger = readJson(selectionInputPaths(inputs.selectionsDir, 'smaller').ledger);
  const rawFileShas = (requireField(smallerLedger, 'sourceFiles', 'smaller ledger') as readonly { readonly sha256: string }[])
    .map((entry) => entry.sha256)
    .sort();

  return {
    version: 1,
    sources: {
      defaultGraphSha: inputs.defaultGraphSha,
      rawFileShas,
      compilerSourceSha: (selections.find((selection) => selection.id === 'smaller') as SelectionResult).compilerSourceSha256,
      producer: selectionRobustnessProducer()
    },
    selections,
    overall: buildOverall(selections),
    host: { arch: process.arch, node: process.version }
  };
};

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------

/** Add/overwrite the manifest's `selectionRobustness` key in place -- mirrors `task-generality-report.ts`'s `updateManifestWithTaskGenerality`. */
export const updateManifestWithSelectionRobustness = (
  manifestPath: string,
  entry: { readonly artifact: string; readonly sha256: string }
): void => {
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8')) as Record<string, unknown>;
  manifest.selectionRobustness = entry;
  atomicWriteFileSync(manifestPath, `${JSON.stringify(sortKeysDeep(manifest), null, 2)}\n`);
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

export interface SelectionReportArgs {
  readonly selectionsDir: string;
  readonly manifest: string;
  readonly out: string;
  readonly reportMd: string;
  readonly skipManifestUpdate: boolean;
}

export const parseSelectionReportArgs = (argv: readonly string[]): SelectionReportArgs => {
  let selectionsDir = DEFAULT_SELECTIONS_DIR;
  let manifest = DEFAULT_MANIFEST;
  let out = DEFAULT_OUT;
  let reportMd = DEFAULT_REPORT_MD;
  let skipManifestUpdate = false;

  let i = 0;
  while (i < argv.length) {
    const flag = argv[i];
    if (flag === '--selections-dir') {
      selectionsDir = resolve(process.cwd(), requireValue(flag, argv[i + 1]));
      i += 2;
    } else if (flag === '--manifest') {
      manifest = resolve(process.cwd(), requireValue(flag, argv[i + 1]));
      i += 2;
    } else if (flag === '--out') {
      out = resolve(process.cwd(), requireValue(flag, argv[i + 1]));
      i += 2;
    } else if (flag === '--report-md') {
      reportMd = resolve(process.cwd(), requireValue(flag, argv[i + 1]));
      i += 2;
    } else if (flag === '--skip-manifest-update') {
      skipManifestUpdate = true;
      i += 1;
    } else {
      throw new Error(`Unknown argument: ${flag}`);
    }
  }

  if (resolve(out) === resolve(reportMd)) {
    throw new Error('selection-report: --out and --report-md must not be the same path');
  }
  return { selectionsDir, manifest, out, reportMd, skipManifestUpdate };
};

export interface RunSelectionReportResult {
  readonly out: string;
  readonly reportMdPath: string;
  readonly artifactSha256: string;
  readonly artifact: SelectionRobustnessArtifact;
}

export const runSelectionReport = (args: Readonly<SelectionReportArgs>): RunSelectionReportResult => {
  const manifest = JSON.parse(readFileSync(args.manifest, 'utf8')) as { readonly binarySha256?: string };
  if (!manifest.binarySha256) throw new Error(`selection-report: ${args.manifest} is missing binarySha256`);

  const artifact = buildSelectionRobustnessArtifact({ selectionsDir: args.selectionsDir, defaultGraphSha: manifest.binarySha256 });

  if (!args.skipManifestUpdate) verifyManifestRoundTrips(args.manifest);

  const artifactContents = JSON.stringify(artifact);
  const artifactSha256 = sha256Hex(artifactContents);
  const reportMdContents = renderSelectionRobustnessReportMarkdown(artifact);

  mkdirSync(dirname(args.out), { recursive: true });
  atomicWriteFileSync(args.out, artifactContents);

  if (!args.skipManifestUpdate) {
    updateManifestWithSelectionRobustness(args.manifest, { artifact: basename(args.out), sha256: artifactSha256 });
  }

  mkdirSync(dirname(args.reportMd), { recursive: true });
  atomicWriteFileSync(args.reportMd, reportMdContents);

  return { out: args.out, reportMdPath: args.reportMd, artifactSha256, artifact };
};

const main = (): void => {
  try {
    const args = parseSelectionReportArgs(process.argv.slice(2));
    const result = runSelectionReport(args);
    // eslint-disable-next-line no-console -- CLI tool: this is its user-facing output.
    console.log(
      `selection-report: wrote ${result.out} (sha256 ${result.artifactSha256}) and ${result.reportMdPath}\n` +
        `robustToSize=${result.artifact.overall.robustToSize.verdict} robustToMethod=${result.artifact.overall.robustToMethod.verdict} ` +
        `mapping=${result.artifact.overall.mapping.verdict}`
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    // eslint-disable-next-line no-console -- CLI tool: this is its user-facing error output.
    console.error(`selection-report failed: ${message}`);
    process.exit(1);
  }
};

if (process.argv[1] === fileURLToPath(import.meta.url)) main();
