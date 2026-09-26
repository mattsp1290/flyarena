import { mkdirSync, readFileSync } from 'node:fs';
import { basename, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { requireNonNegativeInt, requirePositiveInt, requireValue } from '../training/cli';
import { atomicWriteFileSync, sha256Hex } from '../training/fsio';
import { collectRepoRelativeDependencies, computeSourceIdentitySha256 } from '../lib/import-graph';
import { sortKeysDeep, verifyManifestRoundTrips } from './null-report';
import {
  parseGraphListIndexInfo,
  type ArmDistribution,
  type BiologicalReproductionCheck,
  type InterventionStatistics,
  type OutcomeCategory
} from './intervention-report';
import { parseAttribution, parseIndexGraphTransfer, TRANSFER_INPUT_CHANNELS, TRANSFER_OUTPUT_POPULATIONS, type Transfer3x8 } from './intervention-attribution';
import {
  buildTrainedStatistics,
  P_TRAINER_SEEDS,
  type PTrainerSeed,
  type TrainedOutcomeCategory,
  type TrainedSeedResult,
  type TrainedArmDistribution
} from './intervention-report-trained';
import type { NullTrainedInterventionEvaluationRaw } from './null-trained-evaluate-graph-list';

/**
 * `.agents/plans/pathway-interventions/04-report-and-ledger.md`'s WP4
 * artifact/report producer: combines WP2's `statistics.json`
 * (`intervention-report.ts`), WP1's `attribution.json`/`index.json`
 * (`intervention-attribution.ts`), and WP3's rescored `trained.json`
 * (`intervention-report-trained.ts`) into `public/data/pathway-interventions-v1.json`
 * (the `pathwayInterventions` manifest key) and `docs/pathway-interventions-report.md`.
 *
 * A separate CLI/file from `intervention-report.ts` on purpose, not an edit
 * to that file (04-report-and-ledger.md's own "extend it into the artifact
 * and report producer... keep it under 1000 lines by splitting into
 * modules if needed" — `intervention-report.ts` is already ~740 lines of
 * WP2 statistics with its own hardened test suite; folding WP4's artifact/
 * report/manifest logic into it directly would both cross the threshold
 * and risk touching WP2's already-reviewed, already-tested surface).
 * Collectively, this file plus `intervention-report.ts`/
 * `intervention-report-trained.ts`/`intervention-attribution.ts` *are* "the
 * artifact and report producer" the plan asks for — split the same way
 * `graph-list-index.ts`/`null-trained-evaluate-graph-list.ts` were already
 * split out of `null-evaluate.ts`/`null-trained-evaluate.ts` for the exact
 * same 1000-line reason.
 *
 * Deterministic and re-run-safe by construction: every input is read once
 * from disk (never re-simulated), `sortKeysDeep`+`JSON.stringify` never
 * iterate a `Map`/`Set` when producing array output, and nothing here reads
 * the clock — running this CLI twice against the same inputs produces
 * byte-identical `pathway-interventions-v1.json` bytes (04-report-and-ledger.md's
 * "Regenerating the artifact twice gives byte-identical output" acceptance
 * criterion).
 */

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '../..');

const DEFAULT_STATISTICS = resolve(repoRoot, 'training/runs/interventions/statistics.json');
const DEFAULT_ATTRIBUTION = resolve(repoRoot, 'training/runs/interventions/attribution.json');
const DEFAULT_TRAINED = resolve(repoRoot, 'training/runs/interventions/trained.json');
const DEFAULT_INDEX = resolve(repoRoot, 'training/runs/interventions/index.json');
const DEFAULT_REWIRING_NULL = resolve(repoRoot, 'public/data/rewiring-null-v1.json');
export const DEFAULT_MANIFEST = resolve(repoRoot, 'public/data/malecns-arena-v1.manifest.json');
export const DEFAULT_OUT = resolve(repoRoot, 'public/data/pathway-interventions-v1.json');
export const DEFAULT_REPORT_MD = resolve(repoRoot, 'docs/pathway-interventions-report.md');

/** `'PATH'` — this study's own fixed bootstrap seed, matching `intervention-report.ts`'s `DEFAULT_BOOTSTRAP_SEED` convention (same constant value, reused here for the trained arm's own per-seed bootstrap CIs). */
const DEFAULT_BOOTSTRAP_SEED = 0x50415448;
const DEFAULT_BOOTSTRAP_RESAMPLES = 10000;

// ---------------------------------------------------------------------------
// Producer code identity
// ---------------------------------------------------------------------------

export interface PathwayInterventionsProducer {
  readonly script: string;
  readonly sourceSha256: string;
  readonly dependencies: readonly string[];
}

/**
 * This run's TS code-identity block, hashed over the *real, walked* import
 * graph from this file (`scripts/lib/import-graph.ts`'s
 * `collectRepoRelativeDependencies`) — the same scheme
 * `scripts/null/regime-check.ts`'s `regimeProducer()` already established
 * for the null-explanation study, reused verbatim per this bean's own
 * "follow their patterns, including producer code identity as sha256 over
 * the import-graph closure" instruction. Pulls in every module this file
 * actually imports (`intervention-report.ts`, `intervention-report-trained.ts`,
 * `intervention-attribution.ts`, `null-report.ts` for the manifest helpers,
 * and everything *they* import in turn), not a hand-maintained filename
 * list.
 */
export const pathwayInterventionsProducer = (): PathwayInterventionsProducer => {
  const dependencies = collectRepoRelativeDependencies(fileURLToPath(import.meta.url), repoRoot);
  return {
    script: 'scripts/null/intervention-artifact.ts',
    sourceSha256: computeSourceIdentitySha256(repoRoot, dependencies),
    dependencies
  };
};

// ---------------------------------------------------------------------------
// Artifact shape
// ---------------------------------------------------------------------------

export interface PathwayInterventionsSources {
  /** The biological graph's compiled binary sha256 — `manifest.binarySha256`, cross-checked against every other input's own recorded source-graph sha. */
  readonly biologicalSha: string;
  readonly rewiringNullSha: string;
  readonly nullExplanationSha: string;
  readonly indexSha: string;
  /**
   * `git rev-parse HEAD` at the moment the authored (`null-evaluate.ts
   * --graph-list`) and trained (`null-trained-evaluate.ts --graph-list`)
   * runs actually executed — `null` when either evaluator ran outside a git
   * checkout. Thermo-methodology review I2: `producer.sourceSha256` below
   * is a build-time snapshot of the producer's own dependency files on
   * disk, not a stamp from the moment either multi-hour simulation run
   * itself produced its scores; these two fields close that gap. The two
   * commonly differ (the authored run is re-run closer to publish time than
   * the trained rescore, which is expensive to redo) — this is expected and
   * not cross-asserted equal; each is independently meaningful provenance
   * for its own run.
   */
  readonly authoredEvaluatorGitRev: string | null;
  readonly trainedEvaluatorGitRev: string | null;
  readonly producer: PathwayInterventionsProducer;
}

/** Shared by `interventions.P`/`interventions.Q` — WP1's swap-search outcome plus WP2's rank/category statistics for that one graph. */
export interface PathwayInterventionGraphSummary {
  readonly id: 'P' | 'Q';
  readonly score: number;
  readonly swaps: number;
  readonly targetReached: boolean;
  readonly stopReason: string;
  readonly percentileInPublishedNull: number;
}

export interface PathwayInterventionsArtifact {
  readonly version: 1;
  readonly sources: PathwayInterventionsSources;
  readonly k: number;
  readonly kQ: number;
  readonly interventions: {
    readonly P: PathwayInterventionGraphSummary & { readonly pRankAmongC: number; readonly pRankAmongM: number; readonly category: OutcomeCategory };
    readonly Q: PathwayInterventionGraphSummary & { readonly qRankAmongMQ: number; readonly channelSpecific: boolean };
    readonly R: { readonly applicable: boolean; readonly edgeCount: number; readonly reason: string };
  };
  readonly controls: {
    readonly k: number;
    readonly C: ArmDistribution;
    readonly M: ArmDistribution;
    readonly MQ: ArmDistribution & { readonly kQ: number };
  };
  readonly authored: {
    readonly category: OutcomeCategory;
    readonly pRankAmongC: number;
    readonly pRankAmongM: number;
    readonly qRankAmongMQ: number;
    readonly channelSpecific: boolean;
    readonly publishedNullFloor: number;
    readonly biologicalReproduction: BiologicalReproductionCheck;
    readonly decoder: 'authored';
    readonly seeds: { readonly start: number; readonly count: number };
    readonly ticks: number;
  };
  readonly trained: {
    readonly perSeed: Readonly<Record<PTrainerSeed, TrainedSeedResult>>;
    readonly trainedRobust: boolean;
    readonly controls: { readonly C: TrainedArmDistribution; readonly M: TrainedArmDistribution };
    readonly evaluatorGitRev: string | null;
    readonly d: number;
    readonly note: string;
  };
  readonly transferBeforeAfter: {
    readonly populations: typeof TRANSFER_OUTPUT_POPULATIONS;
    readonly channels: typeof TRANSFER_INPUT_CHANNELS;
    readonly biological: Transfer3x8;
    readonly P: Transfer3x8;
    readonly Q: Transfer3x8;
  };
  readonly host: { readonly arch: string; readonly node: string };
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

export interface BuildArtifactInputs {
  readonly statistics: Readonly<InterventionStatistics>;
  readonly attributionText: string;
  readonly attributionLabel: string;
  readonly trainedRaw: Readonly<NullTrainedInterventionEvaluationRaw>;
  readonly indexText: string;
  readonly indexLabel: string;
  readonly indexBytes: Buffer;
  readonly rewiringNullBytes: Buffer;
  readonly rewiringNullParsed: {
    readonly sourceGraphSha256: string;
    readonly trained?: { readonly rewired: readonly { readonly score: number }[] };
  };
  readonly manifestBiologicalSha: string;
  readonly manifestRewiringNullSha: string;
  readonly manifestNullExplanationSha: string;
  readonly bootstrapSeed: number;
  readonly bootstrapResamples: number;
}

/**
 * Pure builder: every input is already-read bytes/already-parsed objects
 * (no `readFileSync` inside this function) so `runPathwayInterventionsArtifact`
 * below can hash the exact bytes it also parses (the same TOCTOU-avoidance
 * discipline `intervention-report.ts`'s own doc comment documents), and so
 * this function is directly unit-testable against synthetic fixtures with
 * no filesystem at all.
 *
 * Cross-checks every input against every other before building anything —
 * a stale `statistics.json` built against a different `index.json`/
 * published null, or a manifest re-pinned to a different published null,
 * would otherwise silently publish an artifact whose `sources.*Sha` fields
 * describe a different experiment than the one the numbers actually came
 * from (the same class of bug this codebase's other loaders/producers
 * guard against throughout).
 */
export const buildPathwayInterventionsArtifact = (inputs: Readonly<BuildArtifactInputs>): PathwayInterventionsArtifact => {
  const { statistics } = inputs;
  if (statistics.version !== 1) {
    throw new Error(`intervention-artifact: statistics.json has unsupported version ${String(statistics.version)}, expected 1`);
  }
  // Checked independently of `diagnosticOnly` (a maintainability-review
  // finding): `diagnosticOnly` is only ever set by `intervention-report.ts`
  // itself when `--allow-reproduction-mismatch` was passed, so a
  // hand-edited/older `statistics.json` with `biologicalReproduction.matches:
  // false` but no `diagnosticOnly` flag would otherwise sail through and get
  // published.
  if (statistics.diagnosticOnly || statistics.biologicalReproduction?.matches !== true) {
    throw new Error(
      'intervention-artifact: statistics.json is diagnosticOnly, or its biologicalReproduction.matches is not true -- refusing to publish an artifact built from it'
    );
  }
  if (
    !statistics.inputs ||
    typeof statistics.inputs.indexSha256 !== 'string' ||
    typeof statistics.inputs.publishedNullSha256 !== 'string'
  ) {
    throw new Error('intervention-artifact: statistics.json is missing inputs.indexSha256/inputs.publishedNullSha256');
  }

  const indexSha = sha256Hex(inputs.indexBytes);
  if (indexSha !== statistics.inputs.indexSha256) {
    throw new Error(
      `intervention-artifact: ${inputs.indexLabel} sha256 ${indexSha} does not match statistics.json's recorded indexSha256 (${statistics.inputs.indexSha256}) -- stale statistics.json?`
    );
  }
  // `trained.json`'s own recorded `graphListSha256` (`assembleInterventionRaw`
  // in `null-trained-evaluate-graph-list.ts`) must describe this same
  // `index.json` -- otherwise a `trained.json` scored against an older
  // `interventions.py` run (new P/C/M graphs reusing the same ids) would
  // publish silently, with nothing to flag that the trained arm no longer
  // describes the same graphs the authored statistics/transfer tables do
  // (a maintainability-review finding).
  if (inputs.trainedRaw.graphListSha256 !== indexSha) {
    throw new Error(
      `intervention-artifact: trained.json graphListSha256 (${inputs.trainedRaw.graphListSha256}) does not match ${inputs.indexLabel} sha256 (${indexSha}) -- trained.json was scored against a different graph list`
    );
  }
  const rewiringNullSha = sha256Hex(inputs.rewiringNullBytes);
  if (rewiringNullSha !== statistics.inputs.publishedNullSha256) {
    throw new Error(
      `intervention-artifact: the published null sha256 ${rewiringNullSha} does not match statistics.json's recorded publishedNullSha256 (${statistics.inputs.publishedNullSha256}) -- stale statistics.json?`
    );
  }
  if (rewiringNullSha !== inputs.manifestRewiringNullSha) {
    throw new Error(
      `intervention-artifact: the published null sha256 ${rewiringNullSha} does not match the manifest's rewiringNull.sha256 (${inputs.manifestRewiringNullSha}) -- stale manifest?`
    );
  }
  if (inputs.manifestBiologicalSha !== inputs.rewiringNullParsed.sourceGraphSha256) {
    throw new Error(
      `intervention-artifact: manifest.binarySha256 (${inputs.manifestBiologicalSha}) does not match the published null's sourceGraphSha256 (${inputs.rewiringNullParsed.sourceGraphSha256})`
    );
  }

  const attribution = parseAttribution(inputs.attributionText, inputs.attributionLabel);
  if (attribution.biological.sourceSha256 !== inputs.manifestBiologicalSha) {
    throw new Error(
      `intervention-artifact: ${inputs.attributionLabel}'s biological.sourceSha256 (${attribution.biological.sourceSha256}) does not match manifest.binarySha256 (${inputs.manifestBiologicalSha})`
    );
  }

  const pAfter = parseIndexGraphTransfer(inputs.indexText, inputs.indexLabel, 'P');
  const qAfter = parseIndexGraphTransfer(inputs.indexText, inputs.indexLabel, 'Q');
  if (pAfter.swaps !== attribution.P.swaps) {
    throw new Error(`intervention-artifact: P's swap count disagrees between ${inputs.indexLabel} (${pAfter.swaps}) and ${inputs.attributionLabel} (${attribution.P.swaps})`);
  }
  if (qAfter.swaps !== attribution.Q.swaps) {
    throw new Error(`intervention-artifact: Q's swap count disagrees between ${inputs.indexLabel} (${qAfter.swaps}) and ${inputs.attributionLabel} (${attribution.Q.swaps})`);
  }

  // A missing/empty `trained` section (or an absent `--rewiring-null` file
  // that somehow parsed) must fail loudly, not silently degrade into an
  // `undefined` p25/`Infinity` percentile resolution that `JSON.stringify`
  // then quietly drops from the published artifact (a maintainability-review
  // finding, reproduced: `sortedPublished[Math.floor(0.25 * 0)]` is
  // `undefined`, and `1 / 0` is `Infinity`).
  const publishedTrainedNullRewired = inputs.rewiringNullParsed.trained?.rewired;
  if (!publishedTrainedNullRewired || publishedTrainedNullRewired.length === 0) {
    throw new Error(
      'intervention-artifact: the published null has no trained.rewired scores -- cannot report the trained-null context'
    );
  }

  const indexInfo = parseGraphListIndexInfo(inputs.indexText, inputs.indexLabel);
  const trained = buildTrainedStatistics(
    inputs.trainedRaw,
    indexInfo,
    publishedTrainedNullRewired.map((r) => r.score),
    inputs.bootstrapSeed,
    inputs.bootstrapResamples
  );

  return {
    version: 1,
    sources: {
      biologicalSha: inputs.manifestBiologicalSha,
      rewiringNullSha,
      nullExplanationSha: inputs.manifestNullExplanationSha,
      indexSha,
      authoredEvaluatorGitRev: statistics.evaluatorGitRev,
      trainedEvaluatorGitRev: trained.evaluatorGitRev,
      producer: pathwayInterventionsProducer()
    },
    k: attribution.P.swaps,
    kQ: attribution.Q.swaps,
    interventions: {
      P: {
        id: 'P',
        score: statistics.p.score,
        swaps: attribution.P.swaps,
        targetReached: attribution.P.targetReached,
        stopReason: attribution.P.stopReason,
        percentileInPublishedNull: statistics.p.percentileInPublishedNull,
        pRankAmongC: statistics.p.pRankAmongC,
        pRankAmongM: statistics.p.pRankAmongM,
        category: statistics.p.category
      },
      Q: {
        id: 'Q',
        score: statistics.q.score,
        swaps: attribution.Q.swaps,
        targetReached: attribution.Q.targetReached,
        stopReason: attribution.Q.stopReason,
        percentileInPublishedNull: statistics.q.percentileInPublishedNull,
        qRankAmongMQ: statistics.q.qRankAmongMQ,
        channelSpecific: statistics.q.channelSpecific
      },
      R: attribution.R
    },
    controls: {
      k: attribution.P.swaps,
      C: statistics.controls.C,
      M: statistics.controls.M,
      MQ: { ...statistics.controls.MQ, kQ: attribution.Q.swaps }
    },
    authored: {
      category: statistics.p.category,
      pRankAmongC: statistics.p.pRankAmongC,
      pRankAmongM: statistics.p.pRankAmongM,
      qRankAmongMQ: statistics.q.qRankAmongMQ,
      channelSpecific: statistics.q.channelSpecific,
      publishedNullFloor: statistics.publishedNullFloor,
      biologicalReproduction: statistics.biologicalReproduction,
      decoder: 'authored',
      seeds: statistics.seeds,
      ticks: statistics.ticks
    },
    trained: {
      perSeed: trained.perSeed,
      trainedRobust: trained.trainedRobust,
      controls: trained.controls,
      evaluatorGitRev: trained.evaluatorGitRev,
      d: trained.d,
      note: trained.note
    },
    transferBeforeAfter: {
      populations: TRANSFER_OUTPUT_POPULATIONS,
      channels: TRANSFER_INPUT_CHANNELS,
      biological: attribution.biological.transfer.full3x8,
      P: pAfter.transfer,
      Q: qAfter.transfer
    },
    host: statistics.host
  };
};

// ---------------------------------------------------------------------------
// Report markdown
// ---------------------------------------------------------------------------

const pct = (value: number): string => `${(value * 100).toFixed(1)}%`;
const fmt = (value: number): string => value.toFixed(4);

/** One 3x8 transfer table as a markdown table, rows=output populations, columns=input channels. */
const renderTransferTable = (table: Transfer3x8): string => {
  const header = `| population | ${TRANSFER_INPUT_CHANNELS.join(' | ')} |`;
  const divider = `| --- | ${TRANSFER_INPUT_CHANNELS.map(() => '---').join(' | ')} |`;
  const rows = TRANSFER_OUTPUT_POPULATIONS.map((population, i) => `| ${population} | ${table[i].map(fmt).join(' | ')} |`);
  return [header, divider, ...rows].join('\n');
};

/**
 * `00-overview.md`'s "Predeclared outcome categories (authored decoder)"
 * bullets, quoted byte-verbatim (thermo-methodology review, Suggestion: an
 * earlier version substituted ASCII `--` for the plan's real em dash and
 * dropped its mid-sentence `**and**`/`**but**` emphasis — this is now a
 * direct copy of the plan's own markdown, not a paraphrase). Only the
 * leading `- ` list-item marker and the bullet label's own bold/colon
 * styling are supplied by the call site below (`00-overview.md` writes
 * `- **Pathway supported:** ...`; this record holds everything from
 * `**Pathway supported:**` onward, unchanged).
 */
const AUTHORED_CATEGORY_PROSE: Record<OutcomeCategory, string> = {
  'pathway-supported': "**Pathway supported:** P's score is at or above the null's 25th percentile **and** above the 95th percentile of both the C and M distributions.",
  'edge-class-effect': "**Edge-class effect:** P is at or above the null's 25th percentile and above C's 95th percentile, **but** at or below M's 95th percentile. Any input→thrust edges of this class help about equally, and the specific optimized edges do not matter.",
  'generic-rewiring-effect': "**Generic rewiring effect:** P is at or above the null's 25th percentile **but** at or below C's 95th percentile. Any `k` swaps help about equally.",
  'not-supported': "**Not supported:** P stays below the null's 25th percentile."
};

const TRAINED_CATEGORY_LABEL: Record<TrainedOutcomeCategory, string> = {
  'pathway-supported': 'pathway-supported',
  'edge-class-effect': 'edge-class-effect',
  'no-specific-effect': 'no-specific-effect (neither pathway-supported nor edge-class; the generic-vs-not-supported split is undetermined)'
};

/**
 * The Outcome section's trained-decoder clause, one entry per category —
 * every claim here must be derivable from the artifact's own data, never a
 * literal that happens to match today's run (a dual-review finding: an
 * earlier version hard-coded "P shows no advantage ... at any of the three
 * trainer seeds tested" as fixed prose, which would have printed a
 * self-contradicting report for any other trained result).
 */
const TRAINED_OUTCOME_PROSE: Record<TrainedOutcomeCategory, string> = {
  'pathway-supported': 'P outperforms both freshly-trained control arms',
  'edge-class-effect': 'P outperforms the freshly-trained unrestricted (C) arm but not the class-matched (M) arm',
  'no-specific-effect': 'P shows no advantage over either freshly-trained control arm'
};

export const renderPathwayInterventionsReportMarkdown = (artifact: Readonly<PathwayInterventionsArtifact>): string => {
  const { authored, trained, interventions, controls, transferBeforeAfter } = artifact;

  const perSeedRows = P_TRAINER_SEEDS.map((seed) => {
    const s = trained.perSeed[seed];
    return `| ${seed} | ${fmt(s.score)} | ${s.aboveC ? 'yes' : 'no'} | ${s.aboveM ? 'yes' : 'no'} | ${TRAINED_CATEGORY_LABEL[s.category]} | ${s.context.abovePublishedNullP25 ? 'above' : 'below'} |`;
  }).join('\n');

  // Every sentence below is derived from `trained.perSeed`/`trained.trainedRobust`
  // -- never a literal describing only today's run (see `TRAINED_OUTCOME_PROSE`'s
  // own doc comment).
  const representativeSeed = P_TRAINER_SEEDS[0];
  const trainedOutcomeSentence = trained.trainedRobust
    ? `${TRAINED_OUTCOME_PROSE[trained.perSeed[representativeSeed].category]} at all ${P_TRAINER_SEEDS.length} trainer seeds tested (**${TRAINED_CATEGORY_LABEL[trained.perSeed[representativeSeed].category]}**, robust: true)`
    : `the ${P_TRAINER_SEEDS.length} trainer seeds disagree on the category (${P_TRAINER_SEEDS.map((seed) => `seed ${seed}: ${trained.perSeed[seed].category}`).join(', ')}; robust: false)`;

  const contextAboveSeeds = P_TRAINER_SEEDS.filter((seed) => trained.perSeed[seed].context.abovePublishedNullP25);
  const contextBelowSeeds = P_TRAINER_SEEDS.filter((seed) => !trained.perSeed[seed].context.abovePublishedNullP25);
  const contextSeedRobust = contextAboveSeeds.length === 0 || contextBelowSeeds.length === 0;
  const contextSentence = contextSeedRobust
    ? `**seed-consistent**: every trainer seed falls ${contextAboveSeeds.length > 0 ? 'above' : 'below'} it`
    : `**not seed-robust**: below at trainer seed(s) ${contextBelowSeeds.join('/')} and above at trainer seed(s) ${contextAboveSeeds.join('/')}`;
  const contextPercentileResolution = trained.perSeed[representativeSeed].context.percentileResolution;

  return `# Clearance -> thrust pathway interventions

## Question

\`docs/null-explanation-report.md\` (on \`origin/main\`) associates biological's 0th-percentile score with the
clearance->thrust pathway:

- \`T:rightClearance->thrust\`: biological 0.0031 versus null median 0.0204, rho 0.467;
- \`T:forwardClearance->thrust\`: 0.0022 versus 0.0195, rho 0.353;
- input-restricted \`weightedInDegree:thrust\`: biological **0.0** versus null median 123.3, rho 0.394.

In the biological graph, no input-labeled neuron synapses directly onto a thrust neuron. Degree-preserving
rewiring creates such edges. This experiment edits the biological graph along this pathway with predeclared
interventions, with degree-preserving controls matched in size, and scores it under both the authored decoder
and CEM-retrained readouts.

## Method

The interventions below are quoted byte-verbatim from \`.agents/plans/pathway-interventions/00-overview.md\`'s
"Predeclared interventions and prediction" section (thermo-methodology review, Suggestion: an earlier version of
this report substituted ASCII \`->\`/\`--\` for the plan's real \`→\`/\`—\` characters and dropped its mid-sentence
bold emphasis; this section and "Predeclared outcome categories" below are now direct copies).

- **Primary — targeted degree-preserving swaps (P):** starting from the biological graph, repeatedly apply double-edge swaps \`(a→b, c→d) → (a→d, c→b)\` where \`a\` is input-labeled, \`d\` is a thrust-population neuron, and both collateral endpoints \`b\` and \`c\` are **bridge** neurons (neither input nor output assigned). That keeps the collateral edges added and removed away from the output populations, and every control arm shares the same class restriction. Choose greedily by the largest first-order increase of \`T:rightClearance→thrust + T:forwardClearance→thrust\`. Stop when both entries reach at least the null's 25th percentile, or at 200 swaps. This preserves in-degree, out-degree, the weight multiset, and the presynaptic signs (per-neuron, \`presynapticSigns\`). Record the number of swaps \`k\`.
- **Control — random degree-preserving swaps (C):** 100 graphs, each with exactly \`k\` uniformly random valid double-edge swaps anywhere in the graph (seeds \`0…99\`), under the same validity rules. This tests "any perturbation of this size".
- **Control — class-matched random swaps (M):** 100 graphs, each with exactly \`k\` valid swaps drawn **uniformly** (not greedily) from the same candidate class as P (\`a\` input-labeled, \`d\` thrust, \`b\` and \`c\` bridge), seeds \`1000…1099\`. This separates "these specific, optimized edges" from "any edges of this class", and it matches P's collateral-edge class.
- **Secondary — magnitude-matched removal (R):** remove the attributed edges into thrust neurons from the input side that carry at least 50% of the first-order transfer. Because the biological graph has none, R is expected to be empty. If so, it is reported as not applicable. It is kept only to check the premise.
- **Secondary — clearance-only targeting (Q):** the same as P, but only \`rightClearance\` and \`forwardClearance\` input neurons may be the source \`a\`. This tests channel specificity rather than generic input→thrust wiring.

### Predeclared outcome categories (authored decoder)

- ${AUTHORED_CATEGORY_PROSE['pathway-supported']}
- ${AUTHORED_CATEGORY_PROSE['edge-class-effect']}
- ${AUTHORED_CATEGORY_PROSE['generic-rewiring-effect']}
- ${AUTHORED_CATEGORY_PROSE['not-supported']}
- **Channel-specific (modifier, authored decoder only):** Q (clearance-channel sources only) is above the null's 25th percentile and above the 95th percentile of **its own** size-matched class control MQ: 100 graphs, each with exactly \`k_Q = |Q swaps|\` uniform valid swaps from Q's candidate class, seeds \`2000…2099\`. Q is never compared against the P-sized C or M arms. Q is not evaluated with trained readouts, and the report states this.

Claim language: a positive result means "the net effect of this accepted swap set", not an isolated single-edge causal effect. The report states this.

### Trained decoder

Trained decoder: the same categories. The **governing reference** is the freshly trained controls: 5 C graphs (C000–C004) and 5 M graphs (M1000–M1004), each at trainer seed 101. The published trained null (\`rewiring-null-v1.json\` \`trained\`, 20 full rewirings) is reported for context only and does not decide the category. With 5 graphs per arm, the trained cutoff is "above the maximum of that arm's 5". The report states that this resolution is coarse. The result is robust only if all three P trainer seeds (101/202/303) agree on the category, because the trained null was trainer-seed-sensitive.

The C/M comparisons above are the same ones the authored side uses; the authored floor's own null-percentile
prong is not applied on the trained side (the trained null is context-only, not a decisive threshold — see the
disclosure immediately below, which is this report's own addition, not part of the quoted plan text above).

${trained.note}

## Graph construction

- P: \`k = ${artifact.k}\` accepted swaps, target reached: ${interventions.P.targetReached} (${interventions.P.stopReason}).
- Q: \`k_Q = ${artifact.kQ}\` accepted swaps, target reached: ${interventions.Q.targetReached} (${interventions.Q.stopReason}).
- R: applicable = ${interventions.R.applicable} (${interventions.R.reason}).

### Transfer matrix T, before (biological)

${renderTransferTable(transferBeforeAfter.biological)}

### Transfer matrix T, after P

${renderTransferTable(transferBeforeAfter.P)}

### Transfer matrix T, after Q

${renderTransferTable(transferBeforeAfter.Q)}

Every entry of T may move, including yaw and brake channels unrelated to the targeted pathway -- the tables
above show the full 3x8 matrix precisely so that is checkable; the category below uses only each graph's
\`movementScore\`, never a T entry directly.

## Authored results

Empirical p is the \`(k+1)/(n+1)\` rank statistic against a 100-graph control arm (\`03-evaluation.md\`'s own rank
statistic): the smallest value it can take is 1/101 ≈ 0.0099, which means the intervention exceeded every one of
the 100 control graphs -- not "near the bottom" of the arm.

| graph | score | percentile in published null | empirical p vs C \`(k+1)/(n+1)\` | empirical p vs M | empirical p vs MQ |
| --- | --- | --- | --- | --- | --- |
| P | ${fmt(interventions.P.score)} | ${pct(interventions.P.percentileInPublishedNull)} | ${fmt(authored.pRankAmongC)} | ${fmt(authored.pRankAmongM)} | -- |
| Q | ${fmt(interventions.Q.score)} | ${pct(interventions.Q.percentileInPublishedNull)} | -- | -- | ${fmt(authored.qRankAmongMQ)} |

- P's category: ${AUTHORED_CATEGORY_PROSE[authored.category]}
- Q's channel-specific modifier: **${authored.channelSpecific ? 'holds' : 'does not hold'}** (Q above the null's
  25th percentile and above its own MQ control's 95th percentile).
- Control arms (mean \`movementScore\`, n=${controls.C.n} each): C p50=${fmt(controls.C.p50)} p95=${fmt(controls.C.p95)};
  M p50=${fmt(controls.M.p50)} p95=${fmt(controls.M.p95)}; MQ (n=${controls.MQ.n}, k_Q=${controls.MQ.kQ}) p50=${fmt(controls.MQ.p50)} p95=${fmt(controls.MQ.p95)}.
- Biological reproduction check: computed ${fmt(authored.biologicalReproduction.computedScore)} vs published
  ${fmt(authored.biologicalReproduction.publishedScore)} (matches: ${authored.biologicalReproduction.matches}).
- Multiple comparisons disclosed: P vs C, P vs M, Q vs MQ, and (below) ${P_TRAINER_SEEDS.length} trainer seeds --
  none of these comparisons is corrected against the others; each is reported and read on its own predeclared
  terms.

## Trained results

| P trainer seed | score | above C max | above M max | category | vs published trained-null p25 (context only) |
| --- | --- | --- | --- | --- | --- |
${perSeedRows}

- Control arms (mean \`movementScore\`, trainer seed 101, n=${trained.controls.C.n} each): C max=${fmt(trained.controls.C.max)}; M max=${fmt(trained.controls.M.max)}.
- \`trainedRobust\`: **${trained.trainedRobust}** (every P trainer seed above ${trained.trainedRobust ? 'agrees' : 'does not agree'} on the category).
- The published trained null's 25th percentile is context only (percentile resolution ${pct(contextPercentileResolution)}) and
  does not decide the category -- see the disclosure above. Informally, the finer generic-vs-not-supported split
  this context value would suggest is ${contextSentence}.
- Q is not evaluated with trained readouts (authored-decoder only, per the predeclared method above).

## Outcome

Under the **authored decoder**, this study's mechanical outcome is **${authored.category}**, with the
channel-specific modifier **${authored.channelSpecific ? 'holding' : 'not holding'}**. This authored-decoder
verdict is bound to the hand-written decoder; it does not by itself say what a trained readout finds. Under
**trained readouts**, ${trainedOutcomeSentence}.

## Limitations

- This experiment covers this model only: the authored encoder, rate dynamics, arena, and decoders. It makes no
  biological claim, and a real fly's sensory neurons not synapsing directly onto descending neurons is expected
  anatomy, not a defect.
- The authored decoder is hand-written, not trained and not biology.
- Every claim here is descriptive and bound to this model only; no causal claim is made about the real fly.
- A positive result means the net effect of this accepted swap set, not a single-edge causal effect.
- The method is a greedy targeted search against 100 unrestricted (C) and 100 class-matched (M) random controls;
  it does not prove the targeted edges are individually necessary or sufficient.
- The trained arm compares against only 5 controls per arm (a coarse resolution), and its result depends on the
  trainer seed -- reported as robust only when all three P trainer seeds agree on the category.
- The channel-specific test (Q against MQ) is authored-decoder only; Q was not evaluated with trained readouts.
- The predeclared C/M-max comparison decides pathway-supported/edge-class-effect for the trained decoder, but
  00-overview.md does not say how to report the remaining generic-vs-not-supported split when P does not clear
  the C arm (that needs a trained-null percentile floor, and this study's trained null is context-only).
  'no-specific-effect' is a reporting convention this study's coordinator adopted after the trained scores were
  known (see the disclosure above), not itself a predeclared category; this report does not force an unlicensed
  generic/not-supported label.
- P at trainer seeds 202/303 is compared against C/M control arms trained only at seed 101, so those two
  comparisons mix a graph difference with a trainer-seed difference; only the seed-101 comparison is seed-matched.
- Multiple comparisons (P vs C, P vs M, Q vs MQ, and ${P_TRAINER_SEEDS.length} trainer seeds) are disclosed above
  and are not corrected against each other.
`;
};

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------

/** Add/overwrite the manifest's `pathwayInterventions` key in place -- mirrors `null-report.ts`'s `updateManifestWithRewiringNull` (same re-serialization scheme, same "caller already ran `verifyManifestRoundTrips`" contract). */
export const updateManifestWithPathwayInterventions = (
  manifestPath: string,
  entry: { readonly artifact: string; readonly sha256: string }
): void => {
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8')) as Record<string, unknown>;
  manifest.pathwayInterventions = entry;
  atomicWriteFileSync(manifestPath, `${JSON.stringify(sortKeysDeep(manifest), null, 2)}\n`);
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

export interface IntoArtifactArgs {
  readonly statistics: string;
  readonly attribution: string;
  readonly trained: string;
  readonly index: string;
  readonly rewiringNull: string;
  readonly manifest: string;
  readonly out: string;
  readonly reportMd: string;
  readonly bootstrapSeed: number;
  readonly bootstrapResamples: number;
}

export const parseIntoArtifactArgs = (argv: readonly string[]): IntoArtifactArgs => {
  let statistics = DEFAULT_STATISTICS;
  let attribution = DEFAULT_ATTRIBUTION;
  let trained = DEFAULT_TRAINED;
  let index = DEFAULT_INDEX;
  let rewiringNull = DEFAULT_REWIRING_NULL;
  let manifest = DEFAULT_MANIFEST;
  let out = DEFAULT_OUT;
  let reportMd = DEFAULT_REPORT_MD;
  let bootstrapSeed = DEFAULT_BOOTSTRAP_SEED;
  let bootstrapResamples = DEFAULT_BOOTSTRAP_RESAMPLES;

  let i = 0;
  while (i < argv.length) {
    const flag = argv[i];
    if (flag === '--statistics') {
      statistics = resolve(process.cwd(), requireValue(flag, argv[i + 1]));
      i += 2;
    } else if (flag === '--attribution') {
      attribution = resolve(process.cwd(), requireValue(flag, argv[i + 1]));
      i += 2;
    } else if (flag === '--trained') {
      trained = resolve(process.cwd(), requireValue(flag, argv[i + 1]));
      i += 2;
    } else if (flag === '--index') {
      index = resolve(process.cwd(), requireValue(flag, argv[i + 1]));
      i += 2;
    } else if (flag === '--rewiring-null') {
      rewiringNull = resolve(process.cwd(), requireValue(flag, argv[i + 1]));
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
    } else if (flag === '--bootstrap-seed') {
      bootstrapSeed = requireNonNegativeInt(flag, argv[i + 1]);
      i += 2;
    } else if (flag === '--bootstrap-resamples') {
      bootstrapResamples = requirePositiveInt(flag, argv[i + 1]);
      i += 2;
    } else {
      throw new Error(`Unknown argument: ${flag}`);
    }
  }

  // `--report-md` is a write target too, not merely an input -- it must not
  // collide with `--out` (which would overwrite the just-published artifact
  // after its sha was already written to the manifest, leaving a manifest
  // sha that never matches what is on disk) or with any input file (a
  // maintainability-review finding: only `--out` was checked before).
  for (const target of [out, reportMd]) {
    for (const input of [statistics, attribution, trained, index, rewiringNull, manifest]) {
      if (resolve(target) === resolve(input)) {
        throw new Error(`intervention-artifact: ${target === out ? '--out' : '--report-md'} must not overwrite an input file (${input})`);
      }
    }
  }
  if (resolve(out) === resolve(reportMd)) {
    throw new Error('intervention-artifact: --out and --report-md must not be the same path');
  }

  return { statistics, attribution, trained, index, rewiringNull, manifest, out, reportMd, bootstrapSeed, bootstrapResamples };
};

export interface RunPathwayInterventionsArtifactResult {
  readonly out: string;
  readonly reportMdPath: string;
  readonly artifactSha256: string;
  readonly artifact: PathwayInterventionsArtifact;
}

export const runPathwayInterventionsArtifact = (args: Readonly<IntoArtifactArgs>): RunPathwayInterventionsArtifactResult => {
  const statisticsBytes = readFileSync(args.statistics);
  const statistics = JSON.parse(statisticsBytes.toString('utf8')) as InterventionStatistics;
  const attributionBytes = readFileSync(args.attribution);
  const trainedBytes = readFileSync(args.trained);
  const trainedRaw = JSON.parse(trainedBytes.toString('utf8')) as NullTrainedInterventionEvaluationRaw;
  const indexBytes = readFileSync(args.index);
  const rewiringNullBytes = readFileSync(args.rewiringNull);
  const rewiringNullParsed = JSON.parse(rewiringNullBytes.toString('utf8')) as {
    readonly sourceGraphSha256: string;
    readonly trained?: { readonly rewired: readonly { readonly score: number }[] };
  };
  const manifest = JSON.parse(readFileSync(args.manifest, 'utf8')) as {
    readonly binarySha256?: string;
    readonly rewiringNull?: { readonly sha256?: string };
    readonly nullExplanation?: { readonly sha256?: string };
  };
  if (!manifest.binarySha256) throw new Error(`intervention-artifact: ${args.manifest} is missing binarySha256`);
  if (!manifest.rewiringNull?.sha256) throw new Error(`intervention-artifact: ${args.manifest} is missing rewiringNull.sha256`);
  if (!manifest.nullExplanation?.sha256) {
    throw new Error(`intervention-artifact: ${args.manifest} is missing nullExplanation.sha256 (publish the null-explanation artifact first)`);
  }

  const artifact = buildPathwayInterventionsArtifact({
    statistics,
    attributionText: attributionBytes.toString('utf8'),
    attributionLabel: args.attribution,
    trainedRaw,
    indexText: indexBytes.toString('utf8'),
    indexLabel: args.index,
    indexBytes,
    rewiringNullBytes,
    rewiringNullParsed,
    manifestBiologicalSha: manifest.binarySha256,
    manifestRewiringNullSha: manifest.rewiringNull.sha256,
    manifestNullExplanationSha: manifest.nullExplanation.sha256,
    bootstrapSeed: args.bootstrapSeed,
    bootstrapResamples: args.bootstrapResamples
  });

  verifyManifestRoundTrips(args.manifest);

  const artifactContents = JSON.stringify(artifact);
  const artifactSha256 = sha256Hex(artifactContents);
  const reportMdContents = renderPathwayInterventionsReportMarkdown(artifact);

  mkdirSync(dirname(args.out), { recursive: true });
  atomicWriteFileSync(args.out, artifactContents);

  updateManifestWithPathwayInterventions(args.manifest, { artifact: basename(args.out), sha256: artifactSha256 });

  mkdirSync(dirname(args.reportMd), { recursive: true });
  atomicWriteFileSync(args.reportMd, reportMdContents);

  return { out: args.out, reportMdPath: args.reportMd, artifactSha256, artifact };
};

const main = (): void => {
  try {
    const args = parseIntoArtifactArgs(process.argv.slice(2));
    const result = runPathwayInterventionsArtifact(args);
    // eslint-disable-next-line no-console -- CLI tool: this is its user-facing output.
    console.log(
      `intervention-artifact: wrote ${result.out} (sha256 ${result.artifactSha256}) and ${result.reportMdPath}\n` +
        `authored.category=${result.artifact.authored.category} channelSpecific=${result.artifact.authored.channelSpecific} ` +
        `trained.trainedRobust=${result.artifact.trained.trainedRobust}`
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    // eslint-disable-next-line no-console -- CLI tool: this is its user-facing error output.
    console.error(`intervention-artifact failed: ${message}`);
    process.exit(1);
  }
};

if (process.argv[1] === fileURLToPath(import.meta.url)) main();
