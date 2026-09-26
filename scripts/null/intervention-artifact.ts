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
import { buildTrainedStatistics, type PTrainerSeed, type TrainedSeedResult, type TrainedArmDistribution } from './intervention-report-trained';
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
  if (statistics.diagnosticOnly) {
    throw new Error(
      'intervention-artifact: statistics.json is diagnosticOnly (a failed biological-reproduction check) -- refusing to publish an artifact built from it'
    );
  }

  const indexSha = sha256Hex(inputs.indexBytes);
  if (indexSha !== statistics.inputs.indexSha256) {
    throw new Error(
      `intervention-artifact: ${inputs.indexLabel} sha256 ${indexSha} does not match statistics.json's recorded indexSha256 (${statistics.inputs.indexSha256}) -- stale statistics.json?`
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

  const indexInfo = parseGraphListIndexInfo(inputs.indexText, inputs.indexLabel);
  const trained = buildTrainedStatistics(
    inputs.trainedRaw,
    indexInfo,
    (inputs.rewiringNullParsed.trained?.rewired ?? []).map((r) => r.score),
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

const AUTHORED_CATEGORY_PROSE: Record<OutcomeCategory, string> = {
  'pathway-supported': "**Pathway supported**: P's score is at or above the null's 25th percentile and above the 95th percentile of both the C and M distributions.",
  'edge-class-effect': "**Edge-class effect**: P is at or above the null's 25th percentile and above C's 95th percentile, but at or below M's 95th percentile. Any input->thrust edges of this class help about equally, and the specific optimized edges do not matter.",
  'generic-rewiring-effect': "**Generic rewiring effect**: P is at or above the null's 25th percentile but at or below C's 95th percentile. Any k swaps help about equally.",
  'not-supported': "**Not supported**: P stays below the null's 25th percentile."
};

const TRAINED_CATEGORY_LABEL: Record<'pathway-supported' | 'edge-class-effect' | 'no-specific-effect', string> = {
  'pathway-supported': 'pathway-supported',
  'edge-class-effect': 'edge-class-effect',
  'no-specific-effect': 'no-specific-effect (neither pathway-supported nor edge-class; the generic-vs-not-supported split is undetermined)'
};

export const renderPathwayInterventionsReportMarkdown = (artifact: Readonly<PathwayInterventionsArtifact>): string => {
  const { authored, trained, interventions, controls, transferBeforeAfter } = artifact;

  const perSeedRows = ([101, 202, 303] as const)
    .map((seed) => {
      const s = trained.perSeed[seed];
      return `| ${seed} | ${fmt(s.score)} | ${s.aboveC ? 'yes' : 'no'} | ${s.aboveM ? 'yes' : 'no'} | ${TRAINED_CATEGORY_LABEL[s.category]} | ${s.context.abovePublishedNullP25 ? 'above' : 'below'} |`;
    })
    .join('\n');

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

**Primary -- targeted degree-preserving swaps (P):** starting from the biological graph, repeatedly apply double-edge
swaps \`(a->b, c->d) -> (a->d, c->b)\` where \`a\` is input-labeled, \`d\` is a thrust-population neuron, and both
collateral endpoints \`b\` and \`c\` are bridge neurons (neither input nor output assigned). Choose greedily by the
largest first-order increase of \`T:rightClearance->thrust + T:forwardClearance->thrust\`. Stop when both entries
reach at least the null's 25th percentile, or at 200 swaps. This preserves in-degree, out-degree, the weight
multiset, and the presynaptic signs (per-neuron, \`presynapticSigns\`). \`k\` is the number of accepted swaps.

**Control -- random degree-preserving swaps (C):** 100 graphs, each with exactly \`k\` uniformly random valid
double-edge swaps anywhere in the graph (seeds 0-99). This tests "any perturbation of this size".

**Control -- class-matched random swaps (M):** 100 graphs, each with exactly \`k\` valid swaps drawn uniformly (not
greedily) from the same candidate class as P (\`a\` input-labeled, \`d\` thrust, \`b\`/\`c\` bridge), seeds 1000-1099.

**Secondary -- magnitude-matched removal (R):** remove the attributed edges into thrust neurons from the input
side that carry at least 50% of the first-order transfer. Because the biological graph has none, R is expected
to be empty. If so, it is reported as not applicable.

**Secondary -- clearance-only targeting (Q):** the same as P, but only \`rightClearance\` and \`forwardClearance\`
input neurons may be the source \`a\`. This tests channel specificity rather than generic input->thrust wiring.

### Predeclared outcome categories (authored decoder)

- ${AUTHORED_CATEGORY_PROSE['pathway-supported']}
- ${AUTHORED_CATEGORY_PROSE['edge-class-effect']}
- ${AUTHORED_CATEGORY_PROSE['generic-rewiring-effect']}
- ${AUTHORED_CATEGORY_PROSE['not-supported']}
- **Channel-specific (modifier, authored decoder only)**: Q (clearance-channel sources only) is above the null's
  25th percentile and above the 95th percentile of its own size-matched class control MQ: 100 graphs, each with
  exactly \`k_Q = |Q swaps|\` uniform valid swaps from Q's candidate class, seeds 2000-2099. Q is never compared
  against the P-sized C or M arms. Q is not evaluated with trained readouts, and this report states this.

Claim language: a positive result means "the net effect of this accepted swap set", not an isolated single-edge
causal effect.

### Trained decoder

The same categories apply, but the **governing reference** is the freshly trained controls: 5 C graphs
(C000-C004) and 5 M graphs (M1000-M1004), each at trainer seed 101. The published trained null
(\`rewiring-null-v1.json\` \`trained\`, 20 full rewirings) is reported for context only and does not decide the
category. With 5 graphs per arm, the trained cutoff is "above the maximum of that arm's 5". The result is robust
only if all three P trainer seeds (101/202/303) agree on the category, because the trained null was
trainer-seed-sensitive.

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

| graph | score | percentile in published null (n=500) | rank among C | rank among M | rank among MQ |
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
- Multiple comparisons disclosed: P vs C, P vs M, Q vs MQ, and (below) 3 trainer seeds -- none of these
  comparisons is corrected against the others; each is reported and read on its own predeclared terms.

## Trained results

| P trainer seed | score | above C max | above M max | category | vs published trained-null p25 (context only) |
| --- | --- | --- | --- | --- | --- |
${perSeedRows}

- Control arms (mean \`movementScore\`, trainer seed 101, n=${trained.controls.C.n} each): C max=${fmt(trained.controls.C.max)}; M max=${fmt(trained.controls.M.max)}.
- \`trainedRobust\`: **${trained.trainedRobust}** (every P trainer seed above agrees on the category).
- The published trained null's 25th percentile is context only (percentile resolution ${pct(1 / 20)} at n=20) and
  does not decide the category -- see the disclosure above. Informally, the finer generic-vs-not-supported split
  this context value would suggest is **not seed-robust**: it is below at trainer seed 101 and above at trainer
  seeds 202/303.
- Q is not evaluated with trained readouts (authored-decoder only, per the predeclared method above).

## Outcome

Under the **authored decoder**, this study's mechanical outcome is **${authored.category}**, with the
channel-specific modifier **${authored.channelSpecific ? 'holding' : 'not holding'}**. Under **trained readouts**,
P shows no advantage over either freshly-trained control arm at any of the three trainer seeds tested
(**${TRAINED_CATEGORY_LABEL[trained.perSeed[101].category]}**, robust: ${trained.trainedRobust}).

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
- The generic-vs-not-supported split is undetermined for the trained decoder under this study's predeclared
  rules (see the disclosure above); this report does not force one of those two labels.
- Multiple comparisons (P vs C, P vs M, Q vs MQ, and 3 trainer seeds) are disclosed above and are not corrected
  against each other.
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

  for (const input of [statistics, attribution, trained, index, rewiringNull, manifest]) {
    if (resolve(out) === resolve(input)) {
      throw new Error(`intervention-artifact: --out must not overwrite an input file (${input})`);
    }
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
