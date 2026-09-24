import { ARM_NAMES, type ArmName } from './arms';
import type { ArmProvenance } from './export-arms';
import type { ConditionStats, PairedStats } from './stats';

/**
 * `docs/trained-readout-report.md` generation, from the exact in-memory
 * report data `scripts/training/evaluate.ts`'s `runEvaluate` builds (not a
 * re-parse of the written `report.json` — the same object, so there is no
 * JSON round-trip to introduce drift). Per
 * `.agents/plans/trained-readout/04-authoritative-evaluation-and-artifacts.md`'s
 * Artifacts table: "method, parameter accounting table, results table,
 * limitations, 'what this does not show'". No significance/superiority
 * language beyond the reported CIs, anywhere in this file — the numbers
 * speak for themselves; this module must not editorialize about direction.
 *
 * Deterministic: `renderReportMarkdown` is a pure function of its `report`
 * argument (no wall-clock, no git rev, no object-iteration-order
 * dependence beyond `report`'s own already-sorted arrays/maps) — the same
 * `report` always renders to byte-identical markdown, matching
 * `report.json`'s own byte-identical-across-runs requirement.
 */

export interface ArmReplicaReport {
  readonly H: number;
  readonly parameterCount: number;
  readonly weightsSha256: string;
  readonly env: unknown;
  readonly trained: ConditionStats;
  readonly silenced: ConditionStats;
  readonly pairedTrainedVsAuthored: PairedStats;
  readonly pairedTrainedVsSilenced: PairedStats;
}

export interface ArmReport {
  readonly D: number;
  readonly provenance: ArmProvenance | undefined;
  readonly armBundleSha256: string | undefined;
  readonly authored: ConditionStats;
  /** Keyed by `String(trainerSeed)`, e.g. `"101"`. */
  readonly replicas: Readonly<Record<string, ArmReplicaReport>>;
  /**
   * A topological (not statistical) guarantee, computed by `evaluate.ts`'s
   * `graphGuaranteesZeroReadoutInput` from the arm's own graph arrays, that
   * this arm's readout input is exactly zero on every tick of every
   * episode. See that function's doc comment for the exact condition
   * (`edgeCount === 0` and no output-assigned neuron is input-channel-
   * mapped). Used by `structurallyZeroReadoutInputArms` below.
   */
  readonly structurallyZeroInput: boolean;
}

export interface ArmPairReport {
  readonly condition: 'authored' | 'trained' | 'silenced';
  readonly trainerSeed: number | null;
  readonly armA: ArmName;
  readonly armB: ArmName;
  readonly pairedDifference: PairedStats;
}

export interface SideBySideReport {
  readonly label: 'authored-side-by-side' | 'trained-side-by-side';
  readonly leftArm: ArmName;
  readonly rightArm: ArmName;
  readonly replica: number | null;
  readonly left: ConditionStats;
  readonly right: ConditionStats;
  readonly pairedLeftMinusRight: PairedStats;
}

export interface EvaluationReport {
  readonly formatVersion: 1;
  readonly graph: {
    readonly source: 'artifact' | 'trace-graph-fixture';
    readonly path: string | null;
    readonly sha256: string;
  };
  readonly evaluation: {
    readonly ticks: number;
    readonly substeps: number;
    readonly heldOutSeeds: { readonly start: number; readonly count: number };
    readonly bootstrap: { readonly resamples: number; readonly seed: number };
    readonly opponentParked: boolean;
  };
  /** Keyed by `ArmName`; only arms actually evaluated are present. */
  readonly arms: Readonly<Record<string, ArmReport>>;
  readonly armPairs: readonly ArmPairReport[];
  readonly sideBySide: readonly SideBySideReport[];
  readonly warnings: readonly string[];
}

/**
 * The required "what this does not show" disclosure
 * (`04-authoritative-evaluation-and-artifacts.md`, `00-overview.md`'s
 * key-decisions table). Every word of this sentence is load-bearing for
 * `tests/unit/report.test.ts`'s regression check — do not reword it without
 * updating that test to match, and re-confirm the new wording still
 * satisfies the plan's "must state verbatim that the headline per-arm
 * numbers were measured single-agent with the opponent parked, which
 * differs from the shipped two-agent side-by-side default, and point to the
 * side-by-side table" requirement.
 */
export const OPPONENT_PARKED_DISCLOSURE =
  'The headline per-arm numbers above were measured single-agent with the opponent parked, ' +
  'which differs from the shipped two-agent side-by-side default; see the Side-by-side section ' +
  'of this report for the shipped two-agent condition.';

/**
 * Stable arm ordering for every table: fixed `ArmName` order (`ARM_NAMES`,
 * `./arms` — the same list `export-arms.ts`, `run-dir.ts`, and `evaluate.ts`
 * share), not object-key insertion order.
 */
const armsInOrder = (arms: Readonly<Record<string, ArmReport>>): Array<[ArmName, ArmReport]> =>
  ARM_NAMES.filter((arm) => arms[arm] !== undefined).map((arm) => [arm, arms[arm]!]);

/** Ascending-trainerSeed replica entries for one arm's `replicas` map. */
const replicasInOrder = (arm: ArmReport): Array<[number, ArmReplicaReport]> =>
  Object.entries(arm.replicas)
    .map(([seed, replica]) => [Number(seed), replica] as [number, ArmReplicaReport])
    .sort(([a], [b]) => a - b);

/** Fixed-precision, locale-independent number formatting so table cells are deterministic and diffable. */
const fmt = (value: number): string => value.toFixed(4);
const fmtCI = (ci95: readonly [number, number]): string => `[${fmt(ci95[0])}, ${fmt(ci95[1])}]`;

const mdTable = (headers: readonly string[], rows: ReadonlyArray<readonly string[]>): string => {
  const headerRow = `| ${headers.join(' | ')} |`;
  const separatorRow = `| ${headers.map(() => '---').join(' | ')} |`;
  const bodyRows = rows.map((row) => `| ${row.join(' | ')} |`);
  return [headerRow, separatorRow, ...bodyRows].join('\n');
};

/** `10000` -> `10,000`, without depending on `toLocaleString`'s ICU-dependent behavior (deterministic, plain ASCII). */
const thousands = (n: number): string => n.toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, ',');

const renderMethod = (report: EvaluationReport): string => {
  const { evaluation, graph } = report;
  const evaluatedArms = armsInOrder(report.arms).map(([arm]) => arm);
  const lines = [
    '## Method',
    '',
    `Each evaluated arm (${evaluatedArms.join(', ')}) is scored on ${evaluation.heldOutSeeds.count} held-out ` +
      `seeds (seeds ${evaluation.heldOutSeeds.start}–${
        evaluation.heldOutSeeds.start + evaluation.heldOutSeeds.count - 1
      }), running ${evaluation.ticks} ticks at ${evaluation.substeps} neural substeps per tick, with the opponent ` +
      'parked (zero action) unless stated otherwise in the Side-by-side section.',
    '',
    'Three decoder conditions are reported per arm/replica:',
    '',
    '- **trained**: the readout loaded from that replica’s `theta_final`.',
    '- **authored**: the current `aggregateOutputs → decodeAction` path (no trained readout).',
    '- **silenced**: the trained readout with its input vector forced to zero every tick — a ' +
      'circuit-silenced control.',
    '',
    'For every condition: mean, median, population standard deviation, and a 95% bootstrap confidence ' +
      `interval of the mean (${thousands(evaluation.bootstrap.resamples)} seeded resamples, ` +
      `bootstrap seed ${evaluation.bootstrap.seed}). For every arm pair and every trained-vs-authored/` +
      'trained-vs-silenced comparison: a paired difference on the same seeds, with its own 95% bootstrap CI. ' +
      'No significance or superiority language is used beyond these confidence intervals.',
    '',
    `Graph source: \`${graph.source}\`${graph.path ? ` (\`${graph.path}\`)` : ''}, sha256 \`${graph.sha256}\`.`
  ];
  return lines.join('\n');
};

/**
 * The distinct values of `pick(replica)` across an arm's replicas, joined
 * for display. A WP3 run is expected to use one training config per arm
 * except `arm`/`replica-seed` (`03-cem-training.md`), so this is normally a
 * single value — but nothing in `runEvaluate` enforces H/parameterCount
 * equality across an arm's own replicas (only across the shipped replica
 * 101's three arms), so a real mismatch is surfaced here rather than
 * silently hidden by only reading the first replica.
 */
const distinctReplicaValues = (
  replicas: ReadonlyArray<[number, ArmReplicaReport]>,
  pick: (replica: ArmReplicaReport) => number
): string => {
  if (replicas.length === 0) return 'n/a (no trained replica)';
  const distinct = [...new Set(replicas.map(([, replica]) => pick(replica)))];
  return distinct.length === 1 ? String(distinct[0]) : `${distinct.join(', ')} (differs across this arm's replicas)`;
};

const renderParameterAccounting = (report: EvaluationReport): string => {
  const rows = armsInOrder(report.arms).map(([arm, armReport]) => {
    const replicas = replicasInOrder(armReport);
    return [
      arm,
      String(armReport.D),
      distinctReplicaValues(replicas, (replica) => replica.H),
      distinctReplicaValues(replicas, (replica) => replica.parameterCount)
    ];
  });
  return [
    '## Parameter accounting',
    '',
    'D (input size, output-neuron count) is gated equal across arms at export time ' +
      '(`export-arms.ts`’s node-set gate). H (hidden size) and parameter count are taken from each ' +
      'arm’s replicas, which share one training config except `arm`/`replica-seed` and are therefore ' +
      'expected equal across arms as well.',
    '',
    mdTable(['Arm', 'D', 'H', 'Parameter count'], rows)
  ].join('\n');
};

const renderResults = (report: EvaluationReport): string => {
  const sections = armsInOrder(report.arms).map(([arm, armReport]) => {
    const rows: string[][] = [
      [
        'authored',
        '—',
        String(armReport.authored.n),
        fmt(armReport.authored.mean),
        fmt(armReport.authored.median),
        fmt(armReport.authored.std),
        fmtCI(armReport.authored.ci95)
      ]
    ];
    for (const [trainerSeed, replica] of replicasInOrder(armReport)) {
      rows.push([
        'trained',
        String(trainerSeed),
        String(replica.trained.n),
        fmt(replica.trained.mean),
        fmt(replica.trained.median),
        fmt(replica.trained.std),
        fmtCI(replica.trained.ci95)
      ]);
      rows.push([
        'silenced',
        String(trainerSeed),
        String(replica.silenced.n),
        fmt(replica.silenced.mean),
        fmt(replica.silenced.median),
        fmt(replica.silenced.std),
        fmtCI(replica.silenced.ci95)
      ]);
    }
    return [
      `### ${arm}`,
      '',
      mdTable(['Condition', 'Replica', 'n', 'Mean', 'Median', 'Std', '95% CI'], rows)
    ].join('\n');
  });
  return [
    '## Results',
    '',
    'Per arm, per replica: trained, authored, and silenced condition statistics.',
    '',
    sections.join('\n\n')
  ].join('\n');
};

const renderPairedDifferences = (report: EvaluationReport): string => {
  const withinArmRows: string[][] = [];
  for (const [arm, armReport] of armsInOrder(report.arms)) {
    for (const [trainerSeed, replica] of replicasInOrder(armReport)) {
      withinArmRows.push([
        arm,
        String(trainerSeed),
        'trained − authored',
        String(replica.pairedTrainedVsAuthored.n),
        fmt(replica.pairedTrainedVsAuthored.meanDifference),
        fmtCI(replica.pairedTrainedVsAuthored.ci95)
      ]);
      withinArmRows.push([
        arm,
        String(trainerSeed),
        'trained − silenced',
        String(replica.pairedTrainedVsSilenced.n),
        fmt(replica.pairedTrainedVsSilenced.meanDifference),
        fmtCI(replica.pairedTrainedVsSilenced.ci95)
      ]);
    }
  }

  const acrossArmRows = report.armPairs.map((pair) => [
    pair.condition,
    pair.trainerSeed === null ? '—' : String(pair.trainerSeed),
    pair.armA,
    pair.armB,
    String(pair.pairedDifference.n),
    fmt(pair.pairedDifference.meanDifference),
    fmtCI(pair.pairedDifference.ci95)
  ]);

  return [
    '## Paired differences',
    '',
    'Within-arm: trained vs. authored, and trained vs. silenced, on the same held-out seeds.',
    '',
    mdTable(['Arm', 'Replica', 'Comparison', 'n', 'Mean difference', '95% CI'], withinArmRows),
    '',
    'Across arms: paired difference on the same held-out seeds, per reported condition.',
    '',
    mdTable(['Condition', 'Replica', 'Arm A', 'Arm B', 'n', 'Mean difference', '95% CI'], acrossArmRows)
  ].join('\n');
};

const renderSideBySide = (report: EvaluationReport): string => {
  if (report.sideBySide.length === 0) {
    return [
      '## Side-by-side',
      '',
      'No side-by-side condition was evaluated: this run did not include both a `biological` and a ' +
        '`rewired` arm (see Warnings).'
    ].join('\n');
  }
  const byLabel = (label: SideBySideReport['label']): SideBySideReport[] =>
    report.sideBySide.filter((entry) => entry.label === label);

  const renderLabel = (label: SideBySideReport['label'], title: string): string => {
    const entries = byLabel(label);
    if (entries.length === 0) return `### ${title}\n\nNot evaluated this run.`;
    const rows = entries.map((entry) => [
      entry.replica === null ? '—' : String(entry.replica),
      `${entry.leftArm} (left)`,
      String(entry.left.n),
      fmt(entry.left.mean),
      fmtCI(entry.left.ci95),
      `${entry.rightArm} (right)`,
      String(entry.right.n),
      fmt(entry.right.mean),
      fmtCI(entry.right.ci95),
      fmt(entry.pairedLeftMinusRight.meanDifference),
      fmtCI(entry.pairedLeftMinusRight.ci95)
    ]);
    return [
      `### ${title}`,
      '',
      mdTable(
        [
          'Replica',
          'Left arm',
          'Left n',
          'Left mean',
          'Left 95% CI',
          'Right arm',
          'Right n',
          'Right mean',
          'Right 95% CI',
          'Left − right mean diff',
          'Left − right 95% CI'
        ],
        rows
      )
    ].join('\n');
  };

  return [
    '## Side-by-side',
    '',
    'The shipped default view: both agents driven, same seeds, biological vs. rewired.',
    '',
    renderLabel('trained-side-by-side', 'trained-side-by-side'),
    '',
    renderLabel('authored-side-by-side', 'authored-side-by-side')
  ].join('\n');
};

/**
 * The plan's default held-out range (`04-authoritative-evaluation-and-artifacts.md`:
 * `30001…30100`), disjoint by construction from its training (`1..10000`)
 * and validation (`20001..20064`) ranges. `--held-out-start`/`--held-out-count`
 * are free CLI inputs this evaluator does not cross-check against a training
 * seed range, so the disjointness claim below is only made when the range
 * actually in use matches the plan's default — otherwise it would be an
 * unchecked claim about arbitrary CLI input.
 */
const DEFAULT_HELD_OUT_START = 30001;
const DEFAULT_HELD_OUT_COUNT = 100;

const renderLimitations = (report: EvaluationReport): string => {
  const { evaluation } = report;
  const heldOutRange = `${evaluation.heldOutSeeds.start}–${
    evaluation.heldOutSeeds.start + evaluation.heldOutSeeds.count - 1
  }`;
  const isPlanDefaultHeldOutRange =
    evaluation.heldOutSeeds.start === DEFAULT_HELD_OUT_START && evaluation.heldOutSeeds.count === DEFAULT_HELD_OUT_COUNT;
  const heldOutBullet = isPlanDefaultHeldOutRange
    ? `- Held-out seeds (${heldOutRange}) are the plan's default range, disjoint from its training ` +
      '(1–10000) and validation (20001–20064) seed ranges, but are a fixed, finite sample ' +
      '(not the full seed space).'
    : `- This run's held-out seeds (${heldOutRange}) are a non-default range; disjointness from any ` +
      'training/validation seed range was not checked for it. They are, in any case, a fixed, finite ' +
      'sample (not the full seed space).';
  const bullets = [
    '- Headline per-arm statistics (Results, Paired differences) are measured single-agent, opponent ' +
      'parked — see "What this does not show" below.',
    '- Statistics are descriptive (mean/median/std) plus bootstrap confidence intervals; no significance ' +
      'test or superiority claim is made or implied.',
    '- The `silenced` control forces the trained readout’s input vector to zero every tick; it does ' +
      'not silence the recurrent connectome dynamics themselves.',
    heldOutBullet
  ];
  if (report.warnings.length > 0) {
    bullets.push('- This run recorded the following warnings:');
    for (const warning of report.warnings) bullets.push(`  - ${warning}`);
  }
  return ['## Limitations', '', ...bullets].join('\n');
};

const renderWhatThisDoesNotShow = (): string =>
  ['## What this does not show', '', OPPONENT_PARKED_DISCLOSURE].join('\n');

/**
 * Distinct from `nearInputIndependentPolicyArms` below: an arm whose readout
 * input is *provably* zero on every tick of every episode. This must NOT be
 * inferred from `trained`/`silenced` score identity alone — saturated
 * `tanh`/`sigmoid` readout units, or a fitness function insensitive to small
 * action differences, could produce identical scores for a genuinely nonzero
 * input too. Instead this requires BOTH: (a) `armReport.structurallyZeroInput`
 * — `evaluate.ts`'s `graphGuaranteesZeroReadoutInput`, a topological proof
 * from the graph's own arrays (no edges, and no output-assigned neuron is
 * directly input-channel-mapped) that the readout's gathered rates cannot
 * ever be nonzero, independent of any weights or seeds — AND (b) every
 * replica's `trained`/`silenced` paired difference is empirically exactly 0
 * (CI exactly `[0, 0]`), as a corroborating check: if the topological proof
 * held but the empirical scores somehow differed, that would indicate a bug
 * elsewhere, and this finding must stay silent rather than assert something
 * that contradicts the run's own data. The `disconnected` control arm (zero
 * edges) is the expected case both conditions catch together.
 */
const armHasStructurallyZeroReadoutInput = (armReport: Readonly<ArmReport>): boolean => {
  if (!armReport.structurallyZeroInput) return false;
  const replicas = replicasInOrder(armReport);
  if (replicas.length === 0) return false;
  return replicas.every(
    ([, replica]) =>
      replica.pairedTrainedVsSilenced.meanDifference === 0 &&
      replica.pairedTrainedVsSilenced.ci95[0] === 0 &&
      replica.pairedTrainedVsSilenced.ci95[1] === 0
  );
};

/** Exported alongside `nearInputIndependentPolicyArms` for direct testing. */
export const structurallyZeroReadoutInputArms = (report: Readonly<EvaluationReport>): readonly ArmName[] =>
  armsInOrder(report.arms)
    .filter(([, armReport]) => armHasStructurallyZeroReadoutInput(armReport))
    .map(([arm]) => arm);

const renderStructurallyZeroReadoutInputFinding = (report: Readonly<EvaluationReport>): string | null => {
  const arms = structurallyZeroReadoutInputArms(report);
  if (arms.length === 0) return null;
  return [
    '## Finding: readout input was structurally zero',
    '',
    `For ${arms.join(', ')}, the graph itself guarantees the readout's input was exactly zero on every ` +
      'tick of every episode: the graph has no edges, and none of its output-assigned neurons is itself ' +
      'directly wired to an input channel, so their rates can never leave their zero starting value, ' +
      'independent of weights or seeds. Every replica’s `trained` and `silenced` scores were also ' +
      'numerically identical on every held-out seed (paired difference exactly 0, 95% CI exactly ' +
      '[0, 0]), consistent with that guarantee. For that arm, `trained` and `silenced` computed the ' +
      'identical function; whatever score the readout achieved came entirely from its learned bias ' +
      'terms — a fixed, input-independent action — never from sensory information. This finding is ' +
      'descriptive only: it does not rank or compare arms against each other.'
  ].join('\n');
};

/**
 * Whether one arm's replicas all look like a near-input-independent policy:
 * every replica's `trained` mean falls inside the arm's `authored` 95% CI,
 * AND every replica's `trained` vs. `silenced` paired-difference 95% CI
 * includes zero (not distinguishable from no difference). An arm with no
 * replicas evaluated is never "near-input-independent" (there is nothing to
 * judge).
 */
const armLooksNearInputIndependent = (armReport: Readonly<ArmReport>): boolean => {
  const replicas = replicasInOrder(armReport);
  if (replicas.length === 0) return false;
  return replicas.every(([, replica]) => {
    const trainedWithinAuthoredCI =
      replica.trained.mean >= armReport.authored.ci95[0] && replica.trained.mean <= armReport.authored.ci95[1];
    const silencedIndistinguishableFromTrained =
      replica.pairedTrainedVsSilenced.ci95[0] <= 0 && replica.pairedTrainedVsSilenced.ci95[1] >= 0;
    return trainedWithinAuthoredCI && silencedIndistinguishableFromTrained;
  });
};

/**
 * `00-overview.md`'s risk: "If every trained arm scores within the authored
 * decoder's CI and the silenced control matches trained, the report must
 * state that the readout learned a near-input-independent policy." Also
 * used by `05-production-run.md`'s WP5 acceptance risk (identical wording).
 * Exported so `tests/unit/report.test.ts` can exercise the detection logic
 * directly against constructed report fixtures, independent of markdown
 * rendering.
 */
export const nearInputIndependentPolicyArms = (report: Readonly<EvaluationReport>): readonly ArmName[] =>
  armsInOrder(report.arms)
    .filter(([, armReport]) => armLooksNearInputIndependent(armReport))
    .map(([arm]) => arm);

const renderNearInputIndependentFinding = (report: Readonly<EvaluationReport>): string | null => {
  const evaluatedArms = armsInOrder(report.arms).map(([arm]) => arm);
  if (evaluatedArms.length === 0) return null;
  const nearIndependentArms = nearInputIndependentPolicyArms(report);
  if (nearIndependentArms.length === 0) return null;

  const lines = ['## Finding: near-input-independent policy', ''];
  if (nearIndependentArms.length === evaluatedArms.length) {
    lines.push(
      'Every evaluated arm’s trained readout scored within its authored decoder’s 95% confidence ' +
        'interval, and every replica’s trained-vs-silenced paired difference was not distinguishable ' +
        'from zero (its 95% CI includes 0). Per the plan’s stated risk, this means the trained readout ' +
        'learned a near-input-independent policy for every arm: its behavior did not depend on the ' +
        'gathered per-neuron output rates in a way that materially changed the held-out score, ' +
        'relative to both the authored decoder and a version of itself with its input silenced. This ' +
        'finding is descriptive only: it does not rank or compare arms against each other.'
    );
  } else {
    lines.push(
      `For ${nearIndependentArms.join(', ')} (but not every evaluated arm), the trained readout scored ` +
        'within the authored decoder’s 95% confidence interval, and its trained-vs-silenced paired ' +
        'difference was not distinguishable from zero (its 95% CI includes 0) for every replica of that ' +
        'arm — the trained readout for that arm looks near-input-independent. This finding is ' +
        'descriptive only: it does not rank or compare arms against each other.'
    );
  }
  return lines.join('\n');
};

/**
 * Render `report` (the exact object `runEvaluate` builds, before it is
 * `JSON.stringify`'d to `trained-readout-v1.report.json`) as
 * `docs/trained-readout-report.md`'s markdown contents.
 */
export const renderReportMarkdown = (report: EvaluationReport): string => {
  const nearIndependentFinding = renderNearInputIndependentFinding(report);
  const structurallyZeroFinding = renderStructurallyZeroReadoutInputFinding(report);
  return [
    '# Trained-Readout Evaluation Report',
    '',
    renderMethod(report),
    '',
    renderParameterAccounting(report),
    '',
    renderResults(report),
    '',
    renderPairedDifferences(report),
    '',
    renderSideBySide(report),
    '',
    renderLimitations(report),
    '',
    ...(nearIndependentFinding !== null ? [nearIndependentFinding, ''] : []),
    ...(structurallyZeroFinding !== null ? [structurallyZeroFinding, ''] : []),
    renderWhatThisDoesNotShow(),
    ''
  ].join('\n');
};
