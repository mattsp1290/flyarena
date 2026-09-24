import type { ArmName, ArmProvenance } from './export-arms';
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
  'below for the shipped two-agent condition.';

const ARM_ORDER: readonly ArmName[] = ['biological', 'rewired', 'disconnected'];

/** Stable arm ordering for every table: fixed `ArmName` order, not object-key insertion order. */
const armsInOrder = (arms: Readonly<Record<string, ArmReport>>): Array<[ArmName, ArmReport]> =>
  ARM_ORDER.filter((arm) => arms[arm] !== undefined).map((arm) => [arm, arms[arm]!]);

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

const renderMethod = (report: EvaluationReport): string => {
  const { evaluation, graph } = report;
  const lines = [
    '## Method',
    '',
    `Each arm (biological, rewired, disconnected) is scored on ${evaluation.heldOutSeeds.count} held-out seeds ` +
      `(seeds ${evaluation.heldOutSeeds.start}–${
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
      `interval of the mean (${evaluation.bootstrap.resamples.toLocaleString('en-US')} seeded resamples, ` +
      `bootstrap seed ${evaluation.bootstrap.seed}). For every arm pair and every trained-vs-authored/` +
      'trained-vs-silenced comparison: a paired difference on the same seeds, with its own 95% bootstrap CI. ' +
      'No significance or superiority language is used beyond these confidence intervals.',
    '',
    `Graph source: \`${graph.source}\`${graph.path ? ` (\`${graph.path}\`)` : ''}, sha256 \`${graph.sha256}\`.`
  ];
  return lines.join('\n');
};

const renderParameterAccounting = (report: EvaluationReport): string => {
  const rows = armsInOrder(report.arms).map(([arm, armReport]) => {
    const replicas = replicasInOrder(armReport);
    const first = replicas[0]?.[1];
    return [
      arm,
      String(armReport.D),
      first ? String(first.H) : 'n/a (no trained replica)',
      first ? String(first.parameterCount) : 'n/a (no trained replica)'
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

const renderLimitations = (report: EvaluationReport): string => {
  const bullets = [
    '- Headline per-arm statistics (Results, Paired differences) are measured single-agent, opponent ' +
      'parked — see "What this does not show" below.',
    '- Statistics are descriptive (mean/median/std) plus bootstrap confidence intervals; no significance ' +
      'test or superiority claim is made or implied.',
    '- The `silenced` control forces the trained readout’s input vector to zero every tick; it does ' +
      'not silence the recurrent connectome dynamics themselves.',
    '- Held-out seeds are disjoint from training/validation seed ranges, but are a fixed, finite sample ' +
      '(not the full seed space).'
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
 * Render `report` (the exact object `runEvaluate` builds, before it is
 * `JSON.stringify`'d to `trained-readout-v1.report.json`) as
 * `docs/trained-readout-report.md`'s markdown contents.
 */
export const renderReportMarkdown = (report: EvaluationReport): string =>
  [
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
    renderWhatThisDoesNotShow(),
    ''
  ].join('\n');
