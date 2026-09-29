import type { RobustnessVerdict, SelectionId, SelectionResult, SelectionRobustnessArtifact } from './selection-report';

/**
 * `renderSelectionRobustnessReportMarkdown`, split out of `./selection-report.ts`
 * on the same precedent `scripts/atlas/repertoire-report.ts`/
 * `repertoire-report-markdown.ts` already set, so the producer file stays
 * well under the repo's 1000-line limit. `03-artifact-and-findings.md`'s
 * required structure: Question; the selection table verbatim; predeclared
 * rules verbatim; per-selection results; overall; limitations.
 */

const pct = (value: number): string => `${(value * 100).toFixed(1)}%`;

/** `00-overview.md`'s own selection table, quoted verbatim (this file's own doc comment below reproduces the exact source cells; do not paraphrase). */
const SELECTION_TABLE_MD = `| id | Change versus \`compile.py\` defaults | Size |
| --- | --- | --- |
| \`larger\` | \`BRIDGE_TARGET 1600\` | about 1,808 neurons |
| \`smaller\` | \`BRIDGE_TARGET 400\` | about 608 neurons |
| \`random-bridge\` | Bridge neurons are sampled uniformly from the eligible candidates with a seeded RNG (seed \`20260927\`) instead of by degree rank, size 800 | 1,008 |
| \`alt-sensory-mapping\` | The same neurons, but sensory→channel assignment uses a seeded permutation (seed \`20260927\`) of the sorted sensory ids instead of contiguous body-id blocks. The descending mapping is unchanged | 1,008 |`;

/** `00-overview.md`'s "Predeclared outcome per selection" section, quoted verbatim. */
const PREDECLARED_RULES_MD = `- **Null holds:** biological is below that selection's rewired-null 25th percentile.
- **Explanation replicates:** at least one of the two original gate-passing **transfer** entries (\`T:rightClearance->thrust\`, \`T:forwardClearance->thrust\`), which are the ones P's search optimizes, passes both predeclared gates in that selection's explanation. \`weightedInDegree:thrust\` is reported separately as \`structuralReplicates\` and never decides \`replicates\`. \`selection-report.ts\` recomputes both flags **in TypeScript** from the raw \`spearman\` and \`bioPercentile\` values and the published thresholds. It does not trust the flags from \`explain.py\`.
- **Pathway supported:** P for that selection passes the original rule (at or above that selection's null 25th percentile, and above the 95th percentile of both C and M).
- A selection whose null, C, or M is degenerate (the IQR guard) is not categorized.
- **Overall:** reported on two axes, never as "4 independent selections". \`larger\`, \`default\`, and \`smaller\` are nested cuts of one degree ranking, so together they test **bridge-population size**. \`random-bridge\` tests **selection method**. \`alt-sensory-mapping\` tests **the authored channel mapping**, not topology. Each verdict is \`true\`, \`false\`, or \`"indeterminate"\` with a reason. "Robust to size" is \`true\` if all three findings hold for both \`larger\` and \`smaller\`, \`false\` if any finding fails in a categorized one, and \`"indeterminate"\` if either is uncategorized and none failed. "Robust to method" follows the same rule for \`random-bridge\`. The mapping result is reported by itself. Degenerate selections are not categorized.
- **Search-budget disclosure:** each selection reports P's swap count \`k\`, \`targetReached\`, and the size of its bridge-candidate pool. A P that stopped at the 200-swap cap without reaching its target is reported as "search-limited", not "not supported".
- **Coverage check (\`random-bridge\`, and recorded for all):** per input channel and output population, the number of selected bridge neurons with at least one edge from that channel's sensory neurons and at least one edge to that population's descending neurons. Any channel or population with zero coverage is flagged, and a selection with a flagged channel is not categorized for the pathway finding.`;

const verdictText = (verdict: RobustnessVerdict): string =>
  verdict.verdict === true ? '**true**' : verdict.verdict === false ? `**false** -- ${verdict.reason}` : `**indeterminate** -- ${verdict.reason}`;

const renderNullSection = (selection: SelectionResult): string => {
  const { null: nullResult } = selection;
  const holdsText = nullResult.holds ? 'holds' : 'does not hold';
  return (
    `Null ${holdsText}: biological's score ${nullResult.bioScore.toFixed(4)} vs. the rewired-null 25th percentile ` +
    `${nullResult.p25.toFixed(4)} (bioPercentile ${pct(nullResult.bioPercentile)} of 500 rewirings, i.e. biological ranks ` +
    `below ${Math.round(nullResult.bioPercentile * 500)} of 500).`
  );
};

const renderExplanationSection = (selection: SelectionResult): string => {
  const { explanation } = selection;
  const passingList =
    explanation.passing.length === 0
      ? 'no metric independently passes both predeclared gates'
      : explanation.passing.map((metric) => `\`${metric.name}\` (rho=${metric.spearman.toFixed(3)}, ${pct(metric.bioPercentile)} of null)`).join('; ');
  const lines = [
    `Explanation ${explanation.replicates ? 'replicates' : 'does not replicate'} (structuralReplicates: ${explanation.structuralReplicates}): ${passingList}.`,
    `Mirrored decoder (flip-both): biological moves to the ${pct(explanation.mirroredBioPercentile)} percentile.`
  ];
  if (explanation.singleAxis) {
    lines.push(
      `The mirrored run cleared the conditional 25% threshold, so the single-axis variants ran: flip-thrust ` +
        `${pct(explanation.singleAxis.flipThrust)}, flip-yaw ${pct(explanation.singleAxis.flipYaw)}.`
    );
  }
  lines.push(`Exploratory feature: omitted (${explanation.exploratoryOmittedReason}).`);
  return lines.join(' ');
};

const renderPathwaySection = (selection: SelectionResult): string => {
  const { pathway } = selection;
  if (pathway.cDegenerate || pathway.mDegenerate) {
    return (
      `Pathway P/Q: **degenerate**, not categorized. ${pathway.degenerateMechanism}. ` +
      `P score ${pathway.pScore.toFixed(4)}, Q score ${pathway.qScore.toFixed(4)}, published null floor ` +
      `${selection.null.p25.toFixed(4)}, k=${pathway.k}, qK=${pathway.qK}, bridge-candidate pool ${pathway.bridgePoolSize}.`
    );
  }
  const supportedText = pathway.searchLimited ? 'search-limited' : pathway.supported ? 'pathway-supported' : 'not-supported';
  return (
    `Pathway P: **${supportedText}** -- score ${pathway.pScore.toFixed(4)} vs. null floor ${selection.null.p25.toFixed(4)}, ` +
    `C p95 ${pathway.cP95.toFixed(4)}, M p95 ${pathway.mP95.toFixed(4)}, k=${pathway.k} of ${pathway.maxSwaps} swaps, ` +
    `targetReached=${pathway.targetReached}, bridge-candidate pool ${pathway.bridgePoolSize}. ` +
    `Q channel-specific: ${pathway.channelSpecific} (score ${pathway.qScore.toFixed(4)}, k=${pathway.qK}).`
  );
};

const renderCoverageSection = (selection: SelectionResult): string => {
  const { coverage } = selection;
  return coverage.flagged
    ? `Coverage: **flagged** -- at least one sensory channel or descending population has zero bridge coverage.`
    : `Coverage: not flagged (every sensory channel and descending population has at least one covered bridge neuron).`;
};

const SELECTION_TITLE: Readonly<Record<SelectionId, string>> = {
  larger: 'Larger (bridge target 1600, ~1,808 neurons)',
  smaller: 'Smaller (bridge target 400, ~608 neurons)',
  'random-bridge': 'Random bridge (seeded-uniform bridge sample, seed 20260927)',
  'alt-sensory-mapping': 'Alt sensory mapping (seeded-permutation channel assignment, seed 20260927)'
};

const renderSelectionSection = (selection: SelectionResult): string => {
  const lines = [
    `### \`${selection.id}\` -- ${SELECTION_TITLE[selection.id]}`,
    '',
    `Graph sha \`${selection.graphSha.slice(0, 12)}\`, global gain ${selection.globalGain}, ${selection.counts.finalNodeCount} neurons, ` +
      `compiled from git revision \`${selection.compiledFromGitRevision.slice(0, 12)}\` (compiler source sha \`${selection.compilerSourceSha256.slice(0, 12)}\`).`,
    '',
    renderNullSection(selection),
    '',
    renderExplanationSection(selection),
    '',
    renderPathwaySection(selection),
    '',
    renderCoverageSection(selection),
    '',
    selection.categorized ? 'Categorized: yes.' : `Categorized: no -- ${selection.categorizedReason}.`
  ];
  return lines.join('\n');
};

const STATIC_LIMITATIONS = [
  "`larger`, `default`, and `smaller` are nested cuts of one degree ranking, testing size, not independent selections.",
  'The 200-swap cap is not scaled to the bridge-pool size.',
  'The exploratory unrestricted feature is not recomputed for any selection (`--selection-mode` records `exploratory: null`).',
  'One seed per seeded variant (`random-bridge`, `alt-sensory-mapping`).',
  'All selections use the same candidate pools and synapse threshold, so `bridgePoolSize` is a property of that ' +
    'shared candidate set, not of the selection -- it is expected to be identical across all four rows, not a bug.',
  '`alt-sensory-mapping` changes the authored channel assignment; it is not a topology test.',
  'Authored decoder and the default arena task only.',
  'Uncorrected comparisons across 4 selections x 3 findings each (12 comparisons); no multiple-comparisons adjustment is applied.',
  'This model only -- no biological claim about the real fly is made anywhere in this report.'
];

const renderLimitations = (): string => STATIC_LIMITATIONS.map((line) => `- ${line}`).join('\n');

export const renderSelectionRobustnessReportMarkdown = (artifact: SelectionRobustnessArtifact): string => {
  const lines: string[] = [];
  lines.push('# Subgraph-selection robustness (under this model)');
  lines.push('');
  lines.push(
    '**Question.** Do the headline authored-decoder findings -- biological at the bottom of its degree-preserving ' +
      'null, the clearance->thrust metrics explaining the gap, and pathway P being pathway-supported -- depend on ' +
      '*how* the 1,008-neuron subgraph was selected? This report tests four predeclared alternative selections, ' +
      'all compiled from the same pinned MaleCNS raw files, under the authored decoder and the default arena task ' +
      'only. Every result below is descriptive and bound to this model; none is a claim about the real fly.'
  );
  lines.push('');
  lines.push('## Selections');
  lines.push('');
  lines.push(SELECTION_TABLE_MD);
  lines.push('');
  lines.push('## Predeclared outcome rules');
  lines.push('');
  lines.push('Quoted verbatim from `.agents/plans/selection-robustness/00-overview.md`:');
  lines.push('');
  lines.push(PREDECLARED_RULES_MD);
  lines.push('');
  lines.push('## Per-selection results');
  lines.push('');
  lines.push(...artifact.selections.map((selection) => renderSelectionSection(selection) + '\n'));
  lines.push('## Overall');
  lines.push('');
  lines.push(`- **Robust to size** (\`smaller\` + \`larger\`): ${verdictText(artifact.overall.robustToSize)}.`);
  lines.push(`- **Robust to method** (\`random-bridge\`): ${verdictText(artifact.overall.robustToMethod)}.`);
  lines.push(`- **Channel mapping** (\`alt-sensory-mapping\`): ${verdictText(artifact.overall.mapping)}.`);
  lines.push('');
  lines.push(
    'This is reported on two axes plus the mapping result by itself, never as "N of 4 selections agree" -- ' +
      '`larger`/`smaller` are nested cuts of one degree ranking (a size test), `random-bridge` is a method test, ' +
      'and `alt-sensory-mapping` is a channel-mapping test, not a topology test. See the Predeclared outcome rules ' +
      'above for the exact aggregation rule.'
  );
  lines.push('');
  lines.push('## Provenance');
  lines.push('');
  lines.push(`- Default (shipped) graph sha: \`${artifact.sources.defaultGraphSha.slice(0, 12)}\`.`);
  lines.push(`- Raw MaleCNS file shas (pinned, \`scripts/data/download.py\`): ${artifact.sources.rawFileShas.map((sha) => `\`${sha.slice(0, 12)}\``).join(', ')}.`);
  lines.push(`- Current compiler source sha (\`scripts/data/compile.py\` + \`scripts/data/selections.py\`): \`${artifact.sources.compilerSourceSha.slice(0, 12)}\`.`);
  lines.push(
    `- Producer: \`${artifact.sources.producer.script}\`, source sha \`${artifact.sources.producer.sourceSha256.slice(0, 12)}\`, ` +
      `${artifact.sources.producer.dependencies.length} dependency files.`
  );
  lines.push('- Each selection above records its own `compiledFromGitRevision`/`compilerSourceSha256` -- see "Per-selection results".');
  lines.push('');
  lines.push('## Limitations');
  lines.push('');
  lines.push(renderLimitations());
  lines.push('');
  return lines.join('\n');
};
