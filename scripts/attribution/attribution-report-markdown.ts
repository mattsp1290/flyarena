import type {
  DescendingTypeEntry,
  HypothesisResult,
  ReadoutAttributionArtifact,
  ReadoutAttributionEntry
} from './attribution-report';

/**
 * `./attribution-report.ts`'s report renderer, split out following
 * `scripts/selections/selection-report.ts`/`selection-report-markdown.ts`'s
 * own precedent, so `attribution-report.ts` stays well under the repo's
 * 1000-line-per-file limit. Pure: every function here takes already-built
 * data and returns a string, never reads a file or the clock.
 *
 * The predeclared analyses and H1-H3 outcome rules below are quoted
 * verbatim from `.agents/plans/readout-attribution/00-overview.md`
 * (WP3's own change-surface spec, `03-artifact-and-findings.md`, requires
 * this). H1 has no "not-supported" branch anywhere in the plan's rule text
 * -- only "supported" or "inconclusive" -- so this report states the rule
 * exactly as written rather than inventing a symmetric third branch.
 */

const fmt = (value: number, digits = 4): string => value.toFixed(digits);
const fmtCI = (ci: readonly [number, number] | null, digits = 4): string =>
  ci === null ? 'n/a' : `[${fmt(ci[0], digits)}, ${fmt(ci[1], digits)}]`;
const fmtBool = (value: boolean): string => (value ? 'yes' : 'no');

// ---------------------------------------------------------------------------
// Descending cell-type naming
// ---------------------------------------------------------------------------

/**
 * `d` is a **D-space** index (0..47, `outputNeuronIndices(graph)`'s own
 * ascending position -- the same indexing every `saliency`/`ablation`
 * `input`/D-space field in this artifact uses), NOT
 * `descending-types-v1.json`'s own `index` field (the raw graph-node
 * index, which ranges well past 47 -- e.g. 445). `descending_types.py`'s
 * own docstring: the sidecar's `neurons` array is written "in graph index
 * order", i.e. already ascending by graph-node index -- exactly
 * `outputNeuronIndices`'s own ordering -- so `types[d]` (array position,
 * `loadDescendingTypeNames`'s caller-side sort by `index` ascending makes
 * this safe even if the file's own on-disk order ever changed) is the
 * correct D-space lookup, never `types.find((t) => t.index === d)` (a
 * bug an earlier version of this function had: matching D against the
 * raw graph-node index silently mislabeled every one of the 41 descending
 * neurons whose graph-node index differs from its D-space position, which
 * on this graph is every position past the seventh).
 */
const nameForIndex = (types: readonly DescendingTypeEntry[], d: number): string => {
  const entry = types[d];
  if (!entry) return `#${d}`;
  const side = entry.somaSide ? `_${entry.somaSide}` : '';
  return `${entry.type ?? 'unknown'}${side}`;
};

// ---------------------------------------------------------------------------
// Predeclared method text (verbatim from 00-overview.md)
// ---------------------------------------------------------------------------

const PREDECLARED_ANALYSES = `1. **Input saliency per descending neuron:** for each readout and each of the D = 48 output-neuron inputs, compute the mean absolute input-gradient of the readout's thrust and yaw outputs along that readout's own held-out trajectories (authoritative TS episodes, seeds 30001-30010, T 1800). The full chain rule is \`d(out)/d(r_d) = diag(f_out'(W2*h + b2)) * W2 * diag(1 - h^2) * W1[:, d]\`, where \`f_out'\` is \`1 - tanh^2\` for thrust and yaw. A secondary **variance-weighted** saliency (gradient x that input's trajectory std) is also reported.
2. **Readout-input ablation:** for the top-8 and bottom-8 descending inputs by saliency, per readout, zero that input in the **readout's input only**, leaving the dynamics intact. Measure the paired score change on 100 held-out seeds. This needs a new trained-decoder input mask. Zero is the rate model's equilibrium (leak toward 0; rates clamped to +/-2). Each ablated input's empirical trajectory mean and std are reported next to its effect, so an ablation that pushes a neuron outside its operating range is visible.
3. **Input-independence share:** per readout, \`mean(silenced) / mean(trained)\` over the 100 seeds (a ratio of means, never per-episode). If \`mean(trained) <= 1\`, the share is reported as undefined, which the report discloses. It measures how much of the policy ignores the circuit.
4. **Pathway linkage:** per descending input d, the steady-state clearance transfer \`|T_clear,d|\`, which is the d-row analogue of the explanation's transfer from the clearance channels to that neuron, computed on the readout's graph.`;

const PREDECLARED_HYPOTHESES = `- **H1 (consistent with routing around):** in biological readouts, saliency correlates with the clearance signal that bridge paths deliver. Spearman rho between per-neuron saliency and \`|T_clear,d|\` is >= 0.3, with a **cluster bootstrap** CI excluding 0. The resampling unit is the annotated \`group\` from the WP1c sidecar, or the output population if \`group\` is missing, not the 48 neurons treated as independent. H1 must hold in **all 3** biological trainer seeds, matching the repo's unanimous-seed convention (\`trainedRobust\`). If the cluster-bootstrap and neuron-level-bootstrap CIs disagree on excluding 0, H1 is inconclusive. **Regime gate:** H1 is evaluated only if the trained-readout trajectories pass the explanation study's regime thresholds (clamp fraction <= 20%, steady-state distance <= 0.5, measured by \`regime-task.ts\` logic on those trajectories). Otherwise H1 is inconclusive (regime-invalid). H1's rule defines only two outcomes, "supported" or "inconclusive" -- it has no "not-supported" branch.
- **H2 (constant policy, equivalence):** the input-independence share is equivalent between biological and rewired. Pairing is same-trainer-seed (101/202/303). "Supported" means the 90% CI of the mean paired difference lies inside +/-0.10, which is the predeclared smallest share difference of interest, a TOST-style test. "Not supported" means the CI excludes 0 and lies outside +/-0.10. Anything else is inconclusive. With n = 3 pairs the resolution is coarse, and the report says so.
- **H3 (P redundancy, equivalence or less):** in P's readouts (seeds 101/202/303), saliency on P's **newly connected thrust neurons**, identified from the archived P swap edges, is not larger than in the same-seed biological readouts. "Supported" means the upper 90% bound of the mean paired ratio (P/biological) is <= 1.25. "Not supported" means the lower bound is above 1.25. Anything else is inconclusive.
- No multiple-comparison correction is applied, and the number of hypotheses is disclosed. H1 is always described in correlational language ("consistent with"). Only the ablation results support causal wording, and then only about the readout's reliance on its inputs.`;

// ---------------------------------------------------------------------------
// Coverage
// ---------------------------------------------------------------------------

const renderCoverage = (artifact: ReadoutAttributionArtifact): string => {
  const rows = artifact.readouts
    .map((r) => `| \`${r.id}\` | ${r.arm} | ${r.graphId} | ${r.trainerSeed} | ${r.arenaTask} |`)
    .join('\n');
  return `## Coverage

This artifact covers ${artifact.coverage.ids.length} default-task archived readouts. The committed archive (\`training/archive/trained-readouts-v1.json\`) holds 75 entries total; 52 \`kind: "task-intervention"\` entries across 4 non-default arena tasks (\`crowded\`/\`hazard-heavy\`/\`no-movement\`/\`sparse-food\`, from WP1b) are **excluded** here -- every predeclared analysis and H1-H3 rule in \`00-overview.md\` is scoped to fixed default-task ids (\`biological-seed101\`, \`P-seed202\`, ...) and was never defined over the per-task entries. \`coverage.perTaskIncluded\` is \`false\` in the JSON artifact.

| id | arm | graphId | trainerSeed | arenaTask |
| --- | --- | --- | --- | --- |
${rows}
`;
};

// ---------------------------------------------------------------------------
// Saliency (top-10 combined thrust+yaw score per readout)
// ---------------------------------------------------------------------------

const top10ByCombinedScore = (readout: ReadoutAttributionEntry, types: readonly DescendingTypeEntry[]): string => {
  const combined = readout.saliency.thrust.map((t, d) => ({ d, score: t + readout.saliency.yaw[d] }));
  combined.sort((a, b) => b.score - a.score);
  return combined
    .slice(0, 10)
    .map((entry) => `${nameForIndex(types, entry.d)} (#${entry.d}, ${fmt(entry.score, 3)})`)
    .join('; ');
};

const renderSaliency = (artifact: ReadoutAttributionArtifact, types: readonly DescendingTypeEntry[]): string => {
  const rows = artifact.readouts
    .map((r) => `| \`${r.id}\` | ${top10ByCombinedScore(r, types)} |`)
    .join('\n');
  return `## Saliency

Analysis 1 (\`00-overview.md\`): mean absolute input-gradient of thrust/yaw along each readout's own 10 held-out trajectories (seeds 30001-30010, 1800 ticks). Named against \`descending-types-v1.json\`'s pinned cell-type annotations -- labels only, not a claim about fly descending-neuron function. Below, per readout, the top 10 of the 48 descending inputs by combined (thrust + yaw) saliency score, in descending order:

| id | top 10 by combined thrust+yaw saliency |
| --- | --- |
${rows}

Full per-input thrust/yaw/variance-weighted saliency arrays (all 48 inputs, every readout) are in the JSON artifact's \`readouts[].saliency\`/\`saliencyVarWeighted\`.
`;
};

// ---------------------------------------------------------------------------
// Ablation (biological + P headline readouts get full detail; the rest point at the JSON)
// ---------------------------------------------------------------------------

const HEADLINE_ID_PATTERN = /^(biological|P)-seed(101|202|303)$/;

const renderAblationTable = (readout: ReadoutAttributionEntry, types: readonly DescendingTypeEntry[]): string => {
  const rows = [...readout.ablation]
    .sort((a, b) => (a.rank === b.rank ? b.saliencyScore - a.saliencyScore : a.rank === 'top' ? -1 : 1))
    .map(
      (a) =>
        `| ${a.rank} | ${nameForIndex(types, a.input)} (#${a.input}) | ${fmt(a.saliencyScore, 3)} | ${fmt(a.inputMean, 4)} | ${fmt(
          a.inputStd,
          4
        )} | ${fmt(a.effect, 3)} | ${fmtCI(a.ci, 3)} |`
    )
    .join('\n');
  return `#### \`${readout.id}\` (baseline held-out score: see \`independence.json\`'s \`trainedMean\`; n=${readout.ablation[0]?.n ?? 100} paired seeds per input)

| rank | input | saliency score | input mean | input std | effect (paired mean diff) | 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
${rows}
`;
};

const renderAblation = (artifact: ReadoutAttributionArtifact, types: readonly DescendingTypeEntry[]): string => {
  const headline = artifact.readouts.filter((r) => HEADLINE_ID_PATTERN.test(r.id));
  const tables = headline.map((r) => renderAblationTable(r, types)).join('\n');
  return `## Ablation

Analysis 2 (\`00-overview.md\`): for the top-8 and bottom-8 descending inputs by saliency (thrust+yaw summed), zero that input in the readout's input only (dynamics untouched), and measure the paired score change on 100 held-out seeds (seeds 30001-30100). Full detail below for the 6 headline readouts central to H1/H3 (biological and P, all three trainer seeds each); every readout's full 16-input ablation table is in the JSON artifact's \`readouts[].ablation\`.

${tables}`;
};

// ---------------------------------------------------------------------------
// Independence
// ---------------------------------------------------------------------------

const renderIndependence = (artifact: ReadoutAttributionArtifact): string => {
  const rows = artifact.readouts
    .map(
      (r) =>
        `| \`${r.id}\` | ${fmt(r.independence.trainedMean, 3)} | ${fmt(r.independence.silencedMean, 3)} | ${
          r.independence.defined ? fmt(r.independence.ratio as number, 4) : 'undefined'
        } | ${fmtBool(r.independence.defined)} |`
    )
    .join('\n');
  return `## Input independence

Analysis 3 (\`00-overview.md\`): \`mean(silenced) / mean(trained)\` over 100 held-out seeds (a ratio of means, never per-episode); undefined when \`mean(trained) <= 1\`.

| id | trained mean | silenced mean | ratio | defined |
| --- | --- | --- | --- | --- |
${rows}
`;
};

// ---------------------------------------------------------------------------
// Linkage
// ---------------------------------------------------------------------------

const renderLinkage = (artifact: ReadoutAttributionArtifact): string => {
  const rows = artifact.readouts
    .map(
      (r) =>
        `| \`${r.id}\` | ${r.linkage.degenerate ? 'degenerate' : fmt(r.linkage.rhoThrust as number, 3)} | ${
          r.linkage.degenerate ? 'n/a' : fmt(r.linkage.rhoYaw as number, 3)
        } | ${fmtCI(r.linkage.ciCluster, 3)} | ${fmtCI(r.linkage.ciNeuron, 3)} | ${r.linkage.clusterCount} |`
    )
    .join('\n');
  return `## Pathway linkage

Analysis 4 (\`00-overview.md\`): per descending input d, the steady-state clearance transfer \`|T_clear,d|\`, correlated (Spearman) against saliency, with a cluster bootstrap over the pinned annotations' \`group\` (28 clusters for the 48 neurons: 20 bilateral pairs and 8 singletons) and a neuron-level bootstrap reported as a sensitivity check.

| id | rho (thrust) | rho (yaw) | 90% CI (cluster) | 90% CI (neuron) | cluster count |
| --- | --- | --- | --- | --- | --- |
${rows}
`;
};

// ---------------------------------------------------------------------------
// Regime
// ---------------------------------------------------------------------------

const renderRegime = (artifact: ReadoutAttributionArtifact): string => {
  const rows = artifact.readouts
    .map(
      (r) =>
        `| \`${r.id}\` | ${fmt(r.regime.clampFraction, 4)} | ${fmt(r.regime.steadyStateDistance, 4)} | ${fmtBool(r.regime.valid)} |`
    )
    .join('\n');
  return `## Regime check (H1's gate)

The same clamp-fraction/steady-state-distance statistic, at the same granularity, as the explanation study (\`regime-metrics.ts\`, extracted from \`regime-task.ts\`); H1 is evaluated only for a biological seed whose trajectories pass both thresholds (clamp fraction <= 20%, steady-state distance <= 0.5).

| id | clamp fraction | steady-state distance | valid |
| --- | --- | --- | --- |
${rows}
`;
};

// ---------------------------------------------------------------------------
// Hypothesis outcomes
// ---------------------------------------------------------------------------

interface H1SeedEvidence {
  readonly id: string;
  readonly regimeValid: boolean;
  readonly degenerate: boolean;
  readonly rho: number | null;
  readonly ciCluster: readonly [number, number] | null;
  readonly ciNeuron: readonly [number, number] | null;
  readonly meetsThreshold: boolean;
}
interface H1Evidence {
  readonly rhoThreshold: number;
  readonly seeds: readonly H1SeedEvidence[];
}
interface H2Evidence {
  readonly equivalenceBound: number;
  readonly differences: readonly number[];
  readonly ci: readonly [number, number];
  readonly n: number;
}
interface H3RatioEntry {
  readonly seed: number;
  readonly pMean: number;
  readonly bioMean: number;
  readonly ratio: number;
}
interface H3Evidence {
  readonly ratioBound: number;
  readonly ratios: readonly H3RatioEntry[];
  readonly ci: readonly [number, number];
  readonly newlyConnectedThrustDIndices: readonly number[];
}

const renderH1 = (result: HypothesisResult): string => {
  const evidence = result.evidence as H1Evidence;
  const rows = evidence.seeds
    .map(
      (s) =>
        `| \`${s.id}\` | ${fmtBool(s.regimeValid)} | ${s.degenerate ? 'degenerate' : fmt(s.rho as number, 3)} | ${fmtCI(
          s.ciCluster,
          3
        )} | ${fmtCI(s.ciNeuron, 3)} | ${fmtBool(s.meetsThreshold)} |`
    )
    .join('\n');
  return `### H1 (consistent with routing around)

**Outcome: ${result.outcome}**${result.reason ? ` (${result.reason})` : ''}. Threshold: rho >= ${evidence.rhoThreshold}, unanimous across all 3 biological trainer seeds, cluster-bootstrap CI excluding 0, agreeing with the neuron-level bootstrap.

| seed | regime valid | rho (thrust) | 90% CI (cluster) | 90% CI (neuron) | meets threshold |
| --- | --- | --- | --- | --- | --- |
${rows}

None of the 3 biological trainer seeds' rho meets the >= 0.3 threshold (seed101: ${fmt(evidence.seeds[0]?.rho ?? NaN, 3)}, seed202: ${fmt(
    evidence.seeds[1]?.rho ?? NaN,
    3
  )}, seed303: ${fmt(evidence.seeds[2]?.rho ?? NaN, 3)}) -- two of the three are even negative. H1's rule has no "not-supported" branch (see the predeclared rules above); with the threshold unmet in all seeds, the mechanical result is **inconclusive**, stated exactly as the plan's rule defines it, never read as a stronger "refuted" claim.
`;
};

const renderH2 = (result: HypothesisResult): string => {
  const evidence = result.evidence as H2Evidence;
  return `### H2 (constant policy, equivalence)

**Outcome: ${result.outcome}**${result.reason ? ` (${result.reason})` : ''}. Equivalence bound: +/-${evidence.equivalenceBound}. Paired differences (biological minus rewired independence share, same trainer seed): ${evidence.differences
    .map((d) => fmt(d, 4))
    .join(', ')} (n=${evidence.n}). t-based 90% CI of the mean paired difference: ${fmtCI(evidence.ci, 4)}.

The CI (${fmtCI(evidence.ci, 4)}) neither sits entirely inside +/-${evidence.equivalenceBound} (which "supported" would require) nor entirely outside it while excluding 0 (which "not supported" would require) -- it straddles both boundaries. With n = 3 paired seeds the resolution is coarse, exactly as \`00-overview.md\` anticipates; the mechanical result is **inconclusive**, read as inconclusive, never as "not consistent" or any stronger claim.
`;
};

const renderH3 = (result: HypothesisResult): string => {
  const evidence = result.evidence as H3Evidence;
  const rows = evidence.ratios
    .map((r) => `| ${r.seed} | ${fmt(r.pMean, 6)} | ${fmt(r.bioMean, 6)} | ${fmt(r.ratio, 2)} |`)
    .join('\n');
  return `### H3 (P redundancy, equivalence or less)

**Outcome: ${result.outcome}**${result.reason ? ` (${result.reason})` : ''}. Ratio bound: <= ${evidence.ratioBound}. P's newly connected thrust neurons (D-space indices, from the archived swap edges): ${evidence.newlyConnectedThrustDIndices.join(', ')}.

| seed | P mean saliency | biological mean saliency | ratio (P / biological) |
| --- | --- | --- | --- |
${rows}

t-based 90% CI of the mean paired ratio: ${fmtCI(evidence.ci, 2)}. The per-seed ratios are highly skewed: seed 202's ratio (${fmt(
    evidence.ratios[1]?.ratio ?? NaN,
    2
  )}) is driven by a near-zero biological mean saliency denominator (${fmt(
    evidence.ratios[1]?.bioMean ?? NaN,
    6
  )}) in that one seed, not by a genuinely large P effect -- stated plainly, not smoothed over. The CI's negative lower bound (${fmt(
    evidence.ci[0],
    2
  )}) is an artifact of fitting a t-interval to 3 highly skewed values, not evidence that the ratio is ever negative (a mean saliency ratio cannot be negative by construction). With the CI this wide and straddling the <= 1.25 bound in both directions, the mechanical result is **inconclusive**.
`;
};

// ---------------------------------------------------------------------------
// What remains unexplained (required paragraph, 03-artifact-and-findings.md)
// ---------------------------------------------------------------------------

const WHAT_REMAINS_UNEXPLAINED = `## What remains unexplained

This study set out to explain, under this model, why CEM-retrained readouts erase the biological graph's deficit: trained biological scores 62.7-72.2 versus rewired scores 68.0-74.1 (\`docs/trained-readout-report.md\`), and why the pathway intervention P gains nothing under trained readouts (\`docs/pathway-interventions-report.md\`). All three predeclared hypotheses came back **inconclusive**, so this evidence does not account for either result -- not partially, not with caveats folded in, but genuinely not.

For the ~5-10 point score-convergence gap: H1 (routing-around) found no rho >= 0.3 in any of the 3 biological trainer seeds -- two of three seeds even show a small negative correlation between saliency and the clearance transfer. H2 (constant-policy equivalence) could neither confirm nor rule out that biological and rewired trained readouts rely on their circuit to the same degree; its CI straddles both the equivalence and non-equivalence boundaries at n=3. Neither result points at a mechanism for why training closes the gap -- the combined evidence leaves that question exactly where it started, not partially resolved.

For P's null trained result: H3 (P-redundancy) could neither confirm nor rule out that saliency on P's newly-connected thrust neurons is no larger than biological's, again at the coarse n=3 resolution and with one seed's ratio dominated by a near-zero denominator, not a real effect. This study offers no positive account of why P gains nothing under trained readouts either.

What this study does establish: the readout-input ablation results (a causal claim about a readout's own reliance on its inputs, not merely correlational) are real, measured effects on each readout's own held-out score, reported per readout above and in the JSON artifact -- these are the one part of this study's evidence that supports causal language, and even they speak only to each readout's reliance on its own inputs, not to why training converges the two topologies' scores.
`;

// ---------------------------------------------------------------------------
// Limitations (required content, 03-artifact-and-findings.md)
// ---------------------------------------------------------------------------

const LIMITATIONS = `## Limitations

- Trainer-seed variance is large and every hypothesis rule is evaluated per seed; the seed count (3) is disclosed throughout.
- Readouts are the only trained parameters -- the dynamics and topology are fixed and authored/measured, not trained.
- An analytic saliency (the exact chain-rule gradient of the readout's own forward pass) is not a causal claim about the circuit; the readout-input ablation results are included specifically because they, unlike saliency, do support causal language about a readout's own reliance on its inputs.
- Cell-type names (\`descending-types-v1.json\`) are annotation labels copied from the pinned MaleCNS table, never a claim about fly descending-neuron function.
- Three hypotheses (H1, H2, H3) are evaluated with no multiple-comparison correction; this is disclosed, not adjusted for.
- H2 and H3 use n = 3 paired trainer seeds and predeclared equivalence bounds (+/-0.10 independence-share difference for H2, <= 1.25 mean saliency ratio for H3) -- a coarse resolution, stated in \`00-overview.md\` itself, and confirmed coarse in practice by this study's own straddling CIs.
- H1 is stated only in correlational language ("consistent with"), never as a causal claim.
- The linkage cluster bootstrap resamples 28 clusters (20 bilateral pairs, 8 singletons), not the 48 neurons independently; a neuron-level bootstrap is reported alongside as a sensitivity check, and H1 is inconclusive whenever the two disagree on excluding 0.
- The 90% CIs for H2/H3 use a one-sample t interval (df = 2) rather than a percentile bootstrap: at n = 3, a percentile bootstrap of the mean cannot extend beyond the sample's own min/max (only 10 distinct resample multisets exist), which would understate real sampling uncertainty and make a "supported" verdict artificially easy to reach. The t-interval is the conventional TOST construction for a small paired sample and is used here instead -- a deliberate deviation from a percentile-bootstrap default, disclosed here.
- Coverage is scoped to the 23 default-task archive entries; the 52 per-task entries (\`kind: "task-intervention"\`, WP1b) are excluded because none of this study's predeclared analyses or hypothesis rules were ever defined over them (see Coverage above).
- This model only: no claim is made about the real fly's descending-neuron function or the biological plausibility of CEM training.
`;

// ---------------------------------------------------------------------------
// Top-level renderer
// ---------------------------------------------------------------------------

export const renderReadoutAttributionReportMarkdown = (
  artifact: ReadoutAttributionArtifact,
  descendingTypes: readonly DescendingTypeEntry[]
): string => {
  const sections = [
    `# Readout attribution (under this model)

**Question.** Why do CEM-retrained readouts erase the biological graph's deficit against its degree-preserving rewirings (trained biological scores 62.7-72.2 versus rewired scores 68.0-74.1), and why does the pathway intervention P gain nothing under trained readouts? This report tests three predeclared hypotheses against per-readout input saliency, readout-input ablation, input-independence share, and pathway linkage, computed on the ${artifact.coverage.ids.length} default-task readouts this artifact covers. Every result below is descriptive and bound to this model; none is a claim about the real fly.
`,
    `## Predeclared analyses

Quoted verbatim from \`.agents/plans/readout-attribution/00-overview.md\`:

${PREDECLARED_ANALYSES}
`,
    `## Predeclared hypotheses and outcome rules

Quoted verbatim from \`.agents/plans/readout-attribution/00-overview.md\`:

${PREDECLARED_HYPOTHESES}
`,
    renderCoverage(artifact),
    renderSaliency(artifact, descendingTypes),
    renderAblation(artifact, descendingTypes),
    renderIndependence(artifact),
    renderLinkage(artifact),
    renderRegime(artifact),
    `## Hypothesis outcomes

Hypothesis count: ${artifact.hypotheses.hypothesisCount}. Multiple-comparison correction: ${artifact.hypotheses.multipleComparisonCorrection}.

${renderH1(artifact.hypotheses.H1)}
${renderH2(artifact.hypotheses.H2)}
${renderH3(artifact.hypotheses.H3)}`,
    WHAT_REMAINS_UNEXPLAINED,
    LIMITATIONS
  ];
  return `${sections.join('\n')}`;
};
