<script lang="ts">
  import type { NullExplanationLoadResult, NullExplanationQualifyingMetric } from '../experiment/nullExplanation';
  import {
    P_TRAINER_SEEDS,
    type PathwayInterventionsAuthoredCategory,
    type PathwayInterventionsLoadResult,
    type PathwayInterventionsTrainedCategory
  } from '../experiment/pathwayInterventions';
  import { githubDocUrl } from './links';
  import { formatPercentile } from '../findings/format';

  /**
   * WP4 of `.agents/plans/null-explanation` (`04-ledger-note.md`): the
   * finding note that explains, under this model only, why biological
   * scored where it did among the graph's rewired versions above. Extracted
   * out of `LedgerPanel.svelte` (thermo-maintainability review, Important
   * I1) — this block grew that panel 320 -> 545 lines by inlining exactly
   * the kind of self-contained derived-artifact block `NullHistogram.svelte`
   * (the rewiring-null histogram just above this note) was already pulled
   * out of the same panel for; this component follows that same precedent:
   * it consumes only `nullExplanation` and (for the mirrored-decoder
   * baseline and the lead sentence's rewiring count) two primitives its
   * caller already has in scope from `rewiringNull.data`, and owns its own
   * markup/CSS/derived formatting.
   *
   * `LedgerPanel.svelte` only renders this once `rewiringNull.status ===
   * 'ok'` (mirroring `NullHistogram.svelte`'s own contract), so
   * `baselinePercentile`/`rewiringCount` are always real numbers here, never
   * placeholders.
   */
  interface Props {
    /** `undefined` while `ExperimentController#initialize()`'s null-explanation load has not yet resolved. */
    nullExplanation: NullExplanationLoadResult | undefined;
    /** `rewiringNull.data.bioPercentile` — the un-mirrored baseline the mirrored-decoder clause below compares against. */
    baselinePercentile: number;
    /** `rewiringNull.data.null.n` — the null set size (the report's `500` rewired versions), read from the verified artifact rather than hard-coded so a future re-run with a different count can never leave this note silently describing the wrong run. */
    rewiringCount: number;
    /** `undefined` while `ExperimentController#initialize()`'s pathway-interventions load (WP4 of `.agents/plans/pathway-interventions`) has not yet resolved. */
    pathwayInterventions: PathwayInterventionsLoadResult | undefined;
  }

  let { nullExplanation, baselinePercentile, rewiringCount, pathwayInterventions }: Props = $props();

  /** WP4's report link, built the same way `NullHistogram.svelte`'s own `GITHUB_REPORT_URL` is (`./links.ts#githubDocUrl`) — `docs/` is not part of the deployed static site, so a relative link would 404 under any base path. */
  const NULL_EXPLANATION_REPORT_URL = githubDocUrl('null-explanation-report.md');
  /**
   * The definition-sensitivity disclosure lives under
   * `docs/null-explanation-report.md`'s "Structural features" heading (the
   * feature-6/`weightedInDegree` adjudication) — GitHub slugifies that
   * heading to this same anchor. There is no more specific heading to link:
   * the adjudication itself is inline bold text within that section, not its
   * own markdown heading.
   */
  const NULL_EXPLANATION_DISCLOSURE_URL = `${NULL_EXPLANATION_REPORT_URL}#structural-features`;

  /**
   * Turns a lowerCamelCase channel/population identifier into lowercase
   * words separated by spaces (e.g. "rightClearance" -> "right clearance",
   * "foodBearing" -> "food bearing") — used only to phrase a qualifying
   * metric's own name in plain words below; every sentence built from it is
   * still generated from the verified artifact's own metric names, never a
   * hard-coded per-metric string table.
   */
  const humanizeIdentifier = (identifier: string): string => identifier.replace(/([a-z0-9])([A-Z])/g, '$1 $2').toLowerCase();

  /**
   * Plain-words phrasing for one qualifying metric
   * (`NullExplanationFinding.qualifyingMetrics[]`), generated from the
   * verified artifact's own `kind`/`name` — never a hard-coded per-metric
   * string table. Two name shapes get their own plain-English phrasing per
   * the bean's non-negotiables ("explain T entries and 'weighted in-degree
   * from input neurons' plainly"); any other kind/name combination (none
   * currently qualify, but the formatter must not silently drop a future
   * one) falls back to a generic, still-honest phrasing built from the same
   * fields.
   */
  const describeQualifyingMetric = (metric: NullExplanationQualifyingMetric): string => {
    const transferMatch = metric.kind === 'transfer' ? /^T:([A-Za-z0-9]+)->([A-Za-z0-9]+)$/.exec(metric.name) : null;
    if (transferMatch) {
      const [, channel, population] = transferMatch;
      return `linear signal gain from ${humanizeIdentifier(channel)} input to ${humanizeIdentifier(population)} output`;
    }
    if (metric.kind === 'feature' && metric.name.startsWith('weightedInDegree:')) {
      const population = metric.name.slice('weightedInDegree:'.length);
      return `weighted in-degree from input neurons to ${humanizeIdentifier(population)} output`;
    }
    return `${metric.kind} metric "${humanizeIdentifier(metric.name)}"`;
  };

  /**
   * `explain.py`'s own `definitionSensitive` flag is computed only from its
   * single top structural candidate (`structuralDetail`, the
   * highest-|rho| structural metric), and is `true` only when *that one*
   * metric's name starts with `weightedInDegree:` — never from
   * `finding.qualifyingMetrics` as a whole, which can independently list
   * several `feature`-kind metrics at once (`scripts/analysis/explain.py`'s
   * `qualifying_metrics = linear_candidates + structural_candidates`).
   * Flagging every `feature`-kind qualifying metric whenever
   * `definitionSensitive` is true (an earlier version of this file did)
   * would mislabel any other qualifying structural feature (e.g.
   * `reciprocity`) with a disclosure link that says nothing about it —
   * round-2 dual review, Important. Restricting the flag to this same name
   * family keeps it sound even if a future re-run qualifies more than one
   * structural feature at once.
   */
  const isWeightedInDegreeFeature = (metric: NullExplanationQualifyingMetric): boolean =>
    metric.kind === 'feature' && metric.name.startsWith('weightedInDegree:');

  interface QualifyingMetricLine {
    key: string;
    text: string;
    /** True only for the `weightedInDegree:*` qualifying metric, and only when the finding's own `definitionSensitive` flag is set — see `isWeightedInDegreeFeature`'s doc comment. */
    sensitive: boolean;
  }

  const qualifyingMetricLines = $derived<QualifyingMetricLine[]>(
    nullExplanation?.status === 'ok'
      ? nullExplanation.data.finding.qualifyingMetrics.map((metric, index) => ({
          key: `${metric.kind}:${metric.name}:${index}`,
          text: `${describeQualifyingMetric(metric)} (ρ = ${metric.spearman.toFixed(3)})`,
          sensitive: nullExplanation.data.finding.definitionSensitive && isWeightedInDegreeFeature(metric)
        }))
      : []
  );

  /**
   * Data-driven lead sentence (thermo review I1 fix): replaces a verbatim
   * paste of `explain.py`'s own report-writing `finding.summarySentence`
   * (raw metric IDs, an ASCII `--`, `rho=0.467`) — written for a technical
   * report, not a product UI — which was immediately followed by
   * `qualifyingMetricLines` restating the same two findings in plain words.
   * This short lead sentence carries only the count, and the bullets below
   * (already correct, already plain-language) carry every specific.
   */
  const leadSentence = $derived(
    `Biological's low score lines up with ${qualifyingMetricLines.length} metric${qualifyingMetricLines.length === 1 ? '' : 's'} that fall outside the range seen across the graph's ${rewiringCount} rewired versions:`
  );

  /**
   * States the mirrored decoder-convention check's result (`variants.flipBoth`,
   * the one required re-scoring with thrust and yaw signs both flipped)
   * against the *un-mirrored* baseline percentile (`baselinePercentile`,
   * threaded in from `rewiringNull.data.bioPercentile` — the caller only
   * ever passes this once `rewiringNull.status === 'ok'`).
   * Round-2 dual review, Important: an earlier version compared only the
   * mirrored value against a hard-coded "0", so the "still" wording was true
   * for the current shipped data by coincidence, not because it was actually
   * derived from the baseline — a re-run where the baseline itself moved
   * while the mirrored value stayed at 0 would have rendered a false
   * "still". Both sides are now read from their own verified artifact.
   *
   * Compares the *formatted* labels, not the raw floats, for "same
   * percentile" (thermo-maintainability review, carried-over correctness
   * Suggestion): two independently-produced artifacts' floats can differ in
   * a way invisible at the one-decimal precision actually shown, so a bare
   * `===` could report "moves… to…" between two values that render
   * identically, or vice versa a hair of float noise could flip which
   * sentence a bit-identical-looking rerun gets.
   */
  const mirroredDecoderClause = $derived(
    nullExplanation?.status === 'ok'
      ? (() => {
          const mirrored = nullExplanation.data.variants.flipBoth.bioPercentile;
          const baselineLabel = formatPercentile(baselinePercentile);
          const mirroredLabel = formatPercentile(mirrored);
          if (baselinePercentile <= 0 && mirrored <= 0) {
            return `Mirroring the decoder's thrust and yaw signs still leaves biological at the bottom of the null distribution (${mirroredLabel}).`;
          }
          return mirroredLabel === baselineLabel
            ? `Mirroring the decoder's thrust and yaw signs leaves biological at the same ${mirroredLabel} of the null distribution.`
            : `Mirroring the decoder's thrust and yaw signs moves biological from the ${baselineLabel} to the ${mirroredLabel} of the null distribution.`;
        })()
      : ''
  );

  /**
   * States the regime-check outcome (`finding.regimeInvalid`) in one clause.
   * The passed-gate wording (thermo review S2 fix) says "applicable to this
   * model's dynamics," matching `docs/null-explanation-report.md`'s own
   * careful phrasing — the report explicitly does not call the analysis
   * "valid," only that the regime gate "licenses treating the linear
   * analysis as applicable"; "valid" reads to a lay audience as "the
   * finding above is correct," which the regime gate alone does not
   * establish. The failed-gate branch is unchanged — it was already
   * accurately worded.
   */
  const regimeClause = $derived(
    nullExplanation?.status === 'ok'
      ? nullExplanation.data.finding.regimeInvalid
        ? 'The linear-regime check failed, so the linear-transfer analysis is reported as regime-invalid (inconclusive).'
        : "The linear-regime check passed, so the linear-transfer analysis above is treated as applicable to this model's dynamics."
      : ''
  );

  /**
   * WP4 of `.agents/plans/pathway-interventions`: the tested-outcome
   * sentence, templated from the verified artifact's own predeclared
   * category (`00-overview.md`'s vocabulary, stated mechanically — this
   * never reads as a stronger or weaker claim than the category itself
   * licenses). One template per authored category, each carrying the
   * "net effect of the accepted swap set, not a single-edge effect"
   * framing in its own wording so a reader never mistakes this for a
   * single-edge causal claim.
   */
  const PATHWAY_AUTHORED_CLAUSE: Record<PathwayInterventionsAuthoredCategory, string> = {
    'pathway-supported':
      "the pathway-supported category holds: the accepted swap set's net effect outperforms both the unrestricted (C) and class-matched (M) random controls",
    'edge-class-effect':
      "the edge-class-effect category holds: the accepted swap set's net effect outperforms the unrestricted (C) control but not the class-matched (M) control — any edge of this class helps about equally",
    'generic-rewiring-effect':
      "the generic-rewiring-effect category holds: the accepted swap set's net effect does not outperform the unrestricted (C) control — any perturbation of this size helps about equally",
    'not-supported': "the not-supported category holds: the accepted swap set's net effect does not clear the null's 25th percentile"
  };

  /**
   * One template per trained category — see
   * `scripts/null/intervention-report-trained.ts`'s own doc comment for why
   * `'no-specific-effect'` is not a renamed authored category (it is the
   * deliberate merge of `'generic-rewiring-effect'`/`'not-supported'` for
   * the trained side, where this study's predeclared rules cannot decide
   * that finer split). Used only when `trainedRobust` is true — see
   * `pathwayTrainedClause` below for the non-robust wording.
   */
  const PATHWAY_TRAINED_CLAUSE: Record<PathwayInterventionsTrainedCategory, string> = {
    'pathway-supported': 'P also outperforms both freshly-trained control arms across all three trainer seeds tested',
    'edge-class-effect': 'P outperforms the freshly-trained unrestricted (C) arm but not the class-matched (M) arm, across all three trainer seeds tested',
    'no-specific-effect': 'P shows no advantage over either freshly-trained control arm, across all three trainer seeds tested'
  };

  const pathwayChannelSpecificClause = $derived(
    pathwayInterventions?.status === 'ok'
      ? pathwayInterventions.data.authored.channelSpecific
        ? 'the channel-specific modifier holds (Q, using only clearance-channel sources, clears its own class-matched control)'
        : 'the channel-specific modifier does not hold'
      : ''
  );

  const pathwayTrainedClause = $derived(
    pathwayInterventions?.status === 'ok'
      ? pathwayInterventions.data.trained.trainedRobust
        ? `Under trained readouts, ${PATHWAY_TRAINED_CLAUSE[pathwayInterventions.data.trained.perSeedCategory['101']]} (robust)`
        : // Non-robust: state each seed's own category rather than only "do
          // not agree" (thermo review, Suggestion — the per-seed data is
          // already carried on the artifact, and hiding it here would leave
          // a reader with no way to see how the seeds actually diverged).
          `Under trained readouts, the three trainer seeds do not agree on a category (${P_TRAINER_SEEDS.map((seed) => `seed ${seed}: ${pathwayInterventions.data.trained.perSeedCategory[seed]}`).join(', ')}; not robust)`
      : ''
  );

  const PATHWAY_INTERVENTIONS_REPORT_URL = githubDocUrl('pathway-interventions-report.md');
</script>

{#if nullExplanation?.status === 'ok'}
  {@const explanation = nullExplanation.data}
  <div class="detail-box null-explanation-detail">
    <h4>What biological's low score is associated with (under this model)</h4>
    <p>{leadSentence}</p>
    {#if qualifyingMetricLines.length > 0}
      <ul class="metric-list">
        {#each qualifyingMetricLines as line (line.key)}
          <li>
            {line.text}
            {#if line.sensitive}
              — <a href={NULL_EXPLANATION_DISCLOSURE_URL} target="_blank" rel="noreferrer">definition-sensitive, see disclosure</a>
            {/if}
          </li>
        {/each}
      </ul>
    {/if}
    <p>{mirroredDecoderClause} {regimeClause}</p>
    <p class="disclaimer">
      ρ is the rank correlation between a metric and score across the {rewiringCount} rewirings. This is a descriptive association within this model, not a cause. "Authored" means a fixed, hand-written decoder — not biology and not trained.
    </p>
    <ul class="links">
      <li><a href={NULL_EXPLANATION_REPORT_URL} target="_blank" rel="noreferrer">Full explanation report</a></li>
    </ul>
  </div>
{:else if nullExplanation?.status === 'unavailable'}
  <!-- A fetch/network failure or an unexpected runtime error — not a
       claim about the artifact's integrity, so this must not say
       "failed verification" (mirrors `rewiringNull.status ===
       'unavailable'` in `LedgerPanel.svelte`). -->
  <p class="error-message">
    Explanation could not be loaded: {nullExplanation.reason}
  </p>
{:else if nullExplanation?.status === 'invalid'}
  <p class="error-message">
    Explanation failed verification: {nullExplanation.reason}
  </p>
{/if}

<!-- WP4 of `.agents/plans/pathway-interventions`: the tested-outcome
     sentence, in its own box below the explanation note. `pathwayInterventions`
     is a wholly separate load from `nullExplanation` above (its own
     artifact, its own manifest entry, its own cross-checks against the
     manifest rather than against any live-loaded `NullExplanationLoadResult`)
     — this block is therefore a top-level sibling of the `{#if
     nullExplanation...}` block above, not nested inside it, so it renders
     independently of whatever `nullExplanation` itself resolved to
     (thermo-methodology review, Important — an earlier version nested this
     entirely inside `nullExplanation?.status === 'ok'`, so a
     failed/loading/absent explanation silently hid an otherwise
     successfully verified pathway-interventions result too, contradicting
     both `controller.ts`'s "attempted unconditionally, independent of
     whichever status the explanation load itself resolved to" design and
     this component's own doc comment). `'missing'` hides this block
     entirely (nothing was ever shipped); `'unavailable'` and `'invalid'`
     each show their own honestly-worded message, mirroring every other
     sidecar artifact in this panel. -->
{#if pathwayInterventions?.status === 'ok'}
  <!-- `pathway-interventions-detail`, not `null-explanation-detail` (thermo-
       methodology review, Important fix): this box is a sibling of the
       explanation box above, not its content, and now that both can be on
       screen at once (the whole point of the independence fix) a shared
       class would make `.null-explanation-detail` match two elements —
       ambiguous for both e2e locators and any future query. Its own rule
       below (`.pathway-interventions-detail p`) reuses the identical `p`
       styling `.null-explanation-detail p` already declares. -->
  <div class="detail-box pathway-interventions-detail">
    <p>
      Tested under this model, with the authored (hand-written) decoder: {PATHWAY_AUTHORED_CLAUSE[pathwayInterventions.data.authored.category]}, and {pathwayChannelSpecificClause}.
    </p>
    <p>{pathwayTrainedClause}.</p>
    <!-- Restored (fix-verification review finding: dropped when the sentence
         was tightened from three sentences to two, contradicting the commit
         message that landed that change). The authored-decoder result above
         is one decoder's result, not the study's overall finding -- the
         trained-readout sentence right above this one can (and here does)
         disagree with it. -->
    <p class="disclaimer">
      This authored-decoder result is bound to the hand-written decoder; it is not necessarily the overall finding.
    </p>
    <ul class="links">
      <li><a href={PATHWAY_INTERVENTIONS_REPORT_URL} target="_blank" rel="noreferrer">Intervention report</a></li>
    </ul>
  </div>
{:else if pathwayInterventions?.status === 'unavailable'}
  <p class="error-message">
    Intervention test could not be loaded: {pathwayInterventions.reason}
  </p>
{:else if pathwayInterventions?.status === 'invalid'}
  <p class="error-message">
    Intervention test failed verification: {pathwayInterventions.reason}
  </p>
{/if}

<style>
  /* `.detail-box` (margin/padding/border/border-radius/background) is
     shared with `LedgerPanel.svelte`'s `.trained-readout-detail` via
     `src/app.css` (thermo-maintainability review, carried-over Suggestion —
     Svelte's per-component style scoping means neither component could
     otherwise reach a `<style>` block defined in the other). This class
     keeps only this block's own content-specific rules. */
  .null-explanation-detail h4 {
    margin: 0 0 0.4rem;
    color: #cbd8e7;
    font-size: 0.78rem;
    font-weight: 700;
  }

  .null-explanation-detail p,
  .pathway-interventions-detail p {
    margin: 0.5rem 0;
    color: #cbd8e7;
    font-size: 0.8rem;
    line-height: 1.5;
  }

  .disclaimer {
    margin: 0.9rem 0;
    color: #9aacc2;
    font-size: 0.8rem;
    line-height: 1.5;
  }

  /* Overrides the global `.ledger ul`/`.ledger li` term/label row styling
     (`src/app.css`) — that styling is for the top-level ledger rows and the
     `.links` lists, not for a plain, bulleted list of qualifying metrics. */
  .null-explanation-detail .metric-list {
    margin: 0.5rem 0;
    padding-left: 1.2rem;
    list-style: disc;
  }

  .null-explanation-detail .metric-list li {
    display: list-item;
    border: none;
    padding: 0.15rem 0;
    font-size: 0.8rem;
    color: #cbd8e7;
    text-align: left;
  }

  .links {
    margin: 0;
    padding: 0;
    list-style: none;
    display: grid;
    gap: 0.4rem;
  }

  .links li {
    border: none;
    padding: 0;
    font-size: 0.8rem;
    color: #9aacc2;
  }

  .links a {
    color: #79d8d0;
  }

  .error-message {
    margin: 0.9rem 0;
    padding: 0.6rem 0.75rem;
    border: 1px solid #ef476f;
    border-radius: 0.4rem;
    color: #ffd7de;
    background: rgb(239 71 111 / 12%);
    font-size: 0.8rem;
  }
</style>
