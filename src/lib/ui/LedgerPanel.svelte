<script lang="ts">
  import type { ArenaManifest, TrainedReadoutLoadResult } from '../experiment/assets';
  import type { RewiringNullLoadResult } from '../experiment/rewiringNull';
  import type { NullExplanationLoadResult, NullExplanationQualifyingMetric } from '../experiment/nullExplanation';
  import type { DecoderKind } from '../worker/protocol';
  import NullHistogram from './NullHistogram.svelte';
  import { githubDocUrl } from './links';

  /**
   * The model ledger vocabulary (`docs/model-ledger.md`) plus provenance
   * links (WP6 item 7). Never says "brain emulation," and never attributes
   * the authored encoder/decoder to biology — see `docs/data-provenance.md`
   * for the full sourcing detail this panel links out to.
   *
   * The trained-readout row/detail block (WP6) is visible regardless of
   * `decoder` — it documents what the artifact *is*, not merely what is
   * currently selected — matching
   * `.agents/plans/trained-readout/06-browser-integration.md`'s "Visible in
   * both modes" requirement.
   */
  interface Props {
    manifest: ArenaManifest | undefined;
    /** Which decoder both agents currently share; scopes the "Sensory encoder and action decoder" row's label to the active mode. */
    decoder: DecoderKind;
    /** `undefined` while `ExperimentController#initialize()`'s trained-readout step has not yet resolved. */
    trainedReadout: TrainedReadoutLoadResult | undefined;
    /** `undefined` while `ExperimentController#initialize()`'s rewiring-null load (WP4) has not yet resolved. */
    rewiringNull: RewiringNullLoadResult | undefined;
    /** `undefined` while `ExperimentController#initialize()`'s null-explanation load (WP4 of `.agents/plans/null-explanation`) has not yet resolved. */
    nullExplanation: NullExplanationLoadResult | undefined;
  }

  let { manifest, decoder, trainedReadout, rewiringNull, nullExplanation }: Props = $props();

  const LEDGER_ROWS = $derived<readonly { term: string; label: string }[]>([
    { term: 'Graph topology', label: 'Measured' },
    { term: 'Biological annotations', label: 'Annotated' },
    { term: 'Network dynamics', label: 'Authored / literature-derived' },
    { term: 'Global parameters', label: 'Calibrated' },
    // Scoped to the current mode (docs/model-ledger.md's Trained-mode
    // scoping): in Authored mode the decoder is identical across arms; in
    // Trained mode each arm's readout is its own trained weights (matched
    // architecture/parameter count — see the Readout row below), while the
    // encoder and decodeAction stay Authored and identical either way.
    {
      term: 'Sensory encoder and action decoder',
      // "(hand-written)" gloss (thermo review I1): "Authored" alone reads as
      // a neutral engineering label, but readers over-interpret bare
      // "authored" numbers elsewhere in the product (see `NullHistogram.svelte`'s
      // `CAPTION_DISCLAIMER`) — this row is the one place that word is
      // *defined* in-product, without leaving the app.
      label:
        decoder === 'trained'
          ? 'Authored (hand-written, encoder) + Trained (readout, per-arm)'
          : 'Authored (hand-written)'
    },
    { term: '3D presentation', label: 'Synthetic' },
    // WP3 (anatomical activity view): neuron positions come from the MaleCNS
    // soma annotation sidecar (`docs/data-provenance.md`); the colors drawn
    // from them are the authored dynamics' live output, not a measurement.
    { term: 'Neuron positions', label: 'Measured' },
    { term: 'Displayed neural activity', label: 'Computed' },
    // WP3 of `.agents/plans/lesion-atlas`: the activity view's optional
    // "Lesion effect (offline)" color mode paints static per-neuron colors
    // from the offline-computed single-neuron lesion atlas
    // (`docs/lesion-atlas-report.md`) — always offline/static regardless of
    // whether this particular session happens to load it successfully (the
    // per-session load status is instead disclosed in-place by
    // `ActivityPanel.svelte`'s own disabled-reason hint, the same pattern
    // `loadPositions`'s toggle-disable already uses).
    { term: 'Lesion effect map', label: 'Computed (offline)' },
    {
      term: 'Readout (trained mode)',
      label:
        trainedReadout === undefined
          ? 'Loading…'
          : trainedReadout.status === 'ok'
            ? 'Trained (offline)'
            : 'Trained (offline) — unavailable'
    },
    // WP4 (`docs/model-ledger.md`'s new row): descriptive scores of
    // degree-preserving rewirings under the authored decoder, computed
    // offline — never a biological measurement.
    //
    // `'absent'`/`'unavailable'`/`'invalid'` are worded differently (thermo
    // review, Suggestion): "not shipped" is only true when the manifest has
    // no entry at all; a fetch/network failure ("could not load") is not a
    // claim about the artifact's integrity the way "failed verification"
    // is, so this row (and the section below) must not conflate them.
    {
      term: 'Topology null distribution',
      label:
        rewiringNull === undefined
          ? 'Loading…'
          : rewiringNull.status === 'ok'
            ? 'Computed (offline)'
            : rewiringNull.status === 'absent'
              ? 'Computed (offline) — not shipped'
              : rewiringNull.status === 'unavailable'
                ? 'Computed (offline) — could not be loaded'
                : 'Computed (offline) — failed verification'
    },
    // WP4 of `.agents/plans/null-explanation`: a static label, like "Lesion
    // effect map" above rather than "Topology null distribution"'s
    // status-driven one — this row documents what the artifact *is*, not
    // this session's load outcome; a missing/failed-verification note is
    // instead disclosed in place, next to the histogram itself (see the
    // section below), the same "row stays static, per-session status shown
    // in place" split "Lesion effect map"'s own doc comment explains.
    { term: 'Null-result explanation', label: 'Computed (offline)' }
  ]);

  const reportUrl = $derived(`${import.meta.env.BASE_URL}data/trained-readout-v1.report.json`);
  const readoutManifestUrl = $derived(`${import.meta.env.BASE_URL}data/trained-readout-v1.manifest.json`);
  /**
   * A plain GitHub blob link to the human-readable report, since `docs/` is
   * not part of the deployed static site (only `public/` is served) — a
   * relative `docs/trained-readout-report.md` link would 404 under any base
   * path. The JSON links above are the base-path-safe, always-resolvable
   * links; this is offered alongside them for the prose version. Built via
   * `./links.ts#githubDocUrl`, shared with `NullHistogram.svelte`'s own
   * report link (thermo-maintainability review S3).
   */
  const GITHUB_REPORT_URL = githubDocUrl('trained-readout-report.md');

  /**
   * WP4's counterpart to `reportUrl` above, for the rewiring-null artifact's
   * own JSON — `NullHistogram.svelte`'s figcaption links the human-readable
   * report. Built from `manifest.rewiringNull.artifact` (the same field
   * `rewiringNull.ts#loadRewiringNull` itself fetches), not a hardcoded filename
   * (dual review, Suggestion) — the manifest is the single source of truth
   * for this artifact's name, already available here as a prop. Falls back
   * to the conventional filename only for the (impossible in practice) case
   * where this section renders `rewiringNull.status === 'ok'` from a
   * `manifest` that is somehow `undefined` at the same tick.
   */
  const rewiringNullJsonUrl = $derived(
    `${import.meta.env.BASE_URL}data/${manifest?.rewiringNull?.artifact ?? 'rewiring-null-v1.json'}`
  );

  /** WP4 of `.agents/plans/null-explanation`'s report link, built the same way `NullHistogram.svelte`'s own `GITHUB_REPORT_URL` is (`./links.ts#githubDocUrl`). */
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
   * States the mirrored decoder-convention check's result (`variants.flipBoth`,
   * the one required re-scoring with thrust and yaw signs both flipped)
   * against the *un-mirrored* baseline percentile
   * (`rewiringNull.data.bioPercentile` — already in scope, since this note
   * only ever renders inside `{#if rewiringNull.status === 'ok'}` below).
   * Round-2 dual review, Important: an earlier version compared only the
   * mirrored value against a hard-coded "0", so the "still" wording was true
   * for the current shipped data by coincidence, not because it was actually
   * derived from the baseline — a re-run where the baseline itself moved
   * while the mirrored value stayed at 0 would have rendered a false
   * "still". Both sides are now read from their own verified artifact.
   */
  const mirroredDecoderClause = $derived(
    nullExplanation?.status === 'ok' && rewiringNull?.status === 'ok'
      ? (() => {
          const percentileLabel = (value: number): string => `${(value * 100).toFixed(1)}th percentile`;
          const baseline = rewiringNull.data.bioPercentile;
          const mirrored = nullExplanation.data.variants.flipBoth.bioPercentile;
          if (baseline <= 0 && mirrored <= 0) {
            return "Mirroring the decoder's thrust and yaw signs still leaves biological at the bottom of the null distribution (0th percentile).";
          }
          return mirrored === baseline
            ? `Mirroring the decoder's thrust and yaw signs leaves biological at the same ${percentileLabel(mirrored)} of the null distribution.`
            : `Mirroring the decoder's thrust and yaw signs moves biological from the ${percentileLabel(baseline)} to the ${percentileLabel(mirrored)} of the null distribution.`;
        })()
      : ''
  );

  /** States the regime-check outcome (`finding.regimeInvalid`) in one clause. */
  const regimeClause = $derived(
    nullExplanation?.status === 'ok'
      ? nullExplanation.data.finding.regimeInvalid
        ? 'The linear-regime check failed, so the linear-transfer analysis is reported as regime-invalid (inconclusive).'
        : 'The linear-regime check passed, so the linear-transfer analysis is treated as valid under this model.'
      : ''
  );
</script>

<section class="panel ledger" aria-labelledby="ledger-heading">
  <div class="section-heading">
    <h2 id="ledger-heading">Model ledger</h2>
    <span>Inspectability first</span>
  </div>
  <ul>
    {#each LEDGER_ROWS as row (row.term)}
      <li><strong>{row.term}</strong><span>{row.label}</span></li>
    {/each}
  </ul>

  <p class="disclaimer">
    This is a small, descriptive proof-of-concept experiment. The graph
    topology below comes from a real connectome reconstruction; the
    dynamics and sensory encoder are authored engineering choices applied
    identically to every arm, not measurements of biological behavior. In
    Authored mode the action decoder is identical across arms too; in
    Trained mode each arm uses its own trained readout weights, with a
    matched architecture, parameter count, and training procedure (see
    "Readout (trained mode)" below) — the decoder is never a measurement of
    biological behavior in either mode.
  </p>

  {#if trainedReadout?.status === 'ok'}
    {@const readout = trainedReadout}
    <div class="trained-readout-detail">
      <p>
        Weights optimized offline by the cross-entropy method (CEM) against
        the arena score — an engineering artifact, not biology, with
        identical architecture and parameter count across arms. GPU training
        reruns are not bit-identical to the shipped weights (informational
        only; the shipped, hash-pinned artifact below is the authoritative
        result the browser always loads).
      </p>
      <dl>
        <div><dt>Artifact sha256</dt><dd>{readout.manifest.artifactSha256.slice(0, 12)}…</dd></div>
        <div><dt>Parameter count (per arm)</dt><dd>{readout.manifest.parameterCount}</dd></div>
        <div><dt>Architecture</dt><dd>{readout.manifest.D} → {readout.manifest.H} → 3</dd></div>
      </dl>
      <p class="disclaimer">
        Headline results measured single-agent (opponent parked); side-by-side
        results in report.
      </p>
      <ul class="links">
        <li><a href={reportUrl} target="_blank" rel="noreferrer">Trained-readout report (JSON)</a></li>
        <li><a href={readoutManifestUrl} target="_blank" rel="noreferrer">Trained-readout artifact manifest (JSON)</a></li>
        <li><a href={GITHUB_REPORT_URL} target="_blank" rel="noreferrer">Trained-readout report (Markdown, GitHub)</a></li>
      </ul>
    </div>
  {:else if trainedReadout?.status === 'unavailable'}
    <!-- No `role="status"`: the app header's own status span already owns
         that role page-wide (see `ExperimentPanel.svelte`'s matching hint
         for the same reasoning). -->
    <p class="error-message">
      Trained readout artifact failed verification: {trainedReadout.reason}
    </p>
  {/if}

  {#if manifest}
    <dl>
      <div><dt>Source dataset</dt><dd>{manifest.sourceDataset}</dd></div>
      <div><dt>License</dt><dd>{manifest.license}</dd></div>
      <div><dt>Neurons / edges (biological)</dt><dd>{manifest.neuronCount} / {manifest.edgeCount}</dd></div>
    </dl>
  {/if}

  {#if rewiringNull && rewiringNull.status !== 'absent'}
    <!-- One block, not two separately-conditioned `{#if}`s (dual review,
         Suggestion — the earlier version had to keep the heading's own
         `{#if}` in sync with this body's by hand). This also lets
         `rewiringNull.status` narrow inside the `{#if}/{:else if}` below
         without `?.`, since the outer condition already excludes `undefined`
         and `'absent'` (nothing was ever shipped, so there is nothing to say
         here — `'unavailable'`/`'invalid'` are genuine attempts that failed
         and still get a heading plus an honestly-worded message). -->
    <h3>Topology null distribution</h3>
    {#if rewiringNull.status === 'ok'}
      <!-- `NullHistogram`'s own `<figcaption>` already links "Full
           rewiring-null report" (the bean's non-negotiable) — this list adds
           only the machine-readable JSON, matching the "JSON is linked too"
           instruction without duplicating the same link text/target twice on
           one page. -->
      <NullHistogram data={rewiringNull.data} />
      <ul class="links">
        <li><a href={rewiringNullJsonUrl} target="_blank" rel="noreferrer">Rewiring-null result (JSON)</a></li>
      </ul>

      <!-- WP4 of `.agents/plans/null-explanation`: the finding note that
           explains, under this model only, why biological scored where it
           did above. `'missing'` hides this whole block (the bean's own
           contract) — the histogram above still renders on its own,
           unaffected. `'unavailable'`/`'invalid'` each show their own
           honestly-worded message instead of the note, mirroring
           `rewiringNull.status`'s own three-way split just below. -->
      {#if nullExplanation?.status === 'ok'}
        {@const explanation = nullExplanation.data}
        <div class="null-explanation-detail">
          <h4>Why biological scores low (under this model)</h4>
          <p>{explanation.finding.summarySentence}</p>
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
            This is a descriptive association within this model, not a cause. "Authored" means a fixed, hand-written decoder — not biology and not trained.
          </p>
          <ul class="links">
            <li><a href={NULL_EXPLANATION_REPORT_URL} target="_blank" rel="noreferrer">Full explanation report</a></li>
          </ul>
        </div>
      {:else if nullExplanation?.status === 'unavailable'}
        <!-- A fetch/network failure or an unexpected runtime error — not a
             claim about the artifact's integrity, so this must not say
             "failed verification" (mirrors `rewiringNull.status ===
             'unavailable'` just below). -->
        <p class="error-message">
          Explanation could not be loaded: {nullExplanation.reason}
        </p>
      {:else if nullExplanation?.status === 'invalid'}
        <p class="error-message">
          Explanation failed verification: {nullExplanation.reason}
        </p>
      {/if}
    {:else if rewiringNull.status === 'unavailable'}
      <!-- A fetch/network failure or an unexpected runtime error
           (`controller.ts`'s leading `.catch`) — not a claim about the
           artifact's integrity, so this must not say "failed verification"
           (thermo review, Suggestion). -->
      <p class="error-message">
        Null-distribution result could not be loaded: {rewiringNull.reason}
      </p>
    {:else}
      <p class="error-message">
        Null-distribution result failed verification: {rewiringNull.reason}
      </p>
    {/if}
  {/if}

  <h3>Provenance and licensing</h3>
  <ul class="links">
    <li><a href={`${import.meta.env.BASE_URL}data/malecns-arena-v1.manifest.json`} target="_blank" rel="noreferrer">Compiled artifact manifest (JSON)</a></li>
    <li><a href={`${import.meta.env.BASE_URL}data/malecns-arena-v1.ledger.json`} target="_blank" rel="noreferrer">Compiler ledger (JSON)</a></li>
    <li><a href="https://male-cns.janelia.org/" target="_blank" rel="noreferrer">Male CNS Connectome project page</a></li>
    <li>
      <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noreferrer">CC BY 4.0</a>
      — Janelia FlyEM Project (HHMI), MRC Laboratory of Molecular Biology, Google Research
    </li>
  </ul>
</section>

<style>
  .disclaimer {
    margin: 0.9rem 0;
    color: #9aacc2;
    font-size: 0.8rem;
    line-height: 1.5;
  }

  h3 {
    margin: 0.9rem 0 0.4rem;
    color: #cbd8e7;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
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

  .trained-readout-detail {
    margin: 0.9rem 0;
    padding: 0.65rem 0.75rem;
    border: 1px solid #304355;
    border-radius: 0.4rem;
    background: rgb(121 216 208 / 6%);
  }

  .trained-readout-detail dl {
    margin: 0.5rem 0;
    display: grid;
    gap: 0.25rem;
  }

  .trained-readout-detail dl div {
    display: flex;
    justify-content: space-between;
    gap: 0.6rem;
    font-size: 0.8rem;
  }

  .trained-readout-detail dt {
    color: #9aacc2;
  }

  .trained-readout-detail dd {
    margin: 0;
    color: #edf4ff;
    text-align: right;
  }

  /* Mirrors `.trained-readout-detail`'s own box; a distinct class only
     because the two blocks are never both shown at once, so sharing one
     name would be misleading about which content it wraps. */
  .null-explanation-detail {
    margin: 0.9rem 0;
    padding: 0.65rem 0.75rem;
    border: 1px solid #304355;
    border-radius: 0.4rem;
    background: rgb(121 216 208 / 6%);
  }

  .null-explanation-detail h4 {
    margin: 0 0 0.4rem;
    color: #cbd8e7;
    font-size: 0.78rem;
    font-weight: 700;
  }

  .null-explanation-detail p {
    margin: 0.5rem 0;
    color: #cbd8e7;
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
