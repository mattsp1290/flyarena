<script lang="ts">
  import type { ArenaManifest, RewiringNullLoadResult, TrainedReadoutLoadResult } from '../experiment/assets';
  import type { DecoderKind } from '../worker/protocol';
  import NullHistogram from './NullHistogram.svelte';

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
  }

  let { manifest, decoder, trainedReadout, rewiringNull }: Props = $props();

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
      label: decoder === 'trained' ? 'Authored (encoder) + Trained (readout, per-arm)' : 'Authored'
    },
    { term: '3D presentation', label: 'Synthetic' },
    // WP3 (anatomical activity view): neuron positions come from the MaleCNS
    // soma annotation sidecar (`docs/data-provenance.md`); the colors drawn
    // from them are the authored dynamics' live output, not a measurement.
    { term: 'Neuron positions', label: 'Measured' },
    { term: 'Displayed neural activity', label: 'Computed' },
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
    {
      term: 'Topology null distribution',
      label:
        rewiringNull === undefined
          ? 'Loading…'
          : rewiringNull.status === 'ok'
            ? 'Computed (offline)'
            : rewiringNull.status === 'missing'
              ? 'Computed (offline) — not shipped'
              : 'Computed (offline) — unavailable'
    }
  ]);

  const reportUrl = $derived(`${import.meta.env.BASE_URL}data/trained-readout-v1.report.json`);
  const readoutManifestUrl = $derived(`${import.meta.env.BASE_URL}data/trained-readout-v1.manifest.json`);
  /**
   * A plain GitHub blob link to the human-readable report, since `docs/` is
   * not part of the deployed static site (only `public/` is served) — a
   * relative `docs/trained-readout-report.md` link would 404 under any base
   * path. The JSON links above are the base-path-safe, always-resolvable
   * links; this is offered alongside them for the prose version.
   */
  const GITHUB_REPORT_URL = 'https://github.com/mattsp1290/flyarena/blob/main/docs/trained-readout-report.md';

  /**
   * WP4's counterpart to `reportUrl` above, for the rewiring-null artifact's
   * own JSON — `NullHistogram.svelte`'s figcaption links the human-readable
   * report. Built from `manifest.rewiringNull.artifact` (the same field
   * `assets.ts#loadRewiringNull` itself fetches), not a hardcoded filename
   * (dual review, Suggestion) — the manifest is the single source of truth
   * for this artifact's name, already available here as a prop. Falls back
   * to the conventional filename only for the (impossible in practice) case
   * where this section renders `rewiringNull.status === 'ok'` from a
   * `manifest` that is somehow `undefined` at the same tick.
   */
  const rewiringNullJsonUrl = $derived(
    `${import.meta.env.BASE_URL}data/${manifest?.rewiringNull?.artifact ?? 'rewiring-null-v1.json'}`
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

  {#if rewiringNull && rewiringNull.status !== 'missing'}
    <!-- One block, not two separately-conditioned `{#if}`s (dual review,
         Suggestion — the earlier version had to keep the heading's own
         `{#if}` in sync with this body's by hand). This also lets
         `rewiringNull.status` narrow inside the `{#if}/{:else}` below
         without `?.`, since the outer condition already excludes `undefined`
         and `'missing'`. -->
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
