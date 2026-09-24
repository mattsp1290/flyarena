<script lang="ts">
  import type { ArenaManifest } from '../experiment/assets';
  import { publicAssetUrl } from '../paths';

  /**
   * The model ledger vocabulary (`docs/model-ledger.md`) plus provenance
   * links (WP6 item 7). Never says "brain emulation," and never attributes
   * the authored encoder/decoder to biology — see `docs/data-provenance.md`
   * for the full sourcing detail this panel links out to.
   */
  interface Props {
    manifest: ArenaManifest | undefined;
  }

  let { manifest }: Props = $props();

  const LEDGER_ROWS: readonly { term: string; label: string }[] = [
    { term: 'Graph topology', label: 'Measured' },
    { term: 'Biological annotations', label: 'Annotated' },
    { term: 'Network dynamics', label: 'Authored / literature-derived' },
    { term: 'Global parameters', label: 'Calibrated' },
    { term: 'Sensory encoder and action decoder', label: 'Authored' },
    { term: '3D presentation', label: 'Synthetic' }
  ];
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
    dynamics, sensory encoder, and action decoder are authored engineering
    choices applied identically to every arm, not measurements of biological
    behavior.
  </p>

  {#if manifest}
    <dl>
      <div><dt>Source dataset</dt><dd>{manifest.sourceDataset}</dd></div>
      <div><dt>License</dt><dd>{manifest.license}</dd></div>
      <div><dt>Neurons / edges (biological)</dt><dd>{manifest.neuronCount} / {manifest.edgeCount}</dd></div>
    </dl>
  {/if}

  <h3>Provenance and licensing</h3>
  <ul class="links">
    <li><a href={publicAssetUrl('data/malecns-arena-v1.manifest.json')} target="_blank" rel="noreferrer">Compiled artifact manifest (JSON)</a></li>
    <li><a href={publicAssetUrl('data/malecns-arena-v1.ledger.json')} target="_blank" rel="noreferrer">Compiler ledger (JSON)</a></li>
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
</style>
