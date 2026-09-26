<script lang="ts">
  import { onMount } from 'svelte';
  import { loadAtlas } from './assets';
  import { loadRepertoireNull, type RepertoireNullLoadResult } from '../experiment/repertoireNull';
  import type { ArenaManifest } from '../experiment/assets';
  import { githubDocUrl } from '../ui/links';
  import { isRepertoireStale, buildRepertoireStripText } from './repertoireStrip';
  import { COVERAGE_EDGES, TURN_EDGES, HELDOUT_SEEDS, CONTROL_NAMES, average, type Control, type LoadedAtlas } from './types';
  import BehaviorReplay from './BehaviorReplay.svelte';
  import Workbench from '../counterfactual/Workbench.svelte';
  let loaded = $state<LoadedAtlas | null>(null), error = $state(''), selectedId = $state<number | null>(null);
  let repertoireNull = $state<RepertoireNullLoadResult | undefined>(undefined);
  let disposed = false;
  const selected = $derived(loaded?.atlas.cells.find(c => c.id === selectedId));
  const qualityScale = $derived(loaded ? Math.max(1, ...loaded.atlas.cells.map(c => Math.abs(c.quality))) : 1);
  async function load() {
    error = '';
    try {
      const next = await loadAtlas(); if (disposed) return;
      loaded = next; selectedId = next.atlas.cells.reduce((best, cell) => cell.quality > best.quality ? cell : best).id;
    } catch (e) { if (!disposed) error = e instanceof Error ? e.message : 'Unable to load behavior atlas'; }
  }
  /**
   * `03-artifact-and-atlas-strip.md`: "fetch the central manifest
   * independently of `loadAtlas()`, then call `loadRepertoireNull`" — the
   * repertoire strip's own fetch chain, wholly independent of `load()`
   * above (which fetches the *atlas-specific* manifest,
   * `behavior-atlas-v1.manifest.json`, a different file). Its state never
   * delays or fails the atlas load: any failure here only ever sets
   * `repertoireNull` to an honest `unavailable`/`invalid` status, never
   * touches `loaded`/`error` above.
   */
  async function loadRepertoireStrip() {
    const base = `${import.meta.env.BASE_URL}data`;
    try {
      const response = await fetch(`${base}/malecns-arena-v1.manifest.json`);
      if (!response.ok) throw new Error(`Failed to fetch manifest: ${response.status} ${response.statusText}`);
      const manifest = (await response.json()) as ArenaManifest;
      if (disposed) return;
      const result = await loadRepertoireNull(manifest, base);
      if (!disposed) repertoireNull = result;
    } catch (e) {
      if (!disposed) {
        repertoireNull = { status: 'unavailable', reason: e instanceof Error ? e.message : 'Unable to load the manifest' };
      }
    }
  }
  onMount(() => { void load(); void loadRepertoireStrip(); return () => { disposed = true; }; });
  const pct = (n: number) => `${(n * 100).toFixed(0)}%`;
  const controlLabels: Record<Control, string> = { biological: 'Intact circuit', disconnected: 'No recurrent edges', silenced: 'Zero readout inputs' };
  const mean = (name: Control) => selected ? average(selected.heldout[name].map(m => m.movementScore)) : 0;
  /**
   * A maintainability review (Important) found the strip's own doc comment
   * claimed to cross-check this study against the manifest's biological
   * graph and rewiring-null distribution, but never against the shipped
   * atlas itself — even though the study's whole premise
   * (`docs/behavior-repertoire-null-report.md`'s Method section) is that it
   * reuses the shipped atlas search exactly. `loadRepertoireNull` cannot
   * make that check itself (the atlas is pinned by its own separate
   * `behavior-atlas-v1.manifest.json`, not by the central manifest that
   * loader cross-checks against — see that function's own doc comment for
   * why the check belongs here instead). This component is the one place
   * that already loads both: `loaded.sha256` (from `load()`, the shipped
   * atlas's own verified sha256) and `repertoireNull.data.sources.atlasSha256`
   * (the atlas this study was computed against). Neither load waits on the
   * other; this is a display-time check only, once both have resolved. The
   * comparison itself is `isRepertoireStale` (`./repertoireStrip.ts`), a
   * plain, DOM-free function a thermo-methodology review (Important) found
   * had zero test coverage at any level -- `tests/unit/repertoire-strip.test.ts`
   * now covers it directly, and `tests/e2e/atlas-repertoire.spec.ts` covers
   * the full stale-atlas render path end-to-end.
   */
  const repertoireStale = $derived(
    loaded !== null && repertoireNull?.status === 'ok' && isRepertoireStale(repertoireNull.data.sources.atlasSha256, loaded.sha256)
  );
  /**
   * `03-artifact-and-atlas-strip.md`'s one-line strip. `null` when there is
   * nothing to show (still loading, `missing`, which hides the line
   * entirely per spec, or `repertoireStale`, which instead renders the same
   * "failed verification" line an `invalid` load would). The sentence
   * itself is built by `buildRepertoireStripText` (`./repertoireStrip.ts`),
   * a plain function extracted out of this component (thermo-
   * maintainability review, Suggestion) so its template logic is unit-
   * tested directly rather than only through a full Playwright run.
   */
  const repertoireStripText = $derived.by(() => {
    const result = repertoireNull;
    if (result?.status !== 'ok' || repertoireStale) return null;
    return buildRepertoireStripText(result.data);
  });
  const REPERTOIRE_REPORT_URL = githubDocUrl('behavior-repertoire-null-report.md');
</script>
<div class="atlas">
  <header><p class="eyebrow">DGX discovery / interactive circuit experiments</p><h1>Find a behavior.<br /><em>Ask what drives it.</em></h1>
    <p>A repertoire of optimized controllers on measured fly connectivity. Select a behavior, inspect its controls, then silence a circuit group and watch its future change.</p>
    <div class="badges"><span>Measured topology</span><span>GPU-optimized decoder</span><span>Fresh browser experiments</span></div>
    <p class="caveat">The dynamics, sensory encoding, arena and objectives are authored. These are behaviors of a model, not evidence of natural fly behavior or biological causality.</p>
  </header>
  {#if error}<section role="alert">{error} <button onclick={load}>Retry atlas loading</button></section>
  {:else if !loaded}<p role="status">Loading and verifying the behavior atlas…</p>
  {:else}
    <div class="discovery">
      <section aria-label="Behavior map" class="panel">
        <p class="eyebrow">01 / Discover</p><h2>{loaded.atlas.cells.length} behaviors, one circuit</h2>
        <p>Columns show arena area visited. Rows show average turning command, from rightward (+) to leftward (−). Color shows discovery score; empty cells were not found in this search.</p>
        <div class="grid" role="group" aria-label="Select a discovered behavior">
          {#each Array.from({ length: 36 }, (_, i) => (5 - Math.floor(i / 6)) * 6 + i % 6) as coordinate}
            {@const cell = loaded.atlas.cells.find(c => c.cell === coordinate)}
            {@const col = coordinate % 6}
            {@const row = Math.floor(coordinate / 6)}
            {@const bounds = `coverage ${pct(COVERAGE_EDGES[col])}–${pct(COVERAGE_EDGES[col + 1])}, turning ${TURN_EDGES[row].toFixed(2)}–${TURN_EDGES[row + 1].toFixed(2)}`}
            {#if cell}
              <button aria-label={`Controller ${cell.id}, ${bounds}, discovery score ${cell.quality.toFixed(2)}`} aria-pressed={cell.id === selectedId}
                style={`background:rgba(66, 180, 151, ${.12 + .6 * Math.max(0, cell.quality) / qualityScale})`} onclick={() => selectedId = cell.id}>
                <span>#{cell.id}</span><strong>{cell.quality.toFixed(1)}</strong>
              </button>
            {:else}<div class="empty" title={`Undiscovered: ${bounds}`} aria-label={`Undiscovered: ${bounds}`}>·</div>{/if}
          {/each}
        </div>
        <div class="axis">{#each COVERAGE_EDGES.slice(0, -1) as edge}<span>{pct(edge)}+</span>{/each}</div>
        <p class="axis-caption">Area visited →</p>
        <!-- No `role="status"`: the DGX sandbox's own run-status region and
             `App.svelte`'s experiment-status span already own that role
             (`LedgerPanel.svelte`'s identical precedent/doc comment) — a
             second `role="status"` element anywhere on the page breaks
             every unscoped `getByRole('status')` locator across the e2e
             suite (confirmed directly: `tests/e2e/subpath.spec.ts`'s own
             `getByRole('status')` started resolving to two elements once
             this strip gained the role). -->
        {#if repertoireStripText}
          <p class="repertoire-strip">
            {repertoireStripText}
            <a href={REPERTOIRE_REPORT_URL} target="_blank" rel="noreferrer">Full report</a>
          </p>
        {:else if repertoireNull?.status === 'unavailable'}
          <p class="repertoire-strip error-message">Repertoire comparison could not be loaded</p>
        {:else if repertoireNull?.status === 'invalid' || repertoireStale}
          <p class="repertoire-strip error-message">Repertoire comparison failed verification</p>
        {/if}
        <details><summary>How this repertoire was discovered</summary>
          <p>MAP-Elites retained the highest movement score within each of 36 behavior cells. {loaded.atlas.source.options.population} candidates × {loaded.atlas.source.options.generations} generations × 8 discovery seeds. The graph and world equations stayed fixed.</p>
          <p>Device: {loaded.atlas.source.runtime.deviceName}. {loaded.atlas.source.runtime.seconds.toFixed(1)} seconds; peak tensor allocation {(loaded.atlas.source.runtime.peakTensorBytes / 1024 ** 2).toFixed(1)} MiB. Canonical TypeScript evaluation determines the displayed cells and scores.</p>
          <p>Held-out seeds are evaluated after selection. Repeated interactive probes are exploratory; descriptive intervals are not corrected for repeated comparisons.</p>
          <p class="hash">Atlas SHA-256: {loaded.sha256}</p>
          <a href={`${import.meta.env.BASE_URL}data/behavior-atlas-v1.json`} download>Download full atlas evidence</a>
        </details>
      </section>
      {#if selected}
        <section class="panel" aria-label="Selected behavior">
          <p class="eyebrow">02 / Inspect</p><h2>Controller #{selected.id}</h2>
          <p>{pct(selected.coverage)} area visited · mean turning {selected.turning.toFixed(3)} · discovery score {selected.quality.toFixed(2)}</p>
          {#key selected.id}<BehaviorReplay frames={selected.replay} seed={HELDOUT_SEEDS[0]} />{/key}
          <table><caption>Mean score on 12 held-out seeds · same readout</caption><thead><tr><th>Condition</th><th>Score</th><th>Δ from intact</th></tr></thead>
            <tbody>{#each CONTROL_NAMES as key}
              {@const value = mean(key)}
              <tr><th>{controlLabels[key]}</th><td>{value.toFixed(2)}</td><td>{(value - mean('biological')).toFixed(2)}</td></tr>
            {/each}</tbody>
          </table>
          <p>Authored decoder reference: {average(loaded.atlas.authored.map(m => m.movementScore)).toFixed(2)}. These controls expose decoder bias: behavior that survives zero inputs cannot be attributed to the circuit.</p>
        </section>
      {/if}
    </div>
    {#if selected}
      <div class="probe-intro"><p class="eyebrow">03 / Intervene</p><h2>Put controller #{selected.id} to the test</h2><p>Choose a circuit group and a fork tick. This runs a fresh matched experiment with the selected weights.</p></div>
      {#key selected.id}<Workbench selection={{ id: selected.id, atlasSha256: loaded.sha256 }} />{/key}
    {/if}
  {/if}
</div>
<style>
  .atlas { max-width:1500px; margin:auto; color:#d9e8f0; padding:2rem clamp(1rem,4vw,3rem); }
  header { max-width:850px; margin-bottom:2rem; } h1 { font-size:clamp(2.4rem,5vw,4.5rem); line-height:1.04; letter-spacing:-.04em; margin:1rem 0; } em { color:#8ce6c5; font-style:normal; }
  h2 { font-size:1.45rem; margin:.6rem 0; } p { color:#a6bacb; line-height:1.65; font-size:.9rem; }
  .eyebrow { color:#87cab7; text-transform:uppercase; letter-spacing:.14em; font-size:.7rem; } .badges { display:flex; flex-wrap:wrap; gap:.5rem; } .badges span { font-size:.75rem; border:1px solid #365d52; border-radius:20px; padding:.35rem .6rem; } .caveat { font-size:.8rem; }
  .discovery { display:grid; grid-template-columns:1fr 1fr; gap:1.3rem; } .panel { min-width:0; padding:1.4rem; background:#0e1c29; border:1px solid #283e4f; border-radius:10px; }
  .grid { display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:5px; margin-top:1.5rem; }
  .grid button,.empty { min-width:0; height:62px; border:1px solid #31584e; border-radius:5px; display:flex; flex-direction:column; align-items:center; justify-content:center; color:#e3f5ee; }
  .grid button { cursor:pointer; } .grid button span { font-size:.65rem; opacity:.8; } .grid button strong { font-size:.95rem; } .grid button[aria-pressed=true] { outline:3px solid #f1c870; outline-offset:1px; } .empty { color:#516778; border-style:dashed; }
  .axis { display:grid; grid-template-columns:repeat(6,1fr); text-align:center; margin-top:.7rem; font-size:.7rem; color:#a6bacb; } .axis-caption { text-align:center; font-size:.75rem; }
  details { margin-top:1rem; } summary { cursor:pointer; color:#b9d7cc; } .hash { overflow-wrap:anywhere; font-family:monospace; font-size:.7rem; }
  .repertoire-strip { margin-top:.9rem; padding:.6rem .75rem; border:1px solid #365d52; border-radius:.4rem; font-size:.78rem; color:#cbd8e7; }
  .repertoire-strip a { color:#79d8d0; margin-left:.3rem; }
  .repertoire-strip.error-message { border-color:#ef476f; color:#ffd7de; background:rgb(239 71 111 / 12%); }
  table { width:100%; border-collapse:collapse; font-size:.8rem; } caption { text-align:left; padding:.8rem 0; color:#a6bacb; } th,td { text-align:left; padding:.55rem .3rem; border-bottom:1px solid #2a3d4d; } td { font-variant-numeric:tabular-nums; }
  a { color:#a0e9cd; } button:focus-visible,a:focus-visible,summary:focus-visible { outline:2px solid #f1c870; outline-offset:3px; } .probe-intro { margin-top:3rem; }
  @media(max-width:800px) { .discovery { grid-template-columns:1fr; } .panel { padding:1rem; } .grid button,.empty { height:52px; } }
</style>
