<script lang="ts">
  import { onDestroy, untrack } from 'svelte';
  import type { GraphMode } from '../connectome/format';
  import { CounterfactualClient, ExperimentCancelled } from './client';
  import { DEFAULT_REQUEST, BRANCHES, type ExportDocument, type Preparation, type Request } from './types';
  import { serializeEvidence } from './evidence';
  import PairedReplay from './PairedReplay.svelte';

  let { setup }: { setup?: { seed: number; topology: GraphMode } } = $props();
  let options = $state<Request>(untrack(() => ({ ...DEFAULT_REQUEST, ...setup })));
  let preparation = $state<Preparation | null>(null);
  let document = $state<ExportDocument | null>(null);
  let status = $state<'preparing' | 'ready' | 'running' | 'completed' | 'cancelled' | 'error'>('preparing');
  let error = $state('');
  let completed = $state(0);
  let selectedSeed = $state(0);
  const client = new CounterfactualClient();
  let epoch = 0;
  let target = $derived(preparation?.targets.find(t => t.id === options.target));
  let evidence = $derived(document?.evidence);
  let effectScale = $derived(evidence ? Math.max(.01, ...evidence.results.map(r => Math.abs(r.difference.movementScore))) : 1);
  const format = (value: number) => value.toFixed(4);

  async function prepare(topology: GraphMode) {
    const current = ++epoch;
    client.cancel(); preparation = null; error = ''; document = null; status = 'preparing';
    try {
      const next = await client.prepare(topology);
      if (current !== epoch) return;
      preparation = next; status = 'ready';
    } catch (e) {
      if (current !== epoch) return;
      error = e instanceof Error ? e.message : 'Graph preparation failed'; status = 'error';
    }
  }
  $effect(() => { void prepare(options.topology); });
  async function run() {
    const current = ++epoch;
    error = ''; document = null; completed = 0; selectedSeed = 0; status = 'running';
    try {
      const next = await client.run({ ...options }, n => { if (current === epoch) completed = n; });
      if (current !== epoch) return;
      document = next; status = 'completed';
    } catch (e) {
      if (current !== epoch) return;
      status = e instanceof ExperimentCancelled ? 'cancelled' : 'error';
      if (status === 'error') error = e instanceof Error ? e.message : 'Experiment failed';
    }
  }
  function download() {
    if (!document) return;
    const url = URL.createObjectURL(new Blob([serializeEvidence(document)], { type:'application/json' }));
    const anchor = window.document.createElement('a');
    anchor.href = url; anchor.download = `counterfactual-${document.evidence.request.seed}.json`; anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  onDestroy(() => { epoch++; client.cancel(); });
</script>
<div class="counterfactual">
  <header class="intro">
    <p class="eyebrow">FlyArena / counterfactual workbench</p>
    <h1>One past.<br /><em>Different futures.</em></h1>
    <p>Fork the same world and neural state. Silence a circuit group. Follow what changes.</p>
    <div class="badges"><span>Real connectome artifact</span><span>Authored dynamics & decoder</span><span>Runs in your browser</span></div>
  </header>
  <main class="workbench">
    <aside class="controls panel" aria-label="Counterfactual settings">
      <p class="eyebrow">01 / Set the intervention</p><h2>A controlled fork</h2>
      <p class="muted">The left agent uses the authored decoder. The right agent receives zero action. This restarts a seeded setup; it does not capture the live arena.</p>
      <form onsubmit={event => { event.preventDefault(); void run(); }}>
        <fieldset disabled={status === 'running'}>
          <label>Graph topology<select bind:value={options.topology}><option value="biological">Measured biological topology</option><option value="rewired">Rewired seed 0 control</option><option value="disconnected">Disconnected control</option></select></label>
          <label>Silence group<select bind:value={options.target} disabled={!preparation}>
            {#each preparation?.targets ?? [] as item}<option value={item.id} disabled={!item.indices.length}>{item.label} · {item.indices.length} neurons</option>{/each}
          </select></label>
          <div class="fields">
            <label>Base seed<input type="number" min="0" max="4294967295" step="1" bind:value={options.seed} required /></label>
            <label>Paired seeds<input type="number" min="4" max="16" step="1" bind:value={options.seedCount} required /></label>
            <label>Fork tick<input type="number" min="0" max="300" step="1" bind:value={options.warmup} required /></label>
            <label>Future ticks<input type="number" min="30" max="300" step="1" bind:value={options.horizon} required /></label>
          </div>
          <button class="quiet" type="button" onclick={() => { options = { ...options, seedCount:4, warmup:30, horizon:30 }; }}>Use quick probe settings</button>
          <button class="primary" type="submit" disabled={!preparation || !target?.indices.length || status === 'preparing'}>Fork & compare →</button>
        </fieldset>
      </form>
      {#if status === 'running'}<button class="cancel" onclick={() => client.cancel()}>Cancel probe</button>{/if}
      <p class="run-status" role="status" aria-live="polite">{status}{status === 'running' ? ` · ${completed} / ${options.seedCount} seeds` : ''}</p>
      {#if status === 'running'}<progress value={completed} max={options.seedCount}>Running</progress>{/if}
      {#if error}<p role="alert">{error}</p>{#if !preparation}<button onclick={() => prepare(options.topology)}>Retry graph loading</button>{/if}{/if}
      {#if preparation}
        <details><summary>Verified graph & target</summary>
          <p>{preparation.identity.neuronCount} neurons · {preparation.identity.edgeCount} edges</p>
          <p class="hash">SHA-256 {preparation.identity.binarySha256}</p>
          <p>{preparation.identity.sourceDataset} · {preparation.identity.license}</p>
          <p>Groups follow authored model mappings, not anatomy. {target?.indices.length ?? 0} targeted neurons:</p>
          <p class="ids">{target?.bodyIds.join(', ')}</p>
        </details>
      {/if}
      <p class="muted">For GPU training on a separate synthetic circuit, open the <a href="#dgx">DGX sandbox</a>. Its score units and world equations differ.</p>
    </aside>
    <div class="results">
      {#if evidence}
        <section class="panel findings" aria-label="Counterfactual results">
          <div class="result-heading"><div><p class="eyebrow">02 / Measure the difference</p><h2>{evidence.target.label}</h2></div><button onclick={download}>Export counterfactual evidence ↓</button></div>
          <div class="metrics"><div><span>Paired score effect</span><strong>{format(evidence.summary.effect.mean)}</strong></div><div><span>Descriptive 95% interval</span><strong>[{format(evidence.summary.effect.low)}, {format(evidence.summary.effect.high)}]</strong></div><div><span>Sham effect</span><strong>{format(evidence.summary.shamEffect.mean)}</strong></div></div>
          <p class="muted">Silenced minus baseline, using post-fork score increments across {evidence.seeds.length} matched seeds. Descriptive normal intervals are not biological significance tests. No training occurs in this experiment.</p>
          <div class="table-scroll"><table><caption>Mean post-fork outcomes</caption><thead><tr><th>Branch</th><th>Score</th><th>Food</th><th>Hazard contacts</th><th>Distance</th></tr></thead><tbody>
            {#each BRANCHES as branch}{@const mean = evidence.summary.means[branch]}<tr><th>{branch}</th><td>{format(mean.movementScore)}</td><td>{format(mean.foodPickups)}</td><td>{format(mean.hazardContacts)}</td><td>{format(mean.distanceTravelled)}</td></tr>{/each}
          </tbody></table></div>
          <h3>Choose a world to inspect</h3>
          <div class="seed-list">{#each evidence.results as result, i}<button class:selected={selectedSeed === i} aria-pressed={selectedSeed === i} aria-label={`Inspect seed ${result.seed}`} onclick={() => selectedSeed = i}>
            <span>{result.seed}</span><span class="effect-track"><i style:left={`${50 + Math.min(0, result.difference.movementScore / effectScale) * 48}%`} style:width={`${Math.max(.5, Math.abs(result.difference.movementScore / effectScale) * 48)}%`}></i></span><span>{format(result.difference.movementScore)}</span>
          </button>{/each}</div>
          <details><summary>Per-seed raw outcomes</summary><div class="table-scroll"><table><thead><tr><th>Seed</th><th>Baseline score</th><th>Sham score</th><th>Silenced score</th><th>Paired Δ</th></tr></thead><tbody>{#each evidence.results as result}<tr><td>{result.seed}</td><td>{format(result.branches.baseline.outcome.movementScore)}</td><td>{format(result.branches.sham.outcome.movementScore)}</td><td>{format(result.branches.lesion.outcome.movementScore)}</td><td>{format(result.difference.movementScore)}</td></tr>{/each}</tbody></table></div></details>
        </section>
        <section class="panel replay-panel">{#key evidence}<PairedReplay {evidence} {selectedSeed} />{/key}</section>
        <section class="panel evidence"><h2>Reproduce this result</h2><p>Every seed, graph hash, targeted body ID, outcome and sampled frame is in the export. Verify it against the same graph artifacts:</p><code>npm run experiment:counterfactual -- --verify counterfactual-{evidence.request.seed}.json</code><p>Exact reproduction is runtime-dependent. The CLI offers an explicitly labeled numerical diagnostic for small floating-point differences.</p></section>
      {:else}
        <section class="panel empty">
          <p class="eyebrow">A causal question inside an authored model</p><h2>What did this circuit<br />actually contribute?</h2>
          <svg viewBox="0 0 640 240" role="img" aria-label="One checkpoint forks into baseline, sham and silenced futures">
            <path d="M 20 120 H 220 C 350 120 340 40 480 40 H 600 M 220 120 H 600 M 220 120 C 350 120 340 200 480 200 H 600" fill="none" stroke="#446575" stroke-width="3" />
            <path d="M 220 120 C 350 120 340 200 480 200 H 600" fill="none" stroke="#7be5c5" stroke-width="3" />
            <circle cx="220" cy="120" r="11" fill="#f0cb81" /><text x="140" y="160" fill="#f0cb81" font-size="14">Identical checkpoint</text>
            <text x="495" y="30" fill="#c7d8e3" font-size="16">Baseline</text><text x="495" y="110" fill="#c7d8e3" font-size="16">Sham</text><text x="495" y="190" fill="#7be5c5" font-size="16">Silenced</text>
          </svg>
          <div class="principles"><div><h3>Copy the complete past</h3><p>World, random state, contacts and every neural rate are identical at the fork.</p></div><div><h3>Change one thing</h3><p>Keep the selected neurons at zero through every subsequent neural substep.</p></div><div><h3>Follow both futures</h3><p>Compare paired seeds and inspect when their behavior starts to diverge.</p></div></div>
        </section>
      {/if}
      <section class="panel honesty"><strong>Measured topology. Authored experiment.</strong><p>This intervention tests the running model, not a biological animal. Target groups are model assignments, not anatomical regions. Negative and zero effects are valid results.</p></section>
    </div>
  </main>
</div>
<style>
  .counterfactual { color:#deebf3; } .intro { max-width:1500px; margin:auto; padding:2.5rem clamp(1rem,4vw,3rem); } .intro h1 { font-weight:500; font-size:clamp(2.6rem,5vw,4.8rem); line-height:1.05; } h1 em { font-style:normal; color:#7be5c5; } .intro > p:not(.eyebrow) { margin-top:1.3rem; color:#b4c7d5; }
  .badges { display:flex; flex-wrap:wrap; gap:.6rem; margin-top:1.5rem; } .badges span { font-size:.72rem; border:1px solid #36505d; color:#b1cec8; padding:.5rem .7rem; border-radius:20px; }
  .workbench { display:grid; grid-template-columns:300px minmax(0,1fr); max-width:1500px; gap:1.5rem; align-items:start; }
  .controls { padding:1.4rem; } .muted, details p, .evidence p, .honesty p { color:#a1b5c6; font-size:.8rem; line-height:1.7; margin:1rem 0; }
  .fields { display:grid; grid-template-columns:1fr 1fr; gap:1rem; } label { display:grid; gap:.5rem; font-size:.8rem; color:#bfd0df; margin-top:1rem; min-width:0; }
  fieldset { margin:0; padding:0; border:0; min-width:0; } input, select { width:100%; min-width:0; padding:.65rem; border:1px solid #365063; border-radius:5px; background:#0b1825; color:#deebf3; font:inherit; }
  button { padding:.7rem .8rem; border:1px solid #365063; border-radius:6px; background:#162b3a; color:#deebf3; cursor:pointer; } button:disabled { opacity:.45; cursor:default; }
  button:focus-visible, input:focus-visible, select:focus-visible, summary:focus-visible { outline:2px solid #7be5c5; outline-offset:3px; }
  .primary, .quiet, .cancel { width:100%; margin-top:1rem; } .primary { background:#9be9d1; color:#09241d; font-weight:650; } .quiet { font-size:.75rem; background:transparent; } .cancel { color:#ffbfc4; }
  .run-status { text-transform:capitalize; font-size:.85rem; margin-top:1.2rem; } progress { width:100%; accent-color:#7be5c5; } [role=alert] { color:#ffb5bd; overflow-wrap:anywhere; }
  details { margin-top:1rem; border-top:1px solid #2b414e; padding-top:.9rem; } summary { cursor:pointer; font-size:.85rem; } .hash, code { overflow-wrap:anywhere; } .ids { max-height:10rem; overflow:auto; font-size:.7rem; } a { color:#7be5c5; }
  .results { display:grid; gap:1.5rem; min-width:0; } .findings, .replay-panel, .evidence { padding:1.5rem; min-width:0; } .result-heading { display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:1rem; } .result-heading button { font-size:.78rem; }
  .metrics { display:grid; grid-template-columns:1fr 1.5fr 1fr; gap:1rem; margin:1.6rem 0; } .metrics div { display:grid; gap:.6rem; padding:1rem; border:1px solid #294450; border-radius:6px; min-width:0; } .metrics span { color:#a1b8c8; font-size:.72rem; } .metrics strong { font-size:1rem; font-weight:550; overflow-wrap:anywhere; }
  .table-scroll { overflow:auto; } table { width:100%; border-collapse:collapse; text-align:left; font-size:.8rem; font-variant-numeric:tabular-nums; } th, td { padding:.7rem .5rem; border-bottom:1px solid #263d4b; } th { font-weight:500; color:#b1c5d4; } caption { text-align:left; color:#acbdce; margin-bottom:.5rem; } h3 { font-size:.9rem; margin:1.5rem 0 .8rem; }
  .seed-list { display:grid; gap:.3rem; } .seed-list button { display:grid; grid-template-columns:90px minmax(50px,1fr) 90px; gap:1rem; text-align:left; align-items:center; font-size:.8rem; background:transparent; border-color:transparent; } .seed-list button > span:last-child { text-align:right; } .seed-list .selected { border-color:#608e7c; background:#18383055; }
  .effect-track { position:relative; height:20px; background:linear-gradient(90deg,transparent 49.8%,#526675 49.8%,#526675 50.2%,transparent 50.2%); } .effect-track i { position:absolute; top:8px; height:4px; background:#7be5c5; border-radius:3px; }
  .empty { padding:2.5rem; } .empty h2 { font-size:clamp(1.8rem,3.5vw,3rem); font-weight:450; line-height:1.15; } .empty svg { width:100%; margin:2rem 0; } .principles { display:grid; grid-template-columns:repeat(3,1fr); gap:1.5rem; } .principles p { font-size:.8rem; color:#9eafc2; line-height:1.7; } .honesty { padding:1.2rem 1.5rem; } .honesty strong { color:#e1cea3; font-size:.85rem; } .honesty p { margin-bottom:0; }
  @media(max-width:1100px) { .metrics { grid-template-columns:1fr; } }
  @media(max-width:800px) { .workbench { grid-template-columns:1fr; } .empty { padding:1.5rem; } .principles { grid-template-columns:1fr; gap:0; } .seed-list button { grid-template-columns:75px minmax(40px,1fr) 65px; gap:.5rem; } }
</style>
