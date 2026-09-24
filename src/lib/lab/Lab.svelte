<script lang="ts">
  import { onDestroy } from 'svelte';
  import { LabApi } from './api';
  import Replay from './Replay.svelte';
  import type { Job, Options, Result } from './types';
  let endpoint = $state('http://127.0.0.1:8765');
  let token = $state('');
  import './lab.css';
  let options = $state<Options>({ seed: 17, device: 'cuda', population: 32, generations: 12,
    training_seeds: 8, heldout_seeds: 16, ticks: 240 });
  let job = $state<Job | null>(null);
  let result = $state<Result | null>(null);
  let error = $state('');
  let busy = $state(false);
  let selected = $state(3);
  let client: LabApi | null = null;
  let pollStopped = false;
  let controller: AbortController | null = null;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let identifier = $state<string | null>(null);
  let generation = $derived(job?.progress?.generation ?? 0);
  let scale = $derived(result ? Math.max(1, ...result.arms.map(a => Math.max(Math.abs(a.effect.low), Math.abs(a.effect.high)))) : 1);
  let historyPoints = $derived.by(() => {
    if (!result) return '';
    const low = Math.min(...result.history), high = Math.max(...result.history);
    const history = result.history;
    return history.map((v, i) => `${10 + i * 480 / (history.length - 1)},${90 - (v - low) * 70 / Math.max(1, high - low)}`).join(' ');
  });
  const number = (n: number) => n.toFixed(2);
  const active = (status: string) => ['queued', 'running', 'cancelling'].includes(status);
  function apply(next: Job) {
    job = next;
    busy = active(next.status);
    if (next.status === 'completed') result = next.result;
    if (next.status === 'failed') error = next.error ?? 'Experiment failed.';
  }
  async function poll() {
    if (!client || !controller || !identifier) return;
    pollStopped = false;
    try {
      apply(await client.status(identifier, controller.signal));
      if (busy) timer = setTimeout(poll, 600);
    } catch (e) {
      if (controller?.signal.aborted) return;
      pollStopped = true;
      error = `${e instanceof Error ? e.message : 'Connection failed.'} Reconnect to inspect the existing job.`;
      // Keep the job identity; a network error must not create a duplicate experiment.
    }
  }
  async function start() {
    clearTimeout(timer);
    controller?.abort();
    controller = new AbortController();
    error = ''; result = null; job = null; identifier = null; busy = true;
    try {
      client = new LabApi(endpoint, token);
      const created = await client.submit(options, controller.signal);
      identifier = created.id;
      await poll();
    } catch (e) {
      if (controller.signal.aborted) return;
      busy = false;
      error = `${e instanceof Error ? e.message : 'Connection failed.'} Submission is not retried automatically.`;
    }
  }
  async function cancel() {
    if (!client || !identifier || !controller) return;
    try {
      apply(await client.cancel(identifier, controller.signal));
      if (pollStopped) { error = ''; await poll(); }
    }
    catch (e) { error = e instanceof Error ? e.message : 'Cancellation failed.'; }
  }
  function reconnect() { clearTimeout(timer); error = ''; void poll(); }
  function exportResult() {
    if (!result) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(result)], { type: 'application/json' }));
    const a = document.createElement('a'); a.href = url; a.download = `flyarena-seed-${result.options.seed}.json`; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function quick() { options = { ...options, population: 4, generations: 1, training_seeds: 4, heldout_seeds: 8, ticks: 30 }; }
  onDestroy(() => {
    clearTimeout(timer);
    // Preserve the cancellation request when polling's controller is aborted.
    if (busy && client && identifier) {
      void client.cancel(identifier, new AbortController().signal).catch(() => {});
    }
    controller?.abort();
  });
</script>

<div class="dgx-sandbox">
<header class="masthead">
  <div><p class="eyebrow">FlyArena / DGX experiments</p><h1>What changes<br /><em>when a circuit goes quiet?</em></h1>
  <p class="intro">Train one controller. Freeze its weights. Explore eleven versions of the same world.</p></div>
  <span class="status">Authored circuit · synthetic task</span>
</header>
<main class="lab-layout">
  <aside class="panel controls" aria-label="Experiment controls">
    <p class="eyebrow">01 / Design the experiment</p><h2>One brain. Matched worlds.</h2>
    <p class="subtle">64 recurrent units, eight groups, a learned motor readout. Every intervention uses the same held-out seeds.</p>
    <form onsubmit={e => { e.preventDefault(); void start(); }}>
      <fieldset disabled={busy}>
        <details open><summary>Backend connection</summary>
          <label>Backend URL<input type="url" bind:value={endpoint} required /></label>
          <label>Access token<input type="password" bind:value={token} autocomplete="off" required minlength="16" /></label>
          <p class="subtle">Token stays in this page’s memory. Jobs keep running when you switch views. Start the backend with scripts/lab.sh.</p>
        </details>
        <div class="fields">
          <label>Device<select aria-label="Device" bind:value={options.device}><option value="cuda">NVIDIA GPU / CUDA</option><option value="cpu">CPU reference</option></select></label>
          <label>Experiment seed<input type="number" min="0" max="2147483647" step="1" bind:value={options.seed} required /></label>
          <label>Population<input type="number" min="4" max="64" step="1" bind:value={options.population} required /></label>
          <label>Generations<input type="number" min="1" max="40" step="1" bind:value={options.generations} required /></label>
          <label>Training seeds<input type="number" min="4" max="32" step="1" bind:value={options.training_seeds} required /></label>
          <label>Held-out seeds<input type="number" min="8" max="64" step="1" bind:value={options.heldout_seeds} required /></label>
          <label>Episode ticks<input type="number" min="30" max="600" step="1" bind:value={options.ticks} required /></label>
        </div>
        <button class="quiet" type="button" onclick={quick}>Use quick validation settings</button>
        <button class="primary" type="submit">Train & probe circuit <span aria-hidden="true">↗</span></button>
      </fieldset>
    </form>
    {#if busy}<button class="cancel" onclick={cancel} disabled={!identifier || job?.status === 'cancelling'}>Cancel experiment</button>{/if}
    <div class="job-status" role="status" aria-live="polite">
      <strong>{job?.status ?? (busy ? 'Connecting' : 'Ready for an experiment')}</strong>
      {#if identifier}<span data-testid="lab-job-id">Job {identifier}</span>{/if}
      {#if job?.progress.phase}<span>{job.progress.phase} · generation {generation} / {job.progress.generations}</span>{/if}
      {#if busy}<progress max={options.generations + 1} value={generation}>Running</progress>{/if}
    </div>
    {#if error}<p role="alert" class="error">{error}</p>{#if identifier}<button onclick={reconnect}>Reconnect to job</button>{/if}{/if}
    <p class="subtle">Jobs are ephemeral. Export a completed run before restarting the backend.</p>
  </aside>

  <section class="workspace" aria-label="Circuit experiment results">
    {#if result}
      <div class="metrics">
        <div><span>Compute device</span><strong>{result.runtime.device_name}</strong></div>
        <div><span>Training fitness</span><strong>{number(result.history[0]!)} → {number(result.history.at(-1)!)}</strong></div>
        <div><span>Independent evaluation</span><strong>{result.heldout_seeds.length} seeds × 11 arms</strong></div>
        <div><span>Compute time</span><strong>{number(result.runtime.training_seconds + result.runtime.evaluation_seconds)}s</strong></div>
      </div>
      <section class="panel atlas" aria-labelledby="atlas-heading">
        <div class="section-heading"><div><p class="eyebrow">02 / Intervene, then compare</p><h2 id="atlas-heading">Circuit lesion atlas</h2></div><button onclick={exportResult}>Export evidence ↓</button></div>
        <p class="subtle">Score difference from baseline on paired held-out worlds. Select an arm to replay it. Bars show descriptive 95% intervals over seed differences.</p>
        <div class="atlas-key"><span>Intervention</span><span>← Lower score · Higher score →</span><span>Mean Δ</span></div>
        {#each result.arms as arm, index}
          <button class="arm-row" class:selected={selected === index} onclick={() => selected = index} aria-pressed={selected === index} aria-label={`Inspect ${arm.name}`}>
            <span>{arm.name.startsWith('G') ? `Silence ${arm.name}` : arm.name}</span>
            <span class="effect-track"><span class="zero"></span><span class="interval" style:left={`${50 + arm.effect.low / scale * 46}%`} style:width={`${Math.max(0.3, (arm.effect.high - arm.effect.low) / scale * 46)}%`}></span><span class="dot" style:left={`${50 + arm.effect.mean / scale * 46}%`}></span></span>
            <span>{number(arm.effect.mean)}</span>
          </button>
        {/each}
        <p class="subtle selected-detail">{result.arms[selected]!.name}: mean score {number(result.arms[selected]!.mean_score)} · paired interval [{number(result.arms[selected]!.effect.low)}, {number(result.arms[selected]!.effect.high)}]</p>
      </section>
      <div class="result-bottom">
        <section class="panel replay">{#key result}<Replay {result} arm={selected} />{/key}</section>
        <section class="panel evidence" aria-label="Experiment evidence">
          <p class="eyebrow">03 / Inspect the evidence</p><h2>Training isn’t the test.</h2>
          <p class="subtle">Evolution sees {result.training_seeds.length} training seeds. This atlas uses {result.heldout_seeds.length} different seeds, with no retraining after intervention.</p>
          <svg viewBox="0 0 500 110" role="img" aria-label="Best training fitness by generation"><polyline points={historyPoints} fill="none" stroke="#77f0ca" stroke-width="3" /></svg>
          <p class="subtle">Generation 0 → {result.options.generations} · retained training fitness</p>
          <dl><div><dt>Neural units</dt><dd>64 / 8 groups</dd></div><div><dt>Model</dt><dd>{result.model_version}</dd></div><div><dt>Tensor peak</dt><dd>{result.runtime.peak_tensor_bytes === null ? 'CPU / not measured' : `${number(result.runtime.peak_tensor_bytes / 1024 ** 2)} MiB`}</dd></div><div><dt>Sham control</dt><dd>{number(result.arms[1]!.effect.mean)} Δ</dd></div></dl>
          <p class="subtle">The export contains every weight, seed, per-seed score, sampled path and runtime version. Reevaluate it with the backend CLI.</p>
        </section>
      </div>
    {:else}
      <section class="panel empty" aria-label="Experiment introduction">
        <p class="eyebrow">A counterfactual laboratory</p><h2>A circuit’s role is<br />something you can test.</h2>
        <div class="circuit-art" aria-hidden="true">{#each Array(8) as _, i}<div><span>G{i + 1}</span><div class="neurons">{#each Array(8) as _}<i></i>{/each}</div></div>{/each}</div>
        <div class="steps"><div><b>01</b><h3>Evolve</h3><p>A motor readout learns from parallel synthetic foraging episodes.</p></div><div><b>02</b><h3>Silence</h3><p>Freeze weights, then turn off one recurrent group at a time.</p></div><div><b>03</b><h3>Compare</h3><p>Inspect paired effects and replay the worlds that produced them.</p></div></div>
      </section>
    {/if}
    <section class="panel provenance" aria-label="Model ledger"><strong>Authored, not biological.</strong><p>The circuit topology, dynamics, sensory mappings and task are synthetic engineering choices. This is not a measured fly connectome or a brain emulation. Intervals describe sampled synthetic worlds, not biological significance. The task is separate from the browser arena model.</p></section>
  </section>
</main>

</div>
