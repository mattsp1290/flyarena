<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import './graphlab.css';
  import { GraphLabApi } from './api';
  import { GraphLabSession, initialSession, sessionActive } from './session';
  import {
    MAX_REWIRED_SEED,
    MAX_SEARCH_SEED,
    MAX_SEED,
    MIN_TOKEN_LENGTH,
    parseNeuronList,
    parseNeuronToken,
    validateAtlas,
    validateLesion,
    validateSwapset
  } from './forms';
  import type {
    AtlasResult,
    HealthResponse,
    JobKind,
    JobRequest,
    LesionResult,
    SwapsetResult
  } from './types';

  // -- Connection: endpoint and token live only in these two `$state`
  // variables (component memory) -- never written to `localStorage`,
  // `sessionStorage`, a cookie, or the URL, and only ever sent as this
  // page's own fetch calls to the endpoint the user typed
  // (`tests/unit/graphlab-storage.test.ts` asserts this directly).
  let endpoint = $state('http://127.0.0.1:8766');
  let token = $state('');
  let health = $state<HealthResponse | null>(null);
  let connection = $state<'idle' | 'checking' | 'connected' | 'unavailable'>('idle');
  let connectionError = $state('');
  let siteGraphSha = $state<string | null>(null);
  let bodyIds = $state<readonly string[] | undefined>(undefined);
  let disposed = false;

  async function checkHealth() {
    connection = 'checking';
    connectionError = '';
    try {
      const api = new GraphLabApi(endpoint, token);
      const response = await api.health(new AbortController().signal);
      if (disposed) return;
      health = response;
      connection = 'connected';
    } catch (error) {
      if (disposed) return;
      health = null;
      connection = 'unavailable';
      connectionError = error instanceof Error ? error.message : 'Connection failed.';
    }
  }
  function disconnect() {
    token = '';
    health = null;
    connection = 'idle';
    connectionError = '';
  }
  const graphShaMatches = $derived(
    health?.graphSha256 && siteGraphSha ? health.graphSha256 === siteGraphSha : null
  );

  onMount(() => {
    // The site's own manifest is public, unauthenticated, static data --
    // fetched to compare against the backend's `/health` `graphSha256`,
    // never sent anywhere.
    const base = `${import.meta.env.BASE_URL}data`;
    fetch(`${base}/malecns-arena-v1.manifest.json`)
      .then((response) => (response.ok ? response.json() : null))
      .then((manifest: { binarySha256?: string } | null) => {
        if (!disposed && manifest?.binarySha256) siteGraphSha = manifest.binarySha256;
      })
      .catch(() => {});
    fetch(`${base}/malecns-arena-v1.positions.json`)
      .then((response) => (response.ok ? response.json() : null))
      .then((positions: { bodyIds?: string[] } | null) => {
        if (!disposed && positions?.bodyIds) bodyIds = positions.bodyIds;
      })
      .catch(() => {});
    return () => {
      disposed = true;
    };
  });

  // -- Job forms -------------------------------------------------------
  type GraphMode = 'biological' | 'disconnected' | 'rewired';
  let tab = $state<JobKind>('lesion');
  const graphId = (mode: GraphMode, seed: number) => (mode === 'rewired' ? `rewired:${seed}` : mode);

  let lesionGraphMode = $state<GraphMode>('biological');
  let lesionRewiredSeed = $state(0);
  let lesionSetsText = $state('7');
  let lesionSeedStart = $state(30001);
  let lesionSeedCount = $state(4);
  let lesionTicks = $state(300);
  const lesionParsed = $derived.by(() => {
    const lines = lesionSetsText.split('\n').map((line) => line.trim()).filter((line) => line.length > 0);
    const sets: number[][] = [];
    const parseErrors: string[] = [];
    for (const line of lines) {
      try {
        sets.push(parseNeuronList(line, bodyIds));
      } catch (error) {
        parseErrors.push(error instanceof Error ? error.message : String(error));
      }
    }
    return { sets, parseErrors };
  });
  const lesionErrors = $derived([
    ...lesionParsed.parseErrors,
    ...validateLesion({
      graph: graphId(lesionGraphMode, lesionRewiredSeed),
      sets: lesionParsed.sets,
      seedStart: lesionSeedStart,
      seedCount: lesionSeedCount,
      ticks: lesionTicks
    })
  ]);

  let atlasGraphMode = $state<GraphMode>('biological');
  let atlasRewiredSeed = $state(0);
  let atlasSearchSeed = $state(1);
  let atlasPopulation = $state(8);
  let atlasGenerations = $state(2);
  let atlasTicks = $state(300);
  const atlasErrors = $derived(
    validateAtlas({
      graph: graphId(atlasGraphMode, atlasRewiredSeed),
      searchSeed: atlasSearchSeed,
      population: atlasPopulation,
      generations: atlasGenerations,
      ticks: atlasTicks
    })
  );

  interface SwapRow {
    a: string;
    b: string;
    c: string;
    d: string;
  }
  let swapRows = $state<SwapRow[]>([{ a: '', b: '', c: '', d: '' }]);
  let swapsetControls = $state(0);
  let swapsetSeedStart = $state(30001);
  let swapsetSeedCount = $state(4);
  let swapsetTicks = $state(300);
  function addSwapRow() {
    if (swapRows.length < 50) swapRows = [...swapRows, { a: '', b: '', c: '', d: '' }];
  }
  function removeSwapRow(index: number) {
    swapRows = swapRows.filter((_, i) => i !== index);
  }
  const swapsetParsed = $derived.by(() => {
    const swaps: { a: number; b: number; c: number; d: number }[] = [];
    const parseErrors: string[] = [];
    swapRows.forEach((row, index) => {
      try {
        // Each field accepts a raw index or a body id, exactly like the
        // lesion sets' own `parseNeuronList` (`03-frontend-route.md`'s
        // "parser for neuron sets by index or body id"). `parseNeuronToken`
        // throws for anything unrecognized, so a typo surfaces as a named
        // parse error here rather than a generic out-of-range message.
        swaps.push({
          a: parseNeuronToken(row.a, bodyIds),
          b: parseNeuronToken(row.b, bodyIds),
          c: parseNeuronToken(row.c, bodyIds),
          d: parseNeuronToken(row.d, bodyIds)
        });
      } catch (error) {
        parseErrors.push(`swap ${index + 1}: ${error instanceof Error ? error.message : String(error)}`);
      }
    });
    return { swaps, parseErrors };
  });
  const swapsetErrors = $derived([
    ...swapsetParsed.parseErrors,
    ...validateSwapset({
      swaps: swapsetParsed.swaps,
      controls: swapsetControls,
      seedStart: swapsetSeedStart,
      seedCount: swapsetSeedCount,
      ticks: swapsetTicks
    })
  ]);

  const currentErrors = $derived(tab === 'lesion' ? lesionErrors : tab === 'atlas' ? atlasErrors : swapsetErrors);

  // -- Session -----------------------------------------------------------
  let sessionState = $state(initialSession());
  const session = new GraphLabSession((next) => {
    sessionState = next;
  });
  let submittedGraph = $state('');
  let submittedKind = $state<JobKind | null>(null);
  let submittedBundleSha = $state<string | null>(null);
  const busy = $derived(sessionActive(sessionState.status));
  const job = $derived(sessionState.job);
  const result = $derived(job?.status === 'completed' ? job.result : null);
  /**
   * Every graph-lab job kind emits its own shape of progress line (lesion:
   * `{completed, total}`; atlas/swapset: `{stage}` -- see `jobs.py`'s
   * `_drain`, `entry-lesion.ts`'s `printProgress`, and `engine_atlas.py`'s/
   * `engine_swapset.py`'s `_progress`). Rather than special-casing each
   * shape, render whatever keys the current job's progress object actually
   * has -- so the user gets *some* feedback across a run that can take
   * minutes to tens of minutes (a code-review finding: this route
   * previously showed only the bare status word for the whole run).
   */
  const progressText = $derived.by(() => {
    const progress = job?.progress;
    if (!progress || !busy) return '';
    return Object.entries(progress)
      .map(([key, value]) => `${key}: ${typeof value === 'object' ? JSON.stringify(value) : String(value)}`)
      .join(' · ');
  });
  const canSubmit = $derived(
    connection === 'connected' && token.length >= MIN_TOKEN_LENGTH && !busy && currentErrors.length === 0
  );

  function submit() {
    if (!canSubmit) return;
    // A stale selection from a previous atlas run must never silently carry
    // over: a new result can legally contain a cell with the same `id`
    // (cell ids are per-search, not globally unique), which would otherwise
    // resolve `selectedCell` against the new result without the user having
    // clicked anything (a code-review finding).
    selectedCellId = null;
    let request: JobRequest;
    if (tab === 'lesion') {
      request = {
        kind: 'lesion',
        graph: graphId(lesionGraphMode, lesionRewiredSeed),
        sets: lesionParsed.sets,
        seedStart: lesionSeedStart,
        seedCount: lesionSeedCount,
        ticks: lesionTicks
      };
    } else if (tab === 'atlas') {
      request = {
        kind: 'atlas',
        graph: graphId(atlasGraphMode, atlasRewiredSeed),
        searchSeed: atlasSearchSeed,
        population: atlasPopulation,
        generations: atlasGenerations,
        ticks: atlasTicks
      };
    } else {
      request = {
        kind: 'swapset',
        graph: 'biological',
        swaps: swapsetParsed.swaps,
        controls: swapsetControls,
        seedStart: swapsetSeedStart,
        seedCount: swapsetSeedCount,
        ticks: swapsetTicks
      };
    }
    submittedGraph = request.graph;
    submittedKind = request.kind;
    submittedBundleSha = health?.bundleSha256 ?? null;
    void session.start(endpoint, token, request);
  }

  function short(sha: string | null | undefined): string {
    return sha ? `${sha.slice(0, 12)}…` : 'unknown';
  }
  /**
   * Not every job kind's result reports its own `graph`/`graphSha256`
   * (only `LesionResult` does -- see `types.ts`'s own doc comment): this
   * falls back to the graph id the user actually submitted, and to the
   * connected backend's own `/health` `graphSha256` only when that graph
   * was `"biological"` (the only case `/health`'s single sha value
   * describes). Never invents a hash the engine did not itself report.
   */
  const provenance = $derived.by(() => {
    if (!result || !submittedKind) return '';
    const graphShaKnown = 'graphSha256' in result ? result.graphSha256 : submittedGraph === 'biological' ? health?.graphSha256 : undefined;
    return [
      `graph ${submittedGraph}`,
      `graph sha ${short(graphShaKnown)}`,
      `engine bundle sha ${short(submittedBundleSha)}`,
      `host ${result.host.arch}/${result.host.node}`
    ].join(' · ');
  });

  function exportResult() {
    if (!job?.result) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(job.result)], { type: 'application/json' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = `graph-lab-${submittedKind}-${Date.now()}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  let selectedCellId = $state<number | null>(null);
  const atlasResult = $derived(submittedKind === 'atlas' ? (result as AtlasResult | null) : null);
  const selectedCell = $derived(atlasResult?.cells.find((cell) => cell.id === selectedCellId));
  const qualityScale = $derived(
    atlasResult ? Math.max(1, ...atlasResult.cells.map((cell) => Math.abs(cell.quality))) : 1
  );

  onDestroy(() => session.dispose());
</script>

<div class="graph-lab">
  <header class="masthead">
    <p class="eyebrow">Private lab · DGX Spark</p>
    <h1>Real-graph experiments,<br /><em>never published as-is.</em></h1>
    <p class="intro">
      Runs against the measured MaleCNS connectome on a private tailnet backend. Every result below is
      exploratory and model-bound -- not the pinned, published artifacts this site otherwise shows.
    </p>
  </header>

  <section class="panel connection" aria-label="Backend connection">
    <div class="fields">
      <label
        >Backend URL<input type="url" bind:value={endpoint} disabled={busy} required /></label
      >
      <label
        >Access token<input
          type="password"
          bind:value={token}
          autocomplete="off"
          disabled={busy}
          minlength={MIN_TOKEN_LENGTH}
          required
        /></label
      >
    </div>
    <p class="subtle">
      The endpoint and token stay in this page's memory only -- never saved, logged, or put in the URL.
      They are sent only to the backend URL above.
    </p>
    <div class="actions">
      <button type="button" onclick={checkHealth} disabled={connection === 'checking'}>Connect</button>
      {#if connection !== 'idle'}<button type="button" class="quiet" onclick={disconnect} disabled={busy}
          >Disconnect</button
        >{/if}
    </div>
    {#if connection === 'connected' && health}
      <dl class="health">
        <div><dt>Status</dt><dd>{health.status}</dd></div>
        <div><dt>Model version</dt><dd>{health.modelVersion}</dd></div>
        <div><dt>Bundle SHA-256</dt><dd class="hash">{short(health.bundleSha256)}</dd></div>
        <div><dt>Graph SHA-256</dt><dd class="hash">{short(health.graphSha256)}</dd></div>
        <div><dt>GPU</dt><dd>{health.gpu.available ? 'available' : 'unavailable or busy'}</dd></div>
      </dl>
      {#if graphShaMatches === false}
        <p role="alert" class="warning">
          This backend's graph SHA-256 does not match the graph loaded by this site. Results may not be
          comparable to the published artifacts.
        </p>
      {:else if graphShaMatches === true}
        <p class="subtle">Graph SHA-256 matches this site's loaded manifest.</p>
      {/if}
    {:else if connection === 'unavailable'}
      <p role="alert" class="error">Backend unavailable. {connectionError}</p>
    {/if}
  </section>

  {#if connection !== 'connected'}
    <section class="panel empty" aria-label="Job forms unavailable">
      <p class="subtle">Connect to a running graph-lab backend to submit jobs.</p>
    </section>
  {:else}
    <section class="panel jobs" aria-label="Submit a job">
      <div class="tabs" role="tablist" aria-label="Job kind">
        <button role="tab" aria-selected={tab === 'lesion'} disabled={busy} onclick={() => (tab = 'lesion')}
          >Lesion sweep</button
        >
        <button role="tab" aria-selected={tab === 'atlas'} disabled={busy} onclick={() => (tab = 'atlas')}
          >Atlas search</button
        >
        <button role="tab" aria-selected={tab === 'swapset'} disabled={busy} onclick={() => (tab = 'swapset')}
          >Swap-set intervention</button
        >
      </div>

      <form
        onsubmit={(e) => {
          e.preventDefault();
          submit();
        }}
      >
        <fieldset disabled={busy}>
          {#if tab === 'lesion'}
            <div class="fields">
              <label
                >Graph<select bind:value={lesionGraphMode}
                  ><option value="biological">Biological</option><option value="disconnected"
                    >Disconnected</option
                  ><option value="rewired">Rewired (seed)</option></select
                ></label
              >
              {#if lesionGraphMode === 'rewired'}
                <label
                  >Rewired seed<input
                    type="number"
                    min="0"
                    max={MAX_REWIRED_SEED}
                    bind:value={lesionRewiredSeed}
                  /></label
                >
              {/if}
              <label
                >Seed start<input type="number" min="0" max={MAX_SEED} bind:value={lesionSeedStart} /></label
              >
              <label>Seed count<input type="number" min="4" max="100" bind:value={lesionSeedCount} /></label>
              <label>Ticks<input type="number" min="300" max="1800" bind:value={lesionTicks} /></label>
            </div>
            <label
              >Lesion sets (one per line, neuron index or body id, space/comma separated)
              <textarea rows="4" bind:value={lesionSetsText}></textarea>
            </label>
          {:else if tab === 'atlas'}
            <div class="fields">
              <label
                >Graph<select bind:value={atlasGraphMode}
                  ><option value="biological">Biological</option><option value="disconnected"
                    >Disconnected</option
                  ><option value="rewired">Rewired (seed)</option></select
                ></label
              >
              {#if atlasGraphMode === 'rewired'}
                <label
                  >Rewired seed<input
                    type="number"
                    min="0"
                    max={MAX_REWIRED_SEED}
                    bind:value={atlasRewiredSeed}
                  /></label
                >
              {/if}
              <label
                >Search seed<input
                  type="number"
                  min="0"
                  max={MAX_SEARCH_SEED}
                  bind:value={atlasSearchSeed}
                /></label
              >
              <label>Population<input type="number" min="4" max="64" bind:value={atlasPopulation} /></label>
              <label
                >Generations<input type="number" min="1" max="48" bind:value={atlasGenerations} /></label
              >
              <label>Ticks<input type="number" min="300" max="900" bind:value={atlasTicks} /></label>
            </div>
            <p class="subtle">Refused by the backend as "GPU busy" if free GPU memory is below 2 GiB.</p>
          {:else}
            <p class="subtle">Base graph is always biological. Each swap retargets (a→b, c→d) to (a→d, c→b).</p>
            <div class="table-scroll">
              <table class="swap-rows">
                <thead><tr><th>a</th><th>b</th><th>c</th><th>d</th><th></th></tr></thead>
                <tbody>
                  {#each swapRows as row, index}
                    <tr>
                      <td><input aria-label={`Swap ${index + 1} a`} bind:value={row.a} /></td>
                      <td><input aria-label={`Swap ${index + 1} b`} bind:value={row.b} /></td>
                      <td><input aria-label={`Swap ${index + 1} c`} bind:value={row.c} /></td>
                      <td><input aria-label={`Swap ${index + 1} d`} bind:value={row.d} /></td>
                      <td
                        ><button
                          type="button"
                          onclick={() => removeSwapRow(index)}
                          disabled={swapRows.length <= 1}>Remove</button
                        ></td
                      >
                    </tr>
                  {/each}
                </tbody>
              </table>
            </div>
            <button type="button" class="quiet" onclick={addSwapRow} disabled={swapRows.length >= 50}
              >Add swap</button
            >
            <div class="fields">
              <label>Controls<input type="number" min="0" max="100" bind:value={swapsetControls} /></label>
              <label
                >Seed start<input
                  type="number"
                  min="0"
                  max={MAX_SEED}
                  bind:value={swapsetSeedStart}
                /></label
              >
              <label
                >Seed count<input type="number" min="4" max="100" bind:value={swapsetSeedCount} /></label
              >
              <label>Ticks<input type="number" min="300" max="1800" bind:value={swapsetTicks} /></label>
            </div>
          {/if}
          {#if currentErrors.length > 0}
            <ul class="form-errors">
              {#each currentErrors as error}<li>{error}</li>{/each}
            </ul>
          {/if}
          <button class="primary" type="submit" disabled={!canSubmit}>Submit job</button>
        </fieldset>
      </form>

      {#if busy}<button class="cancel" onclick={() => session.cancel()} disabled={job?.status === 'cancelling'}
          >Cancel job</button
        >{/if}
      <div class="job-status" role="status" aria-live="polite">
        <strong>{sessionState.status === 'idle' ? 'Ready' : sessionState.status}</strong>
        {#if sessionState.id}<span data-testid="graph-lab-job-id">Job {sessionState.id}</span>{/if}
        {#if progressText}<span class="subtle">{progressText}</span>{/if}
      </div>
      {#if sessionState.error}
        <p role="alert" class="error">{sessionState.error}</p>
        {#if sessionState.id}<button onclick={() => session.reconnect()}>Reconnect to job</button>{/if}
      {/if}
    </section>

    {#if result}
      <section class="panel result" aria-label="Job result">
        <div class="section-heading">
          <h2>Result</h2>
          <button onclick={exportResult}>Export evidence ↓</button>
        </div>
        <p class="provenance-label">{result.label}</p>
        <p class="subtle provenance">{provenance}</p>

        {#if submittedKind === 'lesion'}
          {@const lesion = result as LesionResult}
          <p class="subtle">Baseline: n={lesion.baseline.n}, mean {lesion.baseline.mean.toFixed(3)}</p>
          <div class="table-scroll">
            <table>
              <thead
                ><tr><th>Set</th><th>Neurons</th><th>Body ids</th><th>Mean Δ</th><th>95% CI</th><th>n</th></tr
                ></thead
              >
              <tbody>
                {#each lesion.sets as set, index}
                  <tr>
                    <td>{index + 1}</td>
                    <td>{set.indices.join(', ')}</td>
                    <td>{set.bodyIds.join(', ')}</td>
                    <td>{set.effect.meanDifference.toFixed(3)}</td>
                    <td>[{set.effect.ci95[0].toFixed(3)}, {set.effect.ci95[1].toFixed(3)}]</td>
                    <td>{set.n}</td>
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
        {:else if submittedKind === 'atlas' && atlasResult}
          <div class="grid" role="group" aria-label="Select a discovered behavior">
            {#each Array.from({ length: 36 }, (_, i) => (5 - Math.floor(i / 6)) * 6 + (i % 6)) as coordinate}
              {@const cell = atlasResult.cells.find((c) => c.cell === coordinate)}
              {#if cell}
                <button
                  aria-label={`Cell ${cell.cell}, quality ${cell.quality.toFixed(2)}`}
                  aria-pressed={cell.id === selectedCellId}
                  style={`background:rgba(66, 180, 151, ${0.12 + (0.6 * Math.max(0, cell.quality)) / qualityScale})`}
                  onclick={() => (selectedCellId = cell.id)}
                  ><span>#{cell.id}</span><strong>{cell.quality.toFixed(1)}</strong></button
                >
              {:else}
                <div class="empty" aria-label={`Undiscovered cell ${coordinate}`}>·</div>
              {/if}
            {/each}
          </div>
          {#if selectedCell}
            <p class="subtle">
              Cell #{selectedCell.cell}: quality {selectedCell.quality.toFixed(2)}, coverage {(
                selectedCell.coverage * 100
              ).toFixed(0)}%, turning {selectedCell.turning.toFixed(3)}
              {#if selectedCell.heldout.biological}
                · held-out own score {(
                  selectedCell.heldout.biological.reduce((sum, m) => sum + m.movementScore, 0) /
                  selectedCell.heldout.biological.length
                ).toFixed(2)}
              {/if}
            </p>
          {/if}
          <p class="subtle">
            {atlasResult.cells.length} of 36 cells occupied · GPU archive size {atlasResult.gpuArchiveSize} ·
            {atlasResult.collisions} collisions
          </p>
        {:else if submittedKind === 'swapset'}
          {@const swapset = result as SwapsetResult}
          <div class="table-scroll">
            <table>
              <thead><tr><th>Graph</th><th>n</th><th>Mean</th></tr></thead>
              <tbody>
                {#each swapset.scores as score}
                  <tr><td>{score.graphId}</td><td>{score.n}</td><td>{score.mean.toFixed(3)}</td></tr>
                {/each}
              </tbody>
            </table>
          </div>
          {#if swapset.paired.length > 0}
            <div class="table-scroll">
              <table>
                <thead><tr><th>Graph</th><th>Mean Δ vs {swapset.baselineGraphId}</th><th>95% CI</th></tr></thead>
                <tbody>
                  {#each swapset.paired as entry}
                    <tr>
                      <td>{entry.graphId}</td>
                      <td>{entry.effect.meanDifference.toFixed(3)}</td>
                      <td>[{entry.effect.ci95[0].toFixed(3)}, {entry.effect.ci95[1].toFixed(3)}]</td>
                    </tr>
                  {/each}
                </tbody>
              </table>
            </div>
          {/if}
          {#if swapset.controlDistribution}
            <p class="subtle">
              Control distribution (n={swapset.controlDistribution.n}): p5 {swapset.controlDistribution.p5.toFixed(
                3
              )}, p50 {swapset.controlDistribution.p50.toFixed(3)}, p95 {swapset.controlDistribution.p95.toFixed(3)}
            </p>
          {/if}
          {#if swapset.candidateRankAmongControls}
            <p class="subtle">
              Candidate rank among controls: bio-percentile {(
                swapset.candidateRankAmongControls.bioPercentile * 100
              ).toFixed(1)}%
            </p>
          {/if}
          <p class="subtle">
            Published-null percentile:
            {#if swapset.publishedNullPercentile === 'not comparable'}not comparable (seeds/ticks did not match
              the published study)
            {:else}
              {(swapset.publishedNullPercentile.bioPercentile * 100).toFixed(1)}%
            {/if}
          </p>
        {/if}
      </section>
    {/if}
  {/if}

  <section class="panel provenance-notice" aria-label="Model ledger">
    <strong>Private, exploratory, and model-bound.</strong>
    <p>
      These results run on the measured MaleCNS connectome, but through the same authored dynamics, sensory
      encoding, arena, and task as the rest of this site. They are computed on a private DGX Spark, are not
      published artifacts, and are not reproduced or verified the way the pinned site data is. Treat every
      number here as exploratory.
    </p>
  </section>
</div>
