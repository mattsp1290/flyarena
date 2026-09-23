<svelte:head>
  <title>FlyArena — 3D Connectome Arena</title>
  <meta
    name="description"
    content="A transparent, client-only connectome arena proof of concept."
  />
</svelte:head>

<header class="masthead">
  <div>
    <p class="eyebrow">Connectome experiment · proof of concept</p>
    <h1>FlyArena</h1>
  </div>
  <span class="status" aria-label="Experiment status: scaffold">Scaffold</span>
</header>

<main>
  <section class="arena panel" aria-labelledby="arena-heading">
    <div class="section-heading">
      <div>
        <p class="eyebrow">Synthetic presentation</p>
        <h2 id="arena-heading">3D Connectome Arena</h2>
      </div>
      <span>Renderer pending</span>
    </div>
    <div class="canvas-placeholder" role="img" aria-label="Arena canvas placeholder">
      <p>The deterministic arena will render here.</p>
    </div>
  </section>

  <aside class="sidebar" aria-label="Experiment information">
    <section class="panel" aria-labelledby="controls-heading">
      <div class="section-heading">
        <h2 id="controls-heading">Experiment controls</h2>
        <span>Authored</span>
      </div>
      <p>Run controls and deterministic seed selection will appear in this region.</p>
      <div class="button-row" aria-hidden="true">
        <button type="button" disabled>Start</button>
        <button type="button" disabled>Reset</button>
      </div>
    </section>

    <section class="panel" aria-labelledby="telemetry-heading">
      <div class="section-heading">
        <h2 id="telemetry-heading">Telemetry</h2>
        <span>Awaiting runtime</span>
      </div>
      <dl>
        <div><dt>Simulation</dt><dd>Not started</dd></div>
        <div><dt>Neural step</dt><dd>—</dd></div>
        <div><dt>Seed</dt><dd>—</dd></div>
      </dl>
    </section>

    <section class="panel ledger" aria-labelledby="ledger-heading">
      <div class="section-heading">
        <h2 id="ledger-heading">Model ledger</h2>
        <span>Inspectability first</span>
      </div>
      <ul>
        <li><strong>Graph topology</strong><span>Measured</span></li>
        <li><strong>Biological annotations</strong><span>Annotated</span></li>
        <li><strong>Network dynamics</strong><span>Authored / literature-derived</span></li>
        <li><strong>Global parameters</strong><span>Calibrated</span></li>
        <li><strong>Encoder and decoder</strong><span>Authored</span></li>
        <li><strong>3D presentation</strong><span>Synthetic</span></li>
      </ul>
    </section>
  </aside>
</main>
