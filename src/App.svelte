<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import { ARENA_CONFIG } from './lib/arena/config';
  import { createSnapshot, createWorld } from './lib/arena/world';
  import type { AgentId } from './lib/arena/types';
  import type { ArenaManifest } from './lib/experiment/assets';
  import { loadArenaArtifacts } from './lib/experiment/assets';
  import { buildGraphBufferForMode, createWorkerAgentBinding } from './lib/experiment/bindings';
  import { ExperimentRunner, type ExperimentTelemetry } from './lib/experiment/runner';
  import { transition, type ExperimentStatus } from './lib/experiment/state';
  import type { GraphMode } from './lib/connectome/format';
  import { createWorkerClient, type WorkerClient } from './lib/worker/client';
  import ExperimentPanel from './lib/ui/ExperimentPanel.svelte';
  import TelemetryPanel from './lib/ui/TelemetryPanel.svelte';
  import LedgerPanel from './lib/ui/LedgerPanel.svelte';
  // Type-only: `three`/OrbitControls are large enough to warrant their own
  // chunk (see docs/architecture.md's load-budget note), so the actual
  // `./lib/render/ArenaScene` module is loaded via a dynamic `import()`
  // inside `onMount` below instead of statically here.
  import type { ArenaScene as ArenaSceneInstance, FrameTelemetry } from './lib/render/ArenaScene';

  let canvasEl: HTMLCanvasElement | undefined;
  let scene: ArenaSceneInstance | undefined;
  let rafId: number | undefined;
  // Set the instant the component unmounts, so any in-flight async work
  // (renderer chunk load, artifact fetch/verify, topology switch) can't
  // construct a scene, mutate reactive state, or reach into a disposed
  // runner after teardown has already run.
  let destroyed = false;
  let reducedMotionQuery: MediaQueryList | undefined;

  let fps = $state(0);
  let rendererError = $state<string | undefined>(undefined);

  const DEFAULT_SEED = 20260923;
  /** 90 seconds of simulated time at the arena's fixed 30 Hz tick — see the closed-loop bean's Definition of done. */
  const TOTAL_TICKS = Math.round(90 / ARENA_CONFIG.fixedDeltaSeconds);

  let status = $state<ExperimentStatus>('loading');
  let errorMessage = $state<string | undefined>(undefined);
  let manifest = $state<ArenaManifest | undefined>(undefined);
  let seed = $state(DEFAULT_SEED);
  let topology = $state<Record<AgentId, GraphMode>>({ left: 'biological', right: 'rewired' });
  let telemetry = $state<ExperimentTelemetry | undefined>(undefined);

  const controlsLocked = $derived(status === 'running' || status === 'loading');

  // Plain (non-reactive) orchestration handles: the runner/worker clients own
  // their own internal state and only ever reach the UI through the $state
  // variables above, via the runner's onStatusChange/onTelemetry/onError
  // callbacks — these do not need to be reactive themselves.
  let runner: ExperimentRunner | undefined;
  let workerClients: Record<AgentId, WorkerClient> | undefined;
  let biologicalGraphBuffer: ArrayBuffer | undefined;
  let rewiredGraphBuffer: ArrayBuffer | undefined;
  /** Rendered while `runner` does not exist yet (during asset loading) so the canvas has something real to draw immediately. */
  const idleWorld = createWorld(DEFAULT_SEED);

  const createNeuralWorker = (): Worker =>
    new Worker(new URL('./lib/worker/neural.worker.ts', import.meta.url), { type: 'module' });

  /**
   * WP6 item 2: fetch both graph artifacts, gunzip, and sha256-verify them
   * against the manifest before anything is allowed to start. WP6 item 3:
   * one dedicated Worker per arm (see `docs/architecture.md`), initialized
   * with the default biological (left) vs rewired (right) topology.
   */
  const initializeExperiment = async (): Promise<void> => {
    let artifacts: Awaited<ReturnType<typeof loadArenaArtifacts>>;
    try {
      artifacts = await loadArenaArtifacts();
    } catch (error) {
      if (destroyed) return;
      errorMessage = error instanceof Error ? error.message : String(error);
      status = transition(status, { type: 'assetsFailed' });
      return;
    }
    if (destroyed) return;

    manifest = artifacts.manifest;
    biologicalGraphBuffer = artifacts.biological;
    rewiredGraphBuffer = artifacts.rewired;

    try {
      const left = createNeuralWorker();
      const right = createNeuralWorker();
      workerClients = { left: createWorkerClient(left), right: createWorkerClient(right) };

      const [leftBinding, rightBinding] = await Promise.all([
        createWorkerAgentBinding(
          workerClients.left,
          buildGraphBufferForMode(biologicalGraphBuffer, rewiredGraphBuffer, topology.left),
          topology.left
        ),
        createWorkerAgentBinding(
          workerClients.right,
          buildGraphBufferForMode(biologicalGraphBuffer, rewiredGraphBuffer, topology.right),
          topology.right
        )
      ]);
      if (destroyed) return;

      runner = new ExperimentRunner({
        seed,
        totalTicks: TOTAL_TICKS,
        agents: { left: leftBinding, right: rightBinding },
        onStatusChange: (next) => {
          if (!destroyed) status = next;
        },
        onTelemetry: (next) => {
          if (!destroyed) telemetry = next;
        },
        onError: (error) => {
          if (!destroyed) errorMessage = error.message;
        }
      });
      telemetry = runner.getTelemetry();
      status = runner.getStatus();
    } catch (error) {
      if (destroyed) return;
      errorMessage = error instanceof Error ? error.message : String(error);
      status = transition(status, { type: 'assetsFailed' });
    }
  };

  const handleStart = (): void => runner?.start();
  const handlePause = (): void => runner?.pause();

  const handleReset = (): void => {
    runner?.reset(seed);
    // reset() doesn't go through the tick loop's onTelemetry callback (there
    // may be no further tick at all if the run never restarts), so refresh
    // the telemetry snapshot explicitly — otherwise the panel would keep
    // showing the previous run's scores/tick count after Reset.
    if (runner) telemetry = runner.getTelemetry();
  };

  const handleSeedInput = (nextSeed: number): void => {
    seed = nextSeed;
    if (runner && runner.getStatus() !== 'running') {
      runner.reset(nextSeed);
      telemetry = runner.getTelemetry();
    }
  };

  /** Re-initializes just one arm's Worker with a freshly derived graph buffer for the chosen topology; disallowed while running (enforced both here and by `ExperimentPanel`'s disabled selects). */
  const handleTopologyChange = (agentId: AgentId, mode: GraphMode): void => {
    if (!runner || !workerClients || !biologicalGraphBuffer || !rewiredGraphBuffer) return;
    if (runner.getStatus() === 'running') return;
    topology = { ...topology, [agentId]: mode };
    const client = workerClients[agentId];
    const buffer = buildGraphBufferForMode(biologicalGraphBuffer, rewiredGraphBuffer, mode);
    void (async () => {
      try {
        await client.dispose();
        const binding = await createWorkerAgentBinding(client, buffer, mode);
        if (destroyed || !runner) return;
        runner.setAgentBinding(agentId, binding);
        telemetry = runner.getTelemetry();
      } catch (error) {
        if (destroyed) return;
        errorMessage = error instanceof Error ? error.message : String(error);
        status = 'error';
      }
    })();
  };

  const handleDownloadReplay = (): void => {
    if (!runner) return;
    const replay = runner.getReplayExport();
    const blob = new Blob([JSON.stringify(replay, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `flyarena-replay-seed${replay.seed}-tick${replay.finalSummary.ticks}.json`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  };

  const frame = (nowMs: number): void => {
    try {
      const snapshot = runner ? runner.getSnapshot(nowMs) : createSnapshot(idleWorld, 1);
      scene?.update(snapshot, nowMs);
    } catch (error) {
      // A throw here must not just silently freeze the canvas on the last
      // rendered frame — surface it the same way a WebGL failure is
      // surfaced, and stop cleanly.
      console.error('FlyArena render loop stopped unexpectedly', error);
      rendererError = `The arena stopped: ${error instanceof Error ? error.message : String(error)}`;
      rafId = undefined;
      const failed = scene;
      scene = undefined;
      failed?.dispose();
      return;
    }
    rafId = requestAnimationFrame(frame);
  };

  const handleReducedMotionChange = (event: MediaQueryListEvent): void => {
    scene?.setReducedMotion(event.matches);
  };

  onMount(() => {
    void initializeExperiment();

    if (!canvasEl) return;
    reducedMotionQuery = window.matchMedia?.('(prefers-reduced-motion: reduce)');
    const reducedMotion = reducedMotionQuery?.matches ?? false;
    reducedMotionQuery?.addEventListener('change', handleReducedMotionChange);

    const start = async (): Promise<void> => {
      let renderModule: typeof import('./lib/render/ArenaScene');
      try {
        renderModule = await import('./lib/render/ArenaScene');
      } catch (error) {
        if (destroyed) return;
        console.error('FlyArena renderer module failed to load', error);
        rendererError = `The 3D renderer failed to load (${error instanceof Error ? error.message : String(error)}).`;
        return;
      }
      if (destroyed || !canvasEl) return;

      const { ArenaScene, ArenaSceneUnavailableError } = renderModule;
      try {
        scene = new ArenaScene({
          canvas: canvasEl,
          arenaConfig: ARENA_CONFIG,
          reducedMotion,
          onFrame: (frameTelemetry: FrameTelemetry) => {
            fps = frameTelemetry.fps;
          },
          onContextLost: () => {
            rendererError = 'The WebGL context was lost.';
            if (rafId !== undefined) cancelAnimationFrame(rafId);
            rafId = undefined;
            const lost = scene;
            scene = undefined;
            queueMicrotask(() => lost?.dispose());
          }
        });
      } catch (error) {
        console.error('FlyArena renderer failed to start', error);
        rendererError =
          error instanceof ArenaSceneUnavailableError
            ? error.message
            : `The 3D renderer failed to start (${error instanceof Error ? error.message : String(error)}).`;
        return;
      }

      rafId = requestAnimationFrame(frame);
    };

    void start();
  });

  onDestroy(() => {
    destroyed = true;
    reducedMotionQuery?.removeEventListener('change', handleReducedMotionChange);
    reducedMotionQuery = undefined;
    if (rafId !== undefined) cancelAnimationFrame(rafId);
    rafId = undefined;
    runner?.dispose();
    workerClients?.left.terminate();
    workerClients?.right.terminate();
    scene?.dispose();
  });
</script>

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
  <span class="status" aria-label={`Experiment status: ${status}`}>{status}</span>
</header>

<main>
  <section class="arena panel" aria-labelledby="arena-heading">
    <div class="section-heading">
      <div>
        <p class="eyebrow">Synthetic presentation</p>
        <h2 id="arena-heading">3D Connectome Arena</h2>
      </div>
      <span>{rendererError ? 'Renderer unavailable' : 'Spectator camera · drag to orbit'}</span>
    </div>
    <div class="canvas-region">
      <canvas
        bind:this={canvasEl}
        class="arena-canvas"
        aria-label="3D connectome arena"
        style:visibility={rendererError ? 'hidden' : 'visible'}
      ></canvas>
      {#if rendererError}
        <div class="canvas-placeholder" role="img" aria-label="Arena canvas unavailable">
          <p>3D rendering is unavailable. {rendererError}</p>
        </div>
      {:else}
        <div class="fps-overlay" aria-hidden="true">{Math.round(fps)} fps</div>
      {/if}
    </div>
  </section>

  <aside class="sidebar" aria-label="Experiment information">
    <ExperimentPanel
      {status}
      {errorMessage}
      {seed}
      {topology}
      {controlsLocked}
      onStart={handleStart}
      onPause={handlePause}
      onReset={handleReset}
      onSeedInput={handleSeedInput}
      onTopologyChange={handleTopologyChange}
      onDownloadReplay={handleDownloadReplay}
    />

    {#if telemetry}
      <TelemetryPanel {telemetry} />
    {/if}

    <LedgerPanel {manifest} />
  </aside>
</main>
