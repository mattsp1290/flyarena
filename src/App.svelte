<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import { ARENA_CONFIG } from './lib/arena/config';
  import { createSnapshot, createWorld } from './lib/arena/world';
  import type { AgentId } from './lib/arena/types';
  import { loadPositions, type ArenaManifest, type PositionsLoadResult } from './lib/experiment/assets';
  import { ExperimentController } from './lib/experiment/controller';
  import type { ExperimentRunner, ExperimentTelemetry } from './lib/experiment/runner';
  import type { ExperimentStatus } from './lib/experiment/state';
  import type { GraphMode } from './lib/connectome/format';
  import ExperimentPanel from './lib/ui/ExperimentPanel.svelte';
  import TelemetryPanel from './lib/ui/TelemetryPanel.svelte';
  import LedgerPanel from './lib/ui/LedgerPanel.svelte';
  import ActivityPanel from './lib/ui/ActivityPanel.svelte';
  // Type-only: `three`/OrbitControls are large enough to warrant their own
  // chunk (see docs/architecture.md's load-budget note), so the actual
  // `./lib/render/ArenaScene` module is loaded via a dynamic `import()`
  // inside `onMount` below instead of statically here.
  import type { ArenaScene as ArenaSceneInstance, FrameTelemetry } from './lib/render/ArenaScene';

  let { onProbe }: { onProbe?: (setup: { seed: number; topology: GraphMode }) => void } = $props();

  let canvasEl: HTMLCanvasElement | undefined;
  let scene: ArenaSceneInstance | undefined;
  let rafId: number | undefined;
  // Set the instant the component unmounts, so any in-flight async work
  // (renderer chunk load, artifact fetch/verify, topology switch) can't
  // construct a scene, mutate reactive state, or reach into a disposed
  // controller/runner after teardown has already run.
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
  /** The anatomical activity view's soma-position sidecar, loaded independently of the graph artifacts (positions are optional presentation, not a Start gate — see `ActivityPanel.svelte`'s doc comment). `undefined` until `onManifest` fires and this load kicks off. */
  let positionsStatus = $state<PositionsLoadResult | undefined>(undefined);
  /** Mirrors `controller.getRunner()` into `$state` once `initialize()` resolves, so `ActivityPanel` (a reactive consumer) can be handed the runner without polling a plain, non-reactive handle. */
  let runner = $state<ExperimentRunner | undefined>(undefined);
  let seed = $state(DEFAULT_SEED);
  let topology = $state<Record<AgentId, GraphMode>>({ left: 'biological', right: 'rewired' });
  let telemetry = $state<ExperimentTelemetry | undefined>(undefined);
  /**
   * Number of in-flight (queued or running) topology-switch operations per
   * arm, mirrored from `ExperimentController`'s own internal counters via
   * its `onTopologySwitchCountChange` callback. A count rather than a
   * boolean: a second switch for the same arm can be queued behind a first
   * one that is still running, and the first's own completion must not
   * clear the "busy" flag out from under the still-pending second one.
   */
  let topologySwitchCount = $state<Record<AgentId, number>>({ left: 0, right: 0 });
  const topologySwitchPending = $derived(topologySwitchCount.left > 0 || topologySwitchCount.right > 0);

  /** True while running/loading — locks Start and Seed. Pause/Reset are governed by `topologySwitchPending` directly instead (see `ExperimentPanel`): they must stay clickable for the entire duration of a run, which is most of what this flag being true actually means. */
  const controlsLocked = $derived(status === 'running' || status === 'loading' || topologySwitchPending);
  /** Topology selectors additionally require `ready`/`finished` — a switch is never allowed mid-run (see `ExperimentRunner#setAgentBinding`), including while merely `paused`. */
  const topologyControlsLocked = $derived(controlsLocked || (status !== 'ready' && status !== 'finished'));

  // Plain (non-reactive) orchestration handle: `ExperimentController`
  // (`./lib/experiment/controller.ts`) owns asset loading, Worker/binding
  // construction, the `ExperimentRunner` itself, and topology-switch
  // serialization. It only ever reaches this component through the
  // callbacks passed to its constructor below, which assign into the
  // `$state` variables above — so this handle does not need to be reactive
  // itself. See that module's doc comment for why this logic lives there
  // and not here.
  let controller: ExperimentController | undefined;
  /** Rendered while the controller's runner does not exist yet (during asset loading) so the canvas has something real to draw immediately. */
  const idleWorld = createWorld(DEFAULT_SEED);

  const createNeuralWorker = (): Worker =>
    new Worker(new URL('./lib/worker/neural.worker.ts', import.meta.url), { type: 'module' });

  const handleStart = (): void => controller?.getRunner()?.start();
  const handlePause = (): void => controller?.getRunner()?.pause();

  const handleReset = (): void => {
    const runner = controller?.getRunner();
    runner?.reset(seed);
    // reset() doesn't go through the tick loop's onTelemetry callback (there
    // may be no further tick at all if the run never restarts), so refresh
    // the telemetry snapshot explicitly — otherwise the panel would keep
    // showing the previous run's scores/tick count after Reset.
    if (runner) telemetry = runner.getTelemetry();
  };

  /**
   * Only auto-applies immediately when the run is idle at `ready` (nothing
   * to lose). While `paused`/`finished`, the new seed is just stored — it
   * takes effect the next time the user presses Reset (`handleReset` above
   * always resets with the current `seed`) — rather than silently
   * discarding a paused run or a finished run's not-yet-downloaded replay.
   */
  const handleSeedInput = (nextSeed: number): void => {
    seed = nextSeed;
    const runner = controller?.getRunner();
    if (runner && runner.getStatus() === 'ready') {
      runner.reset(nextSeed);
      telemetry = runner.getTelemetry();
    }
  };

  const handleTopologyChange = (agentId: AgentId, mode: GraphMode): void => {
    controller?.changeTopology(agentId, mode);
  };

  const handleDownloadReplay = (): void => {
    const runner = controller?.getRunner();
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
    // Deferred rather than revoked synchronously: some browsers (older
    // Safari/Firefox) cancel an in-flight download if the object URL is
    // revoked in the same tick as the triggering click.
    setTimeout(() => URL.revokeObjectURL(url), 0);
  };

  const frame = (nowMs: number): void => {
    try {
      const runner = controller?.getRunner();
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
    controller = new ExperimentController({
      seed,
      totalTicks: TOTAL_TICKS,
      initialTopology: topology,
      createWorker: createNeuralWorker,
      callbacks: {
        onStatusChange: (next) => {
          if (!destroyed) status = next;
        },
        onTelemetry: (next) => {
          if (!destroyed) telemetry = next;
        },
        onError: (message) => {
          if (!destroyed) errorMessage = message;
        },
        onManifest: (nextManifest) => {
          if (destroyed) return;
          manifest = nextManifest;
          // Independent of graph-artifact loading/Worker construction below:
          // the activity view's positions are optional presentation, not a
          // Start gate, so this proceeds even if the rest of `initialize()`
          // goes on to fail.
          //
          // `loadPositions` documents itself as "never throws", but this
          // `.catch` enforces that contract at the call site too (dual
          // review finding) — without it, an unexpected throw anywhere in
          // its chain (a future edit, or `sha256Hex`/`crypto.subtle` itself)
          // would become an unhandled rejection and leave `positionsStatus`
          // `undefined` forever, showing "Loading soma positions…" with no
          // way to recover short of a reload.
          void loadPositions(nextManifest, `${import.meta.env.BASE_URL}data`)
            .catch(
              (error: unknown): PositionsLoadResult => ({
                status: 'invalid',
                reason: `unexpected error while loading positions: ${error instanceof Error ? error.message : String(error)}`
              })
            )
            .then((result) => {
              if (!destroyed) positionsStatus = result;
            });
        },
        onTopologyApplied: (agentId, mode) => {
          if (destroyed) return;
          topology = { ...topology, [agentId]: mode };
          // No-ops safely if the scene hasn't been constructed yet (e.g.
          // asset loading finished before the renderer chunk did) — the
          // scene-construction path below syncs to the then-current
          // `topology` itself once it exists, so no update is ever lost.
          scene?.setAgentTopology(agentId, mode);
        },
        onTopologySwitchCountChange: (counts) => {
          if (!destroyed) topologySwitchCount = { ...counts };
        }
      }
    });
    void controller.initialize().then(() => {
      // `initialize()` never throws; a failure routes through `onError`/
      // `onStatusChange` instead (see that method's doc comment) and leaves
      // `getRunner()` undefined, which this simply mirrors as `undefined` —
      // `ActivityPanel`'s own `!runner` guard already treats that as "not
      // ready yet."
      if (!destroyed) runner = controller?.getRunner();
    });

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

      // Sync both labels to whatever the *current* topology actually is at
      // the moment the scene starts existing — not a hardcoded default.
      // Asset loading and this renderer-chunk load race independently, so
      // by the time the scene is ready, `topology` may already reflect a
      // switch the controller applied before the scene existed (in which
      // case `onTopologyApplied` above already no-op'd against an
      // undefined `scene`). This is the other half of that guarantee — see
      // `ArenaScene#setAgentTopology`'s doc comment for why both calls
      // matter.
      scene?.setAgentTopology('left', topology.left);
      scene?.setAgentTopology('right', topology.right);

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
    controller?.dispose();
    scene?.dispose();
  });
</script>



<header class="masthead">
  <div>
    <p class="eyebrow">Connectome experiment · proof of concept</p>
    <h1>FlyArena</h1>
  </div>
  <!-- `role="status"` before `aria-label`: ARIA prohibits naming a plain
       generic element (a bare `<span>`'s implicit role), so without it the
       label would be silently ignored by assistive tech. -->
  <span class="status" role="status" aria-label={`Experiment status: ${status}`}>{status}</span>
</header>

{#if onProbe}
  <div class="probe-setup">
    <button onclick={() => onProbe?.({ seed, topology: topology.left })} disabled={controlsLocked}>Probe this setup (authored decoder)</button>
    <span>Starts matched worlds from this seed and left topology; live state is not copied.</span>
  </div>
{/if}
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

  <ActivityPanel {runner} {positionsStatus} {telemetry} {topologySwitchPending} />

  <aside class="sidebar" aria-label="Experiment information">
    <ExperimentPanel
      {status}
      {errorMessage}
      {seed}
      {topology}
      {controlsLocked}
      {topologySwitchPending}
      {topologyControlsLocked}
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

<style>
  .probe-setup { max-width:1440px; margin:0 auto 1rem; padding:0 clamp(1rem,4vw,3rem); display:flex; flex-wrap:wrap; align-items:center; gap:.8rem; }
  .probe-setup span { color:#a2b5c8; font-size:.75rem; }
</style>
