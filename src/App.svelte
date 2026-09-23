<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import { ARENA_CONFIG } from './lib/arena/config';
  import type { ActionsByAgent } from './lib/arena/types';
  import { createSnapshot, createWorld, stepWorld } from './lib/arena/world';
  // Type-only: `three`/OrbitControls are large enough to warrant their own
  // chunk (see docs/architecture.md's load-budget note), so the actual
  // `./lib/render/ArenaScene` module is loaded via a dynamic `import()`
  // inside `onMount` below instead of statically here.
  import type { ArenaScene as ArenaSceneInstance, FrameTelemetry } from './lib/render/ArenaScene';

  let canvasEl: HTMLCanvasElement | undefined;
  let scene: ArenaSceneInstance | undefined;
  let rafId: number | undefined;
  // Set the instant the component unmounts, so the dynamic `import()` race
  // in `onMount` can't construct a scene (or write component state) after
  // teardown has already run.
  let destroyed = false;
  let reducedMotionQuery: MediaQueryList | undefined;

  let fps = $state(0);
  let rendererError = $state<string | undefined>(undefined);

  // Fixed placeholder seed for the demo loop below. The closed-loop bean
  // (work package 6) will replace this with a user-selected experiment seed.
  const DEMO_SEED = 20260923;
  const MAX_CATCHUP_SECONDS = ARENA_CONFIG.fixedDeltaSeconds * 5;

  let world = createWorld(DEMO_SEED);
  let accumulatorSeconds = 0;
  let lastFrameTimeMs: number | undefined;

  /**
   * PLACEHOLDER demo driver for the closed-loop bean (work package 6). Until
   * the neural Worker exists, something deterministic has to move the agents
   * so the renderer has real snapshots to draw. This scripted S-curve reads
   * only the tick count — no sensor values, no food/hazard positions, no
   * privileged map coordinates, and no pathfinding — so it never bypasses
   * the declared observation contract described in docs/architecture.md.
   */
  const demoActionsForTick = (tick: number): ActionsByAgent => {
    const yaw = Math.sin(tick / 45) * 0.6;
    return {
      left: { thrust: 0.45, yaw, brake: 0 },
      right: { thrust: 0.45, yaw: -yaw, brake: 0 }
    };
  };

  const frame = (nowMs: number): void => {
    try {
      if (lastFrameTimeMs !== undefined) {
        // Clamp to >= 0: a non-monotonic rAF timestamp (bfcache restore, a
        // mocked clock in tests) must never drive the accumulator negative,
        // which would otherwise stall ticking until real time "paid it back".
        const deltaSeconds = Math.max(0, (nowMs - lastFrameTimeMs) / 1000);
        accumulatorSeconds = Math.min(accumulatorSeconds + deltaSeconds, MAX_CATCHUP_SECONDS);
        while (accumulatorSeconds >= ARENA_CONFIG.fixedDeltaSeconds) {
          world = stepWorld(world, demoActionsForTick(world.tick));
          accumulatorSeconds -= ARENA_CONFIG.fixedDeltaSeconds;
        }
      }
      lastFrameTimeMs = nowMs;
      const alpha = accumulatorSeconds / ARENA_CONFIG.fixedDeltaSeconds;
      scene?.update(createSnapshot(world, alpha), nowMs);
    } catch (error) {
      // A throw here (e.g. stepWorld's own state validation) must not just
      // silently freeze the canvas on the last rendered frame — surface it
      // the same way a WebGL failure is surfaced, and stop cleanly.
      console.error('FlyArena demo loop stopped unexpectedly', error);
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
      // The component may have unmounted while the chunk was in flight —
      // never construct a scene (or touch `canvasEl`, which may already be
      // torn down) after that.
      if (destroyed || !canvasEl) return;

      const { ArenaScene, ArenaSceneUnavailableError } = renderModule;
      try {
        scene = new ArenaScene({
          canvas: canvasEl,
          arenaConfig: world.config,
          reducedMotion,
          onFrame: (telemetry: FrameTelemetry) => {
            fps = telemetry.fps;
          },
          onContextLost: () => {
            rendererError = 'The WebGL context was lost.';
            if (rafId !== undefined) cancelAnimationFrame(rafId);
            rafId = undefined;
            const lost = scene;
            scene = undefined;
            // Defer so disposal never runs from inside the renderer's own
            // 'webglcontextlost' event dispatch.
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
  <span class="status" aria-label="Experiment status: renderer and demo loop live, neural runtime not yet wired in">Renderer live</span>
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
        <span>Renderer live · neural pending</span>
      </div>
      <dl>
        <div><dt>Simulation</dt><dd>Demo motion (scripted placeholder)</dd></div>
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
