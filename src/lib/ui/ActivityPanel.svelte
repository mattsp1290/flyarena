<script lang="ts">
  import { onDestroy } from 'svelte';
  import type { AgentId } from '../arena/types';
  import type { PositionsLoadResult } from '../experiment/assets';
  import type { ExperimentRunner, ExperimentTelemetry } from '../experiment/runner';
  // Type-only: `ActivityScene` pulls in the same `three`/OrbitControls chunk
  // `ArenaScene` does, so it is loaded via a dynamic `import()` inside
  // `toggle()` below, not statically here — this panel (and its collapsed
  // placeholder) must be cheap to mount even before the view is ever
  // expanded.
  import type { ActivityScene as ActivitySceneInstance, ActivitySceneUnavailableError as ActivitySceneUnavailableErrorType } from '../render/ActivityScene';

  /**
   * Collapsible "Neural activity" panel (WP3): one WebGL canvas showing both
   * arms' 1,008 neurons at their MaleCNS soma positions, colored by live
   * computed rate, with role shown by point shape. Collapsed by default.
   *
   * Streaming discipline: `runner.setActivityStreaming(true)` only while
   * *this panel* is expanded, and `setActivityStreaming(false)` on collapse
   * or unmount — never left on when the view isn't visible (see
   * `docs/architecture.md`'s streaming note and
   * `ExperimentRunner#setActivityStreaming`'s own doc comment). The toggle
   * itself (not the already-open view) is disabled while a topology switch
   * is in flight (`topologySwitchPending`) — an open view keeps rendering
   * and streaming straight through a switch of either arm; only opening/
   * closing is locked, so a close can't race the switch's own re-apply of
   * streaming (`ExperimentController#changeTopology`).
   */

  interface Props {
    runner: ExperimentRunner | undefined;
    positionsStatus: PositionsLoadResult | undefined;
    telemetry: ExperimentTelemetry | undefined;
    topologySwitchPending: boolean;
  }

  let { runner, positionsStatus, telemetry, topologySwitchPending }: Props = $props();

  let expanded = $state(false);
  let canvasEl = $state<HTMLCanvasElement | undefined>(undefined);
  let scene: ActivitySceneInstance | undefined;
  let rafId: number | undefined;
  let sceneError = $state<string | undefined>(undefined);
  let contextLostMessage = $state<string | undefined>(undefined);
  let lastUpdateTick = $state<Record<AgentId, number>>({ left: 0, right: 0 });

  let destroyed = false;
  let reducedMotion = false;
  let reducedMotionQuery: MediaQueryList | undefined;
  const lastRatesSeen: Record<AgentId, Float32Array | undefined> = { left: undefined, right: undefined };
  let lastColorUpdateMs = 0;
  /** Reduced-motion color-update cadence cap (plan: "at most 5 times per second"). */
  const REDUCED_MOTION_UPDATE_INTERVAL_MS = 200;

  const canExpand = $derived(positionsStatus?.status === 'ok');
  const disabledReason = $derived.by(() => {
    if (!positionsStatus) return 'Loading soma positions…';
    if (positionsStatus.status === 'missing') return `Positions unavailable: ${positionsStatus.reason}`;
    if (positionsStatus.status === 'invalid') return `Positions failed an integrity check: ${positionsStatus.reason}`;
    return undefined;
  });
  const toggleDisabled = $derived(topologySwitchPending || (!expanded && (!canExpand || !runner)));

  const handleReducedMotionChange = (event: MediaQueryListEvent): void => {
    reducedMotion = event.matches;
  };

  const stopLoop = (): void => {
    if (rafId !== undefined) cancelAnimationFrame(rafId);
    rafId = undefined;
  };

  /** Tear down the scene and stop streaming. Safe to call whether or not a scene/loop currently exists. */
  const teardown = (): void => {
    stopLoop();
    void runner?.setActivityStreaming(false);
    reducedMotionQuery?.removeEventListener('change', handleReducedMotionChange);
    reducedMotionQuery = undefined;
    const closing = scene;
    scene = undefined;
    closing?.dispose();
    lastRatesSeen.left = undefined;
    lastRatesSeen.right = undefined;
  };

  const frame = (nowMs: number): void => {
    if (destroyed || !scene) return;
    const throttled = reducedMotion && nowMs - lastColorUpdateMs < REDUCED_MOTION_UPDATE_INTERVAL_MS;
    if (!throttled && runner) {
      lastColorUpdateMs = nowMs;
      for (const agentId of ['left', 'right'] as const) {
        const rates = runner.getLatestRates(agentId);
        if (rates && rates !== lastRatesSeen[agentId]) {
          lastRatesSeen[agentId] = rates;
          scene.update(agentId, rates);
          lastUpdateTick = { ...lastUpdateTick, [agentId]: telemetry?.tick ?? lastUpdateTick[agentId] };
        }
      }
    }
    scene.render(nowMs);
    rafId = requestAnimationFrame(frame);
  };

  const expand = async (): Promise<void> => {
    if (!runner || !positionsStatus || positionsStatus.status !== 'ok') return;
    expanded = true;
    sceneError = undefined;
    contextLostMessage = undefined;

    let module: typeof import('../render/ActivityScene');
    try {
      module = await import('../render/ActivityScene');
    } catch (error) {
      if (destroyed) return;
      // Deliberately leaves `expanded` true: the fallback placeholder below
      // only renders inside the `{#if expanded}` block, matching
      // `App.svelte`'s own WebGL-unavailable fallback, which stays visible
      // (not collapsed away) so the reason is actually shown. Clicking
      // "Collapse" (still available — see `collapse()`) is how the user
      // dismisses/retries it.
      sceneError = `The activity renderer failed to load (${error instanceof Error ? error.message : String(error)}).`;
      return;
    }
    if (destroyed || !canvasEl) return;

    reducedMotionQuery = window.matchMedia?.('(prefers-reduced-motion: reduce)');
    reducedMotion = reducedMotionQuery?.matches ?? false;
    reducedMotionQuery?.addEventListener('change', handleReducedMotionChange);

    const { ActivityScene, ActivitySceneUnavailableError } = module as {
      ActivityScene: typeof ActivitySceneInstance;
      ActivitySceneUnavailableError: typeof ActivitySceneUnavailableErrorType;
    };
    try {
      scene = new ActivityScene({
        canvas: canvasEl,
        positions: positionsStatus.positions,
        rateMin: positionsStatus.rateMin,
        rateMax: positionsStatus.rateMax,
        reducedMotion,
        onContextLost: () => {
          contextLostMessage = 'Activity view lost its graphics context — collapse and reopen to retry.';
          teardown();
        }
      });
    } catch (error) {
      // Deliberately leaves `expanded` true — see the module-load catch
      // block above for why.
      sceneError =
        error instanceof ActivitySceneUnavailableError
          ? error.message
          : `The activity renderer failed to start (${error instanceof Error ? error.message : String(error)}).`;
      return;
    }

    lastUpdateTick = { left: 0, right: 0 };
    await runner.setActivityStreaming(true);
    if (destroyed) {
      teardown();
      return;
    }
    rafId = requestAnimationFrame(frame);
  };

  const collapse = (): void => {
    expanded = false;
    teardown();
  };

  const toggle = (): void => {
    if (toggleDisabled) return;
    if (expanded) collapse();
    else void expand();
  };

  onDestroy(() => {
    destroyed = true;
    teardown();
  });
</script>

<section class="panel activity" aria-labelledby="activity-heading">
  <div class="section-heading">
    <div>
      <p class="eyebrow">Anatomical activity view</p>
      <h2 id="activity-heading">Neural activity</h2>
    </div>
    <button type="button" onclick={toggle} disabled={toggleDisabled} aria-expanded={expanded}>
      {expanded ? 'Collapse' : 'Expand'}
    </button>
  </div>

  {#if !expanded}
    <p class="reason">{disabledReason ?? 'Expand to show both arms’ neurons at their soma positions, colored by live computed rate.'}</p>
  {/if}

  {#if expanded}
    <p class="labels">
      Positions: <strong>Measured</strong> (MaleCNS soma annotation) &middot;
      Color: <strong>Computed</strong> rate, not measured activity &middot;
      Roles: <strong>Annotated</strong>
    </p>
    <p class="labels">Both arms share neuron positions; only connections differ.</p>
    {#if positionsStatus?.status === 'ok'}
      <p class="coverage">
        Positioned: {positionsStatus.positions.coverage.soma} soma, {positionsStatus.positions.coverage.tosoma} soma-tract,
        {positionsStatus.positions.coverage.none} unavailable
        <span class="units">(units: {positionsStatus.positions.units})</span>
      </p>
    {/if}

    <div class="canvas-region">
      <canvas
        bind:this={canvasEl}
        class="activity-canvas"
        aria-label="Neural activity at soma positions"
        data-last-update-tick-left={lastUpdateTick.left}
        data-last-update-tick-right={lastUpdateTick.right}
        style:visibility={sceneError ? 'hidden' : 'visible'}
      ></canvas>
      {#if sceneError}
        <div class="canvas-placeholder" role="img" aria-label="Neural activity view unavailable">
          <p>{sceneError}</p>
        </div>
      {/if}
    </div>

    {#if contextLostMessage}
      <p class="context-lost" role="alert">{contextLostMessage}</p>
    {/if}

    {#if positionsStatus?.status === 'ok'}
      <div class="legend">
        <div class="legend-scale" aria-hidden="true">
          <span>{positionsStatus.rateMin.toFixed(2)}</span>
          <span class="legend-bar"></span>
          <span>{positionsStatus.rateMax.toFixed(2)}</span>
        </div>
        <ul class="legend-roles">
          <li><span class="shape circle" aria-hidden="true"></span>Sensory</li>
          <li><span class="shape square" aria-hidden="true"></span>Bridge</li>
          <li><span class="shape triangle" aria-hidden="true"></span>Descending</li>
        </ul>
      </div>
    {/if}
  {/if}
</section>

<style>
  /* `App.svelte`'s `<main>` is a two-column grid (arena + sidebar); this
     panel spans both columns and sits below the arena, per the plan's
     placement (`03-activity-view.md`). */
  .activity {
    grid-column: 1 / -1;
  }

  .reason {
    margin: 0.6rem 0 0;
    color: #9aacc2;
    font-size: 0.82rem;
  }

  .labels {
    margin: 0.6rem 0 0;
    color: #cbd8e7;
    font-size: 0.78rem;
    line-height: 1.5;
  }

  .coverage {
    margin: 0.3rem 0 0;
    color: #9aacc2;
    font-size: 0.78rem;
  }

  .units {
    color: #74889d;
  }

  .canvas-region {
    position: relative;
    margin-top: 0.8rem;
    aspect-ratio: 16 / 9;
    border-radius: 0.6rem;
    overflow: hidden;
    background: #05070c;
  }

  .activity-canvas {
    width: 100%;
    height: 100%;
    display: block;
  }

  .canvas-placeholder {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 1rem;
    text-align: center;
    color: #cbd8e7;
    font-size: 0.85rem;
  }

  .context-lost {
    margin: 0.6rem 0 0;
    padding: 0.6rem 0.75rem;
    border: 1px solid #ef476f;
    border-radius: 0.4rem;
    color: #ffd7de;
    background: rgb(239 71 111 / 12%);
    font-size: 0.82rem;
  }

  .legend {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 1rem;
    margin-top: 0.7rem;
    font-size: 0.78rem;
    color: #9aacc2;
  }

  .legend-scale {
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }

  .legend-bar {
    display: inline-block;
    width: 90px;
    height: 8px;
    border-radius: 4px;
    background: linear-gradient(to right, #440154, #3b528b, #21918c, #5ec962, #fde725);
  }

  .legend-roles {
    display: flex;
    gap: 0.8rem;
    margin: 0;
    padding: 0;
    list-style: none;
  }

  .legend-roles li {
    display: flex;
    align-items: center;
    gap: 0.35rem;
  }

  .shape {
    display: inline-block;
    width: 10px;
    height: 10px;
    background: #cbd8e7;
  }

  .shape.circle {
    border-radius: 50%;
  }

  .shape.triangle {
    width: 0;
    height: 0;
    background: transparent;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-bottom: 10px solid #cbd8e7;
  }

  button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
</style>
