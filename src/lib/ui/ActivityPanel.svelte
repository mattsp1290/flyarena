<script lang="ts">
  import { onDestroy, tick } from 'svelte';
  import type { AgentId } from '../arena/types';
  import type { ArenaManifest, PositionsLoadResult } from '../experiment/assets';
  import type { ExperimentRunner, ExperimentTelemetry } from '../experiment/runner';
  import type { ConnectomeGraph, GraphMode } from '../connectome/format';
  import { lesionAtlasGraphKeyForTopology, loadLesionAtlas, type LesionAtlasLoadResult } from '../experiment/lesionAtlas';
  import { githubDocUrl } from './links';
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
    /** `undefined` until `App.svelte`'s `onManifest` fires. Needed (with `biologicalGraph`) to lazily load the lesion atlas the first time the lesion-effect color mode is selected — see `ensureLesionAtlasLoaded`. */
    manifest?: ArenaManifest;
    /** The already-verified, already-parsed biological graph `App.svelte` mirrors from `onManifest` — threaded into `loadLesionAtlas` for its `bodyIds` cross-check, the same graph `loadPositions` already reuses. */
    biologicalGraph?: ConnectomeGraph;
    /** Each arm's current topology (`App.svelte`'s own `topology` state) — maps to which lesion-atlas graph (if any) an arm's static colors come from; see `lesionAtlasGraphKeyForTopology`. Defaults match `App.svelte`'s own initial value so existing callers/tests that don't pass this prop keep working unchanged. */
    topology?: Record<AgentId, GraphMode>;
  }

  let {
    runner,
    positionsStatus,
    telemetry,
    topologySwitchPending,
    manifest,
    biologicalGraph,
    topology = { left: 'biological', right: 'rewired' }
  }: Props = $props();

  let expanded = $state(false);
  let canvasEl = $state<HTMLCanvasElement | undefined>(undefined);
  let scene: ActivitySceneInstance | undefined;
  /** `bind:this` targets must be `$state` in Svelte 5 (`svelte/non_reactive_update`) even though these two are only ever read/written imperatively — see `handleColorModeInputChange`. */
  let liveRadioEl = $state<HTMLInputElement | undefined>(undefined);
  let lesionRadioEl = $state<HTMLInputElement | undefined>(undefined);
  let rafId: number | undefined;
  let sceneError = $state<string | undefined>(undefined);
  let contextLostMessage = $state<string | undefined>(undefined);
  // A `$state` record, mutated in place (`lastUpdateTick[agentId] = ...`),
  // never reassigned wholesale (thermo-architecture S2 fix): a fresh rate
  // array can arrive up to ~30 Hz per arm, and this debug-only DOM attribute
  // (see `data-last-update-tick-*` below, read by `tests/e2e/arena.spec.ts`'s
  // `waitForActivityUpdateTick`) is the only reason it exists. Svelte 5's
  // `$state` proxy makes a single-property mutation fine-grained-reactive on
  // its own — the earlier per-tick allocation came from spreading a *new*
  // object (`{ ...lastUpdateTick, [agentId]: ... }`) on every update, not
  // from using a record at all, so mutating the existing one in place (the
  // same pattern `lastRatesSeen` below already uses) removes the allocation
  // without needing a separate scalar variable, and an `if (agentId ===
  // 'left') ... else ...` branch, per arm.
  let lastUpdateTick = $state<Record<AgentId, number>>({ left: 0, right: 0 });
  /**
   * `runner.supportsActivityStreaming(agentId)` (thermo-architecture S1
   * fix), checked every frame while the panel is open and mirrored into
   * `$state` only on change (same discipline as `lastUpdateTick` above):
   * whether *this arm's current binding* can stream full-neuron rates at
   * all, as distinct from whether rates simply haven't arrived yet. Every
   * browser-constructed binding supports it today (`createOracleAgentBinding`,
   * the one binding type that doesn't, is never reachable from any UI
   * topology option), so this stays `true` in practice — but if that ever
   * changes, the panel now shows an honest "streaming unavailable" reason
   * for that arm instead of leaving it at the ambiguous "No data yet" grey
   * indefinitely.
   */
  let streamingSupported = $state<Record<AgentId, boolean>>({ left: true, right: true });

  /** `'live'` (default): per-tick computed rate colors, as before. `'lesion'`: static diverging colors from the offline lesion atlas — see `switchColorMode`. */
  let colorMode = $state<'live' | 'lesion'>('live');
  /** `undefined` until the lesion-effect mode is first selected (lazy load — see `ensureLesionAtlasLoaded`). Never set to an `'unavailable'` result — see `lesionAtlasTransientReason` below. */
  let lesionAtlasStatus = $state<LesionAtlasLoadResult | undefined>(undefined);
  /** Memoizes the in-flight/completed load so a second mode switch (or a second arm) never re-fetches — plain (non-reactive): only `lesionAtlasStatus` above needs to drive the UI. Cleared (not left memoized) after an `'unavailable'` outcome so the next selection retries the fetch. */
  let lesionAtlasLoadPromise: Promise<LesionAtlasLoadResult> | undefined;
  /**
   * The reason from the most recent `'unavailable'` (fetch/network failure)
   * outcome, shown as a non-blocking hint — round-2 dual review (Important):
   * unlike `lesionAtlasStatus`, a fetch failure is retryable, so it must
   * never disable the radio the way `lesionOptionDisabledReason` does for a
   * genuinely missing entry or a hash/shape failure. Cleared on any
   * subsequent load attempt or a later non-`'unavailable'` outcome.
   */
  let lesionAtlasTransientReason = $state<string | undefined>(undefined);
  /**
   * True once the current `scene` has finished constructing (set at the end
   * of `expand()`, cleared in `teardown()`). Gates the color-mode radio
   * group: switching mode before a scene exists would have nothing to paint
   * — mirrors `toggleDisabled`'s own "don't offer a control whose action
   * would silently no-op" discipline.
   */
  let sceneReady = $state(false);
  /**
   * Debug-only, mirrors which write path last painted each arm's colors —
   * `data-color-source-{left,right}` below, read by
   * `tests/e2e/arena.spec.ts` to prove that repeated animation frames in
   * lesion mode never let a live-mode `update()`/`clear()` repaint over the
   * static lesion colors (both are no-ops in lesion mode — see
   * `ActivityScene.ts` — but this attribute is the *observable* proof of
   * that, the same role `lastUpdateTick` plays for live-mode streaming).
   * Mutated in place, never reassigned wholesale — same allocation
   * discipline as `lastUpdateTick` above.
   */
  let colorSource = $state<Record<AgentId, 'live' | 'lesion'>>({ left: 'live', right: 'live' });

  let destroyed = false;
  let reducedMotion = false;
  let reducedMotionQuery: MediaQueryList | undefined;
  const lastRatesSeen: Record<AgentId, Float32Array | undefined> = { left: undefined, right: undefined };
  let lastColorUpdateMs = 0;
  /**
   * Bumped by every `expand()` call and by `collapse()`/`onDestroy`; each
   * `expand()` invocation captures its own value and checks it after every
   * `await` (dual review finding). Without this, a fast Expand -> Collapse
   * -> Expand can resume a superseded `expand()` call after a *newer* one
   * has already built (or is building) its own scene — the stale call would
   * either build a second `ActivityScene` on the same canvas (leaking the
   * first renderer/`ResizeObserver`/listeners) or start a second rAF loop
   * rendering over the current scene. `isStaleExpand()` is the single check
   * every resume point uses.
   */
  let openGeneration = 0;
  const isStaleExpand = (generation: number): boolean => destroyed || generation !== openGeneration;
  /** Reduced-motion color-update cadence cap (plan: "at most 5 times per second"). */
  const REDUCED_MOTION_UPDATE_INTERVAL_MS = 200;
  /** Plain GitHub blob link to the human-readable lesion-atlas report (`docs/` is not part of the deployed static site) — shared `githubDocUrl` helper, base-path-safe by construction. */
  const LESION_REPORT_URL = githubDocUrl('lesion-atlas-report.md');

  const canExpand = $derived(positionsStatus?.status === 'ok');
  const disabledReason = $derived.by(() => {
    if (!positionsStatus) return 'Loading soma positions…';
    if (positionsStatus.status === 'missing') return `Positions unavailable: ${positionsStatus.reason}`;
    if (positionsStatus.status === 'invalid') return `Positions failed an integrity check: ${positionsStatus.reason}`;
    // positionsStatus.status === 'ok' past this point.
    if (topologySwitchPending) return 'Waiting for the topology switch to finish…';
    if (!runner) return 'Waiting for the experiment to finish loading…';
    return undefined;
  });
  const toggleDisabled = $derived(topologySwitchPending || (!expanded && (!canExpand || !runner)));

  /**
   * Cheap, synchronous check against the already-loaded manifest (no
   * fetch): a manifest known to have no `lesionAtlas` entry at all disables
   * the lesion-effect radio up front, before the user ever tries selecting
   * it — the same way `loadPositions`'s "no-entry" outcome disables the
   * whole panel. A manifest that *does* have an entry stays enabled until an
   * actual load attempt (lazy — see `ensureLesionAtlasLoaded`) proves it
   * `invalid`.
   */
  const lesionAtlasEntryMissing = $derived(manifest !== undefined && !manifest.lesionAtlas);
  const lesionOptionDisabledReason = $derived.by(() => {
    if (lesionAtlasEntryMissing) return 'no lesion atlas was shipped with this build';
    if (lesionAtlasStatus && lesionAtlasStatus.status !== 'ok') return lesionAtlasStatus.reason;
    return undefined;
  });

  /** Which atlas graph (if any) each arm's current topology maps to — the plan's "topology mapping incl. disconnected no-data", also exposed as `data-lesion-source-{left,right}` below for e2e coverage. */
  const lesionSourceLeft = $derived(lesionAtlasGraphKeyForTopology(topology.left) ?? 'none');
  const lesionSourceRight = $derived(lesionAtlasGraphKeyForTopology(topology.right) ?? 'none');

  /** Screen-reader-only summary of FDR-significant neuron counts per arm while lesion mode is active — the plan's accessibility non-negotiable ("sr-only summary… counts of FDR-significant neurons shown"), independent of the WebGL canvas the rest of this mode's content lives on. */
  const lesionSignificantSummary = $derived.by(() => {
    if (colorMode !== 'lesion' || !lesionAtlasStatus || lesionAtlasStatus.status !== 'ok') return undefined;
    // Captured into a local `const` (rather than closing over the `let`
    // directly) so the `status === 'ok'` narrowing above survives into the
    // nested `describeArm` closure below.
    const okStatus = lesionAtlasStatus;
    const describeArm = (label: string, key: 'biological' | 'rewiredSeed0' | 'none'): string => {
      if (key === 'none') return `${label} arm: no lesion data (disconnected)`;
      const graph = okStatus.data.graphs[key];
      const count = graph.fdrSignificant.reduce((total: number, significant: boolean) => total + (significant ? 1 : 0), 0);
      const graphLabel = key === 'biological' ? 'biological' : 'rewired seed 0';
      return `${label} arm (${graphLabel}): ${count} of ${graph.fdrSignificant.length} neurons FDR-significant`;
    };
    return `Lesion effect mode. ${describeArm('Left', lesionSourceLeft)}. ${describeArm('Right', lesionSourceRight)}.`;
  });

  const handleReducedMotionChange = (event: MediaQueryListEvent): void => {
    reducedMotion = event.matches;
    scene?.setReducedMotion(event.matches);
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
    sceneReady = false;
    closing?.dispose();
    lastRatesSeen.left = undefined;
    lastRatesSeen.right = undefined;
  };

  const frame = (nowMs: number): void => {
    if (destroyed || !scene) return;
    try {
      // Lesion mode is static (colors come from `setStaticColors`, applied
      // by the `$effect` below whenever `colorMode`/topology change) — the
      // whole rates-polling block is skipped entirely while it's active, not
      // merely left to no-op inside `scene.update()`/`scene.clear()` (both
      // are no-ops in lesion mode too — see `ActivityScene.ts` — but relying
      // on that alone would still call `setActivityStreaming`-adjacent
      // bookkeeping like `lastRatesSeen`/`streamingSupported` every frame
      // for no reason, and `scene.clear()` firing here would otherwise only
      // be *coincidentally* harmless rather than structurally impossible).
      if (colorMode === 'live') {
        if (runner) {
          // Cheap boolean check (no allocation), independent of the
          // reduced-motion color-update throttle below — support can change
          // (e.g. across a topology switch) whether or not a fresh rate
          // happens to be due this frame.
          for (const agentId of ['left', 'right'] as const) {
            const supported = runner.supportsActivityStreaming(agentId);
            if (supported !== streamingSupported[agentId]) streamingSupported[agentId] = supported;
          }
        }
        const throttled = reducedMotion && nowMs - lastColorUpdateMs < REDUCED_MOTION_UPDATE_INTERVAL_MS;
        if (!throttled && runner) {
          lastColorUpdateMs = nowMs;
          for (const agentId of ['left', 'right'] as const) {
            const rates = runner.getLatestRates(agentId);
            if (rates && rates !== lastRatesSeen[agentId]) {
              lastRatesSeen[agentId] = rates;
              scene.update(agentId, rates);
              colorSource[agentId] = 'live';
              lastUpdateTick[agentId] = telemetry?.tick ?? lastUpdateTick[agentId];
            } else if (!rates && lastRatesSeen[agentId]) {
              // The arm had rates and now doesn't (e.g. `runner.reset()`
              // cleared `latestRates`, or this arm's binding stopped
              // supporting streaming mid-topology-switch) — repaint it to the
              // neutral "no data" color rather than leaving the previous
              // run's final colors on screen under a "Computed rate" label
              // that no longer describes them.
              lastRatesSeen[agentId] = undefined;
              scene.clear(agentId);
              colorSource[agentId] = 'live';
            }
          }
        }
      }
      scene.render(nowMs);
    } catch (error) {
      // Mirrors `App.svelte`'s own render-loop guard (`frame`'s doc
      // comment there): a throw here must not just silently freeze the
      // canvas on its last frame while streaming quietly stays on.
      console.error('Activity view render loop stopped unexpectedly', error);
      sceneError = `The activity view stopped: ${error instanceof Error ? error.message : String(error)}`;
      teardown();
      return;
    }
    rafId = requestAnimationFrame(frame);
  };

  const expand = async (): Promise<void> => {
    if (!runner || !positionsStatus || positionsStatus.status !== 'ok') return;
    const generation = ++openGeneration;
    expanded = true;
    sceneError = undefined;
    contextLostMessage = undefined;

    let module: typeof import('../render/ActivityScene');
    try {
      module = await import('../render/ActivityScene');
    } catch (error) {
      if (isStaleExpand(generation)) return;
      // Deliberately leaves `expanded` true: the fallback placeholder below
      // only renders inside the `{#if expanded}` block, matching
      // `App.svelte`'s own WebGL-unavailable fallback, which stays visible
      // (not collapsed away) so the reason is actually shown. Clicking
      // "Collapse" (still available — see `collapse()`) is how the user
      // dismisses/retries it.
      sceneError = `The activity renderer failed to load (${error instanceof Error ? error.message : String(error)}).`;
      return;
    }
    if (isStaleExpand(generation)) return;
    // `expanded = true` (set above) only *schedules* the `{#if expanded}`
    // canvas to mount — `await tick()` flushes that pending DOM update so
    // `canvasEl` is deterministically bound by the time it's read below,
    // rather than relying on the dynamic `import()` having already yielded
    // enough microtasks for Svelte's own effect flush to land first.
    await tick();
    if (isStaleExpand(generation) || !canvasEl) return;

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
          // Deferred, matching `App.svelte`'s equivalent `ArenaScene`
          // handling: `teardown()` -> `scene.dispose()` -> `renderer.dispose()`
          // would otherwise run synchronously from inside the
          // `webglcontextlost` event's own dispatch, removing Three's own
          // listener for that same event mid-dispatch.
          queueMicrotask(() => teardown());
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

    lastUpdateTick.left = 0;
    lastUpdateTick.right = 0;
    // A fresh `ActivityScene` always starts in its own internal `'live'`
    // mode (`ActivityScene.ts`'s own default) — sync it to whatever mode
    // this panel is already in (colorMode persists across a collapse ->
    // expand cycle, e.g. a user who chose Lesion effect, collapsed, then
    // reopened). The `$effect` below (which reads `sceneReady` and
    // `colorMode` unconditionally, so it reliably reacts to `sceneReady`
    // flipping true here) re-applies static colors for both arms if
    // `colorMode` is already `'lesion'`.
    scene.setMode(colorMode);
    sceneReady = true;
    // Live-mode streaming is skipped entirely while already in lesion mode
    // (this panel's own `frame()` never polls rates then either) — starting
    // it here anyway would immediately be followed by a
    // `setActivityStreaming(false)` the next time lesion-mode logic ran;
    // simplest to just not enable it in the first place. The render loop
    // itself still starts regardless of mode (`scene.render()` — camera
    // responsiveness/damping must keep working in lesion mode too).
    if (colorMode === 'lesion') {
      if (isStaleExpand(generation)) return;
      rafId = requestAnimationFrame(frame);
      return;
    }
    await runner.setActivityStreaming(true);
    if (isStaleExpand(generation)) {
      // Deliberately does NOT call `teardown()` here: whatever `collapse()`
      // call invalidated this generation already tore down *this* call's
      // own scene (it was still the current `scene` at the moment that
      // collapse ran, since nothing else can assign `scene` without an
      // `await` in between). A later `expand()` may by now already own a
      // newer scene of its own — calling `teardown()` again here would
      // dispose *that* one out from under it instead of anything this call
      // built.
      return;
    }
    rafId = requestAnimationFrame(frame);
  };

  const collapse = (): void => {
    openGeneration += 1;
    expanded = false;
    teardown();
  };

  const toggle = (): void => {
    if (toggleDisabled) return;
    if (expanded) collapse();
    else void expand();
  };

  /**
   * Fetches, sha256-verifies, and cross-checks the lesion atlas the first
   * time it is needed (WP3's "load the atlas only when the mode is first
   * selected" non-negotiable) and memoizes the result so a later mode
   * switch, or the reactive `$effect` below re-applying colors after a
   * topology switch, never re-fetches. Returns `undefined` (never a
   * `status`) only when `manifest`/`biologicalGraph` have not arrived from
   * `App.svelte` yet — a real, if narrow, race the caller must also handle,
   * since a user could in principle click the radio before `onManifest` has
   * fired.
   */
  const ensureLesionAtlasLoaded = async (): Promise<LesionAtlasLoadResult | undefined> => {
    if (lesionAtlasStatus) return lesionAtlasStatus;
    if (!manifest || !biologicalGraph) return undefined;
    if (!lesionAtlasLoadPromise) {
      lesionAtlasTransientReason = undefined; // a fresh attempt supersedes any earlier transient failure hint
      lesionAtlasLoadPromise = loadLesionAtlas(manifest, `${import.meta.env.BASE_URL}data`, biologicalGraph).catch(
        (error: unknown): LesionAtlasLoadResult => ({
          status: 'invalid',
          reason: `unexpected error while loading the lesion atlas: ${error instanceof Error ? error.message : String(error)}`
        })
      );
    }
    const result = await lesionAtlasLoadPromise;
    if (destroyed) return result;
    if (result.status === 'unavailable') {
      // Round-2 dual review (Important): a fetch/network failure is
      // retryable, unlike a genuinely missing manifest entry or a hash/
      // shape failure — clearing the memoized promise (but not returning
      // the result as-is) lets the *next* mode selection try the fetch
      // again instead of permanently disabling the mode for the rest of
      // the session over one dropped request.
      lesionAtlasLoadPromise = undefined;
      lesionAtlasTransientReason = result.reason;
    } else {
      lesionAtlasStatus = result;
    }
    return result;
  };

  /**
   * Paints `agentId`'s static lesion-effect colors for `graphMode` (that
   * arm's *current* topology, passed explicitly rather than re-read from
   * `topology[agentId]` so the `$effect` below can establish its own
   * reactive dependency on the exact value it already captured). A no-op if
   * the scene isn't ready yet or the atlas hasn't loaded successfully —
   * both are covered by this function's only caller, the `$effect` below,
   * which gates on both first.
   */
  const applyLesionColorsForArm = (agentId: AgentId, graphMode: GraphMode): void => {
    if (!scene || !lesionAtlasStatus || lesionAtlasStatus.status !== 'ok') return;
    const key = lesionAtlasGraphKeyForTopology(graphMode);
    if (!key) {
      // Disconnected (or any future topology the atlas doesn't cover):
      // honest "no lesion data" — never a fabricated color.
      scene.setNoLesionData(agentId);
    } else {
      const graph = lesionAtlasStatus.data.graphs[key];
      scene.setStaticColors(agentId, graph.effect, graph.fdrSignificant, lesionAtlasStatus.absMax);
    }
    colorSource[agentId] = 'lesion';
  };

  /**
   * Re-applies both arms' static lesion colors whenever lesion mode is
   * active, the scene is ready, and either arm's topology changes — the
   * plan's "a topology switch while in lesion mode re-applies the static
   * colors for that arm". Every dependency (`colorMode`, `sceneReady`,
   * `topology.left`, `topology.right`) is read up front, before the
   * early-return branch — clearer to read at a glance than relying on
   * Svelte 5's own per-run dependency tracking (which, for the record, does
   * still correctly resubscribe on a short-circuited `if (a || b) return`
   * whenever `a` — the value that caused *that* run's early return — later
   * changes; the failure mode this structure guards against would only be a
   * *later* dependency read inside a branch that never executes, not the
   * short-circuit itself).
   */
  $effect(() => {
    const mode = colorMode;
    const ready = sceneReady;
    const leftTopology = topology.left;
    const rightTopology = topology.right;
    if (mode !== 'lesion' || !ready) return;
    applyLesionColorsForArm('left', leftTopology);
    applyLesionColorsForArm('right', rightTopology);
  });

  /**
   * A native radio input flips its own `checked` DOM property (and unchecks
   * its named-group sibling) synchronously on click/keyboard selection,
   * *before* Svelte's own reactive `checked={colorMode === value}` binding
   * ever runs. That reactive binding only re-renders when `colorMode`
   * itself actually changes — so if `switchColorMode` below ends up staying
   * on the current mode (e.g. the lazily-loaded atlas turns out
   * missing/invalid), `colorMode` never changes value, Svelte has no
   * dependency change to react to, and the browser is left showing the
   * *disabled* lesion radio as visually checked while the app's own state
   * still says Live — a real, reproduced bug (caught by this WP's own e2e
   * coverage). Both radios' native `checked` are force-synced back to the
   * *current* (pre-switch) `colorMode` right here, synchronously, before
   * `switchColorMode` runs; if it does end up changing `colorMode` for
   * real, the reactive `checked={...}` bindings correct both radios again
   * on the next render, same as always.
   */
  const handleColorModeInputChange = (next: 'live' | 'lesion'): void => {
    if (liveRadioEl) liveRadioEl.checked = colorMode === 'live';
    if (lesionRadioEl) lesionRadioEl.checked = colorMode === 'lesion';
    void switchColorMode(next);
  };

  /**
   * Bumped on every `switchColorMode` call and captured as `request` at its
   * start; checked again after the one `await` inside it. Round-2 dual
   * review (Important, race condition): switching to Lesion awaits the lazy
   * atlas load (`ensureLesionAtlasLoaded`, a real network round trip on
   * first selection) — without this guard, a *second* switch attempt that
   * lands while the first is still pending (e.g. a double-click on Lesion —
   * `handleColorModeInputChange`'s own DOM "unstick" reset means a plain
   * second click on the already-visually-checked Live radio does not
   * re-fire a native `change` event mid-load, so this specific window is
   * reached by a re-entrant/duplicate switch request more than by a literal
   * second click on the other radio) would see the earlier call's stale
   * resume silently overwrite whatever the later call decided, once it
   * finally resumes: `switchColorMode('live')` (or a second
   * `switchColorMode('lesion')`) started while the first is still pending
   * has no way to mark that first call's eventual resume as superseded
   * without this counter. Same discipline as `openGeneration`/
   * `isStaleExpand` above, for the same class of "an awaited call resumes
   * after a newer one already changed what the user wants" bug.
   */
  let colorModeRequest = 0;

  /**
   * Switches the activity view's color mode. Switching to lesion effect
   * lazily loads the atlas (`ensureLesionAtlasLoaded`) if needed; if that
   * load doesn't succeed, the mode stays on Live and `lesionOptionDisabledReason`
   * (derived above) now explains why. Switching either direction disables
   * live rate streaming while lesion mode is active (`setActivityStreaming(false)`).
   * Returning to Live also repaints both arms to the neutral "no data" color
   * (`scene.clear()`) *before* resetting `lastRatesSeen` — round-2 dual
   * review (Important): without an explicit `clear()` here, an experiment
   * that is `ready`/`paused` (no ticks arriving) would never call
   * `frame()`'s own `update()`/`clear()` branches again, leaving the lesion
   * mode's static diverging colors on screen indefinitely under the "Color:
   * Computed rate" label and viridis legend — a real mislabeled-provenance
   * state. Resetting `lastRatesSeen` after `clear()` still ensures the very
   * next frame that *does* have fresh rates repaints from them rather than
   * skipping a repaint because the last-seen array reference happens to be
   * unchanged.
   */
  const switchColorMode = async (next: 'live' | 'lesion'): Promise<void> => {
    const request = ++colorModeRequest;
    if (colorMode === next) return;
    if (next === 'lesion') {
      const result = await ensureLesionAtlasLoaded();
      if (destroyed || request !== colorModeRequest) return; // superseded by a later color-mode choice
      if (!result || result.status !== 'ok') return; // stays on Live; lesionOptionDisabledReason now explains why
    }
    colorMode = next;
    scene?.setMode(next);
    if (next === 'live') {
      scene?.clear('left');
      scene?.clear('right');
      lastRatesSeen.left = undefined;
      lastRatesSeen.right = undefined;
      colorSource.left = 'live';
      colorSource.right = 'live';
      void runner?.setActivityStreaming(true);
    } else {
      void runner?.setActivityStreaming(false);
    }
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

    <fieldset class="field color-mode-field" disabled={!sceneReady}>
      <legend>Color</legend>
      <label class="radio-option">
        <input
          bind:this={liveRadioEl}
          type="radio"
          name="activity-color-mode"
          value="live"
          checked={colorMode === 'live'}
          onchange={() => handleColorModeInputChange('live')}
        />
        Live rate
      </label>
      <label class="radio-option">
        <input
          bind:this={lesionRadioEl}
          type="radio"
          name="activity-color-mode"
          value="lesion"
          checked={colorMode === 'lesion'}
          disabled={lesionOptionDisabledReason !== undefined}
          onchange={() => handleColorModeInputChange('lesion')}
        />
        Lesion effect (offline)
      </label>
      {#if colorMode === 'live' && lesionOptionDisabledReason}
        <!-- `aria-live="polite"` (matching `ExperimentPanel.svelte`'s decoder
             fieldset precedent): announces why the lesion option is
             unavailable to assistive tech, not just sighted users reading
             the fieldset's visible hint text. -->
        <p class="hint" aria-live="polite">Lesion effect (offline) unavailable: {lesionOptionDisabledReason}</p>
      {:else if colorMode === 'live' && lesionAtlasTransientReason}
        <!-- A fetch/network failure, not a verification failure — the radio
             stays enabled (see `lesionAtlasTransientReason`'s own doc
             comment) since selecting the mode again retries the fetch. -->
        <p class="hint" aria-live="polite">Lesion effect (offline) could not be loaded (retrying is available): {lesionAtlasTransientReason}</p>
      {/if}
    </fieldset>

    {#if colorMode === 'lesion'}
      <p class="labels lesion-label">
        Lesion effect: <strong>Computed (offline)</strong> — effect on this model's score when this neuron's rate is
        clamped to 0 for the whole episode (authored decoder, opponent parked, 100 held-out seeds). FDR q&nbsp;=&nbsp;0.05;
        faded/outlined neurons did not survive false-discovery correction and are not reliable effects. Sensory
        neurons are this model's hand-wired encoder inputs. This is not a claim about the real fly's neural
        function — see the
        <a href={LESION_REPORT_URL} target="_blank" rel="noreferrer">full report</a>.
      </p>
      {#if lesionSourceLeft === 'none'}
        <p class="coverage streaming-unavailable" role="status">Left arm: no lesion data (disconnected).</p>
      {/if}
      {#if lesionSourceRight === 'none'}
        <p class="coverage streaming-unavailable" role="status">Right arm: no lesion data (disconnected).</p>
      {/if}
      {#if lesionSignificantSummary}
        <p class="sr-only" aria-live="polite">{lesionSignificantSummary}</p>
      {/if}
    {/if}

    {#if positionsStatus?.status === 'ok'}
      <p class="coverage">
        Positioned: {positionsStatus.positions.coverage.soma} soma, {positionsStatus.positions.coverage.tosoma} soma-tract,
        {positionsStatus.positions.coverage.none} unavailable
        <span class="units">(units: {positionsStatus.positions.units})</span>
      </p>
      {#if positionsStatus.positions.coverage.none > 0}
        <p class="coverage strip-label">
          The rows of points along the bottom of each arm are the
          <strong>position unavailable</strong> strip — {positionsStatus.positions.coverage.none} neurons with no soma
          annotation, laid out for visibility only. It is not an anatomical location.
        </p>
      {/if}
    {/if}

    {#if !streamingSupported.left}
      <p class="coverage streaming-unavailable" role="status">Left arm: streaming unavailable for this arm.</p>
    {/if}
    {#if !streamingSupported.right}
      <p class="coverage streaming-unavailable" role="status">Right arm: streaming unavailable for this arm.</p>
    {/if}

    <div class="canvas-region">
      <canvas
        bind:this={canvasEl}
        class="activity-canvas"
        aria-label="Neural activity at soma positions"
        data-last-update-tick-left={lastUpdateTick.left}
        data-last-update-tick-right={lastUpdateTick.right}
        data-color-source-left={colorSource.left}
        data-color-source-right={colorSource.right}
        data-lesion-source-left={lesionSourceLeft}
        data-lesion-source-right={lesionSourceRight}
        style:visibility={sceneError ? 'hidden' : 'visible'}
      ></canvas>
      {#if sceneError}
        <!--
          `role="img"` with a static `aria-label` previously wrapped this
          box, which tells assistive tech to treat the whole subtree as one
          image whose accessible name is that fixed string — the actual,
          specific `sceneError` text inside was never reliably exposed to
          screen readers, and this wasn't a live region either (thermo
          maintainability I2). Dropped in favor of `role="alert"` directly
          on the paragraph carrying the real message, matching the
          `contextLostMessage` pattern immediately below.
        -->
        <div class="canvas-placeholder">
          <p role="alert">{sceneError}</p>
        </div>
      {/if}
    </div>

    {#if contextLostMessage}
      <p class="context-lost" role="alert">{contextLostMessage}</p>
    {/if}

    {#if positionsStatus?.status === 'ok' && colorMode === 'live'}
      <div class="legend">
        <div class="legend-scale" aria-hidden="true">
          <span>{positionsStatus.rateMin.toFixed(2)}</span>
          <span class="legend-bar"></span>
          <span>{positionsStatus.rateMax.toFixed(2)}</span>
        </div>
        <!-- Matches `ActivityScene.ts#NO_DATA_COLOR` — grey, deliberately
             outside the viridis bar above, shown before the first tick and
             whenever an arm's rates go from present to absent (e.g. after
             a reset). -->
        <span class="legend-no-data" aria-hidden="true"><span class="swatch"></span>No data yet</span>
        <ul class="legend-roles">
          <li><span class="shape circle" aria-hidden="true"></span>Sensory</li>
          <li><span class="shape square" aria-hidden="true"></span>Bridge</li>
          <li><span class="shape triangle" aria-hidden="true"></span>Descending</li>
        </ul>
      </div>
    {:else if colorMode === 'lesion' && lesionAtlasStatus?.status === 'ok'}
      {@const absMax = lesionAtlasStatus.absMax}
      <div class="legend lesion-legend">
        <div class="legend-scale" aria-hidden="true">
          <span>{(-absMax).toFixed(3)}</span>
          <span class="legend-bar diverging"></span>
          <span>+{absMax.toFixed(3)}</span>
        </div>
        <span class="legend-caption">score change when silenced (0 = no effect)</span>
        <!-- Non-color-only FDR-significance marker: `ActivityScene.ts`'s
             outline ring, drawn at every non-FDR-significant neuron's
             position (shape, not hue) — the color blend
             `activity-layout.ts#writeEffectColors` also applies is a second,
             color-based cue for the same fact, never the only one. -->
        <span class="legend-outline" aria-hidden="true"><span class="swatch outline"></span>Outlined/faded = not FDR-significant (q=0.05)</span>
        <span class="legend-no-data" aria-hidden="true"><span class="swatch"></span>No lesion data (disconnected)</span>
      </div>
    {/if}
  {/if}
</section>

<style>
  /* `App.svelte`'s `<main>` is a two-column grid (arena + sidebar); this
     panel sits directly below the arena, in the arena's own column, per the
     plan's placement (`03-activity-view.md`). Deliberately column 1 only
     (not `1 / -1`): a full-width span here would collide with `.sidebar`'s
     own `grid-row: 1 / span 2` pin in `src/app.css` and push this panel to
     a third row instead of sitting under the arena — see that rule's own
     comment for the exact placement trace.

     `align-self: start` (overriding the grid's default `stretch`): row 2 is
     `main`'s `1fr` track, sized to whatever height `.sidebar`'s two-row
     span needs beyond the arena's own row-1 height (see `main`'s own
     comment in `src/app.css`) — usually taller than this panel's actual
     content, especially collapsed. Without `align-self: start`, default
     stretch alignment fills that whole row-2 height with this panel
     regardless, leaving a large empty bordered box below the arena — a
     real regression a round-2 review caught and confirmed in a browser. */
  .activity {
    grid-column: 1;
    align-self: start;
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

  .streaming-unavailable {
    color: #ffcf8a;
  }

  .canvas-region {
    /* Sizing (a definite `height`, not `aspect-ratio`) comes from the
       shared, unscoped `.canvas-region` rule in `src/app.css` — the same
       one `App.svelte`'s arena canvas uses, and for the same reason (see
       that rule's own comment): a definite height is what lets the canvas
       element's percentage height resolve at all, so `ActivityScene.resize()`
       sizes it correctly instead of the canvas growing unboundedly on
       repeated resizes. An `aspect-ratio` declared here would be silently
       overridden by that definite height and do nothing, so it is
       deliberately omitted rather than left in as dead CSS. */
    position: relative;
    margin-top: 0.8rem;
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

  .legend-no-data {
    display: flex;
    align-items: center;
    gap: 0.35rem;
  }

  .legend-no-data .swatch {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 2px;
    /* Matches `ActivityScene.ts#NO_DATA_COLOR` (0.32, 0.35, 0.4 in linear
       0-1 RGB) converted to an sRGB hex approximation for CSS. */
    background: #6b7484;
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

  /* Matches `ExperimentPanel.svelte`'s decoder-fieldset styling (`.field`/
     `.radio-option`/`.hint`) — Svelte's per-component style scoping means
     that CSS is not reachable from here, so it is intentionally duplicated
     rather than shared. */
  .field {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    margin-top: 0.75rem;
  }

  .color-mode-field {
    border: 1px solid #304355;
    border-radius: 0.4rem;
    padding: 0.5rem 0.6rem 0.65rem;
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }

  .color-mode-field legend {
    padding: 0 0.3rem;
    color: #cbd8e7;
    font-size: 0.82rem;
  }

  .radio-option {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    color: #edf4ff;
    font-size: 0.85rem;
  }

  .hint {
    margin: 0.2rem 0 0;
    color: #ffd7de;
    font-size: 0.75rem;
  }

  .lesion-label {
    color: #cbd8e7;
  }

  .lesion-label a {
    color: #79d8d0;
  }

  .legend-caption {
    color: #9aacc2;
    font-size: 0.75rem;
  }

  .legend-bar.diverging {
    background: linear-gradient(to right, #0072b2, #ffffff, #d55e00);
  }

  .legend-outline {
    display: flex;
    align-items: center;
    gap: 0.35rem;
  }

  .legend-outline .swatch.outline {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: transparent;
    border: 2px solid #ffe08a;
  }

  /* Visually hidden, never `display:none` — same "sr-only" clip pattern
     `NullHistogram.svelte`'s own `.sr-only` rule documents, kept reachable
     by screen-reader/keyboard navigation while invisible on screen. */
  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }
</style>
