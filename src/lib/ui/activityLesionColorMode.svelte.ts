import type { AgentId } from '../arena/types';
import type { ArenaManifest } from '../experiment/assets';
import type { ExperimentRunner } from '../experiment/runner';
import type { ConnectomeGraph, GraphMode } from '../connectome/format';
import type { ActivityScene as ActivitySceneInstance } from '../render/ActivityScene';
import { lesionAtlasGraphKeyForTopology, loadLesionAtlas, type LesionAtlasGraphKey, type LesionAtlasLoadResult } from '../experiment/lesionAtlas';

/**
 * The lesion-effect color mode's load/retry/stale-guard state machine (WP3
 * of `.agents/plans/lesion-atlas`), extracted out of `ActivityPanel.svelte`
 * (thermo-architecture/thermo-maintainability review: that file crossed the
 * 1000-line threshold by inlining this alongside its own WebGL scene
 * lifecycle and live-rate streaming loop — `LesionColorMode.svelte`'s own doc
 * comment covers the presentational half this module is paired with, the
 * same `NullHistogram.svelte`/`LedgerPanel.svelte` "extract the offline-
 * artifact concern" precedent this codebase already established). A
 * `.svelte.ts` runes module (not a class, not a component) so this state
 * machine is independently unit-testable — mount no component, drive
 * `loadLesionAtlas`'s mock directly, assert against the returned reactive
 * object — the "runes in a plain module" pattern Svelte 5 documents for
 * extracting component-independent reactive logic.
 *
 * `createLesionColorMode` must be called during a component's (or a test's
 * `$effect.root`'s) synchronous setup — same rule as any other rune usage —
 * since it creates one `$effect` (the topology/mode/sceneReady-driven
 * re-apply below) that needs an enclosing effect-root scope to attach to and
 * clean up with. It deliberately does *not* create its own `$effect.root`:
 * doing so would detach the re-apply effect from the caller's own lifecycle
 * (`ActivityPanel.svelte`'s `onDestroy`), leaking it past component teardown.
 *
 * Every dependency this module cannot own itself — `scene` (imperative,
 * built by `ActivityPanel.svelte#expand()`), `sceneReady`, `runner`,
 * `manifest`, `biologicalGraph`, `topology`, and whether the host has been
 * destroyed — is threaded through as a getter function (`LesionColorModeHost`),
 * not a snapshot value, so every read inside this module's derived
 * values/effect stays live against the host's own reactive state. Two
 * callbacks (`onEnterLive`/`onArmPainted`) hand back to the host the two bits
 * of *live-mode* bookkeeping (`lastRatesSeen`/`colorSource`) that still
 * belong to `ActivityPanel.svelte`, not this module — this module never reads
 * or writes them directly.
 */

export type ActivityColorMode = 'live' | 'lesion';

export interface LesionColorModeHost {
  /** `ActivityPanel.svelte`'s current `scene`, or `undefined` before `expand()` finishes / after `teardown()`. */
  scene: () => ActivitySceneInstance | undefined;
  /** Mirrors `ActivityPanel.svelte`'s own `sceneReady` — gates the re-apply effect the same way the fieldset's `disabled` prop does. */
  sceneReady: () => boolean;
  runner: () => ExperimentRunner | undefined;
  manifest: () => ArenaManifest | undefined;
  biologicalGraph: () => ConnectomeGraph | undefined;
  topology: () => Record<AgentId, GraphMode>;
  /** Mirrors `ActivityPanel.svelte`'s own `destroyed` flag — an in-flight `ensureLesionAtlasLoaded` must never write state after the host unmounts. */
  destroyed: () => boolean;
  /** Called once, synchronously, when `switchColorMode` actually transitions to Live — the host resets `lastRatesSeen`/`colorSource`, bookkeeping this module never touches. */
  onEnterLive: () => void;
  /** Called once per arm every time this module (re)paints that arm's static colors — the host mirrors it into its own `colorSource` debug state. */
  onArmPainted: (agentId: AgentId) => void;
}

export interface LesionColorMode {
  readonly colorMode: ActivityColorMode;
  readonly lesionAtlasStatus: LesionAtlasLoadResult | undefined;
  readonly lesionAtlasTransientReason: string | undefined;
  readonly lesionOptionDisabledReason: string | undefined;
  readonly lesionSourceLeft: LesionAtlasGraphKey | 'none';
  readonly lesionSourceRight: LesionAtlasGraphKey | 'none';
  readonly lesionSignificantSummary: string | undefined;
  switchColorMode(next: ActivityColorMode): Promise<void>;
}

export const createLesionColorMode = (host: LesionColorModeHost): LesionColorMode => {
  let colorMode = $state<ActivityColorMode>('live');
  /** `undefined` until the lesion-effect mode is first selected (lazy load — see `ensureLesionAtlasLoaded`). Never set to an `'unavailable'` result — see `lesionAtlasTransientReason` below. */
  let lesionAtlasStatus = $state<LesionAtlasLoadResult | undefined>(undefined);
  /**
   * The reason from the most recent `'unavailable'` (fetch/network failure)
   * outcome, shown as a non-blocking hint — unlike `lesionAtlasStatus`, a
   * fetch failure is retryable, so it must never disable the radio the way
   * `lesionOptionDisabledReason` does for a genuinely missing entry or a
   * hash/shape failure. Cleared on any subsequent load attempt or a later
   * non-`'unavailable'` outcome.
   */
  let lesionAtlasTransientReason = $state<string | undefined>(undefined);
  /** Memoizes the in-flight/completed load — plain (non-reactive): only `lesionAtlasStatus` above needs to drive the UI. Cleared after an `'unavailable'` outcome so the next selection retries the fetch. */
  let lesionAtlasLoadPromise: Promise<LesionAtlasLoadResult> | undefined;
  /**
   * Bumped on every `switchColorMode` call and captured as `request` at its
   * start; checked again after the one `await` inside it. Switching to
   * Lesion awaits the lazy atlas load (`ensureLesionAtlasLoaded`, a real
   * network round trip on first selection) — without this guard, a *second*
   * switch attempt that lands while the first is still pending would see the
   * earlier call's stale resume silently overwrite whatever the later call
   * decided, once it finally resumes.
   */
  let colorModeRequest = 0;

  /**
   * Cheap, synchronous check against the already-loaded manifest (no
   * fetch): a manifest known to have no `lesionAtlas` entry at all disables
   * the lesion-effect radio up front, before the user ever tries selecting
   * it. A manifest that *does* have an entry stays enabled until an actual
   * load attempt (lazy — see `ensureLesionAtlasLoaded`) proves it `invalid`.
   */
  const lesionAtlasEntryMissing = $derived(host.manifest() !== undefined && !host.manifest()?.lesionAtlas);
  const lesionOptionDisabledReason = $derived.by(() => {
    if (lesionAtlasEntryMissing) return 'no lesion atlas was shipped with this build';
    if (lesionAtlasStatus && lesionAtlasStatus.status !== 'ok') return lesionAtlasStatus.reason;
    return undefined;
  });

  /** Which atlas graph (if any) each arm's current topology maps to. */
  const lesionSourceLeft = $derived(lesionAtlasGraphKeyForTopology(host.topology().left) ?? 'none');
  const lesionSourceRight = $derived(lesionAtlasGraphKeyForTopology(host.topology().right) ?? 'none');

  /** Screen-reader-only summary of FDR-significant neuron counts per arm while lesion mode is active — the plan's accessibility non-negotiable. */
  const lesionSignificantSummary = $derived.by(() => {
    if (colorMode !== 'lesion' || !lesionAtlasStatus || lesionAtlasStatus.status !== 'ok') return undefined;
    const okStatus = lesionAtlasStatus;
    const describeArm = (label: string, key: LesionAtlasGraphKey | 'none'): string => {
      if (key === 'none') return `${label} arm: no lesion data (disconnected)`;
      const graph = okStatus.data.graphs[key];
      const count = graph.fdrSignificant.reduce((total: number, significant: boolean) => total + (significant ? 1 : 0), 0);
      const graphLabel = key === 'biological' ? 'biological' : 'rewired seed 0';
      return `${label} arm (${graphLabel}): ${count} of ${graph.fdrSignificant.length} neurons FDR-significant`;
    };
    return `Lesion effect mode. ${describeArm('Left', lesionSourceLeft)}. ${describeArm('Right', lesionSourceRight)}.`;
  });

  /**
   * Fetches, sha256-verifies, and cross-checks the lesion atlas the first
   * time it is needed. An `'ok'`/`'missing'`/`'invalid'` result is memoized
   * into `lesionAtlasStatus` so a later mode switch, or the reactive
   * re-apply effect below after a topology switch, never re-fetches — but an
   * `'unavailable'` result (fetch/network failure) is deliberately *not*
   * memoized this way; see that branch below. Returns `undefined` (never a
   * `status`) only when `manifest`/`biologicalGraph` have not arrived yet.
   */
  const ensureLesionAtlasLoaded = async (): Promise<LesionAtlasLoadResult | undefined> => {
    if (lesionAtlasStatus) return lesionAtlasStatus;
    const manifest = host.manifest();
    const biologicalGraph = host.biologicalGraph();
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
    if (host.destroyed()) return result;
    if (result.status === 'unavailable') {
      lesionAtlasLoadPromise = undefined;
      lesionAtlasTransientReason = result.reason;
    } else {
      lesionAtlasStatus = result;
    }
    return result;
  };

  /**
   * Paints `agentId`'s static lesion-effect colors for `graphMode` (that
   * arm's *current* topology). A no-op if the scene isn't ready yet or the
   * atlas hasn't loaded successfully — both are covered by this function's
   * only caller, the re-apply effect below.
   */
  const applyColorsForArm = (agentId: AgentId, graphMode: GraphMode): void => {
    const scene = host.scene();
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
    host.onArmPainted(agentId);
  };

  /**
   * Re-applies both arms' static lesion colors whenever lesion mode is
   * active, the scene is ready, and either arm's topology changes. Every
   * dependency (`colorMode`, `host.sceneReady()`, `host.topology()`) is read
   * up front, before the early-return branch — clearer to read at a glance
   * than relying on Svelte 5's own per-run dependency tracking.
   */
  $effect(() => {
    const mode = colorMode;
    const ready = host.sceneReady();
    const topology = host.topology();
    const leftTopology = topology.left;
    const rightTopology = topology.right;
    if (mode !== 'lesion' || !ready) return;
    applyColorsForArm('left', leftTopology);
    applyColorsForArm('right', rightTopology);
  });

  /**
   * Switches the activity view's color mode. Switching to lesion effect
   * lazily loads the atlas (`ensureLesionAtlasLoaded`) if needed; if that
   * load doesn't succeed, the mode stays on Live and `lesionOptionDisabledReason`
   * (derived above) now explains why. Switching either direction disables
   * live rate streaming while lesion mode is active. Returning to Live also
   * repaints both arms to the neutral "no data" color (`scene.clear()`)
   * *before* handing control back to `host.onEnterLive()` — without an
   * explicit `clear()` here, an experiment that is `ready`/`paused` (no
   * ticks arriving) would never repaint again, leaving lesion mode's static
   * diverging colors on screen indefinitely under a "Color: Computed rate"
   * label.
   */
  const switchColorMode = async (next: ActivityColorMode): Promise<void> => {
    const request = ++colorModeRequest;
    if (colorMode === next) return;
    if (next === 'lesion') {
      const result = await ensureLesionAtlasLoaded();
      if (host.destroyed() || request !== colorModeRequest) return; // superseded by a later color-mode choice
      if (!result || result.status !== 'ok') return; // stays on Live; lesionOptionDisabledReason now explains why
    }
    colorMode = next;
    host.scene()?.setMode(next);
    if (next === 'live') {
      host.scene()?.clear('left');
      host.scene()?.clear('right');
      host.onEnterLive();
      void host.runner()?.setActivityStreaming(true);
    } else {
      void host.runner()?.setActivityStreaming(false);
    }
  };

  return {
    get colorMode() {
      return colorMode;
    },
    get lesionAtlasStatus() {
      return lesionAtlasStatus;
    },
    get lesionAtlasTransientReason() {
      return lesionAtlasTransientReason;
    },
    get lesionOptionDisabledReason() {
      return lesionOptionDisabledReason;
    },
    get lesionSourceLeft() {
      return lesionSourceLeft;
    },
    get lesionSourceRight() {
      return lesionSourceRight;
    },
    get lesionSignificantSummary() {
      return lesionSignificantSummary;
    },
    switchColorMode
  };
};
