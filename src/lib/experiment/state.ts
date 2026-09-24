/**
 * Explicit, pure state machine for the closed-loop arena experiment (WP6
 * item 1). Deliberately has no async/timer/Worker dependency at all — every
 * transition is a plain function of `(state, event) -> state`, so it is
 * unit-testable independent of timing, fetch, or Worker plumbing. See
 * `tests/unit/experiment-state.test.ts`.
 *
 * `ExperimentRunner` (`./runner.ts`) is the only caller once a run exists:
 * it owns the actual async work (ticking, reset, topology swaps) and calls
 * `transition` to decide its own next `status` after each step.
 *
 * `ExperimentController` (`./controller.ts`) is a deliberate exception,
 * calling `transition` directly during its own pre-runner asset-loading
 * phase (`initialize()`'s `assetsFailed` paths) — there is nothing for a
 * not-yet-constructed runner to own at that point. See that method's doc
 * comment for why this is safe: every failure path *after* the runner
 * exists still goes through `ExperimentRunner#fail()` instead, so this
 * module's transition table is never bypassed once a run is live.
 */

export type ExperimentStatus = 'loading' | 'ready' | 'running' | 'paused' | 'finished' | 'error';

export type ExperimentEvent =
  /** Both graph artifacts fetched, integrity-checked, and parsed. */
  | { type: 'assetsReady' }
  /** Fetch failure, hash mismatch, or a malformed graph buffer. */
  | { type: 'assetsFailed' }
  | { type: 'start' }
  | { type: 'pause' }
  | { type: 'resume' }
  /** Returns to `ready` with a fresh world and zeroed neural state; the already-validated graphs are reused. */
  | { type: 'reset' }
  | { type: 'tickCompleted'; tick: number; totalTicks: number }
  /**
   * A Worker/oracle failure, or any other runtime failure the app routes
   * through `ExperimentRunner#fail()` — including a failed topology switch
   * (`App.svelte`'s `handleTopologyChange`) and a failed post-reset rebind.
   * Every live (non-`loading`) state accepts this, not just `running`/
   * `paused`: a topology switch or a stale reset can fail while the runner
   * is sitting at `ready`/`finished` between runs.
   */
  | { type: 'runtimeError' };

/**
 * The single authoritative transition table. An event with no listed
 * transition for the current state is a no-op (returns `state` unchanged)
 * rather than a throw: callers (e.g. a stray double-click on Start) should
 * not have to guard every dispatch, and an unreachable transition is not a
 * bug worth crashing the experiment over.
 */
const TRANSITIONS: Readonly<Record<ExperimentStatus, Partial<Record<ExperimentEvent['type'], ExperimentStatus>>>> = {
  loading: { assetsReady: 'ready', assetsFailed: 'error' },
  ready: { start: 'running', reset: 'ready', runtimeError: 'error' },
  running: { pause: 'paused', reset: 'ready', runtimeError: 'error' },
  paused: { resume: 'running', reset: 'ready', runtimeError: 'error' },
  finished: { reset: 'ready', runtimeError: 'error' },
  error: {}
};

export const transition = (state: ExperimentStatus, event: ExperimentEvent): ExperimentStatus => {
  if (event.type === 'tickCompleted') {
    // Only a 'running' tick can finish the run; a tick result arriving after
    // a pause/reset raced it (see runner.ts's in-flight-tick handling) must
    // not resurrect 'running' or fast-forward into 'finished'.
    if (state !== 'running') return state;
    return event.tick >= event.totalTicks ? 'finished' : 'running';
  }
  return TRANSITIONS[state][event.type] ?? state;
};

/** True once graphs are validated and idle at the start line — only 'ready' accepts Start; 'finished' requires an explicit Reset first. */
export const canStart = (state: ExperimentStatus): boolean => state === 'ready';
export const canPause = (state: ExperimentStatus): boolean => state === 'running';
export const canResume = (state: ExperimentStatus): boolean => state === 'paused';
export const canReset = (state: ExperimentStatus): boolean =>
  state === 'ready' || state === 'running' || state === 'paused' || state === 'finished';
