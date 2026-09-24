<script lang="ts">
  import type { AgentId } from '../arena/types';
  import type { GraphMode } from '../connectome/format';
  import type { ExperimentStatus } from '../experiment/state';

  /**
   * Start/Pause/Reset, seed input, per-agent topology selectors, and the
   * replay download button (WP6 items 6). Fully keyboard-operable (native
   * `<button>`/`<input>`/`<select>` elements only, no custom widgets) and
   * every control carries an explicit label — see `tests/unit/ui-experiment-panel.test.ts`.
   *
   * Labels the two selectors by *slot* ("Left arm topology" / "Right arm
   * topology") rather than by any topology name: the selector's own value —
   * and, on the 3D canvas, `ArenaScene#setAgentTopology`'s label sprite —
   * are the only things that identify topology. The left/right body
   * shape+color pairing (`render/ArenaScene.ts`'s `AGENT_SLOT_ACCENT`) is
   * fixed for the scene's lifetime and identifies *which agent*, never
   * which topology is currently selected for it — labeling the selector
   * itself with a shape/topology name (e.g. the former "(BIO shape)"
   * suffix) risked implying the two were the same thing, which they are
   * not once a topology switch has happened.
   */

  interface Props {
    status: ExperimentStatus;
    errorMessage?: string;
    seed: number;
    topology: Record<AgentId, GraphMode>;
    /** True while running/loading — locks Start and Seed. Does *not* lock Pause/Reset (see `topologySwitchPending`): those two must stay clickable for the entire duration of a run, which is most of what `controlsLocked` being true actually means. */
    controlsLocked: boolean;
    /** True while any topology switch is in flight. Locks Pause/Reset (a switch always implies a reset already in progress; a fresh explicit reset/pause request would race it) in addition to their own status-based rule; see `topologyControlsLocked` for the selectors' own, stricter lock. */
    topologySwitchPending: boolean;
    /** `controlsLocked`, plus true whenever the run isn't idle at `ready`/`finished` — a topology switch is never allowed mid-run, including merely `paused`. */
    topologyControlsLocked: boolean;
    onStart: () => void;
    onPause: () => void;
    onReset: () => void;
    onSeedInput: (seed: number) => void;
    onTopologyChange: (agentId: AgentId, mode: GraphMode) => void;
    onDownloadReplay: () => void;
  }

  let {
    status,
    errorMessage,
    seed,
    topology,
    controlsLocked,
    topologySwitchPending,
    topologyControlsLocked,
    onStart,
    onPause,
    onReset,
    onSeedInput,
    onTopologyChange,
    onDownloadReplay
  }: Props = $props();

  const TOPOLOGY_OPTIONS: readonly { value: GraphMode; label: string }[] = [
    { value: 'biological', label: 'Biological (measured)' },
    { value: 'rewired', label: 'Rewired control (seed 0)' },
    { value: 'disconnected', label: 'Disconnected (negative control)' }
  ];

  const startLabel = $derived(status === 'paused' ? 'Resume' : 'Start');
  const canStartOrResume = $derived(status === 'ready' || status === 'paused');
  const canPauseNow = $derived(status === 'running');
  const canResetNow = $derived(
    status === 'ready' || status === 'running' || status === 'paused' || status === 'finished'
  );
  const canDownload = $derived(status === 'paused' || status === 'finished');

  const STATUS_LABEL: Record<ExperimentStatus, string> = {
    loading: 'Loading and verifying the connectome artifacts…',
    ready: 'Ready — press Start to begin the 90-second run',
    running: 'Running',
    paused: 'Paused',
    finished: 'Finished',
    error: 'Error'
  };

  /**
   * Commits on `change` (blur, or Enter), not `oninput`: applying every
   * keystroke would silently reset a paused or finished run mid-edit
   * before its replay could be downloaded. Ignores an empty field rather
   * than treating it as `0`, and normalizes with the same `>>> 0` the
   * world itself applies (`arena/world.ts`'s `normalizeSeed`), so the
   * value displayed here always matches the seed actually driving the run.
   */
  const handleSeedCommit = (event: Event): void => {
    const raw = (event.currentTarget as HTMLInputElement).value.trim();
    if (raw === '') return;
    const value = Number(raw);
    if (Number.isFinite(value)) onSeedInput(Math.trunc(value) >>> 0);
  };

  const handleTopologyChange = (agentId: AgentId) => (event: Event): void => {
    onTopologyChange(agentId, (event.currentTarget as HTMLSelectElement).value as GraphMode);
  };
</script>

<section class="panel" aria-labelledby="controls-heading">
  <div class="section-heading">
    <h2 id="controls-heading">Experiment controls</h2>
    <span aria-live="polite">{STATUS_LABEL[status]}</span>
  </div>

  {#if status === 'error' && errorMessage}
    <p class="error-message" role="alert">{errorMessage} Reload the page to try again.</p>
  {/if}

  <div class="button-row">
    <button type="button" onclick={onStart} disabled={!canStartOrResume || controlsLocked}>
      {startLabel}
    </button>
    <button type="button" onclick={onPause} disabled={!canPauseNow || topologySwitchPending}>Pause</button>
    <button type="button" onclick={onReset} disabled={!canResetNow || topologySwitchPending}>Reset</button>
  </div>

  <div class="field">
    <label for="experiment-seed">Seed</label>
    <input
      id="experiment-seed"
      name="seed"
      type="number"
      step="1"
      value={seed}
      disabled={controlsLocked}
      onchange={handleSeedCommit}
    />
  </div>

  <div class="field">
    <label for="topology-left">Left arm topology</label>
    <select
      id="topology-left"
      name="topology-left"
      value={topology.left}
      disabled={topologyControlsLocked}
      onchange={handleTopologyChange('left')}
    >
      {#each TOPOLOGY_OPTIONS as option (option.value)}
        <option value={option.value}>{option.label}</option>
      {/each}
    </select>
  </div>

  <div class="field">
    <label for="topology-right">Right arm topology</label>
    <select
      id="topology-right"
      name="topology-right"
      value={topology.right}
      disabled={topologyControlsLocked}
      onchange={handleTopologyChange('right')}
    >
      {#each TOPOLOGY_OPTIONS as option (option.value)}
        <option value={option.value}>{option.label}</option>
      {/each}
    </select>
  </div>

  <button type="button" class="replay-download" onclick={onDownloadReplay} disabled={!canDownload}>
    Download replay (config + score traces, no connectome)
  </button>
</section>

<style>
  .field {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    margin-top: 0.75rem;
  }

  .field label {
    color: #cbd8e7;
    font-size: 0.82rem;
  }

  .field input,
  .field select {
    padding: 0.45rem 0.6rem;
    border: 1px solid #304355;
    border-radius: 0.4rem;
    color: #edf4ff;
    background: #0e1826;
  }

  .replay-download {
    width: 100%;
    margin-top: 0.9rem;
  }

  .error-message {
    margin: 0 0 0.75rem;
    padding: 0.6rem 0.75rem;
    border: 1px solid #ef476f;
    border-radius: 0.4rem;
    color: #ffd7de;
    background: rgb(239 71 111 / 12%);
    font-size: 0.82rem;
  }

  button:disabled,
  input:disabled,
  select:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
</style>
