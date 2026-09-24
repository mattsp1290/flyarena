import { cleanup, fireEvent, render, screen, within } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import ExperimentPanel from '../../src/lib/ui/ExperimentPanel.svelte';
import TelemetryPanel from '../../src/lib/ui/TelemetryPanel.svelte';
import LedgerPanel from '../../src/lib/ui/LedgerPanel.svelte';
import type { ExperimentTelemetry } from '../../src/lib/experiment/runner';
import type { ArenaManifest, RewiringNullLoadResult, TrainedReadoutLoadResult } from '../../src/lib/experiment/assets';

afterEach(() => cleanup());

describe('ExperimentPanel accessibility and wiring', () => {
  const baseProps = () => ({
    status: 'ready' as const,
    errorMessage: undefined,
    seed: 1234,
    topology: { left: 'biological' as const, right: 'rewired' as const },
    controlsLocked: false,
    topologySwitchPending: false,
    decoderSwitchPending: false,
    topologyControlsLocked: false,
    decoder: 'authored' as const,
    decoderControlsLocked: false,
    trainedDecoderUnavailableReason: undefined,
    onStart: vi.fn(),
    onPause: vi.fn(),
    onReset: vi.fn(),
    onSeedInput: vi.fn(),
    onTopologyChange: vi.fn(),
    onDecoderChange: vi.fn(),
    onDownloadReplay: vi.fn()
  });

  it('exposes every control with an accessible name, all keyboard-operable native elements', () => {
    render(ExperimentPanel, baseProps());

    expect(screen.getByRole('button', { name: /^start$/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^pause$/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^reset$/i })).toBeInTheDocument();
    expect(screen.getByLabelText(/^seed$/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/left arm topology/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/right arm topology/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /download replay/i })).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: /^authored$/i })).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: /trained \(offline\)/i })).toBeInTheDocument();

    // Every interactive control here is a native <button>/<input>/<select>,
    // which are keyboard-operable (Tab/Enter/Space/arrow keys) by the
    // platform itself — no custom `div`-as-button widget to audit.
    for (const element of [
      ...screen.getAllByRole('button'),
      ...screen.getAllByRole('radio'),
      screen.getByLabelText(/^seed$/i),
      screen.getByLabelText(/left arm topology/i),
      screen.getByLabelText(/right arm topology/i)
    ]) {
      expect(['BUTTON', 'INPUT', 'SELECT']).toContain(element.tagName);
    }
  });

  it('the decoder radio group defaults to Authored checked, and calls onDecoderChange on selection', async () => {
    const props = baseProps();
    render(ExperimentPanel, props);

    const authored = screen.getByRole('radio', { name: /^authored$/i });
    const trained = screen.getByRole('radio', { name: /trained \(offline\)/i });
    expect(authored).toBeChecked();
    expect(trained).not.toBeChecked();

    await fireEvent.click(trained);
    expect(props.onDecoderChange).toHaveBeenCalledWith('trained');
  });

  it('decoderControlsLocked disables both decoder radios', () => {
    render(ExperimentPanel, { ...baseProps(), decoderControlsLocked: true });
    expect(screen.getByRole('radio', { name: /^authored$/i })).toBeDisabled();
    expect(screen.getByRole('radio', { name: /trained \(offline\)/i })).toBeDisabled();
  });

  it('shows the Trained option disabled with its reason when the trained-readout artifact is unavailable, without disabling Authored', () => {
    render(ExperimentPanel, {
      ...baseProps(),
      trainedDecoderUnavailableReason: 'trained-readout-v1.json sha256 mismatch'
    });
    expect(screen.getByRole('radio', { name: /^authored$/i })).toBeEnabled();
    expect(screen.getByRole('radio', { name: /trained \(offline\)/i })).toBeDisabled();
    expect(screen.getByText(/trained-readout-v1\.json sha256 mismatch/i)).toBeInTheDocument();
  });

  it('disables Start when not ready, and enables Pause only while running', () => {
    const { unmount } = render(ExperimentPanel, { ...baseProps(), status: 'running' });
    expect(screen.getByRole('button', { name: /^start$/i })).toBeDisabled();
    expect(screen.getByRole('button', { name: /^pause$/i })).toBeEnabled();
    unmount();

    render(ExperimentPanel, { ...baseProps(), status: 'ready' });
    expect(screen.getByRole('button', { name: /^start$/i })).toBeEnabled();
    expect(screen.getByRole('button', { name: /^pause$/i })).toBeDisabled();
  });

  it('shows Resume instead of Start while paused', () => {
    render(ExperimentPanel, { ...baseProps(), status: 'paused' });
    expect(screen.getByRole('button', { name: /^resume$/i })).toBeInTheDocument();
  });

  it('calls onStart/onPause/onReset/onDownloadReplay on click', async () => {
    const props = { ...baseProps(), status: 'paused' as const };
    render(ExperimentPanel, props);

    await fireEvent.click(screen.getByRole('button', { name: /^resume$/i }));
    expect(props.onStart).toHaveBeenCalledTimes(1);

    await fireEvent.click(screen.getByRole('button', { name: /^reset$/i }));
    expect(props.onReset).toHaveBeenCalledTimes(1);

    await fireEvent.click(screen.getByRole('button', { name: /download replay/i }));
    expect(props.onDownloadReplay).toHaveBeenCalledTimes(1);
  });

  it('commits the seed on change (not every keystroke) with a parsed, unsigned-normalized integer, and ignores an empty commit', async () => {
    const props = baseProps();
    render(ExperimentPanel, props);
    const seedInput = screen.getByLabelText(/^seed$/i);

    // typing alone (input events) must not commit — only change/blur does.
    await fireEvent.input(seedInput, { target: { value: '99' } });
    expect(props.onSeedInput).not.toHaveBeenCalled();

    await fireEvent.change(seedInput, { target: { value: '99' } });
    expect(props.onSeedInput).toHaveBeenCalledWith(99);

    await fireEvent.change(seedInput, { target: { value: '-1' } });
    // Normalized the same way `arena/world.ts` normalizes a seed (`>>> 0`),
    // so the displayed value always matches what actually drives the run.
    expect(props.onSeedInput).toHaveBeenCalledWith(4294967295);

    props.onSeedInput.mockClear();
    await fireEvent.change(seedInput, { target: { value: '' } });
    expect(props.onSeedInput).not.toHaveBeenCalled();
  });

  it('calls onTopologyChange with the agent id and mode', async () => {
    const props = baseProps();
    render(ExperimentPanel, props);

    await fireEvent.change(screen.getByLabelText(/left arm topology/i), { target: { value: 'disconnected' } });
    expect(props.onTopologyChange).toHaveBeenCalledWith('left', 'disconnected');
  });

  it('shows the error message with recovery guidance as an alert only in the error state', () => {
    const { unmount } = render(ExperimentPanel, { ...baseProps(), status: 'error', errorMessage: 'boom' });
    const alert = screen.getByRole('alert');
    expect(alert).toHaveTextContent('boom');
    expect(alert).toHaveTextContent(/reload the page/i);
    unmount();

    render(ExperimentPanel, { ...baseProps(), status: 'ready', errorMessage: 'boom' });
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('controlsLocked disables Start/Seed even when their own status would otherwise allow them', () => {
    render(ExperimentPanel, { ...baseProps(), status: 'paused', controlsLocked: true });
    expect(screen.getByRole('button', { name: /^resume$/i })).toBeDisabled();
    expect(screen.getByLabelText(/^seed$/i)).toBeDisabled();
  });

  // Regression coverage for a real shipped bug: Pause/Reset were briefly
  // wired to `disabled={!canX || controlsLocked}` — the same blanket flag
  // Start/Seed use — and `controlsLocked` is unconditionally `true` for the
  // entire duration of every run (it exists to lock Start while running),
  // so both buttons were disabled for the whole run and could never
  // actually be clicked. `fireEvent.click` (used by `tests/App.lifecycle.test.ts`)
  // does not itself respect `disabled`, so only an explicit `toBeDisabled`/
  // `toBeEnabled` assertion like this one catches the regression.
  it('controlsLocked alone does NOT disable Pause/Reset — only topologySwitchPending does (regression)', () => {
    render(ExperimentPanel, {
      ...baseProps(),
      status: 'running',
      controlsLocked: true,
      topologySwitchPending: false
    });
    expect(screen.getByRole('button', { name: /^pause$/i })).toBeEnabled();
    expect(screen.getByRole('button', { name: /^reset$/i })).toBeEnabled();
  });

  it('topologySwitchPending disables Pause/Reset independent of controlsLocked', () => {
    render(ExperimentPanel, {
      ...baseProps(),
      status: 'running',
      controlsLocked: false,
      topologySwitchPending: true
    });
    expect(screen.getByRole('button', { name: /^pause$/i })).toBeDisabled();
    expect(screen.getByRole('button', { name: /^reset$/i })).toBeDisabled();
  });

  it('decoderSwitchPending disables Reset (but not Pause) independent of controlsLocked (round-2 dual review)', () => {
    render(ExperimentPanel, {
      ...baseProps(),
      status: 'paused',
      controlsLocked: false,
      topologySwitchPending: false,
      decoderSwitchPending: true
    });
    // Pause needs no guard here: `canPauseNow` already requires `running`,
    // which `ExperimentController#setDecoder` never allows a switch to be
    // in flight during — see `ExperimentPanel.svelte`'s own doc comment.
    expect(screen.getByRole('button', { name: /^pause$/i })).toBeDisabled();
    expect(screen.getByRole('button', { name: /^reset$/i })).toBeDisabled();
  });

  it('topologyControlsLocked disables only the topology selectors, independent of controlsLocked', () => {
    render(ExperimentPanel, {
      ...baseProps(),
      status: 'ready',
      controlsLocked: false,
      topologyControlsLocked: true
    });
    expect(screen.getByLabelText(/left arm topology/i)).toBeDisabled();
    expect(screen.getByLabelText(/right arm topology/i)).toBeDisabled();
    // Start etc. are still governed by controlsLocked/status, not this flag.
    expect(screen.getByRole('button', { name: /^start$/i })).toBeEnabled();
  });

  /**
   * Thermo-maintainability review (a11y, Important): a screen-reader user
   * who activates the Trained radio previously heard nothing further while
   * the decoder fieldset was disabled for the duration of the switch —
   * indistinguishable from the click having silently failed. This asserts
   * an `aria-live="polite"` status announcement exists exactly while
   * `decoderSwitchPending` is true, and is gone once it clears.
   */
  it('shows an aria-live="polite" "Switching decoder…" status while decoderSwitchPending is true, and not otherwise', () => {
    const { unmount } = render(ExperimentPanel, { ...baseProps(), decoderSwitchPending: true });
    const status = screen.getByText(/switching decoder/i);
    expect(status).toBeInTheDocument();
    expect(status).toHaveAttribute('aria-live', 'polite');
    unmount();

    render(ExperimentPanel, { ...baseProps(), decoderSwitchPending: false });
    expect(screen.queryByText(/switching decoder/i)).not.toBeInTheDocument();
  });

  /**
   * Same review finding: the "Trained unavailable" hint was visible to
   * sighted users but carried no live-region role at all, so a
   * screen-reader user was never told Trained had become unavailable
   * unless they happened to be focused inside the fieldset already.
   */
  it('shows the Trained-unavailable hint with aria-live="polite"', () => {
    render(ExperimentPanel, {
      ...baseProps(),
      trainedDecoderUnavailableReason: 'trained-readout-v1.json sha256 mismatch'
    });
    const hint = screen.getByText(/trained-readout-v1\.json sha256 mismatch/i);
    expect(hint).toHaveAttribute('aria-live', 'polite');
  });
});

describe('TelemetryPanel', () => {
  const telemetry: ExperimentTelemetry = {
    status: 'running',
    tick: 45,
    totalTicks: 2700,
    elapsedSimulatedSeconds: 1.5,
    seed: 777,
    behindRealtime: false,
    agents: {
      left: {
        topology: 'biological',
        neuronCount: 1008,
        edgeCount: 46311,
        foodPickups: 2,
        hazardContacts: 1,
        distanceTravelled: 10.4,
        movementScore: 5.2,
        activeFraction: 0.31,
        meanRate: 0.05,
        lastStepLatencyMs: 1.2,
        medianStepLatencyMs: 1.1
      },
      right: {
        topology: 'rewired',
        neuronCount: 1008,
        edgeCount: 46311,
        foodPickups: 0,
        hazardContacts: 3,
        distanceTravelled: 8.1,
        movementScore: -1.0,
        activeFraction: 0.12,
        meanRate: 0.02,
        lastStepLatencyMs: 1.4,
        medianStepLatencyMs: 1.3
      }
    }
  };

  it('renders elapsed time, seed, and both arms’ scores, node/edge counts, active-rate, and latency', () => {
    render(TelemetryPanel, { telemetry });

    expect(screen.getByText('1.5s')).toBeInTheDocument();
    expect(screen.getByText('777')).toBeInTheDocument();
    expect(screen.getByText(/left \(bio shape\)/i)).toBeInTheDocument();
    expect(screen.getByText(/right \(rewired shape\)/i)).toBeInTheDocument();
    expect(screen.getAllByText('1008 / 46311').length).toBe(2);
    expect(screen.getByText('biological')).toBeInTheDocument();
    expect(screen.getByText('rewired')).toBeInTheDocument();
    expect(screen.getByText('31.0%')).toBeInTheDocument();
    expect(screen.getByText('12.0%')).toBeInTheDocument();
  });

  it('shows the tick count and the behind-realtime warning together, never one instead of the other', () => {
    render(TelemetryPanel, { telemetry: { ...telemetry, behindRealtime: true } });
    const heading = screen.getByText(/slower than 30/i);
    // Regression coverage: this used to be a ternary that replaced the tick
    // counter with the warning, hiding the one number a viewer would want
    // to see keep changing during a live demo that's fallen behind pace.
    expect(heading).toHaveTextContent(`Tick ${telemetry.tick} / ${telemetry.totalTicks}`);
    expect(heading).toHaveTextContent(/slower than 30 hz/i);
  });
});

describe('LedgerPanel', () => {
  const manifest: ArenaManifest = {
    artifact: 'malecns-arena-v1.bin.gz',
    binaryBytes: 1,
    binarySha256: 'x',
    edgeCount: 46311,
    formatVersion: 1,
    gzipBytes: 1,
    gzipSha256: 'x',
    inputChannelCount: 8,
    license: 'CC-BY-4.0',
    neuronCount: 1008,
    outputPopulationCount: 3,
    sourceDataset: 'male-cns:v1.0 (Janelia FlyEM Male CNS connectome)',
    rewiredArms: {}
  };

  // Scoped by row (rather than a bare `getByText(label)`) because WP3 added
  // a second "Measured" row (Neuron positions, alongside Graph topology) —
  // an unscoped query would now match more than one element.
  const ledgerRow = (container: HTMLElement, term: string): HTMLElement => {
    const strong = within(container).getByText(term, { selector: 'strong' });
    const row = strong.closest('li');
    if (!row) throw new Error(`Ledger row for "${term}" is not inside an <li>`);
    return row as HTMLElement;
  };

  it('renders the ledger vocabulary, never says brain emulation, and links to the manifest/ledger/license', () => {
    const { container } = render(LedgerPanel, { manifest, decoder: 'authored', trainedReadout: undefined, rewiringNull: undefined });

    expect(ledgerRow(container, 'Graph topology')).toHaveTextContent('Measured');
    expect(ledgerRow(container, 'Biological annotations')).toHaveTextContent('Annotated');
    expect(ledgerRow(container, 'Network dynamics')).toHaveTextContent('Authored / literature-derived');
    expect(ledgerRow(container, 'Global parameters')).toHaveTextContent('Calibrated');
    expect(ledgerRow(container, 'Sensory encoder and action decoder')).toHaveTextContent('Authored');
    expect(ledgerRow(container, '3D presentation')).toHaveTextContent('Synthetic');
    expect(ledgerRow(container, 'Neuron positions')).toHaveTextContent('Measured');
    expect(ledgerRow(container, 'Displayed neural activity')).toHaveTextContent('Computed');
    expect(screen.queryByText(/brain emulation/i)).not.toBeInTheDocument();
    expect(screen.getByRole('link', { name: /manifest/i })).toHaveAttribute(
      'href',
      '/data/malecns-arena-v1.manifest.json'
    );
    expect(screen.getByRole('link', { name: /ledger/i })).toHaveAttribute(
      'href',
      '/data/malecns-arena-v1.ledger.json'
    );
    expect(screen.getByRole('link', { name: /cc by 4\.0/i })).toHaveAttribute(
      'href',
      'https://creativecommons.org/licenses/by/4.0/'
    );
  });

  it('renders the static ledger vocabulary even without a loaded manifest', () => {
    const { container } = render(LedgerPanel, { manifest: undefined, decoder: 'authored', trainedReadout: undefined, rewiringNull: undefined });
    expect(ledgerRow(container, 'Graph topology')).toHaveTextContent('Measured');
  });

  // LedgerPanel only reads `manifest` off an `'ok'` result, never
  // `weightsByMode` — so this test double omits it (it exists purely to
  // satisfy the real type's structural shape for the decoded readout
  // weights, which the component has no reason to render).
  const trainedReadoutOk = {
    status: 'ok',
    manifest: { version: 1, artifactSha256: 'abcdef0123456789abcdef0123456789', D: 48, H: 16, parameterCount: 835 }
  } as unknown as TrainedReadoutLoadResult;

  it('shows the "Readout (trained mode)" row and provenance detail (hash prefix, param count, D -> H -> 3, report links) when the artifact is ok, visible even in Authored mode', () => {
    const { container } = render(LedgerPanel, { manifest, decoder: 'authored', trainedReadout: trainedReadoutOk, rewiringNull: undefined });

    expect(ledgerRow(container, 'Readout (trained mode)')).toHaveTextContent('Trained (offline)');
    expect(screen.getByText('835')).toBeInTheDocument();
    expect(screen.getByText(/48 → 16 → 3/)).toBeInTheDocument();
    expect(screen.getByText(/abcdef012345/)).toBeInTheDocument();
    expect(screen.getByText(/opponent parked/i)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /report \(json\)/i })).toHaveAttribute(
      'href',
      '/data/trained-readout-v1.report.json'
    );
    expect(screen.getByRole('link', { name: /github/i })).toHaveAttribute(
      'href',
      'https://github.com/mattsp1290/flyarena/blob/main/docs/trained-readout-report.md'
    );
  });

  it('scopes the "Sensory encoder and action decoder" row label to Trained mode', () => {
    const { container } = render(LedgerPanel, { manifest, decoder: 'trained', trainedReadout: trainedReadoutOk, rewiringNull: undefined });
    const row = ledgerRow(container, 'Sensory encoder and action decoder');
    expect(row).toHaveTextContent(/authored/i);
    expect(row).toHaveTextContent(/trained/i);
  });

  it('shows the artifact-failed-verification message when the trained-readout artifact is unavailable', () => {
    render(LedgerPanel, {
      manifest,
      decoder: 'authored',
      trainedReadout: { status: 'unavailable', reason: 'trained-readout-v1.json sha256 mismatch' },
      rewiringNull: undefined
    });
    const message = screen.getByText(/artifact failed verification/i);
    expect(message).toHaveTextContent(/sha256 mismatch/i);
  });

  const rewiringNullOk = {
    status: 'ok',
    data: {
      version: 1,
      condition: 'authored, opponent parked',
      seeds: { start: 30001, count: 100 },
      ticks: 1800,
      substeps: 4,
      sourceGraphSha256: 'a'.repeat(64),
      rewireSourceSha256: 'b'.repeat(64),
      shards: 18,
      biological: { score: -0.22, median: -1.4, std: 3.28, ci: [-0.85, 0.45] },
      disconnected: { score: -1.86, median: -2, std: 1.79, ci: [-2.22, -1.52] },
      rewired: [
        { seed: 0, gzipSha256: 'c'.repeat(64), score: 1.02, median: 0.62, std: 3.73, ci: [0.3, 1.77], acceptedSwaps: 764340, attempts: 926220 },
        { seed: 1, gzipSha256: 'd'.repeat(64), score: 2.5, median: 2.5, std: 1.0, ci: [2.0, 3.0], acceptedSwaps: 764340, attempts: 926220 }
      ],
      null: { n: 2, mean: 1.76, median: 1.76, std: 1.05, p2_5: 1.05, p97_5: 2.47, iqr: 1.48, degenerate: false },
      bioPercentile: 0,
      pLow: 0.33,
      pHigh: 1,
      bins: { edges: [-2, 0, 2, 4], counts: [0, 1, 1] }
    }
  } as unknown as RewiringNullLoadResult;

  it('shows the "Topology null distribution" section with the histogram and percentile sentence when the artifact is ok', () => {
    const { container } = render(LedgerPanel, {
      manifest,
      decoder: 'authored',
      trainedReadout: undefined,
      rewiringNull: rewiringNullOk
    });

    expect(ledgerRow(container, 'Topology null distribution')).toHaveTextContent('Computed (offline)');
    expect(screen.getByRole('heading', { name: /topology null distribution/i })).toBeInTheDocument();
    // The percentile sentence is duplicated onto the SVG's own accessible
    // name/`<desc>` (for screen-reader users who never reach the visible
    // `<figcaption>`) — scope this assertion to the visible figcaption
    // specifically, rather than an unscoped `getByText` that would match
    // both and fail with "multiple elements found".
    // The caption is built from the fixture's own `data.null.n`/`data.seeds.count`
    // (2 and 100 here), not a hardcoded "500" — regression coverage for the
    // dual review finding that an earlier version hardcoded these numbers.
    const figcaption = container.querySelector('figcaption');
    expect(figcaption).toHaveTextContent(/scored above 0\.0% of 2 degree-preserving rewirings/i);
    expect(figcaption).toHaveTextContent(/opponent parked/i);
    expect(screen.getByRole('link', { name: /rewiring-null result \(json\)/i })).toHaveAttribute(
      'href',
      '/data/rewiring-null-v1.json'
    );
    expect(screen.getByRole('link', { name: /full rewiring-null report/i })).toBeInTheDocument();
  });

  it('hides the "Topology null distribution" section entirely when the artifact is missing (the ledger row itself, like "Readout (trained mode)", still shows)', () => {
    const { container } = render(LedgerPanel, {
      manifest,
      decoder: 'authored',
      trainedReadout: undefined,
      rewiringNull: { status: 'missing', reason: 'no entry' }
    });
    expect(ledgerRow(container, 'Topology null distribution')).toHaveTextContent('Computed (offline) — not shipped');
    expect(screen.queryByRole('heading', { name: /topology null distribution/i })).not.toBeInTheDocument();
    expect(screen.queryByRole('link', { name: /rewiring-null result \(json\)/i })).not.toBeInTheDocument();
  });

  it('shows the honest verification-failure message when the rewiring-null artifact is invalid, and labels the ledger row "unavailable"', () => {
    const { container } = render(LedgerPanel, {
      manifest,
      decoder: 'authored',
      trainedReadout: undefined,
      rewiringNull: { status: 'invalid', reason: 'rewiring-null artifact sha256 mismatch' }
    });
    const message = screen.getByText(/null-distribution result failed verification/i);
    expect(message).toHaveTextContent(/sha256 mismatch/i);
    expect(ledgerRow(container, 'Topology null distribution')).toHaveTextContent('Computed (offline) — unavailable');
    expect(screen.getByRole('heading', { name: /topology null distribution/i })).toBeInTheDocument();
  });

  it('shows "Loading…" for the ledger row and no section at all while rewiringNull is still undefined', () => {
    const { container } = render(LedgerPanel, {
      manifest,
      decoder: 'authored',
      trainedReadout: undefined,
      rewiringNull: undefined
    });
    expect(ledgerRow(container, 'Topology null distribution')).toHaveTextContent('Loading…');
    expect(screen.queryByRole('heading', { name: /topology null distribution/i })).not.toBeInTheDocument();
  });
});
