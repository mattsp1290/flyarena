import { cleanup, fireEvent, render, screen, within } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import ExperimentPanel from '../../src/lib/ui/ExperimentPanel.svelte';
import TelemetryPanel from '../../src/lib/ui/TelemetryPanel.svelte';
import LedgerPanel from '../../src/lib/ui/LedgerPanel.svelte';
import type { ExperimentTelemetry } from '../../src/lib/experiment/runner';
import type { ArenaManifest, TrainedReadoutLoadResult } from '../../src/lib/experiment/assets';
import type { RewiringNullLoadResult } from '../../src/lib/experiment/rewiringNull';
import type { NullExplanationLoadResult } from '../../src/lib/experiment/nullExplanation';

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
    const { container } = render(LedgerPanel, { manifest, decoder: 'authored', trainedReadout: undefined, rewiringNull: undefined, nullExplanation: undefined });

    expect(ledgerRow(container, 'Graph topology')).toHaveTextContent('Measured');
    expect(ledgerRow(container, 'Biological annotations')).toHaveTextContent('Annotated');
    expect(ledgerRow(container, 'Network dynamics')).toHaveTextContent('Authored / literature-derived');
    expect(ledgerRow(container, 'Global parameters')).toHaveTextContent('Calibrated');
    // Thermo review I1: "authored" carries a short "(hand-written)" gloss
    // in-product, defining the word where a reader would otherwise take it
    // as a neutral engineering label.
    expect(ledgerRow(container, 'Sensory encoder and action decoder')).toHaveTextContent('Authored (hand-written)');
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
    const { container } = render(LedgerPanel, { manifest: undefined, decoder: 'authored', trainedReadout: undefined, rewiringNull: undefined, nullExplanation: undefined });
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
    const { container } = render(LedgerPanel, { manifest, decoder: 'authored', trainedReadout: trainedReadoutOk, rewiringNull: undefined, nullExplanation: undefined });

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
    const { container } = render(LedgerPanel, { manifest, decoder: 'trained', trainedReadout: trainedReadoutOk, rewiringNull: undefined, nullExplanation: undefined });
    const row = ledgerRow(container, 'Sensory encoder and action decoder');
    expect(row).toHaveTextContent(/authored/i);
    expect(row).toHaveTextContent(/trained/i);
  });

  it('shows the artifact-failed-verification message when the trained-readout artifact is unavailable', () => {
    render(LedgerPanel, {
      manifest,
      decoder: 'authored',
      trainedReadout: { status: 'unavailable', reason: 'trained-readout-v1.json sha256 mismatch' },
      rewiringNull: undefined,
      nullExplanation: undefined
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
  } as unknown as Extract<RewiringNullLoadResult, { status: 'ok' }>;

  const nullExplanationOk = {
    status: 'ok',
    data: {
      version: 1,
      sources: { rewiringNullSha256: 'a'.repeat(64) },
      variants: { flipBoth: { bioPercentile: 0 } },
      regime: { gatePassed: true },
      finding: {
        categories: ['linearPathway', 'structuralFeature'],
        definitionSensitive: true,
        qualifyingMetrics: [
          { kind: 'transfer', name: 'T:rightClearance->thrust', spearman: 0.4669248436993748 },
          { kind: 'feature', name: 'weightedInDegree:thrust', spearman: 0.3943192055100449 },
          { kind: 'transfer', name: 'T:forwardClearance->thrust', spearman: 0.3534047736190945 }
        ],
        regimeInvalid: false,
        summarySentence:
          "Biological's low score is associated with: the linear transfer entry T:rightClearance->thrust sits outside the null's 2.5-97.5% range (rank correlation with score rho=0.467); the structural feature weightedInDegree:thrust sits outside the null's 2.5-97.5% range (rank correlation with score rho=0.394) -- a descriptive correlation, not a causal claim."
      }
    }
  } as unknown as Extract<NullExplanationLoadResult, { status: 'ok' }>;

  it('shows the "Topology null distribution" section with the histogram and percentile sentence when the artifact is ok', () => {
    const { container } = render(LedgerPanel, {
      manifest,
      decoder: 'authored',
      trainedReadout: undefined,
      rewiringNull: rewiringNullOk,
      nullExplanation: undefined
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
    expect(figcaption).toHaveTextContent(/ranks at the 0\.0% percentile.*among 2 degree-preserving rewirings/i);
    expect(figcaption).toHaveTextContent(/opponent parked/i);
    // Thermo review I1: the static disclaimer is visible in-product, not
    // only in the linked report.
    expect(figcaption).toHaveTextContent(/fixed, hand-written mapping/i);
    expect(screen.getByRole('link', { name: /rewiring-null result \(json\)/i })).toHaveAttribute(
      'href',
      '/data/rewiring-null-v1.json'
    );
    expect(screen.getByRole('link', { name: /full rewiring-null report/i })).toBeInTheDocument();
  });

  it('hides the "Topology null distribution" section entirely when the artifact is absent (the ledger row itself, like "Readout (trained mode)", still shows)', () => {
    const { container } = render(LedgerPanel, {
      manifest,
      decoder: 'authored',
      trainedReadout: undefined,
      rewiringNull: { status: 'absent', reason: 'no entry' },
      nullExplanation: undefined
    });
    expect(ledgerRow(container, 'Topology null distribution')).toHaveTextContent('Computed (offline) — not shipped');
    expect(screen.queryByRole('heading', { name: /topology null distribution/i })).not.toBeInTheDocument();
    expect(screen.queryByRole('link', { name: /rewiring-null result \(json\)/i })).not.toBeInTheDocument();
  });

  it('shows the honest verification-failure message when the rewiring-null artifact is invalid, and labels the ledger row "failed verification"', () => {
    const { container } = render(LedgerPanel, {
      manifest,
      decoder: 'authored',
      trainedReadout: undefined,
      rewiringNull: { status: 'invalid', reason: 'rewiring-null artifact sha256 mismatch' },
      nullExplanation: undefined
    });
    const message = screen.getByText(/null-distribution result failed verification/i);
    expect(message).toHaveTextContent(/sha256 mismatch/i);
    expect(ledgerRow(container, 'Topology null distribution')).toHaveTextContent('Computed (offline) — failed verification');
    expect(screen.getByRole('heading', { name: /topology null distribution/i })).toBeInTheDocument();
  });

  // Thermo review, Suggestion: distinct wording from "invalid" above — a
  // genuine fetch/network failure is not a claim that verification failed.
  it('shows a "could not be loaded" message (not "failed verification") when the rewiring-null artifact is unavailable, and labels the ledger row accordingly', () => {
    const { container } = render(LedgerPanel, {
      manifest,
      decoder: 'authored',
      trainedReadout: undefined,
      rewiringNull: { status: 'unavailable', reason: 'network error (test)' },
      nullExplanation: undefined
    });
    const message = screen.getByText(/null-distribution result could not be loaded/i);
    expect(message).toHaveTextContent(/network error \(test\)/i);
    expect(screen.queryByText(/failed verification/i)).not.toBeInTheDocument();
    expect(ledgerRow(container, 'Topology null distribution')).toHaveTextContent('Computed (offline) — could not be loaded');
    expect(screen.getByRole('heading', { name: /topology null distribution/i })).toBeInTheDocument();
  });

  it('shows "Loading…" for the ledger row and no section at all while rewiringNull is still undefined', () => {
    const { container } = render(LedgerPanel, {
      manifest,
      decoder: 'authored',
      trainedReadout: undefined,
      rewiringNull: undefined,
      nullExplanation: undefined
    });
    expect(ledgerRow(container, 'Topology null distribution')).toHaveTextContent('Loading…');
    expect(screen.queryByRole('heading', { name: /topology null distribution/i })).not.toBeInTheDocument();
  });

  // WP4 of `.agents/plans/null-explanation`: the ledger's finding note,
  // rendered next to the null histogram once both the histogram's own
  // artifact and this note's artifact are `'ok'`.
  it('always shows the static "Null-result explanation" ledger row, independent of nullExplanation status', () => {
    const { container } = render(LedgerPanel, {
      manifest,
      decoder: 'authored',
      trainedReadout: undefined,
      rewiringNull: undefined,
      nullExplanation: undefined
    });
    expect(ledgerRow(container, 'Null-result explanation')).toHaveTextContent('Computed (offline)');
  });

  it('renders the finding sentence, every qualifying metric in plain words with rho, the definition-sensitive flag/link, the mirrored-decoder clause, the regime clause, the non-causal disclaimer, and the report link', () => {
    render(LedgerPanel, {
      manifest,
      decoder: 'authored',
      trainedReadout: undefined,
      rewiringNull: rewiringNullOk,
      nullExplanation: nullExplanationOk
    });

    expect(screen.getByRole('heading', { name: /why biological scores low/i })).toBeInTheDocument();
    expect(screen.getByText(nullExplanationOk.data.finding.summarySentence)).toBeInTheDocument();

    // Every qualifying metric is listed in plain words with its own rho —
    // never a hard-coded per-metric string table (the artifact drives every
    // word here).
    expect(
      screen.getByText(/linear signal gain from right clearance input to thrust output \(ρ = 0\.467\)/i)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/linear signal gain from forward clearance input to thrust output \(ρ = 0\.353\)/i)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/weighted in-degree from input neurons to thrust output \(ρ = 0\.394\)/i)
    ).toBeInTheDocument();

    // Only the structural-feature (weightedInDegree) qualifying metric is
    // flagged definition-sensitive, with a link into the report's
    // disclosure — the two transfer-kind metrics above must not carry it.
    const sensitiveLink = screen.getByRole('link', { name: /definition-sensitive/i });
    expect(sensitiveLink).toHaveAttribute(
      'href',
      'https://github.com/mattsp1290/flyarena/blob/main/docs/null-explanation-report.md#structural-features'
    );

    // Mirrored decoder-convention check: still the bottom (0th percentile).
    expect(screen.getByText(/mirroring the decoder's thrust and yaw signs still leaves biological at the bottom/i)).toBeInTheDocument();
    // Regime check outcome.
    expect(screen.getByText(/linear-regime check passed/i)).toBeInTheDocument();

    // Always-carried framing.
    expect(screen.getByText(/descriptive association within this model, not a cause/i)).toBeInTheDocument();
    expect(screen.getByText(/fixed, hand-written decoder — not biology and not trained/i)).toBeInTheDocument();

    const reportLink = screen.getByRole('link', { name: /full explanation report/i });
    expect(reportLink).toHaveAttribute(
      'href',
      'https://github.com/mattsp1290/flyarena/blob/main/docs/null-explanation-report.md'
    );
  });

  it('states a nonzero mirrored-decoder percentile (moved up from a 0th-percentile baseline) and a failed regime gate in their own words', () => {
    const movedUp = {
      status: 'ok',
      data: {
        ...nullExplanationOk.data,
        variants: { flipBoth: { bioPercentile: 0.337 } },
        finding: { ...nullExplanationOk.data.finding, regimeInvalid: true }
      }
    } as unknown as NullExplanationLoadResult;
    render(LedgerPanel, {
      manifest,
      decoder: 'authored',
      trainedReadout: undefined,
      rewiringNull: rewiringNullOk,
      nullExplanation: movedUp
    });
    // rewiringNullOk.data.bioPercentile is 0 — the baseline is read from the
    // real un-mirrored result, not assumed to be 0 (round-2 dual review,
    // Important).
    expect(screen.getByText(/moves biological from the 0\.0th percentile to the 33\.7th percentile/i)).toBeInTheDocument();
    expect(screen.getByText(/regime-invalid \(inconclusive\)/i)).toBeInTheDocument();
  });

  it('never says "still" when the mirrored percentile moved away from a nonzero baseline, and states both real values', () => {
    const nonzeroBaseline = {
      status: 'ok',
      data: { ...rewiringNullOk.data, bioPercentile: 0.2 }
    } as unknown as RewiringNullLoadResult;
    // (typed via `RewiringNullLoadResult`, the full union — this value is
    // only ever passed as a prop below, never narrowed with `.data` again)
    render(LedgerPanel, {
      manifest,
      decoder: 'authored',
      trainedReadout: undefined,
      rewiringNull: nonzeroBaseline,
      nullExplanation: nullExplanationOk
    });
    expect(screen.getByText(/moves biological from the 20\.0th percentile to the 0\.0th percentile/i)).toBeInTheDocument();
    expect(screen.queryByText(/still leaves biological at the bottom/i)).not.toBeInTheDocument();
  });

  // Round-2 dual review, Important: `explain.py`'s `definitionSensitive`
  // flag is computed only from its single top structural candidate, and is
  // true only when *that* metric is `weightedInDegree:*` — never a blanket
  // flag over every `feature`-kind qualifying metric.
  it('flags only the weightedInDegree qualifying metric as definition-sensitive, never another feature-kind metric', () => {
    const twoFeatureMetrics = {
      status: 'ok',
      data: {
        ...nullExplanationOk.data,
        finding: {
          ...nullExplanationOk.data.finding,
          qualifyingMetrics: [
            { kind: 'feature', name: 'weightedInDegree:thrust', spearman: 0.394 },
            { kind: 'feature', name: 'reciprocity', spearman: 0.35 }
          ]
        }
      }
    } as unknown as NullExplanationLoadResult;
    const { container } = render(LedgerPanel, {
      manifest,
      decoder: 'authored',
      trainedReadout: undefined,
      rewiringNull: rewiringNullOk,
      nullExplanation: twoFeatureMetrics
    });
    const sensitiveLinks = screen.getAllByRole('link', { name: /definition-sensitive/i });
    expect(sensitiveLinks).toHaveLength(1);
    const items = container.querySelectorAll('.metric-list li');
    expect(items).toHaveLength(2);
    expect(items[0].textContent).toMatch(/weighted in-degree/i);
    expect(items[0].textContent).toMatch(/definition-sensitive/i);
    expect(items[1].textContent).toMatch(/reciprocity/i);
    expect(items[1].textContent).not.toMatch(/definition-sensitive/i);
  });

  it('shows the honest "Explanation failed verification" message when the null-explanation artifact is invalid', () => {
    render(LedgerPanel, {
      manifest,
      decoder: 'authored',
      trainedReadout: undefined,
      rewiringNull: rewiringNullOk,
      nullExplanation: { status: 'invalid', reason: 'null-explanation artifact sha256 mismatch' }
    });
    const message = screen.getByText(/explanation failed verification/i);
    expect(message).toHaveTextContent(/sha256 mismatch/i);
    expect(screen.queryByRole('heading', { name: /why biological scores low/i })).not.toBeInTheDocument();
  });

  it('shows a "could not be loaded" message (not "failed verification") when the null-explanation artifact is unavailable', () => {
    render(LedgerPanel, {
      manifest,
      decoder: 'authored',
      trainedReadout: undefined,
      rewiringNull: rewiringNullOk,
      nullExplanation: { status: 'unavailable', reason: 'network error (test)' }
    });
    const message = screen.getByText(/explanation could not be loaded/i);
    expect(message).toHaveTextContent(/network error \(test\)/i);
    expect(screen.queryByText(/explanation failed verification/i)).not.toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: /why biological scores low/i })).not.toBeInTheDocument();
  });

  it('hides the explanation paragraph entirely when the null-explanation artifact is missing, while the histogram above it keeps rendering', () => {
    const { container } = render(LedgerPanel, {
      manifest,
      decoder: 'authored',
      trainedReadout: undefined,
      rewiringNull: rewiringNullOk,
      nullExplanation: { status: 'missing', reason: 'The manifest has no nullExplanation artifact entry.' }
    });
    expect(screen.queryByRole('heading', { name: /why biological scores low/i })).not.toBeInTheDocument();
    expect(screen.queryByText(/explanation failed verification/i)).not.toBeInTheDocument();
    // The histogram (a required, independent load) is unaffected.
    expect(container.querySelector('.null-histogram')).not.toBeNull();
    expect(screen.getByRole('heading', { name: /topology null distribution/i })).toBeInTheDocument();
  });
});
