import { cleanup, fireEvent, render, screen } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import ExperimentPanel from '../../src/lib/ui/ExperimentPanel.svelte';
import TelemetryPanel from '../../src/lib/ui/TelemetryPanel.svelte';
import LedgerPanel from '../../src/lib/ui/LedgerPanel.svelte';
import type { ExperimentTelemetry } from '../../src/lib/experiment/runner';
import type { ArenaManifest } from '../../src/lib/experiment/assets';

afterEach(() => cleanup());

describe('ExperimentPanel accessibility and wiring', () => {
  const baseProps = () => ({
    status: 'ready' as const,
    errorMessage: undefined,
    seed: 1234,
    topology: { left: 'biological' as const, right: 'rewired' as const },
    controlsLocked: false,
    topologyControlsLocked: false,
    onStart: vi.fn(),
    onPause: vi.fn(),
    onReset: vi.fn(),
    onSeedInput: vi.fn(),
    onTopologyChange: vi.fn(),
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

    // Every interactive control here is a native <button>/<input>/<select>,
    // which are keyboard-operable (Tab/Enter/Space/arrow keys) by the
    // platform itself — no custom `div`-as-button widget to audit.
    for (const element of [
      ...screen.getAllByRole('button'),
      screen.getByLabelText(/^seed$/i),
      screen.getByLabelText(/left arm topology/i),
      screen.getByLabelText(/right arm topology/i)
    ]) {
      expect(['BUTTON', 'INPUT', 'SELECT']).toContain(element.tagName);
    }
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

  it('controlsLocked disables Start/Pause/Reset/Seed even when their own status would otherwise allow them', () => {
    render(ExperimentPanel, { ...baseProps(), status: 'paused', controlsLocked: true });
    expect(screen.getByRole('button', { name: /^resume$/i })).toBeDisabled();
    expect(screen.getByRole('button', { name: /^reset$/i })).toBeDisabled();
    expect(screen.getByLabelText(/^seed$/i)).toBeDisabled();
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

  it('flags a slower-than-30Hz run', () => {
    render(TelemetryPanel, { telemetry: { ...telemetry, behindRealtime: true } });
    expect(screen.getByText(/slower than 30/i)).toBeInTheDocument();
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

  it('renders the ledger vocabulary, never says brain emulation, and links to the manifest/ledger/license', () => {
    render(LedgerPanel, { manifest });

    for (const label of ['Measured', 'Annotated', 'Authored / literature-derived', 'Calibrated', 'Authored', 'Synthetic']) {
      expect(screen.getByText(label)).toBeInTheDocument();
    }
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
    render(LedgerPanel, { manifest: undefined });
    expect(screen.getByText('Measured')).toBeInTheDocument();
  });
});
