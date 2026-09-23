import { cleanup, render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from '../src/App.svelte';
import { createPublicDataFetch, FakeNeuralWorker } from './helpers/fake-worker';

// This project does not enable Vitest's implicit test globals (see
// vite.config.ts), so @testing-library/svelte's own auto-cleanup —
// which only registers itself when it finds a global `afterEach` — never
// runs. Register it explicitly so each test starts from an empty DOM.
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

/**
 * Every test stubs `fetch` (serving the real, committed `public/data/*`
 * artifacts from disk — see `tests/helpers/fake-worker.ts`) and `Worker`
 * (a same-thread stand-in driving the real `handleWorkerRequest`) so
 * `App.svelte`'s asset-load -> Worker-init -> `ExperimentRunner` pipeline
 * runs deterministically under jsdom, which implements neither a real
 * `Worker` nor a network.
 */
beforeEach(() => {
  vi.stubGlobal('fetch', createPublicDataFetch());
  vi.stubGlobal('Worker', FakeNeuralWorker as unknown as typeof Worker);
});

describe('App shell', () => {
  it('presents the arena, controls, and ledger regions immediately, and the telemetry region once the experiment is ready', async () => {
    render(App);

    expect(screen.getByRole('heading', { level: 2, name: /arena/i })).toBeInTheDocument();
    expect(screen.getByRole('region', { name: /experiment controls/i })).toBeInTheDocument();
    expect(screen.getByRole('region', { name: /model ledger/i })).toBeInTheDocument();
    expect(screen.getByText('Measured')).toBeInTheDocument();
    expect(screen.getByText('Annotated')).toBeInTheDocument();
    expect(screen.getByText('Authored / literature-derived')).toBeInTheDocument();
    expect(screen.getByText('Calibrated')).toBeInTheDocument();
    expect(screen.getByText('Synthetic')).toBeInTheDocument();

    await screen.findByRole('region', { name: /telemetry/i }, { timeout: 5000 });
    await waitFor(() => expect(screen.getByLabelText(/experiment status: ready/i)).toBeInTheDocument(), {
      timeout: 5000
    });

    const startButton = screen.getByRole('button', { name: /^start$/i });
    expect(startButton).toBeEnabled();
  });

  it('shows a graceful fallback message in the arena region when WebGL is unavailable', async () => {
    // jsdom (this test environment) implements no WebGL, and normally
    // returns null from getContext('webgl2'/'webgl') itself already, which
    // is what makes ArenaScene's construction fail. Force it explicitly
    // instead of relying on that default, so this test keeps covering the
    // fallback path even if a future jsdom version starts stubbing a
    // context object, and so it doesn't depend on jsdom's noisy
    // "Not implemented" console warning.
    const getContextSpy = vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    try {
      render(App);

      const fallback = await screen.findByRole('img', { name: /arena canvas unavailable/i });
      expect(fallback).toBeInTheDocument();
      expect(fallback).toHaveTextContent(/webgl/i);
    } finally {
      getContextSpy.mockRestore();
    }
  });

  it('moves to the error state and reports it when a fetched artifact fails its sha256 check', async () => {
    vi.stubGlobal('fetch', createPublicDataFetch({ corrupt: 'malecns-arena-v1.bin.gz' }));
    render(App);

    await waitFor(() => expect(screen.getByLabelText(/experiment status: error/i)).toBeInTheDocument(), {
      timeout: 5000
    });
    expect(screen.getByRole('alert')).toHaveTextContent(/sha256/i);
    expect(screen.getByRole('button', { name: /^start$/i })).toBeDisabled();
  });

  it('never describes the experiment as a brain emulation and links to the source provenance', () => {
    render(App);

    expect(screen.queryByText(/brain emulation/i)).not.toBeInTheDocument();
    expect(screen.getByRole('link', { name: /cc by 4\.0/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /compiled artifact manifest/i })).toHaveAttribute(
      'href',
      '/data/malecns-arena-v1.manifest.json'
    );
  });
});
