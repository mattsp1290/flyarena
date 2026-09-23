import { cleanup, render, screen } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import App from '../src/App.svelte';

// This project does not enable Vitest's implicit test globals (see
// vite.config.ts), so @testing-library/svelte's own auto-cleanup —
// which only registers itself when it finds a global `afterEach` — never
// runs. Register it explicitly so each test starts from an empty DOM.
afterEach(() => cleanup());

describe('App shell', () => {
  it('presents the arena and each information region', () => {
    render(App);

    expect(screen.getByRole('heading', { level: 2, name: /arena/i })).toBeInTheDocument();
    expect(screen.getByRole('region', { name: /experiment controls/i })).toBeInTheDocument();
    expect(screen.getByRole('region', { name: /telemetry/i })).toBeInTheDocument();
    expect(screen.getByRole('region', { name: /model ledger/i })).toBeInTheDocument();
    expect(screen.getByText('Measured')).toBeInTheDocument();
    expect(screen.getByText('Annotated')).toBeInTheDocument();
    expect(screen.getByText('Authored / literature-derived')).toBeInTheDocument();
    expect(screen.getByText('Calibrated')).toBeInTheDocument();
    expect(screen.getByText('Synthetic')).toBeInTheDocument();
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
});
