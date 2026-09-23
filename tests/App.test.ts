import { render, screen } from '@testing-library/svelte';
import { describe, expect, it } from 'vitest';
import App from '../src/App.svelte';

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
});
