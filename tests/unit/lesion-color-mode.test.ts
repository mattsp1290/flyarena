import { cleanup, render, screen } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import LesionColorMode from '../../src/lib/ui/LesionColorMode.svelte';
import type { LesionAtlasLoadResult } from '../../src/lib/experiment/lesionAtlas';

/**
 * Component-level tests for `LesionColorMode.svelte` in isolation, mounted
 * directly (no `ActivityPanel.svelte`, no mocked `ActivityScene`) — this is
 * the presentational half of the WP3 extraction (see that component's own
 * doc comment); its state comes entirely from props, and it reports mode
 * switches through a single `onSwitchColorMode` callback.
 */

afterEach(() => {
  cleanup();
});

const fakeLesionAtlasOk = (): LesionAtlasLoadResult => ({
  status: 'ok',
  absMax: 1,
  data: {
    version: 1,
    neuronCount: 3,
    bodyIds: ['1000', '1001', '1002'],
    graphs: {
      biological: {
        graphSha256: 'x',
        baseline: 0,
        effect: [0.1, -0.2, 0.3],
        ciLow: [0, 0, 0],
        ciHigh: [0, 0, 0],
        fdrSignificant: [true, false, true]
      },
      rewiredSeed0: {
        graphSha256: 'y',
        baseline: 0,
        effect: [-0.4, 0.5, 0.0],
        ciLow: [0, 0, 0],
        ciHigh: [0, 0, 0],
        fdrSignificant: [false, true, true]
      }
    }
  }
});

describe('LesionColorMode part="controls"', () => {
  it('renders the hint paragraph as an always-present aria-live region, empty when there is nothing to say', () => {
    const { container } = render(LesionColorMode, {
      part: 'controls',
      sceneReady: true,
      colorMode: 'live',
      lesionOptionDisabledReason: undefined,
      lesionAtlasTransientReason: undefined
    });

    const hint = container.querySelector('p.hint[aria-live="polite"]');
    expect(hint).not.toBeNull();
    expect(hint?.textContent).toBe('');
  });

  it('updates the SAME hint node\'s text (not a freshly-inserted one) when the disabled reason changes across a rerender (thermo-maintainability I3)', async () => {
    const { container, rerender } = render(LesionColorMode, {
      part: 'controls',
      sceneReady: true,
      colorMode: 'live',
      lesionOptionDisabledReason: undefined,
      lesionAtlasTransientReason: undefined
    });

    const hintBefore = container.querySelector('p.hint[aria-live="polite"]');
    expect(hintBefore?.textContent).toBe('');

    await rerender({
      part: 'controls',
      sceneReady: true,
      colorMode: 'live',
      lesionOptionDisabledReason: 'no lesion atlas was shipped with this build',
      lesionAtlasTransientReason: undefined
    });

    const hintAfter = container.querySelector('p.hint[aria-live="polite"]');
    expect(hintAfter).toBe(hintBefore); // same node — a mutation, not an insertion
    expect(hintAfter?.textContent).toMatch(/no lesion atlas was shipped/i);
  });

  it('renders the sr-only summary as an always-present aria-live region that updates in place', async () => {
    const { container, rerender } = render(LesionColorMode, {
      part: 'controls',
      sceneReady: true,
      colorMode: 'live',
      lesionSignificantSummary: undefined
    });

    const summaryBefore = container.querySelector('p.sr-only[aria-live="polite"]');
    expect(summaryBefore).not.toBeNull();
    expect(summaryBefore?.textContent).toBe('');

    await rerender({
      part: 'controls',
      sceneReady: true,
      colorMode: 'lesion',
      lesionSignificantSummary: 'Lesion effect mode. Left arm (biological): 2 of 3 neurons FDR-significant.'
    });

    const summaryAfter = container.querySelector('p.sr-only[aria-live="polite"]');
    expect(summaryAfter).toBe(summaryBefore);
    expect(summaryAfter?.textContent).toMatch(/2 of 3 neurons fdr-significant/i);
  });

  it('the hint prefers the disabled reason over a transient reason, and shows nothing while in lesion mode', async () => {
    const { container, rerender } = render(LesionColorMode, {
      part: 'controls',
      sceneReady: true,
      colorMode: 'live',
      lesionOptionDisabledReason: 'no lesion atlas was shipped with this build',
      lesionAtlasTransientReason: 'network hiccup (test)'
    });

    expect(container.querySelector('p.hint')?.textContent).toMatch(/unavailable: no lesion atlas was shipped/i);

    await rerender({
      part: 'controls',
      sceneReady: true,
      colorMode: 'lesion',
      lesionOptionDisabledReason: 'no lesion atlas was shipped with this build',
      lesionAtlasTransientReason: 'network hiccup (test)'
    });

    // Hint only ever concerns the Live-mode-only disabled/retry states.
    expect(container.querySelector('p.hint')?.textContent).toBe('');
  });

  it('clicking the radios reports the intended mode through onSwitchColorMode', async () => {
    const onSwitchColorMode = vi.fn();
    render(LesionColorMode, {
      part: 'controls',
      sceneReady: true,
      colorMode: 'live',
      onSwitchColorMode
    });

    const lesionRadio = screen.getByRole('radio', { name: /lesion effect \(offline\)/i });
    lesionRadio.dispatchEvent(new Event('change', { bubbles: true }));

    expect(onSwitchColorMode).toHaveBeenCalledWith('lesion');
  });
});

describe('LesionColorMode part="legend"', () => {
  it('renders nothing when the atlas has not loaded ok', () => {
    const { container } = render(LesionColorMode, {
      part: 'legend',
      colorMode: 'lesion',
      lesionAtlasStatus: { status: 'missing', reason: 'no entry (test)' }
    });

    expect(container.querySelector('.legend')).toBeNull();
  });

  it('states the shared scale with live per-graph max values, keeping the numeric endpoints', () => {
    render(LesionColorMode, {
      part: 'legend',
      colorMode: 'lesion',
      lesionAtlasStatus: fakeLesionAtlasOk()
    });

    // Numeric endpoints (absMax = 1 in the fixture) are kept.
    expect(screen.getByText('-1.000')).toBeInTheDocument();
    expect(screen.getByText('+1.000')).toBeInTheDocument();

    // Shared-scale honesty caption with live bioMax (0.3)/rewiredMax (0.5).
    expect(screen.getByText(/one scale, shared across both graphs/i)).toBeInTheDocument();
    expect(screen.getByText(/biological: 0\.300, rewired seed 0: 0\.500/i)).toBeInTheDocument();
  });
});
