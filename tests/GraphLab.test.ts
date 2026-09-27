import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import GraphLab from '../src/lib/graphlab/GraphLab.svelte';

function stubFetch(handler: (url: string) => Response | Promise<Response>) {
  const fetch = vi.fn((input: RequestInfo | URL) => {
    try {
      return Promise.resolve(handler(String(input)));
    } catch (error) {
      return Promise.reject(error);
    }
  });
  vi.stubGlobal('fetch', fetch);
  return fetch;
}

const notFound = () => new Response('not found', { status: 404 });

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('GraphLab security: no browser storage or logging of the endpoint/token', () => {
  it('never writes to localStorage or sessionStorage while connecting and typing a token', async () => {
    const localSetItem = vi.spyOn(window.localStorage, 'setItem');
    const sessionSetItem = vi.spyOn(window.sessionStorage, 'setItem');
    const consoleLog = vi.spyOn(console, 'log').mockImplementation(() => {});
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {});
    stubFetch((url) => {
      if (url.includes('/api/graph/v1/health')) {
        return new Response(
          JSON.stringify({
            status: 'ok',
            modelVersion: 'arena-graph-lab-v1',
            bundleSha256: 'a'.repeat(64),
            graphSha256: 'b'.repeat(64),
            gpu: { available: true }
          })
        );
      }
      return notFound();
    });
    const { container } = render(GraphLab);
    const secretToken = 'super-secret-token-0123456789';
    await fireEvent.input(container.querySelector('input[type="url"]')!, {
      target: { value: 'http://127.0.0.1:8766' }
    });
    await fireEvent.input(container.querySelector('input[type="password"]')!, {
      target: { value: secretToken }
    });
    await fireEvent.click(screen.getByRole('button', { name: 'Connect' }));
    await waitFor(() => expect(screen.getByText(/Bundle SHA-256/)).toBeInTheDocument());

    expect(localSetItem).not.toHaveBeenCalled();
    expect(sessionSetItem).not.toHaveBeenCalled();
    expect(document.cookie).toBe('');
    const loggedText = [...consoleLog.mock.calls, ...consoleError.mock.calls].flat().join(' ');
    expect(loggedText).not.toContain(secretToken);
  });

  it('clears the token on disconnect', async () => {
    stubFetch((url) =>
      url.includes('/api/graph/v1/health')
        ? new Response(
            JSON.stringify({ status: 'ok', modelVersion: 'v', bundleSha256: null, graphSha256: null, gpu: { available: false } })
          )
        : notFound()
    );
    const { container } = render(GraphLab);
    const tokenInput = container.querySelector('input[type="password"]') as HTMLInputElement;
    await fireEvent.input(tokenInput, { target: { value: 'a-token-that-should-be-cleared' } });
    await fireEvent.click(screen.getByRole('button', { name: 'Connect' }));
    await waitFor(() => expect(screen.getByText(/Bundle SHA-256/)).toBeInTheDocument());
    await fireEvent.click(screen.getByRole('button', { name: 'Disconnect' }));
    expect(tokenInput.value).toBe('');
  });

  it('disables Disconnect while a job is active, so an in-flight job cannot be stranded with no visible Cancel button', async () => {
    stubFetch((url) =>
      url.includes('/api/graph/v1/health')
        ? new Response(
            JSON.stringify({ status: 'ok', modelVersion: 'v', bundleSha256: null, graphSha256: null, gpu: { available: false } })
          )
        : notFound()
    );
    const { container } = render(GraphLab);
    await fireEvent.input(container.querySelector('input[type="password"]')!, { target: { value: 'a-valid-16-char-token' } });
    await fireEvent.click(screen.getByRole('button', { name: 'Connect' }));
    await waitFor(() => expect(screen.getByText(/Bundle SHA-256/)).toBeInTheDocument());

    const fetch = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/graph/v1/jobs') && init?.method === 'POST') {
        return Promise.resolve(new Response(JSON.stringify({ id: 'job-1', status: 'queued' }), { status: 202 }));
      }
      if (url.includes('/api/graph/v1/jobs/')) {
        return Promise.resolve(
          new Response(
            JSON.stringify({ id: 'job-1', kind: 'lesion', status: 'running', progress: {}, error: null, result: null })
          )
        );
      }
      return Promise.resolve(notFound());
    });
    vi.stubGlobal('fetch', fetch);

    await fireEvent.input(container.querySelector('textarea')!, { target: { value: '7' } });
    await fireEvent.click(screen.getByRole('button', { name: 'Submit job' }));
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('running'));
    expect(screen.getByRole('button', { name: 'Disconnect' })).toBeDisabled();
  });
});

describe('GraphLab backend-unavailable state', () => {
  it('shows a clear message when the health check cannot reach a backend', async () => {
    stubFetch(() => {
      throw new TypeError('Failed to fetch');
    });
    render(GraphLab);
    await fireEvent.click(screen.getByRole('button', { name: 'Connect' }));
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('Backend unavailable'));
    expect(screen.queryByRole('tablist')).toBeNull();
  });
});

describe('GraphLab result rendering', () => {
  it('renders a lesion result with its provenance label', async () => {
    let submittedBody: unknown;
    stubFetch((url) => {
      if (url.includes('/api/graph/v1/health')) {
        return new Response(
          JSON.stringify({
            status: 'ok',
            modelVersion: 'arena-graph-lab-v1',
            bundleSha256: 'c'.repeat(64),
            graphSha256: 'd'.repeat(64),
            gpu: { available: false }
          })
        );
      }
      return notFound();
    });
    const { container } = render(GraphLab);
    await fireEvent.input(container.querySelector('input[type="password"]')!, {
      target: { value: 'a-valid-16-char-token' }
    });
    await fireEvent.click(screen.getByRole('button', { name: 'Connect' }));
    await waitFor(() => expect(screen.getByText(/Bundle SHA-256/)).toBeInTheDocument());

    const fetch = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/graph/v1/jobs') && init?.method === 'POST') {
        submittedBody = init.body;
        return Promise.resolve(new Response(JSON.stringify({ id: 'job-1', status: 'queued' }), { status: 202 }));
      }
      if (url.includes('/api/graph/v1/jobs/')) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              id: 'job-1',
              kind: 'lesion',
              status: 'completed',
              progress: {},
              error: null,
              result: {
                graph: 'biological',
                graphSha256: 'd'.repeat(64),
                host: { arch: 'arm64', node: 'v22.22.3' },
                label: 'Computed on DGX (private, not published)',
                baseline: { n: 4, mean: 1.5 },
                sets: [
                  {
                    indices: [7],
                    bodyIds: ['10010'],
                    effect: { n: 4, meanDifference: -0.4, ci95: [-0.6, -0.2] },
                    n: 4
                  }
                ]
              }
            })
          )
        );
      }
      return Promise.resolve(notFound());
    });
    vi.stubGlobal('fetch', fetch);

    await fireEvent.input(container.querySelector('textarea')!, { target: { value: '7' } });
    await fireEvent.click(screen.getByRole('button', { name: 'Submit job' }));
    await waitFor(() => expect(screen.getByText('Computed on DGX (private, not published)')).toBeInTheDocument());
    expect(screen.getByText(/10010/)).toBeInTheDocument();
    // The token is sent only as the `Authorization` header (`api.test.ts` covers that
    // directly) -- the submitted job body itself must never carry it.
    expect(String(submittedBody)).not.toContain('a-valid-16-char-token');
    expect(String(submittedBody)).toContain('"kind":"lesion"');
  });

  it('resets the selected atlas cell across submits, even when the new result reuses the same cell id', async () => {
    stubFetch((url) =>
      url.includes('/api/graph/v1/health')
        ? new Response(
            JSON.stringify({ status: 'ok', modelVersion: 'v', bundleSha256: null, graphSha256: null, gpu: { available: false } })
          )
        : notFound()
    );
    const atlasResultBody = (cellPosition: number) => ({
      id: 'job-atlas',
      kind: 'atlas',
      status: 'completed',
      progress: {},
      error: null,
      result: {
        dataDir: '/tmp',
        host: { arch: 'arm64', node: 'v22.22.3' },
        label: 'Computed on DGX (private, not published)',
        evaluations: [],
        // Same cell `id` (5) both times -- cell ids are per-search, not
        // globally unique, so a stale `selectedCellId` from a previous run
        // could otherwise resolve against this new result without a click.
        cells: [{ id: 5, cell: cellPosition, quality: 1.0, coverage: 0.5, turning: 0.1, discovery: [], heldout: {}, replay: [] }],
        gpuArchiveSize: 1,
        collisions: 0
      }
    });
    const { container } = render(GraphLab);
    await fireEvent.input(container.querySelector('input[type="password"]')!, { target: { value: 'a-valid-16-char-token' } });
    await fireEvent.click(screen.getByRole('button', { name: 'Connect' }));
    await waitFor(() => expect(screen.getByText(/Bundle SHA-256/)).toBeInTheDocument());
    await fireEvent.click(screen.getByRole('tab', { name: 'Atlas search' }));

    let jobCounter = 0;
    const fetch = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/graph/v1/jobs') && init?.method === 'POST') {
        jobCounter += 1;
        return Promise.resolve(new Response(JSON.stringify({ id: 'job-atlas', status: 'queued' }), { status: 202 }));
      }
      // First run's cell sits at grid position 0; the second run's at 1, so
      // their aria-labels never collide even though `id` repeats.
      return Promise.resolve(new Response(JSON.stringify(atlasResultBody(jobCounter === 1 ? 0 : 1))));
    });
    vi.stubGlobal('fetch', fetch);

    await fireEvent.click(screen.getByRole('button', { name: 'Submit job' }));
    await waitFor(() => expect(screen.getByRole('button', { name: /Cell 0,/ })).toBeInTheDocument());
    await fireEvent.click(screen.getByRole('button', { name: /Cell 0,/ }));
    await waitFor(() => expect(screen.getByText(/Cell #0:/)).toBeInTheDocument());

    await fireEvent.click(screen.getByRole('button', { name: 'Submit job' }));
    await waitFor(() => expect(screen.getByRole('button', { name: /Cell 1,/ })).toBeInTheDocument());
    expect(screen.queryByText(/Cell #/)).toBeNull();
  });
});

describe('GraphLab route-change token lifetime (thermo-security I1)', () => {
  it('cancels an active job and clears the token when the route is left, and never restores it on return', async () => {
    let cancelCalled = false;
    stubFetch((url) =>
      url.includes('/api/graph/v1/health')
        ? new Response(
            JSON.stringify({ status: 'ok', modelVersion: 'v', bundleSha256: 'a'.repeat(64), graphSha256: 'b'.repeat(64), gpu: { available: false } })
          )
        : notFound()
    );
    const { container, rerender } = render(GraphLab, { props: { active: true } });
    const secretToken = 'a-token-that-must-not-outlive-the-route';
    await fireEvent.input(container.querySelector('input[type="password"]')!, { target: { value: secretToken } });
    await fireEvent.click(screen.getByRole('button', { name: 'Connect' }));
    await waitFor(() => expect(screen.getByText(/Bundle SHA-256/)).toBeInTheDocument());

    const fetch = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/graph/v1/jobs') && init?.method === 'POST') {
        return Promise.resolve(new Response(JSON.stringify({ id: 'job-1', status: 'queued' }), { status: 202 }));
      }
      if (url.includes('/api/graph/v1/jobs/') && init?.method === 'DELETE') {
        cancelCalled = true;
        return Promise.resolve(
          new Response(JSON.stringify({ id: 'job-1', kind: 'lesion', status: 'cancelling', progress: {}, error: null, result: null }))
        );
      }
      if (url.includes('/api/graph/v1/jobs/')) {
        return Promise.resolve(
          new Response(JSON.stringify({ id: 'job-1', kind: 'lesion', status: 'running', progress: {}, error: null, result: null }))
        );
      }
      return Promise.resolve(notFound());
    });
    vi.stubGlobal('fetch', fetch);

    await fireEvent.input(container.querySelector('textarea')!, { target: { value: '7' } });
    await fireEvent.click(screen.getByRole('button', { name: 'Submit job' }));
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('running'));

    // Simulate `Shell.svelte` navigating `view` away from `#graph-lab`.
    await rerender({ active: false });

    await waitFor(() => expect(cancelCalled).toBe(true));
    const tokenInput = container.querySelector('input[type="password"]') as HTMLInputElement;
    expect(tokenInput.value).toBe('');
    // The connection/health panel is gone too -- returning to the route
    // requires re-entering the token, exactly like a fresh visit.
    expect(screen.queryByText(/Bundle SHA-256/)).toBeNull();

    // Returning to the route must not resurrect the old token from
    // anywhere: `disconnect()` clears this component's own `token` field
    // *and* calls `session.release()`, which drops the session's own
    // `GraphLabApi` instance (the token's other, previously-uncleared,
    // home -- see `session.ts`'s `release()` doc comment) -- covered
    // directly by the `GraphLabSession` reachability test below.
    await rerender({ active: true });
    expect((container.querySelector('input[type="password"]') as HTMLInputElement).value).toBe('');
  });

  it('does nothing on the initial mount when active is true (no spurious disconnect)', async () => {
    stubFetch((url) =>
      url.includes('/api/graph/v1/health')
        ? new Response(
            JSON.stringify({ status: 'ok', modelVersion: 'v', bundleSha256: null, graphSha256: null, gpu: { available: false } })
          )
        : notFound()
    );
    const { container } = render(GraphLab, { props: { active: true } });
    await fireEvent.input(container.querySelector('input[type="password"]')!, { target: { value: 'a-valid-16-char-token' } });
    await fireEvent.click(screen.getByRole('button', { name: 'Connect' }));
    await waitFor(() => expect(screen.getByText(/Bundle SHA-256/)).toBeInTheDocument());
    expect((container.querySelector('input[type="password"]') as HTMLInputElement).value).toBe('a-valid-16-char-token');
  });
});

describe('GraphLab provenance staleness (thermo-maintainability)', () => {
  it('re-fetches /health at submit and at completion, and flags the label when the bundle sha changed mid-run', async () => {
    let healthCallCount = 0;
    stubFetch((url) => {
      if (url.includes('/api/graph/v1/health')) {
        healthCallCount += 1;
        // First call is the manual "Connect"; second is submit-time; third
        // is the post-completion re-check, which reports a *different*
        // bundle sha, simulating a backend restart mid-run.
        const bundleSha256 = healthCallCount >= 3 ? 'c'.repeat(64) : 'a'.repeat(64);
        return new Response(
          JSON.stringify({ status: 'ok', modelVersion: 'v', bundleSha256, graphSha256: 'b'.repeat(64), gpu: { available: false } })
        );
      }
      return notFound();
    });
    const { container } = render(GraphLab);
    await fireEvent.input(container.querySelector('input[type="password"]')!, { target: { value: 'a-valid-16-char-token' } });
    await fireEvent.click(screen.getByRole('button', { name: 'Connect' }));
    await waitFor(() => expect(screen.getByText(/Bundle SHA-256/)).toBeInTheDocument());
    expect(healthCallCount).toBe(1);

    const fetch = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/graph/v1/health')) {
        healthCallCount += 1;
        const bundleSha256 = healthCallCount >= 3 ? 'c'.repeat(64) : 'a'.repeat(64);
        return Promise.resolve(
          new Response(
            JSON.stringify({ status: 'ok', modelVersion: 'v', bundleSha256, graphSha256: 'b'.repeat(64), gpu: { available: false } })
          )
        );
      }
      if (url.includes('/api/graph/v1/jobs') && init?.method === 'POST') {
        return Promise.resolve(new Response(JSON.stringify({ id: 'job-1', status: 'queued' }), { status: 202 }));
      }
      if (url.includes('/api/graph/v1/jobs/')) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              id: 'job-1',
              kind: 'lesion',
              status: 'completed',
              progress: {},
              error: null,
              result: {
                graph: 'biological',
                graphSha256: 'b'.repeat(64),
                host: { arch: 'arm64', node: 'v22.22.3' },
                label: 'Computed on DGX (private, not published)',
                baseline: { n: 4, mean: 1.5 },
                sets: [{ indices: [7], bodyIds: ['10010'], effect: { n: 4, meanDifference: -0.4, ci95: [-0.6, -0.2] }, n: 4 }]
              }
            })
          )
        );
      }
      return Promise.resolve(notFound());
    });
    vi.stubGlobal('fetch', fetch);

    await fireEvent.input(container.querySelector('textarea')!, { target: { value: '7' } });
    await fireEvent.click(screen.getByRole('button', { name: 'Submit job' }));
    await waitFor(() => expect(screen.getByText('Computed on DGX (private, not published)')).toBeInTheDocument());
    await waitFor(() => expect(document.querySelector('.provenance')).toHaveTextContent(/backend changed during this job/i));
  });

  it('does not flag provenance when the backend identity stays the same across the run', async () => {
    stubFetch((url) =>
      url.includes('/api/graph/v1/health')
        ? new Response(
            JSON.stringify({ status: 'ok', modelVersion: 'v', bundleSha256: 'a'.repeat(64), graphSha256: 'b'.repeat(64), gpu: { available: false } })
          )
        : notFound()
    );
    const { container } = render(GraphLab);
    await fireEvent.input(container.querySelector('input[type="password"]')!, { target: { value: 'a-valid-16-char-token' } });
    await fireEvent.click(screen.getByRole('button', { name: 'Connect' }));
    await waitFor(() => expect(screen.getByText(/Bundle SHA-256/)).toBeInTheDocument());

    const fetch = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/graph/v1/health')) {
        return Promise.resolve(
          new Response(
            JSON.stringify({ status: 'ok', modelVersion: 'v', bundleSha256: 'a'.repeat(64), graphSha256: 'b'.repeat(64), gpu: { available: false } })
          )
        );
      }
      if (url.includes('/api/graph/v1/jobs') && init?.method === 'POST') {
        return Promise.resolve(new Response(JSON.stringify({ id: 'job-1', status: 'queued' }), { status: 202 }));
      }
      if (url.includes('/api/graph/v1/jobs/')) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              id: 'job-1',
              kind: 'lesion',
              status: 'completed',
              progress: {},
              error: null,
              result: {
                graph: 'biological',
                graphSha256: 'b'.repeat(64),
                host: { arch: 'arm64', node: 'v22.22.3' },
                label: 'Computed on DGX (private, not published)',
                baseline: { n: 4, mean: 1.5 },
                sets: [{ indices: [7], bodyIds: ['10010'], effect: { n: 4, meanDifference: -0.4, ci95: [-0.6, -0.2] }, n: 4 }]
              }
            })
          )
        );
      }
      return Promise.resolve(notFound());
    });
    vi.stubGlobal('fetch', fetch);

    await fireEvent.input(container.querySelector('textarea')!, { target: { value: '7' } });
    await fireEvent.click(screen.getByRole('button', { name: 'Submit job' }));
    await waitFor(() => expect(screen.getByText('Computed on DGX (private, not published)')).toBeInTheDocument());
    await waitFor(() => expect(document.querySelector('.provenance')).not.toHaveTextContent(/backend changed/i));
  });

  // A fix-verification review finding: when the completion-time re-check
  // itself failed (network error, non-2xx, bad JSON -- `fetchHealthSnapshot`
  // returns `null`), the mismatch flag previously stayed at its default
  // `false`, rendering a clean-looking label for a claim that was never
  // actually re-confirmed. The two tests below cover both failure points
  // (completion-time and submit-time) with their own distinct wording.
  it('marks the label unconfirmed (not silently clean) when the completion-time /health re-check itself fails', async () => {
    let healthCallCount = 0;
    stubFetch((url) => {
      if (url.includes('/api/graph/v1/health')) {
        healthCallCount += 1;
        return new Response(
          JSON.stringify({ status: 'ok', modelVersion: 'v', bundleSha256: 'a'.repeat(64), graphSha256: 'b'.repeat(64), gpu: { available: false } })
        );
      }
      return notFound();
    });
    const { container } = render(GraphLab);
    await fireEvent.input(container.querySelector('input[type="password"]')!, { target: { value: 'a-valid-16-char-token' } });
    await fireEvent.click(screen.getByRole('button', { name: 'Connect' }));
    await waitFor(() => expect(screen.getByText(/Bundle SHA-256/)).toBeInTheDocument());
    expect(healthCallCount).toBe(1);

    const fetch = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/graph/v1/health')) {
        healthCallCount += 1;
        // Call 2 is the submit-time re-fetch (succeeds); call 3 is the
        // completion-time re-check, which fails outright.
        if (healthCallCount >= 3) return Promise.reject(new TypeError('Failed to fetch'));
        return Promise.resolve(
          new Response(
            JSON.stringify({ status: 'ok', modelVersion: 'v', bundleSha256: 'a'.repeat(64), graphSha256: 'b'.repeat(64), gpu: { available: false } })
          )
        );
      }
      if (url.includes('/api/graph/v1/jobs') && init?.method === 'POST') {
        return Promise.resolve(new Response(JSON.stringify({ id: 'job-1', status: 'queued' }), { status: 202 }));
      }
      if (url.includes('/api/graph/v1/jobs/')) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              id: 'job-1',
              kind: 'lesion',
              status: 'completed',
              progress: {},
              error: null,
              result: {
                graph: 'biological',
                graphSha256: 'b'.repeat(64),
                host: { arch: 'arm64', node: 'v22.22.3' },
                label: 'Computed on DGX (private, not published)',
                baseline: { n: 4, mean: 1.5 },
                sets: [{ indices: [7], bodyIds: ['10010'], effect: { n: 4, meanDifference: -0.4, ci95: [-0.6, -0.2] }, n: 4 }]
              }
            })
          )
        );
      }
      return Promise.resolve(notFound());
    });
    vi.stubGlobal('fetch', fetch);

    await fireEvent.input(container.querySelector('textarea')!, { target: { value: '7' } });
    await fireEvent.click(screen.getByRole('button', { name: 'Submit job' }));
    await waitFor(() => expect(screen.getByText('Computed on DGX (private, not published)')).toBeInTheDocument());
    await waitFor(() =>
      expect(document.querySelector('.provenance')).toHaveTextContent(/could not be re-confirmed at completion/i)
    );
    // Never the silent, mismatch-only wording for this failure mode.
    expect(document.querySelector('.provenance')).not.toHaveTextContent(/backend changed during this job/i);
  });

  it('marks the label unconfirmed when the submit-time /health re-fetch itself fails', async () => {
    stubFetch((url) =>
      url.includes('/api/graph/v1/health')
        ? new Response(
            JSON.stringify({ status: 'ok', modelVersion: 'v', bundleSha256: 'a'.repeat(64), graphSha256: 'b'.repeat(64), gpu: { available: false } })
          )
        : notFound()
    );
    const { container } = render(GraphLab);
    await fireEvent.input(container.querySelector('input[type="password"]')!, { target: { value: 'a-valid-16-char-token' } });
    await fireEvent.click(screen.getByRole('button', { name: 'Connect' }));
    await waitFor(() => expect(screen.getByText(/Bundle SHA-256/)).toBeInTheDocument());

    const fetch = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      // Every /health call from this point on fails -- including the
      // submit-time re-fetch and the completion-time re-check.
      if (url.includes('/api/graph/v1/health')) return Promise.reject(new TypeError('Failed to fetch'));
      if (url.includes('/api/graph/v1/jobs') && init?.method === 'POST') {
        return Promise.resolve(new Response(JSON.stringify({ id: 'job-1', status: 'queued' }), { status: 202 }));
      }
      if (url.includes('/api/graph/v1/jobs/')) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              id: 'job-1',
              kind: 'lesion',
              status: 'completed',
              progress: {},
              error: null,
              result: {
                graph: 'biological',
                graphSha256: 'b'.repeat(64),
                host: { arch: 'arm64', node: 'v22.22.3' },
                label: 'Computed on DGX (private, not published)',
                baseline: { n: 4, mean: 1.5 },
                sets: [{ indices: [7], bodyIds: ['10010'], effect: { n: 4, meanDifference: -0.4, ci95: [-0.6, -0.2] }, n: 4 }]
              }
            })
          )
        );
      }
      return Promise.resolve(notFound());
    });
    vi.stubGlobal('fetch', fetch);

    await fireEvent.input(container.querySelector('textarea')!, { target: { value: '7' } });
    await fireEvent.click(screen.getByRole('button', { name: 'Submit job' }));
    await waitFor(() => expect(screen.getByText('Computed on DGX (private, not published)')).toBeInTheDocument());
    await waitFor(() =>
      expect(document.querySelector('.provenance')).toHaveTextContent(/could not be confirmed at submit time/i)
    );
    // A failed submit-time snapshot renders "unknown" shas, not a stale
    // cached value from the earlier manual Connect click.
    expect(document.querySelector('.provenance')?.textContent).toContain('unknown');
  });
});

describe('GraphLab ARIA tabs: keyboard navigation and roving tabindex', () => {
  it('supports ArrowRight/ArrowLeft/Home/End with a roving tabindex, and keeps the tabpanel in sync', async () => {
    stubFetch((url) =>
      url.includes('/api/graph/v1/health')
        ? new Response(
            JSON.stringify({ status: 'ok', modelVersion: 'v', bundleSha256: null, graphSha256: null, gpu: { available: false } })
          )
        : notFound()
    );
    const { container } = render(GraphLab);
    await fireEvent.input(container.querySelector('input[type="password"]')!, { target: { value: 'a-valid-16-char-token' } });
    await fireEvent.click(screen.getByRole('button', { name: 'Connect' }));
    await waitFor(() => expect(screen.getByRole('tablist')).toBeInTheDocument());

    const tablist = screen.getByRole('tablist');
    const lesionTab = screen.getByRole('tab', { name: 'Lesion sweep' });
    const atlasTab = screen.getByRole('tab', { name: 'Atlas search' });
    const swapsetTab = screen.getByRole('tab', { name: 'Swap-set intervention' });
    const tabpanel = document.getElementById('graphlab-tabpanel');

    expect(lesionTab).toHaveAttribute('aria-selected', 'true');
    expect(lesionTab).toHaveAttribute('tabindex', '0');
    expect(atlasTab).toHaveAttribute('tabindex', '-1');
    expect(swapsetTab).toHaveAttribute('tabindex', '-1');
    expect(tabpanel).toHaveAttribute('aria-labelledby', 'graphlab-tab-lesion');
    expect(lesionTab).toHaveAttribute('aria-controls', 'graphlab-tabpanel');

    await fireEvent.keyDown(tablist, { key: 'ArrowRight' });
    expect(atlasTab).toHaveAttribute('aria-selected', 'true');
    expect(atlasTab).toHaveAttribute('tabindex', '0');
    expect(lesionTab).toHaveAttribute('aria-selected', 'false');
    expect(lesionTab).toHaveAttribute('tabindex', '-1');
    expect(tabpanel).toHaveAttribute('aria-labelledby', 'graphlab-tab-atlas');

    await fireEvent.keyDown(tablist, { key: 'ArrowRight' });
    expect(swapsetTab).toHaveAttribute('aria-selected', 'true');
    expect(swapsetTab).toHaveAttribute('tabindex', '0');

    // Wraps from the last tab back to the first.
    await fireEvent.keyDown(tablist, { key: 'ArrowRight' });
    expect(lesionTab).toHaveAttribute('aria-selected', 'true');

    // Wraps from the first tab back to the last.
    await fireEvent.keyDown(tablist, { key: 'ArrowLeft' });
    expect(swapsetTab).toHaveAttribute('aria-selected', 'true');

    await fireEvent.keyDown(tablist, { key: 'Home' });
    expect(lesionTab).toHaveAttribute('aria-selected', 'true');
    expect(lesionTab).toHaveAttribute('tabindex', '0');

    await fireEvent.keyDown(tablist, { key: 'End' });
    expect(swapsetTab).toHaveAttribute('aria-selected', 'true');
    expect(swapsetTab).toHaveAttribute('tabindex', '0');
    expect(tabpanel).toHaveAttribute('aria-labelledby', 'graphlab-tab-swapset');

    // A key this handler doesn't own must not change the selection.
    await fireEvent.keyDown(tablist, { key: 'Tab' });
    expect(swapsetTab).toHaveAttribute('aria-selected', 'true');
  });
});
