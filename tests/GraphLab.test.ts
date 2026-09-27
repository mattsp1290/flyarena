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
