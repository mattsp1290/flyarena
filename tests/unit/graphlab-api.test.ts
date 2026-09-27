import { afterEach, describe, expect, it, vi } from 'vitest';
import { GraphLabApi } from '../../src/lib/graphlab/api';

afterEach(() => vi.unstubAllGlobals());

describe('GraphLabApi', () => {
  it('rejects credentials and non-http endpoints', () => {
    expect(() => new GraphLabApi('file:///tmp/a', '')).toThrow();
    expect(() => new GraphLabApi('https://user:secret@example.com', '')).toThrow();
    expect(() => new GraphLabApi('http://127.0.0.1:8766?x=1', '')).toThrow();
    expect(() => new GraphLabApi('http://127.0.0.1:8766#frag', '')).toThrow();
  });

  it('reports non-JSON and backend errors without retrying', async () => {
    const fetch = vi.fn().mockResolvedValue(new Response('offline', { status: 502 }));
    vi.stubGlobal('fetch', fetch);
    const api = new GraphLabApi('http://127.0.0.1:8766', 'memory-only-token');
    await expect(api.status('abc', new AbortController().signal)).rejects.toThrow('non-JSON');
    expect(fetch).toHaveBeenCalledTimes(1);
    fetch.mockResolvedValue(new Response(JSON.stringify({ detail: 'busy' }), { status: 409 }));
    await expect(api.status('abc', new AbortController().signal)).rejects.toThrow('409: busy');
  });

  it('falls back to a fixed message when `detail` is not a string (Pydantic 422 shape)', async () => {
    const fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: [{ loc: ['body', 'ticks'], msg: 'too small', type: 'value_error' }] }), {
        status: 422
      })
    );
    vi.stubGlobal('fetch', fetch);
    const api = new GraphLabApi('http://127.0.0.1:8766', 'memory-only-token');
    await expect(api.status('abc', new AbortController().signal)).rejects.toThrow('422: Invalid request settings.');
  });

  it('calls /health without an Authorization header, and every other call with one', async () => {
    const fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ status: 'ok', modelVersion: 'v', bundleSha256: null, graphSha256: null, gpu: { available: false } }))
    );
    vi.stubGlobal('fetch', fetch);
    const api = new GraphLabApi('http://127.0.0.1:8766', 'memory-only-token');
    await api.health(new AbortController().signal);
    const [url, init] = fetch.mock.calls[0];
    expect(String(url)).toContain('/api/graph/v1/health');
    expect((init.headers as Record<string, string>).Authorization).toBeUndefined();

    fetch.mockResolvedValue(new Response(JSON.stringify({ id: 'abc', kind: 'lesion', status: 'queued', progress: {}, result: null, error: null })));
    await api.status('abc', new AbortController().signal);
    const [, statusInit] = fetch.mock.calls[1];
    expect((statusInit.headers as Record<string, string>).Authorization).toBe('Bearer memory-only-token');
  });

  it('never sends the token anywhere but the Authorization header', async () => {
    const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ id: 'abc', status: 'queued' }), { status: 202 }));
    vi.stubGlobal('fetch', fetch);
    const api = new GraphLabApi('http://127.0.0.1:8766', 'super-secret-token-value');
    await api.submit(
      { kind: 'lesion', graph: 'biological', sets: [[0]], seedStart: 1, seedCount: 4, ticks: 300 },
      new AbortController().signal
    );
    const [url, init] = fetch.mock.calls[0];
    expect(String(url)).not.toContain('super-secret-token-value');
    expect(init.body as string).not.toContain('super-secret-token-value');
  });
});
