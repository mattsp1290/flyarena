import { afterEach, expect, it, vi } from 'vitest';
import { GraphLabSession, initialSession, sessionActive } from '../../src/lib/graphlab/session';
import type { JobRequest } from '../../src/lib/graphlab/types';

const job: JobRequest = { kind: 'lesion', graph: 'biological', sets: [[0]], seedStart: 30001, seedCount: 4, ticks: 300 };
const response = (status: string, error: string | null = null) =>
  new Response(JSON.stringify({ id: 'known-job', kind: 'lesion', status, progress: {}, result: null, error }));
const submitted = () => new Response(JSON.stringify({ id: 'known-job', status: 'queued' }), { status: 202 });

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

it('polls at a 1s interval while active and stops once terminal', async () => {
  vi.useFakeTimers();
  const fetch = vi
    .fn()
    .mockResolvedValueOnce(submitted())
    .mockResolvedValueOnce(response('running'))
    .mockResolvedValueOnce(response('completed'));
  vi.stubGlobal('fetch', fetch);
  let state = initialSession();
  const session = new GraphLabSession((next) => {
    state = next;
  });
  const start = session.start('http://localhost:8766', 'token-0123456789ab', job);
  await vi.advanceTimersByTimeAsync(0);
  await start;
  expect(state.status).toBe('running');
  expect(fetch).toHaveBeenCalledTimes(2);
  await vi.advanceTimersByTimeAsync(1000);
  expect(fetch).toHaveBeenCalledTimes(3);
  expect(state.status).toBe('completed');
  session.dispose();
});

it('surfaces a failed job\'s error and a timed-out job the same way', async () => {
  const fetch = vi.fn().mockResolvedValueOnce(submitted()).mockResolvedValueOnce(response('timed-out', 'Job exceeded its time limit'));
  vi.stubGlobal('fetch', fetch);
  let state = initialSession();
  const session = new GraphLabSession((next) => {
    state = next;
  });
  await session.start('http://localhost:8766', 'token-0123456789ab', job);
  expect(state.status).toBe('timed-out');
  expect(state.error).toBe('Job exceeded its time limit');
  session.dispose();
});

it('treats an authoritative 404 as unavailable, retaining the job id', async () => {
  const fetch = vi
    .fn()
    .mockResolvedValueOnce(submitted())
    .mockRejectedValueOnce(new Error('Offline'))
    .mockResolvedValueOnce(new Response(JSON.stringify({ detail: 'Unknown or expired job' }), { status: 404 }));
  vi.stubGlobal('fetch', fetch);
  let state = initialSession();
  const session = new GraphLabSession((next) => {
    state = next;
  });
  await session.start('http://localhost:8766', 'token-0123456789ab', job);
  expect(sessionActive(state.status)).toBe(true);
  await session.reconnect();
  expect(state.status).toBe('unavailable');
  expect(state.id).toBe('known-job');
  session.dispose();
});

it('makes a best-effort cancel on dispose without waiting for it', async () => {
  const fetch = vi.fn().mockResolvedValueOnce(submitted()).mockResolvedValueOnce(response('running')).mockResolvedValue(response('cancelled'));
  vi.stubGlobal('fetch', fetch);
  const session = new GraphLabSession(() => {});
  await session.start('http://localhost:8766', 'token-0123456789ab', job);
  session.dispose();
  const cancelCall = fetch.mock.calls.find(([, init]) => init?.method === 'DELETE');
  expect(cancelCall).toBeDefined();
});
