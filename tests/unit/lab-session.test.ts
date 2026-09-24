import { afterEach, expect, it, vi } from 'vitest';
import { LabSession, initialSession, sessionActive } from '../../src/lib/lab/session';
import type { Options } from '../../src/lib/lab/types';

const options: Options = { seed: 17, device: 'cpu', population: 4, generations: 1, training_seeds: 4, heldout_seeds: 8, ticks: 30 };
const response = (status: string) => new Response(JSON.stringify({ id: 'known-job', status, progress: {}, result: null, error: null }));
const submitted = () => new Response(JSON.stringify({ id: 'known-job' }), { status: 202 });
const deferred = () => { let resolve!: (value: Response) => void; const promise = new Promise<Response>(r => { resolve = r; }); return { promise, resolve }; };
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });

it('invalidates in-flight polling on cancel/reconnect and leaves exactly one polling loop', async () => {
  vi.useFakeTimers();
  const old = deferred(), fresh = deferred();
  const fetch = vi.fn().mockResolvedValueOnce(submitted()).mockReturnValueOnce(old.promise)
    .mockRejectedValueOnce(new Error('Cancel connection failed')).mockReturnValueOnce(fresh.promise)
    .mockResolvedValue(response('cancelled'));
  vi.stubGlobal('fetch', fetch);
  let state = initialSession();
  const updates = vi.fn(next => { state = next; });
  const session = new LabSession(updates);
  const start = session.start('http://localhost:8765', 'local-test-token', options);
  await vi.advanceTimersByTimeAsync(0);
  expect(fetch).toHaveBeenCalledTimes(2);
  await session.cancel();
  expect(state.error).toContain('Cancel connection failed');
  expect(fetch.mock.calls[1][1].signal.aborted).toBe(true);
  const reconnect = session.reconnect();
  const count = updates.mock.calls.length;
  old.resolve(response('running')); // Even a transport that ignores abort cannot publish this.
  await start;
  expect(updates).toHaveBeenCalledTimes(count);
  fresh.resolve(response('running'));
  await reconnect;
  expect(state.status).toBe('running');
  await vi.advanceTimersByTimeAsync(600);
  expect(fetch.mock.calls.filter(([, init]) => init.method === 'GET')).toHaveLength(3);
  expect(state.status).toBe('cancelled');
  await vi.advanceTimersByTimeAsync(1800);
  expect(fetch).toHaveBeenCalledTimes(5);
  session.dispose();
});

it('releases an expired job only after authoritative 404, retaining its diagnostic identity', async () => {
  const fetch = vi.fn().mockResolvedValueOnce(submitted())
    .mockRejectedValueOnce(new Error('Offline'))
    .mockResolvedValueOnce(new Response(JSON.stringify({ detail: 'Unknown or expired job' }), { status: 404 }))
    .mockResolvedValueOnce(submitted()).mockResolvedValueOnce(response('cancelled'));
  vi.stubGlobal('fetch', fetch);
  let state = initialSession();
  const session = new LabSession(next => { state = next; });
  await session.start('http://localhost:8765', 'local-test-token', options);
  expect(sessionActive(state.status)).toBe(true);
  await session.start('http://localhost:8765', 'local-test-token', options);
  expect(fetch).toHaveBeenCalledTimes(2); // Transient disconnection cannot cause duplicate submission.
  await session.reconnect();
  expect(state.status).toBe('unavailable');
  expect(state.id).toBe('known-job');
  expect(state.error).toContain('no longer retained');
  expect(sessionActive(state.status)).toBe(false);
  await session.start('http://localhost:8765', 'local-test-token', options);
  expect(state.status).toBe('cancelled');
  expect(fetch.mock.calls.filter(([, init]) => init.method === 'POST')).toHaveLength(2);
  session.dispose();
});

it('never lets an obsolete GET replace a terminal cancellation or schedule work after disposal', async () => {
  vi.useFakeTimers();
  const old = deferred();
  const fetch = vi.fn().mockResolvedValueOnce(submitted()).mockReturnValueOnce(old.promise)
    .mockResolvedValueOnce(response('cancelled'));
  vi.stubGlobal('fetch', fetch);
  const updates = vi.fn();
  const session = new LabSession(updates);
  const start = session.start('http://localhost:8765', 'local-test-token', options);
  await vi.advanceTimersByTimeAsync(0);
  await session.cancel();
  expect(updates.mock.lastCall?.[0].status).toBe('cancelled');
  const count = updates.mock.calls.length;
  session.dispose();
  old.resolve(response('running'));
  await start;
  await vi.advanceTimersByTimeAsync(1800);
  expect(updates).toHaveBeenCalledTimes(count);
  expect(fetch).toHaveBeenCalledTimes(3);
});
