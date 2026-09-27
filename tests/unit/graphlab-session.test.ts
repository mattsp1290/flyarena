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

// A fix-verification review finding on an earlier version of this module:
// `start()` stores its own `GraphLabApi` instance (and the token inside it)
// in `this.api`, and nothing ever cleared it -- not `cancel()`, not
// `dispose()` -- so the last instance, and its token, stayed reachable
// through the session object for as long as that object itself lived. The
// tests below assert `release()` (and `dispose()`, which now calls it)
// actually drop that reference, using both a direct field check and a
// structural (`JSON.stringify`) inspection per the review's own suggestion:
// `JSON.stringify` naturally omits a property whose value is `undefined`,
// so this doubles as a token-reachability check without needing to know
// every private field name on `GraphLabApi` itself.
it('release() drops the session\'s GraphLabApi instance, so the token is no longer reachable on the session', async () => {
  // A terminal ('completed') status, not 'running': `refresh()` only
  // schedules its next-poll `setTimeout` while `sessionActive(...)` is
  // true, and a scheduled Node `Timeout` is a circularly-linked object
  // (`_idlePrev`/`_idleNext`) that `JSON.stringify` cannot serialize --
  // unrelated to what this test checks, so a terminal status keeps
  // `this.timer` unset and the structural inspection below meaningful.
  const fetch = vi.fn().mockResolvedValueOnce(submitted()).mockResolvedValueOnce(response('completed'));
  vi.stubGlobal('fetch', fetch);
  const session = new GraphLabSession(() => {});
  const secretToken = 'reachability-test-token-0123456789';
  await session.start('http://localhost:8766', secretToken, job);

  // Before release: the token really is reachable this way, proving the
  // check below is meaningful rather than vacuously true.
  expect(JSON.stringify(session)).toContain(secretToken);

  session.release();

  expect((session as unknown as { api?: unknown }).api).toBeUndefined();
  expect(JSON.stringify(session)).not.toContain(secretToken);
  session.dispose();
});

it('release() does not abort a cancel already in flight (matches GraphLab.svelte\'s cancel-then-release order)', async () => {
  const fetch = vi.fn().mockResolvedValueOnce(submitted()).mockResolvedValueOnce(response('running')).mockResolvedValue(response('cancelled'));
  vi.stubGlobal('fetch', fetch);
  const session = new GraphLabSession(() => {});
  await session.start('http://localhost:8766', 'token-0123456789ab', job);

  // `refresh()` captures its own local `api` reference synchronously,
  // before its first `await` -- so `release()` called immediately after
  // `cancel()` (never awaited in between) must not prevent that already-
  // dispatched DELETE from completing.
  const cancelPromise = session.cancel();
  session.release();
  await cancelPromise;

  const cancelCall = fetch.mock.calls.find(([, init]) => init?.method === 'DELETE');
  expect(cancelCall).toBeDefined();
  session.dispose();
});

it('dispose() also releases the api, leaving no token reachable', async () => {
  const fetch = vi.fn().mockResolvedValueOnce(submitted()).mockResolvedValueOnce(response('running')).mockResolvedValue(response('cancelled'));
  vi.stubGlobal('fetch', fetch);
  const session = new GraphLabSession(() => {});
  const secretToken = 'dispose-reachability-token-0123456789';
  await session.start('http://localhost:8766', secretToken, job);

  session.dispose();

  expect((session as unknown as { api?: unknown }).api).toBeUndefined();
  expect(JSON.stringify(session)).not.toContain(secretToken);
});
