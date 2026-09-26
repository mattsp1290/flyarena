import { EventEmitter } from 'node:events';

import { describe, expect, it, vi } from 'vitest';

/**
 * Direct, in-process tests of `scripts/null/sharded-evaluation.ts`'s
 * `runShardedEvaluation` (imported by `entry-lesion.ts`/`entry-swapset.ts`
 * after `scripts/graph-lab/shard.ts` was deleted -- a thermo-review
 * finding: the graph-lab-local scheduler had dropped two of its
 * safeguards). `tests/unit/graph-lab-entries.test.ts` already proves the
 * happy path end to end by spawning the real built `.mjs` bundles as
 * subprocesses; this file proves the two failure paths that are
 * impractical to trigger through a real subprocess:
 *
 * - A genuine spawn-level `error` event (Node's own docs: EMFILE/ENOMEM
 *   during `fork()`, or the child process being un-spawnable for some
 *   other OS-level reason) does not correspond to any reliably
 *   reproducible condition in a portable test -- empirically, forking a
 *   *nonexistent* module path (the most obvious thing to try) still
 *   spawns the child process successfully; the failure happens *inside*
 *   that child (`MODULE_NOT_FOUND`) and surfaces as a nonzero `'exit'`,
 *   not an `'error'` event. So this file mocks `node:child_process`'s
 *   `fork` to hand back a plain `EventEmitter` stand-in and emits
 *   `'error'` on it directly, which is the only reliable way to exercise
 *   `runShardedEvaluation`'s `child.on('error', ...)` handler at all.
 * - A worker exiting cleanly (code 0) while a task is still in flight (no
 *   `result`/`error` message ever arrived for it) is similarly awkward to
 *   provoke from a real worker script without either a race condition or
 *   editing a worker to misbehave on purpose -- the mock lets the test
 *   emit `'exit'` deterministically at the exact moment a task is known
 *   to be in flight.
 *
 * Uses `LesionWorkerTask`-shaped literals (graph-lab's own real generic
 * instantiation), not a synthetic type, so this is a test of graph-lab's
 * actual integration with the shared scheduler, not of the scheduler in
 * the abstract.
 */

class FakeChild extends EventEmitter {
  kill = vi.fn();
  disconnect = vi.fn();
  send = vi.fn();
}

vi.mock('node:child_process', () => {
  const fork = vi.fn();
  return { fork, default: { fork } };
});

const { fork } = await import('node:child_process');
const { runShardedEvaluation } = await import('../../scripts/null/sharded-evaluation');

describe('runShardedEvaluation (graph-lab integration): failure paths a real subprocess cannot deterministically trigger', () => {
  it('a spawn-level child.on("error") fails the whole run instead of hanging until the job-store ceiling', async () => {
    const fakeChild = new FakeChild();
    vi.mocked(fork).mockReturnValue(fakeChild as never);

    const tasks = [{ graphId: 'baseline', heldOutSeeds: [1] }];
    const promise = runShardedEvaluation(tasks, 1, '/fake/worker-lesion.mjs');

    // `runWorker`'s `Promise` executor (and its `assignNext()`, which
    // calls the mocked `fork`) runs synchronously before `runShardedEvaluation`'s
    // first `await`, so `fork` has already been called by this point.
    expect(fork).toHaveBeenCalledTimes(1);
    fakeChild.emit('error', new Error('spawn EMFILE'));

    await expect(promise).rejects.toThrow(/worker process error: spawn EMFILE/);
    // The failure path must still kill every tracked child (the `finally`
    // safety net), not merely resolve and move on.
    expect(fakeChild.kill).toHaveBeenCalled();
  });

  it('a worker exiting cleanly (code 0) while a task is still in flight is reported as a failure, not silently accepted', async () => {
    const fakeChild = new FakeChild();
    vi.mocked(fork).mockReturnValue(fakeChild as never);

    const tasks = [{ graphId: 'set-0', heldOutSeeds: [1, 2] }];
    const promise = runShardedEvaluation(tasks, 1, '/fake/worker-lesion.mjs');

    // The task was already dispatched (`child.send`) by the synchronous
    // `assignNext()` above; simulate the worker exiting cleanly without
    // ever sending back a `result`/`error` message for it.
    expect(fakeChild.send).toHaveBeenCalledTimes(1);
    fakeChild.emit('exit', 0, null);

    await expect(promise).rejects.toThrow(/worker exited unexpectedly \(code 0, signal null\) while running "set-0"/);
  });

  it('control case: a worker exiting cleanly with no task in flight is accepted (proves the check is specifically about in-flight tasks)', async () => {
    const fakeChild = new FakeChild();
    vi.mocked(fork).mockReturnValue(fakeChild as never);

    const tasks = [{ graphId: 'only-task', heldOutSeeds: [1] }];
    const promise = runShardedEvaluation(tasks, 1, '/fake/worker-lesion.mjs');

    // Answer the in-flight task first, exactly like a real worker would,
    // then exit cleanly -- this must resolve, not reject.
    fakeChild.emit('message', { type: 'result', graphId: 'only-task', results: [{ seed: 1 }] });
    fakeChild.emit('exit', 0, null);

    await expect(promise).resolves.toEqual(new Map([['only-task', [{ seed: 1 }]]]));
  });
});
