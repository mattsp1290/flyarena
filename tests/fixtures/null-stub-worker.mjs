// Stub worker for tests/unit/null-evaluate.test.ts's `runShardedEvaluation`
// error/abort-path coverage: a real `null-worker.ts` task takes seconds to
// minutes (it parses a real graph and runs real episodes), too slow to
// exercise "one task fails, every other shard stops immediately" or "a
// worker killed by a signal is treated as a failure, not a silent success"
// in a fast unit test. This file implements just enough of the wire
// protocol (`process.on('message')` -> `process.send({type: 'result'|'error', ...})`)
// to drive those paths directly, with no real graph or episode involved.
process.on('message', (task) => {
  if (task.graphId === 'kill') {
    process.kill(process.pid, 'SIGKILL');
    return;
  }
  if (task.graphId === 'err') {
    process.send({ type: 'error', graphId: task.graphId, message: 'stub-induced failure' });
    return;
  }
  const end = Date.now() + (task.delayMs ?? 0);
  while (Date.now() < end) {
    // Busy-wait: deterministic, no async timer needed for a short delay.
  }
  process.send({ type: 'result', graphId: task.graphId, results: [] });
});
