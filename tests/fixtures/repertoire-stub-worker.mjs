// Stub worker for tests/unit/repertoire.test.ts's `runWorkerScheduler`
// shard-determinism/failure-path coverage -- mirrors
// tests/fixtures/null-stub-worker.mjs's own rationale: a real
// `repertoire-worker.ts` task (`verifyAndEvaluateSearchGraph`) parses a real
// graph and runs real discovery/held-out episodes, too slow to exercise
// "one task fails, every shard stops immediately" or "results are
// independent of completion order" in a fast unit test. Implements just
// enough of the wire protocol (`process.on('message')` ->
// `process.send({type: 'result'|'error', ...})`, with `results` always a
// one-element array, matching `RepertoireWorkerMessage`'s shape) to drive
// those paths directly, with no real search file or episode involved.
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
  // A deterministic, content-derived stand-in `RepertoireEvaluatedEntry` --
  // its `occupied` count echoes the task's own graphId length so the
  // determinism assertion can tell entries apart without depending on
  // completion order.
  process.send({
    type: 'result',
    graphId: task.graphId,
    results: [
      {
        graphId: task.graphId,
        arm: 'rewired',
        rewiringSeed: null,
        searchSeed: 1,
        gpuArchiveSize: task.graphId.length,
        occupied: task.graphId.length,
        collisions: 0,
        cells: []
      }
    ]
  });
});
