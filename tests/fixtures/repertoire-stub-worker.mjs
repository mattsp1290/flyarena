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
//
// Matches on `task.key` (the composite wire/scheduling key,
// `repertoire-plan.ts`'s `planEntryKey`), not `task.graphId` (the plain
// graph identity) -- the real worker's own protocol distinguishes these two
// fields (`repertoire-task.ts`'s `RepertoireWorkerTask` doc comment), and
// this stub's tests rely on `key` being the unique-per-task field.
process.on('message', (task) => {
  if (task.key === 'kill') {
    process.kill(process.pid, 'SIGKILL');
    return;
  }
  if (task.key === 'err') {
    process.send({ type: 'error', key: task.key, message: 'stub-induced failure' });
    return;
  }
  const end = Date.now() + (task.delayMs ?? 0);
  while (Date.now() < end) {
    // Busy-wait: deterministic, no async timer needed for a short delay.
  }
  // A deterministic, content-derived stand-in `RepertoireEvaluatedEntry` --
  // its `occupied`/`gpuArchiveSize` echo the task's own `key` length so the
  // determinism assertion can tell entries apart without depending on
  // completion order.
  process.send({
    type: 'result',
    key: task.key,
    results: [
      {
        graphId: task.graphId ?? task.key,
        arm: 'rewired',
        rewiringSeed: null,
        searchSeed: 1,
        searchOptions: { seed: 1, population: 4, generations: 1, ticks: 30 },
        gpuArchiveSize: task.key.length,
        occupied: task.key.length,
        collisions: 0,
        cells: []
      }
    ]
  });
});
