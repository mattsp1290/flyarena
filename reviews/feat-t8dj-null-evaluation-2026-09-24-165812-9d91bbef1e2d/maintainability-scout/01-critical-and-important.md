# Critical and Important Issues

## Critical

None. The committed artifact looks internally consistent (500 rewired entries, seeds 0..499, finite null summary, `degenerate: false`), and I found no crash on the happy path.

## Important

### I1. A NaN/Infinity score silently becomes `0` in the published statistics

- **Files:** `scripts/null/null-worker.ts:107-120` (`runTask`), `scripts/null/null-evaluate.ts:367` (`assembleRaw`) and `:423` (`writeFileSync(args.out, JSON.stringify(raw))`), `scripts/null/null-report.ts:387` (`runNullReport`)
- **Problem:** Nothing checks `result.left.movementScore` (or `foodPickups`/`hazardContacts`) for finiteness. `JSON.stringify` writes `NaN`/`Infinity` as `null`. When `null-report.ts` parses `authored.json`, `conditionStats` sums with `sum + value`, so each `null` counts as `0`. I verified this: `JSON.parse(JSON.stringify([1, NaN, Infinity]))` reduces to `1`. The same coercion happens in the bootstrap resample sums and in the `(a, b) => a - b` sort. One diverging episode would therefore shift a graph's mean, CI, and null rank with no error. The only finiteness check is in the test (`tests/unit/null-evaluate.test.ts:182`), and it runs only on the fixture.
- **Fix:** Fail loudly in the worker, where the graph and seed are known. Also add a guard in `null-report.ts` for hand-edited or stale files.

```ts
// null-worker.ts, inside runTask's map
const { movementScore, foodPickups, hazardContacts } = result.left;
if (![movementScore, foodPickups, hazardContacts].every(Number.isFinite)) {
  throw new Error(`null-worker: ${task.graphId} seed ${seed} produced a non-finite score (${movementScore})`);
}

// null-report.ts, before buildArtifact's stats
const assertFiniteScores = (label: string, values: readonly unknown[]): void => {
  const bad = values.findIndex((v) => typeof v !== 'number' || !Number.isFinite(v));
  if (bad !== -1) throw new Error(`null-report: ${label} movementScore[${bad}] is not a finite number`);
};
```

### I2. A worker killed by a signal is treated as a clean exit, which misattributes the failure

- **File:** `scripts/null/null-evaluate.ts:296-304` (`runShardedEvaluation`, the `child.on('exit')` handler)
- **Problem:** `if (code !== 0 && code !== null) reject else resolve`. A child killed by a signal has `code === null`, so the parent resolves. That covers the Linux OOM killer (`SIGKILL`), a stray `kill`, or a native crash that exits via `SIGSEGV`/`SIGABRT`. The in-flight task's result is simply lost. The failure only surfaces later, as `assembleRaw`'s `null-evaluate: missing results for "rewired-N"`, after every other task has finished (up to ~8 h in production). That message never mentions a worker death or a signal. The `code !== null` exemption exists only so that the parent's own `child.kill()` (line 292) is not reported as a failure, but it also catches every unexpected signal.
- **Fix:** Track the in-flight task and whether the parent asked for the kill. Report every other exit that leaves a task unfinished.

```ts
let inFlight: NullWorkerTask | undefined;
let killedByParent = false;
// assignNext: inFlight = task; child.send(task);
// on 'result': inFlight = undefined; ...
// on 'error' message: inFlight = undefined; killedByParent = true; child.kill();
child.on('exit', (code, signal) => finish(() => {
  if (inFlight || (code !== 0 && !killedByParent)) {
    rejectWorker(new Error(
      `null-evaluate: worker exited (code ${code}, signal ${signal})` +
      (inFlight ? ` while running ${inFlight.graphId}` : '')
    ));
  } else resolveWorker();
}));
```

### I3. A failing run is fail-slow and all-or-nothing: hours of results are thrown away

- **File:** `scripts/null/null-evaluate.ts:286-315`, `:407-426`
- **Problem:** When a task posts `{type:'error'}`, only that one worker is killed. The shared `nextTaskIndex` lets the surviving workers keep draining the whole queue. When `Promise.all` rejects (a non-zero exit, `:299`), the other forked children are not killed. `main` then calls `process.exit(1)`, which orphans them until each finishes its current synchronous `runTask` (100 episodes). On any failure, `runNullEvaluate` throws before `writeFileSync`, so none of the completed results are kept. For a 50,200-episode run this means one bad graph late in the queue costs the entire run, and there is no resume path.
- **Fix:** Choose one of these explicitly and document it:
  - (a) Fail fast. On the first error, set `nextTaskIndex = tasks.length`, kill every child, and reject. Keep a `children` array so that a `finally` block can `child.kill()` on every exit path.
  - (b) Fail late but keep the work. On error, write the partial results map to `${out}.partial.json` before throwing.
  Either way, add a teardown for every child:

```ts
const children: ChildProcess[] = [];
// in runWorker: children.push(child);
try {
  await Promise.all(Array.from({ length: workerCount }, runWorker));
} finally {
  for (const c of children) if (c.exitCode === null && c.signalCode === null) c.kill();
}
```

### I4. `readRewireIndex` accepts duplicate seeds and duplicate artifacts, which skews the null set

- **File:** `scripts/null/null-evaluate.ts:74-98` (`readRewireIndex`), with effects at `:229-239` (`buildTasks`) and `:378-386` (`assembleRaw`)
- **Problem:** Seeds are not checked for uniqueness or integrality. If `index.json` has the same `seed` twice, for example because two batches were concatenated by hand:
  - `buildTasks` creates two `rewired-N` tasks and wastes shard time on the duplicate.
  - `results.set` makes the second result overwrite the first.
  - `assembleRaw` emits the same entry twice.
  - `null-report.ts` then counts that graph twice in `nullValues`, which biases `nullSummary`/`rankStatistics`, and the report's `|N|` becomes wrong.
  A fractional or negative seed (`1.5`, `-1`) yields graph ids such as `rewired-1.5` and is not caught either. `gzipBytes` is also missing from the type check, so a missing value produces the confusing "`expects undefined`" error at `:120`.
- **Fix:**

```ts
const seen = new Set<number>();
for (const entry of parsed.seeds) {
  if (!entry || typeof entry !== 'object' || !Number.isInteger(entry.seed) || entry.seed < 0 ||
      typeof entry.gzipBytes !== 'number' /* ...existing checks... */) {
    throw new Error(`null-evaluate: ${path} has a malformed seed entry: ${JSON.stringify(entry)}`);
  }
  if (seen.has(entry.seed)) throw new Error(`null-evaluate: ${path} lists rewiring seed ${entry.seed} more than once`);
  seen.add(entry.seed);
}
```

### I5. The published `shards` field is unchecked operator input that defaults to 18

- **Files:** `scripts/null/null-report.ts:46-48, 64-72, 110-113, 258` (`NullReportArgs.shards`, `DEFAULT_SHARDS`, `buildArtifact`), `scripts/null/null-evaluate.ts:336-358` (`NullEvaluationRaw` doc comment)
- **Problem:** The doc comment says `--shards` is left out of `authored.json` because recording it would break the `--shards 1` vs `--shards 3` byte-identity test. That is true, but the workaround means `null-report.ts` publishes `shards: args.shards`, and that value defaults to 18 no matter how the run was actually done. Suppose an operator runs `null:evaluate --shards 8` and then a bare `null:report`. The public artifact and the markdown's "Evaluation shards" row would both say 18, and nothing would flag it. The comment presents this as airtight, but it swaps a test constraint for a provenance field that can silently be wrong. `host` is recorded in the raw file even though it varies across machines, so "operational parameters don't belong in the raw file" is not applied consistently either.
- **Fix:** Pick one of these:
  - (a) Remove `shards` from the published artifact. The claim that results do not depend on shard count is exactly what the test proves, so the field adds nothing scientific.
  - (b) Write the run's operational metadata to a sidecar file (`authored.run.json`: shards, elapsedMs) that the determinism test does not compare, and have `null-report.ts` read it.
  - (c) Make `--shards` required in `null-report.ts`, with no default.
  In every case, update the `NullEvaluationRaw`/`NullReportArgs.shards` doc comments to match.

### I6. `null-report.ts` writes over the shipped artifacts by default, with no guard (drift from `evaluate.ts`)

- **File:** `scripts/null/null-report.ts:37-41` (the `DEFAULT_OUT`/`DEFAULT_REPORT_MD`/`DEFAULT_MANIFEST` constants), `:386-402` (`runNullReport`)
- **Problem:** A bare `npm run null:report` against any `authored.json`, including a trace-graph fixture or a partial dev run, overwrites `public/data/rewiring-null-v1.json`, `docs/rewiring-null-report.md`, and the rewritten `public/data/malecns-arena-v1.manifest.json`. `scripts/training/evaluate.ts:268-320` sets the precedent here: it refuses to write to the default `public/data` in trace-graph mode, because it has already clobbered shipped files in the past. `null-report.ts` has nothing similar. The three writes are also not atomic: artifact, then manifest, then markdown. If the markdown write fails, the manifest already points at the new sha.
- **Fix:** Refuse to write the default `public/data` targets unless the raw file looks like a real run. For example, require `raw.rewired.length >= 500` and `raw.sourceGraphSha256 === <manifest graph sha>`, or require an explicit `--publish` flag. Also render the markdown before writing anything:

```ts
const reportMdContents = renderReportMarkdown(artifact); // render first, then write all three
```

### I7. `null-report.ts` has no tests, including the plan-mandated rerun byte-identity

- **Files:** `tests/unit/` (the only tests are `null-evaluate.test.ts` and `null-stats.test.ts`), `scripts/null/null-report.ts`
- **Problem:** The plan requires that running the report step twice produces byte-identical output, and nothing tests that. None of these are exercised by any test: `buildArtifact`, `parseNullReportArgs`, `updateManifestWithRewiringNull`/`sortKeysDeep`, `renderReportMarkdown`, the seed-0 error path, and the missing-biological error path. The `rewiringNull`-only manifest diff in this branch was checked by hand, not by a test.
- **Fix:** Add `tests/unit/null-report.test.ts` that:
  - builds a small synthetic `NullEvaluationRaw` in memory;
  - runs `runNullReport` twice into a tmpdir (with `--manifest` pointing at a copied manifest) and asserts the artifact, markdown, and manifest are byte-identical;
  - asserts that the manifest diff touches only `rewiringNull`;
  - asserts the missing-seed-0 and missing-`biological` errors are thrown;
  - asserts `parseNullReportArgs` rejects unknown flags.

### I8. `--trained` is a dead CLI flag

- **File:** `scripts/null/null-report.ts:38` (`DEFAULT_TRAINED`), `:57`, `:76`, `:91-93`, `:120`
- **Problem:** `--trained` is parsed, resolved, and returned, but nothing ever reads it. An operator who passes `--trained training/runs/null/trained.json` would reasonably expect the trained-decoder null to be included. It is silently ignored.
- **Fix:** Delete the flag and constant until the WP that consumes it lands. If the flag is kept for a future step, make it fail loudly (`throw new Error('--trained is not implemented yet (WP3)')`) and document that.

### I9. The report hard-codes contiguous rewiring seeds `0..n-1` and "seed 0 = shipped control arm"

- **File:** `scripts/null/null-report.ts:321-323` (Method section), `:333` (Parameters table row "Rewired graphs")
- **Problem:** `renderReportMarkdown` prints `rewiring seeds 0..${artifact.rewired.length - 1}`. Neither `null-evaluate.ts` nor `null-report.ts` requires that the seeds are contiguous from 0. A batch of seeds `{0, 500..998}` (for example, an extension batch) would render as `0..499`, which is false. This is also a published doc.
- **Fix:** Build the description from the data:

```ts
const rewiredSeeds = artifact.rewired.map((e) => e.seed);
const contiguous = rewiredSeeds.every((s, i) => s === rewiredSeeds[0] + i);
const seedRange = contiguous
  ? `${rewiredSeeds[0]}..${rewiredSeeds[rewiredSeeds.length - 1]}`
  : `${rewiredSeeds.length} non-contiguous seeds (${rewiredSeeds[0]}..${rewiredSeeds[rewiredSeeds.length - 1]})`;
```
