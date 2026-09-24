# Suggestions

### S1. Add a deterministic in-process test of `runShardedEvaluation` with a stub worker

- **File:** `tests/unit/null-evaluate.test.ts` (new test); target `scripts/null/null-evaluate.ts:256`

The CLI shard-determinism test (`null-evaluate.test.ts:158-184`) does use the real `fork` path, and 7 tasks across 3 shards is enough for sharding to matter. But it only catches an ordering regression, such as building output from `results` Map order, if the 3-shard run *happens* to finish out of task order. For tiny tasks the finish order depends on how fast each child boots, so the test protects ordering only by luck.

`runShardedEvaluation` already takes `workerPath` as a parameter, so a stub `.mjs` worker is easy to inject. Use it to:
- force reverse-order completion (task *i* sleeps `(n - i) * 20ms`) and assert the Map contains every task;
- assert that a `type: 'error'` message rejects, and after the I2 fix, that it rejects quickly;
- assert that a worker which SIGKILLs itself rejects (covers I1).

```js
// tests/fixtures/null-stub-worker.mjs
process.on('message', (t) => {
  if (t.graphId === 'kill') return process.kill(process.pid, 'SIGKILL');
  if (t.graphId === 'err') return process.send({ type: 'error', graphId: t.graphId, message: 'boom' });
  const end = Date.now() + (t.ms ?? 0);
  while (Date.now() < end) {}
  process.send({ type: 'result', graphId: t.graphId, results: [] });
});
```

### S2. Test `null-report.ts` itself: run it twice, check the manifest diff, check the histogram sum

- **File:** new `tests/unit/null-report.test.ts`; target `scripts/null/null-report.ts:386`

The report-side byte-identity requirement (`null:report` run twice gives the same bytes) is covered only by the manual check described in the commit message. `runNullReport` is exported and accepts every path as an argument. A fixture test could:
- build a small `NullEvaluationRaw` with seeds 0..9;
- run `runNullReport` twice into a temp directory and compare the artifact, markdown, and manifest bytes;
- assert that the only manifest change against a copy of the real manifest is the `rewiringNull` key;
- assert that `sum(bins.counts)` equals the null size (see I3).

### S3. The tamper test only reaches the file-size check

- **File:** `tests/unit/null-evaluate.test.ts:186-217`

Appending a byte (line 191) changes the length, so `verifyRewiredFiles` rejects at the `gzipBytes` comparison (`null-evaluate.ts:119-121`). The sha256 comparison on lines 123-126 is never reached. The regex `/failed verification|gzip sha256/` passes either way.

Two more cases would help:
- flip one byte in place (same length), which reaches the gzip sha256 check;
- give a correct gzip sha with a wrong `binarySha256` in the index, which reaches `null-worker.ts:74-78`, the worker error message, and the parent's error aggregation. None of that is currently tested.

```ts
const bytes = readFileSync(victim);
bytes[bytes.length - 5] ^= 0xff; // same length
writeFileSync(victim, bytes);
```

### S4. Check the shape of the worker's results in the parent

- **File:** `scripts/null/null-evaluate.ts:287-288`

`results.set(message.graphId, message.results)` accepts whatever arrives. A cheap guard keeps a worker bug from producing misaligned columns that `pairedStats` would then pair by index:

```ts
const task = tasksById.get(message.graphId);
if (!task || message.results.length !== task.heldOutSeeds.length ||
    message.results.some((r, i) => r.seed !== task.heldOutSeeds[i])) {
  errors.push(`${message.graphId}: result shape mismatch`);
}
```

### S5. Histogram bin assignment and `edges` are computed separately

- **File:** `scripts/null/null-stats.ts:137` and `:144`

`edges[i] = min + span*i/binCount`, while `bin = floor((v-min)/span*binCount)`. Floating-point rounding can put a value that equals `edges[i]` into bin `i-1`, which contradicts the doc's `[edges[i], edges[i+1])` contract. For example, `(edges[i]-min)/span*binCount` can come out as `i - 1e-15`.

This has no practical effect on 500 continuous means. If exactness matters for the WP4 consumer, assign bins by comparing against `edges`:

```ts
let bin = Math.min(binCount - 1, Math.floor(((value - min) / span) * binCount));
if (bin + 1 < binCount && value >= edges[bin + 1]) bin += 1;
```

### S6. Match the Python writer's manifest format and write atomically

- **File:** `scripts/null/null-report.ts:196-203`

Today the manifest round-trips byte-for-byte apart from the new key; the diff is +4 lines only. The re-serialization still differs from Python's `json.dumps(..., indent=2, sort_keys=True)` in three ways:
- floats: Python writes `1.0` and `1e-07`, JS writes `1` and `1e-7`;
- `ensure_ascii`: Python escapes non-ASCII characters and JS does not;
- the write is not atomic, whereas `positions.py` uses `fsutil.atomic_write_text`.

A future float field or non-ASCII field in the manifest would make `null:report` rewrite unrelated bytes. Either document the constraint or add a guard that compares the re-serialized manifest (minus `rewiringNull`) with the original bytes. Also write to a temporary file and `renameSync` it into place.

### S7. `--graph` without `--biological` is ignored silently

- **File:** `scripts/null/null-evaluate.ts:411`

`args.graph` is used only when `args.biological` is set. Rejecting `--graph` without `--biological` in `parseNullEvaluateArgs` would stop an operator from believing the override took effect.

### S8. The report says "rewiring seeds 0..N-1", which assumes the seeds are contiguous

- **File:** `scripts/null/null-report.ts:321` and the Parameters table row "Rewired graphs ... (seeds 0..N-1)"

Derive the text from `artifact.rewired[0].seed` and `artifact.rewired.at(-1).seed`, or assert contiguity in `buildArtifact`. A partial batch (for example seeds 0..249 plus 300..549) would otherwise be described wrongly.

### S9. The comment "sorted by graph id" is slightly inaccurate

- **File:** `scripts/null/null-evaluate.ts:22-23`

The output is ordered by the canonical task and index order (biological, disconnected, then rewired by numeric seed), not by a lexicographic `graphId` sort. The behavior is correct, and better than a string sort, which would put `rewired-10` before `rewired-2`. The wording could still mislead a maintainer into "fixing" it to a string sort.
