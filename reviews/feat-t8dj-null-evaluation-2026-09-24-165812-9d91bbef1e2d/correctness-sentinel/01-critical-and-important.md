# Critical and Important Issues

## Critical

None. The published numbers are internally consistent:
- `bioPercentile`, `pLow`, and `pHigh` agree with `kBelow = 0`.
- The sha256 values tie out to the manifest.
- The paired comparison uses the same held-out seed order on both sides.

---

## Important

### I1. A worker killed by a signal is treated as success, and its in-flight task silently disappears

- **Severity:** Important (missing error handling; misattributed failure)
- **File:** `scripts/null/null-evaluate.ts:296-304`

```ts
child.on('exit', (code, signal) => {
  finish(() => {
    if (code !== 0 && code !== null) {
      rejectWorker(...);
    } else {
      resolveWorker();          // code === null && signal === 'SIGKILL' lands here
    }
  });
});
```

When a process is terminated by a signal, `code` is `null`. That includes the kernel OOM killer (SIGKILL), an operator's `kill`, and a SIGSEGV in native code. All of these fall into the `resolveWorker()` branch. The in-flight task never gets a result or an error entry. With `--shards 1`, every task still in the queue is dropped as well.

The branch exists so that the parent's own `child.kill()` on the error path (line 292) resolves quietly. But it cannot tell that case apart from an external kill.

**Reproduced:** a stub worker that SIGKILLs itself on task `sigkill` of `['a','sigkill','c']` with 1 shard makes `runShardedEvaluation` **resolve** with `keys=['a']`.

In the real CLI, `assembleRaw`'s `require()` (line 372-376) then throws `missing results for "rewired-N"`. So the run does not publish bad data. But:
1. The error names the wrong cause (it says nothing about a dead worker or OOM).
2. It only fires after every other shard has finished the whole queue, which could be hours into an 8-hour budget.
3. `runShardedEvaluation` is exported, and on its own it breaks its contract that a resolved Map means every task completed.

**Suggested fix:** record whether this process asked for the kill, reject every other signal exit, and check completeness before returning.

```ts
let killedByParent = false;
// in the 'error' message branch:
errors.push(`${message.graphId}: ${message.message}`);
killedByParent = true;
child.kill();

child.on('exit', (code, signal) => {
  finish(() => {
    if (code === 0 || (code === null && killedByParent)) resolveWorker();
    else rejectWorker(new Error(`null-evaluate: worker exited unexpectedly (code ${code}, signal ${signal})`));
  });
});

// after Promise.all, before returning:
const missing = tasks.filter((t) => !results.has(t.graphId)).map((t) => t.graphId);
if (missing.length > 0) throw new Error(`null-evaluate: no result for ${missing.join(', ')}`);
```

---

### I2. The run is slow to fail: one task error lets the rest of the queue run, and the pool shrinks by one shard

- **Severity:** Important (missing error handling / wasted compute on a multi-hour run)
- **File:** `scripts/null/null-evaluate.ts:286-294` and `:309-315`

When a worker reports `type: 'error'`, the parent kills **that** child only. Every other child keeps calling `assignNext()` until the queue is empty. Only after `Promise.all` settles does the function throw on `errors.length > 0`.

On the real run, a single corrupt or unparseable graph found early would cost the whole run's wall time before failing. The pool also has one fewer worker for the rest of that time.

The reverse case has a related leak. When a worker **rejects** (non-zero exit), `Promise.all` rejects at once, but the sibling children are never killed and keep pulling tasks. The CLI hides this because `main()` calls `process.exit(1)`. Any in-process caller of the exported function, such as a future test, would leak them.

**Reproduced:** with 2 shards and 7 tasks (task 1 errors at once, the other 6 take 300 ms each), the rejection arrived after **2026 ms**. The surviving worker ran all six remaining tasks alone.

**Suggested fix:** use a shared abort flag and keep a set of live children.

```ts
const children = new Set<ChildProcess>();
let aborted = false;
const abortAll = (): void => {
  aborted = true;
  for (const c of children) c.kill();
};

const assignNext = (): void => {
  if (aborted || nextTaskIndex >= tasks.length) {
    child.disconnect();
    return;
  }
  ...
};

// in the error branch:
errors.push(...);
abortAll();

// in rejectWorker paths:
abortAll();
```

Pair this with I1's `killedByParent` flag, set inside `abortAll`, so the killed siblings exit cleanly instead of rejecting.

---

### I3. The published histogram's `counts` include the biological and disconnected scores, but it is labeled and consumed as the 500-graph null

- **Severity:** Important (logic error in published data; mismatch with the downstream consumer)
- **File:** `scripts/null/null-report.ts:248`; label at `:344`

```ts
const bins = buildHistogram([...nullValues, biologicalStats.mean, disconnectedStats.mean], args.histogramBins);
```

The plan asks for "30 equal-width histogram bins over `[min(N ∪ {bio, disc}), max(N ∪ {bio, disc})]`". That sets the **range** of the edges from the union. This code also takes the **counts** from the union.

In the committed artifact, `sum(bins.counts) === 502`:
- `counts[0] === 1` is the disconnected control alone (score -1.86). The lowest rewired score is 0.254, which is in bin 7.
- The biological score (-0.22) falls in bin 6.

The markdown heading (`null-report.ts:344`) says "Null distribution of graph scores (500 rewired graphs)". The WP4 plan (`04-ledger-histogram.md`) draws "bars from `bins`, plus vertical markers labeled Biological ... Disconnected". So the UI will show the two reference graphs **as null bars** and **as markers**. For a result whose headline is "biological falls below every rewiring", a bar at the biological score looks like a rewiring scored there. None did.

**Suggested fix:** take the edges from the union and the counts from `N` only.

```ts
// null-stats.ts
export const buildHistogram = (
  values: readonly number[],
  binCount = DEFAULT_HISTOGRAM_BINS,
  range: readonly [number, number] = [Math.min(...values), Math.max(...values)]
): Histogram => { const [min, max] = range; ... };

// null-report.ts
const all = [...nullValues, biologicalStats.mean, disconnectedStats.mean];
const bins = buildHistogram(nullValues, args.histogramBins, [Math.min(...all), Math.max(...all)]);
```

After the change, add a `null-stats` test asserting `sum(counts) === N.length`, then regenerate `rewiring-null-v1.json` and the manifest sha.

If the owner intends union counts, change the report heading and the WP4 contract to say so explicitly.

---

### I4. The published `shards` value is an unchecked operator-typed default, and the plan's required timing is recorded nowhere

- **Severity:** Important (lost or unverifiable provenance; plan requirement unmet)
- **Files:**
  - `scripts/null/null-report.ts:47` (`DEFAULT_SHARDS = 18`) and `:112` (`--shards` flag)
  - `scripts/null/null-evaluate.ts:336-346` (doc comment) and `:417-419`, `:432-438`

Keeping `shards` out of `authored.json` is the right call for the byte-identity guarantee. But the replacement is not a record of what actually ran. `null-report.ts` publishes whatever `--shards` says, or `18` if the flag is missing. If someone runs `null:evaluate --shards 12` and then `null:report` without the flag, the published artifact says `shards: 18` and nothing flags the mismatch.

The plan (`02-authored-null-evaluation.md` line 76) also says: "Record the measured per-episode time and total wall time in the report." `elapsedMs` and the per-episode time are printed to stdout only (`null-evaluate.ts:432-438`). They appear in the commit message (1610.3 s) but not in `authored.json`, `rewiring-null-v1.json`, or `docs/rewiring-null-report.md`.

**Suggested fix:** have `null-evaluate.ts` write a sidecar, for example `authored.run.json`, holding the fields that legitimately vary between runs. `authored.json` stays shard-independent, and `null-report.ts` reads the sidecar instead of a CLI flag.

```ts
// null-evaluate.ts, runNullEvaluate
writeFileSync(args.out, JSON.stringify(raw));
writeFileSync(
  args.out.replace(/\.json$/, '.run.json'),
  JSON.stringify({ shards: args.shards, elapsedMs, perEpisodeMs: elapsedMs / (tasks.length * args.heldOutCount) })
);

// null-report.ts: read the sidecar (fail if absent unless --shards given explicitly),
// publish shards from it, and render wall/per-episode time in the Parameters table.
```

The shard-determinism test keeps comparing only `authored.json`, so it is unaffected.

---

### I5. `readRewireIndex` does not require unique integer seeds; a duplicate seed breaks shard determinism without any error

- **Severity:** Important (missing validation that directly defeats the determinism guarantee)
- **File:** `scripts/null/null-evaluate.ts:85-96` (validation), `:229-239` (`buildTasks`), `:287-288` (`results.set`), `:378-386` (`assembleRaw`)

If `index.json` lists the same `seed` twice, `buildTasks` creates two tasks with the same `graphId` (`rewired-${seed}`), possibly with different `artifact`s. `results.set(message.graphId, ...)` keeps whichever finished **last**, and that depends on shard count and timing. `assembleRaw` then writes **both** entries with that one result.

The output is wrong (duplicated rows, and `N` counted twice) and depends on shard count. `rewire_batch.py` iterates over a `range`, so today's generator cannot produce this. But `readRewireIndex` exists to validate an external file, and a hand-merged or concatenated index would pass it. The same applies to non-integer or negative seeds: `typeof entry.seed !== 'number'` accepts `1.5` and `NaN`.

**Suggested fix:**

```ts
const seen = new Set<number>();
for (const entry of parsed.seeds) {
  if (!Number.isInteger(entry.seed) || entry.seed < 0 || typeof entry.gzipBytes !== 'number' /* ...existing checks */) throw ...;
  if (seen.has(entry.seed)) throw new Error(`null-evaluate: ${path} lists seed ${entry.seed} more than once`);
  seen.add(entry.seed);
}
```

As defense in depth, `results.set` could also throw if `results.has(message.graphId)` is already true.

---

### I6. `null-report.ts` publishes into the manifest without checking that the raw input matches the graph and seeds the manifest describes

- **Severity:** Important (missing validation at a publish boundary)
- **File:** `scripts/null/null-report.ts:237-246` (paired), `:386-395` (`runNullReport`), `:196-203` (manifest update)

There are three gaps:

1. **Paired seed alignment is assumed, not checked.** `pairedStats(raw.biological.movementScore, seed0.movementScore, ...)` pairs values by array index. `pairedStats` (`scripts/training/stats.ts:84`) checks only that the lengths match. The raw file carries `heldOutSeeds` for every graph, but nothing compares `raw.biological.heldOutSeeds` with `seed0.heldOutSeeds`. `null-evaluate.ts` produces them from one shared array today, so this is correct now. But `authored.json` is a file on disk that could be merged or edited between the two steps.
2. **The source graph is not checked against the manifest.** `raw.sourceGraphSha256` is never compared with the `binarySha256` of the manifest being updated. A stale `authored.json` from an earlier compiled graph would get a `rewiringNull` entry attached to the current graph's manifest with no warning. The values match today (both are `f1a0f982...`), but nothing enforces it.
3. **`raw.version` is never checked** (line 387 is a bare cast).

**Suggested fix:** add these checks at the top of `buildArtifact`/`runNullReport`.

```ts
if (raw.version !== 1) throw new Error(`null-report: unsupported authored.json version ${raw.version}`);
const sameSeeds = (a: readonly number[], b: readonly number[]) => a.length === b.length && a.every((s, i) => s === b[i]);
for (const g of [raw.disconnected, ...raw.rewired]) {
  if (!sameSeeds(g.heldOutSeeds, raw.biological.heldOutSeeds)) throw new Error('null-report: held-out seeds differ between graphs');
}
const manifest = JSON.parse(readFileSync(args.manifest, 'utf8'));
if (manifest.binarySha256 !== raw.sourceGraphSha256) {
  throw new Error(`null-report: authored.json was scored on ${raw.sourceGraphSha256}, manifest describes ${manifest.binarySha256}`);
}
```

Optional: also assert `seed0.gzipSha256 === manifest.rewiredArms?.seed0?.gzipSha256`. The report states that seed 0 is "the shipped control arm", and that is true today (`d143d6...`), but nothing checks it.
