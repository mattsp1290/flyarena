## Action Items

### Critical
- (none)

### Important
- [ ] [scripts/null/null-evaluate.ts:296-304] **Signal exits.** A worker exit with `code === null` (SIGKILL/OOM/SIGSEGV) currently resolves as success, and its in-flight task, plus the rest of the queue when `--shards 1`, silently disappears. Reproduced: the function resolved with `keys=['a']` out of 3 tasks. Fix:
  - track a `killedByParent` flag;
  - resolve only on `code === 0` or on a parent-initiated kill, and reject every other exit;
  - after `Promise.all`, assert that every `task.graphId` is in `results`.
- [ ] [scripts/null/null-evaluate.ts:286-294, 309-315] **Fail fast.** On a worker `type: 'error'`, or on any worker rejection:
  - set a shared `aborted` flag so `assignNext` stops dispatching;
  - kill every live child, tracked in a `Set<ChildProcess>`.

  Today one failed task lets the other shards drain the whole queue before the run fails (reproduced: 2026 ms vs immediate), and sibling children leak on the reject path.
- [ ] [scripts/null/null-report.ts:248, scripts/null/null-stats.ts:129] **Histogram counts.** `bins.counts` include the biological and disconnected means (the committed artifact sums to 502; `counts[0]` is the disconnected control alone), but the report labels it as the 500-graph null and WP4 draws it as the null bars. Fix:
  - add a `range` parameter to `buildHistogram`;
  - take the edges from `N ∪ {bio, disc}` and the counts from `nullValues` only;
  - add a test that `sum(counts) === N.length`;
  - regenerate `public/data/rewiring-null-v1.json` and the manifest `rewiringNull.sha256`.

  If union counts are intended, relabel the report and WP4 contract instead.
- [ ] [scripts/null/null-report.ts:47,112; scripts/null/null-evaluate.ts:417-438] **Shard and timing provenance.** The published `shards` comes from an unchecked CLI flag that defaults to 18, and the plan-required wall time and per-episode time are not persisted anywhere. Fix:
  - have `null-evaluate.ts` write a sidecar `authored.run.json` with `{ shards, elapsedMs, perEpisodeMs }`, keeping `authored.json` shard-independent;
  - have `null-report.ts` read `shards` and timing from the sidecar;
  - render the timing in the report's Parameters table.
- [ ] [scripts/null/null-evaluate.ts:85-96] **Seed validation.** Make `readRewireIndex` reject duplicate seeds, non-integer or negative seeds, and a missing `gzipBytes`. A duplicate seed produces two tasks with the same `graphId`, so the output depends on completion order (which breaks the byte-identity guarantee) and contains duplicated rows. Optionally, make `results.set` throw if the `graphId` is already present.
- [ ] [scripts/null/null-report.ts:237-246, 386-395] **Report input checks.** Before publishing:
  - assert `raw.version === 1`;
  - assert every graph's `heldOutSeeds` equals `raw.biological.heldOutSeeds`, since the paired comparison pairs by array index;
  - assert `raw.sourceGraphSha256 === manifest.binarySha256` of the manifest being updated.

  Optionally, also assert that seed 0's `gzipSha256` equals `manifest.rewiredArms.seed0.gzipSha256`.

### Suggestions
- [ ] [tests/unit/null-evaluate.test.ts] Add an in-process `runShardedEvaluation` test with a stub `.mjs` worker, passed through the existing `workerPath` parameter. It should:
  - force reverse-order completion (the current CLI test only catches ordering regressions by timing luck);
  - assert fast rejection on a `type: 'error'` message;
  - assert rejection on a self-SIGKILL.
- [ ] [tests/unit/ (new null-report.test.ts)] Test `runNullReport` on a small fixture:
  - two runs give byte-identical artifact, markdown, and manifest;
  - the only manifest change is the `rewiringNull` key;
  - the histogram counts sum to `N`.
- [ ] [tests/unit/null-evaluate.test.ts:186-217] The tamper test appends a byte, so it only reaches the size check. Add:
  - a same-length byte flip, which reaches the gzip sha256 check;
  - a wrong `binarySha256` with a correct gzip, which reaches the worker's decompressed check and the parent's error aggregation.
- [ ] [scripts/null/null-evaluate.ts:287-288] Check each worker result in the parent: `results.length === task.heldOutSeeds.length`, and each `results[i].seed === task.heldOutSeeds[i]`.
- [ ] [scripts/null/null-stats.ts:137,144] Assign histogram bins against the computed `edges`, not a separate `floor(...)`, so that a value equal to an edge cannot be binned one bin low by rounding.
- [ ] [scripts/null/null-report.ts:196-203] Manifest writes:
  - write atomically (temporary file plus `renameSync`), as `positions.py` does;
  - document or guard the float, `ensure_ascii`, and formatting differences from Python's `json.dumps(sort_keys=True, indent=2)`, so a future float or non-ASCII manifest field isn't silently reformatted.
- [ ] [scripts/null/null-evaluate.ts:411] Reject `--graph` without `--biological` in `parseNullEvaluateArgs` (it is ignored silently today).
- [ ] [scripts/null/null-report.ts:321 and Parameters table] Derive "seeds 0..N-1" from the actual first and last rewired seed, or assert that the seeds are contiguous.
- [ ] [scripts/null/null-evaluate.ts:22-23] Reword "sorted by graph id" to "ordered by canonical task order (numeric seed)" so no one later "fixes" it into a lexicographic `graphId` sort.
