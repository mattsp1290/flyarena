# Suggestions

### S1. Add a per-task wall-clock timeout for hung workers
`scripts/null/null-evaluate.ts:265-307`. The simulation is deterministic and bounded by `ticks`, so a real hang is unlikely. Still, a stuck worker (a deadlock in a future native dependency, a swapped-out host) would stall `Promise.all` forever and never report anything. A generous watchdog, reset on each `assignNext`, is cheap:

```ts
let watchdog: NodeJS.Timeout | undefined;
const arm = (task: NullWorkerTask) => {
  clearTimeout(watchdog);
  watchdog = setTimeout(() => { errors.push(`${task.graphId}: timed out`); child.kill('SIGKILL'); }, TASK_TIMEOUT_MS);
};
```

Clear it in the `exit` handler. Give it an operator-facing flag, or make it a multiple of `ticks * heldOutCount`.

### S2. Check that the result's `graphId` matches the task that was sent
`scripts/null/null-evaluate.ts:287-289`. `results.set(message.graphId, ...)` trusts whatever id the child sends back. If `inFlight` is tracked (see I2), assert `message.graphId === inFlight.graphId` and `message.results.length === inFlight.heldOutSeeds.length`. That makes the protocol self-checking at almost no cost.

### S3. Make the paired comparison's seed alignment explicit
`scripts/null/null-report.ts:237-246` (`buildArtifact`). `pairedStats` pairs by array index and only checks that the lengths match (`scripts/training/stats.ts:84`). Today the alignment holds by construction, but a hand-merged `authored.json` could break it without any error. Add:

```ts
if (raw.biological.heldOutSeeds.join() !== seed0.heldOutSeeds.join()) {
  throw new Error('null-report: biological and rewired seed 0 were scored on different held-out seeds');
}
```

Also validate `raw.version === 1` at `:387` before trusting the shape. It is currently a bare `as NullEvaluationRaw` cast.

### S4. Make the missing-seed-0 error actionable
`scripts/null/null-report.ts:238-240`. Include what to do and what was found:
`rewired seed 0 is missing from ${args.authored} (found seeds ${min}..${max}); the paired comparison needs the shipped control arm, so rerun rewire_batch.py/null-evaluate with a seed range that includes 0`.

### S5. Resolve path flags consistently in `null-evaluate.ts`
`scripts/null/null-evaluate.ts:177-184` vs `:198`. `--out` is resolved against `process.cwd()`, but `--graph`, `--rewired-index`, and `--graphs-dir` are stored raw. They still work, because they are resolved lazily and `fork` inherits the cwd. `null-report.ts` resolves every path flag, though, and mixing the two styles in one file invites bugs if a path ever crosses into the worker before it is resolved. Resolve all four at parse time.

Also, `--graph` without `--biological` is silently ignored (`:411`). Either reject that combination or say so in a usage comment.

### S6. Don't pass `--inspect` to every forked child
`scripts/null/null-evaluate.ts:267`: `fork(workerPath, [], { execArgv: process.execArgv })`. If the parent runs under `--inspect`/`--inspect-brk`, all 18 children inherit the same inspector port and fail with `EADDRINUSE`. Filter out the inspector flags:

```ts
const childExecArgv = process.execArgv.filter((a) => !a.startsWith('--inspect'));
```

### S7. Trailing blank line and degenerate-histogram labels in the markdown
`scripts/null/null-report.ts:378-379`. The degenerate-bullet interpolation is followed by a literal newline, so a non-degenerate report ends with `\n\n` (confirmed: `docs/rewiring-null-report.md` ends with `.\n\n`). Markdown linters (MD012) flag this. Put the conditional on the same line as the last bullet, or `.trimEnd() + '\n'` the result.

`renderHistogramTable` (`:286-296`) uses `fmt(lo, 3)`. When the null is degenerate, the span is often smaller than 1e-3, so all 30 rows render as the same `x.xxx to x.xxx` range. Use `toPrecision(6)` or switch to exponent notation when `span < 1e-2`.

### S8. The manifest re-serializer assumes the Python output never contains floats or non-ASCII
`scripts/null/null-report.ts:177-203` (`sortKeysDeep`/`updateManifestWithRewiringNull`). `sortKeysDeep` handles every value `JSON.parse` can produce (object, array, string, number, boolean, `null`), so nothing is mangled structurally. The byte-level "matching the file's existing (Python-written) format" claim holds only because the manifest currently has no floats and no non-ASCII characters (I grep-verified this). Python's `json.dumps` writes `1.0` and `1e-05` and escapes non-ASCII (`ensure_ascii=True`). JS writes `1`, `0.00001`, and raw UTF-8. If `compile.py` ever adds a float field, running `null:report` would rewrite unrelated bytes. Document this assumption in the doc comment, or narrow the write to a text-level insert and replace of the `rewiringNull` block.

### S9. Test gaps worth filling cheaply
- `tests/unit/null-evaluate.test.ts` covers only the happy path of `parseNullEvaluateArgs` and the missing required flags. Add cases for:
  - an unknown flag (`/Unknown argument/`);
  - a value-less flag followed by another flag (`--shards --out x` → `requires a value`);
  - `--shards 0` and `--held-out-count 0`.
- Test `readRewireIndex` directly (empty seeds, malformed entry, and duplicate seeds once I4 is fixed), and `buildTasks` ordering (unsorted `index.seeds` input → seed-sorted tasks, biological first).
- `graphStats` (`tests/unit/null-stats.test.ts:18-35`) checks mean/median and determinism but not `std` or that the CI contains the mean. A one-line `expect(a.std).toBeCloseTo(Math.sqrt(2), 12)` for `[1..5]` would pin the delegation.
- The determinism test (`null-evaluate.test.ts:158`) uses `timeout: 60_000` per spawn but no vitest `it` timeout. On a slow CI host the default 5 s vitest timeout will kill the test first, unless the vitest config raises it. Pass an explicit timeout to `it(..., 120_000)`.

### S10. `sha256Hex` is duplicated, but this matches repo convention
`scripts/null/null-worker.ts:58`, `null-evaluate.ts:41`, and `null-report.ts:49` each define `sha256Hex`. `scripts/training/evaluate.ts:59` and `scripts/training/export-arms.ts:62` already do the same with a `Uint8Array | string` signature, and `seed-sweep.ts:151` inlines it. This is established practice and not a blocker. The three new copies have *different* signatures (`Uint8Array` vs `string` with explicit `'utf8'`), though. If a `scripts/shared/hash.ts` is ever extracted, these are the obvious candidates. At minimum, use the `Uint8Array | string` signature from `evaluate.ts` for consistency.

### S11. Minor: histogram edge/bin rounding disagreement
`scripts/null/null-stats.ts:137-147`. `edges[i]` is computed as `min + span*i/binCount`, but the bin index comes from `floor((v-min)/span*binCount)`. For a value that lands exactly on an interior edge, floating-point rounding can put it one bin lower than the documented half-open `[edges[i], edges[i+1])` suggests. This is harmless for a descriptive histogram, but the doc comment on `Histogram.edges` promises more precision than the code delivers. Either soften the wording or note it.
