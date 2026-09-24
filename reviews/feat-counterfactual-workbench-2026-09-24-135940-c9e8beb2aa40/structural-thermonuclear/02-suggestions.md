# Suggestions

## S1 — Use the evidence header as the single model-validation boundary

**Location:** `src/lib/counterfactual/engine.ts:101-130`; related duplicate validation in `counterfactual.worker.ts:15-16`.

`experiment()` repeats request, topology, target and seed validation already owned by `evidenceHeader()`, and the Worker validates its run request twice. This is currently correct, but repeated branches can drift as the evidence contract evolves. Build the header once before the seed loop and consume its validated request/seeds. Keep `runSeed`'s own check if it remains independently callable.

```ts
const header = evidenceHeader(prepared, input);
const results: SeedResult[] = [];
for (const seed of header.seeds) {
  results.push(runSeed(prepared, header.request, seed));
  yield results.length;
}
return { ...header, results, summary: /* existing computation */ };
```

Likewise, narrow the Worker's command once and use the validated request's topology. This suggestion is nonblocking; do not introduce another abstraction merely to avoid a short discriminated-union branch.
