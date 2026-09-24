# Action Items

## Critical

None.

## Important

- [x] **T1** `src/lib/lab/Lab.svelte:27-87`: replace independent mutable poll/cancel/reconnect state with one per-job session owner that aborts and invalidates prior reads, applies only current responses, owns one timer and invalidates all callbacks on destruction. Add delayed/out-of-order cancellation + reconnect tests proving one polling loop, stable job identity and no post-destruction polling. Re-run Lab/Shell and live sandbox browser journeys.

## Suggestions

- [x] **S1** `src/lib/counterfactual/engine.ts:101-130`, `counterfactual.worker.ts:15-16`: create the evidence header once at generator entry, use its validated request/seeds, and reuse the Worker's already validated request topology so validation branches cannot drift.

T1 correction independently reviewed and focused tests passed; see the overview addendum for content hashes and approval.

Root remediation: documentation status refreshed and S1 validation consolidation implemented in `63862b4`. Final combined validation is recorded in `docs/counterfactual-validation.md`.
