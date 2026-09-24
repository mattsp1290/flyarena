# Action items

## Critical

None.

## Important

- [x] `src/lib/lab/Lab.svelte:43` and `src/lib/lab/api.ts:45`: expose typed HTTP status and recover explicitly from a current-session 404 missing/expired job by stopping polling and releasing the input lock, without automatic resubmission. Cover status, reconnect and cancellation; test a new deliberate job after expiry and continued ownership on transient errors.

## Suggestions

- [x] `.agents/plans/counterfactual-workbench/00-overview.md:3`: replace the stale “not implemented” status after final review/merge gates with a truthful completion status and validation-record link.

Root remediation: documentation status refreshed and S1 validation consolidation implemented in `63862b4`. Final combined validation is recorded in `docs/counterfactual-validation.md`.
