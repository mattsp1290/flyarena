# Critical and important findings

No Critical findings.

## Important — an expired backend job permanently locks the sandbox

**Location:** `src/lib/lab/Lab.svelte:43–47` and `src/lib/lab/api.ts:45–53`.

After the backend restarts, its documented ephemeral job store loses the tracked ID. The next status call returns HTTP 404, but `poll()` treats this definitive missing-job response as a transient network error: it retains `busy=true`, the ID, and the disabled fieldset. Reconnect repeats the same 404; cancellation also returns 404. Because the new shell deliberately retains the Lab component through navigation, leaving and returning does not recover. The operator must reload the entire app, despite the server being ready for a new job. Eviction of a tracked result while polling is disconnected has the same failure mode.

**Reproduction:** render the actual Lab component, return `{id:'lost-on-restart'}` from POST and `404 {detail:'Unknown or expired job'}` from every subsequent request. Both the training button and backend URL remain disabled after status failure and after reconnect. There is no abandon/reset action. A temporary Vitest test confirmed those assertions.

**Suggested fix:** retain HTTP status as a typed API error and distinguish authoritative job loss from transient connection failures. On 404 for the current request identity, stop polling, release the active lock and show an explicit expired-job message; preserve the missing ID as diagnostic context if desired. Permit an explicit fresh submission. Never automatically resubmit after failure, and never treat a transient network failure as proof that compute ended. Apply the same handling to cancel/reconnect and protect it with the request identity used by the lifecycle owner.

```ts
export class LabHttpError extends Error {
  constructor(readonly status: number, message: string) { super(message); }
}
// API: throw new LabHttpError(response.status, detail).
// Current-session status/cancel error handling:
if (error instanceof LabHttpError && error.status === 404) {
  stopPolling();
  busy = false;
  errorMessage = 'This job expired or the backend restarted. Start a new experiment.';
  return;
}
```

Add a component/controller test covering 404 after successful submission, recovery to a deliberately started new job, and a transient failure that still retains ownership without duplicate POSTs.

**Resolution:** Reviewed the subsequent `LabSession`/`LabApiError` remedy and ran its regression coverage on 2026-09-24 at 14:04 UTC. Authoritative 404 releases the lock; transient errors retain it; explicit restart succeeds without automatic resubmission. The Important finding is resolved. See the overview's follow-up record.
