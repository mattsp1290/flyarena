# Critical and important findings

No Critical findings.

## T1 — Important: one sandbox job needs one owner of polling and response order

**Location:** `src/lib/lab/Lab.svelte:27-73` (`apply`, `poll`, `cancel`, `reconnect`), with teardown at lines 80–87.

The imported component owns `client`, `controller`, `identifier`, `busy`, `pollStopped`, and one `timer` independently. `reconnect()` clears only the most recently stored timer; it neither cancels nor invalidates an in-flight status request. Cancellation can expose the reconnect button while the ordinary poll is still running. Both the old poll and the reconnect then schedule their own next request, overwrite the single timer handle, and continue polling independently. Responses from an older GET can also overwrite a newer cancellation response. The hidden mounted sandbox makes this an ongoing background lifecycle problem, not just a transient display nit.

Concrete reproduction against the exact component functions:

1. Submit a job and hold its first status GET pending.
2. Click Cancel; make DELETE fail with a connection error. The error exposes Reconnect.
3. Click Reconnect while the original GET is pending.
4. Resolve both status GETs with `running`.
5. Observe **two concurrent GETs and two independently scheduled next-poll timers**. Only the last timer is held by `timer` and cleared on teardown.

This violates the reviewed plan's serial-polling and teardown contract. The current happy-path/reconnect tests never overlap cancellation and status requests.

**Suggested fix:** give one submitted job a small explicit session owner. It should own its immutable client/job identity, current operation controller/generation, one timer and the session lifetime. Poll, reconnect and cancellation replace/invalidate the previous read before starting; only the current operation may apply results or schedule another poll. Keep the component responsible for options and rendering. Preserve the existing fresh bounded DELETE on destruction. Do not merely add another component boolean.

Illustrative invariant (adapt to the final owner, not a required class API):

```ts
private beginOperation() {
  clearTimeout(this.timer);
  this.controller?.abort();
  const controller = this.controller = new AbortController();
  const generation = ++this.generation;
  return { signal: controller.signal, current: () =>
    !this.disposed && generation === this.generation };
}

async refresh() {
  const operation = this.beginOperation();
  try {
    const job = await this.api.status(this.id, operation.signal);
    if (!operation.current()) return;
    this.publish(job);
    if (isActive(job.status)) this.timer = setTimeout(() => void this.refresh(), 600);
  } catch (error) {
    if (operation.current()) this.publishConnectionError(error);
  }
}
```

Regression coverage must hold old GETs pending while cancellation/reconnect happen, deliver them out of order even if the mock ignores abort, prove no older state wins, prove at most one next poll is scheduled, and destroy while requests are pending to prove no polling resumes afterward. Retain navigation/job identity and one-submission tests.

**Resolution:** fixed and independently re-reviewed in the session-owner correction; see `00-overview.md` addendum. This original finding is retained as review history, not an outstanding blocker.
