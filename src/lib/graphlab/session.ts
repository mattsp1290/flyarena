import { GraphLabApi, GraphLabApiError } from './api';
import type { Job, JobRequest } from './types';

/**
 * One job session: owns one request and one next-poll timer, including
 * while its view is hidden. Modeled directly on `src/lib/lab/session.ts`
 * (`03-frontend-route.md`: "Modeled on `src/lib/lab/session.ts`"), with a
 * slower 1 s poll (the plan's own number for this route, versus the
 * synthetic lab's 600 ms -- these jobs run for minutes, not seconds, so
 * there is no benefit to polling faster) and the same behavior otherwise:
 * a 404 on refresh means the job is no longer retained by the backend
 * (`jobs.py`'s `MAX_RETAINED_JOBS`), surfaced as `unavailable`, and
 * `dispose()` makes a best-effort cancel of a still-active job without
 * waiting for it.
 */
export interface SessionState {
  id: string | null;
  job: Job | null;
  status: Job['status'] | 'idle' | 'submitting' | 'unavailable';
  error: string;
}

export const initialSession = (): SessionState => ({ id: null, job: null, status: 'idle', error: '' });

export const sessionActive = (status: SessionState['status']): boolean =>
  status === 'submitting' || status === 'queued' || status === 'running' || status === 'cancelling';

const POLL_INTERVAL_MS = 1000;

export class GraphLabSession {
  private state = initialSession();
  private api: GraphLabApi | undefined;
  private controller: AbortController | undefined;
  private timer: ReturnType<typeof setTimeout> | undefined;
  private disposed = false;

  constructor(private readonly changed: (state: SessionState) => void) {}

  private publish(state: SessionState) {
    this.state = state;
    this.changed(state);
  }
  private invalidate() {
    clearTimeout(this.timer);
    this.timer = undefined;
    this.controller?.abort();
    this.controller = undefined;
  }
  private begin() {
    this.invalidate();
    const controller = new AbortController();
    this.controller = controller;
    return controller;
  }
  private current(controller: AbortController) {
    return !this.disposed && this.controller === controller;
  }

  async start(endpoint: string, token: string, job: JobRequest) {
    if (this.disposed || sessionActive(this.state.status)) return;
    const controller = this.begin();
    this.publish({ ...initialSession(), status: 'submitting' });
    try {
      const api = new GraphLabApi(endpoint, token);
      this.api = api;
      const created = await api.submit(job, controller.signal);
      if (!this.current(controller)) return;
      this.publish({ ...this.state, id: created.id, status: 'queued' });
      await this.refresh('status');
    } catch (error) {
      if (!this.current(controller)) return;
      this.publish({
        ...this.state,
        status: 'idle',
        error: `${message(error)} Submission is not retried automatically.`,
      });
    }
  }

  private async refresh(operation: 'status' | 'cancel') {
    const { api } = this,
      { id } = this.state;
    if (this.disposed || !api || !id) return;
    const controller = this.begin();
    this.publish({ ...this.state, error: '' });
    try {
      const job = await api[operation](id, controller.signal);
      if (!this.current(controller)) return;
      this.publish({
        id,
        job,
        status: job.status,
        error: job.status === 'failed' || job.status === 'timed-out' ? (job.error ?? 'Job failed.') : '',
      });
      if (sessionActive(job.status)) {
        this.timer = setTimeout(() => {
          void this.refresh('status');
        }, POLL_INTERVAL_MS);
      }
    } catch (error) {
      if (!this.current(controller)) return;
      const missing = error instanceof GraphLabApiError && error.status === 404;
      this.publish({
        ...this.state,
        status: missing ? 'unavailable' : this.state.status,
        error: missing
          ? 'This job is no longer retained by the backend. You can start a new job.'
          : `${message(error)} Reconnect to inspect the existing job.`,
      });
    }
  }

  reconnect() {
    return this.refresh('status');
  }
  cancel() {
    return this.refresh('cancel');
  }
  /**
   * Drops this session's own `GraphLabApi` instance -- and the token
   * inside it -- so nothing reachable through this session can hold the
   * token in memory once the caller is done with it (a fix-verification
   * review finding: `start()` stores `this.api`, and until this method
   * existed, nothing ever cleared it -- not `dispose()`, not any caller --
   * so the last `GraphLabApi` instance, and its token, stayed reachable for
   * as long as the session object itself did, i.e. for the rest of the
   * tab, contradicting `GraphLab.svelte`'s route-leave doc comment).
   *
   * Safe to call while a `cancel()` (or any `refresh()`) is still in
   * flight: `refresh()` reads `const { api } = this` synchronously, before
   * its first `await` -- calling an async method runs that synchronous
   * prefix immediately, so by the time this method's caller gets control
   * back, the in-flight call already holds its own local reference,
   * independent of `this.api`. Setting `this.api = undefined` here can
   * therefore never abort or corrupt a request already under way; it only
   * prevents any *new* `refresh()` call from finding an `api` to use
   * (`refresh()`'s own `!api` guard then makes it a no-op).
   *
   * A no-op if no job was ever started (`this.api` is already `undefined`).
   * Does not touch `this.state`/polling/`this.disposed` -- callers that
   * also want to stop an active job or tear down the session entirely
   * still call `cancel()`/`dispose()` themselves; this method only ever
   * detaches the API client.
   */
  release() {
    this.api = undefined;
  }
  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.invalidate();
    if (sessionActive(this.state.status) && this.api && this.state.id) {
      // The API bounds this fresh request to 15 seconds; aborting polling cannot cancel it.
      void this.api.cancel(this.state.id, new AbortController().signal).catch(() => {});
    }
    this.release();
  }
}

const message = (error: unknown) => (error instanceof Error ? error.message : 'Connection failed.');
