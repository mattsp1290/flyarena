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
  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.invalidate();
    if (sessionActive(this.state.status) && this.api && this.state.id) {
      // The API bounds this fresh request to 15 seconds; aborting polling cannot cancel it.
      void this.api.cancel(this.state.id, new AbortController().signal).catch(() => {});
    }
  }
}

const message = (error: unknown) => (error instanceof Error ? error.message : 'Connection failed.');
