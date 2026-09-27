import type { HealthResponse, Job, JobRequest } from './types';

/**
 * Client for the private real-graph lab, `/api/graph/v1`
 * (`backend/graph_lab/service.py`). Modeled directly on `src/lib/lab/api.ts`
 * (`03-frontend-route.md`: "modeled on `src/lib/lab/api.ts`"): the same
 * endpoint validation, the same `Authorization: Bearer` header, the same
 * 15 s request timeout. The endpoint and token are held only in this
 * instance's own fields -- never written to `localStorage`, `sessionStorage`,
 * a cookie, or the URL -- and are used only for requests to the
 * caller-supplied `endpoint`.
 */
export class GraphLabApiError extends Error {
  constructor(
    readonly status: number,
    detail: string,
  ) {
    super(`${status}: ${detail}`);
  }
}

export class GraphLabApi {
  private base: string;

  constructor(
    endpoint: string,
    private token: string,
  ) {
    const url = new URL(endpoint);
    if (
      !['http:', 'https:'].includes(url.protocol) ||
      url.username ||
      url.password ||
      url.search ||
      url.hash
    ) {
      throw new Error(
        'Use an HTTP(S) backend URL without credentials, query or fragment.',
      );
    }
    // Built from the already-validated `URL` object's own `origin`/`pathname`
    // (never the raw `endpoint` string) -- a security-review suggestion:
    // this removes any reliance on this class's own checks above and
    // `fetch()`'s later parsing staying byte-for-byte in sync for every
    // input, which two independent re-parses of the same raw string can
    // never fully guarantee.
    this.base = `${url.origin}${url.pathname}`.replace(/\/$/, '');
  }

  private async request<T>(
    path: string,
    method: string,
    signal: AbortSignal,
    options: { body?: unknown; auth?: boolean } = {},
  ): Promise<T> {
    const { body, auth = true } = options;
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    // `auth: false` for `health()` only -- the endpoint has no token to send
    // there is nothing to guard, but never attach the header when it isn't
    // needed either.
    if (auth) headers.Authorization = `Bearer ${this.token}`;
    const response = await fetch(`${this.base}/api/graph/v1${path}`, {
      method,
      signal: AbortSignal.any([signal, AbortSignal.timeout(15000)]),
      headers,
      // The backend URL is user-entered, attacker-influenceable input by
      // design (a security-review suggestion): `no-referrer` means this
      // request never carries this site's own origin as `Referer`, purely
      // defensive tightening beyond the browser's `strict-origin-when-
      // cross-origin` default -- no secret is in the URL either way (the
      // token is header-only), so this closes a hardening gap, not an
      // active leak.
      referrerPolicy: 'no-referrer',
      ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
    });
    let data: unknown;
    try {
      data = await response.json();
    } catch {
      throw new Error(`Backend returned a non-JSON response (${response.status}).`);
    }
    if (!response.ok) {
      // FastAPI's own validation errors carry `detail` as a *list* of
      // `{loc, msg, type}` objects, not a string (Pydantic's default 422
      // shape) -- only ever render a string `detail` verbatim; anything
      // else (a list, an object, `undefined`) falls back to a fixed
      // message, so a malformed/unexpected error body can never smuggle
      // structured data into the rendered error text.
      const record = data as { detail?: unknown };
      const detail = typeof record.detail === 'string' ? record.detail : 'Invalid request settings.';
      throw new GraphLabApiError(response.status, detail);
    }
    return data as T;
  }

  /** `GET /health` is unauthenticated (`service.py`'s own doc comment) -- never sends the token. */
  health(signal: AbortSignal): Promise<HealthResponse> {
    return this.request<HealthResponse>('/health', 'GET', signal, { auth: false });
  }

  submit(job: JobRequest, signal: AbortSignal): Promise<{ id: string; status: string }> {
    return this.request<{ id: string; status: string }>('/jobs', 'POST', signal, { body: job });
  }

  status(id: string, signal: AbortSignal): Promise<Job> {
    return this.request<Job>(`/jobs/${encodeURIComponent(id)}`, 'GET', signal);
  }

  cancel(id: string, signal: AbortSignal): Promise<Job> {
    return this.request<Job>(`/jobs/${encodeURIComponent(id)}`, 'DELETE', signal);
  }
}
