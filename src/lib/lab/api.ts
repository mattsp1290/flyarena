import type { Job, Options } from "./types";

export class LabApi {
  private base: string;
  constructor(
    endpoint: string,
    private token: string,
  ) {
    const url = new URL(endpoint);
    if (
      !["http:", "https:"].includes(url.protocol) ||
      url.username ||
      url.password ||
      url.search ||
      url.hash
    ) {
      throw new Error(
        "Use an HTTP(S) backend URL without credentials, query or fragment.",
      );
    }
    this.base = endpoint.replace(/\/$/, "");
  }
  async request<T>(
    path: string,
    method: string,
    signal: AbortSignal,
    body?: Options,
  ): Promise<T> {
    const response = await fetch(`${this.base}/api/v1${path}`, {
      method,
      signal: AbortSignal.any([signal, AbortSignal.timeout(15000)]),
      headers: {
        Authorization: `Bearer ${this.token}`,
        "Content-Type": "application/json",
      },
      ...(body ? { body: JSON.stringify(body) } : {}),
    });
    let data;
    try {
      data = await response.json();
    } catch {
      throw new Error(
        `Backend returned a non-JSON response (${response.status}).`,
      );
    }
    if (!response.ok) {
      const detail =
        typeof data.detail === "string"
          ? data.detail
          : "Invalid request settings.";
      throw new Error(`${response.status}: ${detail}`);
    }
    return data as T;
  }
  submit(options: Options, signal: AbortSignal) {
    return this.request<{ id: string }>("/jobs", "POST", signal, options);
  }
  status(id: string, signal: AbortSignal) {
    return this.request<Job>(`/jobs/${encodeURIComponent(id)}`, "GET", signal);
  }
  cancel(id: string, signal: AbortSignal) {
    return this.request<Job>(
      `/jobs/${encodeURIComponent(id)}`,
      "DELETE",
      signal,
    );
  }
}
