import { afterEach, describe, expect, it, vi } from "vitest";
import { LabApi } from "../../src/lib/lab/api";

afterEach(() => vi.unstubAllGlobals());
describe("Lab API", () => {
  it("rejects credentials and non-http endpoints", () => {
    expect(() => new LabApi("file:///tmp/a", "")).toThrow();
    expect(() => new LabApi("https://user:secret@example.com", "")).toThrow();
  });
  it("reports non-JSON and backend errors without retrying", async () => {
    const fetch = vi
      .fn()
      .mockResolvedValue(new Response("offline", { status: 502 }));
    vi.stubGlobal("fetch", fetch);
    const api = new LabApi("http://127.0.0.1:8000", "memory-only");
    await expect(
      api.status("abc", new AbortController().signal),
    ).rejects.toThrow("non-JSON");
    expect(fetch).toHaveBeenCalledTimes(1);
    fetch.mockResolvedValue(
      new Response(JSON.stringify({ detail: "busy" }), { status: 409 }),
    );
    await expect(
      api.status("abc", new AbortController().signal),
    ).rejects.toThrow("409: busy");
  });
});
