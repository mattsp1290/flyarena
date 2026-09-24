import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gunzipSync } from 'node:zlib';
import { createWorkerRuntime, handleWorkerRequest } from '../../src/lib/worker/neural.worker';
import type { WorkerRequest, WorkerResponse } from '../../src/lib/worker/protocol';

/**
 * jsdom (this project's Vitest environment) implements no `Worker` global at
 * all, so `src/App.svelte`'s `new Worker(new URL(...), { type: 'module' })`
 * cannot run under test. `FakeNeuralWorker` is a `Worker`-shaped stand-in
 * that drives the exact same `handleWorkerRequest` the real dedicated
 * Worker uses (`src/lib/worker/neural.worker.ts`), delivering responses
 * asynchronously via `queueMicrotask` (never synchronously — App-level code
 * must not accidentally depend on same-tick delivery) and cloning every
 * message both directions with `structuredClone`, matching what a real
 * postMessage structured-clone transfer actually does.
 */
export class FakeNeuralWorker {
  private readonly runtime: ReturnType<typeof createWorkerRuntime> = createWorkerRuntime();
  private readonly listeners = new Map<string, Set<(event: MessageEvent<WorkerResponse>) => void>>();
  terminated = false;

  postMessage(message: WorkerRequest, _transfer?: Transferable[]): void {
    if (this.terminated) return;
    const cloned = structuredClone(message);
    queueMicrotask(() => {
      if (this.terminated) return;
      // `structuredClone` on `{ response, transfer }` clones `response` (and
      // whatever `transfer` references, e.g. a streamed `rates.buffer`)
      // exactly the way a real Worker's own outbound `postMessage(response,
      // transfer)` would — the `transfer` list is not itself sent, only
      // consulted by the transfer/clone algorithm, so only `.response` needs
      // dispatching here.
      const { response } = handleWorkerRequest(this.runtime, cloned);
      this.dispatch('message', new MessageEvent('message', { data: structuredClone(response) }));
    });
  }

  addEventListener(type: string, listener: (event: MessageEvent<WorkerResponse>) => void): void {
    let set = this.listeners.get(type);
    if (!set) {
      set = new Set();
      this.listeners.set(type, set);
    }
    set.add(listener);
  }

  removeEventListener(type: string, listener: (event: MessageEvent<WorkerResponse>) => void): void {
    this.listeners.get(type)?.delete(listener);
  }

  terminate(): void {
    this.terminated = true;
    this.listeners.clear();
  }

  private dispatch(type: string, event: MessageEvent<WorkerResponse>): void {
    for (const listener of this.listeners.get(type) ?? []) listener(event);
  }
}

const here = dirname(fileURLToPath(import.meta.url));
const publicDataDir = resolve(here, '../../public/data');

const toArrayBuffer = (buffer: Buffer): ArrayBuffer =>
  buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength) as ArrayBuffer;

/**
 * A `fetch`-compatible stand-in that serves the real, committed
 * `public/data/malecns-arena-v1.*` files from disk instead of over the
 * network — so tests exercise the real sha256-verification path
 * (`src/lib/experiment/assets.ts`) against real artifact bytes rather than
 * a synthetic fixture. `corrupt` optionally flips one byte of a named file
 * to exercise the hash-mismatch -> error path. `transparentGzipDecode`
 * simulates what `vite preview` actually does for a `.gz` file (serves it
 * with `Content-Encoding: gzip`, which the Fetch API decodes before handing
 * bytes to application code) — verified directly against a real build; see
 * `src/lib/experiment/assets.ts#verifyAndDecompressArtifact`'s doc comment.
 */
export const createPublicDataFetch = (options?: { corrupt?: string; transparentGzipDecode?: boolean }) => {
  return async (input: RequestInfo | URL): Promise<Response> => {
    const url = typeof input === 'string' ? input : input.toString();
    const filename = url.split('/').pop() ?? '';
    const filePath = resolve(publicDataDir, filename);
    let bytes: Buffer;
    try {
      bytes = readFileSync(filePath);
    } catch {
      return new Response(null, { status: 404, statusText: 'Not Found' });
    }
    if (options?.transparentGzipDecode && filename.endsWith('.gz')) {
      bytes = gunzipSync(bytes);
    }
    if (options?.corrupt === filename) {
      bytes = Buffer.from(bytes);
      // Flip a byte well past any leading magic number (gzip's 2-byte
      // magic, or this project's own 4-byte "FANG" format magic) so a
      // corruption test exercises the hash/length check it means to, not
      // an incidental change in which check trips first because the flip
      // happened to land on a magic byte used for gzip-vs-decompressed
      // detection (see `verifyAndDecompressArtifact`).
      const offset = Math.min(64, bytes.length - 1);
      bytes[offset] ^= 0xff;
    }
    const contentType = filename.endsWith('.json') ? 'application/json' : 'application/octet-stream';
    return new Response(toArrayBuffer(bytes), { status: 200, headers: { 'content-type': contentType } });
  };
};
