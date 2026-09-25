import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';

import { buildGraphBufferForMode } from '../../src/lib/experiment/bindings';
import { parseGraphBinary, type ConnectomeGraph, type GraphMode } from '../../src/lib/connectome/format';
import { sha256Hex } from '../training/fsio';

/**
 * Shared between `null-worker.ts` and `null-trained-worker.ts`'s
 * `node:child_process.fork`ed worker leaves: the non-finite-score guard and
 * the `process.on('message', ...)`/`process.send` IPC wrapper were
 * duplicated near-verbatim between the two files (a thermo-maintainability
 * review finding — `null-trained-worker.ts`'s own doc comment even said "See
 * null-worker.ts's identical check" instead of importing it). No import
 * from `./null-worker` (runtime or type-only), so there is no runtime
 * circularity even though `null-worker.ts` itself imports the runtime
 * helpers below from here.
 *
 * Also now home to `loadVerifiedGraphBinary`/`graphFromTaskMode`
 * (originally private to `null-worker.ts`): `scripts/null/regime-task.ts`
 * (WP2) needs the exact same gzip-decompress-and-sha256-verify-then-parse
 * logic against its own `RegimeWorkerTask`, and previously carried a
 * hand-duplicated copy (a thermo-maintainability review finding). They live
 * here rather than being merely exported from `null-worker.ts` because
 * `null-worker.ts`'s own module body calls `runWorkerMain(runTask)` at
 * top level (registering a real `process.on('message', ...)` listener as a
 * side effect of being imported at all) -- `regime-task.ts` is deliberately
 * side-effect-free so `tests/unit/regime-check.test.ts` can import it
 * without also registering a listener on the *test process's* own IPC
 * channel (see `regime-task.ts`'s module doc comment). This file has no
 * top-level side effects (only `export const`/`export type` declarations),
 * so importing from here is safe for both `null-worker.ts` and
 * `regime-task.ts` alike.
 */

/** The three score fields every worker's per-seed result carries, checked for finiteness before it is ever reported back to the parent. */
export interface NullWorkerScoreResult {
  readonly movementScore: number;
  readonly foodPickups: number;
  readonly hazardContacts: number;
}

/**
 * Refuse to report a non-finite `movementScore`/`foodPickups`/
 * `hazardContacts` — a NaN/Infinity score would silently become `null`
 * under `JSON.stringify` and then `0` wherever the published artifact's
 * statistics sum it later (`conditionStats`'s sums, the bootstrap resample
 * sums, even `Array.prototype.sort`'s comparator), shifting a graph's mean,
 * CI, and null rank with no error anywhere downstream. Fail here instead,
 * where the graph/run and seed that produced it are still known (a
 * dual-review finding, originally caught in `null-worker.ts` and later
 * replicated by hand into `null-trained-worker.ts`). `source` is the
 * calling worker's own name (`"null-worker"`/`"null-trained-worker"`), so
 * the thrown message still identifies which worker raised it.
 */
export const assertFiniteScores = (
  source: string,
  graphId: string,
  seed: number,
  result: Readonly<NullWorkerScoreResult>
): void => {
  const { movementScore, foodPickups, hazardContacts } = result;
  if (![movementScore, foodPickups, hazardContacts].every(Number.isFinite)) {
    throw new Error(
      `${source}: ${graphId} seed ${seed} produced a non-finite score ` +
        `(movementScore=${movementScore}, foodPickups=${foodPickups}, hazardContacts=${hazardContacts})`
    );
  }
};

/**
 * Read, decompress, and sha256-verify a graph binary against `expectedSha256`
 * before it is ever fed into an episode — "it verifies each rewired file's
 * sha256 against index.json before scoring" (the plan's own wording). A
 * second, independent check on top of `null-evaluate.ts`'s
 * `verifyRewiredFiles`/`regime-check.ts`'s reuse of it, which verify every
 * rewired file's *gzip* sha256 up front, before forking any shard: this one
 * guards the exact *decompressed* bytes actually handed to
 * `parseGraphBinary` in *this* worker process. `source` is the calling
 * worker's own name (`"null-worker"`/`"regime-worker"`), matching
 * `assertFiniteScores`'s existing `source`-prefixed-message convention
 * above, so the thrown message still identifies which worker raised it.
 */
export const loadVerifiedGraphBinary = (source: string, path: string, expectedSha256: string): ArrayBuffer => {
  const gzipBytes = readFileSync(path);
  const binary = gunzipSync(gzipBytes);
  const actualSha256 = sha256Hex(binary);
  if (actualSha256 !== expectedSha256) {
    throw new Error(`${source}: ${path} decompressed sha256 ${actualSha256} does not match expected ${expectedSha256}`);
  }
  return binary.buffer.slice(binary.byteOffset, binary.byteOffset + binary.byteLength);
};

const EMPTY_BUFFER = new ArrayBuffer(0);

/**
 * Build the `ConnectomeGraph` for one task's `mode`, going through
 * `buildGraphBufferForMode` (`src/lib/experiment/bindings.ts`) exactly as
 * the browser does, per the plan's "reuse the same buffer builder ... so it
 * matches the browser exactly". For `biological`/`rewired`, `graphBinary`
 * is the loaded artifact bytes and passes straight through (a `.slice(0)`
 * inside `buildGraphBufferForMode`); for `disconnected`, `graphBinary` is
 * the *biological* source and `buildGraphBufferForMode` derives the
 * disconnected control from it at runtime (`createDisconnectedGraph`,
 * re-encoded to the wire format), then this re-parses that encoded buffer
 * — the same round trip the browser's Worker `init` path takes. Takes the
 * bare `mode: GraphMode` rather than a whole task object: `NullWorkerTask`
 * and `RegimeWorkerTask` are otherwise structurally unrelated (different
 * extra fields), and `mode` is the only property this function ever reads.
 */
export const graphFromTaskMode = (mode: GraphMode, graphBinary: ArrayBuffer): ConnectomeGraph => {
  const baseBuffer = mode === 'biological' || mode === 'disconnected' ? graphBinary : EMPTY_BUFFER;
  const rewiredBuffer = mode === 'rewired' ? graphBinary : EMPTY_BUFFER;
  const modeBuffer = buildGraphBufferForMode(baseBuffer, rewiredBuffer, mode);
  return parseGraphBinary(modeBuffer);
};

/**
 * Wire up this worker process's `process.on('message', ...)` handler: run
 * `runTask` against the incoming task, `process.send` a `result` message on
 * success or an `error` message (never throwing back out of the handler)
 * on failure. Originally hand-duplicated between `null-worker.ts` and
 * `null-trained-worker.ts` (generic over `Task` only, hard-coding
 * `NullSeedResult`/`NullWorkerMessage`), and then hand-duplicated a third
 * time in `regime-worker.ts` because its `RegimeSeedResult`/
 * `RegimeWorkerMessage` didn't fit that hard-coded pair — generalized here
 * over `Result`/`Message` too (a thermo-maintainability review finding,
 * matching `null-evaluate.ts`'s `runShardedEvaluation<Task, Result,
 * Message>`'s own already-generic pattern), with `Message` defaulted to the
 * same `{type: 'result', graphId, results: readonly Result[]} |
 * {type: 'error', graphId, message}` shape every caller's own message type
 * already uses structurally — so `runWorkerMain(runTask)` still infers
 * `Task`/`Result` from `runTask` alone and no call site needs to spell out
 * type arguments (`NullWorkerMessage`/`RegimeWorkerMessage` are each
 * structurally identical to the default, just with a different `Result`).
 */
export const runWorkerMain = <
  Task extends { readonly graphId: string },
  Result,
  Message extends
    | { readonly type: 'result'; readonly graphId: string; readonly results: readonly Result[] }
    | { readonly type: 'error'; readonly graphId: string; readonly message: string } =
    | { readonly type: 'result'; readonly graphId: string; readonly results: readonly Result[] }
    | { readonly type: 'error'; readonly graphId: string; readonly message: string }
>(
  runTask: (task: Task) => readonly Result[]
): void => {
  process.on('message', (task: Task) => {
    try {
      const results = runTask(task);
      // `Message` is a generic type parameter, so TS can't structurally
      // verify this literal against it even though the literal matches the
      // `type: 'result'` arm of `Message`'s own default constraint above --
      // the cast asserts what every real caller's `Message` type parameter
      // is structurally guaranteed to accept.
      const message = { type: 'result', graphId: task.graphId, results } as Message;
      process.send?.(message);
    } catch (error) {
      // Same reasoning as the `as Message` cast above, for the `type:
      // 'error'` arm.
      const message = {
        type: 'error',
        graphId: task.graphId,
        message: error instanceof Error ? error.message : String(error)
      } as Message;
      process.send?.(message);
    }
  });
};
