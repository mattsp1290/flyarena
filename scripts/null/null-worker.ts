import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';

import { buildGraphBufferForMode } from '../../src/lib/experiment/bindings';
import { parseGraphBinary, type GraphMode } from '../../src/lib/connectome/format';
import { runEpisode } from '../training/episode';
import { sha256Hex } from '../training/fsio';

/**
 * `null-evaluate.ts`'s child process: `node:child_process.fork`s this file
 * once per shard. Each shard is handed one graph at a time over IPC
 * (`process.send`/`process.on('message')`, not `worker_threads`
 * `parentPort` — `fork` gives every child its own process, which is what
 * lets `tsx`'s loader (inherited via `execArgv`) resolve this file's own
 * TypeScript imports without a separate build step), scores it on every
 * held-out seed with the authored decoder against a parked opponent, and
 * reports back the raw per-seed results. `null-evaluate.ts` (the parent)
 * owns all statistics and ordering; this file never sorts or aggregates —
 * it only produces one seed-ordered results array per task, so the parent
 * can reassemble a fully deterministic output regardless of which shard
 * finished which task or in what order.
 */

export type NullTaskMode = GraphMode;

export interface NullWorkerTask {
  readonly graphId: string;
  readonly mode: NullTaskMode;
  /** Path to the gzip-compressed graph binary to load for this task. For `disconnected`, this is the *biological* source graph (the disconnected control is derived from it, matching the browser's own `buildGraphBufferForMode`). */
  readonly path: string;
  /** Expected sha256 of the *decompressed* binary read from `path` — verified before any episode is scored. */
  readonly expectedSha256: string;
  readonly heldOutSeeds: readonly number[];
  readonly ticks: number;
}

export interface NullSeedResult {
  readonly seed: number;
  readonly movementScore: number;
  readonly foodPickups: number;
  readonly hazardContacts: number;
}

export interface NullWorkerResultMessage {
  readonly type: 'result';
  readonly graphId: string;
  readonly results: readonly NullSeedResult[];
}

export interface NullWorkerErrorMessage {
  readonly type: 'error';
  readonly graphId: string;
  readonly message: string;
}

export type NullWorkerMessage = NullWorkerResultMessage | NullWorkerErrorMessage;

/**
 * Read, decompress, and sha256-verify a graph binary against `expectedSha256`
 * before it is ever fed into an episode — "it verifies each rewired file's
 * sha256 against index.json before scoring" (the plan's own wording). This
 * is a second, independent check: `null-evaluate.ts` already verifies every
 * rewired file's gzip sha256 against `index.json` up front, before forking
 * any shard, so a corrupted batch fails fast without spending shard time.
 * This check instead guards the exact bytes actually handed to
 * `parseGraphBinary` in *this* process.
 */
const loadVerifiedGraphBinary = (path: string, expectedSha256: string): ArrayBuffer => {
  const gzipBytes = readFileSync(path);
  const binary = gunzipSync(gzipBytes);
  const actualSha256 = sha256Hex(binary);
  if (actualSha256 !== expectedSha256) {
    throw new Error(
      `null-worker: ${path} decompressed sha256 ${actualSha256} does not match expected ${expectedSha256}`
    );
  }
  return binary.buffer.slice(binary.byteOffset, binary.byteOffset + binary.byteLength);
};

const EMPTY_BUFFER = new ArrayBuffer(0);

/**
 * Build the `ConnectomeGraph` for one task, going through
 * `buildGraphBufferForMode` (`src/lib/experiment/bindings.ts`) exactly as
 * the browser does, per the plan's "reuse the same buffer builder ... so it
 * matches the browser exactly". For `biological`/`rewired`, `graphBinary`
 * is the loaded artifact bytes and passes straight through (a `.slice(0)`
 * inside `buildGraphBufferForMode`); for `disconnected`, `graphBinary` is
 * the *biological* source and `buildGraphBufferForMode` derives the
 * disconnected control from it at runtime (`createDisconnectedGraph`,
 * re-encoded to the wire format), then this re-parses that encoded buffer
 * — the same round trip the browser's Worker `init` path takes.
 */
const graphFromTask = (task: NullWorkerTask, graphBinary: ArrayBuffer) => {
  const baseBuffer = task.mode === 'biological' || task.mode === 'disconnected' ? graphBinary : EMPTY_BUFFER;
  const rewiredBuffer = task.mode === 'rewired' ? graphBinary : EMPTY_BUFFER;
  const modeBuffer = buildGraphBufferForMode(baseBuffer, rewiredBuffer, task.mode);
  return parseGraphBinary(modeBuffer);
};

const runTask = (task: NullWorkerTask): readonly NullSeedResult[] => {
  const graphBinary = loadVerifiedGraphBinary(task.path, task.expectedSha256);
  const graph = graphFromTask(task, graphBinary);

  return task.heldOutSeeds.map((seed) => {
    const result = runEpisode({
      seed,
      ticks: task.ticks,
      left: { decoder: 'authored', graph },
      right: { decoder: 'parked' }
    });
    const { movementScore, foodPickups, hazardContacts } = result.left;
    // A NaN/Infinity score would silently become `null` under
    // `JSON.stringify` and then `0` wherever `authored.json` is later
    // summed (`conditionStats`'s sums, the bootstrap resample sums, even
    // `Array.prototype.sort`'s comparator) — shifting a graph's mean, CI,
    // and null rank with no error anywhere downstream. Fail here instead,
    // where the graph and seed that produced it are still known (a
    // dual-review finding).
    if (![movementScore, foodPickups, hazardContacts].every(Number.isFinite)) {
      throw new Error(
        `null-worker: ${task.graphId} seed ${seed} produced a non-finite score ` +
          `(movementScore=${movementScore}, foodPickups=${foodPickups}, hazardContacts=${hazardContacts})`
      );
    }
    return { seed, movementScore, foodPickups, hazardContacts };
  });
};

process.on('message', (task: NullWorkerTask) => {
  try {
    const results = runTask(task);
    const message: NullWorkerMessage = { type: 'result', graphId: task.graphId, results };
    process.send?.(message);
  } catch (error) {
    const message: NullWorkerMessage = {
      type: 'error',
      graphId: task.graphId,
      message: error instanceof Error ? error.message : String(error)
    };
    process.send?.(message);
  }
});
