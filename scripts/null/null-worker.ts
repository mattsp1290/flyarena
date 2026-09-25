import type { GraphMode } from '../../src/lib/connectome/format';
import { runEpisode } from '../training/episode';
import { assertFiniteScores, graphFromTaskMode, loadVerifiedGraphBinary, runWorkerMain } from './null-worker-shared';

/**
 * `null-evaluate.ts`'s child process: `node:child_process.fork`s this file
 * once per shard. Each shard is handed one graph at a time over IPC
 * (`process.send`/`process.on('message')`, not `worker_threads`
 * `parentPort` — `fork` gives every child its own process, which is what
 * lets `tsx`'s loader (inherited via `execArgv`) resolve this file's own
 * TypeScript imports without a separate build step), scores it on every
 * held-out seed with the authored decoder (or one of its `NullDecoderKind`
 * sign-flip variants, per the task's own `decoder` field — see
 * `.agents/plans/null-explanation/01-decoder-variants.md`) against a parked
 * opponent, and reports back the raw per-seed results. `null-evaluate.ts` (the parent)
 * owns all statistics and ordering; this file never sorts or aggregates —
 * it only produces one seed-ordered results array per task, so the parent
 * can reassemble a fully deterministic output regardless of which shard
 * finished which task or in what order.
 */

export type NullTaskMode = GraphMode;

/**
 * Single source of truth for every authored decoder family kind this
 * study's evaluator ever drives the left agent with
 * (`scripts/training/episode.ts`'s `EpisodeDecoderKind`, restricted to the
 * `isAuthoredFamily` subset) —
 * `.agents/plans/null-explanation/01-decoder-variants.md` WP1's
 * decoder-convention-check variants. `NullDecoderKind` is derived from this
 * array (`as const` + `(typeof ...)[number]`), not declared independently,
 * so adding or renaming a kind here is a compile error everywhere it isn't
 * also updated (`null-evaluate.ts`'s `--decoder` validator, `null-report.ts`'s
 * `CONDITION_LABELS`) instead of a silent runtime gap — a reviewer finding:
 * an earlier version declared the type and this array separately, so a kind
 * added to one without the other would compile.
 */
export const NULL_DECODER_KINDS = [
  'authored',
  'authored-flip-thrust',
  'authored-flip-yaw',
  'authored-flip-both'
] as const;

export type NullDecoderKind = (typeof NULL_DECODER_KINDS)[number];

export interface NullWorkerTask {
  readonly graphId: string;
  readonly mode: NullTaskMode;
  /** Path to the gzip-compressed graph binary to load for this task. For `disconnected`, this is the *biological* source graph (the disconnected control is derived from it, matching the browser's own `buildGraphBufferForMode`). */
  readonly path: string;
  /** Expected sha256 of the *decompressed* binary read from `path` — verified before any episode is scored. */
  readonly expectedSha256: string;
  readonly heldOutSeeds: readonly number[];
  readonly ticks: number;
  /**
   * Left-agent decoder for this task. Defaults to `'authored'` when absent
   * (older callers, and every existing test's hand-built task literal), so
   * this field is additive rather than a breaking change to the wire
   * protocol — see `runTask`'s own default below.
   */
  readonly decoder?: NullDecoderKind;
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
 * `NullWorkerTask` crosses a `fork`/IPC boundary (`JSON.stringify`/
 * `process.send`, per this file's own header comment) where TypeScript's
 * compile-time typing doesn't protect the runtime value -- an IPC payload
 * is just JSON on the wire. In practice every real task is built by
 * `null-evaluate.ts`'s `buildTasks`, which only ever assigns a
 * CLI-flag-validated `args.decoder`, so this isn't reachable through the
 * shipped driver today -- but a stray/corrupted IPC payload must fail with
 * an actionable error here rather than reaching `runEpisode`'s own generic
 * `unknown decoder` throw with no task/graph context (a reviewer finding,
 * still open from the prior review round's S7).
 */
const validateTaskDecoder = (decoder: unknown): NullDecoderKind | undefined => {
  if (decoder === undefined) return undefined;
  if (typeof decoder !== 'string' || !(NULL_DECODER_KINDS as readonly string[]).includes(decoder)) {
    throw new Error(
      `null-worker: task.decoder is not a recognized NullDecoderKind: ${JSON.stringify(decoder)} ` +
        `(expected one of ${NULL_DECODER_KINDS.join(', ')}, or undefined)`
    );
  }
  return decoder as NullDecoderKind;
};

const runTask = (task: NullWorkerTask): readonly NullSeedResult[] => {
  const graphBinary = loadVerifiedGraphBinary('null-worker', task.path, task.expectedSha256);
  const graph = graphFromTaskMode(task.mode, graphBinary);
  const decoder = validateTaskDecoder(task.decoder) ?? 'authored';

  return task.heldOutSeeds.map((seed) => {
    const result = runEpisode({
      seed,
      ticks: task.ticks,
      left: { decoder, graph },
      right: { decoder: 'parked' }
    });
    const { movementScore, foodPickups, hazardContacts } = result.left;
    assertFiniteScores('null-worker', task.graphId, seed, { movementScore, foodPickups, hazardContacts });
    return { seed, movementScore, foodPickups, hazardContacts };
  });
};

runWorkerMain(runTask);
