import { ARENA_CONFIG, createArenaConfigFingerprint, retainArenaConfig, type ArenaConfig } from './config';

/**
 * `.agents/plans/task-generality/01-task-plumbing.md`'s WP1: named `ArenaConfig`
 * variants ("arena tasks", not to be confused with `NullWorkerTask` — a
 * different, pre-existing "task" meaning, one work item for a `null-worker.ts`
 * shard). Every task is a config variant only (`00-overview.md`'s "no new
 * physics, sensors, or reward code"); `default` is `ARENA_CONFIG` itself,
 * unchanged, so every existing caller that never passes an arena task keeps
 * producing byte-identical output.
 *
 * The four predeclared variants (`00-overview.md`'s table) are each built by
 * spreading `ARENA_CONFIG` and overriding only the listed fields, then
 * validated through `retainArenaConfig` at module load — an invalid variant
 * (e.g. one that fails the disk-packing capacity check) fails immediately
 * when this module is first imported, not later at first use.
 */
export const ARENA_TASK_IDS = ['default', 'hazard-heavy', 'sparse-food', 'no-movement', 'crowded'] as const;

export type ArenaTaskId = (typeof ARENA_TASK_IDS)[number];

const isArenaTaskId = (value: string): value is ArenaTaskId =>
  (ARENA_TASK_IDS as readonly string[]).includes(value);

const buildTaskConfig = (overrides: Partial<ArenaConfig>): Readonly<ArenaConfig> =>
  retainArenaConfig({ ...ARENA_CONFIG, ...overrides });

export const ARENA_TASKS: Readonly<Record<ArenaTaskId, Readonly<ArenaConfig>>> = Object.freeze({
  default: ARENA_CONFIG,
  /** Scoring dominated by avoidance. */
  'hazard-heavy': buildTaskConfig({ hazardCount: 4, hazardPenalty: 6 }),
  /** Long-range search. */
  'sparse-food': buildTaskConfig({ foodCount: 1, halfWidth: 18, halfDepth: 12 }),
  /** Score only from food and hazards; removes the distance term thrust feeds. */
  'no-movement': buildTaskConfig({ movementScorePerUnit: 0 }),
  /** Walls are close and clearance signals dominate. */
  crowded: buildTaskConfig({ halfWidth: 8, halfDepth: 5.5 })
});

export interface ResolvedArenaTask {
  readonly id: ArenaTaskId;
  readonly config: Readonly<ArenaConfig>;
  readonly fingerprint: string;
}

/**
 * Resolves an arena task id (`undefined` means `'default'`, matching every
 * existing caller's current behavior) to its config and fingerprint. Throws
 * on an unrecognized id rather than silently falling back to `default` —
 * every call site (the IPC boundary in `null-worker.ts`/`null-trained-worker.ts`,
 * every CLI's `--arena-task` flag, `episode.ts`'s `runEpisode`) needs a typo'd
 * or stale id to fail loudly, not silently score the wrong task.
 */
export const resolveArenaTask = (id?: string): ResolvedArenaTask => {
  const resolvedId: ArenaTaskId = id === undefined ? 'default' : assertArenaTaskId(id);
  const config = ARENA_TASKS[resolvedId];
  return { id: resolvedId, config, fingerprint: createArenaConfigFingerprint(config) };
};

const assertArenaTaskId = (id: string): ArenaTaskId => {
  if (!isArenaTaskId(id)) {
    throw new Error(`resolveArenaTask: unknown arena task id "${id}" (expected one of ${ARENA_TASK_IDS.join(', ')})`);
  }
  return id;
};
