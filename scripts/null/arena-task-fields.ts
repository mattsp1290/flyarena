import { resolveArenaTask } from '../../src/lib/arena/tasks';

/**
 * `--arena-task <id>` output-field convention, shared by every evaluator
 * script that scores against an arena task (`null-evaluate.ts`,
 * `null-trained-evaluate.ts`, `null-trained-evaluate-graph-list.ts`):
 * `arenaTask`/`arenaTaskFingerprint` are omitted entirely from the raw output
 * when no `--arena-task` flag was passed, so a default run's bytes are
 * unchanged (`.agents/plans/task-generality/01-task-plumbing.md`'s "no keys
 * are added for default"). Extracted here (rather than each script
 * reimplementing the same three-line spread) so the omit-when-absent
 * convention can't drift between them, and so this addition costs each
 * caller only one import and one spread instead of its own doc comment and
 * conditional (keeping `null-evaluate.ts` under the 1000-line threshold it
 * was already split once to stay under).
 */
export interface ArenaTaskOutputFields {
  readonly arenaTask?: string;
  readonly arenaTaskFingerprint?: string;
}

/** Validates `--arena-task <id>` at CLI-parse time (throws on an unknown id via `resolveArenaTask`). */
export const parseArenaTaskArg = (id: string): string => {
  resolveArenaTask(id);
  return id;
};

export const arenaTaskOutputFields = (arenaTask: string | undefined): ArenaTaskOutputFields =>
  arenaTask === undefined ? {} : { arenaTask, arenaTaskFingerprint: resolveArenaTask(arenaTask).fingerprint };
