import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { requireNonNegativeInt, requirePositiveInt, requireValue } from '../training/cli';
import { parseArenaTaskArg } from './arena-task-fields';

/**
 * `intervention-report.ts`'s CLI/run-mode glue -- `InterventionReportArgs`
 * parsing, the published-null (`PublishedNull`) parser, and the
 * selection-robustness public/docs refusal guard -- extracted out of that
 * file (a "do not let a file cross 1000 lines without a very strong reason"
 * review blocker; `scripts/null/null-report-variant.ts` was extracted out
 * of `null-report.ts` for the same reason, and this module follows its
 * exact convention: it deliberately does NOT import anything back from
 * `intervention-report.ts` -- everything here is a pure type/parser/CLI
 * helper, called by `intervention-report.ts`, never the reverse. Re-exported
 * from `intervention-report.ts` so its existing external import surface
 * (tests, `intervention-report-validation.ts`) is unchanged.
 */

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '../..');

export const DEFAULT_AUTHORED = resolve(repoRoot, 'training/runs/interventions/authored.json');
export const DEFAULT_INDEX = resolve(repoRoot, 'training/runs/interventions/index.json');
export const DEFAULT_PUBLISHED_NULL = resolve(repoRoot, 'public/data/rewiring-null-v1.json');
/** Exported so tests can exercise the `diagnosticOnly`-vs-default-`--out` guard directly, without needing a real 302-graph run just to get there (matches `null-report.ts`'s `DEFAULT_OUT` export convention). */
export const DEFAULT_OUT = resolve(repoRoot, 'training/runs/interventions/statistics.json');

/** `'PATH'` as a fixed default seed (arbitrary but stable across runs) — this study's own bootstrap seed, independent of `null-report.ts`'s `DEFAULT_BOOTSTRAP_SEED` ('NULL'), matching that file's "fixed default" convention (`03-evaluation.md`: "`--bootstrap-seed` with a fixed default"). */
const DEFAULT_BOOTSTRAP_SEED = 0x50415448;
const DEFAULT_BOOTSTRAP_RESAMPLES = 10000;

// ---------------------------------------------------------------------------
// Published 500-graph authored null
// ---------------------------------------------------------------------------

export interface PublishedNull {
  readonly biologicalScore: number;
  /** The 500 rewired graphs' mean `movementScore` (`rewiring-null-v1.json` `rewired[].score`) — the null set every graph in this study is ranked against. */
  readonly scores: readonly number[];
  readonly sourceGraphSha256: string;
  readonly seeds: { readonly start: number; readonly count: number };
  readonly ticks: number;
  readonly substeps: number;
  /**
   * `.agents/plans/task-generality/02-authored-runs.md`'s WP2: present only
   * when this null was built (via `null-report.ts --variant-out`) from a
   * `--arena-task`-labelled `null-evaluate.ts` run — absent for the shipped
   * default-task `rewiring-null-v1.json`. `runInterventionReport`'s
   * `--stats-only` mode requires this to be present and to match the
   * requested `--arena-task`'s fingerprint: the biological connectome graph
   * (and so `sourceGraphSha256`) is identical across every arena task, so
   * that check alone cannot catch a per-task run ranked against the *wrong*
   * task's null (a dual-review finding — the graph doesn't change per task,
   * only `ArenaConfig` does, so nothing else in this file's existing
   * consistency checks would notice). One exception:
   * `statsOnlyNullMatchesTask` below also accepts an *absent* `arenaTask`
   * when the requested task is `'default'` (selection-robustness WP2) — a
   * selection's own null is a default-task run and so never carries this
   * field either, the same "omit when absent" convention as the shipped
   * default null.
   */
  readonly arenaTask?: { readonly id: string; readonly fingerprint: string };
}

/** Parses already-read JSON text — see `parseGraphListIndexInfo`'s doc comment (`intervention-report.ts`) for why (the same TOCTOU reasoning applies to every input this module hashes into `InterventionStatistics.inputs`). */
export const parsePublishedNull = (text: string, label: string): PublishedNull => {
  const parsed = JSON.parse(text) as {
    biological?: { score?: unknown };
    rewired?: readonly { score?: unknown }[];
    sourceGraphSha256?: unknown;
    seeds?: { start?: unknown; count?: unknown };
    ticks?: unknown;
    substeps?: unknown;
    arenaTask?: { id?: unknown; fingerprint?: unknown };
  };
  const biologicalScore = parsed.biological?.score;
  if (typeof biologicalScore !== 'number' || !Number.isFinite(biologicalScore)) {
    throw new Error(`intervention-report: ${label} is missing a finite biological.score`);
  }
  if (!Array.isArray(parsed.rewired) || parsed.rewired.length === 0) {
    throw new Error(`intervention-report: ${label} has no rewired entries`);
  }
  const scores = parsed.rewired.map((entry, i) => {
    if (typeof entry.score !== 'number' || !Number.isFinite(entry.score)) {
      throw new Error(`intervention-report: ${label} rewired[${i}].score is not a finite number`);
    }
    return entry.score;
  });
  if (typeof parsed.sourceGraphSha256 !== 'string') {
    throw new Error(`intervention-report: ${label} is missing sourceGraphSha256`);
  }
  if (typeof parsed.seeds?.start !== 'number' || typeof parsed.seeds?.count !== 'number') {
    throw new Error(`intervention-report: ${label} is missing seeds.start/seeds.count`);
  }
  if (typeof parsed.ticks !== 'number') {
    throw new Error(`intervention-report: ${label} is missing ticks`);
  }
  if (typeof parsed.substeps !== 'number') {
    throw new Error(`intervention-report: ${label} is missing substeps`);
  }
  // Tolerant of absence (the shipped default-task null never carries this),
  // but a *present* arenaTask must be well-formed -- a malformed one is
  // exactly the kind of silent-corruption case this field exists to guard
  // against elsewhere, so it must not itself pass through unchecked here.
  let arenaTask: PublishedNull['arenaTask'];
  if (parsed.arenaTask !== undefined) {
    if (typeof parsed.arenaTask.id !== 'string' || typeof parsed.arenaTask.fingerprint !== 'string') {
      throw new Error(`intervention-report: ${label} has a malformed arenaTask`);
    }
    arenaTask = { id: parsed.arenaTask.id, fingerprint: parsed.arenaTask.fingerprint };
  }
  return {
    biologicalScore,
    scores,
    sourceGraphSha256: parsed.sourceGraphSha256,
    seeds: { start: parsed.seeds.start, count: parsed.seeds.count },
    ticks: parsed.ticks,
    substeps: parsed.substeps,
    ...(arenaTask !== undefined ? { arenaTask } : {})
  };
};

/** Reads and parses `path` — see `parsePublishedNull`'s doc comment for why the two are split. */
export const readPublishedNull = (path: string): PublishedNull => parsePublishedNull(readFileSync(path, 'utf8'), path);

/**
 * `.agents/plans/selection-robustness/02-per-selection-chain.md` WP2:
 * `--stats-only --arena-task default` (a per-selection chain's own scoring
 * step) is extended to accept a `--null` with no recorded `arenaTask` at
 * all -- a selection's own `rewiring-null.json` is itself a default-task
 * run (`arena-task-fields.ts`'s "omit when absent" convention applies to it
 * exactly the same way it applies to the shipped `rewiring-null-v1.json`),
 * so requiring a *stamped* `arenaTask.id === 'default'` would refuse every
 * legitimate default-task null, selection or shipped alike. Still requires
 * an exact fingerprint match for every other task -- this only widens the
 * `'default'` case, and the null still has to match the run's own graph sha
 * (`assertConsistentInputs`, unaffected by this).
 */
export const statsOnlyNullMatchesTask = (
  publishedNull: Readonly<Pick<PublishedNull, 'arenaTask'>>,
  resolvedArenaTask: { readonly id: string; readonly fingerprint: string }
): boolean => {
  if (publishedNull.arenaTask !== undefined) {
    return publishedNull.arenaTask.fingerprint === resolvedArenaTask.fingerprint;
  }
  return resolvedArenaTask.id === 'default';
};

// ---------------------------------------------------------------------------
// CLI args
// ---------------------------------------------------------------------------

export interface InterventionReportArgs {
  readonly authored: string;
  readonly index: string;
  readonly publishedNull: string;
  readonly out: string;
  readonly bootstrapSeed: number;
  readonly bootstrapResamples: number;
  /** Escape hatch for diagnosis only: without it, `runInterventionReport` refuses to write `--out` when `biologicalReproduction.matches` is false (a dual-review finding — this acceptance gate was previously advisory only: the CLI printed `matches=false` and still wrote a category). */
  readonly allowReproductionMismatch: boolean;
  /** `--arena-task <id>` — labels this run's output by task (see `InterventionStatistics.arenaTask`). Absent means the default task, and the default output is unchanged. */
  readonly arenaTask?: string;
  /**
   * `.agents/plans/task-generality/02-authored-runs.md`'s WP2: `--stats-only`
   * marks this run as a per-task descriptive-statistics computation only
   * (never a WP4 publish combining step). It requires `--arena-task <id>` --
   * a stats-only run with no task makes no sense to derive a path for, and
   * every real call site pairs the two flags (`--arena-task t --stats-only`)
   * -- and, when `--out` was left at its default, derives
   * `training/runs/tasks/<id>/intervention-stats.json` so the caller doesn't
   * have to hand-build that path. An explicit `--out` is never overridden
   * (tracked via a real "was --out passed" flag, not by comparing against
   * the default path -- an operator explicitly choosing the default path
   * is still an explicit choice). This never touches `public/data` (this
   * file has no other write path than `--out` to begin with). Because the
   * derived path is never the default `--out`, `guardCanonicalOutDefault`'s
   * `diagnosticOnly` check (scoped only to the default path) does not apply
   * to a `--stats-only` run even if `--allow-reproduction-mismatch` was
   * also passed -- a diagnostic per-task run still writes normally to its
   * derived path, just carrying both `statsOnly: true` and
   * `diagnosticOnly: true`.
   */
  readonly statsOnly: boolean;
}

export const parseInterventionReportArgs = (argv: readonly string[]): InterventionReportArgs => {
  let authored = DEFAULT_AUTHORED;
  let index = DEFAULT_INDEX;
  let publishedNull = DEFAULT_PUBLISHED_NULL;
  let out = DEFAULT_OUT;
  let bootstrapSeed = DEFAULT_BOOTSTRAP_SEED;
  let bootstrapResamples = DEFAULT_BOOTSTRAP_RESAMPLES;
  let allowReproductionMismatch = false;
  let arenaTask: string | undefined;
  let statsOnly = false;
  /** True only when `--out` was actually passed, not merely defaulted -- see `--stats-only`'s own path-derivation branch below (a dual-review finding: comparing `resolve(out) === resolve(DEFAULT_OUT)` would also redirect an operator's own explicit `--out` that happens to equal the default, contradicting this field's "an explicit --out is never overridden" doc comment). Matches `export-traces.ts`'s `outDirExplicit` convention. */
  let outExplicit = false;

  let i = 0;
  while (i < argv.length) {
    const flag = argv[i];
    if (flag === '--authored') {
      authored = resolve(process.cwd(), requireValue(flag, argv[i + 1]));
      i += 2;
    } else if (flag === '--index') {
      index = resolve(process.cwd(), requireValue(flag, argv[i + 1]));
      i += 2;
    } else if (flag === '--null') {
      publishedNull = resolve(process.cwd(), requireValue(flag, argv[i + 1]));
      i += 2;
    } else if (flag === '--out') {
      out = resolve(process.cwd(), requireValue(flag, argv[i + 1]));
      outExplicit = true;
      i += 2;
    } else if (flag === '--bootstrap-seed') {
      bootstrapSeed = requireNonNegativeInt(flag, argv[i + 1]);
      i += 2;
    } else if (flag === '--bootstrap-resamples') {
      bootstrapResamples = requirePositiveInt(flag, argv[i + 1]);
      i += 2;
    } else if (flag === '--allow-reproduction-mismatch') {
      allowReproductionMismatch = true;
      i += 1;
    } else if (flag === '--arena-task') {
      arenaTask = parseArenaTaskArg(requireValue(flag, argv[i + 1]));
      i += 2;
    } else if (flag === '--stats-only') {
      statsOnly = true;
      i += 1;
    } else {
      throw new Error(`Unknown argument: ${flag}`);
    }
  }

  if (statsOnly) {
    if (arenaTask === undefined) {
      throw new Error(
        'intervention-report: --stats-only requires --arena-task <id> (it derives ' +
          'training/runs/tasks/<id>/intervention-stats.json from the task id)'
      );
    }
    // Only when `--out` was never passed at all -- an explicit `--out` (any
    // path, including one a caller happens to choose that equals the
    // default) is never second-guessed here; see this field's own doc
    // comment. Tracked via `outExplicit`, not `resolve(out) === resolve(DEFAULT_OUT)`
    // (a dual-review finding: that comparison can't distinguish "never passed"
    // from "explicitly passed the default path", so it silently redirected
    // the latter too).
    if (!outExplicit) {
      out = resolve(repoRoot, 'training', 'runs', 'tasks', arenaTask, 'intervention-stats.json');
    }
  }

  // `--out` overwriting one of its own inputs would replace a multi-hour
  // `authored.json` (or `index.json`/the published null) with the much
  // smaller statistics output -- checked before any file is touched (a
  // dual-review finding).
  for (const input of [authored, index, publishedNull]) {
    if (resolve(out) === resolve(input)) {
      throw new Error(`intervention-report: --out must not overwrite an input file (${input})`);
    }
  }

  return {
    authored,
    index,
    publishedNull,
    out,
    bootstrapSeed,
    bootstrapResamples,
    allowReproductionMismatch,
    arenaTask,
    statsOnly
  };
};

// Selection-scratch mode guard (`.agents/plans/selection-robustness/
// 02-per-selection-chain.md` WP2) -- see `./selection-scratch-guard.ts`'s
// doc comment for the full rationale. Extracted there (a thermo-
// maintainability review finding) because this module's `guardSelectionScratchOut`
// duplicated `null-report.ts`'s identical `guardSelectionScratchTarget`;
// `intervention-report.ts` now imports the shared
// `guardSelectionScratchTarget` directly from `./selection-scratch-guard`.
