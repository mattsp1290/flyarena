import { sameSeeds } from './null-stats';
import type { NullGraphListEvaluationRaw } from './null-evaluate';
import type { GraphKind, GraphListIndexInfo, PublishedNull } from './intervention-report';

/**
 * `intervention-report.ts`'s input-consistency preflight, extracted into its
 * own module (a thermo-maintainability review finding: this single function
 * was larger than most whole utility files in this codebase, and it is
 * specifically the piece `intervention-report.ts`'s own doc comment says
 * WP3 will need to parameterize -- "this hard `'authored'` guard and the
 * published-null comparison ... are exactly the two pieces that will need
 * to become parameters rather than fixed to this module's own inputs".
 * Extracting it now means WP3's parameterization change touches this small,
 * focused file instead of the middle of `intervention-report.ts`).
 *
 * Takes `raw`/`info`/`publishedNull` and returns void-or-throw exactly as
 * before this extraction -- a pure move, no behavior change.
 *
 * Only imports `type`s back from `intervention-report.ts` (`GraphKind`/
 * `GraphListIndexInfo`/`PublishedNull`, all already exported there) --
 * type-only imports are erased at compile time, so this creates no runtime
 * circularity even though `intervention-report.ts` itself imports the
 * runtime `assertConsistentInputs` export below.
 */

/**
 * A `null`/`NaN`/`Infinity` `movementScore` would silently become `0` (or
 * throw far from here) once it reaches `mean`/`graphStats`/`pairedStats` —
 * the same class of bug `null-report.ts`'s own `assertFiniteScores`
 * documents. `label` identifies the graph (`"biological"`, or the
 * graph-list `id`) in the thrown message.
 */
const assertFiniteScores = (label: string, movementScore: readonly number[]): void => {
  const badIndex = movementScore.findIndex((value) => typeof value !== 'number' || !Number.isFinite(value));
  if (badIndex !== -1) {
    throw new Error(`intervention-report: ${label}.movementScore[${badIndex}] is not a finite number`);
  }
};

/**
 * `buildInterventionStatistics`'s preflight: everything that must hold for
 * `raw` (a `--graph-list` `authored.json`), `info` (`index.json`'s
 * id -> kind/gzipSha256 map plus its declared `controlCount`), and
 * `publishedNull` (`rewiring-null-v1.json`'s scores + condition fields) to
 * describe the same comparable experiment. None of this was previously
 * checked (a dual-review finding on both `null-evaluate.ts --graph-list` and
 * `intervention-report.ts` — a partial, stale, or wrong-decoder
 * `authored.json` would silently produce a category ranked against an
 * incompatible null):
 *
 * - **Shape.** `raw.graphs` must be an array — catches an operator pointing
 *   `--authored` at a `--rewired-index`-mode `authored.json` (same file
 *   name, different shape: `rewired`/no `graphs`) with an actionable message
 *   instead of a bare `TypeError` several lines later.
 * - **Decoder.** The predeclared categories are authored-decoder only
 *   (`00-overview.md`), and the published null's own `condition` is
 *   `"authored, opponent parked"` — a non-`'authored'` `raw.decoder` is
 *   rejected outright, mirroring `null-report.ts`'s own decoder guard.
 *   (WP3's trained-decoder statistics reuse `armDistribution`/`evaluateCategory`
 *   against a 5-graph-per-arm freshly-trained C/M reference instead of this
 *   published 500-graph null — `03-evaluation.md`'s "the same statistics as
 *   WP2" — so this hard `'authored'` guard and the published-null comparison
 *   above are exactly the two pieces that will need to become parameters
 *   rather than fixed to this module's own inputs; noted here so WP3 doesn't
 *   have to rediscover it.)
 * - **Seeds/ticks/substeps/source graph.** `raw.seeds`/`raw.ticks`/
 *   `raw.substeps`/`raw.sourceGraphSha256` must match `publishedNull`'s own
 *   recorded values — otherwise every rank/percentile below compares scores
 *   from two different conditions.
 * - **Coverage, both directions.** Every id in `info.entries` (the index)
 *   must appear in `raw.graphs`, and every id in `raw.graphs` must appear in
 *   `info.entries` (the acceptance criterion in `03-evaluation.md`:
 *   "`authored.json` has entries for every index id") — checked together so
 *   a coverage mismatch always reports which ids are on which side, rather
 *   than the "extra id" direction failing one graph at a time inside the
 *   per-graph loop below with a message that reads like a missing-`kind`
 *   bug. `raw.graphs` must also not repeat an id — otherwise an arm silently
 *   shrinks or double-counts a graph.
 * - **P/Q counts.** Checked before the C/M/MQ arm-size loop below (a
 *   dual-review finding): the most likely hand-edit/corruption mode is a
 *   mislabeled P or Q entry (e.g. the real "P" id relabeled `kind: 'C'`),
 *   which would otherwise surface as "arm C has 101 graph(s)..." — accurate,
 *   but it points an operator at the wrong arm instead of at P/Q directly.
 * - **Arm size vs the index's own declared `controlCount`.** Each of C/M/MQ
 *   must have exactly `info.controlCount` graphs — `armDistribution` alone
 *   only requires "at least one", so a truncated or dev-sized index (a
 *   smaller `--control-count`) would otherwise silently compute the category
 *   at a coarser percentile resolution with no signal in the output.
 * - **Graph identity.** Each `raw.graphs[i].gzipSha256` must match `info`'s
 *   own `gzipSha256` for that id — catching a stale `authored.json` scored
 *   against an older `interventions.py` run (a different `k`) than the
 *   `index.json` now being read.
 * - **Seed alignment.** `raw.biological.heldOutSeeds` must equal the
 *   `[seeds.start, ..., seeds.start + seeds.count - 1]` range `raw.seeds`
 *   itself claims, and every other graph (biological included) must share
 *   that same sequence, in order — `pairedStats` below only checks array
 *   *length*, not which seeds they are.
 * - **Finite scores.** No `movementScore` entry may be non-finite.
 */
export const assertConsistentInputs = (
  raw: Readonly<NullGraphListEvaluationRaw>,
  info: Readonly<GraphListIndexInfo>,
  publishedNull: Readonly<PublishedNull>
): void => {
  if (!Array.isArray(raw.graphs)) {
    throw new Error(
      'intervention-report: authored.json has no graphs[] -- is this a --rewired-index run? intervention-report ' +
        'needs `null-evaluate.ts --graph-list` output'
    );
  }
  if (raw.decoder !== 'authored') {
    throw new Error(
      `intervention-report: authored.json was scored with decoder "${raw.decoder}"; the predeclared categories ` +
        'are authored-decoder only'
    );
  }
  if (raw.sourceGraphSha256 !== publishedNull.sourceGraphSha256) {
    throw new Error(
      'intervention-report: authored.json and the published null were scored against different biological graphs ' +
        `(authored.json ${raw.sourceGraphSha256}, published null ${publishedNull.sourceGraphSha256})`
    );
  }
  if (
    raw.seeds.start !== publishedNull.seeds.start ||
    raw.seeds.count !== publishedNull.seeds.count ||
    raw.ticks !== publishedNull.ticks ||
    raw.substeps !== publishedNull.substeps
  ) {
    throw new Error(
      'intervention-report: authored.json seeds/ticks/substeps differ from the published null -- ranks would ' +
        `compare different conditions (authored.json: seeds ${raw.seeds.start}..${raw.seeds.start + raw.seeds.count - 1} ` +
        `ticks=${raw.ticks} substeps=${raw.substeps}; published null: seeds ${publishedNull.seeds.start}..` +
        `${publishedNull.seeds.start + publishedNull.seeds.count - 1} ticks=${publishedNull.ticks} ` +
        `substeps=${publishedNull.substeps})`
    );
  }

  const rawIds = raw.graphs.map((g) => g.id);
  const duplicate = rawIds.find((id, i) => rawIds.indexOf(id) !== i);
  if (duplicate !== undefined) {
    throw new Error(`intervention-report: authored.json has a duplicate graph id "${duplicate}"`);
  }
  const rawIdSet = new Set(rawIds);
  const missing = [...info.entries.keys()].filter((id) => !rawIdSet.has(id));
  const extra = rawIds.filter((id) => !info.entries.has(id));
  if (missing.length > 0 || extra.length > 0) {
    const missingText =
      missing.length > 0
        ? `missing ${missing.length} index id(s): ${missing.slice(0, 10).join(', ')}${missing.length > 10 ? ', ...' : ''}`
        : undefined;
    const extraText =
      extra.length > 0
        ? `${extra.length} id(s) not in index.json (stale authored.json or a different index?): ` +
          `${extra.slice(0, 10).join(', ')}${extra.length > 10 ? ', ...' : ''}`
        : undefined;
    throw new Error(
      `intervention-report: authored.json/index.json coverage mismatch -- ${[missingText, extraText].filter(Boolean).join('; ')}`
    );
  }

  if (!raw.biological) {
    throw new Error(
      'intervention-report: authored.json has no biological section (run null-evaluate with --biological)'
    );
  }
  assertFiniteScores('biological', raw.biological.movementScore);
  const referenceSeeds = raw.biological.heldOutSeeds;
  const expectedSeeds = Array.from({ length: raw.seeds.count }, (_, i) => raw.seeds.start + i);
  if (!sameSeeds(referenceSeeds, expectedSeeds)) {
    throw new Error(
      'intervention-report: authored.json biological.heldOutSeeds does not match its own seeds.start/seeds.count'
    );
  }

  const armCounts = new Map<GraphKind, number>();
  for (const graph of raw.graphs) {
    const entryInfo = info.entries.get(graph.id);
    if (!entryInfo) throw new Error(`intervention-report: no kind found for graph id "${graph.id}" in the index`);
    if (entryInfo.gzipSha256 !== graph.gzipSha256) {
      throw new Error(
        `intervention-report: "${graph.id}" was scored from a different graph file than index.json currently ` +
          `lists (authored.json gzipSha256 ${graph.gzipSha256}, index.json ${entryInfo.gzipSha256}) -- stale authored.json?`
      );
    }
    if (!sameSeeds(graph.heldOutSeeds, referenceSeeds)) {
      throw new Error(`intervention-report: "${graph.id}" was scored on different held-out seeds than biological`);
    }
    assertFiniteScores(graph.id, graph.movementScore);
    armCounts.set(entryInfo.kind, (armCounts.get(entryInfo.kind) ?? 0) + 1);
  }

  for (const kind of ['P', 'Q'] as const) {
    const n = armCounts.get(kind) ?? 0;
    if (n !== 1) {
      throw new Error(`intervention-report: expected exactly one graph of kind "${kind}", found ${n}`);
    }
  }

  for (const kind of ['C', 'M', 'MQ'] as const) {
    const n = armCounts.get(kind) ?? 0;
    if (n !== info.controlCount) {
      throw new Error(
        `intervention-report: arm ${kind} has ${n} graph(s), but index.json declares controlCount=${info.controlCount}`
      );
    }
  }
};
