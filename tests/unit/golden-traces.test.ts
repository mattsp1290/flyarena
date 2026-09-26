import { readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import {
  buildGoldenFiles,
  buildSeedTrace,
  DEFAULT_GRAPH_ID,
  TRACE_SEEDS,
  TRACE_SUBSTEPS,
  TRACE_TICKS
} from '../../scripts/training/export-traces';
import { createTraceGraph } from '../fixtures/trace-graph';
import {
  diffCloseEnough,
  FLOAT_ABS_TOLERANCE,
  FLOAT_REL_TOLERANCE,
  GOLDEN_GENERATING_ARCH,
  MAX_INEXACT_LEAVES
} from '../fixtures/cross-arch-tolerance';

/**
 * Golden-trace regression: `scripts/training/export-traces.ts` is TypeScript
 * behavior frozen into committed JSON so a later PyTorch port can be checked
 * against it (`.agents/plans/trained-readout/02-gpu-port-and-parity.md`). If
 * this test fails, the arena/rate-model behavior changed; regenerate the
 * fixtures deliberately with `npm run training:traces` and review the diff
 * before committing it.
 *
 * Cross-architecture note: see `tests/fixtures/cross-arch-tolerance.ts`
 * (the full rationale and measured divergence) and `docs/architecture.md`'s
 * "Determinism scope". Short version: the committed fixtures were generated
 * on `GOLDEN_GENERATING_ARCH` (`linux-arm64`), and the byte-for-byte
 * comparison below is only meaningful there; every other architecture
 * (e.g. GitHub's x86_64 CI runner) falls back to `diffCloseEnough`'s
 * tolerance-based structural comparison, because V8's transcendental
 * `Math.*` functions used on the observation/physics path (`arena/sensors.ts`,
 * `arena/world.ts`) are "implementation-defined rounding" per the
 * ECMAScript spec and not guaranteed bit-identical across architectures —
 * unlike `connectome/model.ts`'s `stepModel`/`aggregateOutputs`, which use
 * only `+`/`-`/`*` and are bit-exact everywhere.
 */

const GOLDEN_DIR = resolve(dirname(fileURLToPath(import.meta.url)), '../fixtures/golden');
const GOLDEN_BUDGET_BYTES = 200_000;
const MAX_REPORTED_MISMATCHES = 20;

const readGoldenText = (fileName: string): string =>
  readFileSync(resolve(GOLDEN_DIR, fileName), 'utf8');

const exactnessLabel =
  process.arch === GOLDEN_GENERATING_ARCH ? 'byte-for-byte' : 'within cross-arch tolerance';

describe('golden trace regeneration', () => {
  it(`regenerates every committed golden file from a fresh build (${exactnessLabel})`, () => {
    const graph = createTraceGraph();
    const files = buildGoldenFiles(graph, DEFAULT_GRAPH_ID, TRACE_SUBSTEPS);

    expect(files.length).toBeGreaterThan(0);

    if (process.arch === GOLDEN_GENERATING_ARCH) {
      // Comparing serialized *text* (not parsed objects with `toEqual`) is
      // deliberate: `JSON.stringify` writes `-0` as `"0"`, but a parsed `-0`
      // (which `decodeAction`'s clamp can produce, e.g. `Math.max(-1, -0)`)
      // is distinct from `0` under `toEqual`'s `Object.is`-based comparison.
      // Comparing text sidesteps that entirely and is a strictly stronger
      // check — "byte-identical to what's committed" — but that equivalence
      // only holds on the architecture that generated the fixtures; see
      // this file's doc comment and `cross-arch-tolerance.ts` for every
      // other arch.
      for (const { fileName, value } of files) {
        expect(JSON.stringify(value)).toBe(readGoldenText(fileName));
      }
    } else {
      // Collected across every file before asserting (rather than failing
      // at the first divergent file) so a failure reports every affected
      // file, not just the first one alphabetically/positionally.
      const allMismatches: string[] = [];
      const overBudget: string[] = [];
      for (const { fileName, value } of files) {
        const expected = JSON.parse(readGoldenText(fileName));
        // Diff against the same JSON round-trip the byte-exact path
        // compares (`JSON.stringify` then re-parsed), not the raw
        // in-memory `value` — keeps both arch paths enforcing the same
        // serialization contract (NaN/Infinity, typed arrays, dropped
        // `undefined`, ... all normalize the same way).
        const actual = JSON.parse(JSON.stringify(value));
        const { mismatches, inexactLeaves } = diffCloseEnough(expected, actual, fileName);
        allMismatches.push(...mismatches);
        if (inexactLeaves > MAX_INEXACT_LEAVES) {
          overBudget.push(`${fileName}: ${inexactLeaves} inexact leaves (budget ${MAX_INEXACT_LEAVES})`);
        }
      }

      // Per-leaf tolerance failures first: when both this and the budget
      // check below would fail, this shows the concrete `expected`/`actual`
      // values, which is the more useful starting point for diagnosis. A
      // dense regression whose leaves each individually stay within
      // FLOAT_ABS_TOLERANCE/FLOAT_REL_TOLERANCE produces no mismatches here
      // (that's exactly the case the budget check below exists to catch).
      const shown = allMismatches.slice(0, MAX_REPORTED_MISMATCHES);
      const omitted = allMismatches.length - shown.length;
      expect(
        allMismatches,
        `Golden fixtures differ from a fresh build beyond cross-arch float tolerance ` +
          `(process.arch=${process.arch}, fixtures generated on ${GOLDEN_GENERATING_ARCH}, ` +
          `abs<=${FLOAT_ABS_TOLERANCE} or rel<=${FLOAT_REL_TOLERANCE}). Showing ${shown.length} of ` +
          `${allMismatches.length}:\n${shown.join('\n')}` +
          (omitted > 0 ? `\n... and ${omitted} more` : '')
      ).toEqual([]);

      // Dense-regression gate: see MAX_INEXACT_LEAVES's doc comment. Leaves
      // within LEAF_NOISE_FLOOR_REL/LEAF_NOISE_FLOOR_ABS of the committed
      // value never count here at all (measured cross-arch noise, including
      // world.ts's float64 feedback loop, stays under that floor); a real
      // behavior change leaves many leaves above it and trips this even
      // when every individual leaf is within FLOAT_ABS_TOLERANCE/
      // FLOAT_REL_TOLERANCE. If this fails, check the magnitude of the
      // above-floor differences (re-run with a temporary console.log of
      // `diffCloseEnough`'s output) before assuming a regression: this
      // budget is sized from a finite set of measurements, not a proof.
      expect(
        overBudget,
        'One or more golden files have more above-noise-floor, not-bit-identical numeric leaves ' +
          `than the measured cross-arch drift budget allows (every measured cross-arch noise case ` +
          `stayed under budget; every measured float32-ULP-scale regression exceeded it). Budget ` +
          `${MAX_INEXACT_LEAVES}:\n${overBudget.join('\n')}`
      ).toEqual([]);
    }

    // No orphaned committed file that buildGoldenFiles no longer produces.
    // `tasks.json` (task-generality WP1's cross-language fingerprint export)
    // and the `tasks/` directory (per-task golden traces, each covered by
    // its own `export-traces.test.ts` check) are committed here too but are
    // not part of this default-graph fixture set, so both are excluded.
    const committedNames = readdirSync(GOLDEN_DIR).filter((name) => name.endsWith('.json') && name !== 'tasks.json');
    expect(committedNames.sort()).toEqual(files.map((f) => f.fileName).sort());
  });

  it('is deterministic: two in-process runs produce identical output for every committed seed', () => {
    const graph = createTraceGraph();
    for (const seed of TRACE_SEEDS) {
      const first = buildSeedTrace(graph, DEFAULT_GRAPH_ID, seed, TRACE_TICKS, TRACE_SUBSTEPS);
      const second = buildSeedTrace(graph, DEFAULT_GRAPH_ID, seed, TRACE_TICKS, TRACE_SUBSTEPS);
      expect(second).toEqual(first);
    }
  });

  it('has a non-zero decoded thrust, yaw, and brake on at least one tick of each committed trace', () => {
    for (const seed of TRACE_SEEDS) {
      const trace = JSON.parse(readGoldenText(`trace-graph-seed-${seed}.json`)) as {
        actions: number[][];
      };
      expect(trace.actions.some((action) => action[0] !== 0)).toBe(true);
      expect(trace.actions.some((action) => action[1] !== 0)).toBe(true);
      expect(trace.actions.some((action) => action[2] !== 0)).toBe(true);
    }
  });

  it('stays inside the committed byte budget', () => {
    const names = readdirSync(GOLDEN_DIR).filter((name) => name.endsWith('.json'));
    const totalBytes = names.reduce(
      (sum, name) => sum + statSync(resolve(GOLDEN_DIR, name)).size,
      0
    );
    expect(
      totalBytes,
      `Committed golden fixtures are ${totalBytes} bytes, over the ${GOLDEN_BUDGET_BYTES}-byte ` +
        'budget. See TRACE_TICKS\'s doc comment in scripts/training/export-traces.ts for how to ' +
        'free budget (e.g. reduce TRACE_TICKS or TRACE_SEEDS) before regenerating and committing.'
    ).toBeLessThanOrEqual(GOLDEN_BUDGET_BYTES);
  });
});
