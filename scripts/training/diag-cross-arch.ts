// TEMPORARY diagnostic script — NOT part of the permanent test suite.
//
// Regenerates every committed golden trace file in-process (same code path
// as `tests/unit/golden-traces.test.ts`) and diffs it field-by-field against
// the committed fixtures under `tests/fixtures/golden/`, reporting the first
// divergent tick and max abs/rel difference per field family. Also probes a
// handful of transcendental Math functions (tanh/exp/sin/cos/atan2/hypot) at
// fixed inputs and prints their IEEE-754 bit patterns, to help identify
// which function is the root cause of any cross-architecture divergence.
//
// Run with: npx tsx scripts/training/diag-cross-arch.ts <output-dir>
//
// This file is removed before the fix lands — see fix/golden-cross-arch PR.

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  buildGoldenFiles,
  DEFAULT_GRAPH_ID,
  TRACE_SUBSTEPS
} from './export-traces';
import { createTraceGraph } from '../../tests/fixtures/trace-graph';

const GOLDEN_DIR = resolve(
  dirname(fileURLToPath(import.meta.url)),
  '../../tests/fixtures/golden'
);

const outDirArg = process.argv[2];
if (!outDirArg) {
  console.error('usage: diag-cross-arch.ts <output-dir>');
  process.exit(1);
}
const outDir = resolve(process.cwd(), outDirArg);
mkdirSync(outDir, { recursive: true });

const f64Bits = (value: number): string => {
  const buf = new ArrayBuffer(8);
  new Float64Array(buf)[0] = value;
  const bytes = new Uint8Array(buf);
  return Array.from(bytes)
    .reverse()
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
};

interface FieldDiff {
  field: string;
  firstDivergentTick: number | null;
  maxAbsDiff: number;
  maxRelDiff: number;
  sampleExpected: number | null;
  sampleActual: number | null;
}

// Diff two equal-length arrays-of-arrays (columnar per-tick data) or flat
// arrays; returns per-field stats. `path` is a dotted/bracket label for
// reporting.
const diffNumeric = (
  expected: unknown,
  actual: unknown,
  path: string,
  out: Record<string, FieldDiff>
): void => {
  if (typeof expected === 'number' && typeof actual === 'number') {
    if (expected === actual) return;
    const absDiff = Math.abs(expected - actual);
    const relDiff = expected !== 0 ? absDiff / Math.abs(expected) : absDiff;
    const key = path.replace(/\[\d+\]/g, '[]');
    const existing = out[key];
    if (!existing) {
      out[key] = {
        field: key,
        firstDivergentTick: extractTick(path),
        maxAbsDiff: absDiff,
        maxRelDiff: relDiff,
        sampleExpected: expected,
        sampleActual: actual
      };
    } else {
      if (existing.firstDivergentTick === null) {
        existing.firstDivergentTick = extractTick(path);
      }
      if (absDiff > existing.maxAbsDiff) {
        existing.maxAbsDiff = absDiff;
        existing.sampleExpected = expected;
        existing.sampleActual = actual;
      }
      if (relDiff > existing.maxRelDiff) existing.maxRelDiff = relDiff;
    }
    return;
  }
  if (Array.isArray(expected) && Array.isArray(actual)) {
    const len = Math.max(expected.length, actual.length);
    for (let i = 0; i < len; i += 1) {
      diffNumeric(expected[i], actual[i], `${path}[${i}]`, out);
    }
    return;
  }
  if (
    expected !== null &&
    actual !== null &&
    typeof expected === 'object' &&
    typeof actual === 'object'
  ) {
    const keys = new Set([...Object.keys(expected), ...Object.keys(actual)]);
    for (const k of keys) {
      diffNumeric((expected as any)[k], (actual as any)[k], `${path}.${k}`, out);
    }
    return;
  }
  if (expected !== actual) {
    // non-numeric mismatch (string/bool/etc) — record as a distinct field
    const key = path.replace(/\[\d+\]/g, '[]');
    out[key] = out[key] ?? {
      field: key,
      firstDivergentTick: extractTick(path),
      maxAbsDiff: NaN,
      maxRelDiff: NaN,
      sampleExpected: null,
      sampleActual: null
    };
  }
};

const extractTick = (path: string): number | null => {
  // Top-level per-tick columns look like `observations[TICK][channel]` —
  // the first bracket index is the tick.
  const match = /^\w+\[(\d+)\]/.exec(path);
  return match ? Number(match[1]) : null;
};

const main = (): void => {
  const graph = createTraceGraph();
  const files = buildGoldenFiles(graph, DEFAULT_GRAPH_ID, TRACE_SUBSTEPS);

  const report: Record<string, unknown> = {
    arch: process.arch,
    platform: process.platform,
    nodeVersion: process.version,
    v8Version: process.versions.v8
  };

  for (const { fileName, value } of files) {
    const freshText = JSON.stringify(value);
    let committedText: string;
    try {
      committedText = readFileSync(resolve(GOLDEN_DIR, fileName), 'utf8');
    } catch {
      report[fileName] = { note: 'no committed file to compare' };
      continue;
    }

    if (freshText === committedText) {
      report[fileName] = { identical: true };
      continue;
    }

    const committedValue = JSON.parse(committedText);
    const diffs: Record<string, FieldDiff> = {};
    diffNumeric(committedValue, value, fileName, diffs);
    report[fileName] = {
      identical: false,
      byteLengthExpected: committedText.length,
      byteLengthActual: freshText.length,
      diffs: Object.values(diffs)
    };
  }

  // Probe transcendental functions at fixed inputs relevant to this repo's
  // math (sensors.ts / world.ts / readout.ts): tanh, exp, sin, cos, atan2,
  // hypot.
  const probeInputs = [
    0, 1, -1, 0.5, -0.5, 1.5707963267948966, 3.14159265358979, 0.30000000000000004,
    12.345678901234, -7.6543219876
  ];
  const mathProbe: Record<string, Record<string, string>> = {};
  for (const fn of ['tanh', 'exp', 'sin', 'cos'] as const) {
    mathProbe[fn] = {};
    for (const x of probeInputs) {
      const y = Math[fn](x);
      mathProbe[fn][String(x)] = `${y} (${f64Bits(y)})`;
    }
  }
  mathProbe.atan2 = {};
  mathProbe.hypot = {};
  for (const [a, b] of [
    [1, 1],
    [0.5, -0.5],
    [12.345678901234, -7.6543219876],
    [0.30000000000000004, 0.1],
    [-3, 4]
  ] as const) {
    const ya = Math.atan2(a, b);
    mathProbe.atan2[`${a},${b}`] = `${ya} (${f64Bits(ya)})`;
    const yh = Math.hypot(a, b);
    mathProbe.hypot[`${a},${b}`] = `${yh} (${f64Bits(yh)})`;
  }
  report.mathProbe = mathProbe;

  const outPath = resolve(outDir, `cross-arch-report-${process.arch}.json`);
  writeFileSync(outPath, JSON.stringify(report, null, 2));
  console.log(`Wrote diagnostic report to ${outPath}`);
};

main();
