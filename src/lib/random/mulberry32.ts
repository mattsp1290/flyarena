/**
 * Deterministic mulberry32 PRNG. A pure utility with no test-only
 * dependencies: `scripts/training/export-traces.ts` (a shipped
 * `package.json`-scripted tool, not a test) uses it to generate golden
 * readout weights, and `tests/fixtures/trace-graph.ts`/`tests/fixtures/tiny-graph.ts`
 * use it to generate deterministic test graphs. Previously duplicated across
 * both fixture files (and imported backward into `export-traces.ts` from
 * `tests/fixtures/`); consolidated here so every caller shares one
 * byte-identical implementation instead of three copies that could drift.
 *
 * Same seed always produces the same output sequence; different seeds
 * produce different (still deterministic) sequences.
 */
export const mulberry32 = (seed: number): (() => number) => {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};
