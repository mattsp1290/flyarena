// @vitest-environment node
import { describe, expect, it } from 'vitest';
import { parseArgs, summarizeByMode, type SeedSampleResult } from '../../scripts/experiments/seed-sweep';

/**
 * WP7 follow-up: `scripts/experiments/seed-sweep.ts` was previously outside
 * every automated gate — not type-checked (fixed separately by adding
 * `scripts/**` to `tsconfig.json`'s `include`) and, before this file, never
 * imported by anything, so a rename in `runner.ts`/`bindings.ts` could break
 * this committed methodology script without any test noticing. Its
 * `process.argv[1] === fileURLToPath(import.meta.url)` guard means
 * importing it here for `parseArgs`/`summarizeByMode` does not invoke
 * `main()` (which would otherwise try to read `public/data/*` and run a
 * real headless sweep during `npm run test:unit`).
 */

describe('parseArgs', () => {
  it('defaults to 20 seeds (1000..1019) and the default output directory', () => {
    const args = parseArgs([]);
    expect(args.seeds).toHaveLength(20);
    expect(args.seeds[0]).toBe(1000);
    expect(args.seeds.at(-1)).toBe(1019);
    expect(args.outDir).toBe('scripts/experiments/out');
  });

  it('accepts an explicit --seeds count >= 20 and builds that many sequential seeds', () => {
    const args = parseArgs(['--seeds', '25']);
    expect(args.seeds).toHaveLength(25);
    expect(args.seeds.at(-1)).toBe(1024);
  });

  it('accepts --out and applies it independent of --seeds', () => {
    const args = parseArgs(['--out', 'tmp/sweep', '--seeds', '30']);
    expect(args.outDir).toBe('tmp/sweep');
    expect(args.seeds).toHaveLength(30);
  });

  it('rejects fewer than 20 seeds, with no --out workaround (a prior version of this message incorrectly implied one)', () => {
    expect(() => parseArgs(['--seeds', '19'])).toThrow(/>= 20/);
    expect(() => parseArgs(['--seeds', '1', '--out', 'anything'])).toThrow(/>= 20/);
  });

  it('rejects a non-integer or negative --seeds value', () => {
    expect(() => parseArgs(['--seeds', '20.5'])).toThrow(/positive integer/);
    expect(() => parseArgs(['--seeds', '-5'])).toThrow(/positive integer/);
    expect(() => parseArgs(['--seeds', 'nope'])).toThrow(/positive integer/);
  });

  it('rejects a flag missing its value', () => {
    expect(() => parseArgs(['--seeds'])).toThrow(/requires a value/);
    expect(() => parseArgs(['--out'])).toThrow(/requires a value/);
  });

  it('rejects an unknown flag', () => {
    expect(() => parseArgs(['--bogus'])).toThrow(/Unknown argument/);
  });
});

describe('summarizeByMode', () => {
  const sample = (
    mode: SeedSampleResult['mode'],
    seed: number,
    arm: SeedSampleResult['arm'],
    movementScore: number
  ): SeedSampleResult => ({
    mode,
    seed,
    arm,
    foodPickups: 0,
    hazardContacts: 0,
    distanceTravelled: 0,
    movementScore
  });

  it('treats the run (seed) as the independent sampling unit: runCount equals the seed count, not the arm-sample count', () => {
    const results: SeedSampleResult[] = [
      sample('biological', 1000, 'left', 10),
      sample('biological', 1000, 'right', 20),
      sample('biological', 1001, 'left', 30),
      sample('biological', 1001, 'right', 40)
    ];
    const [biological] = summarizeByMode(results).filter((entry) => entry.mode === 'biological');
    expect(biological.runCount).toBe(2);
    expect(biological.armSampleCount).toBe(4);
  });

  it('averages the two correlated arms per seed before computing the primary per-run stats', () => {
    const results: SeedSampleResult[] = [
      sample('biological', 1000, 'left', 10),
      sample('biological', 1000, 'right', 20), // seed 1000 average: 15
      sample('biological', 1001, 'left', 30),
      sample('biological', 1001, 'right', 50) // seed 1001 average: 40
    ];
    const [biological] = summarizeByMode(results).filter((entry) => entry.mode === 'biological');
    // Per-run mean of [15, 40] = 27.5 -- not the flat mean of all four raw
    // values ([10,20,30,50] -> 27.5 happens to coincide here only because
    // both seeds have exactly 2 arms; the median below is what actually
    // distinguishes per-run from per-arm grouping.
    expect(biological.movementScore.mean).toBeCloseTo(27.5, 10);
    expect(biological.movementScore.n).toBe(2);
    expect(biological.movementScore.median).toBeCloseTo(27.5, 10); // median of [15, 40]
  });

  it('still reports the raw per-arm breakdown separately, at arm granularity', () => {
    const results: SeedSampleResult[] = [
      sample('biological', 1000, 'left', 10),
      sample('biological', 1000, 'right', 20),
      sample('biological', 1001, 'left', 30),
      sample('biological', 1001, 'right', 50)
    ];
    const [biological] = summarizeByMode(results).filter((entry) => entry.mode === 'biological');
    expect(biological.perArm.movementScore.n).toBe(4);
    expect(biological.perArm.movementScore.mean).toBeCloseTo((10 + 20 + 30 + 50) / 4, 10);
  });

  it('reports one entry per declared mode, even for a mode with zero samples', () => {
    const summary = summarizeByMode([sample('biological', 1000, 'left', 5)]);
    expect(summary.map((entry) => entry.mode)).toEqual(['biological', 'rewired', 'disconnected']);
    const [rewired] = summary.filter((entry) => entry.mode === 'rewired');
    expect(rewired.runCount).toBe(0);
    expect(rewired.movementScore.n).toBe(0);
    expect(rewired.movementScore.mean).toBe(0);
    expect(rewired.movementScore.sampleStdDev).toBe(0);
  });

  it('sample standard deviation uses the n-1 (sample) denominator', () => {
    // Per-run values [10, 20, 30] (three seeds, one arm each so per-run ==
    // per-arm here): mean 20, sample variance = ((10)^2+(0)^2+(10)^2)/(3-1)
    // = 200/2 = 100 -> sd = 10.
    const results: SeedSampleResult[] = [
      sample('disconnected', 1000, 'left', 10),
      sample('disconnected', 1001, 'left', 20),
      sample('disconnected', 1002, 'left', 30)
    ];
    const [disconnected] = summarizeByMode(results).filter((entry) => entry.mode === 'disconnected');
    expect(disconnected.movementScore.sampleStdDev).toBeCloseTo(10, 10);
  });
});
