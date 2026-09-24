import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { gunzipSync } from 'node:zlib';

import { ARENA_CONFIG, createArenaConfigFingerprint } from '../../src/lib/arena/config';
import { NEURAL_SUBSTEPS_PER_TICK } from '../../src/lib/connectome/constants';
import type { GraphMode } from '../../src/lib/connectome/format';
import { buildGraphBufferForMode, createOracleAgentBinding } from '../../src/lib/experiment/bindings';
import type { ArenaManifest } from '../../src/lib/experiment/assets';
import { ExperimentRunner, computeMedian, type ExperimentTelemetry } from '../../src/lib/experiment/runner';

/**
 * WP7 item 3: a headless, descriptive multi-seed study over the real
 * checked-in MaleCNS-derived artifact (`public/data/malecns-arena-v1.*`),
 * for every experimental arm the plan defines (biological, degree-preserving
 * rewired, disconnected negative control) — run directly through the same
 * `ExperimentRunner` + `createOracleAgentBinding` the app's own unit tests
 * and `scripts/training/export-traces.ts` use, never a browser or a Worker,
 * and through the exact same `buildGraphBufferForMode` the product's own
 * `ExperimentController` calls (`src/lib/experiment/bindings.ts`) — not a
 * hand-copied duplicate of its mode-switch logic that could silently drift
 * from the real product path.
 *
 * Every artifact byte this study reads is sha256-verified against the
 * committed manifest before use (`loadVerifiedArtifactBuffers` below) — the
 * same integrity bar the browser's own `assets.ts#loadArenaArtifacts`
 * enforces before a real run is allowed to start. A dual review pass caught
 * that an earlier version of this script skipped that check entirely, which
 * would have let a silently-corrupted or stale artifact on disk get reported
 * as "Biological (measured graph)" results with no signal at all.
 *
 * Per-mode pairing (biological-vs-biological, rewired-vs-rewired,
 * disconnected-vs-disconnected — one run per seed per mode, both arms
 * sharing that mode) rather than the product UI's own default
 * biological-vs-rewired pairing: `world.ts#processContacts` awards a
 * contested food pickup to whichever agent already overlaps it first
 * (`world.agents.find`, left before right), so pairing two *different*
 * topologies against each other in the same run would let each mode's
 * measured score be biased by which topology it happened to be racing that
 * seed, rather than reflecting that topology's own dynamics under the
 * arena's shared authored contract. Self-pairing removes that
 * cross-topology confound; every run still uses the exact same seed, world
 * config, encoder, dynamics, and decoder the product does — only the
 * topology differs between the three groups of runs, per the plan's
 * "control arms differ only in topology" boundary.
 *
 * **Sampling unit.** Each run contributes two raw per-arm samples (its left
 * and right agent), but the two arms of one self-paired run share a single
 * world (same seed, same food/hazard positions and motion, and they compete
 * for the same contested food pickups) — they are correlated, not
 * independent draws. A dual review pass caught that an earlier version of
 * this script and `docs/seed-sweep.md` reported `n = 40` (`seeds x arms`)
 * as if all 40 were independent, which overstates precision exactly where
 * the plan's "no superiority claim without multiple seeds and uncertainty"
 * boundary cares most. `summarizeByMode` below now treats the *run* (one
 * per seed) as the independent sampling unit for its primary statistics —
 * averaging the two correlated arm values per seed first, so `n` equals the
 * seed count — and additionally reports the raw per-arm breakdown
 * (`perArm`, `n = seeds x 2`) for transparency, clearly labeled as
 * correlated pairs rather than independent samples.
 *
 * This produces *descriptive* results only. Per the plan's product
 * boundary ("Results are descriptive for the POC; no superiority claim is
 * made without multiple seeds and uncertainty") and this bean's own
 * instruction, this script and `docs/seed-sweep.md` report means, medians,
 * and sample standard deviations — nothing here claims one topology
 * outperforms another. The rewired arm reflects exactly one rewiring
 * realization (`rewiredArms.seed0`); rewiring-to-rewiring variation across
 * different rewiring seeds is not sampled by this study.
 */

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '../..');
const publicDataDir = resolve(repoRoot, 'public/data');

const TOTAL_TICKS = Math.round(90 / ARENA_CONFIG.fixedDeltaSeconds); // 2,700 ticks = 90 simulated seconds at 30 Hz
const DEFAULT_SEED_COUNT = 20;
/**
 * Fixed, documented seed list rather than `Math.random()`: reproducibility
 * is the entire point of this study. Deliberately avoids seed 0
 * (`world.ts#normalizeSeed` remaps it to a fixed non-zero constant rather
 * than rejecting it — using it here would just be a confusing alias for
 * seed `0x6d2b79f5`).
 */
const buildDefaultSeeds = (count: number): number[] => Array.from({ length: count }, (_, index) => 1000 + index);
const DEFAULT_OUT_DIR = 'scripts/experiments/out';
const MODES: readonly GraphMode[] = ['biological', 'rewired', 'disconnected'];

interface CliArgs {
  seeds: number[];
  outDir: string;
}

const requireValue = (flag: string, value: string | undefined): string => {
  if (value === undefined || value.startsWith('--')) throw new Error(`${flag} requires a value`);
  return value;
};

export const parseArgs = (argv: readonly string[]): CliArgs => {
  let seedCount = DEFAULT_SEED_COUNT;
  let outDir = DEFAULT_OUT_DIR;
  for (let index = 0; index < argv.length; index += 1) {
    const flag = argv[index];
    if (flag === '--seeds') {
      const value = requireValue(flag, argv[index + 1]);
      const parsed = Number(value);
      if (!Number.isInteger(parsed) || parsed < 1) throw new Error(`--seeds must be a positive integer, got "${value}"`);
      seedCount = parsed;
      index += 1;
    } else if (flag === '--out') {
      outDir = requireValue(flag, argv[index + 1]);
      index += 1;
    } else {
      throw new Error(`Unknown argument: ${flag}`);
    }
  }
  if (seedCount < 20) {
    // A prior version of this message suggested a `--out` workaround for a
    // smaller ad hoc run; a dual review pass caught that the floor applies
    // unconditionally regardless of `--out`, making that advice actively
    // wrong. There is no bypass — this floor is the committed methodology.
    throw new Error(`--seeds must be >= 20 (this study's committed methodology; got ${seedCount}).`);
  }
  return { seeds: buildDefaultSeeds(seedCount), outDir };
};

interface VerifiedArtifacts {
  biological: ArrayBuffer;
  rewired: ArrayBuffer;
  manifest: ArenaManifest;
}

/**
 * Read both real artifacts and sha256-verify each decompressed buffer
 * against the committed manifest before returning anything — the same
 * integrity bar `src/lib/experiment/assets.ts#loadArenaArtifacts` enforces
 * for the browser (see this module's doc comment). Throws with a clear
 * message on any mismatch rather than silently proceeding.
 */
const loadVerifiedArtifactBuffers = (): VerifiedArtifacts => {
  const manifest = JSON.parse(
    readFileSync(resolve(publicDataDir, 'malecns-arena-v1.manifest.json'), 'utf-8')
  ) as ArenaManifest;
  const rewiredEntry = manifest.rewiredArms.seed0;
  if (!rewiredEntry) throw new Error('Manifest is missing the seed0 rewired control arm');

  const readVerified = (filename: string, expectedBinarySha256: string): ArrayBuffer => {
    const binary = gunzipSync(readFileSync(resolve(publicDataDir, filename)));
    const actualSha256 = createHash('sha256').update(binary).digest('hex');
    if (actualSha256 !== expectedBinarySha256) {
      throw new Error(
        `${filename}: decompressed sha256 ${actualSha256} does not match manifest (${expectedBinarySha256})`
      );
    }
    return binary.buffer.slice(binary.byteOffset, binary.byteOffset + binary.byteLength);
  };

  return {
    biological: readVerified(manifest.artifact, manifest.binarySha256),
    rewired: readVerified(rewiredEntry.artifact, rewiredEntry.binarySha256),
    manifest
  };
};

/** Drive one `ExperimentRunner` to `finished`, polling its status the same way the project's own unit tests do (`tests/unit/experiment-runner.test.ts#runToFinished`) — no real-time pacing (`targetTickIntervalMs: 0`), since this is a headless batch study, not a real-time demo. */
const runToFinished = (runner: ExperimentRunner): Promise<void> =>
  new Promise((resolveRun, rejectRun) => {
    const poll = setInterval(() => {
      const status = runner.getStatus();
      if (status === 'finished') {
        clearInterval(poll);
        resolveRun();
      } else if (status === 'error') {
        clearInterval(poll);
        rejectRun(new Error('ExperimentRunner entered the error state during the seed sweep'));
      }
    }, 1);
    runner.start();
  });

export interface SeedSampleResult {
  mode: GraphMode;
  seed: number;
  arm: 'left' | 'right';
  foodPickups: number;
  hazardContacts: number;
  distanceTravelled: number;
  movementScore: number;
}

const runOneSelfPairedSeed = async (
  artifacts: VerifiedArtifacts,
  mode: GraphMode,
  seed: number
): Promise<SeedSampleResult[]> => {
  const buildBinding = () =>
    createOracleAgentBinding({
      graphBuffer: buildGraphBufferForMode(artifacts.biological, artifacts.rewired, mode),
      mode
    });
  const runner = new ExperimentRunner({
    seed,
    totalTicks: TOTAL_TICKS,
    agents: { left: buildBinding(), right: buildBinding() },
    substepsPerTick: NEURAL_SUBSTEPS_PER_TICK,
    targetTickIntervalMs: 0
  });
  await runToFinished(runner);
  const telemetry: ExperimentTelemetry = runner.getTelemetry();
  return (['left', 'right'] as const).map((arm) => ({
    mode,
    seed,
    arm,
    foodPickups: telemetry.agents[arm].foodPickups,
    hazardContacts: telemetry.agents[arm].hazardContacts,
    distanceTravelled: telemetry.agents[arm].distanceTravelled,
    movementScore: telemetry.agents[arm].movementScore
  }));
};

const mean = (values: readonly number[]): number =>
  values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length;

/** Sample standard deviation (n-1 denominator); 0 for fewer than 2 samples rather than NaN. */
const sampleStdDev = (values: readonly number[]): number => {
  if (values.length < 2) return 0;
  const m = mean(values);
  const variance = values.reduce((sum, value) => sum + (value - m) ** 2, 0) / (values.length - 1);
  return Math.sqrt(variance);
};

export interface MetricStats {
  n: number;
  mean: number;
  median: number;
  sampleStdDev: number;
}

const statsFor = (values: readonly number[]): MetricStats => ({
  n: values.length,
  mean: mean(values),
  // Reuses the exact function `ExperimentRunner`'s own telemetry median
  // (`runner.ts`) is built on, rather than a second hand-maintained copy of
  // the same sorted-median logic.
  median: computeMedian(values),
  sampleStdDev: sampleStdDev(values)
});

const METRIC_KEYS = ['foodPickups', 'hazardContacts', 'distanceTravelled', 'movementScore'] as const;
type MetricKey = (typeof METRIC_KEYS)[number];

/** One averaged value per seed for `key` — the run/seed is the independent sampling unit; see this module's doc comment. */
const perRunAveraged = (forMode: readonly SeedSampleResult[], key: MetricKey): number[] => {
  const bySeed = new Map<number, number[]>();
  for (const result of forMode) {
    const bucket = bySeed.get(result.seed) ?? [];
    bucket.push(result[key]);
    bySeed.set(result.seed, bucket);
  }
  return [...bySeed.values()].map(mean);
};

export interface ModeMetrics {
  foodPickups: MetricStats;
  hazardContacts: MetricStats;
  distanceTravelled: MetricStats;
  movementScore: MetricStats;
}

export interface ModeSummary extends ModeMetrics {
  mode: GraphMode;
  /** Independent sampling units: one per seed (the shared world/run), not one per arm — see this module's doc comment. The stats above (`foodPickups`..`movementScore`) are computed at this granularity: the two arms of each run are averaged together first. */
  runCount: number;
  /** `runCount * 2`: the raw per-arm sample count backing `perArm` below. */
  armSampleCount: number;
  /**
   * Secondary breakdown at arm granularity, included for transparency only.
   * The two arms within one run are correlated (shared world, contested
   * food), not independent — a round-2 dual review pass caught that an
   * earlier version of this comment mischaracterized the effect as
   * `sampleStdDev` here being an "underestimate" of true variation. It is
   * not: averaging two values before computing a statistic always tends to
   * *reduce* dispersion relative to the raw values (regardless of
   * correlation), which is exactly why `perArm`'s `sampleStdDev` values are
   * larger than the per-run ones above in this study's actual results, not
   * smaller. The real risk with `perArm` is precision, not magnitude: using
   * its `n` (`= seeds x 2`) in any downstream calculation that assumes
   * independent samples — e.g. a standard error of `sampleStdDev /
   * sqrt(n)` — would overstate how precisely known each mean is, because
   * the true number of independent observations is the seed count, not
   * twice it. The per-run stats above (`n` = seed count) are the correct
   * basis for that kind of calculation; `perArm` is reported only so the
   * raw per-agent values are visible, not as a second "safer" statistic.
   */
  perArm: ModeMetrics;
}

export const summarizeByMode = (results: readonly SeedSampleResult[]): ModeSummary[] =>
  MODES.map((mode) => {
    const forMode = results.filter((result) => result.mode === mode);
    const runCount = new Set(forMode.map((result) => result.seed)).size;
    const perRunStats = (key: MetricKey): MetricStats => statsFor(perRunAveraged(forMode, key));
    const perArmStats = (key: MetricKey): MetricStats => statsFor(forMode.map((result) => result[key]));
    return {
      mode,
      runCount,
      armSampleCount: forMode.length,
      foodPickups: perRunStats('foodPickups'),
      hazardContacts: perRunStats('hazardContacts'),
      distanceTravelled: perRunStats('distanceTravelled'),
      movementScore: perRunStats('movementScore'),
      perArm: {
        foodPickups: perArmStats('foodPickups'),
        hazardContacts: perArmStats('hazardContacts'),
        distanceTravelled: perArmStats('distanceTravelled'),
        movementScore: perArmStats('movementScore')
      }
    };
  });

const main = async (): Promise<void> => {
  const args = parseArgs(process.argv.slice(2));
  const artifacts = loadVerifiedArtifactBuffers();

  const outDir = resolve(process.cwd(), args.outDir);
  mkdirSync(outDir, { recursive: true });

  const startedAt = performance.now();
  const results: SeedSampleResult[] = [];
  for (const mode of MODES) {
    for (const seed of args.seeds) {
      // eslint-disable-next-line no-await-in-loop -- deliberately sequential: this is a batch CLI study, not a request path; running seeds concurrently would gain nothing (single CPU core) and would make the per-seed console progress line meaningless.
      const samples = await runOneSelfPairedSeed(artifacts, mode, seed);
      results.push(...samples);
      // eslint-disable-next-line no-console -- CLI tool: user-facing progress output.
      console.log(
        `[seed-sweep] ${mode} seed=${seed}: left movementScore=${samples[0].movementScore.toFixed(2)}, ` +
          `right movementScore=${samples[1].movementScore.toFixed(2)}`
      );
    }
  }
  const elapsedMs = performance.now() - startedAt;

  const summary = summarizeByMode(results);

  const output = {
    generatedAt: new Date().toISOString(),
    totalTicks: TOTAL_TICKS,
    substepsPerTick: NEURAL_SUBSTEPS_PER_TICK,
    // Ties this study's numbers to exactly which sha256-verified compiled
    // artifacts and which arena config produced them (see this module's doc
    // comment and `ExperimentReplayExport.graphBinarySha256`'s equivalent
    // provenance field for the product's own downloaded replays).
    artifacts: {
      biologicalBinarySha256: artifacts.manifest.binarySha256,
      rewiredBinarySha256: artifacts.manifest.rewiredArms.seed0.binarySha256
    },
    configFingerprint: createArenaConfigFingerprint(ARENA_CONFIG),
    seeds: args.seeds,
    elapsedMs,
    results,
    summary
  };

  const outPath = resolve(outDir, 'seed-sweep-results.json');
  writeFileSync(outPath, JSON.stringify(output, null, 2));

  // eslint-disable-next-line no-console -- CLI tool: user-facing summary output.
  console.log(`\n[seed-sweep] ${args.seeds.length} seeds x ${MODES.length} modes finished in ${(elapsedMs / 1000).toFixed(1)}s`);
  // eslint-disable-next-line no-console -- CLI tool: user-facing summary output.
  console.log(`[seed-sweep] wrote ${results.length} samples to ${outPath}`);
  for (const modeSummary of summary) {
    // eslint-disable-next-line no-console -- CLI tool: user-facing summary output.
    console.log(
      `[seed-sweep] ${modeSummary.mode} (runs=${modeSummary.runCount}, arm samples=${modeSummary.armSampleCount}): ` +
        `movementScore mean=${modeSummary.movementScore.mean.toFixed(2)} median=${modeSummary.movementScore.median.toFixed(2)} ` +
        `sd=${modeSummary.movementScore.sampleStdDev.toFixed(2)} (per-run, arms averaged)`
    );
  }
};

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  main().catch((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    // eslint-disable-next-line no-console -- CLI tool: user-facing error output.
    console.error(`seed-sweep failed: ${message}`);
    process.exit(1);
  });
}
