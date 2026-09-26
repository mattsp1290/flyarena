import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { createDisconnectedGraph, encodeGraphBinary } from '../../src/lib/connectome/format';
import { loadLocalAssets } from '../experiments/local-assets';
import { readRewireIndex, type RewireIndex } from '../null/rewire-index';
import { computeArmBundleSha256, deserializeArmBundle, type SerializedArmBundle } from '../training/export-arms';
import { sha256Hex } from '../training/fsio';
import type { ArmName } from '../training/arms';
import type { ExpectedGraphIdentity } from './verify-search-graph';

/**
 * WP2's single source of truth for "the planned 46 (graph, seed) pairs"
 * (`.agents/plans/repertoire-null/02-runs-and-reevaluation.md`), and for
 * each pair's arm bundle path, its search-output path, and its expected
 * per-graph identity (`ExpectedGraphIdentity`, `verify-search-graph.ts`
 * WP1). `repertoire-search.sh` (the GPU search driver) and
 * `repertoire-evaluate.ts` (the TS re-evaluation driver) both need the
 * *exact same* enumeration -- a hand-duplicated copy in bash and in
 * TypeScript would let the two drift (a search run for a pair nothing ever
 * evaluates, or an evaluation expecting a pair nothing ever searched), so
 * this module is imported directly by `repertoire-evaluate.ts` and is the
 * only place `repertoire-search.sh` gets the same data from, via this
 * file's own `--mode emit` CLI (bash cannot `import` a TS module).
 *
 * Naming convention (fixed, both drivers depend on it):
 * - `biological`/`disconnected` bundles live under `<armsDir>/base/<graphArtifactSha256>/<arm>.json`.
 * - Rewiring `N`'s bundle lives under `<armsDir>/seed<N>/<graphArtifactSha256>/rewired.json`
 *   (a *distinct* `--out` per rewiring -- `export-arms.ts` writes
 *   `<out>/<graphArtifactSha256>/<arm>.json`, and `graphArtifactSha256` is
 *   the *biological* parent's sha on every arm from one invocation, so a
 *   shared `--out` across rewirings would overwrite one rewiring's bundle
 *   with the next's).
 * - Search output lives at `<searchDir>/<graphId>-<searchSeed>.json`, where
 *   `graphId` is `biological`, `disconnected`, or `rewired-<N>`.
 */
export interface RepertoirePlanEntry {
  /** `biological` | `disconnected` | `rewired-<N>`. */
  readonly graphId: string;
  readonly arm: ArmName;
  /** The rewiring seed for a `rewired` entry, else `null`. */
  readonly rewiringSeed: number | null;
  readonly searchSeed: number;
  readonly bundlePath: string;
  readonly searchOutputPath: string;
  readonly expected: ExpectedGraphIdentity;
}

/** The shipped search's own seed (`docs/behavior-atlas-validation.md`, `SearchOptions.seed`'s default). Every graph is searched at least once at this seed. */
export const PRIMARY_SEARCH_SEED = 1729;
/** The 4 additional biological search seeds, and the 4 additional seeds rewirings 0-4 are searched at (search-seed robustness, `00-overview.md`'s "Search-seed robustness" section). */
export const EXTRA_SEARCH_SEEDS = [1730, 1731, 1732, 1733];
/** Every rewiring in `[0, REWIRED_COUNT)` is searched once, at `PRIMARY_SEARCH_SEED`. */
export const REWIRED_COUNT = 20;
/** Rewirings `[0, REWIRED_SEED_MATCHED_COUNT)` are additionally searched at every `EXTRA_SEARCH_SEEDS` entry -- the small seed-matched sample for search-seed robustness. */
export const REWIRED_SEED_MATCHED_COUNT = 5;

export interface RepertoirePlanInputs {
  readonly rewireIndex: Readonly<RewireIndex>;
  readonly biologicalBinarySha256: string;
  readonly biologicalGzipSha256: string;
  readonly disconnectedBinarySha256: string;
  readonly armsDir: string;
  readonly searchDir: string;
}

const bundlePathFor = (armsDir: string, graphArtifactSha256: string, arm: ArmName, rewiringSeed: number | null): string =>
  rewiringSeed === null
    ? resolve(armsDir, 'base', graphArtifactSha256, `${arm}.json`)
    : resolve(armsDir, `seed${rewiringSeed}`, graphArtifactSha256, `${arm}.json`);

/**
 * Build the full planned pair list, in the canonical `(graph, seed)` order
 * `repertoire-evaluate.ts`'s `evaluated.json` is sorted by: biological
 * (ascending search seed), disconnected, then rewired (ascending rewiring
 * seed, then ascending search seed). Deliberately never a lexical sort over
 * `graphId` strings, which would put `"rewired-10"` before `"rewired-2"`
 * (the same pitfall `rewire-index.ts`'s `sortedRewireSeeds` documents for
 * `rewire_batch.py`'s own `index.json`).
 */
export const buildRepertoirePlan = (inputs: Readonly<RepertoirePlanInputs>): readonly RepertoirePlanEntry[] => {
  const { rewireIndex, biologicalBinarySha256, biologicalGzipSha256, disconnectedBinarySha256, armsDir, searchDir } =
    inputs;
  const seedEntry = new Map(rewireIndex.seeds.map((entry) => [entry.seed, entry]));
  const entries: RepertoirePlanEntry[] = [];

  const outputPathFor = (graphId: string, searchSeed: number): string => resolve(searchDir, `${graphId}-${searchSeed}.json`);

  for (const searchSeed of [PRIMARY_SEARCH_SEED, ...EXTRA_SEARCH_SEEDS]) {
    entries.push({
      graphId: 'biological',
      arm: 'biological',
      rewiringSeed: null,
      searchSeed,
      bundlePath: bundlePathFor(armsDir, biologicalGzipSha256, 'biological', null),
      searchOutputPath: outputPathFor('biological', searchSeed),
      expected: { arm: 'biological', binarySha256: biologicalBinarySha256, parentGzipSha256: biologicalGzipSha256 }
    });
  }

  entries.push({
    graphId: 'disconnected',
    arm: 'disconnected',
    rewiringSeed: null,
    searchSeed: PRIMARY_SEARCH_SEED,
    bundlePath: bundlePathFor(armsDir, biologicalGzipSha256, 'disconnected', null),
    searchOutputPath: outputPathFor('disconnected', PRIMARY_SEARCH_SEED),
    expected: { arm: 'disconnected', binarySha256: disconnectedBinarySha256, parentGzipSha256: biologicalGzipSha256 }
  });

  for (let rewiringSeed = 0; rewiringSeed < REWIRED_COUNT; rewiringSeed += 1) {
    const seed = seedEntry.get(rewiringSeed);
    if (!seed) throw new Error(`repertoire-plan: index.json has no entry for rewiring seed ${rewiringSeed}`);
    const graphId = `rewired-${rewiringSeed}`;
    const bundlePath = bundlePathFor(armsDir, biologicalGzipSha256, 'rewired', rewiringSeed);
    const expected: ExpectedGraphIdentity = {
      arm: 'rewired',
      binarySha256: seed.binarySha256,
      parentGzipSha256: biologicalGzipSha256
    };
    const searchSeeds =
      rewiringSeed < REWIRED_SEED_MATCHED_COUNT ? [PRIMARY_SEARCH_SEED, ...EXTRA_SEARCH_SEEDS] : [PRIMARY_SEARCH_SEED];
    for (const searchSeed of searchSeeds) {
      entries.push({
        graphId,
        arm: 'rewired',
        rewiringSeed,
        searchSeed,
        bundlePath,
        searchOutputPath: outputPathFor(graphId, searchSeed),
        expected
      });
    }
  }

  return entries;
};

/**
 * The same per-graph binary-identity rule `verify-search-graph.ts`'s
 * `verifyAndEvaluateSearchGraph` applies to a search JSON's embedded
 * bundle, applied directly to a raw `export-arms.ts` bundle file instead --
 * there is no search JSON yet at the point `repertoire-search.sh` needs
 * this (it verifies a bundle immediately before running the GPU search
 * that will read it, per WP2's "verify then search" procedure), so this
 * function takes a bundle path and an already-known `binarySha256`
 * directly rather than deriving a fallback the way
 * `verifyAndEvaluateSearchGraph` does for a `disconnected` search with no
 * `binarySha256` supplied (`repertoire-plan.ts`'s caller always knows the
 * disconnected identity up front too, from `buildRepertoirePlan`'s own
 * `disconnectedBinarySha256` input, so that fallback path is never needed
 * here). Throws on any mismatch; returns nothing on success.
 */
export const verifyBundleIdentity = (bundlePath: string, expected: Readonly<ExpectedGraphIdentity>): void => {
  const bundle = JSON.parse(readFileSync(bundlePath, 'utf8')) as SerializedArmBundle;
  if (bundle.formatVersion !== 1) throw new Error(`repertoire-plan: ${bundlePath} has unexpected formatVersion`);
  if (bundle.arm !== expected.arm) {
    throw new Error(`repertoire-plan: ${bundlePath} arm mismatch: expected ${expected.arm}, bundle says ${bundle.arm}`);
  }
  const { sha256: declaredSha256, ...withoutHash } = bundle;
  if (computeArmBundleSha256(withoutHash) !== declaredSha256) {
    throw new Error(`repertoire-plan: ${bundlePath} is not self-consistent with its own declared sha256`);
  }
  if (bundle.graphArtifactSha256 !== expected.parentGzipSha256) {
    throw new Error(`repertoire-plan: ${bundlePath} parent identity mismatch`);
  }
  if (bundle.arm === 'disconnected' && bundle.metadata.edgeCount !== 0) {
    throw new Error(`repertoire-plan: ${bundlePath} is arm disconnected but edgeCount is not 0`);
  }
  if (!expected.binarySha256) throw new Error(`repertoire-plan: ${bundlePath} has no expected.binarySha256 to check`);
  const actualBinarySha256 = sha256Hex(new Uint8Array(encodeGraphBinary(deserializeArmBundle(bundle))));
  if (actualBinarySha256 !== expected.binarySha256) {
    throw new Error(
      `repertoire-plan: ${bundlePath} binary identity mismatch (expected ${expected.binarySha256}, got ${actualBinarySha256})`
    );
  }
};

/** Load `manifest.json` + the compiled biological graph, and derive the disconnected control's expected binary sha256 from it -- the one piece of `RepertoirePlanInputs` that isn't read straight off a file on disk. */
export const loadPlanInputs = async (
  data: string,
  graphsIndexPath: string,
  armsDir: string,
  searchDir: string
): Promise<RepertoirePlanInputs> => {
  const [assets, rewireIndex] = await Promise.all([
    loadLocalAssets(data),
    Promise.resolve(readRewireIndex(graphsIndexPath))
  ]);
  const disconnectedBinarySha256 = sha256Hex(
    new Uint8Array(encodeGraphBinary(createDisconnectedGraph(assets.parsedBiological)))
  );
  return {
    rewireIndex,
    biologicalBinarySha256: assets.manifest.binarySha256,
    biologicalGzipSha256: assets.manifest.gzipSha256,
    disconnectedBinarySha256,
    armsDir,
    searchDir
  };
};

interface CliArgs {
  readonly mode: 'emit' | 'verify-bundle';
  readonly data: string;
  readonly graphsIndex: string;
  readonly armsDir: string;
  readonly searchDir: string;
  readonly bundle?: string;
  readonly arm?: ArmName;
  readonly binarySha256?: string;
  readonly parentGzipSha256?: string;
}

const requireArg = (argv: readonly string[], flag: string): string | undefined => {
  const index = argv.indexOf(flag);
  if (index === -1) return undefined;
  const value = argv[index + 1];
  if (value === undefined || value.startsWith('--')) throw new Error(`${flag} requires a value`);
  return value;
};

const parseCliArgs = (argv: readonly string[]): CliArgs => {
  const mode = requireArg(argv, '--mode');
  if (mode !== 'emit' && mode !== 'verify-bundle') throw new Error('--mode must be "emit" or "verify-bundle"');
  const arm = requireArg(argv, '--arm');
  if (arm !== undefined && arm !== 'biological' && arm !== 'rewired' && arm !== 'disconnected') {
    throw new Error(`--arm must be biological, rewired, or disconnected, got "${arm}"`);
  }
  return {
    mode,
    data: requireArg(argv, '--data') ?? 'public/data',
    graphsIndex: requireArg(argv, '--graphs-index') ?? 'training/runs/repertoire/graphs/index.json',
    armsDir: requireArg(argv, '--arms-dir') ?? 'training/runs/repertoire/arms',
    searchDir: requireArg(argv, '--search-dir') ?? 'training/runs/repertoire/search',
    bundle: requireArg(argv, '--bundle'),
    arm,
    binarySha256: requireArg(argv, '--binary-sha256'),
    parentGzipSha256: requireArg(argv, '--parent-gzip-sha256')
  };
};

const runCli = async (): Promise<void> => {
  const args = parseCliArgs(process.argv.slice(2));
  if (args.mode === 'emit') {
    const inputs = await loadPlanInputs(args.data, args.graphsIndex, args.armsDir, args.searchDir);
    const plan = buildRepertoirePlan(inputs);
    process.stdout.write(JSON.stringify(plan));
    return;
  }
  if (!args.bundle || !args.arm || !args.parentGzipSha256) {
    throw new Error('--mode verify-bundle requires --bundle, --arm, and --parent-gzip-sha256');
  }
  verifyBundleIdentity(args.bundle, {
    arm: args.arm,
    binarySha256: args.binarySha256,
    parentGzipSha256: args.parentGzipSha256
  });
  // eslint-disable-next-line no-console -- CLI tool: this is its user-facing success output.
  console.log(`repertoire-plan: ${args.bundle} verified (${args.arm})`);
};

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  runCli().catch((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    // eslint-disable-next-line no-console -- CLI tool: this is its user-facing error output.
    console.error(message);
    process.exit(1);
  });
}
