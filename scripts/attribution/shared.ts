import { readFileSync } from 'node:fs';

import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import type { ConnectomeGraph } from '../../src/lib/connectome/format';
import { resolveArenaTask } from '../../src/lib/arena/tasks';
import { readoutFromFlat } from '../../src/lib/connectome/readout-serialization';
import type { ReadoutWeights } from '../../src/lib/connectome/readout';
import { sha256Hex } from '../training/fsio';
import { arenaTaskFingerprintOf, type ArchivedReadout } from './archive-readouts';
import { resolveReadoutGraph, type ResolveGraphConfig } from './resolve-graph';

/**
 * Small pieces every WP2 analysis script (`saliency.ts`, `ablate.ts`,
 * `independence.ts`, `regime.ts`) shares: loading `training/archive/
 * trained-readouts-v1.json`, decoding one entry's `theta` into
 * `ReadoutWeights`, resolving its graph, and the two predeclared held-out
 * seed ranges (`.agents/plans/readout-attribution/00-overview.md`'s
 * predeclared analyses: saliency trajectories on seeds 30001-30010,
 * everything scored on seeds 30001-30100). Extracted here rather than
 * hand-duplicated across the four scripts.
 */

const here = dirname(fileURLToPath(import.meta.url));
export const repoRoot = resolve(here, '../..');

export const DEFAULT_ARCHIVE_PATH = resolve(repoRoot, 'training/archive/trained-readouts-v1.json');
export const DEFAULT_MANIFEST_PATH = resolve(repoRoot, 'public/data/malecns-arena-v1.manifest.json');
export const DEFAULT_ARCHIVED_INTERVENTION_INDEX_PATH = resolve(
  repoRoot,
  'training/archive/intervention-index-v1.json'
);

/** Seeds 30001..30010 -- the saliency-trajectory held-out set (`00-overview.md`'s predeclared analysis 1). */
export const SALIENCY_SEEDS: readonly number[] = Array.from({ length: 10 }, (_, i) => 30001 + i);
/** Seeds 30001..30100 -- the ablation/independence held-out scoring set, matching `null-trained-evaluate.ts`'s `DEFAULT_HELD_OUT_START`/`DEFAULT_HELD_OUT_COUNT`. */
export const SCORING_SEEDS: readonly number[] = Array.from({ length: 100 }, (_, i) => 30001 + i);
export const SALIENCY_TICKS = 1800;
export const SCORING_TICKS = 1800;

interface TrainedReadoutArchive {
  readonly version: number;
  readonly readouts: readonly ArchivedReadout[];
}

const DEFAULT_TASK_FINGERPRINT = resolveArenaTask('default').fingerprint;

/**
 * Load and lightly validate `training/archive/trained-readouts-v1.json`,
 * sorted by id (the archive's own committed order), scoped to the
 * default-task readouts only (`entry.arenaTask === 'default'`) --
 * WP1b's per-task archive (`flyarena-qp2e`, merged) added 52
 * `kind: "task-intervention"` entries across 4 non-default arena tasks
 * (`crowded`/`hazard-heavy`/`no-movement`/`sparse-food`) that this WP's own
 * predeclared analyses and H1-H3 (`00-overview.md`) were never scoped
 * around: every hypothesis rule reads fixed default-task ids
 * (`biological-seed101`, `P-seed202`, ...), and including the per-task
 * entries would roughly triple the ablation compute (75 readouts x 17
 * tasks x 100 seeds vs 23's) for zero effect on any H1-H3 outcome. Scoped
 * here, in one place, rather than per-CLI, so every WP2 analysis stays
 * consistently scoped without repeating the filter five times. Each
 * default-task entry's `arenaTaskFingerprintOf(entry)` is cross-checked
 * against the default task's own fingerprint (not merely trusting the
 * `arenaTask` string label) -- the sanctioned read path
 * (`archive-readouts.ts`'s own doc comment on `arenaTaskFingerprint`),
 * catching a hypothetical future entry mislabeled `arenaTask: "default"`
 * whose stored fingerprint disagrees.
 */
export const loadArchive = (archivePath: string): readonly ArchivedReadout[] => {
  const parsed = JSON.parse(readFileSync(archivePath, 'utf8')) as Partial<TrainedReadoutArchive>;
  if (!Array.isArray(parsed.readouts)) {
    throw new Error(`shared: ${archivePath} is missing a "readouts" array`);
  }
  const defaultTaskReadouts = parsed.readouts.filter((entry) => entry.arenaTask === 'default');
  for (const entry of defaultTaskReadouts) {
    const fingerprint = arenaTaskFingerprintOf(entry);
    if (fingerprint !== DEFAULT_TASK_FINGERPRINT) {
      throw new Error(
        `shared: archive entry "${entry.id}" has arenaTask "default" but arenaTaskFingerprintOf resolves to ` +
          `"${fingerprint}", not the default task's own fingerprint "${DEFAULT_TASK_FINGERPRINT}"`
      );
    }
  }
  return [...defaultTaskReadouts].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
};

/**
 * Decode one archived readout's base64 `theta` into `ReadoutWeights`,
 * validated against its own recorded `D`/`H`/`weightsSha256`
 * (`readoutFromFlat` throws on a parameter-count mismatch or a non-finite
 * value; the `weightsSha256` recheck below catches silent base64/byte
 * corruption that a shape check alone would miss).
 */
export const weightsForEntry = (entry: Readonly<ArchivedReadout>): ReadoutWeights => {
  const buffer = Buffer.from(entry.theta, 'base64');
  // `.slice()` on the underlying `ArrayBuffer` guarantees 4-byte alignment
  // regardless of the `Buffer`'s own pool offset (the same pattern
  // `regime-task.ts`'s `loadSteadyStateMap` uses for its float64 sidecar).
  const aligned = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
  const actualWeightsSha256 = sha256Hex(new Uint8Array(aligned));
  if (actualWeightsSha256 !== entry.weightsSha256) {
    throw new Error(
      `shared: archived readout "${entry.id}"'s decoded theta sha256 ${actualWeightsSha256} does not match its ` +
        `recorded weightsSha256 ${entry.weightsSha256} (corrupt or tampered archive entry?)`
    );
  }
  const theta = new Float32Array(aligned);
  return readoutFromFlat(theta, entry.D, entry.H);
};

/** Resolve one archived readout's `ConnectomeGraph`, per `resolve-graph.ts`'s contract. */
export const graphForEntry = (
  entry: Readonly<ArchivedReadout>,
  config: Readonly<ResolveGraphConfig>
): ConnectomeGraph => resolveReadoutGraph(entry, config);

/**
 * Minimal shared `--flag value` table-driven parser for the WP2 analysis
 * CLIs (`saliency.ts`/`ablate.ts`/`independence.ts`/`regime.ts`/
 * `hypotheses.ts`). `scripts/training/cli.ts`'s own doc comment records a
 * deliberate decision AGAINST a generic flag-table parser for its own three
 * callers ("would add indirection without buying clarity") -- that
 * reasoning was written for three CLIs with heterogeneous, ad hoc flag
 * sets; these five WP2 CLIs instead share a near-identical flag set
 * (archive/manifest/intervention-index/out, plus one or two numeric flags),
 * which is exactly the case a table-driven parser earns its keep for. Every
 * flag takes exactly one RAW value string, handed to `handler` unresolved
 * -- a path-valued flag's own handler calls `resolvePathFlag` (below)
 * itself; a numeric flag's handler calls `cli.ts`'s own
 * `requirePositiveInt`/`requireNonNegativeInt` (never bare `Number(...)`,
 * which would accept `NaN`/non-integers silently) -- so validation still
 * lives in the one place `cli.ts` already owns it, not reimplemented here.
 * Unrecognized flags throw with `toolName` in the message. Kept
 * intentionally tiny (no repeat flags, no boolean flags) -- every one of
 * these CLIs' own flag sets fits this shape.
 */
export const parseFlags = (
  toolName: string,
  argv: readonly string[],
  handlers: Readonly<Record<string, (rawValue: string) => void>>
): void => {
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    const handler = handlers[flag];
    if (!handler) throw new Error(`${toolName}: unrecognized argument "${flag}"`);
    const value = argv[i + 1];
    if (value === undefined) throw new Error(`${toolName}: ${flag} requires a value`);
    handler(value);
    i += 1;
  }
};

/** Resolve a raw CLI value against `process.cwd()` -- call this from a path-flag's own handler passed to `parseFlags`. */
export const resolvePathFlag = (rawValue: string): string => resolve(process.cwd(), rawValue);

/**
 * sha256 of the archive file's own raw bytes -- stamped into every
 * downstream analysis output (`saliency.json`/`independence.json`/
 * `regime.json`/`ablation.json`) and cross-checked on read
 * (`ablate.ts`/`hypotheses.ts`/`linkage.py`), so a stale intermediate left
 * over from a different archive version is a loud, immediate error instead
 * of a silently wrong ablation ranking or hypothesis verdict (a dual-review
 * finding: this pipeline sha-verifies every graph and every theta, but
 * previously had no check tying its OWN intermediate JSON outputs back to
 * the archive that produced them).
 */
export const computeArchiveSha256 = (archivePath: string): string => sha256Hex(readFileSync(archivePath));

/** Throws unless `actual` (an upstream output's own recorded `archiveSha256`) matches `expected` (the current archive's). */
export const assertSameArchive = (toolName: string, inputLabel: string, actual: string | undefined, expected: string): void => {
  if (actual !== expected) {
    throw new Error(
      `${toolName}: ${inputLabel} was produced from a different archive (archiveSha256 ${actual ?? '<missing>'}, ` +
        `expected ${expected}) -- rerun the upstream analysis against the current archive`
    );
  }
};

/** The `ResolveGraphConfig` every analysis CLI builds from its own flags/defaults. */
export const defaultResolveGraphConfig = (args: {
  readonly manifestPath?: string;
  readonly interventionIndexPath?: string;
  readonly archivedInterventionIndexPath?: string;
}): ResolveGraphConfig => ({
  manifestPath: args.manifestPath ?? DEFAULT_MANIFEST_PATH,
  interventionIndexPath: args.interventionIndexPath,
  archivedInterventionIndexPath: args.archivedInterventionIndexPath ?? DEFAULT_ARCHIVED_INTERVENTION_INDEX_PATH
});
