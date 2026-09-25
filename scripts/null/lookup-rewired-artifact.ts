import { fileURLToPath } from 'node:url';

import { readRewireIndex } from './null-evaluate';

/**
 * `scripts/null/train-sample.sh`'s `rewired_artifact_for_seed` previously
 * shelled out to an inline `python3` heredoc that read one seed's
 * `artifact` field straight out of `<graphs-dir>/index.json` with no
 * validation beyond `entry["seed"]`/`entry["artifact"]` dict lookups --
 * `rewire_batch.py`'s `index.json` schema is already declared once,
 * authoritatively, as `RewireIndexSeedEntry`/`RewireIndex` in
 * `null-evaluate.ts`, validated field-by-field by `readRewireIndex`. A
 * malformed/renamed field there throws a clear, specific error; the ad hoc
 * Python lookup would instead either silently misbehave or raise a generic
 * `KeyError` (a thermo-maintainability review finding). This CLI is a thin
 * wrapper so `train-sample.sh` can reuse that one validated schema/reader
 * instead of a second, disconnected implementation of it.
 *
 * Usage: `tsx scripts/null/lookup-rewired-artifact.ts <index.json> <seed>`
 * -- prints the matching entry's `artifact` field to stdout (nothing else)
 * on success, or a one-line error to stderr and exits 1 on failure (index
 * missing/malformed, or no entry for `seed`).
 */

export const lookupRewiredArtifact = (indexPath: string, seed: number): string => {
  const index = readRewireIndex(indexPath);
  const entry = index.seeds.find((candidate) => candidate.seed === seed);
  if (!entry) {
    throw new Error(`lookup-rewired-artifact: seed ${seed} not found in ${indexPath}`);
  }
  return entry.artifact;
};

const main = (): void => {
  const [indexPath, seedArg] = process.argv.slice(2);
  if (!indexPath || seedArg === undefined) {
    // eslint-disable-next-line no-console -- CLI tool: this is its user-facing error output.
    console.error('Usage: lookup-rewired-artifact.ts <index.json> <seed>');
    process.exit(1);
  }
  const seed = Number(seedArg);
  if (!Number.isInteger(seed) || seed < 0) {
    // eslint-disable-next-line no-console -- CLI tool: this is its user-facing error output.
    console.error(`lookup-rewired-artifact: <seed> must be a non-negative integer, got "${seedArg}"`);
    process.exit(1);
  }

  try {
    const artifact = lookupRewiredArtifact(indexPath, seed);
    process.stdout.write(`${artifact}\n`);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    // eslint-disable-next-line no-console -- CLI tool: this is its user-facing error output.
    console.error(message);
    process.exit(1);
  }
};

if (process.argv[1] === fileURLToPath(import.meta.url)) main();
