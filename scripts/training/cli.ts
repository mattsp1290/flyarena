/**
 * Small argv-parsing helpers shared by `scripts/training/{export-traces,export-arms,evaluate}.ts`.
 * Each of those three CLIs still parses its own flags with its own `while`/`for`
 * loop and its own flag `if`/`else if` chain — they have different flag sets
 * and different validation needs, so a generic flag-table parser would add
 * indirection without buying clarity. This module only removes the
 * byte-identical `requireValue` (and friends) that had drifted into three
 * separate copies (round-1's `thermo-correctness` S9 / `graph-provenance`
 * S8-S9, still open at round 2).
 */

/** A missing option value must not silently consume the next flag instead. */
export const requireValue = (flag: string, value: string | undefined): string => {
  if (value === undefined || value.startsWith('--')) {
    throw new Error(`${flag} requires a value`);
  }
  return value;
};

export const requirePositiveInt = (flag: string, value: string | undefined): number => {
  const raw = requireValue(flag, value);
  const parsed = Number(raw);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${flag} must be a positive integer, got "${raw}"`);
  }
  return parsed;
};

/** Like `requirePositiveInt`, but accepts 0 — for seed-like flags, where 0 is a meaningful seed. */
export const requireNonNegativeInt = (flag: string, value: string | undefined): number => {
  const raw = requireValue(flag, value);
  const parsed = Number(raw);
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`${flag} must be a non-negative integer, got "${raw}"`);
  }
  return parsed;
};

/** Any finite real number — for measured/informational values (e.g. `evaluate.ts`'s `--gpu-rerun-*` flags) that may be negative (a signed delta) or non-integer. */
export const requireFloat = (flag: string, value: string | undefined): number => {
  const raw = requireValue(flag, value);
  // `Number('')` (or an all-whitespace string) is `0`, not `NaN` — without
  // this check, an accidentally-empty value would silently record a
  // measurement of exactly 0 (e.g. "zero CUDA rerun drift") instead of
  // failing loudly.
  if (raw.trim() === '') {
    throw new Error(`${flag} must be a finite number, got "${raw}"`);
  }
  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) {
    throw new Error(`${flag} must be a finite number, got "${raw}"`);
  }
  return parsed;
};
