/**
 * Pure, DOM-free validators mirroring `backend/graph_lab/models.py`'s bounds
 * exactly (`.agents/plans/graph-lab/03-frontend-route.md`: "Pure validators
 * mirroring the backend bounds"). These are a *client-side* convenience --
 * cheap, immediate feedback before a submit round-trip -- never the source
 * of truth: the backend re-validates every field independently
 * (`models.py`'s own `extra="forbid"` + `strict=True` models), so a bug here
 * can only ever make the form *stricter* than the server, never looser.
 *
 * Every bound below is a literal copy of the corresponding `Field(...)` in
 * `models.py`; when it changes there, it must change here too. See that
 * file's own field-by-field bounds for the source of truth this mirrors.
 */

/** `models.py`'s `NEURON_COUNT` (`public/data/malecns-arena-v1.manifest.json`'s `neuronCount`). */
export const NEURON_COUNT = 1008;
/** `models.py`'s `MAX_REWIRED_SEED`. */
export const MAX_REWIRED_SEED = 499;
/** `models.py`'s `MAX_SEED` (`2**32 - 1`). */
export const MAX_SEED = 2 ** 32 - 1;
/** `models.py`'s `AtlasJobRequest.search_seed` bound (`2**31 - 1`). */
export const MAX_SEARCH_SEED = 2 ** 31 - 1;
/** `service.py`'s `MIN_TOKEN_LENGTH` -- the shortest token the backend will accept at startup. */
export const MIN_TOKEN_LENGTH = 16;

/**
 * `models.py`'s `GRAPH_PATTERN`: `\A(biological|disconnected|rewired:(0|[1-9]\d{0,2}))\Z`.
 * JavaScript's `^`/`$` (no `m` flag) already anchor to the absolute start
 * and end of the string -- unlike Python's bare `$`, which also matches
 * immediately before a single trailing "\n" (the reason `models.py` uses
 * `\A`/`\Z` instead) -- so a plain `^...$` here is already exact. JS's `\d`
 * is also already ASCII-only (no Unicode decimal-digit matching without an
 * explicit `\p{Nd}` Unicode property, which this pattern never uses), so no
 * analog of Python's `re.ASCII` flag is needed either.
 */
const GRAPH_PATTERN = /^(biological|disconnected|rewired:(0|[1-9]\d{0,2}))$/;

export type GraphId = string;

/** Mirrors `models.py`'s `validate_graph_id`. Returns an error message, or `null` if `value` is valid. */
export function validateGraphId(value: string): string | null {
  const match = GRAPH_PATTERN.exec(value);
  if (!match) return 'graph must be "biological", "disconnected", or "rewired:<seed>"';
  const seedText = match[2];
  if (seedText !== undefined && Number(seedText) > MAX_REWIRED_SEED) {
    return `rewired seed must be <= ${MAX_REWIRED_SEED}`;
  }
  return null;
}

const inRange = (value: number, min: number, max: number): boolean =>
  Number.isInteger(value) && value >= min && value <= max;

/** Mirrors `models.py`'s `_check_seed_range`: the highest seed a request will actually produce
 * (`seedStart + seedCount - 1`) must itself stay within `MAX_SEED`. */
export function validateSeedRange(seedStart: number, seedCount: number): string | null {
  if (seedStart + seedCount - 1 > MAX_SEED) {
    return `seedStart + seedCount - 1 (${seedStart + seedCount - 1}) exceeds the maximum seed ${MAX_SEED}`;
  }
  return null;
}

/** Mirrors `models.py`'s `_validate_unique_indices`. */
export function validateUniqueIndices(
  indices: readonly number[],
  { minLen, maxLen, label }: { minLen: number; maxLen: number; label: string }
): string[] {
  const errors: string[] = [];
  if (indices.length < minLen || indices.length > maxLen) {
    errors.push(`${label} must have ${minLen}-${maxLen} entries`);
  }
  if (new Set(indices).size !== indices.length) {
    errors.push(`${label} entries must be unique`);
  }
  for (const index of indices) {
    if (!inRange(index, 0, NEURON_COUNT - 1)) {
      errors.push(`${label} entry out of range [0, ${NEURON_COUNT})`);
      break;
    }
  }
  return errors;
}

export interface LesionFormValues {
  readonly graph: string;
  readonly sets: readonly (readonly number[])[];
  readonly seedStart: number;
  readonly seedCount: number;
  readonly ticks: number;
}

/** Mirrors `models.py`'s `LesionJobRequest` bounds. */
export function validateLesion(values: LesionFormValues): string[] {
  const errors: string[] = [];
  const graphError = validateGraphId(values.graph);
  if (graphError) errors.push(graphError);
  if (values.sets.length < 1 || values.sets.length > 32) {
    errors.push('sets must have 1-32 entries');
  }
  values.sets.forEach((set, index) => {
    errors.push(
      ...validateUniqueIndices(set, { minLen: 1, maxLen: 64, label: `set ${index + 1}` })
    );
  });
  if (!inRange(values.seedStart, 0, MAX_SEED)) errors.push(`seedStart must be an integer in [0, ${MAX_SEED}]`);
  if (!inRange(values.seedCount, 4, 100)) errors.push('seedCount must be an integer in [4, 100]');
  if (!inRange(values.ticks, 300, 1800)) errors.push('ticks must be an integer in [300, 1800]');
  const seedRangeError = validateSeedRange(values.seedStart, values.seedCount);
  if (seedRangeError) errors.push(seedRangeError);
  return errors;
}

export interface AtlasFormValues {
  readonly graph: string;
  readonly searchSeed: number;
  readonly population: number;
  readonly generations: number;
  readonly ticks: number;
}

/** Mirrors `models.py`'s `AtlasJobRequest` bounds. */
export function validateAtlas(values: AtlasFormValues): string[] {
  const errors: string[] = [];
  const graphError = validateGraphId(values.graph);
  if (graphError) errors.push(graphError);
  if (!inRange(values.searchSeed, 0, MAX_SEARCH_SEED)) errors.push(`searchSeed must be an integer in [0, ${MAX_SEARCH_SEED}]`);
  if (!inRange(values.population, 4, 64)) errors.push('population must be an integer in [4, 64]');
  if (!inRange(values.generations, 1, 48)) errors.push('generations must be an integer in [1, 48]');
  if (!inRange(values.ticks, 300, 900)) errors.push('ticks must be an integer in [300, 900]');
  return errors;
}

export interface SwapFormValue {
  readonly a: number;
  readonly b: number;
  readonly c: number;
  readonly d: number;
}

export interface SwapsetFormValues {
  readonly swaps: readonly SwapFormValue[];
  readonly controls: number;
  readonly seedStart: number;
  readonly seedCount: number;
  readonly ticks: number;
}

/** Mirrors `models.py`'s `SwapsetJobRequest`/`Swap` bounds. `graph` is always `"biological"` (the model's own default; not user-chosen). */
export function validateSwapset(values: SwapsetFormValues): string[] {
  const errors: string[] = [];
  if (values.swaps.length < 1 || values.swaps.length > 50) {
    errors.push('swaps must have 1-50 entries');
  }
  values.swaps.forEach((swap, index) => {
    for (const [key, value] of Object.entries(swap) as [string, number][]) {
      if (!inRange(value, 0, NEURON_COUNT - 1)) {
        errors.push(`swap ${index + 1} field "${key}" must be an integer in [0, ${NEURON_COUNT})`);
      }
    }
  });
  if (!inRange(values.controls, 0, 100)) errors.push('controls must be an integer in [0, 100]');
  if (!inRange(values.seedStart, 0, MAX_SEED)) errors.push(`seedStart must be an integer in [0, ${MAX_SEED}]`);
  if (!inRange(values.seedCount, 4, 100)) errors.push('seedCount must be an integer in [4, 100]');
  if (!inRange(values.ticks, 300, 1800)) errors.push('ticks must be an integer in [300, 1800]');
  const seedRangeError = validateSeedRange(values.seedStart, values.seedCount);
  if (seedRangeError) errors.push(seedRangeError);
  return errors;
}

/**
 * Parses a free-text, comma/whitespace-separated list of neurons into
 * sorted, unique zero-based indices -- "a parser for neuron sets by index or
 * body id, using the loaded positions or graph for the mapping"
 * (`03-frontend-route.md`). A token is read as a raw index when it parses as
 * an integer within `[0, NEURON_COUNT)`; otherwise it is looked up verbatim
 * in `bodyIds` (`public/data/malecns-arena-v1.positions.json`'s own
 * `bodyIds` array, index-aligned with the graph -- every real body id in
 * that file is a 5+ digit MaleCNS identifier, so it can never collide with a
 * valid `[0, 1008)` index). Throws with a message naming the offending
 * token, rather than silently dropping it, so a typo is never scored as
 * "no neurons selected" instead of a form error.
 */
export function parseNeuronToken(token: string, bodyIds?: readonly string[]): number {
  const trimmed = token.trim();
  if (/^\d+$/.test(trimmed)) {
    const index = Number(trimmed);
    if (inRange(index, 0, NEURON_COUNT - 1)) return index;
  }
  if (bodyIds) {
    const index = bodyIds.indexOf(trimmed);
    if (index !== -1) return index;
  }
  throw new Error(`"${trimmed}" is not a recognized neuron index or body id`);
}

export function parseNeuronList(input: string, bodyIds?: readonly string[]): number[] {
  const tokens = input
    .split(/[\s,]+/)
    .map((token) => token.trim())
    .filter((token) => token.length > 0);
  const indices = tokens.map((token) => parseNeuronToken(token, bodyIds));
  return Array.from(new Set(indices)).sort((a, b) => a - b);
}
