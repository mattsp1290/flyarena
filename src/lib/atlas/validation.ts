import { ARENA_CONFIG, createArenaConfigFingerprint } from '../arena/config';
import { readoutFromFlat } from '../connectome/readout-serialization';
import {
  ATLAS_VERSION,
  COVERAGE_EDGES,
  TURN_EDGES,
  DISCOVERY_SEEDS,
  HELDOUT_SEEDS,
  CONTROL_NAMES,
  cellFor,
  type Atlas,
  type SearchArtifact
} from './types';

export const object = (v: unknown): Record<string, unknown> => {
  if (!v || typeof v !== 'object' || Array.isArray(v)) throw new Error('Expected atlas object');
  return v as Record<string, unknown>;
};
const array = (v: unknown, min: number, max = min): unknown[] => {
  if (!Array.isArray(v) || v.length < min || v.length > max)
    throw new Error('Invalid atlas array length');
  return v;
};
const number = (v: unknown, min = -1e9, max = 1e9): number => {
  if (typeof v !== 'number' || !Number.isFinite(v) || v < min || v > max)
    throw new Error('Invalid atlas number');
  return v;
};
const integer = (v: unknown, min: number, max: number): number => {
  const n = number(v, min, max);
  if (!Number.isInteger(n)) throw new Error('Expected atlas integer');
  return n;
};
export const hash = (v: unknown): string => {
  if (typeof v !== 'string' || !/^[a-f0-9]{64}$/.test(v)) throw new Error('Invalid atlas hash');
  return v;
};
const equal = (a: unknown, b: unknown) => {
  if (JSON.stringify(a) !== JSON.stringify(b)) throw new Error('Atlas contract mismatch');
};

/** Bound arbitrary provenance fields as well as the typed public fields. */
function finiteTree(value: unknown, depth = 0, budget = { leaves: 0 }): void {
  if (depth > 16 || ++budget.leaves > 500_000) throw new Error('Atlas structure exceeds bounds');
  if (typeof value === 'number' && !Number.isFinite(value))
    throw new Error('Nonfinite atlas value');
  if (value && typeof value === 'object')
    for (const child of Object.values(value)) finiteTree(child, depth + 1, budget);
}

export function validateSearch(input: unknown): SearchArtifact {
  finiteTree(input);
  const v = object(input),
    options = object(v.options);
  equal(v.schemaVersion, 1);
  equal(v.modelVersion, ATLAS_VERSION);
  integer(options.seed, 0, 2 ** 31 - 1);
  integer(options.population, 4, 128);
  integer(options.generations, 1, 96);
  integer(options.ticks, 30, 1800);
  const d = integer(v.inputSize, 1, 4096);
  equal(v.hiddenSize, 8);
  equal(v.substeps, 4);
  equal(v.discoverySeeds, DISCOVERY_SEEDS);
  equal(v.heldoutSeeds, HELDOUT_SEEDS);
  equal(v.coverageEdges, COVERAGE_EDGES);
  equal(v.turnEdges, TURN_EDGES);
  hash(v.graphArtifactSha256);
  hash(v.bundleSha256);
  object(v.bundle);
  const ids = new Set<number>();
  for (const entry of array(v.candidates, 1, 36)) {
    const c = object(entry),
      id = integer(c.id, 0, Number(options.population) * Number(options.generations) - 1);
    if (ids.has(id)) throw new Error('Duplicate atlas controller');
    ids.add(id);
    const theta = array(c.theta, 8 * d + 8 + 24 + 3).map((x) => number(x, -8, 8));
    readoutFromFlat(theta, d, 8);
    number(c.quality);
    cellFor(number(c.coverage, 0, 1), number(c.turning, -1, 1));
  }
  array(v.history, Number(options.generations)).forEach((entry, i) => {
    const row = object(entry);
    equal(row.generation, i + 1);
    integer(row.occupied, 1, 36);
    number(row.bestQuality);
  });
  equal(v.searchPolicy, {
    initialStd: 0.5,
    mutationScales: [0.05, 0.15, 0.4],
    freshFraction: 0.25,
    weightBound: 8,
    ties: 'earlier candidate',
    rng: 'torch CPU Generator'
  });
  const runtime = object(v.runtime);
  if (runtime.device !== 'cpu' && runtime.device !== 'cuda')
    throw new Error('Unknown atlas compute device');
  for (const key of ['deviceName', 'torch'])
    if (typeof runtime[key] !== 'string') throw new Error('Missing runtime identity');
  if (runtime.cuda !== null && typeof runtime.cuda !== 'string')
    throw new Error('Invalid CUDA version');
  number(runtime.seconds, 0, 86400);
  integer(runtime.peakTensorBytes, 0, 1024 ** 3);
  const config = object(runtime.config);
  for (const [key, value] of Object.entries(ARENA_CONFIG))
    equal(config[key.replace(/[A-Z]/g, (c) => '_' + c.toLowerCase())], value);
  return v as unknown as SearchArtifact;
}

function metrics(value: unknown) {
  const m = object(value);
  number(m.movementScore);
  number(m.distanceTravelled, 0);
  integer(m.foodPickups, 0, 1800);
  integer(m.hazardContacts, 0, 1800);
  number(m.coverage, 0, 1);
  number(m.turning, -1, 1);
}
function frames(value: unknown, ticks: number) {
  const entries = array(value, Math.min(60, ticks) + 1);
  entries.forEach((entry, i) => {
    const frame = object(entry),
      snapshot = object(frame.snapshot);
    equal(snapshot.tick, Math.round((i * ticks) / (entries.length - 1)));
    number(snapshot.timeSeconds, 0, ticks / 30 + 1e-9);
    for (const [key, count] of [
      ['agents', 2],
      ['foods', 4],
      ['hazards', 2]
    ] as const) {
      array(snapshot[key], count).forEach((entity) => {
        const e = object(entity),
          p = object(e.position);
        if (typeof e.id !== 'string') throw new Error('Missing entity identity');
        number(p.x, -12, 12);
        number(p.z, -8, 8);
        if (key !== 'agents') number(e.radius, 0, 1);
        if (key === 'agents') number(e.heading, -Math.PI, Math.PI);
      });
    }
    const scores = object(frame.scores);
    for (const key of ['left', 'right']) {
      const s = object(scores[key]);
      for (const field of ['foodPickups', 'hazardContacts', 'distanceTravelled', 'movementScore'])
        number(s[field]);
    }
  });
}

export function validateAtlas(input: unknown): Atlas {
  finiteTree(input);
  const v = object(input),
    source = validateSearch(v.source);
  equal(v.schemaVersion, 1);
  equal(v.modelVersion, ATLAS_VERSION);
  hash(v.graphBinarySha256);
  equal(v.configFingerprint, createArenaConfigFingerprint(ARENA_CONFIG));
  let previous = -1;
  array(v.outputNeuronIndices, source.inputSize).forEach((x) => {
    const n = integer(x, 0, 100_000);
    if (n <= previous) throw new Error('Unordered output neurons');
    previous = n;
  });
  const candidateIds = new Set(source.candidates.map((c) => c.id)),
    seen = new Set<number>(),
    cells = new Set<number>();
  const evaluations = array(v.evaluations, candidateIds.size),
    evaluated = new Set<number>();
  for (const entry of evaluations) {
    const e = object(entry),
      id = integer(e.id, 0, 12287);
    if (!candidateIds.has(id) || evaluated.has(id)) throw new Error('Invalid candidate evaluation');
    evaluated.add(id);
    array(e.discovery, 8).forEach(metrics);
  }
  for (const entry of array(v.cells, 1, 36)) {
    const c = object(entry),
      id = integer(c.id, 0, 12287),
      cell = integer(c.cell, 0, 35);
    if (!candidateIds.has(id) || seen.has(id) || cells.has(cell))
      throw new Error('Duplicate or unknown atlas cell');
    seen.add(id);
    cells.add(cell);
    equal(cell, cellFor(number(c.coverage, 0, 1), number(c.turning, -1, 1)));
    number(c.quality);
    array(c.discovery, 8).forEach(metrics);
    const heldout = object(c.heldout);
    for (const name of CONTROL_NAMES) array(heldout[name], 12).forEach(metrics);
    frames(c.replay, source.options.ticks);
  }
  array(v.authored, 12).forEach(metrics);
  return v as unknown as Atlas;
}
