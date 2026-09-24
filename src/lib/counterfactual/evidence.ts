import { MODEL_VERSION, validateRequest } from './types';

const record = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value);

/** Validate bounded instructions first; the verifier compares all remaining evidence to a fresh run. */
export function readEvidenceRequest(document: unknown) {
  if (!record(document) || !record(document.evidence)) throw new Error('Expected a counterfactual export');
  const evidence = document.evidence;
  if (evidence.schemaVersion !== 1 || evidence.modelVersion !== MODEL_VERSION) throw new Error('Unsupported evidence version');
  return { request: validateRequest(evidence.request), evidence };
}

export function serializeEvidence(document: unknown): string {
  return JSON.stringify(document, (_key, value: unknown) => {
    if (typeof value === 'number' && !Number.isFinite(value)) throw new Error('Evidence contains nonfinite numbers');
    return value;
  }, 2);
}

export interface Comparison {
  exact: boolean;
  matches: boolean;
  inexactLeaves: number;
  maxAbsoluteError: number;
  maxRelativeError: number;
  mismatches: string[];
}

// Numeric tolerance applies only to continuous result fields, never requests or identities.
function continuous(path: string): boolean {
  if (!path.startsWith('evidence.results[') && !path.startsWith('evidence.summary.')) return false;
  return /\.(distanceTravelled|movementScore|timeSeconds|heading|radius|mean|low|high)$/.test(path)
    || /\.position\.(x|z)$/.test(path);
}

/** Exact structure/discrete fields; explicitly optional conservative numerical diagnostic. */
export function compareEvidence(expected: unknown, actual: unknown, numerical = false): Comparison {
  const result: Comparison = { exact: true, matches: true, inexactLeaves: 0, maxAbsoluteError: 0, maxRelativeError: 0, mismatches: [] };
  const fail = (path: string) => {
    result.matches = false;
    if (result.mismatches.length < 20) result.mismatches.push(path);
  };
  function walk(a: unknown, b: unknown, path: string, depth: number) {
    if (depth > 32) { fail(`${path}: nesting too deep`); return; }
    if (typeof a === 'number' && typeof b === 'number') {
      if (!Number.isFinite(a) || !Number.isFinite(b)) { fail(`${path}: nonfinite value`); return; }
      if (a === b) return;
      result.exact = false;
      const abs = Math.abs(a - b), rel = abs / Math.max(Math.abs(a), Math.abs(b), Number.MIN_VALUE);
      result.maxAbsoluteError = Math.max(result.maxAbsoluteError, abs);
      result.maxRelativeError = Math.max(result.maxRelativeError, rel);
      if (abs > 1e-13 && rel > 1e-13) result.inexactLeaves++;
      if (!numerical || !continuous(path) || (abs > 1e-6 && rel > 1e-6)) fail(`${path}: numeric mismatch`);
      return;
    }
    if (Array.isArray(a) && Array.isArray(b)) {
      if (a.length !== b.length) { fail(`${path}: array length mismatch`); return; }
      a.forEach((v, i) => walk(v, b[i], `${path}[${i}]`, depth + 1));
      return;
    }
    if (record(a) && record(b)) {
      const keys = Object.keys(a).sort();
      if (keys.join('\0') !== Object.keys(b).sort().join('\0')) { fail(`${path}: object keys mismatch`); return; }
      for (const key of keys) walk(a[key], b[key], `${path}.${key}`, depth + 1);
      return;
    }
    if (a !== b) fail(`${path}: value mismatch`);
  }
  walk(expected, actual, 'evidence', 0);
  if (numerical && result.inexactLeaves > 8) fail('Numerical differences exceed the eight-leaf diagnostic budget');
  if (!result.matches) result.exact = false;
  return result;
}
