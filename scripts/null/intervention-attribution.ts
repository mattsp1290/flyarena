import { readFileSync } from 'node:fs';

/**
 * Parses `scripts/analysis/interventions.py`'s `attribution.json` (WP1) and
 * the per-graph `transfer` block of its sibling `index.json` — the two
 * files this study's report needs for the `k`/`k_Q` swap counts, the
 * predeclared 3x8 transfer table before/after each intervention, and R's
 * applicability. Split out of `intervention-report.ts` (see that file's own
 * "keep it under 1000 lines" note in `04-report-and-ledger.md`) — this
 * module owns parsing only; category decisions stay in `intervention-report.ts`/
 * `intervention-report-trained.ts`.
 *
 * `scripts/analysis/transfer.py`'s own row/column vocabulary
 * (`OUTPUT_POPULATION_INDEX`/`OBSERVATION_CHANNEL_INDEX`), restated here so
 * the report/artifact builder can label the 3x8 tables without a second,
 * drifting copy of these names.
 */
export const TRANSFER_OUTPUT_POPULATIONS = ['thrust', 'yaw', 'brake'] as const;
export const TRANSFER_INPUT_CHANNELS = [
  'foodBearing',
  'foodDistance',
  'hazardBearing',
  'hazardDistance',
  'forwardClearance',
  'leftClearance',
  'rightClearance',
  'speed'
] as const;

/** A 3 (population) x 8 (channel) transfer-sensitivity table, `scripts/analysis/transfer.py`'s own `full3x8` shape. */
export type Transfer3x8 = readonly (readonly number[])[];

const isTransfer3x8 = (value: unknown): value is Transfer3x8 =>
  Array.isArray(value) &&
  value.length === TRANSFER_OUTPUT_POPULATIONS.length &&
  value.every((row) => Array.isArray(row) && row.length === TRANSFER_INPUT_CHANNELS.length && row.every((cell) => typeof cell === 'number' && Number.isFinite(cell)));

export interface AttributionInterventionResult {
  readonly kind: 'P' | 'Q';
  /** Accepted double-edge-swap count (`k` for P, `k_Q` for Q). */
  readonly swaps: number;
  readonly targetReached: boolean;
  readonly stopReason: string;
  readonly finalRightClearanceThrust: number;
  readonly finalForwardClearanceThrust: number;
  readonly candidateSampleSeed: number;
  readonly stepCount: number;
}

export interface AttributionR {
  readonly applicable: boolean;
  readonly edgeCount: number;
  readonly reason: string;
}

export interface AttributionBiological {
  readonly transfer: { readonly full3x8: Transfer3x8; readonly rightClearanceThrust: number; readonly forwardClearanceThrust: number };
  readonly sourceSha256: string;
  readonly explanationCrossCheck: Record<string, { readonly computed: number; readonly published: number }>;
}

export interface AttributionNullRegenerationChannel {
  readonly p25: number;
}

export interface AttributionFile {
  readonly version: number;
  readonly P: AttributionInterventionResult;
  readonly Q: AttributionInterventionResult;
  readonly R: AttributionR;
  readonly biological: AttributionBiological;
  /** WP1's own 25th-percentile *stopping-rule targets* for the two transfer entries the greedy search optimizes toward — a different quantity from `intervention-report.ts`'s `publishedNullFloorValue` (a `movementScore` null floor). Reported as context only. */
  readonly nullRegeneration: {
    readonly rightClearanceThrust: AttributionNullRegenerationChannel;
    readonly forwardClearanceThrust: AttributionNullRegenerationChannel;
  };
  readonly producer: { readonly script: string; readonly sourceSha256: string; readonly dependencies: readonly string[]; readonly host: { readonly arch: string; readonly python: string } };
}

const requireNumber = (value: unknown, label: string): number => {
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error(`intervention-attribution: ${label} is not a finite number`);
  return value;
};

const parseInterventionResult = (value: unknown, kind: 'P' | 'Q', label: string): AttributionInterventionResult => {
  if (typeof value !== 'object' || value === null) throw new Error(`intervention-attribution: ${label}.${kind} is not an object`);
  const v = value as Record<string, unknown>;
  if (typeof v.swaps !== 'number' || !Number.isInteger(v.swaps) || v.swaps < 0) {
    throw new Error(`intervention-attribution: ${label}.${kind}.swaps is not a non-negative integer`);
  }
  if (typeof v.targetReached !== 'boolean') throw new Error(`intervention-attribution: ${label}.${kind}.targetReached is not a boolean`);
  if (typeof v.stopReason !== 'string') throw new Error(`intervention-attribution: ${label}.${kind}.stopReason is not a string`);
  if (!Array.isArray(v.steps)) throw new Error(`intervention-attribution: ${label}.${kind}.steps is not an array`);
  return {
    kind,
    swaps: v.swaps,
    targetReached: v.targetReached,
    stopReason: v.stopReason,
    finalRightClearanceThrust: requireNumber(v.finalRightClearanceThrust, `${label}.${kind}.finalRightClearanceThrust`),
    finalForwardClearanceThrust: requireNumber(v.finalForwardClearanceThrust, `${label}.${kind}.finalForwardClearanceThrust`),
    candidateSampleSeed: requireNumber(v.candidateSampleSeed, `${label}.${kind}.candidateSampleSeed`),
    stepCount: v.steps.length
  };
};

const parseR = (value: unknown, label: string): AttributionR => {
  if (typeof value !== 'object' || value === null) throw new Error(`intervention-attribution: ${label}.R is not an object`);
  const v = value as Record<string, unknown>;
  if (typeof v.applicable !== 'boolean') throw new Error(`intervention-attribution: ${label}.R.applicable is not a boolean`);
  if (typeof v.edgeCount !== 'number' || !Number.isInteger(v.edgeCount) || v.edgeCount < 0) {
    throw new Error(`intervention-attribution: ${label}.R.edgeCount is not a non-negative integer`);
  }
  if (typeof v.reason !== 'string') throw new Error(`intervention-attribution: ${label}.R.reason is not a string`);
  return { applicable: v.applicable, edgeCount: v.edgeCount, reason: v.reason };
};

const parseBiological = (value: unknown, label: string): AttributionBiological => {
  if (typeof value !== 'object' || value === null) throw new Error(`intervention-attribution: ${label}.biological is not an object`);
  const v = value as Record<string, unknown>;
  const transfer = v.transfer as Record<string, unknown> | undefined;
  if (!transfer || !isTransfer3x8(transfer.full3x8)) {
    throw new Error(`intervention-attribution: ${label}.biological.transfer.full3x8 is missing or not a 3x8 table`);
  }
  if (typeof v.sourceSha256 !== 'string') throw new Error(`intervention-attribution: ${label}.biological.sourceSha256 is not a string`);
  const crossCheckRaw = v.explanationCrossCheck;
  if (typeof crossCheckRaw !== 'object' || crossCheckRaw === null) {
    throw new Error(`intervention-attribution: ${label}.biological.explanationCrossCheck is not an object`);
  }
  const explanationCrossCheck: Record<string, { computed: number; published: number }> = {};
  for (const [key, entry] of Object.entries(crossCheckRaw as Record<string, unknown>)) {
    if (typeof entry !== 'object' || entry === null) throw new Error(`intervention-attribution: ${label}.biological.explanationCrossCheck.${key} is not an object`);
    const e = entry as Record<string, unknown>;
    explanationCrossCheck[key] = {
      computed: requireNumber(e.computed, `${label}.biological.explanationCrossCheck.${key}.computed`),
      published: requireNumber(e.published, `${label}.biological.explanationCrossCheck.${key}.published`)
    };
  }
  return {
    transfer: {
      full3x8: transfer.full3x8,
      rightClearanceThrust: requireNumber(transfer.rightClearanceThrust, `${label}.biological.transfer.rightClearanceThrust`),
      forwardClearanceThrust: requireNumber(transfer.forwardClearanceThrust, `${label}.biological.transfer.forwardClearanceThrust`)
    },
    sourceSha256: v.sourceSha256,
    explanationCrossCheck
  };
};

const parseNullRegenerationChannel = (value: unknown, label: string): AttributionNullRegenerationChannel => {
  if (typeof value !== 'object' || value === null) throw new Error(`intervention-attribution: ${label} is not an object`);
  const v = value as Record<string, unknown>;
  return { p25: requireNumber(v.p25, `${label}.p25`) };
};

export const parseAttribution = (text: string, label: string): AttributionFile => {
  const parsed = JSON.parse(text) as Record<string, unknown>;
  if (typeof parsed.version !== 'number') throw new Error(`intervention-attribution: ${label} is missing a numeric version`);
  const producer = parsed.producer as Record<string, unknown> | undefined;
  if (
    !producer ||
    typeof producer.script !== 'string' ||
    typeof producer.sourceSha256 !== 'string' ||
    !Array.isArray(producer.dependencies)
  ) {
    throw new Error(`intervention-attribution: ${label}.producer is missing script/sourceSha256/dependencies`);
  }
  const host = producer.host as Record<string, unknown> | undefined;
  if (!host || typeof host.arch !== 'string' || typeof host.python !== 'string') {
    throw new Error(`intervention-attribution: ${label}.producer.host is missing arch/python`);
  }
  const nullRegeneration = parsed.nullRegeneration as Record<string, unknown> | undefined;
  if (!nullRegeneration) throw new Error(`intervention-attribution: ${label} is missing nullRegeneration`);

  return {
    version: parsed.version,
    P: parseInterventionResult(parsed.P, 'P', label),
    Q: parseInterventionResult(parsed.Q, 'Q', label),
    R: parseR(parsed.R, label),
    biological: parseBiological(parsed.biological, label),
    nullRegeneration: {
      rightClearanceThrust: parseNullRegenerationChannel(nullRegeneration.rightClearanceThrust, `${label}.nullRegeneration.rightClearanceThrust`),
      forwardClearanceThrust: parseNullRegenerationChannel(nullRegeneration.forwardClearanceThrust, `${label}.nullRegeneration.forwardClearanceThrust`)
    },
    producer: {
      script: producer.script,
      sourceSha256: producer.sourceSha256,
      dependencies: producer.dependencies as readonly string[],
      host: { arch: host.arch, python: host.python }
    }
  };
};

export const readAttribution = (path: string): AttributionFile => parseAttribution(readFileSync(path, 'utf8'), path);

/**
 * `scripts/analysis/interventions.py`'s `index.json` entries each carry
 * their own post-intervention `transfer.full3x8` (the "after" table) —
 * `attribution.json` itself has no full 3x8 table for P/Q, only their final
 * scalar `T:rightClearance->thrust`/`T:forwardClearance->thrust` values (see
 * `AttributionInterventionResult`). This reads `index.json`'s raw entries
 * (not `intervention-report.ts`'s `readGraphListIndexInfo`, which only
 * extracts `kind`/`gzipSha256`) to get the one requested id's `transfer`
 * table.
 */
export const parseIndexGraphTransfer = (text: string, label: string, id: string): { readonly swaps: number; readonly transfer: Transfer3x8 } => {
  const parsed = JSON.parse(text) as { entries?: readonly Record<string, unknown>[] };
  const entry = (parsed.entries ?? []).find((e) => e.id === id);
  if (!entry) throw new Error(`intervention-attribution: ${label} has no entry with id "${id}"`);
  const transfer = entry.transfer as Record<string, unknown> | undefined;
  if (!transfer || !isTransfer3x8(transfer.full3x8)) {
    throw new Error(`intervention-attribution: ${label} entry "${id}" is missing a 3x8 transfer.full3x8 table`);
  }
  if (typeof entry.swaps !== 'number') throw new Error(`intervention-attribution: ${label} entry "${id}" is missing swaps`);
  return { swaps: entry.swaps, transfer: transfer.full3x8 };
};

/** Reads and parses `indexPath` -- see `parseIndexGraphTransfer`'s doc comment / `parseAttribution`'s own for why callers that already have the bytes in hand (`intervention-artifact.ts`) should prefer parsing them directly instead of a second `readFileSync`. */
export const readIndexGraphTransfer = (indexPath: string, id: string): { readonly swaps: number; readonly transfer: Transfer3x8 } =>
  parseIndexGraphTransfer(readFileSync(indexPath, 'utf8'), indexPath, id);
