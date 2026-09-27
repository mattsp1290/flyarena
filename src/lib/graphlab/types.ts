/**
 * Wire types for the private real-graph lab (`/api/graph/v1`,
 * `backend/graph_lab/service.py`/`models.py`). Field names and casing match
 * the Python/TS side exactly:
 *
 * - Request bodies use `models.py`'s Pydantic `alias` names (`seedStart`,
 *   `seedCount`, `searchSeed`) -- the wire shape, not the Python attribute
 *   names (`seed_start`, `seed_count`, `search_seed`).
 * - Result shapes are copied from what the job actually prints on
 *   completion: `scripts/graph-lab/entry-lesion.ts`'s `printResult` call
 *   (lesion), `scripts/graph-lab/entry-atlas-reeval.ts`'s (atlas, spread
 *   with `evaluateSearchOnGraph`'s `GraphEvaluationResultOwn` from
 *   `scripts/atlas/publish.ts`), and `engine_swapset.py`'s `run` (swapset,
 *   which extends `scripts/graph-lab/entry-swapset.ts`'s own scoring result
 *   with `controlDistribution`/`candidateRankAmongControls`/
 *   `publishedNullPercentile`). None of the three carry a `bundleSha256`
 *   field of their own -- only `GET /api/graph/v1/health` does -- so the
 *   provenance label combines a captured health snapshot with the result's
 *   own `graph`/`graphSha256`/`host`/`label` fields (see `GraphLab.svelte`).
 */

export interface HealthResponse {
  status: string;
  modelVersion: string;
  bundleSha256: string | null;
  graphSha256: string | null;
  gpu: { available: boolean };
}

export type JobKind = 'lesion' | 'atlas' | 'swapset';

export type JobStatus =
  | 'queued'
  | 'running'
  | 'cancelling'
  | 'cancelled'
  | 'timed-out'
  | 'failed'
  | 'completed';

// -- Requests -----------------------------------------------------------

export interface LesionRequest {
  readonly kind: 'lesion';
  readonly graph: string;
  readonly sets: readonly (readonly number[])[];
  readonly seedStart: number;
  readonly seedCount: number;
  readonly ticks: number;
}

export interface AtlasRequest {
  readonly kind: 'atlas';
  readonly graph: string;
  readonly searchSeed: number;
  readonly population: number;
  readonly generations: number;
  readonly ticks: number;
}

export interface SwapValue {
  readonly a: number;
  readonly b: number;
  readonly c: number;
  readonly d: number;
}

export interface SwapsetRequest {
  readonly kind: 'swapset';
  readonly graph: 'biological';
  readonly swaps: readonly SwapValue[];
  readonly controls: number;
  readonly seedStart: number;
  readonly seedCount: number;
  readonly ticks: number;
}

export type JobRequest = LesionRequest | AtlasRequest | SwapsetRequest;

// -- Results --------------------------------------------------------------

/** Shared by every job kind's result (`02-job-engines.md`: "Every result carries..."). */
export interface ResultProvenance {
  readonly host: { readonly arch: string; readonly node: string };
  readonly label: string;
}

export interface PairedEffect {
  readonly n: number;
  readonly meanDifference: number;
  readonly ci95: readonly [number, number];
}

export interface LesionSetResult {
  readonly indices: readonly number[];
  readonly bodyIds: readonly string[];
  readonly effect: PairedEffect;
  readonly n: number;
}

export interface LesionResult extends ResultProvenance {
  readonly graph: string;
  readonly graphSha256: string;
  readonly baseline: { readonly n: number; readonly mean: number };
  readonly sets: readonly LesionSetResult[];
}

/** `scripts/training/evaluate.ts`'s `Metrics` shape -- a per-seed score summary. Kept loose (not every field is rendered). */
export interface AtlasMetrics {
  readonly movementScore: number;
  readonly [key: string]: unknown;
}

/** `scripts/atlas/publish.ts`'s `GraphCellBase & { heldout: Partial<Record<Control, Metrics[]>> }` (the `heldout: 'own'` variant `entry-atlas-reeval.ts` requests). */
export interface AtlasCellResult {
  readonly id: number;
  readonly cell: number;
  readonly quality: number;
  readonly coverage: number;
  readonly turning: number;
  readonly discovery: readonly AtlasMetrics[];
  readonly heldout: Partial<Record<string, readonly AtlasMetrics[]>>;
  /** Full frame replays are intentionally not rendered here (`02-job-engines.md`: "There are no replays, which keeps results small"). */
  readonly replay: unknown;
}

export interface AtlasResult extends ResultProvenance {
  readonly dataDir: string;
  readonly evaluations: readonly { readonly id: number; readonly discovery: readonly AtlasMetrics[] }[];
  readonly cells: readonly AtlasCellResult[];
  readonly gpuArchiveSize: number;
  readonly collisions: number;
}

export interface SwapsetScoreEntry {
  readonly graphId: string;
  readonly n: number;
  readonly mean: number;
}

export interface SwapsetPairedEntry {
  readonly graphId: string;
  readonly effect: PairedEffect;
}

export interface ControlDistribution {
  readonly n: number;
  readonly scores: readonly number[];
  readonly p5: number;
  readonly p50: number;
  readonly p95: number;
}

export interface RankStatistics {
  readonly kBelow: number;
  readonly kEqual: number;
  readonly bioPercentile: number;
  readonly pLow: number;
  readonly pHigh: number;
}

export interface SwapsetResult extends ResultProvenance {
  readonly scores: readonly SwapsetScoreEntry[];
  readonly baselineGraphId?: string;
  readonly paired: readonly SwapsetPairedEntry[];
  readonly controlDistribution: ControlDistribution | null;
  readonly candidateRankAmongControls: RankStatistics | null;
  readonly publishedNullPercentile: RankStatistics | 'not comparable';
}

export type JobResult = LesionResult | AtlasResult | SwapsetResult;

export interface Job {
  readonly id: string;
  readonly kind: JobKind;
  readonly status: JobStatus;
  readonly progress: Record<string, unknown>;
  readonly result: JobResult | null;
  readonly error: string | null;
}
