/**
 * `public/data/lesion-atlas-v1.json` (WP3 of `.agents/plans/lesion-atlas`,
 * `03-activity-lesion-mode.md`): the offline-computed single-neuron lesion
 * atlas the activity view's "Lesion effect (offline)" color mode reads from.
 * The real producer's full type is `scripts/lesion/atlas-report.ts`'s
 * `LesionAtlasArtifact` — that module is a Node-only pipeline (not part of
 * the browser bundle), so this file independently authors and validates just
 * the subset the browser actually renders, the same "reimplemented, not
 * imported" discipline `./rewiringNull.ts`'s own doc comment explains for
 * its `RewiringNullArtifact`. `summary`/`condition`/`seeds`/`ticks`/etc. are
 * real fields on the shipped artifact (see `docs/lesion-atlas-report.md`)
 * but are never read by this WP's rendering path, so they are deliberately
 * left untyped/unvalidated here — mirroring `RewiringNullArtifact.trained`'s
 * own precedent for an artifact section this loader does not consume.
 *
 * Split into its own module rather than added to `./assets.ts` (thermo
 * review precedent: `./rewiringNull.ts` was split out for the same "god
 * module" reason) — depends on nothing from `assets.ts` except the shared
 * `ArenaManifest` type and the `fetchAndVerifySidecarJson` fetch->sha256-
 * verify->JSON.parse helper it shares with `loadPositions`/`loadRewiringNull`.
 */

import type { ConnectomeGraph, GraphMode } from '../connectome/format';
import { fetchAndVerifySidecarJson, type ArenaManifest } from './assets';
import { isFiniteNumber } from './rewiringNull';

/** One graph's per-neuron lesion data — the fields this WP's color mode actually consumes. */
export interface LesionAtlasGraphData {
  /**
   * The *decompressed binary* sha256 of the graph this section was computed
   * against (`manifest.binarySha256` for `biological`,
   * `manifest.rewiredArms.seed0.binarySha256` for `rewiredSeed0`) — per
   * `scripts/lesion/atlas-report.ts`'s own `AtlasGraphArtifact.graphSha256`
   * doc comment ("WP3's loader should compare this against the graph it
   * parsed, the same field"). Deliberately *not* the gzip sha
   * (`ArenaManifest.gzipSha256`) that `loadPositions`' `graphSha256` cross-
   * check uses — a different field on a different artifact.
   */
  readonly graphSha256: string;
  readonly baseline: number;
  /** Length `neuronCount`; paired mean effect on `movementScore` when that neuron is silenced for the whole episode. */
  readonly effect: readonly number[];
  readonly ciLow: readonly number[];
  readonly ciHigh: readonly number[];
  /** Length `neuronCount`; Benjamini-Hochberg FDR significance at `q = 0.05` (per graph, 1008 simultaneous tests). */
  readonly fdrSignificant: readonly boolean[];
}

export interface LesionAtlasArtifact {
  readonly version: 1;
  readonly neuronCount: number;
  /** Length `neuronCount`, MaleCNS body IDs as decimal strings, in the same order as the compiled graph's `biologicalIds` — cross-checked element-wise by `loadLesionAtlas` below. */
  readonly bodyIds: readonly string[];
  readonly graphs: {
    readonly biological: LesionAtlasGraphData;
    readonly rewiredSeed0: LesionAtlasGraphData;
  };
}

/** The two graph keys the shipped atlas covers — never a plain rewired arm at another seed (a stated non-goal; see `.agents/plans/lesion-atlas/00-overview.md`'s "Scope and non-goals"). */
export type LesionAtlasGraphKey = 'biological' | 'rewiredSeed0';

/**
 * Maps an arm's *current* topology (`GraphMode`, `connectome/format.ts`) to
 * the lesion atlas's matching graph key. `'rewired'` always means the one
 * shipped rewired arm (`ArenaManifest.rewiredArms.seed0` — the app offers no
 * UI to select a different rewired seed; see `App.svelte`'s `GraphMode`
 * topology selector), so it always maps to `'rewiredSeed0'`, never any other
 * seed. `'disconnected'` has no atlas coverage at all (a stated non-goal:
 * the atlas covers exactly the two shipped topologies) and maps to
 * `undefined` — callers (`ActivityPanel.svelte`) show an honest "no lesion
 * data" state for that arm instead of a lookup failure.
 */
export const lesionAtlasGraphKeyForTopology = (mode: GraphMode): LesionAtlasGraphKey | undefined => {
  if (mode === 'biological') return 'biological';
  if (mode === 'rewired') return 'rewiredSeed0';
  return undefined;
};

const isGraphData = (value: unknown, neuronCount: number): value is LesionAtlasGraphData => {
  if (typeof value !== 'object' || value === null) return false;
  const v = value as Record<string, unknown>;
  if (typeof v.graphSha256 !== 'string') return false;
  if (!isFiniteNumber(v.baseline)) return false;
  if (!Array.isArray(v.effect) || v.effect.length !== neuronCount || !v.effect.every(isFiniteNumber)) return false;
  if (!Array.isArray(v.ciLow) || v.ciLow.length !== neuronCount || !v.ciLow.every(isFiniteNumber)) return false;
  if (!Array.isArray(v.ciHigh) || v.ciHigh.length !== neuronCount || !v.ciHigh.every(isFiniteNumber)) return false;
  const ciLow = v.ciLow as number[];
  const ciHigh = v.ciHigh as number[];
  for (let index = 0; index < neuronCount; index += 1) {
    // A confidence interval whose bounds are swapped is itself a sign of a
    // producer bug, not a real interval (same discipline as
    // `rewiringNull.ts#isFiniteCiPair`).
    if (ciLow[index] > ciHigh[index]) return false;
  }
  if (
    !Array.isArray(v.fdrSignificant) ||
    v.fdrSignificant.length !== neuronCount ||
    !v.fdrSignificant.every((entry) => typeof entry === 'boolean')
  ) {
    return false;
  }
  return true;
};

const validateLesionAtlasShape = (
  value: unknown
): { ok: true; data: LesionAtlasArtifact } | { ok: false; reason: string } => {
  if (typeof value !== 'object' || value === null) {
    return { ok: false, reason: 'lesion-atlas artifact is not a JSON object' };
  }
  const v = value as Record<string, unknown>;
  if (v.version !== 1) {
    return { ok: false, reason: `lesion-atlas artifact has unsupported version ${String(v.version)}` };
  }
  if (!Number.isInteger(v.neuronCount) || (v.neuronCount as number) <= 0) {
    return { ok: false, reason: 'lesion-atlas artifact has a malformed neuronCount' };
  }
  const neuronCount = v.neuronCount as number;
  if (!Array.isArray(v.bodyIds) || v.bodyIds.length !== neuronCount || !v.bodyIds.every((id) => typeof id === 'string')) {
    return { ok: false, reason: 'lesion-atlas artifact has a malformed bodyIds array' };
  }
  if (typeof v.graphs !== 'object' || v.graphs === null) {
    return { ok: false, reason: 'lesion-atlas artifact is missing "graphs"' };
  }
  const graphs = v.graphs as Record<string, unknown>;
  if (!isGraphData(graphs.biological, neuronCount)) {
    return { ok: false, reason: 'lesion-atlas artifact has a malformed graphs.biological section' };
  }
  if (!isGraphData(graphs.rewiredSeed0, neuronCount)) {
    return { ok: false, reason: 'lesion-atlas artifact has a malformed graphs.rewiredSeed0 section' };
  }
  return { ok: true, data: value as LesionAtlasArtifact };
};

/**
 * `status: 'ok'` also carries `absMax`, the diverging colormap's symmetric
 * range (`.agents/plans/lesion-atlas/00-overview.md`'s "Colors are computed
 * once when the mode is entered" decision): `max(|effect|)` over *both*
 * graphs' `effect` arrays together, computed once at load time so the two
 * graphs — and, within a graph, biological vs. rewired-seed-0 — are always
 * shown on one comparable scale rather than each auto-normalizing to its own
 * range.
 */
export type LesionAtlasLoadResult =
  | { status: 'ok'; data: LesionAtlasArtifact; absMax: number }
  /** No `manifest.lesionAtlas` entry at all — nothing was ever shipped. Never retried (there is nothing to retry). */
  | { status: 'missing'; reason: string }
  /**
   * A fetch/network failure — not a claim about the artifact's integrity.
   * Distinct from `'missing'`/`'invalid'` for the same reason
   * `rewiringNull.ts#RewiringNullLoadResult` keeps its own `'unavailable'`
   * apart from `'absent'`/`'invalid'` (round-2 dual review, Important): a
   * dropped request or a transient 5xx is retryable, unlike a genuinely
   * missing manifest entry or a hash/shape failure, so
   * `ActivityPanel.svelte#ensureLesionAtlasLoaded` deliberately does not
   * memoize this outcome — selecting the mode again retries the fetch.
   */
  | { status: 'unavailable'; reason: string }
  | { status: 'invalid'; reason: string };

/**
 * Fetch, sha256-verify, and structurally validate `lesion-atlas-v1.json`
 * (`manifest.lesionAtlas`), then cross-check it against the manifest's own
 * graph hashes and the already-parsed biological graph's `biologicalIds` —
 * never throws, matching `loadPositions`/`loadRewiringNull`'s "never throws,
 * always return a reasoned status" contract, so a missing/tampered/malformed
 * atlas only ever disables the lesion-effect color mode with an honest
 * reason, never the rest of the activity view or the experiment.
 *
 * Deliberately **not** loaded eagerly the way `loadPositions`
 * (`App.svelte#onManifest`) and `loadRewiringNull`
 * (`ExperimentController#initialize`) are: the lesion atlas is loaded
 * lazily, only when `ActivityPanel.svelte` first selects the lesion-effect
 * color mode (see that component's `ensureLesionAtlasLoaded`), so a session
 * that never opens the mode never pays its fetch/verify cost.
 *
 * `dataBaseUrl` must be the same value the caller passes to
 * `loadArenaArtifacts`/`loadPositions` (`${import.meta.env.BASE_URL}data` in
 * production) so this artifact resolves under the app's real deployment base
 * path too. `biologicalGraph` must be the already-verified, already-parsed
 * graph `ExperimentController.initialize()` produced for this same manifest
 * (threaded through `App.svelte`'s `onManifest` callback), reused here
 * rather than re-fetched/re-parsed — the same discipline `loadPositions`
 * documents for its own `biologicalGraph` parameter.
 */
export const loadLesionAtlas = async (
  manifest: ArenaManifest,
  dataBaseUrl: string,
  biologicalGraph: ConnectomeGraph
): Promise<LesionAtlasLoadResult> => {
  const fetched = await fetchAndVerifySidecarJson(manifest.lesionAtlas, dataBaseUrl, 'lesion-atlas artifact');
  if (fetched.status === 'no-entry') {
    return { status: 'missing', reason: 'The manifest has no lesionAtlas artifact entry.' };
  }
  if (fetched.status === 'fetch-error') {
    return { status: 'unavailable', reason: fetched.reason };
  }
  if (fetched.status === 'hash-mismatch' || fetched.status === 'parse-error') {
    return { status: 'invalid', reason: fetched.reason };
  }

  const validated = validateLesionAtlasShape(fetched.parsed);
  if (!validated.ok) return { status: 'invalid', reason: validated.reason };
  const data = validated.data;

  if (data.neuronCount !== manifest.neuronCount) {
    return {
      status: 'invalid',
      reason: `lesion-atlas neuronCount ${data.neuronCount} does not match the manifest's own declared neuronCount (${manifest.neuronCount})`
    };
  }

  if (data.graphs.biological.graphSha256 !== manifest.binarySha256) {
    return {
      status: 'invalid',
      reason: `lesion-atlas biological graphSha256 does not match the manifest's compiled biological graph (${manifest.binarySha256}) — stale artifact`
    };
  }
  // `loadArenaArtifacts` already throws if the manifest is missing its
  // seed0 rewired arm entirely, so by the time this loader ever runs in
  // production that field exists — this still fails closed rather than
  // silently skipping the cross-check for a hand-built/stale manifest (same
  // fail-closed discipline as `assets.ts#loadArenaArtifacts`'s own
  // `swapStats.edgeCount` guard — round-2 dual review corrected a wrong
  // cross-reference here: `loadRewiringNull` actually does the *opposite*
  // for its own seed-0 check, skipping it when the manifest field is
  // absent; see that function's `shippedSeed0GzipSha256 &&` guard).
  const shippedSeed0BinarySha256 = manifest.rewiredArms.seed0?.binarySha256;
  if (!shippedSeed0BinarySha256) {
    return {
      status: 'invalid',
      reason: 'manifest is missing rewiredArms.seed0.binarySha256, needed to cross-check the lesion atlas'
    };
  }
  if (data.graphs.rewiredSeed0.graphSha256 !== shippedSeed0BinarySha256) {
    return {
      status: 'invalid',
      reason: `lesion-atlas rewiredSeed0 graphSha256 does not match the manifest's shipped rewired seed-0 graph (${shippedSeed0BinarySha256}) — stale artifact`
    };
  }

  if (biologicalGraph.biologicalIds.length !== data.neuronCount) {
    return {
      status: 'invalid',
      reason: `graph biologicalIds length ${biologicalGraph.biologicalIds.length} does not match lesion-atlas neuronCount ${data.neuronCount}`
    };
  }
  for (let index = 0; index < data.neuronCount; index += 1) {
    if (data.bodyIds[index] !== biologicalGraph.biologicalIds[index].toString()) {
      return {
        status: 'invalid',
        reason: `lesion-atlas bodyIds[${index}] "${data.bodyIds[index]}" does not match the graph's biologicalIds[${index}] "${biologicalGraph.biologicalIds[index].toString()}"`
      };
    }
  }

  let absMax = 0;
  for (const graph of [data.graphs.biological, data.graphs.rewiredSeed0]) {
    for (const value of graph.effect) {
      const magnitude = Math.abs(value);
      if (magnitude > absMax) absMax = magnitude;
    }
  }

  return { status: 'ok', data, absMax };
};
