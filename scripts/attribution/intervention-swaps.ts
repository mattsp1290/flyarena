import { mkdirSync, readFileSync } from 'node:fs';
import { dirname } from 'node:path';

import { atomicWriteFileSync, sha256Hex } from '../training/fsio';

/**
 * `training/archive/intervention-swaps-v1.json`: extracts the accepted-swap
 * `addedEdge`/`removedEdge` lists for P and Q from `interventions.py`'s
 * `attribution.json`. H3 uses the P-added edges' thrust endpoints as
 * "newly connected thrust neurons" (`00-overview.md`).
 *
 * Split out of `archive-readouts.ts` (a thermo-maintainability review
 * finding) into its own module -- this concern doesn't depend on
 * `ArchivedReadout`'s schema or on either of the other two copy/extract
 * concerns.
 */

export interface SwapEdge {
  readonly pre: number;
  readonly post: number;
}

interface AttributionStep {
  readonly step: number;
  readonly accepted: boolean;
  readonly addedEdge: SwapEdge;
  readonly addedEdge2: SwapEdge;
  readonly removedEdge: SwapEdge;
  readonly removedEdge2: SwapEdge;
}

interface AttributionEntry {
  readonly kind: string;
  readonly steps: readonly AttributionStep[];
  /** `interventions.py`'s own count of accepted swap steps -- cross-checked against `steps.filter(accepted).length` below, a free integrity check on the parse. */
  readonly swaps?: number;
}

export interface InterventionSwapEntry {
  readonly id: string;
  /** Every accepted step's `addedEdge`/`addedEdge2`, in step order -- H3 uses these edges' thrust endpoints as "newly connected thrust neurons" (`00-overview.md`). */
  readonly addedEdges: readonly SwapEdge[];
  /** Every accepted step's `removedEdge`/`removedEdge2`, in step order. */
  readonly removedEdges: readonly SwapEdge[];
}

export interface InterventionSwapsArchive {
  readonly version: 1;
  /** sha256 of the source `attribution.json`'s raw bytes -- provenance, not a value with a separately-published record to check it against (unlike `intervention-index-v1.json`'s `sourceSha256`, which `pathway-interventions-v1.json`'s `sources.indexSha` does independently confirm). */
  readonly sourceSha256: string;
  /** `P` then `Q`, in that order -- `00-overview.md`'s H3 only ever needs `P`; `Q` is archived alongside it since `interventions.py` produces both from one run and a future analysis may need it. */
  readonly swaps: readonly InterventionSwapEntry[];
}

const isSwapEdge = (value: unknown): value is SwapEdge =>
  typeof value === 'object' &&
  value !== null &&
  Number.isInteger((value as Partial<SwapEdge>).pre) &&
  Number.isInteger((value as Partial<SwapEdge>).post);

const extractSwapsFor = (id: string, entry: AttributionEntry | undefined): InterventionSwapEntry => {
  if (!entry || !Array.isArray(entry.steps)) {
    throw new Error(`archive-readouts: attribution.json has no "${id}" entry with a "steps" array`);
  }
  const accepted = entry.steps.filter((step) => step.accepted === true);
  if (typeof entry.swaps === 'number' && accepted.length !== entry.swaps) {
    throw new Error(
      `archive-readouts: attribution.json's "${id}" entry has ${accepted.length} accepted step(s) but ` +
        `swaps=${entry.swaps}`
    );
  }
  const addedEdges: SwapEdge[] = [];
  const removedEdges: SwapEdge[] = [];
  for (const step of accepted) {
    for (const edge of [step.addedEdge, step.addedEdge2, step.removedEdge, step.removedEdge2]) {
      if (!isSwapEdge(edge)) {
        throw new Error(`archive-readouts: attribution.json's "${id}" step ${step.step} has a malformed edge`);
      }
    }
    addedEdges.push({ pre: step.addedEdge.pre, post: step.addedEdge.post }, { pre: step.addedEdge2.pre, post: step.addedEdge2.post });
    removedEdges.push(
      { pre: step.removedEdge.pre, post: step.removedEdge.post },
      { pre: step.removedEdge2.pre, post: step.removedEdge2.post }
    );
  }
  return { id, addedEdges, removedEdges };
};

/** Read+validate+build (not write) -- split out so `runArchiveReadouts` can validate this input before writing anything, matching `readRawInterventionScores`'s split. */
export const buildInterventionSwapsArchive = (sourcePath: string): InterventionSwapsArchive => {
  const raw = readFileSync(sourcePath);
  const parsed = JSON.parse(raw.toString('utf8')) as { readonly P?: AttributionEntry; readonly Q?: AttributionEntry };
  const swaps = [extractSwapsFor('P', parsed.P), extractSwapsFor('Q', parsed.Q)];
  return { version: 1, sourceSha256: sha256Hex(raw), swaps };
};

export const extractInterventionSwaps = (sourcePath: string, outPath: string): void => {
  const archive = buildInterventionSwapsArchive(sourcePath);
  mkdirSync(dirname(outPath), { recursive: true });
  atomicWriteFileSync(outPath, JSON.stringify(archive));
};
