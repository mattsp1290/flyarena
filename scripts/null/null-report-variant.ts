import { dirname, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

import type { NullDecoderKind } from './null-worker';

/**
 * `.agents/plans/null-explanation/01-decoder-variants.md` WP1's
 * variant-publish machinery, extracted out of `null-report.ts` (a
 * thermo-maintainability review finding: WP1 pushed `null-report.ts` from
 * 748 to 903 lines, repeating -- in miniature -- the exact "self-contained
 * slice bolted into a shared file" pattern a prior thermo review already
 * split out once at this branch's base commit, `null-report-trained.ts`).
 * `RewiringNullCondition`/`CONDITION_LABELS` are a pure type + lookup table;
 * `resolveCondition` and `guardVariantOutPath` are pure validation
 * functions. `null-report.ts` imports from here the same way it already
 * imports `buildTrainedSection`/`renderTrainedSection` from
 * `null-report-trained.ts`. This module deliberately does NOT import
 * anything back from `null-report.ts` -- the shipped default paths
 * `guardVariantOutPath` must reject are passed in by the caller (which
 * already owns `DEFAULT_OUT`/`DEFAULT_REPORT_MD`/`DEFAULT_MANIFEST`) rather
 * than re-derived or duplicated here -- to keep the import boundary
 * one-directional.
 */

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '../..');

/**
 * Every condition a rewiring-null artifact can be labelled with -- the
 * authored condition (unchanged wording, so the shipped
 * `rewiring-null-v1.json` stays byte-identical) plus the three
 * decoder-convention-check variants
 * (`.agents/plans/null-explanation/01-decoder-variants.md` WP1). See
 * `CONDITION_LABELS` for the `NullDecoderKind -> RewiringNullCondition`
 * mapping.
 */
export type RewiringNullCondition =
  | 'authored, opponent parked'
  | 'authored (thrust flipped), opponent parked'
  | 'authored (yaw flipped), opponent parked'
  | 'authored (thrust and yaw flipped), opponent parked';

/** `NullDecoderKind -> RewiringNullCondition`, the single source of truth `resolveCondition` uses to derive a condition label from `raw.decoder`. */
export const CONDITION_LABELS: Record<NullDecoderKind, RewiringNullCondition> = {
  authored: 'authored, opponent parked',
  'authored-flip-thrust': 'authored (thrust flipped), opponent parked',
  'authored-flip-yaw': 'authored (yaw flipped), opponent parked',
  'authored-flip-both': 'authored (thrust and yaw flipped), opponent parked'
};

/**
 * Derive `RewiringNullCondition` from an `authored.json`'s `raw.decoder`.
 * Missing (or explicitly `null`, from a hand-edited/corrupted file) on any
 * `authored.json` produced before this field existed -- treated as
 * `'authored'`, the only value every such file could ever have meant.
 * Anything else that isn't a recognized `NullDecoderKind` throws rather
 * than silently producing a variant artifact with no `condition` (a
 * dual-review finding). `Object.hasOwn` (not `in`, which walks the
 * prototype chain) -- a hand-edited/corrupted `authored.json` with
 * `"decoder": "toString"` or `"constructor"` must be rejected as
 * unrecognized, not silently resolve to a function inherited from
 * `Object.prototype` (a reviewer finding).
 *
 * `sourcePath` is only used to build an actionable error message (the
 * `authored.json` path this decoder value came from) -- it is never read.
 */
export const resolveCondition = (
  rawDecoder: NullDecoderKind | null | undefined,
  sourcePath: string
): RewiringNullCondition => {
  const decoder: NullDecoderKind = rawDecoder ?? 'authored';
  if (!Object.hasOwn(CONDITION_LABELS, decoder)) {
    throw new Error(`null-report: ${sourcePath} has an unrecognized decoder "${String(decoder)}"`);
  }
  return CONDITION_LABELS[decoder];
};

/**
 * `--variant-out` is otherwise unconstrained (see `runNullReport`'s variant
 * branch, which deliberately skips `guardShippedDefault`) -- without this
 * check, `--variant-out public/data/rewiring-null-v1.json` (or the report
 * markdown, the manifest, or even `--authored`'s own input file) would
 * silently overwrite a shipped/input path with a decoder-variant artifact, a
 * dual-review finding on this WP. `.json` is required for the same reason
 * `--out`/`--authored` require it elsewhere in this study: a caller that
 * strips a trailing `.json` for a sidecar path must never collide.
 *
 * `shipped.out`/`shipped.reportMd`/`shipped.manifest` are passed in by the
 * caller (`null-report.ts`'s `DEFAULT_OUT`/`DEFAULT_REPORT_MD`/
 * `DEFAULT_MANIFEST`) rather than re-derived here, so this module has
 * exactly one source of truth for those paths to stay in sync with.
 */
export const guardVariantOutPath = (
  variantOut: string,
  authoredPath: string,
  shipped: { readonly out: string; readonly reportMd: string; readonly manifest: string }
): void => {
  if (!variantOut.endsWith('.json')) {
    throw new Error(`null-report: --variant-out must end with ".json" (got "${variantOut}")`);
  }
  const resolved = resolve(variantOut);
  const forbidden: readonly [string, string][] = [
    [shipped.out, 'the shipped published artifact'],
    [shipped.reportMd, 'the shipped report markdown'],
    [shipped.manifest, 'the shipped manifest'],
    [authoredPath, 'its own --authored input']
  ];
  for (const [path, label] of forbidden) {
    if (resolved === resolve(path)) {
      throw new Error(`null-report: --variant-out must not resolve to ${label} (${path})`);
    }
  }
  // Named-path checks above only ever catch the specific shipped files this
  // study knows about; `public/` and `docs/` both hold other tracked,
  // shipped JSON this study doesn't touch (trained-readout-v1.json, the
  // ledger, positions, lab-benchmark.json, ...) that a copy-pasted or
  // typo'd path could still land on, and a named list would never keep up
  // with future shipped files anyway (a reviewer finding). Block the whole
  // tree instead.
  for (const shippedDir of [resolve(repoRoot, 'public'), resolve(repoRoot, 'docs')]) {
    if (resolved === shippedDir || resolved.startsWith(shippedDir + sep)) {
      throw new Error(`null-report: --variant-out must not be under ${shippedDir} (a shipped tree)`);
    }
  }
};
