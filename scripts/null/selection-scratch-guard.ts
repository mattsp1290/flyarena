import { readFileSync } from 'node:fs';
import { resolve, sep } from 'node:path';

/**
 * `.agents/plans/selection-robustness/02-per-selection-chain.md` WP2's
 * "only the shipped graph may write into the shipped tree" refusal,
 * extracted out of `null-report.ts` and `intervention-report-run-mode.ts`
 * (a thermo-maintainability review finding: both files independently
 * implemented the identical algorithm -- read the shipped manifest's
 * `binarySha256`, allow when it matches the caller's `sourceGraphSha256`,
 * otherwise refuse any candidate path that resolves under `public/` or
 * `docs/`). This module deliberately does NOT import anything back from
 * either caller -- `publicDataDir`/`docsDir` are passed in by the caller
 * (which already owns its own `public`/`docs` resolution), the same
 * one-directional import boundary `null-report-variant.ts` already
 * established for `guardVariantOutPath`.
 *
 * A per-selection chain reuses `null-report.ts`'s and
 * `intervention-report.ts`'s ordinary (non-`--variant-out`) publish paths
 * against a scratch `training/runs/selections/<id>/` output tree, with a
 * real input scored against a *different* biological graph than the one
 * shipped in `public/data/malecns-arena-v1.manifest.json` -- so an
 * exact-path "is this the shipped default file" check (like
 * `guardShippedDefault`/`guardCanonicalOutDefault`) never fires for it, the
 * same way it never fires for any other explicit scratch `--out`. But that
 * exact-path check is the *only* thing standing between an arbitrary
 * explicit `--out` and a write anywhere under `public/data/` or `docs/` --
 * `--out public/data/rewiring-null-selection-larger.json` (a different
 * filename, not the shipped default) sails straight past it.
 *
 * This closes that gap for any artifact scored against a graph other than
 * the one the shipped manifest actually describes: the candidate path may
 * resolve under `publicDataDir`'s parent (`public/`) or `docsDir` only when
 * the scored graph's `sourceGraphSha256` matches the shipped manifest's own
 * `binarySha256` -- the same invariant `scripts/data/selections.py`'s
 * `refuse_unsafe_variant_target` already enforces on the compiler side,
 * mirrored here with a directory-*prefix* check (like
 * `guardVariantOutPath`'s tree block), not an exact-path one. Fails safe:
 * if the shipped manifest cannot be read at all (missing or corrupt --
 * `JSON.parse` failures included), the graph is treated as not-shipped,
 * since there is then no basis to prove the write is safe.
 *
 * Not shared with the Python mirror, `explain_selection_mode.py`'s
 * `guard_selection_scratch_target` -- that one has a real, justified extra
 * branch (the `selection_mode` unconditional-refusal case, needed because
 * `explain.py --selection-mode` can otherwise silently overwrite
 * `null-explanation-v1.json` with a schema the browser's validator rejects)
 * that neither TS caller needs. Cross-language mirroring (not sharing) is
 * an accepted, already-established pattern here -- `refuse_unsafe_variant_target`
 * / `guardVariantOutPath` are likewise mirrored, not shared, across
 * Python/TS.
 */
export const guardSelectionScratchTarget = (
  path: string,
  flagLabel: string,
  sourceGraphSha256: string,
  publicDataDir: string,
  docsDir: string
): void => {
  const shippedManifestPath = resolve(publicDataDir, 'malecns-arena-v1.manifest.json');
  let shippedSha256: string | undefined;
  try {
    const shippedManifest = JSON.parse(readFileSync(shippedManifestPath, 'utf8')) as { binarySha256?: string };
    shippedSha256 = shippedManifest.binarySha256;
  } catch {
    shippedSha256 = undefined;
  }
  if (shippedSha256 !== undefined && shippedSha256 === sourceGraphSha256) return;
  const resolved = resolve(path);
  for (const shippedDir of [resolve(publicDataDir, '..'), resolve(docsDir)]) {
    if (resolved === shippedDir || resolved.startsWith(shippedDir + sep)) {
      throw new Error(
        `${flagLabel} (${resolved}) resolves under ${shippedDir}, but this artifact was scored ` +
          `against a graph with sha256 ${sourceGraphSha256}, which does not match the shipped biological graph` +
          `${shippedSha256 !== undefined ? ` (${shippedSha256})` : ' (the shipped manifest could not be read)'} ` +
          '-- refusing to write a non-shipped-graph artifact into a shipped tree. Pass an explicit scratch path ' +
          'outside public/ and docs/ (e.g. training/runs/selections/<id>/...).'
      );
    }
  }
};
