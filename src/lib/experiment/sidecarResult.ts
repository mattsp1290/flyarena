/**
 * Shared shape for every "fetch, sha256-verify, and structurally validate an
 * optional sidecar artifact" load result in this directory
 * (thermo-maintainability review I3): `./nullExplanation.ts`'s
 * `NullExplanationLoadResult`, `./rewiringNull.ts`'s `RewiringNullLoadResult`,
 * and `./lesionAtlas.ts`'s `LesionAtlasLoadResult` all hand-rolled the same
 * four-state union — `nullExplanation.ts` was the third near-identical copy,
 * and the one time introducing this generic cost nothing extra, since that
 * file was brand new in the same branch.
 *
 * Generic over two things besides the artifact's own data type, so each
 * existing union can become a genuine zero-behavior-change type alias
 * instead of a rename:
 * - `TMissingStatus`: the literal status name for "nothing was ever
 *   shipped" — `'missing'` by default, but `RewiringNullLoadResult` already
 *   ships (and asserts against, in tests) `'absent'` for the same state, so
 *   that vocabulary difference is left as-is rather than forced into one
 *   name here (thermo-maintainability I3's own fix note: renaming `'absent'`
 *   needs its own separate decision).
 * - `TOkExtra`: extra fields alongside `data` on the `'ok'` branch —
 *   `LesionAtlasLoadResult` carries `absMax` there; `NullExplanationLoadResult`
 *   and `RewiringNullLoadResult` carry nothing extra, hence the `unknown`
 *   default (intersecting with `unknown` adds no fields).
 *
 * The four states themselves are the shared, load-bearing part of this type:
 * - `'ok'`: fetched, sha256-verified, and structurally (and, where
 *   applicable, cross-artifact) validated.
 * - `TMissingStatus` (`'missing'`/`'absent'`): the manifest has no entry for
 *   this artifact at all — nothing was ever shipped.
 * - `'unavailable'`: a fetch/network failure, or an unexpected runtime
 *   error anywhere in the load chain — not a claim about the artifact's
 *   integrity.
 * - `'invalid'`: the artifact was actually fetched and failed a real
 *   verification step (sha256, shape, or a cross-field/cross-artifact
 *   consistency check).
 */
export type SidecarLoadResult<TData, TMissingStatus extends string = 'missing', TOkExtra = unknown> =
  | ({ status: 'ok'; data: TData } & TOkExtra)
  | { status: TMissingStatus; reason: string }
  | { status: 'unavailable'; reason: string }
  | { status: 'invalid'; reason: string };
