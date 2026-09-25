/**
 * A plain GitHub blob URL under `docs/`, for a human-readable report that is
 * not part of the deployed static site (only `public/` is served, so a
 * relative `docs/…` link would 404 under any base path). Shared by
 * `LedgerPanel.svelte` (trained-readout report) and `NullHistogram.svelte`
 * (rewiring-null report) so the repo/branch prefix lives in exactly one
 * place (thermo-maintainability review S3: the two components previously
 * hand-typed the same prefix into their own identically-named constant).
 */
export const githubDocUrl = (path: string): string =>
  `https://github.com/mattsp1290/flyarena/blob/main/docs/${path}`;
