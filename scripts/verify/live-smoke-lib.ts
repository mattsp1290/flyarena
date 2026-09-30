/**
 * WP2 of `.agents/plans/consolidated-release` (`02-release.md`)'s extended
 * Findings step-status check: the pure decision logic factored out of
 * `scripts/verify/live-smoke.ts` so it can be unit-tested
 * (`tests/unit/live-smoke-lib.test.ts`) without a real Chromium/Playwright
 * session. Everything here is a plain function over plain values -- no
 * `Page`/`Locator` in sight -- `live-smoke.ts` itself stays the thin
 * Playwright-driving shell that calls these functions and does the actual
 * waiting/navigating.
 *
 * The rules this module encodes (`02-release.md`'s own change-surface row):
 *  - `ok` always passes.
 *  - `missing` passes only if the step id is listed in the (committed,
 *    default-empty) smoke allowlist; otherwise it fails.
 *  - `invalid` fails immediately -- a real verification failure on
 *    already-published data, never retried.
 *  - `unavailable` -- a network-fetch failure against the live origin,
 *    not a verification failure -- gets exactly one reload-and-recheck
 *    before it counts as a failure. `decideStep`'s `isRetry` flag is what
 *    turns a *second* `unavailable` look into a hard failure instead of
 *    asking `live-smoke.ts` for a second retry it was never promised.
 */

/**
 * Mirrors `FindingStepStatus` (`src/lib/findings/steps/shared.ts`) --
 * restated here rather than imported, so this module (and the script that
 * imports it) stays free of any dependency on that step-builder module
 * graph. `live-smoke.ts` itself does import `FINDING_SECTIONS` from
 * `src/lib/findings/sections.ts` directly -- see that script's own doc
 * comment on why that one import is safe (a standalone data module, no
 * runtime imports of its own) while the heavier `steps.ts` orchestrator
 * (which pulls in every study's browser-`fetch`-based loader) is not
 * something a standalone `tsx` script should import.
 */
export type StepStatus = 'loading' | 'ok' | 'missing' | 'unavailable' | 'invalid';

export type StepDecision =
  | { readonly kind: 'ok' }
  | { readonly kind: 'missing-allowed' }
  | { readonly kind: 'retry-unavailable' }
  | { readonly kind: 'fail'; readonly reason: string };

/**
 * Classifies one step's already-settled (non-`'loading'`) status.
 * `isRetry` must be `true` only on the second look at a step that was
 * `'unavailable'` on its first look (after the one allowed reload) -- it
 * never grants a second retry, regardless of how many times the caller
 * passes `true`.
 */
export const decideStep = (
  id: string,
  status: Exclude<StepStatus, 'loading'>,
  allowlist: readonly string[],
  isRetry: boolean
): StepDecision => {
  switch (status) {
    case 'ok':
      return { kind: 'ok' };
    case 'missing':
      return allowlist.includes(id)
        ? { kind: 'missing-allowed' }
        : { kind: 'fail', reason: `Findings step "${id}" is "missing" and not in the smoke allowlist.` };
    case 'invalid':
      return { kind: 'fail', reason: `Findings step "${id}" is "invalid" (failed verification).` };
    case 'unavailable':
      return isRetry
        ? { kind: 'fail', reason: `Findings step "${id}" is still "unavailable" after the one allowed reload-and-recheck.` }
        : { kind: 'retry-unavailable' };
  }
};

/**
 * Parses/validates `scripts/verify/smoke-allow-missing.json`'s already-read
 * text into a readonly array of step-id strings. Throws a fixed,
 * file-content-free message on anything other than a JSON array of
 * strings: the file is committed and reviewed, but a hand-edit mistake
 * should fail the smoke check loudly rather than silently allow every step
 * to be missing (a non-array/non-string entry could otherwise coerce into
 * an always-true or always-false `includes` check depending on the bug).
 */
export const parseAllowlist = (raw: string): readonly string[] => {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new Error('live-smoke: smoke-allow-missing.json is not valid JSON.');
  }
  if (!Array.isArray(parsed) || !parsed.every((entry): entry is string => typeof entry === 'string')) {
    throw new Error('live-smoke: smoke-allow-missing.json must be a JSON array of step-id strings.');
  }
  return parsed;
};
