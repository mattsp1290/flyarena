/**
 * WP2 of `.agents/plans/findings-tour` (`02-verified-redeploy.md`, step 5):
 * a small Chromium smoke check against a real, already-deployed `DEPLOY_URL`
 * -- run by `scripts/deploy.sh`'s `--deploy` mode (still holding its lock)
 * after the byte-for-byte asset verification passes, and again, best-effort,
 * after a rollback. Also runnable standalone: `DEPLOY_URL=... node_modules/.bin/tsx
 * scripts/verify/live-smoke.ts` (or `npx tsx ...`).
 *
 * Reuses the real page structure `tests/e2e/findings.spec.ts` and
 * `tests/e2e/arena-test-helpers.ts` already assert against, rather than
 * inventing new selectors: the ready `[role="status"]` text (`statusRegion`),
 * the Findings panel's own expand toggle (`findingsToggle`), its `li.step`
 * stepper items and `p.sentence` text (`findings.spec.ts`'s step assertions,
 * every sentence ends "under this model."), and the model ledger's
 * `section.ledger` render (`tests/e2e/arena.spec.ts`'s `.ledger` locator and
 * its static "Graph topology" / "Measured" row, present regardless of the
 * live manifest's actual counts).
 *
 * Deliberately does not import `@playwright/test`'s `expect` (or the
 * `arena-test-helpers.ts` exports built on it, e.g. `waitForReady`,
 * `expandFindingsPanel`) -- those assertions are meant to run inside
 * Playwright's own test runner/reporter, not a standalone script launched
 * directly with `tsx`. This script only reuses the *locator* helpers
 * (`statusRegion`, `findingsToggle`), which are plain functions with no
 * dependency on a test-runner context, and does its own waiting/assertions
 * with the plain Playwright API.
 *
 * No credentials are used or required. `DEPLOY_URL` itself is never
 * printed -- only fixed, non-secret progress/failure messages.
 */
import { chromium } from '@playwright/test';
import { findingsToggle, statusRegion } from '../../tests/e2e/arena-test-helpers';

const READY_TIMEOUT_MS = 20_000;
const STEP_TIMEOUT_MS = 10_000;

const deployUrl = process.env.DEPLOY_URL;
if (!deployUrl) {
  console.error('live-smoke: DEPLOY_URL is not set.');
  process.exit(1);
}

/** Polls `check` until it returns true or `timeoutMs` elapses; throws `message` on timeout. Never logs `check`'s inputs. */
const waitUntil = async (check: () => Promise<boolean>, timeoutMs: number, message: string): Promise<void> => {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    if (await check()) return;
    if (Date.now() >= deadline) throw new Error(message);
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
};

const main = async (): Promise<void> => {
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage();
    // Never echo the URL: any thrown error below (e.g. Playwright's own
    // navigation-failure message) may otherwise include it verbatim.
    await page.goto(deployUrl, { waitUntil: 'domcontentloaded', timeout: READY_TIMEOUT_MS }).catch(() => {
      throw new Error('live-smoke: failed to load the deploy target.');
    });

    // (dual review, Important) `.catch(() => null)`, matching every other
    // check below: right after `domcontentloaded`, the Svelte app has
    // likely not hydrated yet, so `[role="status"]` may not exist yet.
    // Without the guard, Playwright's own `textContent()` blocks for its
    // default action timeout (30s) on a still-missing element instead of
    // `waitUntil`'s own polling loop enforcing READY_TIMEOUT_MS, and throws
    // Playwright's generic timeout error instead of this function's own
    // message.
    await waitUntil(
      async () => ((await statusRegion(page).textContent().catch(() => null)) ?? '').trim() === 'ready',
      READY_TIMEOUT_MS,
      'live-smoke: status region never reached "ready".'
    );

    const toggle = findingsToggle(page);
    await waitUntil(
      async () => (await toggle.isEnabled().catch(() => false)) === true,
      STEP_TIMEOUT_MS,
      'live-smoke: Findings toggle never became enabled.'
    );
    await toggle.click();
    await waitUntil(
      async () => /collapse/i.test((await toggle.textContent()) ?? ''),
      STEP_TIMEOUT_MS,
      'live-smoke: Findings panel did not expand.'
    );

    // Step 1's sentence (`findings.spec.ts`'s own invariant: every rendered
    // step sentence is templated from the artifact and ends "under this
    // model.", never hard-coded copy).
    const step1Sentence = page.locator('section.findings li.step').nth(0).locator('p.sentence');
    await waitUntil(
      async () => /under this model\.\s*$/i.test(((await step1Sentence.textContent().catch(() => null)) ?? '').trim()),
      STEP_TIMEOUT_MS,
      'live-smoke: Findings step 1 sentence did not render (or did not match the expected "under this model." pattern).'
    );

    // The model ledger render (`arena.spec.ts`'s `.ledger` locator; "Graph
    // topology" / "Measured" is a static row, present regardless of the
    // live manifest's own counts).
    const ledger = page.locator('section.ledger');
    await waitUntil(
      async () => /graph topology/i.test((await ledger.textContent().catch(() => null)) ?? ''),
      STEP_TIMEOUT_MS,
      'live-smoke: model ledger did not render.'
    );

    console.log('live-smoke: passed (ready, Findings step 1, ledger).');
  } finally {
    await browser.close();
  }
};

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : 'live-smoke: failed.');
  process.exit(1);
});
