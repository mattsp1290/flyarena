/**
 * WP2 of `.agents/plans/findings-tour` (`02-verified-redeploy.md`, step 5),
 * extended by WP2 of `.agents/plans/consolidated-release` (`02-release.md`):
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
 * `02-release.md`'s own extension: after Findings is expanded (every
 * question section renders open by default -- `01-findings-sections.md`),
 * every step id declared in `src/lib/findings/sections.ts#FINDING_SECTIONS`
 * -- the real, tested source of truth for which step ids exist and how
 * they're grouped (`tests/unit/findings-sections.test.ts` already gates
 * "Other findings" staying empty on `main`) -- is polled via its own
 * `data-step-status` attribute (`FindingsPanel.svelte`) until it leaves
 * `'loading'`, then classified by `./live-smoke-lib.ts#decideStep` against
 * the committed `./smoke-allow-missing.json` allowlist. `FINDING_SECTIONS`
 * is imported directly here (not scraped from the DOM) because it is a
 * standalone data module with zero runtime imports of its own (unlike
 * `src/lib/findings/steps.ts`, which pulls in every study's own
 * browser-`fetch`-based loader) -- safe for a plain `tsx` script to import
 * the same way `tests/unit/findings-sections.test.ts` does. The DOM step
 * count is still cross-checked against that import below, so a step that
 * somehow lands unclassified in a fifth "Other findings" section (which
 * `main`'s own unit test already forbids, but a smoke check should never
 * silently trust a unit test it isn't running) fails loudly instead of
 * simply not being checked.
 *
 * Also asserts `#graph-lab` loads its default idle state, "Connect to a
 * running graph-lab backend to submit jobs." (`src/lib/graphlab/GraphLab.svelte`),
 * with no page/console errors -- without ever clicking Connect, so there is
 * no network call and no dependence on the runner's loopback state
 * (`00-overview.md`'s own non-goal: "the graph-lab backend launch and real-
 * origin round trip... still wait on owner setup").
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
 * printed -- only fixed, non-secret progress/failure messages (step ids,
 * declared statuses, and the smoke allowlist's own step ids -- never a
 * hostname, IP, or URL).
 */
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium, type Page } from '@playwright/test';
import { findingsToggle, statusRegion } from '../../tests/e2e/arena-test-helpers';
import { FINDING_SECTIONS } from '../../src/lib/findings/sections';
import { decideStep, parseAllowlist, type StepStatus } from './live-smoke-lib';

const READY_TIMEOUT_MS = 20_000;
const STEP_TIMEOUT_MS = 10_000;
/** `02-release.md`: "poll... until the status leaves loading, with a per-step timeout of 30 s." */
const STEP_STATUS_TIMEOUT_MS = 30_000;
/** `02-release.md`: "gets one reload-and-recheck after 10 s". */
const UNAVAILABLE_RETRY_DELAY_MS = 10_000;
/** `src/lib/graphlab/GraphLab.svelte`'s own idle-state copy, verbatim. */
const GRAPH_LAB_IDLE_TEXT = 'Connect to a running graph-lab backend to submit jobs.';

const deployUrl = process.env.DEPLOY_URL;
if (!deployUrl) {
  console.error('live-smoke: DEPLOY_URL is not set.');
  process.exit(1);
}

const here = dirname(fileURLToPath(import.meta.url));
const allowlist = parseAllowlist(readFileSync(resolve(here, 'smoke-allow-missing.json'), 'utf-8'));

/** Every step id `FindingsPanel.svelte` is expected to render, in `FINDING_SECTIONS`' own declared order -- never a hard-coded list (this file's own doc comment explains why this import, not DOM-scraping, is the source of truth). */
const expectedStepIds = FINDING_SECTIONS.flatMap((section) => section.stepIds);

/** Polls `check` until it returns true or `timeoutMs` elapses; throws `message` on timeout. Never logs `check`'s inputs. */
const waitUntil = async (check: () => Promise<boolean>, timeoutMs: number, message: string): Promise<void> => {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    if (await check()) return;
    if (Date.now() >= deadline) throw new Error(message);
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
};

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

/** Waits for `[role="status"]` to read "ready", then expands the Findings panel -- the sequence every one of this script's checks depends on, and the sequence a `page.reload()` (used by the "unavailable" retry below) always resets, since a reload re-collapses the panel (`02-release.md`: "the recheck repeats the ready wait and the Findings expand click since reload collapses the panel"). */
const waitForReadyAndExpandFindings = async (page: Page): Promise<void> => {
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
  // (thermo review, Important I3/ops-safety) `.catch(() => null)`, matching
  // every other DOM read in this file: without it, a transient exception
  // right after `.click()` (e.g. the toggle briefly detaching during a
  // Svelte re-render) propagates straight out instead of being retried by
  // `waitUntil`'s own polling loop within its timeout budget -- and in
  // deploy.sh a smoke-check failure triggers a rollback, so a spurious
  // failure here has real operational cost, not just log noise.
  await waitUntil(
    async () => /collapse/i.test((await toggle.textContent().catch(() => null)) ?? ''),
    STEP_TIMEOUT_MS,
    'live-smoke: Findings panel did not expand.'
  );
};

const stepLocator = (page: Page, id: string) => page.locator(`section.findings li.step[data-step-id="${id}"]`);

/** Polls one step's `data-step-status` until it leaves `'loading'` (`STEP_STATUS_TIMEOUT_MS` budget), then returns the settled status. */
const waitForStepSettled = async (page: Page, id: string): Promise<Exclude<StepStatus, 'loading'>> => {
  const locator = stepLocator(page, id);
  let settled: Exclude<StepStatus, 'loading'> | null = null;
  await waitUntil(
    async () => {
      const value = await locator.getAttribute('data-step-status').catch(() => null);
      if (value !== null && value !== 'loading') {
        settled = value as Exclude<StepStatus, 'loading'>;
        return true;
      }
      return false;
    },
    STEP_STATUS_TIMEOUT_MS,
    `live-smoke: Findings step "${id}" never left "loading".`
  );
  // `waitUntil` only resolves (rather than throwing) once `settled` above
  // has actually been assigned inside its polling callback -- see its own
  // contract at the top of this file -- so this is never really `null`
  // here. TS's control-flow narrowing can't see across that callback
  // boundary (it only sees the literal `null` from the initial
  // declaration), so the cast goes through `unknown` rather than directly,
  // per TS's own suggestion for this exact "narrows don't cross a closure"
  // shape.
  return settled as unknown as Exclude<StepStatus, 'loading'>;
};

/** Checks one step against `decideStep`'s rules, performing the one allowed reload-and-recheck for an `'unavailable'` status. Throws on any other failure. */
const checkStep = async (page: Page, id: string): Promise<void> => {
  const status = await waitForStepSettled(page, id);
  const decision = decideStep(id, status, allowlist, false);
  if (decision.kind === 'ok' || decision.kind === 'missing-allowed') return;
  if (decision.kind === 'fail') throw new Error(`live-smoke: ${decision.reason}`);

  // `decision.kind === 'retry-unavailable'`: a network-fetch failure
  // against the live origin, not a verification failure -- wait, reload,
  // and look once more. A reload resets the whole SPA (every step, not
  // just this one) back to "loading", so the ready-wait/Findings-expand
  // sequence must repeat in full before this step's attribute can be read
  // again.
  console.log(`live-smoke: Findings step "${id}" is "unavailable"; waiting ${UNAVAILABLE_RETRY_DELAY_MS / 1000}s, then reloading for the one allowed recheck.`);
  await sleep(UNAVAILABLE_RETRY_DELAY_MS);
  await page.reload({ waitUntil: 'domcontentloaded', timeout: READY_TIMEOUT_MS }).catch(() => {
    throw new Error(`live-smoke: reload for Findings step "${id}"'s unavailable-recheck failed to load.`);
  });
  await waitForReadyAndExpandFindings(page);

  const retryStatus = await waitForStepSettled(page, id);
  const retryDecision = decideStep(id, retryStatus, allowlist, true);
  if (retryDecision.kind === 'ok' || retryDecision.kind === 'missing-allowed') return;
  // `retryDecision.kind` is always `'fail'` here (`decideStep` never
  // returns `'retry-unavailable'` when `isRetry` is `true`).
  throw new Error(`live-smoke: ${(retryDecision as { reason: string }).reason}`);
};

/** `#graph-lab`: loads without a page/console error and shows its default idle text, never clicking Connect (no network call). */
const checkGraphLabRoute = async (page: Page): Promise<void> => {
  let errorCount = 0;
  page.on('pageerror', () => {
    errorCount += 1;
  });
  page.on('console', (message) => {
    if (message.type() === 'error') errorCount += 1;
  });

  // Never build this URL by string-concatenating anything printable
  // (nothing here is logged), and never navigate through `expect`/
  // `page.goto`'s own thrown-error text, which can embed the URL verbatim
  // -- same discipline as the initial `deployUrl` navigation below.
  const withTrailingSlash = deployUrl.endsWith('/') ? deployUrl : `${deployUrl}/`;
  await page.goto(`${withTrailingSlash}#graph-lab`, { waitUntil: 'domcontentloaded', timeout: READY_TIMEOUT_MS }).catch(() => {
    throw new Error('live-smoke: failed to load the #graph-lab route.');
  });

  const idle = page.locator('.graph-lab section.panel.empty p.subtle');
  await waitUntil(
    async () => ((await idle.textContent().catch(() => null)) ?? '').trim() === GRAPH_LAB_IDLE_TEXT,
    STEP_TIMEOUT_MS,
    'live-smoke: #graph-lab did not show its default idle text.'
  );

  if (errorCount > 0) {
    throw new Error(`live-smoke: #graph-lab reported ${errorCount} page/console error(s).`);
  }
};

const main = async (): Promise<void> => {
  console.log(
    allowlist.length === 0
      ? 'live-smoke: smoke allowlist (steps allowed to be "missing"): (none)'
      : `live-smoke: smoke allowlist (steps allowed to be "missing"): ${JSON.stringify(allowlist)}`
  );

  const browser = await chromium.launch();
  try {
    const page = await browser.newPage();
    // Never echo the URL: any thrown error below (e.g. Playwright's own
    // navigation-failure message) may otherwise include it verbatim.
    await page.goto(deployUrl, { waitUntil: 'domcontentloaded', timeout: READY_TIMEOUT_MS }).catch(() => {
      throw new Error('live-smoke: failed to load the deploy target.');
    });

    await waitForReadyAndExpandFindings(page);

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

    // `02-release.md`'s extension: every Findings step, by id, until each
    // settles and passes `decideStep`'s rules. The DOM's real `li.step`
    // count is cross-checked against `expectedStepIds` first -- a step
    // landing unclassified in "Other findings" (which `main`'s own unit
    // test already forbids -- `tests/unit/findings-sections.test.ts`) would
    // otherwise silently never be checked here, since it has no
    // `FINDING_SECTIONS` entry to derive its id from.
    const domStepCount = await page.locator('section.findings li.step').count();
    if (domStepCount !== expectedStepIds.length) {
      throw new Error(
        `live-smoke: Findings rendered ${domStepCount} step(s) but FINDING_SECTIONS declares ${expectedStepIds.length}; an unclassified step may have landed in "Other findings".`
      );
    }
    for (const id of expectedStepIds) {
      await checkStep(page, id);
    }
    console.log(`live-smoke: all ${expectedStepIds.length} Findings steps settled ok (or allowlisted missing).`);

    await checkGraphLabRoute(page);
    console.log('live-smoke: #graph-lab idle state OK, no page/console errors.');

    console.log('live-smoke: passed (ready, Findings step 1, ledger, every Findings step, #graph-lab).');
  } finally {
    await browser.close();
  }
};

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : 'live-smoke: failed.');
  process.exit(1);
});
