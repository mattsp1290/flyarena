import { expect, test, type Page } from '@playwright/test';

/**
 * Regression coverage for the root-absolute-asset-URL bug: production is
 * deployed under the non-root `/fly/` base path (`vite build --base /fly/`,
 * via `scripts/deploy.sh`; see `.agents/deployment.md`), but
 * `loadArenaArtifacts`'s old default and `LedgerPanel`'s provenance links
 * hard-coded a root-absolute `/data/...` URL. Under `/fly/` that fetched
 * `/data/malecns-arena-v1.manifest.json` from the origin root instead of
 * `/fly/data/...`, 404'd, and drove the experiment straight to the `error`
 * state — Start never enabled. `tests/e2e/arena.spec.ts` never caught this
 * because `playwright.config.ts`'s original single webServer always
 * built/previewed with the default root base.
 *
 * This spec runs the same asset-load-to-running path and the ledger's
 * provenance links against a *second* build/preview
 * (`npm run build:fly`/`preview:fly`, wired as the `fly-base` Playwright
 * project in `playwright.config.ts`) served at `http://127.0.0.1:4174/fly/`.
 * It's deliberately a small slice of `arena.spec.ts`'s full coverage, not a
 * duplicate of the whole suite — the base path only affects the handful of
 * asset-URL call sites `src/lib/paths.ts#publicAssetUrl` now funnels
 * through, not per-tick simulation behavior, which `arena.spec.ts` already
 * covers thoroughly against the root base.
 */

const statusRegion = (page: Page) => page.locator('[role="status"]');
const startOrResumeButton = (page: Page) => page.locator('.button-row button').nth(0);

const waitForReady = (page: Page): Promise<void> =>
  expect(statusRegion(page)).toHaveText('ready', { timeout: 20_000 });

/**
 * Navigates to this project's `baseURL` (`http://127.0.0.1:4174/fly/`)
 * itself, not a sibling path under it. This is deliberately `page.goto('')`
 * rather than the `page.goto('/')` every other spec in this suite uses:
 * Playwright resolves a relative navigation via the WHATWG `URL`
 * constructor, and a leading `/` is an *absolute-path* reference that
 * replaces baseURL's own path entirely — `new URL('/', 'http://h/fly/')` is
 * `http://h/`, silently dropping back to the origin root and defeating the
 * whole point of this spec. An empty string is not an absolute-path
 * reference, so it resolves to baseURL unchanged:
 * `new URL('', 'http://h/fly/')` is `http://h/fly/`.
 */
const gotoFlyBase = (page: Page): Promise<unknown> => page.goto('');

test.describe('non-root deployment base (/fly/)', () => {
  test('asset load -> ready -> Start -> running, with every /data request served under /fly/data/', async ({
    page
  }) => {
    const dataRequestUrls: string[] = [];
    page.on('request', (request) => {
      const url = request.url();
      if (url.includes('/data/')) dataRequestUrls.push(url);
    });

    await gotoFlyBase(page);
    await waitForReady(page);
    await expect(startOrResumeButton(page)).toBeEnabled();
    await expect(startOrResumeButton(page)).toHaveText('Start');

    await startOrResumeButton(page).click();
    await expect(statusRegion(page)).toHaveText('running');

    // The manifest and both compressed graph artifacts must all have been
    // fetched to get this far — assert every one of those requests actually
    // went to `/fly/data/...`, not `/data/...` at the origin root.
    expect(dataRequestUrls.length).toBeGreaterThanOrEqual(3);
    for (const url of dataRequestUrls) {
      expect(url).toMatch(/^http:\/\/127\.0\.0\.1:4174\/fly\/data\//);
    }
  });

  test('ledger provenance links point at /fly/data/, not the origin root', async ({ page }) => {
    await gotoFlyBase(page);
    await waitForReady(page);

    await expect(page.getByRole('link', { name: 'Compiled artifact manifest (JSON)' })).toHaveAttribute(
      'href',
      '/fly/data/malecns-arena-v1.manifest.json'
    );
    await expect(page.getByRole('link', { name: 'Compiler ledger (JSON)' })).toHaveAttribute(
      'href',
      '/fly/data/malecns-arena-v1.ledger.json'
    );
  });
});
