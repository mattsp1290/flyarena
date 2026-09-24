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
 * project in `playwright.config.ts`) served under `/fly/`. It's
 * deliberately a small slice of `arena.spec.ts`'s full coverage, not a
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
 * Navigates to this project's own `baseURL` (`use.baseURL` in
 * `playwright.config.ts`'s `fly-base` project) itself, not a sibling path
 * under it. This is deliberately `page.goto('')` rather than the
 * `page.goto('/')` every other spec in this suite uses: Playwright resolves
 * a relative navigation via the WHATWG `URL` constructor, and a leading `/`
 * is an *absolute-path* reference that replaces `baseURL`'s own path
 * entirely — `new URL('/', 'http://h/fly/')` is `http://h/`, dropping back
 * to the origin root.
 *
 * `vite preview --base /fly/` (this spec's own dev/test server) happens to
 * 302-redirect a bare `GET /` to `/fly/` — confirmed directly by curling it
 * — so `page.goto('/')` would still *land* on the right final URL here even
 * after that URL-resolution mistake: the browser follows the redirect, and
 * a `toHaveURL(baseURL)` check alone can't tell a direct hit from a
 * followed redirect. Production's Apache config (`.agents/deployment.md`)
 * has no such redirect: it only redirects the exact path `/fly` (no
 * trailing slash) to `/fly/`, never `/` to `/fly/`. `page.goto('')` avoids
 * relying on this server's redirect at all, and the `redirectedFrom()`
 * check below (a round-2 review pass caught that `toHaveURL` alone doesn't
 * actually distinguish the two cases against this specific server) makes
 * that explicit and testable: it fails if navigation ever went through a
 * redirect to get here, not just if it landed somewhere else.
 */
const gotoFlyBase = async (page: Page, baseURL: string): Promise<void> => {
  const response = await page.goto('');
  expect(response, 'page.goto(\'\') must produce a navigation response').not.toBeNull();
  await expect(page).toHaveURL(baseURL);
  const redirectedFrom = response!.request().redirectedFrom();
  expect(
    redirectedFrom?.url() ?? null,
    'navigation must land on baseURL directly, not via a server redirect (see this function\'s doc comment)'
  ).toBeNull();
};

test.describe('non-root deployment base (/fly/)', () => {
  test('asset load -> ready -> Start -> running, with every /data request served under /fly/data/', async ({
    page,
    baseURL
  }) => {
    if (!baseURL) throw new Error('fly-base project must configure use.baseURL');
    const dataRequestUrls: string[] = [];
    page.on('request', (request) => {
      const url = request.url();
      if (url.includes('/data/')) dataRequestUrls.push(url);
    });

    await gotoFlyBase(page, baseURL);
    await waitForReady(page);
    await expect(startOrResumeButton(page)).toBeEnabled();
    await expect(startOrResumeButton(page)).toHaveText('Start');

    await startOrResumeButton(page).click();
    await expect(statusRegion(page)).toHaveText('running');

    // `loadArenaArtifacts` fetches exactly these three files (the manifest,
    // then the two compressed graph artifacts it names) and nothing else —
    // assert the exact set (not just a lower-bound count) so a future
    // refactor that drops or duplicates one of them is caught here, and
    // that every one of those requests actually went under this project's
    // own `/fly/data/...` base, not `/data/...` at the origin root.
    const requestedFilenames = dataRequestUrls.map((url) => new URL(url).pathname.split('/').pop()).sort();
    expect(requestedFilenames).toEqual(
      ['malecns-arena-v1-rewired-seed0.bin.gz', 'malecns-arena-v1.bin.gz', 'malecns-arena-v1.manifest.json'].sort()
    );
    for (const url of dataRequestUrls) {
      expect(url.startsWith(`${baseURL}data/`), url).toBe(true);
    }
  });

  test('ledger provenance links point at /fly/data/ and actually resolve there, not the origin root', async ({
    page,
    baseURL
  }) => {
    if (!baseURL) throw new Error('fly-base project must configure use.baseURL');
    await gotoFlyBase(page, baseURL);
    await waitForReady(page);

    // Derived from `baseURL` (e.g. '/fly/') rather than a hard-coded
    // '/fly/data/...' literal, so this stays correct if the project's base
    // path ever changes without needing a matching edit here.
    const basePath = new URL(baseURL).pathname;
    const manifestLink = page.getByRole('link', { name: 'Compiled artifact manifest (JSON)' });
    const ledgerLink = page.getByRole('link', { name: 'Compiler ledger (JSON)' });
    await expect(manifestLink).toHaveAttribute('href', `${basePath}data/malecns-arena-v1.manifest.json`);
    await expect(ledgerLink).toHaveAttribute('href', `${basePath}data/malecns-arena-v1.ledger.json`);

    // The `href` string alone doesn't prove the file is actually served
    // there — neither the app nor any other test in this spec ever fetches
    // the ledger JSON. Follow both links for real. A round-2 review pass
    // caught that `vite preview` answers *any* missing path under `/fly/`
    // with its SPA fallback (`index.html`, status 200, `text/html`) rather
    // than a 404 — confirmed directly by requesting a nonexistent path — so
    // asserting `status() === 200` alone would still pass if either file
    // were missing entirely. Asserting the JSON content type plus an actual
    // field from each file's real shape (`public/data/malecns-arena-v1.
    // {manifest,ledger}.json`) rules that out.
    // Requested via the same `basePath` already asserted above (not a fresh
    // `getAttribute` round-trip), so these two checks stay tied to the
    // exact hrefs just verified rather than independently re-deriving them.
    const manifestResponse = await page.request.get(new URL(`${basePath}data/malecns-arena-v1.manifest.json`, baseURL).href);
    expect(manifestResponse.status()).toBe(200);
    expect(manifestResponse.headers()['content-type']).toContain('application/json');
    const manifestBody = (await manifestResponse.json()) as { sourceDataset?: unknown };
    expect(typeof manifestBody.sourceDataset).toBe('string');

    const ledgerResponse = await page.request.get(new URL(`${basePath}data/malecns-arena-v1.ledger.json`, baseURL).href);
    expect(ledgerResponse.status()).toBe(200);
    expect(ledgerResponse.headers()['content-type']).toContain('application/json');
    const ledgerBody = (await ledgerResponse.json()) as { compileStats?: unknown };
    expect(typeof ledgerBody.compileStats).toBe('object');
  });
});
