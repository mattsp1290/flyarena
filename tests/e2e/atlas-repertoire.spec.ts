import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { expect, test } from '@playwright/test';
import { publicDataDir } from './arena-test-helpers';

/**
 * WP3 of `.agents/plans/repertoire-null`: e2e coverage for the atlas view's
 * one-line repertoire-comparison strip (`src/lib/atlas/Atlas.svelte`).
 * Split into its own spec file, not added to `tests/e2e/atlas.spec.ts`
 * (already over 1000 lines is `tests/e2e/arena.spec.ts`'s own reason for
 * splitting out `pathway-interventions.spec.ts`/`trained-decoder.spec.ts`;
 * this bean's own instruction is the same "keep every file under 1000
 * lines" discipline, applied before the file grows that large rather than
 * after) — mirrors `pathway-interventions.spec.ts`'s structure: a first
 * describe block against the real, shipped artifact, and a second against
 * deliberately tampered/missing/unavailable routes.
 */

const REAL_ARTIFACT_PATH = resolve(publicDataDir, 'behavior-repertoire-null-v1.json');

test.describe('atlas view: behavior-repertoire comparison strip', () => {
  test('shows the real comparison line under the discovery grid, generated from the verified artifact, linking to the report', async ({
    page
  }) => {
    const errors: string[] = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto('/#atlas');
    await expect(page.getByRole('group', { name: 'Select a discovered behavior' })).toBeVisible();

    const strip = page.locator('.repertoire-strip').first();
    await expect(strip).toBeVisible();
    // The real shipped artifact's own numbers (independently verified,
    // `.agents/plans/repertoire-null/00-overview.md`'s worked example) --
    // never a hardcoded placeholder.
    await expect(strip).toContainText(/Repertoire vs 20 rewirings: biological occupies 28 of 36 cells/i);
    await expect(strip).toContainText(/rewired median 29, range \d+–\d+/i);
    // Methodology review (Important): the category must never be stated
    // next to only `occupied` -- `qd` (the metric with the larger, less
    // ambiguous gap) and the joint basis are now always shown too.
    await expect(strip).toContainText(/qd 637 vs rewired median 763/i);
    await expect(strip).toContainText(/narrower on both metrics at search seed 1729/i);
    await expect(strip).toContainText(/not robust: seeds give 1729: narrower, 1730: narrower, 1731: typical, 1732: typical, 1733: typical/i);
    await expect(strip).toContainText(/Seeds 1730–1733 compare against only 5 seed-matched rewirings/i);

    const reportLink = strip.getByRole('link', { name: /full report/i });
    await expect(reportLink).toHaveAttribute(
      'href',
      'https://github.com/mattsp1290/flyarena/blob/main/docs/behavior-repertoire-null-report.md'
    );
    expect(errors).toEqual([]);
  });
});

test.describe('behavior-repertoire-null hash-mismatch and missing-entry integrity checks', () => {
  test('a tampered behavior-repertoire-null-v1.json shows an honest verification-failure line, while the atlas view itself keeps working', async ({
    page
  }) => {
    const original = readFileSync(REAL_ARTIFACT_PATH);
    const tampered = Buffer.from(original);
    tampered[10] ^= 0xff;

    await page.route('**/data/behavior-repertoire-null-v1.json', (route) =>
      route.fulfill({ status: 200, contentType: 'application/json', body: tampered })
    );

    await page.goto('/#atlas');
    const map = page.getByRole('group', { name: 'Select a discovered behavior' });
    await expect(map).toBeVisible();

    const strip = page.locator('.repertoire-strip.error-message');
    await expect(strip).toHaveText('Repertoire comparison failed verification');
    // The line itself must never render over unverified bytes.
    await expect(page.locator('body')).not.toContainText(/Repertoire vs \d+ rewirings/i);

    // The atlas view keeps working exactly as if the artifact were never routed.
    await expect(page.getByRole('region', { name: 'Selected behavior' })).toBeVisible();
    await map.getByRole('button').first().click();
    await expect(map.getByRole('button').first()).toHaveAttribute('aria-pressed', 'true');
  });

  /**
   * Thermo-methodology review (Important, I-2): the atlas-staleness
   * cross-check (`repertoireStale`, `Atlas.svelte`) had zero test coverage
   * at any level -- neither this file's own tamper/404/missing-entry cases
   * nor `tests/e2e/subpath.spec.ts` ever served a real, hash-valid
   * `behavior-repertoire-null-v1.json` next to a *different* (but still
   * self-consistent) atlas. This is that scenario: the repertoire artifact
   * is untouched (still verifies against the manifest), but the shipped
   * atlas is changed the same way `tests/e2e/atlas.spec.ts`'s own "rejects
   * a valid atlas changed between selection and worker execution" test
   * changes it (bump `source.runtime.seconds`, recompute its manifest so
   * the atlas itself still loads successfully) -- so `loaded.sha256` no
   * longer matches the repertoire artifact's `sources.atlasSha256`.
   */
  test('a repertoire artifact computed against a different (but still valid) atlas shows an honest verification-failure line, and never the live numbers', async ({
    page
  }) => {
    const nextAtlas = JSON.parse(readFileSync(resolve(publicDataDir, 'behavior-atlas-v1.json'), 'utf8'));
    nextAtlas.source.runtime.seconds += 1;
    const body = JSON.stringify(nextAtlas);
    await page.route('**/behavior-atlas-v1.manifest.json', (route) =>
      route.fulfill({
        json: {
          schemaVersion: 1,
          artifact: 'behavior-atlas-v1.json',
          sha256: createHash('sha256').update(body).digest('hex'),
          bytes: Buffer.byteLength(body)
        }
      })
    );
    await page.route('**/behavior-atlas-v1.json', (route) => route.fulfill({ body, contentType: 'application/json' }));

    await page.goto('/#atlas');
    const map = page.getByRole('group', { name: 'Select a discovered behavior' });
    // The atlas itself keeps loading and working -- this is a display-time
    // staleness check on the repertoire strip only, never a failure of the
    // atlas load path.
    await expect(map).toBeVisible();

    const strip = page.locator('.repertoire-strip.error-message');
    await expect(strip).toHaveText('Repertoire comparison failed verification');
    await expect(page.locator('body')).not.toContainText(/Repertoire vs \d+ rewirings/i);
  });

  test('a 404 for behavior-repertoire-null-v1.json shows a "could not be loaded" line, never a verification-failure message', async ({
    page
  }) => {
    await page.route('**/data/behavior-repertoire-null-v1.json', (route) => route.fulfill({ status: 404, body: '' }));
    await page.goto('/#atlas');
    await expect(page.getByRole('group', { name: 'Select a discovered behavior' })).toBeVisible();
    const strip = page.locator('.repertoire-strip.error-message');
    await expect(strip).toHaveText('Repertoire comparison could not be loaded');
  });

  test('a missing behaviorRepertoireNull manifest entry hides the strip entirely, with no message, while the atlas keeps working', async ({
    page
  }) => {
    // Guarded to fire only once: a second matching request (e.g. a browser
    // prefetch/revalidation, unrelated to `Atlas.svelte`'s own single
    // `onMount` fetch) would otherwise race `route.fetch()`'s response
    // object against page teardown once the assertions below already
    // resolved (`apiResponse.json: Response has been disposed` — a
    // Playwright-side artifact of a stray duplicate request, not a real
    // app bug), so any further request is simply passed through unmodified.
    let handled = false;
    await page.route('**/data/malecns-arena-v1.manifest.json', async (route) => {
      if (handled) {
        await route.continue();
        return;
      }
      handled = true;
      const response = await route.fetch();
      const manifest = (await response.json()) as { behaviorRepertoireNull?: unknown };
      delete manifest.behaviorRepertoireNull;
      await route.fulfill({ response, json: manifest });
    });

    await page.goto('/#atlas');
    const map = page.getByRole('group', { name: 'Select a discovered behavior' });
    await expect(map).toBeVisible();
    await expect(page.locator('.repertoire-strip')).toHaveCount(0);
    await expect(page.getByRole('region', { name: 'Selected behavior' })).toBeVisible();
  });

  test('a corrupt/unreachable manifest never blocks the atlas discovery grid, and the strip is simply absent', async ({ page }) => {
    await page.route('**/data/malecns-arena-v1.manifest.json', (route) => route.fulfill({ status: 500, body: '' }));
    await page.goto('/#atlas');
    await expect(page.getByRole('group', { name: 'Select a discovered behavior' })).toBeVisible();
    const strip = page.locator('.repertoire-strip.error-message');
    await expect(strip).toHaveText('Repertoire comparison could not be loaded');
  });
});
