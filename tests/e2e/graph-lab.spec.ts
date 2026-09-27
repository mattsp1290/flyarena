import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';

/**
 * `.agents/plans/graph-lab/03-frontend-route.md`: "e2e `tests/e2e/graph-lab.spec.ts`
 * runs against a locally started container (skipped unless
 * `GRAPH_LAB_E2E_URL` and a token are provided), covering submit, poll,
 * cancel, and render for each kind."
 *
 * The container-backed tests below are deliberately **lesion-only**: a
 * multi-hour GPU training job (`flyarena-train`, bean `s8z8`) is running on
 * this same DGX Spark while this route is implemented, and an `atlas`
 * job here would contend with it for the GPU (it is refused outright as
 * "GPU busy" if free memory drops below 2 GiB -- `02-job-engines.md`'s
 * atlas bound -- but even attempting one risks stealing cycles from that
 * job). `swapset` needs no GPU, but is left out of this first e2e pass too
 * to keep the live-container surface area minimal and fast; its request/
 * response shapes are already covered by `test_engines.py`'s
 * `SwapsetEngineTests` (backend) and `tests/GraphLab.test.ts`'s render
 * tests (frontend), so this gap is a deliberate scope cut, not an
 * untested path.
 *
 * Own file (not `tests/e2e/lab.spec.ts`, and not `tests/e2e/arena.spec.ts`,
 * which is already over the project's 1000-line-per-file review limit).
 */

const ENDPOINT = process.env.GRAPH_LAB_E2E_URL ?? 'http://127.0.0.1:8766';
const TOKEN = process.env.GRAPH_LAB_E2E_TOKEN;

test.describe('graph lab: live container (lesion only)', () => {
  test.skip(!TOKEN, 'Build/run the graph-lab container and set GRAPH_LAB_E2E_TOKEN (see this file\'s own doc comment)');

  test('connects, submits a lesion job, polls to completion, and renders its provenance', async ({ page }, testInfo) => {
    test.setTimeout(180_000);
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await page.goto('/#graph-lab');
    await page.getByRole('link', { name: /05 Real-graph lab/ }).waitFor();

    await page.locator('.graph-lab').getByLabel('Backend URL').fill(ENDPOINT);
    await page.locator('.graph-lab').getByLabel('Access token').fill(TOKEN!);
    await page.locator('.graph-lab').getByRole('button', { name: 'Connect' }).click();
    await expect(page.locator('.graph-lab')).toContainText('Bundle SHA-256', { timeout: 20_000 });

    await page.locator('.graph-lab').getByRole('tab', { name: 'Lesion sweep' }).click();
    await page.locator('.graph-lab textarea').fill('7');
    await page.locator('.graph-lab').getByLabel('Seed count').fill('4');
    await page.locator('.graph-lab').getByLabel('Ticks').fill('300');
    await page.locator('.graph-lab').getByRole('button', { name: 'Submit job' }).click();

    await expect(page.locator('.graph-lab').getByRole('status')).toContainText('completed', { timeout: 120_000 });
    await expect(page.locator('.graph-lab .provenance-label')).toHaveText('Computed on DGX (private, not published)');
    await expect(page.locator('.graph-lab .provenance')).toContainText('graph biological');

    const downloadEvent = page.waitForEvent('download');
    await page.locator('.graph-lab').getByRole('button', { name: /Export evidence/ }).click();
    const download = await downloadEvent;
    const path = await download.path();
    const result = JSON.parse(await readFile(path!, 'utf8'));
    expect(result.sets).toHaveLength(1);
    expect(JSON.stringify(result)).not.toContain(TOKEN!);

    await page.setViewportSize({ width: 390, height: 844 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.screenshot({ path: testInfo.outputPath('graph-lab-mobile.png'), fullPage: true });

    expect(errors).toEqual([]);
  });

  test('cancels an in-flight lesion job', async ({ page }) => {
    test.setTimeout(60_000);
    await page.goto('/#graph-lab');
    await page.locator('.graph-lab').getByLabel('Backend URL').fill(ENDPOINT);
    await page.locator('.graph-lab').getByLabel('Access token').fill(TOKEN!);
    await page.locator('.graph-lab').getByRole('button', { name: 'Connect' }).click();
    await expect(page.locator('.graph-lab')).toContainText('Bundle SHA-256', { timeout: 20_000 });

    await page.locator('.graph-lab').getByRole('tab', { name: 'Lesion sweep' }).click();
    await page.locator('.graph-lab textarea').fill('7');
    await page.locator('.graph-lab').getByLabel('Seed count').fill('100');
    await page.locator('.graph-lab').getByLabel('Ticks').fill('1800');
    await page.locator('.graph-lab').getByRole('button', { name: 'Submit job' }).click();

    await expect(page.locator('.graph-lab').getByRole('button', { name: 'Cancel job' })).toBeEnabled({
      timeout: 20_000
    });
    await page.locator('.graph-lab').getByRole('button', { name: 'Cancel job' }).click();
    await expect(page.locator('.graph-lab').getByRole('status')).toContainText('cancelled', { timeout: 30_000 });
  });
});

test.describe('graph lab: backend unavailable (no container needed)', () => {
  test('shows a clear "backend unavailable" state and leaves the rest of the site working', async ({ page }) => {
    await page.goto('/#graph-lab');
    await page.locator('.graph-lab').getByLabel('Backend URL').fill('http://127.0.0.1:1');
    await page.locator('.graph-lab').getByRole('button', { name: 'Connect' }).click();
    await expect(page.locator('.graph-lab').getByRole('alert')).toContainText('Backend unavailable', {
      timeout: 20_000
    });
    await expect(page.locator('.graph-lab').getByRole('tablist')).toHaveCount(0);

    // The rest of the site is unaffected by a graph-lab connection failure.
    await page.getByRole('link', { name: '01 Arena', exact: true }).click();
    await expect(page.getByRole('status')).toHaveText('ready');
  });
});
