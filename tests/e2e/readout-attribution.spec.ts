import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { expect, test } from '@playwright/test';
import { expandFindingsPanel, publicDataDir, startOrResumeButton, statusRegion, waitForReady, waitForTick } from './arena-test-helpers';

/**
 * WP3 of `.agents/plans/readout-attribution`: e2e coverage for the
 * Findings panel's readout-attribution step, grouped into the "What do
 * trained readouts do?" question section (`src/lib/findings/sections.ts`),
 * alongside the trained-null and trained-interventions steps. Split into
 * its own file rather than added to `tests/e2e/findings.spec.ts` (already
 * over this repo's thermo-review 1000-line threshold) -- the same
 * "split out a self-contained block" precedent
 * `tests/e2e/selection-robustness.spec.ts`/`tests/e2e/task-generality.spec.ts`
 * already establish. `/fly/` subpath coverage lives in
 * `tests/e2e/subpath.spec.ts` per that file's own established pattern (it
 * is the only spec `npm run test:subpath` actually runs).
 */

test.describe('Findings panel (readout attribution)', () => {
  test('the readout-attribution step states the trained decoder, "under this model", and the real (inconclusive) H1-H3 outcomes, generated from the verified artifact', async ({
    page
  }) => {
    await page.goto('/');
    await waitForReady(page);
    await expandFindingsPanel(page);

    const step = page.locator('section.findings li.step[data-step-id="readout-attribution"]');
    await expect(step).toContainText(/readout attribution/i);
    await expect(step).toContainText(/trained/i);

    // The real shipped WP2 results (`docs/readout-attribution-report.md`):
    // every hypothesis came back inconclusive on this model's real data --
    // asserted here, not a hardcoded placeholder, and the sentence must read
    // as honestly "inconclusive", never a stronger "not consistent"/
    // "different"/"used more than" claim an inconclusive result never
    // licenses (the task's own explicit hedging requirement).
    const sentenceEl = step.locator('p.sentence');
    await expect(sentenceEl).toHaveText(/under this model\.$/);
    await expect(sentenceEl).toContainText('trained decoder');
    await expect(sentenceEl).toContainText('(H1: inconclusive)');
    await expect(sentenceEl).toContainText('(H2: inconclusive)');
    await expect(sentenceEl).toContainText('(H3: inconclusive)');
    await expect(sentenceEl).not.toContainText('not consistent with routing around');
    await expect(sentenceEl).not.toContainText('is different between');

    // Per-hypothesis detail lives in its own screen-reader-friendly
    // <ul>/<li> list -- three hypotheses plus a coverage line.
    const detailItems = step.locator('ul.detail-list > li');
    await expect(detailItems).toHaveCount(4);
    const h1Item = detailItems.filter({ hasText: 'H1' });
    await expect(h1Item).toContainText('inconclusive');
    const coverageItem = detailItems.filter({ hasText: 'Coverage' });
    await expect(coverageItem).toContainText('23 default-task readouts');
    await expect(coverageItem).toContainText('not included');

    const reportLink = step.getByRole('link', { name: /report/i });
    await expect(reportLink).toHaveAttribute(
      'href',
      'https://github.com/mattsp1290/flyarena/blob/main/docs/readout-attribution-report.md'
    );
    const jsonLink = step.getByRole('link', { name: 'Pinned JSON' });
    await expect(jsonLink).toHaveAttribute('href', '/data/readout-attribution-v1.json');
  });
});

test.describe('readout-attribution hash-mismatch integrity check', () => {
  test('a tampered readout-attribution-v1.json shows an honest verification-failure line on the readout-attribution step only, while the rest of the panel and the experiment (Start included) keep working', async ({
    page
  }) => {
    const original = readFileSync(resolve(publicDataDir, 'readout-attribution-v1.json'));
    const tampered = Buffer.from(original);
    tampered[10] ^= 0xff;

    await page.route('**/data/readout-attribution-v1.json', (route) =>
      route.fulfill({ status: 200, contentType: 'application/json', body: tampered })
    );

    await page.goto('/');
    await waitForReady(page);
    await expandFindingsPanel(page);

    const panel = page.locator('section.findings');
    const step = (id: string) => panel.locator(`li.step[data-step-id="${id}"]`);
    const readoutAttribution = step('readout-attribution');
    await expect(readoutAttribution).toContainText(/failed verification/i);
    await expect(readoutAttribution).not.toContainText(/inconclusive/i);

    // Every other step is unaffected -- still its own real templated sentence.
    const trainedInterventions = step('trained-interventions');
    const taskGenerality = step('task-generality');
    await expect(trainedInterventions.locator('p.sentence')).toHaveText(/under this model\.$/);
    await expect(trainedInterventions).not.toContainText(/failed verification/i);
    await expect(taskGenerality.locator('p.sentence')).toHaveText(/under this model\.$/);
    await expect(taskGenerality).not.toContainText(/failed verification/i);

    // Start still works -- the panel is optional presentation, never a gate.
    await expect(startOrResumeButton(page)).toBeEnabled();
    await startOrResumeButton(page).click();
    await waitForTick(page, 10);
    await expect(statusRegion(page)).toHaveText('running');
  });

  test('a missing readoutAttribution manifest entry shows "Not yet published" on the readout-attribution step, with no failure message, while the rest of the panel keeps working', async ({
    page
  }) => {
    await page.route('**/data/malecns-arena-v1.manifest.json', async (route) => {
      const response = await route.fetch();
      const manifest = (await response.json()) as { readoutAttribution?: unknown };
      delete manifest.readoutAttribution;
      await route.fulfill({ response, json: manifest });
    });

    await page.goto('/');
    await waitForReady(page);
    await expandFindingsPanel(page);

    const panel = page.locator('section.findings');
    const step = (id: string) => panel.locator(`li.step[data-step-id="${id}"]`);
    const readoutAttribution = step('readout-attribution');
    await expect(readoutAttribution).toContainText(/not yet published/i);
    await expect(readoutAttribution).not.toContainText(/failed verification/i);

    const trainedInterventions = step('trained-interventions');
    await expect(trainedInterventions.locator('p.sentence')).toHaveText(/under this model\.$/);
    await expect(startOrResumeButton(page)).toBeEnabled();
  });
});
