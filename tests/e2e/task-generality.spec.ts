import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { expect, test } from '@playwright/test';
import { expandFindingsPanel, publicDataDir, startOrResumeButton, statusRegion, waitForReady, waitForTick } from './arena-test-helpers';

/**
 * WP4 of `.agents/plans/task-generality`: e2e coverage for the Findings
 * panel's task-generality step (inserted before "Behavior repertoire",
 * which shifts to step 8 of 8). Split into its own file rather than added
 * to `tests/e2e/findings.spec.ts` or `tests/e2e/arena.spec.ts` (both
 * already at/over this repo's thermo-review 1000-line threshold) -- the
 * same "split out a self-contained block" precedent
 * `tests/e2e/pathway-interventions.spec.ts`/`tests/e2e/trained-decoder.spec.ts`
 * already establish. `/fly/` subpath coverage lives in
 * `tests/e2e/subpath.spec.ts` per that file's own established pattern (it
 * is the only spec `npm run test:subpath` actually runs).
 */

test.describe('Findings panel (task generality)', () => {
  test('step 7 states the real authored/trained overall verdicts, generated from the verified artifact', async ({ page }) => {
    await page.goto('/');
    await waitForReady(page);
    await expandFindingsPanel(page);

    const step = page.locator('section.findings li.step').nth(6);
    await expect(step).toContainText(/7\. task generality/i);
    await expect(step).toContainText(/both decoders/i);
    // The real shipped artifact's own overall verdicts, not a hardcoded
    // placeholder (mirrors arena.spec.ts's own "assert the real shipped
    // numbers" discipline): authored is general (3 of 4 non-degenerate
    // tasks hold and generalize), trained is task-dependent (no task
    // reaches a robust pathway-supported category).
    await expect(step).toContainText(/authored: general/);
    await expect(step).toContainText(/trained: task-dependent/);
    // Thermo review (Important, both reviewers): the summary sentence
    // itself stays short (no per-task/per-seed enumeration); "no specific
    // effect" is defined once in the sentence, not repeated per task.
    const sentenceEl = step.locator('p.sentence');
    await expect(sentenceEl).toHaveText(/under this model\.$/);
    const sentenceText = (await sentenceEl.textContent()) ?? '';
    expect(sentenceText.length).toBeLessThan(400);
    expect(sentenceText).toContain('"no specific effect" means neither pathway-supported nor edge-class holds');

    // Per-task detail lives in its own screen-reader-friendly <ul>/<li>
    // list, one item per task. "not robust" and the dissenting seed(s) are
    // stated inline there (never only in a footnote).
    const perTaskItems = step.locator('ul.per-task > li');
    await expect(perTaskItems).toHaveCount(4);
    await expect(step.locator('ul.per-task')).toContainText(/not robust \(seed \d+ pathway-supported\)/);
    // no-movement's degenerate authored result is its own list item, never
    // merged with anything else.
    const noMovementItem = perTaskItems.filter({ hasText: 'no-movement' });
    await expect(noMovementItem).toContainText('degenerate');

    const reportLink = step.getByRole('link', { name: /report/i });
    await expect(reportLink).toHaveAttribute('href', 'https://github.com/mattsp1290/flyarena/blob/main/docs/task-generality-report.md');
    const jsonLink = step.getByRole('link', { name: 'Pinned JSON' });
    await expect(jsonLink).toHaveAttribute('href', '/data/task-generality-v1.json');
  });
});

test.describe('task-generality hash-mismatch integrity check', () => {
  test('a tampered task-generality-v1.json shows an honest verification-failure line on step 7 only, while the rest of the panel and the experiment (Start included) keep working', async ({
    page
  }) => {
    const original = readFileSync(resolve(publicDataDir, 'task-generality-v1.json'));
    const tampered = Buffer.from(original);
    tampered[10] ^= 0xff;

    await page.route('**/data/task-generality-v1.json', (route) =>
      route.fulfill({ status: 200, contentType: 'application/json', body: tampered })
    );

    await page.goto('/');
    await waitForReady(page);
    await expandFindingsPanel(page);

    const panel = page.locator('section.findings');
    const step7 = panel.locator('li.step').nth(6);
    await expect(step7).toContainText(/failed verification/i);
    await expect(page.locator('body')).not.toContainText(/authored: general/);

    // Every other step is unaffected -- still its own real templated sentence.
    const step1 = panel.locator('li.step').nth(0);
    const step8 = panel.locator('li.step').nth(7);
    await expect(step1.locator('p.sentence')).toHaveText(/under this model\.$/);
    await expect(step1).not.toContainText(/failed verification/i);
    await expect(step8.locator('p.sentence')).toHaveText(/under this model\.$/);
    await expect(step8).not.toContainText(/failed verification/i);

    // Start still works -- the panel is optional presentation, never a gate.
    await expect(startOrResumeButton(page)).toBeEnabled();
    await startOrResumeButton(page).click();
    await waitForTick(page, 10);
    await expect(statusRegion(page)).toHaveText('running');
  });

  test('a missing taskGenerality manifest entry shows "Not yet published" on step 7, with no failure message, while the rest of the panel keeps working', async ({
    page
  }) => {
    await page.route('**/data/malecns-arena-v1.manifest.json', async (route) => {
      const response = await route.fetch();
      const manifest = (await response.json()) as { taskGenerality?: unknown };
      delete manifest.taskGenerality;
      await route.fulfill({ response, json: manifest });
    });

    await page.goto('/');
    await waitForReady(page);
    await expandFindingsPanel(page);

    const panel = page.locator('section.findings');
    const step7 = panel.locator('li.step').nth(6);
    await expect(step7).toContainText(/not yet published/i);
    await expect(step7).not.toContainText(/failed verification/i);

    const step8 = panel.locator('li.step').nth(7);
    await expect(step8.locator('p.sentence')).toHaveText(/under this model\.$/);
    await expect(startOrResumeButton(page)).toBeEnabled();
  });
});
