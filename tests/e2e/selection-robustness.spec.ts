import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { expect, test } from '@playwright/test';
import { expandFindingsPanel, publicDataDir, startOrResumeButton, statusRegion, waitForReady, waitForTick } from './arena-test-helpers';

/**
 * WP3 of `.agents/plans/selection-robustness`: e2e coverage for the
 * Findings panel's selection-robustness step, grouped into the "Does it
 * generalize?" question section by WP1 of `.agents/plans/consolidated-release`
 * (`01-findings-sections.md`), alongside the task-generality step. Split
 * into its own file rather than added to `tests/e2e/findings.spec.ts` or
 * `tests/e2e/arena.spec.ts` (both already at/over this repo's thermo-review
 * 1000-line threshold) -- the same "split out a self-contained block"
 * precedent `tests/e2e/task-generality.spec.ts` already establishes.
 * `/fly/` subpath coverage lives in `tests/e2e/subpath.spec.ts` per that
 * file's own established pattern (it is the only spec `npm run test:subpath`
 * actually runs).
 */

test.describe('Findings panel (selection robustness)', () => {
  test('the selection-robustness step states the real overall verdicts, generated from the verified artifact', async ({ page }) => {
    await page.goto('/');
    await waitForReady(page);
    await expandFindingsPanel(page);

    const step = page.locator('section.findings li.step[data-step-id="selection-robustness"]');
    await expect(step).toContainText(/selection robustness/i);
    await expect(step).toContainText(/authored/i);

    // The real shipped artifact's own overall verdicts, not a hardcoded
    // placeholder (mirrors task-generality.spec.ts's own "assert the real
    // shipped numbers" discipline): robust to size holds across the
    // smaller/larger bridge-population sizes, but neither the
    // random-bridge (method) nor alt-sensory-mapping (channel mapping)
    // selection is robust -- see `.agents/plans/selection-robustness/00-overview.md`'s
    // predeclared aggregation rule and `docs/selection-robustness-report.md`
    // for the full per-selection reasoning.
    await expect(step).toContainText(/robust to size is true/);
    await expect(step).toContainText(/robust to method is false/);
    await expect(step).toContainText(/the channel-mapping result is false/);

    const sentenceEl = step.locator('p.sentence');
    await expect(sentenceEl).toHaveText(/under this model\.$/);
    const sentenceText = (await sentenceEl.textContent()) ?? '';
    expect(sentenceText.length).toBeLessThan(400);

    // Per-selection detail lives in its own screen-reader-friendly
    // <ul>/<li> list, one item per selection -- four predeclared
    // alternative selections, never folded into the summary sentence.
    const perSelectionItems = step.locator('ul.per-task > li');
    await expect(perSelectionItems).toHaveCount(4);

    // random-bridge's null holds narrowly and its explanation does not
    // replicate; its pathway is degenerate (P's search needed zero swaps),
    // stated as a mechanism, never a bare "degenerate" tag.
    const randomBridgeItem = perSelectionItems.filter({ hasText: 'random-bridge' });
    await expect(randomBridgeItem).toContainText('null holds');
    await expect(randomBridgeItem).toContainText('explanation does not replicate');
    await expect(randomBridgeItem).toContainText(/pathway P\/Q degenerate, not categorized/);
    await expect(randomBridgeItem).toContainText(/0 swaps/);

    // alt-sensory-mapping: the null holds and the explanation replicates,
    // but the pathway is not-supported -- the key result of this study.
    const altSensoryMappingItem = perSelectionItems.filter({ hasText: 'alt-sensory-mapping' });
    await expect(altSensoryMappingItem).toContainText('null holds');
    await expect(altSensoryMappingItem).toContainText('explanation replicates');
    await expect(altSensoryMappingItem).toContainText('pathway not-supported');

    // smaller and larger are both fully categorized and pathway-supported.
    const smallerItem = perSelectionItems.filter({ hasText: 'smaller' }).filter({ hasNotText: 'alt-sensory' });
    await expect(smallerItem).toContainText('pathway supported');
    const largerItem = perSelectionItems.filter({ hasText: 'larger' });
    await expect(largerItem).toContainText('pathway supported');

    const reportLink = step.getByRole('link', { name: /report/i });
    await expect(reportLink).toHaveAttribute(
      'href',
      'https://github.com/mattsp1290/flyarena/blob/main/docs/selection-robustness-report.md'
    );
    const jsonLink = step.getByRole('link', { name: 'Pinned JSON' });
    await expect(jsonLink).toHaveAttribute('href', '/data/selection-robustness-v1.json');
  });
});

test.describe('selection-robustness hash-mismatch integrity check', () => {
  test('a tampered selection-robustness-v1.json shows an honest verification-failure line on the selection-robustness step only, while the rest of the panel and the experiment (Start included) keep working', async ({
    page
  }) => {
    const original = readFileSync(resolve(publicDataDir, 'selection-robustness-v1.json'));
    const tampered = Buffer.from(original);
    tampered[10] ^= 0xff;

    await page.route('**/data/selection-robustness-v1.json', (route) =>
      route.fulfill({ status: 200, contentType: 'application/json', body: tampered })
    );

    await page.goto('/');
    await waitForReady(page);
    await expandFindingsPanel(page);

    const panel = page.locator('section.findings');
    const step = (id: string) => panel.locator(`li.step[data-step-id="${id}"]`);
    const selectionRobustness = step('selection-robustness');
    await expect(selectionRobustness).toContainText(/failed verification/i);
    await expect(page.locator('body')).not.toContainText(/robust to size is true/);

    // Every other step is unaffected -- still its own real templated sentence.
    const rewiringNull = step('rewiring-null');
    const taskGenerality = step('task-generality');
    await expect(rewiringNull.locator('p.sentence')).toHaveText(/under this model\.$/);
    await expect(rewiringNull).not.toContainText(/failed verification/i);
    await expect(taskGenerality.locator('p.sentence')).toHaveText(/under this model\.$/);
    await expect(taskGenerality).not.toContainText(/failed verification/i);

    // Start still works -- the panel is optional presentation, never a gate.
    await expect(startOrResumeButton(page)).toBeEnabled();
    await startOrResumeButton(page).click();
    await waitForTick(page, 10);
    await expect(statusRegion(page)).toHaveText('running');
  });

  test('a missing selectionRobustness manifest entry shows "Not yet published" on the selection-robustness step, with no failure message, while the rest of the panel keeps working', async ({
    page
  }) => {
    await page.route('**/data/malecns-arena-v1.manifest.json', async (route) => {
      const response = await route.fetch();
      const manifest = (await response.json()) as { selectionRobustness?: unknown };
      delete manifest.selectionRobustness;
      await route.fulfill({ response, json: manifest });
    });

    await page.goto('/');
    await waitForReady(page);
    await expandFindingsPanel(page);

    const panel = page.locator('section.findings');
    const step = (id: string) => panel.locator(`li.step[data-step-id="${id}"]`);
    const selectionRobustness = step('selection-robustness');
    await expect(selectionRobustness).toContainText(/not yet published/i);
    await expect(selectionRobustness).not.toContainText(/failed verification/i);

    const taskGenerality = step('task-generality');
    await expect(taskGenerality.locator('p.sentence')).toHaveText(/under this model\.$/);
    await expect(startOrResumeButton(page)).toBeEnabled();
  });
});
