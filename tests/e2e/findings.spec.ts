import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { expect, test } from '@playwright/test';
import {
  expandFindingsPanel,
  pauseButton,
  publicDataDir,
  startOrResumeButton,
  statusRegion,
  waitForReady,
  waitForTick
} from './arena-test-helpers';

/**
 * WP1 of `.agents/plans/findings-tour` (`01-findings-panel.md`): e2e
 * coverage for `FindingsPanel.svelte` — split into its own file rather than
 * added to `tests/e2e/arena.spec.ts` (already at/over this repo's
 * thermo-review 1000-line threshold on `origin/main`), the same "split out
 * a self-contained block" precedent `tests/e2e/pathway-interventions.spec.ts`/
 * `tests/e2e/trained-decoder.spec.ts` already establish. Subpath (`/fly/`)
 * coverage lives in `tests/e2e/subpath.spec.ts` per this WP's own plan.
 */

test.describe('Findings panel', () => {
  test('expanding walks all seven steps, rendered from the real verified artifacts, with resolvable provenance links', async ({
    page,
    request
  }) => {
    await page.goto('/');
    await waitForReady(page);

    await expandFindingsPanel(page);

    const panel = page.locator('section.findings');
    await expect(panel.getByRole('heading', { name: 'Findings under this model — not claims about the real fly' })).toBeVisible();

    // Step 1: Rewiring null -- the real shipped bioPercentile/null.n.
    const step1 = panel.locator('li.step').nth(0);
    await expect(step1).toContainText(/1\. rewiring null/i);
    await expect(step1).toContainText(/authored \(hand-written\) decoder/i);
    await expect(step1).toContainText(/0\.0th percentile/);
    await expect(step1).toContainText(/500 degree-preserving rewirings/);
    await expect(step1.locator('p.sentence')).toHaveText(/under this model\.$/);

    // Step 2: Mirrored decoder -- real data has both baseline and mirrored
    // bioPercentile at 0, so the "still leaves ... at the bottom" clause.
    const step2 = panel.locator('li.step').nth(1);
    await expect(step2).toContainText(/still leaves biological at the bottom/i);

    // Step 3: Explanation -- the real qualifying metrics, plain-named.
    const step3 = panel.locator('li.step').nth(2);
    await expect(step3).toContainText(/T:rightClearance->thrust/);
    await expect(step3).toContainText(/ρ = 0\.467/);

    // Step 4: Intervention (authored) -- the real category/modifier.
    const step4 = panel.locator('li.step').nth(3);
    await expect(step4).toContainText(/pathway-supported category/);
    await expect(step4).toContainText(/channel-specific modifier holding/);

    // Step 5: Trained null -- real per-seed spread (0th to 40th percentile).
    const step5 = panel.locator('li.step').nth(4);
    await expect(step5).toContainText(/trained decoder/i);
    await expect(step5).toContainText(/0\.0th percentile to 40\.0th percentile/);

    // Step 6: Trained interventions -- the fixed polarity sentence; the
    // real shipped data has authored=pathway-supported, trained (robust)
    // =no-specific-effect, so this must say "does not reproduce", never
    // "matches" (the bean's own "fix the step-6 polarity sentence" ask).
    // Rendered through the shared `describeTrainedCategory` helper (thermo
    // review, methodology I1/I2), never the raw enum slug, plus the
    // artifact's own disclosed reporting-convention caveat.
    const step6 = panel.locator('li.step').nth(5);
    await expect(step6).not.toContainText(/no-specific-effect/);
    await expect(step6).toContainText(/all 3 seeds agree: no specific effect \(neither pathway-supported nor edge-class/);
    await expect(step6).toContainText(/reporting convention adopted after the trained scores were known/);
    await expect(step6).toContainText(/does not reproduce/);
    await expect(step6).not.toContainText(/this matches/);

    // Step 7: Behavior repertoire -- always "Not yet published" in this WP.
    const step7 = panel.locator('li.step').nth(6);
    await expect(step7).toContainText(/not yet published/i);
    await expect(step7.getByRole('link', { name: /behavior atlas/i })).toHaveAttribute('href', '#atlas');

    // Every provenance link (pinned JSON + report) actually resolves.
    const jsonLinks = await panel.getByRole('link', { name: 'Pinned JSON' }).evaluateAll((links) =>
      links.map((link) => (link as HTMLAnchorElement).href)
    );
    expect(jsonLinks.length).toBeGreaterThan(0);
    for (const href of new Set(jsonLinks)) {
      const response = await request.get(href);
      expect(response.status(), href).toBe(200);
    }
    // (dual review, Important) Report links point at github.com/mattsp1290/flyarena
    // (`links.ts#githubDocUrl`) -- a real `request.get` there would make CI
    // network-dependent and flaky (offline runs, GitHub rate limits, or a
    // report doc that exists on this branch but not yet on remote `main`).
    // Assert the href's shape and that the cited doc actually exists in
    // this checkout instead -- same "resolvable" guarantee, no live fetch.
    const reportLinks = await panel.getByRole('link', { name: 'Report' }).evaluateAll((links) =>
      links.map((link) => (link as HTMLAnchorElement).href)
    );
    expect(reportLinks.length).toBeGreaterThan(0);
    for (const href of new Set(reportLinks)) {
      expect(href, href).toMatch(/^https:\/\/github\.com\/mattsp1290\/flyarena\/blob\/main\/docs\/[\w-]+\.md$/);
      const slug = href.split('/docs/')[1];
      // `publicDataDir` is `<repoRoot>/public/data`.
      expect(existsSync(resolve(publicDataDir, '../../docs', slug)), slug).toBe(true);
    }
  });

  test('a tampered pathway-interventions-v1.json fails only steps 4 and 6, while the rest of the panel and the experiment (Start included) keep working', async ({
    page
  }) => {
    const original = readFileSync(resolve(publicDataDir, 'pathway-interventions-v1.json'));
    const tampered = Buffer.from(original);
    tampered[10] ^= 0xff;

    await page.route('**/data/pathway-interventions-v1.json', (route) =>
      route.fulfill({ status: 200, contentType: 'application/json', body: tampered })
    );

    await page.goto('/');
    await waitForReady(page);
    await expandFindingsPanel(page);

    const panel = page.locator('section.findings');
    const step4 = panel.locator('li.step').nth(3);
    const step6 = panel.locator('li.step').nth(5);
    await expect(step4).toContainText(/failed verification/i);
    await expect(step6).toContainText(/failed verification/i);

    // Every other step is unaffected -- still its own real templated sentence.
    const step1 = panel.locator('li.step').nth(0);
    const step3 = panel.locator('li.step').nth(2);
    await expect(step1.locator('p.sentence')).toHaveText(/under this model\.$/);
    await expect(step1).not.toContainText(/failed verification/i);
    await expect(step3.locator('p.sentence')).toHaveText(/under this model\.$/);
    await expect(step3).not.toContainText(/failed verification/i);

    // Start still works -- the panel is optional presentation, never a gate.
    await expect(startOrResumeButton(page)).toBeEnabled();
    await startOrResumeButton(page).click();
    await waitForTick(page, 10);
    await expect(statusRegion(page)).toHaveText('running');
  });

  test('collapsed by default, and never blocks reaching ready', async ({ page }) => {
    await page.goto('/');
    await waitForReady(page);
    await expect(startOrResumeButton(page)).toBeEnabled();
    await expect(page.locator('section.findings ol.steps')).toHaveCount(0);
  });

  /**
   * (thermo review, maintainability Important) A real Tab-key session,
   * driving actual `document.activeElement`, not `fireEvent.click` on the
   * buttons directly (the unit tests' own coverage) — proves the guarantee
   * the code comment above `.stepper-controls` in `FindingsPanel.svelte`
   * actually makes: after `Next` moves focus to a step's heading, Tab no
   * longer walks through *every earlier step's* own links (the DOM-order
   * placement this branch fixed), only the *current* step's own provenance
   * links, in order, before reaching `Previous`/`Next`. Step 2 is used
   * because it has two provenance entries (four links) — the deepest case
   * in the panel — so this also pins down the real, narrower Tab count the
   * comment's prior "reaches Next directly" claim overstated.
   */
  test('real Tab order from a step heading walks that step\'s own provenance links, then Previous, then Next — never an earlier step\'s links', async ({
    page
  }) => {
    await page.goto('/');
    await waitForReady(page);
    await expandFindingsPanel(page);
    await page.getByRole('button', { name: /^next$/i }).click();

    const step2 = page.locator('section.findings li.step').nth(1);
    await expect(step2.locator('h3')).toBeFocused();

    const focusedSummary = async (): Promise<string> =>
      page.evaluate(() => {
        const el = document.activeElement;
        if (!el) return '(none)';
        return `${el.tagName}:${(el.textContent ?? '').trim().slice(0, 40)}`;
      });

    const jsonLinks = step2.getByRole('link', { name: 'Pinned JSON' });
    const reportLinks = step2.getByRole('link', { name: 'Report' });
    await expect(jsonLinks).toHaveCount(2);
    await expect(reportLinks).toHaveCount(2);

    // Four provenance links (one pair per source artifact), in DOM order.
    await page.keyboard.press('Tab');
    expect(await focusedSummary()).toContain('Pinned JSON');
    await page.keyboard.press('Tab');
    expect(await focusedSummary()).toContain('Report');
    await page.keyboard.press('Tab');
    expect(await focusedSummary()).toContain('Pinned JSON');
    await page.keyboard.press('Tab');
    expect(await focusedSummary()).toContain('Report');

    // Then, and only then, Previous and Next — never an earlier step's own
    // links (the bug the prior placement above the `<ol>` had).
    await page.keyboard.press('Tab');
    expect(await focusedSummary()).toContain('Previous');
    await page.keyboard.press('Tab');
    expect(await focusedSummary()).toContain('Next');
  });
});

test.describe('performance gates with the Findings panel expanded', () => {
  // Duplicates of `tests/e2e/arena.spec.ts`'s own two performance-gate
  // tests (`performance gates` describe block), with the Findings panel
  // expanded instead of collapsed — the plan's own acceptance criterion
  // ("Performance gates in `tests/e2e/arena.spec.ts` still pass with the
  // panel expanded") is satisfied here rather than by editing that file,
  // which is already at/over the thermo-review 1000-line threshold (see
  // this file's own doc comment).

  test('median neural step latency stays under the 33ms control budget with the Findings panel expanded', async ({ page }) => {
    await page.goto('/');
    await waitForReady(page);
    await expandFindingsPanel(page);
    await startOrResumeButton(page).click();
    await waitForTick(page, 60);
    await pauseButton(page).click();
    await expect(statusRegion(page)).toHaveText('paused');

    const readLatencyMs = async (agent: 'left' | 'right'): Promise<number> => {
      const armPanel = page.locator('.arm').nth(agent === 'left' ? 0 : 1);
      const text = await armPanel
        .locator('dt', { hasText: 'Neural step latency (median)' })
        .locator('xpath=following-sibling::dd')
        .innerText();
      const match = text.match(/^(\d+(?:\.\d+)?) ms$/);
      if (!match) throw new Error(`Unexpected latency text for ${agent}: "${text}"`);
      const value = Number(match[1]);
      if (!Number.isFinite(value) || value < 0) throw new Error(`Non-finite/negative latency for ${agent}: "${text}"`);
      return value;
    };
    const [leftMs, rightMs] = await Promise.all([readLatencyMs('left'), readLatencyMs('right')]);
    // eslint-disable-next-line no-console -- perf-gate visibility.
    console.log(`[perf] (Findings panel open) median neural step latency: left=${leftMs}ms right=${rightMs}ms (budget < 33ms)`);
    expect(leftMs).toBeLessThan(33);
    expect(rightMs).toBeLessThan(33);
  });

  test('no main-thread stall >= 200ms while running with the Findings panel expanded', async ({ page }) => {
    await page.addInitScript(() => {
      const w = window as unknown as {
        __longTasks: unknown[];
        __longTaskSupported: boolean;
        __longTaskObserver?: PerformanceObserver;
      };
      w.__longTasks = [];
      w.__longTaskSupported = PerformanceObserver.supportedEntryTypes?.includes('longtask') ?? false;
      if (w.__longTaskSupported) {
        const observer = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            w.__longTasks.push({ name: entry.name, duration: entry.duration, startTime: entry.startTime });
          }
        });
        observer.observe({ entryTypes: ['longtask'] });
        w.__longTaskObserver = observer;
      }
    });

    await page.goto('/');
    await waitForReady(page);
    await expandFindingsPanel(page);
    expect(await page.evaluate(() => (window as unknown as { __longTaskSupported: boolean }).__longTaskSupported)).toBe(
      true
    );
    await page.evaluate(() => {
      (window as unknown as { __longTasks: unknown[] }).__longTasks = [];
    });
    await startOrResumeButton(page).click();
    await waitForTick(page, 90);
    await pauseButton(page).click();
    await expect(statusRegion(page)).toHaveText('paused');

    const longTasks = await page.evaluate(() => {
      const w = window as unknown as {
        __longTasks: Array<{ duration: number }>;
        __longTaskObserver?: PerformanceObserver;
      };
      for (const entry of w.__longTaskObserver?.takeRecords() ?? []) {
        w.__longTasks.push({ duration: entry.duration });
      }
      return w.__longTasks ?? [];
    });
    // eslint-disable-next-line no-console -- perf-gate visibility: report what was actually measured either way.
    console.log(`[perf] (Findings panel open) long tasks observed while running: ${JSON.stringify(longTasks)}`);
    for (const task of longTasks) {
      expect(task.duration).toBeLessThan(200);
    }
  });
});
