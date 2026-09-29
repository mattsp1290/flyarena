import { expect, test } from '@playwright/test';
import { expandFindingsPanel, waitForReady } from './arena-test-helpers';

/**
 * WP1 of `.agents/plans/consolidated-release` (`01-findings-sections.md`):
 * e2e coverage for the Findings panel's question-section accordion, split
 * into its own file rather than added to `tests/e2e/findings.spec.ts`
 * (already covers the panel's per-step content and the real-Tab-order
 * guarantee) or `tests/e2e/arena.spec.ts` (already at/over this repo's
 * thermo-review 1000-line threshold — the same "split out a self-contained
 * block" precedent `tests/e2e/pathway-interventions.spec.ts`/
 * `tests/e2e/trained-decoder.spec.ts` already establish).
 *
 * Real section titles from `src/lib/findings/sections.ts#FINDING_SECTIONS`
 * — not re-derived from the app bundle, so these tests fail loudly (wrong
 * accessible name) rather than silently if a title ever drifts from that
 * module.
 */

const SECTION_TITLES = [
  'Is the measured wiring special?',
  'Why does it score low?',
  'Does it generalize?',
  'What do trained readouts do?'
];

test.describe('Findings panel — question sections', () => {
  test('every section renders open by default, each with its own accessible toggle name and aria-controls target', async ({
    page
  }) => {
    await page.goto('/');
    await waitForReady(page);
    await expandFindingsPanel(page);

    const panel = page.locator('section.findings');
    for (const title of SECTION_TITLES) {
      const toggle = panel.getByRole('button', { name: title });
      await expect(toggle).toBeVisible();
      await expect(toggle).toHaveAttribute('aria-expanded', 'true');
      const controlsId = await toggle.getAttribute('aria-controls');
      expect(controlsId).toBeTruthy();
      const controlledOl = page.locator(`#${controlsId}`);
      await expect(controlledOl).toBeVisible();
      expect(await controlledOl.evaluate((el) => el.tagName)).toBe('OL');
    }

    // "Other findings" never appears while every shipped step id is classified.
    await expect(panel.getByRole('button', { name: 'Other findings' })).toHaveCount(0);
  });

  test('each question section groups the real step ids the plan assigns it', async ({ page }) => {
    await page.goto('/');
    await waitForReady(page);
    await expandFindingsPanel(page);

    const sectionStepIds = async (title: string): Promise<string[]> => {
      const toggle = page.locator('section.findings').getByRole('button', { name: title });
      const controlsId = await toggle.getAttribute('aria-controls');
      const items = page.locator(`#${controlsId} > li.step`);
      return items.evaluateAll((els) => els.map((el) => el.getAttribute('data-step-id') ?? ''));
    };

    expect(await sectionStepIds('Is the measured wiring special?')).toEqual(['rewiring-null', 'mirrored-decoder', 'behavior-repertoire']);
    expect(await sectionStepIds('Why does it score low?')).toEqual(['explanation', 'intervention']);
    expect(await sectionStepIds('Does it generalize?')).toEqual(['task-generality']);
    expect(await sectionStepIds('What do trained readouts do?')).toEqual(['trained-null', 'trained-interventions']);
  });

  test('collapsing a section toggle hides its steps and flips aria-expanded to false; activating it again restores them', async ({
    page
  }) => {
    await page.goto('/');
    await waitForReady(page);
    await expandFindingsPanel(page);

    const toggle = page.locator('section.findings').getByRole('button', { name: 'Why does it score low?' });
    const controlsId = await toggle.getAttribute('aria-controls');
    const controlledOl = page.locator(`#${controlsId}`);

    await expect(controlledOl).toBeVisible();
    await toggle.click();
    await expect(toggle).toHaveAttribute('aria-expanded', 'false');
    await expect(controlledOl).toHaveCount(0);

    await toggle.click();
    await expect(toggle).toHaveAttribute('aria-expanded', 'true');
    await expect(controlledOl).toBeVisible();
  });

  test('Next walks across sections: collapsing the second section, then walking Next into it re-expands it and moves focus to the step heading', async ({
    page
  }) => {
    await page.goto('/');
    await waitForReady(page);
    await expandFindingsPanel(page);

    const scoresLowToggle = page.locator('section.findings').getByRole('button', { name: 'Why does it score low?' });
    await scoresLowToggle.click();
    await expect(scoresLowToggle).toHaveAttribute('aria-expanded', 'false');

    // The first section ("Is the measured wiring special?") has three
    // steps (rewiring-null, mirrored-decoder, behavior-repertoire); Next
    // three times reaches the first step of the collapsed second section
    // ("explanation", in "Why does it score low?").
    const next = page.locator('section.findings').getByRole('button', { name: /^next$/i });
    await next.click();
    await next.click();
    await next.click();

    await expect(scoresLowToggle).toHaveAttribute('aria-expanded', 'true');
    const explanationStep = page.locator('section.findings li.step[data-step-id="explanation"]');
    await expect(explanationStep).toBeVisible();
    await expect(explanationStep).toHaveAttribute('aria-current', 'step');
    await expect(explanationStep.locator('h4')).toBeFocused();
  });

  test('collapsing all four sections in sequence, by keyboard, never reaches zero visible steps or a stale live-region announcement', async ({
    page
  }) => {
    // (thermo accessibility review, Important I1) The exact reachable
    // sequence the review flagged: Tab to each section toggle and activate
    // it with the keyboard, left to right. Previously the fourth (last)
    // collapse left zero `<li class="step">` elements, zero Previous/Next
    // controls, and a stale "Step N of M in <section>" live-region
    // announcement describing content that no longer existed anywhere on
    // the page. The last collapse is now a no-op: at least one section
    // stays open throughout.
    await page.goto('/');
    await waitForReady(page);
    await expandFindingsPanel(page);

    const panel = page.locator('section.findings');
    const toggles = SECTION_TITLES.map((title) => panel.getByRole('button', { name: title }));

    for (const toggle of toggles) {
      await toggle.focus();
      await page.keyboard.press('Enter');
    }

    // Exactly one toggle (the last section collapse attempt refused) is
    // still expanded; the other three collapsed successfully.
    const expandedStates = await Promise.all(toggles.map((toggle) => toggle.getAttribute('aria-expanded')));
    expect(expandedStates.filter((state) => state === 'true')).toHaveLength(1);
    expect(expandedStates.filter((state) => state === 'false')).toHaveLength(SECTION_TITLES.length - 1);

    // At least one step is still visible, with working Previous/Next.
    await expect(panel.locator('li.step')).not.toHaveCount(0);
    await expect(panel.getByRole('button', { name: /^previous$/i })).toBeVisible();
    await expect(panel.getByRole('button', { name: /^next$/i })).toBeVisible();

    // The live region still names a real, still-open section -- never
    // blank and never naming a now-collapsed one.
    const stillOpenTitle = SECTION_TITLES[expandedStates.findIndex((state) => state === 'true')];
    await expect(page.getByText(new RegExp(`^Step \\d+ of \\d+ in ${stillOpenTitle.replace(/[?]/g, '\\?')}$`))).toBeVisible();
  });

  test('the live region announces the current step\'s section title', async ({ page }) => {
    await page.goto('/');
    await waitForReady(page);
    await expandFindingsPanel(page);

    await expect(page.getByText(/^Step 1 of \d+ in Is the measured wiring special\?$/)).toBeVisible();

    // Walk to the first step of the second section (explanation, "Why does
    // it score low?") -- three Next clicks past the first section's three
    // steps.
    const next = page.locator('section.findings').getByRole('button', { name: /^next$/i });
    await next.click();
    await next.click();
    await next.click();
    await expect(page.getByText(/^Step 4 of \d+ in Why does it score low\?$/)).toBeVisible();
  });

  test('the Findings panel itself stays within the phone-width viewport with every section expanded', async ({ page }) => {
    // Scoped to `section.findings`'s own box, not `document.documentElement`'s
    // full `scrollWidth` (already covered, in a different route/state
    // combination, by `tests/e2e/subpath.spec.ts`'s own phone-width check) --
    // a pre-existing, unrelated `sr-only` table in `NullHistogram.svelte`
    // (a sibling panel this WP does not touch) can widen the *document's*
    // scrollWidth regardless of the Findings panel's own layout, so
    // asserting against the whole document here would fail on a defect
    // outside this change's scope. This test's own job is only to confirm
    // the accordion sections this WP adds do not themselves cause overflow.
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto('/');
    await waitForReady(page);
    await expandFindingsPanel(page);
    const panelBox = await page.locator('section.findings').boundingBox();
    expect(panelBox).not.toBeNull();
    expect(panelBox!.x + panelBox!.width).toBeLessThanOrEqual(390 + 1);
  });
});
