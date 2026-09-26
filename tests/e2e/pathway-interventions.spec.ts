import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { expect, test } from '@playwright/test';
import { publicDataDir, startOrResumeButton, statusRegion, waitForReady, waitForTick } from './arena-test-helpers';

/**
 * WP4 of `.agents/plans/pathway-interventions`: e2e coverage for the
 * pathway-intervention tested-outcome sentence rendered next to the
 * null-explanation note. Split out of `tests/e2e/arena.spec.ts` (a
 * maintainability-review suggestion: that file was already at the
 * thermo-review's 1000-line threshold on `origin/main`, and these two
 * self-contained blocks pushed it further past it) — mirrors
 * `tests/e2e/trained-decoder.spec.ts`'s own precedent for the identical
 * "split out a self-contained block" reason.
 */

test.describe('model ledger and provenance (pathway interventions)', () => {
  test('shows the pathway-intervention tested-outcome sentence under the explanation note, generated from the verified artifact', async ({
    page
  }) => {
    await page.goto('/');
    await waitForReady(page);

    const ledgerRow = (term: string) => page.locator('.ledger li', { hasText: term });
    await expect(ledgerRow('Pathway intervention test')).toContainText('Computed (offline)');

    const detail = page.locator('.null-explanation-detail');
    await expect(detail).toContainText(/tested under this model/i);
    // The real shipped artifact's own category/modifier/trained result —
    // not a hardcoded placeholder (mirrors arena.spec.ts's own
    // "assert the real shipped numbers" discipline for the sibling
    // null-explanation note).
    await expect(detail).toContainText(/the pathway-supported category holds/i);
    await expect(detail).toContainText(/the channel-specific modifier holds/i);
    await expect(detail).toContainText(/P shows no advantage over either freshly-trained control arm/i);
    await expect(detail).toContainText(/\(robust\)/i);

    const reportLink = detail.getByRole('link', { name: /intervention report/i });
    await expect(reportLink).toHaveAttribute(
      'href',
      'https://github.com/mattsp1290/flyarena/blob/main/docs/pathway-interventions-report.md'
    );
  });
});

test.describe('pathway-interventions hash-mismatch integrity check', () => {
  test('a tampered pathway-interventions-v1.json shows an honest verification-failure message in place of the sentence, while the explanation note above it and the rest of the experiment (Start included) keep working', async ({
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

    await expect(page.locator('.ledger')).toContainText(/intervention test failed verification/i);
    await expect(page.locator('.ledger')).toContainText(/sha256/i);
    // The tested-outcome sentence itself must never render over unverified bytes.
    await expect(page.locator('body')).not.toContainText(/tested under this model/i);

    // The explanation note above it (an independent load) keeps working.
    await expect(page.getByRole('heading', { name: /what biological's low score is associated with/i })).toBeVisible();

    // The required arena graph artifacts are untouched — the experiment
    // keeps working exactly as if the pathway-interventions artifact were
    // never routed.
    await expect(startOrResumeButton(page)).toBeEnabled();
    await startOrResumeButton(page).click();
    await waitForTick(page, 10);
    await expect(statusRegion(page)).toHaveText('running');
  });

  test('a missing pathwayInterventions manifest entry hides the sentence entirely, with no failure message, while the explanation note keeps working', async ({
    page
  }) => {
    await page.route('**/data/malecns-arena-v1.manifest.json', async (route) => {
      const response = await route.fetch();
      const manifest = (await response.json()) as { pathwayInterventions?: unknown };
      delete manifest.pathwayInterventions;
      await route.fulfill({ response, json: manifest });
    });

    await page.goto('/');
    await waitForReady(page);

    await expect(page.locator('.ledger')).not.toContainText(/intervention test failed verification/i);
    await expect(page.locator('body')).not.toContainText(/tested under this model/i);
    await expect(page.getByRole('heading', { name: /what biological's low score is associated with/i })).toBeVisible();
    await expect(startOrResumeButton(page)).toBeEnabled();
  });
});
