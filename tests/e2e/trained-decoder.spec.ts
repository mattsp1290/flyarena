import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { expect, test } from '@playwright/test';
import {
  decoderRadio,
  downloadReplay,
  pauseButton,
  publicDataDir,
  resetButton,
  startOrResumeButton,
  statusRegion,
  waitForReady,
  waitForTick
} from './arena-test-helpers';

/**
 * WP6 decoder-toggle browser coverage, split out of `tests/e2e/arena.spec.ts`
 * when that file crossed the thermo review's 1000-line threshold (see
 * `reviews/feat-nom6-trained-toggle-thermo-2026-09-24-766f09c/thermo-architecture/01-critical-and-important.md`).
 * Shares its locators/polling helpers with `arena.spec.ts` via
 * `./arena-test-helpers.ts` — see that module's doc comment. Picked up by
 * Playwright automatically (default `testMatch` over `testDir:
 * './tests/e2e'`, `playwright.config.ts`); no config change was needed for
 * either local runs or CI (`npm run test:e2e`).
 */

test.describe('trained decoder toggle (WP6)', () => {
  test('toggling to Trained resets to tick 0, changes the score trace versus Authored at the same seed, and running Trained twice at the same seed is identical', async ({
    page
  }) => {
    await page.goto('/');
    await waitForReady(page);
    await expect(decoderRadio(page, 'authored')).toBeChecked();

    const N = 40;

    // Authored baseline (the default) for this seed.
    await startOrResumeButton(page).click();
    await waitForTick(page, N);
    await pauseButton(page).click();
    await expect(statusRegion(page)).toHaveText('paused');
    const authored = await downloadReplay(page);

    await resetButton(page).click();
    await expect(statusRegion(page)).toHaveText('ready');

    // Switch to Trained. The trained-readout artifact ships committed and
    // hash-verified (WP5's production run), so the option must be selectable.
    await expect(decoderRadio(page, 'trained')).toBeEnabled({ timeout: 20_000 });
    await decoderRadio(page, 'trained').click();
    await expect(decoderRadio(page, 'trained')).toBeChecked();
    // The switch resets the run to tick 0 (WP6's "Behavior and invariants").
    await expect(statusRegion(page)).toHaveText('ready');
    await expect(page.locator('.section-heading span', { hasText: /^Tick / })).toHaveText(/^Tick 0 \//);

    await startOrResumeButton(page).click();
    await waitForTick(page, N);
    await pauseButton(page).click();
    await expect(statusRegion(page)).toHaveText('paused');
    const trainedFirst = await downloadReplay(page);

    expect(trainedFirst.seed).toBe(authored.seed);
    // Same seed, same topology, different decoder: the trace must differ.
    expect(trainedFirst.trace.slice(0, N)).not.toEqual(authored.trace.slice(0, N));

    // Same seed twice under Trained: identical trace prefix (determinism
    // holds for the trained decoder exactly as it does for authored).
    await resetButton(page).click();
    await expect(statusRegion(page)).toHaveText('ready');
    await expect(decoderRadio(page, 'trained')).toBeChecked();

    await startOrResumeButton(page).click();
    await waitForTick(page, N);
    await pauseButton(page).click();
    await expect(statusRegion(page)).toHaveText('paused');
    const trainedSecond = await downloadReplay(page);

    expect(trainedSecond.trace.slice(0, N)).toEqual(trainedFirst.trace.slice(0, N));
  });

  test('shows the "Readout (trained mode)" ledger row with the shipped artifact\'s hash/parameter count/architecture, visible without toggling', async ({
    page
  }) => {
    await page.goto('/');
    await waitForReady(page);

    const ledgerRow = (term: string) => page.locator('.ledger li', { hasText: term });
    await expect(ledgerRow('Readout (trained mode)')).toContainText('Trained (offline)');

    const readoutManifest = JSON.parse(
      readFileSync(resolve(publicDataDir, 'trained-readout-v1.manifest.json'), 'utf-8')
    ) as { parameterCount: number; D: number; H: number; artifactSha256: string };
    const detail = page.locator('.trained-readout-detail');
    await expect(detail).toContainText(String(readoutManifest.parameterCount));
    await expect(detail).toContainText(`${readoutManifest.D} → ${readoutManifest.H} → 3`);
    await expect(detail).toContainText(readoutManifest.artifactSha256.slice(0, 12));
    await expect(detail).toContainText(/opponent parked/i);
    await expect(detail.getByRole('link', { name: /report \(json\)/i })).toHaveAttribute(
      'href',
      '/data/trained-readout-v1.report.json'
    );
  });
});

test.describe('trained-readout hash-mismatch integrity check', () => {
  test('a tampered trained-readout-v1.json disables the Trained option with an honest reason, while Authored keeps working', async ({
    page
  }) => {
    const original = readFileSync(resolve(publicDataDir, 'trained-readout-v1.json'));
    const tampered = Buffer.from(original);
    tampered[10] ^= 0xff;

    await page.route('**/data/trained-readout-v1.json', (route) =>
      route.fulfill({ status: 200, contentType: 'application/json', body: tampered })
    );

    await page.goto('/');
    await waitForReady(page);

    await expect(decoderRadio(page, 'trained')).toBeDisabled();
    await expect(page.locator('.hint')).toContainText(/trained unavailable/i);
    await expect(page.locator('.hint')).toContainText(/sha256/i);
    await expect(page.locator('.ledger')).toContainText(/artifact failed verification/i);

    // The required arena graph artifacts are untouched — Authored keeps
    // working exactly as if the trained-readout artifact were never routed.
    await expect(decoderRadio(page, 'authored')).toBeChecked();
    await expect(startOrResumeButton(page)).toBeEnabled();
    await startOrResumeButton(page).click();
    await waitForTick(page, 10);
    await expect(statusRegion(page)).toHaveText('running');
  });
});
