import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';

test('discover, replay, intervene and verify selected-controller evidence', async ({
  page
}, info) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/#atlas');
  const map = page.getByRole('group', { name: 'Select a discovered behavior' });
  await expect(map).toBeVisible();
  await expect(page.getByRole('region', { name: 'Selected behavior' })).toBeVisible();
  await page.getByRole('button', { name: 'Play behavior replay', exact: true }).click();
  await page.getByRole('button', { name: 'Pause behavior replay', exact: true }).click();
  await page.getByRole('slider', { name: 'Behavior replay timeline' }).fill('60');
  await map.getByRole('button').first().focus();
  await page.keyboard.press('Enter');
  await expect(map.getByRole('button').first()).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'Use quick probe settings' }).click();
  await page.getByRole('button', { name: 'Fork & compare' }).click();
  await expect(page.getByRole('region', { name: 'Counterfactual results' })).toBeVisible({
    timeout: 60_000
  });
  await page.getByRole('slider', { name: 'Paired replay timeline' }).fill('30');
  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export counterfactual evidence' }).click();
  const download = await downloadPromise,
    path = info.outputPath('atlas-probe.json');
  await download.saveAs(path);
  const document = JSON.parse(readFileSync(path, 'utf8'));
  expect(document.evidence.decoder).toBe('atlas-trained');
  expect(document.evidence.controller.theta.length).toBe(419);
  expect(document.evidence.summary.shamEffect.mean).toBe(0);
  const result = execFileSync(
    process.execPath,
    ['--import', 'tsx', 'scripts/experiments/counterfactual.ts', '--compare-numerical', path],
    { encoding: 'utf8', timeout: 120_000 }
  );
  const comparison = JSON.parse(result);
  expect(comparison.matches).toBe(true);
  expect(comparison.maxAbsoluteError).toBeLessThan(1e-12);
  await map.getByRole('button').last().click();
  await expect(page.getByRole('region', { name: 'Counterfactual results' })).toHaveCount(0);
  await page.getByLabel('Silence group', { exact: true }).selectOption('output');
  await page.getByRole('button', { name: 'Use quick probe settings' }).click();
  await page.getByLabel('Fork tick', { exact: true }).fill('60');
  await page.getByRole('button', { name: 'Fork & compare' }).click();
  await page.getByRole('button', { name: 'Cancel probe', exact: true }).click();
  await expect(page.getByRole('status').filter({ hasText: 'cancelled' })).toBeVisible();
  await page.getByRole('button', { name: 'Fork & compare' }).click();
  await expect(page.getByRole('region', { name: 'Counterfactual results' })).toBeVisible({
    timeout: 60_000
  });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.getByRole('link', { name: '01 Arena' }).click();
  await page.getByRole('link', { name: '03 Behavior atlas' }).click();
  await expect(map).toBeVisible();
  expect(errors).toEqual([]);
});

test('rejects a valid atlas changed between selection and worker execution', async ({ page }) => {
  await page.goto('/#atlas');
  await expect(page.getByRole('group', { name: 'Select a discovered behavior' })).toBeVisible();
  const next = JSON.parse(readFileSync('public/data/behavior-atlas-v1.json', 'utf8'));
  next.source.runtime.seconds += 1;
  const body = JSON.stringify(next);
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
  await page.route('**/behavior-atlas-v1.json', (route) =>
    route.fulfill({ body, contentType: 'application/json' })
  );
  await page.getByRole('button', { name: 'Use quick probe settings' }).click();
  await page.getByRole('button', { name: 'Fork & compare' }).click();
  await expect(page.getByRole('alert')).toContainText('Atlas changed since selection');
  await expect(page.getByRole('region', { name: 'Counterfactual results' })).toHaveCount(0);
});

test('fails closed on tampered atlas and retries', async ({ page }) => {
  await page.route('**/behavior-atlas-v1.json', (route) =>
    route.fulfill({ body: '{}', contentType: 'application/json' })
  );
  await page.goto('/#atlas');
  await expect(page.getByRole('alert')).toContainText('integrity');
  await page.unroute('**/behavior-atlas-v1.json');
  await page.getByRole('button', { name: 'Retry atlas loading' }).click();
  await expect(page.getByRole('group', { name: 'Select a discovered behavior' })).toBeVisible();
});


test('selection changes terminate a pending probe without a stale result', async ({ page }) => {
  await page.goto('/#atlas');
  const map = page.getByRole('group', { name: 'Select a discovered behavior' });
  await expect(map).toBeVisible();
  let started!: () => void;
  const fetching = new Promise<void>(resolve => { started = resolve; });
  await page.route('**/behavior-atlas-v1.json', async route => {
    started();
    await new Promise(resolve => setTimeout(resolve, 500));
    await route.continue().catch(() => {}); // The old worker is intentionally terminated.
  });
  await page.getByRole('button', { name: 'Use quick probe settings' }).click();
  await page.getByRole('button', { name: 'Fork & compare' }).click();
  await fetching;
  await map.getByRole('button').first().click();
  await expect(page.getByRole('status')).toHaveText('ready');
  await page.waitForTimeout(600);
  await expect(page.getByRole('region', { name: 'Counterfactual results' })).toHaveCount(0);
});

test('missing atlas produces a recoverable loading error', async ({ page }) => {
  await page.route('**/behavior-atlas-v1.json', route => route.fulfill({ status: 404, body: '' }));
  await page.goto('/#atlas');
  await expect(page.getByRole('alert')).toContainText('Atlas request failed (404)');
  await page.unroute('**/behavior-atlas-v1.json');
  await page.getByRole('button', { name: 'Retry atlas loading' }).click();
  await expect(page.getByRole('group', { name: 'Select a discovered behavior' })).toBeVisible();
});
