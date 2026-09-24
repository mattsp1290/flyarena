import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { expect, type Page } from '@playwright/test';

/**
 * Shared Playwright locators/helpers for the closed-loop experiment e2e
 * suite, factored out so `tests/e2e/arena.spec.ts` and
 * `tests/e2e/trained-decoder.spec.ts` (split out of it when it crossed the
 * thermo review's 1000-line threshold — see
 * `reviews/feat-nom6-trained-toggle-thermo-2026-09-24-766f09c/thermo-architecture/01-critical-and-important.md`)
 * do not each hand-roll their own copy of the same locators/polling
 * helpers. Not itself a spec file (no `.spec.ts` suffix), so Playwright's
 * default `testMatch` does not pick it up as a test.
 */

export const here = dirname(fileURLToPath(import.meta.url));
export const publicDataDir = resolve(here, '../../public/data');

export const statusRegion = (page: Page) => page.locator('[role="status"]');
/** Positionally stable: `ExperimentPanel`'s `.button-row` always renders Start/Resume, Pause, Reset in that fixed order. */
export const startOrResumeButton = (page: Page) => page.locator('.button-row button').nth(0);
export const pauseButton = (page: Page) => page.locator('.button-row button').nth(1);
export const resetButton = (page: Page) => page.locator('.button-row button').nth(2);
export const downloadReplayButton = (page: Page) => page.getByRole('button', { name: /Download replay/ });
export const leftTopologySelect = (page: Page) => page.locator('#topology-left');
export const rightTopologySelect = (page: Page) => page.locator('#topology-right');
/** `TelemetryPanel` renders one `.arm` block per agent, in `ARM_IDS = ['left', 'right']` order. */
export const armPanel = (page: Page, agent: 'left' | 'right') => page.locator('.arm').nth(agent === 'left' ? 0 : 1);
export const armMetric = (armLocator: ReturnType<typeof armPanel>, label: string) =>
  armLocator.locator('dt', { hasText: label }).locator('xpath=following-sibling::dd');

export const waitForReady = (page: Page): Promise<void> =>
  expect(statusRegion(page)).toHaveText('ready', { timeout: 20_000 });

/**
 * Confirms the real Three.js renderer actually came up, not merely that the
 * asset-loading/state-machine path succeeded. `App.svelte`'s WebGL-failure
 * fallback is deliberately graceful (asset loading, the state machine, and
 * the Workers keep working even if the renderer fails — see the
 * WebGL-unavailable-fallback test in `arena.spec.ts`, which relies on
 * exactly this), but that graceful degradation means a real renderer
 * regression (a failed chunk import, an `ArenaScene` constructor throw, or
 * a render-loop error) would otherwise show the same fallback message while
 * every other assertion in this suite still passes — including on a CI
 * runner where headless Chromium's SwiftShader software renderer is the
 * only WebGL path available at all. A dual review pass flagged this gap:
 * without this check, no test in this suite ever actually confirmed a
 * frame was drawn.
 */
export const waitForRendererUp = async (page: Page): Promise<void> => {
  // These first two checks only confirm the *absence of a failure signal*
  // so far — `App.svelte`'s "Spectator camera · drag to orbit" text is
  // `rendererError`'s falsy branch, which is also true before the renderer
  // chunk has even started loading (there is no separate "renderer
  // confirmed up" flag, only a "renderer failed" one) — so on their own
  // they would pass immediately, before `ArenaScene` finishes constructing.
  // They're kept here as part of the full contract (and would still catch
  // a case where the fallback message/placeholder appears instead), but the
  // one check below is what actually proves a frame was drawn.
  await expect(page.getByText('Spectator camera · drag to orbit')).toBeVisible({ timeout: 20_000 });
  await expect(page.getByRole('img', { name: 'Arena canvas unavailable' })).toHaveCount(0);
  // Load-bearing: the FPS overlay only renders on the non-error path and
  // only updates away from its initial "0 fps" text once `ArenaScene#update()`
  // has actually run at least once (see `ArenaScene.ts#trackFrameTiming`,
  // and `App.svelte`'s `fps = $state(0)` initial value) — the one part of
  // this helper that is genuine proof a frame was drawn, not just that the
  // canvas element and its container markup exist without an error.
  await expect(page.locator('.fps-overlay')).not.toHaveText('0 fps', { timeout: 10_000 });
};

/**
 * Polls the telemetry heading's "Tick N / 2700" text (`TelemetryPanel`'s
 * `.section-heading span`) until the tick count reaches `minTick`.
 * `polling: 'raf'` (rather than a fixed interval) reacts within one
 * animation frame of the DOM actually updating, since ticks arrive roughly
 * every 33ms (the fixed 30 Hz simulation step) and a slower poll could add a
 * meaningful, variable delay to exactly which tick a subsequent Pause lands
 * on.
 */
export const waitForTick = (page: Page, minTick: number): Promise<unknown> =>
  page.waitForFunction(
    (n) => {
      const match = document.body.textContent?.match(/Tick (\d+) \/ \d+/);
      return Boolean(match && Number(match[1]) >= n);
    },
    minTick,
    { timeout: 20_000, polling: 'raf' }
  );

/**
 * WP3: the anatomical activity view's single collapse/expand toggle (one
 * `<button>` inside `ActivityPanel`'s own `section.activity` root — its
 * accessible name switches between "Expand"/"Collapse" with the panel's
 * open state, so this locates it structurally rather than by name).
 */
export const activityToggle = (page: Page) => page.locator('section.activity button');
export const activityCanvas = (page: Page) => page.getByLabel('Neural activity at soma positions');

/** Expands the activity panel and waits for it to actually be open (real positions loaded and verified, not just clicked). */
export const expandActivityPanel = async (page: Page): Promise<void> => {
  await expect(activityToggle(page)).toBeEnabled({ timeout: 20_000 });
  await activityToggle(page).click();
  await expect(activityToggle(page)).toHaveText(/collapse/i);
  await expect(activityCanvas(page)).toBeVisible();
};

/**
 * Proof the activity scene is actually receiving and drawing fresh rates —
 * not just that the canvas element exists — mirroring `waitForRendererUp`'s
 * own "don't just check for the absence of a failure" discipline. `left`'s
 * `data-last-update-tick-left` debug attribute (set by `ActivityPanel`) only
 * advances once `runner.getLatestRates('left')` has actually returned a
 * fresh array and `ActivityScene#update` was called with it.
 */
export const waitForActivityUpdateTick = (page: Page, agent: 'left' | 'right', minTick: number): Promise<unknown> =>
  page.waitForFunction(
    ({ attr, n }) => {
      const canvas = document.querySelector(`canvas[aria-label="Neural activity at soma positions"]`);
      const value = canvas?.getAttribute(attr);
      return Boolean(value && Number(value) >= n);
    },
    { attr: `data-last-update-tick-${agent}`, n: minTick },
    { timeout: 20_000, polling: 'raf' }
  );

export interface DownloadedReplay {
  schemaVersion: 1;
  seed: number;
  configFingerprint: string;
  topology: Record<'left' | 'right', string>;
  finalSummary: { ticks: number };
  trace: ReadonlyArray<{ tick: number; timeSeconds: number; agents: Record<'left' | 'right', unknown> }>;
  /** The decoder that produced this trace — see `ExperimentReplayExport.decoder`'s doc comment (`src/lib/arena/replay.ts`). */
  decoder: 'authored' | 'trained';
  /** Present only when `decoder === 'trained'`; see `ExperimentReplayExport.trainedReadoutArtifactSha256`'s doc comment. */
  trainedReadoutArtifactSha256?: string;
}

/** Triggers the replay download (only enabled while paused/finished — see `ExperimentPanel`'s `canDownload`) and parses the resulting JSON file from disk. */
export const downloadReplay = async (page: Page): Promise<DownloadedReplay> => {
  const [download] = await Promise.all([page.waitForEvent('download'), downloadReplayButton(page).click()]);
  const filePath = await download.path();
  if (!filePath) throw new Error('Replay download produced no local file path');
  return JSON.parse(readFileSync(filePath, 'utf-8')) as DownloadedReplay;
};

/** `ExperimentPanel`'s decoder radio group, located by accessible name (`Decoder`'s two native `<input type="radio">`s). */
export const decoderRadio = (page: Page, decoder: 'authored' | 'trained'): ReturnType<Page['getByRole']> =>
  page.getByRole('radio', { name: decoder === 'trained' ? /trained \(offline\)/i : /^authored$/i });
