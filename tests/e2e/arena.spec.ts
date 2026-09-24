import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { expect, test, type Page } from '@playwright/test';

/**
 * WP7 item 1 (plus the performance-gate items in WP7 item 4): browser
 * smoke/control/determinism/honesty coverage for the closed-loop experiment,
 * driven against a real production build served by Playwright's own
 * `webServer` (`playwright.config.ts`: `npm run build && npm run preview`).
 *
 * These tests exercise the real, un-mocked app end to end: the real
 * fetch/hash-verify path (`src/lib/experiment/assets.ts`), the real
 * dedicated neural Workers, and the real Three.js renderer (except where a
 * test deliberately breaks one of those on purpose, e.g. the WebGL-fallback
 * and hash-mismatch tests below).
 */

const here = dirname(fileURLToPath(import.meta.url));
const publicDataDir = resolve(here, '../../public/data');

const statusRegion = (page: Page) => page.locator('[role="status"]');
/** Positionally stable: `ExperimentPanel`'s `.button-row` always renders Start/Resume, Pause, Reset in that fixed order. */
const startOrResumeButton = (page: Page) => page.locator('.button-row button').nth(0);
const pauseButton = (page: Page) => page.locator('.button-row button').nth(1);
const resetButton = (page: Page) => page.locator('.button-row button').nth(2);
const downloadReplayButton = (page: Page) => page.getByRole('button', { name: /Download replay/ });
const leftTopologySelect = (page: Page) => page.locator('#topology-left');
const rightTopologySelect = (page: Page) => page.locator('#topology-right');
/** `TelemetryPanel` renders one `.arm` block per agent, in `ARM_IDS = ['left', 'right']` order. */
const armPanel = (page: Page, agent: 'left' | 'right') => page.locator('.arm').nth(agent === 'left' ? 0 : 1);
const armMetric = (armLocator: ReturnType<typeof armPanel>, label: string) =>
  armLocator.locator('dt', { hasText: label }).locator('xpath=following-sibling::dd');

const waitForReady = (page: Page): Promise<void> =>
  expect(statusRegion(page)).toHaveText('ready', { timeout: 20_000 });

/**
 * Confirms the real Three.js renderer actually came up, not merely that the
 * asset-loading/state-machine path succeeded. `App.svelte`'s WebGL-failure
 * fallback is deliberately graceful (asset loading, the state machine, and
 * the Workers keep working even if the renderer fails — see the
 * WebGL-unavailable-fallback test below, which relies on exactly this), but
 * that graceful degradation means a real renderer regression (a failed
 * chunk import, an `ArenaScene` constructor throw, or a render-loop error)
 * would otherwise show the same fallback message while every other
 * assertion in this suite still passes — including on a CI runner where
 * headless Chromium's SwiftShader software renderer is the only WebGL path
 * available at all. A dual review pass flagged this gap: without this
 * check, no test in this suite ever actually confirmed a frame was drawn.
 */
const waitForRendererUp = async (page: Page): Promise<void> => {
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
const waitForTick = (page: Page, minTick: number): Promise<unknown> =>
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
const activityToggle = (page: Page) => page.locator('section.activity button');
const activityCanvas = (page: Page) => page.getByLabel('Neural activity at soma positions');

/** Expands the activity panel and waits for it to actually be open (real positions loaded and verified, not just clicked). */
const expandActivityPanel = async (page: Page): Promise<void> => {
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
const waitForActivityUpdateTick = (page: Page, agent: 'left' | 'right', minTick: number): Promise<unknown> =>
  page.waitForFunction(
    ({ attr, n }) => {
      const canvas = document.querySelector(`canvas[aria-label="Neural activity at soma positions"]`);
      const value = canvas?.getAttribute(attr);
      return Boolean(value && Number(value) >= n);
    },
    { attr: `data-last-update-tick-${agent}`, n: minTick },
    { timeout: 20_000, polling: 'raf' }
  );

interface DownloadedReplay {
  schemaVersion: 1;
  seed: number;
  configFingerprint: string;
  topology: Record<'left' | 'right', string>;
  finalSummary: { ticks: number };
  trace: ReadonlyArray<{ tick: number; timeSeconds: number; agents: Record<'left' | 'right', unknown> }>;
}

/** Triggers the replay download (only enabled while paused/finished — see `ExperimentPanel`'s `canDownload`) and parses the resulting JSON file from disk. */
const downloadReplay = async (page: Page): Promise<DownloadedReplay> => {
  const [download] = await Promise.all([page.waitForEvent('download'), downloadReplayButton(page).click()]);
  const filePath = await download.path();
  if (!filePath) throw new Error('Replay download produced no local file path');
  return JSON.parse(readFileSync(filePath, 'utf-8')) as DownloadedReplay;
};

test.describe('asset load and readiness', () => {
  test('loads and hash-verifies the real connectome artifacts, then reaches ready with Start enabled', async ({
    page
  }) => {
    await page.goto('/');
    await waitForReady(page);
    await waitForRendererUp(page);
    await expect(startOrResumeButton(page)).toBeEnabled();
    await expect(startOrResumeButton(page)).toHaveText('Start');

    // The ledger panel only renders manifest-derived counts once the real
    // manifest has actually been fetched and parsed — a stronger signal
    // than the status text alone that asset loading genuinely completed.
    const manifest = JSON.parse(readFileSync(resolve(publicDataDir, 'malecns-arena-v1.manifest.json'), 'utf-8')) as {
      neuronCount: number;
      edgeCount: number;
    };
    // Scoped to the ledger panel: the default topology (biological left,
    // rewired right) means the rewired arm's telemetry shows the exact same
    // neuron/edge counts too (degree-preserving rewiring), so an unscoped
    // page-wide text match resolves to more than one element.
    await expect(page.locator('.ledger')).toContainText(`${manifest.neuronCount} / ${manifest.edgeCount}`);
  });

  test('the sidebar (controls/telemetry/ledger) stays beside the arena, not pushed below the activity panel (layout regression)', async ({
    page
  }) => {
    // Dual review finding: the activity panel's `.activity { grid-column: 1
    // / -1 }` used to make CSS grid's sparse auto-placement push
    // `aside.sidebar` into a third row *below* the activity panel, leaving
    // the whole column next to the arena empty — Start/Pause/Reset,
    // telemetry, and the ledger all fell below the fold. Confirmed by
    // measuring real layout boxes at a desktop viewport, not just checking
    // that the elements exist.
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.goto('/');
    await waitForReady(page);

    const arenaBox = await page.locator('section.arena').boundingBox();
    const sidebarBox = await page.locator('aside.sidebar').boundingBox();
    const activityBox = await page.locator('section.activity').boundingBox();
    if (!arenaBox || !sidebarBox || !activityBox) throw new Error('Expected all three layout regions to have a bounding box');

    // The sidebar sits to the right of the arena (same row), not below it.
    expect(sidebarBox.x).toBeGreaterThan(arenaBox.x + arenaBox.width - 1);
    expect(Math.abs(sidebarBox.y - arenaBox.y)).toBeLessThan(4);
    // The activity panel sits below the arena, in the arena's own column
    // (not spanning under the sidebar too).
    expect(activityBox.y).toBeGreaterThan(arenaBox.y + arenaBox.height - 1);
    expect(activityBox.x).toBeCloseTo(arenaBox.x, 0);

    // Round-2 review finding: pinning the sidebar to `grid-row: 1 / span 2`
    // (above) fixed *position* but, without `main { grid-template-rows:
    // auto 1fr }` and `.activity { align-self: start }`, CSS grid spread
    // the tall sidebar's height evenly across both rows and stretched the
    // arena and the *collapsed* activity panel to fill it — confirmed in a
    // browser: the empty collapsed panel measured ~700px tall around ~73px
    // of actual content, pushing the Expand toggle below the fold. These
    // height/gap assertions catch that class of regression, which the
    // position-only assertions above cannot.
    expect(activityBox.height).toBeLessThan(200);
    const canvasRegionBox = await page.locator('section.arena .canvas-region').boundingBox();
    if (!canvasRegionBox) throw new Error('Expected the arena canvas region to have a bounding box');
    expect(activityBox.y - (canvasRegionBox.y + canvasRegionBox.height)).toBeLessThan(120);
    expect(activityBox.y).toBeLessThan(1000);
  });
});

test.describe('controls: start/pause/reset', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForReady(page);
  });

  test('Pause and Reset stay clickable for the entire duration of a run (regression: both were previously disabled the whole run)', async ({
    page
  }) => {
    await expect(pauseButton(page)).toBeDisabled(); // not running yet
    await expect(resetButton(page)).toBeEnabled(); // 'ready' is one of Reset's allowed states

    await startOrResumeButton(page).click();
    await expect(statusRegion(page)).toHaveText('running');
    await waitForTick(page, 5);

    // The regression this guards: both buttons were previously wired to the
    // same blanket `controlsLocked` flag Start/Seed use, which is true for
    // the whole run — so neither could ever actually be clicked mid-run.
    await expect(pauseButton(page)).toBeEnabled();
    await expect(resetButton(page)).toBeEnabled();

    await pauseButton(page).click();
    await expect(statusRegion(page)).toHaveText('paused');
    await expect(startOrResumeButton(page)).toHaveText('Resume');
    await expect(startOrResumeButton(page)).toBeEnabled();

    await resetButton(page).click();
    await expect(statusRegion(page)).toHaveText('ready');
    await expect(page.getByText(/^Tick 0 \//)).toBeVisible();
  });
});

test.describe('fixed-seed replay determinism', () => {
  test('running the same seed twice produces an identical replay trace prefix', async ({ page }) => {
    await page.goto('/');
    await waitForReady(page);

    const N = 40;

    await startOrResumeButton(page).click();
    await waitForTick(page, N);
    await pauseButton(page).click();
    await expect(statusRegion(page)).toHaveText('paused');
    const first = await downloadReplay(page);

    await resetButton(page).click();
    await expect(statusRegion(page)).toHaveText('ready');

    await startOrResumeButton(page).click();
    await waitForTick(page, N);
    await pauseButton(page).click();
    await expect(statusRegion(page)).toHaveText('paused');
    const second = await downloadReplay(page);

    expect(first.seed).toBe(second.seed);
    expect(first.configFingerprint).toBe(second.configFingerprint);
    expect(first.topology).toEqual(second.topology);
    // Compare only the shared N-tick prefix rather than the two runs' exact
    // (real-time-paced) stopping points: both runs pause somewhere at or
    // past tick N, and — because the simulation is fixed-step and
    // wall-clock-independent (see `docs/architecture.md`'s closed-loop
    // contract) — every tick up to N must already be byte-identical
    // regardless of exactly when either run's Pause click landed.
    expect(first.trace.length).toBeGreaterThanOrEqual(N);
    expect(second.trace.length).toBeGreaterThanOrEqual(N);
    expect(first.trace.slice(0, N)).toEqual(second.trace.slice(0, N));
  });
});

/** `ExperimentPanel`'s decoder radio group, located by accessible name (`Decoder`'s two native `<input type="radio">`s). */
const decoderRadio = (page: Page, decoder: 'authored' | 'trained'): ReturnType<Page['getByRole']> =>
  page.getByRole('radio', { name: decoder === 'trained' ? /trained \(offline\)/i : /^authored$/i });

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

test.describe('mode switching, including the disconnected negative control', () => {
  test('switching the right arm to disconnected updates its accessible label and its ledger edge count to zero', async ({
    page
  }) => {
    await page.goto('/');
    await waitForReady(page);

    await expect(rightTopologySelect(page)).toHaveValue('rewired');
    await expect(armMetric(armPanel(page, 'right'), 'Graph nodes / edges')).not.toHaveText(/\/\s*0$/);

    // `getByLabel` proves the select's accessible name really is "Right arm
    // topology" (WP6 item 8's accessibility requirement), not just that a
    // `<select id="topology-right">` happens to exist.
    await page.getByLabel('Right arm topology').selectOption('disconnected');
    await expect(rightTopologySelect(page)).toHaveValue('disconnected');

    // The switch is async (Worker dispose -> reinit -> runner.setAgentBinding
    // — see `ExperimentController#changeTopology`); the ledger's own
    // edge-count readout, sourced from the real re-initialized Worker's
    // response, is the observable proxy for it having actually settled
    // (this is also the honesty-critical surface `ArenaScene#setAgentTopology`
    // exists to keep in sync with — see that method's doc comment; the label
    // sprite itself lives on the WebGL canvas and is not otherwise
    // DOM-observable).
    await expect(armMetric(armPanel(page, 'right'), 'Graph nodes / edges')).toHaveText(/\/\s*0$/, {
      timeout: 10_000
    });
    // The left arm's own topology/edge count must be unaffected by the
    // right arm's switch.
    await expect(armMetric(armPanel(page, 'left'), 'Graph nodes / edges')).not.toHaveText(/\/\s*0$/);
    await expect(leftTopologySelect(page)).toHaveValue('biological');
  });

  test('the disconnected control materially changes declared neural features: its arm never moves, while the other arm does', async ({
    page
  }) => {
    await page.goto('/');
    await waitForReady(page);

    await page.getByLabel('Right arm topology').selectOption('disconnected');
    await expect(armMetric(armPanel(page, 'right'), 'Graph nodes / edges')).toHaveText(/\/\s*0$/, {
      timeout: 10_000
    });

    await startOrResumeButton(page).click();
    // The leaky recurrent network starts from zero state and ramps up
    // gradually (bounded leaky integration, not a step function) — at this
    // seed, the biological arm's own decoded thrust stays small enough that
    // "Distance travelled" still rounds to "0.0" through roughly tick 90.
    // Tick 300 (~10 simulated seconds) gives a comfortable margin past that
    // ramp-up so this assertion is about the disconnected-vs-connected
    // contrast, not a race against float-rounding at the display layer.
    await waitForTick(page, 300);
    await pauseButton(page).click();
    await expect(statusRegion(page)).toHaveText('paused');

    // With edgeCount 0 there is no recurrent path from any input-mapped
    // neuron to any output-mapped neuron, so the decoded action is always
    // exactly zero (see `tests/unit/experiment-runner.test.ts`'s
    // "disconnected control never produces a non-zero decoded action" test)
    // and the agent never accelerates away from its start position —
    // "Distance travelled" must read exactly 0.0 the entire run.
    await expect(armMetric(armPanel(page, 'right'), 'Distance travelled')).toHaveText('0.0');
    // The left (biological) arm, driven by the exact same authored encoder/
    // dynamics/decoder contract, must show real (if still modest at this
    // seed/tick count) movement.
    await expect(armMetric(armPanel(page, 'left'), 'Distance travelled')).not.toHaveText('0.0');
  });
});

test.describe('model ledger and provenance', () => {
  test('shows the full ledger vocabulary, real manifest counts, and CC BY attribution', async ({ page }) => {
    await page.goto('/');
    await waitForReady(page);

    const ledgerRow = (term: string) => page.locator('.ledger li', { hasText: term });
    await expect(ledgerRow('Graph topology')).toContainText('Measured');
    await expect(ledgerRow('Biological annotations')).toContainText('Annotated');
    await expect(ledgerRow('Network dynamics')).toContainText('Authored / literature-derived');
    await expect(ledgerRow('Global parameters')).toContainText('Calibrated');
    await expect(ledgerRow('Sensory encoder and action decoder')).toContainText('Authored');
    await expect(ledgerRow('3D presentation')).toContainText('Synthetic');

    const manifest = JSON.parse(readFileSync(resolve(publicDataDir, 'malecns-arena-v1.manifest.json'), 'utf-8')) as {
      neuronCount: number;
      edgeCount: number;
      license: string;
    };
    await expect(page.locator('.ledger')).toContainText(`${manifest.neuronCount} / ${manifest.edgeCount}`);
    await expect(page.locator('.ledger')).toContainText(manifest.license);

    const ccByLink = page.getByRole('link', { name: /CC BY 4\.0/ });
    await expect(ccByLink).toBeVisible();
    await expect(ccByLink).toHaveAttribute('href', 'https://creativecommons.org/licenses/by/4.0/');

    // Never "uploaded a fly brain" / "brain emulation" framing anywhere on
    // the page — the model ledger's whole reason for existing.
    await expect(page.locator('body')).not.toContainText(/brain emulation/i);
  });
});

test.describe('anatomical activity view', () => {
  test('expanding the panel and starting a run shows the canvas, provenance labels, and coverage disclosure', async ({
    page
  }) => {
    await page.goto('/');
    await waitForReady(page);
    await expandActivityPanel(page);

    await expect(page.getByText('Measured', { exact: false }).first()).toBeVisible();
    await expect(page.getByText('Computed', { exact: false }).first()).toBeVisible();
    await expect(page.getByText('Annotated', { exact: false }).first()).toBeVisible();
    await expect(page.getByText(/Both arms share neuron positions; only connections differ/)).toBeVisible();

    const positionsManifest = JSON.parse(
      readFileSync(resolve(publicDataDir, 'malecns-arena-v1.positions.json'), 'utf-8')
    ) as { coverage: { soma: number; tosoma: number; none: number } };
    await expect(page.getByText(/^Positioned:/)).toContainText(
      `${positionsManifest.coverage.soma} soma, ${positionsManifest.coverage.tosoma} soma-tract, ${positionsManifest.coverage.none} unavailable`
    );
    // The "position unavailable" strip must actually be labeled in-product
    // (docs/model-ledger.md's own claim), not just disclosed as a bare
    // count — regression coverage for a dual-review finding.
    await expect(page.getByText(/position unavailable/i)).toBeVisible();
    await expect(page.getByText(new RegExp(`${positionsManifest.coverage.none} neurons with no soma`))).toBeVisible();

    await startOrResumeButton(page).click();
    await waitForTick(page, 10);
    await waitForActivityUpdateTick(page, 'left', 1);
    await waitForActivityUpdateTick(page, 'right', 1);
    await pauseButton(page).click();
    await expect(statusRegion(page)).toHaveText('paused');
  });

  test('closing the panel stops rate streaming: the Worker sends no more `rates` messages after collapse', async ({
    page
  }) => {
    // Instrument both dedicated neural Workers directly (not just the DOM):
    // patch each one's own `postMessage` to count `step` responses that
    // carry a `rates` key (`StepWorkerSuccess.rates` — present only while
    // `set-activity` has most recently enabled streaming for that Worker;
    // see `neural.worker.ts`/`protocol.ts`). `neural.worker.ts` stores
    // `self` itself (not a destructured function reference) as its
    // `workerScope`, so reassigning `self.postMessage` here is visible to
    // every later call the Worker module makes — this is the direct,
    // protocol-level proof the plan's "close -> no rates messages" gate
    // asks for, not just a DOM proxy for it.
    const workers: import('@playwright/test').Worker[] = [];
    // Collected (not fire-and-forget) so the test can `Promise.all` them
    // before trusting any count read below — otherwise a read racing an
    // unlanded patch would see `undefined` for that Worker, and a real leak
    // on that one arm could pass unnoticed (dual review finding).
    const patchPromises: Array<Promise<unknown>> = [];
    page.on('worker', (worker) => {
      workers.push(worker);
      patchPromises.push(
        worker.evaluate(() => {
          const scope = self as unknown as { postMessage: (...args: unknown[]) => void; __rateMessageCount: number };
          scope.__rateMessageCount = 0;
          const original = scope.postMessage.bind(scope);
          scope.postMessage = (...args: unknown[]) => {
            const data = args[0] as { type?: string; rates?: unknown } | undefined;
            if (data && data.type === 'step' && 'rates' in data) scope.__rateMessageCount += 1;
            return original(...args);
          };
        })
      );
    });

    await page.goto('/');
    await waitForReady(page);
    // Both dedicated neural Workers are constructed during `initialize()`,
    // well before `ready` — safe to require exactly two and await their
    // patches landing before any count read is trusted.
    await expect.poll(() => workers.length, { timeout: 10_000 }).toBe(2);
    await Promise.all(patchPromises);

    await expandActivityPanel(page);
    await startOrResumeButton(page).click();
    await waitForActivityUpdateTick(page, 'left', 1);
    await waitForActivityUpdateTick(page, 'right', 1);

    const readCounts = (): Promise<number[]> =>
      Promise.all(
        workers.map((worker) =>
          worker.evaluate(() => (self as unknown as { __rateMessageCount: number }).__rateMessageCount)
        )
      );
    const readTick = (): Promise<number> =>
      page.evaluate(() => Number(document.body.textContent?.match(/Tick (\d+) \/ \d+/)?.[1] ?? NaN));

    const countsWhileOpen = await readCounts();
    expect(countsWhileOpen.every((count) => typeof count === 'number' && count > 0)).toBe(true);

    await activityToggle(page).click();
    await expect(activityToggle(page)).toHaveText(/expand/i);
    // The canvas is removed from the DOM on collapse (this panel only
    // renders it while expanded), which is itself proof the view stopped
    // presenting — the real assertion below is protocol-level.
    expect(await page.locator('canvas[aria-label="Neural activity at soma positions"]').count()).toBe(0);

    const countsAtCollapse = await readCounts();
    // Relative to the tick observed *at* collapse, not an absolute
    // threshold: an absolute `waitForTick(page, 40)` can already be in the
    // past by the time collapse happens on a slow runner, which would let
    // this whole assertion pass without ever observing a post-collapse
    // window (dual review finding).
    const tickAtCollapse = await readTick();
    await waitForTick(page, tickAtCollapse + 30);
    const countsAfterMoreTicks = await readCounts();
    expect(countsAfterMoreTicks).toEqual(countsAtCollapse);

    // Reopen and confirm streaming actually resumes rather than staying
    // wedged off — the gate is "off while closed," not "off forever."
    await expandActivityPanel(page);
    await waitForActivityUpdateTick(page, 'left', 1);
    const countsAfterReopen = await readCounts();
    expect(countsAfterReopen.some((count, index) => count > countsAfterMoreTicks[index])).toBe(true);

    await pauseButton(page).click();
    await expect(statusRegion(page)).toHaveText('paused');
  });

  test('switching the right arm’s topology while the panel stays open keeps both arms’ activity canvas updating on the new topology', async ({
    page
  }) => {
    // `ExperimentRunner#setAgentBinding` only allows a topology switch from
    // `ready`/`finished` (never mid-run) — `ExperimentPanel`'s topology
    // selectors are disabled otherwise (see `topologyControlsLocked` in
    // `App.svelte`), so the real reachable flow is: run a bit, Reset (back
    // to `ready` — the activity panel stays open and streaming throughout;
    // `ExperimentRunner#reset()` deliberately leaves the streaming setting
    // alone, only the world/neural state), switch, then Start again. This
    // exercises the real regression surface: `ExperimentController
    // #changeTopology` must re-apply streaming to the *freshly rebuilt*
    // right-arm Worker binding (a fresh `init` always resets streaming to
    // off), or the activity view would silently go dark for that arm after
    // any topology switch made while it was open.
    await page.goto('/');
    await waitForReady(page);
    await expandActivityPanel(page);

    await startOrResumeButton(page).click();
    await waitForActivityUpdateTick(page, 'left', 5);
    await waitForActivityUpdateTick(page, 'right', 5);
    const rightTickBeforeReset = Number(await activityCanvas(page).getAttribute('data-last-update-tick-right'));
    expect(rightTickBeforeReset).toBeGreaterThanOrEqual(5);

    await resetButton(page).click();
    await expect(statusRegion(page)).toHaveText('ready');

    await page.getByLabel('Right arm topology').selectOption('disconnected');
    await expect(rightTopologySelect(page)).toHaveValue('disconnected', { timeout: 10_000 });
    await expect(statusRegion(page)).toHaveText('ready');

    await startOrResumeButton(page).click();
    // A fresh, low post-restart tick strictly less than the pre-reset value
    // is proof this is genuinely new data on the switched topology, not a
    // stale attribute left over from before the reset.
    await page.waitForFunction(
      ({ attr, ceiling }) => {
        const canvas = document.querySelector('canvas[aria-label="Neural activity at soma positions"]');
        const value = Number(canvas?.getAttribute(attr));
        return Number.isFinite(value) && value > 0 && value < ceiling;
      },
      { attr: 'data-last-update-tick-right', ceiling: rightTickBeforeReset },
      { timeout: 20_000, polling: 'raf' }
    );
    await waitForActivityUpdateTick(page, 'left', 1);
    await expect(armMetric(armPanel(page, 'right'), 'Graph nodes / edges')).toHaveText(/\/\s*0$/);

    await pauseButton(page).click();
    await expect(statusRegion(page)).toHaveText('paused');
  });

  test('a WebGL-unavailable canvas shows a fallback message inside the activity panel, and the experiment still runs', async ({
    page
  }) => {
    await page.addInitScript(() => {
      const proto = HTMLCanvasElement.prototype;
      const original = proto.getContext;
      const stub = function (this: HTMLCanvasElement, type: string, ...args: unknown[]): unknown {
        if (typeof type === 'string' && type.toLowerCase().includes('webgl')) return null;
        return Reflect.apply(original, this, [type, ...args]);
      };
      proto.getContext = stub as typeof proto.getContext;
    });

    await page.goto('/');
    await waitForReady(page);
    // Not `expandActivityPanel` here: that helper asserts the canvas itself
    // becomes visible, which is exactly what must *not* happen on this path
    // (the canvas stays `visibility: hidden` while `sceneError` is set — see
    // `ActivityPanel.svelte`). `ActivityScene`'s construction fails
    // synchronously inside `expand()`, but the panel deliberately stays
    // "open" (`expanded` is not reverted) so this fallback is actually shown
    // instead of silently collapsing away — see that catch block's comment.
    await expect(activityToggle(page)).toBeEnabled({ timeout: 20_000 });
    await activityToggle(page).click();

    // Both the arena canvas and the activity canvas fail the same stubbed
    // `getContext`, so their fallback text is nearly identical
    // ("... could not create a WebGL context") — scope to the activity
    // panel specifically, both to disambiguate from the arena's own
    // "3D rendering is unavailable" message and to prove *this* panel's
    // fallback (not just the arena's, already covered by the
    // 'WebGL-unavailable fallback' describe block) actually rendered.
    const activitySection = page.locator('section.activity');
    // Thermo-maintainability I2 fix: the dynamic `sceneError` text is exposed
    // to assistive tech via `role="alert"` (matching the `contextLostMessage`
    // pattern), not hidden behind a static `role="img"` label — assert the
    // live region's actual accessible content contains the real reason, not
    // just that some visible text on the page happens to match it.
    await expect(activitySection.getByRole('alert')).toContainText(/could not create a WebGL context/i);
    await expect(activityCanvas(page)).toBeHidden();

    await startOrResumeButton(page).click();
    await expect(statusRegion(page)).toHaveText('running');
    await waitForTick(page, 5);
    await pauseButton(page).click();
    await expect(statusRegion(page)).toHaveText('paused');
  });
});

test.describe('performance gates with the activity panel open', () => {
  test('median neural step latency stays under the 33ms control budget with the activity panel expanded and streaming', async ({
    page
  }) => {
    await page.goto('/');
    await waitForReady(page);
    await expandActivityPanel(page);
    await startOrResumeButton(page).click();
    await waitForTick(page, 60);
    await pauseButton(page).click();
    await expect(statusRegion(page)).toHaveText('paused');

    const readLatencyMs = async (agent: 'left' | 'right'): Promise<number> => {
      const text = await armMetric(armPanel(page, agent), 'Neural step latency (median)').innerText();
      const match = text.match(/^(\d+(?:\.\d+)?) ms$/);
      if (!match) throw new Error(`Unexpected latency text for ${agent}: "${text}"`);
      const value = Number(match[1]);
      if (!Number.isFinite(value) || value < 0) throw new Error(`Non-finite/negative latency for ${agent}: "${text}"`);
      return value;
    };
    const [leftMs, rightMs] = await Promise.all([readLatencyMs('left'), readLatencyMs('right')]);
    // eslint-disable-next-line no-console -- perf-gate visibility.
    console.log(`[perf] (activity panel open) median neural step latency: left=${leftMs}ms right=${rightMs}ms (budget < 33ms)`);
    expect(leftMs).toBeLessThan(33);
    expect(rightMs).toBeLessThan(33);
  });

  test('no main-thread stall >= 200ms while running with the activity panel expanded and streaming', async ({ page }) => {
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
    await waitForRendererUp(page);
    await expandActivityPanel(page);
    expect(await page.evaluate(() => (window as unknown as { __longTaskSupported: boolean }).__longTaskSupported)).toBe(
      true
    );
    await page.evaluate(() => {
      (window as unknown as { __longTasks: unknown[] }).__longTasks = [];
    });
    await startOrResumeButton(page).click();
    await waitForActivityUpdateTick(page, 'left', 1);
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
    console.log(`[perf] (activity panel open) long tasks observed while running: ${JSON.stringify(longTasks)}`);
    for (const task of longTasks) {
      expect(task.duration).toBeLessThan(200);
    }
  });
});

test.describe('WebGL-unavailable fallback', () => {
  test('a canvas that cannot produce a WebGL context still lets the experiment run, with a clear fallback message', async ({
    page
  }) => {
    // Stub `getContext` before any app script runs so `THREE.WebGLRenderer`'s
    // own context-creation call fails exactly the way it would on a real
    // WebGL-unavailable browser: `ArenaScene`'s constructor wraps that call
    // and re-throws `ArenaSceneUnavailableError` (see its doc comment),
    // which `App.svelte`'s `onMount` catches into `rendererError` — entirely
    // independent of `ExperimentController`, which never touches the canvas.
    await page.addInitScript(() => {
      const proto = HTMLCanvasElement.prototype;
      const original = proto.getContext;
      const stub = function (this: HTMLCanvasElement, type: string, ...args: unknown[]): unknown {
        if (typeof type === 'string' && type.toLowerCase().includes('webgl')) return null;
        return Reflect.apply(original, this, [type, ...args]);
      };
      proto.getContext = stub as typeof proto.getContext;
    });

    await page.goto('/');
    await expect(page.getByText(/3D rendering is unavailable/)).toBeVisible({ timeout: 20_000 });
    await expect(page.getByRole('img', { name: 'Arena canvas unavailable' })).toBeVisible();

    // The experiment itself must still be fully usable: asset loading, the
    // state machine, and the Workers do not depend on the renderer at all.
    await waitForReady(page);
    await startOrResumeButton(page).click();
    await expect(statusRegion(page)).toHaveText('running');
    await waitForTick(page, 5);
    await pauseButton(page).click();
    await expect(statusRegion(page)).toHaveText('paused');
  });
});

test.describe('hash-mismatch integrity check', () => {
  test('a tampered artifact fails its sha256 check and the experiment moves to the error state (Start stays disabled)', async ({
    page
  }) => {
    const original = readFileSync(resolve(publicDataDir, 'malecns-arena-v1.bin.gz'));
    const tampered = Buffer.from(original);
    // Flip a byte well past the gzip magic number so this exercises the
    // gzip sha256 check specifically, not an incidental gzip-detection
    // change (see `assets.ts#verifyAndDecompressArtifact`'s doc comment).
    tampered[80] ^= 0xff;

    await page.route('**/data/malecns-arena-v1.bin.gz', (route) =>
      route.fulfill({ status: 200, contentType: 'application/octet-stream', body: tampered })
    );

    await page.goto('/');
    await expect(statusRegion(page)).toHaveText('error', { timeout: 20_000 });
    await expect(page.getByRole('alert')).toContainText(/sha256/i);
    await expect(startOrResumeButton(page)).toBeDisabled();
  });
});

test.describe('performance gates', () => {
  // Isolating these three tests from CPU contention with the rest of the
  // suite (each e2e test runs two neural Workers and a SwiftShader
  // renderer of its own) needs the whole run to have no *other* concurrent
  // worker, not just these three tests serialized against each other — see
  // `playwright.config.ts`'s `workers: process.env.CI ? 1 : undefined` for
  // where that's actually enforced, and why an earlier
  // `test.describe.configure({ mode: 'serial' })` attempt scoped to only
  // this block was insufficient (round-2 dual review finding) and has been
  // removed (it also had the unwanted side effect of skipping later tests
  // in this block after any one failure).

  test('median neural step latency stays under the 33ms control budget', async ({ page }) => {
    await page.goto('/');
    await waitForReady(page);
    await startOrResumeButton(page).click();
    await waitForTick(page, 60);
    await pauseButton(page).click();
    await expect(statusRegion(page)).toHaveText('paused');

    const readLatencyMs = async (agent: 'left' | 'right'): Promise<number> => {
      const text = await armMetric(armPanel(page, agent), 'Neural step latency (median)').innerText();
      // Anchored against `TelemetryPanel#formatMs`'s exact `"X.XX ms"` shape
      // rather than stripping all non-digits: that permissive parse would
      // silently turn an empty/unexpected string into `0`, which would then
      // pass the `< 33` budget assertion below for the wrong reason instead
      // of failing on a malformed read.
      const match = text.match(/^(\d+(?:\.\d+)?) ms$/);
      if (!match) throw new Error(`Unexpected latency text for ${agent}: "${text}"`);
      const value = Number(match[1]);
      if (!Number.isFinite(value) || value < 0) throw new Error(`Non-finite/negative latency for ${agent}: "${text}"`);
      return value;
    };
    const [leftMs, rightMs] = await Promise.all([readLatencyMs('left'), readLatencyMs('right')]);
    // eslint-disable-next-line no-console -- perf-gate visibility.
    console.log(`[perf] median neural step latency: left=${leftMs}ms right=${rightMs}ms (budget < 33ms)`);
    expect(leftMs).toBeLessThan(33);
    expect(rightMs).toBeLessThan(33);
  });

  test('no main-thread stall >= 200ms while running (long tasks reported; neural stepping itself runs in a dedicated Worker, never on the main thread)', async ({
    page
  }) => {
    await page.addInitScript(() => {
      const w = window as unknown as {
        __longTasks: unknown[];
        __longTaskSupported: boolean;
        __longTaskObserver?: PerformanceObserver;
      };
      w.__longTasks = [];
      // Recorded explicitly (not swallowed) so the assertion below can
      // require real coverage in Chromium rather than passing vacuously
      // whenever the observer happens not to be supported — a dual review
      // pass caught that an earlier version of this test's `try {
      // observer.observe(...) } catch {}` let an unsupported-entry-type
      // environment report zero long tasks and pass for the wrong reason.
      w.__longTaskSupported = PerformanceObserver.supportedEntryTypes?.includes('longtask') ?? false;
      if (w.__longTaskSupported) {
        const observer = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            w.__longTasks.push({ name: entry.name, duration: entry.duration, startTime: entry.startTime });
          }
        });
        observer.observe({ entryTypes: ['longtask'] });
        // Kept on `window` (not only in this closure) so the read below can
        // call `takeRecords()` on the exact same observer instance to flush
        // any entries the browser has already recorded but not yet
        // delivered to the callback — the callback can lag slightly behind
        // real time, and reading `__longTasks` immediately after Pause
        // could otherwise miss an entry from the run's very last moments.
        w.__longTaskObserver = observer;
      }
    });

    await page.goto('/');
    await waitForReady(page);
    // Wait for the renderer to be up *before* clearing the buffer, so its
    // one-time startup cost (chunk parse, `ArenaScene` construction, first
    // shader compilation) is excluded from this "while running" measurement
    // instead of being misattributed to it.
    await waitForRendererUp(page);
    expect(await page.evaluate(() => (window as unknown as { __longTaskSupported: boolean }).__longTaskSupported)).toBe(
      true
    );
    await page.evaluate(() => {
      (window as unknown as { __longTasks: unknown[] }).__longTasks = [];
    });
    // Cleared *before* the click (not after, as an earlier version of this
    // test did) so the click handler's own work is included in what this
    // measures, not silently excluded from the "while running" window.
    await startOrResumeButton(page).click();
    await waitForTick(page, 90);
    await pauseButton(page).click();
    await expect(statusRegion(page)).toHaveText('paused');

    const longTasks = await page.evaluate(() => {
      const w = window as unknown as {
        __longTasks: Array<{ duration: number }>;
        __longTaskObserver?: PerformanceObserver;
      };
      // Flush any entries already recorded but not yet delivered to the
      // observer's callback before reading — see the `addInitScript` doc
      // comment above.
      for (const entry of w.__longTaskObserver?.takeRecords() ?? []) {
        w.__longTasks.push({ duration: entry.duration });
      }
      return w.__longTasks ?? [];
    });
    // eslint-disable-next-line no-console -- perf-gate visibility: report what was actually measured either way.
    console.log(`[perf] long tasks observed while running: ${JSON.stringify(longTasks)}`);
    // Neural stepping happens exclusively inside `neural.worker.ts` — a
    // dedicated Worker thread — so no long task here can be *caused by*
    // neural stepping specifically; the 200ms ceiling below is a sanity
    // check against a genuine main-thread stall (e.g. rendering or GC
    // pathology), not the neural budget itself (that's the latency test
    // above; long tasks are by definition >= 50ms, well above that 33ms
    // per-step budget, so this test's name reflects the actual assertion
    // rather than overclaiming "no long tasks at all"). Reported regardless
    // of outcome, per the plan's "assert none attributable, or report
    // measured" allowance.
    for (const task of longTasks) {
      expect(task.duration).toBeLessThan(200);
    }
  });

  test('interactive load time under a throttled ("broadband") network profile', async ({ page, browserName }) => {
    test.skip(browserName !== 'chromium', 'Network throttling here uses the Chromium DevTools Protocol.');

    const client = await page.context().newCDPSession(page);
    await client.send('Network.enable');
    await client.send('Network.emulateNetworkConditions', {
      offline: false,
      // ~4 Mbps down / ~1 Mbps up, 40ms latency: a representative "broadband"
      // profile per the plan's <= 10s interactive-load budget, not a
      // worst-case mobile profile.
      downloadThroughput: (4 * 1024 * 1024) / 8,
      uploadThroughput: (1 * 1024 * 1024) / 8,
      latency: 40
    });

    const startedAt = Date.now();
    await page.goto('/');
    await waitForReady(page);
    const elapsedMs = Date.now() - startedAt;
    // eslint-disable-next-line no-console -- perf-gate visibility.
    console.log(`[perf] interactive load time under throttled network: ${elapsedMs}ms (budget <= 10000ms)`);

    if (elapsedMs <= 10_000) {
      expect(elapsedMs).toBeLessThanOrEqual(10_000);
    } else {
      // Per the plan's explicit allowance ("the build either meets published
      // budgets or reports the measured miss explicitly"): report rather
      // than fail the whole CI gate on this one soft budget, while still
      // failing on a genuine hang via the sanity ceiling below.
      console.warn(
        `[perf] MISS: interactive load time ${elapsedMs}ms exceeds the 10s throttled-network budget.`
      );
      expect(elapsedMs).toBeLessThan(60_000);
    }
  });
});
