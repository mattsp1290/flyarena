import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  // A second dual-review pass (round 2) caught that scoping
  // `test.describe.configure({ mode: 'serial' })` to just the
  // `performance gates` block (an earlier fix attempt) only serialized
  // those tests *against each other* — with `fullyParallel: true`, other
  // spec files still ran concurrently on separate Playwright workers, so
  // the latency/long-task measurements could still share CPU with e.g. the
  // ~12s "disconnected control" test. `mode: 'serial'` also has an unwanted
  // side effect for three otherwise-independent gates: a failure in one
  // test skips every later test in the same serial block (Playwright's
  // documented behavior), which would silently skip the throttled-load-time
  // test's own report-or-pass logic if an earlier perf gate failed. Real
  // isolation needs the whole run to have no *other* worker to contend
  // with, not just the one block; restricting workers to 1 in CI (where
  // these gates' thresholds actually matter for CI stability, not merely
  // local dev speed) does that directly, without the serial-block side
  // effect. `test.describe.configure({ mode: 'serial' })` was removed from
  // `tests/e2e/arena.spec.ts`'s performance-gates block accordingly.
  workers: process.env.CI ? 1 : undefined,
  // In CI, `line` alone never writes `playwright-report/` (only the `html`
  // reporter does) — a dual review pass caught that `.github/workflows/ci.yml`'s
  // "Upload Playwright report" step was therefore always uploading nothing.
  // `html` alongside `line` keeps the terminal output CI actually reads live
  // while also producing a real report to upload as a failure artifact.
  reporter: process.env.CI ? [['line'], ['html', { open: 'never' }]] : 'list',
  use: {
    baseURL: 'http://127.0.0.1:4173',
    // Only captured on failure, and only in CI — this is diagnostic evidence
    // for a failed run, not something every local `npm run test:e2e` needs
    // to pay the capture cost for.
    trace: process.env.CI ? 'retain-on-failure' : 'off',
    screenshot: process.env.CI ? 'only-on-failure' : 'off'
  },
  webServer: {
    // CI already runs `npm run build` as its own separate step before
    // `npm run test:e2e` (so a build failure is reported on its own step,
    // not buried inside Playwright's webServer startup log) — building
    // again here would just repeat that work against the same `dist/`
    // output. Locally, `npm run test:e2e` is expected to build fresh.
    command: process.env.CI ? 'npm run preview' : 'npm run build && npm run preview',
    url: 'http://127.0.0.1:4173',
    reuseExistingServer: !process.env.CI
  }
});
