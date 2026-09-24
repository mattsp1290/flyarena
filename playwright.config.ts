import { defineConfig } from '@playwright/test';

// Two independent static servers: the default root-base build (what most of
// the suite exercises) and a second build/preview with `--base /fly/`
// (`npm run build:fly` / `npm run preview:fly`), matching the non-root path
// production is actually deployed under (`scripts/deploy.sh`,
// `.agents/deployment.md`). Each build writes to its own `outDir`
// (`dist`/`dist-fly`) so the two webServers never race on the same output
// directory. This is the regression coverage for the root-absolute-asset-URL
// bug (`src/lib/paths.ts`): CI never caught it before because every existing
// build/preview here used the default root base. Any non-root base exercises
// this bug class equally well; `/fly/` here just mirrors `scripts/deploy.sh`'s
// default `DEPLOY_BASE` for familiarity — it does not need to stay in sync
// with a future `DEPLOY_BASE` change for this coverage to remain valid.
const ROOT_BASE_URL = 'http://127.0.0.1:4173';
const FLY_BASE_URL = 'http://127.0.0.1:4174/fly/';

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
    // Only captured on failure, and only in CI — this is diagnostic evidence
    // for a failed run, not something every local `npm run test:e2e` needs
    // to pay the capture cost for.
    trace: process.env.CI ? 'retain-on-failure' : 'off',
    screenshot: process.env.CI ? 'only-on-failure' : 'off'
  },
  // Each project owns its own `baseURL`. `fly-base` is scoped (via
  // `testMatch`) to just `fly-base.spec.ts` — it only re-covers the
  // base-path-sensitive slice (asset load -> ready -> Start -> running, and
  // the ledger's provenance links) against the non-root build, keeping the
  // added CI time small. `root-base` deliberately uses `testIgnore` rather
  // than a `testMatch` allowlist (a dual review pass caught that a `arena.
  // spec.ts`-only `testMatch` here would make any *future* spec file added
  // under `tests/e2e/` match neither project and silently never run,
  // anywhere, with no warning): every spec except `fly-base.spec.ts` keeps
  // running under `root-base` by default, the same as before this file had
  // more than one project. `workers: 1` in CI (above) serializes both
  // projects' tests, so they still never contend with each other for CPU.
  projects: [
    {
      name: 'root-base',
      testIgnore: /fly-base\.spec\.ts$/,
      use: { baseURL: ROOT_BASE_URL }
    },
    {
      name: 'fly-base',
      testMatch: /fly-base\.spec\.ts$/,
      use: { baseURL: FLY_BASE_URL }
    }
  ],
  webServer: [
    {
      // CI already runs `npm run build` as its own separate step before
      // `npm run test:e2e` (so a build failure is reported on its own step,
      // not buried inside Playwright's webServer startup log) — building
      // again here would just repeat that work against the same `dist/`
      // output. Locally, `npm run test:e2e` is expected to build fresh.
      command: process.env.CI ? 'npm run preview' : 'npm run build && npm run preview',
      url: ROOT_BASE_URL,
      reuseExistingServer: !process.env.CI
    },
    {
      // Unlike the root-base server above, CI has no separate "build with
      // --base /fly/" step, so this one always builds for itself (both in
      // CI and locally) before previewing — `npm run build:fly` is fast
      // (a few hundred ms; this is the same small app) so this stays cheap.
      command: 'npm run build:fly && npm run preview:fly',
      url: FLY_BASE_URL,
      reuseExistingServer: !process.env.CI
    }
  ]
});
