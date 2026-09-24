/**
 * Resolve a runtime public-asset URL against Vite's own base path, so it
 * keeps working whether the app is served from the origin root (dev,
 * `vite preview` with no `--base`, and every existing test) or from a
 * non-root deployment path such as `/fly/` (`scripts/deploy.sh` always
 * builds with `--base /fly/`; see `.agents/deployment.md`).
 *
 * A previous version of `src/lib/experiment/assets.ts` and
 * `src/lib/ui/LedgerPanel.svelte` hard-coded root-absolute URLs
 * (`/data/...`). Under `/fly/` those resolved against the origin root
 * instead of the deployment path, so the manifest fetch 404'd, the
 * experiment state went straight to `error`, and Start never enabled — CI
 * never caught it because `playwright.config.ts` built/previewed with the
 * default root base. Every runtime-fetched public asset URL must be built
 * from this one helper instead of a literal leading slash.
 *
 * `base` defaults to `import.meta.env.BASE_URL`, which Vite guarantees
 * always ends in `/` (root deployments get `'/'`; a `--base /fly/` build
 * gets `'/fly/'`; Vitest/jsdom resolves it to `'/'` too, so this is correct
 * under `npm run test:unit` with no extra setup). It is still an explicit
 * parameter — not read internally on every call — so callers and unit
 * tests can inject an arbitrary base (e.g. `'/fly/'`) without mocking
 * `import.meta.env`.
 */
export const publicAssetUrl = (relative: string, base: string = import.meta.env.BASE_URL): string => {
  const trimmedBase = base.replace(/\/+$/, '');
  const trimmedRelative = relative.replace(/^\/+/, '');
  return `${trimmedBase}/${trimmedRelative}`;
};
