import { flushSync } from 'svelte';

/**
 * Test-only helper for exercising a `.svelte.ts` runes module (e.g.
 * `activityLesionColorMode.svelte.ts`) outside of any mounted component.
 * `$state`/`$derived` work as plain reactive values wherever they're
 * declared, but `$effect` needs an enclosing *effect root* to attach its
 * cleanup to — normally a mounted component provides one implicitly. This
 * file (a `.svelte.ts` module itself, so the rune syntax below is valid) is
 * the one place that boundary is bridged: it must live in a rune-processed
 * file so a plain `tests/**\/*.test.ts` file can call `withEffectRoot`
 * without itself needing rune syntax.
 *
 * `fn` runs synchronously inside the root; `flushSync()` afterward forces any
 * `$effect`s created during that call to run their first pass immediately
 * (Svelte 5 batches effect flushes into a microtask otherwise), so the
 * returned `value` is already fully initialized by the time this returns.
 * `cleanup()` — call it in the test's own `afterEach`/at the end of the
 * test — destroys the root and every effect it owns, mirroring what a real
 * component's unmount does.
 */
export const withEffectRoot = <T>(fn: () => T): { value: T; cleanup: () => void } => {
  let value!: T;
  const cleanup = $effect.root(() => {
    value = fn();
  });
  flushSync();
  return { value, cleanup };
};
