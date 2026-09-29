/**
 * WP1 of `.agents/plans/consolidated-release` (`01-findings-sections.md`),
 * fix pass after thermo review (maintainability S1): the pure "which
 * flattened step is currently visible, and which visible step is nearest a
 * given index" search `FindingsPanel.svelte`'s accordion navigation needs,
 * pulled out of the component so it's unit-testable directly (no Svelte
 * mount, no `jsdom`, no `fireEvent`) and reusable if a future collapsible
 * list ever needs the same shape.
 *
 * Every function here takes `flatSteps`/`collapsedSectionIds` as plain
 * arguments rather than closing over component state, mirroring
 * `./sections.ts`'s own "pure functions over plain arrays, no Svelte
 * imports" convention.
 */

/** The minimal shape these functions need from a flattened step entry -- `FindingsPanel.svelte`'s own richer `{ step, sectionId, sectionTitle }` entries satisfy this structurally. */
export interface FlattenedFindingStep {
  readonly sectionId: string;
}

/** Whether `flatSteps[index]` exists and its section is not in `collapsedSectionIds`. */
export const isStepVisible = (
  flatSteps: readonly FlattenedFindingStep[],
  index: number,
  collapsedSectionIds: ReadonlySet<string>
): boolean => {
  const entry = flatSteps[index];
  return entry !== undefined && !collapsedSectionIds.has(entry.sectionId);
};

/**
 * The nearest still-visible step to `fromIndex` under `collapsedSectionIds`
 * -- searched forward first (the natural "keep reading on" order), falling
 * back to backward when `fromIndex` was the last visible step. `undefined`
 * only when no step in `flatSteps` is visible at all.
 */
export const nearestVisibleIndex = (
  flatSteps: readonly FlattenedFindingStep[],
  fromIndex: number,
  collapsedSectionIds: ReadonlySet<string>
): number | undefined => {
  for (let i = fromIndex + 1; i < flatSteps.length; i += 1) {
    if (isStepVisible(flatSteps, i, collapsedSectionIds)) return i;
  }
  for (let i = fromIndex - 1; i >= 0; i -= 1) {
    if (isStepVisible(flatSteps, i, collapsedSectionIds)) return i;
  }
  return undefined;
};

/**
 * Whether at least one step in `flatSteps` would still be visible under
 * `collapsedSectionIds`. Thermo accessibility review (Important I1):
 * `FindingsPanel.svelte`'s stepper contract (exactly one `aria-current="step"`
 * cursor, with a working Previous/Next pair rendered inside that step's own
 * `<li>`) requires at least one visible step at all times -- `toggleSection`
 * uses this to refuse a collapse that would leave none, rather than reach a
 * state with zero visible steps and a stale live-region announcement.
 */
export const hasAnyVisibleStep = (
  flatSteps: readonly FlattenedFindingStep[],
  collapsedSectionIds: ReadonlySet<string>
): boolean => flatSteps.some((_, index) => isStepVisible(flatSteps, index, collapsedSectionIds));
