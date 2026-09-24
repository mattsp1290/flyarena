# Browser experiment and replay

Prerequisite: validated causal engine from 02.

## New files
New `src/lib/counterfactual/{protocol,client}.ts`, `counterfactual.worker.ts`, `Workbench.svelte`, `PairedReplay.svelte` beneath new counterfactual directory in existing `src/lib/`. New focused unit/component tests under `tests/`, and `tests/e2e/counterfactual.spec.ts` under existing e2e directory.

## Worker boundary and lifetime
A dedicated Worker receives one validated run at a time and loads graph artifacts through existing `loadArenaArtifacts` using explicit BASE_URL-derived URLs. Send bounded progress between seeds; at most one active run. Controller owns an epoch/request id so obsolete results cannot replace a newer run. Cancel/leave terminates the worker immediately and invalidates the id; a subsequent run constructs a fresh worker. Errors including artifact mismatch, worker error/messageerror, invalid settings and unexpected worker exit surface visibly and allow retry. A controller-owned 30-second preparation deadline and 120-second no-progress watchdog bound stalled fetches or silent Workers; valid progress resets the watchdog, and a five-minute total run ceiling remains. Clear every timer on terminal transitions. Expiry terminates the Worker, invalidates its id and presents a retryable timeout. Test silent preparation and a silent run followed by successful retry. Do not silently run numerical work on the main thread. Terminate on success/error, retain only evidence needed for UI, and clear old results when submitting a new run.

Display target counts from verified graph metadata/arrays via a preparation result before enabling execution. Avoid detached shared buffers: only transfer worker-owned copies, or fetch in the experiment worker. Do not reuse the arena's live neural Workers or its state.

## Visitor journey
1. Navigate from Arena or open `#counterfactual`. Arena handoff copies configured seed/topology only, with prominent authored-decoder / zero-action-opponent wording. Returning to Arena initializes it cleanly; no promise to resume a discarded live episode.
2. Select graph topology, group, base seed, seed count, fork tick and horizon; defaults from 02. Show measured topology versus authored mapping/dynamics and the actual group count. No backend connection fields in this mode.
3. Run, observe seed progress, or cancel. Settings lock while running. An unavailable/mismatched graph shows a clear error instead of synthetic fallback.
4. Inspect a table of paired baseline/sham/lesion outcome means, score difference and descriptive interval; group body-ID disclosure; graph hash; explanation of checkpoint copying and post-fork environment divergence. Expose exact seed results as accessible text/table.
5. Choose a seed and scrub a single timeline driving separate baseline/lesion world panels, with each world showing both agents, food, hazards, agent trails and current score. Include fork tick and elapsed simulated time. Read the displayed post-fork score from each selected frame's captured score components minus its fork score, not from final outcomes. A scrub test must check differing intermediate score values against the directly stepped reference. SVG is sufficient; do not depend on WebGL for evidence inspection.
6. Download versioned JSON and see the CLI verification command. A link to DGX sandbox explains that it trains a separate authored model and demonstrates the same experimental method, with incomparable score units.

## Integration coverage
Real Chromium must run the real compiled graph and actual experiment Worker, inspect a nonempty target, complete a small matched sweep, select another seed, scrub replay, verify downloaded graph hash/settings/results and exact sham, cancel a longer sweep, and successfully run again. Exercise hash mismatch, navigation-away cancellation and browser back/forward. Check all modes at 390px and desktop, keyboard labels and no horizontal overflow. Observe long tasks during an active workbench run: no main-thread stall >=200ms. Existing arena performance gates remain unchanged.

Run the imported DGX browser test on `#dgx` with a temporary backend on an unused loopback port; GPU required for final local acceptance, optional/skipped in generic CI when no token configured. Synthetic tests and arena tests must use explicit view-scoped selectors to avoid accidental coverage of the wrong experiment. Default CI proves arena/workbench without Docker. Keep tokens out of traces, exports and committed screenshots.
