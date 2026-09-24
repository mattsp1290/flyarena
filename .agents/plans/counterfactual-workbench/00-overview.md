# Counterfactual workbench

Status: Ready. Two independent reviews and a fresh adversarial review completed; accepted findings are incorporated. This integration has not been implemented. The existing DGX sandbox is implemented on `feat/dgx-counterfactual-lab` and remains a separate source of reusable code.

## Application context
```json
{
  "application_context": {
    "has_active_users": false,
    "backward_compatibility_required": false,
    "feature_flags": "not-applicable",
    "confirmed_at": "2026-09-23T21:56:50.205676+00:00",
    "confirmation_digest": "c30d1dd45ed368d376678081d25bb9bd9ac87f9bb5dc8f79c8b8cd9b2ca8d5e1"
  }
}
```
Carry forward the user's confirmed context from the original counterfactual-lab plan. The newer plans in the main checkout also record no users and no compatibility requirement. No migration or feature flag is needed. Preserving main's arena and planned interfaces is an explicit integration requirement, independent of backward compatibility.

## Outcome and scope
Add a browser counterfactual workbench to the real connectome arena: warm up a seeded episode, fork the complete world and neural state, then compare baseline, sham, and a persistently silenced circuit group. Show paired post-fork effects and synchronized separate-world replay. Put this and the existing DGX synthetic lab in one navigable app, with the existing arena as the default landing view.

Change type: additive experiment capability and integration of an existing branch. Affected areas: new browser experiment engine/Worker/UI, entry-point navigation, a small arena handoff, imported DGX backend/UI, offline packaging and documentation. The plan ends with a branch tested against current main and a documented merge gate, a reviewed merge to main as explicitly requested after planning.

## Repository evidence and complementary work
Grounding snapshot: main `69cb9ed`; original lab `26c422f`. Implementation worktree is on `feat/counterfactual-workbench`, created from main. Resolve worktree locations with `git worktree list`; do not hard-code developer filesystem paths in code.

Read-only `~/git/beans/bin/bn plan list/status` and issue listing show `flyarena-plan-k36d` in progress: contract, GPU port, CEM, and TS evaluator complete; `flyarena-bigq` production training active; `flyarena-nom6` browser decoder integration blocked on it. `flyarena-plan-9yhp` owns the original arena. Additional ready, ignored local plans in the checkout containing main are `.agents/plans/rewiring-null/` (500 rewired graphs, trained sample and ledger histogram) and `.agents/plans/anatomical-activity-view/` (soma positions and rate streaming). They were not yet published in the Beans plan listing at inspection time and are not tracked in main. Locate that checkout with `git worktree list` before reading those relative paths. Do not copy or commit the other agents' planning files.

This work tests **within-circuit interventions from a matched checkpoint**. It does not repeat topology-wide null statistics, training, or anatomical rendering. It uses the same graph artifacts, authored equations, seeds, and provenance as main. The DGX lab remains useful as a trainable synthetic sandbox and is brought into the same app without presenting its scores as arena scores.

Verified interfaces: `src/lib/arena/world.ts` exports deterministic `createWorld`/`stepWorld`/`createSnapshot`; `WorldState.rngState` retains the complete random state. `stepWorld` returns a detached state. `src/lib/connectome/model.ts` exports `stepModel`, `runSubsteps`, and `aggregateOutputs`. `src/lib/connectome/constants.ts` owns K. `loadArenaArtifacts` verifies binary hashes. The real biological artifact currently contains 1,008 neurons and 46,311 edges: 20 neurons per input channel, 800 bridge neurons and 48 outputs. `scripts/training/episode.ts` is the current authored/parked scoring reference; it has a stale bb45 TODO, so equivalence must be tested, not assumed. Main's `App.svelte` owns arena lifecycle and should remain in place to minimize conflicts with the anatomical and trained-decoder plans.

## Design decisions
1. A new entry shell selects Arena, Counterfactual workbench, or DGX sandbox. Keep `App.svelte` as the arena component. Lazy-load optional modes; no backend or token is required by the default app or workbench.
2. Use main's TypeScript world and neural functions directly. A dedicated, short-lived experiment Worker runs bounded causal sweeps. Do not edit neural stepping or the existing live neural Worker protocol, which other plans own.
3. Study one chosen topology and one target group per run across matched seeds. Fork after warmup, with the right agent receiving zero action. Copy RNG, contacts, scores, neural rates and all other world state. Measure subsequent score increments. This is a replayable seeded setup, not a snapshot of the currently playing two-agent episode.
4. Authored decoder only in this milestone. Label this next to controls and in every export. Main's upcoming trained toggle remains independent. Do not silently interpret trained state as authored state in a handoff.
5. Retain the synthetic DGX backend's existing bounded service and verified CUDA runtime. No attempt to compare the 64-unit synthetic model's scores to MaleCNS arena scores, share weight schemas, or imply shared physics.
6. Develop and track locally. Beans is read-only planning input; do not create/claim issues or publish this plan there. Do not deploy during development or disturb existing services. The user subsequently authorized `scripts/deploy.sh` after all work is merged to main.

## Success criteria
- Main's arena still starts at the default URL and passes its full existing correctness, lifecycle, asset-integrity, and browser performance suites.
- A visitor can open the workbench from the arena's seed/topology setup, select an explicitly labeled group, run on the real verified graph without a backend, view baseline/sham/intervention results, scrub both futures, select seeds, and download evidence.
- Baseline equals sham exactly; fork copies cannot alias; selected neurons are zero before the first post-fork recurrent scatter and after every neural substep; baseline scores equal the existing authoritative authored/parked evaluator. Negative or zero effects are valid.
- Export/CLI regeneration reproduces deterministic results and verifies graph identity; tampered input and incompatible versions fail closed. Graph and world identities, exact target body IDs and all seeds are recorded.
- Existing GPU lab remains functional through the new navigation, including a real CUDA browser run, cancellation, replay and export on a temporary local container. Navigation cannot leak polling, workers or renders.
- Integration branch contains the reviewed plan, code, focused tests, validation evidence and docs, has a clean tree, incorporates latest main without losing another plan's changes, and passes the final merge gates.

## Risks and non-goals
A group may have no effect under the authored decoder. Report that honestly, with a sham and positive-control fixture; do not tune on the displayed seeds to manufacture an effect. Groups derive from authored input/output mappings, not anatomical regions. A parked agent still participates in world physics; label it as receiving zero action, not immovable. Matched RNG state is guaranteed at the fork; food placement can diverge after different contacts, so do not claim identical exogenous trajectories after intervention.

No new public backend endpoint, graph regeneration, null study, anatomical view, new model training, trained-readout adapter, live-arena state capture, arbitrary neuron editor or cross-model statistical comparison. No external repository capability is needed; all owners/consumers are in FlyArena.

The user selected the integrated workbench and later authorized scope expansion, required the live Cursor thermonuclear review and merge of all our work to main, and permitted deployment from main afterward. Performance bounds may be reduced only if measured responsiveness requires it, while preserving matched controls and multiple seeds. Coordinate changes through fresh main inspection rather than editing another agent's files.

## Document map
- [01-integration.md](01-integration.md): preserve main, import sandbox, navigation and packaging.
- [02-causal-engine.md](02-causal-engine.md): fork semantics, controls, evidence and replayability.
- [03-browser.md](03-browser.md): experiment Worker, controls, replay, real-browser acceptance.
- [04-execution-handoff.md](04-execution-handoff.md): ordered execution, review and merge gates.
