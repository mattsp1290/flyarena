# Positive notes

- `src/lib/counterfactual/engine.ts:16–20` copies the complete world and rates into branch-owned buffers. This retains contact history, RNG state, previous transforms and accumulated scores, avoiding hidden differences at the fork.
- `engine.ts:24–34` masks rates before the first recurrent scatter and after every neural substep. Masking only after aggregate output would have tested a different intervention; the implementation and handcrafted graph test correctly distinguish it.
- `engine.ts:62–86` records post-fork increments and real per-frame scores; separate branches capture their own food and hazard states. Exact baseline/sham comparison catches accidental asymmetric execution.
- `tests/unit/counterfactual-engine.test.ts` checks both the headless evaluator and the product runner, rather than merely duplicating the new engine in a test oracle.
- `src/lib/counterfactual/evidence.ts` keeps exact verification as the default. The explicitly requested numerical mode preserves structural/discrete identity, limits continuous-field tolerance, and reports the failure to reproduce exactly.
- `src/lib/counterfactual/client.ts` gives each request a fresh Worker with local settled state, termination, bounded deadlines and stale-callback protection; cancellation does not leave background browser computation running.
- `src/Shell.svelte` preserves the arena component and isolates optional modules, while keeping the sandbox instance hidden and inert so navigation does not discard a live job.
- Backend `service.py` enforces one active compute thread, bounded retained jobs, bearer authentication, input limits and cancellation/completion publication under a lock. The optional server never becomes a requirement for real-graph browser experiments.
- The UI and docs consistently separate measured topology from authored dynamics/grouping and keep synthetic sandbox score units separate from arena outcomes.
