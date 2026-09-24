# Positive notes

- `src/lib/counterfactual/engine.ts:15-34`: complete detached world/rate forks reuse canonical observation, neural stepping, decoding and world functions. Masking before the first scatter and after each substep is explicit.
- `src/lib/counterfactual/engine.ts:59-80`: one branch runner serves baseline, sham and lesion; captured scores are actual tick-local state. This avoids three subtly different simulation implementations.
- `src/lib/counterfactual/client.ts:29-70`: short-lived Workers, settled guards, local timer cleanup and active-cancel identity make cancellation ownership clear. Preserve this local ownership principle when correcting the separate sandbox lifecycle.
- `src/Shell.svelte:44-55`: navigation and persistence are centralized. Main arena and workbench dispose naturally; the optional sandbox's hidden/inert wrapper protects job continuity and accessibility.
- `src/lib/counterfactual/evidence.ts:27-73`: exact reproduction remains the default. The explicit numerical diagnostic retains strict discrete identity and reports discrepancies rather than silently rounding evidence.
- `tests/unit/counterfactual-engine.test.ts:22-114`: tests use nonaliasing checks, positive intervention behavior and the preexisting evaluator/product runner, providing independent evidence rather than merely reflecting the new implementation.
- `scripts/deploy.sh:10-18` and `scripts/verify/serve-subpath.mjs:9-15`: offline packaging uses a source allowlist; the strict subpath server rejects root-relative assets instead of letting a permissive development server hide path regressions.
- Documentation consistently distinguishes measured topology, authored dynamics and a separate synthetic model, and records cross-runtime limitations rather than making unsupported exactness claims.
