# Structural Thermonuclear review

- Branch: `feat/counterfactual-workbench`
- Base: `main`, `69cb9ed`
- Reviewed HEAD: `c9e8beb2aa40`
- Date: 2026-09-24
- Reviewer: Structural Thermonuclear
- Slug: `structural-thermonuclear`
- Role: Independently audit ownership, boundary cleanliness, canonical reuse and ambitious structural simplification, applying the complete live Cursor thermonuclear rubric.
- Stats: 60 files changed; 6,083 insertions, 20 deletions; two commits.
- Initial verdict at `c9e8beb2aa40`: **REQUEST_CHANGES**
- Effective verdict after remedy re-review below: **APPROVE**

The branch preserves the default connectome arena, adds a pure TypeScript matched-checkpoint intervention engine and dedicated Worker, supplies reproducible evidence/replay, and imports the separately labeled synthetic GPU sandbox. Most structural choices are strong: the existing arena is not replaced, canonical simulation equations are reused, and no shared training/protocol changes are scattered into unrelated paths. The sandbox's asynchronous job lifecycle still has several independent mutable owners inside its view. A reproducible reconnect/cancellation race starts multiple polling loops, so the lifecycle must be consolidated before merging.

## Complete live rubric application

Source: https://github.com/cursor/plugins/blob/main/cursor-team-kit/skills/thermo-nuclear-code-quality-review/SKILL.md

Retrieved source file: `/tmp/flyarena-cursor-thermo-review.md`; independently checked SHA-256 `7faca08b51b643b2ddd0836f92af15574444024685dcc1e677dbbb39ae8c9e8f`. This review applied the actual rules, not merely read the skill name.

0. **Ambitious simplification:** T1 identifies a concrete ownership reframing: one job session owns requests, timers and invalidation, removing incidental mutable state from the Svelte view. S1 removes redundant model-validation branches rather than relocating them.
1. **1,000-line threshold:** inspected every changed file's line count. No changed source/test file exceeds 1,000 lines. Only the historical numerical data file `docs/lab-benchmark.json` has 2,522 lines; it is data, not a sprawling implementation. Largest new source/style file is the scoped 529-line lab stylesheet.
2. **Spaghetti growth:** App's handoff is narrow and optional; shell routing does not leak mode checks into simulation. T1 flags the existing-in-source-branch but newly imported job flow's poll/cancel/reconnect branches as actual cross-cutting state complexity.
3. **Clean design beyond passing tests:** passing navigation and CPU/GPU journeys do not waive T1. Consolidating ownership is a merge requirement, not a cosmetic suggestion.
4. **Direct, boring code:** graph preparation, dedicated short-lived Worker, evidence and replay modules earn their separation. No framework or generic router is needed. The numerical comparison walks a known regenerated structure and limits tolerant fields; it does not silently accept arbitrary exports. Avoid replacing T1 with a generic reactive state framework.
5. **Types and boundaries:** strict request validation and closed target vocabulary are appropriate. Browser/API result assertions merit care, but the high-conviction issue here is asynchronous session ownership, not a demand to duplicate every Python schema in TypeScript. Narrow internal casts and non-null assertions correspond to canonical world invariants. S1 simplifies repeated boundary checks.
6. **Canonical layers/reuse:** checked the stepping path against canonical world/sensor/decoder/connectome modules. The separate synthetic model has distinct provenance, equations and export shape, and is not smuggled into the connectome engine. Source identity/hash validation uses existing artifact utilities. Replay derives captured score increments rather than inventing a second score evaluator.
7. **Orchestration/atomicity:** independent artifact reads are parallel; per-seed numerical work is intentionally sequential in one Worker. T1 catches non-atomic status application from concurrent requests. Worker client cancellation/deadlines are locally owned and cannot affect later Workers.

All primary questions and aggressive flag categories were considered in the review of the changed code: coupling/state growth, branch growth, justified modules, clear types, canonical ownership, serialization and atomic updates. The preferred remedy is deletion of competing lifecycle owners. There is no request for unrelated rewrites, speculative abstractions, or cosmetic churn.

**Approval bar:** approval requires no clear structural regression, obvious missed dramatic simplification, unjustified file-size explosion, scattered special cases, magical abstraction, needless type churn, canonical-layer leak, or materially missed decomposition. T1 currently fails the ownership, atomicity and decomposition parts of that bar. Re-review the actual fix and its delayed-response regression tests before approval.

## Review evidence

Read the branch diff and complete changed implementation/test files, plus plan and architecture/model contracts. Parsed the historical benchmark artifact and inspected its identity/structure; it is not newly generated experimental evidence. Independently reproduced T1 by extracting the exact `poll`, `cancel` and `reconnect` functions from the reviewed component, transpiling TypeScript and executing them with deferred status promises. The result was two concurrent GETs and two independent scheduled poll timers. No implementation files were edited by this reviewer. Existing full-suite results are recorded by the author in `docs/counterfactual-validation.md`; this review does not claim to have independently rerun those suites.

## Remedy re-review — 2026-09-24 14:05 UTC

Independently inspected the uncommitted correction in `src/lib/lab/session.ts`, its component/API integration and all three delayed-response regression tests. **T1 is resolved.** One small session now owns timer, API, job identity and request controller; controller identity serves as the operation token, so a separate generation counter would be redundant. Reconnect/cancel invalidate the old read first, every continuation checks identity/disposal, and the view no longer owns the polling machinery. This deletes competing ownership rather than merely moving the same branches into a larger abstraction. The explicit unavailable state on authoritative HTTP 404 is justified and preserves identity while permitting a new submission; transient failures retain ownership.

Independently executed `vitest run tests/unit/lab-session.test.ts tests/Lab.test.ts tests/Shell.test.ts tests/unit/lab-api.test.ts` using Node 22.22.3: **four files, nine tests passed**. The deferred mocks deliberately ignore abort and still cannot create multiple loops, overwrite cancellation or restart polling after disposal. Also inspected the newly added real CLI exact/tamper tests and corrected deployment wording; no additional maintainability finding.

Reviewed remedy content SHA-256:

- `session.ts`: `e30ddda0f778ed5c39a37a42f0796fb9bb4b42394977e5bd42d9f721492b1b38`
- `Lab.svelte`: `e38d0d3e415e322df34a88bd246389b4a9643278e797db5d1520da6420115c7b`
- `api.ts`: `6e903c8501fdc63d917a40d44486e7cc90b60eb258852b70d5013a295b31de9a`
- `lab-session.test.ts`: `4861a2b69114b667226174ddcc8c4094edb959b20157905cdd62d07316594073`

The full live rubric's structural approval bar is now met. **APPROVE** the reviewed source plus this correction. S1 is nonblocking. The root agent still owns the complete regression/live-backend rerun and final merge/deployment gates; this approval does not claim those operations have occurred.
