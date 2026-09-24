# Causal Correctness review

- Branch: `feat/counterfactual-workbench`
- Date: 2026-09-24
- Reviewer display name: Causal Correctness
- Reviewer slug: `causal-correctness`
- Role: Independent scientific-invariant, evidence-integrity, and asynchronous-lifecycle review.
- Reviewed revision: `c9e8beb2aa40` against `main` at `69cb9ed`.
- Stats: 60 files changed, 6,083 additions, 20 removals, 2 commits.
- Verdict: **REQUEST_CHANGES**

The branch adds a real-connectome counterfactual engine, bounded Worker execution, paired replay and exact/numerical evidence regeneration, then integrates it with the existing arena and a separate synthetic DGX service. The numerical design preserves the intended fork and intervention semantics and reuses the authoritative equations. A recoverability defect remains in the imported sandbox lifecycle: an expired job leaves the retained component permanently locked. Fix that before merge. No critical issue was identified.

Reviewed changed source, tests, scripts and documents in context; inspected the large historical benchmark JSON by structure, provenance and arm summaries. Checked canonical world, neural and artifact contracts. Reproduced the expired-job lock with the real Svelte component in a temporary Vitest test, which passed assertions demonstrating the bug; removed the scratch test afterward. Existing full-suite evidence is in `docs/counterfactual-validation.md`. Later root additions are outside this pinned review.

## Remediation follow-up — 2026-09-24 14:04 UTC

Inspected the subsequent uncommitted `LabSession` extraction, typed `LabApiError`, updated Lab component, and three new session regression tests. The authoritative 404 transition now releases ownership into `unavailable` while preserving the diagnostic ID; transient failures retain ownership; request-local controller identity rejects obsolete responses. The reported Important finding is resolved. Independently ran `npx vitest run tests/unit/lab-session.test.ts tests/Lab.test.ts tests/Shell.test.ts tests/unit/lab-api.test.ts`: **9 tests passed, 4 files**. Follow-up verdict for the reviewed implementation plus this remedy: **APPROVE**, subject to the root's final combined-suite and merge gates. The original pinned-revision verdict above remains the historical review record.
