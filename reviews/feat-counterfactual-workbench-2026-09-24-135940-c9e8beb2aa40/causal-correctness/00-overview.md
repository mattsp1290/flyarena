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

## Deployment packaging addendum — 2026-09-24

**Verdict: APPROVE.** Independently reviewed the narrow `scripts/deploy.sh` patch on top of merged revision `bba244a`, plus the complete deployment context. Creating the tar output in a `mktemp` file outside `dist` removes the archive-creation mutation of the directory tar is traversing. The archive retains the same source tree and exclusion. `set -e` stops before upload on failure; the EXIT trap removes an unfinished temporary archive; successful `mv` places the expected `dist/flyarena.tar.gz` before clearing that trap. The later public-verification trap remains independent. No deployment target, activation or verification semantics changed. No Critical or Important finding. Independently ran `bash -n scripts/deploy.sh` successfully; root owns the build-only and extracted-byte comparison acceptance run before merging this fix and retrying deployment.

## HTTP artifact hashing addendum — 2026-09-24

**Verdict: APPROVE.** Reviewed the narrow changes on top of `0d86c39`: canonical `sha256Hex`, pinned dependency and lock entry, known-vector/real-artifact tests, and the true non-secure-origin browser test including same-length corruption. The change retains SHA-256, hexadecimal encoding, existing byte-length checks and both compressed/uncompressed digest comparisons. It introduces no skip or origin-based exception. The async public contract stays compatible; the actual computation is synchronous, and the current artifact sizes are bounded and small. The `@noble/hashes` package is pinned at 2.4.0 with lockfile integrity, no runtime dependencies and no installation lifecycle hook; the imported SHA-256 and hex paths do not require Web Crypto. The Chromium hostname mapping makes `isSecureContext === false` a meaningful regression case. Reading checked-in gzip bytes for the corruption response avoids Node resolver dependence and still proves a digest check rather than a length check.

Independently removed global `crypto` and compared the replacement against Node's independent `createHash('sha256')` implementation for both committed graphs, compressed and uncompressed: all four digests matched. Measured Node calls took approximately 3.96, 1.48, 0.45 and 1.53 ms for 107,391 / 399,792 / 118,650 / 399,792 bytes. These local timings are not a browser-performance claim; root is running the existing actual-browser load/long-task gates. No Critical or Important finding. Approval covers the patch; merge/redeployment still requires those combined acceptance checks.
