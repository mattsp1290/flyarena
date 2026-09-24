# Integrated counterfactual validation

Validated on 2026-09-24 in the isolated `feat/counterfactual-workbench` checkout,
against local main `69cb9ed`. Original synthetic lab source: `26c422f`.
Node 22.22.3, Linux ARM64, NVIDIA GB10. No Beans changes or production-service
restarts were made during implementation and validation.

## Completed gates

- `npm ci`, `npm run check` (zero errors/warnings), `npm run build` passed.
- `npm run test:unit`: 394 passed, one pre-existing optional trace test skipped.
- `CI=1 npm run test:e2e` with a temporary CUDA backend: all 17 passed, none skipped.
  Includes unchanged arena journeys, actual graph probes, setup handoff, cancel,
  navigation, corruption/retry, evidence download and CLI comparison, and two
  live sandbox journeys. Navigation kept the same GPU job, permitted cancel on
  return, and exported a job completed while the sandbox was hidden.
- Both sandbox browser tests also passed with CPU execution.
- `npm run test:subpath`: strict `/fly/` server passed all three views, real graph
  execution, provenance links, narrow viewport and return navigation. Root-level
  assets return 404. This gate is included in CI.
- Compiler Python suite: 32 passed. Separate training project: 119 passed,
  including its GPU test. Both used locked uv environments with injected tracing
  disabled; no production training process was changed.
- Rebuilt optional backend image and ran all nine backend unit tests with CUDA
  available, including frozen-weight CPU/CUDA agreement. Image ID:
  `sha256:ed601b4bd36af63ce14edc27e8181fe84c06fa5fc1226534742829b42cfe34b6`.
- `bash -n scripts/deploy.sh scripts/lab.sh` and offline backend packaging passed.
  Archive contained exactly the eight explicitly listed source files, no config
  or credentials. Temporary validation container was stopped after testing.

## Scientific and responsiveness evidence

Fixtures test complete non-aliasing forks, pre-scatter and every-substep masks,
all three graph topologies and identity, body-ID target membership, exact sham,
canonical evaluator and product-runner parity, real intermediate replay scores,
frame caps, request bounds, worker deadlines/retry/stale replies and tampered
exports. Same-runtime repetitions are exact.

Node versus Chromium on this machine showed hazard-coordinate differences up to
`1.1102230246251565e-16`, with zero leaves above the declared noise floor and no
outcome changes. Cross-runtime exact equality is therefore **not** claimed. The
browser journey uses the explicit numerical diagnostic; exact verification stays
strict and is never silently replaced by a tolerance comparison.

Counterfactual main-thread long tasks: none observed. Arena median neural step
latency: 0.6/0.7 ms, below 33 ms. Arena largest observed long task: 61 ms, below
200 ms. Throttled interactive load: 2071 ms, below 10000 ms. Desktop and 390px
mobile screenshots were inspected; paired replay and evidence controls fit.

## Review and merge record

The user-required live [Cursor thermonuclear rubric](https://github.com/cursor/plugins/blob/main/cursor-team-kit/skills/thermo-nuclear-code-quality-review/SKILL.md)
was retrieved on 2026-09-24. Content SHA-256:
`7faca08b51b643b2ddd0836f92af15574444024685dcc1e677dbbb39ae8c9e8f`.
Independent implementation reviews, findings, fixes and the final merge record
will be appended before completion. These checks are local evidence, not a claim
of remote CI success. Historical synthetic-model benchmarks remain separately
identified in `lab-validation.md`; they are not real-connectome GPU benchmarks.
