# Integrated counterfactual validation

Validated on 2026-09-24 in the isolated `feat/counterfactual-workbench` checkout,
against local main `69cb9ed`. Original synthetic lab source: `26c422f`.
Node 22.22.3, Linux ARM64, NVIDIA GB10. No Beans changes or production-service
restarts were made during implementation and validation.

## Completed gates

- `npm ci`, `npm run check` (zero errors/warnings), `npm run build` passed.
- `npm run test:unit`: 404 passed, one pre-existing optional trace test skipped.
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
latency: 0.7/0.7 ms, below 33 ms. Arena largest observed long task: 62 ms, below
200 ms. Throttled interactive load: 2010 ms, below 10000 ms. Desktop and 390px
mobile screenshots were inspected; paired replay and evidence controls fit.

## Review and merge record

The user-required live [Cursor thermonuclear rubric](https://github.com/cursor/plugins/blob/main/cursor-team-kit/skills/thermo-nuclear-code-quality-review/SKILL.md)
was retrieved on 2026-09-24. Content SHA-256:
`7faca08b51b643b2ddd0836f92af15574444024685dcc1e677dbbb39ae8c9e8f`.
The two independent review directories are under
`reviews/feat-counterfactual-workbench-2026-09-24-135940-c9e8beb2aa40/`.
Causal Correctness found one Important expired-job lock; Structural Thermonuclear
found one Important overlapping-poll/late-response race. Both findings were fixed
by the small session owner in `63862b4`, then independently re-reviewed as
APPROVE (each reviewer ran nine focused tests). There were no Critical findings.
Both nonblocking suggestions were also addressed: one canonical evidence header
now owns model validation, and the plan status is current. The exact live rubric
is retained beside the review manifest.

After fixes, check/build, all 404 frontend tests, all 17 browser journeys with
CUDA, both CPU sandbox journeys and the strict subpath journey passed again.
The final browser suite observed no counterfactual long tasks. Temporary
validation services were stopped. Main and freshly fetched origin/main remained
`69cb9ed`; their trees were clean, so the reviewed integration is additive with
no main conflict. Beans was inspected read-only again: trained-readout execution
remains in progress; rewiring-null and anatomy plans are now published and ready.
The workbench does not depend on their unfinished contracts. These checks are local evidence, not a claim
of remote CI success. Historical synthetic-model benchmarks remain separately
identified in `lab-validation.md`; they are not real-connectome GPU benchmarks.


## Regenerable example

```bash
npm run experiment:counterfactual -- --seed 17 --seeds 4 --warmup 30 --horizon 30 --target bridge --topology biological --output /tmp/probe.json
npm run experiment:counterfactual -- --verify /tmp/probe.json
```

Actual graph binary SHA-256:
`f1a0f982ffdfba12ecb2206d064c3dc1ceaae7cffa06a22049093d69a79098f7`.
The local exact Node rerun passed. Mean post-fork baseline score was
0.001280176415256772; silenced score 0.0005642154207826707; paired difference
−0.0007159609944741011 with descriptive interval
[−0.0012024162004193209, −0.00022950578852888142]. Sham difference was exactly zero.
These are small authored-model movements, not a claim about animal behavior.
Seven real CLI tests cover exact regeneration and rejection of edited graph,
body ID, seed list, model version, replay and outcome evidence.
