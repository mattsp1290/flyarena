# Local validation — 2026-09-23

This change was validated locally on the DGX Spark. No static or backend deployment was performed. The other agents' worktrees and the running LLM service were not modified.

## Hardware and runtime

- Architecture: aarch64; NVIDIA GB10; driver 580.126.09.
- Runtime: torch 2.12.0a0+0291f960b6.nv26.04.48445190; CUDA 13.2.
- Container: built from backend/Dockerfile and launched through scripts/lab.sh, with host loopback publishing, unprivileged UID and read-only root filesystem.
- Both GPU matrix multiplication and actual authenticated experiment jobs succeeded in the packaged runtime.

The NVIDIA runtime is pinned by digest in the Dockerfile. [NVIDIA's Spark container guidance](https://docs.nvidia.com/dgx/dgx-spark/ngc.html) and [hardware description](https://docs.nvidia.com/dgx/dgx-spark/hardware.html) informed the ARM64/unified-memory choice; actual local execution established compatibility.

## Measured experiment

The same settings were submitted sequentially through the live API on CUDA and CPU: seed 17, population 64, generations 12, training seeds 32, held-out seeds 32, ticks 240. Training evaluates 2,048 parallel worlds per generation. Held-out evaluation uses 352 independent arm/seed worlds. CPU uses two PyTorch threads. These are individual warm-process measurements under the machine's current load, not a general hardware benchmark.

| Measurement | GB10 CUDA | CPU |
| --- | ---: | ---: |
| Training | 2.609 s | 3.261 s |
| Evaluation | 0.241 s | 0.162 s |
| Initial training fitness | -10.71 | -10.71 |
| Final training fitness | 99.59 | 99.59 |
| Peak PyTorch tensor allocation | 35.00 MiB | Not measured |

Training was about 1.25× faster on GPU in this run; evaluation was faster on CPU. There is no claim that all workloads benefit from GPU execution. The tensor peak excludes driver context and other applications' allocations.

### Held-out CUDA effects

| Arm | Mean score | Paired Δ | Descriptive 95% interval |
| --- | ---: | ---: | --- |
| baseline | 96.96 | 0.00 | [0.00, 0.00] |
| sham | 96.96 | 0.00 | [0.00, 0.00] |
| disconnected | -1.17 | -98.13 | [-103.90, -92.36] |
| G1 | 41.39 | -55.57 | [-69.25, -41.89] |
| G2 | 58.40 | -38.56 | [-53.24, -23.89] |
| G3 | 54.21 | -42.75 | [-53.17, -32.32] |
| G4 | 4.35 | -92.61 | [-98.82, -86.40] |
| G5 | 19.80 | -77.16 | [-85.83, -68.49] |
| G6 | 70.23 | -26.73 | [-33.24, -20.22] |
| G7 | 8.18 | -88.78 | [-95.28, -82.28] |
| G8 | 15.06 | -81.90 | [-89.35, -74.46] |

These describe authored synthetic behavior. They are not biological findings. Raw per-seed metrics, settings, timings and original-export SHA-256 digests are retained in [lab-benchmark.json](lab-benchmark.json). Full original exports are local ignored files under `lab-results/`; no token is included.

## Verification coverage

- `npm run check`: no TypeScript or Svelte errors/warnings.
- `npm test -- --run`: 62 passing frontend/arena tests, including server rejection, reconnect without resubmission, and API errors.
- `npm run build`: production Vite build.
- `docker run --rm --network none --gpus all -v "$PWD/backend/tests:/tests:ro" --entrypoint python flyarena-lab:local -m unittest discover -s /tests -v`: 9 passing numerical and API tests, including actual CUDA parity, deterministic CPU repeat, sham equality, persistent masks, disjoint seeds, invalid weights/options, scalar world equations, retention, authentication, body limits, cancellation/completion races and safe failures.
- `LAB_TEST_TOKEN="$LAB_TOKEN" npm run test:e2e`: actual Chromium against packaged backend; completion, arm selection, replay scrubbing, JSON download, cancellation and 390px layout. The test captures desktop and mobile screenshots; these were visually inspected. CUDA mode uses `LAB_TEST_DEVICE=cuda`.
- The actual `python -m flyarena_lab.cli --reevaluate ... --device cuda` command produced **exactly equal** arms and replay frames in the same pinned runtime. CPU/CUDA small frozen-weight tests use 2e-4 absolute/relative tolerance and equal contact counts.
- `bash -n scripts/deploy.sh scripts/lab.sh` and offline backend packaging/content inspection. The source bundle contains only the eight explicitly listed source/runtime files, with no environment or result files.

## Reproduce the measured job

Launch the backend per README. Submit the following body to `/api/v1/jobs`, using its bearer token, then poll `/api/v1/jobs/<returned-id>` until terminal and retain `result`:

```json
{
  "seed": 17,
  "device": "cuda",
  "population": 64,
  "generations": 12,
  "training_seeds": 32,
  "heldout_seeds": 32,
  "ticks": 240
}
```

Repeat with `device` set to `cpu`. Timings vary with warmup, concurrent workloads and hardware state. No performance threshold is imposed by the tests. The CLI and [experimental contract](counterfactual-lab.md) describe frozen-weight reevaluation.

## Completion audit — 2026-09-24

Revalidated the implementation at `a32a80a` in the separate
`feat/dgx-counterfactual-lab` worktree. The reviewed plan remains under
`.agents/plans/dgx-counterfactual-lab/`; implementation is committed in
`70e0e80`, with subsequent runtime-configuration work in `a32a80a`.

- Node 22.22.3: `npm run check` reported zero errors/warnings;
  `npm test -- --run` passed all 62 tests; `npm run build` succeeded.
- `./scripts/lab.sh --build` reproduced image
  `sha256:ed601b4bd36af63ce14edc27e8181fe84c06fa5fc1226534742829b42cfe34b6`.
  All nine backend tests passed in that image with GPU access, including actual
  CPU/CUDA frozen-weight agreement and the service lifecycle tests.
- Started a separate temporary container through `scripts/lab.sh --cuda` on
  loopback port 18765, using a fresh ephemeral token. Ran the live Playwright
  test once with `LAB_TEST_DEVICE=cuda` and once with `LAB_TEST_DEVICE=cpu`;
  both passed without skips. Each exercised job submission, completion,
  intervention selection, timeline scrubbing, JSON export and cancellation.
  Desktop and 390px screenshots were visually inspected; no horizontal
  overflow or browser errors were reported.
- SHA-256 hashes of both retained benchmark exports match
  `docs/lab-benchmark.json`; their recorded results and runtime measurements
  match the exports. Reexecuting the CLI against the saved CUDA export in the
  pinned runtime reproduced all 11 arms and every sampled replay frame exactly.
- Shell syntax checks and `./scripts/deploy.sh --package-backend` passed.
  The archive contains only the eight intended backend/runtime files.

The temporary browser/backend processes were stopped after verification.
This audit used no Beans workflow and performed no deployment; the preexisting
lab and LLM services were left running unchanged. The historical private
activation is documented separately in `.agents/deployment.md`.
