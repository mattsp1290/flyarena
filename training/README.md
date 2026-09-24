# flyarena-training

A batched PyTorch port of the FlyArena arena, sensors, action decoder, rate
model, and readout MLP, for GPU-accelerated CEM training on the DGX Spark
(NVIDIA GB10) and a tolerance-based parity gate against the TypeScript
implementation's golden traces. See
`.agents/plans/trained-readout/02-gpu-port-and-parity.md` (WP2) for the
full design and `00-overview.md` for the wider trained-readout plan this is
one work package of.

This is a separate `uv` project so `torch` never enters the root
`pyproject.toml`/`uv.lock`, which are reserved for the offline graph
compiler (bean `flyarena-qyxw`).

## Setup on the Spark

```bash
cd training
uv sync
```

`uv sync` reads `.python-version` (pinned to `3.12`, matching this host's
system interpreter) and `[tool.uv.sources]`/`[[tool.uv.index]]` in
`pyproject.toml`, which pin `torch` to the `pytorch-cu128` index
(`https://download.pytorch.org/whl/cu128`). That index ships Linux wheels
only; this project is Spark/GB10-specific by design (see the plan's
"Separate uv project at `training/`" decision), so `uv sync` is not expected
to succeed on a non-Linux dev machine or a CI runner without a matching
wheel.

### GPU check (stop/go gate)

```bash
uv run python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); x=torch.randn(1024,1024,device='cuda'); print((x@x).sum().item())"
```

**Result on this host: PASS on the first index tried (cu128).**

```
NVIDIA GB10
43930.19921875
torch 2.11.0+cu128, CUDA 12.8, device capability (12, 1) [sm_121]
torch.cuda.get_arch_list() = ['sm_80', 'sm_90', 'sm_100', 'sm_120']
```

The GB10 reports SM 12.1 (Blackwell-class); the cu128 wheel ships `sm_120`
kernels, which run on it via PTX/minor-version forward compatibility. No
need was found to try cu129/cu130 — recorded here per the plan in case a
later run needs the fallback path.

### Precision settings

`flyarena_training/__init__.py` sets, at package-import time (before any
simulation tensor op can run):

```python
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
torch.use_deterministic_algorithms(True, warn_only=True)
```

PyTorch enables TF32 for float32 matmul by default on Ampere+ GPUs
(including the GB10), which would silently blow the tolerance-based parity
gate below. The `use_deterministic_algorithms` call is the plan's "Risks and
exclusions" mitigation for `torch.sparse`'s CUDA kernels (the CSR
`torch.sparse.mm` `model.py`'s `step_model` uses) possibly being
nondeterministic across runs; `warn_only=True` degrades an operation with no
deterministic implementation to a logged warning rather than an exception —
none were observed on this host during the full test run.
`tests/test_parity.py::test_precision_settings_applied` asserts all four
settings after import.

## Running the parity suite

```bash
cd training
uv run pytest -v
```

Passes on CPU always; the CUDA-parametrized cases and `test_gpu.py`'s
throughput smoke test run automatically in addition when CUDA is available,
with the *same* tolerances (see `tests/test_parity.py`'s `DEVICES`).

`FLYARENA_TRACE_DIR` overrides the trace directory the parity suite reads
from (default `tests/fixtures/golden/`, resolved relative to the repo
root), so WP5 can point this suite at real-graph traces without code
changes:

```bash
FLYARENA_TRACE_DIR=/path/to/real-graph-traces uv run pytest -v
```

### The `--include-world` trace, and why it's needed

The committed golden fixtures (`tests/fixtures/golden/`) deliberately omit
a per-tick world-state column to stay inside their committed byte budget
(see `scripts/training/export-traces.ts`'s module doc). That's enough for
the rate/output/readout parity checks (they teacher-force from the
observation and rate columns the golden files already have), but the
world-state, observation-beyond-tick-0, exact-event, and free-running
checks need the full per-tick world state.

`tests/conftest.py`'s `include_world_trace_dir` fixture generates it
automatically, into a session-scoped temp directory, by invoking:

```bash
npm run training:traces -- --include-world --out training/runs/traces/trace-graph
```

with Node resolved from `PATH` or, failing that, the pinned nvm location
(`~/.nvm/versions/node/v22.22.3/bin`). If Node can't be found or the export
fails, the fixture `pytest.skip()`s the tests that need it with that exact
command printed, so `uv run pytest` still runs everything it can standalone.
On this host, Node was found and the trace generated successfully every run
during development of this port — the skip path was verified by temporarily
hiding `node`/nvm from `PATH`.

### Config-drift gate — a deliberate deviation from the plan's literal wording

The plan's tolerance table says: *"A config-drift test fails if
`ARENA_CONFIG` in TS changes without a regenerated
`tests/fixtures/golden/arena-config.json`."* No such file exists anywhere
in this repository (confirmed by grepping the tree before writing this
port), and `scripts/training/export-traces.ts`'s `buildGoldenFiles` does not
produce one — `tests/unit/golden-traces.test.ts` (owned by WP1, out of
scope for this work package) asserts the committed file set matches
`buildGoldenFiles`'s output exactly, so adding a new committed file would
require editing that test too, which is out of this work package's
boundaries per its brief.

Every committed golden seed file already embeds a `configFingerprint`
string (`createArenaConfigFingerprint` in `src/lib/arena/config.ts`), which
*is* a TS-produced encoding of every `ARENA_CONFIG` field.
`config.py`'s `assert_matches_fingerprint` parses that string and checks
`config.py`'s mirrored `ARENA_CONFIG` against it field-by-field;
`test_config_matches_every_seed_fingerprint` runs it against all four
committed seeds. This satisfies the drift-detection intent (a TS
`ARENA_CONFIG` change without updating `config.py`'s mirror now fails a
test) without creating a new committed fixture file outside this work
package's change surface.

## Parity results (measured on this host)

All figures from `uv run pytest -v -s` against the committed
`tests/fixtures/golden/` fixtures (the `trace-graph` fixture: 24 neurons,
107 edges, 4 committed seeds × 60 ticks) plus a conftest-generated
`--include-world` trace for the world-state/observation/free-running rows.

| Quantity | Tolerance | Max abs diff observed (CPU) | Max abs diff observed (CUDA) |
| --- | --- | --- | --- |
| World state after `step_world` | abs ≤ 1e-9 | **0.000e+00** (exact) | n/a (CPU-only; see design note) |
| Observation | abs ≤ 1e-9 | 5.83e-16 | n/a |
| Rates after 4 substeps | abs ≤ 1e-5 or rel ≤ 1e-4 | 1.19e-07 | 1.19e-07 |
| Aggregated outputs (from own rates) | abs ≤ 1e-5 | 1.19e-07 | 1.19e-07 |
| Aggregated outputs (from recorded rates, isolated) | abs ≤ 1e-5 | 1.19e-07 | 1.19e-07 |
| Readout outputs | abs ≤ 1e-6 | 5.96e-08 | 1.19e-07 |
| Food respawn / hazard contact events | exact | **exact** (all seeds, all ticks) | n/a |
| Free-running positions, 60 ticks | abs ≤ 1e-4 | **0.000e+00** (all 4 seeds) | n/a |

World-state/observation/event/free-running checks are pure-Python
(`world.py`, `sensors.py`; see those modules' doc comments for why — the
placement rejection sampling is inherently sequential/data-dependent, not a
dense GPU op), so they have no separate CUDA variant; they ran once, on
CPU, and their result applies regardless of the GPU gate. The rate/output
model and readout MLP are the actual GPU-batched pieces and are checked on
both devices where CUDA is available.

All tolerances passed with wide margin — the rate/output/readout residuals
(~1e-7, tolerance 1e-5/1e-6) are consistent with the plan's documented
float32-rounding-order discrepancy between TS's per-edge `Float32Array`
accumulation and a dense sparse matmul (see `model.py`'s module doc); the
world-state/observation/free-running residuals are at or below float64
round-off (both TS and Python use IEEE-754 binary64 for this path, and this
port mirrors each TS line's exact operation order).

## Throughput (informational; no gate)

`tests/test_gpu.py::test_batched_step_b4096_on_cuda`, batched rate-model
step (`run_substeps`, `substeps=4`) at `B = 4096` on the `trace-graph`
fixture (24 neurons, 107 edges), CUDA, `torch==2.11.0+cu128`:

```
2116.0 ticks/s (8464.1 substeps/s, 0.473 ms/tick, 200 ticks measured)
```

This is a very small graph (24 neurons); at this size, throughput is
dominated by CUDA kernel-launch/Python-dispatch overhead per `run_substeps`
call rather than by the sparse matmul's FLOPs. Expect a different profile
on a MaleCNS-scale graph (thousands of neurons, WP5) — re-measure there
before drawing throughput conclusions for the production CEM run. (This
number is with `torch.use_deterministic_algorithms(True, warn_only=True)`
enabled per the plan's reproducibility mitigation, added after the first
review pass — about 30% lower than the ~3170 ticks/s measured without it;
at this tiny graph size the run-to-run variance from kernel-launch overhead
is comparable in magnitude, so treat both figures as order-of-magnitude,
not precise.)

## Environment note (this host)

This host's global Datadog APM auto-injection (`DD_INJECTION_ENABLED=tracer`,
forced onto every Python process's `sys.path` via
`/opt/datadog-packages/datadog-apm-library-python/.../ddtrace/bootstrap`)
ships an IAST instrumentation aspect for `Tensor.expand` that raises
`RuntimeError: Boolean value of Tensor with more than one value is
ambiguous` on any `expand()` call producing more than one element — this
port's `PreparedGraph`/`broadcast_readout_weights` both call `expand()`.
Unrelated to this port's correctness; disable Datadog's Python
instrumentation for `uv run pytest` invocations:

```bash
DD_TRACE_ENABLED=false DD_IAST_ENABLED=false DD_APPSEC_ENABLED=false uv run pytest -v
```

If this project is ever run on a host without this injection, the env vars
are harmless no-ops.

## Deviations from the plan (summary)

- **Config-drift check** reads the `configFingerprint` already embedded in
  each committed golden seed file instead of a new
  `tests/fixtures/golden/arena-config.json` (see "Config-drift gate" above
  for the full reasoning — no such file exists, and creating one would
  require editing `tests/unit/golden-traces.test.ts`, out of this work
  package's change surface).
- **World state, RNG placement/respawn, sensors, and action decoding are
  plain Python (not dense `torch` tensors).** The plan's module list
  describes `world.py` as "batched world state tensors `[B, ...]`"; this
  port batches at the *list-of-instances* level (`WorldItem` per seed)
  rather than as dense arrays, because `placeWithoutOverlap` (used both at
  reset and for food respawn during `stepWorld`) is sequential, data-dependent
  rejection sampling — not vectorizable across a batch without either a
  different algorithm or reduced fidelity. The plan itself directs reset
  placement to run "on CPU per seed"; this port applies the identical
  reasoning to respawn placement, since it is the same function for the
  same underlying reason. The dense, GPU-batched piece is exactly the piece
  the plan's throughput requirement targets: the rate model (`model.py`,
  CSR sparse matmul) and the readout MLP (`readout.py`).
- No GPU fallback was needed: the cu128 wheel worked on the first attempt.
