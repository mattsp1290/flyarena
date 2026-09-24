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

## Environment note (this host) — read this first

This host's global Datadog APM auto-injection (`DD_INJECTION_ENABLED=tracer`,
forced onto every Python process's `sys.path` via
`/opt/datadog-packages/datadog-apm-library-python/.../ddtrace/bootstrap`)
ships an IAST instrumentation aspect for `Tensor.expand` that raises
`RuntimeError: Boolean value of Tensor with more than one value is
ambiguous` on any `expand()` call producing more than one element — this
port's `PreparedGraph`/`broadcast_readout_weights` both call `expand()`.
Unrelated to this port's correctness; disable Datadog's Python
instrumentation for every `uv run` invocation, one of two ways:

```bash
DD_TRACE_ENABLED=false DD_IAST_ENABLED=false DD_APPSEC_ENABLED=false uv run pytest -v
# or, equivalently, for any command (pytest, a future flyarena-train CLI, ...):
training/scripts/run.sh pytest -v
```

`training/scripts/run.sh` wraps the three env vars and `exec`s `uv run
"$@"`, so WP3's automation (a script, a cron job, a CI runner) doesn't have
to rely on the vars being copy-pasted correctly by hand every time — it's a
one-line substitution for `uv run` everywhere in this project. (The env vars
can't be set from inside `training/tests/conftest.py` and have this effect:
ddtrace's auto-injection runs via `sitecustomize`/`PYTHONPATH` before any
user code executes, so by the time `conftest.py` runs it's too late — the
shell invocation is the only correct fix point. `conftest.py` sets them
anyway, defensively, in case some other code path reads `os.environ` at
runtime; see its module doc.)

If this project is ever run on a host without this injection, the env vars
(and `run.sh`) are harmless no-ops.

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

### Longer traces for event coverage

The committed golden fixtures (60 ticks × 4 seeds) never trigger a food
pickup, wall clamp, or hazard contact — a review pass caught that this left
`place_without_overlap`'s respawn path (and `step_world_batched`'s wall-clamp
and hazard-contact branches) untested even indirectly. `training/scripts/generate_long_traces.ts`
reuses `buildSeedTrace` from `scripts/training/export-traces.ts` (the actual
TS `createWorld`/`stepWorld`/rate-model rollout, unchanged) with a
configurable seed list and tick count — the committed exporter's CLI only
accepts its own fixed `TRACE_SEEDS`/`TRACE_TICKS`, so this is a separate
script rather than an edit to that exporter (see its own module doc for the
full reasoning). No action scripting was needed to reach useful coverage: a
hand-picked list of 13 seeds (found via a throwaway sweep over seeds 1..800,
picking for wall-clamp frequency plus a few for extra food/hazard variety;
see the script's `DEFAULT_SEEDS` comment) at 2000 ticks each, under the
existing closed-loop authored policy, produces:

```
food respawns=15, hazard contacts=42, wall-clamp ticks=63
```

`tests/conftest.py`'s `long_trace_dir` fixture generates this the same way
`include_world_trace_dir` does (skips with the manual command if Node isn't
found):

```bash
npx tsx training/scripts/generate_long_traces.ts --out training/runs/traces/event-coverage
```

`tests/test_world_event_coverage.py` uses it for two things: (1) a minimum-
event-count assertion directly against the generated trace (so this
coverage can't silently regress back to zero), and (2) the same
teacher-forced, fully-batched parity check `test_parity.py` runs against the
committed fixtures, but against a trace where the respawn fallback and
wall-clamp/hazard-contact branches actually execute — batching every
(seed, tick) row across all 13 seeds into one `WorldBatch` (`B = 26,000`)
and stepping it in a single `step_world_batched` call. Measured result: max
position/observation abs diff ~9e-16 (float64 round-off), exact event-count
match (respawns/pickups/hazard-contacts) against the recorded trace.

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

World-state/observation/event/free-running checks run through the batched
`step_world_batched`/`observe_batch` (`world.py`, `sensors.py`), on CPU in
these test runs; they work identically on CUDA (`device=` is threaded
through `world_batch_from_items`/`create_world_batch`), but aren't
parametrized onto it in `test_parity.py` since the committed golden traces
are small enough that CPU is already instant — see "World step batching"
below for the CUDA throughput numbers that actually matter for these ops
(B = 4096). The rate/output model and readout MLP are checked on both
devices where CUDA is available (`test_parity.py`'s `DEVICES`).

Also see `tests/test_world_batch.py` (reset-vs-`initialWorld` parity, a
property-based cross-check of `step_world_batched` against the per-item
reference oracle across random actions, and `validate_world_batch` tests)
and `tests/test_world_event_coverage.py` (parity + minimum-event-count
assertions against a longer, event-rich trace — see "Longer traces for
event coverage" below), both added by the batching fix.

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

### World step batching — before/after (thermo-architecture review finding #1)

A first review pass found that `step_world`/`observe_agent` were ported as a
Python `for` loop over per-item dataclasses for *all* physics (movement
integration, wall clamp, hazard motion/bounce, contact detection, score
updates) — not just the rejection-sampling placement the plan scopes to
per-item — which the reviewer measured at ~41,000 item-ticks/s combined
(world + sensors), ~200x slower than the already-batched GPU model step,
projecting to **~67 hours of pure Python object-churn for WP3's full 9-run
CEM sweep** at the plan's declared defaults. That's fixed: `world.py`'s
`step_world_batched`/`sensors.py`'s `observe_batch` are now dense `[B, ...]`
`torch` tensor ops (see `world.py`'s module doc); only `place_without_overlap`
(food-respawn placement, genuinely sequential rejection sampling) stays
per-item, and it now runs as a **masked** CPU fallback only for the batch
items that actually had a pickup that tick, not unconditionally for every
item every tick.

Before/after benchmark (`training/scripts/bench_combined_step.py`), CUDA,
`B = 4096`, `trace-graph` fixture, combined world-step + observe + rate-model
(4 substeps) + readout MLP per tick — "before" reproduces the pre-fix
architecture exactly (the per-item Python `step_world`/`observe_agent`,
preserved as the test oracle at `training/tests/reference_world.py`, driving
the *same* batched model+readout step "after" uses):

```
BEFORE (per-item world/observe + batched model/readout), B=4096:
  8.8 ticks/s, ~36,100 item-ticks/s
AFTER (fully batched world/observe/model/readout), B=4096:
  ~320 ticks/s, ~1,310,000 item-ticks/s
Speedup: ~35x
```

(Two runs measured 34.0x and 36.3x; the "before" number is in the same
order of magnitude as the review's own ~41,000 item-ticks/s world+sensors-only
measurement — this run's number is combined world+observe+model+readout, so
a slightly lower per-item rate is expected.)

**Re-estimated WP3 wall time** at the plan's declared defaults (`P = 256`,
`E = 16` ⇒ `B = 4096`; `T = 1800`; `G = 150`; 9 runs = 3 arms × 3 replicas):

| | Before (per-item) | After (batched) |
| --- | --- | --- |
| Per generation | ~185–204 s | ~5.5–5.6 s |
| Per run (`G = 150`) | ~7.7–8.5 h | ~0.23 h |
| All 9 runs | **~70–77 h (≈3 days)** | **~2.0–2.1 h** |

This does not include per-tick GPU↔CPU synchronization from
`validate_world_batch` (on by default in `step_world_batched`, matching
`stepWorld` always calling `validateStepState`; pass `validate=False` in a
rollout loop that already trusts its state) or WP3's own CEM bookkeeping
(elite selection, seed sampling) — both still TODO for WP3 — but the
Python-object-churn bottleneck the first review pass identified is gone,
and the batched pipeline is fast enough for WP3's calibration loop (many
short, iterable runs), which the plan's "Unresolved decisions" table
requires and the pre-fix architecture could not deliver.

## Hardening (this fix)

Two other review findings, both fixed alongside the batching work above:

- **`validate_world_batch`** (`world.py`) ports TS `stepWorld`'s
  `validateStepState` guard (finite-value + clock-consistency checks) as a
  batched, cheap `torch.isfinite(...).all()`-based check, run by default at
  the start of every `step_world_batched` call (matching `stepWorld` always
  calling `validateStepState`); pass `validate=False` to skip it in a
  perf-critical rollout loop. See `tests/test_world_batch.py`'s
  `test_validate_world_batch_*`/`test_step_world_batched_*` tests.
- **`graph.load_graph_json`** now validates the loaded JSON (required keys,
  array lengths against `neuronCount`/`edgeCount`, index bounds, finite
  values, `presynapticOffsets`' CSR invariants) and raises
  `InvalidGraphJsonError` (a `ValueError`, mirroring `scripts/data/binfmt.py`'s
  `InvalidGraphError` convention) naming the malformed field, instead of a
  bare `KeyError` at the load site or an opaque shape mismatch several
  frames away inside `model.PreparedGraph.__init__`. This matters once
  WP4/WP5 feed real (less-trusted) MaleCNS exports through this loader — see
  `tests/test_graph_validation.py`.

## Deviations from the plan (summary)

- **Config-drift check** reads the `configFingerprint` already embedded in
  each committed golden seed file instead of a new
  `tests/fixtures/golden/arena-config.json` (see "Config-drift gate" above
  for the full reasoning — no such file exists, and creating one would
  require editing `tests/unit/golden-traces.test.ts`, out of this work
  package's change surface).
- **Reset placement (`create_world_item`) and food-respawn placement
  (`place_without_overlap`, called from `step_world_batched`'s masked
  fallback) stay per-item, on CPU.** Both are `placeWithoutOverlap`:
  sequential, data-dependent rejection sampling, not vectorizable across a
  batch without either a different algorithm or reduced fidelity. The plan
  itself directs reset placement to run "on CPU per seed"; this port applies
  the identical reasoning to respawn placement, since it is the same
  function for the same underlying reason — but the respawn fallback now
  runs only for the batch items that actually need it each tick (see "World
  step batching" above), not unconditionally for every item. Everything
  else — movement integration, wall clamping, hazard motion/bounce, contact
  detection, score bookkeeping, sensors, and action decoding — is dense
  `[B, ...]` `torch` tensor ops (`step_world_batched`, `observe_batch`,
  `decode_action_batch`), matching the plan's "batched world state tensors
  `[B, ...]`" module description.
- **`validate_world_batch`'s clock check uses a small float64 tolerance
  (`1e-9`) instead of TS's `Object.is` bit-exact comparison** — see
  `validate_world_batch`'s docstring in `world.py` for why (a `WorldBatch`
  built from externally-serialized JSON has already round-tripped through
  two independent language runtimes, unlike TS's in-process check).
  `schemaVersion` is not tracked (nothing in this port ever sees a
  different one).
- No GPU fallback was needed: the cu128 wheel worked on the first attempt.
