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
env -u PYTHONPATH DD_TRACE_ENABLED=false DD_IAST_ENABLED=false DD_APPSEC_ENABLED=false uv run pytest -v
# or, equivalently, for any command (pytest, a future flyarena-train CLI, ...):
training/scripts/run.sh pytest -v
```

`training/scripts/run.sh` wraps the three `DD_*` env vars, unsets
`PYTHONPATH`, and `exec`s `uv run "$@"`, so WP3's automation (a script, a
cron job, a CI runner) doesn't have to rely on the vars being copy-pasted
correctly by hand every time — it's a one-line substitution for `uv run`
everywhere in this project. (The env vars
can't be set from inside `training/tests/conftest.py` and have this effect:
ddtrace's auto-injection runs via `sitecustomize`/`PYTHONPATH` before any
user code executes, so by the time `conftest.py` runs it's too late — the
shell invocation is the only correct fix point. `conftest.py` sets them
anyway, defensively, in case some other code path reads `os.environ` at
runtime; see its module doc.)

Also run every `training/` command with `PYTHONPATH` unset, not just these
three `DD_*` vars set -- an inherited `PYTHONPATH` from the shell (e.g. left
over from another project) can reintroduce the same auto-injection this
section disables, even with the three `DD_*` vars set, because
`sitecustomize`/`ddtrace`'s bootstrap hooks into whatever is already on
`sys.path`. `training/scripts/run.sh` unsets it for you; if invoking `uv
run` directly instead, unset it yourself. Observed on this host (2026-09-24):
`training/` pytest failed 30/119 tests under the injected IAST
instrumentation with a bare `uv run pytest`; with `DD_IAST_ENABLED=false`
and `PYTHONPATH` unset, all 119 passed (see
`.agents/plans/rewiring-null/00-overview.md`'s "Risks and assumptions").

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

This includes `validate_world_batch`'s per-tick GPU↔CPU synchronization (on
by default in `step_world_batched`, matching `stepWorld` always calling
`validateStepState`); see "`validate_world_batch` host-sync fusion" below
for its now-measured, now-fused cost. It does not include WP3's own CEM
bookkeeping (elite selection, seed sampling) — still TODO for WP3 — but the
Python-object-churn bottleneck the first review pass identified is gone,
and the batched pipeline is fast enough for WP3's calibration loop (many
short, iterable runs), which the plan's "Unresolved decisions" table
requires and the pre-fix architecture could not deliver.

### `validate_world_batch` host-sync fusion (thermo-fix-verification review)

A review pass found `validate_world_batch` written as ~11 sequential Python
`if (<tensor op>).any(): raise ...` statements. Evaluating `if` on a tensor
forces an implicit host sync (the boolean reduction has to be pulled off
the GPU to answer the `if`), so the function issued on the order of 10
separate host↔device round trips per call instead of one; the reviewer
measured this **in isolation** (calling only `validate_world_batch` in a
tight loop, no other queued GPU work) at a 1.48x throughput cost (773.9 vs
1146.5 ticks/s, B=4096, CUDA, GB10). Fixed: every invariant is now computed
as an on-device boolean tensor (no sync), stacked, and reduced with a single
`.any()` that syncs exactly once; only on failure does it fall back to
re-running the checks individually to recover which one failed and raise
its original, specific message (see `world.py`'s `validate_world_batch`
docstring).

Before/after benchmark (`training/scripts/bench_combined_step.py`), CUDA,
`B = 4096`, `trace-graph` fixture, full combined pipeline (world-step +
observe + rate-model + readout), `validate=True` vs `validate=False`, run
against both the pre-fix (sequential-`if`) and post-fix (fused) versions of
`validate_world_batch`:

```
PRE-FIX (sequential `if tensor.any():`), validate_world_batch cost:
  validate=True:  332.97 ticks/s
  validate=False: 383.25 ticks/s
  slowdown from validation: 1.15x

POST-FIX (fused single .any() reduction), validate_world_batch cost:
  validate=True:  329.90 ticks/s
  validate=False: 375.62 ticks/s
  slowdown from validation: 1.14x
```

(A second post-fix run measured 332.67/383.38 ticks/s, 1.15x — consistent
with the pre-fix number.) **The fusion does not measurably change
end-to-end throughput within this pipeline** — both versions cost about
1.14–1.15x. This is not a wasted fix: it still cuts `validate_world_batch`
from ~11 host↔device round trips to 1 (confirmed by code inspection and by
the reviewer's isolated 1.48x measurement, which specifically isolated
those syncs from other GPU work), and it removes the risk noted below of
the two check lists drifting apart. But in the *full* pipeline, `torch`'s
async CUDA queue means a sync mostly waits for already-queued kernels
(model substeps, readout MLP) to finish rather than adding its own
proportional cost — so with substantial other GPU work already queued
every tick, going from ~11 syncs to 1 barely changes wall-clock time here.
The reviewer's larger 1.48x number is real, but it is specific to calling
`validate_world_batch` back-to-back with no other GPU work between calls
(e.g. a validation-heavy inner loop), not to this training pipeline's
per-tick cost. `validate=False` remains available for a rollout loop that
already trusts its state and wants to skip the sync entirely.

## Hardening (this fix)

Two other review findings, both fixed alongside the batching work above:

- **`validate_world_batch`** (`world.py`) ports TS `stepWorld`'s
  `validateStepState` guard (finite-value + clock-consistency checks) as a
  batched, cheap check, run by default at the start of every
  `step_world_batched` call (matching `stepWorld` always calling
  `validateStepState`); pass `validate=False` to skip it in a perf-critical
  rollout loop. Every invariant is computed as an on-device boolean tensor
  and reduced with a single fused `.any()` (one host sync in the common,
  valid case) instead of ~11 sequential `if tensor.any():` syncs; see
  "`validate_world_batch` host-sync fusion" above for the measured cost.
  See `tests/test_world_batch.py`'s
  `test_validate_world_batch_*`/`test_step_world_batched_*` tests.
- **`graph.load_graph_json`** validates the loaded JSON to parity with the
  canonical validators (`docs/graph-format.md`'s "Validation a reader must
  perform", `src/lib/connectome/format.ts`'s `validateGraph`,
  `scripts/data/binfmt.py`'s `validate_graph`): required keys, array lengths
  against `neuronCount`/`edgeCount`, `formatVersion` compatibility, index
  bounds (including channel/population ranges), finite values,
  `contactMagnitudes` finite-and-positive, `presynapticOffsets`' CSR
  invariants (starts at 0, non-decreasing, ends at `edgeCount`), per-row
  strictly-increasing `postsynapticIndices` (duplicate/out-of-order edge
  rejection), `presynapticSigns` exactly ±1, and metadata semantic bounds
  (`timestepSeconds > 0`, `leakRate >= 0`, `rateMin <= rateMax`,
  `inputClampMin <= inputClampMax`, `globalGain >= 0`). Raises
  `InvalidGraphJsonError` (a `ValueError`, mirroring `scripts/data/binfmt.py`'s
  `InvalidGraphError` convention) naming the malformed field, instead of a
  bare `KeyError` at the load site or an opaque shape mismatch several
  frames away inside `model.PreparedGraph.__init__`. This matters once
  WP4/WP5 feed real (less-trusted) MaleCNS exports through this loader — see
  `tests/test_graph_validation.py`.

## WP3 — CEM readout trainer (`flyarena-train`)

Seeded, GPU-batched cross-entropy method (CEM) trainer for the readout MLP
(`.agents/plans/trained-readout/03-cem-training.md`). New modules:
`flyarena_training/cem.py` (the CEM optimizer: elite selection and the
mean/std smoothing update), `flyarena_training/seeds.py` (the seed policy —
training/validation/held-out ranges, per-generation training-seed sampling,
and the held-out guard — imported by both `cem.py` and `rollout.py`, neither
of which imports the other), `flyarena_training/rollout.py` (batched episode
rollout: world + observe + model + readout + decode, `evaluate(theta_batch,
seeds) -> fitness`), and `flyarena_training/cli.py` (the `flyarena-train`
entry point, registered in `pyproject.toml`'s `[project.scripts]`; a thin
orchestrator over `_load_and_validate_bundle`/`_resolve_graph`/
`_build_run_config`/`_capture_env_info`). Tests: `tests/test_cem.py`.

### Usage

```bash
# Trace-graph development (substeps default to TRACE_SUBSTEPS=4):
npm run training:export-arms   # writes training/runs/arms/<sha>/{biological,disconnected}.json
training/scripts/run.sh python -m flyarena_training.cli \
  --arm biological --graph training/runs/arms/<sha>/biological.json \
  --replica-seed 101 --out training/runs/dev-run --generations 5

# Real graph (substeps K is required explicitly; no default):
npm run training:export-arms -- --graph public/data/malecns-arena-v1.bin.gz \
  --rewired public/data/malecns-arena-v1-rewired-seed0.bin.gz
training/scripts/run.sh python -m flyarena_training.cli \
  --arm biological --graph training/runs/arms/<sha>/biological.json \
  --replica-seed 101 --out training/runs/<run-id> --substeps 4
```

`--graph` always takes an `export-arms.ts` bundle (`training/runs/arms/<graph
sha>/<arm>.json`), never a re-derived graph: `graph.load_graph_json` already
reads a bundle correctly with no format-specific branch, because a bundle is
a strict superset of the fields `ConnectomeGraph` needs (it adds `arm`,
`graphId`, `graphSource`, `graphArtifactSha256`, `provenance`, `D`,
`outputNeuronIndices`, and its own self-certifying `sha256` on top of the
graph arrays), and `load_graph_json`'s validator only ever reads its own
known field list, so it silently ignores the extra bundle fields. `cli.py`
reads those extra fields itself (from the raw JSON, once) for `--substeps`
defaulting and for `config.json`/`env.json` provenance.

`--substeps` defaults to `TRACE_SUBSTEPS` (4) only when the loaded bundle's
`graphSource` is `"trace-graph-fixture"`; a bundle with `graphSource:
"artifact"` (a real compiled graph — biological, rewired, or disconnected)
must pass `--substeps` explicitly or the CLI exits non-zero, per the plan's
"the CLI refuses a production graph without an explicit --substeps".

Output (`training/runs/<run-id>/`, gitignored): `config.json` (every
hyperparameter, the seed sets, `D`, `H`, `parameterCount`, `arm`,
`trainerSeed`, `substeps`, `armBundleSha256` — satisfies
`scripts/training/run-dir.ts`'s `RunConfig` contract exactly, plus extra
informational fields `run-dir.ts` ignores), `theta_final.npy`/`theta_best.npy`
(1-D float32, the flat `[w1 (H×D), b1 (H), w2 (3×H), b2 (3)]` layout
`run-dir.ts` documents), `generations.csv` (`generation,meanFitness,
maxFitness,validationFitness` per row), `env.json` (torch/CUDA versions,
device name, git rev, graph bundle sha256, precision flags, wall time).

### CEM defaults and seed policy

Population 256, elites 32, generations 150, smoothing α=0.7 on mean and std,
std floor 0.02, init std 0.5, `E=16` training seeds per generation (`B = P ×
E = 4096`), hidden size `H=16` — all CLI-overridable, defaults matching
03-cem-training.md's table exactly. Training seeds: `E` sampled without
replacement per generation from `seeds.sample_training_seeds`
(`numpy.random.default_rng([trainer_seed, generation])` over `[1, 10000]` —
see "Replica training-seed independence" below for why this deviates from
the plan's literal formula). Validation seeds: the fixed range
`[20001, 20064]`, evaluated on the CEM mean (not the population) every
generation, used only to track the best-ever candidate by validation
fitness — never to select elites. Held-out seeds `[30001, 30100]` are never
sampled by either policy; `seeds.assert_no_held_out_seeds` is called
before every training and validation batch regardless (defense in depth —
see `test_cem_held_out_injection_fires_the_assertion` and
`test_evaluate_fitness_held_out_seed_raises`). The published candidate is
`theta_final` (the final smoothed CEM mean); `theta_best` (the best-ever
candidate by validation mean) is also written but not consumed by the
evaluator.

**α convention (resolved ambiguity):** the plan states only "smoothing α =
0.7 on mean and std" with no formula. This trainer follows the standard
smoothed cross-entropy method convention (De Boer, Kroese, Mannor &
Rubinstein 2005, "A Tutorial on the Cross-Entropy Method"): `α` weights the
*new* elite estimate, `mean = α·new_mean + (1-α)·mean`, so `α=0.7` moves the
search distribution 70% of the way toward each generation's elites. Pinned
by `test_cem_alpha_weights_the_new_elite_estimate`.

**Replica training-seed independence (deliberate deviation from the plan's
literal formula):** 03-cem-training.md's "Seed policy" table specifies
`numpy.random.default_rng(trainer_seed + generation)` (a scalar sum) for
per-generation training-seed sampling. That literal formula aliases whenever
two replicas' `trainer_seed`s differ by less than `G` — at the plan's own
default replica seeds 101/202/303 and `G=150`, replica 101's generation
`g >= 101` drew *exactly* the same 16-seed training set as replica 202's
generation `g - 101` (49/150 = 32.7% of generations for each adjacent pair,
confirmed empirically across three review passes), which directly
contradicts the plan's own "R = 3 independent trainer_seed values per arm"
framing (`.agents/plans/trained-readout/00-overview.md:43`) — two of the
three replica pairs would not be drawing from independent Monte Carlo
training curricula for roughly a third of their generations.

`seeds.sample_training_seeds` therefore keys `numpy.random.default_rng` on
the two-element entropy tuple `[trainer_seed, generation]`
(`numpy.random.SeedSequence` semantics) instead of the literal scalar sum.
This is a deliberate deviation from the plan's literal text, made to honor
the plan's own *stated intent* of three independent replicas, which the
literal formula fails to deliver. It changes no other behavior: same
function signature, same per-generation determinism (the CPU bit-identity
gate — see "Reproducibility" below — still passes with the new formula), and
the same `[1, 10000]` training-seed range disjoint from held-out. The
candidate noise stream was already independent per replica under either
formula (`torch.Generator` seeded from `trainer_seed` alone, with no
`+ generation` term). `test_sample_training_seeds_is_replica_independent_across_defaults`
(`tests/test_cem.py`) asserts no generation of any of the three default
replicas ever draws the same 16-seed training set as any generation of
either of the other two, and that held-out isolation still holds; it also
sanity-checks that the old formula would still show the reviewed 49/150
collision count, so the test would fail if this fix were ever reverted.
`config.json`'s `trainingSeedRng` field records the formula in effect
(`"default_rng([trainerSeed, generation])"`) for any run this trainer
writes.

### Reproducibility (measured)

- **CPU bit-identity (blocking gate, measured):** two CPU runs with the same
  `trainer_seed`, graph bundle, and config produce byte-identical
  `theta_final.npy` — `test_cpu_theta_final_bit_identical_across_two_runs`.
  Verified true on this host.
- **GPU rerun (informational, measured, not gated):** two CUDA runs with the
  same config; `test_gpu_theta_final_rerun_diff_is_informational` measures
  and prints `theta_final`'s max-abs diff (`gpuRerunMaxAbsDiff`) without
  asserting a bound. On this host, at the test's tiny scale (P=16, G=3,
  T=20 ticks, trace graph), the measured diff was `0.0` — CUDA's sparse
  kernels happened to be deterministic at this scale on this run; this is
  not a guarantee at production scale (P=256, G=150, T=1800), where
  `torch.use_deterministic_algorithms(True, warn_only=True)` still allows a
  nondeterministic kernel to fall back with only a warning (see
  `flyarena_training/__init__.py`). WP5 records the real production-scale
  `gpuRerunMaxAbsDiff` in the manifest.

### Calibration (informational; no gate; no full production run performed)

Measured on this host (GB10), CUDA, at the plan's defaults (`P=256, E=16,
T=1800` -> `B=4096`), `H=16`, 2 generations:

| Graph | D (output neurons) | s/generation |
| --- | --- | --- |
| Trace graph (`tests/fixtures/golden/trace-graph.json`) | 6 | ~8.45 |
| Real MaleCNS (`public/data/malecns-arena-v1.bin.gz`, `--substeps 4`) | 48 | ~8.42 |

The two are within measurement noise of each other: at this batch size the
per-tick Python/CUDA-launch overhead of the 1800-tick rollout loop dominates
wall time, not the sparse operator's size (6 vs. 48 output neurons, and the
underlying neuron/edge counts, are both tiny relative to `B=4096`). Projected
full WP5 cost at `G=150`: `9 runs (3 arms × 3 replicas) × 150 generations ×
~8.4 s/generation ≈ 3.2 hours` — comfortably under the plan's 12-hour budget
(`05-production-run.md`), so no `P` reduction is anticipated to be necessary,
though WP5 should still re-calibrate on the merged, real closed-loop `K`.
These are single-arm, few-generation measurements only; no full production
run (`G=150`, all 9 arm×replica combinations) was performed in this work
package, per its exclusions.

### End-to-end contract test

`test_end_to_end_contract_with_ts_evaluator` (`tests/test_cem.py`) trains a
tiny trace-graph run with the Python CLI, then runs the real
`scripts/training/evaluate.ts` (via `npm run training:evaluate`) on it into a
scratch `--out` directory (never `public/data`/`docs/`) and asserts it loads,
validates (including the arm bundle's self-certifying sha256 check inside
`evaluate.ts`'s `loadArmGraphs`), and scores the run — confirming
`training/`'s output run directory is byte-for-byte consumable by the
TypeScript evaluator with zero TS changes. Skips (naming the manual command)
when Node isn't reachable, matching `conftest.py`'s existing convention.
Verified passing on this host.

### Deviations from the WP3 plan

- **Calibration test ticks:** `test_cli_trace_graph_run_writes_all_files` and
  the CPU-bit-identity test use `--ticks 20`, not the plan's `T=1800`
  episode length. The plan's acceptance bullet for this test pins `G=3,
  P=16, E=2` but not `T`; a full 1800-tick episode at these tiny `P`/`E`
  would cost more wall time in CI with no additional coverage of "does the
  CLI write every file correctly". `T=1800` is exercised by the calibration
  runs above (this file) and is the CLI's own default.
- **CLI-only tests use a hand-built synthetic bundle**, not a real
  `export-arms.ts` bundle: `cli.py` never verifies a bundle's own
  self-certifying `sha256` (only `evaluate.ts` does), so a placeholder
  `sha256`/`graphArtifactSha256` is sufficient for exercising the CLI's file
  writing in isolation. The end-to-end contract test uses a real
  `export-arms.ts`-produced bundle instead, since `evaluate.ts` does verify
  it.
- **`elite_candidates.std(dim=0, unbiased=False)`** (population, not sample,
  standard deviation) for the CEM std update: the plan does not specify
  biased vs. unbiased; population std was chosen since `N_e=32` is the
  entire elite population being summarized, not a sample of a larger one.
- **Training seeds are sampled without replacement**
  (`numpy.random.default_rng(...).choice(..., replace=False)`): the plan
  specifies the RNG but not replacement; sampling without replacement avoids
  wasting part of the `E`-seed batch on a duplicate episode.
- **Training-seed RNG is keyed on `[trainer_seed, generation]`, not the
  plan's literal `trainer_seed + generation`:** see "Replica training-seed
  independence" above — the literal scalar-sum formula aliases between
  replicas and contradicts the plan's own "3 independent replicas" intent,
  so this trainer honors that intent instead of the literal formula.
- **`--out` write ordering, not full atomic publish:** `cli.py` creates
  `--out` before training (fails fast on a bad path), immediately removes
  any `config.json` already there (so a *reused* `--out` can't leave an old
  run's `config.json` paired with this run's new weights if training
  crashes), and writes this run's own `config.json` last, after
  `theta_final.npy`/`theta_best.npy`/`generations.csv`/`env.json` all
  succeed — `run-dir.ts` reads `config.json` first, so an interrupted write
  is a clean "file not found" rather than a *complete-looking* but
  mismatched run directory. This is not a full atomic-rename publish (a
  `.partial` staging directory that only replaces `--out` on total success):
  WP3's manual training procedure runs one `flyarena-train` invocation at a
  time, so the remaining residual risk (a genuinely concurrent writer to the
  same `--out` from two processes at once) is out of scope for this pass.
- **Bundle validation requires every `export-arms.ts` `SerializedArmBundle`
  field this CLI reads** (`formatVersion`, `arm`, `graphId`, `graphSource`,
  `graphArtifactSha256`, `sha256`, `D`, `outputNeuronIndices`), and
  cross-checks the bundle's declared `D`/`outputNeuronIndices` against what
  `output_neuron_indices` computes from the loaded graph arrays. A bundle
  missing any of these, or one whose `D`/`outputNeuronIndices` disagree with
  its own graph arrays, is refused before training starts, rather than
  trained against silently and only rejected later by `evaluate.ts`.
  `config.json`'s `armBundleSha256` is always the bundle's real `sha256`
  (never silently dropped for being absent): `evaluate.ts`'s `loadArmGraphs`
  cross-checks that field against the bundle it loads as its "trained
  against the right bundle" integrity check, so a missing or wrong value
  there would silently disable that check.

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
