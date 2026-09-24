"""Before/after benchmark for the CRITICAL batching fix (thermo-architecture
review finding #1): combined world-step + observe + rate-model + readout
throughput at B = 4096 on CUDA, on the `trace-graph` fixture.

"Before" reproduces the pre-fix architecture's actual mixed-device pipeline
cost: a per-item Python loop for `step_world`/`observe_agent` (still
available as the test oracle at `training/tests/reference_world.py`,
`training/tests/reference_world.py`'s per-item `observe_agent_reference`)
combined with the *same* GPU-batched model+readout step the "after" run
uses — this isolates exactly what changed (the world/observe path) rather
than re-measuring the model, which was already batched before this fix.

"After" is the all-batched pipeline this fix introduces:
`step_world_batched` + `observe_batch` (dense tensor ops) + the same
model+readout step.

Also reports the isolated cost of `validate_world_batch` (run by default
inside `step_world_batched`, matching TS `stepWorld` always calling
`validateStepState`): the same "after" pipeline with `validate=True` vs
`validate=False`, to measure the CUDA host-sync overhead of the invariant
checks (thermo-fix-verification review finding: fused from ~11 sequential
per-check syncs into one).

Usage: `uv run python scripts/bench_combined_step.py` (from `training/`).
Requires CUDA (skips with a message otherwise, matching `test_gpu.py`'s
convention).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from flyarena_training.actions import decode_action_batch  # noqa: E402
from flyarena_training.config import ARENA_CONFIG  # noqa: E402
from flyarena_training.graph import load_graph_json, output_neuron_indices  # noqa: E402
from flyarena_training.model import PreparedGraph, create_model_state, run_substeps  # noqa: E402
from flyarena_training.readout import ReadoutWeights, gather_output_rates, readout_forward  # noqa: E402
from flyarena_training.sensors import observe_batch  # noqa: E402
from flyarena_training.world import create_world, create_world_batch, step_world_batched, world_batch_from_items  # noqa: E402

from reference_world import observe_agent_reference, step_world_item_reference  # noqa: E402

BATCH_SIZE = 4096
SUBSTEPS = 4
MEASURED_TICKS = 200
WARMUP_TICKS = 10
# training/scripts/ -> training/ -> repo root -> tests/fixtures/golden/
GRAPH_PATH = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "golden" / "trace-graph.json"


def _random_readout_weights(input_size: int, hidden_size: int, batch_size: int, device: torch.device) -> ReadoutWeights:
    generator = torch.Generator(device="cpu").manual_seed(0)
    return ReadoutWeights(
        input_size=input_size,
        hidden_size=hidden_size,
        w1=torch.randn(batch_size, hidden_size, input_size, generator=generator).to(device),
        b1=torch.randn(batch_size, hidden_size, generator=generator).to(device),
        w2=torch.randn(batch_size, 3, hidden_size, generator=generator).to(device),
        b2=torch.randn(batch_size, 3, generator=generator).to(device),
    )


def _bench_after(device: torch.device, validate: bool = True) -> float:
    graph = load_graph_json(GRAPH_PATH, device=device)
    prepared = PreparedGraph(graph, device)
    indices = output_neuron_indices(graph)
    weights = _random_readout_weights(indices.numel(), hidden_size=8, batch_size=BATCH_SIZE, device=device)

    state = create_world_batch(list(range(1, BATCH_SIZE + 1)), device=device)
    model_state = create_model_state(graph, batch_size=BATCH_SIZE, device=device)

    def one_tick(state):
        observation = observe_batch(state, "left", ARENA_CONFIG).to(torch.float32)
        run_substeps(prepared, model_state, observation, SUBSTEPS)
        gathered = gather_output_rates(graph, model_state.rate, indices)
        readout_out = readout_forward(weights, gathered)
        left_action = decode_action_batch(readout_out.double(), BATCH_SIZE, device)
        right_action = torch.zeros_like(left_action)
        return step_world_batched(state, {"left": left_action, "right": right_action}, ARENA_CONFIG, validate=validate)

    for _ in range(WARMUP_TICKS):
        state = one_tick(state)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(MEASURED_TICKS):
        state = one_tick(state)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return MEASURED_TICKS / elapsed


def _bench_before(device: torch.device, measured_ticks: int) -> float:
    graph = load_graph_json(GRAPH_PATH, device=device)
    prepared = PreparedGraph(graph, device)
    indices = output_neuron_indices(graph)
    weights = _random_readout_weights(indices.numel(), hidden_size=8, batch_size=BATCH_SIZE, device=device)

    items = create_world(list(range(1, BATCH_SIZE + 1)))
    model_state = create_model_state(graph, batch_size=BATCH_SIZE, device=device)

    def one_tick(items):
        observations = [observe_agent_reference(item, "left", ARENA_CONFIG) for item in items]
        observation = torch.tensor(observations, dtype=torch.float32, device=device)
        run_substeps(prepared, model_state, observation, SUBSTEPS)
        gathered = gather_output_rates(graph, model_state.rate, indices)
        readout_out = readout_forward(weights, gathered)
        left_actions = readout_out.double().cpu().tolist()
        return [
            step_world_item_reference(item, {"left": left_actions[i], "right": [0.0, 0.0, 0.0]})
            for i, item in enumerate(items)
        ]

    for _ in range(min(WARMUP_TICKS, 3)):
        items = one_tick(items)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(measured_ticks):
        items = one_tick(items)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return measured_ticks / elapsed


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available; this benchmark is CUDA-only (matches test_gpu.py's convention). Skipping.")
        return
    device = torch.device("cuda")

    # The "before" (per-item Python) path is ~200x slower per the
    # thermo-architecture review's own measurement; a handful of ticks is
    # enough for a stable per-tick estimate without a multi-minute run.
    before_ticks_per_s = _bench_before(device, measured_ticks=5)
    after_ticks_per_s = _bench_after(device)

    before_item_ticks = before_ticks_per_s * BATCH_SIZE
    after_item_ticks = after_ticks_per_s * BATCH_SIZE

    print(f"BEFORE (per-item world/observe + batched model/readout), B={BATCH_SIZE}:")
    print(f"  {before_ticks_per_s:.2f} ticks/s, {before_item_ticks:.0f} item-ticks/s")
    print(f"AFTER (fully batched world/observe/model/readout), B={BATCH_SIZE}:")
    print(f"  {after_ticks_per_s:.2f} ticks/s, {after_item_ticks:.0f} item-ticks/s")
    print(f"Speedup: {after_item_ticks / before_item_ticks:.1f}x")

    # `validate_world_batch` cost, in isolation: same fully-batched pipeline,
    # `step_world_batched`'s default `validate=True` vs the `validate=False`
    # escape hatch (thermo-fix-verification review finding: ~11 sequential
    # `if tensor.any():` host syncs fused into one `.any()` reduction).
    validate_true_ticks_per_s = _bench_after(device, validate=True)
    validate_false_ticks_per_s = _bench_after(device, validate=False)
    print(f"\nvalidate_world_batch cost (fully batched pipeline), B={BATCH_SIZE}:")
    print(f"  validate=True:  {validate_true_ticks_per_s:.2f} ticks/s")
    print(f"  validate=False: {validate_false_ticks_per_s:.2f} ticks/s")
    print(f"  slowdown from validation: {validate_false_ticks_per_s / validate_true_ticks_per_s:.2f}x")

    # WP3 wall-time re-estimate at the plan's declared defaults.
    p, e, g, t, runs = 256, 16, 150, 1800, 9
    b = p * e
    for label, item_ticks_per_s in (("BEFORE", before_item_ticks), ("AFTER", after_item_ticks)):
        per_generation_s = (b * t) / item_ticks_per_s
        per_run_s = per_generation_s * g
        total_s = per_run_s * runs
        print(
            f"{label}: WP3 estimate at P={p}, E={e}, G={g}, T={t}, {runs} runs: "
            f"{per_generation_s:.2f} s/generation, {per_run_s / 3600:.2f} h/run, {total_s / 3600:.2f} h total"
        )


if __name__ == "__main__":
    main()
