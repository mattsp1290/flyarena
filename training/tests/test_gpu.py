"""GPU-only smoke test: skipped unless CUDA is available. Runs the batched
sparse rate-model step at B = 4096 on the trace graph and reports
throughput (informational; no gate), per
`.agents/plans/trained-readout/02-gpu-port-and-parity.md`'s "Tests and
acceptance" ("Throughput note recorded in training/README.md: steps/second
at B = 4096 on the trace graph")."""
from __future__ import annotations

import time

import pytest
import torch

from flyarena_training.graph import load_graph_json
from flyarena_training.model import PreparedGraph, create_model_state, run_substeps

GRAPH_ID = "trace-graph"
BATCH_SIZE = 4096
SUBSTEPS = 4
MEASURED_TICKS = 200


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_batched_step_b4096_on_cuda(trace_dir):
    device = torch.device("cuda")
    graph = load_graph_json(trace_dir / f"{GRAPH_ID}.json", device=device)
    prepared = PreparedGraph(graph, device)
    state = create_model_state(graph, batch_size=BATCH_SIZE, device=device)
    observation = torch.rand(
        BATCH_SIZE, graph.metadata.input_channel_count, device=device, dtype=torch.float32
    )

    # Warm-up: pays first-call CUDA kernel compile/launch overhead outside
    # the timed region.
    run_substeps(prepared, state, observation, substeps=SUBSTEPS)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(MEASURED_TICKS):
        run_substeps(prepared, state, observation, substeps=SUBSTEPS)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ticks_per_second = MEASURED_TICKS / elapsed
    substeps_per_second = ticks_per_second * SUBSTEPS
    print(
        f"[gpu] batched step at B={BATCH_SIZE}: {ticks_per_second:.1f} ticks/s "
        f"({substeps_per_second:.1f} substeps/s, {elapsed / MEASURED_TICKS * 1000:.3f} ms/tick, "
        f"{MEASURED_TICKS} ticks measured, substeps={SUBSTEPS})"
    )

    assert torch.isfinite(state.rate).all()
