"""Batched port of `src/lib/connectome/model.ts`'s rate model.

All neural state is float32, matching the TS `Float32Array` rate/drive
buffers. `stepModel` in TS re-rounds to float32 on *every* accumulation into
`drive` (a `Float32Array`), not just once at the end — a GPU sparse matmul
necessarily accumulates each output element's edges in one shot (in
whatever internal order/precision the kernel picks), so it cannot reproduce
that per-edge rounding bit-for-bit. This is a known, accepted discrepancy
(see `.agents/plans/trained-readout/00-overview.md`'s "Repository findings"),
which is exactly why the rate/output parity tolerance is
`abs <= 1e-5 or rel <= 1e-4` rather than exact.

Requires the TF32-disabling precision settings this package's `__init__`
applies at import time (see `.agents/plans/trained-readout/02-gpu-port-and-parity.md`'s
"Precision settings"): PyTorch enables TF32 for float32 matmul by default on
Ampere+ GPUs (including the GB10), which would silently blow the tolerance
above.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .graph import ConnectomeGraph


@dataclass
class ModelState:
    """`rate`: `[B, neuronCount]` float32, the bounded rate of every neuron
    for every batch item."""

    rate: torch.Tensor


def create_model_state(
    graph: ConnectomeGraph, batch_size: int, device: str | torch.device, dtype: torch.dtype = torch.float32
) -> ModelState:
    return ModelState(rate=torch.zeros(batch_size, graph.metadata.neuron_count, dtype=dtype, device=device))


def reset_model_state(state: ModelState) -> None:
    state.rate.zero_()


class PreparedGraph:
    """Precomputed sparse operator and dense per-neuron arrays for
    `step_model`, built once per `(graph, device)` and reused across every
    subsequent step (matching `model.ts`'s allocation-free-per-step
    convention: `createStepScratch`/`createOutputBuffer` there, this class
    here).

    `operator` is `[N, N]` CSR sparse with `operator[post, pre] =
    contactMagnitude(pre -> post)`, so `drive = operator @ signed_rate`
    (`signed_rate` as `[N, B]` columns) computes, for every post-synaptic
    neuron, the sum over its presynaptic edges of `sign * gain * rate *
    magnitude` — exactly `stepModel`'s recurrent-drive scatter loop,
    reassociated as a sparse matmul.
    """

    def __init__(self, graph: ConnectomeGraph, device: str | torch.device):
        device = torch.device(device)
        self.graph = graph
        self.device = device
        n = graph.metadata.neuron_count

        offsets = graph.presynaptic_offsets.to(device)
        counts = offsets[1:] - offsets[:-1]
        pre_index = torch.repeat_interleave(torch.arange(n, device=device, dtype=torch.int64), counts)
        post_index = graph.postsynaptic_indices.to(device)
        magnitudes = graph.contact_magnitudes.to(device)

        indices = torch.stack([post_index, pre_index], dim=0)
        coo = torch.sparse_coo_tensor(indices, magnitudes, size=(n, n), device=device)
        self.operator = coo.coalesce().to_sparse_csr()

        self.presynaptic_signs = graph.presynaptic_signs.to(device)  # [N], values in {-1, +1}
        self.global_gain = float(graph.metadata.global_gain)
        self.timestep_seconds = float(graph.metadata.timestep_seconds)
        self.leak_rate = float(graph.metadata.leak_rate)
        self.rate_min = float(graph.metadata.rate_min)
        self.rate_max = float(graph.metadata.rate_max)
        self.input_clamp_min = float(graph.metadata.input_clamp_min)
        self.input_clamp_max = float(graph.metadata.input_clamp_max)

        input_channel_index = graph.input_channel_index.to(device)
        self.has_input = input_channel_index >= 0  # [N] bool
        self.input_channel_index = input_channel_index.clamp(min=0)
        self.input_weight = graph.input_weight.to(device)
        # `channel_values` has width `inputChannelCount`; if that's 0 (no
        # input channels declared at all), `index_select` below would index
        # a width-0 dim even though `has_input` is all-False and would mask
        # the result to zero anyway. `stepModel` (model.ts:91-96) has no such
        # failure mode since it just `continue`s past every non-input
        # neuron, so this port skips the gather entirely in that case too.
        self.input_channel_count = graph.metadata.input_channel_count

        self.output_population_index = graph.output_population_index.to(device)
        self.output_weight = graph.output_weight.to(device)
        self.output_population_count = graph.metadata.output_population_count


def step_model(prepared: PreparedGraph, state: ModelState, channel_values: torch.Tensor) -> None:
    """One substep of `graph.metadata.timestepSeconds`. `channel_values`:
    `[B, inputChannelCount]` float32, read-only, indexed by
    `inputChannelIndex` (port of `stepModel`)."""
    rate = state.rate  # [B, N]
    signed_rate = rate * prepared.presynaptic_signs.unsqueeze(0) * prepared.global_gain  # [B, N]
    drive = torch.sparse.mm(prepared.operator, signed_rate.t().contiguous()).t()  # [B, N]

    if prepared.input_channel_count > 0:
        gathered = channel_values.index_select(1, prepared.input_channel_index)  # [B, N]
        clamped = gathered.clamp(prepared.input_clamp_min, prepared.input_clamp_max)
        external = clamped * prepared.input_weight.unsqueeze(0)
        external = torch.where(prepared.has_input.unsqueeze(0), external, torch.zeros_like(external))
        drive = drive + external

    next_rate = rate + prepared.timestep_seconds * (-prepared.leak_rate * rate + drive)
    state.rate = next_rate.clamp(prepared.rate_min, prepared.rate_max)


def aggregate_outputs(prepared: PreparedGraph, state: ModelState) -> torch.Tensor:
    """`[B, outputPopulationCount]` float32 (port of `aggregateOutputs`)."""
    batch_size = state.rate.shape[0]
    weighted = state.rate * prepared.output_weight.unsqueeze(0)  # [B, N]
    valid = prepared.output_population_index >= 0
    safe_index = prepared.output_population_index.clamp(min=0)
    masked = torch.where(valid.unsqueeze(0), weighted, torch.zeros_like(weighted))
    outputs = torch.zeros(
        batch_size, prepared.output_population_count, dtype=state.rate.dtype, device=state.rate.device
    )
    outputs.scatter_add_(1, safe_index.unsqueeze(0).expand(batch_size, -1), masked)
    return outputs


def run_substeps(
    prepared: PreparedGraph, state: ModelState, channel_values: torch.Tensor, substeps: int
) -> torch.Tensor:
    """Port of `runSubsteps`: `substeps` consecutive `step_model` calls with
    the same held-constant `channel_values`, then aggregate."""
    for _ in range(substeps):
        step_model(prepared, state, channel_values)
    return aggregate_outputs(prepared, state)
