"""Batched port of `readoutForward` (`src/lib/connectome/readout.ts`).

Weights are batched over a population of `P` candidates (e.g. WP3's CEM
population), each with its own `D -> H -> 3` MLP. `readout_forward` requires
`rate.shape[0] == P`: one gathered rate row per candidate's own rollout.
`load_readout_weights_json`/`broadcast_readout_weights` cover the other
useful case — one fixed weight set (as committed in the golden readout
fixture) applied to a batch of many rate rows (e.g. every tick of a
teacher-forced seed).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .graph import ConnectomeGraph, output_neuron_indices

# Canonical output order (`OUTPUT_POPULATION`, `src/lib/arena/actions.ts`):
# thrust, yaw, brake.
THRUST_INDEX, YAW_INDEX, BRAKE_INDEX = 0, 1, 2


@dataclass
class ReadoutWeights:
    """`w1`: `[P, H, D]`; `b1`: `[P, H]`; `w2`: `[P, 3, H]`; `b2`: `[P, 3]`."""

    input_size: int
    hidden_size: int
    w1: torch.Tensor
    b1: torch.Tensor
    w2: torch.Tensor
    b2: torch.Tensor


def gather_output_rates(
    graph: ConnectomeGraph, rate: torch.Tensor, indices: torch.Tensor | None = None
) -> torch.Tensor:
    """`rate`: `[B, neuronCount]` -> `[B, D]`, gathered at `outputNeuronIndices(graph)`."""
    if indices is None:
        indices = output_neuron_indices(graph)
    return rate.index_select(1, indices)


def readout_forward(weights: ReadoutWeights, rate: torch.Tensor) -> torch.Tensor:
    """`rate`: `[P, inputSize]`, one already-gathered row per candidate.
    Returns `[P, 3]` in `OUTPUT_POPULATION` order: `tanh(hidden)` for
    thrust/yaw, `sigmoid` for brake, exactly matching `readoutForward`."""
    hidden_pre = torch.einsum("phd,pd->ph", weights.w1, rate) + weights.b1
    hidden = torch.tanh(hidden_pre)
    out_pre = torch.einsum("poh,ph->po", weights.w2, hidden) + weights.b2
    thrust = torch.tanh(out_pre[:, THRUST_INDEX])
    yaw = torch.tanh(out_pre[:, YAW_INDEX])
    brake = torch.sigmoid(out_pre[:, BRAKE_INDEX])
    return torch.stack([thrust, yaw, brake], dim=1)


def load_readout_weights_json(path: str | Path, device: str | torch.device = "cpu") -> ReadoutWeights:
    """Loads a single (unbatched, `P = 1`) `ReadoutWeights` JSON, matching
    `serializeReadoutWeights` in `scripts/training/export-traces.ts` — either
    a readout-case file directly (`{"weights": {...}, ...}`) or a bare
    weights object. Use `broadcast_readout_weights` to apply it to `B > 1`
    rate rows."""
    data = json.loads(Path(path).read_text())
    weights = data["weights"] if "weights" in data else data
    input_size = weights["inputSize"]
    hidden_size = weights["hiddenSize"]
    w1 = torch.tensor(weights["w1"], dtype=torch.float32, device=device).reshape(1, hidden_size, input_size)
    b1 = torch.tensor(weights["b1"], dtype=torch.float32, device=device).reshape(1, hidden_size)
    w2 = torch.tensor(weights["w2"], dtype=torch.float32, device=device).reshape(1, 3, hidden_size)
    b2 = torch.tensor(weights["b2"], dtype=torch.float32, device=device).reshape(1, 3)
    return ReadoutWeights(input_size=input_size, hidden_size=hidden_size, w1=w1, b1=b1, w2=w2, b2=b2)


def broadcast_readout_weights(weights: ReadoutWeights, batch_size: int) -> ReadoutWeights:
    """Expands a `P = 1` weight set to `P = batch_size` (a view, no copy)."""
    return ReadoutWeights(
        input_size=weights.input_size,
        hidden_size=weights.hidden_size,
        w1=weights.w1.expand(batch_size, -1, -1),
        b1=weights.b1.expand(batch_size, -1),
        w2=weights.w2.expand(batch_size, -1, -1),
        b2=weights.b2.expand(batch_size, -1),
    )


def readout_parameter_count(input_size: int, hidden_size: int) -> int:
    """Port of `readoutParameterCount`: total trainable scalars for one
    `D -> H -> 3` readout (`w1 + b1 + w2 + b2`)."""
    return hidden_size * input_size + hidden_size + 3 * hidden_size + 3
