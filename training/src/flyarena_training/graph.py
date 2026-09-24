"""Load a connectome graph exported as JSON.

The schema matches `serializeGraph` in `scripts/training/export-traces.ts`
(the committed `trace-graph.json` golden fixture), which the plan
(`.agents/plans/trained-readout/02-gpu-port-and-parity.md`) says WP4's
`export-arms.ts` CSR bundle shares, so this loader also covers the real
MaleCNS arm export once WP4/WP5 land.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class GraphMetadata:
    format_version: int
    neuron_count: int
    edge_count: int
    input_channel_count: int
    output_population_count: int
    timestep_seconds: float
    leak_rate: float
    rate_min: float
    rate_max: float
    input_clamp_min: float
    input_clamp_max: float
    global_gain: float


@dataclass
class ConnectomeGraph:
    metadata: GraphMetadata
    biological_ids: list[int]
    presynaptic_offsets: torch.Tensor  # int64 [N + 1]
    postsynaptic_indices: torch.Tensor  # int64 [E]
    contact_magnitudes: torch.Tensor  # float32 [E]
    presynaptic_signs: torch.Tensor  # float32 [N], values in {-1, +1}
    input_channel_index: torch.Tensor  # int64 [N], -1 if not an input neuron
    input_weight: torch.Tensor  # float32 [N]
    output_population_index: torch.Tensor  # int64 [N], -1 if not an output neuron
    output_weight: torch.Tensor  # float32 [N]


def load_graph_json(path: str | Path, device: str | torch.device = "cpu") -> ConnectomeGraph:
    data = json.loads(Path(path).read_text())
    metadata_raw = data["metadata"]
    metadata = GraphMetadata(
        format_version=metadata_raw["formatVersion"],
        neuron_count=metadata_raw["neuronCount"],
        edge_count=metadata_raw["edgeCount"],
        input_channel_count=metadata_raw["inputChannelCount"],
        output_population_count=metadata_raw["outputPopulationCount"],
        timestep_seconds=metadata_raw["timestepSeconds"],
        leak_rate=metadata_raw["leakRate"],
        rate_min=metadata_raw["rateMin"],
        rate_max=metadata_raw["rateMax"],
        input_clamp_min=metadata_raw["inputClampMin"],
        input_clamp_max=metadata_raw["inputClampMax"],
        global_gain=metadata_raw["globalGain"],
    )
    return ConnectomeGraph(
        metadata=metadata,
        biological_ids=[int(x) for x in data["biologicalIds"]],
        presynaptic_offsets=torch.tensor(data["presynapticOffsets"], dtype=torch.int64, device=device),
        postsynaptic_indices=torch.tensor(data["postsynapticIndices"], dtype=torch.int64, device=device),
        contact_magnitudes=torch.tensor(data["contactMagnitudes"], dtype=torch.float32, device=device),
        presynaptic_signs=torch.tensor(data["presynapticSigns"], dtype=torch.float32, device=device),
        input_channel_index=torch.tensor(data["inputChannelIndex"], dtype=torch.int64, device=device),
        input_weight=torch.tensor(data["inputWeight"], dtype=torch.float32, device=device),
        output_population_index=torch.tensor(data["outputPopulationIndex"], dtype=torch.int64, device=device),
        output_weight=torch.tensor(data["outputWeight"], dtype=torch.float32, device=device),
    )


def output_neuron_indices(graph: ConnectomeGraph) -> torch.Tensor:
    """Ascending neuron indices where `outputPopulationIndex[i] >= 0`,
    length `D` (port of `outputNeuronIndices`, `src/lib/connectome/readout.ts`)."""
    mask = graph.output_population_index >= 0
    return torch.nonzero(mask, as_tuple=False).flatten()
