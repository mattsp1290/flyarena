"""Load a connectome graph exported as JSON.

The schema matches `serializeGraph` in `scripts/training/export-traces.ts`
(the committed `trace-graph.json` golden fixture), which the plan
(`.agents/plans/trained-readout/02-gpu-port-and-parity.md`) says WP4's
`export-arms.ts` CSR bundle shares, so this loader also covers the real
MaleCNS arm export once WP4/WP5 land — which is exactly why its input is
validated rather than trusted: WP4/WP5 exports are not the hand-controlled
golden fixture this loader was first tested against.

`load_graph_json` mirrors `scripts/data/binfmt.py`'s `InvalidGraphError`
convention (a `ValueError` subclass, descriptive per-field messages) rather
than letting a malformed file surface as a bare `KeyError` at the load site
or an opaque shape-mismatch failure several frames away inside
`model.PreparedGraph.__init__`.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch


class InvalidGraphJsonError(ValueError):
    """Raised by `load_graph_json` for a malformed graph JSON file, mirroring
    `scripts/data/binfmt.py`'s `InvalidGraphError` convention."""


def _invalid(path: str | Path, message: str) -> None:
    raise InvalidGraphJsonError(f"Invalid connectome graph JSON ({path}): {message}")


def _require_key(data: Mapping, key: str, path: str | Path, context: str) -> object:
    if key not in data:
        _invalid(path, f"{context} is missing required field {key!r}")
    return data[key]


def _require_list(data: Mapping, key: str, path: str | Path, context: str) -> Sequence:
    value = _require_key(data, key, path, context)
    if not isinstance(value, list):
        _invalid(path, f"{context} field {key!r} must be a JSON array, got {type(value).__name__}")
    return value  # type: ignore[return-value]


def _require_len(value: Sequence, expected: int, path: str | Path, field_name: str) -> None:
    if len(value) != expected:
        _invalid(path, f"field {field_name!r} has length {len(value)}, expected {expected}")


def _require_finite(value: Sequence[float], path: str | Path, field_name: str) -> None:
    for index, item in enumerate(value):
        if not isinstance(item, (int, float)) or not math.isfinite(item):
            _invalid(path, f"field {field_name!r}[{index}] = {item!r} is not a finite number")


def _require_index_range(value: Sequence[int], count: int, path: str | Path, field_name: str, allow_negative_one: bool) -> None:
    minimum = -1 if allow_negative_one else 0
    for index, item in enumerate(value):
        if not isinstance(item, int) or item < minimum or item >= count:
            _invalid(
                path,
                f"field {field_name!r}[{index}] = {item!r} is out of range "
                f"[{minimum}, {count}) for a graph with {count} neurons",
            )


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


_METADATA_FIELDS: tuple[str, ...] = (
    "formatVersion",
    "neuronCount",
    "edgeCount",
    "inputChannelCount",
    "outputPopulationCount",
    "timestepSeconds",
    "leakRate",
    "rateMin",
    "rateMax",
    "inputClampMin",
    "inputClampMax",
    "globalGain",
)

# Array fields whose length must equal `neuronCount`.
_PER_NEURON_FIELDS: tuple[str, ...] = (
    "biologicalIds",
    "presynapticSigns",
    "inputChannelIndex",
    "inputWeight",
    "outputPopulationIndex",
    "outputWeight",
)

# Array fields whose length must equal `edgeCount`.
_PER_EDGE_FIELDS: tuple[str, ...] = ("postsynapticIndices", "contactMagnitudes")


def _validate_graph_json(data: Mapping, path: str | Path) -> Mapping:
    if not isinstance(data, dict):
        _invalid(path, f"top-level JSON value must be an object, got {type(data).__name__}")

    metadata_raw = _require_key(data, "metadata", path, "graph")
    if not isinstance(metadata_raw, dict):
        _invalid(path, f"'metadata' must be an object, got {type(metadata_raw).__name__}")
    for field_name in _METADATA_FIELDS:
        value = _require_key(metadata_raw, field_name, path, "metadata")
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            _invalid(path, f"metadata field {field_name!r} = {value!r} is not a finite number")

    neuron_count = metadata_raw["neuronCount"]
    edge_count = metadata_raw["edgeCount"]
    input_channel_count = metadata_raw["inputChannelCount"]
    output_population_count = metadata_raw["outputPopulationCount"]
    for count_name, count_value in (
        ("neuronCount", neuron_count),
        ("edgeCount", edge_count),
        ("inputChannelCount", input_channel_count),
        ("outputPopulationCount", output_population_count),
    ):
        if not isinstance(count_value, int) or count_value < 0:
            _invalid(path, f"metadata field {count_name!r} = {count_value!r} must be a non-negative integer")

    for field_name in _PER_NEURON_FIELDS:
        values = _require_list(data, field_name, path, "graph")
        _require_len(values, neuron_count, path, field_name)
    for field_name in _PER_EDGE_FIELDS:
        values = _require_list(data, field_name, path, "graph")
        _require_len(values, edge_count, path, field_name)
    presynaptic_offsets = _require_list(data, "presynapticOffsets", path, "graph")
    _require_len(presynaptic_offsets, neuron_count + 1, path, "presynapticOffsets")

    _require_finite(data["contactMagnitudes"], path, "contactMagnitudes")
    _require_finite(data["inputWeight"], path, "inputWeight")
    _require_finite(data["outputWeight"], path, "outputWeight")

    for index, value in enumerate(presynaptic_offsets):
        if not isinstance(value, int) or value < 0 or value > edge_count:
            _invalid(path, f"field 'presynapticOffsets'[{index}] = {value!r} is out of range [0, {edge_count}]")
    for index in range(len(presynaptic_offsets) - 1):
        if presynaptic_offsets[index] > presynaptic_offsets[index + 1]:
            _invalid(
                path,
                f"field 'presynapticOffsets' is not non-decreasing at index {index}: "
                f"{presynaptic_offsets[index]} > {presynaptic_offsets[index + 1]}",
            )
    if presynaptic_offsets[0] != 0 or presynaptic_offsets[-1] != edge_count:
        _invalid(
            path,
            "field 'presynapticOffsets' must start at 0 and end at edgeCount "
            f"({edge_count}); got [{presynaptic_offsets[0]}, ..., {presynaptic_offsets[-1]}]",
        )

    _require_index_range(data["postsynapticIndices"], neuron_count, path, "postsynapticIndices", allow_negative_one=False)
    _require_index_range(data["inputChannelIndex"], input_channel_count, path, "inputChannelIndex", allow_negative_one=True)
    _require_index_range(
        data["outputPopulationIndex"], output_population_count, path, "outputPopulationIndex", allow_negative_one=True
    )

    for index, value in enumerate(data["presynapticSigns"]):
        if value not in (-1, 1, -1.0, 1.0):
            _invalid(path, f"field 'presynapticSigns'[{index}] = {value!r} must be -1 or 1")

    return data


def load_graph_json(path: str | Path, device: str | torch.device = "cpu") -> ConnectomeGraph:
    """Load and validate a connectome graph JSON file (`serializeGraph`'s
    output, `scripts/training/export-traces.ts`). Raises
    `InvalidGraphJsonError` (a `ValueError`) naming the missing/malformed
    field on any schema violation, rather than letting the caller hit a bare
    `KeyError` here or an opaque shape mismatch later inside
    `model.PreparedGraph.__init__`."""
    try:
        data = json.loads(Path(path).read_text())
    except json.JSONDecodeError as error:
        raise InvalidGraphJsonError(f"Invalid connectome graph JSON ({path}): not valid JSON: {error}") from error

    data = _validate_graph_json(data, path)
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
