"""`load_graph_json` validation (thermo-maintainability review finding I2):
malformed graph JSON must raise `InvalidGraphJsonError` (a `ValueError`)
naming the bad field, not a bare `KeyError` at the load site or an opaque
shape mismatch several frames away inside `model.PreparedGraph.__init__`."""
from __future__ import annotations

import json

import pytest

from flyarena_training.graph import InvalidGraphJsonError, load_graph_json


def _valid_graph_dict() -> dict:
    """A minimal, valid 2-neuron, 1-edge graph, matching `serializeGraph`'s
    schema (`scripts/training/export-traces.ts`)."""
    return {
        "metadata": {
            "formatVersion": 1,
            "neuronCount": 2,
            "edgeCount": 1,
            "inputChannelCount": 1,
            "outputPopulationCount": 1,
            "timestepSeconds": 0.01,
            "leakRate": 0.1,
            "rateMin": 0.0,
            "rateMax": 1.0,
            "inputClampMin": 0.0,
            "inputClampMax": 1.0,
            "globalGain": 1.0,
        },
        "biologicalIds": ["1", "2"],
        "presynapticOffsets": [0, 1, 1],
        "postsynapticIndices": [1],
        "contactMagnitudes": [0.5],
        "presynapticSigns": [1, -1],
        "inputChannelIndex": [0, -1],
        "inputWeight": [1.0, 0.0],
        "outputPopulationIndex": [-1, 0],
        "outputWeight": [0.0, 1.0],
    }


def _write(tmp_path, data) -> str:
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_load_graph_json_accepts_a_valid_graph(tmp_path):
    path = _write(tmp_path, _valid_graph_dict())
    graph = load_graph_json(path, device="cpu")
    assert graph.metadata.neuron_count == 2
    assert graph.metadata.edge_count == 1


def test_load_graph_json_not_json(tmp_path):
    path = tmp_path / "graph.json"
    path.write_text("not json")
    with pytest.raises(InvalidGraphJsonError, match="not valid JSON"):
        load_graph_json(str(path))


def test_load_graph_json_missing_metadata_key(tmp_path):
    data = _valid_graph_dict()
    del data["metadata"]["neuronCount"]
    path = _write(tmp_path, data)
    with pytest.raises(InvalidGraphJsonError, match="neuronCount"):
        load_graph_json(path)


def test_load_graph_json_missing_top_level_field(tmp_path):
    data = _valid_graph_dict()
    del data["postsynapticIndices"]
    path = _write(tmp_path, data)
    with pytest.raises(InvalidGraphJsonError, match="postsynapticIndices"):
        load_graph_json(path)


def test_load_graph_json_wrong_length_presynaptic_offsets(tmp_path):
    data = _valid_graph_dict()
    data["presynapticOffsets"] = [0, 1]  # should be neuronCount + 1 == 3
    path = _write(tmp_path, data)
    with pytest.raises(InvalidGraphJsonError, match="presynapticOffsets"):
        load_graph_json(path)


def test_load_graph_json_wrong_length_per_neuron_field(tmp_path):
    data = _valid_graph_dict()
    data["presynapticSigns"] = [1]  # should have length neuronCount == 2
    path = _write(tmp_path, data)
    with pytest.raises(InvalidGraphJsonError, match="presynapticSigns"):
        load_graph_json(path)


def test_load_graph_json_wrong_length_per_edge_field(tmp_path):
    data = _valid_graph_dict()
    data["contactMagnitudes"] = [0.5, 0.25]  # should have length edgeCount == 1
    path = _write(tmp_path, data)
    with pytest.raises(InvalidGraphJsonError, match="contactMagnitudes"):
        load_graph_json(path)


def test_load_graph_json_out_of_range_postsynaptic_index(tmp_path):
    data = _valid_graph_dict()
    data["postsynapticIndices"] = [5]  # neuronCount is 2, valid range [0, 2)
    path = _write(tmp_path, data)
    with pytest.raises(InvalidGraphJsonError, match="postsynapticIndices"):
        load_graph_json(path)


def test_load_graph_json_non_finite_contact_magnitude(tmp_path):
    data = _valid_graph_dict()
    data["contactMagnitudes"] = [float("nan")]
    path = _write(tmp_path, data)
    with pytest.raises(InvalidGraphJsonError, match="contactMagnitudes"):
        load_graph_json(path)


def test_load_graph_json_invalid_presynaptic_sign(tmp_path):
    data = _valid_graph_dict()
    data["presynapticSigns"] = [1, 0]  # must be -1 or 1
    path = _write(tmp_path, data)
    with pytest.raises(InvalidGraphJsonError, match="presynapticSigns"):
        load_graph_json(path)


def test_load_graph_json_presynaptic_offsets_not_starting_at_zero(tmp_path):
    data = _valid_graph_dict()
    data["presynapticOffsets"] = [1, 1, 1]
    path = _write(tmp_path, data)
    with pytest.raises(InvalidGraphJsonError, match="presynapticOffsets"):
        load_graph_json(path)


def test_load_graph_json_presynaptic_offsets_not_non_decreasing(tmp_path):
    data = _valid_graph_dict()
    data["presynapticOffsets"] = [0, 1, 0]
    path = _write(tmp_path, data)
    with pytest.raises(InvalidGraphJsonError):
        load_graph_json(path)
