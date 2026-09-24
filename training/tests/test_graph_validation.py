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


def _valid_graph_dict_two_edges_one_row() -> dict:
    """A 3-neuron graph where neuron 0 has two outgoing edges (to neurons 1
    and 2), for exercising the per-row strictly-increasing
    `postsynapticIndices` check."""
    return {
        "metadata": {
            "formatVersion": 1,
            "neuronCount": 3,
            "edgeCount": 2,
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
        "biologicalIds": ["1", "2", "3"],
        "presynapticOffsets": [0, 2, 2, 2],
        "postsynapticIndices": [1, 2],
        "contactMagnitudes": [0.5, 0.5],
        "presynapticSigns": [1, -1, -1],
        "inputChannelIndex": [0, -1, -1],
        "inputWeight": [1.0, 0.0, 0.0],
        "outputPopulationIndex": [-1, 0, 0],
        "outputWeight": [0.0, 1.0, 1.0],
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


def test_load_graph_json_unsupported_format_version(tmp_path):
    data = _valid_graph_dict()
    data["metadata"]["formatVersion"] = 2
    path = _write(tmp_path, data)
    with pytest.raises(InvalidGraphJsonError, match="formatVersion"):
        load_graph_json(path)


def test_load_graph_json_non_positive_contact_magnitude(tmp_path):
    data = _valid_graph_dict()
    data["contactMagnitudes"] = [0.0]  # finite, but not > 0
    path = _write(tmp_path, data)
    with pytest.raises(InvalidGraphJsonError, match="contactMagnitudes"):
        load_graph_json(path)


def test_load_graph_json_duplicate_postsynaptic_index_in_row(tmp_path):
    data = _valid_graph_dict_two_edges_one_row()
    data["postsynapticIndices"] = [1, 1]  # duplicate edge within row 0
    path = _write(tmp_path, data)
    with pytest.raises(InvalidGraphJsonError, match="postsynapticIndices"):
        load_graph_json(path)


def test_load_graph_json_out_of_order_postsynaptic_index_in_row(tmp_path):
    data = _valid_graph_dict_two_edges_one_row()
    data["postsynapticIndices"] = [2, 1]  # out of order within row 0
    path = _write(tmp_path, data)
    with pytest.raises(InvalidGraphJsonError, match="postsynapticIndices"):
        load_graph_json(path)


def test_load_graph_json_timestep_seconds_not_positive(tmp_path):
    data = _valid_graph_dict()
    data["metadata"]["timestepSeconds"] = 0.0
    path = _write(tmp_path, data)
    with pytest.raises(InvalidGraphJsonError, match="timestepSeconds"):
        load_graph_json(path)


def test_load_graph_json_negative_leak_rate(tmp_path):
    data = _valid_graph_dict()
    data["metadata"]["leakRate"] = -0.1
    path = _write(tmp_path, data)
    with pytest.raises(InvalidGraphJsonError, match="leakRate"):
        load_graph_json(path)


def test_load_graph_json_rate_min_exceeds_rate_max(tmp_path):
    data = _valid_graph_dict()
    data["metadata"]["rateMin"] = 2.0
    data["metadata"]["rateMax"] = 1.0
    path = _write(tmp_path, data)
    with pytest.raises(InvalidGraphJsonError, match="rateMin"):
        load_graph_json(path)


def test_load_graph_json_input_clamp_min_exceeds_max(tmp_path):
    data = _valid_graph_dict()
    data["metadata"]["inputClampMin"] = 2.0
    data["metadata"]["inputClampMax"] = 1.0
    path = _write(tmp_path, data)
    with pytest.raises(InvalidGraphJsonError, match="inputClampMin"):
        load_graph_json(path)


def test_load_graph_json_negative_global_gain(tmp_path):
    data = _valid_graph_dict()
    data["metadata"]["globalGain"] = -1.0
    path = _write(tmp_path, data)
    with pytest.raises(InvalidGraphJsonError, match="globalGain"):
        load_graph_json(path)


def test_load_graph_json_negative_contact_magnitude(tmp_path):
    data = _valid_graph_dict()
    data["contactMagnitudes"] = [-0.5]
    path = _write(tmp_path, data)
    with pytest.raises(InvalidGraphJsonError, match="contactMagnitudes"):
        load_graph_json(path)


def test_load_graph_json_negative_timestep_seconds(tmp_path):
    data = _valid_graph_dict()
    data["metadata"]["timestepSeconds"] = -0.01
    path = _write(tmp_path, data)
    with pytest.raises(InvalidGraphJsonError, match="timestepSeconds"):
        load_graph_json(path)


# --- JSON `true`/`false` must not silently pass as 1/0 (Python `bool` is an
# `int` subclass, but a JSON boolean is not a JSON number; TS's
# `Number.isFinite`/`Number.isInteger` correctly reject it). ---


def test_load_graph_json_boolean_format_version_rejected(tmp_path):
    data = _valid_graph_dict()
    data["metadata"]["formatVersion"] = True  # would equal 1, but is not a number
    path = _write(tmp_path, data)
    with pytest.raises(InvalidGraphJsonError, match="formatVersion"):
        load_graph_json(path)


def test_load_graph_json_boolean_contact_magnitude_rejected(tmp_path):
    data = _valid_graph_dict()
    data["contactMagnitudes"] = [True]
    path = _write(tmp_path, data)
    with pytest.raises(InvalidGraphJsonError, match="contactMagnitudes"):
        load_graph_json(path)


def test_load_graph_json_boolean_presynaptic_sign_rejected(tmp_path):
    data = _valid_graph_dict()
    data["presynapticSigns"] = [True, -1]  # would equal 1, but is not a number
    path = _write(tmp_path, data)
    with pytest.raises(InvalidGraphJsonError, match="presynapticSigns"):
        load_graph_json(path)


def test_load_graph_json_boolean_postsynaptic_index_rejected(tmp_path):
    data = _valid_graph_dict()
    data["postsynapticIndices"] = [True]  # would equal 1, in range, but is not an int
    path = _write(tmp_path, data)
    with pytest.raises(InvalidGraphJsonError, match="postsynapticIndices"):
        load_graph_json(path)


def test_load_graph_json_boolean_neuron_count_rejected(tmp_path):
    data = _valid_graph_dict()
    data["metadata"]["neuronCount"] = True  # would equal 1, not 2, but must not silently coerce
    path = _write(tmp_path, data)
    with pytest.raises(InvalidGraphJsonError, match="neuronCount"):
        load_graph_json(path)


# --- Boundary values that must be ACCEPTED (inclusive bounds), not rejected. ---


def test_load_graph_json_accepts_rate_min_equal_rate_max(tmp_path):
    data = _valid_graph_dict()
    data["metadata"]["rateMin"] = 1.0
    data["metadata"]["rateMax"] = 1.0
    path = _write(tmp_path, data)
    load_graph_json(path)  # must not raise


def test_load_graph_json_accepts_input_clamp_min_equal_max(tmp_path):
    data = _valid_graph_dict()
    data["metadata"]["inputClampMin"] = 0.5
    data["metadata"]["inputClampMax"] = 0.5
    path = _write(tmp_path, data)
    load_graph_json(path)  # must not raise


def test_load_graph_json_accepts_zero_leak_rate(tmp_path):
    data = _valid_graph_dict()
    data["metadata"]["leakRate"] = 0.0
    path = _write(tmp_path, data)
    load_graph_json(path)  # must not raise


def test_load_graph_json_accepts_zero_global_gain(tmp_path):
    data = _valid_graph_dict()
    data["metadata"]["globalGain"] = 0.0
    path = _write(tmp_path, data)
    load_graph_json(path)  # must not raise


def test_load_graph_json_accepts_two_edges_one_row_fixture(tmp_path):
    """Sanity check on the fixture used by the duplicate/out-of-order
    row-ordering tests: its unmodified, valid form must load cleanly."""
    data = _valid_graph_dict_two_edges_one_row()
    path = _write(tmp_path, data)
    graph = load_graph_json(path)
    assert graph.metadata.edge_count == 2
