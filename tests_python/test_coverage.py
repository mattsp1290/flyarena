"""`.agents/plans/selection-robustness/02-per-selection-chain.md` WP2, step
4: `scripts/selections/coverage.py`'s `compute_coverage` -- a synthetic
6-neuron graph (2 sensory, 2 bridge, 2 descending) with edges wired through
only 2 of the 8 input channels and 2 of the 3 output populations, so the
other 6 channels and 1 population are exercised as the "zero coverage"
flagged case. Also covers the conjunctive (channel AND population) coverage
definition directly, including a dead-end bridge that must not count for
its input channel despite having an incoming edge from it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "selections"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "data"))

import binfmt  # noqa: E402
import coverage  # noqa: E402
from coverage import compute_coverage  # noqa: E402


def _tiny_graph() -> "binfmt.GraphArrays":
    # neurons: 0=sensory(foodBearing), 1=sensory(foodDistance),
    #          2=bridge, 3=bridge, 4=descending(thrust), 5=descending(yaw)
    # edges (CSR, pre-ordered): 0->2, 1->3, 2->4, 3->5
    neuron_count = 6
    presynaptic_offsets = np.array([0, 1, 2, 3, 4, 4, 4], dtype=np.uint32)
    postsynaptic_indices = np.array([2, 3, 4, 5], dtype=np.uint32)
    contact_magnitudes = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    presynaptic_signs = np.array([1, 1, 1, 1, 1, 1], dtype=np.int8)
    input_channel_index = np.array([0, 1, -1, -1, -1, -1], dtype=np.int32)
    input_weight = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    output_population_index = np.array([-1, -1, -1, -1, 0, 1], dtype=np.int32)
    output_weight = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    metadata = {
        "formatVersion": 1,
        "neuronCount": neuron_count,
        "edgeCount": 4,
        "inputChannelCount": 8,
        "outputPopulationCount": 3,
    }
    return binfmt.GraphArrays(
        metadata=metadata,
        biological_ids=np.arange(neuron_count, dtype=np.uint64),
        presynaptic_offsets=presynaptic_offsets,
        postsynaptic_indices=postsynaptic_indices,
        contact_magnitudes=contact_magnitudes,
        presynaptic_signs=presynaptic_signs,
        input_channel_index=input_channel_index,
        input_weight=input_weight,
        output_population_index=output_population_index,
        output_weight=output_weight,
    )


def test_bridge_selected_count_excludes_sensory_and_descending() -> None:
    coverage = compute_coverage(_tiny_graph())
    assert coverage["bridgeSelectedCount"] == 2


def test_per_channel_coverage_and_flagging() -> None:
    coverage = compute_coverage(_tiny_graph())
    by_channel = {entry["channel"]: entry for entry in coverage["perChannel"]}
    assert len(by_channel) == 8
    assert by_channel["foodBearing"]["coveredBridgeCount"] == 1
    assert by_channel["foodBearing"]["flagged"] is False
    assert by_channel["foodDistance"]["coveredBridgeCount"] == 1
    assert by_channel["foodDistance"]["flagged"] is False
    for unwired in ("hazardBearing", "hazardDistance", "forwardClearance", "leftClearance", "rightClearance", "speed"):
        assert by_channel[unwired]["coveredBridgeCount"] == 0
        assert by_channel[unwired]["flagged"] is True


def test_per_population_coverage_and_flagging() -> None:
    coverage = compute_coverage(_tiny_graph())
    by_population = {entry["population"]: entry for entry in coverage["perPopulation"]}
    assert len(by_population) == 3
    assert by_population["thrust"]["coveredBridgeCount"] == 1
    assert by_population["thrust"]["flagged"] is False
    assert by_population["yaw"]["coveredBridgeCount"] == 1
    assert by_population["yaw"]["flagged"] is False
    assert by_population["brake"]["coveredBridgeCount"] == 0
    assert by_population["brake"]["flagged"] is True


def test_overall_flagged_true_when_any_channel_or_population_has_zero_coverage() -> None:
    coverage = compute_coverage(_tiny_graph())
    assert coverage["flagged"] is True


def test_fully_covered_graph_is_not_flagged() -> None:
    """Every sensory neuron feeds a bridge neuron that feeds every
    descending population, and every channel/population is wired -- the
    "clean" case, verifying `flagged` can be `False`."""
    neuron_count = 11  # 8 sensory (one per channel) + 1 bridge + ... simplified below
    # Simplify: one sensory neuron per of the 8 channels, all feeding one
    # shared bridge neuron, which feeds one descending neuron per population.
    sensory_count = 8
    bridge_index = sensory_count
    descending_indices = [sensory_count + 1 + p for p in range(3)]
    neuron_count = sensory_count + 1 + 3

    pre_lists: list[list[int]] = [[] for _ in range(neuron_count)]
    for c in range(sensory_count):
        pre_lists[c].append(bridge_index)
    for descending in descending_indices:
        pre_lists[bridge_index].append(descending)

    offsets = [0]
    post: list[int] = []
    for pre in range(neuron_count):
        post.extend(pre_lists[pre])
        offsets.append(len(post))

    presynaptic_offsets = np.array(offsets, dtype=np.uint32)
    postsynaptic_indices = np.array(post, dtype=np.uint32)
    contact_magnitudes = np.ones(len(post), dtype=np.float32)
    presynaptic_signs = np.ones(neuron_count, dtype=np.int8)

    input_channel_index = np.full(neuron_count, -1, dtype=np.int32)
    input_channel_index[:sensory_count] = np.arange(sensory_count, dtype=np.int32)
    input_weight = np.zeros(neuron_count, dtype=np.float32)
    input_weight[:sensory_count] = 1.0

    output_population_index = np.full(neuron_count, -1, dtype=np.int32)
    for p, descending in enumerate(descending_indices):
        output_population_index[descending] = p
    output_weight = np.zeros(neuron_count, dtype=np.float32)
    for descending in descending_indices:
        output_weight[descending] = 1.0

    metadata = {
        "formatVersion": 1,
        "neuronCount": neuron_count,
        "edgeCount": len(post),
        "inputChannelCount": 8,
        "outputPopulationCount": 3,
    }
    graph = binfmt.GraphArrays(
        metadata=metadata,
        biological_ids=np.arange(neuron_count, dtype=np.uint64),
        presynaptic_offsets=presynaptic_offsets,
        postsynaptic_indices=postsynaptic_indices,
        contact_magnitudes=contact_magnitudes,
        presynaptic_signs=presynaptic_signs,
        input_channel_index=input_channel_index,
        input_weight=input_weight,
        output_population_index=output_population_index,
        output_weight=output_weight,
    )
    coverage = compute_coverage(graph)
    assert coverage["bridgeSelectedCount"] == 1
    assert coverage["flagged"] is False
    assert all(entry["coveredBridgeCount"] == 1 for entry in coverage["perChannel"])
    assert all(entry["coveredBridgeCount"] == 1 for entry in coverage["perPopulation"])
    # Every channel pairs with every population through the one shared bridge.
    assert coverage["pairCoverage"] == [[1] * 3 for _ in range(8)]


def test_dead_end_bridge_does_not_count_for_its_input_channel() -> None:
    """A dual-review finding on an earlier version of `compute_coverage`:
    it counted a bridge for channel c merely because it had an incoming
    edge from a channel-c sensory neuron, even if that bridge had no
    outgoing edge to any descending neuron at all -- exactly the "dead-end
    bridge" `compile.py` can produce (bridge candidates are chosen on the
    full traced-edge graph, before `select_edges` drops edges below
    `SYNAPSE_THRESHOLD`). `foodBearing` must NOT count a bridge that is
    reached but goes nowhere."""
    # neurons: 0=sensory(foodBearing), 1=bridge (dead end, no out-edge),
    #          2=bridge (real path), 3=descending(thrust)
    # edges: 0->1 (dead end), 0->2, 2->3
    neuron_count = 4
    presynaptic_offsets = np.array([0, 2, 2, 3, 3], dtype=np.uint32)
    postsynaptic_indices = np.array([1, 2, 3], dtype=np.uint32)
    contact_magnitudes = np.ones(3, dtype=np.float32)
    presynaptic_signs = np.ones(neuron_count, dtype=np.int8)
    input_channel_index = np.array([0, -1, -1, -1], dtype=np.int32)
    input_weight = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    output_population_index = np.array([-1, -1, -1, 0], dtype=np.int32)
    output_weight = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    metadata = {
        "formatVersion": 1,
        "neuronCount": neuron_count,
        "edgeCount": 3,
        "inputChannelCount": 8,
        "outputPopulationCount": 3,
    }
    graph = binfmt.GraphArrays(
        metadata=metadata,
        biological_ids=np.arange(neuron_count, dtype=np.uint64),
        presynaptic_offsets=presynaptic_offsets,
        postsynaptic_indices=postsynaptic_indices,
        contact_magnitudes=contact_magnitudes,
        presynaptic_signs=presynaptic_signs,
        input_channel_index=input_channel_index,
        input_weight=input_weight,
        output_population_index=output_population_index,
        output_weight=output_weight,
    )
    coverage = compute_coverage(graph)
    assert coverage["bridgeSelectedCount"] == 2  # both bridge 1 (dead end) and bridge 2 are selected
    by_channel = {entry["channel"]: entry for entry in coverage["perChannel"]}
    # Only bridge 2 has BOTH an in-edge from foodBearing and an out-edge to
    # a descending neuron; bridge 1 has the in-edge but no out-edge, so it
    # must not be counted -- the conjunctive marginal, not a plain "reached
    # from foodBearing" count (which would wrongly be 2).
    assert by_channel["foodBearing"]["coveredBridgeCount"] == 1
    assert by_channel["foodBearing"]["flagged"] is False
    # The joint (channel, population) matrix: foodBearing x thrust has
    # exactly one covering bridge (bridge 2); every other pair is 0.
    channel_row = coverage["pairCoverage"][0]  # foodBearing is index 0
    assert channel_row == [1, 0, 0]  # thrust, yaw, brake


# ---------------------------------------------------------------------------
# `coverage.py main()`'s selection-scratch guard (thermo-maintainability
# review finding I3): every invocation of this script is, by construction,
# a selection-scratch invocation (there is no shipped `coverage-v1.json`
# under `public/data`, and no CLI mode distinguishing "ordinary republish"
# from "selection scratch" the way `explain.py --selection-mode` does), so
# `guard_selection_scratch_target` must be called with `selection_mode=True`
# unconditionally -- mirroring `test_explain.py`'s
# `test_selection_mode_guard_refuses_public_data_out_even_when_graph_sha_matches_shipped`.
# ---------------------------------------------------------------------------


def _write_tiny_graph_gz(path: Path) -> str:
    """Encodes `_tiny_graph()` (adding the dynamics metadata fields
    `compute_coverage` doesn't need but `binfmt.validate_graph`/
    `encode_graph_binary` require) to a deterministic gzip file at `path`,
    returning its decompressed sha256 (the value `--manifest`'s
    `binarySha256` and the shipped manifest fixture must both carry for
    `load_verified_graph`/the guard's sha comparison to line up)."""
    graph = _tiny_graph()
    graph.metadata.update(
        {
            "timestepSeconds": 1.0 / 30.0,
            "leakRate": 0.1,
            "rateMin": -2.0,
            "rateMax": 2.0,
            "inputClampMin": -1.0,
            "inputClampMax": 1.0,
            "globalGain": 1.0,
        }
    )
    binary = binfmt.encode_graph_binary(binfmt.validate_graph(graph))
    binfmt.write_gzip_deterministic(binary, path)
    return binfmt.sha256_hex(binary)


def test_main_guard_refuses_public_data_out_even_when_graph_sha_matches_shipped(tmp_path, monkeypatch) -> None:
    graph_path = tmp_path / "graph.bin.gz"
    graph_sha256 = _write_tiny_graph_gz(graph_path)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"binarySha256": graph_sha256}))

    # The fake "shipped" manifest's binarySha256 deliberately MATCHES the
    # selection's own graph sha -- proving the guard still refuses (unlike
    # explain.py's non-selection-mode path, which would allow this).
    fake_public_data = tmp_path / "shipped" / "public" / "data"
    fake_docs = tmp_path / "shipped" / "docs"
    fake_public_data.mkdir(parents=True)
    fake_docs.mkdir(parents=True)
    (fake_public_data / "malecns-arena-v1.manifest.json").write_text(json.dumps({"binarySha256": graph_sha256}))
    monkeypatch.setattr(coverage, "PUBLIC_DATA_DIR", fake_public_data)
    monkeypatch.setattr(coverage, "DOCS_DIR", fake_docs)

    out_path = fake_public_data / "coverage-selection-larger.json"
    with pytest.raises(ValueError, match="resolves under"):
        coverage.main(["--graph", str(graph_path), "--manifest", str(manifest_path), "--out", str(out_path)])
    assert not out_path.exists()


def test_main_guard_allows_a_scratch_out_outside_public_and_docs(tmp_path, monkeypatch) -> None:
    graph_path = tmp_path / "graph.bin.gz"
    graph_sha256 = _write_tiny_graph_gz(graph_path)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"binarySha256": graph_sha256}))

    fake_public_data = tmp_path / "shipped" / "public" / "data"
    fake_docs = tmp_path / "shipped" / "docs"
    fake_public_data.mkdir(parents=True)
    fake_docs.mkdir(parents=True)
    (fake_public_data / "malecns-arena-v1.manifest.json").write_text(json.dumps({"binarySha256": graph_sha256}))
    monkeypatch.setattr(coverage, "PUBLIC_DATA_DIR", fake_public_data)
    monkeypatch.setattr(coverage, "DOCS_DIR", fake_docs)

    out_path = tmp_path / "selections" / "fake-id" / "coverage.json"
    coverage.main(["--graph", str(graph_path), "--manifest", str(manifest_path), "--out", str(out_path)])
    assert out_path.exists()
    assert json.loads(out_path.read_text())["sourceGraphSha256"] == graph_sha256
