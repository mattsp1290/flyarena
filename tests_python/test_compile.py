"""Compiler invariant tests, driven entirely from tiny CSV fixtures under
`tests_python/fixtures/` -- no MaleCNS raw data required, so this suite runs
in CI without downloading anything.

Covers:
- header/section byte offsets matching docs/graph-format.md's worked
  example exactly,
- duplicate `(pre, post)` row aggregation (summed magnitudes),
- dropped-endpoint accounting,
- self-loop accounting,
- strictly-increasing CSR row ordering,
- deterministic byte-identical compiler output across repeated runs,
- and every rewiring invariant `scripts/data/rewire.py` claims to preserve
  (in-degree, out-degree, edge-weight multiset, sign ownership, node set,
  edge count), plus its self-loop/duplicate policy.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "data"))

import binfmt  # noqa: E402
import compile as compiler  # noqa: E402
import rewire  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

DEFAULT_METADATA_PARAMS = {
    "timestepSeconds": 1.0 / 30.0,
    "leakRate": 0.35,
    "rateMin": -2.0,
    "rateMax": 2.0,
    "inputClampMin": -1.0,
    "inputClampMax": 1.0,
    "globalGain": 0.5,
}


def _load_csv_fixture(nodes_name: str, edges_name: str):
    """Load a tiny nodes/edges CSV pair into the inputs `compile_graph`
    expects: node id list, signs map, input/output assignment maps, and an
    edges DataFrame with columns pre/post/weight (int/int/float)."""
    nodes_path = FIXTURES_DIR / nodes_name
    edges_path = FIXTURES_DIR / edges_name

    node_ids: list[int] = []
    signs: dict[int, int] = {}
    input_assignment: dict[int, tuple[int, float]] = {}
    output_assignment: dict[int, tuple[int, float]] = {}

    with open(nodes_path, newline="") as fh:
        for row in csv.DictReader(fh):
            body = int(row["body_id"])
            node_ids.append(body)
            signs[body] = int(row["sign"])
            if row["input_channel"].strip() != "":
                input_assignment[body] = (int(row["input_channel"]), float(row["input_weight"]))
            if row["output_population"].strip() != "":
                output_assignment[body] = (int(row["output_population"]), float(row["output_weight"]))

    edges = pd.read_csv(edges_path)
    return node_ids, signs, input_assignment, output_assignment, edges


def _compile_fixture(nodes_name: str, edges_name: str, **metadata_overrides):
    node_ids, signs, input_assignment, output_assignment, edges = _load_csv_fixture(
        nodes_name, edges_name
    )
    metadata_params = {**DEFAULT_METADATA_PARAMS, **metadata_overrides}
    graph, stats = compiler.compile_graph(
        node_ids=node_ids,
        edges=edges,
        signs=signs,
        input_assignment=input_assignment,
        output_assignment=output_assignment,
        input_channel_count=2,
        output_population_count=2,
        metadata_params=metadata_params,
    )
    return graph, stats


# ---------------------------------------------------------------------------
# Header / section layout vs. docs/graph-format.md's worked example.
# ---------------------------------------------------------------------------


def test_worked_example_layout_matches_doc():
    """docs/graph-format.md's "Worked offset example" table, for
    neuronCount=4, edgeCount=2, verbatim."""
    layout = binfmt.compute_graph_layout(neuron_count=4, edge_count=2)

    assert (layout.biological_ids.byte_offset, layout.biological_ids.byte_length) == (56, 32)
    assert (layout.presynaptic_offsets.byte_offset, layout.presynaptic_offsets.byte_length) == (88, 20)
    assert (layout.postsynaptic_indices.byte_offset, layout.postsynaptic_indices.byte_length) == (112, 8)
    assert (layout.contact_magnitudes.byte_offset, layout.contact_magnitudes.byte_length) == (120, 8)
    assert (layout.presynaptic_signs.byte_offset, layout.presynaptic_signs.byte_length) == (128, 4)
    assert (layout.input_channel_index.byte_offset, layout.input_channel_index.byte_length) == (136, 16)
    assert (layout.input_weight.byte_offset, layout.input_weight.byte_length) == (152, 16)
    assert (layout.output_population_index.byte_offset, layout.output_population_index.byte_length) == (168, 16)
    assert (layout.output_weight.byte_offset, layout.output_weight.byte_length) == (184, 16)
    assert layout.total_bytes == 200


def test_tiny_fixture_compiles_to_exact_worked_example_bytes():
    graph, stats = _compile_fixture("tiny_nodes.csv", "tiny_edges.csv")

    assert stats == {
        "neuronCount": 4,
        "edgeCount": 2,
        "droppedEndpointCount": 0,
        "duplicateAggregatedCount": 0,
        "selfLoopCount": 0,
        "isolatedNodeCount": 0,
    }

    # Node order is sorted-ascending body id: 1001, 1002, 1003, 1004 -> 0..3.
    assert graph.biological_ids.tolist() == [1001, 1002, 1003, 1004]
    assert graph.presynaptic_offsets.tolist() == [0, 1, 2, 2, 2]
    assert graph.postsynaptic_indices.tolist() == [2, 3]
    assert graph.contact_magnitudes.tolist() == [2.0, 1.5]
    assert graph.presynaptic_signs.tolist() == [1, -1, 1, 1]
    assert graph.input_channel_index.tolist() == [0, 1, -1, -1]
    assert graph.output_population_index.tolist() == [-1, -1, 0, 1]

    binary = binfmt.encode_graph_binary(graph)
    assert len(binary) == 200


def test_tiny_fixture_header_fields_round_trip():
    graph, _ = _compile_fixture("tiny_nodes.csv", "tiny_edges.csv")
    binary = binfmt.encode_graph_binary(graph)

    import struct

    magic, version, neuron_count, edge_count, input_channels, output_populations = struct.unpack_from(
        "<4sIIIII", binary, 0
    )
    assert magic == b"FANG"
    assert version == 1
    assert neuron_count == 4
    assert edge_count == 2
    assert input_channels == 2
    assert output_populations == 2


# ---------------------------------------------------------------------------
# Aggregation / dropped-endpoint / self-loop / sort-order invariants.
# ---------------------------------------------------------------------------


def test_duplicate_edges_are_aggregated_by_summing():
    graph, stats = _compile_fixture("dup_nodes.csv", "dup_edges.csv")

    assert stats["neuronCount"] == 5
    assert stats["droppedEndpointCount"] == 1  # 2001 -> 2999 (2999 is not a node)
    assert stats["duplicateAggregatedCount"] == 1  # (2001, 2003) appears twice
    assert stats["selfLoopCount"] == 1  # 2003 -> 2003
    assert stats["isolatedNodeCount"] == 0
    assert stats["edgeCount"] == 6  # 7 kept rows - 1 duplicate collapsed

    # body ids 2001..2005 sort to indices 0..4.
    index_of = {body: i for i, body in enumerate(graph.biological_ids.tolist())}
    assert index_of == {2001: 0, 2002: 1, 2003: 2, 2004: 3, 2005: 4}

    def magnitude_of(pre_body: int, post_body: int) -> float:
        pre_idx = index_of[pre_body]
        post_idx = index_of[post_body]
        start = graph.presynaptic_offsets[pre_idx]
        end = graph.presynaptic_offsets[pre_idx + 1]
        row_posts = graph.postsynaptic_indices[start:end].tolist()
        row_mags = graph.contact_magnitudes[start:end].tolist()
        return row_mags[row_posts.index(post_idx)]

    assert magnitude_of(2001, 2003) == pytest.approx(3.0)  # 1.0 + 2.0
    assert magnitude_of(2003, 2003) == pytest.approx(4.0)
    assert magnitude_of(2003, 2004) == pytest.approx(1.0)
    assert magnitude_of(2003, 2001) == pytest.approx(0.5)


def test_csr_rows_are_strictly_increasing_after_scrambled_input_order():
    # dup_edges.csv inserts pre=2003's targets out of order (2004, then the
    # self-loop 2003, then 2001 last); the compiled row must still be
    # strictly ascending by postsynaptic index regardless of input order.
    graph, _ = _compile_fixture("dup_nodes.csv", "dup_edges.csv")
    index_of = {body: i for i, body in enumerate(graph.biological_ids.tolist())}
    pre_idx = index_of[2003]
    start = graph.presynaptic_offsets[pre_idx]
    end = graph.presynaptic_offsets[pre_idx + 1]
    row = graph.postsynaptic_indices[start:end].tolist()
    assert row == sorted(row)
    assert len(row) == len(set(row))  # strictly increasing, not merely sorted


def test_validate_graph_accepts_every_fixture():
    for nodes, edges in (("tiny_nodes.csv", "tiny_edges.csv"), ("dup_nodes.csv", "dup_edges.csv")):
        graph, _ = _compile_fixture(nodes, edges)
        binfmt.validate_graph(graph)  # must not raise


# ---------------------------------------------------------------------------
# Determinism.
# ---------------------------------------------------------------------------


def test_compiler_output_is_byte_identical_across_runs():
    graph_a, _ = _compile_fixture("dup_nodes.csv", "dup_edges.csv")
    graph_b, _ = _compile_fixture("dup_nodes.csv", "dup_edges.csv")
    assert binfmt.encode_graph_binary(graph_a) == binfmt.encode_graph_binary(graph_b)


def test_compiler_output_is_byte_identical_under_shuffled_edge_input_order():
    """Aggregation + CSR sorting must make output order-independent: feeding
    the same edge rows in a different order must still produce the same
    bytes."""
    node_ids, signs, input_assignment, output_assignment, edges = _load_csv_fixture(
        "dup_nodes.csv", "dup_edges.csv"
    )
    shuffled = edges.sample(frac=1.0, random_state=42).reset_index(drop=True)

    graph_a, _ = compiler.compile_graph(
        node_ids, edges, signs, input_assignment, output_assignment, 2, 2, DEFAULT_METADATA_PARAMS
    )
    graph_b, _ = compiler.compile_graph(
        node_ids, shuffled, signs, input_assignment, output_assignment, 2, 2, DEFAULT_METADATA_PARAMS
    )
    assert binfmt.encode_graph_binary(graph_a) == binfmt.encode_graph_binary(graph_b)


# ---------------------------------------------------------------------------
# Validation negative cases (mirrors format.ts's validateGraph failure
# modes, from the Python side).
# ---------------------------------------------------------------------------


def test_negative_global_gain_rejected():
    # compile_graph validates internally, so the invalid metadata is
    # rejected at compile time rather than needing a separate validate call.
    with pytest.raises(binfmt.InvalidGraphError):
        _compile_fixture("tiny_nodes.csv", "tiny_edges.csv", globalGain=-1.0)


def test_out_of_order_postsynaptic_row_rejected():
    graph, _ = _compile_fixture("tiny_nodes.csv", "tiny_edges.csv")
    graph.postsynaptic_indices[:] = graph.postsynaptic_indices[::-1]
    # only meaningful if this actually breaks ascending order somewhere;
    # for the 2-edge tiny fixture reversing swaps rows, still ascending per
    # row (each row has exactly one edge) so force a genuine violation:
    graph.presynaptic_offsets[:] = [0, 2, 2, 2, 2]
    graph.postsynaptic_indices = np.array([3, 2], dtype=np.uint32)  # descending within row 0
    graph.metadata = {**graph.metadata, "edgeCount": 2}
    with pytest.raises(binfmt.InvalidGraphError):
        binfmt.validate_graph(graph)


# ---------------------------------------------------------------------------
# Rewiring invariants.
# ---------------------------------------------------------------------------


def _synthetic_graph_for_rewiring(seed: int = 7, neuron_count: int = 40, edge_count_target: int = 220):
    """A larger deterministic synthetic graph (not from a fixture file) with
    enough edges to exercise the double-edge-swap algorithm meaningfully,
    including at least one self-loop and no duplicates."""
    rng = np.random.default_rng(seed)
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < edge_count_target:
        pre = int(rng.integers(0, neuron_count))
        post = int(rng.integers(0, neuron_count))
        pairs.add((pre, post))
    # Guarantee at least one self-loop so the self-loop policy is exercised.
    pairs.add((3, 3))

    pre_arr = np.array([p for p, _ in pairs], dtype=np.int64)
    post_arr = np.array([q for _, q in pairs], dtype=np.uint32)
    weight_arr = rng.uniform(0.1, 5.0, size=len(pairs)).astype(np.float32)

    order = np.lexsort((post_arr, pre_arr))
    pre_sorted = pre_arr[order]
    post_sorted = post_arr[order]
    weight_sorted = weight_arr[order]

    offsets = np.zeros(neuron_count + 1, dtype=np.uint32)
    counts = np.bincount(pre_sorted, minlength=neuron_count)
    offsets[1:] = np.cumsum(counts)

    metadata = {
        "formatVersion": binfmt.SUPPORTED_FORMAT_VERSION,
        "neuronCount": neuron_count,
        "edgeCount": len(pairs),
        "inputChannelCount": 4,
        "outputPopulationCount": 3,
        **DEFAULT_METADATA_PARAMS,
    }
    signs = np.where(rng.random(neuron_count) < 0.5, -1, 1).astype(np.int8)
    input_channel_index = np.full(neuron_count, -1, dtype=np.int32)
    input_channel_index[:4] = np.arange(4)
    input_weight = np.where(input_channel_index >= 0, 1.0, 0.0).astype(np.float32)
    output_population_index = np.full(neuron_count, -1, dtype=np.int32)
    output_population_index[4:7] = np.arange(3)
    output_weight = np.where(output_population_index >= 0, 1.0, 0.0).astype(np.float32)

    graph = binfmt.GraphArrays(
        metadata=metadata,
        biological_ids=np.arange(1_000_000, 1_000_000 + neuron_count, dtype=np.uint64),
        presynaptic_offsets=offsets,
        postsynaptic_indices=post_sorted,
        contact_magnitudes=weight_sorted,
        presynaptic_signs=signs,
        input_channel_index=input_channel_index,
        input_weight=input_weight,
        output_population_index=output_population_index,
        output_weight=output_weight,
    )
    binfmt.validate_graph(graph)
    return graph


def _degree_sequences(graph: binfmt.GraphArrays):
    neuron_count = int(graph.metadata["neuronCount"])
    out_degree = np.diff(graph.presynaptic_offsets.astype(np.int64))
    in_degree = np.bincount(graph.postsynaptic_indices.astype(np.int64), minlength=neuron_count)
    return out_degree, in_degree


def test_rewire_preserves_degree_sequences_and_node_set():
    graph = _synthetic_graph_for_rewiring()
    out_before, in_before = _degree_sequences(graph)

    rewired, stats = rewire.rewire_graph(graph, seed=123)

    assert stats["acceptedSwaps"] > 0  # sanity: the graph is big enough to actually rewire

    out_after, in_after = _degree_sequences(rewired)
    assert np.array_equal(out_before, out_after)
    assert np.array_equal(in_before, in_after)

    assert np.array_equal(graph.biological_ids, rewired.biological_ids)
    assert np.array_equal(graph.presynaptic_signs, rewired.presynaptic_signs)
    assert np.array_equal(graph.input_channel_index, rewired.input_channel_index)
    assert np.array_equal(graph.input_weight, rewired.input_weight)
    assert np.array_equal(graph.output_population_index, rewired.output_population_index)
    assert np.array_equal(graph.output_weight, rewired.output_weight)
    assert int(graph.metadata["edgeCount"]) == int(rewired.metadata["edgeCount"])


def test_rewire_preserves_edge_weight_multiset_per_presynaptic_neuron():
    """Sign ownership + weight travel together with an edge's `pre`; verify
    the multiset of weights *leaving each neuron* (not just globally) is
    unchanged, which is the precise claim "sign ownership is preserved"
    depends on."""
    graph = _synthetic_graph_for_rewiring()
    rewired, _ = rewire.rewire_graph(graph, seed=99)

    neuron_count = int(graph.metadata["neuronCount"])
    for pre in range(neuron_count):
        s0, e0 = graph.presynaptic_offsets[pre], graph.presynaptic_offsets[pre + 1]
        s1, e1 = rewired.presynaptic_offsets[pre], rewired.presynaptic_offsets[pre + 1]
        before = sorted(graph.contact_magnitudes[s0:e0].tolist())
        after = sorted(rewired.contact_magnitudes[s1:e1].tolist())
        assert before == pytest.approx(after)


def test_rewire_produces_no_duplicate_edges():
    graph = _synthetic_graph_for_rewiring()
    rewired, _ = rewire.rewire_graph(graph, seed=5)
    neuron_count = int(rewired.metadata["neuronCount"])
    pairs = []
    for pre in range(neuron_count):
        start, end = rewired.presynaptic_offsets[pre], rewired.presynaptic_offsets[pre + 1]
        for post in rewired.postsynaptic_indices[start:end].tolist():
            pairs.append((pre, post))
    assert len(pairs) == len(set(pairs))


def test_rewire_disallows_self_loops_by_default_even_when_source_has_one():
    graph = _synthetic_graph_for_rewiring()
    assert int((graph.postsynaptic_indices == np.repeat(
        np.arange(int(graph.metadata["neuronCount"])),
        np.diff(graph.presynaptic_offsets.astype(np.int64)),
    )).sum()) >= 1  # the synthetic graph does contain a self-loop (3 -> 3)

    rewired, stats = rewire.rewire_graph(graph, seed=5, allow_self_loops=False)
    pre_of_edge = np.repeat(
        np.arange(int(rewired.metadata["neuronCount"])),
        np.diff(rewired.presynaptic_offsets.astype(np.int64)),
    )
    self_loops_after = int((pre_of_edge == rewired.postsynaptic_indices.astype(np.int64)).sum())
    assert self_loops_after == 0
    assert stats["allowSelfLoops"] is False


def test_rewire_allow_self_loops_true_does_not_forbid_them():
    graph = _synthetic_graph_for_rewiring()
    rewired, stats = rewire.rewire_graph(graph, seed=5, allow_self_loops=True)
    binfmt.validate_graph(rewired)  # the format itself always permits self-loops
    assert stats["allowSelfLoops"] is True


def test_rewire_is_deterministic_for_a_given_seed():
    graph = _synthetic_graph_for_rewiring()
    rewired_a, _ = rewire.rewire_graph(graph, seed=42)
    rewired_b, _ = rewire.rewire_graph(graph, seed=42)
    assert binfmt.encode_graph_binary(rewired_a) == binfmt.encode_graph_binary(rewired_b)


def test_rewire_round_trips_through_binary_encoding():
    graph = _synthetic_graph_for_rewiring()
    rewired, _ = rewire.rewire_graph(graph, seed=17)
    binary = binfmt.encode_graph_binary(rewired)
    decoded = rewire.decode_graph_binary(binary)
    assert np.array_equal(decoded.biological_ids, rewired.biological_ids)
    assert np.array_equal(decoded.postsynaptic_indices, rewired.postsynaptic_indices)
    assert np.array_equal(decoded.contact_magnitudes, rewired.contact_magnitudes)


def test_rewire_swap_stat_buckets_sum_to_attempts():
    """Every attempted (i, j) draw must land in exactly one counted bucket:
    accepted, or one of the four rejection reasons (including the
    same-index draw, which an earlier version of this compiler silently
    `continue`d past without counting)."""
    graph = _synthetic_graph_for_rewiring()
    _, stats = rewire.rewire_graph(graph, seed=5)
    bucket_sum = (
        stats["acceptedSwaps"]
        + stats["rejectedSelfLoop"]
        + stats["rejectedDuplicate"]
        + stats["rejectedDegenerate"]
        + stats["rejectedSameIndex"]
    )
    assert bucket_sum == stats["attempts"]


def test_decode_graph_binary_rejects_truncated_buffer():
    graph, _ = _compile_fixture("tiny_nodes.csv", "tiny_edges.csv")
    binary = binfmt.encode_graph_binary(graph)
    with pytest.raises(binfmt.InvalidGraphError):
        rewire.decode_graph_binary(binary[:-1])
    with pytest.raises(binfmt.InvalidGraphError):
        rewire.decode_graph_binary(binary[:10])  # shorter than the fixed header


# ---------------------------------------------------------------------------
# assign_signs: duplicate-body-id crash and missing/missing_value
# distinction (review findings from both reviewers on
# feat/qyxw-malecns-graph -- see reviews/feat-qyxw-malecns-graph-*/).
# ---------------------------------------------------------------------------


def test_assign_signs_rejects_duplicate_body_in_neurotransmitter_table():
    nt = pd.DataFrame(
        {"body": [1, 1, 2], "consensus_nt": ["acetylcholine", "gaba", "gaba"]}
    )
    with pytest.raises(ValueError, match="duplicate body"):
        compiler.assign_signs([1, 2], nt)


def test_assign_signs_distinguishes_missing_row_from_missing_value():
    # body 3: no row at all. body 4: a row exists but consensus_nt is NaN.
    nt = pd.DataFrame(
        {
            "body": [1, 2, 4, 5],
            "consensus_nt": ["acetylcholine", "gaba", float("nan"), "unclear"],
        }
    )
    signs, unknown_by_label = compiler.assign_signs([1, 2, 3, 4, 5], nt)

    assert signs == {1: 1, 2: -1, 3: 1, 4: 1, 5: 1}
    assert unknown_by_label == {"missing": 1, "missing_value": 1, "unclear": 1}


def test_select_subgraph_rejects_duplicate_body_id_in_annotations():
    annotations = pd.DataFrame(
        {
            "bodyId": [1, 1, 2],
            "status": ["Traced", "Traced", "Traced"],
            "superclass": ["vnc_sensory", "vnc_sensory", "descending_neuron"],
            "class": ["mechanosensory_tactile", "mechanosensory_tactile", None],
        }
    )
    weights = pd.DataFrame({"body_pre": [1], "body_post": [2], "weight": [5]})
    with pytest.raises(ValueError, match="duplicate bodyId"):
        compiler.select_subgraph(annotations, weights)


# ---------------------------------------------------------------------------
# assign_channels: dedup, empty-input guards, and the boundary case where
# the candidate count is not evenly divisible (or is smaller than) the
# channel/population count.
# ---------------------------------------------------------------------------


def test_assign_channels_dedupes_input_ids():
    # A duplicate id (e.g. from an upstream data-quality issue) must not
    # consume two partition slots for the same body.
    input_assignment, output_assignment = compiler.assign_channels(
        sensory_ids=[10, 10, 20, 30, 40, 50, 60, 70],
        descending_ids=[100, 100, 200, 300],
    )
    assert set(input_assignment.keys()) == {10, 20, 30, 40, 50, 60, 70}
    assert set(output_assignment.keys()) == {100, 200, 300}


def test_assign_channels_rejects_empty_candidate_lists():
    with pytest.raises(ValueError):
        compiler.assign_channels(sensory_ids=[], descending_ids=[1, 2, 3])
    with pytest.raises(ValueError):
        compiler.assign_channels(sensory_ids=[1, 2, 3], descending_ids=[])


def test_assign_channels_handles_fewer_candidates_than_channels():
    # 3 sensory neurons, 8 input channels: every neuron gets a channel, no
    # crash, and every assigned channel index is in range.
    input_assignment, _ = compiler.assign_channels(sensory_ids=[1, 2, 3], descending_ids=[100])
    assert len(input_assignment) == 3
    for channel, _weight in input_assignment.values():
        assert 0 <= channel < compiler.INPUT_CHANNEL_COUNT


def test_calibrate_global_gain_rejects_empty_edges():
    with pytest.raises(ValueError):
        compiler.calibrate_global_gain(pd.DataFrame({"pre": [], "post": [], "weight": []}))


# ---------------------------------------------------------------------------
# Duplicate aggregation determinism under 3+-way ties (a 2-way tie is
# commutative regardless of summation order and cannot exercise this).
# ---------------------------------------------------------------------------


def test_three_way_duplicate_aggregation_is_order_independent():
    node_ids = [1, 2]
    signs = {1: 1, 2: 1}
    input_assignment = {1: (0, 1.0)}
    output_assignment = {2: (0, 1.0)}
    metadata_params = DEFAULT_METADATA_PARAMS

    # Three rows for the same (pre=1, post=2) pair, with magnitudes spanning
    # enough orders of magnitude that float64 summation order could
    # plausibly matter, in two different input orders.
    weights = [1e-8, 1.0, 1e8]
    import itertools

    results = set()
    for permutation in itertools.permutations(weights):
        edges = pd.DataFrame({"pre": [1, 1, 1], "post": [2, 2, 2], "weight": list(permutation)})
        graph, _ = compiler.compile_graph(
            node_ids, edges, signs, input_assignment, output_assignment, 1, 1, metadata_params
        )
        results.add(binfmt.encode_graph_binary(graph))

    assert len(results) == 1, "aggregation order must not depend on input row order"
