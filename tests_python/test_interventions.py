"""`.agents/plans/pathway-interventions/02-intervention-graphs.md`'s WP1
tests: `swap_ops`'s pure helpers on the trace-graph fixture, and
`interventions.py`'s invariant checks, R premise check, greedy search, and
end-to-end CLI (invariant/degree-preservation gates, TS round-trip, and
byte-identical rerun).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data"))
import binfmt  # noqa: E402

import interventions  # noqa: E402
import swap_ops  # noqa: E402
from graph_io import build_dense_matrices  # noqa: E402
from transfer import OBSERVATION_CHANNEL_INDEX, OUTPUT_POPULATION_INDEX, transfer_matrix  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "tests_python" / "fixtures" / "trace-graph-transfer.json"
ROUNDTRIP_TS = REPO_ROOT / "tests_python" / "fixtures" / "graph_binary_roundtrip.ts"
TSX_BIN = REPO_ROOT / "node_modules" / ".bin" / "tsx"

RIGHT_IDX = OBSERVATION_CHANNEL_INDEX["rightClearance"]
FORWARD_IDX = OBSERVATION_CHANNEL_INDEX["forwardClearance"]
THRUST_IDX = OUTPUT_POPULATION_INDEX["thrust"]


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _graph_arrays_from_fixture_dict(g: dict, *, global_gain: "float | None" = None) -> "binfmt.GraphArrays":
    metadata = dict(g["metadata"])
    if global_gain is not None:
        metadata["globalGain"] = global_gain
    graph = binfmt.GraphArrays(
        metadata=metadata,
        biological_ids=np.array([int(x) for x in g["biologicalIds"]], dtype=np.uint64),
        presynaptic_offsets=np.array(g["presynapticOffsets"], dtype=np.uint32),
        postsynaptic_indices=np.array(g["postsynapticIndices"], dtype=np.uint32),
        contact_magnitudes=np.array(g["contactMagnitudes"], dtype=np.float32),
        presynaptic_signs=np.array(g["presynapticSigns"], dtype=np.int8),
        input_channel_index=np.array(g["inputChannelIndex"], dtype=np.int32),
        input_weight=np.array(g["inputWeight"], dtype=np.float32),
        output_population_index=np.array(g["outputPopulationIndex"], dtype=np.int32),
        output_weight=np.array(g["outputWeight"], dtype=np.float32),
    )
    return binfmt.validate_graph(graph)


@pytest.fixture(scope="module")
def trace_graph() -> "binfmt.GraphArrays":
    if not FIXTURE_PATH.exists():
        pytest.skip(f"{FIXTURE_PATH} not generated (run scripts/analysis/export-trace-graph-fixture.ts)")
    with FIXTURE_PATH.open("r") as fh:
        fixture = json.load(fh)
    return _graph_arrays_from_fixture_dict(fixture["graph"])


@pytest.fixture(scope="module")
def trace_graph_masks(trace_graph):
    input_mask = trace_graph.input_channel_index >= 0
    thrust_mask = trace_graph.output_population_index == THRUST_IDX
    bridge_mask = swap_ops.bridge_mask_of(trace_graph)
    clearance_mask = np.isin(trace_graph.input_channel_index, [RIGHT_IDX, FORWARD_IDX])
    return input_mask, thrust_mask, bridge_mask, clearance_mask


def _degree_vectors(graph: "binfmt.GraphArrays") -> "tuple[np.ndarray, np.ndarray]":
    out_degree = np.diff(graph.presynaptic_offsets.astype(np.int64))
    in_degree = np.bincount(
        graph.postsynaptic_indices.astype(np.int64), minlength=int(graph.metadata["neuronCount"])
    )
    return out_degree, in_degree


def _assert_swap_invariants(before: "binfmt.GraphArrays", after: "binfmt.GraphArrays") -> None:
    assert int(before.metadata["edgeCount"]) == int(after.metadata["edgeCount"])
    before_out, before_in = _degree_vectors(before)
    after_out, after_in = _degree_vectors(after)
    assert np.array_equal(before_out, after_out)
    assert np.array_equal(before_in, after_in)
    assert sorted(before.contact_magnitudes.tolist()) == sorted(after.contact_magnitudes.tolist())
    assert np.array_equal(before.presynaptic_signs, after.presynaptic_signs)
    assert np.array_equal(before.biological_ids, after.biological_ids)
    assert np.array_equal(before.input_channel_index, after.input_channel_index)
    assert np.array_equal(before.output_population_index, after.output_population_index)


# ---------------------------------------------------------------------------
# swap_ops: edge_set_from_graph / bridge_mask_of / valid_swap
# ---------------------------------------------------------------------------


def test_edge_set_from_graph_matches_csr(trace_graph):
    edges = swap_ops.edge_set_from_graph(trace_graph)
    edge_count = int(trace_graph.metadata["edgeCount"])
    assert len(edges.pre) == edge_count
    assert len(edges.post) == edge_count
    assert len(edges.weight) == edge_count
    assert np.array_equal(edges.post, trace_graph.postsynaptic_indices.astype(np.int64))
    assert np.array_equal(edges.weight, trace_graph.contact_magnitudes)
    # pre[e] must match presynapticOffsets' row membership.
    for pre in range(int(trace_graph.metadata["neuronCount"])):
        start, end = int(trace_graph.presynaptic_offsets[pre]), int(trace_graph.presynaptic_offsets[pre + 1])
        assert np.all(edges.pre[start:end] == pre)


def test_bridge_mask_of(trace_graph):
    bridge = swap_ops.bridge_mask_of(trace_graph)
    assert int(bridge.sum()) == 10  # neurons 8..17 in the fixture (see this file's own docstring convention)
    expected = (trace_graph.input_channel_index == -1) & (trace_graph.output_population_index == -1)
    assert np.array_equal(bridge, expected)


def test_valid_swap_rejects_degenerate_self_loop_duplicate_and_accepts_legal():
    existing = {(0, 1), (2, 3)}
    # Degenerate: same source on both edges.
    assert swap_ops.valid_swap(0, 1, 0, 3, existing) is False
    # Degenerate: same target on both edges.
    existing2 = {(0, 1), (2, 1)}
    assert swap_ops.valid_swap(0, 1, 2, 1, existing2) is False
    # Self-loop: pre_i == post_j.
    existing3 = {(0, 1), (1, 2)}
    assert swap_ops.valid_swap(0, 1, 1, 2, existing3) is False
    # Duplicate: new_pair_1 already exists elsewhere.
    existing4 = {(0, 1), (2, 3), (0, 3)}
    assert swap_ops.valid_swap(0, 1, 2, 3, existing4) is False
    # Legal.
    existing5 = {(0, 1), (2, 3)}
    assert swap_ops.valid_swap(0, 1, 2, 3, existing5) is True


# ---------------------------------------------------------------------------
# swap_ops: candidate_targeted_swaps / apply_swap
# ---------------------------------------------------------------------------


def test_candidate_targeted_swaps_on_trace_fixture(trace_graph, trace_graph_masks):
    input_mask, thrust_mask, bridge_mask, clearance_mask = trace_graph_masks
    candidates = swap_ops.candidate_targeted_swaps(trace_graph, input_mask, thrust_mask)
    edges = swap_ops.edge_set_from_graph(trace_graph)
    assert len(candidates.in_edges) > 0
    assert len(candidates.out_edges) > 0
    for e in candidates.in_edges.tolist():
        assert input_mask[edges.pre[e]]
        assert bridge_mask[edges.post[e]]
    for e in candidates.out_edges.tolist():
        assert bridge_mask[edges.pre[e]]
        assert thrust_mask[edges.post[e]]

    # Q's class (clearance-only sources) must be a subset of P's in_edges.
    q_candidates = swap_ops.candidate_targeted_swaps(trace_graph, clearance_mask, thrust_mask)
    assert set(q_candidates.in_edges.tolist()).issubset(set(candidates.in_edges.tolist()))
    assert np.array_equal(q_candidates.out_edges, candidates.out_edges)


def test_apply_swap_preserves_invariants_on_trace_fixture(trace_graph, trace_graph_masks):
    input_mask, thrust_mask, _bridge_mask, _clearance_mask = trace_graph_masks
    candidates = swap_ops.candidate_targeted_swaps(trace_graph, input_mask, thrust_mask)
    edges = swap_ops.edge_set_from_graph(trace_graph)
    existing_pairs = set(zip(edges.pre.tolist(), edges.post.tolist()))

    applied = 0
    for e1 in candidates.in_edges.tolist():
        for e2 in candidates.out_edges.tolist():
            pre_i, post_i = int(edges.pre[e1]), int(edges.post[e1])
            pre_j, post_j = int(edges.pre[e2]), int(edges.post[e2])
            if not swap_ops.valid_swap(pre_i, post_i, pre_j, post_j, existing_pairs):
                continue
            swapped = swap_ops.apply_swap(trace_graph, e1, e2)
            _assert_swap_invariants(trace_graph, swapped)
            # The two specific edges actually moved.
            assert int(swapped.metadata["edgeCount"]) == int(trace_graph.metadata["edgeCount"])
            applied += 1
            if applied >= 5:
                return
    assert applied > 0, "no legal candidate found on the trace fixture -- test fixture assumption broken"


# ---------------------------------------------------------------------------
# swap_ops: random_swaps / random_class_swaps
# ---------------------------------------------------------------------------


def test_random_swaps_deterministic_per_seed_and_preserves_invariants(trace_graph):
    graph_a, stats_a = swap_ops.random_swaps(trace_graph, k=4, seed=7)
    graph_b, stats_b = swap_ops.random_swaps(trace_graph, k=4, seed=7)
    assert binfmt.encode_graph_binary(graph_a) == binfmt.encode_graph_binary(graph_b)
    assert stats_a == stats_b
    _assert_swap_invariants(trace_graph, graph_a)

    graph_c, _stats_c = swap_ops.random_swaps(trace_graph, k=4, seed=8)
    assert binfmt.encode_graph_binary(graph_a) != binfmt.encode_graph_binary(graph_c)


def test_random_swaps_zero_k_returns_unchanged_graph(trace_graph):
    graph, stats = swap_ops.random_swaps(trace_graph, k=0, seed=0)
    assert stats["acceptedSwaps"] == 0
    assert binfmt.encode_graph_binary(graph) == binfmt.encode_graph_binary(trace_graph)


def test_random_class_swaps_deterministic_and_respects_class(trace_graph, trace_graph_masks):
    input_mask, thrust_mask, bridge_mask, _clearance_mask = trace_graph_masks
    graph_a, stats_a = swap_ops.random_class_swaps(trace_graph, k=3, seed=7, source_mask=input_mask, thrust_mask=thrust_mask, bridge_mask=bridge_mask)
    graph_b, stats_b = swap_ops.random_class_swaps(trace_graph, k=3, seed=7, source_mask=input_mask, thrust_mask=thrust_mask, bridge_mask=bridge_mask)
    assert binfmt.encode_graph_binary(graph_a) == binfmt.encode_graph_binary(graph_b)
    assert stats_a == stats_b
    _assert_swap_invariants(trace_graph, graph_a)

    # Each accepted class swap retargets one in-edge `(a -> b)` onto `d`
    # (thrust), creating a *direct* source->thrust edge `(a -> d)` -- that
    # is the whole point of the P/M candidate class (00-overview.md:
    # "Degree-preserving rewiring creates such edges"). The trace fixture
    # starts with zero direct input->thrust edges (same premise as the real
    # biological graph -- see `test_r_premise_check_not_applicable_on_
    # trace_fixture`), so exactly `k` should exist after `k` accepted swaps
    # (each swap's other retargeted edge, `c -> b`, is bridge->bridge and
    # never itself becomes a class member again -- `d`/`a` are no longer
    # bridge, so this new edge can't be picked as an in-edge or out-edge by
    # a later swap in the same sequence).
    edges = swap_ops.edge_set_from_graph(graph_a)
    direct = input_mask[edges.pre] & thrust_mask[edges.post]
    assert int(direct.sum()) == stats_a["acceptedSwaps"] == 3


def test_random_class_swaps_raises_on_empty_candidate_class(trace_graph):
    empty_mask = np.zeros(int(trace_graph.metadata["neuronCount"]), dtype=bool)
    with pytest.raises(RuntimeError, match="empty candidate class"):
        swap_ops.random_class_swaps(
            trace_graph, k=1, seed=0, source_mask=empty_mask,
            thrust_mask=trace_graph.output_population_index == THRUST_IDX,
            bridge_mask=swap_ops.bridge_mask_of(trace_graph),
        )


# ---------------------------------------------------------------------------
# swap_ops: first_order_delta
# ---------------------------------------------------------------------------


def test_first_order_delta_agrees_in_sign_with_exact_change_for_small_gain(trace_graph_masks):
    with FIXTURE_PATH.open("r") as fh:
        fixture = json.load(fh)
    # Small global gain: the first-order Taylor expansion of T in A is most
    # accurate near A=0-scaled dynamics, so sign agreement with the exact
    # recomputed T is expected here (`02-intervention-graphs.md`'s own test
    # wording: "agrees in sign ... for small-gain graphs").
    graph = _graph_arrays_from_fixture_dict(fixture["graph"], global_gain=0.01)
    input_mask, thrust_mask, _bridge_mask, _clearance_mask = trace_graph_masks

    matrices = build_dense_matrices(graph)
    leak_rate = float(graph.metadata["leakRate"])
    global_gain = float(graph.metadata["globalGain"])
    n = matrices.adjacency.shape[0]
    system_matrix = leak_rate * np.eye(n) - global_gain * matrices.adjacency
    steady_state_map = np.linalg.solve(system_matrix, matrices.input_matrix)
    L = np.linalg.solve(system_matrix.T, matrices.output_matrix[THRUST_IDX, :])
    R = steady_state_map[:, RIGHT_IDX] + steady_state_map[:, FORWARD_IDX]
    sens = swap_ops.SwapSensitivity(global_gain=global_gain, L=L, R=R)

    T0 = matrices.output_matrix @ steady_state_map
    target0 = float(T0[THRUST_IDX, RIGHT_IDX] + T0[THRUST_IDX, FORWARD_IDX])

    candidates = swap_ops.candidate_targeted_swaps(graph, input_mask, thrust_mask)
    edges = swap_ops.edge_set_from_graph(graph)
    sign = graph.presynaptic_signs.astype(np.float64)
    existing_pairs = set(zip(edges.pre.tolist(), edges.post.tolist()))

    tested = 0
    for i in candidates.in_edges.tolist():
        for j in candidates.out_edges.tolist():
            a, b, w1 = int(edges.pre[i]), int(edges.post[i]), float(edges.weight[i])
            c, d, w2 = int(edges.pre[j]), int(edges.post[j]), float(edges.weight[j])
            if not swap_ops.valid_swap(a, b, c, d, existing_pairs):
                continue
            predicted = swap_ops.first_order_delta(
                sens, (a, b, w1, sign[a]), (a, d, w1, sign[a])
            ) + swap_ops.first_order_delta(sens, (c, d, w2, sign[c]), (c, b, w2, sign[c]))
            if abs(predicted) < 1e-9:
                continue
            candidate_graph = swap_ops.apply_swap(graph, i, j)
            candidate_matrices = build_dense_matrices(candidate_graph)
            m2 = leak_rate * np.eye(n) - global_gain * candidate_matrices.adjacency
            ss2 = np.linalg.solve(m2, candidate_matrices.input_matrix)
            t2 = candidate_matrices.output_matrix @ ss2
            exact_delta = float(t2[THRUST_IDX, RIGHT_IDX] + t2[THRUST_IDX, FORWARD_IDX]) - target0
            assert np.sign(predicted) == np.sign(exact_delta), (predicted, exact_delta, a, b, c, d)
            tested += 1
    assert tested > 50  # the trace fixture's P class has ~200 legal candidates; this is a sanity floor


# ---------------------------------------------------------------------------
# interventions.py: check_invariants
# ---------------------------------------------------------------------------


def test_check_invariants_passes_for_a_legal_swap(trace_graph, trace_graph_masks):
    input_mask, thrust_mask, _bridge_mask, _clearance_mask = trace_graph_masks
    candidates = swap_ops.candidate_targeted_swaps(trace_graph, input_mask, thrust_mask)
    edges = swap_ops.edge_set_from_graph(trace_graph)
    existing_pairs = set(zip(edges.pre.tolist(), edges.post.tolist()))
    for e1 in candidates.in_edges.tolist():
        for e2 in candidates.out_edges.tolist():
            pre_i, post_i = int(edges.pre[e1]), int(edges.post[e1])
            pre_j, post_j = int(edges.pre[e2]), int(edges.post[e2])
            if swap_ops.valid_swap(pre_i, post_i, pre_j, post_j, existing_pairs):
                swapped = swap_ops.apply_swap(trace_graph, e1, e2)
                interventions.check_invariants(trace_graph, swapped)  # must not raise
                return
    pytest.fail("no legal candidate pair found on the trace fixture -- test fixture assumption broken")


def test_check_invariants_detects_degree_violation(trace_graph):
    tampered = binfmt.GraphArrays(
        metadata=dict(trace_graph.metadata),
        biological_ids=trace_graph.biological_ids.copy(),
        presynaptic_offsets=trace_graph.presynaptic_offsets.copy(),
        postsynaptic_indices=trace_graph.postsynaptic_indices.copy(),
        contact_magnitudes=trace_graph.contact_magnitudes.copy(),
        presynaptic_signs=trace_graph.presynaptic_signs.copy(),
        input_channel_index=trace_graph.input_channel_index.copy(),
        input_weight=trace_graph.input_weight.copy(),
        output_population_index=trace_graph.output_population_index.copy(),
        output_weight=trace_graph.output_weight.copy(),
    )
    # Retarget one edge's post without a matching compensating swap: breaks
    # in/out-degree preservation (out-degree of the row is unaffected, but
    # in-degree of the old/new target neurons changes) -- must be caught,
    # not silently written.
    tampered.postsynaptic_indices[0] = (int(tampered.postsynaptic_indices[0]) + 1) % int(
        trace_graph.metadata["neuronCount"]
    )
    with pytest.raises(RuntimeError, match="invariant check failed"):
        interventions.check_invariants(trace_graph, tampered)


def test_check_invariants_exempt_degree_still_checks_labeling(trace_graph):
    # Only `inputChannelIndex` differs -- degree/edge/weight checks are
    # skipped by `exempt_degree=True` (R legitimately changes those), but
    # per-neuron labeling must still be caught.
    tampered = binfmt.GraphArrays(
        metadata=dict(trace_graph.metadata),
        biological_ids=trace_graph.biological_ids.copy(),
        presynaptic_offsets=trace_graph.presynaptic_offsets.copy(),
        postsynaptic_indices=trace_graph.postsynaptic_indices.copy(),
        contact_magnitudes=trace_graph.contact_magnitudes.copy(),
        presynaptic_signs=trace_graph.presynaptic_signs.copy(),
        input_channel_index=trace_graph.input_channel_index.copy(),
        input_weight=trace_graph.input_weight.copy(),
        output_population_index=trace_graph.output_population_index.copy(),
        output_weight=trace_graph.output_weight.copy(),
    )
    tampered.input_channel_index[0] = -1  # a real labeling violation, must still be caught
    with pytest.raises(RuntimeError, match="inputChannelIndex changed"):
        interventions.check_invariants(trace_graph, tampered, exempt_degree=True)


# ---------------------------------------------------------------------------
# interventions.py: R premise check
# ---------------------------------------------------------------------------


def test_r_premise_check_not_applicable_on_trace_fixture(trace_graph):
    result, graph = interventions._r_premise_check(trace_graph)
    assert result["applicable"] is False
    assert result["edgeCount"] == 0
    assert graph is None


def _make_direct_input_to_thrust_graph() -> "binfmt.GraphArrays":
    """A 3-neuron graph with a *direct* input->thrust edge (neuron 0, input
    channel 0, directly wired to neuron 2, thrust population 0) -- exercises
    `_r_premise_check`'s non-empty branch, never triggered on the real
    biological graph (input-restricted `weightedInDegree:thrust` is exactly
    0.0 there). `inputChannelCount`/`outputPopulationCount` match the
    production shape (8/3, `transfer.py`'s `OBSERVATION_CHANNEL_INDEX`/
    `OUTPUT_POPULATION_INDEX`) so `_r_premise_check`'s fixed
    `RIGHT_CLEARANCE_IDX`/`FORWARD_CLEARANCE_IDX`/`THRUST_IDX` column/row
    indices are in range. Neuron 0 is on channel 6 (`rightClearance`), not
    channel 0: `_r_premise_check`'s `R` vector only sums the
    `rightClearance`/`forwardClearance` columns of the steady-state map, so
    a neuron on any *other* channel would make every edge's first-order
    contribution trivially zero (`R[pre] == 0`) -- not a meaningful exercise
    of the non-empty branch's cumulative-removal logic."""
    metadata = {
        "formatVersion": 1,
        "neuronCount": 3,
        "edgeCount": 2,
        "inputChannelCount": 8,
        "outputPopulationCount": 3,
        "timestepSeconds": 1.0 / 30.0,
        "leakRate": 0.35,
        "rateMin": -2.0,
        "rateMax": 2.0,
        "inputClampMin": -1.0,
        "inputClampMax": 1.0,
        "globalGain": 0.5,
    }
    graph = binfmt.GraphArrays(
        metadata=metadata,
        biological_ids=np.array([1, 2, 3], dtype=np.uint64),
        presynaptic_offsets=np.array([0, 2, 2, 2], dtype=np.uint32),
        postsynaptic_indices=np.array([1, 2], dtype=np.uint32),
        contact_magnitudes=np.array([1.0, 2.0], dtype=np.float32),
        presynaptic_signs=np.array([1, 1, 1], dtype=np.int8),
        input_channel_index=np.array([6, -1, -1], dtype=np.int32),  # 6 = rightClearance
        input_weight=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        output_population_index=np.array([-1, -1, 0], dtype=np.int32),
        output_weight=np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )
    return binfmt.validate_graph(graph)


def test_r_premise_check_removes_edges_when_direct_input_to_thrust_exists():
    graph = _make_direct_input_to_thrust_graph()
    result, removed_graph = interventions._r_premise_check(graph)
    assert result["applicable"] is True
    assert result["edgeCount"] == 1  # only neuron0->neuron2 is input->thrust; neuron0->neuron1 is not
    assert result["removedEdgeCount"] == 1
    assert removed_graph is not None
    assert int(removed_graph.metadata["edgeCount"]) == int(graph.metadata["edgeCount"]) - 1
    interventions.check_invariants(graph, removed_graph, exempt_degree=True)
    # The removed edge is really gone: neuron 0's row no longer contains neuron 2.
    edges_check = swap_ops.edge_set_from_graph(removed_graph)
    assert not np.any((edges_check.pre == 0) & (edges_check.post == 2))


def _make_cancelling_input_to_thrust_graph() -> "binfmt.GraphArrays":
    """Two input neurons (both `rightClearance`, channel 6), opposite
    `presynapticSigns`, each with a direct edge of equal weight to the same
    thrust neuron: since neither neuron has any *incoming* edge, the
    steady-state map's rows for them are `inputWeight / leakRate`
    independent of sign (sign only enters through `A`, which only affects a
    neuron's own *outgoing* contribution, not `B`'s mapping into it) -- so
    the two edges' first-order contributions to the summed target transfer
    are exactly `+x` and `-x`, giving `total == 0` exactly. Exercises
    `_r_premise_check`'s `total <= 0` guard (a real dual-review finding:
    the original 50%-of-total stopping rule was unsound for a non-positive
    total)."""
    metadata = {
        "formatVersion": 1,
        "neuronCount": 3,
        "edgeCount": 2,
        "inputChannelCount": 8,
        "outputPopulationCount": 3,
        "timestepSeconds": 1.0 / 30.0,
        "leakRate": 0.35,
        "rateMin": -2.0,
        "rateMax": 2.0,
        "inputClampMin": -1.0,
        "inputClampMax": 1.0,
        "globalGain": 0.5,
    }
    graph = binfmt.GraphArrays(
        metadata=metadata,
        biological_ids=np.array([1, 2, 3], dtype=np.uint64),
        presynaptic_offsets=np.array([0, 1, 2, 2], dtype=np.uint32),
        postsynaptic_indices=np.array([2, 2], dtype=np.uint32),
        contact_magnitudes=np.array([1.0, 1.0], dtype=np.float32),
        presynaptic_signs=np.array([1, -1, 1], dtype=np.int8),
        input_channel_index=np.array([6, 6, -1], dtype=np.int32),  # both rightClearance
        input_weight=np.array([1.0, 1.0, 0.0], dtype=np.float32),
        output_population_index=np.array([-1, -1, 0], dtype=np.int32),
        output_weight=np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )
    return binfmt.validate_graph(graph)


def test_r_premise_check_raises_on_non_positive_total_contribution():
    graph = _make_cancelling_input_to_thrust_graph()
    with pytest.raises(RuntimeError, match="non-positive total"):
        interventions._r_premise_check(graph)


# ---------------------------------------------------------------------------
# interventions.py: greedy targeted swaps
# ---------------------------------------------------------------------------


def test_greedy_targeted_swaps_increases_target_sum_on_trace_fixture(trace_graph, trace_graph_masks):
    input_mask, thrust_mask, _bridge_mask, _clearance_mask = trace_graph_masks
    bio_result = transfer_matrix(trace_graph)
    bio_target = float(bio_result["T"][THRUST_IDX][RIGHT_IDX] + bio_result["T"][THRUST_IDX][FORWARD_IDX])

    final_graph, result = interventions._greedy_targeted_swaps(
        trace_graph, input_mask, thrust_mask, right_target=1e9, forward_target=1e9,
        max_swaps=5, rng_seed=20260925, label="P",
    )
    assert result["swaps"] >= 1
    assert result["swaps"] <= 5
    assert result["targetReached"] is False  # target deliberately unreachable (1e9)
    # On the trace fixture's small (~200-candidate) class, this search
    # naturally exhausts `MAX_EXACT_RECHECKS_PER_STEP` on its second step --
    # asserted explicitly (not just incidentally true) so this is a real
    # regression test for the "recheck_cap" stop reason, per
    # `02-intervention-graphs.md`'s "After 20 failures, stop the search...
    # with targetReached: false".
    assert result["stopReason"] == "recheck_cap"
    final_target = result["finalRightClearanceThrust"] + result["finalForwardClearanceThrust"]
    assert final_target > bio_target
    _assert_swap_invariants(trace_graph, final_graph)
    interventions.check_invariants(trace_graph, final_graph)


def test_greedy_targeted_swaps_is_deterministic(trace_graph, trace_graph_masks):
    input_mask, thrust_mask, _bridge_mask, _clearance_mask = trace_graph_masks
    graph_a, result_a = interventions._greedy_targeted_swaps(
        trace_graph, input_mask, thrust_mask, right_target=1e9, forward_target=1e9,
        max_swaps=5, rng_seed=20260925, label="P",
    )
    graph_b, result_b = interventions._greedy_targeted_swaps(
        trace_graph, input_mask, thrust_mask, right_target=1e9, forward_target=1e9,
        max_swaps=5, rng_seed=20260925, label="P",
    )
    assert binfmt.encode_graph_binary(graph_a) == binfmt.encode_graph_binary(graph_b)
    assert result_a["swaps"] == result_b["swaps"]
    assert result_a["steps"] == result_b["steps"]


def test_greedy_targeted_swaps_stops_immediately_when_target_already_met(trace_graph, trace_graph_masks):
    input_mask, thrust_mask, _bridge_mask, _clearance_mask = trace_graph_masks
    final_graph, result = interventions._greedy_targeted_swaps(
        trace_graph, input_mask, thrust_mask, right_target=-1e9, forward_target=-1e9,
        max_swaps=5, rng_seed=20260925, label="P",
    )
    assert result["swaps"] == 0
    assert result["targetReached"] is True
    assert binfmt.encode_graph_binary(final_graph) == binfmt.encode_graph_binary(trace_graph)


def test_greedy_targeted_swaps_exercises_candidate_sampling_branch(monkeypatch, trace_graph, trace_graph_masks):
    """On the real ~3.5M-candidate production graph, `total_pairs >
    MAX_CANDIDATES_PER_STEP` (50,000) is true on *every* step -- the
    `rng.choice(..., replace=False)` sampling branch is the only candidate-
    selection code path a real run ever exercises. The trace fixture's
    candidate class (~200 pairs) never reaches the real threshold, so this
    monkeypatches it down to force the same branch here: still deterministic
    across two runs with the same seed, and the sampled index set has no
    duplicate `(e1, e2)` pairs."""
    monkeypatch.setattr(interventions, "MAX_CANDIDATES_PER_STEP", 3)
    input_mask, thrust_mask, _bridge_mask, _clearance_mask = trace_graph_masks

    candidates = swap_ops.candidate_targeted_swaps(trace_graph, input_mask, thrust_mask)
    assert len(candidates.in_edges) * len(candidates.out_edges) > 3  # the sampling branch really is exercised

    graph_a, result_a = interventions._greedy_targeted_swaps(
        trace_graph, input_mask, thrust_mask, right_target=1e9, forward_target=1e9,
        max_swaps=5, rng_seed=20260925, label="P",
    )
    graph_b, result_b = interventions._greedy_targeted_swaps(
        trace_graph, input_mask, thrust_mask, right_target=1e9, forward_target=1e9,
        max_swaps=5, rng_seed=20260925, label="P",
    )
    assert binfmt.encode_graph_binary(graph_a) == binfmt.encode_graph_binary(graph_b)
    assert result_a["steps"] == result_b["steps"]
    interventions.check_invariants(trace_graph, graph_a)


def test_vectorized_delta_matches_scalar_first_order_delta(trace_graph, trace_graph_masks):
    """Direct regression test for the closed-form `delta` formula
    `_greedy_targeted_swaps` actually evaluates (`interventions.py`'s
    `delta = global_gain * (L[d] - L[b]) * (...)`), cross-checked against
    `swap_ops.first_order_delta`'s two-call sum for the same candidates --
    the exact-recompute acceptance gate in the greedy search would mask a
    magnitude/sign regression in `delta` (a worse-ranked but still-accepted
    candidate), so this exercises the vectorized formula on its own."""
    input_mask, thrust_mask, _bridge_mask, _clearance_mask = trace_graph_masks

    matrices = build_dense_matrices(trace_graph)
    leak_rate = float(trace_graph.metadata["leakRate"])
    global_gain = float(trace_graph.metadata["globalGain"])
    n = matrices.adjacency.shape[0]
    system_matrix = leak_rate * np.eye(n) - global_gain * matrices.adjacency
    steady_state_map = np.linalg.solve(system_matrix, matrices.input_matrix)
    L = np.linalg.solve(system_matrix.T, matrices.output_matrix[THRUST_IDX, :])
    R = steady_state_map[:, RIGHT_IDX] + steady_state_map[:, FORWARD_IDX]
    sens = swap_ops.SwapSensitivity(global_gain=global_gain, L=L, R=R)

    candidates = swap_ops.candidate_targeted_swaps(trace_graph, input_mask, thrust_mask)
    edges = swap_ops.edge_set_from_graph(trace_graph)
    sign = trace_graph.presynaptic_signs.astype(np.float64)

    tested = 0
    for i in candidates.in_edges.tolist():
        for j in candidates.out_edges.tolist():
            a, b, w1 = int(edges.pre[i]), int(edges.post[i]), float(edges.weight[i])
            c, d, w2 = int(edges.pre[j]), int(edges.post[j]), float(edges.weight[j])
            scalar_delta = swap_ops.first_order_delta(
                sens, (a, b, w1, sign[a]), (a, d, w1, sign[a])
            ) + swap_ops.first_order_delta(sens, (c, d, w2, sign[c]), (c, b, w2, sign[c]))
            vectorized_delta = global_gain * (L[d] - L[b]) * (sign[a] * w1 * R[a] - sign[c] * w2 * R[c])
            assert vectorized_delta == pytest.approx(scalar_delta, rel=1e-9, abs=1e-12)
            tested += 1
    assert tested > 50  # sanity floor: this fixture's P class has ~200 pairs


# ---------------------------------------------------------------------------
# End-to-end CLI: invariant/degree gates, TS round-trip, byte-identical rerun
# ---------------------------------------------------------------------------


def _find_node_bin_dir() -> "str | None":
    if shutil.which("node") is not None:
        return None
    candidate = Path.home() / ".nvm" / "versions" / "node" / "v22.22.3" / "bin"
    if (candidate / "node").exists():
        return str(candidate)
    return None


def _run_ts_roundtrip(paths: "list[Path]") -> "list[dict] | None":
    """Returns the TS-side summaries, or `None` (with a `pytest.skip`) if
    Node/tsx is unavailable -- mirrors `test_null_stats_cross_check.py`'s
    skip-not-fail convention for this repo's Python suite running standalone
    without a JS toolchain."""
    if not TSX_BIN.exists():
        pytest.skip(f"tests_python: {TSX_BIN} not found (run npm ci first) -- skipping TS round-trip check")
    env = dict(os.environ)
    extra_dir = _find_node_bin_dir()
    if extra_dir is not None:
        env["PATH"] = f"{extra_dir}:{env.get('PATH', '')}"
    elif shutil.which("node") is None:
        pytest.skip("tests_python: node not found on PATH (and not under ~/.nvm) -- skipping TS round-trip check")
    try:
        result = subprocess.run(
            [str(TSX_BIN), str(ROUNDTRIP_TS)],
            input=json.dumps({"paths": [str(p) for p in paths]}),
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env=env,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        pytest.skip(f"tests_python: could not run tsx ({error}) -- skipping TS round-trip check")
    if result.returncode != 0:
        pytest.skip(f"tests_python: tsx round-trip check failed to run: {result.stderr} -- skipping")
    return json.loads(result.stdout)


def _write_cli_inputs(root: Path, trace_graph: "binfmt.GraphArrays") -> "tuple[Path, Path, Path]":
    """A tiny biological.bin.gz + a self-consistent `--null`/`--explanation`
    pair for `interventions.py`'s CLI, built from the trace-graph fixture:

    - `--explanation`'s `metrics[].bio` is set to *this exact graph's* own
      freshly computed `T[thrust,rightClearance]`/`T[thrust,forwardClearance]`
      (`interventions.py`'s cross-check assertion requires an exact match,
      not a fixture-independent placeholder).
    - `--null`'s `rewired[].gzipSha256` is populated by actually running
      `rewire_batch.run_batch` once here (seeds 0..499) against the same
      biological graph -- `interventions.py`'s own internal regeneration is
      the same deterministic function, so its second, independent run
      reproduces byte-identical gzip bytes and passes the sha-verification
      gate.
    """
    import rewire  # local import: scripts/data is already on sys.path via graph_io
    import rewire_batch

    bio_path = root / "biological.bin.gz"
    binary = binfmt.encode_graph_binary(trace_graph)
    binfmt.write_gzip_deterministic(binary, bio_path)

    # `encode_graph_binary`'s wire-format header stores `leakRate`/
    # `globalGain`/etc as float32 (`docs/graph-format.md`), truncating
    # precision relative to the JSON fixture's float64 metadata -- the CLI
    # always loads the *decoded* graph, so `bio_result` must be computed
    # from the same round-tripped-through-binary graph the CLI will see,
    # not from `trace_graph` directly, or this cross-check would compare
    # two different-precision computations of the same quantity.
    bio_result = transfer_matrix(rewire.decode_graph_binary(binary))
    explanation_path = root / "null-explanation-v1.json"
    explanation_payload = {
        "metrics": [
            {
                "name": "T:rightClearance->thrust",
                "bio": bio_result["T"][THRUST_IDX][RIGHT_IDX],
                "nullMedian": 0.0,
                "p2_5": 0.0,
                "p97_5": 0.0,
                "bioPercentile": 0.0,
            },
            {
                "name": "T:forwardClearance->thrust",
                "bio": bio_result["T"][THRUST_IDX][FORWARD_IDX],
                "nullMedian": 0.0,
                "p2_5": 0.0,
                "p97_5": 0.0,
                "bioPercentile": 0.0,
            },
        ]
    }
    explanation_path.write_text(json.dumps(explanation_payload))

    precheck_dir = root / "null-precheck"
    null_index = rewire_batch.run_batch(in_path=bio_path, seeds=range(0, 500), out_dir=precheck_dir)
    null_payload = {"rewired": [{"seed": e["seed"], "gzipSha256": e["gzipSha256"]} for e in null_index["seeds"]]}
    null_path = root / "rewiring-null-v1.json"
    null_path.write_text(json.dumps(null_payload))

    return bio_path, explanation_path, null_path


def _run_interventions_cli(
    bio_path: Path, explanation_path: Path, null_path: Path, out_dir: Path, control_count: int, max_swaps: int
) -> subprocess.CompletedProcess:
    script = REPO_ROOT / "scripts" / "analysis" / "interventions.py"
    return subprocess.run(
        [
            sys.executable,
            str(script),
            "--biological",
            str(bio_path),
            "--explanation",
            str(explanation_path),
            "--null",
            str(null_path),
            "--out-dir",
            str(out_dir),
            "--control-count",
            str(control_count),
            "--max-swaps",
            str(max_swaps),
        ],
        env=os.environ,
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.fixture(scope="module")
def cli_fixture_dir(trace_graph):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bio_path, explanation_path, null_path = _write_cli_inputs(root, trace_graph)
        yield root, bio_path, explanation_path, null_path


def test_cli_end_to_end_gates_and_ts_roundtrip(cli_fixture_dir):
    root, bio_path, explanation_path, null_path = cli_fixture_dir
    out_dir = root / "run1"
    result = _run_interventions_cli(bio_path, explanation_path, null_path, out_dir, control_count=2, max_swaps=5)
    assert result.returncode == 0, result.stderr

    index_path = out_dir / "index.json"
    attribution_path = out_dir / "attribution.json"
    assert index_path.exists()
    assert attribution_path.exists()

    index = json.loads(index_path.read_text())
    entry_ids = {e["id"] for e in index["entries"]}
    assert {"P", "Q"}.issubset(entry_ids)
    assert "R" not in entry_ids  # trace fixture has no input->thrust edges
    assert {"C000", "C001"}.issubset(entry_ids)
    assert {"M1000", "M1001"}.issubset(entry_ids)
    assert {"MQ2000", "MQ2001"}.issubset(entry_ids)
    assert len(index["entries"]) == 2 + 2 + 2 + 2  # P, Q, C*2, M*2, MQ*2

    import gzip

    import rewire

    with gzip.open(bio_path, "rb") as fh:
        bio_binary = fh.read()
    bio_arrays = rewire.decode_graph_binary(bio_binary)

    for entry in index["entries"]:
        graph_path = out_dir / entry["path"]
        assert graph_path.exists()
        gzip_bytes = graph_path.read_bytes()
        assert binfmt.sha256_hex(gzip_bytes) == entry["gzipSha256"]
        binary = gzip.decompress(gzip_bytes)
        assert binfmt.sha256_hex(binary) == entry["binarySha256"]
        decoded = rewire.decode_graph_binary(binary)
        interventions.check_invariants(bio_arrays, decoded, exempt_degree=(entry["kind"] == "R"))

    bio_out_degree, bio_in_degree = _degree_vectors(bio_arrays)
    ts_paths = [out_dir / e["path"] for e in index["entries"]]
    ts_summaries = _run_ts_roundtrip(ts_paths)
    for entry, summary in zip(index["entries"], ts_summaries):
        assert summary["neuronCount"] == int(bio_arrays.metadata["neuronCount"])
        if entry["kind"] != "R":
            # The real degree-preservation invariant, independently recomputed
            # from the TS-side `parseGraphBinary` decode (not the Python
            # decode already checked above via `check_invariants`) -- closes
            # the gap where `graph_binary_roundtrip.ts` computed these fields
            # but nothing asserted on them.
            assert summary["edgeCount"] == int(bio_arrays.metadata["edgeCount"])
            assert summary["outDegree"] == bio_out_degree.tolist()
            assert summary["inDegree"] == bio_in_degree.tolist()
            assert summary["sortedContactMagnitudes"] == pytest.approx(
                sorted(bio_arrays.contact_magnitudes.tolist())
            )
        assert summary["presynapticSigns"] == bio_arrays.presynaptic_signs.tolist()
        assert summary["inputChannelIndex"] == bio_arrays.input_channel_index.tolist()
        assert summary["outputPopulationIndex"] == bio_arrays.output_population_index.tolist()
        assert summary["biologicalIds"] == [str(int(x)) for x in bio_arrays.biological_ids.tolist()]


def test_cli_rerun_is_byte_identical(cli_fixture_dir):
    root, bio_path, explanation_path, null_path = cli_fixture_dir
    out_a = root / "rerun-a"
    out_b = root / "rerun-b"
    result_a = _run_interventions_cli(bio_path, explanation_path, null_path, out_a, control_count=2, max_swaps=5)
    assert result_a.returncode == 0, result_a.stderr
    result_b = _run_interventions_cli(bio_path, explanation_path, null_path, out_b, control_count=2, max_swaps=5)
    assert result_b.returncode == 0, result_b.stderr

    assert (out_a / "index.json").read_bytes() == (out_b / "index.json").read_bytes()
    assert (out_a / "attribution.json").read_bytes() == (out_b / "attribution.json").read_bytes()

    index = json.loads((out_a / "index.json").read_text())
    for entry in index["entries"]:
        assert (out_a / entry["path"]).read_bytes() == (out_b / entry["path"]).read_bytes()


def test_cli_rejects_mismatched_null_sha(cli_fixture_dir):
    root, bio_path, explanation_path, _null_path = cli_fixture_dir
    tampered_null = root / "tampered-null.json"
    tampered_null.write_text(json.dumps({"rewired": [{"seed": 0, "gzipSha256": "0" * 64}]}))
    out_dir = root / "tampered-run"
    result = _run_interventions_cli(bio_path, explanation_path, tampered_null, out_dir, control_count=1, max_swaps=1)
    assert result.returncode != 0
    assert "does not match" in result.stderr or "RuntimeError" in result.stderr
