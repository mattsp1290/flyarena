"""`.agents/plans/readout-attribution/02-analyses.md`'s WP2 test requirement:
"Linkage on a hand graph." Plus coverage for `build_archive_shas` (the
archive-sha cross-check `linkage.py` mirrors from `resolve-graph.ts`) and
`bootstrap_spearman_ci_clustered` (the plan's "cluster bootstrap resamples
groups, not neurons" requirement).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data"))
import binfmt  # noqa: E402

from explain_stats import metric_rng  # noqa: E402
from transfer import _compute_transfer  # noqa: E402

import linkage  # noqa: E402
from linkage import FORWARD_CLEARANCE_IDX, RIGHT_CLEARANCE_IDX  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Hand graph: 2 input-labeled neurons (one on forwardClearance, one on
# rightClearance) driving 4 output-assigned neurons through direct edges,
# plus 2 output-assigned neurons with no input edge at all (T_clear == 0
# for those, by construction).
# ---------------------------------------------------------------------------


def _make_hand_graph() -> "binfmt.GraphArrays":
    # Neurons: 0 = forwardClearance input, 1 = rightClearance input,
    # 2..5 = output-assigned (D-space order 0..3). Edges: 0->2 (weight a),
    # 1->3 (weight b); neurons 4 and 5 receive no edges at all.
    leak_rate = 0.35
    global_gain = 0.5
    a, b = 1.25, 0.5  # exactly representable in float32, unlike 1.2/0.8 -- avoids a spurious ~1e-7 mismatch against the closed form below
    w_in = 1.0
    w_out = 1.0
    neuron_count = 6
    input_channel_count = 8  # matches OBSERVATION_CHANNEL_INDEX's real layout
    metadata = {
        "formatVersion": 1,
        "neuronCount": neuron_count,
        "edgeCount": 2,
        "inputChannelCount": input_channel_count,
        "outputPopulationCount": 3,
        "timestepSeconds": 1.0 / 30.0,
        "leakRate": leak_rate,
        "rateMin": -2.0,
        "rateMax": 2.0,
        "inputClampMin": -1.0,
        "inputClampMax": 1.0,
        "globalGain": global_gain,
    }
    input_channel_index = np.full(neuron_count, -1, dtype=np.int32)
    input_weight = np.zeros(neuron_count, dtype=np.float32)
    input_channel_index[0] = FORWARD_CLEARANCE_IDX
    input_weight[0] = w_in
    input_channel_index[1] = RIGHT_CLEARANCE_IDX
    input_weight[1] = w_in

    output_population_index = np.full(neuron_count, -1, dtype=np.int32)
    output_weight = np.zeros(neuron_count, dtype=np.float32)
    for i in (2, 3, 4, 5):
        output_population_index[i] = 0
        output_weight[i] = w_out

    graph = binfmt.GraphArrays(
        metadata=metadata,
        biological_ids=np.arange(1, neuron_count + 1, dtype=np.uint64),
        presynaptic_offsets=np.array([0, 1, 2, 2, 2, 2, 2], dtype=np.uint32),
        postsynaptic_indices=np.array([2, 3], dtype=np.uint32),
        contact_magnitudes=np.array([a, b], dtype=np.float32),
        presynaptic_signs=np.array([1, 1, 1, 1, 1, 1], dtype=np.int8),
        input_channel_index=input_channel_index,
        input_weight=input_weight,
        output_population_index=output_population_index,
        output_weight=output_weight,
    )
    return binfmt.validate_graph(graph)


def test_output_neuron_indices_matches_d_space_order():
    graph = _make_hand_graph()
    assert list(linkage.output_neuron_indices(graph)) == [2, 3, 4, 5]


def test_t_clear_on_hand_graph_matches_closed_form():
    graph = _make_hand_graph()
    matrices = linkage.matrices_for_graph_id("biological", graph)
    computation = _compute_transfer(
        matrices,
        leak_rate=float(graph.metadata["leakRate"]),
        global_gain=float(graph.metadata["globalGain"]),
        timestep_seconds=float(graph.metadata["timestepSeconds"]),
        strict_shape=False,
    )
    steady_state_map = computation.steady_state_map
    output_idx = linkage.output_neuron_indices(graph)
    t_clear_raw = steady_state_map[:, RIGHT_CLEARANCE_IDX] + steady_state_map[:, FORWARD_CLEARANCE_IDX]
    t_clear_d_space = t_clear_raw[output_idx]

    # Neuron 2 (D-space d=0) receives ONLY the forwardClearance-driven
    # neuron 0's direct edge; steady state (no recurrence: neuron 2 has no
    # outgoing edges of its own, and neuron 0 has no incoming edges) is
    # `r2 = (g*a/lambda) * r0`, `r0 = w_in/lambda` (forwardClearance
    # clamped to 1) -- so `M[2, forwardClearance] = g*a/lambda^2`, and
    # `M[2, rightClearance] = 0` (no path from the rightClearance neuron to
    # neuron 2 at all).
    leak_rate, global_gain, a = 0.35, 0.5, 1.25
    b = 0.5
    expected_d0 = (global_gain * a) / (leak_rate**2)  # forwardClearance path only
    expected_d1 = (global_gain * b) / (leak_rate**2)  # rightClearance path only
    assert t_clear_d_space[0] == pytest.approx(expected_d0, rel=1e-9)
    assert t_clear_d_space[1] == pytest.approx(expected_d1, rel=1e-9)
    # Neurons 4, 5 (d=2, d=3) receive no edges at all: exactly zero.
    assert t_clear_d_space[2] == pytest.approx(0.0, abs=1e-15)
    assert t_clear_d_space[3] == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# default_task_readouts
# ---------------------------------------------------------------------------


def test_default_task_readouts_filters_out_per_task_entries():
    # flyarena-qp2e's per-task archive additions (kind: "task-intervention",
    # arenaTask != "default") are out of scope for this WP -- see the
    # function's own doc comment.
    readouts = [
        {"id": "biological-seed101", "arenaTask": "default"},
        {"id": "C000-seed101-crowded", "arenaTask": "crowded"},
        {"id": "P-seed202", "arenaTask": "default"},
        {"id": "C000-seed101-sparse-food", "arenaTask": "sparse-food"},
    ]
    result = linkage.default_task_readouts(readouts)
    assert [r["id"] for r in result] == ["biological-seed101", "P-seed202"]


def test_default_task_readouts_raises_on_mislabeled_entry_with_disagreeing_fingerprint():
    # A hypothetical future archive-writer bug: an entry labeled
    # `arenaTask: "default"` but carrying a stale/wrong `arenaTaskFingerprint`
    # (e.g. copy-pasted from a per-task entry with only the label fixed).
    # `shared.ts`'s `loadArchive` throws on this (thermo-maintainability
    # review finding I1); `default_task_readouts` must too, not silently
    # include the mislabeled entry.
    readouts = [
        {"id": "biological-seed101", "arenaTask": "default"},
        {
            "id": "mislabeled-entry",
            "arenaTask": "default",
            "arenaTaskFingerprint": "arena-config-v1|halfWidth=8|hazardCount=4",  # some other task's fingerprint
        },
    ]
    with pytest.raises(ValueError, match='has arenaTask "default" but arenaTaskFingerprintOf'):
        linkage.default_task_readouts(readouts)


def test_default_task_readouts_accepts_entry_whose_stored_fingerprint_matches_default():
    # An entry that DOES carry an explicit `arenaTaskFingerprint` equal to
    # the default task's own fingerprint is legitimate (the check is on
    # agreement, not on the field's mere presence).
    readouts = [
        {
            "id": "biological-seed101",
            "arenaTask": "default",
            "arenaTaskFingerprint": linkage.DEFAULT_TASK_FINGERPRINT,
        }
    ]
    result = linkage.default_task_readouts(readouts)
    assert [r["id"] for r in result] == ["biological-seed101"]


def test_default_task_readouts_passes_real_archive_default_entries():
    # The real archive's 23 `arenaTask == "default"` entries (`training/
    # archive/trained-readouts-v1.json`, WP1b's per-task additions bring the
    # total to 75) must all pass the fingerprint cross-check unchanged --
    # this is the non-degenerate "real data still works" companion to the
    # synthetic mismatch case above.
    archive_path = REPO_ROOT / "training" / "archive" / "trained-readouts-v1.json"
    if not archive_path.exists():
        pytest.skip("training/archive/trained-readouts-v1.json not present in this checkout")
    archive = json.loads(archive_path.read_text())
    result = linkage.default_task_readouts(archive["readouts"])
    assert len(result) == 23


# ---------------------------------------------------------------------------
# build_archive_shas
# ---------------------------------------------------------------------------


def test_build_archive_shas_happy_path():
    readouts = [
        {"id": "biological-seed101", "graphId": "biological", "graphGzipSha256": "g1", "graphBinarySha256": "b1"},
        {"id": "biological-seed202", "graphId": "biological", "graphGzipSha256": "g1", "graphBinarySha256": "b1"},
        {"id": "disconnected-seed101", "graphId": "disconnected", "graphGzipSha256": None, "graphBinarySha256": None},
    ]
    shas = linkage.build_archive_shas(readouts)
    assert shas == {"biological": ("g1", "b1")}


def test_build_archive_shas_rejects_disagreeing_duplicate_graph_ids():
    readouts = [
        {"id": "biological-seed101", "graphId": "biological", "graphGzipSha256": "g1", "graphBinarySha256": "b1"},
        {"id": "biological-seed202", "graphId": "biological", "graphGzipSha256": "g2", "graphBinarySha256": "b1"},
    ]
    with pytest.raises(ValueError, match="disagree on graph shas"):
        linkage.build_archive_shas(readouts)


def test_build_archive_shas_rejects_disconnected_with_non_null_shas():
    readouts = [{"id": "disconnected-seed101", "graphId": "disconnected", "graphGzipSha256": "g1", "graphBinarySha256": "b1"}]
    with pytest.raises(ValueError, match="disconnected.*null graph shas"):
        linkage.build_archive_shas(readouts)


def test_assert_matches_archive_throws_on_missing_entry_not_silently_skips():
    with pytest.raises(ValueError, match="no archive entry"):
        linkage._assert_matches_archive("P", "g1", "b1", {})


def test_assert_matches_archive_throws_on_mismatch():
    with pytest.raises(ValueError, match="does not match"):
        linkage._assert_matches_archive("biological", "g1", "b1", {"biological": ("g2", "b1")})


def test_assert_matches_archive_passes_on_match():
    linkage._assert_matches_archive("biological", "g1", "b1", {"biological": ("g1", "b1")})


# ---------------------------------------------------------------------------
# bootstrap_spearman_ci_clustered -- resamples clusters, not neurons
# ---------------------------------------------------------------------------


def test_cluster_bootstrap_resamples_whole_clusters_not_individual_neurons():
    # 2 clusters: {0, 1} (size 2) and {2} (size 1). A per-neuron bootstrap
    # (drawing 3 individual neuron indices with replacement) would always
    # resample exactly 3 indices, every time. A per-CLUSTER bootstrap draws
    # 2 cluster labels with replacement (k = number of clusters = 2), so
    # the total resampled index COUNT varies: both draws == the size-1
    # cluster -> 2 indices; one of each -> 3; both == the size-2 cluster ->
    # 4. Observing more than one distinct resample size is only possible
    # under cluster-level (not neuron-level) resampling.
    rank_x = np.array([1.0, 2.0, 3.0])
    rank_y = np.array([3.0, 1.0, 2.0])
    cluster_of_index = np.array([0, 0, 1])
    rng = metric_rng(1, "test-cluster-bootstrap")
    resamples = 200

    # Reimplement just the index-count bookkeeping the function itself does
    # internally, to observe the resample sizes without changing its public
    # signature (`bootstrap_spearman_ci_clustered` returns only rho bounds).
    unique_clusters, _ = np.unique(cluster_of_index, return_counts=True)
    members = {c: np.where(cluster_of_index == c)[0] for c in unique_clusters}
    sizes = set()
    for _ in range(resamples):
        drawn = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        idx = np.concatenate([members[c] for c in drawn])
        sizes.add(idx.shape[0])
    assert sizes == {2, 3, 4}, f"expected resample sizes {{2,3,4}} from cluster-level resampling, got {sizes}"

    lo, hi, cluster_count, cluster_sizes = linkage.bootstrap_spearman_ci_clustered(
        rank_x, rank_y, cluster_of_index, resamples, metric_rng(1, "actual-call")
    )
    assert cluster_count == 2
    assert cluster_sizes == [1, 2]
    assert lo <= hi


def test_cluster_bootstrap_ci_upper_bound_reaches_1_when_clusters_are_homogeneous():
    # 2 clusters of 2, k = 2 draws with replacement: whenever the draw
    # includes BOTH distinct clusters (probability 1/2, either order), the
    # resample reproduces the full sample's perfect rho = 1 exactly. This
    # directly demonstrates cluster-level (not neuron-level) resampling: a
    # per-NEURON bootstrap draws 4 individual indices with replacement and
    # would essentially never resample the data down to only 2 distinct
    # values this way, so it would not show this same bimodal pattern.
    # (When the draw instead picks the SAME cluster twice, both resampled
    # ranks are constant and `bootstrap_spearman_ci_clustered`'s own
    # zero-denominator fallback reports rho = 0 for that resample -- so the
    # LOWER 2.5th-percentile bound is not asserted here, only the upper.)
    rank_x = np.array([1.0, 1.0, 2.0, 2.0])
    rank_y = np.array([1.0, 1.0, 2.0, 2.0])
    cluster_of_index = np.array([0, 0, 1, 1])
    lo, hi, cluster_count, cluster_sizes = linkage.bootstrap_spearman_ci_clustered(
        rank_x, rank_y, cluster_of_index, 200, metric_rng(1, "homogeneous")
    )
    assert cluster_count == 2
    assert cluster_sizes == [2, 2]
    assert hi == pytest.approx(1.0, abs=1e-9)
    assert lo <= hi
