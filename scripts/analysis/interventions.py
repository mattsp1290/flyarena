#!/usr/bin/env python3
"""Predeclared pathway-intervention and control graphs
(`.agents/plans/pathway-interventions/02-intervention-graphs.md`, WP1).

Builds, deterministically, every graph `00-overview.md`'s predeclared
experiment needs to test whether the clearance->thrust pathway (the
0th-percentile finding in `docs/null-explanation-report.md`) is causally
supported under this model:

- **P** (primary): greedy, degree-preserving double-edge swaps that route
  input-labeled neurons onto thrust neurons through bridge collaterals,
  chosen by largest first-order increase of `T[thrust, rightClearance] +
  T[thrust, forwardClearance]`, stopping once both entries reach the null's
  25th percentile (or at `--max-swaps`).
- **Q**: the same search, restricted to `rightClearance`/`forwardClearance`
  input neurons as the swap source (channel specificity).
- **R** (premise check): the input-labeled->thrust edges already present in
  the biological graph, expected empty (see `_r_premise_check`'s doc
  comment) -- kept only to verify the premise, not as a real intervention.
- **C000-C099** / **M1000-M1099**: 100 graphs each, exactly `k = |P's
  swaps|` valid double-edge swaps, drawn uniformly from *anywhere* (C) or
  from *P's own candidate class* (M) -- `random_swaps`/`random_class_swaps`
  in `scripts/analysis/swap_ops.py`.
- **MQ2000-MQ2099**: 100 graphs, exactly `k_Q = |Q's swaps|` valid swaps
  uniformly from Q's candidate class.

Every graph is a pure edge-rewiring of the biological source (in/out-degree,
weight multiset, `presynapticSigns`, node set, and edge count preserved by
construction -- see `swap_ops.py`'s docstring); this module additionally
hard-fails (`_check_invariants`) if any written graph ever violates that,
rather than shipping a silently-broken artifact. R is exempt from the
degree/edge-count/weight-multiset checks only (it removes edges by design).

Run with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DD_IAST_ENABLED=false` and `PYTHONPATH` unset, same as every other
`scripts/analysis/` CLI (`env_guard.assert_single_threaded_blas`, checked
before `numpy` is imported).
"""

from __future__ import annotations

import argparse
import gzip
import json
import platform
import sys
from pathlib import Path

from env_guard import assert_single_threaded_blas

assert_single_threaded_blas()

import numpy as np  # noqa: E402

import graph_io  # noqa: E402
from graph_io import PUBLIC_DATA_DIR, load_verified_graph, write_canonical_json  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
import binfmt  # noqa: E402
import rewire  # noqa: E402
import rewire_batch  # noqa: E402

import swap_ops  # noqa: E402
from explain_stats import quantile_index  # noqa: E402
from transfer import OBSERVATION_CHANNEL_INDEX, OUTPUT_POPULATION_INDEX, transfer_matrix  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

#: This module's own real import-graph closure -- see `transfer.py`'s
#: `TRANSFER_SOURCE_DIR` doc comment for why (walked, not hand-maintained).
INTERVENTIONS_ENTRY = Path(__file__).resolve()
INTERVENTIONS_SOURCE_DIR = INTERVENTIONS_ENTRY.parent
INTERVENTIONS_SEARCH_DIRS: tuple[Path, ...] = (INTERVENTIONS_SOURCE_DIR, INTERVENTIONS_SOURCE_DIR.parent / "data")


def interventions_producer() -> dict:
    """See `transfer.py`'s `transfer_producer()` doc comment -- identical
    shape and rationale, this module's own real import-graph closure."""
    dependencies = graph_io.python_dependency_closure(INTERVENTIONS_ENTRY, REPO_ROOT, INTERVENTIONS_SEARCH_DIRS)
    return {
        "script": "scripts/analysis/interventions.py",
        "sourceSha256": graph_io.source_identity_sha256(REPO_ROOT, dependencies),
        "dependencies": dependencies,
        "host": {"arch": platform.machine(), "python": platform.python_version()},
    }


#: `00-overview.md`'s predeclared target metrics: the summed first-order
#: sensitivity/exact-recompute target is `T[thrust, rightClearance] +
#: T[thrust, forwardClearance]`.
RIGHT_CLEARANCE_IDX = OBSERVATION_CHANNEL_INDEX["rightClearance"]
FORWARD_CLEARANCE_IDX = OBSERVATION_CHANNEL_INDEX["forwardClearance"]
THRUST_IDX = OUTPUT_POPULATION_INDEX["thrust"]

#: The greedy search's deterministic candidate-sampling seed (predeclared,
#: `02-intervention-graphs.md`'s "Greedy targeted swaps (P and Q)" section) --
#: the *same* constant for both P and Q, as separate `np.random.Generator`
#: instances (they sample from disjoint candidate spaces).
CANDIDATE_SAMPLE_SEED = 20260925
MAX_CANDIDATES_PER_STEP = 50_000
MAX_EXACT_RECHECKS_PER_STEP = 20

#: Predeclared control seed ranges (`00-overview.md`): C (anywhere) 0..99,
#: M (P's class) 1000..1099, MQ (Q's class) 2000..2099.
C_SEED_BASE = 0
M_SEED_BASE = 1000
MQ_SEED_BASE = 2000

DEFAULT_BIOLOGICAL = PUBLIC_DATA_DIR / "malecns-arena-v1.bin.gz"
DEFAULT_NULL = PUBLIC_DATA_DIR / "rewiring-null-v1.json"
DEFAULT_EXPLANATION = PUBLIC_DATA_DIR / "null-explanation-v1.json"
DEFAULT_OUT_DIR = REPO_ROOT / "training" / "runs" / "interventions"

#: `null-explanation-v1.json`'s `metrics[].name` for the two target
#: entries -- used only for the biological-reproduction sanity cross-check
#: below (`_cross_check_against_published_explanation`), never for the
#: percentile targets themselves (those are always freshly recomputed from
#: 500 regenerated rewirings -- the published artifact carries no per-graph
#: values, see this module's `_regenerate_null_targets` doc comment).
EXPLANATION_METRIC_NAMES = {
    "rightClearanceThrust": "T:rightClearance->thrust",
    "forwardClearanceThrust": "T:forwardClearance->thrust",
}


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def _degree_vectors(graph: "binfmt.GraphArrays") -> "tuple[np.ndarray, np.ndarray]":
    """`(outDegree, inDegree)`, each length `neuronCount`. Out-degree is
    each CSR row's length (`presynapticOffsets`' consecutive difference);
    in-degree is the count of edges terminating at each node
    (`postsynapticIndices`' bincount) -- neither depends on edge order, so
    this is insensitive to `apply_swap`'s post-swap CSR re-sort."""
    out_degree = np.diff(graph.presynaptic_offsets.astype(np.int64))
    neuron_count = int(graph.metadata["neuronCount"])
    in_degree = np.bincount(graph.postsynaptic_indices.astype(np.int64), minlength=neuron_count)
    return out_degree, in_degree


def check_invariants(bio_graph: "binfmt.GraphArrays", graph: "binfmt.GraphArrays", *, exempt_degree: bool = False) -> None:
    """Hard-fails (`RuntimeError`) unless `graph` preserves, relative to
    `bio_graph`: node set (`biologicalIds`), every per-neuron input/output
    labeling and weight, and `presynapticSigns` -- always. With
    `exempt_degree=False` (every graph except R), also: edge count, the
    in-degree vector, the out-degree vector, and the contact-magnitude
    multiset (sorted comparison, since a swap only ever changes *which*
    edge a weight sits on, never the weight itself or the multiset of
    weights) -- the same invariants `rewire.py`'s docstring lists for its
    own rewired arm, since every intervention/control here is built the
    same way (a pure double-edge-swap sequence), R excepted (it removes
    edges by design, per `00-overview.md`'s predeclared "magnitude-matched
    removal")."""
    errors: list[str] = []
    if int(graph.metadata["neuronCount"]) != int(bio_graph.metadata["neuronCount"]):
        errors.append("neuronCount changed")
    if not exempt_degree:
        if int(graph.metadata["edgeCount"]) != int(bio_graph.metadata["edgeCount"]):
            errors.append("edgeCount changed")
        bio_out, bio_in = _degree_vectors(bio_graph)
        out, indeg = _degree_vectors(graph)
        if not np.array_equal(bio_out, out):
            errors.append("out-degree vector changed")
        if not np.array_equal(bio_in, indeg):
            errors.append("in-degree vector changed")
        if sorted(bio_graph.contact_magnitudes.tolist()) != sorted(graph.contact_magnitudes.tolist()):
            errors.append("contact-magnitude (weight) multiset changed")
    if not np.array_equal(bio_graph.presynaptic_signs, graph.presynaptic_signs):
        errors.append("presynapticSigns changed")
    if not np.array_equal(bio_graph.biological_ids, graph.biological_ids):
        errors.append("node set (biologicalIds) changed")
    if not np.array_equal(bio_graph.input_channel_index, graph.input_channel_index):
        errors.append("inputChannelIndex changed")
    if not np.array_equal(bio_graph.input_weight, graph.input_weight):
        errors.append("inputWeight changed")
    if not np.array_equal(bio_graph.output_population_index, graph.output_population_index):
        errors.append("outputPopulationIndex changed")
    if not np.array_equal(bio_graph.output_weight, graph.output_weight):
        errors.append("outputWeight changed")
    if errors:
        raise RuntimeError(f"interventions: invariant check failed: {'; '.join(errors)}")


# ---------------------------------------------------------------------------
# R: removal premise check
# ---------------------------------------------------------------------------


def _remove_edges(graph: "binfmt.GraphArrays", remove_indices: "set[int]") -> "binfmt.GraphArrays":
    """Build a new graph with the CSR-expanded edges at `remove_indices`
    deleted (used only by R's non-empty branch -- see `_r_premise_check`).
    Unlike every swap helper in `swap_ops.py`, this changes `edgeCount` and
    the degree vectors by design; `presynapticOffsets` is rebuilt from the
    surviving edges' `pre` row membership."""
    neuron_count = int(graph.metadata["neuronCount"])
    edges = swap_ops.edge_set_from_graph(graph)
    keep_mask = np.ones(len(edges.pre), dtype=bool)
    keep_mask[list(remove_indices)] = False  # order is not load-bearing for a boolean-mask assignment

    new_pre = edges.pre[keep_mask]
    new_post = edges.post[keep_mask]
    new_weight = edges.weight[keep_mask]
    new_edge_count = int(keep_mask.sum())

    row_counts = np.bincount(new_pre, minlength=neuron_count)
    new_offsets = np.zeros(neuron_count + 1, dtype=np.uint32)
    new_offsets[1:] = np.cumsum(row_counts).astype(np.uint32)

    order = np.lexsort((new_post, new_pre))
    sorted_post = new_post[order].astype(np.uint32)
    sorted_weight = new_weight[order]

    new_metadata = dict(graph.metadata)
    new_metadata["edgeCount"] = new_edge_count

    removed = binfmt.GraphArrays(
        metadata=new_metadata,
        biological_ids=graph.biological_ids.copy(),
        presynaptic_offsets=new_offsets,
        postsynaptic_indices=sorted_post,
        contact_magnitudes=sorted_weight,
        presynaptic_signs=graph.presynaptic_signs.copy(),
        input_channel_index=graph.input_channel_index.copy(),
        input_weight=graph.input_weight.copy(),
        output_population_index=graph.output_population_index.copy(),
        output_weight=graph.output_weight.copy(),
    )
    return binfmt.validate_graph(removed)


def _r_premise_check(bio_graph: "binfmt.GraphArrays") -> "tuple[dict, binfmt.GraphArrays | None]":
    """List the input-labeled->thrust edges in the biological graph
    (`00-overview.md`'s "Secondary — magnitude-matched removal (R)"). The
    explanation's input-restricted `weightedInDegree:thrust` is exactly
    0.0, so this is expected to find none, in which case R is reported
    `applicable: False` and no graph is written or returned (the caller
    writes no R entry into `index.json`, per the plan). If it is
    non-empty (not expected on the real biological graph -- this branch
    exists so the check is a genuine premise test, not an assumption, and
    is exercised by a synthetic fixture in `tests_python/
    test_interventions.py`), remove the smallest set of those edges
    (ranked by first-order contribution to the summed target transfer,
    largest first) carrying at least 50% of the total first-order target
    transfer, and return the resulting (degree-changing) graph."""
    input_mask = bio_graph.input_channel_index >= 0
    thrust_mask = bio_graph.output_population_index == THRUST_IDX
    edges = swap_ops.edge_set_from_graph(bio_graph)
    edge_indices = np.nonzero(input_mask[edges.pre] & thrust_mask[edges.post])[0]

    if edge_indices.size == 0:
        return (
            {
                "applicable": False,
                "edgeCount": 0,
                "reason": "no input-labeled -> thrust edges in the biological graph",
            },
            None,
        )

    matrices = graph_io.build_dense_matrices(bio_graph)
    leak_rate = float(bio_graph.metadata["leakRate"])
    global_gain = float(bio_graph.metadata["globalGain"])
    n = matrices.adjacency.shape[0]
    system_matrix = leak_rate * np.eye(n) - global_gain * matrices.adjacency
    try:
        steady_state_map = np.linalg.solve(system_matrix, matrices.input_matrix)
        L = np.linalg.solve(system_matrix.T, matrices.output_matrix[THRUST_IDX, :])
    except np.linalg.LinAlgError as error:
        # Matches `transfer.py`'s own `_compute_transfer` policy (flag and
        # fail loudly rather than silently propagate a NaN/inf-laced
        # solution -- see that module's `_finite_or_none` doc comment for
        # the two dual-review rounds that established this convention).
        # `main()` already verifies `bio_graph` itself is non-singular
        # (`bio_transfer["singular"]`) before ever calling this function, so
        # this can only fire when `_r_premise_check` is called directly
        # against an untested graph.
        raise RuntimeError(f"interventions: R premise check's biological graph is singular: {error}") from error
    R = steady_state_map[:, RIGHT_CLEARANCE_IDX] + steady_state_map[:, FORWARD_CLEARANCE_IDX]
    sign = bio_graph.presynaptic_signs.astype(np.float64)

    contributions: list[tuple[int, float]] = []
    for e in edge_indices.tolist():
        pre = int(edges.pre[e])
        post = int(edges.post[e])
        weight = float(edges.weight[e])
        contribution = sign[pre] * weight * global_gain * L[post] * R[pre]
        contributions.append((e, contribution))
    total = sum(c for _, c in contributions)
    # "At least 50% of the total first-order target transfer" is only a
    # well-defined stopping rule when `total > 0`: for `total == 0`
    # (contributions cancel), `cumulative >= 0.5 * total == 0` is satisfied
    # by the very first (possibly tiny) contribution only if that
    # contribution alone is already >= 0, which is not "the smallest set
    # carrying >= 50%" in any meaningful sense, and for `total < 0` the
    # comparison's sense flips (a highly negative `0.5 * total` makes the
    # loop stop almost immediately). Both are real dual-review findings
    # (round 1) on a branch this study's real biological graph never
    # reaches (it has zero input-labeled->thrust edges) -- raising loudly
    # here means a future graph that *does* reach this branch gets an
    # explicit policy decision instead of a silently wrong edge set.
    if total <= 0:
        raise RuntimeError(
            f"interventions: R premise check's total first-order target contribution is {total!r} "
            "(<= 0); the 'remove edges carrying >= 50% of the total' rule is not well-defined for a "
            "non-positive total and needs an explicit policy decision"
        )
    contributions.sort(key=lambda item: -item[1])

    remove_indices: set[int] = set()
    cumulative = 0.0
    for e, c in contributions:
        remove_indices.add(e)
        cumulative += c
        if cumulative >= 0.5 * total:
            break

    removed_graph = _remove_edges(bio_graph, remove_indices)
    result = {
        "applicable": True,
        "edgeCount": int(edge_indices.size),
        "removedEdgeCount": len(remove_indices),
        "removedFraction": len(remove_indices) / int(edge_indices.size),
        "cumulativeContributionFraction": (cumulative / total) if total != 0 else None,
        "degreeChanging": True,
    }
    return result, removed_graph


# ---------------------------------------------------------------------------
# Null regeneration: 500 rewirings -> 25th-percentile targets
# ---------------------------------------------------------------------------


def _regenerate_null_targets(
    biological_path: Path,
    source_binary: bytes,
    bio_graph: "binfmt.GraphArrays",
    published_null_path: Path,
    out_dir: Path,
) -> dict:
    """`00-overview.md`'s repository findings: "`rewiring-null-v1.json`
    carries only aggregates ... WP1 therefore always regenerates the 500
    rewirings (`rewire_batch.py --seeds 0:500`) ... to get the
    25th-percentile targets." Regenerates (never reuses a cached copy),
    sha-verifies every regenerated seed's gzip bytes against the published
    artifact's `rewired[].gzipSha256` (a mismatch means this run's
    `rewire.py`/`binfmt.py`/numpy is not the same code+environment that
    produced the published null, and the percentile targets below would be
    silently wrong), then computes the exact `transfer_matrix` for each of
    the 500 regenerated graphs to get the two target entries' empirical
    25th percentile (`explain_stats.quantile_index`, the same low-tail-floor
    convention `null-stats.ts`/`explain.py` already use elsewhere in this
    study)."""
    with published_null_path.open("r") as fh:
        published_null = json.load(fh)
    published_by_seed = {int(entry["seed"]): entry for entry in published_null["rewired"]}

    null_index = rewire_batch.run_batch(
        in_path=biological_path,
        seeds=range(0, 500),
        out_dir=out_dir,
        preloaded_source=(source_binary, bio_graph),
    )

    mismatches: list[str] = []
    for entry in null_index["seeds"]:
        published_entry = published_by_seed.get(entry["seed"])
        if published_entry is None:
            mismatches.append(f"seed {entry['seed']}: not present in {published_null_path}")
        elif published_entry["gzipSha256"] != entry["gzipSha256"]:
            mismatches.append(
                f"seed {entry['seed']}: gzipSha256 {entry['gzipSha256']} != published "
                f"{published_entry['gzipSha256']}"
            )
    if mismatches:
        raise RuntimeError(
            "interventions: regenerated null does not match "
            f"{published_null_path} ({len(mismatches)} mismatch(es)): " + "; ".join(mismatches[:5])
        )

    right_values: list[float] = []
    forward_values: list[float] = []
    for entry in null_index["seeds"]:
        graph = load_verified_graph(out_dir / entry["artifact"], entry["binarySha256"])
        result = transfer_matrix(graph)
        if result["singular"]:
            raise RuntimeError(f"interventions: regenerated null seed {entry['seed']} is singular")
        T = result["T"]
        right_values.append(float(T[THRUST_IDX][RIGHT_CLEARANCE_IDX]))
        forward_values.append(float(T[THRUST_IDX][FORWARD_CLEARANCE_IDX]))

    n = len(right_values)
    right_p25 = sorted(right_values)[quantile_index(n, 0.25)]
    forward_p25 = sorted(forward_values)[quantile_index(n, 0.25)]

    return {
        "seedCount": n,
        "sourceSha256": null_index["sourceSha256"],
        "rewireSourceSha256": null_index["rewireSourceSha256"],
        "shaVerifiedAgainst": str(published_null_path),
        "rightClearanceThrust": {"values": right_values, "p25": right_p25},
        "forwardClearanceThrust": {"values": forward_values, "p25": forward_p25},
    }


def _cross_check_against_published_explanation(explanation_path: Path, bio_transfer: dict) -> dict:
    """Sanity check (not one of `00-overview.md`'s invariant gates, but
    cheap and highly protective): this run's own freshly computed
    biological `T[thrust, rightClearance]`/`T[thrust, forwardClearance]`
    must match `null-explanation-v1.json`'s published `metrics[].bio` for
    those same two entries, within float64 tolerance. A mismatch means this
    module's transfer computation, or the biological graph it loaded, has
    silently diverged from the finding this whole experiment is testing --
    hard-fails rather than proceeding to build interventions against a
    target that no longer matches the premise."""
    with explanation_path.open("r") as fh:
        explanation = json.load(fh)
    metrics_by_name = {entry["name"]: entry for entry in explanation["metrics"]}

    bio_right = float(bio_transfer["T"][THRUST_IDX][RIGHT_CLEARANCE_IDX])
    bio_forward = float(bio_transfer["T"][THRUST_IDX][FORWARD_CLEARANCE_IDX])
    published_right = float(metrics_by_name[EXPLANATION_METRIC_NAMES["rightClearanceThrust"]]["bio"])
    published_forward = float(metrics_by_name[EXPLANATION_METRIC_NAMES["forwardClearanceThrust"]]["bio"])

    if not np.isclose(bio_right, published_right, rtol=1e-9, atol=1e-12):
        raise RuntimeError(
            f"interventions: biological T[thrust,rightClearance]={bio_right!r} does not match "
            f"published {published_right!r} ({explanation_path})"
        )
    if not np.isclose(bio_forward, published_forward, rtol=1e-9, atol=1e-12):
        raise RuntimeError(
            f"interventions: biological T[thrust,forwardClearance]={bio_forward!r} does not match "
            f"published {published_forward!r} ({explanation_path})"
        )
    return {
        "rightClearanceThrust": {"computed": bio_right, "published": published_right},
        "forwardClearanceThrust": {"computed": bio_forward, "published": published_forward},
    }


# ---------------------------------------------------------------------------
# Greedy targeted swaps (P, Q)
# ---------------------------------------------------------------------------


def _greedy_targeted_swaps(
    bio_graph: "binfmt.GraphArrays",
    source_mask: np.ndarray,
    thrust_mask: np.ndarray,
    right_target: float,
    forward_target: float,
    max_swaps: int,
    rng_seed: int,
    label: str,
) -> "tuple[binfmt.GraphArrays, dict]":
    """Greedy targeted-swap search shared by P (`source_mask = every
    input-labeled neuron`) and Q (`source_mask = clearance-channel input
    neurons only`), per `02-intervention-graphs.md`'s "Greedy targeted
    swaps (P and Q)" algorithm:

    Each step: recompute the first-order sensitivity (`swap_ops.
    SwapSensitivity`) from the *current* graph (a first-order approximation
    is only locally valid, and prior accepted swaps already changed the
    graph); score up to `MAX_CANDIDATES_PER_STEP` candidate pairs (sampled
    deterministically when the candidate class exceeds that, by a single
    `np.random.Generator` seeded once at `rng_seed` and advanced across
    steps -- not reseeded per step); accept the best positive-delta
    candidate whose *exact* recomputed target sum actually increases,
    trying up to `MAX_EXACT_RECHECKS_PER_STEP` candidates (in descending
    first-order-delta order) before giving up on this step. Stops when both
    target entries reach their null 25th percentile, at `max_swaps`, when no
    positive-delta candidate exists, or after `MAX_EXACT_RECHECKS_PER_STEP`
    failed exact rechecks in one step.
    """
    n = int(bio_graph.metadata["neuronCount"])
    leak_rate = float(bio_graph.metadata["leakRate"])
    global_gain = float(bio_graph.metadata["globalGain"])
    identity = np.eye(n)

    current_graph = bio_graph
    matrices = graph_io.build_dense_matrices(current_graph)
    system_matrix = leak_rate * identity - global_gain * matrices.adjacency
    try:
        steady_state_map = np.linalg.solve(system_matrix, matrices.input_matrix)
    except np.linalg.LinAlgError as error:
        # `main()` already verifies `bio_graph` is non-singular
        # (`bio_transfer["singular"]`) before calling this function, so this
        # can only fire when `_greedy_targeted_swaps` is called directly
        # against an untested graph -- matches `transfer.py`'s own
        # fail-loud-rather-than-propagate-garbage policy.
        raise RuntimeError(f"interventions: {label}'s starting graph is singular: {error}") from error
    T = matrices.output_matrix @ steady_state_map
    current_target_sum = float(T[THRUST_IDX, RIGHT_CLEARANCE_IDX] + T[THRUST_IDX, FORWARD_CLEARANCE_IDX])

    rng = np.random.default_rng(rng_seed)
    steps: list[dict] = []
    swaps_applied = 0
    target_reached = False
    stop_reason: str | None = None

    while True:
        right_val = float(T[THRUST_IDX, RIGHT_CLEARANCE_IDX])
        forward_val = float(T[THRUST_IDX, FORWARD_CLEARANCE_IDX])
        if right_val >= right_target and forward_val >= forward_target:
            target_reached = True
            stop_reason = "target_reached"
            break
        if swaps_applied >= max_swaps:
            stop_reason = "max_swaps"
            break

        system_matrix = leak_rate * identity - global_gain * matrices.adjacency
        try:
            L = np.linalg.solve(system_matrix.T, matrices.output_matrix[THRUST_IDX, :])
        except np.linalg.LinAlgError as error:
            # `current_graph` only ever becomes the accepted candidate whose
            # own `candidate_steady_state` solve (below) already succeeded,
            # so this is provably unreachable given this function's own
            # control flow -- guarded anyway for the same fail-loud policy
            # as every other solve in this module.
            raise RuntimeError(
                f"interventions: {label}'s current graph became singular after {swaps_applied} swaps: {error}"
            ) from error
        R = steady_state_map[:, RIGHT_CLEARANCE_IDX] + steady_state_map[:, FORWARD_CLEARANCE_IDX]

        candidate_class = swap_ops.candidate_targeted_swaps(current_graph, source_mask, thrust_mask)
        in_edges = candidate_class.in_edges
        out_edges = candidate_class.out_edges
        if len(in_edges) == 0 or len(out_edges) == 0:
            stop_reason = "no_candidates"
            break

        edges = swap_ops.edge_set_from_graph(current_graph)
        total_pairs = len(in_edges) * len(out_edges)
        if total_pairs > MAX_CANDIDATES_PER_STEP:
            flat_indices = rng.choice(total_pairs, size=MAX_CANDIDATES_PER_STEP, replace=False)
        else:
            flat_indices = np.arange(total_pairs)
        i_idx = flat_indices // len(out_edges)
        j_idx = flat_indices % len(out_edges)
        e1 = in_edges[i_idx]  # a -> b
        e2 = out_edges[j_idx]  # c -> d

        a = edges.pre[e1]
        b = edges.post[e1]
        w1 = edges.weight[e1].astype(np.float64)
        c = edges.pre[e2]
        d = edges.post[e2]
        w2 = edges.weight[e2].astype(np.float64)
        sign = current_graph.presynaptic_signs.astype(np.float64)

        # Closed form for the sum of two `first_order_delta` calls (removing
        # (a,b)/adding (a,d), removing (c,d)/adding (c,b)); see
        # `swap_ops.first_order_delta`'s doc comment for the derivation.
        # Vectorized here across every sampled candidate at once, rather
        # than calling `first_order_delta` in a Python loop.
        delta = global_gain * (L[d] - L[b]) * (sign[a] * w1 * R[a] - sign[c] * w2 * R[c])

        order = np.argsort(-delta)
        order = order[delta[order] > 0]
        if order.size == 0:
            stop_reason = "no_positive_candidate"
            break

        existing_pairs = set(zip(edges.pre.tolist(), edges.post.tolist()))
        accepted_this_step = False
        rechecks = 0
        for idx in order.tolist():
            if rechecks >= MAX_EXACT_RECHECKS_PER_STEP:
                stop_reason = "recheck_cap"
                break
            ei1 = int(e1[idx])
            ei2 = int(e2[idx])
            pre_i, post_i = int(edges.pre[ei1]), int(edges.post[ei1])
            pre_j, post_j = int(edges.pre[ei2]), int(edges.post[ei2])
            if not swap_ops.valid_swap(pre_i, post_i, pre_j, post_j, existing_pairs):
                continue
            rechecks += 1

            candidate_graph = swap_ops.apply_swap(current_graph, ei1, ei2)
            candidate_matrices = graph_io.build_dense_matrices(candidate_graph)
            candidate_system_matrix = leak_rate * identity - global_gain * candidate_matrices.adjacency
            try:
                candidate_steady_state = np.linalg.solve(candidate_system_matrix, candidate_matrices.input_matrix)
            except np.linalg.LinAlgError:
                # A singular candidate cannot be scored, so by definition it
                # cannot be an "improvement" -- reject it exactly like a
                # candidate whose exact target sum did not increase (this
                # still consumes one of the `MAX_EXACT_RECHECKS_PER_STEP`
                # slots: an exact recheck really was attempted and failed),
                # rather than crashing the whole run over one bad candidate
                # among up to 50,000 sampled per step.
                steps.append(
                    {
                        "step": swaps_applied,
                        "recheck": rechecks,
                        "removedEdge": {"pre": pre_i, "post": post_i},
                        "addedEdge": {"pre": pre_i, "post": post_j},
                        "removedEdge2": {"pre": pre_j, "post": post_j},
                        "addedEdge2": {"pre": pre_j, "post": post_i},
                        "predictedDelta": float(delta[idx]),
                        "singular": True,
                        "accepted": False,
                    }
                )
                continue
            candidate_T = candidate_matrices.output_matrix @ candidate_steady_state
            candidate_target_sum = float(
                candidate_T[THRUST_IDX, RIGHT_CLEARANCE_IDX] + candidate_T[THRUST_IDX, FORWARD_CLEARANCE_IDX]
            )
            improved = candidate_target_sum > current_target_sum

            steps.append(
                {
                    "step": swaps_applied,
                    "recheck": rechecks,
                    "removedEdge": {"pre": pre_i, "post": post_i},
                    "addedEdge": {"pre": pre_i, "post": post_j},
                    "removedEdge2": {"pre": pre_j, "post": post_j},
                    "addedEdge2": {"pre": pre_j, "post": post_i},
                    "predictedDelta": float(delta[idx]),
                    "exactTargetSumBefore": current_target_sum,
                    "exactTargetSumAfter": candidate_target_sum,
                    "singular": False,
                    "accepted": improved,
                }
            )
            if improved:
                current_graph = candidate_graph
                matrices = candidate_matrices
                steady_state_map = candidate_steady_state
                T = candidate_T
                current_target_sum = candidate_target_sum
                swaps_applied += 1
                accepted_this_step = True
                break

        if not accepted_this_step:
            if stop_reason is None:
                stop_reason = "candidates_exhausted"
            break

    result = {
        "kind": label,
        "swaps": swaps_applied,
        "targetReached": target_reached,
        "stopReason": stop_reason,
        "steps": steps,
        "finalRightClearanceThrust": float(T[THRUST_IDX, RIGHT_CLEARANCE_IDX]),
        "finalForwardClearanceThrust": float(T[THRUST_IDX, FORWARD_CLEARANCE_IDX]),
        "candidateSampleSeed": rng_seed,
    }
    return current_graph, result


# ---------------------------------------------------------------------------
# Graph writing + index entries
# ---------------------------------------------------------------------------


def _transfer_summary(graph: "binfmt.GraphArrays") -> dict:
    result = transfer_matrix(graph)
    if result["singular"]:
        return {"rightClearanceThrust": None, "forwardClearanceThrust": None, "full3x8": None, "singular": True}
    T = result["T"]
    return {
        "rightClearanceThrust": float(T[THRUST_IDX][RIGHT_CLEARANCE_IDX]),
        "forwardClearanceThrust": float(T[THRUST_IDX][FORWARD_CLEARANCE_IDX]),
        "full3x8": T,
        "singular": False,
    }


def _write_graph_entry(
    graphs_dir: Path,
    entry_id: str,
    kind: str,
    graph: "binfmt.GraphArrays",
    swaps: int,
    target_reached: "bool | None",
) -> dict:
    binary = binfmt.encode_graph_binary(graph)
    binary_sha256 = binfmt.sha256_hex(binary)
    path = graphs_dir / f"{entry_id}.bin.gz"
    binfmt.write_gzip_deterministic(binary, path)
    gzip_sha256 = binfmt.sha256_hex(path.read_bytes())

    return {
        "id": entry_id,
        "kind": kind,
        "path": f"graphs/{entry_id}.bin.gz",
        "gzipSha256": gzip_sha256,
        "binarySha256": binary_sha256,
        "swaps": swaps,
        "targetReached": target_reached,
        "transfer": _transfer_summary(graph),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: "list[str]") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--biological", type=Path, default=DEFAULT_BIOLOGICAL)
    parser.add_argument("--null", type=Path, default=DEFAULT_NULL)
    parser.add_argument("--explanation", type=Path, default=DEFAULT_EXPLANATION)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--control-count", type=int, default=100)
    parser.add_argument("--max-swaps", type=int, default=200)
    args = parser.parse_args(argv)
    if args.control_count < 0:
        parser.error(f"--control-count must be >= 0, got {args.control_count}")
    if args.max_swaps < 0:
        parser.error(f"--max-swaps must be >= 0, got {args.max_swaps}")
    return args


def main(argv: "list[str] | None" = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    out_dir: Path = args.out_dir
    graphs_dir = out_dir / "graphs"
    null_regen_dir = out_dir / "null-500"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    # Removed up front, before any graph is written: mirrors
    # `rewire_batch.py`'s own "a run interrupted partway through leaves the
    # directory with no index (detectable), not the previous run's stale
    # index describing files this run may have partially overwritten"
    # convention (`write_gzip_deterministic` is not an atomic write, unlike
    # `fsutil.atomic_write_text`, so a killed rerun really can leave a
    # graph's bytes not matching either run's recorded hash).
    (out_dir / "index.json").unlink(missing_ok=True)
    (out_dir / "attribution.json").unlink(missing_ok=True)

    # Mirrors `rewire_batch._load_source_graph` exactly (a private helper of
    # that module; inlined here rather than reached into, so this module
    # only depends on `rewire_batch`'s public `run_batch`).
    with gzip.open(args.biological, "rb") as fh:
        source_binary = fh.read()
    bio_graph = rewire.decode_graph_binary(source_binary)
    bio_sha256 = binfmt.sha256_hex(source_binary)

    input_mask = bio_graph.input_channel_index >= 0
    thrust_mask = bio_graph.output_population_index == THRUST_IDX
    bridge_mask = swap_ops.bridge_mask_of(bio_graph)
    clearance_channel_indices = {RIGHT_CLEARANCE_IDX, FORWARD_CLEARANCE_IDX}
    clearance_mask = np.array(
        [idx in clearance_channel_indices for idx in bio_graph.input_channel_index.tolist()], dtype=bool
    )

    print(f"interventions: biological loaded from {args.biological} ({bio_sha256[:16]}...)")

    bio_transfer = transfer_matrix(bio_graph)
    if bio_transfer["singular"]:
        raise RuntimeError("interventions: biological graph's transfer matrix is singular")
    explanation_cross_check = _cross_check_against_published_explanation(args.explanation, bio_transfer)
    print("interventions: biological T cross-check against published null-explanation-v1.json passed")

    print("interventions: regenerating 500 null rewirings for 25th-percentile targets ...")
    null_targets = _regenerate_null_targets(args.biological, source_binary, bio_graph, args.null, null_regen_dir)
    right_target = null_targets["rightClearanceThrust"]["p25"]
    forward_target = null_targets["forwardClearanceThrust"]["p25"]
    print(
        f"interventions: targets -- rightClearance->thrust p25={right_target!r}, "
        f"forwardClearance->thrust p25={forward_target!r}"
    )

    r_result, r_graph = _r_premise_check(bio_graph)
    print(f"interventions: R premise check -- {r_result}")

    print("interventions: P greedy search (input-labeled sources) ...")
    p_graph, p_result = _greedy_targeted_swaps(
        bio_graph, input_mask, thrust_mask, right_target, forward_target, args.max_swaps, CANDIDATE_SAMPLE_SEED, "P"
    )
    print(
        f"interventions: P done -- swaps={p_result['swaps']}, targetReached={p_result['targetReached']}, "
        f"stopReason={p_result['stopReason']}"
    )
    if p_result["swaps"] > args.max_swaps:
        # Structural invariant, not live `--max-swaps` validation (that
        # happens in `_parse_args`): `_greedy_targeted_swaps`'s own loop
        # already stops the instant `swaps_applied >= max_swaps`, so this
        # can only fire if a future refactor breaks that loop invariant.
        raise RuntimeError(
            f"interventions: internal invariant violated -- P applied {p_result['swaps']} swaps, "
            f"more than --max-swaps {args.max_swaps}"
        )
    bio_target_sum = float(
        bio_transfer["T"][THRUST_IDX][RIGHT_CLEARANCE_IDX] + bio_transfer["T"][THRUST_IDX][FORWARD_CLEARANCE_IDX]
    )
    p_target_sum = p_result["finalRightClearanceThrust"] + p_result["finalForwardClearanceThrust"]
    # `00-overview.md`'s WP1 gate ("P's recomputed target T entries
    # increased relative to biological") presumes P actually needed to act:
    # on the real biological graph the premise is that it starts well below
    # the null's 25th percentile, so P always applies at least one swap.
    # If the target is *already* met with zero swaps (`p_result["swaps"] ==
    # 0`, `targetReached` True at the top of the search -- not expected on
    # the real data, but possible on a small/atypical graph), there is
    # nothing P could have increased, and the gate is vacuous rather than
    # violated.
    if p_result["swaps"] > 0 and p_target_sum <= bio_target_sum:
        raise RuntimeError(
            f"interventions: P's exact target sum ({p_target_sum!r}) did not increase relative to "
            f"biological ({bio_target_sum!r})"
        )

    print("interventions: Q greedy search (clearance-channel sources only) ...")
    q_graph, q_result = _greedy_targeted_swaps(
        bio_graph, clearance_mask, thrust_mask, right_target, forward_target, args.max_swaps, CANDIDATE_SAMPLE_SEED, "Q"
    )
    print(
        f"interventions: Q done -- swaps={q_result['swaps']}, targetReached={q_result['targetReached']}, "
        f"stopReason={q_result['stopReason']}"
    )
    if q_result["swaps"] > args.max_swaps:
        raise RuntimeError(
            f"interventions: internal invariant violated -- Q applied {q_result['swaps']} swaps, "
            f"more than --max-swaps {args.max_swaps}"
        )

    k = p_result["swaps"]
    k_q = q_result["swaps"]

    check_invariants(bio_graph, p_graph)
    check_invariants(bio_graph, q_graph)
    if r_graph is not None:
        check_invariants(bio_graph, r_graph, exempt_degree=True)

    entries: list[dict] = []
    entries.append(_write_graph_entry(graphs_dir, "P", "P", p_graph, k, p_result["targetReached"]))
    entries.append(_write_graph_entry(graphs_dir, "Q", "Q", q_graph, k_q, q_result["targetReached"]))
    if r_graph is not None:
        # R removes edges rather than swapping them: its `swaps` field is
        # reused to carry `removedEdgeCount` (there is no swap count to
        # report), unlike every other entry where `swaps` really is an
        # accepted-double-edge-swap count.
        entries.append(_write_graph_entry(graphs_dir, "R", "R", r_graph, r_result["removedEdgeCount"], None))

    print(f"interventions: building {args.control_count} C (anywhere) control graphs, k={k} ...")
    for seed in range(C_SEED_BASE, C_SEED_BASE + args.control_count):
        control_graph, _stats = swap_ops.random_swaps(bio_graph, k, seed)
        check_invariants(bio_graph, control_graph)
        entry_id = f"C{seed:03d}"
        entries.append(_write_graph_entry(graphs_dir, entry_id, "C", control_graph, k, None))

    print(f"interventions: building {args.control_count} M (P's class) control graphs, k={k} ...")
    for seed in range(M_SEED_BASE, M_SEED_BASE + args.control_count):
        control_graph, _stats = swap_ops.random_class_swaps(bio_graph, k, seed, input_mask, thrust_mask, bridge_mask)
        check_invariants(bio_graph, control_graph)
        entry_id = f"M{seed}"
        entries.append(_write_graph_entry(graphs_dir, entry_id, "M", control_graph, k, None))

    print(f"interventions: building {args.control_count} MQ (Q's class) control graphs, k_Q={k_q} ...")
    for seed in range(MQ_SEED_BASE, MQ_SEED_BASE + args.control_count):
        control_graph, _stats = swap_ops.random_class_swaps(
            bio_graph, k_q, seed, clearance_mask, thrust_mask, bridge_mask
        )
        check_invariants(bio_graph, control_graph)
        entry_id = f"MQ{seed}"
        entries.append(_write_graph_entry(graphs_dir, entry_id, "MQ", control_graph, k_q, None))

    index_payload = {
        "version": 1,
        "sourceArtifact": args.biological.name,
        "sourceSha256": bio_sha256,
        "controlCount": args.control_count,
        "maxSwaps": args.max_swaps,
        "kP": k,
        "kQ": k_q,
        "entries": entries,
        "producer": interventions_producer(),
    }
    index_path = out_dir / "index.json"
    write_canonical_json(index_path, index_payload)
    print(f"interventions: wrote {index_path} ({len(entries)} entries)")

    attribution_payload = {
        "version": 1,
        "biological": {
            "path": str(args.biological),
            "sourceSha256": bio_sha256,
            "explanationCrossCheck": explanation_cross_check,
            "transfer": {
                "rightClearanceThrust": float(bio_transfer["T"][THRUST_IDX][RIGHT_CLEARANCE_IDX]),
                "forwardClearanceThrust": float(bio_transfer["T"][THRUST_IDX][FORWARD_CLEARANCE_IDX]),
                "full3x8": bio_transfer["T"],
            },
        },
        "nullRegeneration": null_targets,
        "R": r_result,
        "P": p_result,
        "Q": q_result,
        "producer": interventions_producer(),
    }
    attribution_path = out_dir / "attribution.json"
    write_canonical_json(attribution_path, attribution_payload)
    print(f"interventions: wrote {attribution_path}")


if __name__ == "__main__":
    main()
