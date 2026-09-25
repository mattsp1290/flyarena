#!/usr/bin/env python3
"""Predeclared structural graph features
(`.agents/plans/null-explanation/02-transfer-and-features.md`, WP2's
"Structural features" section), computed per graph on the dense signed
adjacency built by `scripts/analysis/graph_io.py`.

Fixed list (predeclared before any analysis ran, per the overview plan's "no
exploratory feature search" decision -- do not add/remove/reorder features
after the first real run; changing this list requires editing the plan
first):

1. For each input channel `c` (8, `src/lib/arena/sensors.ts`'s
   `OBSERVATION_CHANNELS` order) and output population `p` (3,
   `src/lib/arena/actions.ts`'s `OUTPUT_POPULATION` order): the shortest
   directed-edge-count path length from *any* `c`-input neuron to *any*
   `p`-output neuron (24 values, multi-source BFS over the unweighted
   directed edge graph -- edge direction is `pre -> post`, the same
   direction `stepModel` scatters drive along). Reported per pair plus the
   mean over the reachable pairs.
2. Count of directed *walks* (adjacency-matrix-power sense, not
   necessarily simple/self-avoiding paths -- the standard, computationally
   tractable definition for this kind of motif-count feature) of length <=3
   from any input neuron to any neuron in output population `p`, split by
   net sign (excitatory: sign product `+1`; inhibitory: sign product `-1`),
   per population (6 values = 3 populations x 2 signs). A path's sign
   product is `presynapticSigns[source] * presynapticSigns[n1] * ...` over
   every *traversed presynaptic* neuron (Dale's law: sign is a per-neuron
   property applied uniformly to every outgoing edge, so a path's net sign
   never depends on which particular edge is taken, only which neurons are
   traversed -- see `stepModel`'s `signedRate = presynapticSigns[pre] *
   presynapticRate * globalGain`).
3. Reciprocity: fraction of directed edges `(a -> b)` with `(b -> a)` also
   present (a self-loop `(a -> a)` is trivially reciprocal to itself).
4. Signed-weight balance into each output population: `sum(sign[pre] *
   magnitude)` over every edge terminating at a neuron in that population
   (3 values).
5. Motif counts: the number of unordered node pairs with edges in both
   directions ("2-cycles") and the number of ordered `(a, b, c)` triples
   (all distinct) with edges `a->b`, `b->c`, and `a->c` ("feed-forward
   triangles") -- both computed with self-loops excluded (a 3-node motif, by
   definition, needs 3 distinct nodes; see `motif_counts`'s doc comment for
   why zeroing the adjacency diagonal is sufficient to exclude them
   correctly, not just approximately).
6. Mean **input-restricted** weighted in-degree for each output population:
   the mean, over the neurons in that population, of `sum(magnitude)` over
   only the edges terminating at that neuron whose *presynaptic* neuron is
   input-labeled (`input_channel_index >= 0`, the same `input_mask`
   `_signed_path_counts` already computes for feature 2 above) -- not every
   presynaptic neuron in the graph, unsigned or otherwise.

   ADJUDICATION NOTE (owner-delegated decision, feature list frozen before
   the first production run): the plan's wording, "Mean input->output
   weighted in-degree", is ambiguous against item 4 ("Signed-weight balance
   of edges into output neurons", unqualified, i.e. *every* presynaptic
   neuron) -- a thermo-architecture review flagged this and recommended
   confirming with the plan owner before running the full batch. The plan
   owner confirmed the input-restricted reading on the plan-text grounds
   that item 6's "input->" qualifier, like items 1/2's explicit "from any
   input neuron" restriction, is meaningless unless it restricts the
   *source* of the counted edges -- item 4's parallel, unqualified "edges
   into output neurons" phrasing is what an unrestricted reading of item 6
   would instead look like. An unrestricted ("any presynaptic neuron")
   variant was also computed during adjudication for comparison; it is
   *not* a predeclared feature, must never be added as one after this
   decision, and if reported at all must be labeled exploratory, not
   part of the frozen feature list.

Total: 25 (24 pair path lengths + their mean) + 6 + 1 + 3 + 2 + 3 = 40
features, matching this study's overview's "about 66 metrics" tally (the
other 26 -- 24 transfer entries + 2 derived predictors -- come from
`transfer.py`).

Run with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DD_IAST_ENABLED=false` and `PYTHONPATH` unset (`.agents/plans/
null-explanation/05-execution-handoff.md`'s required run environment). This
module refuses to run unless the three thread-count variables above are all
`"1"` (`env_guard.assert_single_threaded_blas`, checked before `numpy` does
any work) -- `DD_IAST_ENABLED`/`PYTHONPATH` are not independently enforced
here (a dual-review finding: an earlier version of this docstring implied
they were).
"""

from __future__ import annotations

import argparse
import platform
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Mapping

from env_guard import assert_single_threaded_blas

assert_single_threaded_blas()

import numpy as np  # noqa: E402

import graph_io  # noqa: E402
from graph_io import (  # noqa: E402
    build_dense_matrices,
    disconnected_graph_arrays,
    load_verified_graph,
    write_canonical_json,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
import binfmt  # noqa: E402

#: `src/lib/arena/sensors.ts`'s `OBSERVATION_CHANNELS`, index-for-index.
OBSERVATION_CHANNELS: tuple[str, ...] = (
    "foodBearing",
    "foodDistance",
    "hazardBearing",
    "hazardDistance",
    "forwardClearance",
    "leftClearance",
    "rightClearance",
    "speed",
)

#: `src/lib/arena/actions.ts`'s `OUTPUT_POPULATION`.
OUTPUT_POPULATIONS: tuple[str, ...] = ("thrust", "yaw", "brake")

#: `features.json`'s own producer sha is hashed over the *real, walked*
#: Python import graph from this file -- see `transfer.py`'s identically-
#: shaped `TRANSFER_SOURCE_DIR` doc comment for why (a thermo-fix-
#: verification review finding's structural fix, replacing the old
#: hand-maintained `FEATURES_SOURCE_FILENAMES` flat list, which silently
#: omitted `scripts/data/rewire.py`/`binfmt.py`). The pre-adjudication
#: `--features-exploratory-unrestricted` run is a deliberate exception: it
#: is exempt from `explain.py`'s code-identity check by design (a pinned
#: historical, stale-code snapshot -- see `explain.py`'s `verify_
#: provenance` doc comment), so this producer block applies to it the same
#: as any other `features.py` output; it is `explain.py`'s policy, not this
#: module's, that decides which check to run on which input.
FEATURES_ENTRY = Path(__file__).resolve()
FEATURES_SOURCE_DIR = FEATURES_ENTRY.parent
FEATURES_SEARCH_DIRS: tuple[Path, ...] = (FEATURES_SOURCE_DIR, FEATURES_SOURCE_DIR.parent / "data")
REPO_ROOT = FEATURES_SOURCE_DIR.parents[1]


def features_producer() -> dict:
    """See `transfer.py`'s `transfer_producer()` doc comment -- identical
    shape and rationale, `features.py`'s own real import-graph closure."""
    dependencies = graph_io.python_dependency_closure(FEATURES_ENTRY, REPO_ROOT, FEATURES_SEARCH_DIRS)
    return {
        "script": "scripts/analysis/features.py",
        "sourceSha256": graph_io.source_identity_sha256(REPO_ROOT, dependencies),
        "dependencies": dependencies,
        "host": {"arch": platform.machine(), "python": platform.python_version()},
    }


def _multi_source_bfs(edge_bool: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Shortest directed-edge-count distance from *any* node in `sources`
    (a boolean mask, length `n`) to every node, following `edge_bool[post,
    pre]` in the `pre -> post` direction. Unreachable nodes get `-1`.
    `sources` with no `True` entries returns all `-1`."""
    n = edge_bool.shape[0]
    distance = np.full(n, -1, dtype=np.int64)
    queue: deque[int] = deque()
    for node in np.nonzero(sources)[0]:
        distance[node] = 0
        queue.append(int(node))
    # `edge_bool[post, pre]`: outgoing neighbors of `pre` are the `post`
    # indices with `edge_bool[post, pre]` true, i.e. `np.nonzero(edge_bool[:, pre])[0]`.
    while queue:
        pre = queue.popleft()
        for post in np.nonzero(edge_bool[:, pre])[0]:
            post_i = int(post)
            if distance[post_i] == -1:
                distance[post_i] = distance[pre] + 1
                queue.append(post_i)
    return distance


def _path_length_features(edge_bool: np.ndarray, graph: "binfmt.GraphArrays") -> dict:
    channel_index = graph.input_channel_index
    population_index = graph.output_population_index

    pair_lengths: dict[str, "int | None"] = {}
    reachable_values: list[int] = []
    for c, channel_name in enumerate(OBSERVATION_CHANNELS):
        sources = channel_index == c
        if not np.any(sources):
            for population_name in OUTPUT_POPULATIONS:
                pair_lengths[f"{channel_name}->{population_name}"] = None
            continue
        distance = _multi_source_bfs(edge_bool, sources)
        for p, population_name in enumerate(OUTPUT_POPULATIONS):
            targets = population_index == p
            target_distances = distance[targets]
            target_distances = target_distances[target_distances >= 0]
            if target_distances.size == 0:
                pair_lengths[f"{channel_name}->{population_name}"] = None
            else:
                length = int(np.min(target_distances))
                pair_lengths[f"{channel_name}->{population_name}"] = length
                reachable_values.append(length)

    mean_path_length = float(np.mean(reachable_values)) if reachable_values else None
    return {
        "pathLengths": pair_lengths,
        "meanPathLength": mean_path_length,
        "reachablePairCount": len(reachable_values),
        "totalPairCount": len(OBSERVATION_CHANNELS) * len(OUTPUT_POPULATIONS),
    }


def _signed_path_counts(edge_pos: np.ndarray, edge_neg: np.ndarray, graph: "binfmt.GraphArrays") -> dict:
    """`edge_pos[post, pre]`/`edge_neg[post, pre]`: booleans, true where an
    edge `pre -> post` exists and `presynapticSigns[pre]` is `+1`/`-1`
    respectively. Counts directed walks of length 1..3 from any input
    neuron to any neuron in each output population, split by the sign
    product of every traversed presynaptic neuron (see this module's doc
    comment).

    Only the `n_inputs` source columns (8 in production, never more than
    `neuronCount`) are ever read from `c_pos`/`c_neg` -- propagating the
    full `n x n` matrix at every step wasted almost all of the work (a
    dual-review finding: on the real biological graph, n=1008, this made
    `graph_features` ~6.4s/graph, dominated by 8 `int64` `n x n` matmuls;
    NumPy's integer matmul does not use BLAS, unlike the float64 solves in
    `transfer.py`). Propagating only the input columns turns those into
    `n x n_inputs` matmuls -- the same walk counts, in a fraction of the
    time. The earlier `@ identity` for the length-1 step was also a no-op
    full `n x n` matmul that just returned its input unchanged.
    """
    input_mask = graph.input_channel_index >= 0
    population_index = graph.output_population_index

    edge_pos_int = edge_pos.astype(np.int64)
    edge_neg_int = edge_neg.astype(np.int64)

    # c_pos_k[post, source]: # length-k walks source -> ... -> post whose
    # sign product (over every traversed presynaptic neuron) is +1, for
    # `source` ranging only over input neurons. c_neg_k: sign product -1.
    # Recursion extends by one more hop: a positive-so-far walk extended by
    # a `+` edge stays positive; extended by a `-` edge becomes negative
    # (and symmetrically for a negative-so-far walk) -- this is exactly
    # matrix multiplication by the signed *unweighted* adjacency, split into
    # its positive/negative parts.
    c_pos = edge_pos_int[:, input_mask]  # length-1: c_pos_1 = edge_pos restricted to input columns
    c_neg = edge_neg_int[:, input_mask]

    totals_pos = {name: 0 for name in OUTPUT_POPULATIONS}
    totals_neg = {name: 0 for name in OUTPUT_POPULATIONS}

    def accumulate(c_pos_k: np.ndarray, c_neg_k: np.ndarray) -> None:
        for p, population_name in enumerate(OUTPUT_POPULATIONS):
            target_mask = population_index == p
            totals_pos[population_name] += int(np.sum(c_pos_k[target_mask, :]))
            totals_neg[population_name] += int(np.sum(c_neg_k[target_mask, :]))

    accumulate(c_pos, c_neg)
    for _ in range(2):  # extend to length 2, then length 3
        next_pos = edge_pos_int @ c_pos + edge_neg_int @ c_neg
        next_neg = edge_pos_int @ c_neg + edge_neg_int @ c_pos
        c_pos, c_neg = next_pos, next_neg
        accumulate(c_pos, c_neg)

    return {
        "excitatoryPathCount": totals_pos,
        "inhibitoryPathCount": totals_neg,
    }


def _reciprocity(edge_bool: np.ndarray, edge_count: int) -> float:
    if edge_count == 0:
        return 0.0
    reciprocal = edge_bool & edge_bool.T
    return float(np.sum(reciprocal)) / edge_count


def _weight_balance(adjacency: np.ndarray, population_index: np.ndarray) -> dict:
    """`None` (not `0.0`) for a population with no neurons: `0.0` is a
    plausible real measurement ("perfectly balanced"), so it must not also
    mean "not applicable" -- matches `_weighted_in_degree`'s convention
    below (a dual-review finding; every production graph has all three
    populations, so this was not a live bug, but a trap for reuse)."""
    balance: dict[str, float | None] = {}
    for p, population_name in enumerate(OUTPUT_POPULATIONS):
        rows = population_index == p
        balance[population_name] = float(np.sum(adjacency[rows, :])) if np.any(rows) else None
    return balance


def motif_counts(edge_bool: np.ndarray) -> dict:
    """Zeroing the diagonal before counting excludes self-loops from both
    motifs *exactly*, not just approximately: for the "feed-forward
    triangle" two-step count `(E @ E)[a, c] = sum_b E[a, b] * E[b, c]`, the
    degenerate `b == a` term is `E[a, a] * E[a, c]` and the degenerate `b ==
    c` term is `E[a, c] * E[c, c]` -- both vanish once `E[a, a]` and `E[c,
    c]` are forced to `0`, for every `a`/`c`, with no separate correction
    term needed."""
    n = edge_bool.shape[0]
    edges = edge_bool.copy()
    np.fill_diagonal(edges, False)

    reciprocal = edges & edges.T
    two_cycle_count = int(np.sum(np.triu(reciprocal, k=1)))

    edges_int = edges.astype(np.int64)
    # `edges_int` is `[post, pre]` (matching `edge_bool`'s own convention
    # throughout this module), so `(edges_int @ edges_int)[x, z] = sum_y
    # edges_int[x, y] * edges_int[y, z]` counts walks `z -> y -> x`, not
    # `x -> y -> z` -- a dual-review finding: an earlier version of this
    # comment mislabeled the direction. The feed-forward-triangle count
    # itself is unaffected (correct either way): summing `two_step *
    # edges_int` over all `(x, z)` is the same total whether each term is
    # read as counting `z->y->x` (with the direct edge `edges_int[x,z]`,
    # i.e. `z->x`) or, equivalently, as counting `x->y->z` with a direct
    # edge `x->z` under the transposed reading -- the two readings visit
    # the same multiset of (source, mid, sink) triples, just labeled
    # oppositely, so the sum is transpose-invariant.
    two_step = edges_int @ edges_int
    feed_forward_triangle_count = int(np.sum(two_step * edges_int))

    return {"twoCycleCount": two_cycle_count, "feedForwardTriangleCount": feed_forward_triangle_count}


def _weighted_in_degree(adjacency: np.ndarray, population_index: np.ndarray, input_mask: np.ndarray) -> dict:
    """Input-restricted (see this module's docstring, feature 6's
    adjudication note): only edges whose presynaptic neuron is
    input-labeled (`input_mask`, `graph.input_channel_index >= 0`) count
    toward a neuron's in-degree -- `unsigned[:, input_mask]` keeps only
    those columns before summing, mirroring `_signed_path_counts`'s own
    `input_mask`-restricted column slice for the same reason (propagating
    only the input columns, not the full `n x n` matrix)."""
    unsigned = np.abs(adjacency)
    restricted = unsigned[:, input_mask]  # keep only input-labeled presynaptic columns
    in_degree = np.sum(restricted, axis=1)  # per-neuron, sum over input-labeled pre only
    result = {}
    for p, population_name in enumerate(OUTPUT_POPULATIONS):
        rows = population_index == p
        result[population_name] = float(np.mean(in_degree[rows])) if np.any(rows) else None
    return result


def graph_features(graph: "binfmt.GraphArrays") -> dict:
    matrices = build_dense_matrices(graph)
    adjacency = matrices.adjacency
    edge_bool = adjacency != 0
    edge_count = int(graph.metadata["edgeCount"])

    signs = graph.presynaptic_signs
    positive_pre = signs == 1
    edge_pos = edge_bool & positive_pre[np.newaxis, :]
    edge_neg = edge_bool & (~positive_pre)[np.newaxis, :]
    input_mask = graph.input_channel_index >= 0

    path = _path_length_features(edge_bool, graph)
    signed_paths = _signed_path_counts(edge_pos, edge_neg, graph)
    reciprocity = _reciprocity(edge_bool, edge_count)
    weight_balance = _weight_balance(adjacency, graph.output_population_index)
    motifs = motif_counts(edge_bool)
    in_degree = _weighted_in_degree(adjacency, graph.output_population_index, input_mask)

    return {
        "pathLengths": path["pathLengths"],
        "meanPathLength": path["meanPathLength"],
        "reachablePairCount": path["reachablePairCount"],
        "totalPairCount": path["totalPairCount"],
        "excitatoryPathCount": signed_paths["excitatoryPathCount"],
        "inhibitoryPathCount": signed_paths["inhibitoryPathCount"],
        "reciprocity": reciprocity,
        "weightBalance": weight_balance,
        "twoCycleCount": motifs["twoCycleCount"],
        "feedForwardTriangleCount": motifs["feedForwardTriangleCount"],
        "weightedInDegree": in_degree,
        "edgeCount": edge_count,
        "neuronCount": int(graph.metadata["neuronCount"]),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _one_graph(graph_id: str, path: Path, expected_sha256: str) -> tuple[str, dict]:
    graph = load_verified_graph(path, expected_sha256)
    return graph_id, graph_features(graph)


def _one_disconnected(graph_id: str, path: Path, expected_sha256: str) -> tuple[str, dict]:
    graph = load_verified_graph(path, expected_sha256)
    return graph_id, graph_features(disconnected_graph_arrays(graph))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    return graph_io.base_arg_parser(__doc__, out_help="combined features.json output path").parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.biological is not None and args.skip_biological:
        raise SystemExit("features: --biological and --skip-biological are mutually exclusive")
    index = graph_io.read_rewire_index(args.index, "features")

    jobs = graph_io.build_jobs(args, index, "features")
    graph_io.verify_jobs(jobs, "features")

    results: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [
            pool.submit(_one_disconnected if is_disconnected else _one_graph, graph_id, path, sha)
            for graph_id, path, sha, is_disconnected in jobs
        ]
        try:
            for future in futures:
                graph_id, result = future.result()
                results[graph_id] = result
        except BaseException:
            # See transfer.py's identical guard's doc comment: without this,
            # every already-queued graph still runs to completion before the
            # error is raised (a dual-review finding).
            pool.shutdown(wait=False, cancel_futures=True)
            raise

    out_payload = {
        "version": 1,
        "sourceGraphSha256": index["sourceSha256"],
        "rewireSourceSha256": index["rewireSourceSha256"],
        "graphs": results,
        "producer": features_producer(),
    }
    write_canonical_json(args.out, out_payload)
    print(f"features: wrote {args.out} ({len(results)} graphs)")


if __name__ == "__main__":
    main()
