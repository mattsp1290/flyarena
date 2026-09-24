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
6. Mean unsigned weighted in-degree for each output population: the mean,
   over the neurons in that population, of `sum(magnitude)` over every edge
   terminating at that neuron (3 values).

Total: 25 (24 pair path lengths + their mean) + 6 + 1 + 3 + 2 + 3 = 40
features, matching this study's overview's "about 66 metrics" tally (the
other 26 -- 24 transfer entries + 2 derived predictors -- come from
`transfer.py`).

Run with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DD_IAST_ENABLED=false` and `PYTHONPATH` unset; refuses to run otherwise
(`env_guard.assert_single_threaded_blas`, checked before `numpy` does any
work).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Mapping

from env_guard import assert_single_threaded_blas

assert_single_threaded_blas()

import numpy as np  # noqa: E402

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
    comment)."""
    n = edge_pos.shape[0]
    input_mask = graph.input_channel_index >= 0
    population_index = graph.output_population_index

    # c_pos_k[post, source]: # length-k walks source -> ... -> post whose
    # sign product (over every traversed presynaptic neuron) is +1.
    # c_neg_k: sign product -1. Recursion extends by one more hop: a
    # positive-so-far walk extended by a `+` edge stays positive; extended
    # by a `-` edge becomes negative (and symmetrically for a
    # negative-so-far walk) -- this is exactly matrix multiplication by the
    # signed *unweighted* adjacency, split into its positive/negative parts.
    identity = np.eye(n, dtype=np.int64)
    c_pos = edge_pos.astype(np.int64) @ identity  # length-1: c_pos_1 = edge_pos
    c_neg = edge_neg.astype(np.int64) @ identity  # length-1: c_neg_1 = edge_neg

    totals_pos = {name: 0 for name in OUTPUT_POPULATIONS}
    totals_neg = {name: 0 for name in OUTPUT_POPULATIONS}

    def accumulate(c_pos_k: np.ndarray, c_neg_k: np.ndarray) -> None:
        for p, population_name in enumerate(OUTPUT_POPULATIONS):
            target_mask = population_index == p
            totals_pos[population_name] += int(np.sum(c_pos_k[np.ix_(target_mask, input_mask)]))
            totals_neg[population_name] += int(np.sum(c_neg_k[np.ix_(target_mask, input_mask)]))

    accumulate(c_pos, c_neg)
    for _ in range(2):  # extend to length 2, then length 3
        next_pos = edge_pos.astype(np.int64) @ c_pos + edge_neg.astype(np.int64) @ c_neg
        next_neg = edge_pos.astype(np.int64) @ c_neg + edge_neg.astype(np.int64) @ c_pos
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
    balance = {}
    for p, population_name in enumerate(OUTPUT_POPULATIONS):
        rows = population_index == p
        balance[population_name] = float(np.sum(adjacency[rows, :])) if np.any(rows) else 0.0
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
    two_step = edges_int @ edges_int  # [a, c] = # of distinct b with a->b->c
    feed_forward_triangle_count = int(np.sum(two_step * edges_int))

    return {"twoCycleCount": two_cycle_count, "feedForwardTriangleCount": feed_forward_triangle_count}


def _weighted_in_degree(adjacency: np.ndarray, population_index: np.ndarray) -> dict:
    unsigned = np.abs(adjacency)
    in_degree = np.sum(unsigned, axis=1)  # per-neuron, sum over pre
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

    path = _path_length_features(edge_bool, graph)
    signed_paths = _signed_path_counts(edge_pos, edge_neg, graph)
    reciprocity = _reciprocity(edge_bool, edge_count)
    weight_balance = _weight_balance(adjacency, graph.output_population_index)
    motifs = motif_counts(edge_bool)
    in_degree = _weighted_in_degree(adjacency, graph.output_population_index)

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


def _read_rewire_index(path: Path) -> dict:
    with path.open("r") as fh:
        index = json.load(fh)
    for required in ("sourceArtifact", "sourceSha256", "rewireSourceSha256", "seeds"):
        if required not in index:
            raise ValueError(f"features: {path} is missing '{required}'")
    return index


def _one_graph(graph_id: str, path: Path, expected_sha256: str) -> tuple[str, dict]:
    graph = load_verified_graph(path, expected_sha256)
    return graph_id, graph_features(graph)


def _one_disconnected(graph_id: str, path: Path, expected_sha256: str) -> tuple[str, dict]:
    graph = load_verified_graph(path, expected_sha256)
    return graph_id, graph_features(disconnected_graph_arrays(graph))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True, help="rewire_batch.py index.json")
    parser.add_argument("--graphs-dir", type=Path, required=True, help="directory holding rewired .bin.gz files")
    parser.add_argument(
        "--biological",
        type=Path,
        default=None,
        help="biological source .bin.gz (default: skip biological/disconnected)",
    )
    parser.add_argument("--out", type=Path, required=True, help="combined features.json output path")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="process-pool workers (each pinned to single-threaded BLAS); default min(8, cpu_count)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    index = _read_rewire_index(args.index)

    jobs: list[tuple[str, Path, str, bool]] = []  # (graphId, path, expectedSha256, isDisconnected)
    if args.biological is not None:
        if not args.biological.exists():
            raise SystemExit(f"features: --biological path {args.biological} does not exist")
        jobs.append(("biological", args.biological, index["sourceSha256"], False))
        jobs.append(("disconnected", args.biological, index["sourceSha256"], True))

    seeds = sorted(index["seeds"], key=lambda entry: entry["seed"])
    for entry in seeds:
        jobs.append((f"rewired-{entry['seed']}", args.graphs_dir / entry["artifact"], entry["binarySha256"], False))

    results: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [
            pool.submit(_one_disconnected if is_disconnected else _one_graph, graph_id, path, sha)
            for graph_id, path, sha, is_disconnected in jobs
        ]
        for future in futures:
            graph_id, result = future.result()
            results[graph_id] = result

    out_payload = {
        "version": 1,
        "sourceGraphSha256": index["sourceSha256"],
        "rewireSourceSha256": index["rewireSourceSha256"],
        "graphs": results,
    }
    write_canonical_json(args.out, out_payload)
    print(f"features: wrote {args.out} ({len(results)} graphs)")


if __name__ == "__main__":
    main()
