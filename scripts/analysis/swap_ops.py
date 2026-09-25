"""Pure degree-preserving swap helpers for `scripts/analysis/interventions.py`
(WP1, `.agents/plans/pathway-interventions/02-intervention-graphs.md`).

This module implements the same double-edge-swap legality rules as
`scripts/data/rewire.py`'s `rewire_graph` (no self-loop, no duplicate, not
degenerate; `pre` never moves, only which edge's `post` it is paired with) --
see that module's docstring for why those rules make every invariant
(in/out-degree, weight multiset, `presynapticSigns` ownership, node set,
edge count) hold by construction. Unlike `rewire.rewire_graph`, which
rewires the *whole* graph for a fixed attempt budget, every generator here
is restricted to a specific candidate *class* of edges (the P/Q pathway
class, or "anywhere") and stops after exactly `k` *accepted* swaps, not a
fixed attempt count.

A "bridge" neuron is one that is neither input- nor output-labeled
(`inputChannelIndex == -1` and `outputPopulationIndex == -1`) -- the
collateral-endpoint restriction `00-overview.md`'s predeclared intervention
P uses to keep swapped edges away from the output populations.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_DATA_DIR = str(Path(__file__).resolve().parents[1] / "data")
if _DATA_DIR not in sys.path:
    sys.path.insert(0, _DATA_DIR)

import binfmt  # noqa: E402

#: Default cap on random-attempt loops (`random_swaps`/`random_class_swaps`),
#: as a multiple of `k`: large enough that a real candidate pool (thousands
#: of edges for "anywhere", hundreds for the P/Q class -- see
#: `interventions.py`'s repository-evidence candidate counts) never comes
#: close to exhausting it in practice, but finite so a genuinely-empty or
#: near-exhausted candidate class fails loudly instead of spinning forever.
DEFAULT_MAX_ATTEMPTS_MULTIPLIER = 1000


@dataclass(frozen=True)
class EdgeSet:
    """Flat, CSR-row-expanded `(pre, post, weight)` view of a graph's edges,
    index-aligned the same way `rewire.rewire_graph` expands them
    internally: `pre[e]` is the presynaptic neuron owning edge `e` (derived
    from `presynapticOffsets`, one row-length-repeated index per row) and
    never changes; `post[e]`/`weight[e]` are the postsynaptic neuron/contact
    magnitude for edge `e`. This is the indexing every function below
    (`candidate_targeted_swaps`, `apply_swap`, `valid_swap`, the random
    generators) operates on."""

    pre: np.ndarray  # int64, len == edgeCount
    post: np.ndarray  # int64, len == edgeCount (owned copy, safe to mutate)
    weight: np.ndarray  # float32, len == edgeCount


def edge_set_from_graph(graph: "binfmt.GraphArrays") -> EdgeSet:
    neuron_count = int(graph.metadata["neuronCount"])
    pre = np.repeat(
        np.arange(neuron_count, dtype=np.int64),
        np.diff(graph.presynaptic_offsets.astype(np.int64)),
    )
    post = graph.postsynaptic_indices.astype(np.int64).copy()
    weight = graph.contact_magnitudes.copy()
    return EdgeSet(pre=pre, post=post, weight=weight)


def bridge_mask_of(graph: "binfmt.GraphArrays") -> np.ndarray:
    """Boolean mask, length `neuronCount`: neither input- nor
    output-labeled -- the collateral-endpoint restriction shared by P, Q,
    and their matched controls M/MQ (`00-overview.md`'s primary-intervention
    definition)."""
    return (graph.input_channel_index == -1) & (graph.output_population_index == -1)


@dataclass(frozen=True)
class TargetedCandidateClass:
    """The P/Q candidate class, as edge indices into a fixed `EdgeSet`: `in_edges`
    are edges `(a -> b)` with `a` in `source_mask` and `b` a bridge neuron;
    `out_edges` are edges `(c -> d)` with `c` a bridge neuron and `d` in
    `thrust_mask`. A full swap candidate pairs one in-edge with one
    out-edge -- `len(in_edges) * len(out_edges)` such pairs (up to ~3.5M on
    the real biological graph), never materialized as a full list here;
    `interventions.py`'s greedy step samples directly from this product
    space (a flat index `i * len(out_edges) + j`, or `np.random.Generator.
    integers` on each axis independently)."""

    in_edges: np.ndarray  # int64 indices into `edges.pre`/`edges.post`
    out_edges: np.ndarray  # int64 indices into `edges.pre`/`edges.post`


def candidate_targeted_swaps(
    graph: "binfmt.GraphArrays", source_mask: np.ndarray, thrust_mask: np.ndarray
) -> TargetedCandidateClass:
    """`source_mask`/`thrust_mask` select edge `a`/`d` endpoints (for P,
    every input-labeled neuron; for Q, only the clearance-channel input
    neurons). The bridge restriction on `b`/`c` is fixed (`bridge_mask_of`),
    not parameterized, since every predeclared intervention/control in this
    study uses the same collateral-endpoint restriction."""
    edges = edge_set_from_graph(graph)
    bridge = bridge_mask_of(graph)
    in_edges = np.nonzero(source_mask[edges.pre] & bridge[edges.post])[0]
    out_edges = np.nonzero(bridge[edges.pre] & thrust_mask[edges.post])[0]
    return TargetedCandidateClass(in_edges=in_edges, out_edges=out_edges)


def valid_swap(
    pre_i: int,
    post_i: int,
    pre_j: int,
    post_j: int,
    existing_pairs: set[tuple[int, int]],
    allow_self_loops: bool = False,
) -> bool:
    """The exact legality rules `rewire.rewire_graph` applies inline to a
    candidate swap of edges `(pre_i -> post_i)` and `(pre_j -> post_j)` into
    `(pre_i -> post_j)` and `(pre_j -> post_i)`: not degenerate (same
    source, or same target, on both edges), no self-loop (unless
    `allow_self_loops`), no duplicate against `existing_pairs` (the full
    current edge set -- a candidate that already exists anywhere in the
    graph, in or out of the swap's own class, is rejected, matching the
    wire format's "no duplicate directed edge" invariant)."""
    if pre_i == pre_j or post_i == post_j:
        return False
    new_pair_1 = (pre_i, post_j)
    new_pair_2 = (pre_j, post_i)
    if not allow_self_loops and (new_pair_1[0] == new_pair_1[1] or new_pair_2[0] == new_pair_2[1]):
        return False
    if new_pair_1 in existing_pairs or new_pair_2 in existing_pairs:
        return False
    return True


def _rebuild_graph(graph: "binfmt.GraphArrays", pre: np.ndarray, post: np.ndarray, weight: np.ndarray) -> "binfmt.GraphArrays":
    """Shared CSR-rebuild tail for `apply_swap`/the random generators below:
    `pre` never moved (row membership/counts are unchanged from `graph`'s
    own `presynapticOffsets`), so only each row's postsynaptic order needs
    re-sorting after retargeting -- the exact rebuild `rewire.rewire_graph`
    performs."""
    order = np.lexsort((post, pre))
    new_post = post[order].astype(np.uint32)
    new_weight = weight[order]
    rebuilt = binfmt.GraphArrays(
        metadata=dict(graph.metadata),
        biological_ids=graph.biological_ids.copy(),
        presynaptic_offsets=graph.presynaptic_offsets.copy(),
        postsynaptic_indices=new_post,
        contact_magnitudes=new_weight,
        presynaptic_signs=graph.presynaptic_signs.copy(),
        input_channel_index=graph.input_channel_index.copy(),
        input_weight=graph.input_weight.copy(),
        output_population_index=graph.output_population_index.copy(),
        output_weight=graph.output_weight.copy(),
    )
    return binfmt.validate_graph(rebuilt)


def apply_swap(graph: "binfmt.GraphArrays", e1: int, e2: int) -> "binfmt.GraphArrays":
    """Apply one degree-preserving double-edge swap to `graph`: edges `e1`
    and `e2` (indices into the CSR-expanded flat edge list -- the same
    indexing `edge_set_from_graph`/`candidate_targeted_swaps` use) exchange
    postsynaptic targets. Does not itself check legality (callers must call
    `valid_swap` first); preserves in/out-degree, the weight multiset,
    `presynapticSigns` ownership, node set, and edge count by construction
    (see this module's docstring)."""
    edges = edge_set_from_graph(graph)
    post = edges.post
    post[e1], post[e2] = int(edges.post[e2]), int(edges.post[e1])
    return _rebuild_graph(graph, edges.pre, post, edges.weight)


def random_swaps(
    graph: "binfmt.GraphArrays",
    k: int,
    seed: int,
    max_attempts_multiplier: int = DEFAULT_MAX_ATTEMPTS_MULTIPLIER,
) -> "tuple[binfmt.GraphArrays, dict]":
    """The **C** control: exactly `k` accepted double-edge swaps, drawn
    uniformly from *anywhere* in the graph (no class restriction), seeded
    deterministically by `seed`. Unlike `rewire.rewire_graph` (a fixed
    attempt budget, whatever accepts), this stops as soon as `k` swaps have
    been accepted -- `00-overview.md`'s C/M/MQ arms are defined by an exact
    accepted-swap count, not an attempt count."""
    edges = edge_set_from_graph(graph)
    pre = edges.pre
    post = edges.post
    weight = edges.weight
    edge_count = len(pre)

    existing_pairs = set(zip(pre.tolist(), post.tolist()))
    rng = np.random.default_rng(seed)
    accepted = 0
    attempts = 0
    max_attempts = max(1, k) * max_attempts_multiplier

    while accepted < k:
        if edge_count < 2:
            raise RuntimeError(f"random_swaps: graph has fewer than 2 edges (edgeCount={edge_count})")
        if attempts >= max_attempts:
            raise RuntimeError(
                f"random_swaps: only found {accepted}/{k} valid swaps within {max_attempts} attempts (seed {seed})"
            )
        attempts += 1
        i, j = (int(x) for x in rng.integers(0, edge_count, size=2))
        if i == j:
            continue
        pre_i, post_i = int(pre[i]), int(post[i])
        pre_j, post_j = int(pre[j]), int(post[j])
        if not valid_swap(pre_i, post_i, pre_j, post_j, existing_pairs):
            continue
        existing_pairs.discard((pre_i, post_i))
        existing_pairs.discard((pre_j, post_j))
        existing_pairs.add((pre_i, post_j))
        existing_pairs.add((pre_j, post_i))
        post[i] = post_j
        post[j] = post_i
        accepted += 1

    rewired = _rebuild_graph(graph, pre, post, weight)
    stats = {"seed": seed, "acceptedSwaps": accepted, "attempts": attempts, "kind": "anywhere"}
    return rewired, stats


def random_class_swaps(
    graph: "binfmt.GraphArrays",
    k: int,
    seed: int,
    source_mask: np.ndarray,
    thrust_mask: np.ndarray,
    bridge_mask: np.ndarray,
    max_attempts_multiplier: int = DEFAULT_MAX_ATTEMPTS_MULTIPLIER,
) -> "tuple[binfmt.GraphArrays, dict]":
    """The **M**/**MQ** controls: exactly `k` accepted double-edge swaps,
    drawn *uniformly* (not greedily) from the same candidate class P/Q use
    (`source_mask` selects `a`/`c`... -- an in-edge `a -> b` with `a` in
    `source_mask`/`b` a bridge neuron, an out-edge `c -> d` with `c` a
    bridge neuron/`d` in `thrust_mask`), seeded deterministically by `seed`.

    The candidate class (`in_edges`/`out_edges`, which specific edges
    currently realize it) is recomputed after every *accepted* swap, not
    every attempt: node labels (`source_mask`/`thrust_mask`/`bridge_mask`)
    never change, but which edges satisfy the class membership does, since a
    swap retargets two edges' posts. Recomputing only on acceptance (not on
    every rejected attempt, which never changes `post`) keeps this cheap --
    a masked `np.nonzero` over the whole edge array, not a per-attempt cost."""
    edges = edge_set_from_graph(graph)
    pre = edges.pre
    post = edges.post
    weight = edges.weight

    existing_pairs = set(zip(pre.tolist(), post.tolist()))
    rng = np.random.default_rng(seed)
    accepted = 0
    attempts = 0
    max_attempts = max(1, k) * max_attempts_multiplier

    while accepted < k:
        in_edges = np.nonzero(source_mask[pre] & bridge_mask[post])[0]
        out_edges = np.nonzero(bridge_mask[pre] & thrust_mask[post])[0]
        if len(in_edges) == 0 or len(out_edges) == 0:
            raise RuntimeError(
                f"random_class_swaps: empty candidate class after {accepted}/{k} accepted swaps "
                f"(seed {seed}, in_edges={len(in_edges)}, out_edges={len(out_edges)})"
            )
        found = False
        while not found:
            if attempts >= max_attempts:
                raise RuntimeError(
                    f"random_class_swaps: only found {accepted}/{k} valid swaps within "
                    f"{max_attempts} attempts (seed {seed})"
                )
            attempts += 1
            e1 = int(in_edges[rng.integers(0, len(in_edges))])
            e2 = int(out_edges[rng.integers(0, len(out_edges))])
            pre_i, post_i = int(pre[e1]), int(post[e1])
            pre_j, post_j = int(pre[e2]), int(post[e2])
            if not valid_swap(pre_i, post_i, pre_j, post_j, existing_pairs):
                continue
            existing_pairs.discard((pre_i, post_i))
            existing_pairs.discard((pre_j, post_j))
            existing_pairs.add((pre_i, post_j))
            existing_pairs.add((pre_j, post_i))
            post[e1] = post_j
            post[e2] = post_i
            accepted += 1
            found = True

    rewired = _rebuild_graph(graph, pre, post, weight)
    stats = {"seed": seed, "acceptedSwaps": accepted, "attempts": attempts, "kind": "class"}
    return rewired, stats


@dataclass(frozen=True)
class SwapSensitivity:
    """The first-order sensitivity of the summed target transfer entries
    (`T[thrust, rightClearance] + T[thrust, forwardClearance]`) to each
    adjacency entry, factored per `00-overview.md`'s sensitivity formula
    `dT[p,c]/dA[i,j] = g * [O(lambda I - g A)^-1]_{p,i} * [(lambda I - g
    A)^-1 B]_{j,c}` (`i` = post/row index of `A`, `j` = pre/column index,
    matching `graph_io.build_dense_matrices`'s `A[post, pre]` convention),
    summed over `p = thrust` and `c` in `{rightClearance, forwardClearance}`:

        S[i, j] = g * L[i] * R[j],  where
        L = [O (lambda I - g A)^-1]_{thrust, :}  (indexed by post `i`)
        R = sum_c [(lambda I - g A)^-1 B]_{:, c}  (indexed by pre `j`)

    `interventions.py` computes `L`/`R` fresh (two dense solves) at the
    start of every greedy step, from the graph state as of that step -- a
    first-order approximation is only locally valid, and the graph has
    already changed after any prior accepted swap."""

    global_gain: float
    L: np.ndarray  # length neuronCount, indexed by post
    R: np.ndarray  # length neuronCount, indexed by pre


def first_order_delta(
    sens: SwapSensitivity,
    e_removed: "tuple[int, int, float, int]",
    e_added: "tuple[int, int, float, int]",
) -> float:
    """First-order estimate of the change in the summed target transfer
    entries from retargeting one edge: `e_removed`/`e_added` are each
    `(pre, post, weight, sign)` -- `sign` is `presynapticSigns[pre]` (Dale's
    law: the sign belongs to `pre`, so it is unchanged by which `post` that
    edge currently targets). A full swap retargets two edges; callers sum
    two calls (one per retargeted edge) for the swap's total first-order
    delta -- see `interventions.py`'s greedy candidate scoring, which
    vectorizes this same closed form (`g * (L[d] - L[b]) * (sign_a * w_ab *
    R[a] - sign_c * w_cd * R[c])`, derived by expanding two calls to this
    function algebraically) across many candidates at once rather than
    calling this scalar function in a Python loop."""
    pre_r, post_r, weight_r, sign_r = e_removed
    pre_a, post_a, weight_a, sign_a = e_added
    removed_term = sign_r * weight_r * sens.global_gain * sens.L[post_r] * sens.R[pre_r]
    added_term = sign_a * weight_a * sens.global_gain * sens.L[post_a] * sens.R[pre_a]
    return float(added_term - removed_term)
