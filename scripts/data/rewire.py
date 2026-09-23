"""Seeded, directed, degree-preserving rewiring: the "rewired" control arm.

Implements the classic directed double-edge-swap (configuration-model)
algorithm: repeatedly pick two distinct edges `(a -> b)` and `(c -> d)` and,
if legal, replace them with `(a -> d)` and `(c -> b)`. Each edge's
presynaptic ("pre") endpoint never moves -- only which edge's *target* it
gets swapped with -- which is what makes every invariant below hold by
construction rather than needing a post-hoc repair pass:

- **In-degree of every node is preserved.** A swap only exchanges which
  edge terminates at `b` vs. `d`; the total count of edges terminating at
  each node is unchanged.
- **Out-degree of every node is preserved.** `a` and `c` keep exactly the
  same outgoing edge *slots* (`a`'s edge to `b` becomes `a`'s edge to `d`);
  the count of edges leaving each node never changes, and neither does the
  CSR row structure (`presynapticOffsets` is copied through unmodified).
- **Edge-weight multiset is preserved.** A weight travels with its edge; a
  swap only changes an edge's target, never its `contactMagnitudes` value.
- **Sign ownership per presynaptic neuron is preserved.** `presynapticSigns`
  is indexed by node and is never touched; since an edge's `pre` endpoint
  never moves between edges, "which neuron owns this edge's sign" cannot
  change either.
- **Node set is preserved.** `biologicalIds`, `inputChannelIndex`,
  `inputWeight`, `outputPopulationIndex`, `outputWeight` are copied through
  unmodified.
- **Edge count is preserved.** A swap retargets two existing edges; it never
  adds or removes one.

Self-loop / duplicate policy for the rewired arm (docs/graph-format.md takes
no position on this; it is a compiler-level decision, made and enforced
here):

- **Duplicates are always disallowed** -- the wire format cannot represent
  two edges between the same ordered pair (see docs/graph-format.md's
  "Canonical row ordering and duplicate edges"), so a candidate swap whose
  new pair already exists elsewhere in the edge set is always rejected.
- **Self-loops are disallowed in the rewired arm by default**
  (`allow_self_loops=False`). The biological arm's self-loop count (if any)
  is a measured quantity -- a real autapse the source data recorded; a
  rewired null-model control has no such measurement to preserve, and
  rejecting self-loop-creating swaps is the standard convention for
  degree-preserving configuration-model rewiring. This compiler's own
  MaleCNS selection happens to retain zero self-loops already (see the
  ledger's `compileStats.selfLoopCount`), so the policy is exercised by
  `tests_python/test_compile.py`'s synthetic fixtures rather than by the
  real artifact.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import binfmt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_DATA_DIR = REPO_ROOT / "public" / "data"
ARTIFACT_NAME = "malecns-arena-v1"

#: Number of *attempted* swaps as a multiple of edge count. A standard
#: heuristic for adequate mixing of a double-edge-swap Markov chain is on
#: the order of several times the edge count; not every attempt succeeds
#: (some are rejected for creating a duplicate/self-loop/no-op), so the
#: realized accepted-swap count is reported in the stats rather than assumed.
DEFAULT_SWAP_ATTEMPTS_MULTIPLIER = 20


def decode_graph_binary(buffer: bytes) -> binfmt.GraphArrays:
    """Minimal decoder (Python-side counterpart to `parseGraphBinary`),
    sufficient for round-tripping this compiler's own output. Assumes
    `buffer` was produced by `encode_graph_binary`/is already valid; use
    `binfmt.validate_graph` on the result if that's not guaranteed."""
    if len(buffer) < binfmt.HEADER_BYTES:
        raise binfmt.InvalidGraphError(
            f"buffer is smaller than the fixed header ({len(buffer)} < {binfmt.HEADER_BYTES} bytes)"
        )
    header = binfmt._HEADER_STRUCT.unpack_from(buffer, 0)
    magic, format_version, neuron_count, edge_count, input_channel_count, output_population_count, \
        timestep_seconds, leak_rate, rate_min, rate_max, input_clamp_min, input_clamp_max, global_gain, flags = header
    if magic != binfmt.MAGIC:
        raise binfmt.InvalidGraphError(f"bad magic {magic!r}")

    layout = binfmt.compute_graph_layout(neuron_count, edge_count)
    if len(buffer) < layout.total_bytes:
        raise binfmt.InvalidGraphError(
            f"buffer is truncated ({len(buffer)} bytes, expected at least {layout.total_bytes})"
        )

    def read(section, dtype, count):
        return np.frombuffer(
            buffer, dtype=dtype, count=count, offset=section.byte_offset
        ).copy()

    metadata = {
        "formatVersion": format_version,
        "neuronCount": neuron_count,
        "edgeCount": edge_count,
        "inputChannelCount": input_channel_count,
        "outputPopulationCount": output_population_count,
        "timestepSeconds": timestep_seconds,
        "leakRate": leak_rate,
        "rateMin": rate_min,
        "rateMax": rate_max,
        "inputClampMin": input_clamp_min,
        "inputClampMax": input_clamp_max,
        "globalGain": global_gain,
    }

    graph = binfmt.GraphArrays(
        metadata=metadata,
        biological_ids=read(layout.biological_ids, np.uint64, neuron_count),
        presynaptic_offsets=read(layout.presynaptic_offsets, np.uint32, neuron_count + 1),
        postsynaptic_indices=read(layout.postsynaptic_indices, np.uint32, edge_count),
        contact_magnitudes=read(layout.contact_magnitudes, np.float32, edge_count),
        presynaptic_signs=read(layout.presynaptic_signs, np.int8, neuron_count),
        input_channel_index=read(layout.input_channel_index, np.int32, neuron_count),
        input_weight=read(layout.input_weight, np.float32, neuron_count),
        output_population_index=read(layout.output_population_index, np.int32, neuron_count),
        output_weight=read(layout.output_weight, np.float32, neuron_count),
    )
    return binfmt.validate_graph(graph)


def rewire_graph(
    graph: binfmt.GraphArrays,
    seed: int,
    allow_self_loops: bool = False,
    swap_attempts_multiplier: int = DEFAULT_SWAP_ATTEMPTS_MULTIPLIER,
) -> "tuple[binfmt.GraphArrays, dict]":
    binfmt.validate_graph(graph)
    neuron_count = int(graph.metadata["neuronCount"])
    edge_count = int(graph.metadata["edgeCount"])

    # Expand CSR rows into a flat edge list. `pre_of_edge[e]` never changes
    # for the lifetime of this function -- only `post_of_edge[e]` does.
    pre_of_edge = np.repeat(
        np.arange(neuron_count, dtype=np.int64),
        np.diff(graph.presynaptic_offsets.astype(np.int64)),
    )
    post_of_edge = graph.postsynaptic_indices.astype(np.int64).copy()
    weight_of_edge = graph.contact_magnitudes.copy()

    existing_pairs = set(zip(pre_of_edge.tolist(), post_of_edge.tolist()))
    assert len(existing_pairs) == edge_count, "input graph must not contain duplicate edges"

    rng = np.random.default_rng(seed)
    attempts = swap_attempts_multiplier * max(edge_count, 1)
    accepted = 0
    rejected_self_loop = 0
    rejected_duplicate = 0
    rejected_degenerate = 0
    rejected_same_index = 0

    for _ in range(attempts):
        if edge_count < 2:
            break
        i, j = rng.integers(0, edge_count, size=2)
        if i == j:
            rejected_same_index += 1
            continue
        pre_i, post_i = int(pre_of_edge[i]), int(post_of_edge[i])
        pre_j, post_j = int(pre_of_edge[j]), int(post_of_edge[j])

        if pre_i == pre_j or post_i == post_j:
            # Degenerate: swapping targets between two edges from the same
            # source (or into the same target) cannot change the edge set,
            # or reduces to a relabeling that is not a meaningful swap.
            rejected_degenerate += 1
            continue

        new_pair_1 = (pre_i, post_j)
        new_pair_2 = (pre_j, post_i)

        if not allow_self_loops and (new_pair_1[0] == new_pair_1[1] or new_pair_2[0] == new_pair_2[1]):
            rejected_self_loop += 1
            continue

        if new_pair_1 in existing_pairs or new_pair_2 in existing_pairs:
            rejected_duplicate += 1
            continue

        # Accept: retarget both edges, keeping each edge's `pre` and weight.
        existing_pairs.discard((pre_i, post_i))
        existing_pairs.discard((pre_j, post_j))
        existing_pairs.add(new_pair_1)
        existing_pairs.add(new_pair_2)
        post_of_edge[i] = post_j
        post_of_edge[j] = post_i
        accepted += 1

    # Rebuild CSR: pre never moved, so row membership/counts are identical
    # to the input graph's presynapticOffsets; only each row's postsynaptic
    # order needs re-sorting after retargeting.
    order = np.lexsort((post_of_edge, pre_of_edge))
    new_post = post_of_edge[order].astype(np.uint32)
    new_weight = weight_of_edge[order]

    rewired = binfmt.GraphArrays(
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
    binfmt.validate_graph(rewired)

    stats = {
        "seed": seed,
        "edgeCount": edge_count,
        "attempts": attempts,
        "acceptedSwaps": accepted,
        "rejectedSelfLoop": rejected_self_loop,
        "rejectedDuplicate": rejected_duplicate,
        "rejectedDegenerate": rejected_degenerate,
        "rejectedSameIndex": rejected_same_index,
        "allowSelfLoops": allow_self_loops,
    }
    return rewired, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-path", type=Path, default=PUBLIC_DATA_DIR / f"{ARTIFACT_NAME}.bin.gz")
    parser.add_argument("--out-dir", type=Path, default=PUBLIC_DATA_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--manifest-path", type=Path, default=None)
    args = parser.parse_args(argv)

    with gzip.open(args.in_path, "rb") as fh:
        binary = fh.read()
    graph = decode_graph_binary(binary)

    rewired, stats = rewire_graph(graph, seed=args.seed)
    print(f"rewire stats: {json.dumps(stats, indent=2)}")

    rewired_binary = binfmt.encode_graph_binary(rewired)
    rewired_sha256 = binfmt.sha256_hex(rewired_binary)

    out_path = args.out_dir / f"{ARTIFACT_NAME}-rewired-seed{args.seed}.bin.gz"
    binfmt.write_gzip_deterministic(rewired_binary, out_path)
    gzip_bytes = out_path.read_bytes()
    gzip_sha256 = binfmt.sha256_hex(gzip_bytes)

    print(f"Wrote {out_path} ({len(gzip_bytes)} bytes, {len(gzip_bytes) / 1e6:.3f} MB)")
    print(f"binary sha256: {rewired_sha256}")
    print(f"gzip sha256:   {gzip_sha256}")

    manifest_path = args.manifest_path or (args.out_dir / f"{ARTIFACT_NAME}.manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest.setdefault("rewiredArms", {})[f"seed{args.seed}"] = {
            "artifact": out_path.name,
            "binarySha256": rewired_sha256,
            "binaryBytes": len(rewired_binary),
            "gzipSha256": gzip_sha256,
            "gzipBytes": len(gzip_bytes),
            "swapStats": stats,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"Updated {manifest_path} with rewired-arm hashes")

    return 0


if __name__ == "__main__":
    sys.exit(main())
