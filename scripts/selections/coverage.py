#!/usr/bin/env python3
"""`.agents/plans/selection-robustness/02-per-selection-chain.md` WP2, step
4: per-selection bridge coverage, as predeclared in `00-overview.md`'s
"Coverage check" bullet.

For each of the 8 predeclared input channels, counts the selected bridge
neurons with at least one edge *from* a sensory neuron assigned to that
channel. For each of the 3 predeclared output populations, counts the
selected bridge neurons with at least one edge *to* a descending neuron
assigned to that population. A channel or population with zero coverage is
flagged -- `00-overview.md`: "Any channel or population with zero coverage
is flagged, and a selection with a flagged channel is not categorized for
the pathway finding" (the categorization decision itself is
`selection-report.ts`'s job, WP3; this script only computes and records the
counts and the flag).

"Bridge neuron" is derived from the compiled graph alone, not from any
raw-data node list: `compile.py`'s `select_subgraph` builds
`node_ids = sensory_ids | descending_ids | bridge_ids` with `bridge_ids`
disjoint from both of the others by construction (`select_subgraph`'s
`bridge_candidates = ... - set(sensory_ids) - set(descending_ids)`), and
every sensory/descending neuron gets a non-negative `inputChannelIndex`/
`outputPopulationIndex` while every bridge neuron gets neither (`assign_
channels` only ever assigns those two fields to sensory/descending ids).
So "bridge" is exactly "neither input- nor output-assigned" in the compiled
binary -- the same invariant `graph_io.build_dense_matrices` already relies
on for `input_matrix`/`output_matrix`, reused here rather than threading the
raw sensory/bridge/descending id lists (which `compile.py`'s ledger/manifest
deliberately do not persist -- only their counts) through an extra CLI
input.

Run with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DD_IAST_ENABLED=false` and `PYTHONPATH` unset, matching every other
`scripts/analysis/` CLI (`env_guard.assert_single_threaded_blas`, checked
before `numpy` does any work, below).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ANALYSIS_DIR = str(Path(__file__).resolve().parents[1] / "analysis")
if _ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, _ANALYSIS_DIR)

from env_guard import assert_single_threaded_blas  # noqa: E402

assert_single_threaded_blas()

import numpy as np  # noqa: E402

import graph_io  # noqa: E402
from features import OBSERVATION_CHANNELS  # noqa: E402
from transfer import OBSERVATION_CHANNEL_INDEX, OUTPUT_POPULATION_INDEX  # noqa: E402

VERSION = 1


def compute_coverage(graph: "graph_io.binfmt.GraphArrays") -> dict:
    """Pure function of the decoded graph. Returns the artifact body (no
    `version`/`sourceGraphSha256`/`producer` -- `main` adds those)."""
    neuron_count = int(graph.metadata["neuronCount"])
    channel_index = graph.input_channel_index
    population_index = graph.output_population_index
    is_bridge = (channel_index < 0) & (population_index < 0)

    offsets = graph.presynaptic_offsets.astype(np.int64)
    post_indices = graph.postsynaptic_indices.astype(np.int64)
    if neuron_count > 0 and int(offsets[-1]) > 0:
        row_lengths = np.diff(offsets)
        pre_of_edge = np.repeat(np.arange(neuron_count, dtype=np.int64), row_lengths)
        post_of_edge = post_indices
    else:
        pre_of_edge = np.empty(0, dtype=np.int64)
        post_of_edge = np.empty(0, dtype=np.int64)

    bridge_selected_count = int(np.count_nonzero(is_bridge))

    per_channel = []
    for channel_name in OBSERVATION_CHANNELS:
        c = OBSERVATION_CHANNEL_INDEX[channel_name]
        # Edges whose presynaptic neuron is a channel-c sensory neuron and
        # whose postsynaptic neuron is a selected bridge neuron.
        from_channel = channel_index[pre_of_edge] == c
        to_bridge = is_bridge[post_of_edge]
        covered_bridge_ids = np.unique(post_of_edge[from_channel & to_bridge])
        count = int(covered_bridge_ids.size)
        per_channel.append({"channel": channel_name, "coveredBridgeCount": count, "flagged": count == 0})

    per_population = []
    for population_name in OUTPUT_POPULATION_INDEX:
        p = OUTPUT_POPULATION_INDEX[population_name]
        # Edges whose presynaptic neuron is a selected bridge neuron and
        # whose postsynaptic neuron is a population-p descending neuron.
        from_bridge = is_bridge[pre_of_edge]
        to_population = population_index[post_of_edge] == p
        covered_bridge_ids = np.unique(pre_of_edge[from_bridge & to_population])
        count = int(covered_bridge_ids.size)
        per_population.append({"population": population_name, "coveredBridgeCount": count, "flagged": count == 0})

    flagged = any(entry["flagged"] for entry in per_channel) or any(entry["flagged"] for entry in per_population)

    return {
        "bridgeSelectedCount": bridge_selected_count,
        "perChannel": per_channel,
        "perPopulation": per_population,
        "flagged": flagged,
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True, help="selection's compiled .bin.gz")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="selection's own manifest.json (for --graph's expected binarySha256, and echoed as sourceGraphSha256)",
    )
    parser.add_argument("--out", type=Path, required=True, help="coverage.json output path (no default: always explicit)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    with args.manifest.open("r") as fh:
        manifest = json.load(fh)
    expected_sha256 = manifest["binarySha256"]

    graph = graph_io.load_verified_graph(args.graph, expected_sha256)
    coverage = compute_coverage(graph)

    artifact = {
        "version": VERSION,
        "sourceGraphSha256": expected_sha256,
        **coverage,
    }
    graph_io.write_canonical_json(args.out, artifact)
    flagged_channels = [c["channel"] for c in artifact["perChannel"] if c["flagged"]]
    flagged_populations = [p["population"] for p in artifact["perPopulation"] if p["flagged"]]
    print(
        f"coverage: wrote {args.out} (bridgeSelectedCount={artifact['bridgeSelectedCount']}, "
        f"flagged={artifact['flagged']}, flaggedChannels={flagged_channels}, flaggedPopulations={flagged_populations})"
    )


if __name__ == "__main__":
    main()
