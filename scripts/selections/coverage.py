#!/usr/bin/env python3
"""`.agents/plans/selection-robustness/02-per-selection-chain.md` WP2, step
4: per-selection bridge coverage, as predeclared in `00-overview.md`'s
"Coverage check" bullet.

`00-overview.md`: "per input channel and output population, the number of
selected bridge neurons with at least one edge from that channel's sensory
neurons **and** at least one edge to that population's descending neurons.
Any channel or population with zero coverage is flagged, and a selection
with a flagged channel is not categorized for the pathway finding" (the
categorization decision itself is `selection-report.ts`'s job, WP3; this
script only computes and records the counts and the flag).

A dual-review finding on an earlier version of this file: it counted each
side of that "and" independently (bridges reached from a channel, bridges
reaching a population), which is a weaker, non-conjunctive statistic -- a
bridge reached from channel c but with no path onward to *any* descending
population would still count as "covering" c. `compile.py`'s bridge
candidates are computed on the full traced-edge graph (`select_subgraph`'s
`forward_from_sensory & backward_from_descending`) *before* `select_edges`
drops edges below `SYNAPSE_THRESHOLD`, so a compiled bridge can genuinely
keep its sensory input while losing every edge to a descending neuron (or
the reverse) -- exactly the "dead-end bridge" the predeclared check exists
to catch, and exactly what `random-bridge` (uniform sampling, which favors
low-degree candidates) makes more likely. This version computes the full
8x3 joint (channel, population) matrix (`pairCoverage`) and derives each
marginal conjunctively: a bridge counts for channel c only if it also has
an edge to *some* descending population (not only to `c`'s own paired
population -- `perChannel`/`perPopulation` are one-dimensional summaries,
matching WP3's published `coverage: {perChannel[8], perPopulation[3],
flagged}` schema (`03-artifact-and-findings.md`), and `pairCoverage` is
recorded alongside for WP3 to consume at whichever granularity it needs.

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
import platform
import sys
from pathlib import Path

_ANALYSIS_DIR = str(Path(__file__).resolve().parents[1] / "analysis")
if _ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, _ANALYSIS_DIR)

from env_guard import assert_single_threaded_blas  # noqa: E402

assert_single_threaded_blas()

import numpy as np  # noqa: E402

import graph_io  # noqa: E402
import explain_selection_mode  # noqa: E402
from features import OBSERVATION_CHANNELS  # noqa: E402
from transfer import OBSERVATION_CHANNEL_INDEX, OUTPUT_POPULATION_INDEX  # noqa: E402

VERSION = 1

#: This script's own real import-graph closure -- `scripts/selections/`
#: (itself), `scripts/analysis/` (`graph_io`/`features`/`transfer`/
#: `explain_selection_mode`/`env_guard`), and `scripts/data/` (`graph_io.py`'s
#: own `sys.path.insert` of that directory, which is how `binfmt`/`rewire`
#: resolve) -- matching `transfer.py`'s `TRANSFER_ENTRY`/`TRANSFER_SEARCH_DIRS`
#: convention exactly, so a `coverage.json` can be told apart from one
#: produced by stale code the same way every other producer in this study
#: already is.
COVERAGE_ENTRY = Path(__file__).resolve()
COVERAGE_SOURCE_DIR = COVERAGE_ENTRY.parent
COVERAGE_SEARCH_DIRS: tuple[Path, ...] = (COVERAGE_SOURCE_DIR, COVERAGE_SOURCE_DIR.parent / "analysis")
REPO_ROOT = COVERAGE_SOURCE_DIR.parents[1]
PUBLIC_DATA_DIR = REPO_ROOT / "public" / "data"
DOCS_DIR = REPO_ROOT / "docs"


def coverage_producer() -> dict:
    """See `transfer.py`'s `transfer_producer()` doc comment -- identical shape and rationale."""
    dependencies = graph_io.python_dependency_closure(COVERAGE_ENTRY, REPO_ROOT, COVERAGE_SEARCH_DIRS)
    return {
        "script": "scripts/selections/coverage.py",
        "sourceSha256": graph_io.source_identity_sha256(REPO_ROOT, dependencies),
        "dependencies": dependencies,
        "host": {"arch": platform.machine(), "python": platform.python_version()},
    }


def compute_coverage(graph: "graph_io.binfmt.GraphArrays") -> dict:
    """Pure function of the decoded graph. Returns the artifact body (no
    `version`/`sourceGraphSha256`/`producer` -- `main` adds those)."""
    neuron_count = int(graph.metadata["neuronCount"])
    channel_index = graph.input_channel_index
    population_index = graph.output_population_index
    is_bridge = (channel_index < 0) & (population_index < 0)
    channel_count = len(OBSERVATION_CHANNEL_INDEX)
    population_count = len(OUTPUT_POPULATION_INDEX)

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

    # `bridge_in_from_channel[c, b]`: bridge neuron b has >=1 incoming edge
    # from a channel-c sensory neuron. `bridge_out_to_population[p, b]`:
    # bridge neuron b has >=1 outgoing edge to a population-p descending
    # neuron. Both are boolean per-(index, bridge-neuron) membership, built
    # once from the full edge list (not per channel/population in a Python
    # loop), then combined below for both the joint matrix and the
    # conjunctive marginals.
    bridge_in_from_channel = np.zeros((channel_count, neuron_count), dtype=bool)
    in_mask = is_bridge[post_of_edge] & (channel_index[pre_of_edge] >= 0)
    bridge_in_from_channel[channel_index[pre_of_edge[in_mask]], post_of_edge[in_mask]] = True

    bridge_out_to_population = np.zeros((population_count, neuron_count), dtype=bool)
    out_mask = is_bridge[pre_of_edge] & (population_index[post_of_edge] >= 0)
    bridge_out_to_population[population_index[post_of_edge[out_mask]], pre_of_edge[out_mask]] = True

    has_any_in = bridge_in_from_channel.any(axis=0)  # bridge has *some* sensory input
    has_any_out = bridge_out_to_population.any(axis=0)  # bridge has *some* descending output

    # The predeclared joint statistic: for each (channel, population) pair,
    # the number of bridges with an edge from that channel AND an edge to
    # that population.
    joint = bridge_in_from_channel[:, None, :] & bridge_out_to_population[None, :, :]  # (channel, population, N)
    pair_coverage = joint.sum(axis=2).astype(np.int64)  # (channel, population)

    per_channel = []
    for c, channel_name in enumerate(OBSERVATION_CHANNELS):
        # Conjunctive marginal: a bridge counts for channel c only if it also
        # reaches *some* descending population (a complete, if not
        # necessarily population-c-paired, sensory->bridge->descending path)
        # -- not a plain count of bridges merely reached from channel c,
        # which is what a dual-review finding flagged as under-strict.
        count = int(np.count_nonzero(bridge_in_from_channel[c] & has_any_out))
        per_channel.append({"channel": channel_name, "coveredBridgeCount": count, "flagged": count == 0})

    per_population = []
    for population_name in OUTPUT_POPULATION_INDEX:
        p = OUTPUT_POPULATION_INDEX[population_name]
        count = int(np.count_nonzero(bridge_out_to_population[p] & has_any_in))
        per_population.append({"population": population_name, "coveredBridgeCount": count, "flagged": count == 0})

    flagged = any(entry["flagged"] for entry in per_channel) or any(entry["flagged"] for entry in per_population)

    return {
        "bridgeSelectedCount": bridge_selected_count,
        "perChannel": per_channel,
        "perPopulation": per_population,
        # `pairCoverage[c][p]`: bridges with an edge from channel c AND an
        # edge to population p (the plan's literal, fully joint reading),
        # in `OBSERVATION_CHANNELS` x `OUTPUT_POPULATION_INDEX` order.
        "pairCoverage": pair_coverage.tolist(),
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

    # Selection-robustness WP2: the same "only the shipped graph may write
    # into the shipped tree" guard the other three chain producers carry
    # (dual-review finding: coverage.py was new write-path code left
    # unguarded).
    explain_selection_mode.guard_selection_scratch_target(
        args.out, "--out", expected_sha256, public_data_dir=PUBLIC_DATA_DIR, docs_dir=DOCS_DIR
    )

    graph = graph_io.load_verified_graph(args.graph, expected_sha256)
    coverage = compute_coverage(graph)

    artifact = {
        "version": VERSION,
        "sourceGraphSha256": expected_sha256,
        "producer": coverage_producer(),
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
