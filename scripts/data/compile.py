"""Offline compiler: pinned MaleCNS v1.0 export -> `public/data/malecns-arena-v1.bin.gz`.

Two halves:

- `compile_graph(...)` is the format-level core: given an already-selected
  node list, an edge list (possibly with duplicate `(pre, post)` rows and
  edges that reference bodies outside the node set), per-node signs, and an
  authored input/output channel assignment, it aggregates duplicates, builds
  the presynaptic CSR, and returns a validated `binfmt.GraphArrays` plus a
  stats dict. This half has no knowledge of MaleCNS, feather files, or
  neuPrint -- `tests_python/test_compile.py` drives it directly from a tiny
  CSV fixture, and this module's own `main()` drives it from the real data.
- Everything else in this module (`load_*`, `select_subgraph`,
  `assign_channels`, `assign_signs`, `calibrate_global_gain`, `main`) is the
  MaleCNS-specific selection policy: which neurons go in, which channels
  they're wired to, and how the global dynamics parameters are calibrated.
  Every constant here is also recorded in the emitted ledger so the policy
  is reproducible from the ledger alone; see docs/data-provenance.md for the
  prose explanation of *why* each rule was chosen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.feather as feather

sys.path.insert(0, str(Path(__file__).resolve().parent))
import binfmt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"
PUBLIC_DATA_DIR = REPO_ROOT / "public" / "data"

ARTIFACT_NAME = "malecns-arena-v1"

# ---------------------------------------------------------------------------
# Selection policy constants. All of these are echoed into the emitted
# ledger's "selectionPolicy" block verbatim, so the ledger alone documents
# exactly how the artifact was produced.
# ---------------------------------------------------------------------------

#: Only neurons with this `status` (body-annotations table) are eligible at
#: all. "Traced" is the FlyEM status for a body considered essentially
#: complete; it excludes "Orphan" (disconnected fragments), "Glia",
#: "Unimportant", "Assign" (still under review), and "Anchor" (placeholder)
#: bodies.
ELIGIBLE_STATUS = "Traced"

#: Sensory input candidates: real VNC (ventral nerve cord) sensory neurons,
#: restricted to the two classes most analogous to this POC's abstracted
#: egocentric channels (wall clearance, speed, bearing/distance) -- tactile
#: and proprioceptive mechanosensation -- rather than the optic-lobe visual
#: pathway (`ol_sensory`) or central-brain chemosensory pathways
#: (`cb_sensory`), which this arena's declared 8-channel contract
#: (src/lib/arena/sensors.ts) does not attempt to model at the level of real
#: photoreceptor/olfactory-receptor input.
SENSORY_SUPERCLASS = "vnc_sensory"
SENSORY_CLASSES = ("mechanosensory_tactile", "mechanosensory_proprioceptive")

#: Descending-neuron output candidates: DNs are the real anatomical class
#: whose axons leave the brain through the neck connective to drive motor
#: circuits in the VNC -- the direct biological analog of this POC's
#: thrust/yaw/brake output populations (src/lib/arena/actions.ts).
DESCENDING_SUPERCLASS = "descending_neuron"

#: How many sensory/descending candidates survive the initial superclass
#: filter are ranked by total measured degree (sum of `weight` over every
#: edge touching that body, computed over the traced-only subgraph) and the
#: top N are kept. This keeps the artifact tractable while preferring the
#: best-connected (least likely to be a reconstruction fragment) neurons.
SENSORY_TARGET = 160  # 20 per input channel (8 channels)
DESCENDING_TARGET = 48  # 16 per output population (3 populations)

#: Bridge/intermediate population: neurons that are simultaneously (a) a
#: direct postsynaptic partner of some selected sensory neuron and (b) a
#: direct presynaptic partner of some selected descending neuron -- i.e. the
#: middle node of at least one real 2-edge sensory -> bridge -> descending
#: path. Ranked by the same total-degree measure and capped at
#: BRIDGE_TARGET. A 1-hop-each-direction bridge (rather than a longer BFS)
#: was chosen because it already yields thousands of candidate bridge nodes
#: (see docs/data-provenance.md); a deeper search was not needed to reach a
#: densely-connected, budget-sized subgraph.
BRIDGE_TARGET = 800

#: Minimum aggregate synapse count for a connection to be retained. This is
#: the standard connectomics practice of dropping very-low-count contacts
#: that are more likely to be reconstruction/segmentation noise than a
#: functionally meaningful synapse; the MaleCNS flat-connectome tables are
#: already synapse-confidence filtered at minconf 0.5, this is an additional
#: connection-level (not synapse-level) threshold applied on top.
SYNAPSE_THRESHOLD = 3

#: Presynaptic sign policy (Dale's law: this format applies one +-1 sign per
#: presynaptic neuron to every edge it emits). `consensus_nt` is the source
#: dataset's per-neuron aggregate neurotransmitter prediction (an
#: *annotation*, not a topology measurement). The +-1 mapping itself is an
#: authored policy:
#:   - acetylcholine -> excitatory (+1): the dominant excitatory fast
#:     transmitter in the fly CNS.
#:   - gaba, glutamate -> inhibitory (-1): GABA is the dominant fast
#:     inhibitory transmitter; glutamate is treated as inhibitory here
#:     (ionotropic glutamate-gated chloride receptors are the dominant
#:     glutamate receptor class in the fly CNS), following the same
#:     convention used in prior connectome-constrained rate-model work.
#:   - anything else (dopamine, octopamine, serotonin, histamine, "unclear",
#:     a row whose `consensus_nt` is missing/NaN, or a body with no
#:     neurotransmitter-table row at all) -> defaulted to excitatory (+1)
#:     and counted separately (by label) in `unknownTransmitters` in the
#:     ledger, since these are neuromodulatory or low-confidence calls this
#:     POC does not attempt to model directionally. `assign_signs` further
#:     splits "no row at all" (`missing`) from "row present but
#:     `consensus_nt` is NaN" (`missing_value`) rather than conflating them.
SIGN_BY_TRANSMITTER: Mapping[str, int] = {
    "acetylcholine": 1,
    "gaba": -1,
    "glutamate": -1,
}

#: Input/output channel counts (must match src/lib/arena/sensors.ts's
#: `OBSERVATION_CHANNELS` length and src/lib/arena/actions.ts's
#: `OUTPUT_POPULATION` size).
INPUT_CHANNEL_COUNT = 8
OUTPUT_POPULATION_COUNT = 3

#: Global dynamics defaults shared with the rest of this codebase's tiny/
#: random test fixtures (tests/fixtures/tiny-graph.ts's `createRandomGraph`),
#: chosen here for consistency rather than re-derived: a 1/30s substep,
#: continuous leak rate 0.35/s (so per-substep decay 0.35/30 ~= 0.0117,
#: comfortably under the format's <=1 monotonic-decay guidance), and rate/
#: input clamps of [-2, 2] / [-1, 1].
TIMESTEP_SECONDS = 1.0 / 30.0
LEAK_RATE = 0.35
RATE_MIN = -2.0
RATE_MAX = 2.0
INPUT_CLAMP_MIN = -1.0
INPUT_CLAMP_MAX = 1.0

#: `globalGain` is calibrated (not authored as a fixed literal) from the
#: compiled edge set itself: see `calibrate_global_gain`'s docstring for the
#: exact formula. `GLOBAL_GAIN_TARGET_DRIVE` is the one authored knob in
#: that formula -- the target total per-substep recurrent drive contribution
#: (in rate units) for a fully-active (`rate == rateMax`) neuron at the 95th
#: percentile of total outgoing contact-magnitude mass.
GLOBAL_GAIN_TARGET_DRIVE = 0.5
GLOBAL_GAIN_PERCENTILE = 95

#: Input/output per-neuron weights are left at a flat, uncalibrated 1.0 for
#: input neurons (channel values are already normalized to
#: [-1, 1]/[0, 1] by src/lib/arena/sensors.ts before this weight is
#: applied). Output neuron weight is 1 / (population size) so a
#: population's aggregated output is population-size-invariant rather than
#: scaling with how many neurons happen to be assigned to it.
INPUT_WEIGHT = 1.0


def git_revision() -> str:
    """`git rev-parse HEAD` *at compile time* -- purely informational. This
    is inherently self-referential (the commit that ships the regenerated
    artifact necessarily comes after the commit this function reads, since
    the artifact's own bytes can't be part of the commit that produced
    them), so it should never be treated as "the commit whose code produced
    this artifact" -- use `compiler_source_sha256()` for that. Recorded in
    the ledger as `compiledFromGitRevision`, documented there as informational.
    """
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
            .decode("ascii")
            .strip()
        )
    except Exception:  # pragma: no cover - only when git is unavailable
        return "unknown"


#: Directory containing this compiler's own Python source, hashed by
#: `compiler_source_sha256()`. A plain module-level constant (rather than
#: always deriving it from `__file__` inline) so tests can assert against
#: the same value the real pipeline uses.
COMPILER_SOURCE_DIR = Path(__file__).resolve().parent


#: The exact filenames "the compiler" consists of, for
#: `compiler_source_sha256()` below: the modules whose code actually
#: determines the compiled `.bin`/`.bin.gz` bytes. Deliberately an explicit
#: allowlist rather than a `*.py` directory glob (which this function's
#: first version used): `scripts/data/positions.py` is a sidecar script
#: that joins the pinned annotations' soma columns onto the compiled
#: graph's own `biologicalIds` after the fact -- it never influences
#: `compile_graph`'s aggregation/CSR logic or the emitted `.bin.gz` bytes,
#: and must not change this hash or force an unrelated recompile when it is
#: added or edited. See docs/data-provenance.md's "Soma positions sidecar"
#: section.
COMPILER_SOURCE_FILENAMES: tuple[str, ...] = ("binfmt.py", "compile.py", "download.py", "rewire.py")

#: Every `scripts/data/*.py` file that is *not* part of "the compiler" --
#: i.e. every file `compiler_source_sha256()` deliberately excludes.
#: `test_every_scripts_data_module_is_classified` in `tests_python/
#: test_compile.py` asserts that `COMPILER_SOURCE_FILENAMES` and this tuple
#: partition the directory's actual `*.py` files exactly, so a new module
#: dropped into `scripts/data/` (compiler or sidecar) can never be silently
#: left out of both -- unlike the old `*.py` glob, an allowlist fails open
#: by default; this test is what makes it fail closed instead.
NON_COMPILER_SIDECAR_FILENAMES: tuple[str, ...] = ("positions.py",)


def compiler_source_sha256(source_dir: Path = COMPILER_SOURCE_DIR) -> str:
    """sha256 over this compiler's own Python source
    (`COMPILER_SOURCE_FILENAMES`: `binfmt.py`, `compile.py`, `download.py`,
    `rewire.py` -- not the generated `__pycache__`, and not any other
    sidecar script that happens to also live in `scripts/data/`), recorded
    as `compilerSourceSha256` in both the manifest and the ledger.

    Unlike `git_revision()`'s self-referential git SHA (see its docstring),
    this value is derived directly from the code that ran: recompiling
    after *any* change to these files -- even one that happens not to
    change the compiled bytes -- changes this hash. That makes "did the
    committed artifact get regenerated after this code change" a
    mechanically checkable property (see
    `tests_python/test_compile.py::test_compiler_source_sha256_matches_committed_ledger_and_manifest`
    and `tests/unit/malecns-artifact.test.ts`'s TypeScript equivalent)
    instead of something that can silently drift, which is exactly the
    class of bug a self-referential git SHA field had.

    Scheme (must exactly match the TypeScript recomputation in
    `tests/unit/malecns-artifact.test.ts`): for each name in
    `COMPILER_SOURCE_FILENAMES` sorted ascending, feed one sha256 hasher:
    the filename (UTF-8 bytes), then a single NUL byte, then the file's raw
    bytes. Including the filename means two files swapping content is not
    an accidental collision; the NUL byte gives an unambiguous
    filename/content boundary.
    """
    hasher = hashlib.sha256()
    for name in sorted(COMPILER_SOURCE_FILENAMES):
        path = source_dir / name
        hasher.update(path.name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Format-level core: reusable by both the real pipeline and
# tests_python/test_compile.py's tiny CSV fixture.
# ---------------------------------------------------------------------------


def _aggregate_duplicate_edges(
    pre_idx: np.ndarray, post_idx: np.ndarray, weight: np.ndarray, neuron_count: int
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]":
    """Aggregate duplicate `(pre, post)` rows by summing their magnitudes
    (see docs/graph-format.md's "Canonical row ordering and duplicate
    edges") and build the presynaptic CSR offsets from the same sorted
    order, since the `pair_key`-ascending order this function establishes
    (pre-major, post-minor) is already exactly the CSR order the format
    requires -- no separate re-sort is needed downstream.

    Sorts primarily by `pair_key` and secondarily by weight value (not
    input row order): floating-point addition is not associative, so
    summing a duplicate group in input order would make the result depend
    on which order the *source rows* happened to arrive in. Sorting each
    group's weights ascending before `reduceat` makes the summation order
    -- and therefore the output bytes -- a function of the edge set alone.

    Returns `(agg_pre, agg_post, aggregated_weight, presynaptic_offsets,
    duplicate_aggregated_count)`.
    """
    kept_row_count = len(pre_idx)
    pair_key = pre_idx.astype(np.int64) * neuron_count + post_idx.astype(np.int64)
    order = np.lexsort((weight, pair_key))
    pair_key_sorted = pair_key[order]
    weight_sorted = weight[order]
    unique_keys, start_positions = np.unique(pair_key_sorted, return_index=True)
    aggregated_weight = np.add.reduceat(weight_sorted, start_positions)
    duplicate_aggregated_count = int(kept_row_count - len(unique_keys))

    agg_pre = (unique_keys // neuron_count).astype(np.uint32)
    agg_post = (unique_keys % neuron_count).astype(np.uint32)

    # unique_keys is already sorted ascending, and since post = key % N for a
    # fixed pre (key // N), rows for the same pre are already
    # post-ascending too -- exactly the CSR order the format requires.
    presynaptic_offsets = np.zeros(neuron_count + 1, dtype=np.uint32)
    row_counts = np.bincount(agg_pre.astype(np.int64), minlength=neuron_count)
    presynaptic_offsets[1:] = np.cumsum(row_counts)

    return agg_pre, agg_post, aggregated_weight, presynaptic_offsets, duplicate_aggregated_count


def compile_graph(
    node_ids: Sequence[int],
    edges: pd.DataFrame,  # columns: pre, post, weight (raw body ids, may duplicate / reference outside node_ids)
    signs: Mapping[int, int],
    input_assignment: Mapping[int, "tuple[int, float]"],
    output_assignment: Mapping[int, "tuple[int, float]"],
    input_channel_count: int,
    output_population_count: int,
    metadata_params: Mapping[str, float],
) -> "tuple[binfmt.GraphArrays, dict]":
    """Core aggregation + CSR construction, independent of where `node_ids`/
    `edges`/`signs` came from. Returns `(graph, stats)`; `stats` feeds the
    manifest/ledger (real pipeline) or test assertions (fixture pipeline).
    """
    node_order = sorted(set(int(n) for n in node_ids))
    index_of = {body: i for i, body in enumerate(node_order)}
    neuron_count = len(node_order)

    pre_raw = edges["pre"].to_numpy()
    post_raw = edges["post"].to_numpy()
    weight_raw = edges["weight"].to_numpy(dtype=np.float64)

    in_node_set = np.array(
        [p in index_of and q in index_of for p, q in zip(pre_raw.tolist(), post_raw.tolist())],
        dtype=bool,
    )
    dropped_endpoint_count = int((~in_node_set).sum())

    pre_idx = np.array([index_of[p] for p in pre_raw[in_node_set].tolist()], dtype=np.int64)
    post_idx = np.array([index_of[p] for p in post_raw[in_node_set].tolist()], dtype=np.int64)
    weight = weight_raw[in_node_set]

    self_loop_count = int((pre_idx == post_idx).sum())

    agg_pre, agg_post, aggregated_weight, presynaptic_offsets, duplicate_aggregated_count = (
        _aggregate_duplicate_edges(pre_idx, post_idx, weight, neuron_count)
    )
    edge_count = len(agg_pre)

    biological_ids = np.array(node_order, dtype=np.uint64)
    presynaptic_signs = np.array(
        [signs.get(body, 1) for body in node_order], dtype=np.int8
    )

    input_channel_index = np.full(neuron_count, -1, dtype=np.int32)
    input_weight = np.zeros(neuron_count, dtype=np.float32)
    for body, (channel, weight_value) in input_assignment.items():
        idx = index_of[body]
        input_channel_index[idx] = channel
        input_weight[idx] = weight_value

    output_population_index = np.full(neuron_count, -1, dtype=np.int32)
    output_weight = np.zeros(neuron_count, dtype=np.float32)
    for body, (population, weight_value) in output_assignment.items():
        idx = index_of[body]
        output_population_index[idx] = population
        output_weight[idx] = weight_value

    metadata = {
        "formatVersion": binfmt.SUPPORTED_FORMAT_VERSION,
        "neuronCount": neuron_count,
        "edgeCount": edge_count,
        "inputChannelCount": input_channel_count,
        "outputPopulationCount": output_population_count,
        **metadata_params,
    }

    graph = binfmt.GraphArrays(
        metadata=metadata,
        biological_ids=biological_ids,
        presynaptic_offsets=presynaptic_offsets,
        postsynaptic_indices=agg_post,
        contact_magnitudes=aggregated_weight.astype(np.float32),
        presynaptic_signs=presynaptic_signs,
        input_channel_index=input_channel_index,
        input_weight=input_weight,
        output_population_index=output_population_index,
        output_weight=output_weight,
    )
    binfmt.validate_graph(graph)

    touched = np.zeros(neuron_count, dtype=bool)
    touched[agg_pre.astype(np.int64)] = True
    touched[agg_post.astype(np.int64)] = True
    isolated_node_count = int(neuron_count - touched.sum())

    stats = {
        "neuronCount": neuron_count,
        "edgeCount": edge_count,
        "droppedEndpointCount": dropped_endpoint_count,
        "duplicateAggregatedCount": duplicate_aggregated_count,
        "selfLoopCount": self_loop_count,
        "isolatedNodeCount": isolated_node_count,
    }
    return graph, stats


# ---------------------------------------------------------------------------
# MaleCNS-specific selection policy.
# ---------------------------------------------------------------------------


def load_annotations(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    return feather.read_table(
        raw_dir / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
    ).to_pandas()


def load_neurotransmitters(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    return feather.read_table(
        raw_dir / "body-neurotransmitters-male-cns-v1.0.feather"
    ).to_pandas()


def load_weights(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    return feather.read_table(
        raw_dir / "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        columns=["body_pre", "body_post", "weight"],
    ).to_pandas()


def _build_adjacency(weights: pd.DataFrame):
    pre = weights["body_pre"].to_numpy()
    post = weights["body_post"].to_numpy()

    order_f = np.argsort(pre, kind="stable")
    pre_sorted = pre[order_f]
    post_by_pre = post[order_f]
    uniq_pre, start_f = np.unique(pre_sorted, return_index=True)
    end_f = np.append(start_f[1:], len(pre_sorted))
    forward_range = dict(zip(uniq_pre.tolist(), zip(start_f.tolist(), end_f.tolist())))

    order_b = np.argsort(post, kind="stable")
    post_sorted = post[order_b]
    pre_by_post = pre[order_b]
    uniq_post, start_b = np.unique(post_sorted, return_index=True)
    end_b = np.append(start_b[1:], len(post_sorted))
    backward_range = dict(zip(uniq_post.tolist(), zip(start_b.tolist(), end_b.tolist())))

    def forward_neighbors(node: int) -> np.ndarray:
        r = forward_range.get(node)
        if r is None:
            return np.empty(0, dtype=np.int64)
        s, e = r
        return post_by_pre[s:e]

    def backward_neighbors(node: int) -> np.ndarray:
        r = backward_range.get(node)
        if r is None:
            return np.empty(0, dtype=np.int64)
        s, e = r
        return pre_by_post[s:e]

    return forward_neighbors, backward_neighbors


def _rank_by_degree(candidates: pd.DataFrame, degree: pd.Series) -> pd.DataFrame:
    """Rank `candidates` (a DataFrame with a `bodyId` column) by total
    measured degree descending, breaking ties by ascending `bodyId` for a
    fully deterministic, ledger-reproducible order. Shared by all three of
    `select_subgraph`'s rankings (sensory, descending, bridge) so a re-pin
    to different source data can never produce an undocumented tie-break
    that depends on an implementation-detail sort algorithm's stability
    rather than an explicit rule.
    """
    return candidates.assign(
        degree=candidates["bodyId"].map(degree).fillna(0)
    ).sort_values(["degree", "bodyId"], ascending=[False, True])


def select_subgraph(annotations: pd.DataFrame, weights: pd.DataFrame) -> dict:
    """Implements the SENSORY_* / DESCENDING_* / BRIDGE_TARGET policy
    documented above. Returns a dict with `sensory_ids`, `descending_ids`,
    `bridge_ids`, `node_ids` (their union, deduplicated) and the before/
    after candidate counts the ledger records."""
    duplicate_body_ids = annotations.loc[annotations["bodyId"].duplicated(keep=False), "bodyId"].unique()
    if len(duplicate_body_ids) > 0:
        raise ValueError(
            "body-annotations table has duplicate bodyId(s), expected exactly one row per "
            f"neuron: {sorted(duplicate_body_ids.tolist())[:10]}"
            + (" ..." if len(duplicate_body_ids) > 10 else "")
        )

    traced = annotations[annotations["status"] == ELIGIBLE_STATUS].copy()

    sensory_candidates = traced[
        (traced["superclass"] == SENSORY_SUPERCLASS) & (traced["class"].isin(SENSORY_CLASSES))
    ]
    descending_candidates = traced[traced["superclass"] == DESCENDING_SUPERCLASS]

    traced_ids = set(traced["bodyId"].tolist())
    traced_mask = np.isin(weights["body_pre"].to_numpy(), list(traced_ids)) & np.isin(
        weights["body_post"].to_numpy(), list(traced_ids)
    )
    traced_edges = weights[traced_mask]

    degree = pd.concat(
        [
            traced_edges.groupby("body_pre")["weight"].sum(),
            traced_edges.groupby("body_post")["weight"].sum(),
        ]
    ).groupby(level=0).sum()

    sensory_ranked = _rank_by_degree(sensory_candidates, degree)
    descending_ranked = _rank_by_degree(descending_candidates, degree)

    sensory_ids = sensory_ranked["bodyId"].head(SENSORY_TARGET).astype(np.int64).tolist()
    descending_ids = descending_ranked["bodyId"].head(DESCENDING_TARGET).astype(np.int64).tolist()

    forward_neighbors, backward_neighbors = _build_adjacency(traced_edges)

    def one_hop(seed_ids, neighbor_fn) -> set:
        visited: set = set()
        for node in seed_ids:
            for neighbor in neighbor_fn(node):
                visited.add(int(neighbor))
        return visited

    forward_from_sensory = one_hop(sensory_ids, forward_neighbors)
    backward_from_descending = one_hop(descending_ids, backward_neighbors)
    bridge_candidates = (
        (forward_from_sensory & backward_from_descending) - set(sensory_ids) - set(descending_ids)
    )

    bridge_frame = pd.DataFrame({"bodyId": sorted(bridge_candidates)})
    bridge_ranked = _rank_by_degree(bridge_frame, degree)
    bridge_ids = bridge_ranked["bodyId"].head(BRIDGE_TARGET).astype(np.int64).tolist()

    node_ids = sorted(set(sensory_ids) | set(descending_ids) | set(bridge_ids))

    return {
        "sensory_ids": sensory_ids,
        "descending_ids": descending_ids,
        "bridge_ids": bridge_ids,
        "node_ids": node_ids,
        "counts": {
            "tracedBodyCount": len(traced_ids),
            "sensoryCandidateCount": len(sensory_candidates),
            "sensorySelectedCount": len(sensory_ids),
            "descendingCandidateCount": len(descending_candidates),
            "descendingSelectedCount": len(descending_ids),
            "bridgeCandidateCount": len(bridge_candidates),
            "bridgeSelectedCount": len(bridge_ids),
            "finalNodeCount": len(node_ids),
        },
    }


def select_edges(weights: pd.DataFrame, node_ids: Sequence[int]) -> pd.DataFrame:
    node_arr = np.array(sorted(set(node_ids)), dtype=np.int64)
    mask = np.isin(weights["body_pre"].to_numpy(), node_arr) & np.isin(
        weights["body_post"].to_numpy(), node_arr
    )
    sub = weights[mask]
    thresholded = sub[sub["weight"] >= SYNAPSE_THRESHOLD]
    return thresholded.rename(columns={"body_pre": "pre", "body_post": "post"})[
        ["pre", "post", "weight"]
    ].reset_index(drop=True)


def assign_signs(node_ids: Sequence[int], neurotransmitters: pd.DataFrame) -> "tuple[dict, dict]":
    """Returns `(signs, unknown_by_label)`. `unknown_by_label` distinguishes
    `"missing"` (no row at all for that body in the neurotransmitters table)
    from `"missing_value"` (a row exists but `consensus_nt` is NaN/empty) --
    docs/data-provenance.md's policy section defines "missing" as the former
    only, so the two must not be folded together even though both currently
    default to the same excitatory sign.
    """
    duplicate_bodies = neurotransmitters.loc[
        neurotransmitters["body"].duplicated(keep=False), "body"
    ].unique()
    if len(duplicate_bodies) > 0:
        raise ValueError(
            "body-neurotransmitters table has duplicate body id(s), expected exactly one row "
            f"per neuron: {sorted(duplicate_bodies.tolist())[:10]}"
            + (" ..." if len(duplicate_bodies) > 10 else "")
        )

    nt_by_body = neurotransmitters.set_index("body")["consensus_nt"]
    has_row = set(nt_by_body.index.tolist())
    signs: dict[int, int] = {}
    unknown_by_label: dict[str, int] = {}
    for body in node_ids:
        label = nt_by_body.get(body) if body in has_row else None
        sign = SIGN_BY_TRANSMITTER.get(label)
        if sign is None:
            signs[body] = 1
            if body not in has_row:
                key = "missing"
            elif label is None or pd.isna(label):
                key = "missing_value"
            else:
                key = str(label)
            unknown_by_label[key] = unknown_by_label.get(key, 0) + 1
        else:
            signs[body] = sign
    return signs, unknown_by_label


def assign_channels(
    sensory_ids: Sequence[int], descending_ids: Sequence[int]
) -> "tuple[dict, dict]":
    """Authored (not biologically derived) input/output wiring: neurons are
    sorted by ascending body ID and partitioned into equal contiguous
    blocks, one block per channel/population. This is an arbitrary,
    deterministic, fully documented assignment -- see docs/data-provenance.md
    -- not a claim that a given real neuron "is" e.g. the food-bearing
    sensor.

    `sensory_ids`/`descending_ids` are deduplicated before partitioning
    (defense in depth: `select_subgraph` should never produce duplicates
    itself, since annotation `bodyId` is asserted unique there, but a
    duplicate slipping through here would otherwise silently consume two
    partition slots for one body and skew the channel/population boundary
    math for every neuron after it).
    """
    if len(sensory_ids) == 0:
        raise ValueError("assign_channels: sensory_ids must be non-empty")
    if len(descending_ids) == 0:
        raise ValueError("assign_channels: descending_ids must be non-empty")

    sensory_sorted = sorted(set(sensory_ids))
    input_assignment: dict[int, "tuple[int, float]"] = {}
    for position, body in enumerate(sensory_sorted):
        # Contiguous equal-ish blocks: body at sorted position p goes to
        # channel floor(p * INPUT_CHANNEL_COUNT / len(sensory_sorted)).
        channel = min(
            position * INPUT_CHANNEL_COUNT // len(sensory_sorted), INPUT_CHANNEL_COUNT - 1
        )
        input_assignment[body] = (channel, INPUT_WEIGHT)

    descending_sorted = sorted(set(descending_ids))
    output_assignment: dict[int, "tuple[int, float]"] = {}
    counts_per_population = [0] * OUTPUT_POPULATION_COUNT
    populations = []
    for position, body in enumerate(descending_sorted):
        population = min(
            position * OUTPUT_POPULATION_COUNT // len(descending_sorted), OUTPUT_POPULATION_COUNT - 1
        )
        populations.append(population)
        counts_per_population[population] += 1
    for body, population in zip(descending_sorted, populations):
        output_assignment[body] = (population, 1.0 / counts_per_population[population])

    return input_assignment, output_assignment


def calibrate_global_gain(edges: pd.DataFrame) -> float:
    """`globalGain` is the one dynamics parameter derived from the compiled
    edge set rather than fixed a priori. Per docs/graph-format.md's Dynamics
    section, one substep's recurrent drive contribution from neuron `pre` to
    all its targets combined is
    `globalGain * rate[pre] * sum(contactMagnitudes over pre's row)`.
    This computes the GLOBAL_GAIN_PERCENTILE-th percentile of that
    per-neuron summed-magnitude ("out_sum"), then solves for the globalGain
    that would make a fully-active (`rate == rateMax` conceptually
    normalized to 1 activation unit) neuron at that percentile contribute
    exactly GLOBAL_GAIN_TARGET_DRIVE rate-units of total outgoing drive in
    one substep -- i.e. `globalGain = GLOBAL_GAIN_TARGET_DRIVE / out_sum_p95`.
    This keeps the bulk of the network's recurrent drive within a bounded,
    non-saturating range without hand-picking a magic constant.
    """
    if len(edges) == 0:
        raise ValueError("calibrate_global_gain: edges must be non-empty")
    out_sum = edges.groupby("pre")["weight"].sum()
    percentile_value = float(np.percentile(out_sum.to_numpy(), GLOBAL_GAIN_PERCENTILE))
    if percentile_value <= 0:
        return 0.0
    return GLOBAL_GAIN_TARGET_DRIVE / percentile_value


def build_manifest_and_ledger(
    graph: binfmt.GraphArrays,
    stats: dict,
    selection: dict,
    unknown_by_label: dict,
    binary_sha256: str,
    binary_gzip_sha256: str,
    binary_gzip_size: int,
    binary_size: int,
    compiler_source_sha256_value: str,
) -> "tuple[dict, dict]":
    from download import SOURCE_FILES  # local import to avoid a hard dependency for fixture tests

    meta = graph.metadata
    manifest = {
        "formatVersion": meta["formatVersion"],
        "artifact": f"{ARTIFACT_NAME}.bin.gz",
        "neuronCount": stats["neuronCount"],
        "edgeCount": stats["edgeCount"],
        "inputChannelCount": meta["inputChannelCount"],
        "outputPopulationCount": meta["outputPopulationCount"],
        "binarySha256": binary_sha256,
        "binaryBytes": binary_size,
        "gzipSha256": binary_gzip_sha256,
        "gzipBytes": binary_gzip_size,
        "license": "CC-BY-4.0",
        "sourceDataset": "male-cns:v1.0 (Janelia FlyEM Male CNS connectome)",
        # sha256 over COMPILER_SOURCE_FILENAMES at compile time -- see
        # compiler_source_sha256()'s docstring. Echoed into the ledger too
        # (below) so either file alone proves which compiler code produced
        # this artifact.
        "compilerSourceSha256": compiler_source_sha256_value,
    }

    ledger = {
        "artifact": f"{ARTIFACT_NAME}.bin.gz",
        # sha256 over COMPILER_SOURCE_FILENAMES (binfmt.py, compile.py,
        # download.py, rewire.py) at compile time -- see
        # compiler_source_sha256()'s docstring in scripts/data/compile.py.
        # Unlike compiledFromGitRevision below, this is derived directly
        # from the code that ran, so it cannot go stale the way a
        # self-referential git SHA can.
        "compilerSourceSha256": compiler_source_sha256_value,
        # Informational only: `git rev-parse HEAD` *at compile time*. This is
        # inherently self-referential -- the commit that ships this
        # regenerated ledger necessarily comes after the commit this field
        # names, since the artifact's own bytes can't be part of the commit
        # that produced them -- so it may predate (and never equals) the
        # commit that actually ships this file. Do not use it to answer
        # "which commit's compiler produced this artifact"; use
        # `compilerSourceSha256` for that.
        "compiledFromGitRevision": git_revision(),
        "sourceDataset": {
            "name": "Male CNS Connectome",
            "version": "v1.0",
            "publisher": "Janelia FlyEM Project (HHMI), MRC Laboratory of Molecular Biology, Google Research",
            "url": "https://male-cns.janelia.org/",
            "license": "CC BY 4.0",
            "licenseUrl": "https://creativecommons.org/licenses/by/4.0/",
        },
        "sourceFiles": [
            {
                "filename": source.filename,
                "url": source.url,
                "sha256": source.sha256,
                "sizeBytes": source.size_bytes,
            }
            for source in SOURCE_FILES
        ],
        "selectionPolicy": {
            "eligibleStatus": ELIGIBLE_STATUS,
            "sensorySuperclass": SENSORY_SUPERCLASS,
            "sensoryClasses": list(SENSORY_CLASSES),
            "descendingSuperclass": DESCENDING_SUPERCLASS,
            "sensoryTarget": SENSORY_TARGET,
            "descendingTarget": DESCENDING_TARGET,
            "bridgeTarget": BRIDGE_TARGET,
            "synapseThreshold": SYNAPSE_THRESHOLD,
        },
        "selectionCounts": selection["counts"],
        "compileStats": stats,
        "unknownTransmitters": {
            "totalCount": sum(unknown_by_label.values()),
            "byLabel": unknown_by_label,
            "policy": "defaulted to excitatory sign (+1); see SIGN_BY_TRANSMITTER in scripts/data/compile.py",
        },
        "dynamics": {
            "timestepSeconds": TIMESTEP_SECONDS,
            "leakRate": LEAK_RATE,
            "rateMin": RATE_MIN,
            "rateMax": RATE_MAX,
            "inputClampMin": INPUT_CLAMP_MIN,
            "inputClampMax": INPUT_CLAMP_MAX,
            "globalGain": meta["globalGain"],
            "globalGainCalibration": {
                "targetDrive": GLOBAL_GAIN_TARGET_DRIVE,
                "percentile": GLOBAL_GAIN_PERCENTILE,
            },
        },
        "binarySha256": binary_sha256,
        "binaryBytes": binary_size,
        "gzipSha256": binary_gzip_sha256,
        "gzipBytes": binary_gzip_size,
        "license": "CC-BY-4.0",
    }
    return manifest, ledger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=PUBLIC_DATA_DIR)
    args = parser.parse_args(argv)

    print("Loading raw MaleCNS tables...")
    annotations = load_annotations(args.raw_dir)
    neurotransmitters = load_neurotransmitters(args.raw_dir)
    weights = load_weights(args.raw_dir)

    print("Selecting subgraph...")
    selection = select_subgraph(annotations, weights)
    print(f"  selection counts: {json.dumps(selection['counts'], indent=2)}")

    edges = select_edges(weights, selection["node_ids"])
    print(f"  final edges after synapse threshold: {len(edges)}")

    signs, unknown_by_label = assign_signs(selection["node_ids"], neurotransmitters)
    input_assignment, output_assignment = assign_channels(
        selection["sensory_ids"], selection["descending_ids"]
    )
    global_gain = calibrate_global_gain(edges)
    print(f"  calibrated globalGain = {global_gain}")

    metadata_params = {
        "timestepSeconds": TIMESTEP_SECONDS,
        "leakRate": LEAK_RATE,
        "rateMin": RATE_MIN,
        "rateMax": RATE_MAX,
        "inputClampMin": INPUT_CLAMP_MIN,
        "inputClampMax": INPUT_CLAMP_MAX,
        "globalGain": global_gain,
    }

    graph, stats = compile_graph(
        node_ids=selection["node_ids"],
        edges=edges,
        signs=signs,
        input_assignment=input_assignment,
        output_assignment=output_assignment,
        input_channel_count=INPUT_CHANNEL_COUNT,
        output_population_count=OUTPUT_POPULATION_COUNT,
        metadata_params=metadata_params,
    )
    print(f"  compile stats: {json.dumps(stats, indent=2)}")

    binary = binfmt.encode_graph_binary(graph)
    binary_sha256 = binfmt.sha256_hex(binary)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bin_gz_path = args.out_dir / f"{ARTIFACT_NAME}.bin.gz"
    binfmt.write_gzip_deterministic(binary, bin_gz_path)
    gzip_bytes = bin_gz_path.read_bytes()
    gzip_sha256 = binfmt.sha256_hex(gzip_bytes)

    compiler_source_sha256_value = compiler_source_sha256()
    print(f"compiler source sha256: {compiler_source_sha256_value}")

    manifest, ledger = build_manifest_and_ledger(
        graph=graph,
        stats=stats,
        selection=selection,
        unknown_by_label=unknown_by_label,
        binary_sha256=binary_sha256,
        binary_gzip_sha256=gzip_sha256,
        binary_gzip_size=len(gzip_bytes),
        binary_size=len(binary),
        compiler_source_sha256_value=compiler_source_sha256_value,
    )

    (args.out_dir / f"{ARTIFACT_NAME}.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / f"{ARTIFACT_NAME}.ledger.json").write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n"
    )

    print(f"\nWrote {bin_gz_path} ({len(gzip_bytes)} bytes, {len(gzip_bytes) / 1e6:.3f} MB)")
    print(f"binary sha256: {binary_sha256}")
    print(f"gzip sha256:   {gzip_sha256}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
