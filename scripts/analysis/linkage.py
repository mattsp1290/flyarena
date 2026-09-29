#!/usr/bin/env python3
"""Pathway linkage: per readout, the steady-state clearance transfer
`|T_clear,d|` for each of the readout's 48 descending-neuron inputs, and its
Spearman correlation against that readout's own saliency
(`.agents/plans/readout-attribution/02-analyses.md`'s WP2 analysis 4 /
H1, `00-overview.md`).

`T_clear,d` is the d-row analogue of `scripts/analysis/interventions.py`'s
own established "clearance" transfer quantity -- that script's swap search
targets exactly `T[thrust, rightClearance] + T[thrust, forwardClearance]`
(`RIGHT_CLEARANCE_IDX`/`FORWARD_CLEARANCE_IDX`, both imported from
`transfer.py`) and, at the per-neuron level, already computes
`R = steady_state_map[:, RIGHT_CLEARANCE_IDX] + steady_state_map[:,
FORWARD_CLEARANCE_IDX]` (`interventions.py:280`/`:526`) -- this module
reuses that exact same pair of input channels (not the full
`CLEARANCE_CHANNELS` set `scripts/null/task-clearance.ts` uses, which also
includes `foodDistance`, a food-seeking rather than wall-avoidance signal)
for `|T_clear,d| = |M[d, rightClearance] + M[d, forwardClearance]|`, where
`M = (lambda I - g A)^-1 B` is the same per-graph steady-state input map
`transfer.py`/`regime-task.ts` already use.

H1's correlation is against **thrust** saliency specifically (not yaw): the
established clearance-bridge pathway this whole pathway-interventions study
searches for is explicitly thrust-targeted (`interventions.py`'s own
`RIGHT_CLEARANCE_IDX`/`FORWARD_CLEARANCE_IDX` -> thrust target). Yaw's rho
is also reported, as a secondary diagnostic, clearly labeled as not the
predeclared H1 statistic -- a plan judgment call this module's own doc
comment discloses (the report must disclose it too).

Run with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DD_IAST_ENABLED=false` and `PYTHONPATH` unset, per every other `scripts/
analysis/*.py` module in this study (`transfer.py`'s own module doc
comment) -- enforced by `env_guard.assert_single_threaded_blas` below.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

from env_guard import assert_single_threaded_blas

assert_single_threaded_blas()

import numpy as np  # noqa: E402

from graph_io import (  # noqa: E402
    PUBLIC_DATA_DIR,
    DenseGraphMatrices,
    build_dense_matrices,
    load_verified_graph,
    sha256_hex,
    write_canonical_json,
)
from explain_stats import _rank, bootstrap_spearman_ci, metric_rng, quantile_index, spearman_rho  # noqa: E402
from transfer import (  # noqa: E402
    OBSERVATION_CHANNEL_INDEX,
    _compute_transfer,
    disconnected_matrices,
    transfer_producer,
    write_steady_state_sidecar,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
import binfmt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

RIGHT_CLEARANCE_IDX = OBSERVATION_CHANNEL_INDEX["rightClearance"]
FORWARD_CLEARANCE_IDX = OBSERVATION_CHANNEL_INDEX["forwardClearance"]

DEFAULT_RESAMPLES = 2000
#: Mirrors `scripts/attribution/shared.ts`'s `DEFAULT_ARCHIVED_INTERVENTION_INDEX_PATH` --
#: the committed archive copy every physical intervention index must byte-match
#: (`_load_intervention_index`). Defaulted here too (a dual-review finding: an
#: earlier version defaulted this flag to `None`, so passing `--intervention-index`
#: alone silently dropped the byte-equality check entirely instead of running it
#: against the one archive copy that always exists).
DEFAULT_ARCHIVED_INTERVENTION_INDEX = REPO_ROOT / "training" / "archive" / "intervention-index-v1.json"


# ---------------------------------------------------------------------------
# Graph resolution -- a Python-side mirror of `scripts/attribution/
# resolve-graph.ts`'s contract: bigq ids from the manifest, intervention ids
# from a physical `interventions.py` index.json (verified byte-identical to
# the committed `training/archive/intervention-index-v1.json`), disconnected
# derived from the sha-verified biological graph. Every id's resolved graph
# is additionally cross-checked against the ARCHIVE's own recorded
# `graphGzipSha256`/`graphBinarySha256` for that graphId (never
# `armBundleSha256` -- the jfgv provenance finding, 23c9557).
# ---------------------------------------------------------------------------


def _read_manifest(manifest_path: Path) -> dict:
    return json.loads(manifest_path.read_text())


def _load_intervention_index(index_path: Path, archived_index_path: Path) -> dict:
    physical_bytes = index_path.read_bytes()
    archived_bytes = archived_index_path.read_bytes()
    physical_sha = sha256_hex(physical_bytes)
    archived_sha = sha256_hex(archived_bytes)
    if physical_sha != archived_sha:
        raise ValueError(
            f"linkage: physical intervention index {index_path} (sha256 {physical_sha}) does not match the "
            f"committed archive copy {archived_index_path} (sha256 {archived_sha})"
        )
    index = json.loads(physical_bytes.decode("utf-8"))
    return {entry["id"]: entry for entry in index["entries"]}


def build_archive_shas(readouts: list[dict]) -> dict[str, tuple]:
    """One `(graphGzipSha256, graphBinarySha256)` per unique `graphId`, refusing
    (not silently keeping the first) if two archive entries for the SAME
    graphId disagree, and enforcing the `disconnected` null-sha invariant --
    a dual-review finding: an earlier version used `dict.setdefault`, which
    kept only the first entry per graphId and never checked `disconnected`
    at all, so a later entry recording a different (or a non-null
    `disconnected`) sha would go unnoticed while the TypeScript side
    (`resolve-graph.ts`'s `assertIdentityMatchesArchive`, called once per
    entry) would have thrown on it."""
    archive_shas: dict[str, tuple] = {}
    for entry in readouts:
        graph_id = entry["graphId"]
        shas = (entry["graphGzipSha256"], entry["graphBinarySha256"])
        if graph_id == "disconnected":
            if shas != (None, None):
                raise ValueError(f"linkage: \"disconnected\" archive entry \"{entry['id']}\" must have null graph shas")
            continue
        prior = archive_shas.setdefault(graph_id, shas)
        if prior != shas:
            raise ValueError(
                f"linkage: archive entries for graphId \"{graph_id}\" disagree on graph shas: {prior} vs {shas} "
                f"(entry \"{entry['id']}\")"
            )
    return archive_shas


def _assert_matches_archive(graph_id: str, gzip_sha256: str, binary_sha256: str, archive_shas: Mapping[str, tuple]) -> None:
    """Throws (never silently skips) unless `graph_id` has an archive entry
    with non-null shas matching the just-resolved graph -- mirrors
    `resolve-graph.ts`'s `assertIdentityMatchesArchive`, which throws in
    both of these cases rather than returning early (a dual-review finding
    on an earlier version of this function)."""
    expected = archive_shas.get(graph_id)
    if expected is None:
        raise ValueError(f"linkage: no archive entry for graphId \"{graph_id}\" to verify against")
    expected_gzip, expected_binary = expected
    if gzip_sha256 != expected_gzip or binary_sha256 != expected_binary:
        raise ValueError(
            f"linkage: resolved graph for \"{graph_id}\" (gzipSha256 {gzip_sha256}, binarySha256 {binary_sha256}) "
            f"does not match the archive's own graphGzipSha256 ({expected_gzip}) / graphBinarySha256 "
            f"({expected_binary})"
        )


def resolve_graph(
    graph_id: str,
    manifest: dict,
    manifest_dir: Path,
    intervention_index: Mapping[str, dict] | None,
    intervention_graphs_dir: Path | None,
    archive_shas: Mapping[str, tuple],
) -> "binfmt.GraphArrays":
    """Returns the decoded `graph`, sha-verified against the archive's own `graphGzipSha256`/`graphBinarySha256` for `graph_id` (never `armBundleSha256`)."""
    if graph_id == "biological" or graph_id == "rewired-seed0":
        if graph_id == "biological":
            artifact = manifest["artifact"]
            expected_gzip = manifest["gzipSha256"]
            expected_binary = manifest["binarySha256"]
        else:
            seed0 = manifest["rewiredArms"]["seed0"]
            artifact = seed0["artifact"]
            expected_gzip = seed0["gzipSha256"]
            expected_binary = seed0["binarySha256"]
        path = manifest_dir / artifact
        actual_gzip = sha256_hex(path.read_bytes())
        if actual_gzip != expected_gzip:
            raise ValueError(f"linkage: {path} gzip sha256 {actual_gzip} does not match manifest's {expected_gzip}")
        graph = load_verified_graph(path, expected_binary)
        _assert_matches_archive(graph_id, actual_gzip, expected_binary, archive_shas)
        return graph

    if graph_id == "disconnected":
        # No separate artifact: verified via the biological graph above,
        # then the connectivity zeroed for the dense-matrix step
        # (`disconnected_matrices`, applied by the caller) -- matching
        # `resolve-graph.ts`'s own "derive from the sha-verified biological
        # graph" pattern. `output_population_index` (what THIS function
        # exists to expose) is identical to biological's, unaffected by
        # zeroing `A`.
        return resolve_graph(
            "biological", manifest, manifest_dir, intervention_index, intervention_graphs_dir, archive_shas
        )

    if intervention_index is None or intervention_graphs_dir is None:
        raise ValueError(f"linkage: graphId \"{graph_id}\" needs --intervention-index (a physical interventions.py index.json)")
    entry = intervention_index.get(graph_id)
    if entry is None:
        raise ValueError(f"linkage: the intervention index has no entry for id \"{graph_id}\"")
    path = intervention_graphs_dir / entry["path"]
    actual_gzip = sha256_hex(path.read_bytes())
    if actual_gzip != entry["gzipSha256"]:
        raise ValueError(f"linkage: {path} gzip sha256 {actual_gzip} does not match index's {entry['gzipSha256']}")
    graph = load_verified_graph(path, entry["binarySha256"])
    _assert_matches_archive(graph_id, actual_gzip, entry["binarySha256"], archive_shas)
    return graph


def matrices_for_graph_id(graph_id: str, graph: "binfmt.GraphArrays") -> DenseGraphMatrices:
    matrices = build_dense_matrices(graph)
    return disconnected_matrices(matrices) if graph_id == "disconnected" else matrices


def output_neuron_indices(graph: "binfmt.GraphArrays") -> np.ndarray:
    """Ascending indices where `outputPopulationIndex[i] >= 0` -- the exact same D-space order `src/lib/connectome/readout.ts`'s `outputNeuronIndices` produces."""
    return np.where(graph.output_population_index >= 0)[0]


# ---------------------------------------------------------------------------
# Cluster bootstrap -- resamples the sidecar's `group` (or output population
# if a group is ever missing), not the 48 neurons independently. Mirrors
# `explain_stats.bootstrap_spearman_ci`'s own convention (resamples the
# already-computed FIXED ranks, not a full per-resample re-rank -- see that
# function's module doc comment for why).
# ---------------------------------------------------------------------------


def bootstrap_spearman_ci_clustered(
    rank_x: np.ndarray, rank_y: np.ndarray, cluster_of_index: np.ndarray, resamples: int, rng: np.random.Generator
) -> tuple[float, float, int, list[int]]:
    unique_clusters, cluster_sizes = np.unique(cluster_of_index, return_counts=True)
    members = {c: np.where(cluster_of_index == c)[0] for c in unique_clusters}
    k = len(unique_clusters)
    rhos = np.empty(resamples, dtype=np.float64)
    for r in range(resamples):
        drawn = rng.choice(unique_clusters, size=k, replace=True)
        idx = np.concatenate([members[c] for c in drawn])
        rx = rank_x[idx]
        ry = rank_y[idx]
        rx_c = rx - rx.mean()
        ry_c = ry - ry.mean()
        denom = float(np.sqrt(np.sum(rx_c * rx_c) * np.sum(ry_c * ry_c)))
        rhos[r] = float(np.sum(rx_c * ry_c) / denom) if denom > 0 else 0.0
    sorted_rhos = np.sort(rhos)
    lo = sorted_rhos[quantile_index(resamples, 0.025)]
    hi = sorted_rhos[quantile_index(resamples, 0.975)]
    return float(lo), float(hi), k, sorted(int(s) for s in cluster_sizes)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="training/archive/trained-readouts-v1.json")
    parser.add_argument("--saliency", type=Path, required=True, help="saliency.ts's output (saliency.json)")
    parser.add_argument("--descending-types", type=Path, required=True, help="public/data/descending-types-v1.json")
    parser.add_argument("--manifest", type=Path, default=PUBLIC_DATA_DIR / "malecns-arena-v1.manifest.json")
    parser.add_argument("--intervention-index", type=Path, default=None, help="physical interventions.py index.json")
    parser.add_argument("--archived-intervention-index", type=Path, default=DEFAULT_ARCHIVED_INTERVENTION_INDEX)
    parser.add_argument("--steady-state-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    # Resolve every path argument to absolute against the CWD this process
    # was invoked from, up front -- `sidecar_path.relative_to(REPO_ROOT)`
    # below requires an absolute `--steady-state-dir`, and every other path
    # is read from, so resolving once here (rather than at each call site)
    # keeps a relative `--out`/`--manifest`/etc. working regardless of cwd.
    for path_attr in (
        "archive",
        "saliency",
        "descending_types",
        "manifest",
        "intervention_index",
        "archived_intervention_index",
        "steady_state_dir",
        "out",
    ):
        value = getattr(args, path_attr)
        if value is not None:
            setattr(args, path_attr, value.resolve())

    archive_bytes = args.archive.read_bytes()
    archive_sha256 = sha256_hex(archive_bytes)
    archive = json.loads(archive_bytes.decode("utf-8"))
    readouts = archive["readouts"]
    saliency_doc = json.loads(args.saliency.read_text())
    # `saliency.ts`'s own `archiveSha256` stamp -- a dual-review finding:
    # this pipeline sha-verifies every graph and every theta, but previously
    # had no check tying its OWN intermediate JSON inputs back to the
    # archive that produced them, so a stale `saliency.json` left over from
    # a different archive version would silently drive this module's
    # correlation with no error at all.
    saliency_archive_sha256 = saliency_doc.get("archiveSha256")
    if saliency_archive_sha256 != archive_sha256:
        raise ValueError(
            f"linkage: {args.saliency} was produced from a different archive (archiveSha256 "
            f"{saliency_archive_sha256!r}, expected {archive_sha256!r}) -- rerun saliency.ts against the current archive"
        )
    saliency_by_id = {e["id"]: e for e in saliency_doc["entries"]}
    descending_types = json.loads(args.descending_types.read_text())
    # `descending-types-v1.json`'s `neurons[i].index` is the RAW graph
    # neuron id (0..neuronCount-1, e.g. 387/405/445 for this study's
    # biological graph -- NOT a contiguous 0..47 D-space index by itself).
    # Sorting by `index` ascending recovers `readout.ts`'s D-space order:
    # position `d` in this sorted list is exactly `outputNeuronIndices(graph)[d]`
    # for any graph sharing the same node identity (rewiring preserves
    # neuron count/order/population assignment) -- cross-checked per graph
    # below (`output_idx` must equal `raw_index_by_d`), not merely assumed.
    neurons = sorted(descending_types["neurons"], key=lambda n: n["index"])
    raw_index_by_d = np.array([n["index"] for n in neurons], dtype=np.int64)
    if len(set(raw_index_by_d.tolist())) != len(neurons):
        raise ValueError("linkage: descending-types-v1.json has duplicate neuron indices")
    cluster_of_index = np.array(
        [n["group"] if n["group"] is not None else n["population"] for n in neurons], dtype=np.int64
    )

    # `--steady-state-dir` must be inside the repo: `sidecarPath` below is
    # recorded relative to `REPO_ROOT` (`regime.ts` resolves it the same
    # way, against its own `repoRoot`, unconditionally -- see that file's
    # `runRegime`), so a directory outside the repo could never be read
    # back correctly regardless of whether this check catches it. Checked
    # here, before any graph is loaded or any dense O(n^3) solve runs (a
    # dual-review finding: an earlier version only discovered this via
    # `sidecar_path.relative_to(REPO_ROOT)` raising AFTER the first graph's
    # solve, whose cost this check is meant to avoid wasting).
    try:
        args.steady_state_dir.relative_to(REPO_ROOT)
    except ValueError:
        raise ValueError(
            f"linkage: --steady-state-dir {args.steady_state_dir} must be inside the repository ({REPO_ROOT}) -- "
            "linkage.json records sidecar paths repo-root-relative, and regime.ts resolves them against the repo root"
        ) from None

    manifest = _read_manifest(args.manifest)
    manifest_dir = args.manifest.parent
    intervention_index = (
        _load_intervention_index(args.intervention_index, args.archived_intervention_index)
        if args.intervention_index is not None
        else None
    )

    archive_shas = build_archive_shas(readouts)

    unique_graph_ids = sorted({entry["graphId"] for entry in readouts})
    t_clear_d_space_by_graph: dict[str, np.ndarray] = {}
    graphs_manifest: dict[str, dict] = {}
    args.steady_state_dir.mkdir(parents=True, exist_ok=True)
    intervention_graphs_dir = args.intervention_index.parent if args.intervention_index else None

    for graph_id in unique_graph_ids:
        graph = resolve_graph(graph_id, manifest, manifest_dir, intervention_index, intervention_graphs_dir, archive_shas)
        matrices = matrices_for_graph_id(graph_id, graph)
        meta = graph.metadata
        computation = _compute_transfer(
            matrices,
            leak_rate=float(meta["leakRate"]),
            global_gain=float(meta["globalGain"]),
            timestep_seconds=float(meta["timestepSeconds"]),
            strict_shape=False,
        )
        if computation.result["singular"]:
            raise ValueError(f"linkage: graph \"{graph_id}\" is singular -- cannot compute a steady-state map")
        steady_state_map = computation.steady_state_map

        output_idx = output_neuron_indices(graph)
        if output_idx.shape[0] != len(neurons) or not np.array_equal(output_idx, raw_index_by_d):
            raise ValueError(
                f"linkage: graph \"{graph_id}\"'s output-assigned raw neuron indices do not match "
                "descending-types-v1.json's own (sorted-by-index) D-space order -- rewiring must preserve the "
                "output population assignment and neuron identity"
            )

        sidecar_path = args.steady_state_dir / f"{graph_id}.steadystate.f64"
        write_steady_state_sidecar(sidecar_path, steady_state_map)
        graphs_manifest[graph_id] = {
            "sidecarPath": str(sidecar_path.relative_to(REPO_ROOT)),
            "sidecarSha256": sha256_hex(sidecar_path.read_bytes()),
            "neuronCount": int(steady_state_map.shape[0]),
            "inputChannelCount": int(steady_state_map.shape[1]),
        }
        # Row `output_idx[d]` (a raw neuron id) is `readout.ts`'s D-space
        # input `d` -- gather into D-space order (0..47) before combining
        # the two clearance columns, matching `interventions.py`'s own
        # `steady_state_map[:, RIGHT] + steady_state_map[:, FORWARD]` row
        # selection, restricted here to the 48 descending-output rows.
        t_clear_raw = steady_state_map[:, RIGHT_CLEARANCE_IDX] + steady_state_map[:, FORWARD_CLEARANCE_IDX]
        t_clear_d_space_by_graph[graph_id] = t_clear_raw[output_idx]

    out_entries = []
    for entry in sorted(readouts, key=lambda e: e["id"]):
        graph_id = entry["graphId"]
        saliency_entry = saliency_by_id.get(entry["id"])
        if saliency_entry is None:
            raise ValueError(f"linkage: no saliency entry for readout \"{entry['id']}\" -- run saliency.ts first")
        t_clear_abs = np.abs(t_clear_d_space_by_graph[graph_id])
        thrust = np.array(saliency_entry["thrust"])
        yaw = np.array(saliency_entry["yaw"])

        # A constant `|T_clear|` (e.g. "disconnected", whose A is zeroed --
        # confirmed empirically in this study's own archive: every
        # descending neuron's clearance transfer is exactly 0 there) or a
        # constant saliency vector makes Spearman's rho mathematically
        # undefined (zero variance in one of the two ranked variables).
        # `explain_stats._pearson` (via `spearman_rho`) returns a bare `0.0`
        # for that case (`denom == 0`), and the bootstrap CIs collapse to
        # `[0.0, 0.0]` the same way -- both would otherwise look exactly
        # like a real, measured "no correlation" in the shipped
        # `linkage.json` (a dual-review finding). Flagged here instead:
        # `rho*`/`ci*` are `null` and `degenerate: true` when this happens,
        # so a downstream reader (the WP3 report) can tell "no correlation"
        # from "not a meaningful question for this readout".
        t_clear_degenerate = bool(np.ptp(t_clear_abs) == 0.0)
        thrust_degenerate = t_clear_degenerate or bool(np.ptp(thrust) == 0.0)
        yaw_degenerate = t_clear_degenerate or bool(np.ptp(yaw) == 0.0)

        rho_thrust = None if thrust_degenerate else spearman_rho(thrust, t_clear_abs)
        rho_yaw = None if yaw_degenerate else spearman_rho(yaw, t_clear_abs)

        if thrust_degenerate:
            ci_cluster: list[float] | None = None
            ci_neuron: list[float] | None = None
            cluster_count = int(np.unique(cluster_of_index).size)
            _, counts = np.unique(cluster_of_index, return_counts=True)
            cluster_sizes = sorted(int(c) for c in counts)
        else:
            rank_saliency = _rank(thrust)
            rank_t_clear = _rank(t_clear_abs)
            rng_cluster = metric_rng(args.bootstrap_seed, f"cluster|{entry['id']}")
            rng_neuron = metric_rng(args.bootstrap_seed, f"neuron|{entry['id']}")
            ci_cluster_lo, ci_cluster_hi, cluster_count, cluster_sizes = bootstrap_spearman_ci_clustered(
                rank_saliency, rank_t_clear, cluster_of_index, args.resamples, rng_cluster
            )
            ci_neuron_lo, ci_neuron_hi = bootstrap_spearman_ci(rank_saliency, rank_t_clear, args.resamples, rng_neuron)
            ci_cluster = [ci_cluster_lo, ci_cluster_hi]
            ci_neuron = [ci_neuron_lo, ci_neuron_hi]

        out_entries.append(
            {
                "id": entry["id"],
                "graphId": graph_id,
                "degenerate": thrust_degenerate,
                "rhoThrust": rho_thrust,
                "rhoYaw": rho_yaw,
                "ciCluster": ci_cluster,
                "ciNeuron": ci_neuron,
                "clusterCount": cluster_count,
                "clusterSizes": cluster_sizes,
                "n": len(neurons),
                "tClear": [float(v) for v in t_clear_abs],
            }
        )

    payload = {
        "version": 1,
        "archiveSha256": archive_sha256,
        "producer": transfer_producer(),
        "clearanceChannels": ["rightClearance", "forwardClearance"],
        "resamples": args.resamples,
        "bootstrapSeed": args.bootstrap_seed,
        "graphs": graphs_manifest,
        "readouts": out_entries,
    }
    write_canonical_json(args.out, payload)
    print(f"linkage: wrote {args.out} ({len(out_entries)} readouts, {len(unique_graph_ids)} unique graphs)")


if __name__ == "__main__":
    main()
