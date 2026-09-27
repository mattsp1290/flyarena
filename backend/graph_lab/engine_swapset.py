"""Swap-set engine: build the intervened graph and its class-matched random
controls from a user-submitted swap list
(`.agents/plans/graph-lab/02-job-engines.md`'s swap-set engine: "build the
graphs with `swap_ops.py` in tmpfs and verify the invariants"), plus the
presentation statistics (control distribution quantiles/rank, published-null
percentile) `service.py`'s `default_runner` attaches to `entry-swapset.mjs`'s
own scoring result.

Split into two phases, run at two different times (a dual-review finding):

- `build_candidate` (fast, pure-numpy, one swap application + one graph
  rebuild): runs *synchronously* inside `service.py`'s `_swapset_argv`, at
  `Jobs.submit()` time, so an invalid swap set is still rejected with a
  clean 422 before any job is accepted -- unchanged behavior from the
  original design.
- `build_controls` (up to 100 `random_class_swaps` calls, each recomputing
  the candidate class over the whole edge set, plus 100 graph rebuilds and
  gzip writes): measured at ~46s at this request kind's own upper bounds
  (`swaps` max 50, `controls` max 100) -- too slow to run inside
  `Jobs.submit()`'s lock, which `Jobs.get` (status polling) and
  cancellation also need; moved into `run` below, this job's own
  supervised async subprocess, where the existing wall-clock ceiling and
  `os.killpg`-based cancellation already apply to it like any other engine
  work.

Never reimplements science: graph construction is `scripts/analysis/
swap_ops.py`'s own `apply_explicit_swaps`/`random_class_swaps`/
`bridge_mask_of` (loaded through `py_scripts.load_swap_ops`), and the two
small statistics helpers below (`quantile_index`/`rank_statistics`) are a
Python port of `scripts/null/null-stats.ts`'s own `quantileIndex`/
`rankStatistics` formulas -- the exact same convention
`scripts/null/intervention-report.ts`'s `armDistribution`/`rankStatistics`
already use for this study's published control-arm/null-rank statistics --
not a new statistical definition. Only ported because this presentation
step runs after the TS scoring step (`entry-swapset.mjs`) has already
returned control-of-flow to this Python engine; the underlying formula is
unchanged.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import py_scripts


def _progress(payload: "dict[str, Any]") -> None:
    print(json.dumps({"type": "progress", "progress": payload}), flush=True)


def _result(payload: "dict[str, Any]") -> None:
    print(json.dumps({"type": "result", "result": payload}), flush=True)


def _error(message: str) -> None:
    print(json.dumps({"type": "error", "message": message}), flush=True)


def quantile_index(n: int, p: float) -> int:
    """Mirrors `scripts/null/null-stats.ts`'s `quantileIndex` exactly:
    `p <= 0.5 ? floor(p * n) : min(n - 1, ceil(p * n) - 1)`."""
    import math

    if p <= 0.5:
        return int(math.floor(p * n))
    return min(n - 1, int(math.ceil(p * n)) - 1)


def rank_statistics(null_values: "list[float]", score: float) -> "dict[str, float]":
    """Mirrors `scripts/null/null-stats.ts`'s `rankStatistics` exactly:
    `kBelow`/`kEqual` counts, `bioPercentile = (kBelow + 0.5*kEqual) / n`,
    `pLow = (kBelow + kEqual + 1) / (n + 1)`, `pHigh = (n - kBelow + 1) / (n + 1)`."""
    n = len(null_values)
    if n == 0:
        raise ValueError("rank_statistics requires a non-empty null set")
    k_below = sum(1 for value in null_values if value < score)
    k_equal = sum(1 for value in null_values if value == score)
    return {
        "kBelow": k_below,
        "kEqual": k_equal,
        "bioPercentile": (k_below + 0.5 * k_equal) / n,
        "pLow": (k_below + k_equal + 1) / (n + 1),
        "pHigh": (n - k_below + 1) / (n + 1),
    }


def control_distribution(scores: "list[float]") -> "dict[str, Any]":
    """Mirrors `scripts/null/intervention-report.ts`'s `armDistribution`:
    `n`, `scores` sorted ascending, and the `p5`/`p50`/`p95` quantile
    *values* (`quantile_index` above)."""
    if not scores:
        raise ValueError("control_distribution requires at least one score")
    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    return {
        "n": n,
        "scores": sorted_scores,
        "p5": sorted_scores[quantile_index(n, 0.05)],
        "p50": sorted_scores[quantile_index(n, 0.5)],
        "p95": sorted_scores[quantile_index(n, 0.95)],
    }


# The published seeds/ticks the swap-set percentile is only ever comparable
# against (`02-job-engines.md`'s Result section: "recomputed on the same
# seeds only when the request uses the published seeds 30001-30100 and
# T 1800"), matching `training/src/flyarena_training/seeds.py`'s
# `HELD_OUT_SEED_START = 30001` and the published null's own 100-seed count.
PUBLISHED_SWAPSET_SEED_START = 30001
PUBLISHED_SWAPSET_SEED_COUNT = 100
PUBLISHED_SWAPSET_TICKS = 1800


def published_null_comparable(seed_start: int, seed_count: int, ticks: int) -> bool:
    return (
        seed_start == PUBLISHED_SWAPSET_SEED_START
        and seed_count == PUBLISHED_SWAPSET_SEED_COUNT
        and ticks == PUBLISHED_SWAPSET_TICKS
    )


def _write_gzip_graph(binfmt: Any, graph: Any, path: Path) -> "tuple[str, int]":
    binary = binfmt.encode_graph_binary(graph)
    binfmt.write_gzip_deterministic(binary, path)
    return hashlib.sha256(binary).hexdigest(), len(binary)


def _load_verified_biological(swap_ops: Any, data_dir: Path) -> "tuple[Any, dict, bytes]":
    manifest = json.loads((data_dir / "malecns-arena-v1.manifest.json").read_text())
    biological_gzip_path = data_dir / manifest["artifact"]
    with gzip.open(biological_gzip_path, "rb") as fh:
        biological_binary = fh.read()
    if hashlib.sha256(biological_binary).hexdigest() != manifest["binarySha256"]:
        raise ValueError("biological graph binary does not match the manifest's recorded sha256")
    bio_graph = swap_ops_decode(swap_ops, biological_binary)
    return bio_graph, manifest, biological_binary


def build_candidate(
    *,
    scripts_dir: Path,
    data_dir: Path,
    job_dir: Path,
    swaps: "list[tuple[int, int, int, int]]",
) -> "dict[str, Any]":
    """Fast, synchronous (submit-time) half: validate `swaps` against the
    real biological graph (`swap_ops.apply_explicit_swaps`, sequential,
    against the graph state as of each step) and write the resulting
    candidate graph. Raises `ValueError` for any invalid swap -- the only
    failure mode reachable at this phase, so `service.py`'s `_swapset_argv`
    can still map it to a clean 422 before accepting the job."""
    swap_ops = py_scripts.load_swap_ops(scripts_dir)
    binfmt = swap_ops.binfmt
    bio_graph, manifest, _biological_binary = _load_verified_biological(swap_ops, data_dir)

    candidate_graph, swap_stats = swap_ops.apply_explicit_swaps(bio_graph, swaps)

    candidate_path = job_dir / "candidate.bin.gz"
    candidate_sha256, _size = _write_gzip_graph(binfmt, candidate_graph, candidate_path)

    return {
        "biologicalGraph": {
            "graphId": "biological",
            "path": str(data_dir / manifest["artifact"]),
            "expectedSha256": manifest["binarySha256"],
        },
        "candidateGraph": {
            "graphId": "candidate",
            "path": str(candidate_path),
            "expectedSha256": candidate_sha256,
        },
        "swapStats": swap_stats,
        # The source class is every distinct `a` across `swaps`; the target
        # class is every distinct `d` (`02-job-engines.md`: "the source
        # class is the set of `a` roles and the target class the set of `d`
        # roles, as in `random_class_swaps`") -- both derived from the
        # *submitted* swap list, never from `interventions.py`'s own fixed
        # input/thrust masks (those are specific to the P/M pathway study;
        # a graph-lab swap set is arbitrary user input, so its class-matched
        # controls are built from *its own* roles, per the plan's own
        # wording -- independent of whether the submitted swaps' own `b`/`c`
        # happen to be bridge-restricted, which the plan does not require).
        "sourceIndices": sorted({a for (a, _b, _c, _d) in swaps}),
        "targetIndices": sorted({d for (_a, _b, _c, d) in swaps}),
        "k": len(swaps),
    }


def build_controls(
    *,
    scripts_dir: Path,
    data_dir: Path,
    job_dir: Path,
    source_indices: "list[int]",
    target_indices: "list[int]",
    k: int,
    controls: int,
) -> "list[dict[str, Any]]":
    """The expensive half (moved out of `Jobs.submit()`'s lock -- see this
    module's own doc comment): `controls` class-matched random controls
    (`swap_ops.random_class_swaps`, seeds `0..controls-1`, deterministic
    and documented here -- the plan specifies "0-100 class-matched random
    controls with k = |swaps|" but not a seed convention of its own, unlike
    `interventions.py`'s study-specific `M_SEED_BASE`). Can raise
    `RuntimeError` (via `random_class_swaps`) if the bridge-restricted
    candidate class for this graph's `source_indices`/`target_indices` is
    empty -- surfaced as a job failure (not a 422, which can only apply at
    submit time, before this phase ever runs), with a message that only
    ever echoes indices/counts (audited safe)."""
    if controls == 0:
        return []
    swap_ops = py_scripts.load_swap_ops(scripts_dir)
    binfmt = swap_ops.binfmt
    bio_graph, _manifest, _biological_binary = _load_verified_biological(swap_ops, data_dir)

    import numpy as np  # local import: only this function needs array masks

    neuron_count = int(bio_graph.metadata["neuronCount"])
    source_mask = np.zeros(neuron_count, dtype=bool)
    source_mask[source_indices] = True
    target_mask = np.zeros(neuron_count, dtype=bool)
    target_mask[target_indices] = True
    bridge_mask = swap_ops.bridge_mask_of(bio_graph)

    controls_list: "list[dict[str, Any]]" = []
    for seed in range(controls):
        control_graph, _stats = swap_ops.random_class_swaps(
            bio_graph, k, seed, source_mask, target_mask, bridge_mask
        )
        control_path = job_dir / f"control-{seed}.bin.gz"
        control_sha256, _size = _write_gzip_graph(binfmt, control_graph, control_path)
        controls_list.append(
            {"graphId": f"control-{seed}", "path": str(control_path), "expectedSha256": control_sha256}
        )
    return controls_list


def swap_ops_decode(swap_ops: Any, biological_binary: bytes) -> Any:
    """`swap_ops.py` has no graph *decoder* of its own (it only operates on
    an already-decoded `binfmt.GraphArrays`) -- `rewire.py`'s
    `decode_graph_binary` is the one place this repo decodes a raw graph
    binary into that shape, so this reuses it rather than duplicating the
    header/CSR-array parsing (`py_scripts.load_rewire` loads it from the
    same `scripts_dir` `swap_ops` itself came from, so both always resolve
    to the same repo/image checkout)."""
    scripts_dir = Path(swap_ops.__file__).resolve().parents[1]
    rewire = py_scripts.load_rewire(scripts_dir)
    return rewire.decode_graph_binary(biological_binary)


def _load_verified_published_null(data_dir: Path) -> dict:
    """Sha-verify `rewiring-null-v1.json` against the manifest's own
    recorded value before trusting its `rewired[].score` values for the
    published-null percentile -- mirrors `engine_lesion.regenerate_rewired_graph`'s
    identical check on the same file (a dual-review finding: this file
    previously read it unverified)."""
    manifest = json.loads((data_dir / "malecns-arena-v1.manifest.json").read_text())
    published_null_path = data_dir / "rewiring-null-v1.json"
    raw = published_null_path.read_bytes()
    expected_sha256 = manifest.get("rewiringNull", {}).get("sha256")
    if expected_sha256 and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("rewiring-null-v1.json does not match the manifest's recorded sha256")
    return json.loads(raw)


def run(args: "dict[str, Any]") -> "dict[str, Any]":
    """The swap-set job's single child process
    (`.agents/plans/graph-lab/02-job-engines.md`'s swap-set engine):
    build the class-matched random controls (the expensive half -- see
    this module's own doc comment), run `entry-swapset.mjs` over the full
    graph list (biological + the candidate `service.py` already built at
    submit time + these controls) as this process's own nested child, then
    attach the presentation statistics the plan's Result section asks for:
    the control distribution (p5/p50/p95) with the candidate's rank within
    it, and -- only when the request used the published seeds/ticks -- the
    candidate's percentile within the published 500-graph authored null.

    Mirrors `engine_atlas.py`'s own "orchestrator IS the single supervised
    child process" pattern (`py_scripts.run_step`/`last_json_message`,
    shared with it) -- see that module's doc comment for the full
    `jobs.py`-contract rationale.
    """
    job_dir = Path(args["jobDir"])
    scripts_dir = Path(args["scriptsDir"])
    data_dir = Path(args["dataDir"])

    _progress({"stage": "controls"})
    controls_list = build_controls(
        scripts_dir=scripts_dir,
        data_dir=data_dir,
        job_dir=job_dir,
        source_indices=args["sourceIndices"],
        target_indices=args["targetIndices"],
        k=args["k"],
        controls=args["controls"],
    )
    graphs = [args["biologicalGraph"], args["candidateGraph"], *controls_list]

    entry_args_path = job_dir / "entry-swapset-args.json"
    entry_args_path.write_text(
        json.dumps(
            {
                "dataDir": args["dataDir"],
                "graphs": graphs,
                "baselineGraphId": "biological",
                "seedStart": args["seedStart"],
                "seedCount": args["seedCount"],
                "ticks": args["ticks"],
            }
        )
    )
    _progress({"stage": "score"})
    stdout = py_scripts.run_step([args["nodeBin"], args["entryPath"], str(entry_args_path)], cwd=job_dir)
    scoring_result = py_scripts.last_json_message(stdout, source="entry-swapset")

    scores_by_graph_id = {entry["graphId"]: entry["mean"] for entry in scoring_result["scores"]}
    candidate_score = scores_by_graph_id["candidate"]
    control_scores = [
        mean for graph_id, mean in scores_by_graph_id.items() if graph_id.startswith("control-")
    ]

    result: "dict[str, Any]" = dict(scoring_result)
    if control_scores:
        distribution = control_distribution(control_scores)
        result["controlDistribution"] = distribution
        result["candidateRankAmongControls"] = rank_statistics(distribution["scores"], candidate_score)
    else:
        result["controlDistribution"] = None
        result["candidateRankAmongControls"] = None

    if published_null_comparable(args["seedStart"], args["seedCount"], args["ticks"]):
        published_null = _load_verified_published_null(data_dir)
        null_scores = [float(entry["score"]) for entry in published_null["rewired"]]
        result["publishedNullPercentile"] = rank_statistics(null_scores, candidate_score)
    else:
        result["publishedNullPercentile"] = "not comparable"

    return result


def main() -> None:
    args = json.loads(Path(sys.argv[1]).read_text())
    try:
        result = run(args)
    except subprocess.CalledProcessError as error:
        # `py_scripts.describe_step_failure`: prefers the failing step's
        # own already-sanitized `{"type":"error",...}` stdout message over
        # a bare exit code -- matches `engine_atlas.py`'s identical
        # handling; see that function's own doc comment.
        _error(py_scripts.describe_step_failure(error))
        sys.exit(1)
    except (ValueError, RuntimeError) as error:
        # This module's own science-code errors (swap/graph validation,
        # `random_class_swaps`'s "empty candidate class") -- audited to
        # only ever echo seeds/indices/counts/sha256 digests, never a file
        # path, so safe to return verbatim.
        _error(str(error))
        sys.exit(1)
    except Exception as error:  # noqa: BLE001 -- anything else (OSError et al.) can include an absolute server-side path in its own str() (a dual-review finding); keep the client-facing message generic and log the real exception server-side instead.
        logging.exception("graph-lab swapset job failed")
        _error("an internal error occurred while building the swap-set job")
        sys.exit(1)
    else:
        _result(result)


if __name__ == "__main__":
    main()
