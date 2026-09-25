"""Shared graph loading, dense-matrix construction, and canonical-JSON output
for `scripts/analysis/transfer.py` and `scripts/analysis/features.py`.

Reuses `scripts/data/binfmt.py` (wire-format decode/validate) and
`scripts/data/rewire.py`'s `decode_graph_binary` (`.agents/plans/
null-explanation/02-transfer-and-features.md`'s repository evidence: "
`scripts/data/rewire.py:74` `decode_graph_binary(buffer) -> binfmt.GraphArrays`
") and `scripts/data/fsutil.py`'s atomic text write -- importing these
existing `scripts/data/` modules does not change `compile.py`'s
`compiler_source_sha256()`, which hashes those files' own bytes, not their
importers.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_DATA_DIR = str(Path(__file__).resolve().parents[1] / "data")
if _DATA_DIR not in sys.path:
    sys.path.insert(0, _DATA_DIR)

import binfmt  # noqa: E402
import fsutil  # noqa: E402
import rewire  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_DATA_DIR = REPO_ROOT / "public" / "data"


class GraphVerificationError(ValueError):
    """A graph file's decompressed sha256 did not match the expected value."""


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_verified_graph(path: Path, expected_sha256: str) -> "binfmt.GraphArrays":
    """Read a gzip-compressed graph binary, verify its *decompressed*
    sha256 against `expected_sha256` (the same check `null-worker.ts`'s
    `loadVerifiedGraphBinary` makes on the TypeScript side, and the same
    field name -- `binarySha256`/`sourceSha256` -- `rewire_batch.py`'s
    `index.json` and `rewiring-null-v1.json` both already carry), decode it,
    and return the validated `binfmt.GraphArrays`. Raises
    `GraphVerificationError` on a mismatch rather than silently computing a
    transfer matrix or feature set for the wrong bytes."""
    gzip_bytes = path.read_bytes()
    binary = gzip.decompress(gzip_bytes)
    actual = sha256_hex(binary)
    if actual != expected_sha256:
        raise GraphVerificationError(
            f"{path}: decompressed sha256 {actual} does not match expected {expected_sha256}"
        )
    return rewire.decode_graph_binary(binary)


@dataclass(frozen=True)
class DenseGraphMatrices:
    """Dense float64 views of one graph's dynamics, per
    `.agents/plans/null-explanation/02-transfer-and-features.md`'s "Linear
    transfer" section and `docs/graph-format.md`'s "Dynamics" section
    (mirrored by `src/lib/connectome/model.ts`'s `stepModel`):

    - `adjacency` (`A`), shape `(neuronCount, neuronCount)`: `A[post, pre] =
      presynapticSigns[pre] * contactMagnitudes[edge]` for every CSR edge
      `pre -> post`; `stepModel`'s recurrent drive is `globalGain * A @
      rate`.
    - `input_matrix` (`B`), shape `(neuronCount, inputChannelCount)`:
      `B[i, c] = inputWeight[i]` where `inputChannelIndex[i] == c`, else 0.
    - `output_matrix` (`O`), shape `(outputPopulationCount, neuronCount)`:
      `O[p, i] = outputWeight[i]` where `outputPopulationIndex[i] == p`,
      else 0.

    All three are computed once per graph in float64 (the wire format's own
    `contactMagnitudes`/`inputWeight`/`outputWeight` are float32; widening to
    float64 here, before any dense linear algebra, is what lets `transfer.py`
    match a float64 fixed-point iteration to 1e-9 -- see
    `tests_python/test_transfer.py`).
    """

    adjacency: np.ndarray
    input_matrix: np.ndarray
    output_matrix: np.ndarray


def build_dense_matrices(graph: "binfmt.GraphArrays") -> DenseGraphMatrices:
    neuron_count = int(graph.metadata["neuronCount"])
    input_channel_count = int(graph.metadata["inputChannelCount"])
    output_population_count = int(graph.metadata["outputPopulationCount"])

    adjacency = np.zeros((neuron_count, neuron_count), dtype=np.float64)
    offsets = graph.presynaptic_offsets
    post_indices = graph.postsynaptic_indices
    magnitudes = graph.contact_magnitudes.astype(np.float64)
    signs = graph.presynaptic_signs.astype(np.float64)
    if neuron_count > 0 and int(offsets[-1]) > 0:
        # `pre_of_edge[e]` is the presynaptic neuron owning CSR edge `e`,
        # derived from `presynapticOffsets` exactly as `rewire.py`'s own
        # `rewire_graph` derives it (see that function's "Expand CSR rows
        # into a flat edge list" comment) -- one row-length-repeated index
        # array, not a Python loop over `neuronCount` rows.
        row_lengths = np.diff(offsets.astype(np.int64))
        pre_of_edge = np.repeat(np.arange(neuron_count, dtype=np.int64), row_lengths)
        adjacency[post_indices.astype(np.int64), pre_of_edge] = signs[pre_of_edge] * magnitudes

    input_matrix = np.zeros((neuron_count, input_channel_count), dtype=np.float64)
    channel_index = graph.input_channel_index
    has_channel = channel_index >= 0
    input_matrix[np.nonzero(has_channel)[0], channel_index[has_channel]] = graph.input_weight[
        has_channel
    ].astype(np.float64)

    output_matrix = np.zeros((output_population_count, neuron_count), dtype=np.float64)
    population_index = graph.output_population_index
    has_population = population_index >= 0
    output_matrix[population_index[has_population], np.nonzero(has_population)[0]] = (
        graph.output_weight[has_population].astype(np.float64)
    )

    return DenseGraphMatrices(adjacency=adjacency, input_matrix=input_matrix, output_matrix=output_matrix)


def disconnected_graph_arrays(graph: "binfmt.GraphArrays") -> "binfmt.GraphArrays":
    """The disconnected negative control, mirroring `format.ts`'s
    `createDisconnectedGraph`: same neuron set, same `presynapticSigns`/
    `inputChannelIndex`/`inputWeight`/`outputPopulationIndex`/`outputWeight`,
    every recurrent edge removed (`edgeCount = 0`, `presynapticOffsets` all
    zero). Used by `features.py`'s CLI so the disconnected control's feature
    set is computed through the exact same `graph_features` code path as
    every other graph, rather than a second, hand-maintained list of "what
    every feature trivially evaluates to with no edges" that could silently
    drift from the real feature list."""
    neuron_count = int(graph.metadata["neuronCount"])
    return binfmt.GraphArrays(
        metadata={**graph.metadata, "edgeCount": 0},
        biological_ids=graph.biological_ids.copy(),
        presynaptic_offsets=np.zeros(neuron_count + 1, dtype=np.uint32),
        postsynaptic_indices=np.zeros(0, dtype=np.uint32),
        contact_magnitudes=np.zeros(0, dtype=np.float32),
        presynaptic_signs=graph.presynaptic_signs.copy(),
        input_channel_index=graph.input_channel_index.copy(),
        input_weight=graph.input_weight.copy(),
        output_population_index=graph.output_population_index.copy(),
        output_weight=graph.output_weight.copy(),
    )


def canonical_json_text(payload: object) -> str:
    """Sorted-key, 2-space-indented JSON with a trailing newline -- matches
    `rewire_batch.py`/`compile.py`/`positions.py`'s own
    `json.dumps(..., indent=2, sort_keys=True) + "\\n"` convention (float
    formatting is Python's default `repr`-based `json` encoder, which this
    study's WP2 change surface calls "fixed float formatting via `repr`").
    Every value must already be a native `str`/`int`/`float`/`bool`/`None`/
    `list`/`dict` -- callers must convert numpy scalars (`float(x)`,
    `int(x)`, `bool(x)`) before calling this, since `json` does not know how
    to serialize `numpy.float64`/`numpy.bool_` on its own.

    `allow_nan=False`: a singular or overflowing computation (an `inf`
    condition number, a `nan` transfer entry) must fail loudly here, at
    write time, rather than silently emitting `Infinity`/`NaN` -- not valid
    JSON, and a downstream `JSON.parse` would fail far from the actual
    cause. A review finding (dual review, WP2).
    """
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_canonical_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fsutil.atomic_write_text(path, canonical_json_text(payload))


# ---------------------------------------------------------------------------
# CLI plumbing shared by `transfer.py` and `features.py`
#
# Both CLIs read the same `rewire_batch.py` index.json, build the same
# (biological, disconnected, rewired-<seed>...) job list against it, and
# pre-flight-verify every job's graph file's sha256 before submitting any
# work to a process pool -- extracted here (a thermo-maintainability review
# finding on the WP2 diff: both files are new in the same PR, so there is no
# cross-PR ownership reason to have duplicated this ~70 lines of identical
# logic instead of sharing it from the start). Each CLI still owns its own
# `_parse_args` (via `base_arg_parser`, extended with any tool-specific
# flags such as `transfer.py`'s `--steady-state-dir`) and its own `main()`
# job-dispatch loop, since what happens *to* each job (a transfer solve vs a
# feature computation, and what gets written out) is genuinely different
# between the two tools.
# ---------------------------------------------------------------------------


def base_arg_parser(description: str | None) -> argparse.ArgumentParser:
    """The `--index`/`--graphs-dir`/`--biological`/`--skip-biological`/
    `--out`/`--workers` flags every analysis CLI takes, identically named
    and defaulted. Callers call `parser.add_argument(...)` for any
    tool-specific flags (e.g. `transfer.py`'s `--steady-state-dir`) after
    this returns, then `parser.parse_args(argv)`."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--index", type=Path, required=True, help="rewire_batch.py index.json")
    parser.add_argument("--graphs-dir", type=Path, required=True, help="directory holding rewired .bin.gz files")
    parser.add_argument(
        "--biological",
        type=Path,
        default=None,
        help="biological source .bin.gz (default: public/data/<index.sourceArtifact>); "
        "also computes the disconnected control",
    )
    parser.add_argument(
        "--skip-biological",
        action="store_true",
        help="omit biological/disconnected entirely (rewired graphs only)",
    )
    parser.add_argument("--out", type=Path, required=True, help="combined output JSON path")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="process-pool workers (each pinned to single-threaded BLAS); default min(8, cpu_count)",
    )
    return parser


def read_rewire_index(path: Path, tool_name: str) -> dict:
    """Load and shallow-validate a `rewire_batch.py` `index.json`. `tool_name`
    (e.g. `"transfer"`/`"features"`) prefixes the error message, matching
    each CLI's own error-message convention."""
    with path.open("r") as fh:
        index = json.load(fh)
    for required in ("sourceArtifact", "sourceSha256", "rewireSourceSha256", "seeds"):
        if required not in index:
            raise ValueError(f"{tool_name}: {path} is missing '{required}'")
    return index


def build_jobs(args: argparse.Namespace, index: dict, tool_name: str) -> list[tuple[str, Path, str, bool]]:
    """The (graphId, path, expectedSha256, isDisconnected) job list every
    analysis CLI submits to its process pool: the biological source and its
    disconnected control (unless `--skip-biological`), then every rewiring
    in `index["seeds"]`, sorted by seed for deterministic job order (and
    therefore deterministic output across `--workers` settings, since the
    dict each CLI's `main()` populates from these jobs is keyed by
    `graphId`, not insertion order). Does *not* check
    `args.biological`/`args.skip_biological` mutual exclusion -- each
    caller's `main()` does that before calling `read_rewire_index`, so the
    error fires before the (potentially slow) index read rather than after
    it, matching this repo's existing CLI behavior."""
    jobs: list[tuple[str, Path, str, bool]] = []
    if not args.skip_biological:
        # Default matches `regime-check.ts --biological`'s own resolution
        # (`resolve(PUBLIC_DATA_DIR, index.sourceArtifact)`).
        biological_path = args.biological if args.biological is not None else PUBLIC_DATA_DIR / index["sourceArtifact"]
        if not biological_path.exists():
            raise SystemExit(f"{tool_name}: biological source {biological_path} does not exist")
        jobs.append(("biological", biological_path, index["sourceSha256"], False))
        jobs.append(("disconnected", biological_path, index["sourceSha256"], True))

    seeds = sorted(index["seeds"], key=lambda entry: entry["seed"])
    for entry in seeds:
        jobs.append((f"rewired-{entry['seed']}", args.graphs_dir / entry["artifact"], entry["binarySha256"], False))
    return jobs


def verify_jobs(jobs: list[tuple[str, Path, str, bool]], tool_name: str) -> None:
    """Pre-flight decompressed-sha256 verification for every job, before any
    pool worker is submitted -- mirrors `null-evaluate.ts`'s
    `verifyRewiredFiles`'s "fails in seconds rather than after however much
    of a multi-hour run has already completed" up-front check. Aggregates
    every mismatch into one error, the same "report everything wrong at
    once" convention `verifyRewiredFiles` uses."""
    mismatches: list[str] = []
    checked: set[Path] = set()
    for graph_id, path, expected_sha256, _is_disconnected in jobs:
        if path in checked:
            continue  # biological and disconnected share the same source file
        checked.add(path)
        try:
            load_verified_graph(path, expected_sha256)
        except (OSError, GraphVerificationError, binfmt.InvalidGraphError) as error:
            mismatches.append(f"{graph_id}: {error}")
    if mismatches:
        raise GraphVerificationError(
            f"{tool_name}: {len(mismatches)} graph file(s) failed pre-flight verification:\n" + "\n".join(mismatches)
        )
