"""Batch entry point over `rewire.rewire_graph`: generate many degree-preserving
rewirings of one source graph in a single process invocation.

This module adds no rewiring logic of its own -- every graph it writes comes
from calling `rewire.rewire_graph` (the single canonical implementation,
`docs/graph-format.md`/`tests_python/test_compile.py`'s invariant suite) once
per requested seed, with its default `allow_self_loops`/
`swap_attempts_multiplier`. Its only job is process-level batching: read the
source graph once, loop over a seed range, and write one deterministic-gzip
artifact plus one `index.json` describing everything it wrote -- so 500
rewirings run as one `uv run` invocation (amortizing interpreter/venv
startup, `.agents/plans/rewiring-null/01-rewired-graph-generation.md`'s WP1)
instead of 500.

Unlike `rewire.py`'s own `main`, this script never reads or writes
`public/data/malecns-arena-v1.manifest.json` -- its output is a batch
artifact directory (gitignored, e.g. `training/runs/null/graphs/`), not a
shipped product asset. The one shipped rewired control arm
(`public/data/malecns-arena-v1-rewired-seed0.bin.gz` and its manifest entry)
continues to come from `rewire.py`'s own CLI; this script's seed-0 output is
required to be byte-identical to it (see `tests_python/test_rewire_batch.py`)
but is written to a separate path, not over it.

`index.json` is the authoritative listing of what a `--out-dir` contains --
a downstream reader (WP2's sharded evaluation) should read the seeds it
names rather than globbing `*-rewired-seed*.bin.gz` directly. Rerunning this
script into a non-empty `--out-dir` (a smaller seed range, or a different
`--in-path`) does not delete files from a previous run; only `index.json`
itself is replaced (atomically, so a killed process never leaves a
truncated or stale-but-corrupt one), and it always describes exactly the
seeds this invocation wrote. Point `--out-dir` at a fresh directory per
batch to avoid stale `.bin.gz` files sitting unlisted next to a new index.
"""

from __future__ import annotations

import argparse
import gzip
import inspect
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import binfmt  # noqa: E402
import fsutil  # noqa: E402
import rewire  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_DATA_DIR = REPO_ROOT / "public" / "data"
ARTIFACT_NAME = "malecns-arena-v1"

#: `rewire_graph`'s own `allow_self_loops` default, read off its signature
#: rather than duplicated as a literal here -- so if that default ever
#: changes, this batch script's recorded `params.allowSelfLoops` (below)
#: changes with it instead of silently going stale. `rewire.py` has no
#: separate `DEFAULT_ALLOW_SELF_LOOPS` constant (unlike
#: `DEFAULT_SWAP_ATTEMPTS_MULTIPLIER`), so the function's own default
#: parameter value is the single source of truth.
ALLOW_SELF_LOOPS = inspect.signature(rewire.rewire_graph).parameters["allow_self_loops"].default


def parse_seed_range(spec: str) -> range:
    """Parse a `START:END` spec into a half-open `range` (end-exclusive, so
    `0:500` yields the 500 seeds `0..499` -- matches the plan's "rewiring
    seeds 0..499" and Python's own slice convention). `START` must be
    non-negative: `rewire_graph`'s `np.random.default_rng(seed)` rejects a
    negative seed itself, but only after this script has already committed
    to the run, so this is rejected up front instead."""
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError(f"--seeds must be START:END, got {spec!r}")
    try:
        start, end = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"--seeds must be START:END with integer bounds, got {spec!r}") from exc
    if start < 0:
        raise ValueError(f"--seeds START must be non-negative, got {spec!r}")
    if end <= start:
        raise ValueError(f"--seeds END must be greater than START, got {spec!r}")
    return range(start, end)


def file_sha256(path: Path) -> str:
    """sha256 of one file's raw bytes -- used here for `rewireSourceSha256`
    (`rewire.py` itself), the provenance record that lets a regenerated
    batch be tied back to the exact rewiring algorithm that produced it.
    Deliberately not named after `compile.py`'s `compiler_source_sha256`
    (which hashes a whole directory of files, a different scheme) even
    though both feed a `*SourceSha256` provenance field."""
    return binfmt.sha256_hex(path.read_bytes())


def _load_source_graph(in_path: Path) -> "tuple[bytes, binfmt.GraphArrays]":
    """Read and decode `in_path`'s compiled graph, raising `OSError` (e.g.
    missing file), `gzip.BadGzipFile`, or `binfmt.InvalidGraphError` if it
    doesn't exist or doesn't parse as a valid graph. Shared by `main()` --
    to validate `--in-path` before any destructive write, see its call site
    -- and by `run_batch` itself."""
    with gzip.open(in_path, "rb") as fh:
        source_binary = fh.read()
    graph = rewire.decode_graph_binary(source_binary)
    return source_binary, graph


def run_batch(
    in_path: Path,
    seeds: range,
    out_dir: Path,
    *,
    preloaded_source: "tuple[bytes, binfmt.GraphArrays] | None" = None,
) -> dict:
    """Rewire `in_path`'s graph once per seed in `seeds`, writing each
    output to `out_dir` and returning the `index.json`-shaped dict (not yet
    written to disk) describing every seed written, in seed order.

    `preloaded_source`, if given, must be `_load_source_graph(in_path)`'s
    own return value -- passing it lets a caller that already loaded and
    validated `in_path` (e.g. `main()`, which must validate it before
    deleting any existing `index.json`) avoid reading and gzip-decoding the
    same file a second time. Left `None` (the default), `run_batch` loads
    it itself, unchanged from before -- this keeps `run_batch(in_path=...,
    seeds=..., out_dir=...)` a complete, self-sufficient call for every
    existing caller (tests, a future direct import)."""
    source_binary, graph = preloaded_source if preloaded_source is not None else _load_source_graph(in_path)
    source_sha256 = binfmt.sha256_hex(source_binary)
    rewire_source_sha256 = file_sha256(Path(rewire.__file__).resolve())
    # rewire_source_sha256 alone ties a batch back to the swap algorithm's
    # own code, but the output bytes also depend on binfmt.py's encoder
    # (encode_graph_binary/write_gzip_deterministic) and on which numpy
    # version generated the Generator stream `rewire_graph` draws from --
    # neither of which changing would move rewire_source_sha256. Recorded
    # alongside it so the index is self-describing about every input that
    # can change its own output.
    binfmt_source_sha256 = file_sha256(Path(binfmt.__file__).resolve())

    out_dir.mkdir(parents=True, exist_ok=True)

    seed_entries = []
    for seed in seeds:
        rewired, stats = rewire.rewire_graph(graph, seed=seed, allow_self_loops=ALLOW_SELF_LOOPS)
        rewired_binary = binfmt.encode_graph_binary(rewired)
        rewired_sha256 = binfmt.sha256_hex(rewired_binary)

        out_path = out_dir / f"{ARTIFACT_NAME}-rewired-seed{seed}.bin.gz"
        binfmt.write_gzip_deterministic(rewired_binary, out_path)
        gzip_bytes = out_path.read_bytes()
        gzip_sha256 = binfmt.sha256_hex(gzip_bytes)

        seed_entries.append(
            {
                "seed": seed,
                "artifact": out_path.name,
                "binarySha256": rewired_sha256,
                "binaryBytes": len(rewired_binary),
                "gzipSha256": gzip_sha256,
                "gzipBytes": len(gzip_bytes),
                "stats": stats,
            }
        )
        print(f"[seed {seed}] wrote {out_path.name} ({len(gzip_bytes)} bytes), binary sha256={rewired_sha256}")

    return {
        "sourceArtifact": in_path.name,
        "sourceSha256": source_sha256,
        "rewireSourceSha256": rewire_source_sha256,
        "binfmtSourceSha256": binfmt_source_sha256,
        "numpyVersion": np.__version__,
        "params": {
            "allowSelfLoops": ALLOW_SELF_LOOPS,
            "swapAttemptsMultiplier": rewire.DEFAULT_SWAP_ATTEMPTS_MULTIPLIER,
        },
        "seeds": seed_entries,
    }


def main(argv: list[str] | None = None) -> int:
    # Only the module docstring's first paragraph, not the full design
    # rationale (including gitignored-plan-file paths and provenance
    # detail), which belongs in the source but is noise in --help output.
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--in-path", type=Path, default=PUBLIC_DATA_DIR / f"{ARTIFACT_NAME}.bin.gz")
    parser.add_argument(
        "--seeds",
        type=str,
        required=True,
        help="Seed range START:END, end-exclusive (e.g. 0:500 for seeds 0..499)",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--index-out",
        type=Path,
        default=None,
        help="Defaults to <out-dir>/index.json",
    )
    args = parser.parse_args(argv)

    seed_range = parse_seed_range(args.seeds)

    # Validated *before* index_out is touched: a bad --in-path (missing,
    # not gzip, not a valid graph) must fail here, not after a previous
    # run's valid index.json has already been deleted below. The loaded
    # (source_binary, graph) pair is kept and handed to run_batch below
    # (preloaded_source) rather than discarded, so a valid --in-path is
    # only ever read and gzip-decoded once per invocation.
    try:
        preloaded_source = _load_source_graph(args.in_path)
    except (OSError, binfmt.InvalidGraphError) as exc:
        parser.error(f"--in-path {args.in_path} is not a readable, valid compiled graph: {exc}")

    # Removed up front, before any rewiring starts: if this run is
    # interrupted partway through, the directory is left with no index
    # (which a reader can detect) rather than the previous run's index --
    # possibly describing a different seed range or source graph than the
    # partial set of .bin.gz files now on disk (see this module's docstring).
    index_out = args.index_out or (args.out_dir / "index.json")
    index_out.parent.mkdir(parents=True, exist_ok=True)
    index_out.unlink(missing_ok=True)

    index = run_batch(
        in_path=args.in_path, seeds=seed_range, out_dir=args.out_dir, preloaded_source=preloaded_source
    )

    # sort_keys + a fixed separator (via indent) matches rewire.py's own
    # manifest-writing convention, so two runs over the same seed range
    # produce byte-identical index.json (tests_python/test_rewire_batch.py).
    # Written atomically (fsutil.atomic_write_text) so a process killed
    # mid-write never leaves a truncated, unparseable index.json.
    fsutil.atomic_write_text(index_out, json.dumps(index, indent=2, sort_keys=True) + "\n")

    print(f"Wrote {index_out} ({len(index['seeds'])} seeds)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
