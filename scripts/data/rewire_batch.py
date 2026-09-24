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
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import binfmt  # noqa: E402
import rewire  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_DATA_DIR = REPO_ROOT / "public" / "data"
ARTIFACT_NAME = "malecns-arena-v1"


def parse_seed_range(spec: str) -> range:
    """Parse a `START:END` spec into a half-open `range` (end-exclusive, so
    `0:500` yields the 500 seeds `0..499` -- matches the plan's "rewiring
    seeds 0..499" and Python's own slice convention)."""
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError(f"--seeds must be START:END, got {spec!r}")
    try:
        start, end = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"--seeds must be START:END with integer bounds, got {spec!r}") from exc
    if end <= start:
        raise ValueError(f"--seeds END must be greater than START, got {spec!r}")
    return range(start, end)


def compiler_source_sha256_of(path: Path) -> str:
    """sha256 of one compiler source file's raw bytes -- used here for
    `rewireSourceSha256` (`rewire.py` itself), the provenance record that
    lets a regenerated batch be tied back to the exact rewiring algorithm
    that produced it, the same way `compile.py`'s `compiler_source_sha256`
    ties a compiled artifact back to the whole compiler."""
    return binfmt.sha256_hex(path.read_bytes())


def run_batch(
    in_path: Path,
    seeds: "range",
    out_dir: Path,
) -> dict:
    """Rewire `in_path`'s graph once per seed in `seeds`, writing each
    output to `out_dir` and returning the `index.json`-shaped dict (not yet
    written to disk) describing every seed written, in seed order."""
    with gzip.open(in_path, "rb") as fh:
        source_binary = fh.read()
    graph = rewire.decode_graph_binary(source_binary)
    source_sha256 = binfmt.sha256_hex(source_binary)
    rewire_source_sha256 = compiler_source_sha256_of(Path(rewire.__file__).resolve())

    out_dir.mkdir(parents=True, exist_ok=True)

    seed_entries = []
    for seed in seeds:
        rewired, stats = rewire.rewire_graph(graph, seed=seed)
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
        "seeds": seed_entries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    index = run_batch(in_path=args.in_path, seeds=seed_range, out_dir=args.out_dir)

    index_out = args.index_out or (args.out_dir / "index.json")
    index_out.parent.mkdir(parents=True, exist_ok=True)
    # sort_keys + a fixed separator (via indent) matches rewire.py's own
    # manifest-writing convention, so two runs over the same seed range
    # produce byte-identical index.json (tests_python/test_rewire_batch.py).
    index_out.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")

    print(f"Wrote {index_out} ({len(index['seeds'])} seeds)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
