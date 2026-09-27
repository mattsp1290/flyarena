"""Lesion engine: `rewired:<seed>` graph regeneration
(`.agents/plans/graph-lab/02-job-engines.md`'s lesion engine: "`rewired:<seed>`
with seed 0-499: regenerated with `rewire.py` `rewire_graph` into the job's
tmpfs, with its gzip sha checked against `rewiring-null-v1.json`
`rewired[seed].gzipSha256`"). `biological`/`disconnected` need no Python-side
graph preparation -- `service.py`'s `default_runner` handles them directly,
unchanged from WP1. This module is only the `rewired:<seed>` branch, split
out so `default_runner` stays a thin dispatcher across all three engines
(lesion/atlas/swapset) rather than one growing function.

Never reimplements the rewiring algorithm: `rewire_graph`/`binfmt.
encode_graph_binary`/`binfmt.write_gzip_deterministic` are `scripts/data/
rewire.py`'s own, loaded through `py_scripts.load_rewire` (never a
hand-copied duplicate of that module).
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from . import py_scripts


def regenerate_rewired_graph(*, scripts_dir: Path, data_dir: Path, job_dir: Path, seed: int) -> "tuple[Path, str]":
    """Regenerate the `rewired:<seed>` graph binary into `job_dir` (the
    job's own tmpfs directory), verified against the published rewiring
    null. Returns `(gzip_path, binary_sha256)` -- `binary_sha256` is the
    **decompressed** binary's own sha256, which is what `entry-lesion.ts`'s
    `expectedSha256` checks (`loadVerifiedGraphBinary`) -- never the gzip
    sha, a different check on different bytes (`verify-search-graph.ts`'s
    own doc comment draws this same distinction for a different artifact).

    Raises `ValueError` (never returns a graph that failed any check) if:
    the biological source artifact's sha256 does not match the manifest;
    `rewiring-null-v1.json`'s own sha256 does not match the manifest's
    recorded value for it; `seed` has no published null entry, or that
    entry's own `seed` field disagrees with its list position; or the
    freshly regenerated graph's gzip sha256 does not match the published
    null's recorded value for this seed. The last case is the one this
    function exists to prevent: this is the only place a `rewired:<seed>`
    graph is ever regenerated for a live job, so silently serving a
    mismatched graph would mean an undetected drift between the rewiring
    algorithm (or its default parameters) and the already-published null
    every swap-set percentile and lesion `rewired` result implicitly
    trusts.
    """
    manifest = json.loads((data_dir / "malecns-arena-v1.manifest.json").read_text())

    rewiring_null_path = data_dir / "rewiring-null-v1.json"
    rewiring_null_raw = rewiring_null_path.read_bytes()
    py_scripts.require_sha256(
        rewiring_null_raw, manifest.get("rewiringNull", {}).get("sha256"), what="rewiring-null-v1.json"
    )
    rewired_entries = json.loads(rewiring_null_raw)["rewired"]
    if not (0 <= seed < len(rewired_entries)):
        raise ValueError(f"rewired seed {seed} has no published null entry")
    expected = rewired_entries[seed]
    if int(expected["seed"]) != seed:
        raise ValueError(f"rewiring-null-v1.json entry at index {seed} has seed {expected['seed']}, expected {seed}")

    biological_gzip_path = data_dir / manifest["artifact"]
    with gzip.open(biological_gzip_path, "rb") as fh:
        biological_binary = fh.read()
    py_scripts.require_sha256(biological_binary, manifest.get("binarySha256"), what="the biological graph binary")

    rewire = py_scripts.load_rewire(scripts_dir)
    graph = rewire.decode_graph_binary(biological_binary)
    rewired, _stats = rewire.rewire_graph(graph, seed=seed)
    rewired_binary = rewire.binfmt.encode_graph_binary(rewired)
    binary_sha256 = hashlib.sha256(rewired_binary).hexdigest()

    gzip_path = job_dir / f"rewired-seed{seed}.bin.gz"
    rewire.binfmt.write_gzip_deterministic(rewired_binary, gzip_path)
    gzip_sha256 = hashlib.sha256(gzip_path.read_bytes()).hexdigest()
    if gzip_sha256 != expected["gzipSha256"]:
        raise ValueError(
            f"regenerated rewired:{seed} gzip sha256 {gzip_sha256} does not match "
            f"the published null's {expected['gzipSha256']}"
        )
    return gzip_path, binary_sha256
