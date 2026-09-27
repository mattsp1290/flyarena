"""Shared plumbing for `scripts/data/`'s "sidecar" scripts (`positions.py`,
`descending_types.py`, and any future ones): joining pinned MaleCNS
annotation columns onto the already-compiled graph after the fact, and
publishing the result as a small, sha-verified JSON artifact plus a
manifest key.

This is itself a **non-compiler sidecar** (`compile.py`'s
`NON_COMPILER_SIDECAR_FILENAMES`): every function here only reads a pinned
source file, the already-compiled graph, or an existing manifest/ledger --
none of it ever writes to or otherwise influences the compiled `.bin.gz`
artifact's bytes -- so adding or editing this file must not change
`compilerSourceSha256`. See docs/data-provenance.md's "Soma positions
sidecar" and "Descending cell-type sidecar" sections.

What stays local to each sidecar (deliberately *not* extracted here --
`descending_types.py`'s own docstring: "this module never imports
positions.py, so the two sidecars can evolve without coupling" -- still
holds for the join logic, just not for this file's plumbing): which
annotation columns are required, which per-row values count as missing, the
join's fallback order or field selection, and the exact shape of the
emitted document. Extracted here is only the generic "safely publish a
verified sidecar artifact" plumbing every sidecar in this family needs
regardless of its own schema -- verifying a pinned source file's hash
before reading it, loading the compiled graph and cross-checking it against
the manifest, and atomically writing a JSON document plus merging one key
into an existing manifest/ledger. Before this module existed,
`positions.py` and `descending_types.py` each carried independent copies of
this ~130-line verification/orchestration block; a bug fixed in one would
not have been fixed in the other, and a third sidecar would have copied it
a third time (thermo maintainability review, WP1c).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.feather as feather

sys.path.insert(0, str(Path(__file__).resolve().parent))
import binfmt  # noqa: E402
import download  # noqa: E402
import fsutil  # noqa: E402
import rewire  # noqa: E402


def pinned_source_sha256(filename: str) -> str:
    """The sha256 `download.py` pins for `filename`, looked up from its
    `SOURCE_FILES`. Raises if `filename` isn't one of the pinned sources --
    the pinned list must be extended (URL + sha256 + size) before any
    sidecar can read it."""
    for source in download.SOURCE_FILES:
        if source.filename == filename:
            return source.sha256
    raise RuntimeError(
        f"{filename} is not among download.py's pinned SOURCE_FILES; "
        "the pinned list must be extended (URL + sha256 + size) before this script can run"
    )


def load_verified_source_table(
    filename: str,
    raw_dir: Path,
    *,
    expected_sha256: str,
    purpose: str,
) -> "tuple[pa.Table, str]":
    """Load `filename` from `raw_dir` as a pyarrow Table, refusing to
    proceed if its sha256 doesn't match `expected_sha256` -- the fail-closed
    check every sidecar in this family performs before joining anything.
    Returns `(table, actual_sha256)`.

    This function only verifies the file's identity; it never inspects the
    table's columns -- each sidecar performs its own schema-specific column
    checks (e.g. `positions.py`'s somaLocation/tosomaLocation
    list&lt;integer&gt; check, `descending_types.py`'s group-is-numeric check)
    on the returned table before converting it to pandas, since those
    checks differ per sidecar and are not this module's concern.

    `expected_sha256` is required, not defaulted here: each caller resolves
    its own default (normally `download.py`'s pin, via
    `pinned_source_sha256`) through a thin per-module wrapper function
    (e.g. `positions._pinned_annotations_sha256`), so that tests can
    monkeypatch that resolution per-module without reaching into this
    shared module."""
    path = raw_dir / filename
    if not path.exists():
        raise RuntimeError(f"{path} does not exist; run scripts/data/download.py first")
    actual_sha256 = download._sha256_of_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{path} sha256 {actual_sha256} does not match the pinned {expected_sha256}; "
            f"refusing to join {purpose} from a stale/tampered/unexpected file"
        )
    table = feather.read_table(path)
    return table, actual_sha256


def load_graph_for_sidecar(graph_path: Path) -> "tuple[binfmt.GraphArrays, str]":
    """Read, decompress, and decode the compiled graph at `graph_path`,
    printing the same progress line every sidecar in this family prints.
    Returns `(graph, graph_sha256)`, where `graph_sha256` is the sha256 of
    the still-gzipped `--graph` bytes -- the same value the manifest records
    as `gzipSha256`."""
    print(f"Loading graph from {graph_path}...")
    gzip_bytes = graph_path.read_bytes()
    graph_sha256 = binfmt.sha256_hex(gzip_bytes)
    binary = gzip.decompress(gzip_bytes)
    graph = rewire.decode_graph_binary(binary)
    print(f"  neuronCount={graph.metadata['neuronCount']}, graph sha256(gzip)={graph_sha256}")
    return graph, graph_sha256


def load_json_if_present(path: Path) -> "Optional[dict]":
    """`json.loads(path.read_text())`, or `None` if `path` doesn't exist --
    the tolerant-missing-manifest/ledger convention every sidecar in this
    family follows (a hand-built test fixture, or a run before `compile.py`
    has ever produced one, simply gets no manifest/ledger update rather than
    a crash)."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def cross_check_manifest_graph_sha(
    manifest: "Optional[dict]",
    graph_sha256: str,
    *,
    graph_path: Path,
    manifest_path: Path,
    entry_name: str,
) -> None:
    """Refuse (raise, write nothing) if `manifest` exists and its own
    `gzipSha256` doesn't match `graph_sha256` -- otherwise a stale or
    mismatched `--graph` could attach `entry_name` to a manifest that
    describes a different compiled artifact than the one the rest of the
    manifest describes. A no-op when `manifest` is `None` (nothing to
    cross-check against yet)."""
    if manifest is not None and manifest.get("gzipSha256") != graph_sha256:
        raise RuntimeError(
            f"--graph {graph_path} has sha256 {graph_sha256}, which does not match "
            f"{manifest_path}'s gzipSha256 ({manifest.get('gzipSha256')!r}); refusing to attach "
            f"{entry_name} to a manifest that describes a different compiled graph"
        )


def write_sidecar_artifact(payload: str, out_path: Path) -> str:
    """Atomically write `payload` (a complete, already-rendered JSON
    document, trailing newline included) to `out_path`, creating its parent
    directory if needed, and return the payload's own sha256 hex digest."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fsutil.atomic_write_text(out_path, payload)
    payload_bytes = payload.encode("utf-8")
    print(f"Wrote {out_path} ({len(payload_bytes)} bytes)")
    return hashlib.sha256(payload_bytes).hexdigest()


def merge_json_entry(
    doc: "Optional[dict]",
    doc_path: Path,
    key: str,
    value: object,
    *,
    label: str,
) -> None:
    """If `doc` is not `None`, set `doc[key] = value` and atomically
    rewrite `doc_path` (sorted keys, matching `compile.py`'s own JSON
    convention), touching only that one key. If `doc` is `None` (no
    manifest/ledger present), print a `[warn]` and skip -- the same
    tolerant-missing convention `load_json_if_present` documents. `label`
    names what's being updated in the printed message (e.g. "positions
    entry", "positionsCoverage")."""
    if doc is not None:
        doc[key] = value
        fsutil.atomic_write_text(doc_path, json.dumps(doc, indent=2, sort_keys=True) + "\n")
        print(f"Updated {doc_path} with the {label}")
    else:
        print(f"[warn] {doc_path} does not exist; skipped {label} update")
