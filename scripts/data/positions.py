"""Sidecar: joins the pinned body-annotations table's soma-location columns
onto the compiled graph's own neuron index order, producing
`public/data/malecns-arena-v1.positions.json` -- never claims a position for
a neuron the source data doesn't actually place.

This is intentionally *not* part of "the compiler"
(`binfmt.py`/`compile.py`/`download.py`/`rewire.py`): it only reads the
already-compiled graph's `biologicalIds` (via `rewire.decode_graph_binary`)
to determine join order and role, and never writes to or otherwise
influences the `.bin.gz` artifact's bytes. `compile.py`'s
`COMPILER_SOURCE_FILENAMES` deliberately excludes this file so that adding
or editing it does not change `compilerSourceSha256` or force an unrelated
recompile -- see docs/data-provenance.md's "Soma positions sidecar" section
for the full rationale.

Units: MaleCNS's own documentation (male-cns.janelia.org/download) states
the dataset's EM/segmentation volume is 8nm isotropic and that the sibling
`syn-points` table's coordinate columns are "voxel units, i.e. 8nm"; the
`somaLocation`/`tosomaLocation` field names are the standard neuPrint
`Neuron.somaLocation` convention, documented upstream (neuprint-python) as
voxel coordinates. However, male-cns.janelia.org's own pages do not
explicitly state the unit for `somaLocation` itself, so per this project's
policy of never guessing silently, `UNITS` below is recorded as
unverified rather than assumed to be 8nm voxels.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather

sys.path.insert(0, str(Path(__file__).resolve().parent))
import binfmt  # noqa: E402
import rewire  # noqa: E402
from download import SOURCE_FILES, _sha256_of_file  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"
PUBLIC_DATA_DIR = REPO_ROOT / "public" / "data"
ARTIFACT_NAME = "malecns-arena-v1"

#: The one annotations file this script reads. Must be one of
#: download.py's SOURCE_FILES so its pinned sha256 can be re-verified here
#: before anything is joined -- see `load_annotations_verified`.
ANNOTATIONS_FILENAME = "body-annotations-male-cns-v1.0-minconf-0.5.feather"

#: See this module's docstring for the citation trail. Recorded verbatim
#: (not silently assumed) per docs/data-provenance.md's units policy.
UNITS = "dataset voxel units (unverified)"

#: Fallback order for a neuron's emitted `xyz` position: the measured soma
#: location first, then the soma-tract location (present for a small
#: minority of bodies whose soma itself wasn't segmented/located), then no
#: position at all. Never centroid-imputed or otherwise fabricated.
POSITION_SOURCE_SOMA = "soma"
POSITION_SOURCE_TOSOMA = "tosoma"
POSITION_SOURCE_NONE = "none"


def _pinned_annotations_sha256() -> str:
    for source in SOURCE_FILES:
        if source.filename == ANNOTATIONS_FILENAME:
            return source.sha256
    raise RuntimeError(
        f"{ANNOTATIONS_FILENAME} is not among download.py's pinned SOURCE_FILES; "
        "the pinned list must be extended (URL + sha256 + size) before this script can run"
    )


#: Columns `build_positions` requires. A table missing any of these would
#: otherwise silently produce an all-`"none"` positions artifact (every
#: neuron "not found") rather than failing -- exactly the "never guess
#: silently" case this module's docstring commits to avoiding. Checked
#: eagerly in `build_positions`, not lazily per-row.
REQUIRED_ANNOTATION_COLUMNS = ("bodyId", "somaLocation", "tosomaLocation")


def load_annotations_verified(
    raw_dir: Path = RAW_DATA_DIR, expected_sha256: Optional[str] = None
) -> "tuple[pd.DataFrame, str]":
    """Load the body-annotations feather table, refusing to proceed if its
    sha256 doesn't match the pinned value (`expected_sha256`, defaulting to
    download.py's pin for the real filename -- overridable so tests can
    point this at a small fixture file with its own expected hash).
    Returns `(dataframe, verified_sha256)`.

    Also asserts, for whichever of `somaLocation`/`tosomaLocation` are
    present, that the column's Arrow type is `list<integer>` before
    converting to pandas: a schema that ever changed to a float or
    variable-precision type could otherwise silently truncate or lose
    precision on the `int(...)` coercion in `_location_or_none` below,
    breaking this module's "coordinates are copied exactly" guarantee
    without ever raising."""
    path = raw_dir / ANNOTATIONS_FILENAME
    if not path.exists():
        raise RuntimeError(f"{path} does not exist; run scripts/data/download.py first")
    actual_sha256 = _sha256_of_file(path)
    pinned_sha256 = expected_sha256 if expected_sha256 is not None else _pinned_annotations_sha256()
    if actual_sha256 != pinned_sha256:
        raise RuntimeError(
            f"{path} sha256 {actual_sha256} does not match the pinned {pinned_sha256}; "
            "refusing to join positions from a stale/tampered/unexpected file"
        )
    table = feather.read_table(path)
    for column in ("somaLocation", "tosomaLocation"):
        if column not in table.column_names:
            continue  # build_positions raises its own clearer error for a fully-missing column
        field_type = table.schema.field(column).type
        if not (pa.types.is_list(field_type) and pa.types.is_integer(field_type.value_type)):
            raise RuntimeError(
                f"{path}'s {column} column has Arrow type {field_type}, expected a "
                "list<integer> column; refusing to coerce coordinates that might not be exact integers"
            )
    return table.to_pandas(), actual_sha256


def _location_or_none(value: object, *, body: int, column: str) -> "Optional[list[int]]":
    """Returns the `[x, y, z]` integer coordinate from `value`, or `None` if
    the cell is genuinely absent. Feather round-trips a missing
    `list[int64]` cell as Python `None` (object dtype), not `NaN` -- a
    stray float NaN is tolerated defensively.

    Raises on anything else that is not a clean 3-element integer
    coordinate (wrong length, or a null/NaN component inside an otherwise-
    present list) rather than silently treating a malformed cell the same
    as a legitimately-absent one -- a malformed cell is a data-integrity
    problem this module should surface, not paper over with a fallback."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    arr = np.asarray(value, dtype=object)
    if arr.shape != (3,):
        raise RuntimeError(
            f"bodyId {body}: {column} has shape {arr.shape} (expected a 3-element "
            f"coordinate or a fully-missing cell): {value!r}"
        )
    coordinate: "list[int]" = []
    for component in arr.tolist():
        if component is None or (isinstance(component, float) and pd.isna(component)):
            raise RuntimeError(
                f"bodyId {body}: {column} has a missing component inside an otherwise-present "
                f"3-element coordinate: {value!r}"
            )
        coordinate.append(int(component))
    return coordinate


def build_positions(graph: binfmt.GraphArrays, annotations: pd.DataFrame) -> dict:
    """Join `annotations` onto `graph.biological_ids`, in that exact index
    order, deriving each neuron's role from the graph's own
    inputChannelIndex/outputPopulationIndex arrays (never from the
    annotations table) and its position from
    somaLocation -> tosomaLocation -> none (never fabricated/imputed).
    Returns the fields `malecns-arena-v1.positions.json` needs, minus the
    top-level provenance fields (`version`/`sourceFile`/`sourceSha256`/
    `graphSha256`/`units`) the caller adds.

    Raises rather than silently degrading to an all-`"none"` artifact when
    the annotations table is missing a required column, or when a compiled
    neuron's bodyId has no row in the table at all -- every graph node
    comes from this same table's `traced` subset (`compile.py`'s
    `select_subgraph`), so a missing row means the graph and the
    annotations table have diverged (or, e.g., `bodyId` was read back with
    an unexpected dtype), not that the neuron legitimately has no data."""
    missing_columns = [c for c in REQUIRED_ANNOTATION_COLUMNS if c not in annotations.columns]
    if missing_columns:
        raise RuntimeError(
            f"annotations table is missing required column(s) {missing_columns}; refusing to "
            "silently emit an all-'none' positions artifact"
        )
    if annotations["bodyId"].duplicated().any():
        dup = sorted(annotations.loc[annotations["bodyId"].duplicated(), "bodyId"].unique().tolist())
        raise RuntimeError(f"annotations table has duplicate bodyId(s): {dup[:10]}")

    by_body = annotations.set_index("bodyId")

    neuron_count = int(graph.metadata["neuronCount"])
    graph_body_ids = [int(b) for b in graph.biological_ids]
    missing_rows = [b for b in graph_body_ids if b not in by_body.index]
    if missing_rows:
        raise RuntimeError(
            f"{len(missing_rows)} compiled neuron bodyId(s) have no row in the annotations table "
            "at all (graph/annotations divergence, or a bodyId dtype mismatch such as strings vs "
            f"integers): {missing_rows[:10]}"
        )

    body_ids: list[str] = []
    roles: list[str] = []
    position_sources: list[str] = []
    xyz: "list[Optional[list[int]]]" = []
    soma_count = 0
    tosoma_count = 0
    none_count = 0

    for i in range(neuron_count):
        body = graph_body_ids[i]
        body_ids.append(str(body))

        is_sensory = int(graph.input_channel_index[i]) >= 0
        is_descending = int(graph.output_population_index[i]) >= 0
        if is_sensory and is_descending:
            raise RuntimeError(
                f"neuron index {i} (bodyId {body}) has both an input channel and an output "
                "population assigned; a neuron must not be both sensory and descending"
            )
        roles.append("sensory" if is_sensory else ("descending" if is_descending else "bridge"))

        row = by_body.loc[body]
        soma = _location_or_none(row["somaLocation"], body=body, column="somaLocation")
        tosoma = _location_or_none(row["tosomaLocation"], body=body, column="tosomaLocation")

        if soma is not None:
            xyz.append(soma)
            position_sources.append(POSITION_SOURCE_SOMA)
            soma_count += 1
        elif tosoma is not None:
            xyz.append(tosoma)
            position_sources.append(POSITION_SOURCE_TOSOMA)
            tosoma_count += 1
        else:
            xyz.append(None)
            position_sources.append(POSITION_SOURCE_NONE)
            none_count += 1

    role_counts = {
        "sensory": roles.count("sensory"),
        "bridge": roles.count("bridge"),
        "descending": roles.count("descending"),
    }
    coverage = {"soma": soma_count, "tosoma": tosoma_count, "none": none_count}

    return {
        "bodyIds": body_ids,
        "role": roles,
        "positionSource": position_sources,
        "xyz": xyz,
        "coverage": coverage,
        "roleCounts": role_counts,
    }


def render_positions_document(
    *,
    source_sha256: str,
    graph_sha256: str,
    fields: dict,
) -> str:
    """Deterministic JSON rendering: sorted keys, fixed separators, trailing
    newline. Two calls with the same inputs produce byte-identical output."""
    document = {
        "version": 1,
        "sourceFile": ANNOTATIONS_FILENAME,
        "sourceSha256": source_sha256,
        "graphSha256": graph_sha256,
        "units": UNITS,
        "bodyIds": fields["bodyIds"],
        "role": fields["role"],
        "positionSource": fields["positionSource"],
        "xyz": fields["xyz"],
        "coverage": fields["coverage"],
        "roleCounts": fields["roleCounts"],
    }
    return json.dumps(document, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"


def _atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` via a same-directory temp file plus
    `os.replace`, so a crash mid-write (or a `KeyboardInterrupt`) can never
    leave `path` truncated or half-written -- `path` either has its old
    contents or its new ones, never something in between."""
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--graph", type=Path, default=PUBLIC_DATA_DIR / f"{ARTIFACT_NAME}.bin.gz")
    parser.add_argument("--out-dir", type=Path, default=PUBLIC_DATA_DIR)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--ledger-path", type=Path, default=None)
    args = parser.parse_args(argv)

    print(f"Loading graph from {args.graph}...")
    gzip_bytes = args.graph.read_bytes()
    graph_sha256 = binfmt.sha256_hex(gzip_bytes)
    binary = gzip.decompress(gzip_bytes)
    graph = rewire.decode_graph_binary(binary)
    print(f"  neuronCount={graph.metadata['neuronCount']}, graph sha256(gzip)={graph_sha256}")

    # Load (but don't yet write) the manifest/ledger up front, so every
    # validation below runs -- and can fail loudly -- before anything on
    # disk is touched, and so the --graph/manifest cross-check and the
    # roleCounts/selectionCounts cross-check both have what they need.
    manifest_path = args.manifest_path or (args.out_dir / f"{ARTIFACT_NAME}.manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if manifest is not None and manifest.get("gzipSha256") != graph_sha256:
        raise RuntimeError(
            f"--graph {args.graph} has sha256 {graph_sha256}, which does not match "
            f"{manifest_path}'s gzipSha256 ({manifest.get('gzipSha256')!r}); refusing to attach "
            "a positions entry to a manifest that describes a different compiled graph"
        )

    ledger_path = args.ledger_path or (args.out_dir / f"{ARTIFACT_NAME}.ledger.json")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else None

    print("Loading + verifying pinned body-annotations...")
    annotations, source_sha256 = load_annotations_verified(args.raw_dir)
    print(f"  {len(annotations)} rows, sha256 verified: {source_sha256}")

    print("Joining soma positions onto the graph's neuron order...")
    fields = build_positions(graph, annotations)
    print(f"  coverage: {json.dumps(fields['coverage'], sort_keys=True)}")
    print(f"  roleCounts: {json.dumps(fields['roleCounts'], sort_keys=True)}")

    # Defense in depth: `fields["roleCounts"]` is derived purely from the
    # graph's own channel/population arrays (see build_positions), so this
    # can only disagree with the ledger if the two describe different
    # compiled graphs, or if a future edit to the bridge/sensory/descending
    # classification above silently drifted from compile.py's
    # `select_subgraph` policy. Either way, that is exactly the class of
    # silent-drift bug this whole artifact exists to prevent.
    selection_counts = (ledger or {}).get("selectionCounts")
    if selection_counts is not None:
        expected_role_counts = {
            "sensory": selection_counts.get("sensorySelectedCount"),
            "bridge": selection_counts.get("bridgeSelectedCount"),
            "descending": selection_counts.get("descendingSelectedCount"),
        }
        if expected_role_counts != fields["roleCounts"]:
            raise RuntimeError(
                f"roleCounts {fields['roleCounts']} does not match {ledger_path}'s "
                f"selectionCounts {expected_role_counts}"
            )

    payload = render_positions_document(
        source_sha256=source_sha256, graph_sha256=graph_sha256, fields=fields
    )
    payload_bytes = payload.encode("utf-8")
    positions_sha256 = hashlib.sha256(payload_bytes).hexdigest()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{ARTIFACT_NAME}.positions.json"
    _atomic_write_text(out_path, payload)
    print(f"Wrote {out_path} ({len(payload_bytes)} bytes)")
    print(f"positions sha256: {positions_sha256}")

    if manifest is not None:
        manifest["positions"] = {
            "artifact": out_path.name,
            "sha256": positions_sha256,
            "coverage": fields["coverage"],
        }
        _atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"Updated {manifest_path} with the positions entry")
    else:
        print(f"[warn] {manifest_path} does not exist; skipped manifest update")

    if ledger is not None:
        ledger["positionsCoverage"] = fields["coverage"]
        _atomic_write_text(ledger_path, json.dumps(ledger, indent=2, sort_keys=True) + "\n")
        print(f"Updated {ledger_path} with positionsCoverage")
    else:
        print(f"[warn] {ledger_path} does not exist; skipped ledger update")

    return 0


if __name__ == "__main__":
    sys.exit(main())
