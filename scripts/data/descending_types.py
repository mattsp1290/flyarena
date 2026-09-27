"""Sidecar: joins the pinned body-annotations table's cell-type columns
(`type`, `class`, `instance`, `group`, `somaSide`) onto the compiled graph's
own 48 descending (output-assigned) neurons, in graph index order,
producing `public/data/descending-types-v1.json` -- names for the readout's
D = 48 input neurons, never a claim about fly descending-neuron function.

Like `scripts/data/positions.py` (see its own docstring for the full
compiler/sidecar rationale), this is intentionally *not* part of "the
compiler" (`binfmt.py`/`compile.py`/`download.py`/`rewire.py`): it only
reads the already-compiled graph's `biologicalIds`/`outputPopulationIndex`
(via `rewire.decode_graph_binary`) to determine join order and population,
and never writes to or otherwise influences the `.bin.gz` artifact's bytes.
`compile.py`'s `COMPILER_SOURCE_FILENAMES` deliberately excludes this file
(it is listed in `NON_COMPILER_SIDECAR_FILENAMES` instead, alongside
`positions.py`) so that adding or editing it does not change
`compilerSourceSha256` or force an unrelated recompile -- see
docs/data-provenance.md's "Soma positions sidecar" section for the
analogous rationale this module follows.

Reads the same pinned MaleCNS annotations table `positions.py` reads
(`body-annotations-male-cns-v1.0-minconf-0.5.feather`, one of `download.py`'s
`SOURCE_FILES`), re-verifying its sha256 independently via the shared
`scripts/data/sidecar_io.py` loader (this module never imports
`positions.py` itself, so the two sidecars' *join logic* can evolve without
coupling -- see `sidecar_io.py`'s docstring for what plumbing the two
sidecars do share).

Cell-type annotations are external metadata, not something this project
measured: `type`/`class`/`instance`/`group`/`somaSide` are copied exactly as
the source table has them for each neuron, including `None` where the
source itself has no value -- this module never infers, imputes, or
otherwise fills in a missing type. The shipped graph (`.bin.gz`) is
unchanged; only this new sidecar and its manifest key are added.

Schema shape note: unlike `positions.json`'s structure-of-arrays
(`bodyIds[]`, `role[]`, ... all parallel-indexed), this artifact is
array-of-objects (`neurons: [{index, bodyId, ...}, ...]`) -- a deliberate
choice for 48 rows of five optional fields each, not an oversight; the two
sidecars' document shapes are allowed to differ per-artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parent))
import binfmt  # noqa: E402
# `rewire` is unused directly in this module's own logic (graph decoding now
# lives in `sidecar_io.load_graph_for_sidecar`) but is re-exported here so
# tests can call `descending_types.rewire.decode_graph_binary` directly,
# matching `positions.py`'s identical convention for `positions.rewire`.
import rewire  # noqa: E402
import sidecar_io  # noqa: E402
from download import _sha256_of_file  # noqa: E402  (re-exported for tests, matching positions.py)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"
PUBLIC_DATA_DIR = REPO_ROOT / "public" / "data"
ARTIFACT_NAME = "malecns-arena-v1"
OUT_ARTIFACT_NAME = "descending-types-v1"

#: The one annotations file this script reads -- the same file
#: `positions.py` reads. Must be one of download.py's SOURCE_FILES so its
#: pinned sha256 can be re-verified here before anything is joined -- see
#: `load_annotations_verified`.
ANNOTATIONS_FILENAME = "body-annotations-male-cns-v1.0-minconf-0.5.feather"

#: Columns `build_descending_types` requires to exist on the table (not to
#: be non-null per row -- see that function's docstring). A table missing
#: any of these would otherwise silently produce an all-null cell-type
#: artifact (every neuron "unlabeled") rather than failing -- exactly the
#: "never guess silently" case `positions.py`'s analogous check avoids.
#: Checked eagerly here, not lazily per-row.
REQUIRED_ANNOTATION_COLUMNS = ("bodyId", "type", "class", "instance", "group", "somaSide")

#: The optional (per-row-nullable) fields `build_descending_types` emits per
#: neuron, in the order `main()` reports missing-annotation counts -- also
#: the key order of the artifact's `nullCounts` summary object.
OPTIONAL_FIELDS = ("type", "class", "instance", "group", "somaSide")


def _pinned_annotations_sha256() -> str:
    """Thin per-module wrapper around `sidecar_io.pinned_source_sha256`,
    kept as a standalone module-level function (rather than inlined at each
    call site) so tests can `monkeypatch.setattr(descending_types,
    "_pinned_annotations_sha256", ...)` to point `load_annotations_verified`
    at a fixture's hash without touching the real `download.py` pin."""
    return sidecar_io.pinned_source_sha256(ANNOTATIONS_FILENAME)


def load_annotations_verified(
    raw_dir: Path = RAW_DATA_DIR, expected_sha256: Optional[str] = None
) -> "tuple[pd.DataFrame, str]":
    """Load the body-annotations feather table via `sidecar_io`'s shared
    sha-verified loader, refusing to proceed if its sha256 doesn't match the
    pinned value (`expected_sha256`, defaulting to `_pinned_annotations_sha256()`
    -- overridable so tests can point this at a small fixture file with its
    own expected hash). Returns `(dataframe, verified_sha256)`.

    Also asserts the `group` column (when present) is a numeric Arrow type,
    so a schema change to a non-numeric representation is caught here
    rather than silently producing garbage group ids on the `int(...)`
    coercion in `_group_or_none` below. This column check is
    `descending_types.py`-specific (the counterpart `positions.py` check is
    a different column, a different Arrow type), so it stays local here
    rather than moving into `sidecar_io`, which only verifies the file's
    identity."""
    pinned_sha256 = expected_sha256 if expected_sha256 is not None else _pinned_annotations_sha256()
    path = raw_dir / ANNOTATIONS_FILENAME
    table, actual_sha256 = sidecar_io.load_verified_source_table(
        ANNOTATIONS_FILENAME, raw_dir, expected_sha256=pinned_sha256, purpose="cell types"
    )
    if "group" in table.column_names:
        field_type = table.schema.field("group").type
        if not (pa.types.is_floating(field_type) or pa.types.is_integer(field_type)):
            raise RuntimeError(
                f"{path}'s group column has Arrow type {field_type}, expected a numeric "
                "type; refusing to coerce a group id that might not be exact"
            )
    return table.to_pandas(), actual_sha256


def _group_or_none(value: object, *, body: int) -> "Optional[int]":
    """Returns the integer `group` id from `value`, or `None` if the cell is
    genuinely absent (feather round-trips a missing double cell as NaN).
    Raises on a present-but-non-integral value rather than silently
    truncating it -- a fractional group id would be a data-integrity
    problem this module should surface, not paper over."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    numeric = float(value)
    as_int = int(numeric)
    if float(as_int) != numeric:
        raise RuntimeError(f"bodyId {body}: group has a non-integral value: {value!r}")
    return as_int


def _string_or_none(value: object) -> "Optional[str]":
    """Returns `value` as a plain `str`, or `None` if the cell is genuinely
    absent. Feather round-trips a missing string cell as Python `None`; a
    stray float NaN is tolerated defensively (mirrors `positions.py`'s
    `_location_or_none`)."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    return str(value)


def build_descending_types(graph: binfmt.GraphArrays, annotations: pd.DataFrame) -> "list[dict]":
    """Join `annotations` onto the graph's descending (output-assigned)
    neurons, in ascending graph-index order (`outputPopulationIndex[i] >=
    0`) -- the same order `src/lib/connectome/readout.ts`'s
    `outputNeuronIndices` produces, and the same order the readout's D = 48
    inputs are indexed by. `index` and `population` are derived purely from
    the graph's own `outputPopulationIndex` (never from the annotations
    table); `type`/`class`/`instance`/`group`/`somaSide` are copied exactly
    as the annotations table has them for that bodyId, with a missing value
    recorded as `None` -- never inferred or filled in.

    Raises rather than silently degrading to an all-null artifact when the
    annotations table is missing a required column, or when a descending
    neuron's bodyId has no row in the table at all -- every graph node
    comes from this same table's `traced` subset (`compile.py`'s
    `select_subgraph`), so a missing row means the graph and the
    annotations table have diverged, not that the neuron legitimately has
    no data."""
    missing_columns = [c for c in REQUIRED_ANNOTATION_COLUMNS if c not in annotations.columns]
    if missing_columns:
        raise RuntimeError(
            f"annotations table is missing required column(s) {missing_columns}; refusing to "
            "silently emit an all-null descending-types artifact"
        )
    if annotations["bodyId"].duplicated().any():
        dup = sorted(annotations.loc[annotations["bodyId"].duplicated(), "bodyId"].unique().tolist())
        raise RuntimeError(f"annotations table has duplicate bodyId(s): {dup[:10]}")

    by_body = annotations.set_index("bodyId")

    neuron_count = int(graph.metadata["neuronCount"])
    graph_body_ids = [int(b) for b in graph.biological_ids]
    output_population_index = graph.output_population_index

    descending_indices = [i for i in range(neuron_count) if int(output_population_index[i]) >= 0]

    descending_body_ids = [graph_body_ids[i] for i in descending_indices]
    missing_rows = [b for b in descending_body_ids if b not in by_body.index]
    if missing_rows:
        raise RuntimeError(
            f"{len(missing_rows)} descending neuron bodyId(s) have no row in the annotations table "
            "at all (graph/annotations divergence, or a bodyId dtype mismatch such as strings vs "
            f"integers): {missing_rows[:10]}"
        )

    neurons: "list[dict]" = []
    for i in descending_indices:
        body = graph_body_ids[i]
        row = by_body.loc[body]
        neurons.append(
            {
                "index": i,
                "bodyId": str(body),
                "population": int(output_population_index[i]),
                "type": _string_or_none(row["type"]),
                "class": _string_or_none(row["class"]),
                "instance": _string_or_none(row["instance"]),
                "group": _group_or_none(row["group"], body=body),
                "somaSide": _string_or_none(row["somaSide"]),
            }
        )
    return neurons


def null_counts(neurons: "list[dict]") -> "dict[str, int]":
    """Per-`OPTIONAL_FIELDS` count of neurons whose value for that field is
    `None`, in `OPTIONAL_FIELDS` order. Embedded in the artifact itself
    (`render_descending_types_document`'s `nullCounts`) so a claim like
    "class is null for all 48" (docs/data-provenance.md) is verifiable
    directly from the published JSON, not only from this script's stdout or
    from hand-counting the `neurons` array -- and stays correct automatically
    if the real annotations ever change."""
    return {field: sum(1 for n in neurons if n[field] is None) for field in OPTIONAL_FIELDS}


def render_descending_types_document(
    *,
    source_sha256: str,
    graph_sha256: str,
    neurons: "list[dict]",
) -> str:
    """Deterministic JSON rendering: sorted keys, fixed separators, trailing
    newline. Two calls with the same inputs produce byte-identical output."""
    document = {
        "version": 1,
        "sourceFile": ANNOTATIONS_FILENAME,
        "sourceSha256": source_sha256,
        "graphSha256": graph_sha256,
        "neurons": neurons,
        "nullCounts": null_counts(neurons),
    }
    return json.dumps(document, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--graph", type=Path, default=PUBLIC_DATA_DIR / f"{ARTIFACT_NAME}.bin.gz")
    parser.add_argument("--out-dir", type=Path, default=PUBLIC_DATA_DIR)
    parser.add_argument("--manifest-path", type=Path, default=None)
    args = parser.parse_args(argv)

    graph, graph_sha256 = sidecar_io.load_graph_for_sidecar(args.graph)

    # Load (but don't yet write) the manifest up front, so every validation
    # below runs -- and can fail loudly -- before anything on disk is
    # touched, matching positions.py's --graph/manifest cross-check.
    manifest_path = args.manifest_path or (args.out_dir / f"{ARTIFACT_NAME}.manifest.json")
    manifest = sidecar_io.load_json_if_present(manifest_path)
    sidecar_io.cross_check_manifest_graph_sha(
        manifest,
        graph_sha256,
        graph_path=args.graph,
        manifest_path=manifest_path,
        entry_name="a descendingTypes entry",
    )

    print("Loading + verifying pinned body-annotations...")
    annotations, source_sha256 = load_annotations_verified(args.raw_dir)
    print(f"  {len(annotations)} rows, sha256 verified: {source_sha256}")

    print("Joining cell types onto the graph's descending neurons...")
    neurons = build_descending_types(graph, annotations)
    print(f"  {len(neurons)} descending neurons")
    counts = null_counts(neurons)
    print("  missing annotation counts: " + ", ".join(f"{field}={counts[field]}" for field in OPTIONAL_FIELDS))

    payload = render_descending_types_document(
        source_sha256=source_sha256, graph_sha256=graph_sha256, neurons=neurons
    )

    out_path = args.out_dir / f"{OUT_ARTIFACT_NAME}.json"
    descending_types_sha256 = sidecar_io.write_sidecar_artifact(payload, out_path)
    print(f"descending-types sha256: {descending_types_sha256}")

    sidecar_io.merge_json_entry(
        manifest,
        manifest_path,
        "descendingTypes",
        {"artifact": out_path.name, "sha256": descending_types_sha256, "neuronCount": len(neurons)},
        label="descendingTypes entry",
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
