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
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.feather as feather

sys.path.insert(0, str(Path(__file__).resolve().parent))
import binfmt  # noqa: E402
import rewire  # noqa: E402
from download import SOURCE_FILES  # noqa: E402

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


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pinned_annotations_sha256() -> str:
    for source in SOURCE_FILES:
        if source.filename == ANNOTATIONS_FILENAME:
            return source.sha256
    raise RuntimeError(
        f"{ANNOTATIONS_FILENAME} is not among download.py's pinned SOURCE_FILES; "
        "the pinned list must be extended (URL + sha256 + size) before this script can run"
    )


def load_annotations_verified(
    raw_dir: Path = RAW_DATA_DIR, expected_sha256: Optional[str] = None
) -> "tuple[pd.DataFrame, str]":
    """Load the body-annotations feather table, refusing to proceed if its
    sha256 doesn't match the pinned value (`expected_sha256`, defaulting to
    download.py's pin for the real filename -- overridable so tests can
    point this at a small fixture file with its own expected hash).
    Returns `(dataframe, verified_sha256)`."""
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
    return table.to_pandas(), actual_sha256


def _is_valid_location(value: object) -> bool:
    """True when `value` is a real 3-element coordinate, not a missing
    (`None`/NaN) list cell. Feather round-trips a missing `list[int64]`
    cell as Python `None` (object dtype), not `NaN` -- this also tolerates
    a stray float NaN defensively."""
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    arr = np.asarray(value)
    return arr.shape == (3,)


def build_positions(graph: binfmt.GraphArrays, annotations: pd.DataFrame) -> dict:
    """Join `annotations` onto `graph.biological_ids`, in that exact index
    order, deriving each neuron's role from the graph's own
    inputChannelIndex/outputPopulationIndex arrays (never from the
    annotations table) and its position from
    somaLocation -> tosomaLocation -> none (never fabricated/imputed).
    Returns the fields `malecns-arena-v1.positions.json` needs, minus the
    top-level provenance fields (`version`/`sourceFile`/`sourceSha256`/
    `graphSha256`/`units`) the caller adds."""
    if "bodyId" not in annotations.columns:
        raise RuntimeError("annotations table has no bodyId column")
    if annotations["bodyId"].duplicated().any():
        dup = sorted(annotations.loc[annotations["bodyId"].duplicated(), "bodyId"].unique().tolist())
        raise RuntimeError(f"annotations table has duplicate bodyId(s): {dup[:10]}")

    by_body = annotations.set_index("bodyId")
    has_soma_column = "somaLocation" in annotations.columns
    has_tosoma_column = "tosomaLocation" in annotations.columns

    neuron_count = int(graph.metadata["neuronCount"])
    body_ids: list[str] = []
    roles: list[str] = []
    position_sources: list[str] = []
    xyz: "list[Optional[list[int]]]" = []
    soma_count = 0
    tosoma_count = 0
    none_count = 0

    for i in range(neuron_count):
        body = int(graph.biological_ids[i])
        body_ids.append(str(body))

        is_sensory = int(graph.input_channel_index[i]) >= 0
        is_descending = int(graph.output_population_index[i]) >= 0
        if is_sensory and is_descending:
            raise RuntimeError(
                f"neuron index {i} (bodyId {body}) has both an input channel and an output "
                "population assigned; a neuron must not be both sensory and descending"
            )
        roles.append("sensory" if is_sensory else ("descending" if is_descending else "bridge"))

        row = by_body.loc[body] if body in by_body.index else None
        soma = row["somaLocation"] if row is not None and has_soma_column else None
        tosoma = row["tosomaLocation"] if row is not None and has_tosoma_column else None

        if _is_valid_location(soma):
            xyz.append([int(v) for v in soma])
            position_sources.append(POSITION_SOURCE_SOMA)
            soma_count += 1
        elif _is_valid_location(tosoma):
            xyz.append([int(v) for v in tosoma])
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

    print("Loading + verifying pinned body-annotations...")
    annotations, source_sha256 = load_annotations_verified(args.raw_dir)
    print(f"  {len(annotations)} rows, sha256 verified: {source_sha256}")

    print("Joining soma positions onto the graph's neuron order...")
    fields = build_positions(graph, annotations)
    print(f"  coverage: {json.dumps(fields['coverage'], sort_keys=True)}")
    print(f"  roleCounts: {json.dumps(fields['roleCounts'], sort_keys=True)}")

    payload = render_positions_document(
        source_sha256=source_sha256, graph_sha256=graph_sha256, fields=fields
    )
    positions_sha256 = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{ARTIFACT_NAME}.positions.json"
    out_path.write_text(payload)
    print(f"Wrote {out_path} ({len(payload.encode('utf-8'))} bytes)")
    print(f"positions sha256: {positions_sha256}")

    manifest_path = args.manifest_path or (args.out_dir / f"{ARTIFACT_NAME}.manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["positions"] = {
            "artifact": out_path.name,
            "sha256": positions_sha256,
            "coverage": fields["coverage"],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"Updated {manifest_path} with the positions entry")
    else:
        print(f"[warn] {manifest_path} does not exist; skipped manifest update")

    ledger_path = args.ledger_path or (args.out_dir / f"{ARTIFACT_NAME}.ledger.json")
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        ledger["positionsCoverage"] = fields["coverage"]
        ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
        print(f"Updated {ledger_path} with positionsCoverage")
    else:
        print(f"[warn] {ledger_path} does not exist; skipped ledger update")

    return 0


if __name__ == "__main__":
    sys.exit(main())
