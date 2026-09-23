"""Python encoder/validator for the FlyArena connectome graph binary format.

This module is the Python-side counterpart to `src/lib/connectome/format.ts`
and must stay byte-for-byte compatible with it. The wire format is specified
in `docs/graph-format.md`: little-endian, a fixed 56-byte header, then 9
sections in a fixed order, each starting at a byte offset that is a multiple
of 8. Every invariant enforced by `validateGraph` in `format.ts` is mirrored
here in `validate_graph()` so the compiler can never emit a file its own
consumer would reject.

Nothing in this module reads or downloads MaleCNS data; it only knows how to
turn already-selected node/edge arrays into the wire format (`compile.py`)
and how to validate them (`compile.py`, `rewire.py`, `tests_python/`).
"""

from __future__ import annotations

import gzip
import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

MAGIC = b"FANG"
SUPPORTED_FORMAT_VERSION = 1
HEADER_BYTES = 56

# '<' = little-endian, no padding.
# 4s magic, then 5x uint32 (formatVersion, neuronCount, edgeCount,
# inputChannelCount, outputPopulationCount), then 7x float32, then 1x uint32
# (flags). 4 + 5*4 + 7*4 + 4 = 56 bytes, matching docs/graph-format.md.
_HEADER_STRUCT = struct.Struct("<4sIIIIIfffffffI")

METADATA_FIELDS = (
    "formatVersion",
    "neuronCount",
    "edgeCount",
    "inputChannelCount",
    "outputPopulationCount",
    "timestepSeconds",
    "leakRate",
    "rateMin",
    "rateMax",
    "inputClampMin",
    "inputClampMax",
    "globalGain",
)


def align8(offset: int) -> int:
    """Round a byte offset up to the next multiple of 8, matching format.ts's
    `alignTo8`. Python ints are arbitrary precision, so there is no 32-bit
    wraparound hazard the TypeScript implementation warns about."""
    return -(-offset // 8) * 8


@dataclass(frozen=True)
class SectionLayout:
    byte_offset: int
    byte_length: int


@dataclass(frozen=True)
class GraphLayout:
    biological_ids: SectionLayout
    presynaptic_offsets: SectionLayout
    postsynaptic_indices: SectionLayout
    contact_magnitudes: SectionLayout
    presynaptic_signs: SectionLayout
    input_channel_index: SectionLayout
    input_weight: SectionLayout
    output_population_index: SectionLayout
    output_weight: SectionLayout
    total_bytes: int


def compute_graph_layout(neuron_count: int, edge_count: int) -> GraphLayout:
    """Section offsets/lengths derived purely from neuron/edge counts. Must
    stay identical to `computeGraphLayout` in format.ts; `tests_python/`
    checks this against docs/graph-format.md's worked example directly."""
    cursor = HEADER_BYTES

    def section(element_bytes: int, count: int) -> SectionLayout:
        nonlocal cursor
        byte_offset = cursor
        byte_length = element_bytes * count
        cursor = align8(byte_offset + byte_length)
        return SectionLayout(byte_offset, byte_length)

    biological_ids = section(8, neuron_count)
    presynaptic_offsets = section(4, neuron_count + 1)
    postsynaptic_indices = section(4, edge_count)
    contact_magnitudes = section(4, edge_count)
    presynaptic_signs = section(1, neuron_count)
    input_channel_index = section(4, neuron_count)
    input_weight = section(4, neuron_count)
    output_population_index = section(4, neuron_count)
    output_weight = section(4, neuron_count)

    return GraphLayout(
        biological_ids=biological_ids,
        presynaptic_offsets=presynaptic_offsets,
        postsynaptic_indices=postsynaptic_indices,
        contact_magnitudes=contact_magnitudes,
        presynaptic_signs=presynaptic_signs,
        input_channel_index=input_channel_index,
        input_weight=input_weight,
        output_population_index=output_population_index,
        output_weight=output_weight,
        total_bytes=cursor,
    )


@dataclass
class GraphArrays:
    """One in-memory graph, ready to validate/encode. All arrays use the
    exact dtypes the wire format requires (see docs/graph-format.md's
    section table); `validate_graph`/`encode_graph_binary` assume this."""

    metadata: Mapping[str, float]
    biological_ids: np.ndarray  # uint64, len == neuronCount
    presynaptic_offsets: np.ndarray  # uint32, len == neuronCount + 1
    postsynaptic_indices: np.ndarray  # uint32, len == edgeCount
    contact_magnitudes: np.ndarray  # float32, len == edgeCount
    presynaptic_signs: np.ndarray  # int8, len == neuronCount
    input_channel_index: np.ndarray  # int32, len == neuronCount
    input_weight: np.ndarray  # float32, len == neuronCount
    output_population_index: np.ndarray  # int32, len == neuronCount
    output_weight: np.ndarray  # float32, len == neuronCount


class InvalidGraphError(ValueError):
    pass


def _invalid(message: str) -> None:
    raise InvalidGraphError(f"Invalid connectome graph: {message}")


def validate_graph(graph: GraphArrays) -> GraphArrays:
    """Mirrors `validateGraph` in `src/lib/connectome/format.ts` exactly,
    field for field and check for check, so a file this function accepts is
    guaranteed to be accepted by the TypeScript parser too."""
    meta = graph.metadata

    if int(meta["formatVersion"]) != SUPPORTED_FORMAT_VERSION:
        _invalid(
            f"unsupported formatVersion {meta['formatVersion']}; expected {SUPPORTED_FORMAT_VERSION}"
        )

    for label in ("neuronCount", "edgeCount", "inputChannelCount", "outputPopulationCount"):
        value = meta[label]
        # Intentionally stricter than format.ts's `Number.isInteger(value)`,
        # which accepts a float64 like 5.0 (JS has no separate integer
        # type). Every count this compiler ever produces is a genuine
        # Python int, so requiring `isinstance(value, (int, np.integer))`
        # only ever rejects inputs TS would also treat as suspicious to
        # construct by hand; it never rejects anything this module emits.
        if not (isinstance(value, (int, np.integer)) and value >= 0):
            _invalid(f"{label} must be a non-negative integer")

    for label in (
        "timestepSeconds",
        "leakRate",
        "rateMin",
        "rateMax",
        "inputClampMin",
        "inputClampMax",
        "globalGain",
    ):
        value = float(meta[label])
        if not np.isfinite(value):
            _invalid(f"{label} must be finite")

    if float(meta["timestepSeconds"]) <= 0:
        _invalid("timestepSeconds must be positive")
    if float(meta["leakRate"]) < 0:
        _invalid("leakRate must be non-negative")
    if float(meta["rateMin"]) > float(meta["rateMax"]):
        _invalid("rateMin must not exceed rateMax")
    if float(meta["inputClampMin"]) > float(meta["inputClampMax"]):
        _invalid("inputClampMin must not exceed inputClampMax")
    if float(meta["globalGain"]) < 0:
        _invalid("globalGain must be non-negative")

    neuron_count = int(meta["neuronCount"])
    edge_count = int(meta["edgeCount"])
    input_channel_count = int(meta["inputChannelCount"])
    output_population_count = int(meta["outputPopulationCount"])

    length_checks = (
        ("biologicalIds", len(graph.biological_ids), neuron_count),
        ("presynapticOffsets", len(graph.presynaptic_offsets), neuron_count + 1),
        ("postsynapticIndices", len(graph.postsynaptic_indices), edge_count),
        ("contactMagnitudes", len(graph.contact_magnitudes), edge_count),
        ("presynapticSigns", len(graph.presynaptic_signs), neuron_count),
        ("inputChannelIndex", len(graph.input_channel_index), neuron_count),
        ("inputWeight", len(graph.input_weight), neuron_count),
        ("outputPopulationIndex", len(graph.output_population_index), neuron_count),
        ("outputWeight", len(graph.output_weight), neuron_count),
    )
    for label, actual, expected in length_checks:
        if actual != expected:
            _invalid(f"{label} length {actual} does not match expected length {expected}")

    # The length check above already guarantees len(offsets) == neuron_count
    # + 1 >= 1, so offsets[0] and offsets[neuron_count] are always in range
    # here; no separate emptiness guard is needed.
    offsets = graph.presynaptic_offsets
    if int(offsets[0]) != 0:
        _invalid("presynapticOffsets must start at 0")
    if int(offsets[neuron_count]) != edge_count:
        _invalid("presynapticOffsets must end at edgeCount")

    post = graph.postsynaptic_indices
    magnitudes = graph.contact_magnitudes
    for pre in range(neuron_count):
        start = int(offsets[pre])
        end = int(offsets[pre + 1])
        if end < start:
            _invalid(f"presynapticOffsets must be non-decreasing at row {pre}")
        previous_post = -1
        for edge in range(start, end):
            p = int(post[edge])
            if p < 0 or p >= neuron_count:
                _invalid(f"postsynapticIndices[{edge}] is out of range")
            if p <= previous_post:
                _invalid(
                    f"postsynapticIndices must be strictly increasing within row {pre} "
                    f"(duplicate or out-of-order postsynaptic index {p} at edge {edge})"
                )
            previous_post = p
            magnitude = float(magnitudes[edge])
            if not np.isfinite(magnitude) or magnitude <= 0:
                _invalid(f"contactMagnitudes[{edge}] must be finite and positive")

    signs = graph.presynaptic_signs
    channel_idx = graph.input_channel_index
    input_weight = graph.input_weight
    population_idx = graph.output_population_index
    output_weight = graph.output_weight
    for neuron in range(neuron_count):
        sign = int(signs[neuron])
        if sign not in (-1, 1):
            _invalid(f"presynapticSigns[{neuron}] must be -1 or 1")
        channel = int(channel_idx[neuron])
        if channel < -1 or channel >= input_channel_count:
            _invalid(f"inputChannelIndex[{neuron}] is out of range")
        if not np.isfinite(float(input_weight[neuron])):
            _invalid(f"inputWeight[{neuron}] must be finite")
        population = int(population_idx[neuron])
        if population < -1 or population >= output_population_count:
            _invalid(f"outputPopulationIndex[{neuron}] is out of range")
        if not np.isfinite(float(output_weight[neuron])):
            _invalid(f"outputWeight[{neuron}] must be finite")

    return graph


def encode_graph_binary(graph: GraphArrays) -> bytes:
    """Encode a validated graph into the exact byte layout docs/graph-format.md
    specifies. Mirrors `encodeGraphBinary` in format.ts."""
    validate_graph(graph)
    meta = graph.metadata
    neuron_count = int(meta["neuronCount"])
    edge_count = int(meta["edgeCount"])
    layout = compute_graph_layout(neuron_count, edge_count)

    buffer = bytearray(layout.total_bytes)

    header = _HEADER_STRUCT.pack(
        MAGIC,
        int(meta["formatVersion"]),
        neuron_count,
        edge_count,
        int(meta["inputChannelCount"]),
        int(meta["outputPopulationCount"]),
        float(meta["timestepSeconds"]),
        float(meta["leakRate"]),
        float(meta["rateMin"]),
        float(meta["rateMax"]),
        float(meta["inputClampMin"]),
        float(meta["inputClampMax"]),
        float(meta["globalGain"]),
        0,
    )
    buffer[0:HEADER_BYTES] = header

    def place(section: SectionLayout, array: np.ndarray, dtype: np.dtype) -> None:
        typed = np.ascontiguousarray(array, dtype=dtype)
        raw = typed.tobytes()
        if len(raw) != section.byte_length:
            raise InvalidGraphError(
                f"internal error: section byte length mismatch ({len(raw)} != {section.byte_length})"
            )
        buffer[section.byte_offset : section.byte_offset + len(raw)] = raw

    place(layout.biological_ids, graph.biological_ids, np.uint64)
    place(layout.presynaptic_offsets, graph.presynaptic_offsets, np.uint32)
    place(layout.postsynaptic_indices, graph.postsynaptic_indices, np.uint32)
    place(layout.contact_magnitudes, graph.contact_magnitudes, np.float32)
    place(layout.presynaptic_signs, graph.presynaptic_signs, np.int8)
    place(layout.input_channel_index, graph.input_channel_index, np.int32)
    place(layout.input_weight, graph.input_weight, np.float32)
    place(layout.output_population_index, graph.output_population_index, np.int32)
    place(layout.output_weight, graph.output_weight, np.float32)

    return bytes(buffer)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_gzip_deterministic(data: bytes, path: Path) -> None:
    """Write gzip bytes with a fixed mtime (0) and no filename in the header
    so the compressed file is byte-identical across runs (gzip embeds the
    source mtime and filename by default, which would otherwise make
    `.bin.gz` output non-reproducible run-to-run).

    This makes the *gzip container* (header fields) deterministic; the
    compressed payload itself is deterministic across runs of the same
    Python/zlib version because DEFLATE compression is a pure function of
    its input bytes and compression level (CPython's `gzip` module doesn't
    vary these by OS or time). It is not a guarantee that two different
    zlib versions/vendors always choose bit-identical DEFLATE encodings for
    the same input -- if that ever needs to be pinned exactly across
    machines, pin the Python/zlib version used to build `public/data/`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        with gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0) as gz:
            gz.write(data)
