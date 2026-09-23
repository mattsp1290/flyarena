# Connectome graph binary format (v1)

This document is the wire-format contract for the `.bin` artifact consumed by
`src/lib/connectome/format.ts` (`parseGraphBinary`/`encodeGraphBinary`). A
later bean implements the offline Python compiler that emits this file from a
pinned MaleCNS export; that implementer should be able to produce a
byte-compatible file from this document alone, without reading the
TypeScript parser.

If anything here and `format.ts` ever disagree, `format.ts` is a bug to fix,
not a spec to follow — but they must always agree, and any real change is a
`formatVersion` bump plus an update to this document.

## Conventions

- All multi-byte integers and floats are **little-endian**.
- All offsets in this document are **byte offsets from the start of the
  file**.
- Every section starts at a byte offset that is a multiple of 8. This is
  sufficient alignment for every dtype used below (`BigUint64Array` needs
  8-byte alignment; `Uint32Array`/`Int32Array`/`Float32Array` need 4-byte
  alignment; `Int8Array` needs none), so every section can be read as a
  zero-copy `TypedArray` view directly over the file's `ArrayBuffer` with no
  copying or realignment.
- There is no explicit section-offset table in the file. Every section's
  offset and length are fully determined by the header's `neuronCount` and
  `edgeCount` fields via the formula in [Section layout](#section-layout)
  below. A reader computes offsets; it does not read them from the file.
- A "neuron" here is any node in the compiled subgraph (its `biologicalId`
  traces it back to the source dataset); it is not necessarily a literal
  biological neuron.

## Header (56 bytes, offset 0)

| Offset | Size | Type    | Field                  | Notes |
|-------:|-----:|---------|-------------------------|-------|
| 0      | 4    | ASCII   | `magic`                 | Exactly the 4 bytes `"FANG"` (0x46 0x41 0x4E 0x47). |
| 4      | 4    | uint32  | `formatVersion`         | Must equal `1` for this document. A reader must reject any other value before reading further sections. |
| 8      | 4    | uint32  | `neuronCount`            | Number of nodes, N. |
| 12     | 4    | uint32  | `edgeCount`              | Number of directed synaptic entries, E (CSR non-zero count). |
| 16     | 4    | uint32  | `inputChannelCount`      | Number of external observation channels (see `src/lib/arena/sensors.ts`'s 8-channel contract for the canonical POC input; this field is not hardcoded to 8). |
| 20     | 4    | uint32  | `outputPopulationCount`  | Number of output populations. Convention: population 0/1/2 map to thrust/yaw/brake to match `decodeAction` in `src/lib/arena/actions.ts`, but the format itself is agnostic. |
| 24     | 4    | float32 | `timestepSeconds`        | Simulated seconds integrated per neural substep. Must be > 0. |
| 28     | 4    | float32 | `leakRate`               | Per-substep leak/decay coefficient. Must be >= 0. |
| 32     | 4    | float32 | `rateMin`                | Lower bound every neuron's rate is clamped to. |
| 36     | 4    | float32 | `rateMax`                | Upper bound every neuron's rate is clamped to. Must be >= `rateMin`. |
| 40     | 4    | float32 | `inputClampMin`          | Lower bound applied to a raw channel value before it is injected as external drive. |
| 44     | 4    | float32 | `inputClampMax`          | Upper bound for the same. Must be >= `inputClampMin`. |
| 48     | 4    | float32 | `globalGain`             | Global multiplier applied to every recurrent synaptic contribution. Calibrated, not measured. |
| 52     | 4    | uint32  | `flags`                  | Reserved for future use. Writers must set this to `0`; readers must not reject a nonzero value on the strength of this field alone (forward-compatible reserve), but no flag bits are defined in format version 1. |

Total header size: 56 bytes (already a multiple of 8; no header padding is
needed).

## Section layout

Let `align8(x) = (x + 7) & ~7` (round up to the next multiple of 8).

Starting at `cursor = 56` (the end of the header), sections appear in this
exact order. Each section's byte length is `elementSize * count`; after
writing a section, advance `cursor = align8(cursor + byteLength)` before
placing the next section (i.e. pad with zero bytes up to the next multiple of
8, then place the next section there).

| # | Section                 | Element type | Element size | Count         | Meaning |
|---|--------------------------|--------------|--------------:|---------------|---------|
| 1 | `biologicalIds`          | uint64       | 8             | `neuronCount` | Opaque per-neuron source-dataset identifier (e.g. a MaleCNS body ID), indexed by internal neuron index `0..neuronCount-1`. |
| 2 | `presynapticOffsets`     | uint32       | 4             | `neuronCount + 1` | CSR row pointers. Row `pre`'s outgoing edges are the half-open range `[presynapticOffsets[pre], presynapticOffsets[pre+1])` into sections 3/4. `presynapticOffsets[0]` must be `0`; `presynapticOffsets[neuronCount]` must equal `edgeCount`; the sequence must be non-decreasing. |
| 3 | `postsynapticIndices`    | uint32       | 4             | `edgeCount`   | Postsynaptic neuron index per edge (CSR column indices). Every value must be `< neuronCount`. |
| 4 | `contactMagnitudes`      | float32      | 4             | `edgeCount`   | Positive contact-count magnitude per edge (unsigned; sign is applied per presynaptic neuron, see section 5). Every value must be `> 0` and finite. |
| 5 | `presynapticSigns`       | int8         | 1             | `neuronCount` | Dale's-law sign applied to every edge leaving that neuron. Every value must be exactly `-1` or `1` (never `0`). |
| 6 | `inputChannelIndex`      | int32        | 4             | `neuronCount` | Observation channel index (`0..inputChannelCount-1`) a neuron receives external drive from, or `-1` if the neuron is not an input neuron. |
| 7 | `inputWeight`            | float32      | 4             | `neuronCount` | Per-neuron scale applied to its assigned channel's (clamped) value. Meaningless when `inputChannelIndex[i] == -1`; writers should emit `0` in that case by convention but readers must not depend on it. |
| 8 | `outputPopulationIndex`  | int32        | 4             | `neuronCount` | Output population index (`0..outputPopulationCount-1`) a neuron's rate contributes to, or `-1` if the neuron does not feed an output population. |
| 9 | `outputWeight`           | float32      | 4             | `neuronCount` | Per-neuron scale applied when aggregating its rate into its output population. Meaningless when `outputPopulationIndex[i] == -1`. |

File length = `cursor` after placing section 9 (i.e. the padded end of the
last section). A reader must reject a file shorter than this computed length
("truncated") and may ignore any trailing bytes beyond it.

### Worked offset example

For `neuronCount = 4`, `edgeCount = 2`:

| Section | Raw bytes | Offset (start) | Offset (end, unpadded) | Next section starts at (aligned) |
|---|---:|---:|---:|---:|
| header | 56 | 0 | 56 | 56 |
| biologicalIds | 4×8=32 | 56 | 88 | 88 |
| presynapticOffsets | 5×4=20 | 88 | 108 | 112 |
| postsynapticIndices | 2×4=8 | 112 | 120 | 120 |
| contactMagnitudes | 2×4=8 | 120 | 128 | 128 |
| presynapticSigns | 4×1=4 | 128 | 132 | 136 |
| inputChannelIndex | 4×4=16 | 136 | 152 | 152 |
| inputWeight | 4×4=16 | 152 | 168 | 168 |
| outputPopulationIndex | 4×4=16 | 168 | 184 | 184 |
| outputWeight | 4×4=16 | 184 | 200 | 200 |

Total file length: 200 bytes.

## Validation a reader must perform before stepping the model

In addition to the structural checks implied by the table above (lengths,
`presynapticOffsets` monotonicity/endpoints, index ranges, sign values,
positive magnitudes, finite floats, `rateMin <= rateMax`,
`inputClampMin <= inputClampMax`, `timestepSeconds > 0`, `leakRate >= 0`),
a reader must reject any `formatVersion` other than the one it implements.
`src/lib/connectome/format.ts`'s `validateGraph` is the executable version of
this checklist; `tests/unit/format.test.ts` exercises each failure mode.

## Topology modes are not part of this format

"Biological", "rewired", and "disconnected" are labels for *how a file was
produced*, not fields inside it. A disconnected control artifact is simply a
valid file per this format with `edgeCount = 0` (and therefore
`presynapticOffsets` entirely `0`, and empty `postsynapticIndices`/
`contactMagnitudes` sections) that otherwise shares the biological arm's
`neuronCount`, `biologicalIds`, and input/output mapping. A rewired control
is a valid file with the same node set and edge-weight multiset as the
biological arm but permuted `postsynapticIndices`. Both are produced by the
offline Python compiler; this parser treats every file identically regardless
of which arm produced it.
