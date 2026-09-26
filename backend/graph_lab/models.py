"""Job request models: a closed-set graph id shared by every kind, and the
per-kind bounds from `.agents/plans/graph-lab/02-job-engines.md`.

Every model is `extra="forbid"` + `strict=True` (matching
`flyarena_lab/service.py`'s `Options` convention): an unexpected field or a
loosely-typed value (e.g. a numeric string for an int field) is rejected
rather than silently coerced or ignored, so a malformed or probing request
body fails validation instead of reaching job dispatch with a wrong shape.

`graph` is deliberately validated with a single, exact anchored regex
(`GRAPH_PATTERN`) rather than a permissive check plus a denylist -- there is
no reasonable transformation of `"biological"`, `"disconnected"`, or
`"rewired:<seed>"` that admits a path-traversal or shell-metacharacter
payload, because the pattern is anchored at both ends (`^...$`) and every
character position is drawn from a fixed alphabet (letters of the three
literal words, or ASCII digits after `rewired:`). A value such as
`"rewired:1;rm -rf /"` or `"../x"` fails the full-string match and is
rejected before it is ever used to build a file path or a subprocess
argument list (`jobs.py` never trusts a request field for path
construction -- see its own `parse_graph_id`, which re-validates
independently of Pydantic having already run).
"""
from __future__ import annotations

import re
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

# `public/data/malecns-arena-v1.manifest.json`'s `neuronCount`. Duplicated
# here (not imported from a JS module) as a lightweight API-layer bound;
# `runEpisode`/`swap_ops.valid_swap` independently re-check every index
# against the real loaded graph's own `metadata.neuronCount`, so a stale
# value here can only make this bound *tighter* than the real graph, never
# looser -- it can reject a request this bound would have allowed, but it
# can never admit an out-of-range index the real engine would accept.
NEURON_COUNT = 1008

MAX_REWIRED_SEED = 499

# `\Z` (absolute end of string), never a bare `$`: Python's `$` also
# matches immediately before a single trailing "\n", so `$` alone would let
# "biological\n" (or "rewired:5\n") through this check while `mode` still
# carries the trailing newline into the job -- and `graphFromTaskMode` on
# the Node side does an exact `===` string comparison against the literal
# `'biological'`, which a value with a stray trailing newline would fail,
# silently falling through to the `disconnected` branch instead of erroring.
# Verified empirically before this fix: `GRAPH_PATTERN.match("biological\n")`
# matched with a bare `$`.
GRAPH_PATTERN = re.compile(r"\A(biological|disconnected|rewired:(\d{1,3}))\Z")


def validate_graph_id(value: str) -> str:
    """Shared by every job kind's `graph` field validator, and re-used
    verbatim (not merely mirrored) by `jobs.parse_graph_id` so there is one
    place that defines what a graph id is allowed to look like."""
    match = GRAPH_PATTERN.match(value)
    if not match:
        raise ValueError('graph must be "biological", "disconnected", or "rewired:<seed>"')
    seed_text = match.group(2)
    if seed_text is not None and int(seed_text) > MAX_REWIRED_SEED:
        raise ValueError(f"rewired seed must be <= {MAX_REWIRED_SEED}")
    return value


def _validate_unique_indices(indices: list[int], *, min_len: int, max_len: int, label: str) -> list[int]:
    if not (min_len <= len(indices) <= max_len):
        raise ValueError(f"{label} must have {min_len}-{max_len} entries")
    if len(set(indices)) != len(indices):
        raise ValueError(f"{label} entries must be unique")
    for index in indices:
        if not (0 <= index < NEURON_COUNT):
            raise ValueError(f"{label} entry out of range [0, {NEURON_COUNT})")
    return indices


class LesionJobRequest(BaseModel):
    """`kind: "lesion"` -- `02-job-engines.md`'s lesion sweep bounds."""

    model_config = ConfigDict(extra="forbid", strict=True)

    kind: Literal["lesion"]
    graph: str
    sets: list[list[int]] = Field(min_length=1, max_length=32)
    seed_start: int = Field(alias="seedStart", ge=0, le=2**32 - 1)
    seed_count: int = Field(alias="seedCount", ge=4, le=100)
    ticks: int = Field(ge=300, le=1800)
    decoder: Literal["authored"] = "authored"

    @field_validator("graph")
    @classmethod
    def _check_graph(cls, value: str) -> str:
        return validate_graph_id(value)

    @field_validator("sets")
    @classmethod
    def _check_sets(cls, value: list[list[int]]) -> list[list[int]]:
        for one_set in value:
            _validate_unique_indices(one_set, min_len=1, max_len=64, label="each lesion set")
        return value


class AtlasJobRequest(BaseModel):
    """`kind: "atlas"` -- `02-job-engines.md`'s atlas search bounds."""

    model_config = ConfigDict(extra="forbid", strict=True)

    kind: Literal["atlas"]
    graph: str
    search_seed: int = Field(alias="searchSeed", ge=0, le=2**31 - 1)
    population: int = Field(ge=4, le=64)
    generations: int = Field(ge=1, le=48)
    ticks: int = Field(ge=300, le=900)

    @field_validator("graph")
    @classmethod
    def _check_graph(cls, value: str) -> str:
        return validate_graph_id(value)


class Swap(BaseModel):
    """One `(a→b, c→d) → (a→d, c→b)` degree-preserving swap."""

    model_config = ConfigDict(extra="forbid", strict=True)

    a: int = Field(ge=0, lt=NEURON_COUNT)
    b: int = Field(ge=0, lt=NEURON_COUNT)
    c: int = Field(ge=0, lt=NEURON_COUNT)
    d: int = Field(ge=0, lt=NEURON_COUNT)


class SwapsetJobRequest(BaseModel):
    """`kind: "swapset"` -- `02-job-engines.md`'s swap-set intervention bounds.
    The base graph is always `biological` (the plan does not offer a choice)."""

    model_config = ConfigDict(extra="forbid", strict=True)

    kind: Literal["swapset"]
    graph: Literal["biological"] = "biological"
    swaps: list[Swap] = Field(min_length=1, max_length=50)
    controls: int = Field(ge=0, le=100)
    seed_start: int = Field(alias="seedStart", ge=0, le=2**32 - 1)
    seed_count: int = Field(alias="seedCount", ge=4, le=100)
    ticks: int = Field(ge=300, le=1800)


JobRequest = Annotated[
    Union[LesionJobRequest, AtlasJobRequest, SwapsetJobRequest],
    Field(discriminator="kind"),
]

# Wall-clock ceilings from `02-job-engines.md` ("Ceiling: ... min"), in
# seconds -- `jobs.Jobs` reads this to time out a running child and mark it
# `"timed-out"` rather than letting a stuck job hold the shared Spark forever.
CEILING_SECONDS: dict[str, int] = {
    "lesion": 20 * 60,
    "atlas": 15 * 60,
    "swapset": 20 * 60,
}
