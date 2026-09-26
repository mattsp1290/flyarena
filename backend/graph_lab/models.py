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

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# `public/data/malecns-arena-v1.manifest.json`'s `neuronCount`. Duplicated
# here (not imported from a JS module) as a lightweight API-layer bound;
# `runEpisode`/`swap_ops.valid_swap` independently re-check every index
# against the real loaded graph's own `metadata.neuronCount`, so a stale
# value here can only make this bound *tighter* than the real graph, never
# looser -- it can reject a request this bound would have allowed, but it
# can never admit an out-of-range index the real engine would accept.
NEURON_COUNT = 1008

MAX_REWIRED_SEED = 499

# `src/lib/arena/world.ts`'s `normalizeSeed` does `seed >>> 0` (an unsigned
# 32-bit wrap) before ever using a seed -- so a `heldOutSeeds` entry built
# as `seedStart + i` (`entry-lesion.ts`'s/`entry-swapset.ts`'s `buildTasks`)
# that exceeds this range doesn't error, it silently wraps to some other,
# smaller seed, which can collide with an earlier entry in the very same
# request and make two "different" seeds simulate the identical episode --
# a data-correctness bug (broken statistical independence), not a crash, so
# nothing else here would have caught it. `seed_start` alone is bounded to
# `[0, MAX_SEED]` by each field's own `Field(le=...)`, but that alone does
# not bound `seed_start + seed_count - 1` -- the actual highest seed a
# request produces -- which is checked by `_check_seed_range` below.
MAX_SEED = 2**32 - 1

# `\Z` (absolute end of string), never a bare `$`: Python's `$` also
# matches immediately before a single trailing "\n", so `$` alone would let
# "biological\n" (or "rewired:5\n") through this check while `mode` still
# carries the trailing newline into the job -- and `graphFromTaskMode` on
# the Node side does an exact `===` string comparison against the literal
# `'biological'`, which a value with a stray trailing newline would fail,
# silently falling through to the `disconnected` branch instead of erroring.
# Verified empirically before this fix: `GRAPH_PATTERN.match("biological\n")`
# matched with a bare `$`.
#
# `re.ASCII`, and `(?:0|[1-9]\d{0,2})` instead of a bare `\d{1,3}`: without
# `re.ASCII`, Python's `\d` matches every Unicode decimal-digit character,
# not just 0-9 (e.g. "rewired:٥" (Arabic-Indic) and "rewired:１２"
# (fullwidth) both matched, and `int()` happily parses them too) -- outside
# this module's own documented "closed set" contract even though it never
# reached an unvalidated path or subprocess argument. The non-capturing
# `(?:0|[1-9]\d{0,2})` also rejects a leading zero ("rewired:007"), which
# `\d{1,3}` alone allowed.
GRAPH_PATTERN = re.compile(r"\A(biological|disconnected|rewired:(0|[1-9]\d{0,2}))\Z", re.ASCII)


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


def _check_seed_range(seed_start: int, seed_count: int) -> None:
    """Shared by `LesionJobRequest`/`SwapsetJobRequest`'s `model_validator`s:
    the highest seed a request will actually produce
    (`seed_start + seed_count - 1`) must itself stay within `MAX_SEED`, not
    just `seed_start` alone."""
    if seed_start + seed_count - 1 > MAX_SEED:
        raise ValueError(
            f"seedStart + seedCount - 1 ({seed_start + seed_count - 1}) exceeds the maximum seed {MAX_SEED}"
        )


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
    seed_start: int = Field(alias="seedStart", ge=0, le=MAX_SEED)
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

    @model_validator(mode="after")
    def _check_seed_range(self) -> "LesionJobRequest":
        _check_seed_range(self.seed_start, self.seed_count)
        return self


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
    seed_start: int = Field(alias="seedStart", ge=0, le=MAX_SEED)
    seed_count: int = Field(alias="seedCount", ge=4, le=100)
    ticks: int = Field(ge=300, le=1800)

    @model_validator(mode="after")
    def _check_seed_range(self) -> "SwapsetJobRequest":
        _check_seed_range(self.seed_start, self.seed_count)
        return self


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
