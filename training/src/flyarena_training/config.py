"""Mirror of `ARENA_CONFIG` (`src/lib/arena/config.ts`).

Field values are copied by hand from the TypeScript source and must stay
numerically identical to it. Rather than re-deriving `createArenaConfigFingerprint`'s
exact JS `Number#toString` canonicalization in Python (a source of its own
subtle drift), the drift check below parses the `configFingerprint` string
TypeScript already computed and embedded in every committed golden trace
file (`tests/fixtures/golden/trace-graph-seed-*.json`, produced by
`scripts/training/export-traces.ts`) and compares each field's *numeric*
value against this module's mirror. No dedicated `arena-config.json` file
exists in this repository (grounded by reading `tests/fixtures/golden/` and
`export-traces.ts` before writing this port); the fingerprint already
embedded in every seed file **is** that "JSON dump produced by
export-traces.ts" for drift-detection purposes, so this reuses it directly
instead of asking WP1's exporter (whose golden-file set is regression-tested
byte-for-byte by `tests/unit/golden-traces.test.ts`, owned by another work
package) to grow a new committed file. See `training/README.md` for the full
rationale.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ArenaConfig:
    fixed_delta_seconds: float
    half_width: float
    half_depth: float
    agent_radius: float
    food_radius: float
    hazard_radius: float
    food_count: int
    hazard_count: int
    spawn_inset: float
    max_speed: float
    acceleration: float
    turn_rate: float
    rolling_drag: float
    brake_drag: float
    movement_score_per_unit: float
    food_score: float
    hazard_penalty: float
    sensor_range: float


ARENA_CONFIG = ArenaConfig(
    fixed_delta_seconds=1.0 / 30.0,
    half_width=12.0,
    half_depth=8.0,
    agent_radius=0.35,
    food_radius=0.25,
    hazard_radius=0.6,
    food_count=4,
    hazard_count=2,
    spawn_inset=1.0,
    max_speed=6.0,
    acceleration=9.0,
    turn_rate=math.pi,
    rolling_drag=0.7,
    brake_drag=8.0,
    movement_score_per_unit=0.1,
    food_score=10.0,
    hazard_penalty=2.0,
    sensor_range=24.0,
)

# (TS field name, ArenaConfig attribute name), in `CONFIG_KEYS`'s TS source
# order (`Object.keys(ARENA_CONFIG)` in config.ts). Order isn't load-bearing
# for the drift check below (it compares by key, not position) but is kept
# aligned for readability against the TS source.
TS_FIELD_ORDER: tuple[tuple[str, str], ...] = (
    ("fixedDeltaSeconds", "fixed_delta_seconds"),
    ("halfWidth", "half_width"),
    ("halfDepth", "half_depth"),
    ("agentRadius", "agent_radius"),
    ("foodRadius", "food_radius"),
    ("hazardRadius", "hazard_radius"),
    ("foodCount", "food_count"),
    ("hazardCount", "hazard_count"),
    ("spawnInset", "spawn_inset"),
    ("maxSpeed", "max_speed"),
    ("acceleration", "acceleration"),
    ("turnRate", "turn_rate"),
    ("rollingDrag", "rolling_drag"),
    ("brakeDrag", "brake_drag"),
    ("movementScorePerUnit", "movement_score_per_unit"),
    ("foodScore", "food_score"),
    ("hazardPenalty", "hazard_penalty"),
    ("sensorRange", "sensor_range"),
)


def parse_fingerprint(fingerprint: str) -> dict[str, float]:
    """Parse a `configFingerprint` string (`arena-config-v1|key=value|...`,
    `createArenaConfigFingerprint` in `src/lib/arena/config.ts`) into a
    `{tsFieldName: float}` dict."""
    prefix, separator, rest = fingerprint.partition("|")
    if prefix != "arena-config-v1" or not separator:
        raise ValueError(f"unsupported or malformed config fingerprint: {fingerprint!r}")
    values: dict[str, float] = {}
    for pair in rest.split("|"):
        key, eq, raw = pair.partition("=")
        if not eq:
            raise ValueError(f"malformed fingerprint field {pair!r} in {fingerprint!r}")
        values[key] = float(raw)
    return values


def assert_matches_fingerprint(fingerprint: str, tolerance: float = 1e-12) -> None:
    """Raise `AssertionError` if `ARENA_CONFIG` disagrees with a TS-produced
    `configFingerprint` string on any field (the config-drift gate)."""
    parsed = parse_fingerprint(fingerprint)
    expected_keys = {ts_name for ts_name, _ in TS_FIELD_ORDER}
    if set(parsed) != expected_keys:
        raise AssertionError(
            "config fingerprint field set does not match TS_FIELD_ORDER: "
            f"fingerprint has {sorted(parsed)}, mirror expects {sorted(expected_keys)}. "
            "ARENA_CONFIG's shape changed in TypeScript; update config.py's ArenaConfig "
            "and TS_FIELD_ORDER to match."
        )
    for ts_name, py_name in TS_FIELD_ORDER:
        expected = parsed[ts_name]
        actual = float(getattr(ARENA_CONFIG, py_name))
        if abs(actual - expected) > tolerance:
            raise AssertionError(
                f"ARENA_CONFIG.{py_name} = {actual} does not match TS ARENA_CONFIG.{ts_name} = "
                f"{expected} (diff {abs(actual - expected):.3e}). Regenerate the mirror in "
                "training/src/flyarena_training/config.py from src/lib/arena/config.ts."
            )


# `.agents/plans/task-generality/01-task-plumbing.md`'s WP1: mirror of
# `src/lib/arena/tasks.ts`'s `ARENA_TASKS`. Field values are copied by hand
# from the TS source the same way `ARENA_CONFIG` itself is (see this
# module's own doc comment) -- `training/tests/test_tasks.py` cross-checks
# every entry against `tests/fixtures/golden/tasks.json` (TS's own exported
# fingerprints) via `parse_fingerprint`/`TS_FIELD_ORDER`, the same drift
# check `assert_matches_fingerprint` already runs for the default task.
ARENA_TASKS: dict[str, ArenaConfig] = {
    "default": ARENA_CONFIG,
    "hazard-heavy": dataclasses.replace(ARENA_CONFIG, hazard_count=4, hazard_penalty=6.0),
    "sparse-food": dataclasses.replace(ARENA_CONFIG, food_count=1, half_width=18.0, half_depth=12.0),
    "no-movement": dataclasses.replace(ARENA_CONFIG, movement_score_per_unit=0.0),
    "crowded": dataclasses.replace(ARENA_CONFIG, half_width=8.0, half_depth=5.5),
}

# `resolveArenaTask`'s fingerprint, per task id (`createArenaConfigFingerprint`
# in `src/lib/arena/config.ts`, computed over each `ARENA_TASKS` entry in
# `src/lib/arena/tasks.ts`) -- hand-copied verbatim from the committed
# `tests/fixtures/golden/tasks.json` rather than reformatted from Python
# floats: JS's `Number#toString` and Python's `float.__str__` do not always
# agree byte-for-byte (e.g. a whole-valued float prints as `"12"` in JS but
# `"12.0"` in Python), so this module never re-derives a fingerprint
# *string* -- only ever parses one TS already computed (see this module's
# own top doc comment for the same reasoning applied to `ARENA_CONFIG`
# itself). `cli.py` writes this string verbatim into `config.json`'s
# `arenaTaskFingerprint`, so `null-trained-worker.ts`'s fifth identity check
# compares two TS-native fingerprint strings, never a Python-formatted one.
ARENA_TASK_FINGERPRINTS: dict[str, str] = {
    "default": (
        "arena-config-v1|fixedDeltaSeconds=0.03333333333333333|halfWidth=12|halfDepth=8|agentRadius=0.35|"
        "foodRadius=0.25|hazardRadius=0.6|foodCount=4|hazardCount=2|spawnInset=1|maxSpeed=6|acceleration=9|"
        "turnRate=3.141592653589793|rollingDrag=0.7|brakeDrag=8|movementScorePerUnit=0.1|foodScore=10|"
        "hazardPenalty=2|sensorRange=24"
    ),
    "hazard-heavy": (
        "arena-config-v1|fixedDeltaSeconds=0.03333333333333333|halfWidth=12|halfDepth=8|agentRadius=0.35|"
        "foodRadius=0.25|hazardRadius=0.6|foodCount=4|hazardCount=4|spawnInset=1|maxSpeed=6|acceleration=9|"
        "turnRate=3.141592653589793|rollingDrag=0.7|brakeDrag=8|movementScorePerUnit=0.1|foodScore=10|"
        "hazardPenalty=6|sensorRange=24"
    ),
    "sparse-food": (
        "arena-config-v1|fixedDeltaSeconds=0.03333333333333333|halfWidth=18|halfDepth=12|agentRadius=0.35|"
        "foodRadius=0.25|hazardRadius=0.6|foodCount=1|hazardCount=2|spawnInset=1|maxSpeed=6|acceleration=9|"
        "turnRate=3.141592653589793|rollingDrag=0.7|brakeDrag=8|movementScorePerUnit=0.1|foodScore=10|"
        "hazardPenalty=2|sensorRange=24"
    ),
    "no-movement": (
        "arena-config-v1|fixedDeltaSeconds=0.03333333333333333|halfWidth=12|halfDepth=8|agentRadius=0.35|"
        "foodRadius=0.25|hazardRadius=0.6|foodCount=4|hazardCount=2|spawnInset=1|maxSpeed=6|acceleration=9|"
        "turnRate=3.141592653589793|rollingDrag=0.7|brakeDrag=8|movementScorePerUnit=0|foodScore=10|"
        "hazardPenalty=2|sensorRange=24"
    ),
    "crowded": (
        "arena-config-v1|fixedDeltaSeconds=0.03333333333333333|halfWidth=8|halfDepth=5.5|agentRadius=0.35|"
        "foodRadius=0.25|hazardRadius=0.6|foodCount=4|hazardCount=2|spawnInset=1|maxSpeed=6|acceleration=9|"
        "turnRate=3.141592653589793|rollingDrag=0.7|brakeDrag=8|movementScorePerUnit=0.1|foodScore=10|"
        "hazardPenalty=2|sensorRange=24"
    ),
}


def resolve_arena_task(task_id: str) -> ArenaConfig:
    """Mirror of `resolveArenaTask` (`src/lib/arena/tasks.ts`): the arena
    config for `task_id`, or a `ValueError` on an unrecognized one (there is
    no "fall back to default" here -- an unrecognized id must fail loudly,
    the same way the TS side does)."""
    try:
        return ARENA_TASKS[task_id]
    except KeyError:
        raise ValueError(f"unknown arena task {task_id!r} (expected one of {sorted(ARENA_TASKS)})") from None


def resolve_arena_task_fingerprint(task_id: str) -> str:
    """The TS-computed fingerprint string for `task_id` (see
    `ARENA_TASK_FINGERPRINTS`'s own doc comment for why this is a hand-copied
    literal, not a Python-formatted string). Raises the same way
    `resolve_arena_task` does on an unrecognized id."""
    try:
        return ARENA_TASK_FINGERPRINTS[task_id]
    except KeyError:
        raise ValueError(f"unknown arena task {task_id!r} (expected one of {sorted(ARENA_TASKS)})") from None
