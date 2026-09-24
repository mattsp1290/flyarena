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
