"""xorshift32 PRNG port of `nextRandomState` (`src/lib/arena/world.ts`).

Bit-for-bit identical to the TypeScript implementation. JS's `>>>`/`<<`/`^`
bitwise operators only ever act on a value's 32-bit two's-complement bit
*pattern* (per-spec, they `ToInt32`/`ToUint32` their operands, then operate,
then return either an `Int32` or `Uint32` result — the sign of the returned
interpretation never changes which physical bits survive an XOR, and `<<`
simply discards bits beyond position 31 regardless of sign). Keeping the
Python `int` state masked into `[0, 2**32)` after every operation therefore
reproduces the exact same bit pattern as the TS code, with no signed/
unsigned reinterpretation needed anywhere.
"""
from __future__ import annotations

MASK32 = 0xFFFFFFFF


def normalize_seed(seed: int) -> int:
    """Port of `normalizeSeed`: masks to uint32, then maps the all-zero
    state (xorshift's one fixed point) to a fixed non-zero constant."""
    value = seed & MASK32
    return 0x6D2B79F5 if value == 0 else value


def next_random_state(state: int) -> int:
    """Port of `nextRandomState`. Returns the next non-zero uint32 state."""
    value = normalize_seed(state)
    value = (value ^ ((value << 13) & MASK32)) & MASK32
    value = (value ^ (value >> 17)) & MASK32
    value = (value ^ ((value << 5) & MASK32)) & MASK32
    return value & MASK32


def random_unit(state: int) -> tuple[int, float]:
    """Port of `randomUnit`: `(nextState, nextState / 2**32)`."""
    next_state = next_random_state(state)
    return next_state, next_state / 4294967296.0
