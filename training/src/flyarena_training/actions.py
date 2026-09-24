"""Port of `decodeAction` (`src/lib/arena/actions.ts`).

Scoped to the array form only (`[thrust, yaw, brake]` in `OUTPUT_POPULATION`
order): every caller in this package (the golden traces, the readout MLP,
and WP3's eventual CEM rollout) always produces or consumes a 3-element
array, never the TS union's `Partial<DecodedAction>` object form, so that
form is not ported here.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch

# Canonical output-population order (`OUTPUT_POPULATION`, actions.ts):
# thrust, yaw, brake.
OUTPUT_POPULATION = {"thrust": 0, "yaw": 1, "brake": 2}


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return minimum if value < minimum else maximum if value > maximum else value


def _finite_or_zero(value: float | None) -> float:
    if value is None or not math.isfinite(value):
        return 0.0
    return float(value)


def decode_action(input_: Sequence[float] | None) -> tuple[float, float, float]:
    """Returns `(thrust, yaw, brake)`, clamped to `[-1, 1]`, `[-1, 1]`,
    `[0, 1]` respectively. `None`/missing values become zero before
    clamping, matching `decodeAction(undefined)`."""
    if input_ is None:
        thrust_raw = yaw_raw = brake_raw = None
    else:
        seq = list(input_)
        thrust_raw = seq[OUTPUT_POPULATION["thrust"]] if len(seq) > OUTPUT_POPULATION["thrust"] else None
        yaw_raw = seq[OUTPUT_POPULATION["yaw"]] if len(seq) > OUTPUT_POPULATION["yaw"] else None
        brake_raw = seq[OUTPUT_POPULATION["brake"]] if len(seq) > OUTPUT_POPULATION["brake"] else None
    return (
        _clamp(_finite_or_zero(thrust_raw), -1.0, 1.0),
        _clamp(_finite_or_zero(yaw_raw), -1.0, 1.0),
        _clamp(_finite_or_zero(brake_raw), 0.0, 1.0),
    )


def decode_action_batch(
    raw: torch.Tensor | None,
    batch_size: int,
    device: str | torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Batched port of `decode_action`/`decodeAction`. `raw`: `[B, >=3]`
    tensor (columns in `OUTPUT_POPULATION` order) or `None`/missing for the
    zero action for every item, matching `decodeAction(undefined)`. Returns
    `[B, 3]`, clamped to `[-1, 1]`, `[-1, 1]`, `[0, 1]` for thrust/yaw/brake;
    non-finite input values become zero before clamping, like
    `_finite_or_zero` above."""
    device = torch.device(device)

    def column(index: int) -> torch.Tensor:
        if raw is None or raw.shape[1] <= index:
            return torch.zeros(batch_size, dtype=dtype, device=device)
        values = raw[:, index].to(dtype=dtype, device=device)
        return torch.where(torch.isfinite(values), values, torch.zeros_like(values))

    thrust = column(OUTPUT_POPULATION["thrust"]).clamp(-1.0, 1.0)
    yaw = column(OUTPUT_POPULATION["yaw"]).clamp(-1.0, 1.0)
    brake = column(OUTPUT_POPULATION["brake"]).clamp(0.0, 1.0)
    return torch.stack([thrust, yaw, brake], dim=1)
