"""flyarena_training: batched PyTorch port of the FlyArena arena, rate
model, and readout MLP for GPU-accelerated CEM training and TS parity
checking (see `.agents/plans/trained-readout/02-gpu-port-and-parity.md`).

The settings below must run before any simulation tensor op. Applying them
here, at package-import time, means every entry point (tests, a future CLI,
notebooks) inherits them without having to remember to call anything.

- TF32 disabling + highest matmul precision: PyTorch enables TF32 for
  float32 matmul by default on Ampere+ GPUs (including the GB10's
  Blackwell-class SM_121 device), which would silently break the
  tolerance-based parity gate against the TypeScript golden traces.
- `torch.use_deterministic_algorithms(True, warn_only=True)`: the plan's
  "Risks and exclusions" section notes `torch.sparse` CUDA kernels (the CSR
  `torch.sparse.mm` `model.py`'s `step_model` uses) may be nondeterministic
  across runs, and directs this exact call, `warn_only=True` so an operation
  with no deterministic implementation degrades to a logged warning instead
  of raising (WP3 defines the reproducibility tolerance this is meant to
  keep small, not exact-match).
"""
from __future__ import annotations

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
torch.use_deterministic_algorithms(True, warn_only=True)

PRECISION_APPLIED = True

__all__ = ["PRECISION_APPLIED"]
