"""Bounded evolution, matched interventions, and portable evidence exports."""
import math
import threading
import time
from typing import Literal
import torch
from pydantic import BaseModel, ConfigDict, Field
from .model import ARMS, VERSION, circuit, simulate


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    seed: int = Field(default=17, ge=0, le=2**31 - 1)
    device: Literal["cpu", "cuda"] = "cuda"
    population: int = Field(default=32, ge=4, le=64)
    generations: int = Field(default=12, ge=1, le=40)
    training_seeds: int = Field(default=8, ge=4, le=32)
    heldout_seeds: int = Field(default=16, ge=8, le=64)
    ticks: int = Field(default=240, ge=30, le=600)


class Cancelled(Exception):
    pass


def configure(device):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; choose CPU explicitly")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, 2 * 1024**3 / total))
        torch.cuda.reset_peak_memory_stats()


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def seeds_for(options):
    # Disjoint, deterministic indices within the portable nonnegative 32-bit range.
    base = options.seed * 97 % (2**31)
    return ([base + i for i in range(options.training_seeds)],
            [base + 10_000 + i for i in range(options.heldout_seeds)])


def summary(values):
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean)**2 for x in values) / (n - 1)
    margin = 1.96 * math.sqrt(variance / n)
    return {"mean": mean, "low": mean - margin, "high": mean + margin}


def evaluate(readout, recurrent, sensory, seeds, ticks, checkpoint):
    outputs = simulate(readout[None].expand(len(ARMS), -1, -1).contiguous(),
                       recurrent, sensory, seeds, ticks, arms=ARMS, checkpoint=checkpoint, capture=True)
    scores = outputs.scores.cpu().tolist()
    arms = []
    for index, arm in enumerate(ARMS):
        differences = [a - b for a, b in zip(scores[index], scores[0])]
        arms.append({"name": arm, "scores": scores[index], "effect": summary(differences),
                     "mean_score": sum(scores[index]) / len(seeds),
                     "pickups": outputs.pickups[index].cpu().tolist(),
                     "contacts": outputs.contacts[index].cpu().tolist()})
    return {"arms": arms, "frames": outputs.paths}


def run(options: Options, progress=lambda value: None, cancel=None):
    cancel = cancel or threading.Event()

    def checkpoint():
        if cancel.is_set():
            raise Cancelled("Experiment cancelled")

    configure(options.device)
    checkpoint()
    # Largest state, temporaries and schedule stay well below this conservative estimate.
    estimate = options.population * options.training_seeds * (64 * 32 + options.ticks * 16) * 4
    if estimate > 2 * 1024**3:
        raise ValueError("Estimated tensor allocation exceeds 2 GiB")
    started = time.perf_counter()
    recurrent, sensory, best = circuit(options.seed, options.device)
    training, heldout = seeds_for(options)
    rng = torch.Generator().manual_seed(options.seed + 42)
    history = []
    with torch.inference_mode():
        initial = simulate(best[None], recurrent, sensory, training, options.ticks, checkpoint=checkpoint)
        best_score = initial.scores.mean().item()
        history.append(best_score)
        for generation in range(options.generations):
            checkpoint()
            noise = torch.randn(options.population, *best.shape, generator=rng).to(options.device)
            sigma = 0.15 * (1 - generation / options.generations) + 0.02
            candidates = best[None] + noise * sigma
            candidates[0] = best
            rollout = simulate(candidates, recurrent, sensory, training, options.ticks, checkpoint=checkpoint)
            fitness = rollout.scores.mean(dim=1)
            winner = fitness.argmax().item()
            best = candidates[winner].clone()
            best_score = fitness[winner].item()
            history.append(best_score)
            progress({"phase": "training", "generation": generation + 1,
                      "generations": options.generations, "best_score": best_score})
        synchronize(options.device)
        trained = time.perf_counter()
        progress({"phase": "interventions", "generation": options.generations,
                  "generations": options.generations, "best_score": best_score})
        evaluation = evaluate(best, recurrent, sensory, heldout, options.ticks, checkpoint)
    synchronize(options.device)
    checkpoint()
    result = {"schema_version": 1, "model_version": VERSION, "provenance": "Authored / synthetic",
              "interventions": [{"name": name, "silenced_units": list(range((ARMS.index(name) - 3) * 8, (ARMS.index(name) - 2) * 8)) if name.startswith("G") else [],
                                 "recurrence_enabled": name != "disconnected"} for name in ARMS],
              "task": {"dt": 1 / 15, "extent": 8, "food_radius": 0.65, "hazard_radius": 0.9,
                       "food_reward": 10, "hazard_entry_penalty": 2, "tick_cost": 0.005,
                       "reward": "distance_before - distance_after + food_reward - hazard_entry_penalty - tick_cost"},
              "options": options.model_dump(), "training_seeds": training, "heldout_seeds": heldout,
              "weights": {"recurrent": recurrent.cpu().tolist(), "sensory": sensory.cpu().tolist(),
                          "readout": best.cpu().tolist()}, "history": history, **evaluation,
              "runtime": {"device": options.device, "device_name": torch.cuda.get_device_name() if options.device == "cuda" else "CPU",
                          "torch": str(torch.__version__), "cuda": torch.version.cuda,
                          "training_seconds": trained - started, "evaluation_seconds": time.perf_counter() - trained,
                          "peak_tensor_bytes": torch.cuda.max_memory_allocated() if options.device == "cuda" else None,
                          "precision": "float32; TF32 disabled; deterministic algorithms"}}
    return result


def reevaluate(document, device="cpu"):
    if document.get("schema_version") != 1 or document.get("model_version") != VERSION:
        raise ValueError("Unsupported experiment export")
    options = Options.model_validate(document["options"])
    expected_training, expected_heldout = seeds_for(options)
    if document.get("training_seeds") != expected_training or document.get("heldout_seeds") != expected_heldout:
        raise ValueError("Seed manifest mismatch")
    tensors = []
    for key, shape in [("readout", (65, 2)), ("recurrent", (64, 64)), ("sensory", (8, 64))]:
        value = document["weights"][key]
        # Bound nested JSON before tensor conversion or CUDA allocation.
        if not isinstance(value, list) or len(value) != shape[0] or any(
            not isinstance(row, list) or len(row) != shape[1] or any(
                type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 1e6 for x in row
            ) for row in value
        ):
            raise ValueError(f"Invalid {key} weights")
        tensors.append(torch.tensor(value, dtype=torch.float32))
    if device not in ("cpu", "cuda"):
        raise ValueError("Unknown device")
    configure(device)
    with torch.inference_mode():
        return evaluate(*(t.to(device) for t in tensors), expected_heldout, options.ticks, lambda: None)
