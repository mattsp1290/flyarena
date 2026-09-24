"""Bounded MAP-Elites search; selection uses discovery data only."""
from dataclasses import dataclass, asdict
import math
import torch

DISCOVERY_SEEDS = list(range(61001, 61009))
HELDOUT_SEEDS = list(range(62001, 62013))
COVERAGE_EDGES = [0, .05, .1, .2, .35, .6, 1]
TURN_EDGES = [-1, -2 / 3, -1 / 3, 0, 1 / 3, 2 / 3, 1]
HIDDEN_SIZE = 8
SUBSTEPS = 4
VERSION = "behavior-atlas-v1"


@dataclass(frozen=True)
class SearchOptions:
    seed: int = 1729
    population: int = 64
    generations: int = 24
    ticks: int = 900

    def __post_init__(self):
        for name, low, high in [("seed", 0, 2**31 - 1), ("population", 4, 128),
                                ("generations", 1, 96), ("ticks", 30, 1800)]:
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")


def bin_index(value: float, edges: list[float]) -> int:
    if not math.isfinite(value) or not edges[0] <= value <= edges[-1]:
        raise ValueError("Descriptor outside its finite range")
    return min(len(edges) - 2, sum(value >= edge for edge in edges[1:]))


def cell_for(coverage: float, turning: float) -> int:
    return bin_index(turning, TURN_EDGES) * 6 + bin_index(coverage, COVERAGE_EDGES)


@dataclass
class Elite:
    id: int
    theta: list[float]
    quality: float
    coverage: float
    turning: float

    @property
    def cell(self):
        return cell_for(self.coverage, self.turning)


def retain(archive: dict[int, Elite], elite: Elite):
    if not math.isfinite(elite.quality):
        raise ValueError("Nonfinite quality")
    cell = elite.cell
    incumbent = archive.get(cell)
    if incumbent is None or elite.quality > incumbent.quality:
        archive[cell] = elite


def search(options: SearchOptions, parameter_count: int, evaluate, progress=lambda row: None):
    """CPU RNG and stable evaluation-order IDs; evaluator returns P×3 metrics."""
    rng = torch.Generator().manual_seed(options.seed)
    archive: dict[int, Elite] = {}
    history = []
    for generation in range(options.generations):
        candidates = torch.randn(options.population, parameter_count, generator=rng) * .5
        if archive:
            parents = list(sorted(archive.values(), key=lambda elite: elite.id))
            # One quarter fresh candidates; the remainder explore existing cells.
            for row in range(options.population // 4, options.population):
                parent = parents[torch.randint(len(parents), (), generator=rng).item()]
                sigma = (.05, .15, .4)[row % 3]
                candidates[row] = torch.tensor(parent.theta) + torch.randn(parameter_count, generator=rng) * sigma
        # A finite domain also bounds serialized weights and arithmetic.
        candidates.clamp_(-8, 8)
        metrics = evaluate(candidates)
        if tuple(metrics.shape) != (options.population, 3) or not torch.isfinite(metrics).all():
            raise ValueError("Evaluator must return finite P×3 [quality, coverage, turning]")
        for row, (quality, coverage, turning) in enumerate(metrics.cpu().tolist()):
            retain(archive, Elite(generation * options.population + row, candidates[row].tolist(),
                                  quality, coverage, turning))
        record = {"generation": generation + 1, "occupied": len(archive),
                  "bestQuality": max(elite.quality for elite in archive.values())}
        history.append(record)
        progress(record)
    return {"options": asdict(options), "history": history,
            "candidates": [asdict(elite) for elite in sorted(archive.values(), key=lambda elite: elite.id)],
            "searchPolicy": {"initialStd": .5, "mutationScales": [.05, .15, .4], "freshFraction": .25,
                             "weightBound": 8, "ties": "earlier candidate", "rng": "torch CPU Generator"}}
