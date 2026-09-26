"""Local offline search, with explicit CUDA selection and exclusive output."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import time
import torch
from .atlas import SearchOptions, search, DISCOVERY_SEEDS, HELDOUT_SEEDS, COVERAGE_EDGES, TURN_EDGES, HIDDEN_SIZE, SUBSTEPS, VERSION
from .atlas_rollout import evaluate_behaviors
from .cli import ARM_NAMES
from .graph import load_graph_json, output_neuron_indices
from .readout import readout_parameter_count
from .rollout import build_rollout_env


def _validate_bundle_arm(bundle: dict) -> str:
    """Validate `--graph`'s bundle looks like a real `export-arms.ts`
    bundle (`formatVersion 1`, `arm` one of `ARM_NAMES`
    (biological/rewired/disconnected)) and return its `arm`. Pure function
    of the parsed JSON object -- independently testable without touching
    `torch`/CUDA/`load_graph_json` (`training/tests/test_atlas.py`).

    Generalizes the original biological-only gate (this CLI previously
    rejected any bundle whose `arm != "biological"`) so the GPU search
    accepts the rewired/disconnected null-graph arms `training:export-arms`
    already produces (`.agents/plans/repertoire-null/01-generalize-atlas-pipeline.md`
    WP1), reusing `flyarena-train`'s own arm vocabulary (`cli.py`'s
    `ARM_NAMES`) rather than redeclaring it.
    """
    if bundle.get("formatVersion") != 1 or bundle.get("arm") not in ARM_NAMES:
        raise ValueError("Use a biological, rewired, or disconnected bundle from training:export-arms")
    return bundle["arm"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    for name, value in asdict(SearchOptions()).items():
        parser.add_argument("--" + name, type=int, default=value)
    args = parser.parse_args()
    options = SearchOptions(**{name: getattr(args, name) for name in asdict(SearchOptions())})
    if args.output.exists():
        parser.error("Output exists; select a new run filename")
    if args.graph.stat().st_size > 8 * 1024**2:
        parser.error("Graph bundle exceeds 8 MiB")
    bundle = json.loads(args.graph.read_text())
    try:
        arm = _validate_bundle_arm(bundle)
    except ValueError as error:
        parser.error(str(error))
    torch.set_num_threads(2)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA requested but unavailable; CPU requires --device cpu")
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1, 1024**3 / total))
        torch.cuda.reset_peak_memory_stats()
    graph = load_graph_json(args.graph, device=args.device)
    indices = output_neuron_indices(graph).tolist()
    if bundle.get("D") != len(indices) or bundle.get("outputNeuronIndices") != indices:
        parser.error("Bundle output-neuron identity mismatch")
    env = build_rollout_env(graph, args.device)
    started = time.perf_counter()
    with torch.inference_mode():
        result = search(options, readout_parameter_count(env.input_size, HIDDEN_SIZE),
                        lambda weights: evaluate_behaviors(env, weights, DISCOVERY_SEEDS, options.ticks),
                        lambda row: print(json.dumps(row), flush=True))
    if args.device == "cuda":
        torch.cuda.synchronize()
    result.update({"schemaVersion": 1, "modelVersion": VERSION, "inputSize": env.input_size,
                   "hiddenSize": HIDDEN_SIZE, "substeps": SUBSTEPS,
                   "discoverySeeds": DISCOVERY_SEEDS, "heldoutSeeds": HELDOUT_SEEDS,
                   "coverageEdges": COVERAGE_EDGES, "turnEdges": TURN_EDGES,
                   "arm": arm,
                   "graphArtifactSha256": bundle["graphArtifactSha256"], "bundleSha256": bundle["sha256"], "bundle": bundle,
                   "runtime": {"device": args.device, "deviceName": torch.cuda.get_device_name() if args.device == "cuda" else "CPU",
                               "torch": str(torch.__version__), "cuda": torch.version.cuda,
                               "seconds": time.perf_counter() - started,
                               "peakTensorBytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else 0,
                               "config": asdict(env.config)}})
    payload = json.dumps(result, allow_nan=False, separators=(",", ":")) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Linking a complete temp file publishes atomically without replacing a prior run.
    with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
    try:
        os.link(temporary, args.output)
    finally:
        temporary.unlink()
    print(f"Saved {args.output} ({len(result['candidates'])} occupied cells)")


if __name__ == "__main__":
    main()
