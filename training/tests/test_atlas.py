from pathlib import Path
import pytest
import torch
from flyarena_training.atlas import SearchOptions, Elite, retain, bin_index, COVERAGE_EDGES, search
from flyarena_training.atlas_cli import _validate_bundle_arm
from flyarena_training.atlas_rollout import evaluate_behaviors
from flyarena_training.graph import load_graph_json
from flyarena_training.rollout import build_rollout_env
from flyarena_training.readout import readout_parameter_count


@pytest.mark.parametrize("arm", ["biological", "rewired", "disconnected"])
def test_validate_bundle_arm_accepts_every_export_arms_arm(arm):
    assert _validate_bundle_arm({"formatVersion": 1, "arm": arm}) == arm


@pytest.mark.parametrize(
    "bundle",
    [
        {"formatVersion": 1, "arm": "trained"},
        {"formatVersion": 1, "arm": "unknown"},
        {"formatVersion": 1},
        {"formatVersion": 2, "arm": "biological"},
        {"arm": "biological"},
    ],
)
def test_validate_bundle_arm_rejects_unknown_arm_or_format_version(bundle):
    with pytest.raises(ValueError):
        _validate_bundle_arm(bundle)


def test_boundaries_and_stable_archive():
    assert [bin_index(x, COVERAGE_EDGES) for x in [0, .05, .1, 1]] == [0, 1, 2, 5]
    archive = {}
    retain(archive, Elite(1, [1], 2, .06, .2))
    retain(archive, Elite(2, [2], 2, .06, .2))
    assert next(iter(archive.values())).id == 1
    retain(archive, Elite(3, [3], 3, .06, .2))
    assert next(iter(archive.values())).id == 3
    for quality, coverage in [(float('nan'), .1), (1, float('inf')), (1, -1)]:
        with pytest.raises(ValueError):
            retain(archive, Elite(4, [], quality, coverage, 0))


@pytest.mark.parametrize("kwargs", [{"population": 129}, {"seed": True}, {"generations": 0}, {"ticks": 1801}])
def test_bounds(kwargs):
    with pytest.raises(ValueError):
        SearchOptions(**kwargs)


def test_seeded_search_and_invalid_evaluator():
    options = SearchOptions(population=4, generations=2, ticks=30)
    def evaluate(theta):
        return torch.stack((theta[:, 0], torch.sigmoid(theta[:, 1]), torch.tanh(theta[:, 2])), 1)
    assert search(options, 5, evaluate) == search(options, 5, evaluate)
    with pytest.raises(ValueError):
        search(options, 5, lambda _: torch.full((4, 3), float('nan')))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_rollout_batch_and_device_parity(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.set_num_threads(2)
    graph = load_graph_json(Path(__file__).parents[2] / 'tests/fixtures/golden/trace-graph.json')
    cpu = build_rollout_env(graph, 'cpu')
    env = build_rollout_env(graph, device)
    theta = torch.randn(2, readout_parameter_count(env.input_size, 8), generator=torch.Generator().manual_seed(1729)) * .3
    with torch.inference_mode():
        baseline = evaluate_behaviors(cpu, theta, [61001, 61002], 30)
        batch = evaluate_behaviors(env, theta, [61001, 61002], 30).cpu()
        singles = torch.cat([evaluate_behaviors(env, row[None], [61001, 61002], 30).cpu() for row in theta])
    torch.testing.assert_close(batch, baseline, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(batch, singles, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_actual_cuda_search_repeatability():
    torch.set_num_threads(2)
    graph = load_graph_json(Path(__file__).parents[2] / 'tests/fixtures/golden/trace-graph.json')
    env = build_rollout_env(graph, 'cuda')
    options = SearchOptions(population=4, generations=2, ticks=30)
    with torch.inference_mode():
        def run():
            return search(options, readout_parameter_count(env.input_size, 8),
                          lambda theta: evaluate_behaviors(env, theta, [61001, 61002], 30))
        assert run() == run()
