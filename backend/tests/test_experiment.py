import copy
import threading
import unittest
import torch
from pydantic import ValidationError
from flyarena_lab.experiment import Options, run, reevaluate, summary, Cancelled
from flyarena_lab.model import circuit, neural_step, simulate, move, reward


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.options = Options(device="cpu", population=4, generations=1, ticks=30, training_seeds=4, heldout_seeds=8)
        cls.result = run(cls.options)

    def test_repeat_sham_seeds_export(self):
        other = run(self.options)
        for key in ("weights", "history", "arms", "frames"):
            self.assertEqual(self.result[key], other[key])
        self.assertFalse(set(other["training_seeds"]) & set(other["heldout_seeds"]))
        self.assertEqual(other["arms"][0]["scores"], other["arms"][1]["scores"])
        self.assertEqual(reevaluate(other)["arms"], other["arms"])
        self.assertEqual(other["interventions"][3]["silenced_units"], list(range(8)))
        self.assertEqual(other["frames"][0]["tick"], 0)
        self.assertEqual(other["frames"][-1]["tick"], 30)
        self.assertLessEqual(len(other["frames"]), 100)
        self.assertGreaterEqual(other["history"][-1], other["history"][0])

    def test_masks_and_disconnection(self):
        rec, sensory, _ = circuit(3, "cpu")
        state = torch.ones(1, 1, 64)
        mask = torch.ones(1, 1, 64)
        mask[..., :8] = 0
        obs = torch.ones(1, 1, 8)
        for _ in range(10):
            state = neural_step(state, obs, rec, sensory, mask, 1)
            self.assertEqual(state[..., :8].abs().sum().item(), 0)
        a = neural_step(state, obs, rec, sensory, mask, 0)
        b = neural_step(state, obs, rec * 100, sensory, mask, 0)
        self.assertTrue(torch.equal(a, b))

    def test_statistics(self):
        stats = summary([1., 2., 3., 4.])
        self.assertEqual(stats["mean"], 2.5)
        self.assertAlmostEqual(stats["high"], 2.5 + 1.96 * (5 / 3 / 4)**0.5)
        for arm in self.result["arms"]:
            paired = [a - b for a, b in zip(arm["scores"], self.result["arms"][0]["scores"])]
            self.assertEqual(arm["effect"], summary(paired))

    def test_limits_cancel_and_malformed(self):
        for settings in ({"ticks": 601}, {"seed": True}, {"population": 1}, {"seed": float("nan")}, {"device": "auto"}):
            with self.assertRaises(ValidationError):
                Options(**settings)
        event = threading.Event()
        event.set()
        with self.assertRaises(Cancelled):
            run(self.options, cancel=event)
        bad = copy.deepcopy(self.result)
        bad["weights"]["readout"][0][0] = float("nan")
        with self.assertRaises(ValueError):
            reevaluate(bad)
        bad = copy.deepcopy(self.result)
        bad["heldout_seeds"][0] += 1
        with self.assertRaises(ValueError):
            reevaluate(bad)

    def test_scalar_stationary_world(self):
        rec, sensory, readout = circuit(4, "cpu")
        readout.zero_()
        readout[-1, 0] = -100  # tanh=-1 means zero thrust
        output = simulate(readout[None], rec, sensory, [11], 30, capture=True)
        self.assertAlmostEqual(output.scores.item(), -30 * 0.005, places=6)
        self.assertEqual(output.pickups.item(), 0)
        self.assertEqual(output.contacts.item(), 0)
        self.assertEqual(output.paths[-1]["agents"], [[[0.0, 0.0]]])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_frozen_weight_agreement(self):
        cuda = reevaluate(self.result, "cuda")
        for expected, actual in zip(self.result["arms"], cuda["arms"]):
            torch.testing.assert_close(torch.tensor(expected["scores"]), torch.tensor(actual["scores"]), atol=2e-4, rtol=2e-4)
            self.assertEqual(expected["pickups"], actual["pickups"])
            self.assertEqual(expected["contacts"], actual["contacts"])

    def test_wall_and_contact_equations(self):
        position, heading, speed = move(torch.tensor([[7.99, 0.]]), torch.tensor([torch.pi / 2]),
                                        torch.tensor([4.]), torch.tensor([[1., 0.]]))
        self.assertEqual(position[0, 0].item(), 8.)
        self.assertEqual(speed.item(), 4.)
        first, hit, contact, entered = reward(torch.tensor([1.]), torch.tensor([0.5]),
                                              torch.tensor([0.5]), torch.tensor([False]))
        self.assertAlmostEqual(first.item(), 8.495, places=5)
        self.assertTrue(hit.item() and contact.item() and entered.item())
        next_score, _, _, entered = reward(torch.tensor([1.]), torch.tensor([1.]),
                                            torch.tensor([0.5]), torch.tensor([True]))
        self.assertFalse(entered.item())
        self.assertAlmostEqual(next_score.item(), -0.005, places=6)
