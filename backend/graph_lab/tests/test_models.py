"""`backend/graph_lab/tests/test_models.py`'s own row in
`.agents/plans/graph-lab/02-job-engines.md`'s Tests section: "covers the
bounds for every field and kind". Constructs `models.py`'s Pydantic models
directly (not through the HTTP layer -- `test_security.py` already covers
the closed-set `graph` pattern and a handful of body-level bounds end to
end; this file is the dedicated, field-by-field bounds sweep for all three
job kinds, using `pydantic.ValidationError` directly rather than an HTTP
status code).
"""
import unittest

from pydantic import ValidationError

from graph_lab.models import (
    NEURON_COUNT,
    AtlasJobRequest,
    LesionJobRequest,
    Swap,
    SwapsetJobRequest,
)

LESION_BODY = {
    "kind": "lesion",
    "graph": "biological",
    "sets": [[0, 1]],
    "seedStart": 30001,
    "seedCount": 4,
    "ticks": 300,
}

ATLAS_BODY = {
    "kind": "atlas",
    "graph": "biological",
    "searchSeed": 1729,
    "population": 8,
    "generations": 2,
    "ticks": 300,
}

SWAPSET_BODY = {
    "kind": "swapset",
    "graph": "biological",
    "swaps": [{"a": 0, "b": 1, "c": 2, "d": 3}],
    "controls": 0,
    "seedStart": 30001,
    "seedCount": 4,
    "ticks": 300,
}


class LesionBoundsTests(unittest.TestCase):
    def test_accepts_the_body_unmodified(self):
        LesionJobRequest(**LESION_BODY)

    def test_sets_length_bounds_1_to_32(self):
        LesionJobRequest(**dict(LESION_BODY, sets=[[0]]))
        LesionJobRequest(**dict(LESION_BODY, sets=[[i] for i in range(32)]))
        with self.assertRaises(ValidationError):
            LesionJobRequest(**dict(LESION_BODY, sets=[]))
        with self.assertRaises(ValidationError):
            LesionJobRequest(**dict(LESION_BODY, sets=[[i] for i in range(33)]))

    def test_each_set_length_bounds_1_to_64(self):
        LesionJobRequest(**dict(LESION_BODY, sets=[list(range(64))]))
        with self.assertRaises(ValidationError):
            LesionJobRequest(**dict(LESION_BODY, sets=[[]]))
        with self.assertRaises(ValidationError):
            LesionJobRequest(**dict(LESION_BODY, sets=[list(range(65))]))

    def test_set_entries_must_be_unique(self):
        with self.assertRaises(ValidationError):
            LesionJobRequest(**dict(LESION_BODY, sets=[[0, 0]]))

    def test_set_entries_must_be_in_range(self):
        LesionJobRequest(**dict(LESION_BODY, sets=[[0, NEURON_COUNT - 1]]))
        with self.assertRaises(ValidationError):
            LesionJobRequest(**dict(LESION_BODY, sets=[[-1]]))
        with self.assertRaises(ValidationError):
            LesionJobRequest(**dict(LESION_BODY, sets=[[NEURON_COUNT]]))

    def test_seed_count_bounds_4_to_100(self):
        LesionJobRequest(**dict(LESION_BODY, seedCount=4))
        LesionJobRequest(**dict(LESION_BODY, seedCount=100))
        with self.assertRaises(ValidationError):
            LesionJobRequest(**dict(LESION_BODY, seedCount=3))
        with self.assertRaises(ValidationError):
            LesionJobRequest(**dict(LESION_BODY, seedCount=101))

    def test_ticks_bounds_300_to_1800(self):
        LesionJobRequest(**dict(LESION_BODY, ticks=300))
        LesionJobRequest(**dict(LESION_BODY, ticks=1800))
        with self.assertRaises(ValidationError):
            LesionJobRequest(**dict(LESION_BODY, ticks=299))
        with self.assertRaises(ValidationError):
            LesionJobRequest(**dict(LESION_BODY, ticks=1801))

    def test_seed_start_bounds_0_to_max_uint32(self):
        LesionJobRequest(**dict(LESION_BODY, seedStart=0, seedCount=4))
        with self.assertRaises(ValidationError):
            LesionJobRequest(**dict(LESION_BODY, seedStart=-1))

    def test_seed_range_must_not_overflow_max_seed(self):
        max_seed = 2**32 - 1
        LesionJobRequest(**dict(LESION_BODY, seedStart=max_seed - 3, seedCount=4))
        with self.assertRaises(ValidationError):
            LesionJobRequest(**dict(LESION_BODY, seedStart=max_seed - 2, seedCount=4))

    def test_decoder_defaults_to_authored_and_rejects_other_values(self):
        request = LesionJobRequest(**LESION_BODY)
        self.assertEqual(request.decoder, "authored")
        with self.assertRaises(ValidationError):
            LesionJobRequest(**dict(LESION_BODY, decoder="parked"))

    def test_extra_field_forbidden(self):
        with self.assertRaises(ValidationError):
            LesionJobRequest(**dict(LESION_BODY, extra="nope"))

    def test_graph_closed_set(self):
        for graph in ("biological", "disconnected", "rewired:0", "rewired:499"):
            LesionJobRequest(**dict(LESION_BODY, graph=graph))
        for graph in ("rewired:500", "not-a-graph", ""):
            with self.assertRaises(ValidationError):
                LesionJobRequest(**dict(LESION_BODY, graph=graph))


class AtlasBoundsTests(unittest.TestCase):
    def test_accepts_the_body_unmodified(self):
        AtlasJobRequest(**ATLAS_BODY)

    def test_search_seed_bounds_0_to_max_int31(self):
        AtlasJobRequest(**dict(ATLAS_BODY, searchSeed=0))
        AtlasJobRequest(**dict(ATLAS_BODY, searchSeed=2**31 - 1))
        with self.assertRaises(ValidationError):
            AtlasJobRequest(**dict(ATLAS_BODY, searchSeed=-1))
        with self.assertRaises(ValidationError):
            AtlasJobRequest(**dict(ATLAS_BODY, searchSeed=2**31))

    def test_population_bounds_4_to_64(self):
        AtlasJobRequest(**dict(ATLAS_BODY, population=4))
        AtlasJobRequest(**dict(ATLAS_BODY, population=64))
        with self.assertRaises(ValidationError):
            AtlasJobRequest(**dict(ATLAS_BODY, population=3))
        with self.assertRaises(ValidationError):
            AtlasJobRequest(**dict(ATLAS_BODY, population=65))

    def test_generations_bounds_1_to_48(self):
        AtlasJobRequest(**dict(ATLAS_BODY, generations=1))
        AtlasJobRequest(**dict(ATLAS_BODY, generations=48))
        with self.assertRaises(ValidationError):
            AtlasJobRequest(**dict(ATLAS_BODY, generations=0))
        with self.assertRaises(ValidationError):
            AtlasJobRequest(**dict(ATLAS_BODY, generations=49))

    def test_ticks_bounds_300_to_900(self):
        AtlasJobRequest(**dict(ATLAS_BODY, ticks=300))
        AtlasJobRequest(**dict(ATLAS_BODY, ticks=900))
        with self.assertRaises(ValidationError):
            AtlasJobRequest(**dict(ATLAS_BODY, ticks=299))
        with self.assertRaises(ValidationError):
            AtlasJobRequest(**dict(ATLAS_BODY, ticks=901))

    def test_graph_closed_set(self):
        for graph in ("biological", "disconnected", "rewired:0", "rewired:499"):
            AtlasJobRequest(**dict(ATLAS_BODY, graph=graph))
        with self.assertRaises(ValidationError):
            AtlasJobRequest(**dict(ATLAS_BODY, graph="rewired:500"))

    def test_extra_field_forbidden(self):
        with self.assertRaises(ValidationError):
            AtlasJobRequest(**dict(ATLAS_BODY, extra="nope"))


class SwapBoundsTests(unittest.TestCase):
    def test_accepts_in_range_indices(self):
        Swap(a=0, b=0, c=NEURON_COUNT - 1, d=NEURON_COUNT - 1)

    def test_rejects_out_of_range_indices(self):
        base = {"a": 0, "b": 1, "c": 2, "d": 3}
        for field in ("a", "b", "c", "d"):
            with self.assertRaises(ValidationError):
                Swap(**dict(base, **{field: -1}))
            with self.assertRaises(ValidationError):
                Swap(**dict(base, **{field: NEURON_COUNT}))


class SwapsetBoundsTests(unittest.TestCase):
    def test_accepts_the_body_unmodified(self):
        SwapsetJobRequest(**SWAPSET_BODY)

    def test_graph_defaults_to_biological_and_rejects_anything_else(self):
        request = SwapsetJobRequest(**{k: v for k, v in SWAPSET_BODY.items() if k != "graph"})
        self.assertEqual(request.graph, "biological")
        with self.assertRaises(ValidationError):
            SwapsetJobRequest(**dict(SWAPSET_BODY, graph="disconnected"))
        with self.assertRaises(ValidationError):
            SwapsetJobRequest(**dict(SWAPSET_BODY, graph="rewired:0"))

    def test_swaps_length_bounds_1_to_50(self):
        one_swap = SWAPSET_BODY["swaps"]
        SwapsetJobRequest(**dict(SWAPSET_BODY, swaps=one_swap * 50))
        with self.assertRaises(ValidationError):
            SwapsetJobRequest(**dict(SWAPSET_BODY, swaps=[]))
        with self.assertRaises(ValidationError):
            SwapsetJobRequest(**dict(SWAPSET_BODY, swaps=one_swap * 51))

    def test_controls_bounds_0_to_100(self):
        SwapsetJobRequest(**dict(SWAPSET_BODY, controls=0))
        SwapsetJobRequest(**dict(SWAPSET_BODY, controls=100))
        with self.assertRaises(ValidationError):
            SwapsetJobRequest(**dict(SWAPSET_BODY, controls=-1))
        with self.assertRaises(ValidationError):
            SwapsetJobRequest(**dict(SWAPSET_BODY, controls=101))

    def test_seed_count_bounds_4_to_100(self):
        SwapsetJobRequest(**dict(SWAPSET_BODY, seedCount=4))
        SwapsetJobRequest(**dict(SWAPSET_BODY, seedCount=100))
        with self.assertRaises(ValidationError):
            SwapsetJobRequest(**dict(SWAPSET_BODY, seedCount=3))
        with self.assertRaises(ValidationError):
            SwapsetJobRequest(**dict(SWAPSET_BODY, seedCount=101))

    def test_ticks_bounds_300_to_1800(self):
        SwapsetJobRequest(**dict(SWAPSET_BODY, ticks=300))
        SwapsetJobRequest(**dict(SWAPSET_BODY, ticks=1800))
        with self.assertRaises(ValidationError):
            SwapsetJobRequest(**dict(SWAPSET_BODY, ticks=299))
        with self.assertRaises(ValidationError):
            SwapsetJobRequest(**dict(SWAPSET_BODY, ticks=1801))

    def test_seed_range_must_not_overflow_max_seed(self):
        max_seed = 2**32 - 1
        SwapsetJobRequest(**dict(SWAPSET_BODY, seedStart=max_seed - 3, seedCount=4))
        with self.assertRaises(ValidationError):
            SwapsetJobRequest(**dict(SWAPSET_BODY, seedStart=max_seed - 2, seedCount=4))

    def test_extra_field_forbidden(self):
        with self.assertRaises(ValidationError):
            SwapsetJobRequest(**dict(SWAPSET_BODY, extra="nope"))


if __name__ == "__main__":
    unittest.main()
