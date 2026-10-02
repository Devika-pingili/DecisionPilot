import unittest

from backend.features.candidates import generate_candidates, measure_candidate_recall
from backend.features.history import HistoryStore
from tests.feature_fixture import make_fixture


class CandidateTests(unittest.TestCase):
    def setUp(self):
        frames = make_fixture()
        self.train_products = frames[2]
        self.history = HistoryStore(frames[0], frames[1], frames[3], frames[4], frames[5])

    def test_union_is_deduplicated_and_deterministic(self):
        first = generate_candidates(
            self.history, 10, 104, 4, recent_order_count=2, candidate_limit=20
        )
        second = generate_candidates(
            self.history, 10, 104, 4, recent_order_count=2, candidate_limit=20
        )

        self.assertEqual(first.to_dict("records"), second.to_dict("records"))
        self.assertFalse(first.duplicated("candidate_product_id").any())
        self.assertEqual(set(first["candidate_product_id"]), {1, 2, 3, 4})

    def test_candidate_limit_is_respected(self):
        candidates = generate_candidates(
            self.history, 10, 104, 4, candidate_limit=2, source_limit=2
        )

        self.assertLessEqual(len(candidates), 2)
        self.assertEqual(
            candidates[["customer_id", "target_order_id", "target_order_number"]].drop_duplicates().shape[0],
            1,
        )

    def test_candidate_recall_reports_missed_products(self):
        candidates = generate_candidates(self.history, 10, 104, 4, candidate_limit=2)
        per_target, aggregate = measure_candidate_recall(candidates, self.train_products)

        self.assertEqual(len(per_target), 2)
        self.assertIn("missed_target_products", per_target.columns)
        self.assertGreaterEqual(aggregate["mean_candidate_recall"], 0.0)
        self.assertLessEqual(aggregate["mean_candidate_recall"], 1.0)

    def test_empty_candidates_are_reported(self):
        empty = generate_candidates(self.history, 20, 202, 2, candidate_limit=1).iloc[0:0]
        per_target, aggregate = measure_candidate_recall(empty, self.train_products)

        self.assertEqual(aggregate["orders_with_zero_candidates"], 2.0)
        self.assertEqual(int(per_target.loc[per_target["target_order_id"] == 202, "candidate_count"].iloc[0]), 0)


if __name__ == "__main__":
    unittest.main()
