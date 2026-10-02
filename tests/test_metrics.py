import math
import unittest

from backend.evaluation.metrics import ranking_metrics


class MetricsTests(unittest.TestCase):
    def test_standard_metrics_at_k(self):
        metrics = ranking_metrics([1, 2, 3], {2, 4}, [2, 3])

        self.assertAlmostEqual(metrics["precision_at_2"], 0.5)
        self.assertAlmostEqual(metrics["recall_at_2"], 0.5)
        self.assertAlmostEqual(metrics["f1_at_2"], 0.5)
        self.assertAlmostEqual(metrics["map_at_2"], 0.25)
        self.assertAlmostEqual(metrics["ndcg_at_2"], 1 / (math.log2(3) + 1))
        self.assertAlmostEqual(metrics["precision_at_3"], 1 / 3)
        self.assertAlmostEqual(metrics["recall_at_3"], 0.5)
        self.assertAlmostEqual(metrics["f1_at_3"], 0.4)
        self.assertAlmostEqual(metrics["map_at_3"], 0.25)

    def test_empty_recommendations_and_targets_are_zero(self):
        metrics = ranking_metrics([], set(), [1, 5])

        self.assertTrue(all(value == 0.0 for value in metrics.values()))

    def test_short_recommendation_list_uses_available_length(self):
        metrics = ranking_metrics([2], {2}, [5])

        self.assertEqual(metrics["precision_at_5"], 1.0)
        self.assertEqual(metrics["recall_at_5"], 1.0)
        self.assertEqual(metrics["f1_at_5"], 1.0)


if __name__ == "__main__":
    unittest.main()
