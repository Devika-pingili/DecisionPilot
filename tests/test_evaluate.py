import tempfile
import unittest
from pathlib import Path

from backend.evaluation.evaluate import evaluate_rolling
from backend.features.history import HistoryStore
from tests.feature_fixture import make_fixture


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        frames = make_fixture()
        self.history = HistoryStore(frames[0], frames[1], frames[3], frames[4], frames[5])

    def test_rolling_evaluation_reports_baselines_metrics_and_segments(self):
        result = evaluate_rolling(
            self.history,
            k_values=[1, 2],
            min_history=1,
            candidate_limit=20,
        )

        self.assertEqual(result.metadata["eligible_target_count"], 2)
        self.assertEqual(result.metadata["excluded_target_count"], 2)
        self.assertEqual(set(result.summary["baseline"]), {
            "customer_frequency", "customer_recency", "previously_purchased",
            "global_popularity", "department_aware_popularity",
        })
        self.assertEqual(set(result.summary["k"]), {1, 2})
        self.assertIn("candidate_recall", result.summary.columns)
        self.assertIn("history_length", result.segments["segment_type"].tolist())
        self.assertIn("target_basket_size", result.segments["segment_type"].tolist())
        self.assertFalse(result.per_target.empty)
        self.assertFalse(result.recommendations.empty)

    def test_minimum_history_exclusions_are_reported(self):
        result = evaluate_rolling(self.history, k_values=[5], min_history=2)

        self.assertEqual(result.metadata["eligible_target_count"], 1)
        self.assertEqual(result.metadata["excluded_target_count"], 3)
        self.assertEqual(result.summary["evaluated_targets"].iloc[0], 1)
        self.assertEqual(result.summary["excluded_targets"].iloc[0], 3)

    def test_evaluation_output_can_be_written(self):
        result = evaluate_rolling(self.history, k_values=[1], min_history=1)
        with tempfile.TemporaryDirectory() as directory:
            result.write(Path(directory))
            self.assertTrue((Path(directory) / "baseline_evaluation_summary.csv").is_file())
            self.assertTrue((Path(directory) / "baseline_evaluation_metadata.json").is_file())


if __name__ == "__main__":
    unittest.main()
