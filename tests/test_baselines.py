import unittest

import pandas as pd

from backend.evaluation.baselines import (
    customer_frequency,
    customer_recency,
    department_aware_popularity,
    global_popularity,
    previously_purchased,
)
from backend.features.history import HistoryStore
from tests.feature_fixture import make_fixture


class BaselineTests(unittest.TestCase):
    def setUp(self):
        frames = make_fixture()
        self.orders, self.prior, _, products, aisles, departments = frames
        self.history = HistoryStore(self.orders, self.prior, products, aisles, departments)
        self.candidates = [1, 2, 3, 4]

    def test_frequency_is_point_in_time_and_tie_breaks_by_product_id(self):
        ranking = customer_frequency(self.history, 10, 104, 4, self.candidates)

        self.assertEqual(ranking, [1, 2, 3, 4])

    def test_recency_excludes_target_and_orders_after_target(self):
        ranking = customer_recency(self.history, 10, 102, 2, [1, 2, 3])

        self.assertEqual(ranking, [1, 2, 3])

    def test_previously_purchased_uses_recency_frequency_reorder_rate(self):
        ranking = previously_purchased(self.history, 10, 104, 4, self.candidates)

        self.assertEqual(ranking, [2, 4, 1, 3])

    def test_global_popularity_is_cutoff_specific(self):
        early = global_popularity(self.history, 10, 102, 2, [1, 2, 3])
        late = global_popularity(self.history, 10, 104, 4, self.candidates)

        self.assertEqual(early, [2, 1, 3])
        self.assertEqual(late, [2, 1, 3, 4])

    def test_department_aware_prioritizes_active_departments(self):
        products = pd.concat([
            self.history.products,
            pd.DataFrame([[6, "Coffee", 13, 102]], columns=self.history.products.columns),
        ], ignore_index=True)
        history = HistoryStore(
            self.orders, self.prior, products, self.history.aisles, self.history.departments
        )

        ranking = department_aware_popularity(history, 10, 104, 4, [6, 1, 2, 3, 4])

        self.assertEqual(ranking[-1], 6)
        self.assertEqual(set(ranking[:-1]), {1, 2, 3, 4})

    def test_rankers_return_only_supplied_candidates(self):
        ranking = customer_frequency(self.history, 10, 104, 4, [4, 99])

        self.assertEqual(set(ranking), {4, 99})


if __name__ == "__main__":
    unittest.main()
