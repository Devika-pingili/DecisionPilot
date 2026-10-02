import unittest

from backend.features.history import HistoryStore
from tests.feature_fixture import make_fixture


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.frames = make_fixture()
        self.history = HistoryStore(
            self.frames[0], self.frames[1], self.frames[3], self.frames[4], self.frames[5]
        )

    def test_target_and_later_orders_are_excluded(self):
        orders, transactions = self.history.history_for_target(10, 103, 3)

        self.assertEqual(orders["order_id"].tolist(), [101, 102])
        self.assertNotIn(103, transactions["order_id"].tolist())
        self.assertNotIn(105, transactions["order_id"].tolist())
        self.assertNotIn(2, transactions.loc[transactions["order_id"] == 103, "product_id"].tolist())

    def test_train_target_does_not_enter_history(self):
        orders, transactions = self.history.history_for_target(10, 104, 4)

        self.assertEqual(orders["order_id"].tolist(), [101, 102, 103])
        self.assertNotIn(104, transactions["order_id"].tolist())
        self.assertEqual(set(transactions["product_id"]), {1, 2, 3, 4})

    def test_cutoff_uses_order_number_not_dataframe_position(self):
        shuffled = self.frames[0].sample(frac=1, random_state=7).reset_index(drop=True)
        history = HistoryStore(
            shuffled, self.frames[1], self.frames[3], self.frames[4], self.frames[5]
        )

        orders, _ = history.history_for_target(10, 104, 4)
        self.assertEqual(orders["order_number"].tolist(), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
