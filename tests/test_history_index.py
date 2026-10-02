import hashlib
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backend.features.candidates import generate_candidates
from backend.features.history import HistoryStore, prepare_history_index
from backend.features.point_in_time import build_point_in_time_features
from tests.feature_fixture import make_fixture


class HistoryIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.raw_dir = self.root / "raw" / "instacart"
        self.raw_dir.mkdir(parents=True)
        frames = make_fixture()
        self.orders, self.prior, self.train, self.products, self.aisles, self.departments = frames
        self.orders.to_csv(self.raw_dir / "orders.csv", index=False)
        self.prior.to_csv(self.raw_dir / "order_products__prior.csv", index=False)
        self.train.to_csv(self.raw_dir / "order_products__train.csv", index=False)
        self.products.to_csv(self.raw_dir / "products.csv", index=False)
        self.aisles.to_csv(self.raw_dir / "aisles.csv", index=False)
        self.departments.to_csv(self.raw_dir / "departments.csv", index=False)
        self.index_path = self.root / "processed" / "history.sqlite"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_chunked_index_matches_dataframe_history_and_features(self):
        prepare_history_index(self.raw_dir, self.index_path, chunk_size=2, progress=False)
        indexed = HistoryStore.from_csv(
            self.raw_dir, index_path=self.index_path, progress=False
        )
        frame = HistoryStore(
            self.orders, self.prior, self.products, self.aisles, self.departments
        )
        candidates_frame = generate_candidates(frame, 10, 104, 4, candidate_limit=20)
        candidates_index = generate_candidates(indexed, 10, 104, 4, candidate_limit=20)
        pd.testing.assert_frame_equal(candidates_frame, candidates_index)
        features_frame = build_point_in_time_features(frame, candidates_frame)
        features_index = build_point_in_time_features(indexed, candidates_index)
        pd.testing.assert_frame_equal(features_frame, features_index, check_dtype=False)
        indexed.close()

    def test_cutoff_statistics_are_as_of_the_cutoff(self):
        prepare_history_index(self.raw_dir, self.index_path, chunk_size=2, progress=False)
        indexed = HistoryStore.from_csv(
            self.raw_dir, index_path=self.index_path, progress=False
        )

        early = indexed.product_statistics(2, [3])
        late = indexed.product_statistics(4, [3])
        self.assertEqual(early.empty, True)
        self.assertEqual(int(late.iloc[0]["product_global_purchase_count"]), 1)
        indexed.close()

    def test_repeated_preparation_is_deterministic(self):
        prepare_history_index(self.raw_dir, self.index_path, chunk_size=2, progress=False)
        first = HistoryStore.from_csv(self.raw_dir, index_path=self.index_path, progress=False)
        first_candidates = generate_candidates(first, 10, 104, 4, candidate_limit=20)
        first_features = build_point_in_time_features(first, first_candidates)
        first.close()
        first_hash = hashlib.sha256(self.index_path.read_bytes()).hexdigest()

        prepare_history_index(self.raw_dir, self.index_path, chunk_size=2, progress=False)
        second = HistoryStore.from_csv(self.raw_dir, index_path=self.index_path, progress=False)
        second_candidates = generate_candidates(second, 10, 104, 4, candidate_limit=20)
        second_features = build_point_in_time_features(second, second_candidates)
        second.close()
        second_hash = hashlib.sha256(self.index_path.read_bytes()).hexdigest()

        self.assertEqual(first_hash, second_hash)
        pd.testing.assert_frame_equal(first_candidates, second_candidates)
        pd.testing.assert_frame_equal(first_features, second_features, check_dtype=False)

    def test_train_file_is_never_read_and_raw_files_remain_unchanged(self):
        for path in self.raw_dir.iterdir():
            path.write_bytes(path.read_bytes())
        raw_hashes = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in self.raw_dir.iterdir()
        }
        (self.raw_dir / "order_products__train.csv").write_text("not a valid CSV")
        prepare_history_index(self.raw_dir, self.index_path, chunk_size=2, progress=False)
        indexed = HistoryStore.from_csv(
            self.raw_dir, index_path=self.index_path, progress=False
        )
        indexed.close()
        self.assertEqual(
            raw_hashes["orders.csv"],
            hashlib.sha256((self.raw_dir / "orders.csv").read_bytes()).hexdigest(),
        )
        self.assertEqual(
            (self.raw_dir / "order_products__train.csv").read_text(),
            "not a valid CSV",
        )


if __name__ == "__main__":
    unittest.main()
