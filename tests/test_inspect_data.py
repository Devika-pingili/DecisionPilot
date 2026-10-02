import io
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from backend import inspect_data


class InspectDataTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.files = ["one.csv", "two.csv"]
        for filename in self.files:
            pd.DataFrame({"id": [1, 2], "value": ["a", "b"]}).to_csv(
                self.data_dir / filename, index=False
            )
        self.patches = ExitStack()
        self.patches.enter_context(patch.object(inspect_data, "DATA_DIR", self.data_dir))
        self.patches.enter_context(patch.object(inspect_data, "CSV_FILES", self.files))
        self.patches.enter_context(patch.object(inspect_data, "CHUNK_SIZE", 1))

    def tearDown(self):
        self.patches.close()
        self.temp_dir.cleanup()

    def test_successful_inspection(self):
        output = io.StringIO()
        with redirect_stdout(output):
            inspect_data.main()

        self.assertIn("FILE: one.csv", output.getvalue())
        self.assertIn("Inspection finished. No files were modified.", output.getvalue())

    def test_missing_input_exits_nonzero(self):
        (self.data_dir / "two.csv").unlink()

        with self.assertRaises(SystemExit) as raised:
            inspect_data.main()

        self.assertEqual(raised.exception.code, 1)

    def test_malformed_input_exits_nonzero(self):
        original_read_csv = inspect_data.pd.read_csv

        def read_csv(path, *args, **kwargs):
            if Path(path).name == "two.csv":
                raise ValueError("malformed CSV")
            return original_read_csv(path, *args, **kwargs)

        with patch.object(inspect_data.pd, "read_csv", side_effect=read_csv):
            with self.assertRaises(SystemExit) as raised:
                inspect_data.main()

        self.assertEqual(raised.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
