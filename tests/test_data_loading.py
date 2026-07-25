from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import polars as pl

from wequant import data_loading


class ReadDataTests(TestCase):
    @patch("wequant.data_loading.pl.read_parquet")
    def test_reads_a_path_as_a_parquet_file(self, read_parquet):
        expected = pl.DataFrame({"value": [1]})
        read_parquet.return_value = expected

        actual = data_loading.read_data(Path("fixtures") / "prices.parquet")

        read_parquet.assert_called_once_with("fixtures/prices.parquet")
        self.assertIs(actual, expected)


class ResolveDataPathTests(TestCase):
    def test_resolves_a_managed_filename_from_the_data_directory(self):
        with TemporaryDirectory() as directory:
            expected = Path(directory) / "raw_pricelist.parquet"
            expected.touch()

            with patch.object(data_loading, "DATA_DIR", Path(directory)):
                actual = data_loading.resolve_data_path("raw_pricelist.parquet")

        self.assertEqual(actual, expected)

    def test_rejects_an_unmanaged_filename(self):
        with self.assertRaisesRegex(ValueError, "wequantで管理していない"):
            data_loading.resolve_data_path("unknown.parquet")

    @patch("wequant.data_loading.read_data")
    @patch("wequant.data_loading.resolve_data_path")
    def test_loads_the_resolved_data_file(self, resolve_data_path, read_data):
        resolved = Path("data") / "kessan.parquet"
        expected = pl.DataFrame({"value": [1]})
        resolve_data_path.return_value = resolved
        read_data.return_value = expected

        actual = data_loading.load_data_file("kessan.parquet")

        resolve_data_path.assert_called_once_with("kessan.parquet")
        read_data.assert_called_once_with(resolved)
        self.assertIs(actual, expected)
