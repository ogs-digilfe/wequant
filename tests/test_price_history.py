from datetime import date, timedelta
from unittest import TestCase
from unittest.mock import patch

import polars as pl
from polars.testing import assert_frame_equal
from typer.testing import CliRunner

from wequant.cli import app
from wequant.data_processing import PricelistPl
from wequant.flows.price_history import price_history_flow
from wequant.tasks.price_history import OUTPUT_COLUMNS, select_price_history


def prices(codes, dates):
    return PricelistPl(pl.DataFrame({
        "mcode": codes, "p_key": dates,
        "p_open": [100.125] * len(codes), "p_high": [110.0] * len(codes),
        "p_low": [90.0] * len(codes), "p_close": [None] * len(codes),
        "volume": [1000.0] * len(codes),
    }))


class PriceHistoryTests(TestCase):
    def test_boundaries_sort_and_code_types_without_mutation(self):
        for code in (7203, "130A"):
            with self.subTest(code=code):
                source = prices(
                    [code, code, code, code, 9999 if isinstance(code, int) else "9999"],
                    [date(2026, 1, d) for d in (3, 1, 2, 4, 2)],
                )
                original = source.df.clone()
                result = select_price_history(
                    source, code=str(code), start_date=date(2026, 1, 1), end_date=date(2026, 1, 3),
                )
                self.assertEqual(result.columns, list(OUTPUT_COLUMNS))
                self.assertEqual(result["date"].to_list(), [date(2026, 1, d) for d in (1, 2, 3)])
                self.assertEqual(result["code"].to_list(), [str(code)] * 3)
                assert_frame_equal(source.df, original)

    @patch("wequant.tasks.price_history.PricelistPl.from_file")
    def test_cli_flow_sources_and_complete_output(self, load):
        dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(40)]
        load.return_value = prices(["130A"] * 40, dates[::-1])
        for source in (None, "raw", "reviced"):
            with self.subTest(source=source):
                args = ["price-history", "--code", "130A", "--start-date", "2026-01-01",
                        "--end-date", "2026-03-31"]
                if source:
                    args += ["--source", source]
                result = CliRunner().invoke(app, args)
                self.assertEqual(result.exit_code, 0, result.output)
                load.assert_called_with(f"{source or 'reviced'}_pricelist.parquet")
                lines = result.stdout.splitlines()
                self.assertEqual(len(lines), 41)
                self.assertEqual(lines[0].split(), list(OUTPUT_COLUMNS))
                self.assertEqual(lines[1].split(), ["130A", "2026-01-01", "100.125", "110.0", "90.0", "null", "1000.0"])
                self.assertIn("2026-02-09", lines[-1])
                self.assertEqual(result.stderr, "")

    @patch("wequant.tasks.price_history.PricelistPl.from_file")
    def test_empty_result(self, load):
        load.return_value = prices([7203], [date(2026, 1, 1)])
        result = CliRunner().invoke(app, [
            "price-history", "--code", "130A", "--start-date", "2026-01-01", "--end-date", "2026-01-01",
        ])
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.stdout.split(), list(OUTPUT_COLUMNS))
        self.assertIn("該当データなし", result.stderr)

    @patch("wequant.cli.price_history_flow")
    def test_invalid_arguments_do_not_load(self, flow):
        base = ["price-history", "--code", "7203", "--start-date", "2026-01-01", "--end-date", "2026-01-31"]
        cases = [
            ["price-history"],
            base + ["--source", "other"],
            base + ["--code", "13-A"],
            base + ["--code", ""],
            base + ["--start-date", "2026-02-30"],
            base + ["--start-date", "2026-02-01"],
        ]
        for args in cases:
            with self.subTest(args=args):
                result = CliRunner().invoke(app, args)
                self.assertNotEqual(result.exit_code, 0)
        flow.assert_not_called()

    @patch("wequant.tasks.price_history.PricelistPl.from_file")
    def test_file_errors(self, load):
        for exc in (ValueError("ファイルがありません"), OSError("読み込み失敗"),
                    pl.exceptions.ColumnNotFoundError("close")):
            with self.subTest(exc=exc):
                load.side_effect = exc
                result = CliRunner().invoke(app, [
                    "price-history", "--code", "7203", "--start-date", "2026-01-01", "--end-date", "2026-01-31",
                ])
                self.assertEqual(result.exit_code, 1)
                self.assertEqual(result.stdout, "")
                self.assertIn("株価履歴を取得できません", result.stderr)

    @patch("wequant.tasks.price_history.PricelistPl.from_file")
    def test_flow_rejects_invalid_range_and_source_before_loading(self, load):
        with self.assertRaises(ValueError):
            price_history_flow(code="130A", start_date=date(2026, 2, 1), end_date=date(2026, 1, 1))
        with self.assertRaises(ValueError):
            price_history_flow(code="130A", start_date=date(2026, 1, 1), end_date=date(2026, 1, 1), source="other")
        load.assert_not_called()
