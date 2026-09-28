from datetime import date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import polars as pl
from polars.testing import assert_frame_equal
from typer.testing import CliRunner

from wequant.cli import app
from wequant.data_processing import IndexPricelistPl, MeigaralistPl
from wequant.tasks.quarterly_valuation import OUTPUT_COLUMNS
from wequant.tasks.ds_quarterly_valuation import build_ds_quarterly_valuation, save_ds_quarterly_valuation
from wequant.flows.ds_quarterly_valuation import ds_quarterly_valuation_flow
from test_quarterly_valuation import kessan, settlement, quotes, prices


def inputs():
    settlements = kessan([
        settlement(1, 2024, 100, 10),
        settlement(1, 2024, 80, 8, month=9),
        settlement(1, 2025, 120, 14),
        settlement(1, 2025, 160, 20, announced=date(2025, 8, 1)),
        settlement(1, 2025, 150, 18, month=9),
        # 前年の将来訂正は2025年7月の特徴量に入れない。
        settlement(2, 2025, 200, 20),
        settlement(2, 2025, 250, 25, month=9),
    ])
    # 初期化後の訂正履歴を追加し、既存スクレイパー補正と独立にas-of条件を検証。
    correction = settlements.df.head(1).with_columns(
        pl.lit(date(2025, 8, 2)).alias("announcement_date"),
        pl.lit(999.).alias("sales"), pl.lit(99.).alias("operating_income"),
    )
    settlements.df = pl.concat([settlements.df, correction])
    q = quotes([(1, date(2025, 7, 18), 10, 2), (1, date(2025, 7, 21), 999, 99)])
    p = prices([
        (1, date(2024, 7, 22), 50), (1, date(2024, 10, 21), 60),
        (1, date(2025, 7, 18), 80), (1, date(2025, 7, 20), 100),
        (1, date(2025, 7, 21), 110),
        (1, date(2025, 10, 20), 125), (1, date(2025, 10, 21), 150),
        (2, date(2025, 7, 21), 200),
        (2, date(2025, 10, 20), 240), (2, date(2025, 10, 21), 260),
    ])
    p.df = p.df.with_columns(pl.col("close").alias("open"))
    names = MeigaralistPl(pl.DataFrame({"code": [1, 2], "name": ["一", "二"]}))
    index = IndexPricelistPl(pl.DataFrame({
        "date": [date(2024, 7, 22), date(2024, 10, 21), date(2025, 7, 18),
                 date(2025, 7, 20), date(2025, 7, 21), date(2025, 10, 20), date(2025, 10, 21)],
        "open": [50., 60., 80., 100., 110., 125., 150.],
    }))
    return settlements, q, p, names, index


class DatasetTests(TestCase):
    def test_all_periods_first_announcement_asof_and_state(self):
        args = inputs()
        snapshots = [obj.df.clone() for obj in args]
        result = build_ds_quarterly_valuation(*args)
        self.assertEqual(result.columns, list(OUTPUT_COLUMNS))
        self.assertGreater(result.filter(pl.col("code") == 1).height, 1)
        self.assertFalse(result.select("code", "setd").is_duplicated().any())
        row = result.filter((pl.col("code") == 1) & (pl.col("setd") == date(2025, 6, 30))).row(0, named=True)
        self.assertEqual(row["annd"], date(2025, 7, 20))
        self.assertEqual(row["sls"], 120)
        self.assertAlmostEqual(row["grsl"], 20)
        self.assertAlmostEqual(row["dgrp"], .2)
        self.assertAlmostEqual(row["pr"], 14 / 120 * 100)
        self.assertAlmostEqual(row["ngrpr"], 40)
        self.assertAlmostEqual(row["PER"], 12.5)
        self.assertAlmostEqual(row["divr"], 1.6)
        self.assertAlmostEqual(row["perf"], (150 / 110 - 1) * 100)
        self.assertAlmostEqual(row["bm"], row["perf"])
        self.assertIsNone(result.filter(pl.col("code") == 2)["PER"][0])
        for obj, before in zip(args, snapshots):
            assert_frame_equal(obj.df, before)

    def test_announcement_profit_and_missing_benchmark(self):
        args = inputs()
        args[0].df = args[0].df.with_columns((pl.col("ordinary_profit") * 2).alias("ordinary_profit"))
        result = build_ds_quarterly_valuation(*args, profit="ordinary", perf_period="announcement")
        row = result.filter((pl.col("code") == 1) & (pl.col("setd") == date(2025, 6, 30))).row(0, named=True)
        self.assertEqual(row["prft"], 28)
        self.assertAlmostEqual(row["perf"], 20)
        self.assertAlmostEqual(row["bm"], 20)
        args[4].df = args[4].df.filter(pl.col("date") != date(2025, 10, 21))
        result = build_ds_quarterly_valuation(*args)
        self.assertTrue(result.filter(pl.col("setd") == date(2025, 6, 30)).is_empty())

    def test_empty_and_no_quotes(self):
        args = inputs()
        args[1].df = args[1].df.clear()
        result = build_ds_quarterly_valuation(*args)
        self.assertGreater(result.height, 0)
        self.assertEqual(result["PER"].null_count(), result.height)
        args[0].df = args[0].df.clear()
        empty = build_ds_quarterly_valuation(*args)
        self.assertEqual(empty.schema, result.schema)
        self.assertTrue(empty.is_empty())

    def test_no_future_trade_date_and_conflicting_duplicates(self):
        args = inputs()
        args[2].df = args[2].df.clear()
        self.assertTrue(build_ds_quarterly_valuation(*args).is_empty())
        args = inputs()
        conflict = args[0].df.head(1).with_columns(pl.lit(999.).alias("sales"))
        args[0].df = pl.concat([args[0].df, conflict])
        with self.assertRaises(ValueError):
            build_ds_quarterly_valuation(*args)

    def test_flow_save_roundtrip_and_collision(self):
        with TemporaryDirectory() as temp, patch("wequant.tasks.ds_quarterly_valuation.DATA_DIR", Path(temp)), patch(
            "wequant.tasks.ds_quarterly_valuation.datetime"
        ) as clock, patch("wequant.flows.ds_quarterly_valuation.load_quarterly_valuation_inputs", return_value=inputs()):
            clock.now.return_value = datetime(2026, 9, 27, 12, 34, 56)
            frame, path = ds_quarterly_valuation_flow()
            self.assertEqual(path.name, "ds-quarterly-valuation-operating-quarter-20260927_123456.parquet")
            self.assertEqual(path.parent, Path(temp) / "datasets")
            assert_frame_equal(frame, pl.read_parquet(path))
            with self.assertRaises(FileExistsError):
                save_ds_quarterly_valuation(frame)
            assert_frame_equal(frame, pl.read_parquet(path))

    def test_cli_options_preview_and_validation(self):
        frame = build_ds_quarterly_valuation(*inputs())
        with patch("wequant.flows.ds_quarterly_valuation.ds_quarterly_valuation_flow", return_value=(frame, Path("sample.parquet"))) as flow:
            runner = CliRunner()
            result = runner.invoke(app, ["ds-quarterly-valuation", "--profit", "ordinary", "--perf-period", "announcement"])
            self.assertEqual(result.exit_code, 0, result.output)
            flow.assert_called_once_with(profit="ordinary", perf_period="announcement")
            for text in ("head 10", "tail 10", "sample.parquet", "PER", "pr"):
                self.assertIn(text, result.output)
            flow.reset_mock()
            self.assertNotEqual(runner.invoke(app, ["ds-quarterly-valuation", "--profit", "invalid"]).exit_code, 0)
            flow.assert_not_called()
