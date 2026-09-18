from contextlib import redirect_stdout
from datetime import date
from io import StringIO
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import plotly.graph_objects as go
import polars as pl
from polars.testing import assert_frame_equal

from wequant.data_processing import (
    CreditbalancePl,
    FinancequotePl,
    IndexPricelistPl,
    MeigaralistPl,
    PricelistPl,
    read_data,
)
from wequant.graph_processing import (
    PricelistFig,
    get_fig_actual_performance_progress_rate_pycharts,
)


class ReadDataTests(TestCase):
    @patch("wequant.data_processing.pl.read_parquet")
    def test_reads_a_path_as_a_parquet_file(self, read_parquet):
        expected = pl.DataFrame({"value": [1]})
        read_parquet.return_value = expected

        actual = read_data(Path("fixtures") / "prices.parquet")

        read_parquet.assert_called_once_with("fixtures/prices.parquet")
        self.assertIs(actual, expected)


class DataFrameWrapperTests(TestCase):
    def test_pricelist_renames_source_columns_and_finds_latest_price(self):
        source = pl.DataFrame(
            {
                "mcode": [1001, 1001, 1001, 2002],
                "p_key": [
                    date(2024, 1, 4),
                    date(2024, 1, 5),
                    date(2024, 1, 9),
                    date(2024, 1, 5),
                ],
                "p_open": [100.0, 105.0, 120.0, 200.0],
                "p_high": [110.0, 115.0, 125.0, 210.0],
                "p_low": [95.0, 100.0, 118.0, 190.0],
                "p_close": [108.0, 112.0, 121.0, 205.0],
                "volume": [10, 20, 30, 40],
            }
        )

        prices = PricelistPl(source)

        self.assertEqual(
            prices.df.columns,
            ["code", "date", "open", "high", "low", "close", "volume"],
        )
        self.assertEqual(
            prices.get_latest_dealingdate_and_price(1001, date(2024, 1, 8)),
            (date(2024, 1, 5), 112.0),
        )

    def test_index_pricelist_uses_first_and_last_trading_days_in_range(self):
        source = pl.DataFrame(
            {
                "date": [
                    date(2024, 1, 4),
                    date(2024, 1, 5),
                    date(2024, 1, 9),
                ],
                "open": [100.0, 105.0, 120.0],
                "high": [110.0, 115.0, 125.0],
                "low": [95.0, 100.0, 118.0],
                "close": [108.0, 112.0, 121.0],
            }
        )

        prices = IndexPricelistPl(source)

        self.assertEqual(
            prices.get_updown_rate(
                date(2024, 1, 6),
                date(2024, 1, 10),
                start_point="open",
                end_point="close",
            ),
            0.83,
        )

    def test_finance_quotes_rename_columns_and_select_latest_available_date(self):
        source = pl.DataFrame(
            {
                "mcode": [1001, 2002, 1001],
                "p_key": [
                    date(2024, 1, 4),
                    date(2024, 1, 5),
                    date(2024, 1, 9),
                ],
                "expected_PER": [10.0, 20.0, 11.0],
            }
        )
        quotes = FinancequotePl(source)

        actual = quotes.get_finance_quotes(valuation_date=date(2024, 1, 8))

        expected = pl.DataFrame(
            {
                "code": [2002],
                "date": [date(2024, 1, 5)],
                "expected_PER": [20.0],
            }
        )
        assert_frame_equal(actual, expected)
        self.assertEqual(quotes.df.shape, (3, 3))

    def test_meigaralist_renames_columns_and_returns_company_name(self):
        companies = MeigaralistPl(
            pl.DataFrame(
                {
                    "mcode": [1001, 2002],
                    "mname": ["テスト商事", "サンプル工業"],
                }
            )
        )

        self.assertEqual(companies.df.columns, ["code", "name"])
        self.assertEqual(companies.get_name(2002), "サンプル工業")


class GraphBehaviorTests(TestCase):
    @patch("wequant.graph_processing.KessanPl")
    def test_actual_progress_chart_has_four_donuts_and_expected_layout(
        self, kessan_class
    ):
        progress = pl.DataFrame(
            {
                "code": [1001],
                "announcement_date": [date(2024, 5, 10)],
                "yearly_settlement_date": [date(2025, 3, 31)],
                "quater": [1],
                "sales_pr(%)": [25.0],
                "operating_income_pr(%)": [40.0],
                "ordinary_profit_pr(%)": [50.0],
                "final_profit_pr(%)": [80.0],
            }
        )
        kessan_class.return_value.get_actual_quatery_settlements_progress_rate.return_value = (
            progress
        )
        companies = pl.DataFrame({"code": [1001], "name": ["テスト商事"]})
        output = StringIO()

        with redirect_stdout(output):
            figure = get_fig_actual_performance_progress_rate_pycharts(
                1001,
                date(2024, 6, 1),
                pl.DataFrame({"code": [1001]}),
                companies,
            )

        self.assertIsInstance(figure, go.Figure)
        self.assertEqual(len(figure.data), 4)
        self.assertEqual(
            [list(trace.values) for trace in figure.data],
            [[25.0, 75.0], [40.0, 60.0], [50.0, 50.0], [80.0, 20.0]],
        )
        self.assertTrue(all(trace.hole == 0.5 for trace in figure.data))
        self.assertFalse(figure.layout.showlegend)
        self.assertEqual(
            [annotation.text for annotation in figure.layout.annotations],
            ["売上高進捗率(%)", "営業利益進捗率(%)", "経常利益進捗率(%)", "純利益進捗率(%)"],
        )
        self.assertEqual(
            output.getvalue().strip(),
            "テスト商事(1001)の2025年3月期第1四半期決算進捗率(評価日：2024-06-01)",
        )

    def test_pricelist_figure_contains_candlestick_and_volume_traces(self):
        prices = pl.DataFrame(
            {
                "code": [1001, 1001],
                "date": [date(2024, 1, 4), date(2024, 1, 5)],
                "open": [100.0, 108.0],
                "high": [110.0, 115.0],
                "low": [95.0, 105.0],
                "close": [108.0, 112.0],
                "volume": [1000, 1500],
            }
        )
        companies = pl.DataFrame({"code": [1001], "name": ["テスト商事"]})

        chart = PricelistFig(
            1001,
            pricelist_df=prices,
            meigaralist_df=companies,
            start_date=date(2024, 1, 4),
            end_date=date(2024, 1, 5),
        )

        self.assertEqual(len(chart.fig.data), 2)
        self.assertEqual(chart.fig.data[0].type, "candlestick")
        self.assertEqual(chart.fig.data[0].name, "株価")
        self.assertEqual(list(chart.fig.data[0].close), [108.0, 112.0])
        self.assertEqual(chart.fig.data[1].type, "bar")
        self.assertEqual(chart.fig.data[1].name, "出来高")
        self.assertEqual(list(chart.fig.data[1].y), [1000, 1500])
        self.assertEqual(chart.fig.layout.yaxis.title.text, "株価")
        self.assertEqual(chart.fig.layout.yaxis2.title.text, "出来高")


class RowPreservingColumnsTests(TestCase):
    def test_moving_average_preserves_rows_and_separates_interleaved_codes(self):
        source = pl.DataFrame({
            "code": [1, 2, 1, 2, 1, 2],
            "date": [date(2024, 1, day) for day in [1, 1, 2, 2, 3, 3]],
            "close": [10., 100., 20., 200., 30., 300.],
        })
        before = source.clone()
        prices = PricelistPl(source)
        self.assertIsNone(prices.with_columns_moving_average(2))
        assert_frame_equal(prices.df.select(source.columns), before)
        self.assertEqual(prices.df["ma2"].to_list(), [None, None, 15., 150., 25., 250.])
        assert_frame_equal(source, before)
        expected = prices.df.clone()
        prices.with_columns_moving_average(2)
        assert_frame_equal(prices.df, expected)

    def test_moving_average_null_windows_short_history_and_custom_column(self):
        source = pl.DataFrame({"code": [1]*5 + [2], "volume": [10., None, 30., 40., 50., 60.]})
        prices = PricelistPl(source)
        prices.with_columns_moving_average(2, col="volume")
        self.assertEqual(prices.df["ma2"].to_list(), [None, None, None, 35., 45., None])
        assert_frame_equal(prices.df.select(source.columns), source)
        prices.with_columns_moving_average(1, col="volume")
        self.assertEqual(prices.df["ma1"].to_list(), source["volume"].to_list())
        prices.with_columns_moving_average(10, col="volume")
        self.assertEqual(prices.df["ma10"].null_count(), source.height)

    def test_margin_ratio_keeps_codes_and_handles_zero_and_missing_values(self):
        source = pl.DataFrame({
            "code": [1, 2, 1, 2, 3, 4, 5],
            "date": [date(2024, 1, day) for day in [1, 1, 2, 2, 1, 1, 1]],
            "purchase_margin": [10., 10., 20., 0., 10., None, 0.],
            "unsold_margin": [3., 0., 0., 2., None, 2., 0.],
        })
        before = source.clone()
        credit = CreditbalancePl(source)
        self.assertIsNone(credit.with_columns_margin_ratio())
        self.assertEqual(credit.df["margin_ratio"].to_list(), [3.33, None, None, 0., None, None, None])
        assert_frame_equal(credit.df.select(source.columns), before)
        assert_frame_equal(source, before)
        expected = credit.df.clone()
        credit.with_columns_margin_ratio()
        assert_frame_equal(credit.df, expected)

    def test_empty_frames_keep_schema_and_add_result_column(self):
        cases = [
            (PricelistPl, {"code": pl.Int64, "close": pl.Float64}, "with_columns_moving_average", (2,), "ma2"),
            (CreditbalancePl, {"code": pl.Int64, "purchase_margin": pl.Float64, "unsold_margin": pl.Float64}, "with_columns_margin_ratio", (), "margin_ratio"),
        ]
        for wrapper, schema, method, args, result in cases:
            with self.subTest(method=method):
                source = pl.DataFrame(schema=schema)
                instance = wrapper(source)
                self.assertIsNone(getattr(instance, method)(*args))
                self.assertEqual(instance.df.height, 0)
                self.assertEqual(instance.df.schema, {**schema, result: pl.Float64})
