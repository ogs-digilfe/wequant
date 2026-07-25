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

        actual = quotes.filter_finance_quotes_by_date(date(2024, 1, 8))

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
