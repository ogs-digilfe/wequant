"""取得APIと更新専用のfilter APIを、人工データで検証する。"""

from datetime import date
from unittest import TestCase
from unittest.mock import patch

import polars as pl
from polars.testing import assert_frame_equal

from wequant.data_processing import FinancequotePl, KessanPl, PortfolioManager


class GetterTests(TestCase):
    def setUp(self):
        # 最新日は銘柄別ではなく全体で選ぶ。順序・同日の複数行も保持する。
        self.source = pl.DataFrame({
            "date": [date(2024, 1, 9), date(2024, 1, 5), date(2024, 1, 4), date(2024, 1, 5), None],
            "code": [1001, 2002, 3003, 1001, 4004],
            "value": [90.0, 50.0, 40.0, 51.0, 0.0],
        })
        self.expected = pl.DataFrame({
            "date": [date(2024, 1, 5), date(2024, 1, 5)],
            "code": [2002, 1001],
            "value": [50.0, 51.0],
        })
        self.apis = [
            (FinancequotePl, "get_finance_quotes", "filter_finance_quotes_by_date", "valuation_date"),
            (PortfolioManager, "get_portfolio_as_of_specific_date", "filter_portfolio_as_of_specific_date", "specific_date"),
        ]

    def test_getters_preserve_state_and_select_latest_date_inclusively(self):
        for cls, getter_name, _, date_arg in self.apis:
            for cutoff in (date(2024, 1, 5), date(2024, 1, 8)):
                with self.subTest(cls=cls.__name__, cutoff=cutoff):
                    instance = cls(self.source)
                    before = instance.df.clone()
                    actual = getattr(instance, getter_name)(**{date_arg: cutoff})
                    assert_frame_equal(actual, self.expected)
                    assert_frame_equal(instance.df, before)
                    # A later query still sees rows beyond the previous cutoff.
                    later = getattr(instance, getter_name)(date(2024, 1, 9))
                    assert_frame_equal(later, self.source.head(1))

    def test_getters_return_empty_frame_with_same_schema_when_no_rows_match(self):
        for cls, getter_name, _, _ in self.apis:
            for frame in (self.source, self.source.clear(), self.source.with_columns(pl.lit(None, dtype=pl.Date).alias("date"))):
                with self.subTest(cls=cls.__name__, rows=frame.height, dates=frame["date"].null_count()):
                    instance = cls(frame)
                    actual = getattr(instance, getter_name)(date(2024, 1, 1))
                    assert_frame_equal(actual, frame.clear())
                    assert_frame_equal(instance.df, frame)

    def test_filters_update_state_and_return_none(self):
        for cls, _, filter_name, _ in self.apis:
            for cutoff, expected in ((date(2024, 1, 5), self.expected), (date(2024, 1, 1), self.source.clear())):
                with self.subTest(cls=cls.__name__, cutoff=cutoff):
                    instance = cls(self.source)
                    actual = getattr(instance, filter_name)(specific_date=cutoff)
                    self.assertIsNone(actual)
                    assert_frame_equal(instance.df, expected)

    def test_filters_no_longer_accept_inplace(self):
        for cls, _, filter_name, _ in self.apis:
            with self.subTest(cls=cls.__name__):
                instance = cls(self.source)
                with self.assertRaises(TypeError):
                    getattr(instance, filter_name)(
                        specific_date=date(2024, 1, 8),
                        inplace=False,
                    )
                assert_frame_equal(instance.df, self.source)

    def test_portfolio_default_date_is_evaluated_at_each_call(self):
        portfolio = PortfolioManager(self.source)
        with patch("wequant.data_processing.date") as date_class:
            date_class.today.side_effect = [date(2024, 1, 5), date(2024, 1, 9)]
            assert_frame_equal(portfolio.get_portfolio_as_of_specific_date(), self.expected)
            assert_frame_equal(portfolio.get_portfolio_as_of_specific_date(), self.source.head(1))
        assert_frame_equal(portfolio.df, self.source)


class PortfolioGetterCallerTests(TestCase):
    def setUp(self):
        self.source = pl.DataFrame({
            "date": [date(2024, 1, 9), date(2024, 1, 5), date(2024, 1, 5), date(2024, 1, 4)],
            "ticker_code": ["1001", "2002", "3003", "4004"],
            "銘柄名": ["未来商事", "テスト商事", "テストETF", "旧商事"],
            "instrument_type": ["個別株", "個別株", "ETF", "個別株"],
        })

    def test_individual_stocks_uses_getter_without_changing_portfolio(self):
        portfolio = PortfolioManager(self.source)
        with patch.object(portfolio, "filter_portfolio_as_of_specific_date", side_effect=AssertionError("legacy filter called")):
            actual = portfolio.get_individual_stocks(date(2024, 1, 8), columns_selected=["ticker_code", "銘柄名"])
        assert_frame_equal(actual, pl.DataFrame({"ticker_code": ["2002"], "銘柄名": ["テスト商事"]}))
        assert_frame_equal(portfolio.df, self.source)

    def test_stock_info_uses_quote_getter_at_the_requested_date(self):
        portfolio = PortfolioManager(self.source)
        quotes = FinancequotePl(pl.DataFrame({
            "code": [2002, 2002],
            "date": [date(2024, 1, 5), date(2024, 1, 9)],
            "expected_PER": [10.0, 99.0],
            "expected_dividend_yield": [2.0, 9.0],
        }))
        quotes_before = quotes.df.clone()
        settlements = pl.DataFrame({
            "code": [2002], "settlement_date": [date(2023, 12, 31)],
            "gr_sales": [5.0], "ordinary_profit": [100.0], "gr_ordinary_profit": [6.0],
        })
        with patch("wequant.data_processing.FinancequotePl.from_file", return_value=quotes), \
             patch("wequant.data_processing.KessanPl.from_file") as load_settlements, \
             patch.object(quotes, "filter_finance_quotes_by_date", side_effect=AssertionError("legacy filter called")):
            load_settlements.return_value.df = settlements
            actual = portfolio.get_individual_stocks_info(date(2024, 1, 8))
        expected = pl.DataFrame({
            "date": [date(2024, 1, 5)], "code": ["2002"], "name": ["テスト商事"],
            "fq-PER": [10.0], "fq-配当率": [2.0], "q-sett": [date(2023, 12, 31)],
            "q-sgr": [5.0], "q-op": [100.0], "q-pgr": [6.0],
        })
        assert_frame_equal(actual, expected)
        assert_frame_equal(portfolio.df, self.source)
        assert_frame_equal(quotes.df, quotes_before)

class KessanGetterStateTests(TestCase):
    columns = [
        "code",
        "settlement_date",
        "settlement_type",
        "announcement_date",
        "sales",
        "operating_income",
        "ordinary_profit",
        "final_profit",
        "reviced_eps",
        "dividend",
        "quater",
    ]

    def setUp(self):
        progress_rows = [
            (1001, date(2024, 12, 31), "本", date(2025, 2, 15), 1000, 200, 180, 120, 50.0, 20.0, -1),
            (1001, date(2024, 3, 31), "四", date(2024, 5, 15), 200, 30, 25, 15, 10.0, 5.0, 1),
            (1001, date(2024, 6, 30), "四", date(2024, 8, 15), 250, 40, 35, 20, 12.0, 5.0, 2),
            (1001, date(2024, 9, 30), "四", date(2024, 11, 15), 300, 50, 45, 30, 15.0, 5.0, 3),
            (1001, date(2024, 12, 31), "予", date(2024, 8, 10), 1100, 220, 200, 130, 55.0, 22.0, -2),
            # 内部の累積処理では除外される行。getter後も元の状態には残る必要がある。
            (9999, date(2024, 3, 31), "四", date(2024, 5, 10), None, 10, 9, 8, 1.0, 1.0, 1),
        ]
        forecast_rows = [
            (1001, date(2022, 12, 31), "本", date(2023, 2, 15), 800, 100, 90, 60, 30.0, 10.0, -1),
            (1001, date(2023, 12, 31), "本", date(2024, 2, 15), 1000, 150, 130, 90, 40.0, 15.0, -1),
            (1001, date(2024, 12, 31), "予", date(2024, 8, 10), 1200, 190, 170, 120, 50.0, 20.0, -2),
        ]
        self.progress_source = pl.DataFrame(progress_rows, schema=self.columns, orient="row")
        self.forecast_source = pl.DataFrame(forecast_rows, schema=self.columns, orient="row")

    def test_progress_getters_preserve_all_source_rows_and_order(self):
        kessan = KessanPl(self.progress_source)
        before = kessan.df.clone()

        actual = kessan.get_actual_quatery_settlements_progress_rate()
        assert_frame_equal(
            actual.select(["settlement_date", "q_sales", "sales_pr(%)"]),
            pl.DataFrame({
                "settlement_date": [date(2024, 3, 31), date(2024, 6, 30), date(2024, 9, 30)],
                "q_sales": [200, 450, 750],
                "sales_pr(%)": [20.0, 45.0, 75.0],
            }),
        )
        assert_frame_equal(kessan.df, before)

        expected = kessan.get_expected_quatery_settlements_progress_rate(date(2024, 9, 1))
        assert_frame_equal(
            expected.select(["settlement_date", "q_sales", "sales_pr(%)"]),
            pl.DataFrame({
                "settlement_date": [date(2024, 3, 31), date(2024, 6, 30)],
                "q_sales": [200, 450],
                "sales_pr(%)": [18.2, 40.9],
            }),
        )
        assert_frame_equal(kessan.df, before)

    def test_diff_growth_forecast_getter_preserves_state(self):
        kessan = KessanPl(self.forecast_source)
        before = kessan.df.clone()

        actual = kessan.get_settlement_forcast_by_diff_growth_rate(date(2024, 9, 1))
        assert_frame_equal(
            actual.select([
                "settlement_type",
                "fcst_dgr_sales",
                "fcst_dgr_operating_income",
                "nxt_settlement_date",
            ]),
            pl.DataFrame({
                "settlement_type": ["本", "予"],
                "fcst_dgr_sales": [1250, 1440],
                "fcst_dgr_operating_income": [215, 240],
                "nxt_settlement_date": [date(2024, 12, 31), date(2025, 3, 31)],
            }),
        )
        assert_frame_equal(kessan.df, before)

    def test_with_columns_methods_remain_update_only(self):
        cases = [
            (self.progress_source, "with_columns_yearly_settlement_date", "yearly_settlement_date"),
            (self.progress_source, "with_columns_accumulated_quaterly_settlement", "acc_sales"),
            (self.forecast_source, "with_columns_diff_growth_rate", "sales_growth_rate"),
            (
                self.forecast_source,
                "with_columns_next_settlement_forcast_by_diff_growth_rate",
                "nxt_settlement_date",
            ),
        ]
        for source, method_name, expected_column in cases:
            with self.subTest(method=method_name):
                kessan = KessanPl(source)
                result = getattr(kessan, method_name)()
                self.assertIsNone(result)
                self.assertIn(expected_column, kessan.df.columns)

