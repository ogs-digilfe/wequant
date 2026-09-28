from contextlib import redirect_stderr
from datetime import date
from io import StringIO
from unittest import TestCase
from unittest.mock import patch

import polars as pl
from polars.testing import assert_frame_equal
from typer.testing import CliRunner

from wequant.cli import app
from wequant.data_processing import PortfolioManager
from wequant.flows.quarterly_valuation import quarterly_valuation_flow
from wequant.tasks.quarterly_valuation import load_portfolio_codes


class PortfolioCodesTests(TestCase):
    @patch("wequant.tasks.quarterly_valuation.PortfolioManager.from_file")
    def test_latest_snapshot_individual_stocks_and_invalid_codes(self, load):
        frame = pl.DataFrame({
            "date": [date(2026, 9, 1)] * 7 + [date(2026, 8, 1), date(2026, 9, 2)],
            "ticker_code": ["7203", "7203", "6758", "1306", "ABCD", None, "1.5", "5334", "9999"],
            "instrument_type": ["個別株"] * 3 + ["ETF"] + ["個別株"] * 5,
        })
        portfolio = load.return_value = PortfolioManager(frame)
        before = portfolio.df.clone()
        err = StringIO()
        with redirect_stderr(err):
            actual = load_portfolio_codes(date(2026, 9, 1))
        self.assertEqual(actual, [6758, 7203])
        self.assertIn("3件除外", err.getvalue())
        self.assertNotIn("ABCD", err.getvalue())
        assert_frame_equal(portfolio.df, before)
        load.assert_called_once_with()

    @patch("wequant.tasks.quarterly_valuation.PortfolioManager.from_file")
    def test_empty_or_latest_etf_only_does_not_revive_old_stock(self, load):
        frame = pl.DataFrame({
            "date": [date(2026, 8, 1), date(2026, 9, 1)],
            "ticker_code": ["5334", "1306"], "instrument_type": ["個別株", "ETF"],
        })
        for cutoff, expected in [(date(2026, 7, 1), []), (date(2026, 8, 31), [5334]),
                                 (date(2026, 9, 1), [])]:
            with self.subTest(cutoff=cutoff):
                load.return_value = PortfolioManager(frame)
                self.assertEqual(load_portfolio_codes(cutoff), expected)
        load.return_value = PortfolioManager(frame.clear())
        self.assertEqual(load_portfolio_codes(date(2026, 9, 1)), [])


class PortfolioFlowTests(TestCase):
    @patch("wequant.flows.quarterly_valuation.build_quarterly_valuation")
    @patch("wequant.flows.quarterly_valuation.load_quarterly_valuation_inputs")
    @patch("wequant.flows.quarterly_valuation.load_portfolio_codes")
    def test_flag_and_intersection(self, holdings, load, build):
        load.return_value = (object(), object(), object(), object(), object())
        holdings.return_value = [5334, 7203]
        for enabled, codes, expected in [
            (False, "all", "all"), (True, "all", [5334, 7203]),
            (True, [7203, 6758, 7203], [7203]), (True, [], []), (True, [6758], []),
        ]:
            with self.subTest(enabled=enabled, codes=codes):
                holdings.reset_mock()
                result = quarterly_valuation_flow(date(2026, 9, 1), codes=codes,
                                                  portfolio=enabled, grsl_min=10)
                self.assertIs(result, build.return_value)
                self.assertEqual(build.call_args.kwargs["codes"], expected)
                self.assertEqual(build.call_args.kwargs["grsl_min"], 10)
                if enabled:
                    holdings.assert_called_once_with(date(2026, 9, 1))
                else:
                    holdings.assert_not_called()
        holdings.return_value = []
        quarterly_valuation_flow(date(2026, 9, 1), portfolio=True)
        self.assertEqual(build.call_args.kwargs["codes"], [])


class PortfolioCliIntegrationTests(TestCase):
    @patch("wequant.data_processing.load_data_file")
    def test_real_flow_filtering_warning_and_empty_output(self, load):
        data = {
            "nh225.parquet": pl.DataFrame(schema={"date": pl.Date, "open": pl.Float64}),
            "base_portfolio.parquet": pl.DataFrame({
                "date": [date(2026, 9, 1)] * 4,
                "ticker_code": ["5334", "7203", "9999", "INVALID"],
                "instrument_type": ["個別株"] * 4,
            }),
            "kessan.parquet": pl.DataFrame({
                "code": [5334, 5334, 7203, 7203, 6758],
                "settlement_date": [date(2025, 6, 30), date(2026, 6, 30)] * 2 + [date(2026, 6, 30)],
                "announcement_date": [date(2025, 7, 31), date(2026, 7, 31)] * 2 + [date(2026, 7, 31)],
                "settlement_type": ["四"] * 5,
                "quater": [1] * 5,
                "operating_income": [8, 16, 8, 12, 16],
                "sales": [100, 150, 100, 105, 200], "ordinary_profit": [10, 20, 10, 15, 20],
            }),
            "finance_quote.parquet": pl.DataFrame(schema={"code": pl.Int64, "date": pl.Date,
                "expected_PER": pl.Float64, "expected_dividend_yield": pl.Float64}),
            "reviced_pricelist.parquet": pl.DataFrame(schema={"code": pl.Int64, "date": pl.Date, "close": pl.Float64, "open": pl.Float64}),
            "meigaralist.parquet": pl.DataFrame({"code": [5334, 7203, 6758], "name": ["A社", "B社", "C社"]}),
        }
        load.side_effect = data.__getitem__
        base_args = ["quarterly-valuation", "--valuation-date", "2026-09-19", "--portfolio"]
        for args, expected in [([], {5334, 7203}), (["--grsl-min", "10"], {5334}),
                               (["--codes", "7203"], {7203}), (["--codes", "9999"], set())]:
            with self.subTest(args=args):
                result = CliRunner().invoke(app, base_args + args)
                self.assertEqual(result.exit_code, 0, result.exception)
                self.assertEqual({int(line.split()[0]) for line in result.stdout.splitlines()[1:]}, expected)
                self.assertIn("1件除外", result.stderr)
                self.assertNotIn("除外", result.stdout)
                self.assertNotIn("INVALID", result.stderr)
