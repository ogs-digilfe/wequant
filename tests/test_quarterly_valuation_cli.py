from datetime import date
from unittest import TestCase
from unittest.mock import patch

import polars as pl
from typer.testing import CliRunner

from wequant.cli import app
from wequant.flows.quarterly_valuation import quarterly_valuation_flow
from wequant.tasks.quarterly_valuation import OUTPUT_COLUMNS, load_quarterly_valuation_inputs


def output_frame(count=1):
    return pl.DataFrame({
        "code": list(range(1000, 1000 + count)),
        "name": ["テスト銘柄"] * count,
        "setd": [date(2026, 6, 30)] * count,
        "annd": [date(2026, 8, 1)] * count,
        "sls": [120] * count,
        "prft": [15] * count,
        "pr": [12.5] * count,
        "grsl": [20.0] * count,
        "dgrp": [0.25678] * count,
        "PER": [12.3456] * count,
        "divr": [None] * count,
        "perf": [10.125] * count,
        "bm": [5.25] * count,
    })


class QuarterlyValuationCliTests(TestCase):
    @patch("wequant.data_processing.load_data_file")
    def test_cli_with_real_flow_and_calculations(self, load):
        data = {
            "nh225.parquet": pl.DataFrame({
                "p_key": [date(2026, 7, 21), date(2026, 10, 19)],
                "p_open": [200., 220.],
            }),
            "meigaralist.parquet": pl.DataFrame({"mcode": [1001], "mname": ["試験銘柄"]}),
            "kessan.parquet": pl.DataFrame({
                "mcode": [1001, 1001],
                "settlement_date": [date(2025, 6, 30), date(2026, 6, 30)],
                "announcement_date": [date(2025, 7, 20), date(2026, 7, 20)],
                "settlement_type": ["四", "四"],
                "sales": [100, 120], "ordinary_profit": [10, 15], "operating_income": [8, 10],
            }),
            "finance_quote.parquet": pl.DataFrame({
                "mcode": [1001], "p_key": [date(2026, 9, 11)],
                "expected_PER": [10.0], "expected_dividend_yield": [4.0],
            }),
            "reviced_pricelist.parquet": pl.DataFrame({
                "mcode": [1001, 1001],
                "p_key": [date(2026, 9, 11), date(2026, 9, 18)],
                "p_close": [100, 120], "p_open": [100, 120],
            }),
        }
        load.side_effect = data.__getitem__
        result = CliRunner().invoke(app, ["quarterly-valuation", "--valuation-date", "2026-09-19"])
        self.assertEqual(result.exit_code, 0, result.exception)
        self.assertEqual(result.stdout.splitlines()[1].split(), [
            "1001", "試験銘柄", "2026-06-30", "2026-07-20", "120", "10", "8.33%", "20.00%", "10.00%", "12.00", "3.33", "null", "null",
        ])
        self.assertEqual(load.call_count, 5)
        result = CliRunner().invoke(app, ["quarterly-valuation", "--valuation-date", "2026-09-19",
                                          "--profit", "ordinary"])
        self.assertEqual(result.exit_code, 0, result.exception)
        self.assertEqual(result.stdout.splitlines()[1].split()[5:9], ["15", "12.50%", "20.00%", "25.00%"])


        for mode, expected_rows in [("operating", 0), ("ordinary", 1)]:
            with self.subTest(profit=mode):
                result = CliRunner().invoke(app, [
                    "quarterly-valuation", "--valuation-date", "2026-09-19",
                    "--profit", mode, "--profit-min", "15", "--profit-max", "15",
                ])
                self.assertEqual(result.exit_code, 0, result.exception)
                self.assertEqual(len(result.stdout.splitlines()), expected_rows + 1)


    @patch("wequant.data_processing.load_data_file")
    def test_perf_with_real_flow_and_future_data(self, load):
        data = {
            "nh225.parquet": pl.DataFrame({
                "p_key": [date(2026, 7, 21), date(2026, 10, 19)],
                "p_open": [200., 220.],
            }),
            "meigaralist.parquet": pl.DataFrame({"code": [1001], "name": ["試験銘柄"]}),
            "kessan.parquet": pl.DataFrame({
                "code": [1001, 1001],
                "settlement_date": [date(2026, 6, 30), date(2026, 9, 30)],
                "announcement_date": [date(2026, 7, 17), date(2026, 10, 16)],
                "settlement_type": ["四", "四"],
                "sales": [100, 120], "ordinary_profit": [10, 15], "operating_income": [8, 10],
            }),
            "finance_quote.parquet": pl.DataFrame(schema={
                "code": pl.Int64, "date": pl.Date,
                "expected_PER": pl.Float64, "expected_dividend_yield": pl.Float64,
            }),
            "reviced_pricelist.parquet": pl.DataFrame({
                "code": [1001, 1001],
                "date": [date(2026, 7, 21), date(2026, 10, 19)],
                "open": [100., 120.], "close": [105., 125.],
            }),
        }
        load.side_effect = data.__getitem__
        result = CliRunner().invoke(app, ["quarterly-valuation", "--valuation-date", "2026-09-19",
                                          "--sort-columns", "bm"])
        self.assertEqual(result.exit_code, 0, result.exception)
        self.assertEqual(result.stdout.splitlines()[0].split()[-2:], ["perf", "bm"])
        self.assertEqual(result.stdout.splitlines()[1].split()[-2:], ["20.00%", "10.00%"])

    @patch("wequant.cli.quarterly_valuation_flow")
    def test_defaults_and_full_output(self, flow):
        flow.return_value = output_frame(30)
        before = date.today()
        result = CliRunner().invoke(app, ["quarterly-valuation"])
        self.assertEqual(result.exit_code, 0, result.exception)
        self.assertIn(flow.call_args.kwargs["valuation_date"], (before, date.today()))
        self.assertEqual(flow.call_args.kwargs["sort_columns"], ["dgrp"])
        self.assertEqual(flow.call_args.kwargs["sort_order"], "desc")
        self.assertEqual(flow.call_args.kwargs["codes"], "all")
        lines = result.stdout.splitlines()
        self.assertEqual(len(lines), 31)
        self.assertEqual(lines[0].split(), list(OUTPUT_COLUMNS))
        self.assertIn("1029", lines[-1])
        self.assertIn("25.68%", lines[-1])
        self.assertIn("12.35", lines[-1])
        self.assertIn("null", lines[-1])
        self.assertEqual(flow.return_value["dgrp"][0], 0.25678)

    @patch("wequant.cli.quarterly_valuation_flow")
    def test_numeric_display_and_rounding_preserves_values(self, flow):
        frame = output_frame(4).with_columns(
            pl.Series("sls", [169927.0, 1234.5, -1234.5, None]),
            pl.Series("prft", [33777.0, 1234.4, -1234.4, None]),
            pl.Series("pr", [19.87654, -10.5, 0.0, None]),
            pl.Series("grsl", [0.27, -10.5, 0.0, None]),
            pl.Series("dgrp", [0.25, -0.125, 0.0, None]),
            pl.Series("perf", [20.125, -10.5, 0.0, None]),
            pl.Series("bm", [20.125, -10.5, 0.0, None]),
        )
        flow.return_value = frame
        before = frame.clone()
        result = CliRunner().invoke(app, ["quarterly-valuation"])
        self.assertEqual(result.exit_code, 0, result.exception)
        cells = [line.split()[4:9] for line in result.stdout.splitlines()[1:]]
        self.assertEqual(cells, [
            ["169,927", "33,777", "19.88%", "0.27%", "25.00%"],
            ["1,235", "1,234", "-10.50%", "-10.50%", "-12.50%"],
            ["-1,235", "-1,234", "0.00%", "0.00%", "0.00%"],
            ["null", "null", "null", "null", "null"],
        ])
        self.assertEqual([line.split()[-1] for line in result.stdout.splitlines()[1:]],
                         ["20.12%", "-10.50%", "0.00%", "null"])
        self.assertTrue(frame.equals(before))

    @patch("wequant.cli.quarterly_valuation_flow")
    def test_options(self, flow):
        flow.return_value = output_frame()
        result = CliRunner().invoke(app, [
            "quarterly-valuation", "--valuation-date", "2026-09-19",
            "--sort-columns", "dgrp", "--sort-columns", "grsl",
            "--sort-order", "asc", "--codes", "7203", "--codes", "6758",
        ])
        self.assertEqual(result.exit_code, 0, result.exception)
        flow.assert_called_once_with(
            valuation_date=date(2026, 9, 19), sort_columns=["dgrp", "grsl"], sort_order="asc", codes=[7203, 6758],
            sls_min=None, sls_max=None, grsl_min=None, grsl_max=None, portfolio=False, start_row=1, end_row=None, profit="operating", profit_min=None, profit_max=None,
        )

    @patch("wequant.cli.quarterly_valuation_flow")
    def test_invalid_options_do_not_read_data(self, flow):
        for args in [
            ["--valuation-date", "2026-02-30"],
            ["--sort-order", "ask"],
            ["--profit", "invalid"],
            ["--dgrp-profit", "ordinary"],
            ["--sort-columns", "ordp"],
            ["--start-row", "0"],
            ["--end-row", "-1"],
            ["--start-row", "3", "--end-row", "2"],
            ["--sls-min", "invalid"],
            ["--profit-min", "invalid"],
            ["--profit-max", "invalid"],
            ["--codes", "all", "--codes", "7203"],
            ["--codes", "invalid"],
            ["--sort-columns", "unknown"],
            ["--sort-columns", "dgrp", "--sort-columns", "dgrp"],
        ]:
            with self.subTest(args=args):
                result = CliRunner().invoke(app, ["quarterly-valuation", *args])
                self.assertEqual(result.exit_code, 2)
        flow.assert_not_called()

    @patch("wequant.cli.quarterly_valuation_flow")
    def test_explicit_all(self, flow):
        flow.return_value = output_frame()
        result = CliRunner().invoke(app, ["quarterly-valuation", "--codes", "all"])
        self.assertEqual(result.exit_code, 0, result.exception)
        self.assertEqual(flow.call_args.kwargs["codes"], "all")

    @patch("wequant.cli.quarterly_valuation_flow")
    def test_range_options(self, flow):
        flow.return_value = output_frame()
        result = CliRunner().invoke(app, [
            "quarterly-valuation", "--sls-min", "100", "--sls-max", "200.5",
            "--grsl-min", "-10", "--grsl-max", "50", "--codes", "7203",
            "--profit-min", "-20.5", "--profit-max", "100",
        ])
        self.assertEqual(result.exit_code, 0, result.exception)
        for key, value in {"sls_min": 100.0, "sls_max": 200.5, "grsl_min": -10.0,
                           "grsl_max": 50.0, "codes": [7203], "profit_min": -20.5, "profit_max": 100.0}.items():
            self.assertEqual(flow.call_args.kwargs[key], value)

    @patch("wequant.cli.quarterly_valuation_flow")
    def test_row_options(self, flow):
        flow.return_value = output_frame()
        result = CliRunner().invoke(app, ["quarterly-valuation", "--start-row", "11", "--end-row", "30"])
        self.assertEqual(result.exit_code, 0, result.exception)
        self.assertEqual(flow.call_args.kwargs["start_row"], 11)
        self.assertEqual(flow.call_args.kwargs["end_row"], 30)

    @patch("wequant.cli.quarterly_valuation_flow")
    def test_ordinary_profit_option(self, flow):
        flow.return_value = output_frame()
        result = CliRunner().invoke(app, ["quarterly-valuation", "--profit", "ordinary"])
        self.assertEqual(result.exit_code, 0, result.exception)
        self.assertEqual(flow.call_args.kwargs["profit"], "ordinary")

    @patch("wequant.cli.quarterly_valuation_flow")
    def test_empty_output_has_header(self, flow):
        flow.return_value = output_frame(0)
        result = CliRunner().invoke(app, ["quarterly-valuation"])
        self.assertEqual(result.exit_code, 0, result.exception)
        self.assertEqual(result.stdout.split(), list(OUTPUT_COLUMNS))


class QuarterlyValuationFlowTests(TestCase):
    @patch("wequant.flows.quarterly_valuation.build_quarterly_valuation")
    @patch("wequant.flows.quarterly_valuation.load_quarterly_valuation_inputs")
    def test_flow_passes_inputs_and_options(self, load, build):
        load.return_value = (object(), object(), object(), object(), object())
        expected = build.return_value = output_frame()
        result = quarterly_valuation_flow(date(2026, 9, 19), ["grsl"], "asc", codes=[7203],
                                          sls_min=100, sls_max=200, grsl_min=-10, grsl_max=50, start_row=2, end_row=5, profit="ordinary", profit_min=-20.5, profit_max=100)
        self.assertIs(result, expected)
        load.assert_called_once_with()
        build.assert_called_once_with(
            *load.return_value, valuation_date=date(2026, 9, 19),
            sort_columns=["grsl"], sort_order="asc", codes=[7203],
            sls_min=100, sls_max=200, grsl_min=-10, grsl_max=50, start_row=2, end_row=5, profit="ordinary", profit_min=-20.5, profit_max=100,
        )

    @patch("wequant.tasks.quarterly_valuation.IndexPricelistPl.from_file")
    @patch("wequant.tasks.quarterly_valuation.MeigaralistPl.from_file")
    @patch("wequant.tasks.quarterly_valuation.PricelistPl.from_file")
    @patch("wequant.tasks.quarterly_valuation.FinancequotePl.from_file")
    @patch("wequant.tasks.quarterly_valuation.KessanPl.from_file")
    def test_loader_uses_adjusted_prices(self, settlements, quotes, prices, names, index):
        self.assertEqual(load_quarterly_valuation_inputs(), (
            settlements.return_value, quotes.return_value, prices.return_value, names.return_value, index.return_value,
        ))
        index.assert_called_once_with()
        names.assert_called_once_with()
        settlements.assert_called_once_with()
        quotes.assert_called_once_with()
        prices.assert_called_once_with("reviced_pricelist.parquet")
