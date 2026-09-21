from datetime import date, timedelta
from unittest import TestCase

import polars as pl
from polars.testing import assert_frame_equal

from wequant.data_processing import FinancequotePl, KessanPl, MeigaralistPl, PricelistPl
from wequant.tasks.quarterly_valuation import OUTPUT_COLUMNS, build_quarterly_valuation


VALUATION_DATE = date(2026, 9, 19)


def settlement(code, year, sales, profit, *, month=6, announced=None, kind="四"):
    return (code, date(year, month, 30), kind, announced or date(year, month + 1, 20), sales, profit)


def kessan(rows):
    # 通常の初期化を通す。日付補正対象でない人工データのみ使用する。
    return KessanPl(pl.DataFrame(rows, schema={
        "code": pl.Int64, "settlement_date": pl.Date, "settlement_type": pl.String,
        "announcement_date": pl.Date, "sales": pl.Float64, "ordinary_profit": pl.Float64,
    }, orient="row").with_columns(pl.col("ordinary_profit").alias("operating_income")))


def quotes(rows):
    return FinancequotePl(pl.DataFrame(rows, schema={
        "code": pl.Int64, "date": pl.Date,
        "expected_PER": pl.Float64, "expected_dividend_yield": pl.Float64,
    }, orient="row"))


def prices(rows):
    return PricelistPl(pl.DataFrame(rows, schema={
        "code": pl.Int64, "date": pl.Date, "close": pl.Float64,
    }, orient="row").with_columns(pl.lit(None, dtype=pl.Float64).alias("open")))


class QuarterlyCalculationTests(TestCase):
    def test_year_month_match_latest_and_date_boundaries(self):
        lower = VALUATION_DATE - timedelta(days=93)
        rows = [
            settlement(1, 2025, 100, 10),
            settlement(1, 2026, 120, 14),
            settlement(1, 2026, 150, 25, announced=date(2026, 8, 1)),
            settlement(1, 2026, 999, 999, announced=VALUATION_DATE),
            # 古い決算期の発表が新しくても、新しい決算期を優先する。
            settlement(1, 2026, 999, 999, month=3, announced=date(2026, 9, 1)),
            settlement(2, 2025, 200, 20, month=3),
            settlement(2, 2026, 240, 40, month=3, announced=lower),
            settlement(3, 2026, 300, 30, month=3, announced=lower - timedelta(days=1)),
            settlement(4, 2026, 400, 40, announced=VALUATION_DATE),
            settlement(5, 2026, 500, 50, announced=VALUATION_DATE + timedelta(days=1)),
            settlement(6, 2026, 600, 60, kind="本"),
            settlement(7, 2025, 700, 70, month=3),  # 前年の別月は対応させない。
            settlement(7, 2026, 770, 77),
        ]
        source = kessan(list(reversed(rows)))
        before = source.df.clone()
        result = source.get_quarterly_valuation(VALUATION_DATE)
        self.assertEqual(result["code"].to_list(), [1, 2, 7])
        self.assertAlmostEqual(result["grsl"][0], 50)
        self.assertAlmostEqual(result["dgrp"][0], 0.3)
        self.assertEqual(result["sls"][0], 150)
        self.assertEqual(result["annd"][1], lower)
        self.assertAlmostEqual(result["dgrp"][1], 0.5)
        self.assertIsNone(result["grsl"][2])
        assert_frame_equal(source.df, before)

    def test_invalid_inputs_and_valid_negative_or_zero_results(self):
        cases = [
            # 前年売上、前年利益、当年売上、当年利益、grsl、dgrp
            (100, 10, 120, 15, 20, 0.25),
            (100, 10, 100, 15, 0, None),
            (100, -10, 120, 15, 20, None),
            (100, 10, 120, -5, 20, None),
            (100, 0, 120, 15, 20, None),
            (100, 10, 120, 0, 20, None),
            (0, 10, 120, 15, None, None),
            (100, 10, 0, 15, None, None),
            (None, 10, 120, 15, None, None),
            (100, 10, 120, None, 20, None),
            (100, 10, 80, 5, -20, 0.25),
            (100, 10, 120, 5, 20, -0.25),
            (100, 10, 120, 10, 20, 0),
            (100, 10, float("nan"), 15, None, None),
            (100, 10, 120, float("inf"), 20, None),
            (-100, 10, 120, 15, None, None),
        ]
        rows = []
        for code, (old_sales, old_profit, sales, profit, _, _) in enumerate(cases):
            rows.extend([settlement(code, 2025, old_sales, old_profit), settlement(code, 2026, sales, profit)])
        result = kessan(rows).get_quarterly_valuation(VALUATION_DATE)
        self.assertEqual(result.height, len(cases))
        for row, case in zip(result.iter_rows(named=True), cases):
            for column, expected in zip(("grsl", "dgrp"), case[-2:]):
                with self.subTest(code=row["code"], column=column):
                    if expected is None:
                        self.assertIsNone(row[column])
                    else:
                        self.assertAlmostEqual(row[column], expected)

    def test_duplicates_and_empty(self):
        row = settlement(1, 2026, 120, 15)
        self.assertEqual(kessan([row, row]).get_quarterly_valuation(VALUATION_DATE).height, 1)
        with self.assertRaisesRegex(ValueError, "異なる決算値"):
            kessan([row, settlement(1, 2026, 130, 15)]).get_quarterly_valuation(VALUATION_DATE)
        for source in (kessan([]), kessan([settlement(1, 2025, 100, 10)])):
            result = source.get_quarterly_valuation(VALUATION_DATE)
            self.assertEqual(result.height, 0)
            self.assertEqual(result.columns, [c for c in OUTPUT_COLUMNS if c not in ("name", "PER", "divr", "perf", "bm")])


class PriceAdjustmentTests(TestCase):
    def test_per_code_latest_and_exact_base_price(self):
        source = quotes([
            (1, date(2026, 9, 11), 10, 4),
            (1, date(2026, 9, 4), 99, 99),
            (1, VALUATION_DATE, 999, 999),
            (2, date(2026, 9, 10), 20, 0),
            (3, date(2026, 9, 10), 10, 2),
        ])
        price_source = prices([
            (1, VALUATION_DATE, 999), (1, date(2026, 9, 18), 120),
            (1, date(2026, 9, 11), 100),
            (2, date(2026, 9, 10), 100), (2, date(2026, 9, 17), 80),
            (3, date(2026, 9, 9), 100), (3, date(2026, 9, 18), 120),
        ])
        before_quotes, before_prices = source.df.clone(), price_source.df.clone()
        result = source.get_price_adjusted_valuations(price_source.df, VALUATION_DATE)
        self.assertEqual(result["PER"].to_list(), [12, 16, None])
        self.assertAlmostEqual(result["divr"][0], 4 * 100 / 120)
        self.assertEqual(result["divr"][1], 0)
        self.assertIsNone(result["divr"][2])
        assert_frame_equal(source.df, before_quotes)
        assert_frame_equal(price_source.df, before_prices)

    def test_null_zero_negative_and_nonfinite(self):
        cases = [
            (0, 2, 100, 120, None, 2 * 100 / 120),
            (-1, 2, 100, 120, None, 2 * 100 / 120),
            (10, -1, 100, 120, 12, None),
            (None, None, 100, 120, None, None),
            (10, 2, 0, 120, None, None),
            (10, 2, 100, 0, None, None),
            (10, 2, None, 120, None, None),
            (float("nan"), float("inf"), 100, 120, None, None),
        ]
        source = quotes([(i, date(2026, 9, 11), c[0], c[1]) for i, c in enumerate(cases)])
        ps = prices([row for i, c in enumerate(cases) for row in (
            (i, date(2026, 9, 11), c[2]), (i, date(2026, 9, 18), c[3]),
        )])
        result = source.get_price_adjusted_valuations(ps.df, VALUATION_DATE)
        for row, case in zip(result.iter_rows(named=True), cases):
            for column, expected in zip(("PER", "divr"), case[-2:]):
                with self.subTest(code=row["code"], column=column):
                    if expected is None:
                        self.assertIsNone(row[column])
                    else:
                        self.assertAlmostEqual(row[column], expected)

    def test_empty_missing_and_conflicting_duplicates(self):
        q = (1, date(2026, 9, 11), 10, 2)
        p = (1, date(2026, 9, 11), 100)
        result = quotes([q, q]).get_price_adjusted_valuations(prices([p, p]).df, VALUATION_DATE)
        self.assertEqual(result.height, 1)
        self.assertEqual(result["PER"][0], 10)
        self.assertEqual(quotes([]).get_price_adjusted_valuations(prices([]).df, VALUATION_DATE).height, 0)
        result = quotes([q]).get_price_adjusted_valuations(prices([]).df, VALUATION_DATE)
        self.assertEqual(result["PER"].to_list(), [None])
        with self.assertRaises(ValueError):
            quotes([q, (1, q[1], 20, 2)]).get_price_adjusted_valuations(prices([p]).df, VALUATION_DATE)
        with self.assertRaises(ValueError):
            quotes([q]).get_price_adjusted_valuations(prices([p, (1, p[1], 200)]).df, VALUATION_DATE)


class QuarterlyBuildTests(TestCase):
    def test_join_sort_null_last_and_state(self):
        source = kessan([
            settlement(3, 2026, 120, 15),  # 比較先なしも行を残す。
            settlement(2, 2025, 100, 10), settlement(2, 2026, 120, 15),
            settlement(1, 2025, 100, 10), settlement(1, 2026, 140, 20),
        ])
        q = quotes([(1, date(2026, 9, 11), 10, 2)])
        p = prices([(1, date(2026, 9, 11), 100), (1, date(2026, 9, 18), 120)])
        names = MeigaralistPl(pl.DataFrame({"code": [1, 2], "name": ["一社", "二社"]}))
        before = [obj.df.clone() for obj in (source, q, p, names)]
        for order in ("asc", "desc"):
            result = build_quarterly_valuation(source, q, p, names, valuation_date=VALUATION_DATE, sort_order=order)
            self.assertEqual(result.columns, list(OUTPUT_COLUMNS))
            self.assertEqual(result["code"].to_list(), [1, 2, 3])
            self.assertEqual(result["PER"].to_list(), [12, None, None])
            self.assertEqual(result["name"].to_list(), ["一社", "二社", None])
        result = build_quarterly_valuation(source, q, p, names, valuation_date=VALUATION_DATE,
                                          sort_columns=["dgrp", "grsl"], sort_order="asc")
        self.assertEqual(result["code"].to_list(), [2, 1, 3])
        for obj, original in zip((source, q, p, names), before):
            assert_frame_equal(obj.df, original)
        result = build_quarterly_valuation(kessan([]), quotes([]), prices([]),
                                          MeigaralistPl(pl.DataFrame(schema={"code": pl.Int64, "name": pl.String})), valuation_date=VALUATION_DATE)
        self.assertEqual(result.height, 0)
        self.assertEqual(result.columns, list(OUTPUT_COLUMNS))

    def test_bad_sort_options(self):
        for kwargs in ({"sort_columns": []}, {"sort_columns": ["unknown"]},
                       {"sort_columns": ["code", "code"]}, {"sort_order": "ask"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_quarterly_valuation(kessan([]), quotes([]), prices([]),
                                          MeigaralistPl(pl.DataFrame(schema={"code": pl.Int64, "name": pl.String})),
                                          valuation_date=VALUATION_DATE, **kwargs)


    def test_codes_filter_and_name_sort(self):
        source = kessan([settlement(code, 2026, 120, 15) for code in (1, 2, 3)])
        names = MeigaralistPl(pl.DataFrame({"code": [1, 1, 2], "name": ["B社", "B社", "A社"]}))
        for codes, expected in [("all", [1, 2, 3]), ([2], [2]), ([3, 1, 1], [1, 3]), ([], []), ([999], [])]:
            with self.subTest(codes=codes):
                result = build_quarterly_valuation(source, quotes([]), prices([]), names,
                                                  valuation_date=VALUATION_DATE, codes=codes)
                self.assertEqual(result["code"].to_list(), expected)
                self.assertEqual(result.columns, list(OUTPUT_COLUMNS))
        result = build_quarterly_valuation(source, quotes([]), prices([]), names,
                                          valuation_date=VALUATION_DATE, sort_columns=["name"], sort_order="asc")
        self.assertEqual(result["code"].to_list(), [2, 1, 3])
        for codes in ("1", ["1"], None, [True]):
            with self.subTest(codes=codes), self.assertRaises(ValueError):
                build_quarterly_valuation(source, quotes([]), prices([]), names,
                                         valuation_date=VALUATION_DATE, codes=codes)

    def test_value_ranges_after_latest_selection(self):
        source = kessan([
            settlement(1, 2025, 100, 10), settlement(1, 2026, 150, 15),
            settlement(2, 2025, 100, 10), settlement(2, 2026, 50, 15),
            settlement(3, 2025, 100, 10), settlement(3, 2026, 100, 15),
            settlement(4, 2025, 100, 10), settlement(4, 2026, None, 15),
            settlement(5, 2026, 100, 15),
            settlement(6, 2026, 150, 15, month=3, announced=date(2026, 7, 1)),
            settlement(6, 2026, 200, 20),
        ])
        names = MeigaralistPl(pl.DataFrame(schema={"code": pl.Int64, "name": pl.String}))
        before = source.df.clone()
        cases = [
            ({}, [1, 2, 3, 4, 5, 6]),
            ({"sls_min": 100, "sls_max": 150}, [1, 3, 5]),
            ({"sls_max": 100}, [2, 3, 5]),
            ({"grsl_min": 0, "grsl_max": 50}, [1, 3]),
            ({"grsl_max": 0}, [2, 3]),
            ({"grsl_min": -50, "grsl_max": -50}, [2]),
            ({"sls_min": 100, "grsl_min": 0, "codes": [1, 2]}, [1]),
            ({"sls_min": 0}, [1, 2, 3, 5, 6]),
            ({"sls_min": 201}, []),
            ({"sls_min": 200, "sls_max": 100}, []),
        ]
        for options, expected in cases:
            with self.subTest(options=options):
                result = build_quarterly_valuation(
                    source, quotes([]), prices([]), names, valuation_date=VALUATION_DATE,
                    sort_columns=["code"], sort_order="asc", **options,
                )
                self.assertEqual(result["code"].to_list(), expected)
                self.assertEqual(result.columns, list(OUTPUT_COLUMNS))
        assert_frame_equal(source.df, before)

    def test_profit_ranges_use_selected_profit_before_rounding_and_row_slice(self):
        source = kessan([
            settlement(1, 2026, 100, 10),
            settlement(2, 2026, 200, 20),
            settlement(3, 2026, 300, None),
            settlement(4, 2026, 400, -5),
            settlement(5, 2026, 500, 0),
            settlement(6, 2026, 600, 10.4),
            settlement(7, 2026, 700, 10, month=3, announced=date(2026, 7, 1)),
            settlement(7, 2026, 700, 30),
        ])
        source.df = source.df.with_columns(
            (pl.col("ordinary_profit") * 2).alias("operating_income")
        )
        names = MeigaralistPl(pl.DataFrame(schema={"code": pl.Int64, "name": pl.String}))
        before = source.df.clone()
        cases = [
            ({}, [1, 2, 3, 4, 5, 6, 7]),
            ({"profit_min": 10, "profit_max": 20}, [1, 2, 6]),
            ({"profit_min": 20}, [2, 7]),
            ({"profit_max": 0}, [4, 5]),
            ({"profit_min": -5, "profit_max": 0}, [4, 5]),
            ({"profit_min": 10, "profit_max": 10}, [1]),
            ({"profit_min": 20, "profit_max": 10}, []),
            ({"profit_min": 10, "profit_max": 20, "sls_min": 200,
              "codes": [1, 2, 3, 4, 5, 6]}, [2, 6]),
            ({"profit_min": 10, "profit_max": 20, "start_row": 2, "end_row": 2}, [2]),
        ]
        for mode, factor in [("ordinary", 1), ("operating", 2)]:
            for options, expected in cases:
                options = dict(options)
                for key in ("profit_min", "profit_max"):
                    if key in options:
                        options[key] *= factor
                with self.subTest(profit=mode, options=options):
                    result = build_quarterly_valuation(
                        source, quotes([]), prices([]), names,
                        valuation_date=VALUATION_DATE, profit=mode,
                        sort_columns=["code"], sort_order="asc", **options,
                    )
                    self.assertEqual(result["code"].to_list(), expected)
                    self.assertEqual(result.columns, list(OUTPUT_COLUMNS))
        assert_frame_equal(source.df, before)

    def test_row_range_after_filter_and_sort(self):
        source = kessan([settlement(code, 2026, code * 100, 15) for code in [1, 4, 2, 3]])
        names = MeigaralistPl(pl.DataFrame(schema={"code": pl.Int64, "name": pl.String}))
        for start, end, expected in [(1, None, [4, 3, 2]), (2, 3, [3, 2]),
                                     (2, 2, [3]), (2, None, [3, 2]),
                                     (2, 99, [3, 2]), (9, 99, [])]:
            with self.subTest(start=start, end=end):
                result = build_quarterly_valuation(
                    source, quotes([]), prices([]), names, valuation_date=VALUATION_DATE,
                    codes=[1, 2, 3, 4], sls_min=200, sort_columns=["sls"],
                    start_row=start, end_row=end,
                )
                self.assertEqual(result["code"].to_list(), expected)
                self.assertEqual(result.columns, list(OUTPUT_COLUMNS))
        for start, end in [(0, None), (1, 0), (3, 2)]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                build_quarterly_valuation(source, quotes([]), prices([]), names,
                                         valuation_date=VALUATION_DATE, start_row=start, end_row=end)

    def test_profit_selection_and_selected_profit_null_rules(self):
        cases = [
            (10, 20, 10, 15, 0.5, 0.25),
            (-1, 20, 10, 15, None, 0.25),
            (10, 0, 10, 15, None, 0.25),
            (None, 20, 10, 15, None, 0.25),
            (10, 20, -1, 15, 0.5, None),
            (10, 20, 10, None, 0.5, None),
        ]
        for old_op, op, old_ord, ordinary, expected_op, expected_ord in cases:
            with self.subTest(old_op=old_op, op=op, old_ord=old_ord, ordinary=ordinary):
                source = kessan([settlement(1, 2025, 100, old_ord), settlement(1, 2026, 120, ordinary)])
                source.df = source.df.with_columns(pl.Series("operating_income", [old_op, op], dtype=pl.Float64))
                before = source.df.clone()
                for mode, expected in [("operating", expected_op), ("ordinary", expected_ord)]:
                    result = source.get_quarterly_valuation(VALUATION_DATE, profit=mode)
                    self.assertEqual(result["dgrp"][0], expected)
                    selected = op if mode == "operating" else ordinary
                    self.assertEqual(result["prft"][0], selected)
                    if selected is None:
                        self.assertIsNone(result["pr"][0])
                    else:
                        self.assertAlmostEqual(result["pr"][0], selected / 120 * 100)
                self.assertEqual(source.get_quarterly_valuation(VALUATION_DATE)["dgrp"][0], expected_op)
                assert_frame_equal(source.df, before)
        with self.assertRaises(ValueError):
            source.get_quarterly_valuation(VALUATION_DATE, profit="invalid")
        # 従来の経常利益モードでは営業利益列は不要。
        source = kessan([settlement(1, 2025, 100, 10), settlement(1, 2026, 120, 15)])
        source.df = source.df.drop("operating_income")
        self.assertEqual(source.get_quarterly_valuation(VALUATION_DATE, profit="ordinary")["dgrp"][0], 0.25)

    def test_profit_margin_without_previous_data_and_invalid_values(self):
        cases = [
            (120, 15, 12.5), (120, -15, -12.5), (120, 0, 0),
            (0, 15, None), (-120, 15, None), (None, 15, None),
            (120, None, None), (float("nan"), 15, None),
            (float("inf"), 15, None), (120, float("nan"), None),
            (120, float("inf"), None), (120, -float("inf"), None),
            (1e-300, 1e300, None),
        ]
        source = kessan([settlement(i, 2026, sales, value)
                         for i, (sales, value, _) in enumerate(cases)])
        before = source.df.clone()
        for mode in ("operating", "ordinary"):
            result = source.get_quarterly_valuation(VALUATION_DATE, profit=mode)
            self.assertEqual(result.columns, ["code", "setd", "annd", "sls", "prft", "pr", "grsl", "dgrp"])
            self.assertEqual(result["pr"].to_list(), [expected for _, _, expected in cases])
            self.assertEqual(result["dgrp"].null_count(), len(cases))
        assert_frame_equal(source.df, before)

    def test_sort_by_profit_margin(self):
        source = kessan([settlement(1, 2026, 100, -10), settlement(2, 2026, 200, 30),
                         settlement(3, 2026, 0, 10)])
        names = MeigaralistPl(pl.DataFrame(schema={"code": pl.Int64, "name": pl.String}))
        result = build_quarterly_valuation(
            source, quotes([]), prices([]), names, valuation_date=VALUATION_DATE,
            sort_columns=["pr"], profit="ordinary",
        )
        self.assertEqual(result["code"].to_list(), [2, 1, 3])
        self.assertEqual(result.columns, list(OUTPUT_COLUMNS))


class QuarterlyPerformanceTests(TestCase):
    def test_next_period_first_announcement_trading_days_and_sort(self):
        rows = []
        for code in (1, 2):
            rows.extend([
                settlement(code, 2026, 100, 10, announced=date(2026, 7, 17)),
                # 同じ決算期の将来の訂正は次回扱いしない。
                settlement(code, 2026, 110, 11, announced=date(2026, 9, 25)),
                settlement(code, 2026, 120, 12, month=9, announced=date(2026, 10, 16)),
                settlement(code, 2026, 125, 12, month=9, announced=date(2026, 10, 23)),
                settlement(code, 2026, 120, 12, month=9, announced=date(2026, 10, 1), kind="予"),
            ])
        source = kessan(list(reversed(rows)))
        price_df = pl.DataFrame({
            "code": [1] * 5 + [2] * 3,
            "date": [date(2026, 7, 17), date(2026, 7, 21), date(2026, 10, 16),
                     date(2026, 10, 19), date(2026, 10, 26),
                     date(2026, 7, 22), date(2026, 10, 20), date(2026, 10, 26)],
            "open": [999., 100., 999., 120., 999., 200., 180., 999.],
            "close": [500.] * 8,
        }).reverse()
        quarterly = source.get_quarterly_valuation(VALUATION_DATE)
        before = [df.clone() for df in (source.df, quarterly, price_df)]
        result = source.get_quarterly_performance(quarterly, price_df)
        self.assertEqual(result["code"].to_list(), [1, 2])
        self.assertAlmostEqual(result["perf"][0], 20)
        self.assertAlmostEqual(result["perf"][1], -10)
        for actual, expected in zip((source.df, quarterly, price_df), before):
            assert_frame_equal(actual, expected)
        names = MeigaralistPl(pl.DataFrame({"code": [1, 2], "name": ["一", "二"]}))
        result = build_quarterly_valuation(source, quotes([]), PricelistPl(price_df), names,
                                          valuation_date=VALUATION_DATE,
                                          sort_columns=["perf"], sort_order="asc")
        self.assertEqual(result["code"].to_list(), [2, 1])
        self.assertEqual(result.columns[-2:], ["perf", "bm"])

    def test_missing_and_invalid_prices_do_not_skip_first_trading_day(self):
        cases = [(100., 100., 0.), (100., 80., -20.), (None, 120., None),
                 (0., 120., None), (-1., 120., None), (float("nan"), 120., None),
                 (float("inf"), 120., None), (100., None, None), (100., 0., None),
                 (100., float("inf"), None), (100., float("nan"), None)]
        for start, end, expected in cases:
            with self.subTest(start=start, end=end):
                source = kessan([
                    settlement(1, 2026, 100, 10),
                    settlement(1, 2026, 120, 12, month=9),
                ])
                current = source.get_quarterly_valuation(VALUATION_DATE)
                prices_df = pl.DataFrame({
                    "code": [1, 1, 1],
                    "date": [date(2026, 7, 21), date(2026, 10, 21), date(2026, 10, 22)],
                    "open": pl.Series([start, end, 999.], dtype=pl.Float64),
                })
                result = source.get_quarterly_performance(current, prices_df)
                if expected is None:
                    self.assertIsNone(result["perf"][0])
                else:
                    self.assertAlmostEqual(result["perf"][0], expected)
        for price_input in (prices_df.clear(), prices_df.head(1)):
            self.assertIsNone(source.get_quarterly_performance(current, price_input)["perf"][0])
        no_next = kessan([settlement(1, 2026, 100, 10)])
        self.assertIsNone(no_next.get_quarterly_performance(current, prices_df)["perf"][0])
        self.assertEqual(source.get_quarterly_performance(current.clear(), prices_df).schema,
                         {"code": pl.Int64, "perf": pl.Float64})

    def test_announcement_on_valuation_date_is_only_used_as_next(self):
        source = kessan([
            settlement(1, 2026, 100, 10, month=3, announced=date(2026, 7, 1)),
            settlement(1, 2026, 120, 12, announced=VALUATION_DATE),
        ])
        current = source.get_quarterly_valuation(VALUATION_DATE)
        self.assertEqual(current["annd"][0], date(2026, 7, 1))
        price_df = pl.DataFrame({"code": [1, 1],
                                 "date": [date(2026, 7, 2), date(2026, 9, 24)],
                                 "open": [100., 130.]})
        result = source.get_quarterly_performance(current, pl.concat([price_df, price_df]))
        self.assertAlmostEqual(result["perf"][0], 30)
        conflicting = pl.concat([price_df, price_df.with_columns(pl.lit(99.).alias("open"))])
        with self.assertRaisesRegex(ValueError, "異なる始値"):
            source.get_quarterly_performance(current, conflicting)


class QuarterlyBenchmarkTests(TestCase):
    def setUp(self):
        self.source = kessan([
            settlement(code, 2026, 100, 10, month=month)
            for code in (1, 2) for month in (6, 9)
        ])
        self.current = self.source.get_quarterly_valuation(VALUATION_DATE)
        self.prices = pl.DataFrame({
            "code": [1, 1, 2, 2],
            "date": [date(2026, 7, 21), date(2026, 10, 21),
                     date(2026, 7, 22), date(2026, 10, 22)],
            "open": [100., 120., 100., 90.], "close": [100.] * 4,
        })
        self.index = pl.DataFrame({
            "date": [date(2026, 7, 20), date(2026, 7, 21), date(2026, 7, 22),
                     date(2026, 10, 21), date(2026, 10, 22)],
            "open": [999., 200., 400., 220., 360.],
        }).reverse()

    def test_exact_stock_dates_sort_and_no_mutation(self):
        from wequant.data_processing import IndexPricelistPl

        frames = [self.source.df, self.current, self.prices, self.index]
        before = [df.clone() for df in frames]
        result = self.source.get_quarterly_performance(
            self.current, self.prices.reverse(), pl.concat([self.index, self.index]),
        )
        self.assertAlmostEqual(result["bm"][0], 10)
        self.assertAlmostEqual(result["bm"][1], -10)
        names = MeigaralistPl(pl.DataFrame({"code": [1, 2], "name": ["A", "B"]}))
        result = build_quarterly_valuation(
            self.source, quotes([]), PricelistPl(self.prices), names, IndexPricelistPl(self.index),
            valuation_date=VALUATION_DATE, sort_columns=["bm"], sort_order="asc",
        )
        self.assertEqual(result["code"].to_list(), [2, 1])
        self.assertEqual(result.columns[-2:], ["perf", "bm"])
        for actual, expected in zip(frames, before):
            assert_frame_equal(actual, expected)

    def test_missing_invalid_and_empty(self):
        for value in (None, 0., -1., float("nan"), float("inf")):
            for boundary in (date(2026, 7, 21), date(2026, 10, 21)):
                with self.subTest(value=value, boundary=boundary):
                    index = self.index.with_columns(
                        pl.when(pl.col("date") == boundary).then(pl.lit(value, dtype=pl.Float64))
                        .otherwise(pl.col("open")).alias("open")
                    )
                    result = self.source.get_quarterly_performance(self.current, self.prices, index)
                    self.assertIsNone(result["bm"][0])
                    self.assertAlmostEqual(result["perf"][0], 20)
        # 隣接日のデータがあっても、当日が欠けていれば補完しない。
        missing = self.index.filter(pl.col("date") != date(2026, 7, 21))
        self.assertIsNone(self.source.get_quarterly_performance(
            self.current, self.prices, missing,
        )["bm"][0])
        for prices_df, index in ((self.prices.clear(), self.index),
                                 (self.prices, self.index.clear())):
            result = self.source.get_quarterly_performance(self.current, prices_df, index)
            self.assertEqual(result["bm"].to_list(), [None, None])
        no_next = kessan([settlement(1, 2026, 100, 10)])
        self.assertIsNone(no_next.get_quarterly_performance(
            self.current.head(1), self.prices, self.index,
        )["bm"][0])
        empty = self.source.get_quarterly_performance(self.current.clear(), self.prices, self.index)
        self.assertEqual(empty.schema, {"code": pl.Int64, "perf": pl.Float64, "bm": pl.Float64})
        self.assertEqual(empty.height, 0)

    def test_invalid_stock_open_does_not_remove_dates_and_conflicts_fail(self):
        result = self.source.get_quarterly_performance(
            self.current, self.prices.with_columns(pl.lit(None, dtype=pl.Float64).alias("open")),
            self.index,
        )
        self.assertEqual(result["perf"].to_list(), [None, None])
        self.assertAlmostEqual(result["bm"][0], 10)
        conflicting = pl.concat([
            self.index, self.index.head(1).with_columns(pl.lit(9999.).alias("open")),
        ])
        with self.assertRaisesRegex(ValueError, "異なる始値"):
            self.source.get_quarterly_performance(self.current, self.prices, conflicting)
