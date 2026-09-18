"""人工データと読み込みモックによる生成API・互換性の検証。"""

from datetime import date
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import polars as pl
from polars.testing import assert_frame_equal

from wequant.data_processing import (
    DATA_DIR,
    CreditbalancePl,
    FinancequotePl,
    IndexPricelistPl,
    KessanPl,
    MeigaralistPl,
    PortfolioManager,
    PricelistPl,
    ShikihoOnlinePl,
)


def construction_cases():
    prices = pl.DataFrame({
        "mcode": [1001], "p_key": [date(2024, 1, 5)],
        "p_open": [100.0], "p_high": [110.0],
        "p_low": [95.0], "p_close": [108.0], "volume": [10],
    })
    expected_prices = pl.DataFrame({
        "code": [1001], "date": [date(2024, 1, 5)],
        "open": [100.0], "high": [110.0],
        "low": [95.0], "close": [108.0], "volume": [10],
    })
    settlements = pl.DataFrame({
        "mcode": [1001], "settlement_date": [date(2024, 3, 31)],
        "settlement_type": ["本"], "announcement_date": [date(2024, 5, 15)],
    })
    return [
        (CreditbalancePl, "creditbalance.parquet",
         pl.DataFrame({"code": ["1001"], "purchase_margin": [200]}),
         pl.DataFrame({"code": [1001], "purchase_margin": [200]})),
        (FinancequotePl, "finance_quote.parquet",
         pl.DataFrame({"mcode": [1001], "p_key": [date(2024, 1, 5)]}),
         pl.DataFrame({"code": [1001], "date": [date(2024, 1, 5)]})),
        (IndexPricelistPl, "nh225.parquet", prices.drop("mcode"),
         expected_prices.drop("code")),
        (PricelistPl, "reviced_pricelist.parquet", prices, expected_prices),
        (KessanPl, "kessan.parquet", settlements,
         settlements.rename({"mcode": "code"})),
        (MeigaralistPl, "meigaralist.parquet",
         pl.DataFrame({"mcode": [1001], "mname": ["テスト商事"]}),
         pl.DataFrame({"code": [1001], "name": ["テスト商事"]})),
        (PortfolioManager, "base_portfolio.parquet",
         pl.DataFrame({"ticker_code": ["1001"], "holdings": [100]}),
         pl.DataFrame({"ticker_code": ["1001"], "holdings": [100]})),
        (ShikihoOnlinePl, "shikiho_online.parquet",
         pl.DataFrame({"mcode": [1001], "mname": ["テスト商事"], "issue": [date(2024, 1, 1)]}),
         pl.DataFrame({"code": [1001], "name": ["テスト商事"], "issue": [date(2024, 1, 1)]})),
    ]


class ConstructionTests(TestCase):
    def test_dataframe_inputs_normalize_without_reading_or_changing_source(self):
        with patch("wequant.data_processing.load_data_file") as load, \
             patch("wequant.data_processing.read_data") as read:
            for cls, _, source, expected in construction_cases():
                for empty in (False, True):
                    for keyword in (False, True):
                        with self.subTest(cls=cls.__name__, empty=empty, keyword=keyword):
                            frame = source.clear() if empty else source
                            before = frame.clone()
                            actual = cls(df=frame) if keyword else cls(frame)
                            assert_frame_equal(actual.df, expected.clear() if empty else expected)
                            assert_frame_equal(frame, before)
            load.assert_not_called()
            read.assert_not_called()

    def test_from_file_defaults_read_once_and_apply_normalization(self):
        for cls, filename, source, expected in construction_cases():
            with self.subTest(cls=cls.__name__), \
                 patch("wequant.data_processing.load_data_file", return_value=source) as load, \
                 patch("wequant.data_processing.read_data") as read:
                actual = cls.from_file()
                self.assertIsInstance(actual, cls)
                assert_frame_equal(actual.df, expected)
                load.assert_called_once_with(filename)
                read.assert_not_called()

    def test_from_file_accepts_filename_and_path(self):
        for cls, filename, source, expected in construction_cases():
            for fp in ("tmp_snapshot.parquet", Path("fixtures") / filename):
                with self.subTest(cls=cls.__name__, fp=fp), \
                     patch("wequant.data_processing.load_data_file", return_value=source) as load:
                    actual = cls.from_file(fp=fp)
                    assert_frame_equal(actual.df, expected)
                    load.assert_called_once_with(fp)

    def test_from_file_uses_data_loading_path_resolution(self):
        source = pl.DataFrame({"code": ["1001"]})
        with patch("wequant.data_loading.Path.exists", return_value=True), \
             patch("wequant.data_loading.read_data", return_value=source) as read:
            actual = CreditbalancePl.from_file("creditbalance.parquet")
        read.assert_called_once_with(DATA_DIR / "creditbalance.parquet")
        assert_frame_equal(actual.df, pl.DataFrame({"code": [1001]}))

    def test_from_file_rejects_unmanaged_names_before_reading(self):
        with patch("wequant.data_loading.read_data") as read:
            for cls, _, _, _ in construction_cases():
                with self.subTest(cls=cls.__name__), self.assertRaises(ValueError):
                    cls.from_file("unknown.parquet")
            read.assert_not_called()

    def test_from_file_propagates_loading_failure(self):
        for cls, _, _, _ in construction_cases():
            with self.subTest(cls=cls.__name__), \
                 patch("wequant.data_processing.load_data_file", side_effect=OSError("read failed")):
                with self.assertRaisesRegex(OSError, "read failed"):
                    cls.from_file()

    def test_from_file_constructs_the_requested_subclass(self):
        class CustomPrices(PricelistPl):
            pass

        with patch("wequant.data_processing.load_data_file", return_value=pl.DataFrame({"code": [1001]})):
            self.assertIsInstance(CustomPrices.from_file(), CustomPrices)

    def test_kessan_preserves_date_repair_and_old_row_exclusion(self):
        source = pl.DataFrame({
            "mcode": [1001, 2002, 3003, 4004],
            "settlement_date": [date(2023, 3, 31), date(2024, 3, 31), date(2016, 3, 31), date(2017, 3, 31)],
            "settlement_type": ["本", "本", "本", "予"],
            "announcement_date": [date(2024, 5, 15), date(2024, 5, 16), date(2016, 5, 15), date(2017, 2, 15)],
            "sales": [100, 200, 300, 400],
            "operating_income": [10, 20, 30, 40],
            "ordinary_profit": [10, 20, 30, 40],
            "final_profit": [10, 20, 30, 40],
            "reviced_eps": [1.0, 2.0, 3.0, 4.0],
            "dividend": [1.0, 2.0, 3.0, 4.0],
            "quater": [4, 4, 4, 4],
        })
        expected = source.head(2).rename({"mcode": "code"}).with_columns(
            pl.lit(date(2024, 3, 31)).alias("settlement_date")
        )
        before = source.clone()
        with patch("wequant.data_processing.load_data_file", return_value=source):
            assert_frame_equal(KessanPl.from_file().df, expected)
        assert_frame_equal(KessanPl(source).df, expected)
        assert_frame_equal(source, before)


class LegacyConstructionTests(TestCase):
    def test_default_constructors_keep_their_loading_routes(self):
        for cls, filename, source, expected in construction_cases():
            with self.subTest(cls=cls.__name__), \
                 patch("wequant.data_processing.load_data_file", return_value=source) as load, \
                 patch("wequant.data_processing.read_data", return_value=source) as read:
                assert_frame_equal(cls().df, expected)
                if cls is KessanPl:
                    read.assert_called_once_with(DATA_DIR / filename)
                    load.assert_not_called()
                else:
                    load.assert_called_once_with(filename)
                    read.assert_not_called()

    def test_price_classes_keep_positional_and_fp_file_inputs(self):
        for cls, filename, source, expected in construction_cases():
            if cls not in (PricelistPl, IndexPricelistPl):
                continue
            for fp in (filename, Path("fixtures") / filename):
                for keyword in (False, True):
                    with self.subTest(cls=cls.__name__, fp=fp, keyword=keyword), \
                         patch("wequant.data_processing.load_data_file", return_value=source) as load:
                        actual = cls(fp=fp) if keyword else cls(fp)
                        assert_frame_equal(actual.df, expected)
                        load.assert_called_once_with(fp)

    def test_price_classes_keep_fp_dataframe_without_reading(self):
        with patch("wequant.data_processing.load_data_file") as load:
            for cls, _, source, expected in construction_cases():
                if cls in (PricelistPl, IndexPricelistPl):
                    with self.subTest(cls=cls.__name__):
                        assert_frame_equal(cls(fp=source).df, expected)
            load.assert_not_called()

    def test_price_classes_reject_ambiguous_df_and_fp(self):
        with patch("wequant.data_processing.load_data_file") as load:
            for cls in (PricelistPl, IndexPricelistPl):
                with self.subTest(cls=cls.__name__), self.assertRaises(TypeError):
                    cls(df=pl.DataFrame(), fp="nh225.parquet")
            load.assert_not_called()

    def test_creditbalance_keeps_arbitrary_path_and_dataframe_precedence(self):
        source = pl.DataFrame({"code": ["1001"]})
        expected = pl.DataFrame({"code": [1001]})
        fp = Path("fixtures") / "custom-credit.parquet"
        with patch("wequant.data_processing.read_data", return_value=source) as read, \
             patch("wequant.data_processing.load_data_file") as load:
            assert_frame_equal(CreditbalancePl(fp=fp).df, expected)
            read.assert_called_once_with(fp)
            read.reset_mock()
            assert_frame_equal(CreditbalancePl(source, fp).df, expected)
            read.assert_not_called()
            load.assert_not_called()


class ColumnNameNormalizationTests(TestCase):
    def test_normalized_inputs_can_be_wrapped_again_without_reading(self):
        with patch("wequant.data_processing.load_data_file") as load, \
             patch("wequant.data_processing.read_data") as read:
            for cls, _, _, expected in construction_cases():
                for empty in (False, True):
                    with self.subTest(cls=cls.__name__, empty=empty):
                        frame = expected.clear() if empty else expected
                        before = frame.clone()
                        actual = cls(df=frame)
                        assert_frame_equal(actual.df, frame)
                        assert_frame_equal(cls(actual.df).df, frame)
                        assert_frame_equal(frame, before)
            load.assert_not_called()
            read.assert_not_called()

    def test_mixed_old_and_new_names_normalize_independently(self):
        for cls, _, source, expected in construction_cases():
            for old, new in zip(source.columns, expected.columns, strict=True):
                if old == new:
                    continue
                for empty in (False, True):
                    with self.subTest(cls=cls.__name__, renamed=new, empty=empty):
                        frame = source.rename({old: new})
                        if empty:
                            frame = frame.clear()
                        before = frame.clone()
                        assert_frame_equal(cls(frame).df, expected.clear() if empty else expected)
                        assert_frame_equal(frame, before)

    def test_rejects_alias_collisions_even_with_identical_values(self):
        for cls, _, source, expected in construction_cases():
            for old, new in zip(source.columns, expected.columns, strict=True):
                if old == new:
                    continue
                for empty in (False, True):
                    with self.subTest(cls=cls.__name__, column=new, empty=empty):
                        frame = source.with_columns(pl.col(old).alias(new))
                        if empty:
                            frame = frame.clear()
                        before = frame.clone()
                        with self.assertRaises(ValueError) as raised:
                            cls(frame)
                        self.assertIn(old, str(raised.exception))
                        self.assertIn(new, str(raised.exception))
                        assert_frame_equal(frame, before)

    def test_from_file_accepts_normalized_input_and_rejects_collisions(self):
        for cls, _, source, expected in construction_cases():
            with self.subTest(cls=cls.__name__), \
                 patch("wequant.data_processing.load_data_file", return_value=expected):
                assert_frame_equal(cls.from_file().df, expected)
            if source.columns == expected.columns:
                continue
            old, new = source.columns[0], expected.columns[0]
            frame = source.with_columns(pl.col(old).alias(new))
            with self.subTest(cls=cls.__name__, collision=True), \
                 patch("wequant.data_processing.load_data_file", return_value=frame), \
                 self.assertRaises(ValueError):
                cls.from_file()

    def test_preserves_extra_columns_and_their_order(self):
        for cls, _, source, expected in construction_cases():
            with self.subTest(cls=cls.__name__):
                frame = source.with_columns(pl.lit("keep").alias("custom_label"))
                result = expected.with_columns(pl.lit("keep").alias("custom_label"))
                columns = ["custom_label"] + source.columns
                result_columns = ["custom_label"] + expected.columns
                # Kessan's existing repair path has separate column-position assumptions;
                # these rows do not require repair, so only renaming is exercised here.
                assert_frame_equal(cls(frame.select(columns)).df, result.select(result_columns))

    def test_absent_aliases_do_not_introduce_required_column_checks(self):
        cases = [
            (FinancequotePl, pl.DataFrame({"note": ["keep"]}), pl.DataFrame({"note": ["keep"]})),
            (PricelistPl, pl.DataFrame({"p_close": [100.0]}), pl.DataFrame({"close": [100.0]})),
            (IndexPricelistPl, pl.DataFrame({"p_close": [100.0]}), pl.DataFrame({"close": [100.0]})),
            (MeigaralistPl, pl.DataFrame({"mname": ["テスト商事"]}), pl.DataFrame({"name": ["テスト商事"]})),
            (ShikihoOnlinePl, pl.DataFrame({"mname": ["テスト商事"]}), pl.DataFrame({"name": ["テスト商事"]})),
        ]
        for cls, source, expected in cases:
            with self.subTest(cls=cls.__name__):
                assert_frame_equal(cls(source).df, expected)

    def test_renaming_does_not_change_types_or_apply_other_classes_aliases(self):
        source = pl.DataFrame({
            "mcode": pl.Series(["01001"], dtype=pl.String),
            "p_key": pl.Series([date(2024, 1, 5)], dtype=pl.Date),
            "p_close": pl.Series([100.0], dtype=pl.Float32),
        })
        expected = pl.DataFrame({
            "code": pl.Series(["01001"], dtype=pl.String),
            "date": pl.Series([date(2024, 1, 5)], dtype=pl.Date),
            "close": pl.Series([100.0], dtype=pl.Float32),
        })
        assert_frame_equal(PricelistPl(source).df, expected)
        # FinancequotePl does not define the price alias p_close -> close.
        quote_expected = expected.rename({"close": "p_close"})
        assert_frame_equal(FinancequotePl(source).df, quote_expected)

    def test_classes_without_aliases_keep_their_column_names(self):
        credit = pl.DataFrame({"code": [1001], "mcode": [2002]})
        portfolio = pl.DataFrame({"ticker_code": ["1001"], "code": [2002]})
        assert_frame_equal(CreditbalancePl(credit).df, credit)
        assert_frame_equal(PortfolioManager(portfolio).df, portfolio)
