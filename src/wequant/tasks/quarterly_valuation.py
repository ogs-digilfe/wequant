"""四半期銘柄評価の読み込みと加工を担当するTask。"""

from datetime import date
import sys
from typing import Literal

import polars as pl

from wequant.data_processing import IndexPricelistPl, FinancequotePl, KessanPl, MeigaralistPl, PortfolioManager, PricelistPl


OUTPUT_COLUMNS = ("code", "name", "setd", "annd", "sls", "prft", "pr", "grsl", "dgrp", "ngrpr", "PER", "divr", "perf", "bm")


def load_quarterly_valuation_inputs() -> tuple[KessanPl, FinancequotePl, PricelistPl, MeigaralistPl, IndexPricelistPl]:
    """管理対象のローカルParquetを読み込む。通信・保存は行わない。"""
    return (
        KessanPl.from_file(),
        FinancequotePl.from_file(),
        PricelistPl.from_file("reviced_pricelist.parquet"),
        MeigaralistPl.from_file(),
        IndexPricelistPl.from_file(),
    )


def load_portfolio_codes(valuation_date: date) -> list[int]:
    """評価日以前の最新ポートフォリオの個別株コードを重複なく返す。

    評価日当日を含み、全体の最新日を先に選ぶ。ETF・過去の保有銘柄は含めない。
    整数に変換できないticker_code（nullを含む）は除外し、重複除去後の
    除外件数を標準エラーへ通知する。該当なしは空リスト。状態やファイルは変更しない。
    """
    holdings = PortfolioManager.from_file().get_individual_stocks(
        specific_date=valuation_date, columns_selected=["ticker_code"], unique=True,
    )
    codes = holdings.select(
        pl.col("ticker_code").cast(pl.String).str.strip_chars().cast(pl.Int64, strict=False)
    ).to_series()
    excluded = codes.null_count()
    if excluded:
        print(f"ポートフォリオ: 整数に変換できない銘柄コードを{excluded}件除外しました。", file=sys.stderr)
    return codes.drop_nulls().unique().sort().to_list()


def build_quarterly_valuation(
    settlements: KessanPl,
    quotes: FinancequotePl,
    prices: PricelistPl,
    meigaras: MeigaralistPl,
    index_prices: IndexPricelistPl | None = None,
    *,
    valuation_date: date,
    sort_columns: list[str] | None = None,
    sort_order: Literal["asc", "desc"] = "desc",
    codes: Literal["all"] | list[int] = "all",
    sls_min: float | None = None,
    sls_max: float | None = None,
    grsl_min: float | None = None,
    grsl_max: float | None = None,
    start_row: int = 1,
    end_row: int | None = None,
    profit: Literal["operating", "ordinary"] = "operating",
    profit_min: float | None = None,
    profit_max: float | None = None,
    dgrp_min: float | None = None,
    dgrp_max: float | None = None,
    perf_period: Literal["quarter", "announcement"] = "quarter",
    pr_min: float | None = None,
    pr_max: float | None = None,
) -> pl.DataFrame:
    """入力を変更せず14列の評価一覧を返す。nullは末尾、同値はcode昇順。

    perf_periodはperf・bmの期間（quarter: 発表間、announcement: 次回発表当日→翌取引日）。
    index_pricesはnh225の指数。省略時はbmをnullにする（既存呼び出し互換）。
    start_row/end_rowは抽出・ソート後の1始まりの範囲（両端を含む）。
    end_row=Noneは末尾まで。範囲が件数を超えた分は切り詰める。
    codes="all"は全銘柄、整数リストは指定銘柄のみ（空リストは0件）。
    sls/grsl/prft/dgrp/prの上下限は境界を含むAND条件。Noneは制限なし。
    条件を指定した列のnullは除外。sls・prftは元単位、grslは%で丸め前に比較する。
    dgrp_min/dgrp_maxは%単位。内部倍率へ変換し、丸め前のdgrpと比較する。
    pr_min/pr_maxはprofitで選んだ利益率（pr）の%単位で、丸め前に比較する。
    profit_min/profit_maxはprofitで選んだ利益（prft）に適用する。
    銘柄名がない行も保持する。同一codeに異なる名前があれば結合エラー。"""
    if start_row < 1 or (end_row is not None and (end_row < 1 or start_row > end_row)):
        raise ValueError("行番号は1以上、start_rowはend_row以下にしてください。")
    if codes != "all" and (
        not isinstance(codes, list) or any(type(code) is not int for code in codes)
    ):
        raise ValueError("codesはallまたは整数のリストを指定してください。")
    columns = ["dgrp"] if sort_columns is None else list(sort_columns)
    if not columns or any(column not in OUTPUT_COLUMNS for column in columns):
        raise ValueError("sort_columnsには出力列名を1つ以上指定してください。")
    if len(columns) != len(set(columns)):
        raise ValueError("sort_columnsに同じ列を重複指定できません。")
    if sort_order not in ("asc", "desc"):
        raise ValueError("sort_orderはascまたはdescを指定してください。")

    quarterly = settlements.get_quarterly_valuation(valuation_date, profit=profit)
    performance = settlements.get_quarterly_performance(
        quarterly, prices.df,
        index_prices.df if index_prices is not None else None,
        perf_period=perf_period,
    )
    if index_prices is None:
        performance = performance.with_columns(pl.lit(None, dtype=pl.Float64).alias("bm"))
    quarterly = quarterly.join(performance, on="code", how="left", validate="1:1")
    valuations = quotes.get_price_adjusted_valuations(prices.df, valuation_date)
    result = quarterly.join(valuations, on="code", how="left", validate="1:1")
    if codes != "all":
        result = result.filter(pl.col("code").is_in(codes))
    # 最新決算の選択後、丸め前の値で指定された全条件を適用する。
    for column, lower, upper in (
        ("sls", sls_min, sls_max),
        ("grsl", grsl_min, grsl_max),
        ("prft", profit_min, profit_max),
        ("pr", pr_min, pr_max),
        ("dgrp", None if dgrp_min is None else dgrp_min / 100,
         None if dgrp_max is None else dgrp_max / 100),
    ):
        if lower is not None:
            result = result.filter(pl.col(column) >= lower)
        if upper is not None:
            result = result.filter(pl.col(column) <= upper)
    names = meigaras.df.select("code", "name").unique()
    result = result.join(names, on="code", how="left", validate="m:1")
    descending = [sort_order == "desc"] * len(columns)
    if "code" not in columns:
        columns.append("code")
        descending.append(False)
    result = result.select(OUTPUT_COLUMNS).sort(
        columns, descending=descending, nulls_last=True
    )
    return result.slice(start_row - 1, None if end_row is None else end_row - start_row + 1)
