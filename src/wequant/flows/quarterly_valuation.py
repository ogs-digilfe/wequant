"""四半期銘柄評価の読み込みと加工を組み立てるFlow。"""

from datetime import date
from typing import Literal

import polars as pl

from wequant.tasks.quarterly_valuation import (
    build_quarterly_valuation,
    load_quarterly_valuation_inputs,
    load_portfolio_codes,
)


def quarterly_valuation_flow(
    valuation_date: date | None = None,
    sort_columns: list[str] | None = None,
    sort_order: Literal["asc", "desc"] = "desc",
    codes: Literal["all"] | list[int] = "all",
    sls_min: float | None = None,
    sls_max: float | None = None,
    grsl_min: float | None = None,
    grsl_max: float | None = None,
    portfolio: bool = False,
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
    """評価日省略時は呼び出し時の今日を使い、評価一覧を返す。

    portfolio=Trueは最新ポートフォリオの個別株とcodesの共通部分に絞る。
    対象なしは空の評価一覧。Falseではポートフォリオを読み込まない。"""
    valuation_date = date.today() if valuation_date is None else valuation_date
    if portfolio:
        if codes != "all" and (
            not isinstance(codes, list) or any(type(code) is not int for code in codes)
        ):
            raise ValueError("codesはallまたは整数のリストを指定してください。")
        holdings = load_portfolio_codes(valuation_date)
        codes = holdings if codes == "all" else sorted(set(codes).intersection(holdings))
    inputs = load_quarterly_valuation_inputs()
    return build_quarterly_valuation(
        *inputs,
        valuation_date=valuation_date,
        sort_columns=sort_columns,
        sort_order=sort_order,
        codes=codes,
        sls_min=sls_min,
        sls_max=sls_max,
        grsl_min=grsl_min,
        grsl_max=grsl_max,
        start_row=start_row,
        end_row=end_row,
        profit=profit,
        pr_min=pr_min,
        pr_max=pr_max,
        profit_min=profit_min,
        profit_max=profit_max,
        dgrp_min=dgrp_min,
        dgrp_max=dgrp_max,
        perf_period=perf_period,
    )
