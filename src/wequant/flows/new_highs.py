"""新高値抽出を組み立てるFlow。"""
from datetime import date

import polars as pl

from wequant.tasks.new_highs import build_new_highs, load_new_highs_inputs


def new_highs_flow(
    *, start_date: date | None = None, end_date: date | None = None,
    min_prev_high: int | None = None,
) -> pl.DataFrame:
    """指定条件に合う新高値一覧を返す。保存は行わない。"""
    prices, meigaras = load_new_highs_inputs()
    return build_new_highs(
        prices, meigaras, start_date=start_date, end_date=end_date,
        min_prev_high=min_prev_high,
    )
