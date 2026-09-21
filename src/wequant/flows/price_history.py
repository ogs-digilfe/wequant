"""株価履歴の読み込みと抽出を組み立てるFlow。"""

from datetime import date
from typing import Literal

import polars as pl

from wequant.tasks.price_history import load_price_history, select_price_history


def price_history_flow(
    *, code: str, start_date: date, end_date: date,
    source: Literal["reviced", "raw"] = "reviced",
) -> pl.DataFrame:
    """指定銘柄・期間の株価一覧を返す。"""
    if start_date > end_date:
        raise ValueError("開始日は終了日以下にしてください。")
    return select_price_history(
        load_price_history(source), code=code, start_date=start_date, end_date=end_date,
    )
