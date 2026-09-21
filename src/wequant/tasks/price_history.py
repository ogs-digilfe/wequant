"""株価履歴の読み込みと抽出を担当するTask。"""

from datetime import date
from typing import Literal

import polars as pl

from wequant.data_processing import PricelistPl

OUTPUT_COLUMNS = ("code", "date", "open", "high", "low", "close", "volume")


def load_price_history(source: Literal["reviced", "raw"] = "reviced") -> PricelistPl:
    """選択されたローカルファイルを読み込む。"""
    if source not in ("reviced", "raw"):
        raise ValueError("sourceはrevicedまたはrawを指定してください。")
    return PricelistPl.from_file(f"{source}_pricelist.parquet")


def select_price_history(
    prices: PricelistPl, *, code: str, start_date: date, end_date: date,
) -> pl.DataFrame:
    """両端を含む期間を日付昇順で返す。入力は変更しない。"""
    if start_date > end_date:
        raise ValueError("開始日は終了日以下にしてください。")
    return (
        prices.df.with_columns(pl.col("code").cast(pl.String))
        .filter(
            (pl.col("code") == code)
            & pl.col("date").is_between(start_date, end_date, closed="both")
        )
        .select(OUTPUT_COLUMNS)
        .sort("date")
    )
