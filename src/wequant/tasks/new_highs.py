"""新高値抽出の読み込みと銘柄名の結合。"""
from datetime import date

import polars as pl

from wequant.data_processing import MeigaralistPl, PricelistPl


def load_new_highs_inputs() -> tuple[PricelistPl, MeigaralistPl]:
    """調整済み株価と銘柄一覧を読み込む。"""
    return PricelistPl.from_file("reviced_pricelist.parquet"), MeigaralistPl.from_file()


def build_new_highs(
    prices: PricelistPl, meigaras: MeigaralistPl, *,
    start_date: date | None = None, end_date: date | None = None,
    min_prev_high: int | None = None,
) -> pl.DataFrame:
    """銘柄名を左結合しdate昇順・prev-high降順で返す。入力は変更しない。"""
    result = prices.get_new_highs(
        start_date=start_date, end_date=end_date, min_prev_high=min_prev_high,
    )
    names = meigaras.df.select("code", "name").with_columns(
        pl.col("code").cast(result.schema["code"])
    )
    return result.join(names, on="code", how="left", validate="m:1").select(
        "date", "code", "name", "prev-high",
    ).sort(["date", "prev-high"], descending=[False, True], maintain_order=True)
