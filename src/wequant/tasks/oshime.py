"""押し目候補の読み込みと銘柄名の結合。"""
from datetime import date

import polars as pl

from wequant.data_processing import MeigaralistPl, PricelistPl


def load_oshime_inputs() -> tuple[PricelistPl, MeigaralistPl]:
    """ローカルの調整済み株価と銘柄一覧を読み込む。"""
    return PricelistPl.from_file("reviced_pricelist.parquet"), MeigaralistPl.from_file()


def build_oshime_candidates(
    prices: PricelistPl, meigaras: MeigaralistPl, *, valuation_date: date, decline_rate: float = 5.0,
) -> pl.DataFrame:
    """候補に銘柄名を左結合しcode昇順で返す。入力は変更しない。"""
    candidates = prices.get_oshime_candidates(valuation_date=valuation_date, decline_rate=decline_rate)
    names = meigaras.df.select("code", "name").with_columns(
        pl.col("code").cast(candidates.schema["code"])
    )
    return candidates.join(names, on="code", how="left", validate="m:1").sort("code")
