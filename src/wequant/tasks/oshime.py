"""押し目候補の読み込みと銘柄名の結合。"""
from datetime import date

import polars as pl

from wequant.data_processing import MeigaralistPl, PricelistPl


def load_oshime_inputs() -> tuple[PricelistPl, MeigaralistPl]:
    """ローカルの調整済み株価と銘柄一覧を読み込む。"""
    return PricelistPl.from_file("reviced_pricelist.parquet"), MeigaralistPl.from_file()


def build_oshime_candidates(
    prices: PricelistPl, meigaras: MeigaralistPl, *, valuation_date: date, min_decline_rate: float = 5.0, max_decline_rate: float = 12.0,
    ma25_up_days: int = 50,
    min_pullback_days: int = 3, max_pullback_days: int = 16,
    min_low_age_days: int = 1, max_low_age_days: int = 6,
    max_rebound_rate: float = 10.0,
) -> pl.DataFrame:
    """候補に銘柄名を左結合しcode昇順で返す。入力は変更しない。"""
    candidates = prices.get_oshime_candidates(valuation_date=valuation_date, min_decline_rate=min_decline_rate, max_decline_rate=max_decline_rate, ma25_up_days=ma25_up_days, min_pullback_days=min_pullback_days, max_pullback_days=max_pullback_days, min_low_age_days=min_low_age_days, max_low_age_days=max_low_age_days, max_rebound_rate=max_rebound_rate)
    names = meigaras.df.select("code", "name").with_columns(
        pl.col("code").cast(candidates.schema["code"])
    )
    return candidates.join(names, on="code", how="left", validate="m:1").sort("code")
