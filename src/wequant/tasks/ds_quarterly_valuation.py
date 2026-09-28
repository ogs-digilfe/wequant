"""全期間の四半期評価データセットの加工・保存Task。"""
from datetime import datetime
from pathlib import Path
from typing import Literal

import polars as pl

from wequant.data_loading import DATA_DIR
from wequant.data_processing import KessanPl, FinancequotePl, PricelistPl, MeigaralistPl, IndexPricelistPl
from wequant.tasks.quarterly_valuation import OUTPUT_COLUMNS


def build_ds_quarterly_valuation(
    settlements: KessanPl, quotes: FinancequotePl, prices: PricelistPl,
    meigaras: MeigaralistPl, index_prices: IndexPricelistPl, *,
    profit: Literal["operating", "ordinary"] = "operating",
    perf_period: Literal["quarter", "announcement"] = "quarter",
) -> pl.DataFrame:
    """入力を変更せず15列を返す。perf/bm欠損だけを除外しcode/setd順にする。"""
    quarterly = settlements.get_quarterly_valuation_dataset(prices.df, profit=profit)
    performance = settlements.get_quarterly_performance_dataset(
        quarterly, prices.df, index_prices.df, perf_period=perf_period,
    )
    valuations = quotes.get_price_adjusted_valuations_for_dates(prices.df, quarterly)
    return quarterly.join(performance, on=["code", "setd"], how="left", validate="1:1").join(
        valuations, on=["code", "setd"], how="left", validate="1:1",
    ).join(
        meigaras.df.select("code", "name").unique(), on="code", how="left", validate="m:1",
    ).drop_nulls(["perf", "bm"]).select(OUTPUT_COLUMNS).sort(["code", "setd"])


def save_ds_quarterly_valuation(
    df: pl.DataFrame, *, profit: Literal["operating", "ordinary"] = "operating",
    perf_period: Literal["quarter", "announcement"] = "quarter",
) -> Path:
    """data/datasetsへ丸めず保存。同名ファイルは上書きしない。空結果も保存する。"""
    if profit not in ("operating", "ordinary") or perf_period not in ("quarter", "announcement"):
        raise ValueError("profitまたはperf_periodが不正です。")
    directory = DATA_DIR / "datasets"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"ds-quarterly-valuation-{profit}-{perf_period}-{datetime.now():%Y%m%d_%H%M%S}.parquet"
    with path.open("xb") as stream:
        try:
            df.write_parquet(stream)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
    return path
