"""売買シミュレーションの読み込み・出力組み立て・保存。"""
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from wequant.data_loading import DATA_DIR
from wequant.data_processing import IndexPricelistPl, MeigaralistPl, PricelistPl

OUTPUT_COLUMNS = ["code", "name", "requested_entry_date", "entry_date", "entry_price",
                  "exit_date", "exit_price", "exit_reason", "status", "updown", "benchmark"]


def load_simulation_inputs() -> tuple[PricelistPl, MeigaralistPl, IndexPricelistPl]:
    """調整済み株価、銘柄一覧、日経平均を読み込む。"""
    return (PricelistPl.from_file("reviced_pricelist.parquet"),
            MeigaralistPl.from_file(), IndexPricelistPl.from_file())


def build_threshold_simulation(
    prices: PricelistPl, names: MeigaralistPl, index: IndexPricelistPl,
    entries: list[tuple[date, str]], *, take_profit: float, stop_loss: float,
    entry_mode: str = "next-open", exit_mode: str = "next-open",
) -> pl.DataFrame:
    """入力順を保ち、銘柄名と規定の出力列を付ける。入力は変更しない。"""
    result = prices.get_threshold_simulation(entries, index_df=index.df, take_profit=take_profit,
                                             stop_loss=stop_loss, entry_mode=entry_mode, exit_mode=exit_mode)
    names_df = names.df.select(pl.col("code").cast(pl.String), "name")
    return result.join(names_df, on="code", how="left", validate="m:1",
                       maintain_order="left").select(OUTPUT_COLUMNS)


def save_threshold_simulation(result: pl.DataFrame) -> Path:
    """日本時間の実行日時でCSV保存する。同名ファイルは上書きしない。"""
    directory = DATA_DIR / "output" / "sim"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d%H%M%S")
    path = directory / f"threshold-{stamp}.csv"
    with path.open("x", encoding="utf-8", newline="") as stream:
        result.write_csv(stream)
    return path
