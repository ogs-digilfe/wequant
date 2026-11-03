"""閾値売買シミュレーションのFlow。"""
from datetime import date
from pathlib import Path

import polars as pl

from wequant.tasks.simulation import (
    build_threshold_simulation, load_simulation_inputs, save_threshold_simulation,
)


def threshold_simulation_flow(
    entries: list[tuple[date, str]], *, take_profit: float, stop_loss: float,
    entry_mode: str = "next-open", exit_mode: str = "next-open",
    output_mode: str = "standard",
) -> tuple[pl.DataFrame, Path | None]:
    """読み込み・計算・必要時の保存を行い、結果と保存先を返す。"""
    if output_mode not in ("standard", "file"):
        raise ValueError("出力モードが不正です。")
    prices, names, index = load_simulation_inputs()
    result = build_threshold_simulation(
        prices, names, index, entries, take_profit=take_profit, stop_loss=stop_loss,
        entry_mode=entry_mode, exit_mode=exit_mode,
    )
    path = save_threshold_simulation(result) if output_mode == "file" else None
    return result, path
