"""全期間の四半期評価データセットを作成・保存するFlow。"""
from pathlib import Path
from typing import Literal

import polars as pl

from wequant.tasks.quarterly_valuation import load_quarterly_valuation_inputs
from wequant.tasks.ds_quarterly_valuation import build_ds_quarterly_valuation, save_ds_quarterly_valuation


def ds_quarterly_valuation_flow(
    profit: Literal["operating", "ordinary"] = "operating",
    perf_period: Literal["quarter", "announcement"] = "quarter",
) -> tuple[pl.DataFrame, Path]:
    """全期間の評価一覧と保存先を返す。通信なし、ローカルへの保存あり。"""
    if profit not in ("operating", "ordinary") or perf_period not in ("quarter", "announcement"):
        raise ValueError("profitまたはperf_periodが不正です。")
    result = build_ds_quarterly_valuation(
        *load_quarterly_valuation_inputs(), profit=profit, perf_period=perf_period,
    )
    path = save_ds_quarterly_valuation(result, profit=profit, perf_period=perf_period)
    return result, path
