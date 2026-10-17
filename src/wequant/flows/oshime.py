"""押し目候補の抽出を組み立てるFlow。"""
from datetime import date

import polars as pl

from wequant.tasks.oshime import build_oshime_candidates, load_oshime_inputs


def get_oshime_flow(*, valuation_date: date, decline_rate: float = 5.0) -> pl.DataFrame:
    """評価日以前の最新取引日の候補一覧を返す。"""
    prices, meigaras = load_oshime_inputs()
    return build_oshime_candidates(prices, meigaras, valuation_date=valuation_date, decline_rate=decline_rate)
