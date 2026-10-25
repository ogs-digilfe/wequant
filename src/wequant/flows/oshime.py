"""押し目候補の抽出を組み立てるFlow。"""
from datetime import date

import polars as pl

from wequant.tasks.oshime import build_oshime_candidates, load_oshime_inputs


def get_oshime_flow(*, valuation_date: date, min_decline_rate: float = 5.0, max_decline_rate: float = 12.0, ma25_up_days: int = 50, min_pullback_days: int = 3, max_pullback_days: int = 16, min_low_age_days: int = 1, max_low_age_days: int = 6, max_rebound_rate: float = 10.0) -> pl.DataFrame:
    """評価日以前の最新取引日の候補一覧を返す。"""
    prices, meigaras = load_oshime_inputs()
    return build_oshime_candidates(prices, meigaras, valuation_date=valuation_date, min_decline_rate=min_decline_rate, max_decline_rate=max_decline_rate, ma25_up_days=ma25_up_days, min_pullback_days=min_pullback_days, max_pullback_days=max_pullback_days, min_low_age_days=min_low_age_days, max_low_age_days=max_low_age_days, max_rebound_rate=max_rebound_rate)
