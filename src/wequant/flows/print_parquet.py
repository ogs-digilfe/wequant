"""Parquetの読み込みと行選択を組み立てるFlow。"""

import polars as pl
from wequant.tasks.print_parquet import load_parquet, select_parquet_rows, validate_row_limits


def print_parquet_flow(
    *, file: str, head: int | None = None, tail: int | None = None,
) -> pl.DataFrame:
    """表示対象を返す。表示・保存は行わない。"""
    validate_row_limits(head=head, tail=tail)
    return select_parquet_rows(load_parquet(file), head=head, tail=tail)
