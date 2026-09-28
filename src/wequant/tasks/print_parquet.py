"""任意のローカルParquetの読み込みと行選択。"""

import polars as pl
from wequant import data_loading


def validate_row_limits(*, head: int | None = None, tail: int | None = None) -> None:
    """ライブラリからの呼び出しでも行数指定を検証する。"""
    if head is not None and tail is not None:
        raise ValueError("headとtailは同時に指定できません。")
    for value in (head, tail):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError("行数は1以上の整数を指定してください。")


def load_parquet(file: str) -> pl.DataFrame:
    """data直下の拡張子なしファイル名から全体を読み込む。"""
    if (not file or file in (".", "..") or "/" in file or chr(92) in file
            or chr(0) in file or file.lower().endswith(".parquet")):
        raise ValueError("fileはディレクトリや.parquet拡張子を含まないファイル名を指定してください。")
    return data_loading.read_data(data_loading.DATA_DIR / f"{file}.parquet")


def select_parquet_rows(
    df: pl.DataFrame, *, head: int | None = None, tail: int | None = None,
) -> pl.DataFrame:
    """行順・列・型を保持して選択する。入力は変更しない。"""
    validate_row_limits(head=head, tail=tail)
    if head is not None:
        return df.head(head)
    if tail is not None:
        return df.tail(tail)
    return df.clone()
