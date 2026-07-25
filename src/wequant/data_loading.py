"""Parquetファイルのパス解決と読み込みを担当する。"""

from pathlib import Path
from typing import Union

import polars as pl

from wequant.data_files import DOWNLOADABLE_FILES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def read_data(file_path: Union[str, Path]) -> pl.DataFrame:
    """指定されたParquetファイルを読み込む。"""
    return pl.read_parquet(str(file_path))


def resolve_data_path(file_path: Union[str, Path]) -> Path:
    """wequantが管理するデータファイルのパスを検証して返す。"""
    path = Path(file_path)
    if path.parent == Path("."):
        path = DATA_DIR / path

    filename = path.name
    if filename not in DOWNLOADABLE_FILES and "tmp_" not in filename:
        raise ValueError(
            f"ファイル名{filename}は、wequantで管理していないファイルです。"
            "ファイル名を確認してください。"
        )

    if not path.exists():
        raise ValueError(
            f"""
                ファイル{filename}が、データ保存フォルダ{path.parent}にダウンロードされていません。
                wq dl-pqを実行するなどしてデータをダウンロードしてください。
                """
        )

    return path


def load_data_file(file_path: Union[str, Path]) -> pl.DataFrame:
    """wequantが管理するParquetファイルを検証して読み込む。"""
    return read_data(resolve_data_path(file_path))
