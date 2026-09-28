"""四半期評価データセットの学習前加工と、加工結果・条件の保存。"""
from dataclasses import dataclass, field
from datetime import date, datetime
import json
from math import isfinite
from pathlib import Path
from typing import Literal

import polars as pl

from wequant.data_loading import DATA_DIR


FEATURE_COLUMNS = ("sls", "prft", "pr", "grsl", "dgrp", "ngrpr", "PER", "divr")
IDENTIFIER_COLUMNS = ("code", "setd", "annd", "qtr")
TARGET_EXPRESSIONS = {"perf": "perf", "excess-return": "perf - bm"}


@dataclass(frozen=True)
class PreparationOptions:
    """加工条件。数値範囲は元データの単位（dgrpは倍率）のまま指定する。"""

    features: tuple[str, ...]
    target: Literal["perf", "excess-return"]
    setd_from: date | None = None
    setd_to: date | None = None
    qtr: tuple[int, ...] = ()
    numeric_ranges: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)

    def validate(self) -> None:
        if self.target not in TARGET_EXPRESSIONS:
            raise ValueError("targetはperfまたはexcess-returnを指定してください。")
        if not self.features or any(c not in FEATURE_COLUMNS for c in self.features):
            raise ValueError(f"特徴量は次から1列以上選択してください: {', '.join(FEATURE_COLUMNS)}")
        if len(set(self.features)) != len(self.features):
            raise ValueError("特徴量の重複指定はできません。")
        if any(type(q) is not int or q not in (1, 2, 3, 4) for q in self.qtr):
            raise ValueError("qtrは1〜4を指定してください。")
        for value in (self.setd_from, self.setd_to):
            if value is not None and type(value) is not date:
                raise ValueError("setdの境界はdate型で指定してください。")
        if self.setd_from and self.setd_to and self.setd_from > self.setd_to:
            raise ValueError("setd-fromはsetd-to以下にしてください。")
        for column, (lower, upper) in self.numeric_ranges.items():
            if column not in FEATURE_COLUMNS:
                raise ValueError(f"数値フィルターに指定できない列です: {column}")
            if any(v is not None and not isfinite(v) for v in (lower, upper)):
                raise ValueError(f"{column}の境界は有限の数値を指定してください。")
            if lower is not None and upper is not None and lower > upper:
                raise ValueError(f"{column}の下限は上限以下にしてください。")


def prepare_quarterly_valuation(df: pl.DataFrame, options: PreparationOptions) -> pl.DataFrame:
    """入力を変更せず、管理4列・指定順の特徴量・targetを入力の行順で返す。

    setd/anndはDate型、特徴量・範囲指定列・目的変数の元列は数値型を前提とする。
    境界を含めてANDで絞り、qtrの複数値はORで扱う。範囲指定列のnull/非有限値は
    除外し、その他の特徴量は型・値・欠損を保持する。目的変数にnull/非有限値が
    あればエラー。perfは%、excess-returnはパーセントポイント。空結果も同じ列を返す。
    ファイルの読み書き、追加特徴量の計算、期間分割は行わない。
    """
    options.validate()
    ranges = {c: bounds for c, bounds in options.numeric_ranges.items() if bounds != (None, None)}
    target_columns = ("perf", "bm") if options.target == "excess-return" else ("perf",)
    numeric = set(options.features) | set(ranges) | set(target_columns)
    required = set(IDENTIFIER_COLUMNS) | numeric
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"必要な列がありません: {', '.join(sorted(missing))}")
    for column in numeric:
        if not df.schema[column].is_numeric() and df.schema[column] != pl.Null:
            raise ValueError(f"{column}は数値型である必要があります。")
    if any(df.schema[c] != pl.Date for c in ("setd", "annd")):
        raise ValueError("setd/anndはDate型である必要があります。")
    if not df.schema["qtr"].is_integer():
        raise ValueError("qtrは整数型である必要があります。")

    result = df
    if options.setd_from is not None:
        result = result.filter(pl.col("setd") >= options.setd_from)
    if options.setd_to is not None:
        result = result.filter(pl.col("setd") <= options.setd_to)
    if options.qtr:
        result = result.filter(pl.col("qtr").is_in(options.qtr))
    for column, (lower, upper) in ranges.items():
        value = pl.col(column).cast(pl.Float64)
        condition = value.is_finite()
        if lower is not None:
            condition &= value >= lower
        if upper is not None:
            condition &= value <= upper
        result = result.filter(condition)

    target = pl.col("perf").cast(pl.Float64)
    if options.target == "excess-return":
        target = target - pl.col("bm").cast(pl.Float64)
    result = result.select(*IDENTIFIER_COLUMNS, *options.features, target.alias("target"))
    if result.filter(~pl.col("target").is_finite().fill_null(False)).height:
        raise ValueError("目的変数に欠損または非有限値があります。原本データを確認してください。")
    return result


def save_prepared_quarterly_valuation(
    df: pl.DataFrame, metadata: dict, output_path: Path | None = None,
) -> Path:
    """Parquetと同名JSONを排他的に新規保存する。失敗時は今回作成したファイルだけ削除。"""
    path = output_path if output_path is not None else (
        DATA_DIR / "datasets" / "prepared" / f"quarterly-valuation-{datetime.now():%Y%m%d_%H%M%S}.parquet"
    )
    if path.suffix != ".parquet":
        raise ValueError("出力ファイルの拡張子は.parquetにしてください。")
    metadata_path = path.with_suffix(".json")
    payload = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        with path.open("xb") as parquet_stream:
            created.append(path)
            with metadata_path.open("x", encoding="utf-8") as metadata_stream:
                created.append(metadata_path)
                df.write_parquet(parquet_stream)
                metadata_stream.write(payload)
    except BaseException:
        for created_path in reversed(created):
            created_path.unlink(missing_ok=True)
        raise
    return path
