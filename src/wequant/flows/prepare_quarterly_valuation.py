"""指定Parquetの読み込み、学習前加工、結果と条件の保存を組み立てる。"""
from datetime import datetime
from hashlib import file_digest
from pathlib import Path

import polars as pl

from wequant.data_loading import PROJECT_ROOT
from wequant.tasks.prepare_quarterly_valuation import (
    IDENTIFIER_COLUMNS,
    TARGET_EXPRESSIONS,
    PreparationOptions,
    prepare_quarterly_valuation,
    save_prepared_quarterly_valuation,
)


def prepare_quarterly_valuation_flow(
    input_path: Path, options: PreparationOptions, output_path: Path | None = None,
) -> tuple[pl.DataFrame, Path]:
    """相対パスはリポジトリルート基準。元ファイルを変更せず、加工結果と保存先を返す。"""
    options.validate()
    source = (PROJECT_ROOT / input_path).resolve()
    destination = (PROJECT_ROOT / output_path).resolve() if output_path is not None else None
    if destination is not None and destination.suffix != ".parquet":
        raise ValueError("出力ファイルの拡張子は.parquetにしてください。")
    with source.open("rb") as stream:
        source_hash = file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        original = pl.read_parquet(stream)
    result = prepare_quarterly_valuation(original, options)
    metadata = {
        "schema_version": 1,
        "dataset_kind": "quarterly-valuation",
        "created_at": datetime.now().astimezone().isoformat(),
        "input_path": str(source),
        "input_sha256": source_hash,
        "features": list(options.features),
        "identifier_columns": list(IDENTIFIER_COLUMNS),
        "target": {"column": "target", "kind": options.target, "expression": TARGET_EXPRESSIONS[options.target]},
        "filters": {
            "setd_from": options.setd_from.isoformat() if options.setd_from else None,
            "setd_to": options.setd_to.isoformat() if options.setd_to else None,
            "qtr": list(options.qtr),
            "numeric_ranges": {c: {"min": lo, "max": hi} for c, (lo, hi) in options.numeric_ranges.items()},
        },
        "rows_before": original.height,
        "rows_after": result.height,
    }
    path = save_prepared_quarterly_valuation(result, metadata, destination)
    return result, path
