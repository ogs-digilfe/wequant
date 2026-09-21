# wequqnt/cli.py
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import re
from typing import Annotated, Literal
from unicodedata import combining, east_asian_width

import polars as pl
import typer

from wequant.flows.download_data import download_data_flow
from wequant.flows.price_history import price_history_flow
from wequant.flows.quarterly_valuation import quarterly_valuation_flow
from wequant.tasks.quarterly_valuation import OUTPUT_COLUMNS

app = typer.Typer(help="wequant CLI")

@app.command()
def get_app_name():
    """appの名前を表示する"""
    print("wequant")

@app.command()
def describe():
    """appの説明を表示する"""
    print("analysis tool for stock market")


@app.command()
def dl_pq():
    """最新データをdeliverサーバからdownloadする"""
    print("Downloading latest data from deliver server...")
    download_data_flow()
    print("Download complete.")


class ProfitType(str, Enum):
    operating = "operating"
    ordinary = "ordinary"


class SortOrder(str, Enum):
    asc = "asc"
    desc = "desc"


@app.command()
def quarterly_valuation(
    valuation_date: Annotated[
        datetime | None,
        typer.Option(formats=["%Y-%m-%d"], help="評価日（YYYY-MM-DD）。省略時は今日。"),
    ] = None,
    sort_columns: Annotated[
        list[str] | None,
        typer.Option(help="ソートする出力列。複数列はオプションを繰り返して指定。"),
    ] = None,
    sort_order: Annotated[SortOrder, typer.Option(help="昇順asc / 降順desc。")] = SortOrder.desc,
    codes: Annotated[
        list[str] | None,
        typer.Option(help="allまたは銘柄コード。複数銘柄はオプションを繰り返して指定。"),
    ] = None,
    sls_min: Annotated[float | None, typer.Option(help="売上高の下限（境界を含む）。")] = None,
    sls_max: Annotated[float | None, typer.Option(help="売上高の上限（境界を含む）。")] = None,
    grsl_min: Annotated[float | None, typer.Option(help="売上成長率の下限（%）（境界を含む）。")] = None,
    grsl_max: Annotated[float | None, typer.Option(help="売上成長率の上限（%）（境界を含む）。")] = None,
    portfolio: Annotated[bool, typer.Option("--portfolio", help="評価日以前の最新ポートフォリオの個別株に絞る。")] = False,
    start_row: Annotated[int, typer.Option(min=1, help="抽出・ソート後の開始行（1始まり）。")] = 1,
    end_row: Annotated[int | None, typer.Option(min=1, help="終了行（含む）。省略時は末尾まで。")] = None,
    profit: Annotated[ProfitType, typer.Option(help="prft・pr・dgrpの利益種別：operating=営業利益、ordinary=経常利益。")] = ProfitType.operating,
    profit_min: Annotated[float | None, typer.Option(help="--profitで選んだ利益の下限（出力と同じ単位、境界を含む）。")] = None,
    profit_max: Annotated[float | None, typer.Option(help="--profitで選んだ利益の上限（出力と同じ単位、境界を含む）。")] = None,
):
    """四半期ベースの銘柄評価一覧を表示する。"""
    if end_row is not None and start_row > end_row:
        raise typer.BadParameter("開始行は終了行以下にしてください。", param_hint="--start-row")
    selected_codes: Literal["all"] | list[int] = "all"
    if codes is not None and codes != ["all"]:
        if "all" in codes:
            raise typer.BadParameter("allと個別コードは混在できません。", param_hint="--codes")
        try:
            selected_codes = [int(code) for code in codes]
        except ValueError:
            raise typer.BadParameter("allまたは整数の銘柄コードを指定してください。", param_hint="--codes")
    columns = ["dgrp"] if sort_columns is None else sort_columns
    if any(column not in OUTPUT_COLUMNS for column in columns):
        raise typer.BadParameter(
            "出力列から指定してください: " + ", ".join(OUTPUT_COLUMNS),
            param_hint="--sort-columns",
        )
    if len(columns) != len(set(columns)):
        raise typer.BadParameter("同じ列を重複指定できません。", param_hint="--sort-columns")
    result = quarterly_valuation_flow(
        valuation_date=date.today() if valuation_date is None else valuation_date.date(),
        sort_columns=columns,
        sort_order=sort_order.value,
        codes=selected_codes,
        sls_min=sls_min,
        sls_max=sls_max,
        grsl_min=grsl_min,
        grsl_max=grsl_max,
        portfolio=portfolio,
        start_row=start_row,
        end_row=end_row,
        profit=profit.value,
        profit_min=profit_min,
        profit_max=profit_max,
    )

    # 端末幅・Polarsの表示設定に依存せず、すべての行と列を表示する。
    def format_value(column: str, value: object) -> str:
        if value is None:
            return "null"
        if column in ("sls", "prft"):
            rounded = Decimal(str(value)).to_integral_value(rounding=ROUND_HALF_UP)
            return f"{rounded:,f}"
        if column == "dgrp":
            return f"{value * 100:.2f}%"
        if column in ("grsl", "pr", "perf", "bm"):
            return f"{value:.2f}%"
        if column in ("PER", "divr"):
            return f"{value:.2f}"
        return str(value)

    rows = [
        [format_value(column, value) for column, value in zip(result.columns, row)]
        for row in result.iter_rows()
    ]
    def display_width(value: str) -> int:
        return sum(0 if combining(c) else 2 if east_asian_width(c) in ("W", "F") else 1 for c in value)

    def pad(value: str, width: int, *, left: bool = False) -> str:
        spaces = " " * (width - display_width(value))
        return value + spaces if left else spaces + value

    widths = [
        max(display_width(column), max((display_width(row[i]) for row in rows), default=0))
        for i, column in enumerate(result.columns)
    ]
    typer.echo("  ".join(pad(column, width, left=True) for column, width in zip(result.columns, widths)))
    for row in rows:
        typer.echo("  ".join(pad(value, width) for value, width in zip(row, widths)))


class PriceSource(str, Enum):
    reviced = "reviced"
    raw = "raw"


@app.command()
def price_history(
    code: Annotated[str, typer.Option(help="銘柄コード（半角英数字）。")],
    start_date: Annotated[datetime, typer.Option(formats=["%Y-%m-%d"], help="開始日（含む）。")],
    end_date: Annotated[datetime, typer.Option(formats=["%Y-%m-%d"], help="終了日（含む）。")],
    source: Annotated[PriceSource, typer.Option(help="reviced: 調整済み / raw: 未調整。")] = PriceSource.reviced,
):
    """指定銘柄・期間の四本値と出来高を日付昇順で表示する。"""
    if re.fullmatch(r"[A-Za-z0-9]+", code) is None:
        raise typer.BadParameter("半角英数字の銘柄コードを指定してください。", param_hint="--code")
    if start_date > end_date:
        raise typer.BadParameter("開始日は終了日以下にしてください。", param_hint="--start-date")
    try:
        result = price_history_flow(
            code=code, start_date=start_date.date(), end_date=end_date.date(), source=source.value,
        )
    except (ValueError, OSError, pl.exceptions.PolarsError) as exc:
        typer.echo(f"株価履歴を取得できません: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    rows = [
        ["null" if value is None else str(value) for value in row]
        for row in result.iter_rows()
    ]
    widths = [
        max(len(column), max((len(row[i]) for row in rows), default=0))
        for i, column in enumerate(result.columns)
    ]
    typer.echo("  ".join(column.ljust(width) for column, width in zip(result.columns, widths)))
    for row in rows:
        typer.echo("  ".join(value.rjust(width) for value, width in zip(row, widths)))
    if result.is_empty():
        typer.echo("該当データなし", err=True)
