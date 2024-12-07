# パスのセット
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).parent
PJROOT_DIR = CURRENT_DIR.parent
WORKSPACE_DIR = PJROOT_DIR.parent
DATA_DIR = PJROOT_DIR / "data"

sys.path.append(str(PJROOT_DIR))

# オブジェクトのインポート
import os, calendar
import polars as pl
import pandas as pd
from typing import Union, Literal
from datetime import date
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from plotly.graph_objects import Figure

# global
DOWNLOADABLE_FILES = [
    "finance_quote.parquet",
    "kessan.parquet",
    "meigaralist.parquet",
    "nh225.parquet",
    "raw_pricelist.parquet",
    "reviced_pricelist.parquet"
]

DATEFORMAT = "%Y-%m-%d"
DATEFORMAT2 = "%Y年%m月%d日"

# utility functions
def read_data(fp: Union[str, Path]) -> pl.DataFrame:
    fp = str(fp)
    
    return pl.read_parquet(fp)

# valuation_dateで指定した日の最新通期決算と決算予想をpl.DataFrameで返す
def get_df_latest_yearly_performance(code: int, valuation_date: date=date.today()) -> pl.DataFrame:
    fp = DATA_DIR/"kessan.parquet"
    df = read_data(fp)
    KPL = KessanPl(df)
    KPL.with_columns_financtial_period()

    df1 = KPL.get_latest_yearly_settlements(reference_date=valuation_date, settlement_type="予")
    df1 = df1.with_columns([
    (pl.col("決算期") + pl.lit("(予)")).alias("決算期")
    ])

    df2 = KPL.get_latest_yearly_settlements(reference_date=valuation_date, settlement_type="本")
    df = pl.concat([df1, df2])
    selected_cols = [df.columns[-1]] + df.columns[3:10]
    df = df.filter(pl.col("code")==code)\
        .select(selected_cols)

    rename_map_dct = {
        "announcement_date": "決算発表日",
        "sales": "売上高",
        "operating_income": "営業利益",
        "ordinary_profit": "経常利益",
        "final_profit": "純利益",
        "reviced_eps": "EPS",
        "dividend": "1株配当"
    }
    df = df.rename(rename_map_dct)

    # 出力
    companyname = get_companyname(code)
    print(f'{companyname}({code})の通期決算(評価日：{valuation_date.strftime(DATEFORMAT2)})')

    return df

# codeで指定した銘柄のevaluation_dateで指定した時点での最新の四半期決算発表に基づく
# 売上高~純利益の決算進捗率を円グラフで表示するためのfigを返す
# valuation_dateで指定する日において、当年度の本決算が発表されていない日を指定した場合は前期末第4四半期の進捗率を表示するので、進捗率100%で表示される。
def get_fig_actual_performance_progress_rate_pycharts(code: int, evaluation_date: date, kessan_df: pl.DataFrame, meigaralit_df: pl.DataFrame) -> Figure:
    df = kessan_df
    df = df.filter(pl.col("code")==code)

    KPL = KessanPl(df)
    df = KPL.get_actual_quatery_settlements_progress_rate()
    df = df.filter(pl.col("announcement_date")<evaluation_date)

    df = df.select([
        "code",
        'yearly_settlement_date',
        "quater",
        "sales_pr(%)",
        "operating_income_pr(%)",
        "ordinary_profit_pr(%)",
        "final_profit_pr(%)"
    ])

    pandas_df = df.to_pandas()
    df = pandas_df
    rec_idx = df.shape[0] - 1

    fyear = df.loc[rec_idx, "yearly_settlement_date"]
    quater = df.loc[rec_idx, "quater"]

    # グラフ出力オプション
    pio.renderers.default = 'iframe'

    # 出力グラフのplot設定(1行4列 -> 横並びに4つ表示)
    specs = [
        [{"type": "pie"}, {"type": "pie"}, {"type": "pie"}, {"type": "pie"}]
    ]
    fig = make_subplots(rows=1, cols=4, specs=specs)

    # pychartオブジェクトのセット
    for i in range(4):
        # pychartデータのセット(pandas.DataFrameにセットする)
        labels = ["進捗率(%)", " "]
        pr = df.loc[rec_idx, df.columns[i+3]]
        values = [pr, 100-pr]

        chart_df_data = {
            "labels": labels,
            "values": values
        }
        chart_df = pd.DataFrame(chart_df_data)

        # pychartオブジェクトの設定
        data_set = go.Pie(
            labels = chart_df["labels"],
            values = chart_df["values"],
            hole = 0.5,
            sort = False,
            marker = dict(colors=["aqua", "lightgrey"]),
            textinfo='percent',  # 全体の表示設定
            texttemplate=['%{percent}', '']
        )
        fig.add_trace(data_set, row=1, col=i+1)

    # レイアウトの設定
    items = ["売上高進捗率(%)", "営業利益進捗率(%)", "経常利益進捗率(%)", "純利益進捗率(%)"]
    left_gap = 0.07
    right_gap = 0.93
    gap_correction = 0.01
    gap = (right_gap - left_gap) / 3
    annotations = []
    for i in range(4):
        x = left_gap + gap * i
        if i == 1:
            x = x + gap_correction
        elif i == 2:
            x = x - gap_correction
        annotations.append(
            dict(text=items[i], x=x, y=0.5, font_size=12, showarrow=False)
        )

    # 設定したレイアウトをpychartオブジェクトにセット
    fig.update_layout(
        showlegend=False, # 凡例出力をoff
        annotations=annotations
    )

    # 出力
    MPL = MeigaralistPl(meigaralit_df)
    name = MPL.get_name(code)
    print(f'{name}({code})の{fyear.year}年{fyear.month}月期第{quater}四半期決算進捗率(評価日：{evaluation_date})')

    return fig

# plotly return graph object functions
# codeで指定した銘柄のevaluation_dateで指定した時点での最新の年度決算予想に基づく
# 売上高~純利益の決算進捗率を円グラフで表示するためのfigを返す
def get_fig_expected_performance_progress_rate_pycharts(code: int, evaluation_date: date=date.today()) -> Figure:
    fp1 = DATA_DIR / "kessan.parquet"
    fp2 = DATA_DIR / "meigaralist.parquet"
    df1 = read_data(fp1)
    KPL = KessanPl(df1)
    df2 = read_data(fp2)
    MPL = MeigaralistPl(df2)

    df = KPL.get_expected_quatery_settlements_progress_rate(evaluation_date)
    df = df.filter(pl.col("code")==code)

    df = df.select([
        "code",
        'yearly_settlement_date',
        "quater",
        "sales_pr(%)",
        "operating_income_pr(%)",
        "ordinary_profit_pr(%)",
        "final_profit_pr(%)"
    ])

    
    pandas_df = df.to_pandas()
    pldf = df
    df = pandas_df
    rec_idx = df.shape[0] - 1

    fyear = df.loc[rec_idx, "yearly_settlement_date"]
    quater = df.loc[rec_idx, "quater"]

    # 予想データがなければコメントを表示
    pdf = pldf[-1:]
    pdf = pdf.drop_nulls()
    name = MPL.get_name(code)
    if pdf.shape[0] == 0:
        print(f'{evaluation_date}における{name}({code})の{fyear.year}年{fyear.month}月期の決算予想が公表されていないため、決算進捗率を表示できません。')

    # グラフ出力オプション
    pio.renderers.default = 'iframe'

    # 出力グラフのplot設定(1行4列 -> 横並びに4つ表示)
    specs = [
        [{"type": "pie"}, {"type": "pie"}, {"type": "pie"}, {"type": "pie"}]
    ]
    fig = make_subplots(rows=1, cols=4, specs=specs)

    # pychartオブジェクトのセット
    for i in range(4):
        # pychartデータのセット(pandas.DataFrameにセットする)
        labels = ["進捗率(%)", " "]
        pr = df.loc[rec_idx, df.columns[i+3]]
        values = [pr, 100-pr]

        chart_df_data = {
            "labels": labels,
            "values": values
        }
        chart_df = pd.DataFrame(chart_df_data)

        # pychartオブジェクトの設定
        data_set = go.Pie(
            labels = chart_df["labels"],
            values = chart_df["values"],
            hole = 0.5,
            sort = False,
            marker = dict(colors=["aqua", "lightgrey"]),
            textinfo='percent',  # 全体の表示設定
            texttemplate=['%{percent}', '']
        )
        fig.add_trace(data_set, row=1, col=i+1)
        
    # レイアウトの設定
    items = ["売上高進捗率(%)", "営業利益進捗率(%)", "経常利益進捗率(%)", "純利益進捗率(%)"]
    left_gap = 0.07
    right_gap = 0.93
    gap_correction = 0.01
    gap = (right_gap - left_gap) / 3
    annotations = []
    for i in range(4):
        x = left_gap + gap * i
        if i == 1:
            x = x + gap_correction
        elif i == 2:
            x = x - gap_correction
        annotations.append(
            dict(text=items[i], x=x, y=0.5, font_size=12, showarrow=False)
        )

    # 設定したレイアウトをpychartオブジェクトにセット
    fig.update_layout(
        showlegend=False, # 凡例出力をoff
        annotations=annotations
    )

    # 出力
    print(f'{name}({code})の{fyear.year}年{fyear.month}月期第{quater}四半期決算進捗率(評価日：{evaluation_date})')

    return fig

def get_companyname(code: int) -> str:
    fp = DATA_DIR / "meigaralist.parquet"
    df = read_data(fp)
    MPL = MeigaralistPl(df)

    return MPL.get_name(code)

# 指定したcodeの最新株価(終値)を取得する
# いつの最新かをvaludation_dateで指定できる(過去日)。
# valudation_dateを指定した場合、株式分割は考慮されないので、要注意。
def get_latest_stockprice(code: int, valudation_date: date=date.today()) -> float:
    fp = DATA_DIR / "raw_pricelist.parquet"
    df = read_data(fp)
    df = PricelistPl(df).df

    df = df.filter(pl.col("code")==code)\
        .filter(pl.col("date")<=valudation_date)
    df = df.filter(pl.col("date")==pl.col("date").max())

    close = df.row(0)[5]
    
    # 出力
    name = get_companyname(code)
    dealing_date = df.row(0)[1].strftime(DATEFORMAT2)
    print(f'{name}({code})の{dealing_date}終値')
    
    return close

# 指定したcodeの指定した日における株価と各種ファンダメンタルズデータをまとめて標準出力する
# pricelist_dfは、raw_pricelistかreviced_pricelistかケースに応じて使い分ける。
def print_finance_quote(
        pricelist_df: pl.DataFrame,
        finance_quote_df: pl.DataFrame,
        code: int, 
        valuation_date: date=date.today()
    ) -> None:
    
    # タイトル
    company_name = get_companyname(code)
    print(f'{company_name}({code})の銘柄情報\n')

    # 株価
    df = pricelist_df
    KPL = PricelistPl(df)
    tup = KPL.get_latest_dealingdate_and_price(code, valuation_date)
    stock_price = tup[1]
    print(f'終値: {stock_price}円({tup[0].strftime(DATEFORMAT2)})')

    # その他指標
    df = finance_quote_df
    df = df.filter(pl.col("code")==code)\
        .filter(pl.col("date")<=valuation_date)
    df = df.filter(pl.col("date")==pl.col("date").max())
    quoted_date = df.select(["date"]).row(0)[0]
    # 予想配当利回り
    expected_dividened_yield = df.select(["expected_dividend_yield"]).row(0)[0]
    print(f'予想配当利回り: {expected_dividened_yield}%({quoted_date.strftime(DATEFORMAT2)})')
    # 予想PER
    expected_PER = df.select(["expected_PER"]).row(0)[0]
    print(f'予想PER: {expected_PER}倍({quoted_date.strftime(DATEFORMAT2)})')
    # 実績PBR
    actual_PBR = df.select(["actual_PBR"]).row(0)[0]
    print(f'実績PBR: {actual_PBR}倍({quoted_date.strftime(DATEFORMAT2)})')
    # 自己資本比率
    actual_CAR = df.select(["actual_CAR"]).row(0)[0]
    print(f'自己資本比率: {actual_CAR}%({quoted_date.strftime(DATEFORMAT2)})')
    # 予想ROE
    actual_BPS = df.select(["actual_BPS"]).row(0)[0]
    expected_EPS = df.select(["expected_EPS"]).row(0)[0]
    expected_ROE = 100*(expected_EPS / actual_BPS)
    expected_ROE = round(expected_ROE, 2)
    print(f'予想ROE: {expected_ROE}%({quoted_date.strftime(DATEFORMAT2)})')
    # 予想ROA
    expected_ROA = expected_ROE * (actual_CAR/100)
    expected_ROA = round(expected_ROA, 2)
    print(f'予想ROA: {expected_ROA}%({quoted_date.strftime(DATEFORMAT2)})')

# mapped functions
# KessanPl
def revice_last_date(dataframe_row) -> date:
    r = dataframe_row
    d1 = r[-1]

    y = d1.year
    m = d1.month
    d = calendar.monthrange(y, m)[1]

    r = list(r)
    r[-1] = date(y, m, d)
    r = tuple(r)

    return r


def get_yearly_settlement_date(dataframe_row) -> date:
    r = dataframe_row

    settlement_date_idx = r[-2]
    quater_idx = r[-1]


    quater = r[quater_idx]
    if quater == 1:
        delta_m = 9
    elif quater == 2:
        delta_m = 6
    elif quater == 3:
        delta_m = 3
    elif quater == 4:
        delta_m = 0
    else:
        delta_m = 0

    d0 = r[settlement_date_idx]
    d1 = date(d0.year, d0.month, 1)
    d1 += relativedelta(months=delta_m)

    y = d1.year
    m = d1.month
    d = calendar.monthrange(y, m)[1]

    return r+(date(y, m, d),)



# private classes
# 日々の財務データの加工/分析
class FinancequotePl():
    def __init__(self, df: pl.DataFrame):
        rename_map_dct = {
            "mcode": "code",
            "p_key": "date",
        }
        df = df.rename(rename_map_dct)

        self.df = df
    
    # 指定したcodeの指定した日における各種ファンダメンタルズのレコードをpl.DataFrameで返す
    def get_finance_quote(self, code: int, valuation_date: date=date.today()) -> pl.DataFrame:
        df = self.df

        df = df.filter(pl.col("code")==code)\
            .filter(pl.col("date")<=valuation_date)
        df = df.filter(pl.col("date")==pl.col("date").max())

        return df
    
    # PricelistPlとtotal_shares_numを使って時価総額列(百万円)を追加する
    # pricelist_dfを引数で渡さない場合はdataファイルを読み込む
    def with_columns_market_cap(self, pricelist_df: Union[pl.DataFrame, None]=None) -> None:
        if not pricelist_df:
            pricelist_df = self._read_raw_pricelist()
        
        df = self.df
        original_cols = df.columns
        
        df = df.join(pricelist_df, on=["code", "date"], how="left")
        df = df.select(original_cols+["close"])
        df = df.with_columns([
            (pl.col("total_shares_num")*pl.col("close")/pl.lit(1000000)).alias("market_cap")
        ])
        df = df.select(original_cols+["market_cap"])
        df = df.with_columns([
            pl.col("market_cap").cast(pl.Int64).alias("market_cap")
        ])

        self.df = df
    
    # raw_pricelist.parquetを読み込み
    def _read_raw_pricelist(self) -> pl.DataFrame:
        fp = DATA_DIR/"raw_pricelist.parquet"
        df = read_data(fp)
        
        return PricelistPl(df).df

class PricelistPl():
    # fp = filenameの場合、dirはDATA_DIR
    # fp = filepathの場合、fpはfilepathとして処理
    # fp = pl.DataFrameの場合はそのままPricelistPl.dfにpl.DataFrameをセット
    def __init__(self, fp: Union[str, Path, pl.DataFrame]):
        if type(fp) == type(pl.DataFrame()):
            self.df = fp
        else:
            fp = str(fp)
            data_dir, filename = os.path.split(fp)
            # filenameのみ指定された場合は、DATA_DIR
            if data_dir == "":
                data_dir = str(DATA_DIR)
                fp = str(DATA_DIR/fp)
            
            # 管理対象外ファイルの場合、raise ValueError
            # ただし、tmp_で始まるfile名はok
            if (not filename in DOWNLOADABLE_FILES) and (not "tmp_" in filename):
                raise ValueError(f'ファイル名{filename}は、wequantで管理していないファイルです。ファイル名を確認してください。')
            
            # ファイルをダウンロードしていなかったらraise FileNotFoundError
            utility_fp = str(PJROOT_DIR/"download_data.py")
            if not os.path.exists(fp):
                message = f'''
                ファイル{filename}が、データ保存フォルダ{data_dir}にダウンロードされていません。
                {utility_fp}を実行するなどしてデータをダウンロードしてください。
                '''
                raise ValueError(message)
            
            self.df = pl.read_parquet(fp)

        # 列のrenameをしてない場合は、rename。
        if "mcode" in self.df.columns:
            rename_map_dct = {
                "mcode": "code",
                "p_key": "date",
                "p_open": "open",
                "p_high": "high",
                "p_low": "low",
                "p_close": "close"
            }
            self.df = self.df.rename(rename_map_dct)

    # 指定したコードの指定した日付における最新の終値の株価を、(日付, 株価)のタプルで返す
    def get_latest_dealingdate_and_price(self, code: int, valuation_date: date = date.today()) -> tuple:
        df = self.df

        df = df.filter(pl.col("code")==code)\
            .filter(pl.col("date")<=valuation_date)
        df = df.filter(pl.col("date")==pl.col("date").max())

        dealing_date = df.select(["date"]).row(0)[0]
        price = df.select(["close"]).row(0)[0]

        return dealing_date, price
    
    # items_dfにpl.DataFrame.columns = ["code", "start_date", "end_date"]のpl.DataFrameを与えると、
    # 各レコードのstart_dateからend_dateまでの株価騰落率の列を追加して返す
    # *_pointは、起点(start)と終点(end)において、日足ローソクのどの時点の株価を起点、または終点とするか選択する。
    def get_stockprice_change_rate(self, 
        items_df: pl.DataFrame,
        start_point: Literal["open", "high", "low", "close"] = "open",
        end_point: Literal["open", "high", "low", "close"] = "open"
    ) -> pl.DataFrame:
        
        idf1 = items_df.with_columns([
            pl.col("start_date").alias("date")
        ]).select(["code", "date"])
        df1 = idf1.join(self.df, on=idf1.columns, how="left")
        
        return df1
        

    # colで指定した列のterm日の移動平均列を、25日移動平均であれば、ma25の
    # ような列名(maの後ろに移動平均の日数)で追加する。
    # termで指定した日数での移動平均が計算できない初期のレコードは、dropされてなくなる
    # 全データで実施すると、かなりメモリを消費するので、200日移動平均などを取得する場合は、
    # PricelistPl(filename).dfをfilterしてから実施しないとメモリが足りなくなるかもしれない。
    # メモリが不足して実行プロセスがダウンした場合は、例外も出力されない。
    def with_columns_moving_average(self, term, col="close"):
        df = self.df
        
        # term数shiftする
        df = df.with_columns([pl.col(col).alias('s0')])
        for i in range(1, term-1):
            df = df.with_columns([pl.col(col).shift(i).alias(f's{str(i)}')])
        last_col_shift_num = term - 1
        df = df.with_columns([
            pl.col(col).shift(last_col_shift_num).alias(f's{str(last_col_shift_num)}'),
            pl.col("code").shift(last_col_shift_num).alias("code_r")
        ])
        
        # mcode == mcode_rの行のみfilter(抽出)する
        df = df.filter(pl.col("code")==pl.col("code_r"))
        
        # 移動平均を計算
        df = df.with_columns([pl.lit(0).alias("sum")])
        for i in range(term):
            col = f's{str(i)}'
            df = df.with_columns([
                (pl.col(col) + pl.col("sum")).alias("sum")
            ])
        moving_average_col_name = f'ma{term}'
        df = df.with_columns([
            (pl.col("sum") / pl.lit(term)).alias(moving_average_col_name)
        ])
        
        # 必要な列だけ残す
        df = df.select(self.df.columns + [moving_average_col_name])
    
        self.df = df

class KessanPl():
    def __init__(self, df: pl.DataFrame):
        # 列名を変更
        if "mcode" in df.columns:
            df = df.rename({
                "mcode": "code"
            })
        
        self.df = df

        # スクレイパーによるbugによる誤データの修正。
        # バグはすでに修正されているが、databaseのレコードも修正であり、暫定的に読み込んだpolaras.DataFrameを修正することとした
        self.df = self.revice_settlement_date_bug()
        
    def filter_settlement_type(self, settlement_type: Literal["quaterly", "yearly"]) -> None:
        df = self.df
        
        if settlement_type == "quaterly":
            t = "四"
        elif settlement_type == "yearly":
            t = "本"
            
        self.df = df.filter(pl.col("settlement_type")==t)
    
    def filter_code(self, code: int) -> None:
        self.df = self.df.filter(pl.col("code")==code)

    # codeで指定した銘柄の年決算のリスト(履歴)を返す
    # valuation_dateを指定すると、指定日時点までの年決算を返す。
    # get_latest_forcast = Trueとした場合、valuation_date時点の最新の決算予想を返す
    def get_target_stock_yearly_settlements(self, code: int, get_latest_forcast=True, valuation_date: date=date.today()) -> pl.DataFrame:
        df = self.df
        df = df.filter(pl.col("code")==code)\
            .filter(pl.col("settlement_type")=="本")\
            .filter(pl.col("announcement_date")<valuation_date)
        
        if not get_latest_forcast:
            return df
        
        fdf = self.df
        fdf = fdf.filter(pl.col("code")==code)\
            .filter(pl.col("settlement_type")=="予")\
            .filter(pl.col("settlement_date")>df["settlement_date"].max())
        fdf = fdf.filter(pl.col("settlement_date")==pl.col("settlement_date").min())

        rdf = pl.concat([df, fdf])

        rdf = rdf.sort([pl.col("settlement_date")])
        
        return rdf

    def get_latest_yearly_settlements(self, 
            reference_date: date=date.today(),
            settlement_type: Literal["本", "予"]="本"
        ) -> pl.DataFrame:
        df = self.df

        df = df.filter(pl.col("settlement_type")==settlement_type)\
            .filter(pl.col("announcement_date")<reference_date)
        
        df = df.with_columns([
            pl.col("code").shift(-1).alias("tmp")
        ])
        
        df = df.filter(pl.col("code")!=pl.col("tmp"))
        
        # 決算データの更新されているもののみを抽出する
        term = relativedelta(months=13)
        cut_date = reference_date - term
        df = df.filter(pl.col("announcement_date")>=cut_date)
        
        df = df.drop(["tmp"])
        
        return df
    
    # 年度決算の実績値における、当該年度の四半期決算の進捗率をpl.DataFrameで作成し、返す
    def get_actual_quatery_settlements_progress_rate(self) -> pl.DataFrame:
        # 四半期単体決算のsales～filal_profitの同一決算期における累積列を追加
        self.with_columns_accumulated_quaterly_settlement()
        df = self.df

        # 本決算(年度決算)のみ抽出
        ydf = df.filter(pl.col("settlement_type")=="本")

        # 四半期決算のみ抽出
        qdf = df.filter(pl.col("settlement_type")=="四")

        # 四半期決算と本決算を連結
        df = qdf.join(ydf, on=["code", "yearly_settlement_date"], how="left")

        # nullは削除する
        df = df.drop_nulls()

        # 決算進捗率列を追加
        target_cols = [
            "sales",
            "operating_income",
            "ordinary_profit",
            "final_profit"
        ]
        for c in target_cols:
            df = df.with_columns([
                (100*pl.col(f'acc_{c}')/pl.col(f'{c}_right')).round(1).alias(f'{c}_pr(%)')
            ])

        # 列の絞り込み
        selected_cols = [
            "code",
            "settlement_date",
            "yearly_settlement_date",
            "quater",
            "announcement_date",
            "sales_pr(%)",
            "operating_income_pr(%)",
            "ordinary_profit_pr(%)",
            "final_profit_pr(%)",
            "acc_sales",
            "acc_operating_income",
            "acc_ordinary_profit",
            "acc_final_profit",
            "announcement_date_right",	
            "sales_right",
            "operating_income_right",
            "ordinary_profit_right",
            "final_profit_right"	
        ]
        df = df.select(selected_cols)

        # 列名を変更
        rename_target_cols = selected_cols[-5:]
        rename_map = {}
        for c in rename_target_cols:
            rename_map[c] = f'yearly_{c.replace("_right", "")}'
        rename_map["acc_sales"] = "q_sales"
        rename_map["acc_operating_income"] = "q_operating_income"
        rename_map["acc_ordinary_profit"] = "q_ordinary_profit"
        rename_map["acc_final_profit"] = "q_final_profit"

        df = df.rename(rename_map)

        # 冒頭のwith_columns_accumulated_quaterly_settlementで計算のために追加した列を削除する
        self.df = self.df.select(self.df.columns[:-5])

        return df







    # evaluation_dateで指定した日における、決算進捗率が取得可能な全銘柄の四半期決算進捗率をpl.DataFrameで作成し、返す
    # 進捗率は、evaluation_date時における当期最新決算予想に対する四半期決算の進捗率。
    def get_expected_quatery_settlements_progress_rate(self, valuation_date: date=date.today()) -> pl.DataFrame:
        # 四半期単体決算のsales～filal_profitの同一決算期における累積列を追加
        self.with_columns_accumulated_quaterly_settlement()

        # 決算発表日はvaludation_dateよりも前
        df = self.df
        df = df.filter(pl.col("announcement_date")<valuation_date)

        # yearly_settlement_dateはvaludation_date以降
        df = df.filter(pl.col("yearly_settlement_date")>=valuation_date)

        # valuation_date直近の決算予想dfのみ抽出
        exdf = df.filter(pl.col("settlement_type")=="予")
        exdf = exdf.group_by(["code", "yearly_settlement_date"]).agg([
            pl.col("settlement_date").last().alias("settlement_date"),
            pl.col("settlement_type").last().alias("settlement_type"),
            pl.col("announcement_date").last().alias("announcement_date"),
            pl.col("sales").last().alias("sales"),
            pl.col("operating_income").last().alias("operating_income"),
            pl.col("ordinary_profit").last().alias("ordinary_profit"),
            pl.col("final_profit").last().alias("final_profit")
        ])

        # 決算予想と決算予想決算期対象の四半期決算を連結
        qdf = df.filter(pl.col("settlement_type")=="四")
        df = qdf.join(exdf, on=["code", "yearly_settlement_date"], how="left")

        # 決算進捗率列を追加
        target_cols = [
            "sales",
            "operating_income",
            "ordinary_profit",
            "final_profit"
        ]
        for c in target_cols:
            df = df.with_columns([
                (100*pl.col(f'acc_{c}')/pl.col(f'{c}_right')).round(1).alias(f'{c}_pr(%)')
            ])

        # 列の絞り込み
        selected_cols = [
            "code",
            "settlement_date",
            "yearly_settlement_date",
            "quater",
            "announcement_date",
            "sales_pr(%)",
            "operating_income_pr(%)",
            "ordinary_profit_pr(%)",
            "final_profit_pr(%)",
            "acc_sales",
            "acc_operating_income",
            "acc_ordinary_profit",
            "acc_final_profit",
            "announcement_date_right",	
            "sales_right",
            "operating_income_right",
            "ordinary_profit_right",
            "final_profit_right"	
        ]
        df = df.select(selected_cols)

        # 列名を変更
        rename_target_cols = selected_cols[-5:]
        rename_map = {}
        for c in rename_target_cols:
            rename_map[c] = f'forcast_{c.replace("_right", "")}'
        rename_map["acc_sales"] = "q_sales"
        rename_map["acc_operating_income"] = "q_operating_income"
        rename_map["acc_ordinary_profit"] = "q_ordinary_profit"
        rename_map["acc_final_profit"] = "q_final_profit"

        df = df.rename(rename_map)

        # 冒頭のwith_columns_accumulated_quaterly_settlementで計算のために追加した列を削除する
        self.df = self.df.select(self.df.columns[:-5])

        return df
    
    # KessanPlの四半期決算、および本決算の決算発表日から翌決算発表日までの株価の騰落率列と同期間の日経平均の騰落率列を追加したpl.DataFrameを返す
    # pricelist_dfが空のdataframe(初期値)の場合、parquetファイルから読み込んでくる。
    # overnight_biginingをTrueにセットすると、起点の株価として決算発表日当日の株価をセットし、Falseにセットすると、決算発表日翌営業日の株価をセットする。
    # overnight_endをTrueにセットすると、終点の株価として決算発表日翌営業日の株価をセットし、Falseにセットすると、決算発表日当日の株価をセットする。
    # *_pointは、期首(bigining)と期末(end)において、日足ローソクのどの時点の株価を起点、または終点とするか選択する。
    def get_settlement_performance(self,
        pricelist_df: pl.DataFrame = pl.DataFrame(),
        overnight_bigining: bool = False,
        overnight_end: bool = True,
        bigining_point: Literal["open", "high", "low", "close"] = "open",
        end_point: Literal["open", "high", "low", "close"] = "open"
    ) -> pl.DataFrame:
        df = self.df
        
        # precelist_df
        if pricelist_df.shape[0] == 0:
            fp = DATA_DIR/"reviced_pricelist.parquet"
            df = read_data(fp)
            RPL = PricelistPl(df)
            pricelist_df = RPL.df
            
        
        # 本決算
        yitems_df = self.get_settlement_performance_items_df("本", pricelist_df)            
        
        return yitems_df
    
    # 決算期間中における株価騰落を求めるための引数一覧をpl.DataFrameで取得する
    # 取得されるdfの列は、"code", "start_date", "end_date"
    def get_settlement_performance_items_df(self,
        settlement_type: Literal["本", "四"],
        pricelist_df: pl.DataFrame,
        overnight_bigining: bool = False,
        overnight_end: bool = True,
    ) -> pl.DataFrame:
        
        df = self.df

        df = df.filter(pl.col("settlement_type")==settlement_type)
        df = df.with_columns([
            pl.col("code").shift(-1).alias("ncode"),
            pl.col("announcement_date").shift(-1).alias("end_date")
        ])
        
        if not overnight_bigining:
            df = df.with_columns([
                (pl.col("announcement_date") + pl.duration(days=1)).alias("start_date")
            ])
        
        if overnight_end:
            df = df.with_columns([
                (pl.col("end_date") + pl.duration(days=1)).alias("end_date")
            ])
        
        # 騰落率を取得するための引数表を作成
        df = df.select([
            "code",
            "start_date",
            "end_date"
        ])
        
        df = df.drop_nulls()
        
        # start_date
        df1 = df.select(["code", "start_date"])
        df2 = pricelist_df.select(["code", "date"])
        df3 = df1.join(df2, on="code", how="inner")
        df3 = df3.filter(pl.col("start_date")<=pl.col("date"))
        df3 = df3.group_by(["code", "start_date"]).agg([
            pl.col("date").min()
        ])
        df3 = df3.with_columns([pl.col("date").alias("start_date")])
        
        #ここから
        
        
        return df3

    # 決算データスクレイピング時のバグを修正。
    # バグはすでに修正されているが、databaseのレコードが修正されていないため、暫定的にpolars.DataFrameを読み込んだ後に修正する
    def revice_settlement_date_bug(self) -> None:
        df = self.df

        # 日付の差分をとる
        df = df.with_columns([
            (pl.col("announcement_date")-pl.col("settlement_date")).alias("delta_days")
        ])

        # 修正対象レコードを作成
        # 対象がない場合はそのままself.dfを元に戻して返す
        reviced_recs_df = df.filter(pl.col("delta_days")>=pl.duration(days=365))
        if reviced_recs_df.shape[0] == 0:
            return df.select(df.columns[:-1])
        reviced_recs_df = reviced_recs_df.with_columns([
            (pl.col("settlement_date")+pl.duration(days=360)).alias("new_sett_date")
        ])
        reviced_recs_df = reviced_recs_df.map_rows(revice_last_date)

        # 列名をただす
        rename_map = {
            'column_0': 'code',
            'column_1': 'settlement_date',
            'column_2': 'settlement_type',
            'column_3': 'announcement_date',
            'column_4': 'sales',
            'column_5': 'operating_income',
            'column_6': 'ordinary_profit',
            'column_7': 'final_profit',
            'column_8': 'reviced_eps',
            'column_9': 'dividend',
            'column_10':'quater',
            'column_11':'delta_days',
            'column_12':'new_sett_date'
        }
        reviced_recs_df = reviced_recs_df.rename(rename_map)
        reviced_recs_df = reviced_recs_df.with_columns([
            pl.col("new_sett_date").alias("settlement_date")
        ])
        reviced_recs_df = reviced_recs_df.select(reviced_recs_df.columns[:11])
        
        # 元のデータで、delta_daysが365日を超えているものは誤データなので、消す
        df = df.filter(pl.col("delta_days")< pl.duration(days=360))
        df = df.select(df.columns[:11])

        # concatしてdrop_duplicatesしてsort
        df = pl.concat([df, reviced_recs_df])

        # settlement_type=="四"のdrop_duplicate
        df1 = df.filter(pl.col("settlement_type")=="四")
        df2 = df.filter(pl.col("settlement_type")!="四")
        df1 = df1.unique(subset=["code", "settlement_date", "settlement_type"])
        df = pl.concat([df1, df2])

        df = df.sort([
            pl.col("code"),
            pl.col("announcement_date")
        ])

        # 2020年のうるう年のバグを修正
        df = df.filter(pl.col("settlement_date")!=date(2020, 2, 28))
        
        return df

    def with_columns_yearly_settlement_date(self) -> None:
        df = self.df
        original_cols = df.columns

        # 最終行の1つ前にsettlement_dateの列indexを、
        # 最終行にquaterの列indexを追加してget_yearly_settlement_dateで
        # yearly_settlement_date列を追加できるようにする。
        sd_idx = original_cols.index("settlement_date")
        qt_idx = original_cols.index("quater")

        df = df.with_columns([
            pl.lit(sd_idx).alias("sd_idx"),
            pl.lit(qt_idx).alias("qt_idx")
        ])

        df = df.map_rows(get_yearly_settlement_date)

        # 列名を元に戻す
        col_dct = {}
        num_original_cols = len(self.df.columns)
        for i in range(num_original_cols):
            c1 = f"column_{str(i)}"
            col_dct[c1] = original_cols[i]
        
        # 最終列の列名を変更
        num_new_cols = len(df.columns)
        col_dct[f'column_{str(num_new_cols-1)}'] = "yearly_settlement_date"

        # 計算のために追加したいらない列(sd_idxとqt_idx)を削除する
        df = df.drop([
            f'column_{str(num_new_cols-2)}',
            f'column_{str(num_new_cols-3)}'
        ])

        self.df = df.rename(col_dct)

    def with_columns_accumulated_quaterly_settlement(self) -> None:
        # KessanPl.dfに年度決算日列を追加
        self.with_columns_yearly_settlement_date()

        df = self.df
        tcol = "settlement_type"
        target_cols = ["sales", "operating_income", "ordinary_profit", "final_profit"]
        on_keys = ["code", "yearly_settlement_date"]

        pdfs = []

        # 年度決算レコード(四半期決算以外)
        # sales ～ final_profitをコピーするだけ
        y_df = df.filter(pl.col(tcol)!="四")
        for c in target_cols:
            y_df = y_df.with_columns([
                pl.col(c).alias(f'acc_{c}')
            ])
        pdfs.append(y_df)
        
        # 第1四半期決算レコード
        # sales ～ final_profitをコピーするだけ
        qcol = "quater"
        q1df = df.filter(pl.col(qcol)==1)
        for c in target_cols:
            q1df = q1df.with_columns([
                pl.col(c).alias(f'acc_{c}')
            ])
        q1_df = q1df
        pdfs.append(q1_df)
        
        # 第4四半期決算レコード
        # 本決算からコピー
        original_cols = self.df.columns
        q4df = df.filter(pl.col(tcol)=="四").filter(pl.col(qcol)==4)
        qydf = df.filter(pl.col(tcol)=="本")
        pdf = q4df.join(qydf, on=on_keys, how="left")
        colmap = {}
        added_cols = []
        for c in target_cols:
            added_col = f'acc_{c}'
            colmap[f'{c}_right'] = added_col
            added_cols.append(added_col)
        pdf = pdf.rename(colmap)
        pdf = pdf.select(original_cols+added_cols)

        q4_df = pdf
        pdfs.append(q4_df)

        # 第2四半期、第3四半期
        q1df = df.filter(pl.col(qcol)==1)
        q2df = df.filter(pl.col(qcol)==2)
        q3df = df.filter(pl.col(qcol)==3)

        # 第2四半期(後ろに前を連結するのでhowはrightにして、なるべくnull値がないようにする)
        pdf = q2df.join(q1df, on=on_keys, how="right")

        for c in target_cols:
            pdf = pdf.with_columns([
                (pl.col(c)+pl.col(f'{c}_right')).alias(f'acc_{c}')
            ])
        pdf = pdf.select(original_cols+added_cols)
        q2_df = pdf
        pdfs.append(q2_df)

        # 第3四半期
        pdf = q3df.join(q2_df, on=on_keys, how="right")

        for c in target_cols:
            pdf = pdf.with_columns([
                (pl.col(c)+pl.col(f'acc_{c}')).alias(f'acc_{c}')
            ])
        pdf = pdf.select(original_cols+added_cols)
        q3_df = pdf
        pdfs.append(q3_df)

        # 各部分dfをconcat
        df = pdfs[0]
        for adf in pdfs[1:]:
            df = pl.concat([df, adf])
        
        # sort
        df = df.sort([
            pl.col("code"),
            pl.col("settlement_type"),
            pl.col("announcement_date")
        ])

        # nullは削除する
        df = df.drop_nulls()

        self.df = df

    # 前年同期と比較して、差分利益率：(今年度利益率-昨年度利益率)/(今年度売上高-昨年度売上高)
    # を営業利益～純利益の各差分利益について計算して列を追加する。
    # 売上高に対しては、売上高伸び率列を追加する。
    # 利益改善度合いを見るために利用する。
    # 決算予想の場合は、昨年度の実績に対して出す。
    # 次期移行の予想はnull。
    def with_columns_diff_growthrate(self) -> None:
        df = self.df
        ori_cols = df.columns

        # 四半期
        qdf = df.filter(pl.col("settlement_type")=="四")
        # 昨年度の列を同じレコードに連結
        for c in qdf.columns:
            qdf = qdf.with_columns([
                pl.col(c).shift(4).alias(f'ly_{c}')
            ])
        qdf = qdf.with_columns([
            (pl.col("settlement_date")-pl.col("ly_settlement_date")).alias("diff_sett")
        ])

        # 前年同期が比較できるものだけ、filterする
        qdf = qdf.filter(pl.col("quater")==pl.col("ly_quater"))\
            .filter(pl.col("diff_sett")>=pl.duration(days=365))\
            .filter(pl.col("diff_sett")<=pl.duration(days=366))

        # 追加列を計算する
        # 売上高伸び率
        qdf = qdf.with_columns([
            ((pl.lit(100)*(pl.col("sales")-pl.col("ly_sales"))/pl.col("ly_sales")).round(1)).alias("sales_growthrate")
        ])
        # 差分利益成率
        target_cols = ["operating_income", "ordinary_profit", "final_profit"]
        for c in target_cols:
            qdf = qdf.with_columns([
                ((pl.lit(100)*(pl.col(c)-pl.col(f"ly_{c}"))/(pl.col("sales")-pl.col("ly_sales"))).round(1)).alias(f"diff_{c}_growthrate")
            ])

        # select
        qdf = qdf.select(ori_cols+qdf.columns[-4:])

        # 本決算
        ydf = df.filter(pl.col("settlement_type")=="本")
        # 昨年度の列を同じレコードに連結
        for c in ydf.columns:
            ydf = ydf.with_columns([
                pl.col(c).shift(1).alias(f'ly_{c}')
            ])
        ydf = ydf.with_columns([
            (pl.col("settlement_date")-pl.col("ly_settlement_date")).alias("diff_sett")
        ])
        # 前年同期が比較できるものだけ、filterする
        ydf = ydf.filter(pl.col("quater")==pl.col("ly_quater"))\
            .filter(pl.col("diff_sett")>=pl.duration(days=365))\
            .filter(pl.col("diff_sett")<=pl.duration(days=366))
        # 追加列を計算する
        # 売上高伸び率
        ydf = ydf.with_columns([
            ((pl.lit(100)*(pl.col("sales")-pl.col("ly_sales"))/pl.col("ly_sales")).round(1)).alias("sales_growthrate")
        ])
        # 差分利益率
        target_cols = ["operating_income", "ordinary_profit", "final_profit"]
        for c in target_cols:
            ydf = ydf.with_columns([
                ((pl.lit(100)*(pl.col(c)-pl.col(f"ly_{c}"))/(pl.col("sales")-pl.col("ly_sales"))).round(1)).alias(f"diff_{c}_growthrate")
            ])

        # select
        ydf = ydf.select(ori_cols+qdf.columns[-4:])

        # 決算予想
        fdf = df.filter(pl.col("settlement_type")=="予")
        fdf = fdf.with_columns([
            pl.col("settlement_date").alias("key")
        ])

        pydf = df.filter(pl.col("settlement_type")=="本")
        pydf_cols = pydf.columns
        pydf = pydf.with_columns([
            (pl.col("settlement_date") + pl.duration(days=364)).alias("key")
        ])
        pydf = pydf.map_rows(revice_last_date)
        pydf.columns = pydf_cols + ["key"]
        rename_cols = pydf.columns[1:-1]
        rename_map = {}
        for c in rename_cols:
            rename_map[c] = f'ly_{c}'
        pydf = pydf.rename(rename_map)
        # 連結
        key_cols = ["code", "key"]
        fdf = fdf.join(pydf, on=key_cols, how="left")
        # 売上高伸び率
        fdf = fdf.with_columns([
            ((pl.lit(100)*(pl.col("sales")-pl.col("ly_sales"))/pl.col("ly_sales")).round(1)).alias("sales_growthrate")
        ])
        # 差分利益率
        target_cols = ["operating_income", "ordinary_profit", "final_profit"]
        for c in target_cols:
            fdf = fdf.with_columns([
                ((pl.lit(100)*(pl.col(c)-pl.col(f"ly_{c}"))/(pl.col("sales")-pl.col("ly_sales"))).round(1)).alias(f"diff_{c}_growthrate")
            ])
        # select
        fdf = fdf.select(ori_cols+qdf.columns[-4:])

        # それぞれのdfをconcat
        df = pl.concat([qdf, ydf, fdf])

        # なくなったレコードを元に戻す
        df2 = self.df
        df2 = df2.join(df, on=["code", "settlement_date", "announcement_date", "settlement_type"], how="anti")
        added_cols = df.columns[-4:]
        for c in added_cols:
            df2 = df2.with_columns([
                pl.lit(None, dtype=pl.Float64).alias(c)
            ])
        df = pl.concat([df, df2])

        self.df = df
        self._sort_df()

    def with_columns_expected_quatery_settlements_progress_rate(self, valuation_date: date=date.today()) -> None:
        # 四半期単体決算のsales～filal_profitの同一決算期における累積列を追加
        self.with_columns_accumulated_quaterly_settlement()

        # 決算発表日はvaludation_dateよりも前
        df = self.df
        df = df.filter(pl.col("announcement_date")<valuation_date)

        # yearly_settlement_dateはvaludation_date以降
        df = df.filter(pl.col("yearly_settlement_date")>=valuation_date)

        # valuation_date直近の決算予想dfのみ抽出
        exdf = df.filter(pl.col("settlement_type")=="予")
        exdf = exdf.group_by(["code", "yearly_settlement_date"]).agg([
            pl.col("settlement_date").last().alias("settlement_date"),
            pl.col("settlement_type").last().alias("settlement_type"),
            pl.col("announcement_date").last().alias("announcement_date"),
            pl.col("sales").last().alias("sales"),
            pl.col("operating_income").last().alias("operating_income"),
            pl.col("ordinary_profit").last().alias("ordinary_profit"),
            pl.col("final_profit").last().alias("filan_profit")
        ])

        # 決算予想と決算予想決算期対象の四半期決算を連結
        qdf = df.filter(pl.col("settlement_type")=="四")
        df = qdf.join(exdf, on=["code", "yearly_settlement_date"], how="left")

        # 決算進捗率列を追加
        # ここから

        

        self.df = df

    # 結果出力をしやすいように、決算期の表記を日本語にした列を追加
    # add_settlement_type_string=Trueの場合、「〇年〇月期決算」決算の後ろに、決算種別を追加する。Falseの場合は〇年〇月期までしか表示しない。
      # 本決算 -> 〇年〇月期通期決算
      # 四半期決算 -> 〇年〇月第〇(単体|累積)四半期決算
    # KessanPl.DataFrameに四半期決算データが単体データか累積データ化識別できないので、かっこわるいが指定する。
    def with_columns_financtial_period(self, add_settlement_type_string=False, quaterly_settlement_type: Literal["単体", "累積"]="単体") -> None:
        self.with_columns_yearly_settlement_date()
        df = self.df

        # quater値の修正
        df = df.with_columns([
            pl.when(pl.col("quater")==-2)
            .then(4)
            .otherwise(pl.col("quater")).alias("quater")
        ])
        
        # 列を追加
        df = df.with_columns([
            (pl.col("yearly_settlement_date").dt.year()).alias("fy").cast(pl.Utf8),
            (pl.col("yearly_settlement_date").dt.month()).alias("fm").cast(pl.Utf8),
        ])

        # 文字列を連結して列を追加
        df = df.with_columns(
            pl.concat_str(["fy", pl.lit("年"), "fm", pl.lit("月期")]).alias("決算期")
        )

        # 通期/四半期を追加
        if add_settlement_type_string:
            df = df.with_columns([
                pl.when(pl.col("settlement_type")=="四")
                .then(pl.col("決算期")+pl.lit("第")+pl.col("quater").cast(pl.Utf8)+pl.lit(quaterly_settlement_type)+pl.lit("四半期決算"))
                .otherwise(pl.col("決算期")+pl.lit("通期決算"))
                .alias("決算期")
            ])


        self.df = df

    # 売上高～純利益までの前年同期からの成長率列を追加する
    def with_columns_growthrate_lastyear(self):
        df = self.df
        ori_cols = df.columns

        # 四半期
        qdf = df.filter(pl.col("settlement_type")=="四")
        # 昨年度の列を同じレコードに連結
        for c in qdf.columns:
            qdf = qdf.with_columns([
                pl.col(c).shift(4).alias(f'ly_{c}')
            ])
        qdf = qdf.with_columns([
            (pl.col("settlement_date")-pl.col("ly_settlement_date")).alias("diff_sett")
        ])
        # 前年同期が比較できるものだけ、filterする
        qdf = qdf.filter(pl.col("quater")==pl.col("ly_quater"))\
            .filter(pl.col("diff_sett")>=pl.duration(days=365))\
            .filter(pl.col("diff_sett")<=pl.duration(days=366))
        # 追加列を計算する
        target_cols = ["sales", "operating_income", "ordinary_profit", "final_profit"]
        # 成長率
        for c in target_cols:
            qdf = qdf.with_columns([
                ((pl.lit(100)*(pl.col(c)-pl.col(f"ly_{c}"))/pl.col(f"ly_{c}")).round(1)).alias(f"{c}_growthrate")
            ])
        # select
        qdf = qdf.select(ori_cols+qdf.columns[-4:])

        # 本決算
        ydf = df.filter(pl.col("settlement_type")=="本")
        # 昨年度の列を同じレコードに連結
        for c in ydf.columns:
            ydf = ydf.with_columns([
                pl.col(c).shift(1).alias(f'ly_{c}')
            ])
        ydf = ydf.with_columns([
            (pl.col("settlement_date")-pl.col("ly_settlement_date")).alias("diff_sett")
        ])
        # 追加列を計算する
        # 伸び率
        target_cols = ["sales", "operating_income", "ordinary_profit", "final_profit"]
        for c in target_cols:
            ydf = ydf.with_columns([
                ((pl.lit(100)*(pl.col(c)-pl.col(f"ly_{c}"))/pl.col(f"ly_{c}")).round(1)).alias(f"{c}_growthrate")
            ])
        # select
        ydf = ydf.select(ori_cols+qdf.columns[-4:])


        # 決算予想
        fdf = df.filter(pl.col("settlement_type")=="予")
        fdf = fdf.with_columns([
            pl.col("settlement_date").alias("key")
        ])
        pydf = df.filter(pl.col("settlement_type")=="本")
        pydf_cols = pydf.columns
        pydf = pydf.with_columns([
            (pl.col("settlement_date") + pl.duration(days=364)).alias("key")
        ])
        pydf = pydf.map_rows(revice_last_date)
        pydf.columns = pydf_cols + ["key"]
        rename_cols = pydf.columns[1:-1]
        rename_map = {}
        for c in rename_cols:
            rename_map[c] = f'ly_{c}'
        pydf = pydf.rename(rename_map)
        # 連結
        key_cols = ["code", "key"]
        fdf = fdf.join(pydf, on=key_cols, how="left")
        # 追加列を計算する
        # 伸び率
        target_cols = ["sales", "operating_income", "ordinary_profit", "final_profit"]
        for c in target_cols:
            fdf = fdf.with_columns([
                ((pl.lit(100)*(pl.col(c)-pl.col(f"ly_{c}"))/pl.col(f"ly_{c}")).round(1)).alias(f"{c}_growthrate")
            ])
        # select
        fdf = fdf.select(ori_cols+qdf.columns[-4:])

        # それぞれのdfをconcat
        df = pl.concat([qdf, ydf, fdf])

        # なくなったレコードを元に戻す
        df2 = self.df
        df2 = df2.join(df, on=["code", "settlement_date", "announcement_date", "settlement_type"], how="anti")
        added_cols = df.columns[-4:]
        for c in added_cols:
            df2 = df2.with_columns([
                pl.lit(None, dtype=pl.Float64).alias(c)
            ])
        df = pl.concat([df, df2])
        
        self.df = df
        self._sort_df()




    # 作りかけ
    def with_columns_settlements_progress_rate(self) -> None:
        # KessanPl.dfに年度決算日列を追加
        self.with_columns_yearly_settlement_date()

        # 決算実績と決算予想でdfを分割
        df0 = self.df.filter(pl.col("settlement_type")!="予")
        df1 = self.df.filter(pl.col("settlement_type") =="予")

        # joinしてnull列を削除
        df = df0.join(df1, on=["code", "yearly_settlement_date"], how="left")
        df = df.filter(pl.col("settlement_type_right")=="予")

        # レコードの決算発表時に発表済のレコードのみ抽出
        df = df.filter(pl.col("announcement_date")>pl.col("announcement_date_right"))

        self.df = df

    def _sort_df(self):
        df = self.df

        df = df.sort([
            pl.col("code"),
            pl.col("announcement_date"),
            pl.col("settlement_type")
        ])

        self.df = df

class MeigaralistPl():
    def __init__(self, df: pl.DataFrame):
        # 列名を変更
        if "mcode" in df.columns:
            df = df.rename({
                "mcode": "code",
                "mname": "name"
            })
        
        self.df = df
    
    # 証券コードから、会社名を取得して返す
    def get_name(self, code: int) -> str:
        return self.df.filter(pl.col("code")==code).select(["name"]).to_series().item()

# shikiho.parquetを読みこんでデータの抽出、加工、分析などを行う
class ShikihoPl():
    def __init__(self, df: pl.DataFrame):
        # 列名を変更
        if "mcode" in df.columns:
            df = df.rename({
                "mcode": "code",
                "mname": "name"
            })

        self.df = df
    
    # target_dateで指定した日における最新発行済のデータを抽出する
    def get_latest_df(self, target_date: date=date.today()) -> pl.DataFrame:
        df = self.df
        
        df = df.filter(pl.col("issue")<target_date)
        df = df.filter(pl.col("issue")==pl.col("issue").max())
        
        return df
    
    # codeで指定した銘柄のtarget_dateで指定した日における最新発行済のデータを抽出する
    def get_latest_stock_df(self, code: int ,target_date: date=date.today()) -> pl.DataFrame:
        df = self.get_latest_df(target_date)
        
        return df.filter(pl.col("code")==code)
    
    # codeで指定した銘柄のtarget_dateで指定した日における最新発行済のデータを標準出力する
    def print_latest_stock_df(self, code: int ,target_date: date=date.today()) -> None:
        df = self.get_latest_stock_df(code, target_date)
        
        cols = df.columns
        
        map_dct = {}
        for col in cols:
            map_dct[col] = df.row(0)[cols.index(col)]
        
        print(f'{map_dct["code"]}({map_dct["name"]})の{map_dct["issue"].strftime(DATEFORMAT2)}発行四季報データ')
        print(f'{map_dct["title1"]}')
        print(f'  {map_dct["comment1"]}')
        print(f'{map_dct["title2"]}')
        print(f'  {map_dct["comment2"]}')
    
    # codeで指定した銘柄の過去も含めた四季報のtitle1/comment1, title2/comment2を標準出力する
    # valuation_dateで指定した日以前のものを出力する。
    # numで指定した数だけ出力。
    # num=0を指定すると、すべてのデータを出力する。
    def print_stock_df(self, code: int, valuation_date: date=date.today(), num: int=0) -> None:
        df = self.df
        df = df.filter(pl.col("code")==code)\
            .filter(pl.col("issue")<=valuation_date)\
            .sort(by=["issue"], descending=[True])
        if num !=0:
            df = df.with_row_count(name="index")
            df = df.filter(pl.col("index")<num)
            df = df.select(df.columns[1:])

        name = df["name"][0]
        print(f'{code}({name})の四季報データ履歴')
        
        for i in range(df.shape[0]):
            print()
            self._print_row(df.row(i))
            
            
        
    # ShikihoPl.dfの行を標準出力する
    def _print_row(self, row):
        cols = self.df.columns

        map_dct = {}
        for col in cols:
            map_dct[col] = row[cols.index(col)]
        
        print(f'発行日: {map_dct["issue"].strftime(DATEFORMAT2)}')
        print(f'{map_dct["title1"]}')
        print(f'  {map_dct["comment1"]}')
        print(f'{map_dct["title2"]}')
        print(f'  {map_dct["comment2"]}')
        
        
        
        

# 決算推移グラフを描画する
class KessanFig():
    def __init__(self, 
            code: int, 
            settlement_type: Literal["通期", "四半期"], 
            output_target: str = "jupyter",
            start_settlement_date: date = date(1900, 1, 1),
            end_settlement_date: date = date(2999, 12, 31)
        ):
        
        fp = DATA_DIR / "kessan.parquet"
        self.original_df = read_data(fp)
        df = self.original_df

        # スクレイピング時のバグを修正
        KPL = KessanPl(df)
        KPL.revice_settlement_date_bug()

        
        df = df.rename({
            "mcode": "code"
        })
        
        if settlement_type == "通期":
            st = "本"
        elif settlement_type == "四半期":
            st = "四"
        
        df = df.filter(pl.col("code")==code)\
            .filter(pl.col("settlement_type")==st)\
            .filter(pl.col("settlement_date")>=start_settlement_date)\
            .filter(pl.col("settlement_date")<=end_settlement_date)
        KPL = KessanPl(df)
        KPL.with_columns_financtial_period()
        self.df = KPL.df
        
        self.code = code
        self.settlement_type = settlement_type
        self.start_settlement_date = start_settlement_date
        self.end_settlement_date = end_settlement_date
        
        today = date.today()
        if end_settlement_date >= today:
            self.end_settlement_date = today
        else:
            self.end_settlement_date = end_settlement_date
        self.name = get_companyname(code)
        
        # jupyterにグラフを描画する場合は、pio.renderers.defalutを'iframe'に設定する 
        if output_target == "jupyter":
            pio.renderers.default = 'iframe'
        
        # 売上高棒グラフのグラフオブジェクトを生成
        if settlement_type == "通期":
            self.fig = self.yearly_settlement_trend_barchart()
        elif settlement_type == "四半期":
            self.fig = self.quaterly_settlement_trend_barchart()
    
    def quaterly_settlement_trend_barchart(self) -> Figure:
        df = self.df

        # x軸のラベル用に列をカスタマイズして追加
        df = df.with_columns([
            pl.col("quater").cast(pl.Utf8),
            pl.col("fy").cast(pl.Utf8),
            pl.col("fm").cast(pl.Utf8)
        ])
        df = df.with_columns([
            (pl.col("fy")+pl.lit("-")+pl.col("fm")+pl.lit("-")+pl.col("quater")+pl.lit("Q")).alias("xlabels")
        ])
        
        self.df = df

        pandas_df = df.to_pandas()
        sales_df = pandas_df[["xlabels", "sales"]]

        # 棒グラフのセット
        graph_data = [
            go.Bar(
                x = sales_df["xlabels"],
                y = sales_df["sales"],
                marker = dict(color="skyblue"),
                name = "売上高"
            )
        ]
        fig = go.Figure(graph_data)
        
        # 年度の区切り線を引く
        q4_df = pandas_df[pandas_df["quater"]=="4"]
        vline_x_positions = q4_df.index
        for q4x in vline_x_positions:
            xpos = int(q4x) + 0.5
            
            fig.add_vline(
                x=xpos,  # 棒の間に対応する位置
                line=dict(color='gray', width=1),
                annotation_text="",  # ラベル（任意）
                annotation_position="top"
            )

        # グラフレイアウトの設定
        fig.update_layout(
            title=f'{self.name}({self.code})四半期業績推移({self.end_settlement_date.strftime(DATEFORMAT2)}時点)',
            xaxis=dict(title='年度'),
            yaxis=dict(title='売上高 (百万円)'),
            legend=dict(
                x=1.05,  # 凡例をグラフの外側に配置
                y=1,    # 上部に配置
                xanchor='left',  # 凡例の左端をx座標に揃える
                yanchor='top'    # 凡例の上端をy座標に揃える
            ),
            bargap=0.2  # 棒の間隔
        )
        
        return fig
        
    def yearly_settlement_trend_barchart(self) -> Figure:
        df = self.df
        
        df = df.select(["決算期", "sales"])
        sales_df = df.to_pandas()
        
        # 棒グラフのセット
        graph_data = [
            go.Bar(
                x = sales_df["決算期"],
                y = sales_df["sales"],
                marker = dict(color="skyblue"),
                name = "売上高"
            )
        ]
        fig = go.Figure(graph_data)

        # self.end_settlement_dateにおける最新forcastを追加
        KPL = KessanPl(self.original_df)
        fdf = KPL.get_latest_yearly_settlements(
                reference_date=self.end_settlement_date,
                settlement_type="予"
        )
        fdf = fdf.filter(pl.col("code")==self.code)

        # 決算予想が存在しない場合は、次年度予想のconcatをスキップ。
        if fdf.shape[0] != 0:
            KPL = KessanPl(fdf)
            KPL.with_columns_financtial_period()
            fdf = KPL.df
            fdf = fdf.with_columns([
                (pl.col("決算期")+pl.lit("(予)")).alias("決算期")
            ])
            
            # 他のメソッドで利用するために決算予想をconcatする
            self.df = pl.concat([self.df, fdf])
            
            # 元のグラフオブジェクトに決算予想の売上高をadd_traceする
            pandas_fdf = fdf.to_pandas()
            fig.add_trace(go.Bar(
                x = pandas_fdf["決算期"].iloc[-1:],
                y = pandas_fdf["sales"].iloc[-1:],
                name = "売上高(予)",
                marker = dict(color="lightpink")
                
            ))
        

        # グラフレイアウトの設定
        fig.update_layout(
            title=f'{self.name}({self.code})通期業績推移({self.end_settlement_date.strftime(DATEFORMAT2)}時点)',
            xaxis=dict(title='年度'),
            yaxis=dict(title='売上高 (百万円)'),
            legend=dict(
                x=1.05,  # 凡例をグラフの外側に配置
                y=1,    # 上部に配置
                xanchor='left',  # 凡例の左端をx座標に揃える
                yanchor='top'    # 凡例の上端をy座標に揃える
            ),
            bargap=0.2  # 棒の間隔
        )

        # add_trace_*メソッドでグラフを重ねられるように、"決算期"列を"xlabels"列にコピー。
        self.df = self.df.with_columns([
            pl.col("決算期").alias("xlabels")
        ])
        
        return fig
    

    # 右にy軸をとって各利益(営業利益～純利益)の折れ線グラフを追加する
    def add_trace_profits(self):
        df = self.df
        pandas_df = df.to_pandas()
        
        column_idx = 0
        label_idx = 1
        color_idx = 2
        line_trace_cols_attrs = [
            ['operating_income', '営業利益', 'orange'],
            ['ordinary_profit', '経常利益', 'lightgreen'],
            ['final_profit', '純利益', 'purple']
        ]

        for a in line_trace_cols_attrs:
            self.fig.add_trace(go.Scatter(
                x=pandas_df['xlabels'],
                y=pandas_df[a[column_idx]],
                mode='lines',
                name=a[label_idx],
                yaxis = 'y2',
                line=dict(color=a[color_idx], width=2)
            ))

        # レイアウトの設定
        self.fig.update_layout(
            title=f'{self.name}({self.code})四半期業績推移({self.end_settlement_date.strftime(DATEFORMAT2)}時点)',
            xaxis=dict(title='年度'),
            yaxis=dict(title='売上高 (百万円)'),
            yaxis2=dict(
                title="利益(百万円)",
                overlaying="y", # 左のY軸に重ねる
                side="right"
            ),
            legend=dict(
                x=1.05,  # 凡例をグラフの外側に配置
                y=1,    # 上部に配置
                xanchor='left',  # 凡例の左端をx座標に揃える
                yanchor='top'    # 凡例の上端をy座標に揃える
            ),
            bargap=0.2  # 棒の間隔
        )

# codeで指定した銘柄のローソク足チャートを描画する
# pricelist_dfを指定しない場合、dataファイルから読み込む
# ローソク足チャートの表示期間をstart_dateとend_dateで指定できる
# fig_typeを指定して表示するチャートの型を日足、週足、月足を選択できる。
class PricelistFig():
    def __init__(self,
        code: int,
        pricelist_df: Union[pl.DataFrame, None] = None,
        meigaralist_df: Union[pl.DataFrame, None] = None,
        start_date: date = date(1900, 1, 1),
        end_date: date = date(2999, 12, 31),
        fig_type: Literal["daily", "weekly", "monthly"] = "daily"
    ):
        # PricelistFigのプロパティ
        self.code = code
        self.start_date = start_date
        self.end_date = end_date
        self.ticknum = 10
        self.tickangle = 45
        
        if type(pricelist_df) != pl.DataFrame:
            fp = DATA_DIR / "raw_pricelist.parquet"
            pricelist_df = read_data(fp)
        PPL =  PricelistPl(pricelist_df)
        if type(meigaralist_df) != pl.DataFrame:
            fp = DATA_DIR / "meigaralist.parquet"
            meigaralist_df = read_data(fp)
        PPL = PricelistPl(pricelist_df)
        MPL = MeigaralistPl(meigaralist_df)
        
        self.name = MPL.get_name(code)
        
        df = PPL.df
        df = df.filter(pl.col("code")==code)\
            .filter(pl.col("date")>=start_date)\
            .filter(pl.col("date")<=end_date)
            
        self.datanum = df.shape[0]
        
        df = df.with_columns([
            pl.col("date").cast(pl.Utf8)
        ])
        
        PPL = PricelistPl(df)
        
        # weeklyとmonthly。別途作成する
        if fig_type == "weekly":
            df = PPL.get_weekly_df()
        elif fig_type == "monthly":
            df = PPL.get_monthly_df()
        
        self.df = df
        self.set_fig()
        
    def set_fig(self):
        pddf = self.df.to_pandas()
        
        fig = make_subplots(
            rows = 2, cols = 1,
            shared_xaxes = True,
            vertical_spacing=0.02,  # 各行間の間隔
            # specs=[[{}, {}], [{}, {}]],  # 2行2列目以外のセルにプロットを配置
            row_heights=[1, 0.3]  # 上段70%, 下段30%
        )

        # ローソクチャートを追加
        fig.add_trace(
            go.Candlestick(
                x=pddf["date"],
                open=pddf["open"],
                high=pddf["high"],
                low=pddf["low"],
                close=pddf["close"],
                name="株価"
            ),
            row=1, col=1
        )
        
        # x1軸の日付ラベルをセット
        row_num = self.df.shape[0]
        if row_num <= self.ticknum:
            self.tickvals = pddf["date"]
        else:
            step = int(row_num/self.ticknum)
            self.tickvals = pddf["date"][::step]

        # 出来高のバーグラフを追加
        fig.add_trace(
            go.Bar(
                x=pddf["date"],
                y=pddf["volume"],
                name="出来高",
                marker=dict(color="blue")
            ),
            row=2, col=1
        )
        
        self.fig = fig
        self._update_layout()
    
    # 指定した日に縦線を引く
    # ただし、start_date > target_date or end_date < target_dateの場合は無視される
    def add_vline(self, target_date: date) -> None:
        if self.start_date > target_date or self.end_date < target_date:
            return

        self.fig.add_vline(
            x=target_date.strftime(DATEFORMAT),
            line=dict(color='grey', width=1),  # 色やスタイルのカスタマイズ
            opacity=0.5
        )
        
        
    
    def _update_layout(self):
        fig = self.fig

        # レイアウトの設定
        chart_start = self.df["date"].min()
        chart_end = self.df["date"].max()
        fig.update_layout(
            title=f'{self.name}({self.code})株価ローソクチャートと出来高{chart_start} ～ {chart_end}',
            xaxis_rangeslider_visible=False,  # レンジスライダーを非表示
            xaxis=dict(
                type='category'
                # type="linear" # x軸を連続データとして扱う
            ),  # 下段のX軸にタイトルを設定
            xaxis2=dict(
                title="取引日",
                type='category',
                tickvals=self.tickvals,
                tickangle = self.tickangle
                # type="linear" # x軸を連続データとして扱う
            ),  # 下段のX軸にタイトルを設定
            yaxis=dict(title="株価"),  # 上段のY軸
            yaxis2=dict(title="出来高"),  # 下段のY軸
            # showlegend=False  # 凡例を非表示
            height= 600  #高さの設定
        )
        
        self.fig = fig
        
        
        
    
        
# debug
if __name__ == '__main__':
    code = 1301
    end_date = date.today()
    start_date = end_date + relativedelta(months=-3)
    
    PFIG = PricelistFig(code, start_date=start_date, end_date=end_date)
    
    
    
    
    
    
            
