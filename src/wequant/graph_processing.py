import pandas as pd
import polars as pl
from datetime import date
from typing import Literal, Union

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots

from wequant.data_loading import DATA_DIR, load_data_file, read_data
from wequant.data_processing import (
    DATEFORMAT,
    DATEFORMAT2,
    IndexPricelistPl,
    KessanPl,
    MeigaralistPl,
    PricelistPl,
    get_companyname,
)

# グラフ生成関数
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

class IndexPricelistFig():
    def __init__(self,
        name: Literal["nh225"] = "nh225",
        start_date: date = date(1900, 1, 1),
        end_date: date = date(2999, 12, 31),
        fig_type: Literal["daily", "weekly", "monthly"] = "daily"   #weekly, monthly未作成
    ):

        # IndexPricelistFigのプロパティ
        if name == "nh225":
            self.display_name = "日経225平均"
        self.start_date = start_date
        self.end_date = end_date
        self.ticknum = 10
        self.tickangle = 45
        
        
        # データの読み込みとfilter
        df = load_data_file(f"{name}.parquet")
        df = IndexPricelistPl(df).df
        df = df.filter(pl.col("date")>=start_date)\
            .filter(pl.col("date")<=end_date)
        df = df.with_columns([
            pl.col("date").dt.strftime(DATEFORMAT).alias("date")
        ])
        
        self.df = df
        self.set_fig()
    
    def set_fig(self):
        pddf = self.df.to_pandas()

        # ローソクチャートを追加
        fig = go.Figure(data=[
            go.Candlestick(
                x=pddf["date"],
                open=pddf["open"],
                high=pddf["high"],
                low=pddf["low"],
                close=pddf["close"],
                name="株価"
            )
        ])

        # x軸の日付ラベルをセット
        row_num = self.df.shape[0]
        if row_num <= self.ticknum:
            self.tickvals = pddf["date"]
        else:
            step = int(row_num/self.ticknum)
            self.tickvals = pddf["date"][::step]
        
        # layoutのセット
        chart_start = self.df["date"].min()
        chart_end = self.df["date"].max()

        fig.update_layout(
            title=f'{self.display_name}株価ローソクチャート{chart_start} ～ {chart_end}',
            xaxis_rangeslider_visible=False,  # レンジスライダーを非表示
            xaxis=dict(
                title="取引日",
                type='category',
                tickvals=self.tickvals,
                tickangle = self.tickangle
                # type="linear" # x軸を連続データとして扱う
            ),  # 下段のX軸にタイトルを設定
            yaxis=dict(title="株価"),  # 上段のY軸
            # showlegend=False  # 凡例を非表示
            height= 600  #高さの設定
        )
        
        self.fig = fig

        
        
        
        
        

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
                valuation_date=self.end_settlement_date,
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
            fp = DATA_DIR / "reviced_pricelist.parquet"
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
    # colorで、引く線の色を指定できるが、plotlyのcolorlabelでのみ指定可能とした。
    # ただし、start_date > target_date or end_date < target_dateの場合は無視される
    def add_vline(self, target_date: date, color: str="grey") -> None:
        if self.start_date > target_date or self.end_date < target_date:
            return

        self.fig.add_vline(
            x=target_date.strftime(DATEFORMAT),
            line=dict(color=color, width=1),  # 色やスタイルのカスタマイズ
            opacity=0.5
        )

    # 決算発表日にvlineを引く
    # weekly, monthly未対応    
    def add_vline_announcement_date(self, color: str="orange") -> None:
        start_date = self.start_date
        end_date = self.end_date

        # 決算発表日の取得
        fp = DATA_DIR/"kessan.parquet"
        df = read_data(fp)
        KPL = KessanPl(df)
        df = KPL.df
        df = df.filter(pl.col("code")==self.code)\
            .filter(pl.col("announcement_date")>=start_date)\
            .filter(pl.col("announcement_date")<=end_date)
        dates = df["announcement_date"].to_list()

        # 決算発表日にvlineを引く
        for d in dates:
            self.add_vline(d, color) 
    
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

#
# 散布図
#
class ScatterPlotFig():
    def __init__(
        self, 
        df: Union[pl.DataFrame, pd.DataFrame],
        x_col: str,
        y_col: str,
        title = ""
    ):
        
        # pl.DataFrameは型変換してself.dfをセット
        if type(df) == pd.DataFrame:
            df = pl.from_pandas()
        self.df = df

        # x列とy列のセット
        self.x_col = x_col
        self.y_col = y_col
        
        # titleのセット
        if title == "":
            title = f'x: {x_col} / y: {y_col}'
        self.title = title
        
    def get_fig(self):
        df = self.df
        pddf = df.to_pandas()

        # figインスタンスの生成
        if "category" not in df.columns:
            return px.scatter(
                pddf, 
                x = self.x_col, 
                y = self.y_col, 
                title = self.title
            )
        else:
            return px.scatter(
                pddf, 
                x = self.x_col, 
                y = self.y_col,
                color = "category", 
                title = self.title
            )
    
    # 散布図を見やすくするためのはずれ異常値処理につかう。
    # rangeを指定すると、小さいはずれ値は指定したmin_x, 大きい外れ値はmax_xに書き換えられる
    def set_x_range(
        self,
        min_x: Union[int, float, None] = None,
        max_x: Union[int, float, None] = None
    ) -> None:
        df = self.df

        if min_x is None:
            min_x = df[self.x_col].min()
        if max_x is None:
            max_x = df[self.x_col].max()

        # 小さい方をカット
        df = df.with_columns([
            pl.when(pl.col(self.x_col) < min_x)
            .then(pl.lit(min_x))
            .otherwise(pl.col(self.x_col))
            .alias(self.x_col)
        ])

        # 大きい方をカット
        df = df.with_columns([
            pl.when(pl.col(self.x_col) > max_x)
            .then(pl.lit(max_x))
            .otherwise(pl.col(self.x_col))
            .alias(self.x_col)
        ])

        self.df = df
    
    # 散布図を見やすくするためのはずれ異常値処理につかう。
    # rangeを指定すると、小さいはずれ値は指定したmin_y, 大きい外れ値はmax_yに書き換えられる
    def set_y_range(
        self,
        min_y: Union[int, float, None] = None,
        max_y: Union[int, float, None] = None
    ) -> None:
        df = self.df

        if min_y is None:
            min_y = df[self.y_col].min()
        if max_y is None:
            max_y = df[self.y_col].max()

        # 小さい方をカット
        df = df.with_columns([
            pl.when(pl.col(self.y_col) < min_y)
            .then(pl.lit(min_y))
            .otherwise(pl.col(self.y_col))
            .alias(self.y_col)
        ])

        # 大きい方をカット
        df = df.with_columns([
            pl.when(pl.col(self.y_col) > max_y)
            .then(pl.lit(max_y))
            .otherwise(pl.col(self.y_col))
            .alias(self.y_col)
        ])
    
    # y列の閾値で2つのグループに分類し、category列を追加してプロットの色をカテゴリごとに変える
    # "category"列がある場合は、処理しない。
    def with_columns_category(
        self,
        y_min: Union[int, float, None] = None,
        y_max: Union[int, float, None] = None,
    ) -> None:
    
        df = self.df
        col = "category"
        if col in df.columns:
            print(f'列{col}がすでにあります。新たにcategory列で分類したい場合はScatterPlotFig.dfの列{col}を削除してください。')
            return
        if (y_min is None) and (y_max is None):
            df = df.with_columns([
                pl.lit(1)
            ])
        elif (y_min is None) and (y_max is not None):
            df = df.with_columns([
                pl.when(pl.col(self.y_col) <= y_max)
                .then(pl.lit(2))
                .otherwise(pl.lit(1))
                .alias(col)
            ])
        elif (y_min is not None) and (y_max is None):
            df = df.with_columns([
                pl.when(pl.col(self.y_col) >= y_min)
                .then(pl.lit(2))
                .otherwise(pl.lit(1))
                .alias(col)
            ])
        else:
            df = df.with_columns([
                pl.when(pl.col(self.y_col) < y_min)
                .then(pl.lit(1))
                .when(pl.col(self.y_col) > y_max)
                .then(pl.lit(3))
                .otherwise(pl.lit(2))
                .alias(col)
            ])
        
        self.df = df
