# パスのセット
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).parent
PJROOT_DIR = CURRENT_DIR.parent
WORKSPACE_DIR = PJROOT_DIR.parent
DATA_DIR = PJROOT_DIR / "data"

sys.path.append(str(PJROOT_DIR))

# オブジェクトのインポート
import os
import polars as pl
from typing import Union

# global
DOWNLOADABLE_FILES = [
    "finance_quote.parquet",
    "kessan.parquet",
    "meigaralist.parquet",
    "nh225.parquet",
    "raw_pricelist.parquet",
    "reviced_pricelist.parquet"
]

class PricelistPl():
    # fp = filenameの場合、dirはDATA_DIR
    # fp = filepathの場合、fpはfilepathとして処理
    def __init__(self, fp: Union[str, Path]):
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
    
    # colで指定した列のterm日の移動平均列を、25日移動平均であれば、ma25の
    # ような列名(maの後ろに移動平均の日数)で追加する。
    # termで指定した日数での移動平均が計算できない初期のレコードは、dropされてなくなる
    # 全データで実施すると、かなりメモリを消費するので、200日移動平均などを取得する場合は、
    # PricelistPl(filename).dfをfilterしてから実施しないとメモリが足りなくなるかもしれない。
    # メモリが不足して実行プロセスがダウンした場合は、例外も出力されない。
    def with_columns_moving_average(self, term, col="p_close"):
        df = self.df
        
        # term数shiftする
        df = df.with_columns([pl.col(col).alias('s0')])
        for i in range(1, term-1):
            df = df.with_columns([pl.col(col).shift(i).alias(f's{str(i)}')])
        last_col_shift_num = term - 1
        df = df.with_columns([
            pl.col(col).shift(last_col_shift_num).alias(f's{str(last_col_shift_num)}'),
            pl.col("mcode").shift(last_col_shift_num).alias("mcode_r")
        ])
        
        # mcode == mcode_rの行のみfilter(抽出)する
        df = df.filter(pl.col("mcode")==pl.col("mcode_r"))
        
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
        
# debug
if __name__ == '__main__':
    revPL = PricelistPl("reviced_pricelist.parquet")
    print(revPL.df)
    
    # dataを2020年以降に絞る
    from datetime import date
    df = revPL.df
    df = df.filter(pl.col("p_key")>date(2020, 1, 1))
    revPL.df = df

    # 移動平均を計算して列を追加。ここでは25日、75日、200日
    revPL.with_columns_moving_average(25)    
    revPL.with_columns_moving_average(75)
    revPL.with_columns_moving_average(200)
    print(revPL.df)
    
    
    
    
    
    
    
            
