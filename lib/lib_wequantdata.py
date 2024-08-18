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
from lib import DOWNLOADABLE_FILES
import polars as pl

class PricelistPl():
    def __init__(self, filename):
        fp = str(DATA_DIR/filename)
        data_dir = str(DATA_DIR)
        
        # 管理対象外ファイルの場合、raise ValueError
        if not filename in DOWNLOADABLE_FILES:
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
        
        # debug
        print(df.columns)  
        
        # 平均をとる
        
        
    
    

# debug
if __name__ == '__main__':
    rawPL = PricelistPl("reviced_pricelist.parquet")
    print(rawPL.df)
    
    rawPL.with_columns_moving_average(25)
    
    
            
