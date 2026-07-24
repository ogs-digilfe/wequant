# Pathの設定
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).parent
PJ_DIR = CURRENT_DIR.parent
LIB_DIR = PJ_DIR / "lib"

sys.path.append(str(LIB_DIR))

# import object
from lib_dataprocess import read_data, PricelistPl

# dataの読み込み
fp = PJ_DIR / "data" / "raw_pricelist.parquet"
df = read_data(fp)

# dataを分割修正して保存
RawPL = PricelistPl(df)
reviced_priceliset_df = RawPL.get_reviced_pricelist()
reviced_priceliset_df.write_parquet(PJ_DIR / "data" / "reviced_pricelist.parquet")